//! A consumer's log of the host pages written in a no-copy object's memory (a runtime extension:
//! Metal has none), by which every copy of those pages into memory of the device's own copies in
//! only what changed since its last copy ([`HostMemory::with_dirty_pages`]).
//!
//! A consumer that logs every write to its pages (a VMM, from its hypervisor's dirty log) says
//! which pages changed through [`DirtyPages`], given with the memory. The runtime keeps it with
//! every import made from that memory, on every device, discrete and integrated alike.
//!
//! - **The stamps.** A [`DirtyStamp`] names one of the source's publications of its log. The
//!   runtime asks for a stamp at a command buffer's commit ([`DirtyPages::request`]) where one of
//!   its copies takes the pages again, and asks whether that publication exists as the command
//!   buffer's commands are recorded for the GPU, at its submission ([`DirtyPages::published`],
//!   which never blocks the scheduler's thread where the source can say at once): where it does
//!   not yet, the copy takes every row. A copy then syncs to that stamp: every write made after it
//!   is stamped later.
//! - **The pages changed** ([`DirtyPages::changed_except`]) since the stamp a copy synced to, by
//!   any writer but that copy's own, are the only bytes it may lack: they are copied in, the rest
//!   skipped.
//! - **The runtime's own writes** into the pages (a shadow's copy back or merge) are reported
//!   ([`DirtyPages::wrote_as`]), naming the copy as their writer, so other copies of the same
//!   pages see them as changed and the copy's own copies in, under a log that keeps writers
//!   apart, do not.
//! - **In the order the GPU runs them.** A shadow's hold (the stamp as of which its image holds
//!   every texel of the truth) is decided, let go of and taken in the order the scheduler submits
//!   command buffers, which is the order its tracker plans their accesses in, as each one's first
//!   batch is submitted ([`DirtyShadows::planned`]): a command buffer that waits for an event may
//!   be submitted after work committed later, and that work's writes then run before its copies
//!   in. The stamp a copy in syncs to is still asked at the commit ([`DirtyShadows::stamped`]),
//!   so every CPU write made before the commit is in the publication it waits for, and that
//!   publication has most often come by the submission. A command buffer whose commands lie in
//!   more than one batch may run its later batches after work submitted after its first: its
//!   copies in count on no hold, it takes none, and its writes keep every hold they meet let go
//!   of until it completes.
//! - **Correctness first.** A wait that answers `false`, a source that cannot say, and a copy
//!   that holds nothing it may count on all copy every byte.
//!
//! [`HostMemory::with_dirty_pages`]: crate::HostMemory::with_dirty_pages

use super::device::lock;
use super::resource::ResourceKey;
use super::shadow::ShadowCopies;
use super::tracker::Use;
use ash::vk;
use std::fmt;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

/// One publication of a [`DirtyPages`] source's log, in the order the source publishes them,
/// which every source over the same host bytes shares: a later publication's stamp is larger.
/// Opaque to the runtime, which only compares and hands back stamps a source gave it. No Metal
/// counterpart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DirtyStamp(pub u64);

/// A consumer's log of the writes to the host pages of a no-copy object's memory, which lets
/// every copy the runtime makes of those pages into memory of the device's own copy in only the
/// pages written since its last copy ([`HostMemory::with_dirty_pages`]). No Metal counterpart:
/// Metal's GPU reads a no-copy object's pages in place.
///
/// **Offsets** are bytes from the memory's pointer ([`HostMemory::pointer`]), the first byte
/// the consumer gave. A page that starts before the pointer is reported from 0; a range reported
/// past the bytes the runtime reads is cut to them.
///
/// **Its promises.** It logs at a granularity of whole pages, of up to 256 KiB. A page counts as
/// changed for any stamp earlier than the start of the log's continuous watch over it
/// ([`watch`](Self::watch)), which the first watch over the page begins, whatever source asks;
/// a page no one watches any more counts as changed for every stamp until one watches it again.
/// A write the host makes (the consumer's own, and those [`wrote`](Self::wrote) reports) is
/// stamped at once. The sources over the same host bytes share one order of stamps: each answers
/// [`changed`](Self::changed) for a stamp another of them gave as for its own. A closed source
/// (the consumer's reset, pause or stop) answers [`wait`](Self::wait) with `false`, so the
/// runtime takes every page as changed.
///
/// **Writers** (optional: both methods default to the plain ones). The runtime reports a copy's
/// own copy back or merge naming the copy as its writer ([`wrote_as`](Self::wrote_as)), and asks
/// for that copy's copy in which pages changed by any writer but it
/// ([`changed_except`](Self::changed_except)): it already holds what it wrote, so a page only it
/// wrote since is no change to it. A writer is a [`NonZeroU64`] no other writer in the process
/// is ever given; 0 is the consumer's own writes and every write reported through
/// [`wrote`](Self::wrote), which no copy's question leaves out. The sources over the same host
/// bytes share their writers as their stamps. One way to keep it: per page, its latest stamp and
/// that write's writer, and the latest stamp of any other writer; `changed_except(since, w)`
/// counts the page where its latest writer is not `w` and its latest stamp is later than
/// `since`, or where it is and the other writers' latest stamp is.
///
/// **Threads.** Called on the runtime's threads, never under a lock the source's own calls
/// take: [`watch`](Self::watch) where a copy of the pages is made, [`request`](Self::request) on
/// the thread that commits a command buffer, [`wait`](Self::wait), [`changed`](Self::changed)
/// and [`changed_except`](Self::changed_except) on the thread that records its commands for the
/// GPU, [`wrote`](Self::wrote) and [`wrote_as`](Self::wrote_as) at the commit and at the
/// completion. A [`Watch`](Self::Watch) may be dropped on a completion thread, so its drop takes
/// no lock the consumer holds while it calls the runtime.
///
/// [`HostMemory::with_dirty_pages`]: crate::HostMemory::with_dirty_pages
/// [`HostMemory::pointer`]: crate::HostMemory::pointer
pub trait DirtyPages: Send + Sync {
    /// A watch of a range of the memory's pages, ended by its drop.
    type Watch: Send + Sync;

    /// Starts watching the memory's bytes `range` (offsets from its pointer), for the watch's
    /// life: the consumer logs only watched pages, so a page no copy reads costs the guest
    /// nothing. [`changed`](Self::changed) reports watched pages alone; an unwatched page reads
    /// as changed at every stamp. The runtime watches the bytes of each copy it keeps, for the
    /// copy's life.
    fn watch(&self, range: Range<u64>) -> Self::Watch;

    /// A stamp a future publication reaches: every write made before this call shows in
    /// [`changed`](Self::changed) once [`wait`](Self::wait) on it returns `true`. Never blocks.
    /// The wait for it comes before the command buffer's commands are recorded: a publication
    /// that needs that command buffer's own work never comes within it.
    fn request(&self) -> DirtyStamp;

    /// Block until publication `stamp` exists, bounded (about 4 ms). `false`: the source cannot
    /// say (or is closed), so the runtime takes every page as changed (a whole copy in). The
    /// runtime calls it only through [`published`](Self::published)'s default.
    fn wait(&self, stamp: DirtyStamp) -> bool;

    /// Whether publication `stamp` exists now, as [`wait`](Self::wait) would answer once it
    /// returns, without blocking: the runtime asks it as a command buffer is submitted, on the
    /// device's scheduler thread, inside its lock, where a wait would hold back every submission
    /// of the device. `false` (not yet, or the source cannot say) makes the copy take every row.
    /// By default [`wait`](Self::wait)'s answer, which blocks: a source that knows at once which
    /// publication it reached overrides it.
    fn published(&self, stamp: DirtyStamp) -> bool {
        self.wait(stamp)
    }

    /// How long the runtime may hold a command buffer's submission for the publication of the
    /// stamp its gated copies in sync to, where a hold would let them copy only the rows changed
    /// (about the bound a publication takes, 4 ms for a VMM that flushes its vCPUs' logs): the
    /// scheduler holds that command buffer alone, its queue's later work and the work that
    /// conflicts with it behind it, asking [`published`](Self::published) about every half
    /// millisecond, and submits it at the bound whatever the answer (every row then). Only for a
    /// source whose [`published`](Self::published) answers at once, never waiting. `None` (the
    /// default): no hold.
    fn publication_hold(&self) -> Option<Duration> {
        None
    }

    /// Append to `out` the byte runs, as offsets from the memory's pointer, of the watched pages
    /// stamped later than `since`, and of those whose continuous watch began after it.
    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>);

    /// The runtime wrote the memory's bytes `range` (offsets from its pointer) from the GPU (a
    /// shadow's copy back, a merge), so other copies of the same pages see it as changed.
    fn wrote(&self, range: Range<u64>);

    /// [`wrote`](Self::wrote), naming its writer: the copy whose copy back or merge it is (the
    /// trait's **Writers**). By default, `wrote`.
    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        let _ = writer;
        self.wrote(range);
    }

    /// [`changed`](Self::changed), leaving out a page whose every write stamped later than
    /// `since` was `writer`'s ([`wrote_as`](Self::wrote_as)); a page another writer wrote since
    /// (the consumer among them), or one whose watch began after `since`, still counts (the
    /// trait's **Writers**). By default, `changed`.
    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        let _ = writer;
        self.changed(since, out);
    }
}

/// A [`DirtyPages`] source of any watch type, as the runtime keeps it: its watches boxed.
pub(crate) trait DirtyHook: Send + Sync {
    /// [`DirtyPages::watch`], boxed.
    fn watch(&self, range: Range<u64>) -> Box<dyn Send + Sync>;
    /// [`DirtyPages::request`].
    fn request(&self) -> DirtyStamp;
    /// [`DirtyPages::published`].
    fn published(&self, stamp: DirtyStamp) -> bool;
    /// [`DirtyPages::publication_hold`].
    fn publication_hold(&self) -> Option<Duration>;
    /// [`DirtyPages::wrote`].
    fn wrote_as_consumer(&self, range: Range<u64>);
    /// [`DirtyPages::wrote_as`].
    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64);
    /// [`DirtyPages::changed_except`].
    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>);
}

/// A consumer's source, shared.
struct Hooked<D: ?Sized>(Arc<D>);

impl<D> DirtyHook for Hooked<D>
where
    D: DirtyPages + ?Sized,
    D::Watch: 'static,
{
    fn watch(&self, range: Range<u64>) -> Box<dyn Send + Sync> {
        Box::new(self.0.watch(range))
    }
    fn request(&self) -> DirtyStamp {
        self.0.request()
    }
    fn published(&self, stamp: DirtyStamp) -> bool {
        self.0.published(stamp)
    }
    fn publication_hold(&self) -> Option<Duration> {
        self.0.publication_hold()
    }
    fn wrote_as_consumer(&self, range: Range<u64>) {
        self.0.wrote(range);
    }
    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.0.wrote_as(range, writer);
    }
    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.0.changed_except(since, writer, out);
    }
}

/// `source`, as the runtime keeps it.
pub(crate) fn hook<D>(source: Arc<D>) -> Arc<dyn DirtyHook>
where
    D: DirtyPages + ?Sized + 'static,
    D::Watch: 'static,
{
    Arc::new(Hooked(source))
}

impl fmt::Debug for dyn DirtyHook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("DirtyPages")
    }
}

/// A consumer's [`DirtyPages`] as the runtime keeps it with the pages it came with: the source,
/// the host address its offsets count from, and the latest publication a wait on it found, so a
/// wait on an earlier one is not made again.
pub(crate) struct DirtySource {
    hook: Arc<dyn DirtyHook>,
    /// The memory's pointer, offset 0 of the source.
    pointer: u64,
    /// One past the latest stamp a wait returned `true` for; 0 for none.
    published: AtomicU64,
    /// One past the stamp the latest `false` answer was for (the same commit's other copies in
    /// ask no more); 0 for none.
    unanswered: AtomicU64,
}

impl DirtySource {
    /// `hook`, whose offsets count from `pointer`.
    pub(crate) fn new(hook: Arc<dyn DirtyHook>, pointer: u64) -> Arc<DirtySource> {
        Arc::new(DirtySource {
            hook,
            pointer,
            published: AtomicU64::new(0),
            unanswered: AtomicU64::new(0),
        })
    }

    /// [`DirtyPages::request`], counted in `tally`.
    fn request(&self, tally: &DirtyTally) -> DirtyStamp {
        let stamp = self.hook.request();
        tally.stamps.fetch_add(1, Ordering::Relaxed);
        stamp
    }

    /// The source's offset of the host address `address`; `None` below its pointer.
    pub(crate) fn offset_of(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.pointer)
    }

    /// Whether publication `stamp` exists: at once where an answer for it or a later one was
    /// `true`, `false` at once where the answer for it was `false` (its commit's other copies in);
    /// else [`DirtyPages::published`]'s answer, which a source that knows at once gives without
    /// blocking the scheduler's thread. Not `false` for an earlier stamp: a command buffer
    /// submitted long after its commit finds its stamp published (counted in `tally`).
    fn published(&self, stamp: DirtyStamp, tally: &DirtyTally) -> bool {
        if self.published.load(Ordering::Acquire) > stamp.0 {
            return true;
        }
        let target = stamp;
        if self.unanswered.load(Ordering::Acquire) == target.0.saturating_add(1) {
            return false;
        }
        let started = std::time::Instant::now();
        let published = self.hook.published(target);
        let nanos = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        tally.waits.fetch_add(1, Ordering::Relaxed);
        tally.wait_nanos.fetch_add(nanos, Ordering::Relaxed);
        match published {
            true => {
                (self.published).fetch_max(target.0.saturating_add(1), Ordering::AcqRel);
            }
            false => {
                tally.waits_unanswered.fetch_add(1, Ordering::Relaxed);
                (self.unanswered).store(target.0.saturating_add(1), Ordering::Release);
            }
        }
        published
    }

    /// Whether publication `stamp` exists, asked of a source that answers at once
    /// ([`DirtyPages::publication_hold`]) as the scheduler looks at a held command buffer:
    /// uncounted, and a `false` remembered nowhere, the answer being "not yet".
    fn published_now(&self, stamp: DirtyStamp) -> bool {
        if self.published.load(Ordering::Acquire) > stamp.0 {
            return true;
        }
        let published = self.hook.published(stamp);
        if published {
            (self.published).fetch_max(stamp.0.saturating_add(1), Ordering::AcqRel);
        }
        published
    }

    /// [`DirtyPages::wrote`] of the source's offsets `range` by the runtime's own CPU (a
    /// `replaceRegion` of a texture over the pages, writer 0), where non-empty.
    pub(crate) fn wrote(&self, range: Range<u64>) {
        if range.start < range.end {
            self.hook.wrote_as_consumer(range);
        }
    }
}

impl fmt::Debug for DirtySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DirtySource")
            .field("pointer", &self.pointer)
            .finish()
    }
}

/// A shadow over logged host pages (`shadow.rs`), as its copies in ask the log: the source, the
/// shadow's writer, where its truth lies in the source, the watch over those bytes, and the stamp
/// as of which its image held every texel of the truth, where it holds one ([`DirtyShadows`]).
pub(crate) struct ShadowLog {
    copies: Weak<ShadowCopies>,
    source: Arc<DirtySource>,
    writer: NonZeroU64,
    /// The `VkBuffer` the truth lay in when the log was made, and the source's offset of that
    /// `VkBuffer`'s byte 0 (wrapping: it may lie below the pointer).
    raw: vk::Buffer,
    delta: u64,
    /// The truth's bytes, as the source's offsets.
    truth: Range<u64>,
    _watch: Box<dyn Send + Sync>,
    /// The stamp as of which the image holds every texel of the truth, but for the pages the log
    /// reports changed since (the module docs); `None` where it holds nothing a copy may count on.
    hold: Mutex<Option<DirtyStamp>>,
    /// The submitted command buffers whose writes meet the truth or the shadow from a batch that
    /// may run after work submitted later, until each completes: while any does, the hold stays
    /// let go of and no copy in counts on it ([`DirtyShadows::planned`]).
    unsettled: AtomicU64,
}

impl ShadowLog {
    /// The source's offsets of the `VkBuffer` bytes `range`.
    fn offsets(&self, range: Range<u64>) -> Range<u64> {
        range.start.wrapping_add(self.delta)..range.end.wrapping_add(self.delta)
    }

    /// Report a copy back or merge of the shadow into the `VkBuffer` bytes `range`
    /// ([`DirtyPages::wrote_as`], the shadow its writer).
    pub(crate) fn wrote_back(&self, range: Range<u64>) {
        let range = self.offsets(range);
        let (start, end) = (
            range.start.max(self.truth.start),
            range.end.min(self.truth.end),
        );
        if start < end {
            self.source.hook.wrote_as(start..end, self.writer);
        }
    }
}

/// The logs of a device's shadows over logged host pages, by the shadow's key: made at a shadow's
/// first copy in over such pages, and let go of once the shadow went.
#[derive(Default)]
pub(crate) struct DirtyShadows {
    logs: Mutex<Vec<Arc<ShadowLog>>>,
    /// Whether any log was ever made: a command buffer collects its writes only then.
    any: AtomicBool,
    tally: Arc<DirtyTally>,
}

/// The gated copies in, as `WorkCounts::dirty_copies_in_gated` and kin count them.
#[derive(Default)]
pub(crate) struct DirtyTally {
    pub(crate) gated: AtomicU64,
    pub(crate) whole: AtomicU64,
    pub(crate) skipped: AtomicU64,
    pub(crate) skipped_shared: AtomicU64,
    pub(crate) bytes_whole: AtomicU64,
    pub(crate) bytes_copied: AtomicU64,
    pub(crate) stamps: AtomicU64,
    pub(crate) waits: AtomicU64,
    pub(crate) waits_unanswered: AtomicU64,
    pub(crate) wait_nanos: AtomicU64,
}

/// The writers no other shadow in the process is ever given.
static NEXT_WRITER: AtomicU64 = AtomicU64::new(1);

impl DirtyShadows {
    /// The gated copies in so far.
    pub(crate) fn tally(&self) -> &DirtyTally {
        &self.tally
    }

    /// Whether any shadow over logged host pages was copied in on the device.
    pub(crate) fn any(&self) -> bool {
        self.any.load(Ordering::Acquire)
    }

    /// The log of `copies`, whose truth lies at `truth` of the `VkBuffer` `raw` in pages logged by
    /// `source` (`raw`'s byte `v` at the host address `v + base`): kept, or made now, its watch
    /// begun. A kept log of another source or `VkBuffer` (a rebind gave the buffer other pages)
    /// is made again.
    pub(crate) fn log(
        &self,
        copies: &Arc<ShadowCopies>,
        source: &Arc<DirtySource>,
        raw: vk::Buffer,
        base: u64,
        truth: Range<u64>,
    ) -> Option<Arc<ShadowLog>> {
        let delta = base.wrapping_sub(source.pointer);
        let offsets = truth.start.wrapping_add(delta)..truth.end.wrapping_add(delta);
        // A truth below the pointer, or wrapping, is no range of the source's.
        if offsets.start > offsets.end || base.checked_add(truth.start)? < source.pointer {
            return None;
        }
        let mut logs = lock(&self.logs);
        logs.retain(|log| log.copies.strong_count() > 0);
        let kept = logs.iter().position(|log| {
            std::ptr::eq(log.copies.as_ptr(), Arc::as_ptr(copies))
                && Arc::ptr_eq(&log.source, source)
                && log.raw == raw
                && log.truth == offsets
        });
        if let Some(at) = kept {
            return Some(Arc::clone(&logs[at]));
        }
        logs.retain(|log| !std::ptr::eq(log.copies.as_ptr(), Arc::as_ptr(copies)));
        let writer = NonZeroU64::new(NEXT_WRITER.fetch_add(1, Ordering::Relaxed))?;
        let log = Arc::new(ShadowLog {
            copies: Arc::downgrade(copies),
            source: Arc::clone(source),
            writer,
            raw,
            delta,
            truth: offsets.clone(),
            _watch: source.hook.watch(offsets),
            hold: Mutex::new(None),
            unsettled: AtomicU64::new(0),
        });
        logs.push(Arc::clone(&log));
        self.any.store(true, Ordering::Release);
        Some(log)
    }

    /// A command buffer whose commit was `commit` completed, `failed` or not: its shadows' copies
    /// back and merges are reported again ([`DirtyPages::wrote_as`]), now they ran, so a stamp
    /// asked between the commit and the GPU's write still finds them; the holds its writes
    /// kept unsettled may be taken again; where it failed, the holds its copies in set are let go
    /// of, its copies having perhaps not run.
    pub(crate) fn completed(&self, commit: &mut DirtyCommit, failed: bool) {
        for (log, range) in &commit.wrote {
            log.wrote_back(range.clone());
        }
        for log in commit.unsettling.drain(..) {
            log.unsettled.fetch_sub(1, Ordering::AcqRel);
        }
        if !failed {
            return;
        }
        for gate in &commit.gates {
            let Some(decision) = *lock(&gate.decision) else {
                continue;
            };
            let mut hold = lock(&gate.log.hold);
            if *hold == Some(decision.stamp) {
                *hold = None;
            }
        }
    }

    /// The log kept for `copies`, where its copies in made one.
    pub(crate) fn kept(&self, copies: &ShadowCopies) -> Option<Arc<ShadowLog>> {
        let logs = lock(&self.logs);
        (logs.iter())
            .find(|log| std::ptr::eq(log.copies.as_ptr(), copies))
            .cloned()
    }

    /// A committed command buffer's `commit` ([`DirtyCommit`]), at its commit: a stamp of each
    /// source its gated copies in read, one per source, asked now, so every CPU write made before
    /// the commit, which Metal shows the command buffer, is in the publication its copies in wait
    /// for ([`planned`](Self::planned) decides what they copy); a stamp too for the logs of the
    /// shadows its end left holding every texel of their truth (`held`) with no gated copy in
    /// (a pass that cleared every texel), so their holds count from the commit on; and its copies
    /// back and merges, reported ([`DirtyPages::wrote_as`]) before any stamp a later commit asks.
    pub(crate) fn stamped(&self, commit: &mut DirtyCommit, held: &[Arc<ShadowCopies>]) {
        for (log, range) in &commit.wrote {
            log.wrote_back(range.clone());
        }
        let mut stamps: Vec<(Arc<DirtySource>, DirtyStamp)> = Vec::new();
        let mut stamp_of = |source: &Arc<DirtySource>| match (stamps.iter())
            .find(|(s, _)| Arc::ptr_eq(s, source))
        {
            Some((_, stamp)) => *stamp,
            None => {
                let stamp = source.request(&self.tally);
                stamps.push((Arc::clone(source), stamp));
                stamp
            }
        };
        for gate in &commit.gates {
            *lock(&gate.stamp) = Some(stamp_of(&gate.log.source));
        }
        for copies in held {
            let gated = (commit.gates.iter()).any(|gate| gate.of(copies));
            if let (false, Some(log)) = (gated, self.kept(copies)) {
                let stamp = stamp_of(&log.source);
                commit.held_logs.push((log, stamp));
            }
        }
    }

    /// How long `commit`'s command buffer may be held for the publication of a stamp one of its
    /// gated copies in syncs to (the longest [`DirtyPages::publication_hold`] of them), where
    /// one of them could count on a hold as things stand and its source has not published its
    /// stamp yet; `None` where none waits, and for a command buffer of more than one batch, whose
    /// copies in count on no hold ([`planned`](Self::planned)).
    pub(crate) fn awaits_publication(&self, commit: &DirtyCommit) -> Option<Duration> {
        (commit.gates.iter())
            .filter_map(|gate| {
                let stamp = (*lock(&gate.stamp))?;
                let source = &gate.log.source;
                let bound = source.hook.publication_hold()?;
                let counts = !gate.written_before
                    && gate.log.unsettled.load(Ordering::Acquire) == 0
                    && lock(&gate.log.hold).is_some();
                (counts && !source.published_now(stamp)).then_some(bound)
            })
            .max()
    }

    /// A committed command buffer's `commit` ([`DirtyCommit`]), as its first batch is submitted,
    /// in the order the scheduler submits command buffers and its tracker plans them (the module
    /// docs): its gates take the holds as they stand, each with the stamp its commit asked (no hold
    /// where the command buffer wrote the truth or the shadow before the copy, where it is `split`
    /// over more than one batch, whose later ones may run after work submitted later, or where
    /// another such command buffer's writes keep the hold unsettled); its writes let go of the
    /// holds of the shadows they meet, a `split` one's until it completes; and the shadows its end
    /// left holding every texel of the truth (`held`) hold from their stamp on, but for a `split`
    /// one's.
    pub(crate) fn planned(
        &self,
        commit: &mut DirtyCommit,
        held: &[Arc<ShadowCopies>],
        split: bool,
    ) {
        let mut synced: Vec<(Arc<ShadowLog>, DirtyStamp)> = Vec::new();
        for gate in &commit.gates {
            let Some(stamp) = *lock(&gate.stamp) else {
                continue;
            };
            let unsettled = gate.log.unsettled.load(Ordering::Acquire) > 0;
            let since = match gate.written_before || split || unsettled {
                true => None,
                false => *lock(&gate.log.hold),
            };
            *lock(&gate.decision) = Some(Decision { since, stamp });
            synced.push((Arc::clone(&gate.log), stamp));
        }
        let logs: Vec<Arc<ShadowLog>> = lock(&self.logs).clone();
        for log in &logs {
            let Some(copies) = log.copies.upgrade() else {
                continue;
            };
            if !commit.wrote_over(&copies) {
                continue;
            }
            *lock(&log.hold) = None;
            if split {
                log.unsettled.fetch_add(1, Ordering::AcqRel);
                commit.unsettling.push(Arc::clone(log));
            }
        }
        if split {
            return;
        }
        synced.append(&mut commit.held_logs);
        for (log, stamp) in synced {
            let holds = log
                .copies
                .upgrade()
                .is_some_and(|copies| held.iter().any(|h| Arc::ptr_eq(h, &copies)));
            if holds && log.unsettled.load(Ordering::Acquire) == 0 {
                *lock(&log.hold) = Some(stamp);
            }
        }
    }
}

/// What a command buffer did that the logs' holds hear of as it is submitted
/// ([`DirtyShadows::planned`]): its commands' writes, its gated copies in, its shadows' copies
/// back and merges into logged pages, and the logs its writes keep unsettled until it completes.
#[derive(Default)]
pub(crate) struct DirtyCommit {
    written: Vec<Use<ResourceKey>>,
    gates: Vec<Arc<CopyInGate>>,
    wrote: Vec<(Arc<ShadowLog>, Range<u64>)>,
    unsettling: Vec<Arc<ShadowLog>>,
    /// The logs of the shadows its end held whole with no gated copy in, and their stamps
    /// ([`DirtyShadows::stamped`]).
    held_logs: Vec<(Arc<ShadowLog>, DirtyStamp)>,
}

/// The most writes a command buffer keeps apart, past which they are one write of everything.
const WRITES: usize = 4096;

impl DirtyCommit {
    /// Note a command's `uses`: its writes, each once.
    pub(crate) fn note(&mut self, uses: &[Use<ResourceKey>]) {
        for used in uses.iter().filter(|used| !used.write.is_empty()) {
            let seen = (self.written.iter().rev().take(8))
                .any(|w| w.key == used.key && w.range == used.range);
            if !seen && self.written.len() < WRITES {
                self.written.push(used.clone());
            }
        }
    }

    /// Whether a write noted so far meets `copies`' truth or its shadow (every one, past
    /// [`WRITES`]).
    fn wrote_over(&self, copies: &ShadowCopies) -> bool {
        self.written.len() >= WRITES
            || (self.written.iter()).any(|used| used.key == copies.shadow_key || copies.meets(used))
    }

    /// A gate for a copy in of `log`'s shadow `copies` recorded now ([`CopyInGate`]), counted in
    /// `shadows`' tally.
    pub(crate) fn gate(
        &mut self,
        (log, copies, texel): (Arc<ShadowLog>, &ShadowCopies, u64),
        shadows: &DirtyShadows,
    ) -> Arc<CopyInGate> {
        let gate = Arc::new(CopyInGate {
            log,
            texel,
            tally: Arc::clone(&shadows.tally),
            written_before: self.wrote_over(copies),
            stamp: Mutex::new(None),
            decision: Mutex::new(None),
        });
        self.gates.push(Arc::clone(&gate));
        gate
    }

    /// Note a copy back or merge of `log`'s shadow into its `VkBuffer`'s bytes `range`.
    pub(crate) fn wrote_back(&mut self, log: Arc<ShadowLog>, range: Range<u64>) {
        self.wrote.push((log, range));
    }

    /// Whether it holds a gated copy in, whose regions its submission decides
    /// ([`DirtyShadows::planned`]): its replay waits for that (`Replays::defer`).
    pub(crate) fn gated(&self) -> bool {
        !self.gates.is_empty()
    }
}

/// A shadow's copy in recorded with what its submission decides ([`DirtyShadows::planned`]): the
/// stamp its log's hold stood at, where it held one, and the stamp it syncs to.
#[derive(Clone, Copy)]
struct Decision {
    since: Option<DirtyStamp>,
    stamp: DirtyStamp,
}

/// A copy in of a shadow over logged host pages, its regions decided as its command buffer's
/// commands are recorded for the GPU (`recorder`): every region where its submission found no
/// hold, where it is recorded before its submission decided, or where its log cannot say, else
/// the rows of the pages changed since the hold.
pub(crate) struct CopyInGate {
    log: Arc<ShadowLog>,
    /// The bytes of one texel in the buffer.
    texel: u64,
    /// The command buffer wrote the truth or the shadow before the copy.
    written_before: bool,
    tally: Arc<DirtyTally>,
    /// The stamp its commit asked of the log's source ([`DirtyShadows::stamped`]).
    stamp: Mutex<Option<DirtyStamp>>,
    decision: Mutex<Option<Decision>>,
}

/// A run of rows more than this many rows from the next is copied apart.
const ROW_GAP: u32 = 8;
/// The most regions a gated copy in records, past which it copies every region.
const REGIONS: usize = 1024;

impl CopyInGate {
    /// Whether it gates a copy in of `copies`.
    fn of(&self, copies: &Arc<ShadowCopies>) -> bool {
        std::ptr::eq(self.log.copies.as_ptr(), Arc::as_ptr(copies))
    }

    /// The regions of `whole` to copy now, into `out`, counted: `whole` itself where the decision
    /// holds no stamp, the wait cannot say, or the rows changed come to half the rows or more;
    /// none where no page changed.
    pub(crate) fn regions(
        &self,
        whole: &[vk::BufferImageCopy],
        out: &mut Vec<vk::BufferImageCopy>,
    ) {
        let (total, copied) = self.chosen(whole, out);
        let tally = &self.tally;
        tally.gated.fetch_add(1, Ordering::Relaxed);
        tally.bytes_whole.fetch_add(total, Ordering::Relaxed);
        tally.bytes_copied.fetch_add(copied, Ordering::Relaxed);
        if out.is_empty() {
            tally.skipped.fetch_add(1, Ordering::Relaxed);
            let shared = (self.log.copies.upgrade())
                .is_some_and(|copies| copies.shared.load(Ordering::Relaxed));
            if shared {
                tally.skipped_shared.fetch_add(1, Ordering::Relaxed);
            }
        } else if copied == total {
            tally.whole.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// [`regions`](Self::regions), uncounted: the bytes `whole` and the chosen regions move.
    fn chosen(
        &self,
        whole: &[vk::BufferImageCopy],
        out: &mut Vec<vk::BufferImageCopy>,
    ) -> (u64, u64) {
        out.clear();
        let bytes = |regions: &[vk::BufferImageCopy]| -> u64 {
            (regions.iter())
                .map(|r| {
                    u64::from(r.image_extent.width)
                        * u64::from(r.image_extent.height)
                        * u64::from(r.image_extent.depth.max(1))
                        * self.texel
                })
                .sum()
        };
        let total = bytes(whole);
        let decision = *lock(&self.decision);
        let changed = decision.and_then(|Decision { since, stamp }| {
            let since = since?;
            if !self.log.source.published(stamp, &self.tally) {
                return None;
            }
            let mut runs = Vec::new();
            (self.log.source.hook).changed_except(since, self.log.writer, &mut runs);
            Some(runs)
        });
        let Some(runs) = changed else {
            out.extend_from_slice(whole);
            return (total, total);
        };
        for region in whole {
            if !self.bands(region, &runs, out) {
                out.clear();
                out.extend_from_slice(whole);
                return (total, total);
            }
        }
        if out.len() > REGIONS {
            out.clear();
            out.extend_from_slice(whole);
            return (total, total);
        }
        (total, bytes(out))
    }

    /// The full-width bands of `region`'s rows that the source's changed `runs` meet, into
    /// `out`; `false` where the region is copied whole (rows changed come to half or more, or
    /// its shape is not one of rows).
    fn bands(
        &self,
        region: &vk::BufferImageCopy,
        runs: &[Range<u64>],
        out: &mut Vec<vk::BufferImageCopy>,
    ) -> bool {
        let extent = region.image_extent;
        if extent.depth > 1 || extent.height == 0 || extent.width == 0 {
            return false;
        }
        let row_texels = match region.buffer_row_length {
            0 => extent.width,
            row => row,
        };
        let pitch = u64::from(row_texels) * self.texel;
        let row_bytes = u64::from(extent.width) * self.texel;
        let first = region.buffer_offset.wrapping_add(self.log.delta);
        let height = u64::from(extent.height);
        let end = first + (height - 1) * pitch + row_bytes;
        let mut rows: Vec<(u32, u32)> = Vec::new();
        for run in runs {
            let (start, stop) = (run.start.max(first), run.end.min(end));
            if start >= stop {
                continue;
            }
            let top = (start - first) / pitch;
            let bottom = ((stop - first).div_ceil(pitch)).min(height);
            rows.push((top as u32, bottom as u32));
        }
        rows.sort_unstable();
        let mut bands: Vec<(u32, u32)> = Vec::new();
        for (top, bottom) in rows {
            match bands.last_mut() {
                Some(last) if top <= last.1 + ROW_GAP => last.1 = last.1.max(bottom),
                _ => bands.push((top, bottom)),
            }
        }
        let changed: u64 = bands.iter().map(|(t, b)| u64::from(b - t)).sum();
        if 2 * changed >= height {
            return false;
        }
        out.extend(bands.into_iter().map(|(top, bottom)| vk::BufferImageCopy {
            buffer_offset: region.buffer_offset + u64::from(top) * pitch,
            buffer_image_height: 0,
            image_offset: vk::Offset3D {
                y: region.image_offset.y + top as i32,
                ..region.image_offset
            },
            image_extent: vk::Extent3D {
                height: bottom - top,
                ..extent
            },
            ..*region
        }));
        true
    }
}
