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
//! - **Watched while in use.** A shadow's log watches its truth's bytes from its first gated copy
//!   in, and lets go of the watch once no copy in was recorded or submitted over it for
//!   [`IDLE_WATCH_NANOS`] (a command buffer waiting long in the scheduler keeps it), its hold with
//!   it (an unwatched page reads as changed at every stamp): a VMM's harvest costs in what is
//!   watched, and a surface a pool keeps but nothing draws holds nothing a copy needs. Its next
//!   copy in watches again and takes every row (`WorkCounts::dirty_watched_bytes`,
//!   `dirty_watched_bytes_released`).
//! - **Correctness first.** A wait that answers `false`, a source that cannot say, and a copy
//!   that holds nothing it may count on all copy every byte.
//!
//! [`HostMemory::with_dirty_pages`]: crate::HostMemory::with_dirty_pages

mod deferred;

use super::device::lock;
use super::gpu_time;
use super::resource::ResourceKey;
use super::shadow::{ShadowCopies, ShadowSource, ShadowTarget};
use super::tracker::Use;
use ash::vk;
pub(crate) use deferred::{BackGate, BackKind};
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

    /// The identity of the log these answers come from, where sources of several memories over
    /// the same host pages answer from one: two sources that give the same `Some` share its
    /// stamps, its writers and its pages' watches, so a stamp one requested, a writer one was
    /// told of, and a page's continuous watch begun through one are the other's too, each
    /// answering in offsets from its own memory's pointer. The runtime then copies a shadow over
    /// one memory in from another shadow's image over the same host bytes of the other, and
    /// only the rows changed since that shadow's hold (`dirty_pages.rs`). `None` (the default):
    /// no source but itself.
    fn log_identity(&self) -> Option<u64> {
        None
    }

    /// When the runtime writes what the GPU wrote through a shadow over the memory back into it
    /// ([`WriteBack`]). [`WriteBack::Eager`] (the default): as each encoder that wrote ends, as a
    /// Metal shared resource holds the GPU's writes once its command buffer completes.
    fn write_back(&self) -> WriteBack {
        WriteBack::Eager
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

/// When the runtime writes what the GPU wrote through a shadow back into the memory under it
/// ([`DirtyPages::write_back`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum WriteBack {
    /// As each encoder that wrote through the shadow ends: the memory holds the GPU's writes once
    /// the command buffer completes, as a Metal shared resource's does.
    #[default]
    Eager,
    /// When asked ([`Device::write_back_deferred`](crate::Device::write_back_deferred)), before
    /// a use of the memory's bytes other than through the shadow, and before the shadow goes; the
    /// runtime may write back earlier. The source's owner promises that no CPU reads the bytes
    /// the GPU wrote before it asks (an IOSurface whose users lock it before they read), so the
    /// memory may hold older bytes than the shadow until then. Until the runtime keeps such
    /// writes, it writes back as [`Eager`](Self::Eager) does.
    OnRequest,
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
    /// [`DirtyPages::log_identity`].
    fn log_identity(&self) -> Option<u64>;
    /// [`DirtyPages::write_back`].
    fn write_back(&self) -> WriteBack;
    /// The caller's source this hooks, by its address: one source's hooks share its answers.
    fn source(&self) -> *const ();
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
    fn log_identity(&self) -> Option<u64> {
        self.0.log_identity()
    }
    fn write_back(&self) -> WriteBack {
        self.0.write_back()
    }
    fn source(&self) -> *const () {
        Arc::as_ptr(&self.0).cast()
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
    /// [`DirtyPages::write_back`] of its source.
    pub(crate) fn write_back(&self) -> WriteBack {
        self.hook.write_back()
    }

    /// Whether `other`'s stamps, writers and watches are this source's: the caller's same
    /// source's, or sources of one log ([`DirtyPages::log_identity`]).
    fn shares_log(&self, other: &DirtySource) -> bool {
        std::ptr::eq(self.hook.source(), other.hook.source())
            || (self.hook.log_identity()).is_some_and(|id| other.hook.log_identity() == Some(id))
    }

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

    /// The latest stamp the runtime found published, `None` before any: every guest write made
    /// after its harvest began is stamped later (a source's ordering), so a copy that ran after it
    /// was found may sync to it.
    fn latest_published(&self) -> Option<DirtyStamp> {
        match self.published.load(Ordering::Acquire) {
            0 => None,
            past => Some(DirtyStamp(past - 1)),
        }
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
    /// The watch over the truth's bytes; `None` once no copy in used the log for
    /// [`IDLE_WATCH_NANOS`], its hold let go of with it, until its next use watches again.
    watch: Mutex<Option<Box<dyn Send + Sync>>>,
    /// The host nanoseconds of its last use, a gated copy in recorded ([`DirtyShadows::log`]).
    last_used: AtomicU64,
    tally: Arc<DirtyTally>,
    /// What its copies moved since the census last printed it (`METAL2VULKAN_LOG_DIRTY_CENSUS`).
    census: Census,
    /// The stamp as of which the image holds every texel of the truth, but for the pages the log
    /// reports changed since (the module docs); `None` where it holds nothing a copy may count on.
    hold: Mutex<Hold>,
    /// The submitted command buffers whose writes meet the truth or the shadow from a batch that
    /// may run after work submitted later, until each completes: while any does, the hold stays
    /// let go of and no copy in counts on it ([`DirtyShadows::planned`]).
    unsettled: AtomicU64,
    /// Whether it owns deferred rows (`deferred.rs`): its watch is kept while it does.
    owns_deferred: AtomicBool,
}

/// Another texture's home over the same host bytes as a shadow, through another buffer
/// (`texture/homes.rs`): a gated copy in whose shadow holds
/// nothing may take that home's image, held at a stamp, in place of the bytes, and copy from the
/// pages only the rows changed since ([`DirtyShadows::planned`], [`CopyInGate::regions`]).
/// `_keep` holds its image until the command buffer that copies from it goes.
pub(crate) struct Sibling {
    pub(crate) copies: Arc<ShadowCopies>,
    pub(crate) _keep: Arc<dyn Send + Sync>,
}

/// A copy in's base: its sibling's image ([`Sibling`]) held at `since`, its writer's own writes
/// in it, and its image and region for the copy.
#[derive(Clone, Copy)]
struct Base {
    since: DirtyStamp,
    writer: NonZeroU64,
    image: vk::Image,
}

/// What one shadow's gated copies in and copies back moved since the census last printed them,
/// and why each copy in that took every row did ([`DirtyShadows::log`],
/// `METAL2VULKAN_LOG_DIRTY_CENSUS`): a diagnostic, which nothing reads to decide.
#[derive(Default)]
struct Census {
    copies: AtomicU64,
    /// Recorded before its submission decided (a settle's replay).
    unplanned: AtomicU64,
    /// No hold to count on: let go of by a write, or never taken.
    no_hold: AtomicU64,
    /// Its command buffer wrote the truth or the shadow before the copy.
    written_before: AtomicU64,
    /// Its command buffer lies in more than one batch.
    split: AtomicU64,
    /// Another command buffer's writes keep the hold unsettled.
    unsettled: AtomicU64,
    /// Its stamp not published at its replay.
    unpublished: AtomicU64,
    /// The rows changed came to half or more, or to too many bands.
    most_rows: AtomicU64,
    skipped: AtomicU64,
    bytes_in: AtomicU64,
    bytes_back: AtomicU64,
    /// Took a sibling's image as its base.
    from_sibling: AtomicU64,
}

/// Why a gated copy in took every row ([`Census`]).
#[derive(Clone, Copy)]
enum Whole {
    Unplanned,
    NoHold,
    WrittenBefore,
    Split,
    Unsettled,
    Unpublished,
    MostRows,
}

impl Census {
    fn whole(&self, why: Whole) {
        let count = match why {
            Whole::Unplanned => &self.unplanned,
            Whole::NoHold => &self.no_hold,
            Whole::WrittenBefore => &self.written_before,
            Whole::Split => &self.split,
            Whole::Unsettled => &self.unsettled,
            Whole::Unpublished => &self.unpublished,
            Whole::MostRows => &self.most_rows,
        };
        count.fetch_add(1, Ordering::Relaxed);
    }

    /// Its counts, each set back to 0.
    fn take(&self) -> [u64; 12] {
        [
            &self.copies,
            &self.unplanned,
            &self.no_hold,
            &self.unpublished,
            &self.most_rows,
            &self.skipped,
            &self.bytes_in,
            &self.bytes_back,
            &self.written_before,
            &self.split,
            &self.unsettled,
            &self.from_sibling,
        ]
        .map(|count| count.swap(0, Ordering::Relaxed))
    }
}

/// Whether the census of the logged shadows' copies prints (`METAL2VULKAN_LOG_DIRTY_CENSUS`).
pub(crate) fn census_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(metal2vulkan::env_vars::log_dirty_census)
}

/// How often the census prints.
const CENSUS_NANOS: u64 = 1_000_000_000;

/// The census lines of `logs`, the most bytes copied in first, eight at most, and a total.
fn census_lines(logs: &[Arc<ShadowLog>], seconds: f64) -> Vec<String> {
    let mib = |bytes: u64| bytes as f64 / f64::from(1 << 20) / seconds;
    let rate = |count: u64| (count as f64 / seconds).round();
    let mut rows: Vec<(String, [u64; 12])> = (logs.iter())
        .map(|log| (log.census_name(), log.census.take()))
        .filter(|(_, counts)| counts.iter().any(|&count| count != 0))
        .collect();
    rows.sort_by_key(|(_, counts)| std::cmp::Reverse(counts[6]));
    let mut total = [0u64; 12];
    for (_, counts) in &rows {
        for (sum, count) in total.iter_mut().zip(counts) {
            *sum += count;
        }
    }
    let line = |name: &str, c: &[u64; 12]| {
        format!(
            "m2v dirty census {name}: in {:.1} MiB/s, back {:.1} MiB/s; copies in {}/s: whole for \
             no hold {}, written before {}, split {}, unsettled {}, unpublished {}, most rows {}, \
             unplanned {}; from a sibling's image {}; skipped {}",
            mib(c[6]),
            mib(c[7]),
            rate(c[0]),
            rate(c[2]),
            rate(c[8]),
            rate(c[9]),
            rate(c[10]),
            rate(c[3]),
            rate(c[4]),
            rate(c[1]),
            rate(c[11]),
            rate(c[5]),
        )
    };
    let mut lines: Vec<String> = (rows.iter().take(8))
        .map(|(name, counts)| line(name, counts))
        .collect();
    lines.push(line(&format!("all {} shadows", rows.len()), &total));
    lines
}

/// How long a shadow's log goes without a gated copy in before it lets go of its watch.
const IDLE_WATCH_NANOS: u64 = 250_000_000;
/// How often at most the logs are looked over for watches to let go of.
const IDLE_SWEEP_NANOS: u64 = 50_000_000;

impl Drop for ShadowLog {
    fn drop(&mut self) {
        let watched = self.watch.get_mut().map_or(true, |watch| watch.is_some());
        if watched {
            (self.tally.watched_bytes).fetch_sub(self.bytes(), Ordering::Relaxed);
        }
    }
}

impl ShadowLog {
    /// Its shadow's shape for the census: texels, rows' bytes, its truth's bytes and host address,
    /// and whether a home is shared or watched.
    fn census_name(&self) -> String {
        let shape = self
            .copies
            .upgrade()
            .and_then(|copies| match &copies.source {
                ShadowSource::Bytes {
                    target: ShadowTarget::Image { regions, texel, .. },
                    ..
                } => {
                    let region = regions.first()?;
                    let texel = (*texel)?;
                    let extent = region.image_extent;
                    let rows = match region.buffer_row_length {
                        0 => u64::from(extent.width) * texel,
                        row => u64::from(row) * texel,
                    };
                    let shared = copies.shared.load(Ordering::Relaxed);
                    Some(format!(
                        "{}x{} rows {rows}{}",
                        extent.width,
                        extent.height,
                        if shared { " shared" } else { "" }
                    ))
                }
                _ => None,
            });
        let watched = lock(&self.watch).is_some();
        format!(
            "{} ({:.1} MB at {:#x}{})",
            shape.unwrap_or_else(|| "gone".to_owned()),
            self.bytes() as f64 / 1e6,
            self.source.pointer.wrapping_add(self.truth.start),
            if watched { "" } else { ", unwatched" }
        )
    }

    /// How many bytes it watches while it watches.
    fn bytes(&self) -> u64 {
        self.truth.end.saturating_sub(self.truth.start)
    }

    /// A gated copy in uses it at the host nanoseconds `now`: watched again, where it had let go of
    /// its watch.
    fn used(&self, now: u64) {
        self.last_used.store(now, Ordering::Relaxed);
        let mut watch = lock(&self.watch);
        if watch.is_none() {
            *watch = Some(self.source.hook.watch(self.truth.clone()));
            (self.tally.watched_bytes).fetch_add(self.bytes(), Ordering::Relaxed);
        }
    }

    /// Let go of its watch where no copy in used it for [`IDLE_WATCH_NANOS`] by `now`, and of
    /// its hold with it: the watch, for the caller to drop once it let go of the logs.
    fn idle_watch(&self, now: u64) -> Option<Box<dyn Send + Sync>> {
        let idle = now.saturating_sub(self.last_used.load(Ordering::Relaxed)) >= IDLE_WATCH_NANOS;
        if !idle || self.owns_deferred.load(Ordering::Relaxed) {
            return None;
        }
        let watch = lock(&self.watch).take()?;
        self.set_hold(None);
        let bytes = self.bytes();
        (self.tally.watched_bytes).fetch_sub(bytes, Ordering::Relaxed);
        (self.tally.watched_bytes_released).fetch_add(bytes, Ordering::Relaxed);
        Some(watch)
    }

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

/// A shadow's hold ([`ShadowLog`]), and how many times it was set, by which a raise or a release
/// of a command buffer's own setting is told from a later command buffer's.
#[derive(Default)]
struct Hold {
    stamp: Option<DirtyStamp>,
    generation: u64,
}

impl ShadowLog {
    /// The host addresses of its truth's bytes.
    pub(crate) fn host(&self) -> Range<u64> {
        let pointer = self.source.pointer;
        pointer.wrapping_add(self.truth.start)..pointer.wrapping_add(self.truth.end)
    }

    /// The hold as it stands.
    fn hold(&self) -> Option<DirtyStamp> {
        lock(&self.hold).stamp
    }

    /// Set the hold to `stamp`: its generation.
    fn set_hold(&self, stamp: Option<DirtyStamp>) -> u64 {
        let mut hold = lock(&self.hold);
        hold.generation += 1;
        hold.stamp = stamp;
        hold.generation
    }

    /// Raise the hold of `generation` to `stamp`, where nothing set it since: the generation then.
    fn raise_hold(&self, generation: u64, stamp: DirtyStamp) -> Option<u64> {
        let mut hold = lock(&self.hold);
        let raises = hold.generation == generation && hold.stamp.is_none_or(|held| held < stamp);
        if raises {
            hold.generation += 1;
            hold.stamp = Some(stamp);
        }
        raises.then_some(hold.generation)
    }

    /// Let go of the hold of `generation`, where nothing set it since.
    fn release_hold(&self, generation: u64) {
        let mut hold = lock(&self.hold);
        if hold.generation == generation {
            hold.generation += 1;
            hold.stamp = None;
        }
    }
}

/// Print the census's lines, on stderr.
fn print_census(lines: &[String]) {
    for line in lines {
        eprintln!("{line}");
    }
}

/// The logs of a device's shadows over logged host pages, by the shadow's key: made at a shadow's
/// first copy in over such pages, and let go of once the shadow went.
#[derive(Default)]
pub(crate) struct DirtyShadows {
    logs: Mutex<Vec<Arc<ShadowLog>>>,
    /// The deferred rows of homes over pages whose source writes back on request.
    deferred: Arc<deferred::Deferred>,
    /// Whether any log was ever made: a command buffer collects its writes only then.
    any: AtomicBool,
    tally: Arc<DirtyTally>,
    /// The host nanoseconds the logs were last looked over for idle watches.
    swept: AtomicU64,
    /// The host nanoseconds the census last printed.
    census_at: AtomicU64,
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
    pub(crate) unanswered_after_holds: AtomicU64,
    pub(crate) watched_bytes: AtomicU64,
    pub(crate) watched_bytes_released: AtomicU64,
    pub(crate) from_siblings: AtomicU64,
    pub(crate) deferred_rows: AtomicU64,
    pub(crate) flushes: AtomicU64,
    pub(crate) write_back_bytes: AtomicU64,
    pub(crate) lost_rows: AtomicU64,
    pub(crate) unordered: AtomicU64,
    pub(crate) flushes_unfinished: AtomicU64,
    pub(crate) deferrals_refused: AtomicU64,
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
    /// begun (again, where it let go of it). A kept log of another source or `VkBuffer` (a
    /// rebind gave the buffer other pages) is made again. At most every [`IDLE_SWEEP_NANOS`], the
    /// logs no copy in used for [`IDLE_WATCH_NANOS`] let go of their watches.
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
        let now = gpu_time::host_nanos();
        let mut logs = lock(&self.logs);
        logs.retain(|log| log.copies.strong_count() > 0);
        // Dropped once the logs are let go of: a watch's drop is the consumer's code.
        let mut idle: Vec<Box<dyn Send + Sync>> = Vec::new();
        let mut census = Vec::new();
        if now.saturating_sub(self.swept.load(Ordering::Relaxed)) >= IDLE_SWEEP_NANOS {
            self.swept.store(now, Ordering::Relaxed);
            idle.extend(logs.iter().filter_map(|log| log.idle_watch(now)));
            let last = self.census_at.load(Ordering::Relaxed);
            if census_on() && now.saturating_sub(last) >= CENSUS_NANOS {
                self.census_at.store(now, Ordering::Relaxed);
                let seconds = match last {
                    0 => 1.0,
                    last => now.saturating_sub(last) as f64 / 1e9,
                };
                census = census_lines(&logs, seconds);
            }
        }
        let kept = logs.iter().position(|log| {
            std::ptr::eq(log.copies.as_ptr(), Arc::as_ptr(copies))
                && Arc::ptr_eq(&log.source, source)
                && log.raw == raw
                && log.truth == offsets
        });
        if let Some(at) = kept {
            let log = Arc::clone(&logs[at]);
            drop(logs);
            drop(idle);
            print_census(&census);
            log.used(now);
            return Some(log);
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
            watch: Mutex::new(Some(source.hook.watch(offsets.clone()))),
            last_used: AtomicU64::new(now),
            owns_deferred: AtomicBool::new(false),
            tally: Arc::clone(&self.tally),
            census: Census::default(),
            hold: Mutex::new(Hold::default()),
            unsettled: AtomicU64::new(0),
        });
        let bytes = offsets.end.saturating_sub(offsets.start);
        (self.tally.watched_bytes).fetch_add(bytes, Ordering::Relaxed);
        logs.push(Arc::clone(&log));
        self.any.store(true, Ordering::Release);
        drop(logs);
        drop(idle);
        print_census(&census);
        Some(log)
    }

    /// A command buffer whose commit was `commit` completed, `failed` or not: its shadows' copies
    /// back and merges are reported again ([`DirtyPages::wrote_as`]), now they ran, so a stamp
    /// asked between the commit and the GPU's write still finds them; the holds its writes
    /// kept unsettled may be taken again; where it failed, the holds its copies in set are let go
    /// of, its copies having perhaps not run.
    pub(crate) fn completed(&self, commit: &mut DirtyCommit, failed: bool) {
        if let Some(reach) = &commit.reach {
            reach.done();
        }
        for (log, range) in &commit.wrote {
            log.wrote_back(range.clone());
        }
        // A source that answers at once is asked again whether the stamps its gated copies in
        // synced to are published, by now most often: what it answers counts for later holds
        // alone (`planned`), each a stamp published before a later copy in runs. Without it a
        // shadow whose command buffers go as they commit, asking only before a harvest could
        // publish, would never find a stamp published, nor take a hold.
        for gate in &commit.gates {
            let source = &gate.log.source;
            if let (Some(stamp), Some(_)) = (*lock(&gate.stamp), source.hook.publication_hold()) {
                source.published_now(stamp);
            }
        }
        for log in commit.unsettling.drain(..) {
            log.unsettled.fetch_sub(1, Ordering::AcqRel);
        }
        if !failed {
            return;
        }
        for gate in &commit.gates {
            if let Some(generation) = *lock(&gate.held_at) {
                gate.log.release_hold(generation);
            }
        }
        for (log, generation) in &commit.held_at {
            log.release_hold(*generation);
        }
    }

    /// `commit`'s gates, taken out as its command buffer completed: their siblings' images and
    /// buffers (`Sibling`), and the copies back's homes' (`deferred.rs`), which the command
    /// buffer's object would hold as long as the program keeps it. The caller drops them once it
    /// let go of the record's lock, as a gate may hold a buffer's last handle, whose going runs
    /// its deallocator.
    pub(crate) fn let_go(
        &self,
        commit: &mut DirtyCommit,
    ) -> (Vec<Arc<CopyInGate>>, Vec<Arc<BackGate>>) {
        (
            std::mem::take(&mut commit.gates),
            std::mem::take(&mut commit.backs),
        )
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
    /// for ([`planned`](Self::planned) decides what they copy); the logs of the shadows its end
    /// left holding every texel of their truth (`held`) with no gated copy in (a pass that
    /// cleared every texel), whose holds its submission takes, asking no stamp; and its copies
    /// back and merges, reported ([`DirtyPages::wrote_as`]) before any stamp a later commit asks.
    pub(crate) fn stamped(&self, commit: &mut DirtyCommit, held: &[Arc<ShadowCopies>]) {
        for (log, range) in &commit.wrote {
            log.wrote_back(range.clone());
            let bytes = range.end.saturating_sub(range.start);
            log.census.bytes_back.fetch_add(bytes, Ordering::Relaxed);
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
                commit.held_logs.push(log);
            }
        }
    }

    /// The base a gated copy in whose shadow holds nothing takes, as its command buffer is
    /// submitted: its sibling's image ([`Sibling`]) where the sibling's log holds a stamp, no
    /// write keeps it unsettled, its source shares this one's log, and both shadows' texels lie
    /// at the same host addresses, laid out alike, in the `VkBuffer`s their logs were made over
    /// (neither buffer rebound since). The tracker orders the copy after every write of the sibling's image
    /// submitted before it and before every one after (the copy's read of it,
    /// `EncoderCore::shadow_copies`), so the image holds the bytes as of that stamp but for the
    /// pages other writers changed since, which the copy takes from the pages.
    fn base(&self, gate: &CopyInGate, owner: Option<&Arc<ShadowLog>>) -> Option<Base> {
        let sibling = gate.sibling.as_ref()?;
        let log = self.kept(&sibling.copies)?;
        // Over deferred rows, only their owner's image.
        if owner.is_some_and(|owner| !Arc::ptr_eq(owner, &log)) {
            return None;
        }
        // Its stamp and writer are asked of this log's source, of the same host bytes.
        let alike = (gate.log.source).shares_log(&log.source) && gate.log.host() == log.host();
        if !alike || log.unsettled.load(Ordering::Acquire) > 0 {
            return None;
        }
        let since = log.hold()?;
        let mine = gate.log.copies.upgrade()?;
        let (my_raw, my_layout) = crate::texture::shadow_bytes(&mine)?;
        let (their_raw, their_layout) = crate::texture::shadow_bytes(&sibling.copies)?;
        // Each buffer's bytes lie where its log was made (not rebound since), laid out alike.
        let same = my_raw == gate.log.raw && their_raw == log.raw && my_layout == their_layout;
        let ShadowSource::Bytes {
            target: ShadowTarget::Image { image, .. },
            ..
        } = &sibling.copies.source
        else {
            return None;
        };
        same.then_some(Base {
            since,
            writer: log.writer,
            image: *image,
        })
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
                    && gate.log.hold().is_some();
                let awaits = counts && !source.published_now(stamp);
                if awaits {
                    gate.held.store(true, Ordering::Relaxed);
                }
                awaits.then_some(bound)
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
        // A log in use until its command buffers are submitted: one encoded long before keeps
        // its watch while it waits (`IDLE_WATCH_NANOS`).
        let now = gpu_time::host_nanos();
        // Its copies in and copies back in its order: a copy back decides what the copies in
        // after it find deferred.
        let mut backs = commit.backs.iter().peekable();
        for gate in &commit.gates {
            while let Some(back) = backs.next_if(|back| back.seq < gate.seq) {
                self.plan_back(back, commit.reach.as_ref());
            }
            gate.log.last_used.fetch_max(now, Ordering::Relaxed);
            let Some(stamp) = *lock(&gate.stamp) else {
                continue;
            };
            let unsettled = gate.log.unsettled.load(Ordering::Acquire) > 0;
            let (since, unheld) = match (gate.written_before, split, unsettled) {
                (true, _, _) => (None, Whole::WrittenBefore),
                (_, true, _) => (None, Whole::Split),
                (_, _, true) => (None, Whole::Unsettled),
                _ => (gate.log.hold(), Whole::NoHold),
            };
            // Deferred rows over its bytes another home owns come from that home's image, never
            // from the pages: its own hold counts for nothing of them (`deferred.rs`).
            let skip = self.deferred.skip(&gate.log);
            let foreign = skip
                .as_ref()
                .filter(|skip| !Arc::ptr_eq(&skip.owner, &gate.log));
            let since = since.filter(|_| foreign.is_none());
            let base = match (since, unheld) {
                (None, Whole::NoHold) => self.base(gate, foreign.map(|skip| &skip.owner)),
                _ => None,
            };
            let skip = match (foreign, &base) {
                (Some(_), None) => {
                    self.tally.unordered.fetch_add(1, Ordering::Relaxed);
                    None
                }
                _ => skip,
            };
            *lock(&gate.decision) = Some(Decision {
                since,
                stamp,
                unheld,
                base,
                skip,
            });
        }
        for back in backs {
            self.plan_back(back, commit.reach.as_ref());
        }
        let logs: Vec<Arc<ShadowLog>> = lock(&self.logs).clone();
        for log in &logs {
            let Some(copies) = log.copies.upgrade() else {
                continue;
            };
            // A flush writes its owner's image into the pages: its hold stands (`deferred.rs`).
            if !commit.wrote_over(&copies) || commit.flushes(&copies) {
                continue;
            }
            log.set_hold(None);
            if split {
                log.unsettled.fetch_add(1, Ordering::AcqRel);
                commit.unsettling.push(Arc::clone(log));
            }
        }
        if split {
            return;
        }
        // A shadow its end held whole holds from the latest stamp its source is known to have
        // published by now, which every copy of it ran after, and at most from its gate's own:
        // a guest write made after a stamp's request but before its harvest began is stamped
        // with it, so a copy may sync to a stamp only once it read the bytes after that stamp's
        // publication. Its replay raises the hold to its gate's stamp once it finds that published
        // ([`CopyInGate::regions`]). A source that answers at once is asked now.
        for gate in &commit.gates {
            let log = &gate.log;
            let holds = held.iter().any(|copies| gate.of(copies));
            let Some(stamp) = *lock(&gate.stamp) else {
                continue;
            };
            if !holds || log.unsettled.load(Ordering::Acquire) > 0 {
                continue;
            }
            if log.source.hook.publication_hold().is_some() {
                log.source.published_now(stamp);
            }
            let from = log.source.latest_published().map(|known| known.min(stamp));
            *lock(&gate.held_at) = Some(log.set_hold(from));
        }
        let DirtyCommit {
            held_logs, held_at, ..
        } = commit;
        for log in held_logs.drain(..) {
            if log.unsettled.load(Ordering::Acquire) == 0 {
                let generation = log.set_hold(log.source.latest_published());
                held_at.push((log, generation));
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
    /// Its gated copies back (`deferred.rs`), and the order of every gate so far.
    backs: Vec<Arc<BackGate>>,
    seq: u32,
    /// The host bytes its commands reach another way than through a home (`deferred.rs`).
    reach: Option<Arc<deferred::Reach>>,
    wrote: Vec<(Arc<ShadowLog>, Range<u64>)>,
    unsettling: Vec<Arc<ShadowLog>>,
    /// The logs of the shadows its end held whole with no gated copy in ([`DirtyShadows::stamped`]).
    held_logs: Vec<Arc<ShadowLog>>,
    /// Their holds as its submission set them, which its failure lets go of.
    held_at: Vec<(Arc<ShadowLog>, u64)>,
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
        (log, copies, texel, sibling): (Arc<ShadowLog>, &ShadowCopies, u64, Option<Sibling>),
        shadows: &DirtyShadows,
    ) -> Arc<CopyInGate> {
        let gate = Arc::new(CopyInGate {
            _counted: crate::live::Counted::new(crate::live::Kind::CopyInGate),
            seq: self.next_seq(),
            deferred: Arc::clone(&shadows.deferred),
            emptied: Mutex::new(Vec::new()),
            log,
            texel,
            tally: Arc::clone(&shadows.tally),
            written_before: self.wrote_over(copies),
            stamp: Mutex::new(None),
            decision: Mutex::new(None),
            held_at: Mutex::new(None),
            held: AtomicBool::new(false),
            sibling,
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
        !self.gates.is_empty() || !self.backs.is_empty()
    }

    /// The next gate's place in the command buffer's order.
    fn next_seq(&mut self) -> u32 {
        self.seq += 1;
        self.seq
    }
}

/// A shadow's copy in recorded with what its submission decides ([`DirtyShadows::planned`]): the
/// stamp its log's hold stood at, where it held one, and the stamp it syncs to.
#[derive(Clone)]
struct Decision {
    since: Option<DirtyStamp>,
    stamp: DirtyStamp,
    /// Why `since` is `None` (the census's).
    unheld: Whole,
    /// Where `since` is `None` for want of a hold, a sibling's image held at a stamp
    /// ([`Sibling`]).
    base: Option<Base>,
    /// The deferred rows over its bytes, which it takes not from the pages (`deferred.rs`).
    skip: Option<deferred::Skip>,
}

/// A copy in of a shadow over logged host pages, its regions decided as its command buffer's
/// commands are recorded for the GPU (`recorder`): every region where its submission found no
/// hold, where it is recorded before its submission decided, or where its log cannot say, else
/// the rows of the pages changed since the hold.
pub(crate) struct CopyInGate {
    _counted: crate::live::Counted,
    /// Its place in its command buffer's order.
    seq: u32,
    /// The device's deferred rows (`deferred.rs`).
    deferred: Arc<deferred::Deferred>,
    /// The group of deferred rows its CPU's rows emptied, which holds its owner's image and
    /// buffer: dropped with the gate, past the locks its replay ran under.
    emptied: Mutex<Vec<deferred::Group>>,
    log: Arc<ShadowLog>,
    /// The bytes of one texel in the buffer.
    texel: u64,
    /// The command buffer wrote the truth or the shadow before the copy.
    written_before: bool,
    tally: Arc<DirtyTally>,
    /// The stamp its commit asked of the log's source ([`DirtyShadows::stamped`]).
    stamp: Mutex<Option<DirtyStamp>>,
    decision: Mutex<Option<Decision>>,
    /// The generation of the log's hold its submission set, or its replay raised, which its
    /// failure lets go of ([`DirtyShadows::planned`]).
    held_at: Mutex<Option<u64>>,
    /// The scheduler held its command buffer for its stamp's publication
    /// ([`DirtyShadows::awaits_publication`]), by which an answer `false` at its replay is counted
    /// apart (`WorkCounts::dirty_waits_unanswered_after_holds`).
    held: AtomicBool,
    /// Another texture's home over the same bytes, found as the copy was recorded: the base its
    /// submission may take ([`DirtyShadows::planned`]).
    sibling: Option<Sibling>,
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
    ) -> Option<(vk::Image, vk::ImageCopy)> {
        let (total, copied, base) = self.chosen(whole, out);
        let from = base.and_then(|image| {
            let region = whole.first()?;
            let subresource = vk::ImageSubresourceLayers {
                aspect_mask: region.image_subresource.aspect_mask,
                mip_level: region.image_subresource.mip_level,
                base_array_layer: region.image_subresource.base_array_layer,
                layer_count: region.image_subresource.layer_count,
            };
            let copy = vk::ImageCopy {
                src_subresource: subresource,
                src_offset: region.image_offset,
                dst_subresource: subresource,
                dst_offset: region.image_offset,
                extent: region.image_extent,
            };
            self.tally.from_siblings.fetch_add(1, Ordering::Relaxed);
            Some((image, copy))
        });
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
        from
    }

    /// [`regions`](Self::regions), uncounted: the bytes `whole` and the chosen regions move, and
    /// the sibling's image the copy takes first where it takes one.
    fn chosen(
        &self,
        whole: &[vk::BufferImageCopy],
        out: &mut Vec<vk::BufferImageCopy>,
    ) -> (u64, u64, Option<vk::Image>) {
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
        let census = &self.log.census;
        census.copies.fetch_add(1, Ordering::Relaxed);
        // Every row, but deferred ones where any are (`deferred.rs`), after the base where one is.
        let every_row = |out: &mut Vec<vk::BufferImageCopy>,
                         why: Whole,
                         skip: Option<&[Range<u32>]>,
                         taken: Option<vk::Image>| {
            census.whole(why);
            out.clear();
            match (skip, whole) {
                (Some(skip), [region]) if !skip.is_empty() || taken.is_some() => {
                    let top = u32::try_from(region.image_offset.y).unwrap_or(0);
                    let mut rows: Vec<Range<u32>> =
                        std::iter::once(top..top + region.image_extent.height).collect();
                    for cut in skip {
                        deferred::subtract(&mut rows, cut);
                    }
                    let rows: Vec<Range<u32>> = (rows.iter())
                        .map(|row| row.start - top..row.end - top)
                        .collect();
                    deferred::row_regions(region, &rows, self.texel, out);
                    let copied = bytes(out);
                    census.bytes_in.fetch_add(copied, Ordering::Relaxed);
                    (total, copied, taken)
                }
                _ => {
                    out.extend_from_slice(whole);
                    census.bytes_in.fetch_add(total, Ordering::Relaxed);
                    (total, total, None)
                }
            }
        };
        let Some(decision) = lock(&self.decision).clone() else {
            // Its own deferred rows stay in its image.
            let own = (self.deferred.skip(&self.log))
                .filter(|skip| Arc::ptr_eq(&skip.owner, &self.log))
                .map(|skip| skip.rows);
            return every_row(out, Whole::Unplanned, own.as_deref(), None);
        };
        let Decision {
            since,
            stamp,
            unheld,
            base,
            skip,
        } = decision;
        let skip_rows = skip.as_ref().map(|skip| skip.rows.clone());
        // Asked whatever the hold, so the copy, which runs after this, may raise the hold its
        // submission set to its own stamp ([`DirtyShadows::planned`]).
        let published = self.log.source.published(stamp, &self.tally);
        if published {
            let mut held_at = lock(&self.held_at);
            if let Some(raised) = (*held_at).and_then(|at| self.log.raise_hold(at, stamp)) {
                *held_at = Some(raised);
            }
        }
        if !published && self.held.load(Ordering::Relaxed) {
            (self.tally.unanswered_after_holds).fetch_add(1, Ordering::Relaxed);
        }
        // Its own hold, else the sibling's image as of its stamp, but for what other writers
        // changed since.
        let (since, writer, taken) = match (since, base) {
            (Some(since), _) => (since, self.log.writer, None),
            (None, Some(base)) => (base.since, base.writer, Some(base.image)),
            (None, None) => return every_row(out, unheld, skip_rows.as_deref(), None),
        };
        if !published {
            return every_row(out, Whole::Unpublished, skip_rows.as_deref(), taken);
        }
        let mut runs = Vec::new();
        (self.log.source.hook).changed_except(since, writer, &mut runs);
        // Rows the CPU wrote since that a home deferred: the CPU's win them (`deferred.rs`).
        let skip_rows = match (&skip, skip_rows) {
            (Some(skip), Some(mut rows)) => {
                let changed = self.run_rows(whole, &runs);
                let won: Vec<Range<u32>> = (changed.iter())
                    .flat_map(|c| {
                        (rows.iter())
                            .map(move |r| c.start.max(r.start)..c.end.min(r.end))
                            .filter(|w| w.start < w.end)
                    })
                    .collect();
                if !won.is_empty() {
                    let (lost, emptied) = self.deferred.won(&skip.owner, &won, skip.epoch);
                    lock(&self.emptied).extend(emptied);
                    self.tally.lost_rows.fetch_add(lost, Ordering::Relaxed);
                    for row in &won {
                        deferred::subtract(&mut rows, row);
                    }
                }
                Some(rows)
            }
            _ => None,
        };
        let banded = whole.iter().all(|region| self.bands(region, &runs, out));
        if !banded || out.len() > REGIONS {
            return every_row(out, Whole::MostRows, skip_rows.as_deref(), taken);
        }
        let copied = bytes(out);
        census.bytes_in.fetch_add(copied, Ordering::Relaxed);
        match (taken, out.is_empty()) {
            (Some(_), _) => census.from_sibling.fetch_add(1, Ordering::Relaxed),
            (None, true) => census.skipped.fetch_add(1, Ordering::Relaxed),
            (None, false) => 0,
        };
        (total, copied, taken)
    }

    /// The image's rows of `whole` that the source's changed `runs` meet.
    fn run_rows(&self, whole: &[vk::BufferImageCopy], runs: &[Range<u64>]) -> Vec<Range<u32>> {
        let mut rows = Vec::new();
        for region in whole {
            let extent = region.image_extent;
            let row_texels = match region.buffer_row_length {
                0 => extent.width,
                row => row,
            };
            let pitch = u64::from(row_texels) * self.texel;
            if pitch == 0 {
                continue;
            }
            let first = region.buffer_offset.wrapping_add(self.log.delta);
            let height = u64::from(extent.height);
            let end = first + height * pitch;
            let top = u32::try_from(region.image_offset.y).unwrap_or(0);
            for run in runs {
                let (start, stop) = (run.start.max(first), run.end.min(end));
                if start >= stop {
                    continue;
                }
                let first_row = ((start - first) / pitch) as u32;
                let last_row = ((stop - first).div_ceil(pitch)).min(height) as u32;
                rows.push(top + first_row..top + last_row);
            }
        }
        rows
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

#[cfg(test)]
impl CopyInGate {
    /// A gate of no shadow over a source that logs nothing, for tests of what holds gates.
    pub(crate) fn for_test() -> Arc<CopyInGate> {
        struct Nothing;
        impl DirtyPages for Nothing {
            type Watch = ();
            fn watch(&self, _range: Range<u64>) {}
            fn request(&self) -> DirtyStamp {
                DirtyStamp(0)
            }
            fn wait(&self, _stamp: DirtyStamp) -> bool {
                true
            }
            fn changed(&self, _since: DirtyStamp, _out: &mut Vec<Range<u64>>) {}
            fn wrote(&self, _range: Range<u64>) {}
        }
        let source = DirtySource::new(hook(Arc::new(Nothing)), 0);
        let tally = Arc::new(DirtyTally::default());
        let log = Arc::new(ShadowLog {
            copies: Weak::new(),
            source,
            writer: NonZeroU64::MIN,
            raw: vk::Buffer::null(),
            delta: 0,
            truth: 0..0,
            watch: Mutex::new(None),
            last_used: AtomicU64::new(0),
            tally: Arc::clone(&tally),
            census: Census::default(),
            hold: Mutex::new(Hold::default()),
            unsettled: AtomicU64::new(0),
            owns_deferred: AtomicBool::new(false),
        });
        Arc::new(CopyInGate {
            _counted: crate::live::Counted::new(crate::live::Kind::CopyInGate),
            seq: 0,
            deferred: Arc::default(),
            emptied: Mutex::new(Vec::new()),
            log,
            texel: 4,
            written_before: false,
            tally,
            stamp: Mutex::new(None),
            decision: Mutex::new(None),
            held_at: Mutex::new(None),
            held: AtomicBool::new(false),
            sibling: None,
        })
    }
}
