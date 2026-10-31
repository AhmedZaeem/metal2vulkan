//! Deferred write-back ([`WriteBack::OnRequest`]): where a source asks for it, a home's writes
//! stay in its image, newer than the host pages, until a flush writes them back
//! (`Device::write_back_deferred`).
//!
//! - **A gated copy back.** At the end of an encoder that wrote a home over such pages, its copy
//!   back is recorded as ever, with the uses an eager one declares, so the tracker orders it, and
//!   the holds it meets let go, as before. Its submission decides, in planning order
//!   ([`DirtyShadows::planned`]), what it copies ([`BackGate`]): nothing, its rows then deferred,
//!   where no other home over the same bytes owns deferred rows; else every row, as before. A
//!   flush is the same copy, of every deferred row of its group.
//! - **One owner a surface.** The deferred rows of the shadows over the same host bytes, laid out
//!   alike, are one group's: one home owns them, whose image holds the newest bytes there. A
//!   write of the bytes by another home is copied back eagerly, and takes its rows out of the
//!   group, newer than the owner's.
//! - **Never from the pages.** A gated copy in over a group's bytes copies no deferred row from
//!   the pages: the owner's copy in leaves them in its image, and another home's copies in from
//!   the owner's image (`Sibling`) and then the rest. A copy in that would take every row takes
//!   every row but those. Rows the log reports changed since the hold, unlocked CPU writes, win
//!   over the GPU's: copied in, and out of the group
//!   (`WorkCounts::deferred_rows_lost_to_unlocked_cpu_writes`).
//! - **Held until written.** A group holds its owner's image and buffer until its rows are
//!   written back, as work in flight holds what it uses, and its log's watch.

use super::{lock, DirtyShadows, DirtyTally, ShadowLog};
use crate::shadow::ShadowCopies;
use crate::texture::{shadow_bytes, ShadowLayout};
use ash::vk;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// The rows a group's owner holds newer than the pages, and what holds them.
pub(super) struct Group {
    _counted: crate::live::Counted,
    /// The host address of the first texel, and the layout ([`shadow_bytes`]).
    host: u64,
    /// The host addresses of the owner's truth.
    bytes: Range<u64>,
    layout: ShadowLayout,
    owner: Arc<ShadowLog>,
    /// The owner's shadow, image and buffer, held until the rows are written back.
    copies: Arc<ShadowCopies>,
    keep: Arc<dyn Send + Sync>,
    /// Disjoint, in order, each with the epoch of the deferral that last wrote it.
    rows: Vec<(Range<u32>, u64)>,
    /// The epoch of its latest deferral.
    epoch: u64,
    /// An unwaited flush of it, its owner orphaned, was committed (`Device::flush_orphans`).
    flushing: bool,
}

/// The device's groups of deferred rows ([`DirtyShadows`]), and the host bytes the command
/// buffers in flight reach another way than through a home.
#[derive(Default)]
pub(super) struct Deferred {
    groups: Mutex<Vec<Group>>,
    reaches: Mutex<Vec<Weak<Reach>>>,
    /// Whether a buffer over pages whose source writes back on request was made.
    seen: AtomicBool,
}

/// The host bytes a command buffer's commands reach another way than through a home (bound,
/// copied, or in its resident set), from its first such command until it completes: a deferral
/// planned meanwhile is made at once instead (the module's).
#[derive(Default)]
pub(crate) struct Reach {
    ranges: Mutex<Vec<Range<u64>>>,
    done: AtomicBool,
}

impl Reach {
    fn meets(&self, bytes: &Range<u64>) -> bool {
        let ranges = lock(&self.ranges);
        ranges
            .iter()
            .any(|r| r.start < bytes.end && bytes.start < r.end)
    }

    /// Its command buffer completed (or went uncommitted).
    pub(super) fn done(&self) {
        self.done.store(true, Ordering::Relaxed);
    }
}

/// What a gated copy back is ([`BackGate`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BackKind {
    /// The copy back at the end of an encoder that wrote the home.
    End,
    /// A flush of every deferred row of the home's group.
    Flush,
}

/// A copy back of a home over pages whose source writes back on request, its rows decided as
/// its command buffer is submitted (the module's).
pub(crate) struct BackGate {
    _counted: crate::live::Counted,
    pub(super) seq: u32,
    log: Arc<ShadowLog>,
    copies: Arc<ShadowCopies>,
    kind: BackKind,
    /// The home's image and buffer, which a group it starts holds.
    keep: Arc<dyn Send + Sync>,
    /// The rows of the home it copies, set as it records.
    rows: Mutex<Option<Range<u32>>>,
    /// The rows it copies, decided at its submission; none where its rows were deferred.
    decision: Mutex<Option<Vec<Range<u32>>>>,
    tally: Arc<DirtyTally>,
    /// The groups its submission took out, which hold their owners' images and buffers: dropped
    /// with it, outside the locks its submission ran under (a buffer's last handle going runs
    /// its deallocator).
    held: Mutex<Vec<Group>>,
}

/// A copy in's deferred rows ([`Deferred::skip`]): the group's owner, its rows then, and its
/// epoch then, past which a later deferral's rows are newer than what the copy in reads.
#[derive(Clone)]
pub(super) struct Skip {
    pub(super) owner: Arc<ShadowLog>,
    pub(super) rows: Vec<Range<u32>>,
    pub(super) epoch: u64,
}

/// Add `add` to the disjoint ordered `rows`, deferred at `epoch`.
fn insert(rows: &mut Vec<(Range<u32>, u64)>, add: Range<u32>, epoch: u64) {
    if add.is_empty() {
        return;
    }
    cut_rows(rows, &add, u64::MAX);
    rows.push((add, epoch));
    rows.sort_unstable_by_key(|(row, _)| row.start);
}

/// Take `cut` out of the disjoint ordered `rows`, of those deferred at `upto` or before: the
/// rows taken.
fn cut_rows(rows: &mut Vec<(Range<u32>, u64)>, cut: &Range<u32>, upto: u64) -> u64 {
    let mut taken = 0;
    let mut kept = Vec::with_capacity(rows.len() + 1);
    for (row, epoch) in rows.drain(..) {
        let (start, end) = (row.start.max(cut.start), row.end.min(cut.end));
        if start >= end || epoch > upto {
            kept.push((row, epoch));
            continue;
        }
        taken += u64::from(end - start);
        if row.start < start {
            kept.push((row.start..start, epoch));
        }
        if end < row.end {
            kept.push((end..row.end, epoch));
        }
    }
    *rows = kept;
    taken
}

/// The ranges of `rows`.
fn ranges(rows: &[(Range<u32>, u64)]) -> Vec<Range<u32>> {
    rows.iter().map(|(row, _)| row.clone()).collect()
}

/// Take `cut` out of the disjoint ordered `rows`: the rows taken.
pub(super) fn subtract(rows: &mut Vec<Range<u32>>, cut: &Range<u32>) -> u64 {
    let mut taken = 0;
    let mut kept = Vec::with_capacity(rows.len() + 1);
    for row in rows.drain(..) {
        let (start, end) = (row.start.max(cut.start), row.end.min(cut.end));
        if start >= end {
            kept.push(row);
            continue;
        }
        taken += u64::from(end - start);
        if row.start < start {
            kept.push(row.start..start);
        }
        if end < row.end {
            kept.push(end..row.end);
        }
    }
    *rows = kept;
    taken
}

/// The bands of `region` that copy `rows` of it (counted from its first row).
pub(super) fn row_regions(
    region: &vk::BufferImageCopy,
    rows: &[Range<u32>],
    texel: u64,
    out: &mut Vec<vk::BufferImageCopy>,
) {
    let extent = region.image_extent;
    let row_texels = match region.buffer_row_length {
        0 => extent.width,
        row => row,
    };
    let pitch = u64::from(row_texels) * texel;
    for row in rows {
        let (top, bottom) = (row.start, row.end.min(extent.height));
        if top >= bottom {
            continue;
        }
        out.push(vk::BufferImageCopy {
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
        });
    }
}

/// What a group is known by: its log's host address and its shadow's layout.
fn key_of(log: &ShadowLog, copies: &ShadowCopies) -> Option<(u64, ShadowLayout)> {
    Some((log.host().start, shadow_bytes(copies)?.1))
}

impl Deferred {
    /// The deferred rows over `log`'s bytes, where any are, and their owner.
    pub(super) fn skip(&self, log: &ShadowLog) -> Option<Skip> {
        let copies = log.copies.upgrade()?;
        let (host, layout) = key_of(log, &copies)?;
        let groups = lock(&self.groups);
        let group = groups
            .iter()
            .find(|group| group.host == host && group.layout == layout)?;
        Some(Skip {
            owner: Arc::clone(&group.owner),
            rows: ranges(&group.rows),
            epoch: group.epoch,
        })
    }

    /// The owner of the deferred rows over the bytes `copies` lie over as `log`'s, other than
    /// `copies`, with what holds its image: the sibling a copy in of `copies` takes them from.
    pub(super) fn owner_of(
        &self,
        log: &ShadowLog,
        copies: &Arc<ShadowCopies>,
    ) -> Option<(Arc<ShadowCopies>, Arc<dyn Send + Sync>)> {
        let (host, layout) = key_of(log, copies)?;
        let groups = lock(&self.groups);
        let group = groups
            .iter()
            .find(|group| group.host == host && group.layout == layout)?;
        (!Arc::ptr_eq(&group.copies, copies))
            .then(|| (Arc::clone(&group.copies), Arc::clone(&group.keep)))
    }

    /// Rows a copy in found changed since its hold (`changed`) that `owner` deferred at `upto` or
    /// before, the copy in's submission's epoch: the CPU's, taken out of the group (the
    /// module's). A later deferral's rows stay: newer than the CPU's. How many.
    pub(super) fn won(
        &self,
        owner: &Arc<ShadowLog>,
        changed: &[Range<u32>],
        upto: u64,
    ) -> (u64, Option<Group>) {
        let mut groups = lock(&self.groups);
        let Some(at) = groups.iter().position(|g| Arc::ptr_eq(&g.owner, owner)) else {
            return (0, None);
        };
        let lost: u64 = changed
            .iter()
            .map(|row| cut_rows(&mut groups[at].rows, row, upto))
            .sum();
        // A group the CPU's rows emptied goes to the caller, to hold past the lock.
        let emptied = groups[at].rows.is_empty().then(|| groups.swap_remove(at));
        if let Some(group) = &emptied {
            group.owner.owns_deferred.store(false, Ordering::Relaxed);
        }
        (lost, emptied)
    }

    /// The owners whose groups no flush was committed for that `orphaned` says no texture holds
    /// any more: marked flushing.
    fn orphans(&self, orphaned: impl Fn(&Arc<ShadowCopies>) -> bool) -> Vec<Arc<ShadowCopies>> {
        let mut groups = lock(&self.groups);
        (groups.iter_mut())
            .filter(|g| !g.flushing && orphaned(&g.copies))
            .map(|g| {
                g.flushing = true;
                Arc::clone(&g.copies)
            })
            .collect()
    }

    /// The image and buffer the group of `copies` holds, which its flush holds in turn.
    fn keep_of(&self, copies: &Arc<ShadowCopies>) -> Option<Arc<dyn Send + Sync>> {
        let groups = lock(&self.groups);
        (groups.iter())
            .find(|g| Arc::ptr_eq(&g.copies, copies))
            .map(|g| Arc::clone(&g.keep))
    }

    /// The owners of every group over `host` (every group for `None`), for a flush.
    pub(super) fn owners(&self, host: Option<Range<u64>>) -> Vec<Arc<ShadowCopies>> {
        let groups = lock(&self.groups);
        let over = |group: &&Group| {
            (host.as_ref()).is_none_or(|h| h.start < group.bytes.end && group.bytes.start < h.end)
        };
        groups
            .iter()
            .filter(over)
            .map(|g| Arc::clone(&g.copies))
            .collect()
    }

    /// Whether a command buffer in flight other than `own`, or `own`, reaches `bytes` another way.
    fn reached(&self, bytes: &Range<u64>, own: Option<&Arc<Reach>>) -> bool {
        if own.is_some_and(|own| own.meets(bytes)) {
            return true;
        }
        let mut reaches = lock(&self.reaches);
        reaches.retain(|reach| {
            reach
                .upgrade()
                .is_some_and(|r| !r.done.load(Ordering::Relaxed))
        });
        reaches
            .iter()
            .filter_map(Weak::upgrade)
            .any(|reach| !own.is_some_and(|own| Arc::ptr_eq(own, &reach)) && reach.meets(bytes))
    }

    /// Decide `gate`'s rows as its command buffer is submitted (the module's), its command
    /// buffer's own reach `own`.
    fn plan(&self, gate: &BackGate, own: Option<&Arc<Reach>>) {
        let rows = lock(&gate.rows).clone().unwrap_or(0..0);
        // Work in flight that reaches the bytes another way, or its own command buffer's, reads
        // them as they are when it runs: the copy back is made at once.
        // The groups taken out go to the gate, to drop with it past the locks.
        let mut removed = Vec::new();
        if gate.kind == BackKind::End && self.reached(&gate.log.host(), own) {
            (gate.tally.deferrals_refused).fetch_add(1, Ordering::Relaxed);
            let mut groups = lock(&self.groups);
            if let Some((host, layout)) = key_of(&gate.log, &gate.copies) {
                let at = groups
                    .iter()
                    .position(|g| g.host == host && g.layout == layout);
                if let Some(at) = at {
                    cut_rows(&mut groups[at].rows, &rows, u64::MAX);
                    if groups[at].rows.is_empty() {
                        let group = groups.swap_remove(at);
                        group.owner.owns_deferred.store(false, Ordering::Relaxed);
                        removed.push(group);
                    }
                }
            }
            drop(groups);
            lock(&gate.held).extend(removed);
            *lock(&gate.decision) = Some(std::iter::once(rows).collect());
            return;
        }
        let key = key_of(&gate.log, &gate.copies);
        let mut groups = lock(&self.groups);
        let at = key.and_then(|(host, layout)| {
            (groups.iter()).position(|group| group.host == host && group.layout == layout)
        });
        let tally = &gate.tally;
        let copy = match (gate.kind, at, key) {
            (BackKind::End, None, Some((host, layout))) => {
                gate.log.owns_deferred.store(true, Ordering::Relaxed);
                let deferred = u64::from(rows.end - rows.start);
                tally.deferred_rows.fetch_add(deferred, Ordering::Relaxed);
                groups.push(Group {
                    _counted: crate::live::Counted::new(crate::live::Kind::Group),
                    host,
                    bytes: gate.log.host(),
                    layout,
                    owner: Arc::clone(&gate.log),
                    copies: Arc::clone(&gate.copies),
                    keep: Arc::clone(&gate.keep),
                    rows: std::iter::once((rows, 1)).collect(),
                    epoch: 1,
                    flushing: false,
                });
                Vec::new()
            }
            (BackKind::End, Some(at), _) if Arc::ptr_eq(&groups[at].owner, &gate.log) => {
                let deferred = u64::from(rows.end - rows.start);
                tally.deferred_rows.fetch_add(deferred, Ordering::Relaxed);
                let group = &mut groups[at];
                group.epoch += 1;
                insert(&mut group.rows, rows, group.epoch);
                Vec::new()
            }
            // Another home's write: eager, and newer than the owner's rows there.
            (BackKind::End, Some(at), _) => {
                cut_rows(&mut groups[at].rows, &rows, u64::MAX);
                if groups[at].rows.is_empty() {
                    let group = groups.swap_remove(at);
                    group.owner.owns_deferred.store(false, Ordering::Relaxed);
                    removed.push(group);
                }
                std::iter::once(rows).collect()
            }
            (BackKind::End, None, None) => std::iter::once(rows).collect(),
            (BackKind::Flush, Some(at), _) if Arc::ptr_eq(&groups[at].owner, &gate.log) => {
                let group = groups.swap_remove(at);
                group.owner.owns_deferred.store(false, Ordering::Relaxed);
                tally.flushes.fetch_add(1, Ordering::Relaxed);
                let rows = ranges(&group.rows);
                removed.push(group);
                rows
            }
            (BackKind::Flush, _, _) => Vec::new(),
        };
        drop(groups);
        lock(&gate.held).extend(removed);
        *lock(&gate.decision) = Some(copy);
    }
}

impl DirtyShadows {
    /// A gated copy back of `copies`, whose log is `log`, recorded now ([`BackGate`]), its
    /// image and buffer held by `keep`.
    pub(crate) fn back_gate(
        &self,
        commit: &mut super::DirtyCommit,
        (log, copies, keep): (Arc<ShadowLog>, Arc<ShadowCopies>, Arc<dyn Send + Sync>),
        kind: BackKind,
    ) -> Arc<BackGate> {
        let gate = Arc::new(BackGate {
            _counted: crate::live::Counted::new(crate::live::Kind::BackGate),
            seq: commit.next_seq(),
            log,
            copies,
            kind,
            keep,
            rows: Mutex::new(None),
            decision: Mutex::new(None),
            tally: Arc::clone(&self.tally),
            held: Mutex::new(Vec::new()),
        });
        commit.backs.push(Arc::clone(&gate));
        gate
    }

    /// Decide `gate`'s rows (planning order, [`DirtyShadows::planned`]), its command buffer's
    /// reach `own`.
    pub(super) fn plan_back(&self, gate: &BackGate, own: Option<&Arc<Reach>>) {
        self.deferred.plan(gate, own);
    }

    /// A buffer over pages whose source writes back on request was made: command buffers note
    /// what they reach from now on.
    pub(crate) fn saw_on_request(&self) {
        self.deferred.seen.store(true, Ordering::Relaxed);
    }

    /// Whether command buffers note what they reach ([`saw_on_request`](Self::saw_on_request)).
    pub(crate) fn on_request_seen(&self) -> bool {
        self.deferred.seen.load(Ordering::Relaxed)
    }

    /// Note that `commit`'s command buffer reaches the host bytes `ranges` another way than
    /// through a home: registered as in flight with its first.
    pub(crate) fn reach(&self, commit: &mut super::DirtyCommit, ranges: &[Range<u64>]) {
        let reach = commit.reach.get_or_insert_with(|| {
            let reach = Arc::new(Reach::default());
            lock(&self.deferred.reaches).push(Arc::downgrade(&reach));
            reach
        });
        lock(&reach.ranges).extend(ranges.iter().cloned());
    }

    /// Whether any row is deferred.
    pub(crate) fn any_deferred(&self) -> bool {
        !lock(&self.deferred.groups).is_empty()
    }

    /// The homes that own deferred rows that `orphaned` says no texture holds any more, marked
    /// flushing (`Device::flush_orphans`).
    pub(crate) fn orphans(
        &self,
        orphaned: impl Fn(&Arc<ShadowCopies>) -> bool,
    ) -> Vec<Arc<ShadowCopies>> {
        self.deferred.orphans(orphaned)
    }

    /// The groups of `owners`, whose flush failed to commit, are no longer flushing.
    pub(crate) fn unmark_orphans(&self, owners: &[Arc<ShadowCopies>]) {
        let mut groups = lock(&self.deferred.groups);
        for group in groups.iter_mut() {
            if owners.iter().any(|o| Arc::ptr_eq(o, &group.copies)) {
                group.flushing = false;
            }
        }
    }

    /// The image and buffer the group of `copies` holds (a flush's, its home's texture gone).
    pub(crate) fn deferred_keep(&self, copies: &Arc<ShadowCopies>) -> Option<Arc<dyn Send + Sync>> {
        self.deferred.keep_of(copies)
    }

    /// Whether any deferred row lies over the host bytes `bytes`.
    pub(crate) fn deferred_over(&self, bytes: &Range<u64>) -> bool {
        let groups = lock(&self.deferred.groups);
        (groups.iter()).any(|g| g.bytes.start < bytes.end && bytes.start < g.bytes.end)
    }

    /// The homes that own deferred rows over `host` (every one for `None`): a flush's
    /// (`Device::write_back_deferred`).
    pub(crate) fn deferred_owners(&self, host: Option<Range<u64>>) -> Vec<Arc<ShadowCopies>> {
        self.deferred.owners(host)
    }

    /// The owner of the deferred rows over the bytes of `copies`, whose log is `log`, other
    /// than `copies`: the sibling its copy in takes them from (`super::Sibling`).
    pub(crate) fn deferred_sibling(
        &self,
        log: &ShadowLog,
        copies: &Arc<ShadowCopies>,
    ) -> Option<super::Sibling> {
        let (copies, keep) = self.deferred.owner_of(log, copies)?;
        Some(super::Sibling {
            copies,
            _keep: keep,
        })
    }
}

impl super::DirtyCommit {
    /// Whether its command buffer flushes `copies`' deferred rows ([`BackKind::Flush`]).
    pub(super) fn flushes(&self, copies: &Arc<ShadowCopies>) -> bool {
        (self.backs.iter())
            .any(|gate| gate.kind == BackKind::Flush && Arc::ptr_eq(&gate.copies, copies))
    }
}

impl BackGate {
    /// Note the rows of its home the copy back copies, as it records: `rows` of the image.
    pub(crate) fn recorded(&self, rows: Range<u32>) {
        *lock(&self.rows) = Some(rows);
    }

    /// The regions to record of a copy back of `whole` (one region, its rows those
    /// [`recorded`](Self::recorded)), as its command buffer replays, into `out`.
    pub(crate) fn regions(
        &self,
        whole: &[vk::BufferImageCopy],
        out: &mut Vec<vk::BufferImageCopy>,
    ) {
        out.clear();
        let decision = lock(&self.decision).clone();
        let ([region], Some(rows), Some(texel)) = (whole, decision, self.copies.texel()) else {
            out.extend_from_slice(whole);
            return;
        };
        // The rows count from the image's first; the region from its own.
        let first = u32::try_from(region.image_offset.y).unwrap_or(0);
        let rows: Vec<Range<u32>> = (rows.iter())
            .map(|row| row.start.saturating_sub(first)..row.end.saturating_sub(first))
            .collect();
        row_regions(region, &rows, texel, out);
        let row_bytes = u64::from(region.image_extent.width) * texel;
        let bytes: u64 = (out.iter())
            .map(|r| u64::from(r.image_extent.height) * row_bytes)
            .sum();
        (self.tally.write_back_bytes).fetch_add(bytes, Ordering::Relaxed);
    }
}
