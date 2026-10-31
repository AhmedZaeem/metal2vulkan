//! Homes: on a discrete GPU, one device-local image for every texture over the same bytes of host
//! pages whose memory carries a log of its writes (`HostMemory::with_dirty_pages`), laid out alike.
//!
//! Such a texture is a shadow of the bytes in the device's memory (a device copy, or a shadow that
//! renders), whose copies in the log gates (`dirty_pages.rs`). Without homes each texture over the
//! bytes is a shadow of its own: a pass rendering into one merges its texels back into the pages,
//! a GPU write the log does not show, so every other texture's next copy in reads the rows back
//! across the bus, once per frame and texture. With a home, a texture made over the same bytes,
//! of the same buffer, at the same offset, rows and texel size, extent and tracking, whose image
//! the home's serves (format, or a format the home's image is viewed as, usage and creation flags
//! within the home's), takes the home's image, texels and copies (`Storage::Shadow`, each shared):
//! one image, one log and one hold, one resource to the tracker. What one texture renders, the
//! others sample in place in the device's memory; the merge back into the pages stays, as Metal
//! shows the CPU what the GPU wrote.
//!
//! - **The texels are the truth's, as before.** Every texture over the home presents the same
//!   bytes of the same buffer, laid out alike, so a write through any of them is a write through
//!   the shadow they share, and its copies in, keeps and merges are the shadow's: nothing the
//!   texture's own shadow would have done differs but the images there are no more of.
//! - **Made as wide as is free.** A home's image takes the render target and sampled usage a
//!   texture that may take it later needs, where that costs the image nothing and keeps its
//!   format (`Device::if_free`), so a texture that renders can take a home a sampled texture
//!   made, and the other way.
//! - **Kept a while** (the afterlife). The registry holds a home's image and copies, and its
//!   texels weakly, so a home's last texture going leaves it: for [`AFTERLIFE_NANOS`], within
//!   [`AFTERLIFE_BYTES`] of such homes' images in the process (every device's) and
//!   [`AFTERLIFE_HOMES`] a device, the oldest going first, a texture made
//!   over the same bytes again takes it with its log, hold and watch unbroken, or a copy in over
//!   them takes its image as a base (`dirty_pages.rs`'s `Sibling`). A program that makes its
//!   textures over a surface anew each frame copies only the rows that changed. It holds no
//!   buffer and no pages: a home whose buffer went, or whose bytes a rebind moved, goes at once
//!   (its import's going sweeps the registry), so no guest memory goes later than it did.
//!
//! Counted: `WorkCounts::shadow_homes_made` and `shadow_homes_taken`, the afterlife's
//! `home_afterlives_*` and `home_afterlife_bytes`, and of the gated copies in, those that copied
//! nothing over a home more than one texture took (`dirty_copies_in_skipped_in_shared_homes`).
//! Under the dirty census (`METAL2VULKAN_LOG_DIRTY_CENSUS`) a line a second gives their rates.

use super::{Image, ImagePlan, Storage};
use crate::buffer_texture::Texels;
use crate::device::lock;
use crate::dirty_pages::Sibling;
use crate::resource::ResourceKey;
use crate::shadow::{ShadowCopies, ShadowSource, ShadowTarget};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

/// What a texture over logged pages shares a home by: its buffer's bytes' identity, the offset,
/// rows and texel size its texels lie at, its extent, and whether it is tracked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct HomeKey {
    buffer: ResourceKey,
    offset: u64,
    row_pitch: u64,
    texel: u64,
    extent: [u32; 3],
    tracked: bool,
}

impl HomeKey {
    /// The key of a texture laid out as `plan` over `texels`, `extent` texels wide, high and deep.
    pub(super) fn of(texels: &Texels, plan: &ImagePlan, tracked: bool) -> HomeKey {
        let (buffer, offset, row_pitch, texel) = texels.home_parts();
        HomeKey {
            buffer,
            offset,
            row_pitch,
            texel,
            extent: plan.extent,
            tracked,
        }
    }
}

/// One home: its image and copies, and its texels, held weakly: a texture holds it while they
/// live, and its afterlife begins as they go (the module's).
struct Home {
    key: HomeKey,
    /// The host addresses of its bytes, where known (the census's miss lines).
    host: Option<std::ops::Range<u64>>,
    image: Arc<Image>,
    texels: Weak<Texels>,
    copies: Arc<ShadowCopies>,
    /// The `VkBuffer` its texels lay in when it was made: a rebind gives its buffer another.
    raw: ash::vk::Buffer,
    /// Its image's bytes.
    bytes: u64,
    /// The host nanoseconds no texture held it from, in its afterlife.
    orphaned: Option<u64>,
}

/// The device's homes ([`Registry`]), which the process's reaper sweeps while any is kept
/// ([`reap`]).
#[derive(Default)]
pub(super) struct ShadowHomes {
    inner: Arc<Registry>,
}

/// The homes of a device, and what they did.
#[derive(Default)]
struct Registry {
    homes: Mutex<Vec<Home>>,
    tally: AfterlifeTally,
    /// Whether the reaper sweeps it ([`reap`]): from its first home until it holds none, set
    /// under the homes' lock.
    reaped: std::sync::atomic::AtomicBool,
    /// The host nanoseconds of its last sweep ([`HOT_SWEEP_NANOS`]).
    swept_at: AtomicU64,
    /// Its share of the process's counts after its last sweep: bytes kept, held, orphaned.
    share: Mutex<[u64; 3]>,
}

impl Drop for Registry {
    fn drop(&mut self) {
        let share = *self.share.get_mut().unwrap_or_else(|e| e.into_inner());
        let [bytes, held, orphaned] = share;
        KEPT_BYTES.fetch_sub(bytes, Ordering::Relaxed);
        HELD.fetch_sub(held, Ordering::Relaxed);
        ORPHANED.fetch_sub(orphaned, Ordering::Relaxed);
    }
}

/// How often the reaper sweeps the registries that keep homes: a home its last texture left
/// goes within [`AFTERLIFE_NANOS`] and this, with or without work on its device, which its
/// image holds.
const REAP_EVERY: std::time::Duration = std::time::Duration::from_millis(100);

/// Have the process's reaper sweep `registry` every [`REAP_EVERY`] until it holds no home: a
/// thread of its own, made at the first.
fn reap(registry: &Arc<Registry>) {
    use std::sync::mpsc::{channel, RecvTimeoutError, Sender};
    static REAPER: Mutex<Option<Sender<Weak<Registry>>>> = Mutex::new(None);
    let mut reaper = lock(&REAPER);
    let sender = reaper.get_or_insert_with(|| {
        let (sender, received) = channel::<Weak<Registry>>();
        let builder = std::thread::Builder::new().name("m2v-home-reaper".to_owned());
        let spawned = crate::exit_guard::spawn(builder, move || {
            let mut registries: Vec<Weak<Registry>> = Vec::new();
            loop {
                let next = match registries.is_empty() {
                    true => received.recv().map_err(|_| RecvTimeoutError::Disconnected),
                    false => received.recv_timeout(REAP_EVERY),
                };
                match next {
                    Ok(registry) => registries.push(registry),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                registries.retain(|registry| {
                    registry
                        .upgrade()
                        .is_some_and(|registry| registry.sweep_reaped())
                });
            }
        });
        // Without a thread the registries are swept as their devices work.
        if spawned.is_err() {
            registry.reaped.store(false, Ordering::Relaxed);
        }
        sender
    });
    let _ = sender.send(Arc::downgrade(registry));
}

impl Registry {
    /// Whether a sweep is due as textures are made and copied in ([`HOT_SWEEP_NANOS`]).
    fn due(&self) -> bool {
        let now = crate::gpu_time::host_nanos();
        now.saturating_sub(self.swept_at.load(Ordering::Relaxed)) >= HOT_SWEEP_NANOS
    }

    /// The reaper's sweep: whether to sweep it again, where it holds any home.
    fn sweep_reaped(&self) -> bool {
        let (mut gone, mut kept) = (Vec::new(), Vec::new());
        let mut homes = lock(&self.homes);
        self.swept(&mut homes, (&mut gone, &mut kept));
        let again = !homes.is_empty();
        if !again {
            self.reaped.store(false, Ordering::Relaxed);
        }
        drop(homes);
        drop((gone, kept));
        again
    }
}

/// What the homes did ([`ShadowHomes::tally`]): made, taken, and their afterlives kept, taken
/// again, expired, evicted past the budget and dropped as their pages went, and the bytes of
/// the images kept.
#[derive(Default)]
struct AfterlifeTally {
    made: AtomicU64,
    kept: AtomicU64,
    taken: AtomicU64,
    expired: AtomicU64,
    evicted: AtomicU64,
    dropped: AtomicU64,
    bytes: AtomicU64,
    /// The census's last line: when, and the counts then.
    censused: Mutex<(u64, [u64; 6])>,
}

/// How long a home no texture holds stays (the module's **Kept a while**): about a frame and
/// its successors' at low rates, and the time a log keeps its watch unused (`dirty_pages.rs`).
const AFTERLIFE_NANOS: u64 = 250_000_000;

/// The most bytes of images homes no texture holds keep in the process, every device's
/// registry's together ([`KEPT_BYTES`]), the longest kept going first.
const AFTERLIFE_BYTES: u64 = 256 << 20;

/// The most homes no texture holds a device's registry keeps.
const AFTERLIFE_HOMES: usize = 32;

/// How often at most a registry is swept as textures are made and copied in: the reaper sweeps
/// it every [`REAP_EVERY`] besides, and an import's going or a rebind at once.
const HOT_SWEEP_NANOS: u64 = 10_000_000;

/// The bytes of images every registry's homes in their afterlife keep, and how many homes a
/// texture holds and how many none does, in the process (`WorkCounts::home_afterlife_bytes`,
/// `homes_held`, `homes_orphaned`).
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
static HELD: AtomicU64 = AtomicU64::new(0);
static ORPHANED: AtomicU64 = AtomicU64::new(0);

/// The process's homes: the bytes kept, held homes and orphaned homes.
pub(crate) fn process_homes() -> [u64; 3] {
    [&KEPT_BYTES, &HELD, &ORPHANED].map(|count| count.load(Ordering::Relaxed))
}

/// A registry's entry as the afterlife's rules see it ([`sweep`]).
trait Lived {
    /// What a look at its pages takes hold of, which the caller drops once it let go of the
    /// registry's lock: a buffer's last handle going under it would free the buffer's import,
    /// whose going sweeps the registry ([`Registry::pages_gone`]), and run its deallocator.
    type Kept;
    /// Whether a texture holds it.
    fn held(&self) -> bool;
    /// Whether its bytes lie where it was made: its buffer lives, not rebound since; what it
    /// took hold of to tell into `kept`.
    fn pages(&self, kept: &mut Vec<Self::Kept>) -> bool;
    /// Its image's bytes.
    fn bytes(&self) -> u64;
    fn orphaned(&self) -> Option<u64>;
    fn set_orphaned(&mut self, at: Option<u64>);
}

/// What one sweep did ([`sweep`]).
#[derive(Debug, Default, PartialEq, Eq)]
struct Swept {
    kept: u64,
    expired: u64,
    evicted: u64,
    dropped: u64,
    /// The bytes the entries no texture holds keep, after it.
    bytes: u64,
    /// The entries a texture holds, and those none does, after it.
    held: u64,
    orphans: u64,
}

/// Sweep `entries` at the host nanoseconds `now`: an entry whose pages went goes; one a texture
/// holds is no orphan; one no texture holds begins its afterlife, and goes past
/// [`AFTERLIFE_NANOS`] of it; then, past `budget` bytes or `most` orphans, the longest orphaned
/// go. What goes is moved to `gone`, for the caller to drop once it let go of the registry.
fn sweep<L: Lived>(
    entries: &mut Vec<L>,
    now: u64,
    (gone, kept): (&mut Vec<L>, &mut Vec<L::Kept>),
    (budget, most): (u64, usize),
) -> Swept {
    let mut swept = Swept::default();
    let mut at = 0;
    while at < entries.len() {
        let entry = &mut entries[at];
        let goes = if !entry.pages(kept) {
            swept.dropped += 1;
            true
        } else if entry.held() {
            entry.set_orphaned(None);
            false
        } else {
            let since = match entry.orphaned() {
                Some(since) => since,
                None => {
                    swept.kept += 1;
                    entry.set_orphaned(Some(now));
                    now
                }
            };
            let expired = now.saturating_sub(since) > AFTERLIFE_NANOS;
            swept.expired += u64::from(expired);
            expired
        };
        match goes {
            true => gone.push(entries.swap_remove(at)),
            false => at += 1,
        }
    }
    let orphans = |entries: &[L]| -> u64 {
        (entries.iter())
            .filter(|entry| entry.orphaned().is_some())
            .map(Lived::bytes)
            .sum()
    };
    let mut kept = orphans(entries);
    let mut count = (entries.iter()).filter(|e| e.orphaned().is_some()).count();
    while kept > budget || count > most {
        let oldest = (entries.iter().enumerate())
            .filter_map(|(at, entry)| entry.orphaned().map(|since| (since, at)))
            .min()
            .map(|(_, at)| at);
        let Some(oldest) = oldest else {
            break;
        };
        let entry = entries.swap_remove(oldest);
        kept -= entry.bytes();
        count -= 1;
        swept.evicted += 1;
        gone.push(entry);
    }
    swept.bytes = kept;
    swept.orphans = count as u64;
    swept.held = entries.len() as u64 - swept.orphans;
    swept
}

/// A texture takes `entry` again: whether it was in its afterlife.
fn revive<L: Lived>(entry: &mut L) -> bool {
    let orphaned = entry.orphaned().is_some();
    entry.set_orphaned(None);
    orphaned
}

impl Lived for Home {
    type Kept = crate::Buffer;

    fn held(&self) -> bool {
        self.texels.strong_count() > 0
    }

    fn pages(&self, kept: &mut Vec<crate::Buffer>) -> bool {
        shadow_bytes_kept(&self.copies, kept).is_some_and(|(raw, _)| raw == self.raw)
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }

    fn orphaned(&self) -> Option<u64> {
        self.orphaned
    }

    fn set_orphaned(&mut self, at: Option<u64>) {
        self.orphaned = at;
    }
}

/// The most homes the registry keeps: past them a texture's shadow is its own.
const HOMES: usize = 256;

impl Registry {
    /// Another texture's live home whose first texel lies at the host address `host`, as the
    /// first of `copies` does, laid out alike, of its tracking: through any buffer over those
    /// host bytes, from any task. A gated copy in of `copies` may take its image as a base
    /// (`dirty_pages.rs`'s `Sibling`), which its submission checks again.
    fn sibling(&self, copies: &Arc<ShadowCopies>, host: u64) -> Option<Sibling> {
        let (_, mine) = shadow_bytes(copies)?;
        let (mut gone, mut kept) = (Vec::new(), Vec::new());
        let mut homes = lock(&self.homes);
        if self.due() {
            self.swept(&mut homes, (&mut gone, &mut kept));
        }
        let sibling = homes.iter().find_map(|home| {
            if home.host.as_ref().map(|bytes| bytes.start) != Some(host) {
                return None;
            }
            let theirs = &home.copies;
            if Arc::ptr_eq(theirs, copies) || theirs.tracked != copies.tracked {
                return None;
            }
            if !home.pages(&mut kept) || shadow_bytes_kept(theirs, &mut kept)?.1 != mine {
                return None;
            }
            // Its buffer too: a copy in's submission reads it again under the scheduler's lock
            // (`dirty_pages.rs`), where its last handle going would run its deallocator.
            let buffer = kept.last().cloned();
            let keep: Arc<dyn Send + Sync> = Arc::new((Arc::<Image>::clone(&home.image), buffer));
            Some(Sibling {
                copies: Arc::clone(theirs),
                _keep: keep,
            })
        });
        drop(homes);
        drop((gone, kept));
        sibling
    }

    /// Sweep the registry now (its import went, or a rebind moved a buffer's bytes): the homes
    /// whose pages went go at once.
    fn pages_gone(&self) {
        let (mut gone, mut kept) = (Vec::new(), Vec::new());
        // An import's going under a sweep of this registry (its buffer's last handle let go of
        // there) finds the lock held, by its own thread or another: that sweep, or the next,
        // drops the homes whose pages went, and a home is checked as it is taken.
        let mut homes = match self.homes.try_lock() {
            Ok(homes) => homes,
            Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return,
        };
        self.swept(&mut homes, (&mut gone, &mut kept));
        drop(homes);
        drop((gone, kept));
    }

    /// The tally ([`AfterlifeTally`]): made, taken, and afterlives kept, taken, expired,
    /// evicted and dropped, and the bytes kept.
    fn tally(&self) -> [u64; 7] {
        let t = &self.tally;
        [
            &t.made, &t.kept, &t.taken, &t.expired, &t.evicted, &t.dropped, &t.bytes,
        ]
        .map(|count| count.load(Ordering::Relaxed))
    }

    /// [`sweep`] `homes`, under their lock, into `gone`, counted, and the census's line a second.
    fn swept(
        &self,
        homes: &mut Vec<Home>,
        (gone, kept): (&mut Vec<Home>, &mut Vec<crate::Buffer>),
    ) {
        let now = crate::gpu_time::host_nanos();
        self.swept_at.store(now, Ordering::Relaxed);
        let mut share = lock(&self.share);
        // The process's budget, less what the other registries keep.
        let others = KEPT_BYTES.load(Ordering::Relaxed).saturating_sub(share[0]);
        let budget = AFTERLIFE_BYTES.saturating_sub(others);
        let swept = sweep(homes, now, (gone, kept), (budget, AFTERLIFE_HOMES));
        let now_share = [swept.bytes, swept.held, swept.orphans];
        for ((count, old), new) in [&KEPT_BYTES, &HELD, &ORPHANED]
            .iter()
            .zip(*share)
            .zip(now_share)
        {
            count.fetch_add(new, Ordering::Relaxed);
            count.fetch_sub(old, Ordering::Relaxed);
        }
        *share = now_share;
        drop(share);
        let t = &self.tally;
        t.kept.fetch_add(swept.kept, Ordering::Relaxed);
        t.expired.fetch_add(swept.expired, Ordering::Relaxed);
        t.evicted.fetch_add(swept.evicted, Ordering::Relaxed);
        t.dropped.fetch_add(swept.dropped, Ordering::Relaxed);
        t.bytes.store(swept.bytes, Ordering::Relaxed);
        if crate::dirty_pages::census_on() {
            self.census_line(now, homes.len());
        }
    }

    /// Under the dirty census, once a second: the rates of homes made and taken and of their
    /// afterlives, and what the afterlife keeps.
    fn census_line(&self, now: u64, homes: usize) {
        let t = &self.tally;
        let mut censused = lock(&t.censused);
        let (at, before) = *censused;
        if now.saturating_sub(at) < 1_000_000_000 {
            return;
        }
        let counts = [
            &t.made, &t.kept, &t.taken, &t.expired, &t.evicted, &t.dropped,
        ]
        .map(|count| count.load(Ordering::Relaxed));
        *censused = (now, counts);
        drop(censused);
        if at == 0 {
            return;
        }
        let seconds = now.saturating_sub(at) as f64 / 1e9;
        let rate = |i: usize| (counts[i].saturating_sub(before[i]) as f64 / seconds).round();
        eprintln!(
            "m2v homes: made {}/s, {homes} live; afterlife kept {}/s, taken {}/s, expired {}/s, \
             evicted {}/s, dropped as pages went {}/s; {:.1} MiB kept",
            rate(0),
            rate(1),
            rate(2),
            rate(3),
            rate(4),
            rate(5),
            t.bytes.load(Ordering::Relaxed) as f64 / f64::from(1 << 20),
        );
    }

    /// The storage of a live home of `key` whose image serves a texture planned as `plan`, marked
    /// shared; `None` where none does.
    fn take(&self, key: HomeKey, plan: &ImagePlan, texels: &Arc<Texels>) -> Option<Storage> {
        let (mut gone, mut kept) = (Vec::new(), Vec::new());
        let mut homes = lock(&self.homes);
        if self.due() {
            self.swept(&mut homes, (&mut gone, &mut kept));
        }
        if crate::dirty_pages::census_on() {
            miss_lines(&homes, key, plan, texels);
        }
        let taken = (homes.iter_mut())
            .filter(|home| {
                home.key == key && serves(home.image.plan(), plan) && home.pages(&mut kept)
            })
            .map(|home| {
                // A home in its afterlife takes the new texture's texels, laid out as its own
                // over the same buffer (the key's).
                let held = home.texels.upgrade();
                let texels = held.unwrap_or_else(|| Arc::clone(texels));
                if revive(home) {
                    home.texels = Arc::downgrade(&texels);
                    self.tally.taken.fetch_add(1, Ordering::Relaxed);
                }
                home.copies.shared.store(true, Ordering::Relaxed);
                Storage::Shadow {
                    image: Arc::clone(&home.image),
                    texels,
                    copies: Arc::clone(&home.copies),
                }
            })
            .next();
        drop(homes);
        drop((gone, kept));
        taken
    }

    /// `storage`, a shadow just made over logged pages, as the home of `key`: whether the reaper
    /// is to sweep the registry from now on.
    fn offer(&self, key: HomeKey, storage: &Storage) -> bool {
        let Storage::Shadow {
            image,
            texels,
            copies,
        } = storage
        else {
            return false;
        };
        let Some((raw, layout)) = shadow_bytes(copies) else {
            return false;
        };
        let (_, _, [width, height, depth], texel) = layout;
        let bytes = u64::from(width) * u64::from(height) * u64::from(depth.max(1)) * texel;
        let (mut gone, mut kept) = (Vec::new(), Vec::new());
        let mut homes = lock(&self.homes);
        if self.due() || homes.len() >= HOMES {
            self.swept(&mut homes, (&mut gone, &mut kept));
        }
        if homes.len() < HOMES {
            self.tally.made.fetch_add(1, Ordering::Relaxed);
            homes.push(Home {
                key,
                host: texels.host_bytes(),
                image: Arc::clone(image),
                texels: Arc::downgrade(texels),
                copies: Arc::clone(copies),
                raw,
                bytes,
                orphaned: None,
            });
        }
        let reaped = !homes.is_empty() && !self.reaped.swap(true, Ordering::Relaxed);
        drop(homes);
        drop((gone, kept));
        reaped
    }
}

impl ShadowHomes {
    /// [`Registry::sibling`].
    pub(super) fn sibling(&self, copies: &Arc<ShadowCopies>, host: u64) -> Option<Sibling> {
        self.inner.sibling(copies, host)
    }

    /// [`Registry::pages_gone`].
    pub(super) fn pages_gone(&self) {
        self.inner.pages_gone();
    }

    /// [`Registry::tally`].
    pub(super) fn tally(&self) -> [u64; 7] {
        self.inner.tally()
    }

    /// [`Registry::take`].
    pub(super) fn take(
        &self,
        key: HomeKey,
        plan: &ImagePlan,
        texels: &Arc<Texels>,
    ) -> Option<Storage> {
        self.inner.take(key, plan, texels)
    }

    /// [`Registry::offer`], the reaper sweeping the registry from its first home.
    pub(super) fn offer(&self, key: HomeKey, storage: &Storage) {
        if self.inner.offer(key, storage) {
            reap(&self.inner);
        }
    }
}

/// The misses the census printed (`METAL2VULKAN_LOG_DIRTY_CENSUS`), by shape and kind: each
/// once, 200 at most.
static MISSES: Mutex<Vec<([u32; 3], u64, bool)>> = Mutex::new(Vec::new());

/// Under the census, why no live home serves a texture of `key` planned as `plan` over
/// `texels` where one of its shape lies over the same host bytes through another buffer, or over
/// its bytes but serves it not: a diagnostic, each shape and kind printed once.
fn miss_lines(homes: &[Home], key: HomeKey, plan: &ImagePlan, texels: &Texels) {
    let host = texels.host_bytes();
    let overlaps = |theirs: &Option<std::ops::Range<u64>>| match (theirs, &host) {
        (Some(a), Some(b)) => a.start < b.end && b.start < a.end,
        _ => false,
    };
    let near = |home: &&Home| {
        home.key.extent == key.extent
            && home.key.row_pitch == key.row_pitch
            && home.key.texel == key.texel
    };
    for home in homes.iter().filter(near) {
        let theirs = home.image.plan();
        let why = match home.key == key {
            false if !overlaps(&home.host) => continue,
            false => format!(
                "a home of its shape over the same host bytes through another buffer (buffer {:?} \
                 offset {} tracked {}, its {:?} offset {} tracked {})",
                home.key.buffer,
                home.key.offset,
                home.key.tracked,
                key.buffer,
                key.offset,
                key.tracked
            ),
            true if serves(theirs, plan) => continue,
            true => format!(
                "the home's image serves it not: format {:?} views {:?} usage {:?} flags {:?}, \
                 its format {:?} usage {:?} flags {:?}",
                theirs.format,
                theirs.view_formats,
                theirs.usage,
                theirs.flags,
                plan.format,
                plan.usage,
                plan.flags
            ),
        };
        let seen = (key.extent, key.row_pitch, home.key == key);
        {
            let mut misses = lock(&MISSES);
            if misses.contains(&seen) || misses.len() >= 200 {
                continue;
            }
            misses.push(seen);
        }
        eprintln!(
            "m2v home miss {}x{} rows {}: {why}",
            key.extent[0], key.extent[1], key.row_pitch
        );
    }
}

/// The shapes the census named a texture over logged pages that takes no home for, each once.
static NO_HOMES: Mutex<Vec<([u32; 3], u64)>> = Mutex::new(Vec::new());

/// Under the census, why a texture over logged host pages of 2^18 texels or more, planned as
/// `plan` over `texels`, is no texture a home takes: each condition, the first time for its shape.
pub(super) fn no_home_line(
    device: &crate::Device,
    texels: &Texels,
    plan: &ImagePlan,
    layout: &crate::texture::TextureLayout,
) {
    let logged = (texels.buffer.host_pages()).is_some_and(|pages| pages.dirty().is_some());
    let texels_count = u64::from(plan.extent[0]) * u64::from(plan.extent[1]);
    if !logged || texels_count < 1 << 18 {
        return;
    }
    let seen = (plan.extent, texels.layout.row_pitch);
    {
        let mut shapes = lock(&NO_HOMES);
        if shapes.contains(&seen) || shapes.len() >= 200 {
            return;
        }
        shapes.push(seen);
    }
    eprintln!(
        "m2v no home {}x{} rows {}: {}; detached {} type {:?} levels {} layers {} samples {:?} \
         aspects {:#x} depth unit {}",
        plan.extent[0],
        plan.extent[1],
        texels.layout.row_pitch,
        texels.device_copy_why(device),
        texels.buffer.detached(),
        plan.image_type,
        plan.levels,
        plan.layers,
        plan.samples,
        super::texture_aspects(plan).as_raw(),
        super::surface_depth_unit(layout.pixel_format).is_some(),
    );
}

/// Whether a home's image of `home` serves a texture whose own image would be `plan`: of its shape,
/// its format or one the home's image is viewed as, and usage and creation flags within the home's.
fn serves(home: &ImagePlan, plan: &ImagePlan) -> bool {
    home.image_type == plan.image_type
        && home.extent == plan.extent
        && home.levels == plan.levels
        && home.layers == plan.layers
        && home.samples == plan.samples
        && home.usage.contains(plan.usage)
        && home.flags.contains(plan.flags)
        && (home.format == plan.format || home.view_formats.contains(&plan.format))
}

/// How a shadow's texels are laid out in its bytes: rows' and images' texels, the image's
/// aspect, level and layers, the region's offset and extent, and the texel's bytes.
pub(crate) type ShadowLayout = ([u32; 6], [i32; 3], [u32; 3], u64);

/// Where a shadow's texels lie now, for telling two shadows over the same host bytes apart: its
/// `VkBuffer`, and their layout there; `None` where they lie in no one region of a `VkBuffer`,
/// or nowhere (a buffer gone or detached).
/// [`shadow_bytes`], the buffer it reads taken hold of into `kept`: for a caller under a lock,
/// which drops `kept` once it let go of it ([`Lived::Kept`]).
fn shadow_bytes_kept(
    copies: &ShadowCopies,
    kept: &mut Vec<crate::Buffer>,
) -> Option<(ash::vk::Buffer, ShadowLayout)> {
    if let ShadowSource::Bytes {
        buffer: crate::shadow::ShadowBuffer::Rebound(buffer),
        ..
    } = &copies.source
    {
        kept.extend(buffer.upgrade());
    }
    shadow_bytes(copies)
}

pub(crate) fn shadow_bytes(copies: &ShadowCopies) -> Option<(ash::vk::Buffer, ShadowLayout)> {
    let ShadowSource::Bytes {
        buffer,
        target:
            ShadowTarget::Image {
                regions,
                staging: None,
                texel: Some(texel),
                ..
            },
        ..
    } = &copies.source
    else {
        return None;
    };
    let [region] = &regions[..] else {
        return None;
    };
    let raw = buffer.raw();
    if raw == ash::vk::Buffer::null() {
        return None;
    }
    let sub = region.image_subresource;
    let (offset, extent) = (region.image_offset, region.image_extent);
    let layout = (
        [
            region.buffer_row_length,
            region.buffer_image_height,
            sub.aspect_mask.as_raw(),
            sub.mip_level,
            sub.base_array_layer,
            sub.layer_count,
        ],
        [offset.x, offset.y, offset.z],
        [extent.width, extent.height, extent.depth],
        *texel,
    );
    Some((raw, layout))
}

#[cfg(test)]
mod tests {
    use super::{revive, sweep, Lived, Swept, AFTERLIFE_BYTES, AFTERLIFE_HOMES, AFTERLIFE_NANOS};

    /// The process's whole budget and the registry's count.
    const ALL: (u64, usize) = (AFTERLIFE_BYTES, AFTERLIFE_HOMES);

    /// An entry of the afterlife's rules alone.
    #[derive(Debug)]
    struct Entry {
        held: bool,
        pages: bool,
        bytes: u64,
        orphaned: Option<u64>,
    }

    impl Lived for Entry {
        type Kept = ();
        fn held(&self) -> bool {
            self.held
        }
        fn pages(&self, _kept: &mut Vec<()>) -> bool {
            self.pages
        }
        fn bytes(&self) -> u64 {
            self.bytes
        }
        fn orphaned(&self) -> Option<u64> {
            self.orphaned
        }
        fn set_orphaned(&mut self, at: Option<u64>) {
            self.orphaned = at;
        }
    }

    fn entry(held: bool, bytes: u64) -> Entry {
        Entry {
            held,
            pages: true,
            bytes,
            orphaned: None,
        }
    }

    #[test]
    fn a_home_no_texture_holds_is_kept_until_it_expires() {
        let mut entries = vec![entry(false, 1 << 20), entry(true, 1 << 20)];
        let mut gone = Vec::new();
        let first = sweep(&mut entries, 1_000, (&mut gone, &mut Vec::new()), ALL);
        assert_eq!(
            first,
            Swept {
                kept: 1,
                bytes: 1 << 20,
                held: 1,
                orphans: 1,
                ..Swept::default()
            }
        );
        assert_eq!(entries.len(), 2, "kept, its afterlife begun");
        let within = sweep(
            &mut entries,
            1_000 + AFTERLIFE_NANOS,
            (&mut gone, &mut Vec::new()),
            ALL,
        );
        assert_eq!(within.expired, 0, "within its afterlife");
        let past = sweep(
            &mut entries,
            1_001 + AFTERLIFE_NANOS,
            (&mut gone, &mut Vec::new()),
            ALL,
        );
        assert_eq!(past.expired, 1);
        assert_eq!(entries.len(), 1, "the held one stays");
        assert_eq!(gone.len(), 1);
    }

    #[test]
    fn a_home_taken_again_leaves_its_afterlife() {
        let mut entries = vec![entry(false, 1 << 20)];
        let mut gone = Vec::new();
        sweep(&mut entries, 1_000, (&mut gone, &mut Vec::new()), ALL);
        assert!(revive(&mut entries[0]), "taken out of its afterlife");
        entries[0].held = true;
        assert!(!revive(&mut entries[0]), "a held home is in none");
        let later = sweep(
            &mut entries,
            1_000 + 2 * AFTERLIFE_NANOS,
            (&mut gone, &mut Vec::new()),
            ALL,
        );
        let held = Swept {
            held: 1,
            ..Swept::default()
        };
        assert_eq!(later, held, "held, it never expires");
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn past_the_budget_the_longest_kept_go_first() {
        let third = AFTERLIFE_BYTES / 3 + 1;
        let mut entries = Vec::new();
        let mut gone = Vec::new();
        for at in 1..=3 {
            entries.push(entry(false, third));
            sweep(
                &mut entries,
                at,
                (&mut gone, &mut Vec::new()),
                (AFTERLIFE_BYTES, usize::MAX),
            );
        }
        let evicted: u64 = gone.len() as u64;
        assert_eq!(evicted, 1, "the third goes past the budget");
        assert_eq!(gone[0].orphaned, Some(1), "the longest kept");
        assert_eq!(entries.len(), 2);
        let swept = sweep(
            &mut entries,
            4,
            (&mut gone, &mut Vec::new()),
            (AFTERLIFE_BYTES, usize::MAX),
        );
        assert_eq!(swept.bytes, 2 * third);
    }

    #[test]
    fn a_registry_keeps_a_few_homes_and_a_share_of_the_process_s_bytes() {
        let mut entries: Vec<Entry> = (0..AFTERLIFE_HOMES + 3).map(|_| entry(false, 1)).collect();
        let mut gone = Vec::new();
        let swept = sweep(&mut entries, 1_000, (&mut gone, &mut Vec::new()), ALL);
        assert_eq!(swept.evicted, 3, "past the count a registry keeps");
        assert_eq!(entries.len(), AFTERLIFE_HOMES);
        // Other registries keep all but 10 bytes of the process's budget.
        let mut entries: Vec<Entry> = (0..4).map(|_| entry(false, 4)).collect();
        let swept = sweep(
            &mut entries,
            1_000,
            (&mut gone, &mut Vec::new()),
            (10, AFTERLIFE_HOMES),
        );
        assert_eq!(swept.bytes, 8, "within its share");
        assert_eq!(swept.evicted, 2);
    }

    #[test]
    fn a_home_whose_pages_went_goes_at_once() {
        let mut entries = vec![entry(true, 1 << 20), entry(false, 1 << 20)];
        let mut gone = Vec::new();
        sweep(&mut entries, 1_000, (&mut gone, &mut Vec::new()), ALL);
        // A rebind moved the held one's bytes, and the kept one's buffer went.
        entries.iter_mut().for_each(|entry| entry.pages = false);
        let swept = sweep(&mut entries, 1_001, (&mut gone, &mut Vec::new()), ALL);
        assert_eq!(swept.dropped, 2, "both, held or kept, at once");
        assert!(entries.is_empty());
    }

    #[test]
    fn a_sweep_its_own_thread_asks_for_under_the_registry_s_lock_returns() {
        // A buffer's last handle going under a sweep frees its import, whose going asks for a
        // sweep of the same registry on the same thread.
        let registry = super::Registry::default();
        let held = registry.homes.lock().unwrap();
        registry.pages_gone();
        drop(held);
        registry.pages_gone();
    }
}
