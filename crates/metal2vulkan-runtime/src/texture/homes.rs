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
//! - **Held weakly.** The registry holds a home's parts weakly: the home goes with the last
//!   texture over it.
//!
//! Counted: `WorkCounts::shadow_homes_taken`, and of the gated copies in, those that copied
//! nothing over a home more than one texture took (`dirty_copies_in_skipped_in_shared_homes`).

use super::{Image, ImagePlan, Storage};
use crate::buffer_texture::Texels;
use crate::device::lock;
use crate::resource::ResourceKey;
use crate::shadow::ShadowCopies;
use std::sync::atomic::Ordering;
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

/// One home: its image, texels and copies, held weakly.
struct Home {
    key: HomeKey,
    /// The host addresses of its bytes, where known (the census's miss lines).
    host: Option<std::ops::Range<u64>>,
    image: Weak<Image>,
    texels: Weak<Texels>,
    copies: Weak<ShadowCopies>,
}

/// The device's live homes.
#[derive(Default)]
pub(super) struct ShadowHomes {
    homes: Mutex<Vec<Home>>,
}

/// The most homes the registry keeps: past them a texture's shadow is its own.
const HOMES: usize = 256;

impl ShadowHomes {
    /// The storage of a live home of `key` whose image serves a texture planned as `plan`, marked
    /// shared; `None` where none does.
    pub(super) fn take(&self, key: HomeKey, plan: &ImagePlan, texels: &Texels) -> Option<Storage> {
        let mut homes = lock(&self.homes);
        homes.retain(|home| home.copies.strong_count() > 0);
        if crate::dirty_pages::census_on() {
            miss_lines(&homes, key, plan, texels);
        }
        (homes.iter())
            .filter(|home| home.key == key)
            .find_map(|home| {
                let image = home.image.upgrade()?;
                let texels = home.texels.upgrade()?;
                let copies = home.copies.upgrade()?;
                if !serves(image.plan(), plan) {
                    return None;
                }
                copies.shared.store(true, Ordering::Relaxed);
                Some(Storage::Shadow {
                    image,
                    texels,
                    copies,
                })
            })
    }

    /// `storage`, a shadow just made over logged pages, as the home of `key`.
    pub(super) fn offer(&self, key: HomeKey, storage: &Storage) {
        let Storage::Shadow {
            image,
            texels,
            copies,
        } = storage
        else {
            return;
        };
        let mut homes = lock(&self.homes);
        homes.retain(|home| home.copies.strong_count() > 0);
        if homes.len() < HOMES {
            homes.push(Home {
                key,
                host: texels.host_bytes(),
                image: Arc::downgrade(image),
                texels: Arc::downgrade(texels),
                copies: Arc::downgrade(copies),
            });
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
        let Some(image) = home.image.upgrade() else {
            continue;
        };
        let theirs = image.plan();
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
