//! A shadowed texture over a no-copy buffer's host pages whose memory came with a log of the
//! writes to them (`HostMemory::with_dirty_pages`, `dirty_pages.rs`): its copies in after the
//! first copy only the rows of the pages the log reports changed since, leave out the shadow's
//! own copies back, and copy every row where the log cannot say.
//!
//! The texture is 64x64 `RGBA8Unorm` over the buffer's first bytes, rows of 256 bytes, read and
//! written through (so it is a shadow on every device). A kernel inverts its texels into a plain
//! texture, which the test reads back: the texels the shadow held. The log is the test's own
//! ([`Log`]): it reports the writes the test tells it of, and lies by leaving out one it does not
//! tell, so a row the log leaves out shows whether a copy took it.
//!
//! - Frame 1 copies every row (no hold yet).
//! - Frame 2: the CPU writes row 10, told to the log, and row 40, not told. Row 10 reads the new
//!   bytes, row 40 the old ones: the copy took only the told row's band
//!   (`WorkCounts::dirty_copy_in_bytes_copied`).
//! - Frame 3 with no write copies nothing (`dirty_copies_in_skipped`).
//! - Frame 4: the log's wait answers `false`: every row is copied, row 40's new bytes too.
//! - Frame 7: a command buffer that waits for an event, which a command buffer committed after
//!   it signals once its blit fill wrote row 50's bytes (a GPU write no log hears of), runs after
//!   the fill: decided as it is submitted, after the fill let go of its hold, it copies every row
//!   and reads the filled row, as Metal shows a command buffer what the work it waits for wrote.
//! - Frame 8: a command buffer that waits for an event already reached copies nothing.
//! - Frame 9: a command buffer of two batches (an event wait between its encoders) that writes
//!   through the shadow may run its second after work submitted later: no hold counts until it
//!   completes, so the next frame copies every row, and the one after nothing.
//! - Frame 5: a kernel writes through the shadow (copied back, or merged, reported as the
//!   shadow's own write, `wrote_as`, at the commit and again at the completion); frame 6 then
//!   copies nothing in for it (`changed_except` leaves the shadow's own write out).
//!
//! One wait for a log's publication serves every gated copy in of a commit, two shadows over one
//! log among them, and a wait the log answers `false` is not made again for them
//! (`WorkCounts::dirty_waits`). A log that says at once that the stamp is not published yet
//! (`DirtyPages::published`) is not waited for: both copies take every row.
//!
//! A render target over such pages, loaded by a pass whose draws' scissor bounds them to rows 0
//! to 8, copies in every texel the first time and then only the rows the log reports changed: its
//! copy in is not cut to the scissor, so the image holds every texel for the next command buffer,
//! and the pass's own copy back of rows 0 to 8 is no change to it.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`). The kernels are the owned fixture
//! `validation/fixtures/public/examples/runtime_examples.ll` (`llvm-as`).

use super::blit_holds::{device, Host, Scratch, BYTES, WAIT};
use super::encoder_holds::{bytes_of, render_state};
use crate::{
    Buffer, CommandBufferStatus, ComputePipelineState, Device, DirtyPages, DirtyStamp, Error,
    EventRef, HostMemory, Library, LoadAction, PipelineOptions, Region, RenderPassDescriptor,
    ResourceOptions, ScissorRect, Size, StoreAction, Texture, TextureDescriptor,
};
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage};
use std::num::NonZeroU64;
use std::ops::Range;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const SIDE: u64 = 64;
const ROW: u64 = 256;

/// The test's log: every write it is told of, stamped, with its writer (0 the consumer's).
#[derive(Default)]
struct Log {
    next: AtomicU64,
    writes: Mutex<Vec<(u64, Range<u64>, u64)>>,
    /// The wait answers `false` while set.
    unanswered: AtomicBool,
    /// The stamp is not published yet while set: `published` answers `false` at once, waiting
    /// for nothing.
    pending: AtomicBool,
    watches: AtomicU64,
    waits: AtomicU64,
    /// The writers `changed_except` was asked to leave out, and `wrote_as` named.
    asked: Mutex<Vec<u64>>,
    named: Mutex<Vec<u64>>,
}

impl Log {
    fn told(&self, range: Range<u64>, writer: u64) {
        let stamp = self.next.fetch_add(1, Ordering::SeqCst) + 1;
        self.writes.lock().unwrap().push((stamp, range, writer));
    }
}

impl DirtyPages for Log {
    type Watch = ();

    fn watch(&self, _range: Range<u64>) {
        self.watches.fetch_add(1, Ordering::SeqCst);
    }

    fn request(&self) -> DirtyStamp {
        DirtyStamp(self.next.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn wait(&self, _stamp: DirtyStamp) -> bool {
        self.waits.fetch_add(1, Ordering::SeqCst);
        !self.unanswered.load(Ordering::SeqCst)
    }

    fn published(&self, stamp: DirtyStamp) -> bool {
        !self.pending.load(Ordering::SeqCst) && self.wait(stamp)
    }

    fn changed(&self, since: DirtyStamp, out: &mut Vec<Range<u64>>) {
        let writes = self.writes.lock().unwrap();
        out.extend(
            (writes.iter())
                .filter(|w| w.0 > since.0)
                .map(|w| w.1.clone()),
        );
    }

    fn wrote(&self, range: Range<u64>) {
        self.told(range, 0);
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        self.named.lock().unwrap().push(writer.get());
        self.told(range, writer.get());
    }

    fn changed_except(&self, since: DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.asked.lock().unwrap().push(writer.get());
        let writes = self.writes.lock().unwrap();
        let changed = (writes.iter()).filter(|w| w.0 > since.0 && w.2 != writer.get());
        out.extend(changed.map(|w| w.1.clone()));
    }
}

fn state(device: &Device, kernel: &str) -> ComputePipelineState {
    let ll = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/examples/runtime_examples.ll");
    let bitcode = metal2vulkan_devtools::llvm_assemble(&ll).unwrap();
    let library = metal2vulkan::library::Library::from_air(&bitcode).unwrap();
    let library: Library = device.new_library_with_compile_library(library);
    let function = library.new_function_with_name(kernel).unwrap();
    device
        .new_compute_pipeline_state_with_function(&function, &PipelineOptions::default())
        .unwrap()
}

/// Run `kernel` over `SIDE` x `SIDE` texels from `source` into `target`, to completion.
fn run(device: &Device, kernel: &ComputePipelineState, source: &Texture, target: &Texture) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_texture(Some(source), 0).unwrap();
    encoder.set_texture(Some(target), 1).unwrap();
    encoder
        .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The first texel of row `row` of `texture`, as the kernel inverted it.
fn texel(texture: &Texture, row: u64) -> [u8; 4] {
    let mut bytes = [0u8; 4];
    (texture.get_bytes(&mut bytes, 4, 0, Region::new_2d(0, row, 1, 1), 0, 0)).unwrap();
    bytes
}

/// The counts a frame moved: (gated, whole, skipped, bytes whole, bytes copied).
fn counts(device: &Device) -> [u64; 5] {
    let c = device.work_counts();
    [
        c.dirty_copies_in_gated,
        c.dirty_copies_in_whole,
        c.dirty_copies_in_skipped,
        c.dirty_copy_in_bytes_whole,
        c.dirty_copy_in_bytes_copied,
    ]
}

fn delta(device: &Device, before: [u64; 5]) -> [u64; 5] {
    let now = counts(device);
    std::array::from_fn(|i| now[i] - before[i])
}

/// The buffer over `host`, its memory logged by `log`.
fn over_logged(device: &Device, host: &Host, log: &Arc<Log>) -> Option<Buffer> {
    // SAFETY: the memory outlives the buffer (dropped after the device), and only the buffer's
    // commands and the test use it while it lives.
    let memory: HostMemory = unsafe { host.memory() }.with_dirty_pages(Arc::clone(log));
    let made = unsafe {
        device.new_buffer_with_bytes_no_copy(memory, BYTES as u64, ResourceOptions::default(), None)
    };
    match made {
        Ok(buffer) => Some(buffer),
        Err(Error::Unsupported(gap)) => {
            eprintln!("DIRTY-PAGES: the device imports no host memory: {gap:?}");
            None
        }
        Err(error) => panic!("new_buffer_with_bytes_no_copy: {error:?}"),
    }
}

#[test]
fn a_logged_shadow_copies_in_only_the_changed_rows() {
    let scratch = Scratch::new("dirty-pages-rows");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    let span = (SIDE * ROW) as usize;
    host.fill(0x1010_1010);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let source = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures - shadows;
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    plain.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    plain.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&plain).unwrap();
    let invert = state(&device, "invert_texels");
    eprintln!("DIRTY-PAGES: {shadowed} shadow made");
    if shadowed == 0 {
        return;
    }
    let bytes = buffer.contents().unwrap().as_ptr();
    let write_row = |row: u64, value: u8| {
        // SAFETY: a row inside the buffer's first bytes, which no command uses now.
        unsafe { std::ptr::write_bytes(bytes.add((row * ROW) as usize), value, ROW as usize) };
    };
    let inverted = |value: u8| [255 - value, 255 - value, 255 - value, value];

    // Frame 1: every row.
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let first = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 1 (gated, whole, skipped, bytes whole, copied) {first:?}");
    assert_eq!(texel(&target, 10), inverted(0x10));
    assert_eq!(first[0], 1, "the first copy in is gated");
    assert_eq!(first[1], 1, "the first copy in takes every row");
    assert_eq!(first[4], (SIDE * SIDE * 4), "every byte");
    assert!(
        log.watches.load(Ordering::SeqCst) >= 1,
        "the shadow watches its bytes"
    );

    // Frame 2: row 10 told, row 40 not.
    write_row(10, 0x20);
    write_row(40, 0x30);
    log.told(10 * ROW..11 * ROW, 0);
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let second = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 2 {second:?}");
    assert_eq!(
        texel(&target, 10),
        inverted(0x20),
        "the told row was copied"
    );
    assert_eq!(texel(&target, 40), inverted(0x10), "the untold row was not");
    assert_eq!(second[0], 1);
    assert_eq!(second[1], 0, "not a whole copy");
    assert_eq!(second[4], SIDE * 4, "one row's bytes");
    assert!(
        log.waits.load(Ordering::SeqCst) >= 1,
        "the commit's stamp was waited for"
    );

    // Frame 3: nothing changed.
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let third = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 3 {third:?}");
    assert_eq!(third[2], 1, "nothing copied in");
    assert_eq!(third[4], 0);

    // Frame 4: the log cannot say: every row.
    log.unanswered.store(true, Ordering::SeqCst);
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let fourth = delta(&device, before);
    log.unanswered.store(false, Ordering::SeqCst);
    eprintln!("DIRTY-PAGES: frame 4 {fourth:?}");
    assert_eq!(fourth[1], 1, "a whole copy");
    assert_eq!(fourth[4], SIDE * SIDE * 4);
    assert_eq!(
        texel(&target, 40),
        inverted(0x30),
        "the untold row came with it"
    );

    // Frame 5: a kernel writes through the shadow; its copy back is its own.
    run(&device, &invert, &target, &source);
    let named = log.named.lock().unwrap().clone();
    eprintln!("DIRTY-PAGES: frame 5 named writers {named:?}");
    assert!(
        !named.is_empty(),
        "the copy back was reported as the shadow's"
    );
    assert!(
        named.len() >= 2 && named.iter().all(|&w| w == named[0]),
        "reported at the commit and again at the completion, by one writer"
    );
    // Frame 6: the shadow's own write is no change to it.
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let sixth = delta(&device, before);
    let asked = log.asked.lock().unwrap().clone();
    eprintln!("DIRTY-PAGES: frame 6 {sixth:?}, asked writers {asked:?}");
    assert!(
        asked.contains(&named[0]),
        "its copies in leave its own writes out"
    );
    assert_eq!(sixth[4], 0, "nothing copied in after its own copy back");
    // Frame 7: a command buffer that waits for an event a command buffer committed after it
    // signals, once its fill wrote row 50's bytes, which no log hears of: submitted after the
    // fill, it copies every row and reads the fill.
    let event = device.new_shared_event().unwrap();
    let invert_after = |value: u64| {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        (command_buffer.encode_wait_for_event(EventRef::Shared(&event), value)).unwrap();
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&invert).unwrap();
        encoder.set_texture(Some(&source), 0).unwrap();
        encoder.set_texture(Some(&target), 1).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        command_buffer.commit().unwrap();
        command_buffer
    };
    let before = counts(&device);
    let waiting = invert_after(1);
    let queue = device.new_command_queue().unwrap();
    let filler = queue.command_buffer().unwrap();
    let mut blit = filler.blit_command_encoder().unwrap();
    blit.fill_buffer(&buffer, 50 * ROW..51 * ROW, 0x50).unwrap();
    blit.end_encoding().unwrap();
    (filler.encode_signal_event(EventRef::Shared(&event), 1)).unwrap();
    filler.commit().unwrap();
    for command_buffer in [&waiting, &filler] {
        assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    let seventh = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 7 (after an event a later fill signals) {seventh:?}");
    assert_eq!(
        texel(&target, 50),
        inverted(0x50),
        "it reads the fill it waited for"
    );
    assert_eq!(seventh[1], 1, "a whole copy after a write no log hears of");
    // Frame 8: a wait for a value already reached; nothing changed since frame 7's copy.
    let before = counts(&device);
    let reached = invert_after(1);
    assert!(reached.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(reached.status(), CommandBufferStatus::Completed);
    let eighth = delta(&device, before);
    eprintln!("DIRTY-PAGES: frame 8 (after an event already reached) {eighth:?}");
    assert_eq!(eighth[2], 1, "nothing copied in");
    assert_eq!(texel(&target, 50), inverted(0x50));
    // Frame 9: two batches, the first writing through the shadow.
    let split = queue.command_buffer().unwrap();
    for (from, into) in [(&target, &source), (&source, &target)] {
        let mut encoder = split.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(&invert).unwrap();
        encoder.set_texture(Some(from), 0).unwrap();
        encoder.set_texture(Some(into), 1).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
        if std::ptr::eq(into, &source) {
            (split.encode_wait_for_event(EventRef::Shared(&event), 1)).unwrap();
        }
    }
    split.commit().unwrap();
    assert!(split.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(split.status(), CommandBufferStatus::Completed);
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let ninth = delta(&device, before);
    let before = counts(&device);
    run(&device, &invert, &source, &target);
    let tenth = delta(&device, before);
    eprintln!("DIRTY-PAGES: frames after a split command buffer's write {ninth:?} {tenth:?}");
    assert_eq!(
        ninth[1], 1,
        "every row after a write no hold could count on"
    );
    assert_eq!(tenth[2], 1, "then nothing: the write's hold is settled");
    // SAFETY: every command completed; nothing writes the bytes now.
    let now = unsafe { std::slice::from_raw_parts(bytes, span) };
    assert_eq!(now[10 * ROW as usize], 0x20);
    drop(source);
    drop(buffer);
    drop(device);
}

/// One pass over `target` that loads it and draws `color` over every pixel inside rows 0 to 8 (the
/// scissor), to completion.
fn draw_in_scissor(device: &Device, target: &Texture, color: [f32; 4]) {
    let state = render_state(device, "engine/tob_renders.ll", "tob_rect", "tob_solid");
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut pass = RenderPassDescriptor::new();
    let attachment = &mut pass.color_attachments[0];
    attachment.texture = Some(target.clone());
    attachment.load_action = LoadAction::Load;
    attachment.store_action = StoreAction::Store;
    let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
    encoder.set_render_pipeline_state(&state).unwrap();
    let scissor = ScissorRect {
        x: 0,
        y: 0,
        width: SIDE,
        height: 8,
    };
    encoder.set_scissor_rect(scissor).unwrap();
    let corners = bytes_of(&[-1.0f32, -1.0, 1.0, 1.0], f32::to_le_bytes);
    encoder.set_vertex_bytes(&corners, 0).unwrap();
    let color = bytes_of(&color, f32::to_le_bytes);
    encoder.set_fragment_bytes(&color, 0).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 6, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn a_logged_render_target_loaded_in_a_scissor_copies_only_the_changed_rows() {
    let scratch = Scratch::new("dirty-pages-scissor");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    host.fill(0x1010_1010);
    let shadows = device.work_counts().shadowed_textures;
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let shadowed = device.work_counts().shadowed_textures - shadows;
    eprintln!("DIRTY-PAGES: {shadowed} render shadow made");
    if shadowed == 0 {
        return;
    }
    let bytes = buffer.contents().unwrap().as_ptr();
    let row_byte = |row: u64| {
        // SAFETY: a byte inside the buffer's first rows, which no command uses now.
        unsafe { *bytes.add((row * ROW) as usize) }
    };

    // Frame 1: every texel, the image holding all of them after it.
    let before = counts(&device);
    draw_in_scissor(&device, &target, [1.0, 0.0, 0.0, 1.0]);
    let first = delta(&device, before);
    eprintln!(
        "DIRTY-PAGES: scissor frame 1 (gated, whole, skipped, bytes whole, copied) {first:?}"
    );
    assert_eq!(
        first[0], 1,
        "the first copy in is gated, not cut to the scissor"
    );
    assert_eq!(first[4], SIDE * SIDE * 4, "every texel the first time");
    assert_eq!(row_byte(0), 255, "the pass drew row 0");
    assert_eq!(row_byte(20), 0x10, "and left row 20");

    // Frame 2: row 40 told to the log; the pass's own copy back of rows 0 to 8 is its own.
    // SAFETY: a row inside the buffer's first bytes, which no command uses now.
    unsafe { std::ptr::write_bytes(bytes.add((40 * ROW) as usize), 0x40, ROW as usize) };
    log.told(40 * ROW..41 * ROW, 0);
    let before = counts(&device);
    draw_in_scissor(&device, &target, [0.0, 1.0, 0.0, 1.0]);
    let second = delta(&device, before);
    eprintln!("DIRTY-PAGES: scissor frame 2 {second:?}");
    assert_eq!(second[0], 1, "gated again");
    assert_eq!(second[1], 0, "not a whole copy");
    assert_eq!(second[4], SIDE * 4, "the told row alone");
    assert_eq!(row_byte(40), 0x40, "the CPU's row stays");
    assert_eq!(row_byte(0), 0, "the second pass drew row 0");

    // Frame 3: nothing changed but the pass's own copy back.
    let before = counts(&device);
    draw_in_scissor(&device, &target, [0.0, 0.0, 1.0, 1.0]);
    let third = delta(&device, before);
    eprintln!("DIRTY-PAGES: scissor frame 3 {third:?}");
    assert_eq!(third[2], 1, "nothing copied in");
    drop(target);
    drop(buffer);
    drop(device);
}

/// The stamps asked, the waits made and the waits answered `false` so far.
fn waits(device: &Device) -> [u64; 3] {
    let c = device.work_counts();
    [
        c.dirty_stamps_asked,
        c.dirty_waits,
        c.dirty_waits_unanswered,
    ]
}

/// One command buffer whose encoders each run `kernel` from one of `sources` into `target`, to
/// completion.
fn run_each(
    device: &Device,
    kernel: &ComputePipelineState,
    sources: &[&Texture],
    target: &Texture,
) {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    for source in sources {
        let mut encoder = command_buffer.compute_command_encoder().unwrap();
        encoder.set_compute_pipeline_state(kernel).unwrap();
        encoder.set_texture(Some(*source), 0).unwrap();
        encoder.set_texture(Some(target), 1).unwrap();
        encoder
            .dispatch_threads(Size::new(SIDE, SIDE, 1), Size::new(8, 8, 1))
            .unwrap();
        encoder.end_encoding().unwrap();
    }
    command_buffer.commit().unwrap();
    assert!(command_buffer.wait_until_completed_for(WAIT).is_ok());
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

#[test]
fn one_wait_serves_every_gated_copy_in_of_a_commit() {
    let scratch = Scratch::new("dirty-pages-waits");
    let host = Host::new();
    let device = device(&scratch.0);
    eprintln!("DIRTY-PAGES: device {:?}", device.name());
    let log = Arc::new(Log::default());
    let Some(buffer) = over_logged(&device, &host, &log) else {
        return;
    };
    host.fill(0x1010_1010);
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let shadows = device.work_counts().shadowed_textures;
    let first = buffer.new_texture(&descriptor, 0, ROW).unwrap();
    let second = buffer.new_texture(&descriptor, SIDE * ROW, ROW).unwrap();
    if device.work_counts().shadowed_textures - shadows < 2 {
        eprintln!("DIRTY-PAGES: no shadows made");
        return;
    }
    let mut plain = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, SIDE, SIDE, false);
    plain.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    plain.resource_options = ResourceOptions::shared();
    let target = device.new_texture(&plain).unwrap();
    let invert = state(&device, "invert_texels");
    let sources = [&first, &second];

    // Frame 1: no hold, so no wait.
    let before = waits(&device);
    run_each(&device, &invert, &sources, &target);
    let [stamps, waited, _] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    eprintln!("DIRTY-PAGES: waits frame 1: {stamps} stamps, {waited} waits");
    assert_eq!(stamps, 1, "one stamp per log and commit");
    assert_eq!(waited, 0, "no hold, no wait");

    // Frame 2: two gated copies over one log, one stamp and one wait.
    let before = waits(&device);
    let gated = counts(&device);
    run_each(&device, &invert, &sources, &target);
    let [stamps, waited, _] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    let gated = delta(&device, gated);
    eprintln!("DIRTY-PAGES: waits frame 2: {stamps} stamps, {waited} waits, copies {gated:?}");
    assert_eq!(gated[0], 2, "two gated copies in");
    assert_eq!(gated[2], 2, "neither moved a byte");
    assert_eq!(stamps, 1);
    assert_eq!(waited, 1, "one wait for both");

    // Frame 3: the log cannot say: one wait, both copies whole.
    log.unanswered.store(true, Ordering::SeqCst);
    let before = waits(&device);
    let gated = counts(&device);
    run_each(&device, &invert, &sources, &target);
    log.unanswered.store(false, Ordering::SeqCst);
    let [stamps, waited, unanswered] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    let gated = delta(&device, gated);
    eprintln!("DIRTY-PAGES: waits frame 3: {stamps} stamps, {waited} waits ({unanswered} false)");
    assert_eq!(gated[1], 2, "both copies whole");
    assert_eq!(
        (stamps, waited, unanswered),
        (1, 1, 1),
        "one unanswered wait for both"
    );

    // Frame 4: the stamp not published yet, said at once: both copies whole, and no wait.
    log.pending.store(true, Ordering::SeqCst);
    let waits_before = log.waits.load(Ordering::SeqCst);
    let before = waits(&device);
    let gated = counts(&device);
    run_each(&device, &invert, &sources, &target);
    log.pending.store(false, Ordering::SeqCst);
    let [_, asked, unanswered] = std::array::from_fn(|i| waits(&device)[i] - before[i]);
    let gated = delta(&device, gated);
    eprintln!("DIRTY-PAGES: waits frame 4: {asked} asked ({unanswered} false), copies {gated:?}");
    assert_eq!(gated[1], 2, "both copies whole");
    assert_eq!((asked, unanswered), (1, 1), "one answer for both");
    assert_eq!(
        log.waits.load(Ordering::SeqCst),
        waits_before,
        "the log's wait never ran"
    );
    drop((first, second));
    drop(buffer);
    drop(device);
}
