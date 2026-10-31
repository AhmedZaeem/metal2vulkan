//! A command queue that shares its `VkQueue` with another moves off it while idle, where the other
//! queue's work is in flight there (`Device::idle_queue_moves`, `WorkCounts::idle_queues_moved`):
//! its next command buffer runs on a round robin `VkQueue` with none in flight, and waits for none
//! of the busy queue's long command buffer. Where nothing is in flight on its `VkQueue`, and where
//! the caller used that `VkQueue` (`CommandQueue::with_vk_queue`), it stays.
//!
//! # Why
//!
//! A guest's frames, drawn by a queue that submits a frame at a time, waited in order behind the
//! long command buffers of a busier queue on the `VkQueue` they shared (`queue.rs`): a program with
//! more live command queues than the device's round robin has `VkQueue`s shares some, and which
//! ones turn busy is known only once they run. The GPU itself keeps two `VkQueue`s apart: measured
//! on an RTX A5000 Laptop (NVIDIA 615.78.08), an 8 MiB dispatch on one `VkQueue` completed in
//! 1.1 ms at the median and 3.2 ms at worst while another streamed 10.5 GB/s over PCIe in command
//! buffers of 51 ms.
//!
//! The assertions are counts and `VkQueue` indices, never times: the busy command buffer is long
//! enough (256 fills of 256 MiB) to be in flight as the idle queue makes its command buffer on any
//! GPU, and is waited for at the end. A device whose round robin has one `VkQueue` (lavapipe's,
//! RADV's) has nowhere to move to, and the run checks that nothing moves there.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch, WAIT};
use crate::{Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, ResourceOptions};

/// The busy command buffer's fills, and the bytes each fills.
const FILLS: u32 = 256;
const FILLED: u64 = 256 << 20;

/// Command queues made until the last shares a `VkQueue` with an earlier one: the earlier ones,
/// one per round robin `VkQueue`, the last, and the index of the earlier one it shares with.
fn sharing_pair(device: &Device) -> (Vec<CommandQueue>, CommandQueue, usize) {
    let sharing = || device.work_counts().queues_sharing_a_vk_queue;
    let mut kept: Vec<CommandQueue> = Vec::new();
    while sharing() == 0 {
        assert!(
            kept.len() <= 16,
            "a device shares a `VkQueue` by its 17th command queue"
        );
        kept.push(device.new_command_queue().unwrap());
    }
    let late = kept.pop().unwrap();
    let partner = (kept.iter())
        .position(|queue| queue.vk_queue() == late.vk_queue())
        .expect("the queue past the device's `VkQueue`s shares an earlier queue's");
    (kept, late, partner)
}

/// A command buffer of `queue`'s that fills `target` [`FILLS`] times, committed.
fn long(queue: &CommandQueue, target: &Buffer) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for value in 0..FILLS {
        blit.fill_buffer(target, 0..FILLED, value as u8).unwrap();
    }
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
}

/// A command buffer of `queue`'s with one small fill, committed and waited for.
fn short(queue: &CommandQueue, target: &Buffer) {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(target, 0..4096, 0x5a).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// Wait until work is in flight on the device's `VkQueue` at `at`, as the host sees it.
fn in_flight(device: &Device, at: usize) {
    let deadline = std::time::Instant::now() + WAIT;
    while !device.work_in_flight(at) {
        assert!(
            std::time::Instant::now() < deadline,
            "the long batch was never submitted"
        );
        std::thread::yield_now();
    }
}

/// An idle queue on a `VkQueue` with nothing in flight stays; on one where its partner's long
/// command buffer is in flight, its next command buffer moves it, once, and it runs there.
#[test]
fn an_idle_queue_leaves_a_shared_vk_queue_with_work_in_flight() {
    let scratch = Scratch::new("idle-queue-moves");
    let device = device(&scratch.0);
    let moved = || device.work_counts().idle_queues_moved;
    let (kept, late, partner) = sharing_pair(&device);
    let big = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let shared = late.vk_queue();
    // Nothing in flight on the shared `VkQueue` once its partner's command buffer completed, the
    // device's own earlier work there with it: the idle queue stays.
    short(&kept[partner], &small);
    short(&late, &small);
    assert_eq!(moved(), 0, "nothing was in flight, so nothing moved");
    assert_eq!(late.vk_queue(), shared);
    // Its partner's long command buffer in flight there: the idle queue's next command buffer
    // moves it, where the round robin has another `VkQueue`.
    let busy = long(&kept[partner], &big);
    in_flight(&device, shared);
    short(&late, &small);
    if kept.len() == 1 {
        assert_eq!(
            moved(),
            0,
            "a round robin of one `VkQueue` has nowhere to move to"
        );
        assert_eq!(late.vk_queue(), shared);
    } else {
        assert_ne!(
            busy.status(),
            CommandBufferStatus::Completed,
            "the long command buffer must still run as the idle queue made its command buffer; \
             raise FILLS"
        );
        assert_eq!(
            moved(),
            1,
            "the idle queue moved off its partner's busy `VkQueue`, once"
        );
        assert_ne!(
            late.vk_queue(),
            shared,
            "the idle queue runs on another `VkQueue` now"
        );
        assert_eq!(
            kept[partner].vk_queue(),
            shared,
            "the busy queue keeps its `VkQueue`"
        );
        assert!(
            kept.iter().any(|queue| queue.vk_queue() == late.vk_queue()),
            "it moved to another of the round robin's `VkQueue`s, never a presenting queue's"
        );
    }
    busy.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(busy.status(), CommandBufferStatus::Completed);
    // Dropping every queue, and the command buffer that holds its partner, gives each place back:
    // as many new queues share none.
    let sharing = device.work_counts().queues_sharing_a_vk_queue;
    let own = kept.len();
    drop((busy, kept, late));
    let fresh: Vec<CommandQueue> = (0..own)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    assert_eq!(
        device.work_counts().queues_sharing_a_vk_queue,
        sharing,
        "{own} queues made once the others went share none: the move kept the places' count"
    );
    drop(fresh);
}

/// A queue whose `VkQueue` the caller used stays on it, though its partner's work is in flight.
#[test]
fn a_queue_the_caller_used_never_moves() {
    let scratch = Scratch::new("idle-queue-pinned");
    let device = device(&scratch.0);
    let (kept, late, partner) = sharing_pair(&device);
    let big = device
        .new_buffer(FILLED, ResourceOptions::private())
        .unwrap();
    let small = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let shared = late.vk_queue();
    // SAFETY: the closure keeps no handle and calls nothing.
    unsafe { late.with_vk_queue(|_, _| ()) };
    let busy = long(&kept[partner], &big);
    in_flight(&device, shared);
    short(&late, &small);
    assert_eq!(
        device.work_counts().idle_queues_moved,
        0,
        "a pinned queue never moves"
    );
    assert_eq!(late.vk_queue(), shared);
    busy.wait_until_completed_for(WAIT).unwrap();
}
