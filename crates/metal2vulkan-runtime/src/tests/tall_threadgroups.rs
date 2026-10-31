//! A threadgroup past the device's `maxComputeWorkGroupSize` on an axis, whose threads fit
//! `maxComputeWorkGroupInvocations`, runs on the kernel's flat-local variant
//! (`VariantKey::flat_local`): its Vulkan workgroup is `(x·y·z, 1, 1)` and every builtin derives
//! from the linear local index, as Metal answers it. Metal takes any threadgroup of up to 1024
//! threads, whatever its shape; an NVIDIA GPU's workgroup is at most 64 deep, so before the variant
//! a kernel dispatched 128 deep died at the bind (`LocalSize { axis: Some(Z) }`).
//!
//! The kernel stores every kernel builtin per thread, and the test computes Metal's answer for
//! each: a threadgroup's threads are linear `x` fastest, its simdgroups and quadgroups are runs of
//! 32 and 4 of them, and a `dispatchThreads` edge threadgroup reports its own size.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The function is the owned fixture
//! `validation/fixtures/public/engine/gsm_global_builtins.ll`'s `global_builtins_in_a_kernel`
//! (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::library;
use crate::{
    CommandBufferStatus, ComputePipelineState, Device, PipelineOptions, ResourceOptions, Size,
};
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};

/// The words each thread stores.
const WORDS: usize = 29;

/// How a case dispatches: threadgroups of `threads` each, or a grid of threads in threadgroups.
#[derive(Clone, Copy, Debug)]
enum How {
    Threadgroups { groups: [u32; 3], threads: [u32; 3] },
    Threads { grid: [u32; 3], threads: [u32; 3] },
}

fn state(device: &Device) -> ComputePipelineState {
    let function = library(device, "engine/gsm_global_builtins.ll")
        .new_function_with_name("global_builtins_in_a_kernel")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// The grid's threads of `how`.
fn grid(how: How) -> [u32; 3] {
    match how {
        How::Threadgroups { groups, threads } => [0, 1, 2].map(|i| groups[i] * threads[i]),
        How::Threads { grid, .. } => grid,
    }
}

/// The words the device stored for `how`.
fn run(device: &Device, state: &ComputePipelineState, how: How) -> Vec<u32> {
    let g = grid(how);
    let n = (g[0] * g[1] * g[2]) as u64;
    let out = device
        .new_buffer(n * WORDS as u64 * 4, ResourceOptions::default())
        .unwrap();
    let size = |v: [u32; 3]| Size::new(v[0].into(), v[1].into(), v[2].into());
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(state).unwrap();
    encoder.set_buffer(Some(&out), 0, 0).unwrap();
    match how {
        How::Threadgroups { groups, threads } => encoder
            .dispatch_threadgroups(size(groups), size(threads))
            .unwrap(),
        How::Threads { grid, threads } => {
            encoder.dispatch_threads(size(grid), size(threads)).unwrap()
        }
    }
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(
        command_buffer.status(),
        CommandBufferStatus::Completed,
        "{how:?}"
    );
    let contents = out.contents().unwrap().as_ptr().cast::<u32>();
    // SAFETY: the buffer's words, which no command writes now.
    unsafe { std::slice::from_raw_parts(contents, n as usize * WORDS) }.to_vec()
}

/// Metal's words for `how`, thread by thread of the grid (`x` fastest).
fn metal(how: How) -> Vec<u32> {
    let (g, l) = match how {
        How::Threadgroups { threads, .. } | How::Threads { threads, .. } => (grid(how), threads),
    };
    let mut out = Vec::new();
    for z in 0..g[2] {
        for y in 0..g[1] {
            for x in 0..g[0] {
                let p = [x, y, z];
                let group = [0, 1, 2].map(|i| p[i] / l[i]);
                let local = [0, 1, 2].map(|i| p[i] % l[i]);
                // An edge threadgroup of `dispatchThreads` holds the threads left.
                let actual = [0, 1, 2].map(|i| l[i].min(g[i] - group[i] * l[i]));
                let n = actual[0] * actual[1] * actual[2];
                let index = local[0] + actual[0] * (local[1] + actual[1] * local[2]);
                let groups = [0, 1, 2].map(|i| g[i].div_ceil(l[i]));
                let nominal = l[0] * l[1] * l[2];
                out.extend(actual);
                out.extend(local);
                out.extend([index, index % 32, index / 32, n.div_ceil(32)]);
                out.extend([index % 4, index / 4, n.div_ceil(4)]);
                out.extend(p);
                out.extend(group);
                out.extend(g);
                out.extend(groups);
                out.extend(l);
                out.push(nominal.div_ceil(32));
            }
        }
    }
    out
}

/// The builtins' names, by word.
const NAMES: [&str; WORDS] = [
    "threads_per_threadgroup.x",
    "threads_per_threadgroup.y",
    "threads_per_threadgroup.z",
    "thread_position_in_threadgroup.x",
    "thread_position_in_threadgroup.y",
    "thread_position_in_threadgroup.z",
    "thread_index_in_threadgroup",
    "thread_index_in_simdgroup",
    "simdgroup_index_in_threadgroup",
    "simdgroups_per_threadgroup",
    "thread_index_in_quadgroup",
    "quadgroup_index_in_threadgroup",
    "quadgroups_per_threadgroup",
    "thread_position_in_grid.x",
    "thread_position_in_grid.y",
    "thread_position_in_grid.z",
    "threadgroup_position_in_grid.x",
    "threadgroup_position_in_grid.y",
    "threadgroup_position_in_grid.z",
    "threads_per_grid.x",
    "threads_per_grid.y",
    "threads_per_grid.z",
    "threadgroups_per_grid.x",
    "threadgroups_per_grid.y",
    "threadgroups_per_grid.z",
    "dispatch_threads_per_threadgroup.x",
    "dispatch_threads_per_threadgroup.y",
    "dispatch_threads_per_threadgroup.z",
    "dispatch_simdgroups_per_threadgroup",
];

/// Threadgroups 128 deep, past an NVIDIA GPU's 64, and one of 1024 threads, uniform and with
/// `dispatchThreads` edges, answer every builtin as Metal does; where the device's limit passes
/// them, they run on the flat-local variant.
#[test]
fn a_threadgroup_past_an_axis_limit_answers_every_builtin_as_metal() {
    let scratch = Scratch::new("tall-threadgroups");
    let device = device(&scratch.0);
    let state = state(&device);
    eprintln!("TALL-THREADGROUPS: {}", device.name());
    let cases = [
        How::Threadgroups {
            groups: [1, 1, 1],
            threads: [4, 4, 4],
        },
        How::Threadgroups {
            groups: [1, 1, 2],
            threads: [1, 1, 128],
        },
        How::Threadgroups {
            groups: [2, 1, 1],
            threads: [2, 4, 128],
        },
        How::Threads {
            grid: [3, 5, 200],
            threads: [2, 4, 128],
        },
        How::Threads {
            grid: [1, 1, 300],
            threads: [1, 1, 100],
        },
    ];
    for how in cases {
        let before = device.work_counts().variant_pipelines_made;
        let got = run(&device, &state, how);
        let made = device.work_counts().variant_pipelines_made - before;
        let want = metal(how);
        let wrong: Vec<String> = got
            .chunks(WORDS)
            .zip(want.chunks(WORDS))
            .enumerate()
            .flat_map(|(t, (g, w))| {
                (0..WORDS)
                    .filter(move |&i| g[i] != w[i])
                    .map(move |i| format!("thread {t} {}: {} for {}", NAMES[i], g[i], w[i]))
            })
            .take(12)
            .collect();
        eprintln!("TALL-THREADGROUPS: {how:?}: {made} variant pipelines made");
        assert!(wrong.is_empty(), "{how:?}: {wrong:#?}");
    }
}
