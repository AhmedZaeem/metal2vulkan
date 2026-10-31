//! A sample through a runtime sampler that is normalized, nearest, clamped to the edge and reads
//! one level reads the texel `clamp(floor(coord * size))` by a fetch (the translator's
//! `Sampling::exact`), on every device: the texel past a texel edge, as Metal reads it. A local
//! deviation from the device's own sampling (docs/FAST-PATHS.md, the device sampler row).
//!
//! Why: JetBrains Runtime's Java2D Metal mask fill draws each fully covered tile of an
//! antialiased fill from one opaque texel of its `A8Unorm` mask cache (256 x 160), at (224, 128),
//! every vertex's coordinate at that texel's top left corner (224 / 256, 128 / 160), through a
//! nearest, clamp-to-edge runtime sampler. Measured on the NVIDIA RTX A5000 Laptop GPU, its
//! sampler reads (224, 127) there, an unused and transparent texel, and at the edge `y / 160` (its
//! `f32` value, and one `f32` ulp above it) it reads row `y - 1` for every edge but the multiples
//! of 5: the tiles drew nothing. A kernel whose coordinate comes from a buffer read
//! the same rows as a draw, so the interpolation of the constant coordinate is exact.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`), on the kind
//! of device the cfg selects. The functions are the owned fixtures
//! `validation/fixtures/public/reflection/render.ll`'s `reflection_vertex` and
//! `reflection_fragment`, and `engine/mip_level_sampling.ll`'s `mip_bias_kernel` (`llvm-as`).

use super::blit_holds::{device, Scratch, WAIT};
use super::encoder_holds::{library, pass};
use crate::{
    BlitOption, CommandBufferStatus, Device, Origin, PipelineOptions, Region, RenderPipelineState,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage, VertexFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, ComputePipelineDescriptor, RenderPipelineDescriptor, StageSource,
    VertexAttributeDescriptor, VertexBufferLayoutDescriptor, VertexDescriptor,
};

/// Java2D's mask cache.
const WIDTH: u64 = 256;
const HEIGHT: u64 = 160;

/// The opaque texel's corner, as Java2D computes it.
const CORNER: [f32; 2] = [224.0 / 256.0, 128.0 / 160.0];

/// The texels around the corner and their values: the opaque one and its three neighbours.
const AROUND: [(u64, u64, u8); 4] = [
    (224, 128, 0xff),
    (223, 128, 0x40),
    (224, 127, 0x80),
    (223, 127, 0xc0),
];

/// A state of `reflection_vertex` and `reflection_fragment` into `RGBA8Unorm`: each vertex's
/// position (`float4`), coordinate (`float2`) and colour (`half4`) in 32-byte records of buffer 0.
/// The fragment returns its texture's sample through sampler 0 times the tint in buffer 0, plus
/// the colour.
fn state(device: &Device) -> RenderPipelineState {
    let library = library(device, "reflection/render.ll");
    let vertex = library.new_function_with_name("reflection_vertex").unwrap();
    let fragment = library
        .new_function_with_name("reflection_fragment")
        .unwrap();
    let mut descriptor =
        RenderPipelineDescriptor::new(StageSource::Function(vertex.function_descriptor()));
    descriptor.fragment = Some(StageSource::Function(fragment.function_descriptor()));
    descriptor.color_attachments[0] = Some(ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm));
    let mut input = VertexDescriptor::new();
    let formats = [
        (VertexFormat::Float4, 0),
        (VertexFormat::Float2, 16),
        (VertexFormat::Half4, 24),
    ];
    for (i, (format, offset)) in formats.into_iter().enumerate() {
        input
            .attributes
            .insert(i as u32, VertexAttributeDescriptor::new(format, offset, 0));
    }
    input
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
    descriptor.vertex_descriptor = Some(input);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A mask cache as Java2D takes it from its pool: `A8Unorm`, render target and shader read.
fn mask(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::A8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    device.new_texture(&descriptor).unwrap()
}

/// A nearest, clamp-to-edge, normalized sampler, as Java2D binds it.
fn nearest(device: &Device) -> crate::SamplerState {
    device.new_sampler_state(&SamplerDescriptor::new()).unwrap()
}

/// `values` in a new buffer.
fn buffer_of(device: &Device, values: &[u8]) -> crate::Buffer {
    let buffer = device
        .new_buffer(values.len().max(4) as u64, ResourceOptions::default())
        .unwrap();
    // SAFETY: inside the buffer's length; no command uses it yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            buffer.contents().unwrap().as_ptr(),
            values.len(),
        )
    };
    buffer
}

/// `floats` as little-endian bytes.
fn bytes_of(floats: &[f32]) -> Vec<u8> {
    floats.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Commits `command_buffer` and waits for it.
fn run(command_buffer: &crate::CommandBuffer) {
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// The alpha code a draw reads at each coordinate: one pixel of an `n` x 1 target a coordinate,
/// drawn by a quad whose every vertex carries it.
fn drawn(
    device: &Device,
    state: &RenderPipelineState,
    texture: &Texture,
    coords: &[[f32; 2]],
) -> Vec<u8> {
    let n = coords.len() as u64;
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, n, 1, false);
    target.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = device.new_texture(&target).unwrap();
    // The vertex function adds a sample of its texture to the position: one of zero.
    let mut zero = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
    zero.usage = TextureUsage::SHADER_READ;
    let zero = device.new_texture(&zero).unwrap();
    zero.replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0)
        .unwrap();
    let mut vertices = Vec::new();
    for (i, [u, v]) in coords.iter().enumerate() {
        let left = i as f32 / n as f32 * 2.0 - 1.0;
        let right = (i + 1) as f32 / n as f32 * 2.0 - 1.0;
        let corners = [
            [left, 1.0],
            [right, 1.0],
            [right, -1.0],
            [right, -1.0],
            [left, -1.0],
            [left, 1.0],
        ];
        for [x, y] in corners {
            vertices.extend(bytes_of(&[x, y, 0.0, 1.0, *u, *v]));
            vertices.extend([0u8; 8]);
        }
    }
    let vertices = buffer_of(device, &vertices);
    let extra = buffer_of(device, &vec![0u8; (n * 6 * 4) as usize]);
    let identity: Vec<f32> = (0..16)
        .map(|i| if i % 5 == 0 { 1.0 } else { 0.0 })
        .collect();
    let count = device.new_buffer(4, ResourceOptions::default()).unwrap();
    let sampler = nearest(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_buffer(Some(&vertices), 0, 0).unwrap();
    encoder.set_vertex_bytes(&bytes_of(&identity), 1).unwrap();
    encoder.set_vertex_buffer(Some(&extra), 0, 3).unwrap();
    encoder.set_vertex_texture(Some(&zero), 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(&sampler), 0)
        .unwrap();
    encoder.set_fragment_bytes(&bytes_of(&[1.0; 4]), 0).unwrap();
    encoder.set_fragment_buffer(Some(&count), 0, 2).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, (n * 6) as _, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    let mut read = vec![0u8; (n * 4) as usize];
    target
        .get_bytes(&mut read, n * 4, 0, Region::new_2d(0, 0, n, 1), 0, 0)
        .unwrap();
    read.chunks(4).map(|texel| texel[3]).collect()
}

/// The alpha code a kernel reads at each coordinate, which it takes from a buffer
/// (`mip_bias_kernel`, at bias 0).
fn fetched_by_kernel(device: &Device, texture: &Texture, coords: &[[f32; 2]]) -> Vec<u8> {
    let function = library(device, "engine/mip_level_sampling.ll")
        .new_function_with_name("mip_bias_kernel")
        .unwrap();
    let descriptor =
        ComputePipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let n = coords.len() as u64;
    let params: Vec<f32> = coords
        .iter()
        .flat_map(|[u, v]| [*u, *v, 0.0, 0.0])
        .collect();
    let params = buffer_of(device, &bytes_of(&params));
    let out = device
        .new_buffer(n * 16, ResourceOptions::default())
        .unwrap();
    let sampler = nearest(device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(&state).unwrap();
    encoder.set_texture(Some(texture), 0).unwrap();
    encoder.set_sampler_state(Some(&sampler), 0).unwrap();
    encoder.set_buffer(Some(&params), 0, 0).unwrap();
    encoder.set_buffer(Some(&out), 0, 1).unwrap();
    encoder
        .dispatch_threads(Size::new(n, 1, 1), Size::new(1, 1, 1))
        .unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    let contents = out.contents().unwrap().as_ptr().cast::<f32>();
    (0..coords.len())
        // SAFETY: inside the buffer, which no command writes now.
        .map(|i| (unsafe { *contents.add(i * 4 + 3) } * 255.0).round() as u8)
        .collect()
}

/// The texels of `AROUND` in `texture`, as `getBytes` reads them.
fn around(texture: &Texture) -> Vec<u8> {
    AROUND
        .iter()
        .map(|&(x, y, _)| {
            let mut texel = [0u8; 1];
            texture
                .get_bytes(&mut texel, 1, 0, Region::new_2d(x, y, 1, 1), 0, 0)
                .unwrap();
            texel[0]
        })
        .collect()
}

/// Java2D's fully covered tile reads its opaque texel at the texel's corner, whether the texel
/// came by `replaceRegion` (Java2D's call) or by a copy of the whole texture from a buffer (how a
/// paravirtualized guest's texture is paged in).
#[test]
fn a_full_tile_of_java2ds_mask_fill_reads_its_opaque_texel_at_the_corner() {
    let scratch = Scratch::new("nearest-texel-corner");
    let device = device(&scratch.0);
    let state = state(&device);
    eprintln!("NEAREST-EDGES: {}", device.name());
    let want: Vec<u8> = AROUND.iter().map(|&(_, _, value)| value).collect();

    let replaced = mask(&device);
    for &(x, y, value) in &AROUND {
        replaced
            .replace_region(Region::new_2d(x, y, 1, 1), 0, 0, &[value], 1, 0)
            .unwrap();
    }
    assert_eq!(around(&replaced), want, "the texels replaceRegion wrote");
    assert_eq!(
        drawn(&device, &state, &replaced, &[CORNER]),
        [0xff],
        "the corner (224 / 256, 128 / 160) names texel (224, 128)"
    );

    let mut bytes = vec![0u8; (WIDTH * HEIGHT) as usize];
    for &(x, y, value) in &AROUND {
        bytes[(y * WIDTH + x) as usize] = value;
    }
    let source = buffer_of(&device, &bytes);
    let copied = mask(&device);
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        WIDTH,
        WIDTH * HEIGHT,
        Size::new(WIDTH, HEIGHT, 1),
        &copied,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(around(&copied), want, "the texels the copy wrote");
    assert_eq!(drawn(&device, &state, &copied, &[CORNER]), [0xff]);
}

/// At every row edge `y / 160` of the mask cache, and one `f32` ulp above it, a nearest sample
/// reads row `y`, `floor(v * 160)` in `f32`, from a draw and from a kernel alike; at each row's
/// centre too.
#[test]
fn a_nearest_sample_at_a_texel_edge_reads_the_texel_past_it() {
    let scratch = Scratch::new("nearest-texel-edges");
    let device = device(&scratch.0);
    let state = state(&device);
    // Column 224 holds each texel's row, column 223 a value no row has.
    let texture = mask(&device);
    let mut texels = vec![0u8; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        texels[(y * WIDTH + 224) as usize] = y as u8;
        texels[(y * WIDTH + 223) as usize] = 200;
    }
    texture
        .replace_region(Region::new_2d(0, 0, WIDTH, HEIGHT), 0, 0, &texels, WIDTH, 0)
        .unwrap();
    let u = 224.0 / WIDTH as f32;
    let mut cases: Vec<(&str, u64, f32)> = Vec::new();
    for y in 1..HEIGHT {
        let edge = y as f32 / HEIGHT as f32;
        cases.push(("edge", y, edge));
        cases.push(("above", y, f32::from_bits(edge.to_bits() + 1)));
    }
    for y in 0..HEIGHT {
        cases.push(("centre", y, (y as f32 + 0.5) / HEIGHT as f32));
    }
    let coords: Vec<[f32; 2]> = cases.iter().map(|&(_, _, v)| [u, v]).collect();
    let by_draw = drawn(&device, &state, &texture, &coords);
    let by_kernel = fetched_by_kernel(&device, &texture, &coords);
    let mut wrong = Vec::new();
    for (i, &(kind, y, v)) in cases.iter().enumerate() {
        for (how, read) in [("draw", by_draw[i]), ("kernel", by_kernel[i])] {
            if u64::from(read) != y {
                wrong.push(format!("{how} at {kind} {v:?} of row {y} read {read}"));
            }
        }
    }
    eprintln!("NEAREST-EDGES: {} wrong reads", wrong.len());
    assert!(wrong.is_empty(), "{wrong:#?}");
}
