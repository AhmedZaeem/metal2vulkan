//! Exploration: JetBrains Runtime's Java2D Metal mask fill draws each fully covered tile from one
//! opaque texel of its `A8Unorm` mask cache (256 x 160), at (224, 128), with every vertex's
//! texture coordinate at that texel's top left corner (224 / 256, 128 / 160) and a nearest
//! sampler. The texels around the corner hold distinct values, so the alpha drawn names the texel
//! the device picked.
//!
//! A device test: compiled only under the GPU safety cfgs (`m2v_gpu`, `m2v_cpu_icd`). The
//! functions are the owned fixture `validation/fixtures/public/reflection/render.ll`'s
//! `reflection_vertex` and `reflection_fragment` (`llvm-as`).

use super::blit_holds::{Scratch, WAIT, device};
use super::encoder_holds::{library, pass};
use crate::{
    BlitOption, CommandBufferStatus, Device, Origin, PipelineOptions, Region, RenderPipelineState,
    ResourceOptions, Size, Texture, TextureDescriptor,
};
use metal2vulkan::bind::SamplerDescriptor;
use metal2vulkan::metal::{PixelFormat, PrimitiveType, TextureUsage, VertexFormat};
use metal2vulkan::pipeline::{
    ColorAttachmentDescriptor, RenderPipelineDescriptor, StageSource, VertexAttributeDescriptor,
    VertexBufferLayoutDescriptor, VertexDescriptor,
};

const WIDTH: u64 = 256;
const HEIGHT: u64 = 160;
/// The texels around the corner and their values: the opaque one and its three neighbours.
const AROUND: [(u64, u64, u8); 4] = [
    (224, 128, 0xff),
    (223, 128, 0x40),
    (224, 127, 0x80),
    (223, 127, 0xc0),
];
/// The target: one 16 x 8 quad per case.
const TARGET_W: u64 = 48;
const TARGET_H: u64 = 8;

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
    input.attributes.insert(
        0,
        VertexAttributeDescriptor::new(VertexFormat::Float4, 0, 0),
    );
    input.attributes.insert(
        1,
        VertexAttributeDescriptor::new(VertexFormat::Float2, 16, 0),
    );
    input.attributes.insert(
        2,
        VertexAttributeDescriptor::new(VertexFormat::Half4, 24, 0),
    );
    input
        .layouts
        .insert(0, VertexBufferLayoutDescriptor::new(Some(32)));
    descriptor.vertex_descriptor = Some(input);
    device
        .new_render_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

fn mask(device: &Device) -> Texture {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::A8Unorm, WIDTH, HEIGHT, false);
    descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    device.new_texture(&descriptor).unwrap()
}

/// Six vertices of a quad over target columns `x0..x0 + 16`, every one at `uv`.
fn quad(x0: f32, uv: [f32; 2]) -> Vec<u8> {
    let left = x0 / (TARGET_W as f32) * 2.0 - 1.0;
    let right = (x0 + 16.0) / (TARGET_W as f32) * 2.0 - 1.0;
    let corners = [
        [left, 1.0],
        [right, 1.0],
        [right, -1.0],
        [right, -1.0],
        [left, -1.0],
        [left, 1.0],
    ];
    let mut bytes = Vec::new();
    for [x, y] in corners {
        for f in [x, y, 0.0, 1.0, uv[0], uv[1]] {
            bytes.extend_from_slice(&f32::to_le_bytes(f));
        }
        bytes.extend_from_slice(&[0u8; 8]);
    }
    bytes
}

/// The alpha drawn at the middle of each quad.
fn draw(device: &Device, state: &RenderPipelineState, mask: &Texture) -> Vec<u8> {
    let mut target =
        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, TARGET_W, TARGET_H, false);
    target.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = device.new_texture(&target).unwrap();
    let mut zero = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
    zero.usage = TextureUsage::SHADER_READ;
    let zero = device.new_texture(&zero).unwrap();
    zero.replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0)
        .unwrap();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let count = device.new_buffer(4, ResourceOptions::default()).unwrap();
    let corner = [224.0f32 / 256.0, 128.0f32 / 160.0];
    let center = [224.5f32 / 256.0, 128.5f32 / 160.0];
    let mut vertices = quad(0.0, corner);
    vertices.extend(quad(16.0, center));
    vertices.extend(quad(32.0, [223.5f32 / 256.0, 127.5f32 / 160.0]));
    let mut mvp = Vec::new();
    for i in 0..16 {
        let one = if i % 5 == 0 { 1.0f32 } else { 0.0 };
        mvp.extend_from_slice(&one.to_le_bytes());
    }
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder.set_vertex_bytes(&vertices, 0).unwrap();
    encoder.set_vertex_bytes(&mvp, 1).unwrap();
    encoder.set_vertex_bytes(&[0u8; 128], 3).unwrap();
    encoder.set_vertex_texture(Some(&zero), 0).unwrap();
    encoder.set_fragment_texture(Some(mask), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(&sampler), 0)
        .unwrap();
    let mut tint = Vec::new();
    for _ in 0..4 {
        tint.extend_from_slice(&1.0f32.to_le_bytes());
    }
    encoder.set_fragment_bytes(&tint, 0).unwrap();
    encoder.set_fragment_buffer(Some(&count), 0, 2).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, 18, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut read = vec![0u8; (TARGET_W * TARGET_H * 4) as usize];
    target
        .get_bytes(
            &mut read,
            TARGET_W * 4,
            0,
            Region::new_2d(0, 0, TARGET_W, TARGET_H),
            0,
            0,
        )
        .unwrap();
    [8u64, 24, 40]
        .iter()
        .map(|x| read[((4 * TARGET_W + x) * 4 + 3) as usize])
        .collect()
}

fn around(mask: &Texture) -> Vec<u8> {
    AROUND
        .iter()
        .map(|&(x, y, _)| {
            let mut b = [0u8; 1];
            mask.get_bytes(&mut b, 1, 0, Region::new_2d(x, y, 1, 1), 0, 0)
                .unwrap();
            b[0]
        })
        .collect()
}

#[test]
fn explore_the_corner_sample_of_a_mask_cache() {
    let scratch = Scratch::new("mask-cache-corner");
    let device = device(&scratch.0);
    let state = state(&device);
    eprintln!("CORNER: device {}", device.name());

    // As JetBrains Runtime writes it: one replaceRegion per texel, one byte per row.
    let replaced = mask(&device);
    for &(x, y, value) in &AROUND {
        replaced
            .replace_region(Region::new_2d(x, y, 1, 1), 0, 0, &[value], 1, 0)
            .unwrap();
    }
    eprintln!("CORNER: replaceRegion texels {:02x?}", around(&replaced));
    eprintln!(
        "CORNER: replaceRegion drawn corner/center/(223,127) {:02x?}",
        draw(&device, &state, &replaced)
    );

    // As reims-vgpu pages it: the whole texture copied from a buffer.
    let copied = mask(&device);
    let source = device
        .new_buffer(WIDTH * HEIGHT, ResourceOptions::default())
        .unwrap();
    let contents = source.contents().unwrap();
    for &(x, y, value) in &AROUND {
        // SAFETY: inside the buffer's length; no command uses it yet.
        unsafe { *contents.as_ptr().add((y * WIDTH + x) as usize) = value };
    }
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
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    eprintln!("CORNER: copied texels {:02x?}", around(&copied));
    eprintln!(
        "CORNER: copied drawn corner/center/(223,127) {:02x?}",
        draw(&device, &state, &copied)
    );
}

/// A mask whose texel (x, y) holds `y` in column 224 and 200 in column 223.
fn rows(device: &Device) -> Texture {
    let texture = mask(device);
    let mut texels = vec![0u8; (WIDTH * HEIGHT) as usize];
    for y in 0..HEIGHT {
        texels[(y * WIDTH + 224) as usize] = y as u8;
        texels[(y * WIDTH + 223) as usize] = 200;
    }
    texture
        .replace_region(Region::new_2d(0, 0, WIDTH, HEIGHT), 0, 0, &texels, WIDTH, 0)
        .unwrap();
    texture
}

/// For each `v`, the texel code a nearest sample at (224 / 256, v) reads through a kernel whose
/// coordinate comes from a buffer.
fn kernel_rows(device: &Device, texture: &Texture, vs: &[f32]) -> Vec<u8> {
    let function = library(device, "engine/mip_level_sampling.ll")
        .new_function_with_name("mip_bias_kernel")
        .unwrap();
    let descriptor = metal2vulkan::pipeline::ComputePipelineDescriptor::new(StageSource::Function(
        function.function_descriptor(),
    ));
    let state = device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap();
    let n = vs.len() as u64;
    let params = device
        .new_buffer(n * 16, ResourceOptions::default())
        .unwrap();
    let out = device
        .new_buffer(n * 16, ResourceOptions::default())
        .unwrap();
    let p = params.contents().unwrap().as_ptr().cast::<f32>();
    for (i, v) in vs.iter().enumerate() {
        // SAFETY: inside the buffer; no command uses it yet.
        unsafe {
            *p.add(i * 4) = 224.0 / 256.0;
            *p.add(i * 4 + 1) = *v;
            *p.add(i * 4 + 2) = 0.0;
            *p.add(i * 4 + 3) = 0.0;
        }
    }
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
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
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let o = out.contents().unwrap().as_ptr().cast::<f32>();
    // SAFETY: inside the buffer, which no command writes now.
    (0..vs.len())
        .map(|i| (unsafe { *o.add(i * 4 + 3) } * 255.0).round() as u8)
        .collect()
}

/// For each `v`, the texel code a draw reads at a pixel whose quad's every vertex is at
/// (224 / 256, v), one pixel a `v`.
fn drawn_rows(
    device: &Device,
    state: &RenderPipelineState,
    texture: &Texture,
    vs: &[f32],
) -> Vec<u8> {
    let n = vs.len() as u64;
    let mut target = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, n, 1, false);
    target.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    let target = device.new_texture(&target).unwrap();
    let mut zero = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 1, 1, false);
    zero.usage = TextureUsage::SHADER_READ;
    let zero = device.new_texture(&zero).unwrap();
    zero.replace_region(Region::new_2d(0, 0, 1, 1), 0, 0, &[0; 4], 4, 0)
        .unwrap();
    let sampler = device.new_sampler_state(&SamplerDescriptor::new()).unwrap();
    let count = device.new_buffer(4, ResourceOptions::default()).unwrap();
    let mut vertices = Vec::new();
    for (i, v) in vs.iter().enumerate() {
        let left = i as f32 / n as f32 * 2.0 - 1.0;
        let right = (i + 1) as f32 / n as f32 * 2.0 - 1.0;
        for [x, y] in [
            [left, 1.0],
            [right, 1.0],
            [right, -1.0],
            [right, -1.0],
            [left, -1.0],
            [left, 1.0],
        ] {
            for f in [x, y, 0.0, 1.0, 224.0 / 256.0, *v] {
                vertices.extend_from_slice(&f32::to_le_bytes(f));
            }
            vertices.extend_from_slice(&[0u8; 8]);
        }
    }
    let vertex_buffer = device
        .new_buffer(vertices.len() as u64, ResourceOptions::default())
        .unwrap();
    // SAFETY: inside the buffer; no command uses it yet.
    unsafe {
        std::ptr::copy_nonoverlapping(
            vertices.as_ptr(),
            vertex_buffer.contents().unwrap().as_ptr(),
            vertices.len(),
        )
    };
    let extra = device
        .new_buffer(n * 6 * 4, ResourceOptions::default())
        .unwrap();
    let mut mvp = Vec::new();
    for i in 0..16 {
        let one = if i % 5 == 0 { 1.0f32 } else { 0.0 };
        mvp.extend_from_slice(&one.to_le_bytes());
    }
    let mut tint = Vec::new();
    for _ in 0..4 {
        tint.extend_from_slice(&1.0f32.to_le_bytes());
    }
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer
        .render_command_encoder(&pass(&target))
        .unwrap();
    encoder.set_render_pipeline_state(state).unwrap();
    encoder
        .set_vertex_buffer(Some(&vertex_buffer), 0, 0)
        .unwrap();
    encoder.set_vertex_bytes(&mvp, 1).unwrap();
    encoder.set_vertex_buffer(Some(&extra), 0, 3).unwrap();
    encoder.set_vertex_texture(Some(&zero), 0).unwrap();
    encoder.set_fragment_texture(Some(texture), 0).unwrap();
    encoder
        .set_fragment_sampler_state(Some(&sampler), 0)
        .unwrap();
    encoder.set_fragment_bytes(&tint, 0).unwrap();
    encoder.set_fragment_buffer(Some(&count), 0, 2).unwrap();
    encoder
        .draw_primitives(PrimitiveType::Triangle, 0, (n * 6) as _, 1, 0)
        .unwrap();
    encoder.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    let mut read = vec![0u8; (n * 4) as usize];
    target
        .get_bytes(&mut read, n * 4, 0, Region::new_2d(0, 0, n, 1), 0, 0)
        .unwrap();
    (0..n as usize).map(|i| read[i * 4 + 3]).collect()
}

#[test]
fn explore_every_row_edge_of_a_mask_cache() {
    let scratch = Scratch::new("mask-cache-rows");
    let device = device(&scratch.0);
    let state = state(&device);
    let texture = rows(&device);
    // Each row's top edge y / 160, and the f32 one below and one above it.
    let edges: Vec<f32> = (1..HEIGHT).map(|y| y as f32 / HEIGHT as f32).collect();
    let below: Vec<f32> = edges
        .iter()
        .map(|v| f32::from_bits(v.to_bits() - 1))
        .collect();
    let above: Vec<f32> = edges
        .iter()
        .map(|v| f32::from_bits(v.to_bits() + 1))
        .collect();
    // What exact arithmetic reads: floor(v * 160) in f64.
    let want = |vs: &[f32]| -> Vec<u8> {
        vs.iter()
            .map(|v| (f64::from(*v) * 160.0).floor() as u8)
            .collect()
    };
    for (name, vs) in [("edge", &edges), ("below", &below), ("above", &above)] {
        let exact = want(vs);
        let kernel = kernel_rows(&device, &texture, vs);
        let drawn = drawn_rows(&device, &state, &texture, vs);
        let kernel_off: Vec<u64> = (0..vs.len())
            .filter(|&i| kernel[i] != exact[i])
            .map(|i| i as u64 + 1)
            .collect();
        let drawn_off: Vec<u64> = (0..vs.len())
            .filter(|&i| drawn[i] != exact[i])
            .map(|i| i as u64 + 1)
            .collect();
        eprintln!("ROWS {name}: kernel off at rows {kernel_off:?}");
        eprintln!("ROWS {name}: drawn off at rows {drawn_off:?}");
        eprintln!(
            "ROWS {name}: row 128 exact {} kernel {} drawn {}",
            exact[127], kernel[127], drawn[127]
        );
    }
}
