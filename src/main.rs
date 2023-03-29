use std::{num::NonZeroU32, path::PathBuf, sync::Arc, thread::available_parallelism};

use clap::Parser;
use image::{GenericImageView, ImageOutputFormat};
use tracing::{debug, error, Instrument};
use tracing_indicatif::IndicatifLayer;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use wgpu::{util::DeviceExt, Buffer, Extent3d, RenderPipeline};

pub struct Texture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub sampler: wgpu::Sampler,
    pub size: (u32, u32),
}

pub fn texture_bind_group_layout(device: &wgpu::Device, label: &str) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                // This should match the `filterable` field of the
                // corresponding Texture entry above.
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
        label: Some(label),
    })
}

impl Texture {
    pub fn new_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> anyhow::Result<Self> {
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: label.into(),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[wgpu::TextureFormat::R8Unorm],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
            size: (width, height),
        })
    }

    pub fn new_image(
        device: &wgpu::Device,
        dimensions: (u32, u32),
        label: Option<&str>,
    ) -> anyhow::Result<Self> {
        let size = wgpu::Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: 1,
        };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureFormat::Rgba8UnormSrgb,
            ],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            texture,
            view,
            sampler,
            size: dimensions,
        })
    }

    pub fn from_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        img: &image::DynamicImage,
        label: Option<&str>,
    ) -> anyhow::Result<Self> {
        let dimensions = img.dimensions();
        let tex = Self::new_image(device, dimensions, label)?;
        tex.upload_img(queue, img);
        Ok(tex)
    }

    pub fn upload_img(&self, queue: &wgpu::Queue, img: &image::DynamicImage) {
        assert_eq!(self.size.0, img.width());
        assert_eq!(self.size.1, img.height());
        let rgba = img.to_rgba8();
        let size = wgpu::Extent3d {
            width: self.size.0,
            height: self.size.1,
            depth_or_array_layers: 1,
        };
        queue.write_texture(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            &rgba,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * self.size.0),
                rows_per_image: std::num::NonZeroU32::new(self.size.1),
            },
            size,
        );
    }
}

struct AppState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    lumi_pipeline: RenderPipeline,
    vertex_buffer: Buffer,
    index_buffer: Buffer,
    num_indices: u32,
}

struct RenderDesc<'a> {
    output_data: &'a mut Vec<u8>,
    src: &'a Texture,
    dst: &'a Texture,
    output_buffer: &'a Buffer,
    unpadded_bytes_per_row: u32,
    padded_bytes_per_row: u32,
}

impl AppState {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .expect("Failed to create adapter");

        debug!("Choosen adapter: {:?}", adapter);

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let shader = device.create_shader_module(wgpu::include_wgsl!("lumi.wgsl"));

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout(&device, "src")],
                push_constant_ranges: &[],
            });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = INDICES.len() as u32;

        let lumi_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Lumi pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: true,
            },
            multiview: None,
        });

        Self {
            vertex_buffer,
            index_buffer,
            num_indices,
            device,
            queue,
            lumi_pipeline,
        }
    }

    pub async fn render_single_frame(
        &self,
        desc: &mut RenderDesc<'_>,
    ) -> Result<(), wgpu::SurfaceError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
        // FIXME: retain bind groups
        let src_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &texture_bind_group_layout(&self.device, "src"),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&desc.src.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&desc.src.sampler),
                },
            ],
        });
        // render
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: "lumi".into(),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &desc.dst.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(&self.lumi_pipeline);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            pass.set_bind_group(0, &src_bind, &[]);
            pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTextureBase {
                texture: &desc.dst.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &desc.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(desc.padded_bytes_per_row),
                    rows_per_image: NonZeroU32::new(desc.dst.size.1),
                },
            },
            Extent3d {
                width: desc.dst.size.0,
                height: desc.dst.size.1,
                depth_or_array_layers: 1,
            },
        );
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));

        // Create the map request
        let buffer_slice = desc.output_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // wait for the GPU to finish
        self.device.poll(wgpu::Maintain::Wait);

        match rx.receive().await {
            Some(Ok(())) => {
                let padded_data = buffer_slice.get_mapped_range();
                desc.output_data.clear();
                desc.output_data.extend(
                    padded_data
                        .chunks(desc.padded_bytes_per_row as _)
                        .map(|chunk| &chunk[..desc.unpadded_bytes_per_row as _])
                        .flatten()
                        .map(|x| *x),
                );
                drop(padded_data);
                desc.output_buffer.unmap();
            }
            _ => error!("Something went wrong"),
        }

        Ok(())
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
}

impl Vertex {
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

const VERTICES: &[Vertex] = &[
    // A
    Vertex {
        pos: [-1.0, 1.0],
        uv: [0.0, 0.0],
    },
    // B
    Vertex {
        pos: [-1.0, -1.0],
        uv: [0.0, 1.0],
    },
    // C
    Vertex {
        pos: [1.0, -1.0],
        uv: [1.0, 1.0],
    },
    // D
    Vertex {
        pos: [1.0, 1.0],
        uv: [1.0, 0.0],
    },
];

const INDICES: &[u16] = &[3, 2, 1, 3, 1, 0];

async fn worker(
    app: Arc<AppState>,
    rx: async_channel::Receiver<PathBuf>,
    src_root: PathBuf,
    dst_root: PathBuf,
) {
    let mut src = Texture::new_image(&app.device, (1, 1), None).unwrap();
    let mut dst = Texture::new_target(&app.device, 1, 1, "dst").unwrap();
    let buffer_desc = wgpu::BufferDescriptor {
        size: 0,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        label: Some("Output Buffer"),
        mapped_at_creation: false,
    };
    let mut output_buffer = app.device.create_buffer(&buffer_desc);
    let mut unpadded_bytes_per_row = 0;
    let mut padded_bytes_per_row = 0;
    let mut data = Vec::new();

    while let Ok(path) = rx.recv().await {
        debug!("Processing {path:?}");
        let img = match image::open(&path) {
            Ok(img) => img,
            Err(err) => {
                error!("Failed to open image: {err}");
                continue;
            }
        };
        if img.dimensions() != src.size {
            src = Texture::from_image(&app.device, &app.queue, &img, None).unwrap();
            let dst_width = src.size.0;
            let dst_height = src.size.1;
            let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
            unpadded_bytes_per_row = dst_width;
            let padding = (align - unpadded_bytes_per_row % align) % align;
            padded_bytes_per_row = unpadded_bytes_per_row + padding;
            // create a buffer to copy the texture to so we can get the data
            let buffer_size = (padded_bytes_per_row * dst_height) as wgpu::BufferAddress;
            let buffer_desc = wgpu::BufferDescriptor {
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                label: Some("Output Buffer"),
                mapped_at_creation: false,
            };
            output_buffer = app.device.create_buffer(&buffer_desc);
            dst = Texture::new_target(&app.device, img.width(), img.height(), "dst").unwrap();
        } else {
            src.upload_img(&app.queue, &img);
        }
        app.render_single_frame(&mut RenderDesc {
            src: &src,
            dst: &dst,
            output_buffer: &output_buffer,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
            output_data: &mut data,
        })
        .instrument(tracing::error_span!(
            "image-render",
            path = tracing::field::debug(&path)
        ))
        .await
        .expect("Rendering failed");

        let _span =
            tracing::error_span!("image-save", path = tracing::field::debug(&path)).entered();
        let dst_path = path.strip_prefix(&src_root).unwrap();
        let mut dst_path = dst_root.join(dst_path);
        dst_path.set_extension("png");
        std::fs::create_dir_all(dst_path.parent().unwrap()).unwrap_or_default();

        let w = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(dst_path)
            .expect("Failed to open output");
        let mut w = std::io::BufWriter::new(w);

        image::write_buffer_with_format(
            &mut w,
            &data,
            dst.size.0,
            dst.size.1,
            image::ColorType::L8,
            ImageOutputFormat::Png,
        )
        .expect("Failed to write output");
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory to find the images
    #[arg(short, long)]
    src_dir: PathBuf,

    /// Directory where the results are written.
    #[arg(short, long)]
    dst_dir: Option<PathBuf>,

    /// Number of worker threads
    #[arg(short, long)]
    workers: Option<usize>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let dst_dir = args.dst_dir.unwrap_or_else(|| "result".into());
    assert!(args.src_dir.is_dir(), "Input must be an existing directory");
    assert!(
        dst_dir.is_dir() || !dst_dir.exists(),
        "Output must be a directory"
    );

    let indicatif_layer = IndicatifLayer::new();
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer().with_writer(indicatif_layer.get_stderr_writer()))
        .with(indicatif_layer)
        .init();

    let app = AppState::new().await;
    let app = Arc::new(app);

    let workers = args
        .workers
        .unwrap_or_else(|| available_parallelism().map(|x| x.get()).unwrap_or(8));

    let (sender, receiver) = async_channel::bounded(workers * 2);

    for _ in 0..workers {
        tokio::spawn(worker(
            Arc::clone(&app),
            receiver.clone(),
            args.src_dir.clone(),
            dst_dir.clone(),
        ));
    }

    scan_dir(args.src_dir.clone(), &sender).await;
}

async fn scan_dir(src_dir: std::path::PathBuf, sender: &async_channel::Sender<PathBuf>) {
    let mut roots = Vec::new();
    roots.push(src_dir);
    while let Some(dir) = roots.pop() {
        let _span = tracing::error_span!("scanning directory", path = tracing::field::debug(&dir))
            .entered();
        for entry in std::fs::read_dir(dir).expect("Failed to read source directory") {
            match entry {
                Ok(entry) => {
                    let p = entry.path();
                    if p.is_file() {
                        sender.send(p).await.unwrap();
                    } else if p.is_dir() {
                        roots.push(p);
                    }
                }
                Err(err) => {
                    error!("Failed to read directory entry: {err}");
                }
            }
        }
    }
}
