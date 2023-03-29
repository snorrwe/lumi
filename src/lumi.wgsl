struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct Vertex {
    @location(0) pos: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

@group(0) @binding(0)
var src_texture: texture_2d<f32>;
@group(0) @binding(1)
var src_texture_sampler: sampler;

@vertex
fn vs_main(
    model: Vertex,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(model.pos, 0.0, 1.0);
    out.uv = model.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(src_texture, src_texture_sampler, in.uv);
    let luminance = sqrt(color.r * color.r * 0.299 + color.g * color.g * 0.587 + color.b * color.b * 0.114);
    return vec4<f32>(vec3<f32>(luminance), 1.0);
}
