// ═══════════════════════════════════════════════════════════════
// PROMETHEUS EDITOR — Bone Rendering Shader
//
// Simple line/sphere rendering for skeleton visualization.
// Vertex colors passed through, minimal lighting for depth cues.
// ═══════════════════════════════════════════════════════════════

struct Uniforms {
    view_proj: mat4x4<f32>,
    eye_pos: vec4<f32>,
    highlight_color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = u.view_proj * vec4(in.position, 1.0);
    out.world_pos = in.position;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple depth-based darkening for 3D feel
    let dist = length(u.eye_pos.xyz - in.world_pos);
    let fade = clamp(1.0 - dist * 0.003, 0.3, 1.0);
    return vec4(in.color.rgb * fade, in.color.a);
}
