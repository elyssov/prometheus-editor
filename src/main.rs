// ═══════════════════════════════════════════════════════════════
// PROMETHEUS EDITOR — Visual Model Editor
//
// Phase 1: Skeleton Editor
// - 3D viewport with orbit camera
// - Bone rendering (octahedra + joint spheres)
// - Bone selection (click to pick)
// - Properties panel (egui)
// - Drag rotation with constraint clamping
// - Skeleton presets (Human, Cat)
// - JSON save/load
//
// Controls:
//   RMB drag  — orbit camera
//   MMB drag  — pan camera
//   Scroll    — zoom
//   LMB       — select bone / drag rotate
//   F         — focus selected bone
//   1/2/3     — front/side/top view
//   T         — toggle auto-rotate
// ═══════════════════════════════════════════════════════════════

mod model;
mod cli;

use std::sync::Arc;
use glam::{Mat4, Vec3, Vec4, Quat};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::{MouseButton, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use prometheus_engine::core::skeleton::{Skeleton, BoneId, JointConstraint, Bone};
use prometheus_engine::core::render_mesh::{self, MeshUniforms, GpuMesh};
use prometheus_engine::core::sdf_body::SdfBody;
use prometheus_engine::core::meshing;
use prometheus_engine::core::svo::Voxel;

// ─── Bone Mesh Generation ───────────────────────────────────
// Bones are rendered as octahedra (diamond shapes) for clarity.
// Joint spheres at each connection point.

/// Vertex for bone rendering: position + color
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BoneVertex {
    position: [f32; 3],
    color: [f32; 4],
}

/// Uniforms for bone shader
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BoneUniforms {
    view_proj: [[f32; 4]; 4],
    eye_pos: [f32; 4],
    highlight_color: [f32; 4],
}

/// Generate an octahedron mesh for a bone (from start to end)
fn bone_octahedron(start: Vec3, end: Vec3, thickness: f32, color: [f32; 4]) -> Vec<BoneVertex> {
    let dir = end - start;
    let len = dir.length();
    if len < 0.001 { return Vec::new(); }

    let fwd = dir / len;
    // Find perpendicular axes
    let up = if fwd.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let right = fwd.cross(up).normalize() * thickness;
    let up = fwd.cross(right).normalize() * thickness;

    // Octahedron: 4 side vertices at 20% from start, tip at end
    let mid = start + dir * 0.2;
    let v0 = mid + right;
    let v1 = mid + up;
    let v2 = mid - right;
    let v3 = mid - up;

    let mk = |p: Vec3| BoneVertex { position: [p.x, p.y, p.z], color };

    // 8 triangles (4 front faces toward end, 4 back faces toward start)
    vec![
        // Front (toward end)
        mk(v0), mk(v1), mk(end),
        mk(v1), mk(v2), mk(end),
        mk(v2), mk(v3), mk(end),
        mk(v3), mk(v0), mk(end),
        // Back (toward start)
        mk(v1), mk(v0), mk(start),
        mk(v2), mk(v1), mk(start),
        mk(v3), mk(v2), mk(start),
        mk(v0), mk(v3), mk(start),
    ]
}

/// Generate a sphere approximation (icosphere subdivision 1) at a point
fn joint_sphere(center: Vec3, radius: f32, color: [f32; 4]) -> Vec<BoneVertex> {
    let mk = |p: Vec3| BoneVertex { position: [p.x, p.y, p.z], color };

    // Icosahedron vertices
    let t = (1.0 + 5.0_f32.sqrt()) / 2.0;
    let verts_raw = [
        Vec3::new(-1.0,  t, 0.0), Vec3::new( 1.0,  t, 0.0),
        Vec3::new(-1.0, -t, 0.0), Vec3::new( 1.0, -t, 0.0),
        Vec3::new(0.0, -1.0,  t), Vec3::new(0.0,  1.0,  t),
        Vec3::new(0.0, -1.0, -t), Vec3::new(0.0,  1.0, -t),
        Vec3::new( t, 0.0, -1.0), Vec3::new( t, 0.0,  1.0),
        Vec3::new(-t, 0.0, -1.0), Vec3::new(-t, 0.0,  1.0),
    ];
    let verts: Vec<Vec3> = verts_raw.iter().map(|v| center + v.normalize() * radius).collect();

    let faces: [(usize,usize,usize); 20] = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ];

    faces.iter().flat_map(|&(a,b,c)| vec![mk(verts[a]), mk(verts[b]), mk(verts[c])]).collect()
}

/// Generate constraint visualization (arc for hinge, cone for ball-socket)
fn constraint_arc(bone: &Bone, parent_end: Vec3, color: [f32; 4]) -> Vec<BoneVertex> {
    let mk = |p: Vec3| BoneVertex { position: [p.x, p.y, p.z], color };
    let mut verts = Vec::new();

    match &bone.constraint {
        JointConstraint::Hinge { axis, min_angle, max_angle } => {
            let world_axis = bone.world_rotation * *axis;
            let bone_dir = if bone.rest_length > 0.001 {
                (bone.world_end_position - bone.world_position).normalize()
            } else {
                bone.world_rotation * bone.rest_direction
            };
            let arc_radius = bone.rest_length.max(2.0) * 0.3;
            let steps = 16;

            for i in 0..steps {
                let t0 = *min_angle + (*max_angle - *min_angle) * (i as f32 / steps as f32);
                let t1 = *min_angle + (*max_angle - *min_angle) * ((i + 1) as f32 / steps as f32);
                let r0 = Quat::from_axis_angle(world_axis, t0);
                let r1 = Quat::from_axis_angle(world_axis, t1);
                let p0 = parent_end + r0 * bone_dir * arc_radius;
                let p1 = parent_end + r1 * bone_dir * arc_radius;
                // Thin triangle from joint to arc
                verts.push(mk(parent_end));
                verts.push(mk(p0));
                verts.push(mk(p1));
            }
        }
        JointConstraint::BallSocket { cone_angle, .. } => {
            let bone_dir = if bone.rest_length > 0.001 {
                (bone.world_end_position - bone.world_position).normalize()
            } else {
                bone.world_rotation * bone.rest_direction
            };
            let arc_radius = bone.rest_length.max(2.0) * 0.3;
            let steps = 16;

            for i in 0..steps {
                let a0 = std::f32::consts::TAU * (i as f32 / steps as f32);
                let a1 = std::f32::consts::TAU * ((i + 1) as f32 / steps as f32);
                // Rotate bone_dir by cone_angle around perpendicular, then around bone_dir
                let perp = if bone_dir.y.abs() < 0.9 { bone_dir.cross(Vec3::Y).normalize() } else { bone_dir.cross(Vec3::X).normalize() };
                let up = bone_dir.cross(perp).normalize();
                let cone_edge = (bone_dir * cone_angle.cos() + (perp * a0.cos() + up * a0.sin()) * cone_angle.sin()).normalize();
                let cone_edge2 = (bone_dir * cone_angle.cos() + (perp * a1.cos() + up * a1.sin()) * cone_angle.sin()).normalize();
                let p0 = parent_end + cone_edge * arc_radius;
                let p1 = parent_end + cone_edge2 * arc_radius;
                verts.push(mk(parent_end));
                verts.push(mk(p0));
                verts.push(mk(p1));
            }
        }
        _ => {}
    }
    verts
}

/// Render a ring showing the current thickness at a control point
fn thickness_ring(center: Vec3, radius: f32, bone_rot: Quat, color: [f32; 4]) -> Vec<BoneVertex> {
    let mk = |p: Vec3| BoneVertex { position: [p.x, p.y, p.z], color };
    let mut verts = Vec::new();
    let segments = 12;
    let width = 0.2;

    for i in 0..segments {
        let a0 = std::f32::consts::TAU * (i as f32 / segments as f32);
        let a1 = std::f32::consts::TAU * ((i + 1) as f32 / segments as f32);

        // Points on circle in local XZ plane, then rotate by bone orientation
        let p0 = center + bone_rot * Vec3::new(radius * a0.cos(), 0.0, radius * a0.sin());
        let p1 = center + bone_rot * Vec3::new(radius * a1.cos(), 0.0, radius * a1.sin());
        let p0_in = center + bone_rot * Vec3::new((radius - width) * a0.cos(), 0.0, (radius - width) * a0.sin());
        let p1_in = center + bone_rot * Vec3::new((radius - width) * a1.cos(), 0.0, (radius - width) * a1.sin());

        verts.push(mk(p0)); verts.push(mk(p1)); verts.push(mk(p1_in));
        verts.push(mk(p0)); verts.push(mk(p1_in)); verts.push(mk(p0_in));
    }
    verts
}

// ─── Grid Floor ─────────────────────────────────────────────

fn grid_floor(size: f32, spacing: f32, y: f32) -> Vec<BoneVertex> {
    let mut verts = Vec::new();
    let color = [0.2, 0.2, 0.25, 0.5];
    let mk = |p: Vec3| BoneVertex { position: [p.x, p.y, p.z], color };

    let half = size / 2.0;
    let mut x = -half;
    while x <= half {
        // Thin quad along Z axis
        let w = 0.05;
        verts.push(mk(Vec3::new(x - w, y, -half)));
        verts.push(mk(Vec3::new(x + w, y, -half)));
        verts.push(mk(Vec3::new(x + w, y,  half)));
        verts.push(mk(Vec3::new(x - w, y, -half)));
        verts.push(mk(Vec3::new(x + w, y,  half)));
        verts.push(mk(Vec3::new(x - w, y,  half)));
        x += spacing;
    }
    let mut z = -half;
    while z <= half {
        let w = 0.05;
        verts.push(mk(Vec3::new(-half, y, z - w)));
        verts.push(mk(Vec3::new(-half, y, z + w)));
        verts.push(mk(Vec3::new( half, y, z + w)));
        verts.push(mk(Vec3::new(-half, y, z - w)));
        verts.push(mk(Vec3::new( half, y, z + w)));
        verts.push(mk(Vec3::new( half, y, z - w)));
        z += spacing;
    }
    verts
}

// ─── Orbit Camera ───────────────────────────────────────────

struct OrbitCamera {
    yaw: f32,
    pitch: f32,
    distance: f32,
    target: Vec3,
    fov: f32,
}

impl OrbitCamera {
    fn new() -> Self {
        Self {
            yaw: 0.5,
            pitch: 0.3,
            distance: 400.0,
            target: Vec3::new(0.0, 140.0, 0.0),
            fov: 45.0,
        }
    }

    fn eye(&self) -> Vec3 {
        Vec3::new(
            self.target.x + self.distance * self.pitch.cos() * self.yaw.sin(),
            self.target.y + self.distance * self.pitch.sin(),
            self.target.z + self.distance * self.pitch.cos() * self.yaw.cos(),
        )
    }

    fn view(&self) -> Mat4 {
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Y)
    }

    fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov.to_radians(), aspect, 0.1, 5000.0)
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.005;
        self.pitch = (self.pitch + dy * 0.005).clamp(-1.4, 1.4);
    }

    fn pan(&mut self, dx: f32, dy: f32) {
        let right = self.view().row(0).truncate().normalize();
        let up = self.view().row(1).truncate().normalize();
        let speed = self.distance * 0.002;
        self.target += right * (-dx as f32) * speed + up * (dy as f32) * speed;
    }

    fn zoom(&mut self, delta: f32) {
        self.distance = (self.distance * (1.0 - delta * 0.1)).clamp(5.0, 2000.0);
    }

    fn focus(&mut self, pos: Vec3) {
        self.target = pos;
    }
}

// ─── Ray-bone intersection for picking ──────────────────────

fn ray_from_screen(
    screen_x: f32, screen_y: f32,
    width: f32, height: f32,
    view: Mat4, proj: Mat4,
) -> (Vec3, Vec3) {
    let ndc_x = (2.0 * screen_x / width) - 1.0;
    let ndc_y = 1.0 - (2.0 * screen_y / height);
    let inv_vp = (proj * view).inverse();
    let near = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let origin = near.truncate() / near.w;
    let target = far.truncate() / far.w;
    let dir = (target - origin).normalize();
    (origin, dir)
}

/// Distance from ray to line segment (bone)
fn ray_segment_distance(ray_origin: Vec3, ray_dir: Vec3, seg_a: Vec3, seg_b: Vec3) -> f32 {
    let u = ray_dir;
    let v = seg_b - seg_a;
    let w = ray_origin - seg_a;

    let a = u.dot(u);
    let b = u.dot(v);
    let c = v.dot(v);
    let d = u.dot(w);
    let e = v.dot(w);

    let denom = a * c - b * b;
    if denom.abs() < 0.0001 {
        // Parallel lines
        return w.cross(u).length() / a.sqrt();
    }

    let s = (b * e - c * d) / denom;
    let t = ((a * e - b * d) / denom).clamp(0.0, 1.0);

    let closest_ray = ray_origin + u * s;
    let closest_seg = seg_a + v * t;
    (closest_ray - closest_seg).length()
}

/// Pick the closest bone to a screen click
fn pick_bone(
    skeleton: &Skeleton,
    screen_x: f32, screen_y: f32,
    width: f32, height: f32,
    view: Mat4, proj: Mat4,
) -> Option<BoneId> {
    let (ray_o, ray_d) = ray_from_screen(screen_x, screen_y, width, height, view, proj);

    let mut best_id = None;
    let mut best_dist = 5.0; // Max pick distance in world units (adjusted by bone thickness)

    for bone in skeleton.bones() {
        if bone.rest_length < 0.001 { continue; } // Skip root with zero length

        let dist = ray_segment_distance(ray_o, ray_d, bone.world_position, bone.world_end_position);
        let threshold = (bone.rest_length * 0.15).max(2.0);

        if dist < threshold && dist < best_dist {
            best_dist = dist;
            best_id = Some(bone.id);
        }
    }
    best_id
}

/// Pick a control point near screen position
fn pick_control_point(
    state: &EditorState,
    screen_x: f32, screen_y: f32,
    width: f32, height: f32,
    view: Mat4, proj: Mat4,
) -> Option<(BoneId, usize)> {
    if !state.show_voxels { return None; }

    let (ray_o, ray_d) = ray_from_screen(screen_x, screen_y, width, height, view, proj);
    let mut best: Option<(BoneId, usize)> = None;
    let mut best_dist = 3.0; // pick threshold

    for profile in &state.body_profiles {
        let bone = state.skeleton.bone(profile.bone_name.as_str());
        if bone.rest_length < 0.001 { continue; }

        for (idx, cp) in profile.points.iter().enumerate() {
            let pos = bone.world_position
                + (bone.world_end_position - bone.world_position) * cp.t;

            // Distance from ray to point
            let to_point = pos - ray_o;
            let proj_len = to_point.dot(ray_d);
            if proj_len < 0.0 { continue; } // behind camera
            let closest = ray_o + ray_d * proj_len;
            let dist = (closest - pos).length();

            if dist < best_dist {
                best_dist = dist;
                best = Some((bone.id, idx));
            }
        }
    }
    best
}

// ─── Editor State ───────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
enum EditorMode {
    Skeleton,
    // Sculpt, — Phase 2
    // Detail, — Phase 3
    // Animate, — Phase 3
}

// ─── Animation System (Chevrons) ────────────────────────────

/// A single keyframe = snapshot of all bone rotations at a moment
#[derive(Clone)]
pub struct AnimKeyframe {
    pub rotations: Vec<Quat>,
    pub label: String,
}

/// Easing function between keyframes
#[derive(Clone, Copy, PartialEq)]
pub enum Easing {
    Linear,
    EaseInOut, // smoothstep
    EaseIn,    // accelerate
    EaseOut,   // decelerate
}

/// A complete animation clip
#[derive(Clone)]
pub struct Animation {
    pub name: String,
    pub keyframes: Vec<AnimKeyframe>,
    pub easings: Vec<Easing>,
    pub looping: bool,
    pub transition_duration: f32,
}

impl Animation {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            keyframes: Vec::new(),
            easings: Vec::new(),
            looping: false,
            transition_duration: 0.5,
        }
    }

    /// Capture current skeleton state as a keyframe
    fn capture(&mut self, skeleton: &Skeleton, label: &str) {
        let rotations: Vec<Quat> = skeleton.bones().iter()
            .map(|b| b.local_rotation)
            .collect();
        self.keyframes.push(AnimKeyframe {
            rotations,
            label: label.to_string(),
        });
        // Add default easing for new transition
        if self.keyframes.len() > 1 {
            self.easings.push(Easing::EaseInOut);
        }
    }

    fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }
}

/// Animation playback state
pub struct AnimPlayer {
    pub playing: bool,
    pub time: f32,
    pub anim: Option<Animation>,
    pub speed: f32,
}

impl AnimPlayer {
    fn new() -> Self {
        Self { playing: false, time: 0.0, anim: None, speed: 1.0 }
    }

    /// Advance time and apply interpolated pose to skeleton
    fn update(&mut self, dt: f32, skeleton: &mut Skeleton) -> bool {
        if !self.playing { return false; }
        let anim = match &self.anim { Some(a) => a, None => return false };
        if anim.keyframes.len() < 2 { return false; }

        self.time += dt * self.speed;

        let n = anim.keyframes.len();
        let total_transitions = if anim.looping { n } else { n - 1 };
        let total_duration = total_transitions as f32 * anim.transition_duration;

        if self.time >= total_duration {
            if anim.looping {
                self.time %= total_duration;
            } else {
                self.time = total_duration - 0.001;
                self.playing = false;
            }
        }

        // Find which transition we're in
        let transition_idx = (self.time / anim.transition_duration) as usize;
        let local_t = (self.time / anim.transition_duration).fract();

        let from_idx = transition_idx % n;
        let to_idx = (transition_idx + 1) % n;

        // Apply easing
        let easing = if transition_idx < anim.easings.len() {
            anim.easings[transition_idx]
        } else {
            Easing::EaseInOut
        };
        let t = apply_easing(local_t, easing);

        // Interpolate all bone rotations
        let from = &anim.keyframes[from_idx];
        let to = &anim.keyframes[to_idx];
        let bone_count = skeleton.bone_count().min(from.rotations.len()).min(to.rotations.len());

        for i in 0..bone_count {
            let name = skeleton.bones()[i].name.clone();
            let interp = from.rotations[i].slerp(to.rotations[i], t);
            skeleton.set_rotation(&name, interp);
        }
        skeleton.solve_forward();
        true // changed
    }
}

fn apply_easing(t: f32, easing: Easing) -> f32 {
    match easing {
        Easing::Linear => t,
        Easing::EaseInOut => t * t * (3.0 - 2.0 * t), // smoothstep
        Easing::EaseIn => t * t,
        Easing::EaseOut => t * (2.0 - t),
    }
}

/// A control point on a bone for adjusting body thickness
#[derive(Clone)]
pub struct ControlPoint {
    pub t: f32,
    pub radius_mul: f32,
}

/// Per-bone body profile — control points for thickness
#[derive(Clone)]
pub struct BoneBodyProfile {
    pub bone_name: String,
    pub points: Vec<ControlPoint>,
    pub base_radius: f32,
}

/// What the user is currently interacting with
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Selection {
    None,
    Bone(BoneId),
    ControlPoint(BoneId, usize), // bone_id, point index
}

pub struct EditorState {
    pub skeleton: Skeleton,
    pub selection: Selection,
    pub selected_bone: Option<BoneId>,
    pub mode: EditorMode,
    pub show_constraints: bool,
    pub show_names: bool,
    pub show_voxels: bool,
    pub auto_rotate: bool,
    pub skeleton_scale: f32,
    pub preset_name: String,
    pub body_profiles: Vec<BoneBodyProfile>,
    pub animation: Animation,
    pub anim_player: AnimPlayer,
}

impl EditorState {
    fn new() -> Self {
        let scale = 3.0; // 3.0 = ~300 voxels tall, good detail
        let mut skeleton = Skeleton::human(scale);
        skeleton.root_position = Vec3::new(0.0, 140.0, 0.0); // legs need room below
        skeleton.solve_forward();
        let body_profiles = Self::default_human_profiles(scale);

        Self {
            skeleton,
            selection: Selection::None,
            selected_bone: None,
            mode: EditorMode::Skeleton,
            show_constraints: true,
            show_names: true,
            show_voxels: false,
            auto_rotate: false,
            skeleton_scale: scale,
            preset_name: "Human".to_string(),
            body_profiles,
            animation: Animation::new("walk"),
            anim_player: AnimPlayer::new(),
        }
    }

    fn load_preset(&mut self, name: &str) {
        let scale = self.skeleton_scale;
        match name {
            "Human" => {
                self.skeleton = Skeleton::human(scale);
                self.skeleton.root_position = Vec3::new(0.0, scale * 47.0, 0.0);
                self.body_profiles = Self::default_human_profiles(scale);
            }
            "Cat" => {
                self.skeleton = Skeleton::cat(scale);
                self.skeleton.root_position = Vec3::new(0.0, scale * 20.0, 0.0);
                self.body_profiles = Vec::new(); // TODO: cat profiles
            }
            _ => {}
        }
        self.skeleton.solve_forward();
        self.selection = Selection::None;
        self.selected_bone = None;
        self.preset_name = name.to_string();
    }

    /// Default body profiles for human skeleton — matches SdfBody::human_body() radii
    fn default_human_profiles(scale: f32) -> Vec<BoneBodyProfile> {
        let s = scale;
        let mut profiles = Vec::new();

        // Helper: create profile with N evenly-spaced control points
        let make = |name: &str, radius: f32, n: usize| -> BoneBodyProfile {
            let points: Vec<ControlPoint> = (0..n)
                .map(|i| ControlPoint {
                    t: i as f32 / (n - 1).max(1) as f32,
                    radius_mul: 1.0,
                })
                .collect();
            BoneBodyProfile { bone_name: name.to_string(), points, base_radius: radius }
        };

        // Torso bones
        profiles.push(make("spine", 7.0 * s, 4));
        profiles.push(make("chest", 8.0 * s, 4));
        profiles.push(make("neck", 3.0 * s, 3));
        profiles.push(make("head", 5.5 * s, 4));

        // Arms (both sides)
        for suffix in ["_l", "_r"] {
            profiles.push(make(&format!("shoulder{}", suffix), 5.0 * s, 2));
            profiles.push(make(&format!("upper_arm{}", suffix), 3.5 * s, 4));
            profiles.push(make(&format!("forearm{}", suffix), 3.0 * s, 4));
            profiles.push(make(&format!("hand{}", suffix), 2.0 * s, 3));
        }

        // Legs (both sides)
        for suffix in ["_l", "_r"] {
            profiles.push(make(&format!("thigh{}", suffix), 4.5 * s, 5));
            profiles.push(make(&format!("shin{}", suffix), 3.5 * s, 4));
            profiles.push(make(&format!("foot{}", suffix), 3.5 * s, 3));
        }

        profiles
    }

    /// Get profile for a bone by name (if exists)
    fn profile_for_bone(&self, bone_name: &str) -> Option<&BoneBodyProfile> {
        self.body_profiles.iter().find(|p| p.bone_name == bone_name)
    }

    /// Get mutable profile for a bone by name
    fn profile_for_bone_mut(&mut self, bone_name: &str) -> Option<&mut BoneBodyProfile> {
        self.body_profiles.iter_mut().find(|p| p.bone_name == bone_name)
    }

    /// Get the average radius multiplier for a bone (used in SDF rebuild)
    fn avg_radius_mul(&self, bone_name: &str) -> f32 {
        if let Some(profile) = self.profile_for_bone(bone_name) {
            if profile.points.is_empty() { return 1.0; }
            profile.points.iter().map(|p| p.radius_mul).sum::<f32>() / profile.points.len() as f32
        } else {
            1.0
        }
    }

    /// Build all bone geometry for rendering
    fn build_bone_mesh(&self) -> Vec<BoneVertex> {
        let mut verts = Vec::new();

        // Grid floor
        verts.extend(grid_floor(200.0, 10.0, 0.0));

        // Origin axes (thin colored lines)
        let axis_len = 15.0;
        let aw = 0.3;
        // X = red
        verts.extend(make_box(Vec3::ZERO, Vec3::new(axis_len, aw, aw), [0.8, 0.2, 0.2, 1.0]));
        // Y = green
        verts.extend(make_box(Vec3::ZERO, Vec3::new(aw, axis_len, aw), [0.2, 0.8, 0.2, 1.0]));
        // Z = blue
        verts.extend(make_box(Vec3::ZERO, Vec3::new(aw, aw, axis_len), [0.2, 0.2, 0.8, 1.0]));

        for bone in self.skeleton.bones() {
            let is_selected = self.selected_bone == Some(bone.id);

            // Bone color based on constraint type
            let base_color = match &bone.constraint {
                JointConstraint::Fixed => [0.5, 0.5, 0.5, 1.0],
                JointConstraint::Free => [0.3, 0.7, 0.9, 1.0],
                JointConstraint::Hinge { .. } => [0.9, 0.6, 0.2, 1.0],
                JointConstraint::BallSocket { .. } => [0.2, 0.9, 0.4, 1.0],
            };

            let color = if is_selected {
                [1.0, 1.0, 0.2, 1.0] // Yellow highlight
            } else {
                base_color
            };

            // Draw bone octahedron
            if bone.rest_length > 0.001 {
                let thickness = (bone.rest_length * 0.12).max(0.5);
                verts.extend(bone_octahedron(
                    bone.world_position, bone.world_end_position, thickness, color,
                ));
            }

            // Joint sphere
            let joint_radius = if is_selected { 1.5 } else { 0.8 };
            let joint_color = if is_selected {
                [1.0, 0.9, 0.0, 1.0]
            } else {
                [0.9, 0.9, 0.9, 1.0]
            };
            verts.extend(joint_sphere(bone.world_position, joint_radius, joint_color));

            // Constraint visualization
            if self.show_constraints && !is_selected {
                let constraint_color = [base_color[0] * 0.5, base_color[1] * 0.5, base_color[2] * 0.5, 0.3];
                verts.extend(constraint_arc(bone, bone.world_position, constraint_color));
            }
            if self.show_constraints && is_selected {
                let constraint_color = [1.0, 1.0, 0.0, 0.4];
                verts.extend(constraint_arc(bone, bone.world_position, constraint_color));
            }

            // Control points (when voxels mode is on and bone has a profile)
            if self.show_voxels && bone.rest_length > 0.001 {
                if let Some(profile) = self.profile_for_bone(&bone.name) {
                    for (idx, cp) in profile.points.iter().enumerate() {
                        let pos = bone.world_position
                            + (bone.world_end_position - bone.world_position) * cp.t;
                        let is_cp_selected = matches!(self.selection,
                            Selection::ControlPoint(bid, pidx) if bid == bone.id && pidx == idx);

                        // Size proportional to radius multiplier
                        let cp_radius = if is_cp_selected { 1.2 } else { 0.7 };
                        let cp_color = if is_cp_selected {
                            [1.0, 0.3, 0.0, 1.0] // Orange when selected
                        } else if (cp.radius_mul - 1.0).abs() > 0.01 {
                            [0.0, 0.8, 1.0, 0.8] // Cyan when modified
                        } else {
                            [0.6, 0.6, 0.8, 0.6] // Gray default
                        };
                        verts.extend(joint_sphere(pos, cp_radius, cp_color));

                        // Show radius ring (visual feedback of thickness)
                        let actual_radius = profile.base_radius * cp.radius_mul;
                        let ring_color = if is_cp_selected {
                            [1.0, 0.5, 0.0, 0.3]
                        } else {
                            [0.4, 0.4, 0.6, 0.2]
                        };
                        verts.extend(thickness_ring(pos, actual_radius,
                            bone.world_rotation, ring_color));
                    }
                }
            }
        }

        verts
    }
}

/// Simple axis-aligned box (for origin axes)
fn make_box(min: Vec3, max: Vec3, color: [f32; 4]) -> Vec<BoneVertex> {
    let mk = |x: f32, y: f32, z: f32| BoneVertex { position: [x, y, z], color };
    let (x0,y0,z0) = (min.x, min.y, min.z);
    let (x1,y1,z1) = (max.x, max.y, max.z);
    vec![
        // Front
        mk(x0,y0,z1), mk(x1,y0,z1), mk(x1,y1,z1),
        mk(x0,y0,z1), mk(x1,y1,z1), mk(x0,y1,z1),
        // Back
        mk(x1,y0,z0), mk(x0,y0,z0), mk(x0,y1,z0),
        mk(x1,y0,z0), mk(x0,y1,z0), mk(x1,y1,z0),
        // Top
        mk(x0,y1,z0), mk(x0,y1,z1), mk(x1,y1,z1),
        mk(x0,y1,z0), mk(x1,y1,z1), mk(x1,y1,z0),
        // Bottom
        mk(x0,y0,z1), mk(x0,y0,z0), mk(x1,y0,z0),
        mk(x0,y0,z1), mk(x1,y0,z0), mk(x1,y0,z1),
        // Left
        mk(x0,y0,z0), mk(x0,y0,z1), mk(x0,y1,z1),
        mk(x0,y0,z0), mk(x0,y1,z1), mk(x0,y1,z0),
        // Right
        mk(x1,y0,z1), mk(x1,y0,z0), mk(x1,y1,z0),
        mk(x1,y0,z1), mk(x1,y1,z0), mk(x1,y1,z1),
    ]
}

// ─── Bone tree UI (free function to avoid borrow issues) ────

fn draw_bone_tree_ui(
    ui: &mut egui::Ui,
    skeleton: &Skeleton,
    selected: &mut Option<BoneId>,
    needs_rebuild: &mut bool,
    bone_id: BoneId,
    depth: usize,
) {
    let bone = skeleton.bone_by_id(bone_id);
    let children = bone.children.clone();
    let is_selected = *selected == Some(bone_id);

    let indent = "  ".repeat(depth);
    let constraint_icon = match &bone.constraint {
        JointConstraint::Fixed => "[F]",
        JointConstraint::Free => "[*]",
        JointConstraint::Hinge { .. } => "[H]",
        JointConstraint::BallSocket { .. } => "[B]",
    };

    let label = format!("{}{} {}", indent, constraint_icon, bone.name);
    let response = ui.selectable_label(is_selected, &label);

    if response.clicked() {
        *selected = Some(bone_id);
        *needs_rebuild = true;
    }

    for child_id in children {
        draw_bone_tree_ui(ui, skeleton, selected, needs_rebuild, child_id, depth + 1);
    }
}

// ─── Application ────────────────────────────────────────────

struct App {
    // Window & GPU
    window: Option<Arc<Window>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    config: Option<wgpu::SurfaceConfiguration>,
    depth_view: Option<wgpu::TextureView>,

    // Bone rendering pipeline
    bone_pipeline: Option<wgpu::RenderPipeline>,
    bone_bind_group_layout: Option<wgpu::BindGroupLayout>,
    bone_uniform_buffer: Option<wgpu::Buffer>,
    bone_bind_group: Option<wgpu::BindGroup>,
    bone_vertex_buffer: Option<wgpu::Buffer>,
    bone_vertex_count: u32,

    // Body mesh pipeline (for voxel preview)
    body_pipeline: Option<wgpu::RenderPipeline>,
    body_bind_group_layout: Option<wgpu::BindGroupLayout>,
    body_uniform_buffer: Option<wgpu::Buffer>,
    body_bind_group: Option<wgpu::BindGroup>,
    body_gpu_mesh: Option<GpuMesh>,
    body_needs_rebuild: bool,

    // egui
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    egui_renderer: Option<egui_wgpu::Renderer>,

    // Editor
    state: EditorState,
    camera: OrbitCamera,

    // Input
    mmb_dragging: bool,
    lmb_dragging: bool,
    dragging_bone: bool,    // true = LMB dragging a bone, false = LMB orbiting camera
    last_mouse: (f64, f64),
    mouse_pos: (f64, f64),
    needs_rebuild: bool,
    frame: u64,
}

impl App {
    fn new() -> Self {
        Self {
            window: None, device: None, queue: None, surface: None, config: None,
            depth_view: None,
            bone_pipeline: None, bone_bind_group_layout: None,
            bone_uniform_buffer: None, bone_bind_group: None,
            bone_vertex_buffer: None, bone_vertex_count: 0,
            body_pipeline: None, body_bind_group_layout: None,
            body_uniform_buffer: None, body_bind_group: None,
            body_gpu_mesh: None, body_needs_rebuild: true,
            egui_ctx: egui::Context::default(),
            egui_state: None, egui_renderer: None,
            state: EditorState::new(),
            camera: OrbitCamera::new(),
            mmb_dragging: false, lmb_dragging: false, dragging_bone: false,
            last_mouse: (0.0, 0.0), mouse_pos: (0.0, 0.0),
            needs_rebuild: true, frame: 0,
        }
    }

    fn init_gpu(&mut self, window: Arc<Window>) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(), ..Default::default()
        });
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })).expect("No GPU adapter found");

        println!("  GPU: {}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Prometheus Editor"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
            }, None,
        )).unwrap();

        let size = window.inner_size();
        let config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1)).unwrap();
        surface.configure(&device, &config);

        // Bone rendering pipeline
        let bone_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bone Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bone_render.wgsl").into()),
        });

        let bone_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bone BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bone_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bone Pipeline Layout"),
            bind_group_layouts: &[&bone_bgl],
            push_constant_ranges: &[],
        });

        let bone_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Bone Pipeline"),
            layout: Some(&bone_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &bone_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BoneVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 12, shader_location: 1 },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &bone_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for bone vis (see from all angles)
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let uniforms = BoneUniforms {
            view_proj: Mat4::IDENTITY.to_cols_array_2d(),
            eye_pos: [0.0; 4],
            highlight_color: [1.0, 1.0, 0.0, 1.0],
        };
        let bone_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bone Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bone_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bone BG"),
            layout: &bone_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: bone_uniform_buffer.as_entire_binding() }],
        });

        // Body mesh pipeline (reuses engine mesh shader)
        let (body_pipeline, body_bgl) = render_mesh::create_mesh_pipeline(&device, config.format);
        let body_uniforms = MeshUniforms::new(
            Mat4::IDENTITY, Mat4::IDENTITY, Vec3::ZERO, Vec3::new(0.3, -0.8, 0.5).normalize(),
        );
        let body_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Body Uniforms"),
            contents: bytemuck::bytes_of(&body_uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let body_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Body BG"),
            layout: &body_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: body_uniform_buffer.as_entire_binding() }],
        });

        let (_, depth_view) = render_mesh::create_depth_texture(&device, config.width, config.height);

        // egui setup
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

        self.window = Some(window);
        self.bone_pipeline = Some(bone_pipeline);
        self.bone_bind_group_layout = Some(bone_bgl);
        self.bone_uniform_buffer = Some(bone_uniform_buffer);
        self.bone_bind_group = Some(bone_bind_group);
        self.body_pipeline = Some(body_pipeline);
        self.body_bind_group_layout = Some(body_bgl);
        self.body_uniform_buffer = Some(body_uniform_buffer);
        self.body_bind_group = Some(body_bind_group);
        self.depth_view = Some(depth_view);
        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.config = Some(config);
        self.egui_state = Some(egui_state);
        self.egui_renderer = Some(egui_renderer);
        self.needs_rebuild = true;
        self.body_needs_rebuild = true;
    }

    fn rebuild_bone_mesh(&mut self) {
        let device = match &self.device { Some(d) => d, None => return };

        let verts = self.state.build_bone_mesh();
        if verts.is_empty() { return; }

        self.bone_vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bone Vertices"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.bone_vertex_count = verts.len() as u32;
        self.needs_rebuild = false;
    }

    fn rebuild_body_mesh(&mut self) {
        let device = match &self.device { Some(d) => d, None => return };
        if !self.state.show_voxels {
            self.body_gpu_mesh = None;
            self.body_needs_rebuild = false;
            return;
        }

        let start = std::time::Instant::now();

        // Grid 512: at scale=3.0, body ~300 voxels tall — good detail
        let grid_size: usize = 512;
        let scale = self.state.skeleton_scale;
        let sdf_body = if self.state.preset_name == "Human" {
            SdfBody::human_body(&self.state.skeleton, scale)
        } else {
            // For cat/other, just skip for now
            self.body_gpu_mesh = None;
            self.body_needs_rebuild = false;
            return;
        };

        // Sparse grid
        let mut voxels_map = std::collections::HashMap::new();
        let mut min = [u16::MAX; 3];
        let mut max = [0u16; 3];

        sdf_body.rasterize(grid_size, 1.5, |x, y, z, mat, r, g, b| {
            voxels_map.insert((x as u16, y as u16, z as u16), Voxel::solid(mat, r, g, b));
            min[0] = min[0].min(x as u16);
            min[1] = min[1].min(y as u16);
            min[2] = min[2].min(z as u16);
            max[0] = max[0].max(x as u16);
            max[1] = max[1].max(y as u16);
            max[2] = max[2].max(z as u16);
        });

        if voxels_map.is_empty() {
            self.body_gpu_mesh = None;
            self.body_needs_rebuild = false;
            return;
        }

        // Export to flat grid for meshing
        let margin = 3u16;
        let x0 = min[0].saturating_sub(margin) as usize;
        let y0 = min[1].saturating_sub(margin) as usize;
        let z0 = min[2].saturating_sub(margin) as usize;
        let x1 = (max[0] + margin).min(grid_size as u16 - 1) as usize + 1;
        let y1 = (max[1] + margin).min(grid_size as u16 - 1) as usize + 1;
        let z1 = (max[2] + margin).min(grid_size as u16 - 1) as usize + 1;

        let sx = x1 - x0;
        let sy = y1 - y0;
        let sz = z1 - z0;
        let size = sx.max(sy).max(sz).min(384);

        let mut flat = vec![Voxel::empty(); size * size * size];
        for (&(x, y, z), &v) in &voxels_map {
            let lx = x as usize - x0;
            let ly = y as usize - y0;
            let lz = z as usize - z0;
            if lx < size && ly < size && lz < size {
                flat[lz * size * size + ly * size + lx] = v;
            }
        }

        let offset = Vec3::new(x0 as f32, y0 as f32, z0 as f32);
        let mesh = meshing::generate_mesh_smooth_with_ao(&flat, size, offset, 1.0);

        println!("  Body mesh: {} tri, {} voxels, {:.0}ms",
            mesh.triangle_count, voxels_map.len(), start.elapsed().as_secs_f64() * 1000.0);
        self.body_gpu_mesh = GpuMesh::from_chunk_mesh(device, &mesh);
        self.body_needs_rebuild = false;
    }

    fn draw_egui(&mut self) -> egui::FullOutput {
        let raw_input = self.egui_state.as_mut().unwrap().take_egui_input(self.window.as_ref().unwrap());

        // Extract mutable references to avoid borrow issues with self inside closure
        let state = &mut self.state;
        let camera = &mut self.camera;
        let needs_rebuild = &mut self.needs_rebuild;
        let frame_num = self.frame;

        self.egui_ctx.run(raw_input, |ctx| {
            // ── Top toolbar ──
            egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("PROMETHEUS EDITOR");
                    ui.separator();

                    if ui.selectable_label(state.mode == EditorMode::Skeleton, "Skeleton").clicked() {
                        state.mode = EditorMode::Skeleton;
                    }

                    ui.separator();

                    ui.label("Preset:");
                    let preset = state.preset_name.clone();
                    if ui.selectable_label(preset == "Human", "Human").clicked() {
                        state.load_preset("Human");
                        *needs_rebuild = true;
                    }
                    if ui.selectable_label(preset == "Cat", "Cat").clicked() {
                        state.load_preset("Cat");
                        *needs_rebuild = true;
                    }

                    ui.separator();

                    if ui.checkbox(&mut state.show_constraints, "Constraints").changed() {
                        *needs_rebuild = true;
                    }
                    ui.checkbox(&mut state.show_names, "Names");
                    if ui.checkbox(&mut state.show_voxels, "Voxels").changed() {
                        *needs_rebuild = true;
                    }
                    ui.checkbox(&mut state.auto_rotate, "Rotate");
                });
            });

            // ── Left panel: Bone Outliner ──
            egui::SidePanel::left("outliner").default_width(200.0).show(ctx, |ui| {
                ui.heading("Bones");
                ui.separator();

                egui::ScrollArea::vertical().show(ui, |ui| {
                    let root_id = 0u16;
                    draw_bone_tree_ui(ui, &state.skeleton, &mut state.selected_bone, needs_rebuild, root_id, 0);
                });
            });

            // ── Right panel: Properties ──
            egui::SidePanel::right("properties").default_width(300.0).show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Properties");
                    ui.separator();

                    if let Some(bone_id) = state.selected_bone {
                        // Read bone data
                        let bone = state.skeleton.bone_by_id(bone_id);
                        let name = bone.name.clone();
                        let mut length = bone.rest_length;
                        let parent_name = bone.parent.map(|pid| state.skeleton.bone_by_id(pid).name.clone())
                            .unwrap_or_else(|| "(root)".to_string());
                        let children_count = bone.children.len();
                        let world_pos = bone.world_position;
                        let world_end = bone.world_end_position;
                        let constraint = bone.constraint.clone();
                        let local_rot = bone.local_rotation;

                        // ── Identity ──
                        ui.strong(&name);
                        ui.label(format!("Parent: {}  |  Children: {}", parent_name, children_count));
                        ui.separator();

                        // ── Length (editable slider) ──
                        ui.heading("Length");
                        let min_len = (length * 0.2).max(0.5);
                        let max_len = length * 2.0;
                        if ui.add(egui::Slider::new(&mut length, min_len..=max_len)
                            .text("units").step_by(0.1 as f64)).changed() {
                            state.skeleton.bone_mut(&name).rest_length = length;
                            state.skeleton.solve_forward();
                            *needs_rebuild = true;
                        }
                        ui.separator();

                        // ── Constraint (editable) ──
                        ui.heading("Joint Constraint");

                        // Constraint type selector
                        let constraint_type = match &constraint {
                            JointConstraint::Fixed => 0,
                            JointConstraint::Free => 1,
                            JointConstraint::Hinge { .. } => 2,
                            JointConstraint::BallSocket { .. } => 3,
                        };
                        let mut new_type = constraint_type;
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut new_type, 0, "Fixed");
                            ui.selectable_value(&mut new_type, 1, "Free");
                            ui.selectable_value(&mut new_type, 2, "Hinge");
                            ui.selectable_value(&mut new_type, 3, "Ball");
                        });

                        if new_type != constraint_type {
                            // Changed constraint type
                            let new_constraint = match new_type {
                                0 => JointConstraint::Fixed,
                                1 => JointConstraint::Free,
                                2 => JointConstraint::Hinge {
                                    axis: Vec3::X,
                                    min_angle: 0.0,
                                    max_angle: 2.27,
                                },
                                3 => JointConstraint::BallSocket {
                                    cone_angle: 1.0,
                                    twist_min: -0.5,
                                    twist_max: 0.5,
                                },
                                _ => JointConstraint::Free,
                            };
                            state.skeleton.bone_mut(&name).constraint = new_constraint;
                            state.skeleton.solve_forward();
                            *needs_rebuild = true;
                        }

                        // Constraint parameters
                        match constraint {
                            JointConstraint::Fixed => {
                                ui.label("No movement allowed.");
                            }
                            JointConstraint::Free => {
                                ui.label("Full freedom of rotation.");
                            }
                            JointConstraint::Hinge { axis, mut min_angle, mut max_angle } => {
                                let mut min_deg = min_angle.to_degrees();
                                let mut max_deg = max_angle.to_degrees();

                                if ui.add(egui::Slider::new(&mut min_deg, -180.0..=180.0).text("Min").step_by(1.0)).changed() {
                                    min_angle = min_deg.to_radians();
                                    state.skeleton.bone_mut(&name).constraint = JointConstraint::Hinge {
                                        axis, min_angle, max_angle,
                                    };
                                    state.skeleton.solve_forward();
                                    *needs_rebuild = true;
                                }
                                if ui.add(egui::Slider::new(&mut max_deg, -180.0..=180.0).text("Max").step_by(1.0)).changed() {
                                    max_angle = max_deg.to_radians();
                                    state.skeleton.bone_mut(&name).constraint = JointConstraint::Hinge {
                                        axis, min_angle, max_angle,
                                    };
                                    state.skeleton.solve_forward();
                                    *needs_rebuild = true;
                                }

                                // Hinge axis selector
                                ui.label(format!("Axis: ({:.1}, {:.1}, {:.1})", axis.x, axis.y, axis.z));
                                ui.horizontal(|ui| {
                                    if ui.small_button("X").clicked() {
                                        state.skeleton.bone_mut(&name).constraint = JointConstraint::Hinge {
                                            axis: Vec3::X, min_angle, max_angle,
                                        };
                                        *needs_rebuild = true;
                                    }
                                    if ui.small_button("Y").clicked() {
                                        state.skeleton.bone_mut(&name).constraint = JointConstraint::Hinge {
                                            axis: Vec3::Y, min_angle, max_angle,
                                        };
                                        *needs_rebuild = true;
                                    }
                                    if ui.small_button("Z").clicked() {
                                        state.skeleton.bone_mut(&name).constraint = JointConstraint::Hinge {
                                            axis: Vec3::Z, min_angle, max_angle,
                                        };
                                        *needs_rebuild = true;
                                    }
                                });
                            }
                            JointConstraint::BallSocket { mut cone_angle, mut twist_min, mut twist_max } => {
                                let mut cone_deg = cone_angle.to_degrees();
                                if ui.add(egui::Slider::new(&mut cone_deg, 0.0..=180.0).text("Cone").step_by(1.0)).changed() {
                                    cone_angle = cone_deg.to_radians();
                                    state.skeleton.bone_mut(&name).constraint = JointConstraint::BallSocket {
                                        cone_angle, twist_min, twist_max,
                                    };
                                    *needs_rebuild = true;
                                }
                                let mut tw_min_deg = twist_min.to_degrees();
                                let mut tw_max_deg = twist_max.to_degrees();
                                if ui.add(egui::Slider::new(&mut tw_min_deg, -180.0..=0.0).text("Twist min").step_by(1.0)).changed() {
                                    twist_min = tw_min_deg.to_radians();
                                    state.skeleton.bone_mut(&name).constraint = JointConstraint::BallSocket {
                                        cone_angle, twist_min, twist_max,
                                    };
                                    *needs_rebuild = true;
                                }
                                if ui.add(egui::Slider::new(&mut tw_max_deg, 0.0..=180.0).text("Twist max").step_by(1.0)).changed() {
                                    twist_max = tw_max_deg.to_radians();
                                    state.skeleton.bone_mut(&name).constraint = JointConstraint::BallSocket {
                                        cone_angle, twist_min, twist_max,
                                    };
                                    *needs_rebuild = true;
                                }
                            }
                        }
                        ui.separator();

                        // ── Position ──
                        ui.heading("Position");
                        ui.label(format!("Start: ({:.1}, {:.1}, {:.1})", world_pos.x, world_pos.y, world_pos.z));
                        ui.label(format!("End:   ({:.1}, {:.1}, {:.1})", world_end.x, world_end.y, world_end.z));

                        let (axis, angle) = local_rot.to_axis_angle();
                        if angle.abs() > 0.001 {
                            ui.label(format!("Rotation: {:.1} around ({:.2},{:.2},{:.2})",
                                angle.to_degrees(), axis.x, axis.y, axis.z));
                        } else {
                            ui.label("Rotation: rest pose");
                        }
                        ui.separator();

                        // ── Actions ──
                        ui.heading("Actions");
                        ui.horizontal(|ui| {
                            if ui.button("Reset Rotation").clicked() {
                                state.skeleton.set_rotation(&name, Quat::IDENTITY);
                                state.skeleton.solve_forward();
                                *needs_rebuild = true;
                            }
                            if ui.button("Focus (F)").clicked() {
                                let mid = (world_pos + world_end) * 0.5;
                                camera.focus(mid);
                            }
                        });

                        // Add child bone
                        if ui.button("+ Add Child Bone").clicked() {
                            let child_name = format!("bone_{}", state.skeleton.bone_count());
                            let dir = if length > 0.001 {
                                (world_end - world_pos).normalize()
                            } else {
                                Vec3::Y
                            };
                            state.skeleton.add_bone(
                                &child_name, &name,
                                length * 0.5, // half parent length
                                dir,
                                JointConstraint::BallSocket {
                                    cone_angle: 1.0,
                                    twist_min: -0.5,
                                    twist_max: 0.5,
                                },
                            );
                            state.skeleton.solve_forward();
                            *needs_rebuild = true;
                        }

                    } else {
                        ui.label("No bone selected.");
                        ui.label("");
                        ui.label("Click a bone to select it.");
                        ui.label("Drag empty = orbit camera");
                        ui.label("Drag bone = rotate it");
                        ui.label("MMB drag = pan");
                        ui.label("Scroll = zoom");
                        ui.label("F = focus | WASD = fly");
                        ui.label("Space = reset camera");

                        ui.separator();

                        // Add root-level bone (for building from scratch)
                        if ui.button("+ Add Bone to Root").clicked() {
                            let child_name = format!("bone_{}", state.skeleton.bone_count());
                            state.skeleton.add_bone(
                                &child_name, "pelvis",
                                10.0 * state.skeleton_scale,
                                Vec3::Y,
                                JointConstraint::BallSocket {
                                    cone_angle: 1.0,
                                    twist_min: -0.5,
                                    twist_max: 0.5,
                                },
                            );
                            state.skeleton.solve_forward();
                            *needs_rebuild = true;
                        }
                    }

                    ui.separator();
                    ui.heading("Skeleton");
                    ui.label(format!("Bones: {}", state.skeleton.bone_count()));

                    let mut new_scale = state.skeleton_scale;
                    if ui.add(egui::Slider::new(&mut new_scale, 0.5..=10.0).text("Scale").step_by(0.1)).changed() {
                        state.skeleton_scale = new_scale;
                        let preset = state.preset_name.clone();
                        state.load_preset(&preset);
                        *needs_rebuild = true;
                    }

                    // Control point info (when selected)
                    if let Selection::ControlPoint(bid, idx) = state.selection {
                        ui.separator();
                        ui.heading("Control Point");
                        let bone_name = state.skeleton.bone_by_id(bid).name.clone();
                        if let Some(profile) = state.profile_for_bone(&bone_name) {
                            if idx < profile.points.len() {
                                let cp = &profile.points[idx];
                                ui.label(format!("Bone: {}", bone_name));
                                ui.label(format!("Position: {:.0}%", cp.t * 100.0));
                                ui.label(format!("Radius: {:.1}x ({:.1} units)",
                                    cp.radius_mul, profile.base_radius * cp.radius_mul));
                                ui.label("Scroll to adjust thickness");
                            }
                        }
                    }
                });
            });

            // ── Animation Timeline (above status bar) ──
            egui::TopBottomPanel::bottom("timeline").default_height(100.0).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Animation");
                    ui.separator();

                    // Transport controls
                    if state.anim_player.playing {
                        if ui.button("||  Pause").clicked() {
                            state.anim_player.playing = false;
                        }
                    } else {
                        if ui.button("|>  Play").clicked() {
                            if state.animation.keyframe_count() >= 2 {
                                state.anim_player.anim = Some(state.animation.clone());
                                state.anim_player.time = 0.0;
                                state.anim_player.playing = true;
                            }
                        }
                    }

                    if ui.button("[+] Capture Pose").clicked() {
                        let label = format!("Pose {}", state.animation.keyframe_count() + 1);
                        state.animation.capture(&state.skeleton, &label);
                    }

                    ui.separator();
                    ui.checkbox(&mut state.animation.looping, "Loop");

                    let mut dur = state.animation.transition_duration;
                    ui.label("Speed:");
                    if ui.add(egui::Slider::new(&mut dur, 0.1..=3.0).step_by(0.1).suffix("s")).changed() {
                        state.animation.transition_duration = dur;
                    }
                });

                ui.separator();

                // Keyframe chevrons
                if state.animation.keyframes.is_empty() {
                    ui.label("No keyframes. Pose the skeleton, then click [+] Capture Pose.");
                } else {
                    ui.horizontal(|ui| {
                        for (i, kf) in state.animation.keyframes.iter().enumerate() {
                            let is_current = if let Some(ref anim) = state.anim_player.anim {
                                let dur = anim.transition_duration;
                                let idx = (state.anim_player.time / dur) as usize;
                                idx % anim.keyframes.len() == i
                            } else {
                                false
                            };

                            let label = if is_current {
                                format!("[>>{}<<]", kf.label)
                            } else {
                                format!("[{}]", kf.label)
                            };

                            if ui.selectable_label(is_current, &label).clicked() {
                                // Jump to this keyframe — apply its rotations
                                let bone_count = state.skeleton.bone_count().min(kf.rotations.len());
                                for bi in 0..bone_count {
                                    let bname = state.skeleton.bones()[bi].name.clone();
                                    state.skeleton.set_rotation(&bname, kf.rotations[bi]);
                                }
                                state.skeleton.solve_forward();
                                *needs_rebuild = true;
                            }

                            // Easing selector between keyframes
                            if i < state.animation.keyframes.len() - 1 {
                                let easing_idx = i;
                                if easing_idx < state.animation.easings.len() {
                                    let easing_label = match state.animation.easings[easing_idx] {
                                        Easing::Linear => "---",
                                        Easing::EaseInOut => "~S~",
                                        Easing::EaseIn => "~/>",
                                        Easing::EaseOut => "<\\~",
                                    };
                                    if ui.small_button(easing_label).clicked() {
                                        // Cycle easing
                                        state.animation.easings[easing_idx] = match state.animation.easings[easing_idx] {
                                            Easing::Linear => Easing::EaseInOut,
                                            Easing::EaseInOut => Easing::EaseIn,
                                            Easing::EaseIn => Easing::EaseOut,
                                            Easing::EaseOut => Easing::Linear,
                                        };
                                    }
                                }
                            }
                        }

                        // Clear all keyframes
                        if state.animation.keyframe_count() > 0 {
                            ui.separator();
                            if ui.small_button("Clear All").clicked() {
                                state.animation = Animation::new(&state.animation.name);
                                state.anim_player.playing = false;
                            }
                        }
                    });

                    // Progress bar when playing
                    if state.anim_player.playing {
                        let n = state.animation.keyframe_count();
                        let total = if state.animation.looping { n } else { n - 1 };
                        let total_dur = total as f32 * state.animation.transition_duration;
                        let progress = state.anim_player.time / total_dur;
                        ui.add(egui::ProgressBar::new(progress.clamp(0.0, 1.0))
                            .text(format!("{:.1}s / {:.1}s", state.anim_player.time, total_dur)));
                    }
                }
            });
        })
    }

    fn render(&mut self) {
        if self.needs_rebuild {
            self.rebuild_bone_mesh();
            if self.state.show_voxels {
                self.body_needs_rebuild = true;
            }
        }
        if self.body_needs_rebuild {
            self.rebuild_body_mesh();
        }

        if self.state.auto_rotate {
            self.camera.yaw += 0.005;
            self.needs_rebuild = true;
        }

        // Advance animation playback
        if self.state.anim_player.playing {
            let changed = self.state.anim_player.update(1.0 / 60.0, &mut self.state.skeleton);
            if changed {
                self.needs_rebuild = true;
            }
        }

        // Update bone uniforms
        let config = self.config.as_ref().unwrap();
        let aspect = config.width as f32 / config.height as f32;
        let view = self.camera.view();
        let proj = self.camera.proj(aspect);
        let eye = self.camera.eye();
        let uniforms = BoneUniforms {
            view_proj: (proj * view).to_cols_array_2d(),
            eye_pos: [eye.x, eye.y, eye.z, 1.0],
            highlight_color: [1.0, 1.0, 0.0, 1.0],
        };
        self.queue.as_ref().unwrap().write_buffer(
            self.bone_uniform_buffer.as_ref().unwrap(), 0, bytemuck::bytes_of(&uniforms),
        );

        // Update body uniforms (same camera, different lighting)
        if self.state.show_voxels {
            let body_uniforms = MeshUniforms::new(
                view, proj, eye, Vec3::new(0.3, -0.8, 0.5).normalize(),
            );
            self.queue.as_ref().unwrap().write_buffer(
                self.body_uniform_buffer.as_ref().unwrap(), 0, bytemuck::bytes_of(&body_uniforms),
            );
        }

        // Run egui
        let full_output = self.draw_egui();

        let window = self.window.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let surface = self.surface.as_ref().unwrap();
        let config = self.config.as_ref().unwrap();

        self.egui_state.as_mut().unwrap().handle_platform_output(window, full_output.platform_output);
        let tris = self.egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
        for (id, delta) in &full_output.textures_delta.set {
            self.egui_renderer.as_mut().unwrap().update_texture(device, queue, *id, delta);
        }

        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [config.width, config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        let mut encoder = device.create_command_encoder(&Default::default());
        self.egui_renderer.as_mut().unwrap().update_buffers(device, queue, &mut encoder, &tris, &screen);

        // Get frame
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => { surface.configure(device, config); return; }
        };
        let view_tex = frame.texture.create_view(&Default::default());

        // Render pass: bones first
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Bone Render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.08, g: 0.08, b: 0.12, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw body mesh first (behind bones)
            if self.state.show_voxels {
                if let Some(body_mesh) = &self.body_gpu_mesh {
                    pass.set_pipeline(self.body_pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, self.body_bind_group.as_ref().unwrap(), &[]);
                    pass.set_vertex_buffer(0, body_mesh.vertex_buffer.slice(..));
                    pass.set_index_buffer(body_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..body_mesh.index_count, 0, 0..1);
                }
            }

            // Draw bones on top
            pass.set_pipeline(self.bone_pipeline.as_ref().unwrap());
            pass.set_bind_group(0, self.bone_bind_group.as_ref().unwrap(), &[]);
            if let Some(vb) = &self.bone_vertex_buffer {
                pass.set_vertex_buffer(0, vb.slice(..));
                pass.draw(0..self.bone_vertex_count, 0..1);
            }
        }

        // Egui render pass (uses forget_lifetime() for RenderPass<'static>)
        let egui_renderer = self.egui_renderer.as_ref().unwrap();
        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view_tex,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            egui_renderer.render(
                &mut render_pass.forget_lifetime(),
                &tris,
                &screen,
            );
        }

        queue.submit(std::iter::once(encoder.finish()));
        frame.present();

        for id in &full_output.textures_delta.free {
            self.egui_renderer.as_mut().unwrap().free_texture(id);
        }

        self.window.as_ref().unwrap().request_redraw();
        self.frame += 1;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
        if self.window.is_some() { return; }
        let w = Arc::new(el.create_window(
            Window::default_attributes()
                .with_title("PROMETHEUS EDITOR \u{2014} Skeleton Mode")
                .with_inner_size(winit::dpi::LogicalSize::new(1400, 900))
        ).unwrap());
        self.init_gpu(w);
    }

    fn window_event(&mut self, el: &winit::event_loop::ActiveEventLoop, _wid: winit::window::WindowId, event: WindowEvent) {
        // Let egui handle events first
        if let Some(state) = &mut self.egui_state {
            let response = state.on_window_event(self.window.as_ref().unwrap(), &event);
            if response.consumed {
                return; // egui consumed this event
            }
        }

        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => self.render(),

            WindowEvent::Resized(size) => {
                if let (Some(device), Some(surface), Some(config)) =
                    (self.device.as_ref(), self.surface.as_ref(), self.config.as_mut()) {
                    config.width = size.width.max(1);
                    config.height = size.height.max(1);
                    surface.configure(device, config);
                    let (_, dv) = render_mesh::create_depth_texture(device, config.width, config.height);
                    self.depth_view = Some(dv);
                }
            }

            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                let pressed = btn_state.is_pressed();
                match button {
                    MouseButton::Middle => self.mmb_dragging = pressed,
                    MouseButton::Left => {
                        if pressed && !self.lmb_dragging {
                            if let Some(config) = &self.config {
                                let w = config.width as f32;
                                let h = config.height as f32;
                                let mx = self.mouse_pos.0 as f32;
                                let my = self.mouse_pos.1 as f32;
                                let view = self.camera.view();
                                let proj = self.camera.proj(w / h);

                                // Priority 1: try to pick a control point (in voxel mode)
                                let cp_pick = pick_control_point(
                                    &self.state, mx, my, w, h, view, proj,
                                );

                                if let Some((bone_id, cp_idx)) = cp_pick {
                                    self.state.selection = Selection::ControlPoint(bone_id, cp_idx);
                                    self.state.selected_bone = Some(bone_id);
                                    self.dragging_bone = false; // don't orbit, but don't drag-rotate either
                                } else {
                                    // Priority 2: try to pick a bone
                                    let bone_pick = pick_bone(
                                        &self.state.skeleton, mx, my, w, h, view, proj,
                                    );

                                    if let Some(bone_id) = bone_pick {
                                        self.state.selection = Selection::Bone(bone_id);
                                        self.state.selected_bone = Some(bone_id);
                                        self.dragging_bone = true;
                                    } else {
                                        self.state.selection = Selection::None;
                                        self.dragging_bone = false;
                                    }
                                }
                                self.needs_rebuild = true;
                            }
                        }
                        if !pressed {
                            self.dragging_bone = false;
                        }
                        self.lmb_dragging = pressed;
                    }
                    _ => {}
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.last_mouse.0;
                let dy = position.y - self.last_mouse.1;

                if self.mmb_dragging {
                    self.camera.pan(dx as f32, dy as f32);
                }
                if self.lmb_dragging {
                    if self.dragging_bone {
                        // Drag on bone → rotate selected bone
                        if let Some(bone_id) = self.state.selected_bone {
                            let bone = self.state.skeleton.bone_by_id(bone_id);
                            let name = bone.name.clone();
                            let current_rot = bone.local_rotation;

                            let rot_speed = 0.005;
                            let delta = Quat::from_euler(
                                glam::EulerRot::YXZ,
                                -dx as f32 * rot_speed,
                                -dy as f32 * rot_speed,
                                0.0,
                            );
                            let new_rot = delta * current_rot;
                            self.state.skeleton.set_rotation(&name, new_rot);
                            self.state.skeleton.solve_forward();
                            self.needs_rebuild = true;
                        }
                    } else {
                        // Drag on empty space → orbit camera
                        self.camera.orbit(dx as f32, dy as f32);
                    }
                }

                self.last_mouse = (position.x, position.y);
                self.mouse_pos = (position.x, position.y);
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 / 50.0,
                };

                match self.state.selection {
                    Selection::ControlPoint(bone_id, cp_idx) => {
                        // Scroll on control point → adjust radius
                        let bone_name = self.state.skeleton.bone_by_id(bone_id).name.clone();
                        if let Some(profile) = self.state.profile_for_bone_mut(&bone_name) {
                            if cp_idx < profile.points.len() {
                                let step = 0.05;
                                profile.points[cp_idx].radius_mul =
                                    (profile.points[cp_idx].radius_mul + scroll * step)
                                    .clamp(0.1, 5.0);
                                self.needs_rebuild = true;
                                self.body_needs_rebuild = true;
                            }
                        }
                    }
                    Selection::Bone(bone_id) if self.lmb_dragging => {
                        // Scroll while dragging bone → adjust bone length
                        // (This requires modifying rest_length which is tricky
                        //  since Skeleton doesn't expose mutation of rest_length directly.
                        //  For now, we'll use this to adjust all control point radii of this bone)
                        let bone_name = self.state.skeleton.bone_by_id(bone_id).name.clone();
                        if let Some(profile) = self.state.profile_for_bone_mut(&bone_name) {
                            let step = 0.05;
                            for cp in &mut profile.points {
                                cp.radius_mul = (cp.radius_mul + scroll * step).clamp(0.1, 5.0);
                            }
                            self.needs_rebuild = true;
                            self.body_needs_rebuild = true;
                        }
                    }
                    _ => {
                        // No special selection → zoom camera
                        self.camera.zoom(scroll);
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } if event.state.is_pressed() => {
                let fly_speed = self.camera.distance * 0.05;
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::Escape) => el.exit(),
                    // WASD camera fly
                    PhysicalKey::Code(KeyCode::KeyW) => {
                        let fwd = (self.camera.target - self.camera.eye()).normalize();
                        self.camera.target += fwd * fly_speed;
                    }
                    PhysicalKey::Code(KeyCode::KeyS) => {
                        let fwd = (self.camera.target - self.camera.eye()).normalize();
                        self.camera.target -= fwd * fly_speed;
                    }
                    PhysicalKey::Code(KeyCode::KeyA) => {
                        let fwd = (self.camera.target - self.camera.eye()).normalize();
                        let right = fwd.cross(Vec3::Y).normalize();
                        self.camera.target -= right * fly_speed;
                    }
                    PhysicalKey::Code(KeyCode::KeyD) => {
                        let fwd = (self.camera.target - self.camera.eye()).normalize();
                        let right = fwd.cross(Vec3::Y).normalize();
                        self.camera.target += right * fly_speed;
                    }
                    // Space = reset camera to default
                    PhysicalKey::Code(KeyCode::Space) => {
                        self.camera = OrbitCamera::new();
                        if self.state.preset_name == "Cat" {
                            self.camera.target = Vec3::new(0.0, self.state.skeleton_scale * 20.0, 0.0);
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyF) => {
                        if let Some(id) = self.state.selected_bone {
                            let bone = self.state.skeleton.bone_by_id(id);
                            let mid = (bone.world_position + bone.world_end_position) * 0.5;
                            self.camera.focus(mid);
                        }
                    }
                    PhysicalKey::Code(KeyCode::KeyT) => {
                        self.state.auto_rotate = !self.state.auto_rotate;
                    }
                    PhysicalKey::Code(KeyCode::Digit1) => {
                        self.camera.yaw = 0.0;
                        self.camera.pitch = 0.0;
                    }
                    PhysicalKey::Code(KeyCode::Digit2) => {
                        self.camera.yaw = std::f32::consts::FRAC_PI_2;
                        self.camera.pitch = 0.0;
                    }
                    PhysicalKey::Code(KeyCode::Digit3) => {
                        self.camera.yaw = 0.0;
                        self.camera.pitch = 1.3;
                    }
                    PhysicalKey::Code(KeyCode::Delete) => {
                        if let Some(id) = self.state.selected_bone {
                            let name = self.state.skeleton.bone_by_id(id).name.clone();
                            self.state.skeleton.set_rotation(&name, Quat::IDENTITY);
                            self.state.skeleton.solve_forward();
                            self.needs_rebuild = true;
                        }
                    }
                    _ => {}
                }
            }

            _ => {}
        }
    }
}

// ─── Debug Display for EditorMode ───────────────────────────
impl std::fmt::Debug for EditorMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditorMode::Skeleton => write!(f, "Skeleton"),
        }
    }
}

// ─── Entry Point ────────────────────────────────────────────

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();

    // CLI mode for AI control
    if args.iter().any(|a| a == "--cli") {
        let script = args.iter().position(|a| a == "--cli")
            .and_then(|i| args.get(i + 1))
            .map(|s| s.as_str());
        cli::run_cli(script);
        return;
    }

    println!();
    println!("  ═══════════════════════════════════════");
    println!("  PROMETHEUS EDITOR v0.1-alpha");
    println!("  Skeleton + Animation + Voxel Preview");
    println!("  ═══════════════════════════════════════");
    println!();
    println!("  LMB drag empty = orbit | LMB drag bone = rotate");
    println!("  MMB drag = pan | Scroll = zoom | WASD = fly | Space = reset");
    println!("  F = focus | 1/2/3 = views | T = auto-rotate");
    println!("  --cli for AI headless mode");
    println!();

    let el = EventLoop::new().unwrap();
    let mut app = App::new();
    el.run_app(&mut app).unwrap();
}
