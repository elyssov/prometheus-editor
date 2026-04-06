// ═══════════════════════════════════════════════════════════════
// PROMETHEUS EDITOR — CLI API (headless mode for AI control)
//
// Usage: prometheus-editor --cli [script.txt]
//   or:  echo "commands" | prometheus-editor --cli
//
// Commands are line-based, results printed to stdout as JSON.
// Designed for AI agents (Lara) to control the editor via bash.
//
// Commands:
//   load_preset <Human|Cat>
//   set_scale <f32>
//   list_bones
//   select_bone <name>
//   set_rotation <bone> <x> <y> <z> <w>    — quaternion
//   set_hinge <bone> <min_deg> <max_deg>
//   set_ballsocket <bone> <cone_deg>
//   set_length <bone> <length>
//   set_thickness <bone> <point_idx> <multiplier>
//   capture_keyframe [label]
//   play_animation
//   stop_animation
//   export <path.prom.json>
//   import <path.prom.json>
//   install <path.prom.json> <engine_dir>
//   add_bone <parent_name> <name> <length> <dx> <dy> <dz>
//   info
//   quit
// ═══════════════════════════════════════════════════════════════

use std::io::{self, BufRead, Write};
use glam::{Vec3, Quat};
use prometheus_engine::core::skeleton::{Skeleton, JointConstraint};

use crate::model::*;
use crate::{EditorState, Animation, Selection};

/// Run CLI mode — reads commands from stdin, executes, prints results
pub fn run_cli(script_path: Option<&str>) {
    let mut state = EditorState::new();
    println!("{{\"status\":\"ready\",\"bones\":{}}}", state.skeleton.bone_count());

    let input: Box<dyn BufRead> = if let Some(path) = script_path {
        let file = std::fs::File::open(path).expect("Cannot open script file");
        Box::new(io::BufReader::new(file))
    } else {
        Box::new(io::BufReader::new(io::stdin()))
    };

    for line in input.lines() {
        let line = match line {
            Ok(l) => l.trim().to_string(),
            Err(_) => break,
        };
        if line.is_empty() || line.starts_with('#') { continue; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() { continue; }

        let result = execute_command(&mut state, &parts);
        println!("{}", result);
        io::stdout().flush().ok();

        if parts[0] == "quit" { break; }
    }
}

fn execute_command(state: &mut EditorState, parts: &[&str]) -> String {
    match parts[0] {
        "load_preset" => {
            if parts.len() < 2 { return err("usage: load_preset <Human|Cat>"); }
            state.load_preset(parts[1]);
            ok(&format!("Loaded {} ({} bones)", parts[1], state.skeleton.bone_count()))
        }

        "set_scale" => {
            if parts.len() < 2 { return err("usage: set_scale <f32>"); }
            let scale: f32 = match parts[1].parse() { Ok(v) => v, Err(_) => return err("invalid number") };
            state.skeleton_scale = scale;
            let preset = state.preset_name.clone();
            state.load_preset(&preset);
            ok(&format!("Scale set to {}", scale))
        }

        "list_bones" => {
            let bones: Vec<String> = state.skeleton.bones().iter()
                .map(|b| format!("{{\"id\":{},\"name\":\"{}\",\"length\":{:.2},\"parent\":{:?}}}",
                    b.id, b.name, b.rest_length, b.parent))
                .collect();
            format!("{{\"ok\":true,\"bones\":[{}]}}", bones.join(","))
        }

        "select_bone" => {
            if parts.len() < 2 { return err("usage: select_bone <name>"); }
            let bone = state.skeleton.bone(parts[1]);
            state.selected_bone = Some(bone.id);
            state.selection = Selection::Bone(bone.id);
            ok(&format!("Selected {}", parts[1]))
        }

        "set_rotation" => {
            if parts.len() < 6 { return err("usage: set_rotation <bone> <x> <y> <z> <w>"); }
            let x: f32 = parts[2].parse().unwrap_or(0.0);
            let y: f32 = parts[3].parse().unwrap_or(0.0);
            let z: f32 = parts[4].parse().unwrap_or(0.0);
            let w: f32 = parts[5].parse().unwrap_or(1.0);
            state.skeleton.set_rotation(parts[1], Quat::from_xyzw(x, y, z, w));
            state.skeleton.solve_forward();
            ok("Rotation set")
        }

        "set_hinge" => {
            if parts.len() < 4 { return err("usage: set_hinge <bone> <min_deg> <max_deg>"); }
            let min: f32 = parts[2].parse().unwrap_or(0.0);
            let max: f32 = parts[3].parse().unwrap_or(130.0);
            state.skeleton.bone_mut(parts[1]).constraint = JointConstraint::Hinge {
                axis: Vec3::X,
                min_angle: min.to_radians(),
                max_angle: max.to_radians(),
            };
            ok("Hinge set")
        }

        "set_ballsocket" => {
            if parts.len() < 3 { return err("usage: set_ballsocket <bone> <cone_deg>"); }
            let cone: f32 = parts[2].parse().unwrap_or(60.0);
            state.skeleton.bone_mut(parts[1]).constraint = JointConstraint::BallSocket {
                cone_angle: cone.to_radians(),
                twist_min: -0.5,
                twist_max: 0.5,
            };
            ok("BallSocket set")
        }

        "set_length" => {
            if parts.len() < 3 { return err("usage: set_length <bone> <length>"); }
            let len: f32 = parts[2].parse().unwrap_or(10.0);
            state.skeleton.bone_mut(parts[1]).rest_length = len;
            state.skeleton.solve_forward();
            ok(&format!("Length set to {}", len))
        }

        "set_thickness" => {
            if parts.len() < 4 { return err("usage: set_thickness <bone> <point_idx> <mul>"); }
            let idx: usize = parts[2].parse().unwrap_or(0);
            let mul: f32 = parts[3].parse().unwrap_or(1.0);
            if let Some(profile) = state.profile_for_bone_mut(parts[1]) {
                if idx < profile.points.len() {
                    profile.points[idx].radius_mul = mul.clamp(0.1, 5.0);
                    return ok(&format!("Thickness[{}] = {}", idx, mul));
                }
            }
            err("bone or point not found")
        }

        "capture_keyframe" => {
            let label = if parts.len() > 1 { parts[1..].join(" ") }
                else { format!("Pose {}", state.animation.keyframe_count() + 1) };
            state.animation.capture(&state.skeleton, &label);
            ok(&format!("Captured '{}' (total: {})", label, state.animation.keyframe_count()))
        }

        "add_bone" => {
            if parts.len() < 7 { return err("usage: add_bone <parent> <name> <length> <dx> <dy> <dz>"); }
            let len: f32 = parts[3].parse().unwrap_or(10.0);
            let dx: f32 = parts[4].parse().unwrap_or(0.0);
            let dy: f32 = parts[5].parse().unwrap_or(1.0);
            let dz: f32 = parts[6].parse().unwrap_or(0.0);
            state.skeleton.add_bone(
                parts[2], parts[1], len,
                Vec3::new(dx, dy, dz),
                JointConstraint::BallSocket { cone_angle: 1.0, twist_min: -0.5, twist_max: 0.5 },
            );
            state.skeleton.solve_forward();
            ok(&format!("Added bone '{}' (parent: '{}')", parts[2], parts[1]))
        }

        "export" => {
            if parts.len() < 2 { return err("usage: export <path.prom.json>"); }
            let model = build_model(state);
            match model.save(parts[1]) {
                Ok(()) => ok(&format!("Exported to {}", parts[1])),
                Err(e) => err(&e),
            }
        }

        "import" => {
            if parts.len() < 2 { return err("usage: import <path.prom.json>"); }
            match PrometheusModel::load(parts[1]) {
                Ok(model) => {
                    state.skeleton = model.skeleton;
                    state.skeleton_scale = model.scale;
                    state.skeleton.solve_forward();
                    ok(&format!("Imported '{}' ({} bones)", model.name, state.skeleton.bone_count()))
                }
                Err(e) => err(&e),
            }
        }

        "install" => {
            if parts.len() < 3 { return err("usage: install <path.prom.json> <engine_dir>"); }
            match PrometheusModel::install_to_engine(parts[1], parts[2]) {
                Ok(dest) => ok(&format!("Installed to {}", dest)),
                Err(e) => err(&e),
            }
        }

        "info" => {
            format!("{{\"ok\":true,\"bones\":{},\"scale\":{:.1},\"preset\":\"{}\",\"keyframes\":{},\"selection\":\"{:?}\"}}",
                state.skeleton.bone_count(),
                state.skeleton_scale,
                state.preset_name,
                state.animation.keyframe_count(),
                state.selection)
        }

        "quit" => ok("Bye"),

        _ => err(&format!("Unknown command: {}", parts[0])),
    }
}

fn build_model(state: &EditorState) -> PrometheusModel {
    let body_profiles: Vec<BoneProfileData> = state.body_profiles.iter().map(|bp| {
        BoneProfileData {
            bone_name: bp.bone_name.clone(),
            base_radius: bp.base_radius,
            points: bp.points.iter().map(|cp| ControlPointData {
                t: cp.t,
                radius_mul: cp.radius_mul,
            }).collect(),
        }
    }).collect();

    let keyframes: Vec<KeyframeData> = state.animation.keyframes.iter().map(|kf| {
        KeyframeData {
            label: kf.label.clone(),
            rotations: kf.rotations.iter().map(|q| [q.x, q.y, q.z, q.w]).collect(),
        }
    }).collect();

    let easings: Vec<EasingData> = state.animation.easings.iter().map(|e| {
        match e {
            crate::Easing::Linear => EasingData::Linear,
            crate::Easing::EaseInOut => EasingData::EaseInOut,
            crate::Easing::EaseIn => EasingData::EaseIn,
            crate::Easing::EaseOut => EasingData::EaseOut,
        }
    }).collect();

    let anim = AnimationData {
        name: state.animation.name.clone(),
        keyframes,
        easings,
        looping: state.animation.looping,
        transition_duration: state.animation.transition_duration,
    };

    let mut model = PrometheusModel::new(&state.preset_name);
    model.skeleton = state.skeleton.clone();
    model.body_profiles = body_profiles;
    model.animations = if anim.keyframes.is_empty() { vec![] } else { vec![anim] };
    model.scale = state.skeleton_scale;
    model.author = "Prometheus Editor".to_string();
    model
}

fn ok(msg: &str) -> String {
    format!("{{\"ok\":true,\"msg\":\"{}\"}}", msg)
}

fn err(msg: &str) -> String {
    format!("{{\"ok\":false,\"error\":\"{}\"}}", msg)
}
