// ═══════════════════════════════════════════════════════════════
// PROMETHEUS EDITOR — Model File Format (.prom.json)
//
// Serializable model: skeleton + body profiles + animations.
// Compatible with prometheus-engine models/ directory.
// ═══════════════════════════════════════════════════════════════

use serde::{Serialize, Deserialize};
use prometheus_engine::core::skeleton::Skeleton;

/// A control point for body thickness along a bone
#[derive(Clone, Serialize, Deserialize)]
pub struct ControlPointData {
    pub t: f32,
    pub radius_mul: f32,
}

/// Body profile for one bone
#[derive(Clone, Serialize, Deserialize)]
pub struct BoneProfileData {
    pub bone_name: String,
    pub base_radius: f32,
    pub points: Vec<ControlPointData>,
}

/// One animation keyframe
#[derive(Clone, Serialize, Deserialize)]
pub struct KeyframeData {
    pub label: String,
    /// Rotation as [x,y,z,w] quaternion per bone (by index)
    pub rotations: Vec<[f32; 4]>,
}

/// Easing type
#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum EasingData {
    Linear,
    EaseInOut,
    EaseIn,
    EaseOut,
}

/// One animation clip
#[derive(Clone, Serialize, Deserialize)]
pub struct AnimationData {
    pub name: String,
    pub keyframes: Vec<KeyframeData>,
    pub easings: Vec<EasingData>,
    pub looping: bool,
    pub transition_duration: f32,
}

/// The complete model file
#[derive(Clone, Serialize, Deserialize)]
pub struct PrometheusModel {
    /// Format version
    pub version: u32,
    /// Model name
    pub name: String,
    /// Author
    pub author: String,
    /// The skeleton (bones, constraints, hierarchy)
    pub skeleton: Skeleton,
    /// Body profiles (thickness control points per bone)
    pub body_profiles: Vec<BoneProfileData>,
    /// Animations
    pub animations: Vec<AnimationData>,
    /// Scale used when creating
    pub scale: f32,
}

impl PrometheusModel {
    pub fn new(name: &str) -> Self {
        Self {
            version: 1,
            name: name.to_string(),
            author: String::new(),
            skeleton: Skeleton::new("root"),
            body_profiles: Vec::new(),
            animations: Vec::new(),
            scale: 3.0,
        }
    }

    /// Save to JSON file
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Serialize error: {}", e))?;
        std::fs::write(path, json)
            .map_err(|e| format!("Write error: {}", e))?;
        Ok(())
    }

    /// Load from JSON file
    pub fn load(path: &str) -> Result<Self, String> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| format!("Read error: {}", e))?;
        serde_json::from_str(&json)
            .map_err(|e| format!("Parse error: {}", e))
    }

    /// Copy model file to engine's models/ directory
    pub fn install_to_engine(src_path: &str, engine_dir: &str) -> Result<String, String> {
        let models_dir = format!("{}/models", engine_dir);
        std::fs::create_dir_all(&models_dir)
            .map_err(|e| format!("Cannot create models dir: {}", e))?;

        let filename = std::path::Path::new(src_path)
            .file_name()
            .ok_or("Invalid path")?
            .to_str()
            .ok_or("Invalid filename")?;

        let dest = format!("{}/{}", models_dir, filename);
        std::fs::copy(src_path, &dest)
            .map_err(|e| format!("Copy error: {}", e))?;

        Ok(dest)
    }
}
