//! Build configuration and profiles.
//!
//! This module defines build configurations, compiler flags, and
//! target platforms for the Sounio compiler.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Build profile
    pub profile: BuildProfile,

    /// Target triple
    pub target: TargetTriple,

    /// Compiler flags
    pub flags: CompilerFlags,

    /// Feature configuration
    pub features: FeatureConfig,

    /// Output paths
    pub paths: BuildPaths,

    /// Optimization settings
    pub opt: OptimizationConfig,
}

impl BuildConfig {
    /// Create a new build config with defaults
    pub fn new() -> Self {
        BuildConfig {
            profile: BuildProfile::Dev,
            target: TargetTriple::native(),
            flags: CompilerFlags::default(),
            features: FeatureConfig::default(),
            paths: BuildPaths::default(),
            opt: OptimizationConfig::default(),
        }
    }

    /// Create a config for development builds
    pub fn dev() -> Self {
        let mut config = Self::new();
        config.profile = BuildProfile::Dev;
        config.opt.level = OptLevel::O0;
        config.flags.debug_info = true;
        config.flags.incremental = true;
        config
    }

    /// Create a config for release builds
    pub fn release() -> Self {
        let mut config = Self::new();
        config.profile = BuildProfile::Release;
        config.opt.level = OptLevel::O3;
        config.flags.debug_info = false;
        config.flags.incremental = false;
        config.opt.lto = true;
        config
    }

    /// Create a config for test builds
    pub fn test() -> Self {
        let mut config = Self::new();
        config.profile = BuildProfile::Test;
        config.opt.level = OptLevel::O0;
        config.flags.debug_info = true;
        config.flags.test_mode = true;
        config
    }

    /// Create a config for benchmark builds
    pub fn bench() -> Self {
        let mut config = Self::new();
        config.profile = BuildProfile::Bench;
        config.opt.level = OptLevel::O3;
        config.flags.debug_info = true;
        config.opt.lto = true;
        config
    }

    /// Compute a hash of this configuration (for cache keys)
    pub fn hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", self).hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Build profile (dev, release, test, bench)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BuildProfile {
    /// Development build (fast compile, slower runtime)
    Dev,

    /// Release build (slow compile, fast runtime)
    Release,

    /// Test build (with test harness)
    Test,

    /// Benchmark build (optimized with debug info)
    Bench,
}

impl BuildProfile {
    /// Get profile name as string
    pub fn name(&self) -> &'static str {
        match self {
            BuildProfile::Dev => "dev",
            BuildProfile::Release => "release",
            BuildProfile::Test => "test",
            BuildProfile::Bench => "bench",
        }
    }

    /// Parse from string
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "dev" | "debug" => Some(BuildProfile::Dev),
            "release" | "prod" => Some(BuildProfile::Release),
            "test" => Some(BuildProfile::Test),
            "bench" | "benchmark" => Some(BuildProfile::Bench),
            _ => None,
        }
    }
}

/// Target triple (architecture-vendor-os)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TargetTriple {
    /// Architecture (x86_64, aarch64, etc.)
    pub arch: String,

    /// Vendor (unknown, apple, pc, etc.)
    pub vendor: String,

    /// Operating system (linux, darwin, windows, etc.)
    pub os: String,

    /// ABI (gnu, musl, msvc, etc.)
    pub abi: Option<String>,
}

impl TargetTriple {
    /// Create a new target triple
    pub fn new(arch: String, vendor: String, os: String, abi: Option<String>) -> Self {
        TargetTriple {
            arch,
            vendor,
            os,
            abi,
        }
    }

    /// Get native target triple
    pub fn native() -> Self {
        #[cfg(target_arch = "x86_64")]
        let arch = "x86_64".to_string();
        #[cfg(target_arch = "aarch64")]
        let arch = "aarch64".to_string();
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let arch = "unknown".to_string();

        #[cfg(target_vendor = "apple")]
        let vendor = "apple".to_string();
        #[cfg(target_vendor = "pc")]
        let vendor = "pc".to_string();
        #[cfg(not(any(target_vendor = "apple", target_vendor = "pc")))]
        let vendor = "unknown".to_string();

        #[cfg(target_os = "linux")]
        let os = "linux".to_string();
        #[cfg(target_os = "macos")]
        let os = "darwin".to_string();
        #[cfg(target_os = "windows")]
        let os = "windows".to_string();
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        let os = "unknown".to_string();

        #[cfg(target_env = "gnu")]
        let abi = Some("gnu".to_string());
        #[cfg(target_env = "musl")]
        let abi = Some("musl".to_string());
        #[cfg(target_env = "msvc")]
        let abi = Some("msvc".to_string());
        #[cfg(not(any(target_env = "gnu", target_env = "musl", target_env = "msvc")))]
        let abi = None;

        TargetTriple::new(arch, vendor, os, abi)
    }

    /// Parse from string (e.g., "x86_64-unknown-linux-gnu")
    pub fn parse(s: &str) -> Result<Self, ConfigError> {
        let parts: Vec<&str> = s.split('-').collect();

        match parts.as_slice() {
            [arch, vendor, os] => Ok(TargetTriple::new(
                arch.to_string(),
                vendor.to_string(),
                os.to_string(),
                None,
            )),
            [arch, vendor, os, abi] => Ok(TargetTriple::new(
                arch.to_string(),
                vendor.to_string(),
                os.to_string(),
                Some(abi.to_string()),
            )),
            _ => Err(ConfigError::InvalidTarget(s.to_string())),
        }
    }

    /// Convert to string representation
    pub fn to_string(&self) -> String {
        if let Some(abi) = &self.abi {
            format!("{}-{}-{}-{}", self.arch, self.vendor, self.os, abi)
        } else {
            format!("{}-{}-{}", self.arch, self.vendor, self.os)
        }
    }

    /// Check if this is a Windows target
    pub fn is_windows(&self) -> bool {
        self.os == "windows"
    }

    /// Check if this is a Unix target
    pub fn is_unix(&self) -> bool {
        matches!(self.os.as_str(), "linux" | "darwin" | "freebsd" | "openbsd")
    }
}

/// Compiler flags
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilerFlags {
    /// Generate debug information
    pub debug_info: bool,

    /// Enable incremental compilation
    pub incremental: bool,

    /// Test mode (link with test harness)
    pub test_mode: bool,

    /// Verbose output
    pub verbose: bool,

    /// Treat warnings as errors
    pub warnings_as_errors: bool,

    /// Emit LLVM IR
    pub emit_llvm: bool,

    /// Emit assembly
    pub emit_asm: bool,

    /// Strip symbols
    pub strip: bool,

    /// Additional defines
    pub defines: HashMap<String, String>,
}

impl Default for CompilerFlags {
    fn default() -> Self {
        CompilerFlags {
            debug_info: false,
            incremental: true,
            test_mode: false,
            verbose: false,
            warnings_as_errors: false,
            emit_llvm: false,
            emit_asm: false,
            strip: false,
            defines: HashMap::new(),
        }
    }
}

/// Feature configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeatureConfig {
    /// Enabled features
    pub enabled: Vec<String>,

    /// Disabled features
    pub disabled: Vec<String>,
}

impl FeatureConfig {
    /// Check if a feature is enabled
    pub fn is_enabled(&self, feature: &str) -> bool {
        self.enabled.contains(&feature.to_string()) && !self.disabled.contains(&feature.to_string())
    }

    /// Enable a feature
    pub fn enable(&mut self, feature: String) {
        if !self.enabled.contains(&feature) {
            self.enabled.push(feature.clone());
        }
        self.disabled.retain(|f| f != &feature);
    }

    /// Disable a feature
    pub fn disable(&mut self, feature: String) {
        if !self.disabled.contains(&feature) {
            self.disabled.push(feature.clone());
        }
        self.enabled.retain(|f| f != &feature);
    }
}

/// Build output paths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildPaths {
    /// Source directory
    pub source_dir: PathBuf,

    /// Build output directory
    pub build_dir: PathBuf,

    /// Cache directory
    pub cache_dir: PathBuf,

    /// Target directory (final outputs)
    pub target_dir: PathBuf,
}

impl Default for BuildPaths {
    fn default() -> Self {
        BuildPaths {
            source_dir: PathBuf::from("src"),
            build_dir: PathBuf::from("build"),
            cache_dir: PathBuf::from("build/cache"),
            target_dir: PathBuf::from("target"),
        }
    }
}

impl BuildPaths {
    /// Get profile-specific build directory
    pub fn profile_dir(&self, profile: BuildProfile) -> PathBuf {
        self.build_dir.join(profile.name())
    }

    /// Get artifact output path
    pub fn artifact_path(&self, profile: BuildProfile, name: &str) -> PathBuf {
        self.target_dir.join(profile.name()).join(name)
    }
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization level
    pub level: OptLevel,

    /// Enable link-time optimization
    pub lto: bool,

    /// Number of codegen units (more = faster compile, less optimization)
    pub codegen_units: usize,

    /// Enable inlining
    pub inline: bool,

    /// Loop unrolling
    pub unroll_loops: bool,

    /// Vectorization
    pub vectorize: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        OptimizationConfig {
            level: OptLevel::O0,
            lto: false,
            codegen_units: 16,
            inline: true,
            unroll_loops: true,
            vectorize: true,
        }
    }
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Moderate optimization
    O2,
    /// Aggressive optimization
    O3,
    /// Optimize for size
    Os,
    /// Aggressively optimize for size
    Oz,
}

impl OptLevel {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "0" => Some(OptLevel::O0),
            "1" => Some(OptLevel::O1),
            "2" => Some(OptLevel::O2),
            "3" => Some(OptLevel::O3),
            "s" => Some(OptLevel::Os),
            "z" => Some(OptLevel::Oz),
            _ => None,
        }
    }

    /// Convert to string
    pub fn to_str(&self) -> &'static str {
        match self {
            OptLevel::O0 => "0",
            OptLevel::O1 => "1",
            OptLevel::O2 => "2",
            OptLevel::O3 => "3",
            OptLevel::Os => "s",
            OptLevel::Oz => "z",
        }
    }

    /// Get LLVM optimization level
    pub fn llvm_level(&self) -> u8 {
        match self {
            OptLevel::O0 => 0,
            OptLevel::O1 => 1,
            OptLevel::O2 => 2,
            OptLevel::O3 | OptLevel::Os | OptLevel::Oz => 3,
        }
    }
}

/// Configuration errors
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid target triple: {0}")]
    InvalidTarget(String),

    #[error("Invalid optimization level: {0}")]
    InvalidOptLevel(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_profile() {
        assert_eq!(BuildProfile::Dev.name(), "dev");
        assert_eq!(BuildProfile::Release.name(), "release");

        assert_eq!(BuildProfile::from_name("dev"), Some(BuildProfile::Dev));
        assert_eq!(
            BuildProfile::from_name("release"),
            Some(BuildProfile::Release)
        );
        assert_eq!(BuildProfile::from_name("unknown"), None);
    }

    #[test]
    fn test_target_triple() {
        let native = TargetTriple::native();
        assert!(!native.arch.is_empty());
        assert!(!native.os.is_empty());

        let triple_str = native.to_string();
        assert!(triple_str.contains(&native.arch));
        assert!(triple_str.contains(&native.os));
    }

    #[test]
    fn test_target_parse() {
        let triple = TargetTriple::parse("x86_64-unknown-linux-gnu").unwrap();
        assert_eq!(triple.arch, "x86_64");
        assert_eq!(triple.vendor, "unknown");
        assert_eq!(triple.os, "linux");
        assert_eq!(triple.abi, Some("gnu".to_string()));

        let triple2 = TargetTriple::parse("aarch64-apple-darwin").unwrap();
        assert_eq!(triple2.arch, "aarch64");
        assert_eq!(triple2.vendor, "apple");
        assert_eq!(triple2.os, "darwin");
        assert_eq!(triple2.abi, None);
    }

    #[test]
    fn test_opt_level() {
        assert_eq!(OptLevel::from_str("0"), Some(OptLevel::O0));
        assert_eq!(OptLevel::from_str("3"), Some(OptLevel::O3));
        assert_eq!(OptLevel::from_str("s"), Some(OptLevel::Os));
        assert_eq!(OptLevel::from_str("invalid"), None);

        assert_eq!(OptLevel::O0.to_str(), "0");
        assert_eq!(OptLevel::O3.to_str(), "3");

        assert_eq!(OptLevel::O0.llvm_level(), 0);
        assert_eq!(OptLevel::O3.llvm_level(), 3);
    }

    #[test]
    fn test_build_config_presets() {
        let dev = BuildConfig::dev();
        assert_eq!(dev.profile, BuildProfile::Dev);
        assert_eq!(dev.opt.level, OptLevel::O0);
        assert!(dev.flags.debug_info);

        let release = BuildConfig::release();
        assert_eq!(release.profile, BuildProfile::Release);
        assert_eq!(release.opt.level, OptLevel::O3);
        assert!(!release.flags.debug_info);
        assert!(release.opt.lto);
    }

    #[test]
    fn test_feature_config() {
        let mut features = FeatureConfig::default();

        features.enable("smt".to_string());
        assert!(features.is_enabled("smt"));

        features.disable("smt".to_string());
        assert!(!features.is_enabled("smt"));
    }

    #[test]
    fn test_build_paths() {
        let paths = BuildPaths::default();

        let dev_dir = paths.profile_dir(BuildProfile::Dev);
        assert!(dev_dir.ends_with("build/dev"));

        let artifact = paths.artifact_path(BuildProfile::Release, "myapp");
        assert!(artifact.ends_with("target/release/myapp"));
    }

    #[test]
    fn test_config_hash() {
        let config1 = BuildConfig::dev();
        let config2 = BuildConfig::dev();
        let config3 = BuildConfig::release();

        let hash1 = config1.hash();
        let hash2 = config2.hash();
        let hash3 = config3.hash();

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }
}
