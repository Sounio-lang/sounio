//! Build script support (build.d)
//!
//! Provides:
//! - Custom build logic execution
//! - Build environment variables
//! - Generated code and bindings
//! - Build script caching

use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::SystemTime;

/// Build script configuration
#[derive(Debug, Clone)]
pub struct BuildScriptConfig {
    /// Path to build script
    pub script: PathBuf,

    /// Output directory
    pub out_dir: PathBuf,

    /// Additional environment variables
    pub env: HashMap<String, String>,

    /// Package name
    pub pkg_name: String,

    /// Package version
    pub pkg_version: String,

    /// Target triple
    pub target: String,

    /// Build profile (dev, release, etc.)
    pub profile: String,

    /// Enabled features
    pub features: Vec<String>,

    /// Manifest directory
    pub manifest_dir: PathBuf,
}

impl Default for BuildScriptConfig {
    fn default() -> Self {
        BuildScriptConfig {
            script: PathBuf::from("build.sio"),
            out_dir: PathBuf::from("target/build"),
            env: HashMap::new(),
            pkg_name: String::new(),
            pkg_version: String::new(),
            target: "native".into(),
            profile: "dev".into(),
            features: Vec::new(),
            manifest_dir: PathBuf::from("."),
        }
    }
}

/// Build script output
#[derive(Debug, Clone, Default)]
pub struct BuildScriptOutput {
    /// Parsed instructions
    pub instructions: Vec<BuildInstruction>,

    /// Generated files
    pub generated_files: Vec<PathBuf>,

    /// Standard output
    pub stdout: String,

    /// Standard error
    pub stderr: String,

    /// Exit code
    pub exit_code: i32,

    /// Duration
    pub duration: std::time::Duration,
}

impl BuildScriptOutput {
    /// Check if build script succeeded
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    /// Get all errors from instructions
    pub fn errors(&self) -> Vec<&str> {
        self.instructions
            .iter()
            .filter_map(|i| {
                if let BuildInstruction::Error(msg) = i {
                    Some(msg.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Get all warnings from instructions
    pub fn warnings(&self) -> Vec<&str> {
        self.instructions
            .iter()
            .filter_map(|i| {
                if let BuildInstruction::Warning(msg) = i {
                    Some(msg.as_str())
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Build instruction from build script
#[derive(Debug, Clone)]
pub enum BuildInstruction {
    /// Rerun if file changed
    RerunIfChanged(PathBuf),

    /// Rerun if environment variable changed
    RerunIfEnvChanged(String),

    /// Link a library
    LinkLib { kind: LinkKind, name: String },

    /// Add library search path
    LinkSearch { kind: SearchKind, path: PathBuf },

    /// Define a cfg flag
    Cfg(String),

    /// Set environment variable for compilation
    Env { key: String, value: String },

    /// Include directory for FFI headers
    Include(PathBuf),

    /// Define a preprocessor macro
    Define { name: String, value: Option<String> },

    /// Warning message
    Warning(String),

    /// Error message
    Error(String),

    /// Metadata for other packages
    Metadata { key: String, value: String },
}

/// Library link kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkKind {
    /// Static library
    Static,

    /// Dynamic library
    Dylib,

    /// macOS framework
    Framework,
}

/// Library search kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchKind {
    /// Native library search
    Native,

    /// Framework search (macOS)
    Framework,

    /// All search paths
    All,
}

/// Build script runner
pub struct BuildScriptRunner {
    /// Configuration
    config: BuildScriptConfig,

    /// Cache directory
    cache_dir: PathBuf,

    /// Compiler path
    compiler: PathBuf,
}

impl BuildScriptRunner {
    /// Create a new runner
    pub fn new(config: BuildScriptConfig) -> Self {
        BuildScriptRunner {
            config,
            cache_dir: PathBuf::from("target/.build-scripts"),
            compiler: PathBuf::from("dc"),
        }
    }

    /// Set cache directory
    pub fn cache_dir(mut self, path: PathBuf) -> Self {
        self.cache_dir = path;
        self
    }

    /// Set compiler path
    pub fn compiler(mut self, path: PathBuf) -> Self {
        self.compiler = path;
        self
    }

    /// Run the build script
    pub fn run(&self) -> Result<BuildScriptOutput, BuildScriptError> {
        let start = std::time::Instant::now();

        // Ensure directories exist
        fs::create_dir_all(&self.config.out_dir)?;
        fs::create_dir_all(&self.cache_dir)?;

        // Compile script if needed
        let script_exe = self.cache_dir.join("build_script");

        #[cfg(windows)]
        let script_exe = script_exe.with_extension("exe");

        if self.needs_compile(&script_exe) {
            self.compile_script(&script_exe)?;
        }

        // Run script
        let mut output = self.run_script(&script_exe)?;
        output.duration = start.elapsed();

        Ok(output)
    }

    /// Check if script needs recompilation
    fn needs_compile(&self, exe: &Path) -> bool {
        if !exe.exists() {
            return true;
        }

        let script_mtime = fs::metadata(&self.config.script)
            .and_then(|m| m.modified())
            .ok();

        let exe_mtime = fs::metadata(exe).and_then(|m| m.modified()).ok();

        match (script_mtime, exe_mtime) {
            (Some(s), Some(e)) => s > e,
            _ => true,
        }
    }

    /// Compile the build script
    fn compile_script(&self, output: &Path) -> Result<(), BuildScriptError> {
        let status = Command::new(&self.compiler)
            .arg(&self.config.script)
            .arg("-o")
            .arg(output)
            .arg("--profile")
            .arg("dev")
            .status()?;

        if !status.success() {
            return Err(BuildScriptError::Compilation(format!(
                "Build script compilation failed with exit code {:?}",
                status.code()
            )));
        }

        Ok(())
    }

    /// Run the compiled build script
    fn run_script(&self, exe: &Path) -> Result<BuildScriptOutput, BuildScriptError> {
        let mut cmd = Command::new(exe);

        // Set standard environment variables
        cmd.env("OUT_DIR", &self.config.out_dir);
        cmd.env("CARGO_MANIFEST_DIR", &self.config.manifest_dir);
        cmd.env("CARGO_PKG_NAME", &self.config.pkg_name);
        cmd.env("CARGO_PKG_VERSION", &self.config.pkg_version);
        cmd.env("TARGET", &self.config.target);
        cmd.env("PROFILE", &self.config.profile);
        cmd.env("HOST", env::consts::ARCH);
        cmd.env(
            "OPT_LEVEL",
            if self.config.profile == "release" {
                "3"
            } else {
                "0"
            },
        );
        cmd.env(
            "DEBUG",
            if self.config.profile == "dev" {
                "true"
            } else {
                "false"
            },
        );

        // Set feature flags
        for feature in &self.config.features {
            let env_name = format!("CARGO_FEATURE_{}", feature.to_uppercase().replace('-', "_"));
            cmd.env(env_name, "1");
        }

        // Set custom environment
        for (key, value) in &self.config.env {
            cmd.env(key, value);
        }

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        cmd.current_dir(&self.config.manifest_dir);

        let output = cmd.output()?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let instructions = self.parse_instructions(&stdout);
        let generated_files = self.find_generated_files()?;

        Ok(BuildScriptOutput {
            instructions,
            generated_files,
            stdout,
            stderr,
            exit_code: output.status.code().unwrap_or(-1),
            duration: std::time::Duration::ZERO,
        })
    }

    /// Parse instructions from stdout
    fn parse_instructions(&self, stdout: &str) -> Vec<BuildInstruction> {
        stdout
            .lines()
            .filter_map(|line| self.parse_instruction(line))
            .collect()
    }

    /// Parse a single instruction line
    fn parse_instruction(&self, line: &str) -> Option<BuildInstruction> {
        let line = line.trim();

        if !line.starts_with("cargo:") {
            return None;
        }

        let rest = &line[6..];

        // rerun-if-changed
        if let Some(path) = rest.strip_prefix("rerun-if-changed=") {
            return Some(BuildInstruction::RerunIfChanged(PathBuf::from(path)));
        }

        // rerun-if-env-changed
        if let Some(var) = rest.strip_prefix("rerun-if-env-changed=") {
            return Some(BuildInstruction::RerunIfEnvChanged(var.to_string()));
        }

        // rustc-link-lib
        if let Some(lib) = rest.strip_prefix("rustc-link-lib=") {
            let (kind, name) = if let Some(r) = lib.strip_prefix("static=") {
                (LinkKind::Static, r)
            } else if let Some(r) = lib.strip_prefix("dylib=") {
                (LinkKind::Dylib, r)
            } else if let Some(r) = lib.strip_prefix("framework=") {
                (LinkKind::Framework, r)
            } else {
                (LinkKind::Dylib, lib)
            };
            return Some(BuildInstruction::LinkLib {
                kind,
                name: name.to_string(),
            });
        }

        // rustc-link-search
        if let Some(search) = rest.strip_prefix("rustc-link-search=") {
            let (kind, path) = if let Some(r) = search.strip_prefix("native=") {
                (SearchKind::Native, r)
            } else if let Some(r) = search.strip_prefix("framework=") {
                (SearchKind::Framework, r)
            } else if let Some(r) = search.strip_prefix("all=") {
                (SearchKind::All, r)
            } else {
                (SearchKind::All, search)
            };
            return Some(BuildInstruction::LinkSearch {
                kind,
                path: PathBuf::from(path),
            });
        }

        // rustc-cfg
        if let Some(cfg) = rest.strip_prefix("rustc-cfg=") {
            return Some(BuildInstruction::Cfg(cfg.to_string()));
        }

        // rustc-env
        if let Some(env) = rest.strip_prefix("rustc-env=")
            && let Some((key, value)) = env.split_once('=')
        {
            return Some(BuildInstruction::Env {
                key: key.to_string(),
                value: value.to_string(),
            });
        }

        // rustc-cdylib-link-arg and rustc-link-arg (simplified)
        if rest.starts_with("rustc-cdylib-link-arg=") || rest.starts_with("rustc-link-arg=") {
            // These are link arguments, not directly supported
            return None;
        }

        // include
        if let Some(path) = rest.strip_prefix("include=") {
            return Some(BuildInstruction::Include(PathBuf::from(path)));
        }

        // warning
        if let Some(warning) = rest.strip_prefix("warning=") {
            return Some(BuildInstruction::Warning(warning.to_string()));
        }

        // error
        if let Some(error) = rest.strip_prefix("error=") {
            return Some(BuildInstruction::Error(error.to_string()));
        }

        // metadata
        if rest.contains('=')
            && !rest.starts_with("rustc-")
            && let Some((key, value)) = rest.split_once('=')
        {
            return Some(BuildInstruction::Metadata {
                key: key.to_string(),
                value: value.to_string(),
            });
        }

        None
    }

    /// Find generated files in out_dir
    fn find_generated_files(&self) -> Result<Vec<PathBuf>, BuildScriptError> {
        let mut files = Vec::new();

        if self.config.out_dir.exists() {
            self.scan_dir(&self.config.out_dir, &mut files)?;
        }

        Ok(files)
    }

    /// Recursively scan directory for files
    fn scan_dir(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), BuildScriptError> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                files.push(path);
            } else if path.is_dir() {
                self.scan_dir(&path, files)?;
            }
        }

        Ok(())
    }
}

// =============================================================================
// Build Script API
// =============================================================================

/// Build script API for build.d files
pub mod build_api {
    use super::*;

    /// Get output directory
    pub fn out_dir() -> PathBuf {
        PathBuf::from(env::var("OUT_DIR").unwrap_or_else(|_| "target/build".into()))
    }

    /// Get manifest directory
    pub fn manifest_dir() -> PathBuf {
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into()))
    }

    /// Get package name
    pub fn pkg_name() -> String {
        env::var("CARGO_PKG_NAME").unwrap_or_default()
    }

    /// Get package version
    pub fn pkg_version() -> String {
        env::var("CARGO_PKG_VERSION").unwrap_or_default()
    }

    /// Get target triple
    pub fn target() -> String {
        env::var("TARGET").unwrap_or_else(|_| "native".into())
    }

    /// Get build profile
    pub fn profile() -> String {
        env::var("PROFILE").unwrap_or_else(|_| "debug".into())
    }

    /// Check if a feature is enabled
    pub fn feature_enabled(name: &str) -> bool {
        let env_name = format!("CARGO_FEATURE_{}", name.to_uppercase().replace('-', "_"));
        env::var(env_name).is_ok()
    }

    /// Get optimization level
    pub fn opt_level() -> u32 {
        env::var("OPT_LEVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0)
    }

    /// Check if debug mode
    pub fn debug() -> bool {
        env::var("DEBUG")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false)
    }

    /// Print rerun-if-changed instruction
    pub fn rerun_if_changed(path: &Path) {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    /// Print rerun-if-env-changed instruction
    pub fn rerun_if_env_changed(var: &str) {
        println!("cargo:rerun-if-env-changed={}", var);
    }

    /// Link a library
    pub fn link_lib(name: &str) {
        println!("cargo:rustc-link-lib={}", name);
    }

    /// Link a static library
    pub fn link_lib_static(name: &str) {
        println!("cargo:rustc-link-lib=static={}", name);
    }

    /// Link a framework (macOS)
    pub fn link_framework(name: &str) {
        println!("cargo:rustc-link-lib=framework={}", name);
    }

    /// Add library search path
    pub fn link_search(path: &Path) {
        println!("cargo:rustc-link-search={}", path.display());
    }

    /// Add native library search path
    pub fn link_search_native(path: &Path) {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    /// Define a cfg flag
    pub fn cfg(flag: &str) {
        println!("cargo:rustc-cfg={}", flag);
    }

    /// Set environment variable for compilation
    pub fn set_env(key: &str, value: &str) {
        println!("cargo:rustc-env={}={}", key, value);
    }

    /// Print a warning
    pub fn warning(msg: &str) {
        println!("cargo:warning={}", msg);
    }

    /// Print an error
    pub fn error(msg: &str) {
        println!("cargo:error={}", msg);
    }

    /// Set metadata for other packages
    pub fn metadata(key: &str, value: &str) {
        println!("cargo:{}={}", key, value);
    }

    /// Generate code to a file
    pub fn generate_code(filename: &str, code: &str) -> std::io::Result<PathBuf> {
        let path = out_dir().join(filename);
        fs::write(&path, code)?;
        Ok(path)
    }

    /// Read file content
    pub fn read_file(path: &Path) -> std::io::Result<String> {
        fs::read_to_string(path)
    }

    /// Check if file exists
    pub fn file_exists(path: &Path) -> bool {
        path.exists()
    }

    /// Get file modification time
    pub fn file_mtime(path: &Path) -> Option<SystemTime> {
        fs::metadata(path).and_then(|m| m.modified()).ok()
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Build script error
#[derive(Debug)]
pub enum BuildScriptError {
    /// IO error
    Io(std::io::Error),

    /// Compilation error
    Compilation(String),

    /// Runtime error
    Runtime(String),
}

impl From<std::io::Error> for BuildScriptError {
    fn from(e: std::io::Error) -> Self {
        BuildScriptError::Io(e)
    }
}

impl std::fmt::Display for BuildScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildScriptError::Io(e) => write!(f, "IO error: {}", e),
            BuildScriptError::Compilation(s) => write!(f, "Compilation error: {}", s),
            BuildScriptError::Runtime(s) => write!(f, "Runtime error: {}", s),
        }
    }
}

impl std::error::Error for BuildScriptError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_script_config_default() {
        let config = BuildScriptConfig::default();
        assert_eq!(config.script, PathBuf::from("build.sio"));
        assert_eq!(config.profile, "dev");
    }

    #[test]
    fn test_parse_instruction_rerun_if_changed() {
        let runner = BuildScriptRunner::new(BuildScriptConfig::default());
        let inst = runner.parse_instruction("cargo:rerun-if-changed=src/lib.sio");

        assert!(matches!(
            inst,
            Some(BuildInstruction::RerunIfChanged(p)) if p == PathBuf::from("src/lib.sio")
        ));
    }

    #[test]
    fn test_parse_instruction_link_lib() {
        let runner = BuildScriptRunner::new(BuildScriptConfig::default());

        let inst = runner.parse_instruction("cargo:rustc-link-lib=ssl");
        assert!(matches!(
            inst,
            Some(BuildInstruction::LinkLib { kind: LinkKind::Dylib, name }) if name == "ssl"
        ));

        let inst = runner.parse_instruction("cargo:rustc-link-lib=static=crypto");
        assert!(matches!(
            inst,
            Some(BuildInstruction::LinkLib { kind: LinkKind::Static, name }) if name == "crypto"
        ));
    }

    #[test]
    fn test_parse_instruction_cfg() {
        let runner = BuildScriptRunner::new(BuildScriptConfig::default());
        let inst = runner.parse_instruction("cargo:rustc-cfg=feature=\"test\"");

        assert!(matches!(
            inst,
            Some(BuildInstruction::Cfg(s)) if s == "feature=\"test\""
        ));
    }

    #[test]
    fn test_parse_instruction_warning() {
        let runner = BuildScriptRunner::new(BuildScriptConfig::default());
        let inst = runner.parse_instruction("cargo:warning=This is a warning");

        assert!(matches!(
            inst,
            Some(BuildInstruction::Warning(s)) if s == "This is a warning"
        ));
    }

    #[test]
    fn test_build_api_functions() {
        // These will use defaults when env vars aren't set
        let _out = build_api::out_dir();
        let _manifest = build_api::manifest_dir();
        assert!(!build_api::feature_enabled("nonexistent"));
    }
}
