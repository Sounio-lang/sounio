//! Sysroot Management System
//!
//! This module handles sysroot discovery, building, and caching for cross-compilation.
//! A sysroot contains the target-specific libraries, headers, and runtime files needed
//! to compile and link programs for a target platform.

use super::spec::{OperatingSystem, TargetSpec, TargetSpecError, TargetTriple};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};
use thiserror::Error;

/// Errors that can occur during sysroot operations.
#[derive(Debug, Error)]
pub enum SysrootError {
    #[error("Sysroot not found for target: {0}")]
    NotFound(String),

    #[error("Failed to build sysroot: {0}")]
    BuildFailed(String),

    #[error("Sysroot is corrupted or invalid: {0}")]
    Invalid(String),

    #[error("Missing component in sysroot: {0}")]
    MissingComponent(String),

    #[error("Sysroot cache is stale")]
    CacheStale,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Target spec error: {0}")]
    TargetError(#[from] TargetSpecError),

    #[error("Command execution failed: {0}")]
    CommandFailed(String),
}

/// Result type for sysroot operations.
pub type SysrootResult<T> = Result<T, SysrootError>;

/// Sysroot component types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SysrootComponent {
    /// C runtime (libc, crt*.o)
    CRuntime,
    /// C++ runtime (libstdc++, libc++)
    CxxRuntime,
    /// Compiler runtime (compiler-rt, libgcc)
    CompilerRuntime,
    /// Standard library for the D language
    DStdlib,
    /// System headers
    Headers,
    /// Linker scripts
    LinkerScripts,
    /// Debug symbols
    DebugSymbols,
}

impl SysrootComponent {
    /// Get the directory name for this component.
    pub fn dir_name(&self) -> &str {
        match self {
            Self::CRuntime => "lib",
            Self::CxxRuntime => "lib",
            Self::CompilerRuntime => "lib",
            Self::DStdlib => "lib/d",
            Self::Headers => "include",
            Self::LinkerScripts => "lib/ldscripts",
            Self::DebugSymbols => "lib/debug",
        }
    }

    /// Get human-readable name.
    pub fn display_name(&self) -> &str {
        match self {
            Self::CRuntime => "C runtime",
            Self::CxxRuntime => "C++ runtime",
            Self::CompilerRuntime => "Compiler runtime",
            Self::DStdlib => "D standard library",
            Self::Headers => "System headers",
            Self::LinkerScripts => "Linker scripts",
            Self::DebugSymbols => "Debug symbols",
        }
    }
}

/// Sysroot source type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SysrootSource {
    /// System-provided sysroot
    System(PathBuf),
    /// Downloaded from a URL
    Downloaded {
        url: String,
        checksum: Option<String>,
    },
    /// Built from source
    Built { source_dir: PathBuf },
    /// Custom user-provided path
    Custom(PathBuf),
    /// Bundled with the compiler
    Bundled,
}

/// Sysroot metadata and configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SysrootMetadata {
    /// Target triple
    pub target: String,
    /// Sysroot version
    pub version: String,
    /// Creation timestamp
    pub created: u64,
    /// Last verified timestamp
    pub last_verified: u64,
    /// Source of the sysroot
    pub source: SysrootSource,
    /// Available components
    pub components: Vec<SysrootComponent>,
    /// Checksum of the sysroot contents
    pub checksum: Option<String>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

impl SysrootMetadata {
    /// Create new metadata.
    pub fn new(target: &str, source: SysrootSource) -> Self {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            target: target.to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            created: now,
            last_verified: now,
            source,
            components: Vec::new(),
            checksum: None,
            extra: HashMap::new(),
        }
    }

    /// Load metadata from a JSON file.
    pub fn load(path: &Path) -> SysrootResult<Self> {
        let content = fs::read_to_string(path)?;
        let metadata: Self = serde_json::from_str(&content)
            .map_err(|e| SysrootError::Invalid(format!("Failed to parse metadata: {}", e)))?;
        Ok(metadata)
    }

    /// Save metadata to a JSON file.
    pub fn save(&self, path: &Path) -> SysrootResult<()> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| SysrootError::Invalid(format!("Failed to serialize metadata: {}", e)))?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Check if the sysroot is stale.
    pub fn is_stale(&self, max_age: Duration) -> bool {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        now.saturating_sub(self.last_verified) > max_age.as_secs()
    }
}

/// A discovered or constructed sysroot.
#[derive(Debug, Clone)]
pub struct Sysroot {
    /// Root path of the sysroot
    pub path: PathBuf,
    /// Target this sysroot is for
    pub target: TargetTriple,
    /// Sysroot metadata
    pub metadata: Option<SysrootMetadata>,
    /// Discovered library paths
    pub lib_paths: Vec<PathBuf>,
    /// Discovered include paths
    pub include_paths: Vec<PathBuf>,
}

impl Sysroot {
    /// Create a new sysroot at the given path.
    pub fn new(path: PathBuf, target: TargetTriple) -> Self {
        Self {
            path,
            target,
            metadata: None,
            lib_paths: Vec::new(),
            include_paths: Vec::new(),
        }
    }

    /// Open an existing sysroot.
    pub fn open(path: &Path, target: &TargetTriple) -> SysrootResult<Self> {
        if !path.exists() {
            return Err(SysrootError::NotFound(path.display().to_string()));
        }

        let mut sysroot = Self::new(path.to_path_buf(), target.clone());

        // Try to load metadata
        let metadata_path = path.join("sysroot.json");
        if metadata_path.exists() {
            sysroot.metadata = Some(SysrootMetadata::load(&metadata_path)?);
        }

        // Discover paths
        sysroot.discover_paths()?;

        Ok(sysroot)
    }

    /// Discover library and include paths in the sysroot.
    fn discover_paths(&mut self) -> SysrootResult<()> {
        // Common library path patterns
        let lib_patterns = [
            "lib",
            "lib64",
            &format!("lib/{}", self.target),
            &format!("{}/lib", self.target),
            "usr/lib",
            "usr/lib64",
        ];

        for pattern in &lib_patterns {
            let path = self.path.join(pattern);
            if path.exists() && path.is_dir() {
                self.lib_paths.push(path);
            }
        }

        // Common include path patterns
        let include_patterns = [
            "include",
            &format!("include/{}", self.target),
            &format!("{}/include", self.target),
            "usr/include",
        ];

        for pattern in &include_patterns {
            let path = self.path.join(pattern);
            if path.exists() && path.is_dir() {
                self.include_paths.push(path);
            }
        }

        Ok(())
    }

    /// Get the library directory.
    pub fn lib_dir(&self) -> PathBuf {
        self.lib_paths
            .first()
            .cloned()
            .unwrap_or_else(|| self.path.join("lib"))
    }

    /// Get the include directory.
    pub fn include_dir(&self) -> PathBuf {
        self.include_paths
            .first()
            .cloned()
            .unwrap_or_else(|| self.path.join("include"))
    }

    /// Check if a component is available.
    pub fn has_component(&self, component: SysrootComponent) -> bool {
        if let Some(ref metadata) = self.metadata {
            return metadata.components.contains(&component);
        }

        // Probe for the component
        let dir = self.path.join(component.dir_name());
        dir.exists() && dir.is_dir()
    }

    /// Verify the sysroot integrity.
    pub fn verify(&self) -> SysrootResult<()> {
        if !self.path.exists() {
            return Err(SysrootError::NotFound(self.path.display().to_string()));
        }

        // Check for at least one library path
        if self.lib_paths.is_empty() {
            return Err(SysrootError::Invalid(
                "No library paths found in sysroot".to_string(),
            ));
        }

        // Verify metadata checksum if available
        if let Some(ref metadata) = self.metadata
            && let Some(ref _checksum) = metadata.checksum
        {
            // TODO: Implement checksum verification
        }

        Ok(())
    }

    /// Get linker library search paths as arguments.
    pub fn linker_lib_args(&self) -> Vec<String> {
        self.lib_paths
            .iter()
            .map(|p| format!("-L{}", p.display()))
            .collect()
    }

    /// Get compiler include paths as arguments.
    pub fn include_args(&self) -> Vec<String> {
        self.include_paths
            .iter()
            .map(|p| format!("-I{}", p.display()))
            .collect()
    }
}

/// Sysroot manager for discovering, building, and caching sysroots.
#[derive(Debug)]
pub struct SysrootManager {
    /// Cache directory for sysroots
    cache_dir: PathBuf,
    /// System sysroot search paths
    search_paths: Vec<PathBuf>,
    /// Loaded sysroots
    sysroots: HashMap<String, Sysroot>,
    /// Configuration
    config: SysrootConfig,
}

/// Sysroot manager configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SysrootConfig {
    /// Maximum age before sysroot is considered stale
    pub max_age: Duration,
    /// Whether to auto-download missing sysroots
    pub auto_download: bool,
    /// Whether to auto-build missing sysroots
    pub auto_build: bool,
    /// Custom download URLs for targets
    pub download_urls: HashMap<String, String>,
}

impl Default for SysrootConfig {
    fn default() -> Self {
        Self {
            max_age: Duration::from_secs(86400 * 30), // 30 days
            auto_download: false,
            auto_build: false,
            download_urls: HashMap::new(),
        }
    }
}

impl SysrootManager {
    /// Create a new sysroot manager.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self {
            cache_dir,
            search_paths: Vec::new(),
            sysroots: HashMap::new(),
            config: SysrootConfig::default(),
        }
    }

    /// Create with configuration.
    pub fn with_config(cache_dir: PathBuf, config: SysrootConfig) -> Self {
        Self {
            cache_dir,
            search_paths: Vec::new(),
            sysroots: HashMap::new(),
            config,
        }
    }

    /// Add a search path for system sysroots.
    pub fn add_search_path(&mut self, path: PathBuf) {
        if !self.search_paths.contains(&path) {
            self.search_paths.push(path);
        }
    }

    /// Get or create a sysroot for a target.
    pub fn get_sysroot(&mut self, spec: &TargetSpec) -> SysrootResult<&Sysroot> {
        let key = spec.triple.to_string();

        if !self.sysroots.contains_key(&key) {
            let sysroot = self.find_or_create_sysroot(spec)?;
            self.sysroots.insert(key.clone(), sysroot);
        }

        Ok(self.sysroots.get(&key).unwrap())
    }

    /// Find or create a sysroot for a target.
    fn find_or_create_sysroot(&self, spec: &TargetSpec) -> SysrootResult<Sysroot> {
        // 1. Check if user specified a sysroot in the spec
        if let Some(ref path) = spec.os.sysroot {
            return Sysroot::open(path, &spec.triple);
        }

        // 2. Check the cache
        let cache_path = self.cache_dir.join(spec.triple.to_string());
        if cache_path.exists() {
            let sysroot = Sysroot::open(&cache_path, &spec.triple)?;
            if let Some(ref metadata) = sysroot.metadata
                && !metadata.is_stale(self.config.max_age)
            {
                return Ok(sysroot);
            }
        }

        // 3. Search system paths
        if let Some(sysroot) = self.find_system_sysroot(spec)? {
            return Ok(sysroot);
        }

        // 4. Try to create/download sysroot
        self.create_sysroot(spec)
    }

    /// Search for a system-provided sysroot.
    fn find_system_sysroot(&self, spec: &TargetSpec) -> SysrootResult<Option<Sysroot>> {
        let target_str = spec.triple.to_string();

        // Platform-specific search locations
        let mut search_paths = self.search_paths.clone();

        // Add common system paths
        if spec.os.os == OperatingSystem::Linux {
            search_paths.extend([
                PathBuf::from("/"),
                PathBuf::from(format!("/usr/{}", target_str)),
                PathBuf::from(format!("/opt/{}", target_str)),
            ]);
        } else if spec.os.os == OperatingSystem::MacOs {
            // Check for Xcode SDK
            if let Ok(output) = Command::new("xcrun")
                .args(["--sdk", "macosx", "--show-sdk-path"])
                .output()
                && output.status.success()
            {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                search_paths.push(PathBuf::from(path));
            }
        }

        for path in &search_paths {
            if path.exists()
                && let Ok(sysroot) = Sysroot::open(path, &spec.triple)
                && sysroot.verify().is_ok()
            {
                return Ok(Some(sysroot));
            }
        }

        Ok(None)
    }

    /// Create a new sysroot (download or build).
    fn create_sysroot(&self, spec: &TargetSpec) -> SysrootResult<Sysroot> {
        let cache_path = self.cache_dir.join(spec.triple.to_string());

        // Try downloading if enabled
        if self.config.auto_download
            && let Some(url) = self.config.download_urls.get(&spec.triple.to_string())
        {
            return self.download_sysroot(spec, url, &cache_path);
        }

        // Try building if enabled
        if self.config.auto_build {
            return self.build_sysroot(spec, &cache_path);
        }

        Err(SysrootError::NotFound(format!(
            "No sysroot found for target '{}'. Consider installing a cross-compilation \
             toolchain or specifying a sysroot path with --sysroot.",
            spec.triple
        )))
    }

    /// Download a sysroot from a URL.
    fn download_sysroot(
        &self,
        spec: &TargetSpec,
        url: &str,
        dest: &Path,
    ) -> SysrootResult<Sysroot> {
        // Create destination directory
        fs::create_dir_all(dest)?;

        // Download and extract (simplified - in practice would use reqwest or similar)
        let status = Command::new("curl")
            .args(["-L", "-o", "sysroot.tar.gz", url])
            .current_dir(dest)
            .status()
            .map_err(|e| SysrootError::CommandFailed(format!("curl failed: {}", e)))?;

        if !status.success() {
            return Err(SysrootError::CommandFailed(
                "Failed to download sysroot".to_string(),
            ));
        }

        // Extract
        let status = Command::new("tar")
            .args(["xzf", "sysroot.tar.gz"])
            .current_dir(dest)
            .status()
            .map_err(|e| SysrootError::CommandFailed(format!("tar failed: {}", e)))?;

        if !status.success() {
            return Err(SysrootError::CommandFailed(
                "Failed to extract sysroot".to_string(),
            ));
        }

        // Create metadata
        let metadata = SysrootMetadata::new(
            &spec.triple.to_string(),
            SysrootSource::Downloaded {
                url: url.to_string(),
                checksum: None,
            },
        );
        metadata.save(&dest.join("sysroot.json"))?;

        Sysroot::open(dest, &spec.triple)
    }

    /// Build a sysroot from source.
    fn build_sysroot(&self, spec: &TargetSpec, dest: &Path) -> SysrootResult<Sysroot> {
        // Create destination directory
        fs::create_dir_all(dest)?;

        // Build the D standard library for this target
        self.build_stdlib(spec, dest)?;

        // Create metadata
        let mut metadata = SysrootMetadata::new(
            &spec.triple.to_string(),
            SysrootSource::Built {
                source_dir: PathBuf::from("."),
            },
        );
        metadata.components.push(SysrootComponent::DStdlib);
        metadata.save(&dest.join("sysroot.json"))?;

        Sysroot::open(dest, &spec.triple)
    }

    /// Build the D standard library for a target.
    fn build_stdlib(&self, spec: &TargetSpec, dest: &Path) -> SysrootResult<()> {
        let lib_dir = dest.join("lib/d");
        fs::create_dir_all(&lib_dir)?;

        // TODO: Actually build the stdlib
        // For now, create a placeholder
        let placeholder = lib_dir.join("libstd.rlib");
        fs::write(&placeholder, b"placeholder")?;

        Ok(())
    }

    /// List all cached sysroots.
    pub fn list_cached(&self) -> SysrootResult<Vec<(String, SysrootMetadata)>> {
        let mut result = Vec::new();

        if !self.cache_dir.exists() {
            return Ok(result);
        }

        for entry in fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                let metadata_path = path.join("sysroot.json");
                if metadata_path.exists()
                    && let Ok(metadata) = SysrootMetadata::load(&metadata_path)
                {
                    let name = entry.file_name().to_string_lossy().to_string();
                    result.push((name, metadata));
                }
            }
        }

        Ok(result)
    }

    /// Remove a cached sysroot.
    pub fn remove_cached(&mut self, target: &str) -> SysrootResult<()> {
        let cache_path = self.cache_dir.join(target);

        if !cache_path.exists() {
            return Err(SysrootError::NotFound(target.to_string()));
        }

        fs::remove_dir_all(&cache_path)?;
        self.sysroots.remove(target);

        Ok(())
    }

    /// Clean all stale sysroots.
    pub fn clean_stale(&mut self) -> SysrootResult<usize> {
        let mut removed = 0;

        let cached = self.list_cached()?;
        for (name, metadata) in cached {
            if metadata.is_stale(self.config.max_age) {
                self.remove_cached(&name)?;
                removed += 1;
            }
        }

        Ok(removed)
    }
}

/// Builder for creating custom sysroots.
#[derive(Debug)]
pub struct SysrootBuilder {
    target: TargetTriple,
    path: PathBuf,
    components: Vec<SysrootComponent>,
    lib_paths: Vec<PathBuf>,
    include_paths: Vec<PathBuf>,
    source: SysrootSource,
}

impl SysrootBuilder {
    /// Create a new sysroot builder.
    pub fn new(target: TargetTriple, path: PathBuf) -> Self {
        Self {
            target,
            path,
            components: Vec::new(),
            lib_paths: Vec::new(),
            include_paths: Vec::new(),
            source: SysrootSource::Custom(PathBuf::new()),
        }
    }

    /// Add a component to the sysroot.
    pub fn with_component(mut self, component: SysrootComponent) -> Self {
        if !self.components.contains(&component) {
            self.components.push(component);
        }
        self
    }

    /// Add a library path.
    pub fn with_lib_path(mut self, path: PathBuf) -> Self {
        self.lib_paths.push(path);
        self
    }

    /// Add an include path.
    pub fn with_include_path(mut self, path: PathBuf) -> Self {
        self.include_paths.push(path);
        self
    }

    /// Set the source type.
    pub fn with_source(mut self, source: SysrootSource) -> Self {
        self.source = source;
        self
    }

    /// Build the sysroot.
    pub fn build(self) -> SysrootResult<Sysroot> {
        // Create directory structure
        fs::create_dir_all(&self.path)?;

        for component in &self.components {
            let dir = self.path.join(component.dir_name());
            fs::create_dir_all(&dir)?;
        }

        // Create metadata
        let mut metadata = SysrootMetadata::new(&self.target.to_string(), self.source);
        metadata.components = self.components;
        metadata.save(&self.path.join("sysroot.json"))?;

        // Create sysroot
        let mut sysroot = Sysroot::new(self.path.clone(), self.target);
        sysroot.metadata = Some(metadata);
        sysroot.lib_paths = self.lib_paths;
        sysroot.include_paths = self.include_paths;

        // Discover additional paths
        sysroot.discover_paths()?;

        Ok(sysroot)
    }
}

/// Utilities for working with cross-compilation toolchains.
pub mod toolchain {
    use super::*;

    /// Find a cross-compiler for a target.
    pub fn find_cross_compiler(target: &TargetTriple) -> Option<PathBuf> {
        let target_str = target.to_string();

        // Try common cross-compiler naming patterns
        let patterns = [
            format!("{}-gcc", target_str),
            format!("{}-cc", target_str),
            format!("{}-clang", target_str),
        ];

        for pattern in &patterns {
            if let Ok(output) = Command::new("which").arg(pattern).output()
                && output.status.success()
            {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return Some(PathBuf::from(path));
            }
        }

        None
    }

    /// Find the cross-linker for a target.
    pub fn find_cross_linker(target: &TargetTriple) -> Option<PathBuf> {
        let target_str = target.to_string();

        let patterns = [
            format!("{}-ld", target_str),
            format!("{}-ld.lld", target_str),
            format!("{}-ld.gold", target_str),
        ];

        for pattern in &patterns {
            if let Ok(output) = Command::new("which").arg(pattern).output()
                && output.status.success()
            {
                let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
                return Some(PathBuf::from(path));
            }
        }

        None
    }

    /// Get the sysroot from a cross-compiler.
    pub fn get_compiler_sysroot(compiler: &Path) -> Option<PathBuf> {
        if let Ok(output) = Command::new(compiler).arg("-print-sysroot").output()
            && output.status.success()
        {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
        None
    }

    /// Get library search paths from a cross-compiler.
    pub fn get_compiler_lib_paths(compiler: &Path) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        if let Ok(output) = Command::new(compiler).args(["-print-search-dirs"]).output()
            && output.status.success()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            for line in stdout.lines() {
                if line.starts_with("libraries:") {
                    let dirs = line.trim_start_matches("libraries:");
                    let dirs = dirs.trim_start_matches(" =");
                    for dir in dirs.split(':') {
                        let path = PathBuf::from(dir.trim());
                        if path.exists() {
                            paths.push(path);
                        }
                    }
                }
            }
        }

        paths
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn temp_dir() -> PathBuf {
        env::temp_dir().join(format!("sounio_sysroot_test_{}", std::process::id()))
    }

    #[test]
    fn test_sysroot_component() {
        assert_eq!(SysrootComponent::CRuntime.dir_name(), "lib");
        assert_eq!(SysrootComponent::Headers.dir_name(), "include");
        assert_eq!(SysrootComponent::DStdlib.dir_name(), "lib/d");
    }

    #[test]
    fn test_sysroot_metadata() {
        let metadata = SysrootMetadata::new(
            "x86_64-unknown-linux-gnu",
            SysrootSource::System(PathBuf::from("/")),
        );

        assert_eq!(metadata.target, "x86_64-unknown-linux-gnu");
        assert!(metadata.created > 0);
        assert!(!metadata.is_stale(Duration::from_secs(3600)));
    }

    #[test]
    fn test_sysroot_metadata_serialization() {
        let temp = temp_dir();
        // Clean up any previous run first
        let _ = fs::remove_dir_all(&temp);
        fs::create_dir_all(&temp).unwrap();

        let metadata = SysrootMetadata::new(
            "aarch64-unknown-linux-gnu",
            SysrootSource::Downloaded {
                url: "https://example.com/sysroot.tar.gz".to_string(),
                checksum: Some("abc123".to_string()),
            },
        );

        let path = temp.join("metadata.json");
        metadata.save(&path).unwrap();

        let loaded = SysrootMetadata::load(&path).unwrap();
        assert_eq!(loaded.target, metadata.target);

        fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_sysroot_builder() {
        let temp = temp_dir();
        let target = TargetTriple::parse("x86_64-unknown-linux-gnu").unwrap();

        let sysroot = SysrootBuilder::new(target.clone(), temp.clone())
            .with_component(SysrootComponent::CRuntime)
            .with_component(SysrootComponent::Headers)
            .with_source(SysrootSource::Custom(temp.clone()))
            .build()
            .unwrap();

        assert!(sysroot.path.exists());
        assert!(sysroot.has_component(SysrootComponent::CRuntime));
        assert!(sysroot.has_component(SysrootComponent::Headers));

        fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_sysroot_manager() {
        let temp = temp_dir();
        let mut manager = SysrootManager::new(temp.clone());

        manager.add_search_path(PathBuf::from("/usr"));

        let cached = manager.list_cached().unwrap();
        assert!(cached.is_empty());

        fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn test_sysroot_linker_args() {
        let target = TargetTriple::parse("x86_64-unknown-linux-gnu").unwrap();
        let mut sysroot = Sysroot::new(PathBuf::from("/test"), target);
        sysroot.lib_paths = vec![PathBuf::from("/test/lib"), PathBuf::from("/test/lib64")];

        let args = sysroot.linker_lib_args();
        assert_eq!(args.len(), 2);
        assert!(args.contains(&"-L/test/lib".to_string()));
        assert!(args.contains(&"-L/test/lib64".to_string()));
    }

    #[test]
    fn test_sysroot_include_args() {
        let target = TargetTriple::parse("x86_64-unknown-linux-gnu").unwrap();
        let mut sysroot = Sysroot::new(PathBuf::from("/test"), target);
        sysroot.include_paths = vec![PathBuf::from("/test/include")];

        let args = sysroot.include_args();
        assert_eq!(args.len(), 1);
        assert_eq!(args[0], "-I/test/include");
    }
}
