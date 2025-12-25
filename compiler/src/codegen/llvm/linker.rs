//! Linker integration for creating executables
//!
//! This module provides functionality to link object files into executables
//! using the system linker (cc, clang, ld, etc.).

use std::path::{Path, PathBuf};
use std::process::Command;

/// Linker configuration
#[derive(Debug, Clone)]
pub struct Linker {
    /// Linker command (cc, clang, ld, etc.)
    command: String,

    /// Library search paths (-L)
    lib_paths: Vec<PathBuf>,

    /// Libraries to link (-l)
    libs: Vec<String>,

    /// Extra linker flags
    flags: Vec<String>,

    /// Target triple
    target: Option<String>,

    /// Generate position independent executable
    pie: bool,

    /// Strip debug symbols
    strip: bool,

    /// Verbose output
    verbose: bool,
}

impl Linker {
    /// Create a new linker with auto-detected command
    pub fn new() -> Self {
        Self {
            command: Self::detect_linker(),
            lib_paths: Vec::new(),
            libs: Vec::new(),
            flags: Vec::new(),
            target: None,
            pie: true,
            strip: false,
            verbose: false,
        }
    }

    /// Detect available linker
    fn detect_linker() -> String {
        // Try clang first, then cc, then gcc
        for cmd in &["clang", "cc", "gcc"] {
            if Command::new(cmd).arg("--version").output().is_ok() {
                return cmd.to_string();
            }
        }

        // Fall back to cc
        "cc".to_string()
    }

    /// Set the linker command
    pub fn command(mut self, cmd: impl Into<String>) -> Self {
        self.command = cmd.into();
        self
    }

    /// Add a library search path
    pub fn lib_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.lib_paths.push(path.into());
        self
    }

    /// Add multiple library search paths
    pub fn lib_paths(mut self, paths: impl IntoIterator<Item = PathBuf>) -> Self {
        self.lib_paths.extend(paths);
        self
    }

    /// Add a library to link
    pub fn lib(mut self, name: impl Into<String>) -> Self {
        self.libs.push(name.into());
        self
    }

    /// Add multiple libraries to link
    pub fn libs(mut self, libs: impl IntoIterator<Item = String>) -> Self {
        self.libs.extend(libs);
        self
    }

    /// Add extra linker flag
    pub fn flag(mut self, flag: impl Into<String>) -> Self {
        self.flags.push(flag.into());
        self
    }

    /// Add multiple linker flags
    pub fn flags(mut self, flags: impl IntoIterator<Item = String>) -> Self {
        self.flags.extend(flags);
        self
    }

    /// Set target triple
    pub fn target(mut self, triple: impl Into<String>) -> Self {
        self.target = Some(triple.into());
        self
    }

    /// Enable/disable PIE
    pub fn pie(mut self, enable: bool) -> Self {
        self.pie = enable;
        self
    }

    /// Enable/disable stripping
    pub fn strip(mut self, enable: bool) -> Self {
        self.strip = enable;
        self
    }

    /// Enable/disable verbose output
    pub fn verbose(mut self, enable: bool) -> Self {
        self.verbose = enable;
        self
    }

    /// Link object files into an executable
    pub fn link(&self, objects: &[PathBuf], output: &Path) -> Result<(), LinkError> {
        let mut cmd = Command::new(&self.command);

        // Add object files
        for obj in objects {
            if !obj.exists() {
                return Err(LinkError::ObjectNotFound(obj.clone()));
            }
            cmd.arg(obj);
        }

        // Output file
        cmd.arg("-o").arg(output);

        // Target
        if let Some(ref target) = self.target {
            cmd.arg("-target").arg(target);
        }

        // PIE
        if self.pie {
            cmd.arg("-pie");
        } else {
            cmd.arg("-no-pie");
        }

        // Strip
        if self.strip {
            cmd.arg("-s");
        }

        // Library search paths
        for path in &self.lib_paths {
            cmd.arg("-L").arg(path);
        }

        // Libraries
        for lib in &self.libs {
            cmd.arg(format!("-l{}", lib));
        }

        // Extra flags
        for flag in &self.flags {
            cmd.arg(flag);
        }

        if self.verbose {
            cmd.arg("-v");
            eprintln!("Running: {:?}", cmd);
        }

        // Run linker
        let output = cmd
            .output()
            .map_err(|e| LinkError::IoError(e.to_string()))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            Err(LinkError::LinkerFailed(format!("{}\n{}", stderr, stdout)))
        }
    }

    /// Link with standard system libraries
    pub fn link_with_stdlib(&self, objects: &[PathBuf], output: &Path) -> Result<(), LinkError> {
        let mut linker = self.clone();

        // Add standard libraries based on target
        let target = linker.target.as_deref().unwrap_or("");

        if target.contains("linux") {
            linker.libs.push("m".to_string()); // Math library
            linker.libs.push("pthread".to_string()); // Threading
            linker.libs.push("dl".to_string()); // Dynamic loading
            linker.libs.push("c".to_string()); // C library
        } else if target.contains("darwin") || target.contains("macos") {
            linker.libs.push("System".to_string());
        } else if target.contains("windows") {
            linker.libs.push("kernel32".to_string());
            linker.libs.push("user32".to_string());
            linker.libs.push("msvcrt".to_string());
        } else {
            // Generic Unix
            linker.libs.push("m".to_string());
            linker.libs.push("c".to_string());
        }

        linker.link(objects, output)
    }

    /// Link with D runtime library
    pub fn link_with_runtime(
        &self,
        objects: &[PathBuf],
        output: &Path,
        runtime_path: Option<&Path>,
    ) -> Result<(), LinkError> {
        let mut linker = self.clone();

        // Add runtime library path
        if let Some(rt_path) = runtime_path {
            linker.lib_paths.push(rt_path.to_path_buf());
            linker.libs.push("sounio_rt".to_string());
        }

        // Add standard libraries
        linker.link_with_stdlib(objects, output)
    }

    /// Create a shared library
    pub fn link_shared(&self, objects: &[PathBuf], output: &Path) -> Result<(), LinkError> {
        let mut cmd = Command::new(&self.command);

        // Shared library flag
        cmd.arg("-shared");

        // Add object files
        for obj in objects {
            if !obj.exists() {
                return Err(LinkError::ObjectNotFound(obj.clone()));
            }
            cmd.arg(obj);
        }

        // Output file
        cmd.arg("-o").arg(output);

        // Target
        if let Some(ref target) = self.target {
            cmd.arg("-target").arg(target);
        }

        // Library search paths
        for path in &self.lib_paths {
            cmd.arg("-L").arg(path);
        }

        // Libraries
        for lib in &self.libs {
            cmd.arg(format!("-l{}", lib));
        }

        // Extra flags
        for flag in &self.flags {
            cmd.arg(flag);
        }

        if self.verbose {
            cmd.arg("-v");
            eprintln!("Running: {:?}", cmd);
        }

        let output = cmd
            .output()
            .map_err(|e| LinkError::IoError(e.to_string()))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(LinkError::LinkerFailed(stderr.to_string()))
        }
    }

    /// Create a static library
    pub fn archive(&self, objects: &[PathBuf], output: &Path) -> Result<(), LinkError> {
        // Use ar for static libraries
        let mut cmd = Command::new("ar");
        cmd.arg("rcs").arg(output);

        for obj in objects {
            if !obj.exists() {
                return Err(LinkError::ObjectNotFound(obj.clone()));
            }
            cmd.arg(obj);
        }

        if self.verbose {
            eprintln!("Running: {:?}", cmd);
        }

        let output = cmd
            .output()
            .map_err(|e| LinkError::IoError(e.to_string()))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(LinkError::LinkerFailed(stderr.to_string()))
        }
    }

    /// Get the linker command
    pub fn get_command(&self) -> &str {
        &self.command
    }
}

impl Default for Linker {
    fn default() -> Self {
        Self::new()
    }
}

/// Linker errors
#[derive(Debug, Clone)]
pub enum LinkError {
    /// I/O error
    IoError(String),
    /// Linker failed
    LinkerFailed(String),
    /// Object file not found
    ObjectNotFound(PathBuf),
    /// Linker not found
    LinkerNotFound(String),
}

impl std::fmt::Display for LinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinkError::IoError(e) => write!(f, "I/O error: {}", e),
            LinkError::LinkerFailed(e) => write!(f, "Linker failed: {}", e),
            LinkError::ObjectNotFound(p) => write!(f, "Object file not found: {}", p.display()),
            LinkError::LinkerNotFound(c) => write!(f, "Linker not found: {}", c),
        }
    }
}

impl std::error::Error for LinkError {}

/// Quick link function for simple cases
pub fn link(objects: &[PathBuf], output: &Path) -> Result<(), LinkError> {
    Linker::new().link(objects, output)
}

/// Quick link with stdlib
pub fn link_with_stdlib(objects: &[PathBuf], output: &Path) -> Result<(), LinkError> {
    Linker::new().link_with_stdlib(objects, output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linker_creation() {
        let linker = Linker::new();
        assert!(!linker.get_command().is_empty());
    }

    #[test]
    fn test_linker_builder() {
        let linker = Linker::new()
            .command("clang")
            .lib_path("/usr/lib")
            .lib("m")
            .flag("-O2")
            .target("x86_64-unknown-linux-gnu")
            .pie(true)
            .strip(false);

        assert_eq!(linker.get_command(), "clang");
    }

    #[test]
    fn test_link_error_display() {
        let err = LinkError::ObjectNotFound(PathBuf::from("/tmp/test.o"));
        assert!(err.to_string().contains("test.o"));
    }
}
