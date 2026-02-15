//! # Self-Hosted Native Backend
//!
//! Wrapper around the Sounio self-hosted native code generator that produces
//! ELF64 (Linux) and Mach-O (macOS) binaries.
//!
//! This backend invokes the self-hosted compiler via the Poseidon VM interpreter,
//! extracting the generated binary bytes and writing them to disk.

use std::fs;
use std::path::Path;
use std::process::Command;

/// Self-hosted native backend wrapper
pub struct SelfhostedNativeBackend {
    // Placeholder: In Phase 6B+, this will hold the interpreter state
    #[allow(dead_code)]
    initialized: bool,
}

impl SelfhostedNativeBackend {
    /// Create a new self-hosted native backend
    pub fn new() -> Result<Self, String> {
        // Phase 6B TODO: Load self-hosted/native/codegen.sio + dependencies
        // Initialize Poseidon VM interpreter
        // Load compile_to_elf() function
        Ok(Self { initialized: true })
    }

    /// Compile IR module to ELF binary bytes
    ///
    /// # Arguments
    /// * `ir_bytes` - Serialized IR module
    /// * `base_addr` - ELF load address (typically 0x400000 for executables)
    ///
    /// # Returns
    /// Raw ELF64 binary bytes ready to write to disk
    pub fn compile_ir_to_elf(
        &mut self,
        _ir_bytes: &[u8],
        _base_addr: i64,
    ) -> Result<Vec<u8>, String> {
        // Phase 6B TODO:
        // 1. Convert Rust IR to Sounio representation
        // 2. Call interpreter.call_function("compile_to_elf", [ir, base_addr])
        // 3. Extract Elf64Binary.bytes[0..len]
        // 4. Return as Vec<u8>
        Err("SelfhostedNativeBackend not yet implemented in Phase 6B".to_string())
    }

    /// Write ELF binary to disk and make it executable
    pub fn write_elf(&self, elf_bytes: &[u8], output_path: impl AsRef<Path>) -> Result<(), String> {
        let path = output_path.as_ref();

        fs::write(path, elf_bytes)
            .map_err(|e| format!("Failed to write ELF to {}: {}", path.display(), e))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o755);
            fs::set_permissions(path, perms)
                .map_err(|e| format!("Failed to set executable permissions: {}", e))?;
        }

        Ok(())
    }

    /// Test that the generated ELF can be executed
    pub fn test_elf_execution(&self, output_path: impl AsRef<Path>) -> Result<i32, String> {
        let path = output_path.as_ref();

        let output = Command::new(path)
            .output()
            .map_err(|e| format!("Failed to execute ELF: {}", e))?;

        output.status.code().ok_or_else(|| {
            format!(
                "ELF execution terminated by signal. stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            )
        })
    }
}

impl Default for SelfhostedNativeBackend {
    fn default() -> Self {
        Self { initialized: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let result = SelfhostedNativeBackend::new();
        assert!(result.is_ok(), "Backend should initialize without error");
    }

    #[test]
    fn test_backend_default() {
        let backend = SelfhostedNativeBackend::default();
        assert!(!backend.initialized, "Default backend should be uninitialized");
    }
}
