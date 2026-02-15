//! # Poseidon VM - Safe Rust Wrapper
//!
//! This crate provides a safe, idiomatic Rust wrapper around the Poseidon C VM,
//! which executes SOIR (Sounio Intermediate Representation) bytecode.
//!
//! ## Features
//!
//! - Safe Rust API with zero exposed `unsafe` in public interface
//! - Automatic memory management (no leaks)
//! - Comprehensive error handling
//! - Integration with SOIR library
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use poseidon_vm::PoseidonVm;
//! use soir::serialize;
//!
//! // Create VM instance
//! let mut vm = PoseidonVm::new()?;
//!
//! // Load SOIR bytecode
//! let bytecode = serialize(&ir_module)?;
//! vm.load(&bytecode)?;
//!
//! // Execute program
//! let exit_code = vm.execute()?;
//! println!("Program exited with code: {}", exit_code);
//! ```
//!
//! ## Safety
//!
//! All FFI calls to the C VM are wrapped in safe Rust abstractions.
//! Memory safety is ensured through:
//! - Proper lifetime management of C resources
//! - Bounds checking on all register accesses
//! - Validated bytecode loading
//! - No use-after-free (Module is dropped before VMState)

pub mod error;
pub mod ffi;

pub use error::{Result, VmError};

use std::ptr;

/// Default maximum execution steps (prevents infinite loops)
pub const DEFAULT_MAX_STEPS: i64 = 1_000_000;

/// Maximum number of registers available in the VM
pub const MAX_REGISTERS: usize = ffi::MAX_REGISTERS;

/// Maximum call stack depth
pub const MAX_CALL_DEPTH: usize = ffi::MAX_CALL_DEPTH;

/// Poseidon VM instance
///
/// This structure wraps the C VM state and loaded module.
/// It ensures proper cleanup and memory safety.
pub struct PoseidonVm {
    /// VM execution state
    vm_state: Box<ffi::VMState>,
    /// Loaded module (null if not loaded)
    module: *mut ffi::Module,
    /// Maximum execution steps
    max_steps: i64,
}

impl PoseidonVm {
    /// Create a new VM instance
    ///
    /// The VM starts uninitialized. You must call `load()` before `execute()`.
    pub fn new() -> Result<Self> {
        Ok(Self {
            vm_state: Box::new(unsafe { std::mem::zeroed() }),
            module: ptr::null_mut(),
            max_steps: DEFAULT_MAX_STEPS,
        })
    }

    /// Create a new VM with custom max steps limit
    pub fn with_max_steps(max_steps: i64) -> Result<Self> {
        let mut vm = Self::new()?;
        vm.max_steps = max_steps;
        Ok(vm)
    }

    /// Load SOIR bytecode from bytes
    ///
    /// This deserializes the bytecode and prepares the VM for execution.
    /// If a module is already loaded, it will be freed first.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Bytecode is invalid or corrupted
    /// - No main function is found
    /// - Module allocation fails
    pub fn load(&mut self, bytecode: &[u8]) -> Result<()> {
        // Free existing module if any
        self.free_module();

        // Load module from bytecode
        let module = unsafe { ffi::load_soir_buffer(bytecode.as_ptr(), bytecode.len()) };

        if module.is_null() {
            return Err(VmError::InvalidBytecode(
                "Failed to load SOIR bytecode".to_string(),
            ));
        }

        // Find main function
        let main_fn = unsafe { ffi::vm_find_main_fn(module) };
        if main_fn < 0 {
            unsafe {
                ffi::free_module(module);
            }
            return Err(VmError::NoMainFunction);
        }

        // Initialize VM state with main function
        unsafe {
            ffi::vm_init(self.vm_state.as_mut(), main_fn);
        }

        self.module = module;
        Ok(())
    }

    /// Execute the loaded program
    ///
    /// Runs the VM until:
    /// - The program returns (normal exit)
    /// - Max steps is reached (timeout)
    /// - An error occurs
    ///
    /// # Returns
    ///
    /// The exit code from the program (return value of main function)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No module is loaded
    /// - VM is already halted
    /// - Execution times out
    /// - An execution error occurs
    pub fn execute(&mut self) -> Result<i64> {
        if self.module.is_null() {
            return Err(VmError::InvalidBytecode("No module loaded".to_string()));
        }

        if self.vm_state.halted {
            return Err(VmError::AlreadyHalted(self.vm_state.exit_code));
        }

        // Safety: module is non-null and valid
        let result = unsafe { ffi::vm_run(self.vm_state.as_mut(), self.module, self.max_steps) };

        match result {
            -2 => Err(VmError::Timeout(self.max_steps)),
            code => Ok(code),
        }
    }

    /// Execute a single instruction step
    ///
    /// This is useful for debugging or implementing custom execution logic.
    ///
    /// # Returns
    ///
    /// `Ok(true)` if execution continues, `Ok(false)` if halted
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No module is loaded
    /// - An execution error occurs
    pub fn step(&mut self) -> Result<bool> {
        if self.module.is_null() {
            return Err(VmError::InvalidBytecode("No module loaded".to_string()));
        }

        if self.vm_state.halted {
            return Ok(false);
        }

        // Safety: module is non-null and valid
        let result = unsafe { ffi::vm_step(self.vm_state.as_mut(), self.module) };

        if result < 0 {
            return Err(VmError::ExecutionError(format!(
                "Step failed with code {}",
                result
            )));
        }

        Ok(!self.vm_state.halted)
    }

    /// Get the value of a register
    ///
    /// This is useful for debugging and introspection.
    ///
    /// # Errors
    ///
    /// Returns an error if the register index is out of bounds.
    pub fn register(&self, idx: usize) -> Result<i64> {
        if idx >= MAX_REGISTERS {
            return Err(VmError::InvalidRegister(idx, MAX_REGISTERS));
        }
        Ok(self.vm_state.regs[idx])
    }

    /// Get current call stack depth
    pub fn stack_depth(&self) -> usize {
        self.vm_state.call_depth as usize
    }

    /// Get current program counter
    pub fn program_counter(&self) -> i64 {
        self.vm_state.pc
    }

    /// Get current function ID
    pub fn current_function(&self) -> i64 {
        self.vm_state.current_fn
    }

    /// Check if VM is halted
    pub fn is_halted(&self) -> bool {
        self.vm_state.halted
    }

    /// Get exit code (only valid if halted)
    pub fn exit_code(&self) -> Option<i64> {
        if self.vm_state.halted {
            Some(self.vm_state.exit_code)
        } else {
            None
        }
    }

    /// Reset VM state
    ///
    /// This resets the VM to initial state but keeps the loaded module.
    /// You can call `execute()` again after reset.
    ///
    /// # Errors
    ///
    /// Returns an error if no module is loaded.
    pub fn reset(&mut self) -> Result<()> {
        if self.module.is_null() {
            return Err(VmError::InvalidBytecode("No module loaded".to_string()));
        }

        // Find main function again
        let main_fn = unsafe { ffi::vm_find_main_fn(self.module) };
        if main_fn < 0 {
            return Err(VmError::NoMainFunction);
        }

        // Reinitialize VM state
        unsafe {
            ffi::vm_init(self.vm_state.as_mut(), main_fn);
        }

        Ok(())
    }

    /// Set maximum execution steps
    pub fn set_max_steps(&mut self, max_steps: i64) {
        self.max_steps = max_steps;
    }

    /// Get maximum execution steps
    pub fn max_steps(&self) -> i64 {
        self.max_steps
    }

    /// Free the loaded module
    fn free_module(&mut self) {
        if !self.module.is_null() {
            unsafe {
                ffi::free_module(self.module);
            }
            self.module = ptr::null_mut();
        }
    }
}

impl Drop for PoseidonVm {
    fn drop(&mut self) {
        self.free_module();
    }
}

impl Default for PoseidonVm {
    fn default() -> Self {
        Self::new().expect("Failed to create default VM")
    }
}

// Ensure VM is Send and Sync (C VM has no thread-local state)
unsafe impl Send for PoseidonVm {}
// Note: Not Sync because execution mutates state

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_creation() {
        let vm = PoseidonVm::new();
        assert!(vm.is_ok());
    }

    #[test]
    fn test_vm_with_max_steps() {
        let vm = PoseidonVm::with_max_steps(5000).unwrap();
        assert_eq!(vm.max_steps(), 5000);
    }

    #[test]
    fn test_register_bounds_checking() {
        let vm = PoseidonVm::new().unwrap();
        assert!(vm.register(MAX_REGISTERS).is_err());
        assert!(vm.register(MAX_REGISTERS - 1).is_ok());
    }

    #[test]
    fn test_execute_without_load() {
        let mut vm = PoseidonVm::new().unwrap();
        let result = vm.execute();
        assert!(result.is_err());
    }

    #[test]
    fn test_vm_initial_state() {
        let vm = PoseidonVm::new().unwrap();
        assert_eq!(vm.stack_depth(), 0);
        assert!(!vm.is_halted());
        assert_eq!(vm.exit_code(), None);
    }
}
