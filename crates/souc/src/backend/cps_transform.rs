//! CPS (Continuation-Passing Style) Transformation for Native Backend
//!
//! This module implements CPS transformation for effect handlers in compiled code.
//! It transforms normal control flow into continuation-passing style, enabling
//! effect handlers to capture and resume continuations.
//!
//! # Theory Background
//!
//! Based on:
//! - Plotkin & Pretnar (2009) "Handlers of Algebraic Effects"
//! - Leijen (2017) "Type Directed Compilation of Row-typed Algebraic Effects"
//! - Hillerström et al. (2020) "Effekt: Capability-Passing Style for Effect Handlers"
//!
//! # Transformation Strategy
//!
//! We use **selective CPS**: only code that uses effects is transformed to CPS.
//! Pure code remains in direct style for better performance.
//!
//! ## Direct Style (before):
//! ```text
//! fn example() with IO {
//!     let x = compute(5)
//!     println(x)
//!     let y = compute(10)
//!     y
//! }
//! ```
//!
//! ## CPS (after):
//! ```text
//! fn example_cps(k: Continuation) {
//!     let x = compute(5)
//!     perform_effect("IO", "println", [x], |_| {
//!         let y = compute(10)
//!         resume(k, y)
//!     })
//! }
//! ```
//!
//! # Native Code Generation
//!
//! For native backends (AArch64/x86-64), we need to:
//! 1. Save machine state (registers, stack pointer, frame pointer)
//! 2. Store the saved state in a continuation object
//! 3. Call the effect handler with the continuation
//! 4. Handler can resume by restoring state and jumping to return address
//!
//! # Stack Capture
//!
//! We use **shallow stack capture**: only capture the current frame.
//! Deep continuations (full stack capture) are not needed for most effects.
//!
//! Exception: Multi-shot continuations may need deep capture for backtracking.

use crate::effects::continuation::{ContinuationId, ResumePoint};
use crate::hlir::ir::{Op, Program, VReg, Value as HlirValue};
use std::collections::{HashMap, HashSet};

/// Result type for CPS transformation
pub type CpsResult<T> = Result<T, CpsError>;

/// Errors that can occur during CPS transformation
#[derive(Debug, Clone, thiserror::Error)]
pub enum CpsError {
    #[error("Invalid effect operation: {0}")]
    InvalidEffect(String),

    #[error("CPS transformation failed: {0}")]
    TransformFailed(String),

    #[error("Unsupported operation for CPS: {0}")]
    UnsupportedOp(String),
}

/// CPS transformation context
///
/// Tracks which functions and blocks need CPS transformation based on
/// effect usage analysis.
pub struct CpsContext {
    /// Functions that need CPS transformation (use effects)
    effectful_functions: HashSet<String>,

    /// Map from original function to CPS-transformed version
    cps_functions: HashMap<String, String>,

    /// Continuation counter for generating unique IDs
    next_cont_id: usize,

    /// Whether to use selective or full CPS
    selective: bool,
}

impl CpsContext {
    /// Create a new CPS transformation context
    pub fn new() -> Self {
        Self {
            effectful_functions: HashSet::new(),
            cps_functions: HashMap::new(),
            next_cont_id: 0,
            selective: true, // Use selective CPS by default
        }
    }

    /// Enable full CPS transformation (transform all functions)
    pub fn with_full_cps(mut self) -> Self {
        self.selective = false;
        self
    }

    /// Analyze program to determine which functions need CPS
    pub fn analyze(&mut self, program: &Program) {
        for (name, func) in &program.functions {
            if self.uses_effects(func) {
                self.effectful_functions.insert(name.clone());
            }
        }
    }

    /// Check if a function uses effects
    fn uses_effects(&self, func: &crate::hlir::ir::Function) -> bool {
        // Scan all ops in the function for PerformEffect or DispatchEffect
        for block in &func.blocks {
            for stmt in &block.statements {
                if let Op::PerformEffect { .. } | Op::DispatchEffect { .. } = stmt.op {
                    return true;
                }
            }
            // Check terminator
            if let Some(term) = &block.terminator {
                if matches!(
                    term.op,
                    Op::PerformEffect { .. } | Op::DispatchEffect { .. }
                ) {
                    return true;
                }
            }
        }
        false
    }

    /// Generate a fresh continuation ID
    fn fresh_cont_id(&mut self) -> ContinuationId {
        let id = ContinuationId::new();
        self.next_cont_id += 1;
        id
    }

    /// Get the CPS-transformed name for a function
    pub fn cps_name(&self, original: &str) -> Option<&str> {
        self.cps_functions.get(original).map(String::as_str)
    }

    /// Check if a function needs CPS transformation
    pub fn needs_cps(&self, name: &str) -> bool {
        if self.selective {
            self.effectful_functions.contains(name)
        } else {
            true // Full CPS: transform everything
        }
    }
}

impl Default for CpsContext {
    fn default() -> Self {
        Self::new()
    }
}

/// CPS transformation pass
///
/// Transforms effectful code into continuation-passing style.
pub struct CpsTransform {
    ctx: CpsContext,
}

impl CpsTransform {
    /// Create a new CPS transformer
    pub fn new() -> Self {
        Self {
            ctx: CpsContext::new(),
        }
    }

    /// Transform a program into CPS
    pub fn transform(&mut self, program: Program) -> CpsResult<Program> {
        // Phase 1: Analyze which functions use effects
        self.ctx.analyze(&program);

        // Phase 2: Transform effectful functions to CPS
        let mut new_program = program.clone();

        for (name, func) in &program.functions {
            if self.ctx.needs_cps(name) {
                let cps_func = self.transform_function(func)?;
                let cps_name = format!("{}_cps", name);

                self.ctx
                    .cps_functions
                    .insert(name.clone(), cps_name.clone());

                new_program.functions.insert(cps_name, cps_func);
            }
        }

        Ok(new_program)
    }

    /// Transform a single function to CPS
    fn transform_function(
        &mut self,
        _func: &crate::hlir::ir::Function,
    ) -> CpsResult<crate::hlir::ir::Function> {
        // TODO: Implement actual CPS transformation
        //
        // Strategy:
        // 1. Add continuation parameter to function signature
        // 2. Transform each PerformEffect into:
        //    a. Capture continuation (save registers, stack, return address)
        //    b. Call effect handler with continuation
        //    c. Effect handler can resume by calling continuation
        // 3. Transform function calls in effectful context to pass continuation

        Err(CpsError::TransformFailed(
            "CPS transformation not yet implemented".to_string(),
        ))
    }
}

impl Default for CpsTransform {
    fn default() -> Self {
        Self::new()
    }
}

/// Native continuation capture state
///
/// Represents the machine state needed to resume a native continuation.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct NativeContinuation {
    /// Unique ID for this continuation
    pub id: ContinuationId,

    /// Return address (where to jump when resumed)
    pub return_address: usize,

    /// Saved general-purpose registers (x0-x30 for AArch64, rax-r15 for x86-64)
    pub gp_registers: [u64; 32],

    /// Saved floating-point registers (v0-v31 for AArch64, xmm0-xmm15 for x86-64)
    pub fp_registers: [f64; 32],

    /// Stack pointer at capture time
    pub stack_pointer: usize,

    /// Frame pointer at capture time
    pub frame_pointer: usize,

    /// Stack snapshot (captured stack frame)
    pub stack_data: Vec<u8>,

    /// Whether this is a one-shot or multi-shot continuation
    pub is_one_shot: bool,

    /// Resume count (for one-shot enforcement)
    pub resume_count: usize,
}

impl NativeContinuation {
    /// Create a new native continuation
    pub fn new(return_address: usize, is_one_shot: bool) -> Self {
        Self {
            id: ContinuationId::new(),
            return_address,
            gp_registers: [0; 32],
            fp_registers: [0.0; 32],
            stack_pointer: 0,
            frame_pointer: 0,
            stack_data: Vec::new(),
            is_one_shot,
            resume_count: 0,
        }
    }

    /// Capture current machine state
    ///
    /// SAFETY: This is highly architecture-specific and unsafe.
    /// Must be called from assembly stubs that save registers.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn capture() -> Self {
        // TODO: Implement actual register capture via inline assembly
        //
        // Strategy for AArch64:
        // 1. Save x0-x30 (general-purpose registers)
        // 2. Save v0-v31 (SIMD/FP registers)
        // 3. Save SP (stack pointer)
        // 4. Save FP (frame pointer / x29)
        // 5. Capture stack frame (SP to FP range)

        let mut cont = Self::new(0, true);

        // Placeholder - would use inline assembly here
        // Example (pseudo-code):
        // asm!("mov {}, x0", out(reg) cont.gp_registers[0]);
        // ... save x1-x30 ...
        // asm!("mov {}, sp", out(reg) cont.stack_pointer);

        cont
    }

    /// Resume this continuation with a value
    ///
    /// SAFETY: Extremely unsafe - overwrites current execution context
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn resume(&mut self, _value: u64) -> ! {
        // TODO: Implement actual continuation resumption
        //
        // Strategy:
        // 1. Check one-shot constraint
        // 2. Restore stack frame from stack_data
        // 3. Restore FP and SP
        // 4. Restore v0-v31 (SIMD/FP registers)
        // 5. Restore x1-x30 (keep x0 for return value)
        // 6. Jump to return_address

        panic!("Continuation resumption not yet implemented")
    }

    /// Convert to ResumePoint for effect handlers
    pub fn to_resume_point(self) -> ResumePoint {
        ResumePoint::jit(
            self.return_address,
            self.gp_registers.to_vec(),
            self.stack_data,
        )
    }
}

/// Generate CPS runtime support functions
///
/// These are assembly stubs that perform the actual register saving/restoring.
pub mod runtime {
    use super::*;

    /// Capture a continuation at the current point
    ///
    /// This function must be called from generated native code at effect perform sites.
    ///
    /// # Arguments
    /// * `return_address` - Where to resume when continuation is invoked
    ///
    /// # Returns
    /// Pointer to captured NativeContinuation
    #[no_mangle]
    pub extern "C" fn __sounio_capture_continuation(
        return_address: usize,
    ) -> *mut NativeContinuation {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            let cont = NativeContinuation::capture();
            Box::into_raw(Box::new(cont))
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = return_address;
            panic!("Continuation capture only implemented for AArch64")
        }
    }

    /// Resume a continuation
    ///
    /// # Arguments
    /// * `cont_ptr` - Pointer to NativeContinuation
    /// * `value` - Value to resume with (in x0/rax)
    ///
    /// # Safety
    /// This function never returns - it restores the saved execution context
    #[no_mangle]
    pub unsafe extern "C" fn __sounio_resume_continuation(
        cont_ptr: *mut NativeContinuation,
        value: u64,
    ) -> ! {
        let mut cont = Box::from_raw(cont_ptr);
        cont.resume(value)
    }

    /// Free a continuation without resuming it
    #[no_mangle]
    pub unsafe extern "C" fn __sounio_free_continuation(cont_ptr: *mut NativeContinuation) {
        let _ = Box::from_raw(cont_ptr);
        // Drop automatically frees the continuation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cps_context_creation() {
        let ctx = CpsContext::new();
        assert!(ctx.selective);
        assert!(ctx.effectful_functions.is_empty());
    }

    #[test]
    fn test_cps_context_full_mode() {
        let ctx = CpsContext::new().with_full_cps();
        assert!(!ctx.selective);
    }

    #[test]
    fn test_native_continuation_creation() {
        let cont = NativeContinuation::new(0x1000, true);
        assert!(cont.is_one_shot);
        assert_eq!(cont.resume_count, 0);
        assert_eq!(cont.return_address, 0x1000);
    }

    #[test]
    fn test_cps_transform_creation() {
        let _transform = CpsTransform::new();
    }
}
