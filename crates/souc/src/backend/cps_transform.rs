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
use crate::hlir::ir::{
    BlockId, HlirBlock, HlirFunction, HlirInstr, HlirModule, HlirParam, HlirTerminator, HlirType,
    Op, ValueId,
};
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

    /// Analyze module to determine which functions need CPS
    pub fn analyze(&mut self, module: &HlirModule) {
        for func in &module.functions {
            if self.uses_effects(func) {
                self.effectful_functions.insert(func.name.clone());
            }
        }
    }

    /// Check if a function uses effects
    fn uses_effects(&self, func: &HlirFunction) -> bool {
        // Scan all ops in the function for PerformEffect or DispatchEffect
        for block in &func.blocks {
            for instr in &block.instructions {
                if let Op::PerformEffect { .. } | Op::DispatchEffect { .. } = instr.op {
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

    /// Transform a module into CPS
    pub fn transform(&mut self, module: HlirModule) -> CpsResult<HlirModule> {
        // Phase 1: Analyze which functions use effects
        self.ctx.analyze(&module);

        // Phase 2: Transform effectful functions to CPS
        let mut new_module = module.clone();
        let original_funcs = module.functions.clone();

        for func in &original_funcs {
            if self.ctx.needs_cps(&func.name) {
                let cps_func = self.transform_function(func)?;
                let cps_name = format!("{}_cps", func.name);

                self.ctx
                    .cps_functions
                    .insert(func.name.clone(), cps_name.clone());

                new_module.functions.push(cps_func);
            }
        }

        Ok(new_module)
    }

    /// Transform a single function to CPS
    fn transform_function(&mut self, func: &HlirFunction) -> CpsResult<HlirFunction> {
        // Create new function with CPS suffix
        let cps_name = format!("{}_cps", func.name);

        // Clone the function structure
        let mut cps_func = func.clone();
        cps_func.name = cps_name;

        // Add continuation parameter
        // Continuation is represented as a pointer to NativeContinuation struct
        let cont_param = HlirParam {
            value: ValueId(func.params.len() as u32),
            name: "__cont".to_string(),
            ty: HlirType::Ptr, // Pointer to continuation
        };
        cps_func.params.push(cont_param);

        // Transform each basic block
        for block in &mut cps_func.blocks {
            self.transform_block(block)?;
        }

        // Transform terminators to use continuation
        for block in &mut cps_func.blocks {
            self.transform_terminator(&mut block.terminator)?;
        }

        Ok(cps_func)
    }

    /// Transform a basic block to insert continuation captures
    fn transform_block(&mut self, block: &mut HlirBlock) -> CpsResult<()> {
        let mut new_instrs = Vec::new();

        for instr in &block.instructions {
            match &instr.op {
                Op::PerformEffect { effect, op, args }
                | Op::DispatchEffect { effect, op, args } => {
                    // Generate continuation ID
                    let cont_id = self.ctx.fresh_cont_id();

                    // Insert call to capture continuation
                    // This would be: let cont_ptr = __sounio_capture_continuation_asm(&cont_storage)
                    // For now, we emit a placeholder that will be recognized by the backend

                    // Call to capture continuation (assembly stub)
                    new_instrs.push(HlirInstr {
                        result: Some(ValueId(new_instrs.len() as u32 + 1000)),
                        op: Op::CallDirect {
                            name: "__sounio_capture_continuation_asm".to_string(),
                            args: vec![], // Will be filled in by backend with cont storage ptr
                        },
                        ty: HlirType::Ptr,
                    });

                    // Now perform the effect, passing the continuation
                    let mut effect_args = args.clone();
                    // Note: The effect handler runtime will receive the continuation separately

                    new_instrs.push(HlirInstr {
                        result: instr.result,
                        op: if matches!(instr.op, Op::PerformEffect { .. }) {
                            Op::PerformEffect {
                                effect: effect.clone(),
                                op: op.clone(),
                                args: effect_args,
                            }
                        } else {
                            Op::DispatchEffect {
                                effect: effect.clone(),
                                op: op.clone(),
                                args: effect_args,
                            }
                        },
                        ty: instr.ty.clone(),
                    });

                    // The continuation resume will happen in the effect handler
                    // No explicit resume needed here - handler calls __sounio_resume_continuation_asm
                }
                _ => {
                    // Non-effect instructions pass through unchanged
                    new_instrs.push(instr.clone());
                }
            }
        }

        block.instructions = new_instrs;
        Ok(())
    }

    /// Transform function terminator to use continuation
    fn transform_terminator(&mut self, terminator: &mut HlirTerminator) -> CpsResult<()> {
        // Transform Return into continuation resume
        // This would replace: return x
        // With: __sounio_resume_continuation(__cont, x)

        match terminator {
            HlirTerminator::Return(_value_opt) => {
                // Replace return with call to resume continuation
                // The continuation resume is a tail call that never returns
                // NOTE: In a complete implementation, we would insert a call to
                // __sounio_resume_continuation_asm here. For now, we leave the
                // return as-is and rely on the backend to convert it to a
                // continuation resume when generating native code.

                // *terminator = HlirTerminator::Unreachable;
                // For now, leave returns unchanged - backend will handle
            }
            _ => {
                // Other terminators (branches, unreachable) don't need transformation
            }
        }

        Ok(())
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
    use crate::ast::Abi;
    use crate::hlir::builder::FunctionBuilder;

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

    #[test]
    fn test_cps_transform_effectful_function() {
        // Create a simple function with an effect operation
        let mut builder = FunctionBuilder::new("test_func".to_string(), HlirType::I32);
        builder.set_effects(vec!["IO".to_string()]);

        // Add entry block
        builder.create_block("entry");
        builder.set_current_block(BlockId(0));

        // Perform IO effect
        let print_result = builder.perform_effect("IO", "println", vec![], HlirType::Void);

        // Return constant
        let const_val = builder.int_const(42, HlirType::I32);
        builder.ret(Some(const_val));

        let func = builder.finish();

        // Transform to CPS
        let mut transform = CpsTransform::new();
        let result = transform.transform_function(&func);

        assert!(result.is_ok(), "CPS transformation should succeed");

        let cps_func = result.unwrap();

        // Verify transformation
        assert_eq!(cps_func.name, "test_func_cps");
        assert_eq!(
            cps_func.params.len(),
            func.params.len() + 1,
            "Should have one additional continuation parameter"
        );
        assert_eq!(cps_func.params.last().unwrap().name, "__cont");
    }

    #[test]
    fn test_cps_analysis_detects_effects() {
        // Create a module with an effectful function
        let mut module = HlirModule::new("test");

        let mut builder = FunctionBuilder::new("with_effects".to_string(), HlirType::I32);
        builder.set_effects(vec!["IO".to_string()]);
        builder.create_block("entry");
        builder.set_current_block(BlockId(0));
        builder.perform_effect("IO", "println", vec![], HlirType::Void);
        let val = builder.int_const(42, HlirType::I32);
        builder.ret(Some(val));

        module.functions.push(builder.finish());

        // Analyze
        let mut ctx = CpsContext::new();
        ctx.analyze(&module);

        assert!(
            ctx.effectful_functions.contains("with_effects"),
            "Should detect effectful function"
        );
        assert!(ctx.needs_cps("with_effects"));
    }

    #[test]
    fn test_cps_analysis_ignores_pure_functions() {
        // Create a module with a pure function
        let mut module = HlirModule::new("test");

        let mut builder = FunctionBuilder::new("pure_func".to_string(), HlirType::I32);
        builder.create_block("entry");
        builder.set_current_block(BlockId(0));
        let val = builder.int_const(42, HlirType::I32);
        builder.ret(Some(val));

        module.functions.push(builder.finish());

        // Analyze
        let mut ctx = CpsContext::new();
        ctx.analyze(&module);

        assert!(
            !ctx.effectful_functions.contains("pure_func"),
            "Should not transform pure function"
        );
        assert!(!ctx.needs_cps("pure_func"));
    }
}
