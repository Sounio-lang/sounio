//! Native Backend Effect Support
//!
//! This module provides effect dispatch support for the native AArch64/x86-64 backend.
//! It generates machine code that calls into the unified effect runtime (handler_stack.rs).
//!
//! # Architecture
//!
//! ```text
//! Native Code (AArch64/x86-64)
//!     ↓
//! BL/CALL to __sounio_dispatch_* functions
//!     ↓
//! runtime::handler_stack C functions
//!     ↓
//! Effect handlers
//! ```
//!
//! # Implementation Strategy
//!
//! For each effect operation in the code:
//! 1. Load arguments into registers according to calling convention
//! 2. Generate BL (branch-link) instruction to the appropriate runtime function
//! 3. Handle return value in register (f64 result)
//!
//! # Calling Convention (AAPCS64 for AArch64)
//!
//! - Arguments: X0-X7 (integers), D0-D7 (floats)
//! - Return: X0 (integer), D0 (float)
//! - Link register: X30 (LR)
//!
//! All effect dispatch functions return f64, so we use D0 for return values.

use crate::backend::effect_dispatch::{
    EffectDispatch, EffectError, EffectOp, UnifiedEffectRuntime,
};
use crate::backend::native::aarch64::{AArch64Emitter, AArch64Reg};

/// Effect dispatcher for native AArch64 backend
///
/// This struct wraps an AArch64 emitter and provides methods to generate
/// machine code for effect dispatch operations.
pub struct NativeEffectDispatcher {
    /// The underlying unified runtime (used for testing/validation)
    runtime: UnifiedEffectRuntime,
}

impl NativeEffectDispatcher {
    /// Create a new native effect dispatcher
    pub fn new() -> Self {
        Self {
            runtime: UnifiedEffectRuntime::new(),
        }
    }

    /// Generate code to dispatch an effect operation
    ///
    /// This emits AArch64 instructions to:
    /// 1. Load arguments into D0, D1, D2, D3 (f64 calling convention)
    /// 2. Call the appropriate __sounio_dispatch_* function
    /// 3. Leave result in D0
    ///
    /// # Arguments
    /// * `emitter` - The AArch64 code emitter
    /// * `effect` - Effect name (e.g., "IO", "Mut")
    /// * `op` - Operation name (e.g., "print", "get")
    /// * `arg_regs` - Registers containing arguments (as f64 values in D registers)
    /// * `result_reg` - Register to store result (f64)
    pub fn emit_effect_dispatch(
        &self,
        emitter: &mut AArch64Emitter,
        effect: &str,
        op: &str,
        arg_regs: &[AArch64Reg],
        result_reg: AArch64Reg,
    ) -> Result<(), EffectError> {
        let effect_op = EffectOp::parse(effect, op);

        // Get the runtime function name for this effect operation
        let runtime_fn_name = self.get_runtime_function_name(&effect_op)?;

        // Move arguments to the correct registers according to calling convention
        // D0-D7 for f64 arguments
        for (i, &arg_reg) in arg_regs.iter().enumerate().take(8) {
            if i >= 8 {
                return Err(EffectError::InvalidArgs(
                    "Too many arguments (max 8)".to_string(),
                ));
            }

            let target_reg = match i {
                0 => AArch64Reg::V0,
                1 => AArch64Reg::V1,
                2 => AArch64Reg::V2,
                3 => AArch64Reg::V3,
                4 => AArch64Reg::V4,
                5 => AArch64Reg::V5,
                6 => AArch64Reg::V6,
                7 => AArch64Reg::V7,
                _ => unreachable!(),
            };

            // If arg is not already in the target register, move it
            if arg_reg != target_reg {
                emitter.fmov_reg(target_reg, arg_reg);
            }
        }

        // Generate a branch-link (BL) instruction to the runtime function
        // Note: In a real implementation, we'd need to:
        // 1. Resolve the function address (via linking or relocation)
        // 2. Generate the appropriate BL instruction
        //
        // For now, we emit a placeholder that will be resolved by the linker
        emitter.bl_external(&runtime_fn_name);

        // Move result from D0 to result_reg if needed
        if result_reg != AArch64Reg::V0 {
            emitter.fmov(result_reg, AArch64Reg::V0, true);
        }

        Ok(())
    }

    /// Generate code to push a handler onto the stack
    ///
    /// This emits a call to __sounio_push_handler_<effect>()
    pub fn emit_push_handler(
        &self,
        emitter: &mut AArch64Emitter,
        effect: &str,
        _handler_id: u32,
    ) -> Result<(), EffectError> {
        let fn_name = match effect.to_lowercase().as_str() {
            "io" => "__sounio_push_handler_io",
            "mut" => "__sounio_push_handler_mut",
            "div" => "__sounio_push_handler_div",
            "prob" => "__sounio_push_handler_prob",
            "alloc" => "__sounio_push_handler_alloc",
            "panic" => "__sounio_push_handler_panic",
            "async" => "__sounio_push_handler_async",
            "gpu" => "__sounio_push_handler_gpu",
            "grad" => "__sounio_push_handler_grad",
            "network" => "__sounio_push_handler_network",
            "sensor" => "__sounio_push_handler_sensor",
            "exn" => "__sounio_push_handler_exn",
            "causal" => "__sounio_push_handler_causal",
            _ => return Err(EffectError::NoHandler(effect.to_string())),
        };

        emitter.bl_external(fn_name);
        Ok(())
    }

    /// Generate code to pop a handler from the stack
    pub fn emit_pop_handler(&self, emitter: &mut AArch64Emitter) -> Result<(), EffectError> {
        emitter.bl_external("__sounio_pop_handler");
        Ok(())
    }

    /// Get the runtime function name for an effect operation
    fn get_runtime_function_name(&self, effect_op: &EffectOp) -> Result<String, EffectError> {
        let fn_name = match effect_op {
            // IO operations
            EffectOp::IoPrint => "__sounio_dispatch_io_print",
            EffectOp::IoPrintln => "__sounio_dispatch_io_println",
            EffectOp::IoRead => "__sounio_dispatch_io_read",

            // Mut operations
            EffectOp::MutGet => "__sounio_dispatch_mut_get",
            EffectOp::MutSet => "__sounio_dispatch_mut_set",
            EffectOp::MutModify => "__sounio_dispatch_mut_modify",

            // Div operations
            EffectOp::DivDiv => "__sounio_dispatch_div_div",

            // Prob operations
            EffectOp::ProbSample => "__sounio_dispatch_prob_sample",
            EffectOp::ProbObserve => "__sounio_dispatch_prob_observe",

            // Alloc operations
            EffectOp::AllocAlloc => "__sounio_dispatch_alloc_alloc",
            EffectOp::AllocDealloc => "__sounio_dispatch_alloc_dealloc",

            // Panic operations
            EffectOp::PanicPanic => "__sounio_dispatch_panic_panic",

            // GPU operations
            EffectOp::GpuSync => "__sounio_dispatch_gpu_sync",
            EffectOp::GpuLaunch => "__sounio_dispatch_gpu_launch",

            // Async operations
            EffectOp::AsyncSpawn => "__sounio_dispatch_async_spawn",
            EffectOp::AsyncYield => "__sounio_dispatch_async_yield",
            EffectOp::AsyncAwait => "__sounio_dispatch_async_await",

            // Grad operations
            EffectOp::GradForward => "__sounio_dispatch_grad_forward",
            EffectOp::GradReverse => "__sounio_dispatch_grad_reverse",

            // Causal operations
            EffectOp::CausalDo => "__sounio_dispatch_causal_do",
            EffectOp::CausalObserve => "__sounio_dispatch_causal_observe",

            // Generic dispatch for unhandled operations
            _ => "__sounio_dispatch_generic",
        };

        Ok(fn_name.to_string())
    }

    /// Get access to the underlying runtime (for testing)
    pub fn runtime(&self) -> &UnifiedEffectRuntime {
        &self.runtime
    }

    /// Get mutable access to the underlying runtime (for testing)
    pub fn runtime_mut(&mut self) -> &mut UnifiedEffectRuntime {
        &mut self.runtime
    }

    /// Generate code to perform an epistemic operation on Knowledge<T> values
    ///
    /// This emits AArch64 instructions to:
    /// 1. Load Knowledge struct pointers into X0, X1, X2 (System V ABI)
    /// 2. Call the appropriate sounio_epistemic_* function
    /// 3. Handle result in the result struct
    ///
    /// # Arguments
    /// * `emitter` - The AArch64 code emitter
    /// * `op` - Operation name ("add", "sub", "mul", "div")
    /// * `layout` - Layout variant ("full" or "compact")
    /// * `a_ptr_reg` - Register containing pointer to first Knowledge value
    /// * `b_ptr_reg` - Register containing pointer to second Knowledge value
    /// * `result_ptr_reg` - Register containing pointer to result Knowledge value
    pub fn emit_epistemic_op(
        &self,
        emitter: &mut AArch64Emitter,
        op: &str,
        layout: &str,
        a_ptr_reg: AArch64Reg,
        b_ptr_reg: AArch64Reg,
        result_ptr_reg: AArch64Reg,
    ) -> Result<(), EffectError> {
        // Construct function name
        let fn_name = format!("sounio_epistemic_{}_{}", op, layout);

        // Move pointers to correct registers for System V ABI
        // X0 = a, X1 = b, X2 = result
        if a_ptr_reg != AArch64Reg::X0 {
            emitter.mov_reg(AArch64Reg::X0, a_ptr_reg, true);
        }
        if b_ptr_reg != AArch64Reg::X1 {
            emitter.mov_reg(AArch64Reg::X1, b_ptr_reg, true);
        }
        if result_ptr_reg != AArch64Reg::X2 {
            emitter.mov_reg(AArch64Reg::X2, result_ptr_reg, true);
        }

        // Generate call to epistemic runtime function
        emitter.bl_label(&fn_name);

        Ok(())
    }

    /// Generate code to extract value from Knowledge<T>
    ///
    /// Calls sounio_epistemic_get_value to extract the f64 value.
    /// Result is returned in D0 (XMM0 in x86-64).
    pub fn emit_epistemic_get_value(
        &self,
        emitter: &mut AArch64Emitter,
        knowledge_ptr_reg: AArch64Reg,
        result_reg: AArch64Reg,
    ) -> Result<(), EffectError> {
        // Move pointer to X0
        if knowledge_ptr_reg != AArch64Reg::X0 {
            emitter.mov_reg(AArch64Reg::X0, knowledge_ptr_reg, true);
        }

        // Call runtime function
        emitter.bl_label("sounio_epistemic_get_value");

        // Move result from D0 to result_reg if needed
        if result_reg != AArch64Reg::V0 {
            emitter.fmov(result_reg, AArch64Reg::V0, true);
        }

        Ok(())
    }

    /// Generate code to extract confidence from Knowledge<T>
    ///
    /// Calls sounio_epistemic_get_confidence to extract the confidence value.
    /// Result is returned in D0 (XMM0 in x86-64).
    pub fn emit_epistemic_get_confidence(
        &self,
        emitter: &mut AArch64Emitter,
        knowledge_ptr_reg: AArch64Reg,
        result_reg: AArch64Reg,
    ) -> Result<(), EffectError> {
        // Move pointer to X0
        if knowledge_ptr_reg != AArch64Reg::X0 {
            emitter.mov_reg(AArch64Reg::X0, knowledge_ptr_reg, true);
        }

        // Call runtime function
        emitter.bl_label("sounio_epistemic_get_confidence");

        // Move result from D0 to result_reg if needed
        if result_reg != AArch64Reg::V0 {
            emitter.fmov(result_reg, AArch64Reg::V0, true);
        }

        Ok(())
    }
}

impl Default for NativeEffectDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectDispatch for NativeEffectDispatcher {
    fn dispatch_effect(
        &mut self,
        effect: &str,
        op: &str,
        args: &[f64],
    ) -> Result<f64, EffectError> {
        // Delegate to the unified runtime for actual dispatch
        // (This is used for testing/validation, not code generation)
        self.runtime.dispatch_effect(effect, op, args)
    }

    fn push_handler(&mut self, effect: &str, handler_id: u32) -> Result<(), EffectError> {
        self.runtime.push_handler(effect, handler_id)
    }

    fn pop_handler(&mut self) -> Result<(), EffectError> {
        self.runtime.pop_handler()
    }

    fn handler_depth(&self) -> usize {
        self.runtime.handler_depth()
    }
}

// ============================================================================
// AArch64Emitter Extensions
// ============================================================================

impl AArch64Emitter {
    /// FMOV (register to register) for f64 values
    ///
    /// Copies a floating-point register to another floating-point register.
    /// Format: FMOV Vd.D, Vn.D
    pub fn fmov_reg(&mut self, vd: AArch64Reg, vn: AArch64Reg) {
        // FMOV Vd, Vn (vector form, 64-bit)
        // Format: 0001111001100000000001nnnnnddddd
        // Where sf=0, type=01 (64-bit), rm=00000, opcode=000000, rn=nnnnn, rd=ddddd
        let inst = 0b0001111001100000_00000100000_00000 | (vn.encoding() << 5) | vd.encoding();
        self.emit(inst);
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_dispatcher_creation() {
        let mut dispatcher = NativeEffectDispatcher::new();
        dispatcher.runtime_mut().reset(); // Reset global state for parallel test execution
        assert_eq!(dispatcher.handler_depth(), 0);
    }

    #[test]
    fn test_native_dispatcher_push_pop() {
        let mut dispatcher = NativeEffectDispatcher::new();
        dispatcher.runtime_mut().reset(); // Reset global state

        assert!(dispatcher.push_handler("IO", 1).is_ok());
        assert_eq!(dispatcher.handler_depth(), 1);

        assert!(dispatcher.pop_handler().is_ok());
        assert_eq!(dispatcher.handler_depth(), 0);
    }

    #[test]
    fn test_native_dispatcher_dispatch() {
        let mut dispatcher = NativeEffectDispatcher::new();
        dispatcher.runtime_mut().reset(); // Reset global state

        dispatcher.push_handler("Div", 3).unwrap();

        let result = dispatcher.dispatch_effect("Div", "div", &[20.0, 4.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5.0);
    }

    #[test]
    fn test_get_runtime_function_name() {
        let dispatcher = NativeEffectDispatcher::new();

        let fn_name = dispatcher
            .get_runtime_function_name(&EffectOp::IoPrint)
            .unwrap();
        assert_eq!(fn_name, "__sounio_dispatch_io_print");

        let fn_name = dispatcher
            .get_runtime_function_name(&EffectOp::MutGet)
            .unwrap();
        assert_eq!(fn_name, "__sounio_dispatch_mut_get");

        let fn_name = dispatcher
            .get_runtime_function_name(&EffectOp::DivDiv)
            .unwrap();
        assert_eq!(fn_name, "__sounio_dispatch_div_div");
    }

    #[test]
    fn test_emit_push_handler_generates_code() {
        let mut emitter = AArch64Emitter::new();
        let dispatcher = NativeEffectDispatcher::new();

        let initial_len = emitter.offset();
        assert!(dispatcher.emit_push_handler(&mut emitter, "IO", 1).is_ok());

        // Should have emitted at least one instruction (BL)
        assert!(emitter.offset() > initial_len);
    }

    #[test]
    fn test_emit_pop_handler_generates_code() {
        let mut emitter = AArch64Emitter::new();
        let dispatcher = NativeEffectDispatcher::new();

        let initial_len = emitter.offset();
        assert!(dispatcher.emit_pop_handler(&mut emitter).is_ok());

        // Should have emitted at least one instruction (BL)
        assert!(emitter.offset() > initial_len);
    }

    #[test]
    fn test_emit_effect_dispatch_generates_code() {
        let mut emitter = AArch64Emitter::new();
        let dispatcher = NativeEffectDispatcher::new();

        // Simulate dispatching IO.print with one argument in V2
        let arg_regs = vec![AArch64Reg::V2];
        let result_reg = AArch64Reg::V0;

        let initial_len = emitter.offset();
        assert!(
            dispatcher
                .emit_effect_dispatch(&mut emitter, "IO", "print", &arg_regs, result_reg)
                .is_ok()
        );

        // Should have emitted instructions (move + BL at minimum)
        assert!(emitter.offset() > initial_len);
    }
}
