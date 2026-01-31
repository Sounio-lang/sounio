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
    HlirBlock, HlirFunction, HlirInstr, HlirModule, HlirParam, HlirTerminator, HlirType, Op,
    ValueId,
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

    /// Effect operations that require multi-shot continuations
    /// (e.g., Amb::choose, Prob::branch_all, Search::backtrack)
    multi_shot_operations: HashSet<(String, String)>, // (effect, operation)
}

impl CpsContext {
    /// Create a new CPS transformation context
    pub fn new() -> Self {
        let mut ctx = Self {
            effectful_functions: HashSet::new(),
            cps_functions: HashMap::new(),
            next_cont_id: 0,
            selective: true, // Use selective CPS by default
            multi_shot_operations: HashSet::new(),
        };

        // Register known multi-shot operations
        ctx.register_multi_shot("Amb", "choose");
        ctx.register_multi_shot("Amb", "all");
        ctx.register_multi_shot("Prob", "branch_all");
        ctx.register_multi_shot("Prob", "sample_many");
        ctx.register_multi_shot("Search", "backtrack");
        ctx.register_multi_shot("Nondeterminism", "choice");

        ctx
    }

    /// Enable full CPS transformation (transform all functions)
    pub fn with_full_cps(mut self) -> Self {
        self.selective = false;
        self
    }

    /// Register an operation as requiring multi-shot continuations
    pub fn register_multi_shot(&mut self, effect: &str, operation: &str) {
        self.multi_shot_operations
            .insert((effect.to_string(), operation.to_string()));
    }

    /// Check if an operation requires multi-shot continuation
    pub fn is_multi_shot(&self, effect: &str, operation: &str) -> bool {
        self.multi_shot_operations
            .contains(&(effect.to_string(), operation.to_string()))
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
        let cont_param_id = ValueId(func.params.len() as u32);
        let cont_param = HlirParam {
            value: cont_param_id,
            name: "__cont".to_string(),
            ty: HlirType::Ptr(Box::new(HlirType::U64)), // Pointer to continuation (opaque pointer)
        };
        cps_func.params.push(cont_param);

        // Transform each basic block (both instructions and terminator)
        for block in &mut cps_func.blocks {
            self.transform_block(block)?;
            self.transform_block_terminator(block, cont_param_id)?;
        }

        Ok(cps_func)
    }

    /// Transform a basic block to insert continuation captures
    fn transform_block(&mut self, block: &mut HlirBlock) -> CpsResult<()> {
        let mut new_instrs = Vec::new();
        let mut next_temp_value_id = 10000; // Start temp value IDs at 10000 to avoid conflicts

        for instr in &block.instructions {
            match &instr.op {
                Op::PerformEffect { effect, op, args }
                | Op::DispatchEffect { effect, op, args } => {
                    // Generate continuation ID for tracking/debugging
                    let _cont_id = self.ctx.fresh_cont_id();

                    // Check if this operation requires multi-shot continuation
                    let is_multi_shot = self.ctx.is_multi_shot(effect, op);

                    // Before performing the effect, capture the current continuation
                    // Use different functions for one-shot vs multi-shot
                    // - One-shot: shallow capture (current frame only)
                    // - Multi-shot: deep capture (full stack snapshot)
                    let cont_ptr_id = ValueId(next_temp_value_id);
                    next_temp_value_id += 1;

                    let capture_fn = if is_multi_shot {
                        "__sounio_capture_continuation_multi_shot"
                    } else {
                        "__sounio_capture_continuation"
                    };

                    new_instrs.push(HlirInstr {
                        result: Some(cont_ptr_id),
                        op: Op::CallDirect {
                            name: capture_fn.to_string(),
                            args: vec![], // Capture uses implicit state (registers, stack)
                        },
                        ty: HlirType::Ptr(Box::new(HlirType::U64)),
                    });

                    // Store the continuation in thread-local storage
                    // This allows the effect handler to access it
                    // Generated code: __sounio_store_continuation(cont_ptr)
                    new_instrs.push(HlirInstr {
                        result: None,
                        op: Op::CallDirect {
                            name: "__sounio_store_continuation".to_string(),
                            args: vec![cont_ptr_id],
                        },
                        ty: HlirType::Void,
                    });

                    // Now perform the effect
                    // The effect handler will retrieve the continuation from thread-local storage
                    let effect_args = args.clone();

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

                    // After the effect operation, the handler will have:
                    // 1. Retrieved the continuation from storage
                    // 2. Performed the effect operation
                    // 3. Either resumed the continuation immediately (for simple effects)
                    //    or stored it for later resumption (for async effects)
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

    /// Transform block terminator to use continuation
    ///
    /// This transforms control flow operations to use explicit continuation passing:
    /// - Return(value) => __sounio_resume_continuation(__cont, value); unreachable
    /// - CondBranch => Pass continuation to both branches
    fn transform_block_terminator(
        &mut self,
        block: &mut HlirBlock,
        cont_param_id: ValueId,
    ) -> CpsResult<()> {
        match &block.terminator {
            HlirTerminator::Return(value_opt) => {
                // Transform: return value
                // Into: __sounio_resume_continuation(__cont, value); unreachable

                let resume_args = if let Some(value_id) = value_opt {
                    vec![cont_param_id, *value_id]
                } else {
                    // Returning void - pass 0 as the value
                    vec![cont_param_id, ValueId::UNIT]
                };

                // Insert a call to __sounio_resume_continuation before the terminator
                block.instructions.push(HlirInstr {
                    result: None, // Resume never returns, so no result
                    op: Op::CallDirect {
                        name: "__sounio_resume_continuation".to_string(),
                        args: resume_args,
                    },
                    ty: HlirType::Void,
                });

                // Replace return with unreachable (continuation resume doesn't return)
                block.terminator = HlirTerminator::Unreachable;
            }
            HlirTerminator::CondBranch {
                condition: _,
                then_block: _,
                else_block: _,
            } => {
                // Conditional branches in CPS don't need transformation at the HLIR level.
                //
                // Since blocks are within the same function (not separate closures),
                // both then_block and else_block have access to the continuation parameter
                // through the function's scope. The continuation parameter (cont_param_id)
                // is available to all blocks in the function.
                //
                // Example:
                //   fn foo_cps(__cont: *Continuation, x: i32) {
                //     bb0:
                //       br_cond x > 0, bb1, bb2
                //     bb1:  // then branch - __cont is in scope
                //       let result = x + 1
                //       __sounio_resume_continuation(__cont, result)
                //       unreachable
                //     bb2:  // else branch - __cont is in scope
                //       let result = x - 1
                //       __sounio_resume_continuation(__cont, result)
                //       unreachable
                //   }
                //
                // Both bb1 and bb2 can use __cont directly because it's a function parameter.
                // No transformation needed - the terminator remains unchanged.

                // Verify that the target blocks exist (defensive check)
                // In a more complete implementation, we could validate that both branches
                // properly use the continuation, but that's better done in a separate
                // validation pass.
            }
            HlirTerminator::Switch {
                value: _,
                default: _,
                cases: _,
            } => {
                // Switch terminators in CPS also don't need transformation.
                //
                // Like conditional branches, all case blocks are within the same function
                // and have access to the continuation parameter through function scope.
                //
                // Example:
                //   fn handle_opcode_cps(__cont: *Continuation, opcode: i32) {
                //     bb0:
                //       switch opcode, default: bb_default, [
                //         (1, bb_case1),
                //         (2, bb_case2),
                //         (3, bb_case3)
                //       ]
                //     bb_case1:  // __cont is in scope
                //       let result = operation1()
                //       __sounio_resume_continuation(__cont, result)
                //       unreachable
                //     bb_case2:  // __cont is in scope
                //       let result = operation2()
                //       __sounio_resume_continuation(__cont, result)
                //       unreachable
                //     bb_default:  // __cont is in scope
                //       __sounio_resume_continuation(__cont, 0)
                //       unreachable
                //   }
                //
                // All case blocks can use __cont directly. No transformation needed.
            }
            HlirTerminator::Branch(_) | HlirTerminator::Unreachable => {
                // Unconditional branches and unreachable don't need transformation
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
///
/// # One-shot vs Multi-shot
///
/// - **One-shot** (is_one_shot = true): Shallow capture
///   - Only captures the current stack frame
///   - More efficient (less copying)
///   - Cannot be resumed multiple times
///   - Used for most effects (IO, State, Error, etc.)
///
/// - **Multi-shot** (is_one_shot = false): Deep capture
///   - Captures the full call stack from current frame to effect handler
///   - Required for backtracking effects (Amb, Search, Prob::branch_all)
///   - Can be cloned and resumed multiple times
///   - Higher overhead due to stack copying
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
    /// - One-shot: Only current frame (FP to SP)
    /// - Multi-shot: Full call stack (deep capture)
    pub stack_data: Vec<u8>,

    /// Whether this is a one-shot or multi-shot continuation
    pub is_one_shot: bool,

    /// Resume count (for one-shot enforcement)
    pub resume_count: usize,

    /// Stack depth (number of frames captured, for multi-shot)
    pub stack_depth: usize,
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
            stack_depth: if is_one_shot { 1 } else { 0 },
        }
    }

    /// Capture continuation with shallow stack (current frame only)
    ///
    /// Used for one-shot continuations. More efficient than deep capture.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn capture_shallow() -> Self {
        // Use the existing capture() method which does shallow capture
        // It only saves register state via the native backend
        Self::capture()
    }

    /// Capture continuation with deep stack (full call stack)
    ///
    /// Used for multi-shot continuations. Captures entire stack from
    /// current frame up to the effect handler boundary.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn capture_deep() -> Self {
        use crate::backend::native::continuation;

        // First capture shallow state (registers)
        let native_cont = continuation::capture_continuation();

        let mut cont = Self::new(native_cont.return_address(), false);

        // Copy register state from native continuation
        match &native_cont {
            continuation::Continuation::AArch64(aarch64_cont) => {
                // Copy GP registers (X0-X30)
                for i in 0..31 {
                    cont.gp_registers[i] = aarch64_cont.gp_regs[i];
                }

                // Copy SIMD registers (stored as pairs of u64)
                for i in 0..32 {
                    cont.fp_registers[i] = f64::from_bits(aarch64_cont.simd_regs[i * 2]);
                }

                cont.stack_pointer = aarch64_cont.sp as usize;
                cont.frame_pointer = aarch64_cont.fp as usize;

                // Deep stack capture: walk the call stack using frame pointers
                // Each frame on AArch64 has: [FP_prev, LR_prev] at the top
                let mut current_fp = aarch64_cont.fp as usize;
                let sp = aarch64_cont.sp as usize;
                let mut frame_count = 0;
                const MAX_FRAMES: usize = 100; // Safety limit

                // Calculate total stack size by walking frame chain
                let mut total_stack_size = 0;
                let mut temp_fp = current_fp;

                while temp_fp != 0 && frame_count < MAX_FRAMES {
                    if temp_fp < sp {
                        break; // Invalid frame pointer
                    }

                    // Each frame contributes from its FP to previous FP
                    let frame_size = if frame_count == 0 {
                        temp_fp - sp
                    } else {
                        // Read previous FP from current frame
                        let prev_fp_ptr = temp_fp as *const usize;
                        let prev_fp = *prev_fp_ptr;
                        if prev_fp <= temp_fp || prev_fp - temp_fp > 1024 * 1024 {
                            break; // Invalid or too large
                        }
                        prev_fp - temp_fp
                    };

                    total_stack_size += frame_size;
                    frame_count += 1;

                    // Move to previous frame
                    let prev_fp_ptr = temp_fp as *const usize;
                    temp_fp = *prev_fp_ptr;
                }

                // Allocate and copy the deep stack
                cont.stack_data = Vec::with_capacity(total_stack_size);
                let stack_slice = std::slice::from_raw_parts(sp as *const u8, total_stack_size);
                cont.stack_data.extend_from_slice(stack_slice);
                cont.stack_depth = frame_count;
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported continuation type for deep capture"),
        }

        cont
    }

    /// Capture continuation with shallow stack (x86-64)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn capture_shallow() -> Self {
        unsafe { Self::capture() }
    }

    /// Capture continuation with deep stack (x86-64)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn capture_deep() -> Self {
        unsafe {
            use crate::backend::native::continuation;

            let native_cont = continuation::capture_continuation();
            let mut cont = Self::new(native_cont.return_address(), false);

            match &native_cont {
                continuation::Continuation::X86_64(x86_cont) => {
                    // Copy GP registers
                    for i in 0..14 {
                        cont.gp_registers[i] = x86_cont.gp_regs[i];
                    }

                    // Copy SSE registers
                    for i in 0..16 {
                        cont.fp_registers[i] = f64::from_bits(x86_cont.sse_regs[i * 2]);
                    }

                    cont.stack_pointer = x86_cont.rsp as usize;
                    cont.frame_pointer = x86_cont.rbp as usize;

                    // Deep stack capture: walk the call stack using frame pointers
                    // x86-64 frame layout: [saved RBP, return address, ...]
                    let _current_rbp = x86_cont.rbp as usize;
                    let rsp = x86_cont.rsp as usize;
                    let mut frame_count = 0;
                    const MAX_FRAMES: usize = 100;

                    let mut total_stack_size = 0;
                    let mut temp_rbp = _current_rbp;

                    while temp_rbp != 0 && frame_count < MAX_FRAMES {
                        if temp_rbp < rsp {
                            break;
                        }

                        let frame_size = if frame_count == 0 {
                            temp_rbp - rsp
                        } else {
                            let prev_rbp_ptr = temp_rbp as *const usize;
                            let prev_rbp = *prev_rbp_ptr;
                            if prev_rbp <= temp_rbp || prev_rbp - temp_rbp > 1024 * 1024 {
                                break;
                            }
                            prev_rbp - temp_rbp
                        };

                        total_stack_size += frame_size;
                        frame_count += 1;

                        let prev_rbp_ptr = temp_rbp as *const usize;
                        temp_rbp = *prev_rbp_ptr;
                    }

                    cont.stack_data = Vec::with_capacity(total_stack_size);
                    let stack_slice =
                        std::slice::from_raw_parts(rsp as *const u8, total_stack_size);
                    cont.stack_data.extend_from_slice(stack_slice);
                    cont.stack_depth = frame_count;
                }
                #[allow(unreachable_patterns)]
                _ => panic!("Unsupported continuation type for deep capture"),
            }

            cont
        }
    }

    /// Capture current machine state
    ///
    /// SAFETY: This is highly architecture-specific and unsafe.
    /// Must be called from assembly stubs that save registers.
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn capture() -> Self {
        // Use native backend continuation capture
        use crate::backend::native::continuation;

        let native_cont = continuation::capture_continuation();

        let mut cont = Self::new(native_cont.return_address(), true);

        // Copy register state from native continuation
        match native_cont {
            continuation::Continuation::AArch64(aarch64_cont) => {
                // Copy GP registers (X0-X30)
                for i in 0..31 {
                    cont.gp_registers[i] = aarch64_cont.gp_regs[i];
                }

                // Copy SIMD registers (stored as pairs of u64)
                for i in 0..32 {
                    cont.fp_registers[i] = f64::from_bits(aarch64_cont.simd_regs[i * 2]);
                }

                cont.stack_pointer = aarch64_cont.sp as usize;
                cont.frame_pointer = aarch64_cont.fp as usize;
                cont.stack_data = aarch64_cont.stack.clone();
            }
            #[allow(unreachable_patterns)]
            _ => panic!("Unsupported continuation type"),
        }

        cont
    }

    /// Capture current machine state (x86-64)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn capture() -> Self {
        unsafe {
            use crate::backend::native::continuation;

            let native_cont = continuation::capture_continuation();

            let mut cont = Self::new(native_cont.return_address(), true);

            // Copy register state from native continuation
            match native_cont {
                continuation::Continuation::X86_64(x86_cont) => {
                    // Copy GP registers (RAX, RBX, RCX, RDX, RSI, RDI, R8-R15)
                    for i in 0..14 {
                        cont.gp_registers[i] = x86_cont.gp_regs[i];
                    }

                    // Copy SSE registers (XMM0-XMM15, stored as pairs of u64)
                    for i in 0..16 {
                        cont.fp_registers[i] = f64::from_bits(x86_cont.sse_regs[i * 2]);
                    }

                    cont.stack_pointer = x86_cont.rsp as usize;
                    cont.frame_pointer = x86_cont.rbp as usize;
                    cont.stack_data = x86_cont.stack.clone();
                }
                #[allow(unreachable_patterns)]
                _ => panic!("Unsupported continuation type"),
            }

            cont
        }
    }

    /// Resume this continuation with a value
    ///
    /// SAFETY: Extremely unsafe - overwrites current execution context
    #[cfg(target_arch = "aarch64")]
    pub unsafe fn resume(&mut self, value: u64) -> ! {
        // Check one-shot constraint
        if self.is_one_shot && self.resume_count > 0 {
            panic!("Attempted to resume one-shot continuation multiple times");
        }
        self.resume_count += 1;

        // Use native backend continuation resume
        use crate::backend::native::continuation;

        // Reconstruct native continuation from our state
        let mut native_cont =
            continuation::Continuation::AArch64(continuation::AArch64Continuation {
                gp_regs: {
                    let mut regs = [0u64; 31];
                    for i in 0..31 {
                        regs[i] = self.gp_registers[i];
                    }
                    regs
                },
                simd_regs: {
                    let mut regs = [0u64; 64];
                    for i in 0..32 {
                        regs[i * 2] = self.fp_registers[i].to_bits();
                        regs[i * 2 + 1] = 0; // High 64 bits (we only stored low 64)
                    }
                    regs
                },
                sp: self.stack_pointer as u64,
                fp: self.frame_pointer as u64,
                lr: self.return_address as u64,
                stack: self.stack_data.clone(),
                resumed: false,
            });

        continuation::resume_continuation(&mut native_cont, value);
    }

    /// Resume this continuation with a value (x86-64)
    #[cfg(target_arch = "x86_64")]
    pub unsafe fn resume(&mut self, value: u64) -> ! {
        // Check one-shot constraint
        if self.is_one_shot && self.resume_count > 0 {
            panic!("Attempted to resume one-shot continuation multiple times");
        }
        self.resume_count += 1;

        unsafe {
            use crate::backend::native::continuation;

            // Reconstruct native continuation from our state
            let mut native_cont =
                continuation::Continuation::X86_64(continuation::X86_64Continuation {
                    gp_regs: {
                        let mut regs = [0u64; 14];
                        for i in 0..14 {
                            regs[i] = self.gp_registers[i];
                        }
                        regs
                    },
                    sse_regs: {
                        let mut regs = [0u64; 32];
                        for i in 0..16 {
                            regs[i * 2] = self.fp_registers[i].to_bits();
                            regs[i * 2 + 1] = 0; // High 64 bits
                        }
                        regs
                    },
                    rsp: self.stack_pointer as u64,
                    rbp: self.frame_pointer as u64,
                    rip: self.return_address as u64,
                    stack: self.stack_data.clone(),
                    resumed: false,
                });

            continuation::resume_continuation(&mut native_cont, value);
        }
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
    #[unsafe(no_mangle)]
    pub extern "C" fn __sounio_capture_continuation(
        return_address: usize,
    ) -> *mut NativeContinuation {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        unsafe {
            let cont = NativeContinuation::capture();
            Box::into_raw(Box::new(cont))
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = return_address;
            panic!("Continuation capture only implemented for AArch64 and x86-64")
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
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __sounio_resume_continuation(
        cont_ptr: *mut NativeContinuation,
        _value: u64,
    ) -> ! {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        unsafe {
            let mut cont = *Box::from_raw(cont_ptr);
            cont.resume(_value)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = cont_ptr;
            panic!("Continuation resumption only implemented for AArch64 and x86-64")
        }
    }

    /// Free a continuation without resuming it
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __sounio_free_continuation(cont_ptr: *mut NativeContinuation) {
        unsafe {
            let _ = Box::from_raw(cont_ptr);
            // Drop automatically frees the continuation
        }
    }

    /// Capture a multi-shot continuation with deep stack capture
    ///
    /// Multi-shot continuations can be resumed multiple times, which is required
    /// for backtracking effects like Amb, Prob, and Search.
    ///
    /// # Arguments
    /// * `return_address` - Where to resume when continuation is invoked
    ///
    /// # Returns
    /// Pointer to captured NativeContinuation with full call stack
    #[unsafe(no_mangle)]
    pub extern "C" fn __sounio_capture_continuation_multi_shot(
        return_address: usize,
    ) -> *mut NativeContinuation {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        unsafe {
            let cont = NativeContinuation::capture_deep();
            Box::into_raw(Box::new(cont))
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = return_address;
            panic!("Multi-shot continuation capture only implemented for AArch64 and x86-64")
        }
    }

    /// Clone a multi-shot continuation for reuse
    ///
    /// This allows a continuation to be resumed multiple times by creating
    /// independent copies of the captured state.
    ///
    /// # Arguments
    /// * `cont_ptr` - Pointer to NativeContinuation (must be multi-shot)
    ///
    /// # Returns
    /// Pointer to cloned NativeContinuation
    ///
    /// # Safety
    /// The original continuation pointer remains valid after cloning
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __sounio_clone_continuation(
        cont_ptr: *const NativeContinuation,
    ) -> *mut NativeContinuation {
        unsafe {
            let cont = &*cont_ptr;
            if cont.is_one_shot {
                panic!("Cannot clone one-shot continuation");
            }
            Box::into_raw(Box::new(cont.clone()))
        }
    }

    /// Resume a multi-shot continuation without consuming it
    ///
    /// Unlike one-shot resumption, this preserves the continuation for future resumes.
    ///
    /// # Arguments
    /// * `cont_ptr` - Pointer to NativeContinuation (not consumed)
    /// * `value` - Value to resume with (in x0/rax)
    ///
    /// # Safety
    /// This function never returns - it restores the saved execution context.
    /// The continuation pointer remains valid after resumption.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn __sounio_resume_continuation_multi_shot(
        cont_ptr: *const NativeContinuation,
        _value: u64,
    ) -> ! {
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        unsafe {
            let cont = &*cont_ptr;
            if cont.is_one_shot {
                panic!("Cannot multi-shot resume one-shot continuation");
            }
            // Clone the continuation to preserve the original
            let mut cont_clone = cont.clone();
            cont_clone.resume(_value)
        }

        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            let _ = cont_ptr;
            panic!("Multi-shot continuation resumption only implemented for AArch64 and x86-64")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Abi;
    use crate::hlir::builder::FunctionBuilder;
    use crate::hlir::{FunctionId, HlirConstant};

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
        let mut builder = FunctionBuilder::new(FunctionId(0), "test_func", HlirType::I32);

        // Add entry block
        let entry = builder.create_block("entry");
        builder.switch_to_block(entry);

        // Perform IO effect
        let _print_result = builder.build_perform_effect("IO", "println", vec![], HlirType::Void);

        // Return constant
        let const_val = builder.build_const(HlirConstant::Int(42, HlirType::I32), HlirType::I32);
        builder.build_return(Some(const_val));

        let mut func = builder.build();
        // Manually add effects since they're not set through builder
        func.effects.push("IO".to_string());

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

        let mut builder = FunctionBuilder::new(FunctionId(0), "with_effects", HlirType::I32);
        let entry = builder.create_block("entry");
        builder.switch_to_block(entry);
        let _effect_result = builder.build_perform_effect("IO", "println", vec![], HlirType::Void);
        let val = builder.build_const(HlirConstant::Int(42, HlirType::I32), HlirType::I32);
        builder.build_return(Some(val));

        let mut func = builder.build();
        // Manually add effects since they're not set through builder
        func.effects.push("IO".to_string());

        module.functions.push(func);

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

        let mut builder = FunctionBuilder::new(FunctionId(0), "pure_func", HlirType::I32);
        let entry = builder.create_block("entry");
        builder.switch_to_block(entry);
        let val = builder.build_const(HlirConstant::Int(42, HlirType::I32), HlirType::I32);
        builder.build_return(Some(val));

        module.functions.push(builder.build());

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
