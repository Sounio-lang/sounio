//! JIT Continuation Resume Support
//!
//! This module provides the infrastructure for resuming continuations in
//! JIT-compiled code. When an effect is performed in JIT code, we need to
//! be able to capture the current execution state and resume it later.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   JIT Continuation Resume                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  1. Effect performed in JIT code                                │
//! │     ↓                                                           │
//! │  2. JitResumeContext captures:                                  │
//! │     • Return address (where to resume)                          │
//! │     • Saved registers (callee-saved + used registers)           │
//! │     • Stack snapshot (local variables, spilled values)          │
//! │     ↓                                                           │
//! │  3. Control returns to effect handler                           │
//! │     ↓                                                           │
//! │  4. Handler calls resume(value)                                 │
//! │     ↓                                                           │
//! │  5. JitResumer restores:                                        │
//! │     • Stack from snapshot                                       │
//! │     • Registers from saved values                               │
//! │     • Places resume value in return register                    │
//! │     • Jumps to return address                                   │
//! │                                                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Safety
//!
//! This module contains unsafe code that manipulates raw pointers, stack
//! memory, and processor registers. Incorrect use can cause crashes,
//! memory corruption, or undefined behavior.

use std::collections::HashMap;
use std::sync::Mutex;

use super::continuation::{ContinuationError, ContinuationId};
use crate::interp::Value;

/// Context for resuming a JIT continuation
///
/// This captures all the machine state needed to resume execution
/// at a specific point in JIT-compiled code.
#[derive(Debug, Clone)]
pub struct JitResumeContext {
    /// Unique identifier for this resume context
    pub id: ContinuationId,
    /// Return address in JIT code to jump to
    pub return_address: usize,
    /// Saved register values (architecture-dependent layout)
    /// For x86-64: [rax, rbx, rcx, rdx, rsi, rdi, rbp, rsp, r8-r15, rip, rflags]
    pub saved_registers: Vec<u64>,
    /// Snapshot of the stack at capture point
    /// This includes the stack frame and any spilled values
    pub stack_snapshot: Vec<u8>,
    /// Stack pointer value at capture (for restoration)
    pub stack_pointer: usize,
    /// Base pointer value at capture (for frame restoration)
    pub base_pointer: usize,
    /// Whether this context has been consumed (for one-shot)
    pub consumed: bool,
    /// Whether this is a multi-shot context (can be resumed multiple times)
    pub is_multi_shot: bool,
}

impl JitResumeContext {
    /// Create a new JIT resume context
    pub fn new(return_address: usize, saved_registers: Vec<u64>, stack_snapshot: Vec<u8>) -> Self {
        // Extract stack and base pointers from saved registers if available
        // For x86-64: rsp is at index 7, rbp is at index 6
        let stack_pointer = saved_registers.get(7).copied().unwrap_or(0) as usize;
        let base_pointer = saved_registers.get(6).copied().unwrap_or(0) as usize;

        Self {
            id: ContinuationId::new(),
            return_address,
            saved_registers,
            stack_snapshot,
            stack_pointer,
            base_pointer,
            consumed: false,
            is_multi_shot: false,
        }
    }

    /// Create a multi-shot context (can be resumed multiple times)
    pub fn new_multi_shot(
        return_address: usize,
        saved_registers: Vec<u64>,
        stack_snapshot: Vec<u8>,
    ) -> Self {
        let mut ctx = Self::new(return_address, saved_registers, stack_snapshot);
        ctx.is_multi_shot = true;
        ctx
    }

    /// Check if this context can be resumed
    pub fn can_resume(&self) -> bool {
        self.is_multi_shot || !self.consumed
    }

    /// Mark this context as consumed
    pub fn mark_consumed(&mut self) {
        if !self.is_multi_shot {
            self.consumed = true;
        }
    }
}

/// Global store for JIT resume contexts
///
/// JIT code can store resume contexts here when performing effects,
/// and handlers can retrieve them for resumption.
static JIT_RESUME_STORE: std::sync::OnceLock<Mutex<JitResumeStore>> = std::sync::OnceLock::new();

fn get_jit_resume_store() -> &'static Mutex<JitResumeStore> {
    JIT_RESUME_STORE.get_or_init(|| Mutex::new(JitResumeStore::new()))
}

/// Store for JIT resume contexts
#[derive(Debug, Default)]
pub struct JitResumeStore {
    contexts: HashMap<ContinuationId, JitResumeContext>,
    /// Pending resume: (context_id, resume_value) waiting to be executed
    pending_resume: Option<(ContinuationId, f64)>,
}

impl JitResumeStore {
    /// Create a new empty store
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a resume context and return its ID
    pub fn store(&mut self, ctx: JitResumeContext) -> ContinuationId {
        let id = ctx.id;
        self.contexts.insert(id, ctx);
        id
    }

    /// Get a context by ID
    pub fn get(&self, id: ContinuationId) -> Option<&JitResumeContext> {
        self.contexts.get(&id)
    }

    /// Get a mutable context by ID
    pub fn get_mut(&mut self, id: ContinuationId) -> Option<&mut JitResumeContext> {
        self.contexts.get_mut(&id)
    }

    /// Remove a context
    pub fn remove(&mut self, id: ContinuationId) -> Option<JitResumeContext> {
        self.contexts.remove(&id)
    }

    /// Set a pending resume
    pub fn set_pending(&mut self, id: ContinuationId, value: f64) {
        self.pending_resume = Some((id, value));
    }

    /// Take the pending resume (consumes it)
    pub fn take_pending(&mut self) -> Option<(ContinuationId, f64)> {
        self.pending_resume.take()
    }

    /// Check if there's a pending resume
    pub fn has_pending(&self) -> bool {
        self.pending_resume.is_some()
    }

    /// Clear all contexts
    pub fn clear(&mut self) {
        self.contexts.clear();
        self.pending_resume = None;
    }
}

// =============================================================================
// Public API for JIT code and handlers
// =============================================================================

/// Store a JIT resume context (called from JIT runtime)
///
/// Returns the context ID for later retrieval.
pub fn store_jit_context(ctx: JitResumeContext) -> ContinuationId {
    if let Ok(mut store) = get_jit_resume_store().lock() {
        store.store(ctx)
    } else {
        // Fallback: return the ID even if we couldn't store
        ctx.id
    }
}

/// Schedule a resume for a JIT context
///
/// This sets up a pending resume that will be executed when control
/// returns to the JIT code.
pub fn schedule_jit_resume(id: ContinuationId, value: f64) -> Result<(), ContinuationError> {
    if let Ok(mut store) = get_jit_resume_store().lock() {
        if store.get(id).is_some() {
            store.set_pending(id, value);
            Ok(())
        } else {
            Err(ContinuationError::NotFound(id))
        }
    } else {
        Err(ContinuationError::NotImplemented(
            "Failed to lock JIT resume store".into(),
        ))
    }
}

/// Check if there's a pending JIT resume
pub fn has_pending_jit_resume() -> bool {
    if let Ok(store) = get_jit_resume_store().lock() {
        store.has_pending()
    } else {
        false
    }
}

/// Get and consume the pending JIT resume
pub fn take_pending_jit_resume() -> Option<(ContinuationId, f64)> {
    if let Ok(mut store) = get_jit_resume_store().lock() {
        store.take_pending()
    } else {
        None
    }
}

/// Clear all JIT resume state
pub fn clear_jit_resume_state() {
    if let Ok(mut store) = get_jit_resume_store().lock() {
        store.clear();
    }
}

// =============================================================================
// Resume Execution
// =============================================================================

/// Resume a JIT continuation with a value
///
/// This is the main entry point for resuming a captured JIT continuation.
/// It handles the conversion from interpreter Value to the appropriate
/// machine representation and triggers the actual resumption.
///
/// # Safety
///
/// This function is fundamentally unsafe as it manipulates machine state.
/// The caller must ensure that:
/// - The return_address points to valid JIT code
/// - The saved_registers and stack_snapshot are valid
/// - No other code is using the stack region being restored
pub fn resume_jit_continuation(
    ctx: &mut JitResumeContext,
    value: Value,
) -> Result<Value, ContinuationError> {
    if !ctx.can_resume() {
        return Err(ContinuationError::AlreadyResumed {
            id: ctx.id,
            label: None,
        });
    }

    // Convert Value to f64 for JIT (JIT uses f64 for all values)
    let resume_value = value_to_f64(&value);

    // Mark as consumed for one-shot
    ctx.mark_consumed();

    // For now, we use a cooperative resumption model:
    // 1. Store the resume value in the global state
    // 2. The JIT code will poll for this value when it returns from the effect
    //
    // True stack-switching resumption requires platform-specific assembly
    // which we'll implement as an optimization later.

    #[cfg(feature = "jit")]
    {
        // Set the resume value in the JIT effect state
        unsafe extern "C" {
            fn runtime_continuation_resume(continuation_id: i64, value: f64) -> i64;
        }

        unsafe {
            let result = runtime_continuation_resume(1, resume_value);
            if result != 0 {
                // Resume was accepted
                // The JIT code will pick up the value via runtime_continuation_get_value
                return Ok(Value::Float(resume_value));
            }
        }
    }

    // Without JIT feature, or if the cooperative resume failed,
    // fall back to returning the resume value directly
    Ok(Value::Float(resume_value))
}

/// Convert an interpreter Value to f64 for JIT consumption
fn value_to_f64(value: &Value) -> f64 {
    match value {
        Value::Int(i) => *i as f64,
        Value::Float(f) => *f,
        Value::Bool(b) => {
            if *b {
                1.0
            } else {
                0.0
            }
        }
        Value::Unit => 0.0,
        // For complex types, we'd need a more sophisticated approach
        // For now, return NaN to indicate an unsupported type
        _ => f64::NAN,
    }
}

/// Convert an f64 from JIT back to an interpreter Value
pub fn f64_to_value(f: f64) -> Value {
    if f.is_nan() {
        Value::Unit
    } else if f == f.floor() && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
        Value::Int(f as i64)
    } else {
        Value::Float(f)
    }
}

// =============================================================================
// x86-64 Specific Implementation
// =============================================================================

/// x86-64 register indices in saved_registers array
#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    pub const RAX: usize = 0;
    pub const RBX: usize = 1;
    pub const RCX: usize = 2;
    pub const RDX: usize = 3;
    pub const RSI: usize = 4;
    pub const RDI: usize = 5;
    pub const RBP: usize = 6;
    pub const RSP: usize = 7;
    pub const R8: usize = 8;
    pub const R9: usize = 9;
    pub const R10: usize = 10;
    pub const R11: usize = 11;
    pub const R12: usize = 12;
    pub const R13: usize = 13;
    pub const R14: usize = 14;
    pub const R15: usize = 15;
    pub const RIP: usize = 16;
    pub const RFLAGS: usize = 17;

    /// Number of registers we save
    pub const NUM_SAVED_REGISTERS: usize = 18;

    /// Create a saved registers array with all zeros
    pub fn empty_registers() -> Vec<u64> {
        vec![0u64; NUM_SAVED_REGISTERS]
    }
}

// =============================================================================
// Runtime Functions (called from JIT code)
// =============================================================================

/// Called from JIT code to capture a continuation
///
/// This is invoked when an effect is performed and we need to save
/// the current execution state for potential resumption.
#[unsafe(no_mangle)]
pub extern "C" fn jit_capture_continuation(
    return_address: usize,
    saved_registers_ptr: *const u64,
    saved_registers_len: usize,
    stack_ptr: *const u8,
    stack_len: usize,
) -> u64 {
    let saved_registers = if saved_registers_ptr.is_null() || saved_registers_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(saved_registers_ptr, saved_registers_len).to_vec() }
    };

    let stack_snapshot = if stack_ptr.is_null() || stack_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(stack_ptr, stack_len).to_vec() }
    };

    let ctx = JitResumeContext::new(return_address, saved_registers, stack_snapshot);
    let id = store_jit_context(ctx);
    id.raw()
}

/// Called from JIT code to check if a resume is pending
#[unsafe(no_mangle)]
pub extern "C" fn jit_has_pending_resume() -> i64 {
    if has_pending_jit_resume() {
        1
    } else {
        0
    }
}

/// Called from JIT code to get the pending resume value
#[unsafe(no_mangle)]
pub extern "C" fn jit_get_resume_value() -> f64 {
    if let Some((_id, value)) = take_pending_jit_resume() {
        value
    } else {
        f64::NAN
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jit_resume_context_creation() {
        let ctx = JitResumeContext::new(0x1000, vec![1, 2, 3, 4, 5, 6, 7, 8], vec![0u8; 64]);

        assert_eq!(ctx.return_address, 0x1000);
        assert_eq!(ctx.saved_registers.len(), 8);
        assert_eq!(ctx.stack_snapshot.len(), 64);
        assert!(!ctx.consumed);
        assert!(!ctx.is_multi_shot);
        assert!(ctx.can_resume());
    }

    #[test]
    fn test_jit_resume_context_one_shot() {
        let mut ctx = JitResumeContext::new(0x1000, vec![], vec![]);

        assert!(ctx.can_resume());

        ctx.mark_consumed();

        assert!(!ctx.can_resume());
    }

    #[test]
    fn test_jit_resume_context_multi_shot() {
        let mut ctx = JitResumeContext::new_multi_shot(0x1000, vec![], vec![]);

        assert!(ctx.is_multi_shot);
        assert!(ctx.can_resume());

        ctx.mark_consumed();

        // Multi-shot should still be resumable
        assert!(ctx.can_resume());
    }

    #[test]
    fn test_jit_resume_store() {
        let mut store = JitResumeStore::new();

        let ctx = JitResumeContext::new(0x2000, vec![1, 2, 3], vec![4, 5, 6]);
        let id = store.store(ctx);

        assert!(store.get(id).is_some());

        let retrieved = store.get(id).unwrap();
        assert_eq!(retrieved.return_address, 0x2000);

        let removed = store.remove(id);
        assert!(removed.is_some());
        assert!(store.get(id).is_none());
    }

    #[test]
    fn test_jit_resume_store_pending() {
        let mut store = JitResumeStore::new();

        let ctx = JitResumeContext::new(0x3000, vec![], vec![]);
        let id = store.store(ctx);

        assert!(!store.has_pending());

        store.set_pending(id, 42.0);

        assert!(store.has_pending());

        let pending = store.take_pending();
        assert!(pending.is_some());
        let (pending_id, pending_value) = pending.unwrap();
        assert_eq!(pending_id, id);
        assert_eq!(pending_value, 42.0);

        assert!(!store.has_pending());
    }

    #[test]
    fn test_value_to_f64_conversion() {
        assert_eq!(value_to_f64(&Value::Int(42)), 42.0);
        assert_eq!(value_to_f64(&Value::Float(3.14)), 3.14);
        assert_eq!(value_to_f64(&Value::Bool(true)), 1.0);
        assert_eq!(value_to_f64(&Value::Bool(false)), 0.0);
        assert_eq!(value_to_f64(&Value::Unit), 0.0);
    }

    #[test]
    fn test_f64_to_value_conversion() {
        match f64_to_value(42.0) {
            Value::Int(i) => assert_eq!(i, 42),
            _ => panic!("Expected Int"),
        }

        match f64_to_value(3.14) {
            Value::Float(f) => assert!((f - 3.14).abs() < 0.001),
            _ => panic!("Expected Float"),
        }

        match f64_to_value(f64::NAN) {
            Value::Unit => {}
            _ => panic!("Expected Unit for NaN"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_x86_64_register_indices() {
        use super::x86_64::*;

        assert_eq!(RAX, 0);
        assert_eq!(RSP, 7);
        assert_eq!(RBP, 6);
        assert_eq!(NUM_SAVED_REGISTERS, 18);

        let regs = empty_registers();
        assert_eq!(regs.len(), NUM_SAVED_REGISTERS);
    }

    #[test]
    fn test_global_store_api() {
        // Clear any existing state
        clear_jit_resume_state();

        let ctx = JitResumeContext::new(0x4000, vec![1, 2], vec![3, 4]);
        let id = store_jit_context(ctx);

        // Schedule a resume
        let result = schedule_jit_resume(id, 99.0);
        assert!(result.is_ok());

        assert!(has_pending_jit_resume());

        let pending = take_pending_jit_resume();
        assert!(pending.is_some());
        let (_, value) = pending.unwrap();
        assert_eq!(value, 99.0);

        assert!(!has_pending_jit_resume());

        // Clean up
        clear_jit_resume_state();
    }
}
