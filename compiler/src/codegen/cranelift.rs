//! Cranelift JIT backend
//!
//! This module provides fast JIT compilation using Cranelift.
//! Cranelift is optimized for fast compilation rather than peak runtime performance,
//! making it ideal for development and scripting use cases.
//!
//! # Effect Dispatch
//!
//! The JIT supports algebraic effect handling through two mechanisms:
//!
//! 1. **Direct runtime functions**: Specialized functions like `runtime_prob_sample`,
//!    `runtime_io_print`, etc. for common effects.
//!
//! 2. **Registry-based dispatch**: A `HandlerRegistry` can be configured to handle
//!    effects through the `HandlerCapability` trait system.
//!
//! ```ignore
//! use sounio::codegen::cranelift::{configure_jit_registry, CraneliftJit};
//! use sounio::effects::handlers::HandlerRegistry;
//!
//! // Configure with all handlers
//! configure_jit_registry(HandlerRegistry::with_defaults());
//!
//! let jit = CraneliftJit::new();
//! let result = jit.compile_and_run(&module)?;
//! ```

use crate::hlir::HlirModule;
#[cfg(feature = "jit")]
use crate::runtime::handler_stack;
#[cfg(feature = "jit")]
use std::io::Write;
#[cfg(feature = "jit")]
use std::sync::Arc;

// ==================== Native Runtime Functions ====================
// These are called from JIT-compiled code via FFI

/// Print an i64 value
#[cfg(feature = "jit")]
extern "C" fn runtime_print_i64(val: i64) {
    print!("{}", val);
    let _ = std::io::stdout().flush();
}

/// Print an f64 value
#[cfg(feature = "jit")]
extern "C" fn runtime_print_f64(val: f64) {
    print!("{}", val);
    let _ = std::io::stdout().flush();
}

/// Print a newline
#[cfg(feature = "jit")]
extern "C" fn runtime_print_newline() {
    println!();
}

/// Print a string (pointer + length)
#[cfg(feature = "jit")]
extern "C" fn runtime_print_str(ptr: *const u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        if let Ok(s) = std::str::from_utf8(slice) {
            print!("{}", s);
            let _ = std::io::stdout().flush();
        }
    }
}

/// Print a null-terminated C string
#[cfg(feature = "jit")]
extern "C" fn runtime_print_cstr(ptr: *const u8) {
    if !ptr.is_null() {
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr as *const std::ffi::c_char) };
        if let Ok(s) = cstr.to_str() {
            print!("{}", s);
            let _ = std::io::stdout().flush();
        }
    }
}

// Global storage for string constants during JIT execution
#[cfg(feature = "jit")]
use std::sync::Mutex;
#[cfg(feature = "jit")]
static STRING_STORAGE: Mutex<Vec<std::ffi::CString>> = Mutex::new(Vec::new());

/// Print a boolean value
#[cfg(feature = "jit")]
extern "C" fn runtime_print_bool(val: i8) {
    print!("{}", if val != 0 { "true" } else { "false" });
    let _ = std::io::stdout().flush();
}

/// Debug test function - just returns 99 to verify FFI works
#[cfg(feature = "jit")]
extern "C" fn runtime_debug_test() -> i64 {
    99
}

// ==================== Effect Runtime Functions ====================
// These implement algebraic effect operations for JIT-compiled code
// Uses a simplified thread-safe state separate from the interpreter's EffectContext

/// Handler function type for effect operations
/// Takes (effect_name, op_name, args_ptr, args_len) and returns f64
/// Returns NaN to indicate "not handled, use default"
#[cfg(feature = "jit")]
type HandlerFn = extern "C" fn(*const u8, *const u8, *const f64, usize) -> f64;

// ==================== Predefined Handler IDs ====================
// Handler IDs are organized by effect type in ranges of 10

// Prob effect handlers (10-19)
#[cfg(feature = "jit")]
const HANDLER_PROB_DETERMINISTIC: u32 = 10; // Returns distribution parameter
#[cfg(feature = "jit")]
const HANDLER_PROB_IMPORTANCE: u32 = 11; // Importance sampling
#[cfg(feature = "jit")]
const HANDLER_PROB_ENUMERATE: u32 = 12; // Enumeration for discrete

// Causal effect handlers (20-29)
#[cfg(feature = "jit")]
const HANDLER_CAUSAL_SCM: u32 = 20; // Structural causal model
#[cfg(feature = "jit")]
const HANDLER_CAUSAL_BACKDOOR: u32 = 21; // Backdoor adjustment
#[cfg(feature = "jit")]
const HANDLER_CAUSAL_FRONTDOOR: u32 = 22; // Front-door formula

// Grad effect handlers (30-39)
#[cfg(feature = "jit")]
const HANDLER_GRAD_FORWARD: u32 = 30; // Forward-mode autodiff
#[cfg(feature = "jit")]
const HANDLER_GRAD_REVERSE: u32 = 31; // Reverse-mode autodiff
#[cfg(feature = "jit")]
const HANDLER_GRAD_NUMERIC: u32 = 32; // Finite differences

/// A registered effect handler
#[cfg(feature = "jit")]
#[derive(Clone)]
struct JitHandler {
    /// Effect name this handler handles (e.g., "Prob", "IO")
    effect: String,
    /// Handler name for debugging
    name: String,
    /// Handler function pointer
    handler_fn: Option<HandlerFn>,
    /// Handler ID for predefined handlers
    handler_id: u32,
}

#[cfg(feature = "jit")]
impl JitHandler {
    fn new(effect: &str, name: &str, handler_id: u32) -> Self {
        Self {
            effect: effect.to_string(),
            name: name.to_string(),
            handler_fn: None,
            handler_id,
        }
    }

    fn with_fn(effect: &str, name: &str, handler_fn: HandlerFn) -> Self {
        Self {
            effect: effect.to_string(),
            name: name.to_string(),
            handler_fn: Some(handler_fn),
            handler_id: 0,
        }
    }
}

#[cfg(feature = "jit")]
struct JitEffectState {
    rng: crate::runtime::prob::Rng,
    log_prob: f64,
    interventions: Vec<(String, f64)>,
    observations: Vec<(String, f64)>,
    /// Named mutable state store for Mut effect
    mutable_state: std::collections::HashMap<String, f64>,
    /// Tracked allocations for Alloc effect: ptr -> size
    allocations: std::collections::HashMap<usize, usize>,
    /// Total bytes currently allocated
    total_allocated: usize,
    /// Handler stack for continuation-based effect handling
    handler_stack: Vec<JitHandler>,
    /// Continuation state: stores resume data when an effect is performed
    continuation_data: Option<ContinuationData>,
    /// Optional handler registry for registry-based dispatch
    registry: Option<Arc<crate::effects::handlers::HandlerRegistry>>,
}

/// Continuation data for effect handling
#[cfg(feature = "jit")]
#[derive(Clone)]
struct ContinuationData {
    /// The effect that caused the suspension
    effect: String,
    /// The operation name
    operation: String,
    /// Arguments to the operation
    args: Vec<f64>,
    /// Whether the continuation has been resumed
    resumed: bool,
    /// Resume value (set by handler)
    resume_value: f64,
}

#[cfg(feature = "jit")]
impl JitEffectState {
    fn new() -> Self {
        Self {
            rng: crate::runtime::prob::Rng::new(42),
            log_prob: 0.0,
            interventions: Vec::new(),
            observations: Vec::new(),
            mutable_state: std::collections::HashMap::new(),
            allocations: std::collections::HashMap::new(),
            total_allocated: 0,
            handler_stack: Vec::new(),
            continuation_data: None,
            registry: None,
        }
    }

    fn reset(&mut self) {
        self.log_prob = 0.0;
        self.interventions.clear();
        self.observations.clear();
        self.mutable_state.clear();
        self.handler_stack.clear();
        self.continuation_data = None;
        // Note: We don't clear allocations here to avoid memory leaks
        // Use Alloc.clear() explicitly to free all allocations
        // Note: We don't clear the registry - it persists across runs
    }

    /// Set the handler registry for registry-based dispatch
    fn set_registry(&mut self, registry: crate::effects::handlers::HandlerRegistry) {
        self.registry = Some(Arc::new(registry));
    }

    /// Dispatch an effect operation through the registry
    /// Returns Some(value) if handled, None if not
    fn dispatch_via_registry(&self, effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        let registry = self.registry.as_ref()?;
        let handler = registry.get(effect)?;

        // Convert f64 args to Value args
        let value_args: Vec<crate::interp::Value> = args
            .iter()
            .map(|&f| crate::interp::Value::Float(f))
            .collect();

        // Create a fresh handler state for this call (since we can't store it in the global state)
        let mut capability_state = crate::effects::handler_capability::HandlerState::new();

        // Create a continuation
        let continuation = crate::effects::handler_capability::Continuation::new();

        // Call the handler
        let result = handler.handle(op, &value_args, continuation, &mut capability_state);

        // Convert result back to f64
        match result {
            crate::effects::handler_capability::HandlerResult::Resume(v)
            | crate::effects::handler_capability::HandlerResult::Return(v) => {
                match v {
                    crate::interp::Value::Float(f) => Some(f),
                    crate::interp::Value::Int(i) => Some(i as f64),
                    crate::interp::Value::Bool(b) => Some(if b { 1.0 } else { 0.0 }),
                    crate::interp::Value::Unit => Some(0.0),
                    _ => Some(0.0), // Other types default to 0.0
                }
            }
            crate::effects::handler_capability::HandlerResult::Abort(_) => None,
            crate::effects::handler_capability::HandlerResult::Suspend(_) => None,
        }
    }

    /// Push a handler onto the stack
    fn push_handler(&mut self, handler: JitHandler) {
        self.handler_stack.push(handler);
    }

    /// Pop a handler from the stack
    fn pop_handler(&mut self) -> Option<JitHandler> {
        self.handler_stack.pop()
    }

    /// Find a handler for the given effect
    fn find_handler(&self, effect: &str) -> Option<&JitHandler> {
        // Search from top of stack (most recent) to bottom
        self.handler_stack.iter().rev().find(|h| h.effect == effect)
    }

    /// Dispatch an effect operation to the handler stack
    /// Returns Some(value) if a handler handled the effect, None otherwise
    fn dispatch_to_handler(&self, effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        if let Some(handler) = self.find_handler(effect) {
            if let Some(handler_fn) = handler.handler_fn {
                // Call the handler function
                let effect_ptr = effect.as_ptr();
                let op_ptr = op.as_ptr();
                let args_ptr = args.as_ptr();
                let result = handler_fn(effect_ptr, op_ptr, args_ptr, args.len());
                // NaN means "not handled"
                if !result.is_nan() {
                    return Some(result);
                }
            } else {
                // Use predefined handler based on handler_id
                return self.dispatch_predefined_handler(handler.handler_id, effect, op, args);
            }
        }
        None
    }

    /// Dispatch an effect operation, trying handler stack first, then registry
    fn dispatch_effect(&self, effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        // First try the handler stack
        if let Some(result) = self.dispatch_to_handler(effect, op, args) {
            return Some(result);
        }

        // Then try the registry
        self.dispatch_via_registry(effect, op, args)
    }

    /// Dispatch to a predefined handler by ID
    fn dispatch_predefined_handler(&self, handler_id: u32, _effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        match handler_id {
            // Handler ID 1: Constant sampler (always returns first arg or 0)
            1 => {
                if op == "sample" {
                    Some(args.first().copied().unwrap_or(0.0))
                } else {
                    None
                }
            }
            // Handler ID 2: Logging handler (logs and passes through)
            2 => {
                eprintln!("[Effect] {}({:?})", op, args);
                None // Pass through to default
            }
            // Handler ID 3: Deterministic sampler (returns mean)
            3 => {
                if op == "sample" {
                    Some(args.first().copied().unwrap_or(0.0))
                } else {
                    None
                }
            }
            // Prob effect handlers (10-19)
            10..=19 => dispatch_prob_effect(handler_id, op, args),
            // Causal effect handlers (20-29)
            20..=29 => dispatch_causal_effect(handler_id, op, args),
            // Grad effect handlers (30-39)
            30..=39 => dispatch_grad_effect(handler_id, op, args),
            _ => None,
        }
    }
}

// ==================== Prob Effect Dispatch ====================
/// Dispatch Prob effect operations (sample, observe, factor)
#[cfg(feature = "jit")]
fn dispatch_prob_effect(handler_id: u32, op: &str, args: &[f64]) -> Option<f64> {
    match handler_id {
        // Deterministic sampler - returns distribution parameter
        HANDLER_PROB_DETERMINISTIC => {
            match op {
                "sample" => Some(args.first().copied().unwrap_or(0.5)),
                _ if op.starts_with("observe_") => Some(0.0), // No log-likelihood contribution
                "factor" | "score" => Some(args.first().copied().unwrap_or(0.0)),
                _ => None,
            }
        }
        // Importance sampling handler
        HANDLER_PROB_IMPORTANCE => {
            match op {
                "sample" => {
                    // Return proposal value (first arg)
                    Some(args.first().copied().unwrap_or(0.5))
                }
                _ if op.starts_with("observe_") => {
                    // Compute log-likelihood: log(N(value; 0, 1)) = -0.5 * value^2
                    let value = args.first().copied().unwrap_or(0.0);
                    Some(-0.5 * value * value)
                }
                "factor" | "score" => Some(args.first().copied().unwrap_or(0.0)),
                _ => None,
            }
        }
        // Enumeration handler for discrete distributions
        HANDLER_PROB_ENUMERATE => {
            match op {
                "sample" => {
                    // Return first support point (first arg)
                    Some(args.first().copied().unwrap_or(0.0))
                }
                _ if op.starts_with("observe_") => Some(0.0),
                _ => None,
            }
        }
        _ => None,
    }
}

// ==================== Causal Effect Dispatch ====================
/// Dispatch Causal effect operations (do, counterfactual, query, etc.)
#[cfg(feature = "jit")]
fn dispatch_causal_effect(handler_id: u32, op: &str, args: &[f64]) -> Option<f64> {
    // Common causal operations regardless of handler
    let result = match op {
        // Do intervention: do(X = value) returns the intervention value
        _ if op.starts_with("do_") => {
            Some(args.first().copied().unwrap_or(0.0))
        }
        // Counterfactual: counterfactual(factual, intervention, outcome)
        // Default: outcome + intervention - factual (linear approximation)
        "counterfactual" => {
            let factual = args.first().copied().unwrap_or(0.0);
            let intervention = args.get(1).copied().unwrap_or(0.0);
            let outcome = args.get(2).copied().unwrap_or(0.0);
            Some(outcome + intervention - factual)
        }
        // Query: returns target value (placeholder for do-calculus)
        "query" => Some(args.first().copied().unwrap_or(0.0)),
        // Average treatment effect
        "ate" => {
            // ATE = E[Y(1)] - E[Y(0)]
            let treated = args.first().copied().unwrap_or(0.0);
            let control = args.get(1).copied().unwrap_or(0.0);
            Some(treated - control)
        }
        // Conditional average treatment effect
        "cate" => {
            let treated = args.first().copied().unwrap_or(0.0);
            let control = args.get(1).copied().unwrap_or(0.0);
            Some(treated - control)
        }
        // Difference-in-differences: (post_treat - pre_treat) - (post_ctrl - pre_ctrl)
        "did" => {
            let post_treat = args.first().copied().unwrap_or(0.0);
            let pre_treat = args.get(1).copied().unwrap_or(0.0);
            let post_ctrl = args.get(2).copied().unwrap_or(0.0);
            let pre_ctrl = args.get(3).copied().unwrap_or(0.0);
            Some((post_treat - pre_treat) - (post_ctrl - pre_ctrl))
        }
        // Instrumental variable estimation
        "iv" => {
            // IV = Cov(Y, Z) / Cov(X, Z)
            // Simplified: return ratio of first two args
            let cov_yz = args.first().copied().unwrap_or(0.0);
            let cov_xz = args.get(1).copied().unwrap_or(1.0);
            if cov_xz.abs() > 1e-10 {
                Some(cov_yz / cov_xz)
            } else {
                Some(0.0)
            }
        }
        // Regression discontinuity design
        "rdd" => {
            // RDD: returns treatment effect at cutoff
            Some(args.first().copied().unwrap_or(0.0))
        }
        _ => None,
    };

    // Handler-specific behavior (can override defaults)
    match handler_id {
        HANDLER_CAUSAL_SCM => result, // SCM uses default behavior
        HANDLER_CAUSAL_BACKDOOR => {
            // Backdoor adjustment - same as default for now
            result
        }
        HANDLER_CAUSAL_FRONTDOOR => {
            // Front-door formula - same as default for now
            result
        }
        _ => result,
    }
}

// ==================== Grad Effect Dispatch ====================
/// Dispatch Grad effect operations (grad, jacobian, hessian, jvp, vjp)
#[cfg(feature = "jit")]
fn dispatch_grad_effect(handler_id: u32, op: &str, args: &[f64]) -> Option<f64> {
    match handler_id {
        // Forward-mode autodiff (dual numbers)
        HANDLER_GRAD_FORWARD => {
            match op {
                "grad" | "jacobian" | "hessian" => {
                    // Forward-mode is the default - let it pass through to builtins
                    None
                }
                "jvp" => {
                    // Jacobian-vector product: J * v
                    // args: [f_ptr, x_ptr, v_ptr]
                    // For now, return placeholder
                    Some(args.first().copied().unwrap_or(0.0))
                }
                "vjp" => {
                    // Vector-Jacobian product: v^T * J (reverse mode preferred)
                    // Forward mode can still compute it, just less efficiently
                    Some(args.first().copied().unwrap_or(0.0))
                }
                "forward" => Some(1.0), // Signal forward mode is active
                "reverse" => Some(0.0), // Not using reverse mode
                _ => None,
            }
        }
        // Reverse-mode autodiff (backpropagation)
        HANDLER_GRAD_REVERSE => {
            match op {
                "grad" | "jacobian" | "hessian" => {
                    // Reverse mode - let it pass through (future: tape-based impl)
                    None
                }
                "vjp" => {
                    // VJP is efficient in reverse mode
                    Some(args.first().copied().unwrap_or(0.0))
                }
                "jvp" => {
                    // JVP can be computed but less efficiently
                    Some(args.first().copied().unwrap_or(0.0))
                }
                "forward" => Some(0.0), // Not using forward mode
                "reverse" => Some(1.0), // Signal reverse mode is active
                _ => None,
            }
        }
        // Numeric differentiation (finite differences)
        HANDLER_GRAD_NUMERIC => {
            match op {
                "grad" => {
                    // Numeric gradient via finite differences
                    // args: [f_ptr, x, epsilon]
                    // Placeholder - actual implementation would evaluate f(x+h) - f(x-h) / 2h
                    Some(args.get(2).copied().unwrap_or(1e-7))
                }
                "jacobian" | "hessian" => {
                    // Numeric approximation
                    Some(0.0)
                }
                "forward" | "reverse" => Some(0.0), // Neither - using numeric
                "numeric" => Some(1.0), // Signal numeric mode is active
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(feature = "jit")]
use std::sync::OnceLock;
#[cfg(feature = "jit")]
static JIT_EFFECT_STATE: OnceLock<Mutex<JitEffectState>> = OnceLock::new();

#[cfg(feature = "jit")]
fn get_jit_effect_state() -> &'static Mutex<JitEffectState> {
    JIT_EFFECT_STATE.get_or_init(|| Mutex::new(JitEffectState::new()))
}

// ==================== Public JIT Effect Configuration API ====================

/// Configure the JIT effect system with a handler registry.
///
/// This allows JIT-compiled code to dispatch effects through the
/// `HandlerCapability` trait system, supporting all registered handlers.
///
/// # Example
///
/// ```ignore
/// use sounio::codegen::cranelift::configure_jit_registry;
/// use sounio::effects::handlers::HandlerRegistry;
///
/// // Configure with all default handlers
/// configure_jit_registry(HandlerRegistry::with_defaults());
///
/// // Or configure with a custom registry
/// let registry = HandlerRegistryBuilder::new()
///     .with_io()
///     .with_mut()
///     .build();
/// configure_jit_registry(registry);
/// ```
#[cfg(feature = "jit")]
pub fn configure_jit_registry(registry: crate::effects::handlers::HandlerRegistry) {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.set_registry(registry);
    }
}

/// Configure the JIT with all default effect handlers.
///
/// This is a convenience function that configures all 12 built-in handlers.
#[cfg(feature = "jit")]
pub fn configure_jit_all_effects() {
    configure_jit_registry(crate::effects::handlers::HandlerRegistry::with_defaults());
}

/// Reset the JIT effect state.
///
/// This clears all mutable state, observations, and handler stack.
/// The registry is preserved.
#[cfg(feature = "jit")]
pub fn reset_jit_effect_state() {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.reset();
    }
}

/// Stub for non-JIT builds
#[cfg(not(feature = "jit"))]
pub fn configure_jit_registry(_registry: crate::effects::handlers::HandlerRegistry) {}

/// Stub for non-JIT builds
#[cfg(not(feature = "jit"))]
pub fn configure_jit_all_effects() {}

/// Stub for non-JIT builds
#[cfg(not(feature = "jit"))]
pub fn reset_jit_effect_state() {}

// ==================== Generic Effect Dispatch Runtime ====================

/// Generic effect dispatch function callable from JIT code.
///
/// This function dispatches to registered handlers via the registry.
/// Parameters:
/// - effect_ptr: Pointer to effect name (null-terminated)
/// - op_ptr: Pointer to operation name (null-terminated)
/// - args_ptr: Pointer to f64 array of arguments
/// - args_len: Number of arguments
///
/// Returns: Result as f64 (NaN if unhandled)
#[cfg(feature = "jit")]
extern "C" fn runtime_effect_dispatch(
    effect_ptr: *const u8,
    op_ptr: *const u8,
    args_ptr: *const f64,
    args_len: usize,
) -> f64 {
    // Safety: These pointers come from JIT code and should be valid
    let effect = if effect_ptr.is_null() {
        return f64::NAN;
    } else {
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(effect_ptr as *const std::ffi::c_char);
            match cstr.to_str() {
                Ok(s) => s,
                Err(_) => return f64::NAN,
            }
        }
    };

    let op = if op_ptr.is_null() {
        return f64::NAN;
    } else {
        unsafe {
            let cstr = std::ffi::CStr::from_ptr(op_ptr as *const std::ffi::c_char);
            match cstr.to_str() {
                Ok(s) => s,
                Err(_) => return f64::NAN,
            }
        }
    };

    let args = if args_ptr.is_null() || args_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(args_ptr, args_len) }
    };

    // Dispatch through the effect state
    if let Ok(state) = get_jit_effect_state().lock() {
        state.dispatch_effect(effect, op, args).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// Dispatch an effect operation by name with f64 arguments.
///
/// This is a higher-level wrapper for runtime_effect_dispatch.
#[cfg(feature = "jit")]
extern "C" fn runtime_dispatch_effect_f64(
    effect_ptr: *const u8,
    effect_len: usize,
    op_ptr: *const u8,
    op_len: usize,
    arg0: f64,
    arg1: f64,
) -> f64 {
    let effect = if effect_ptr.is_null() || effect_len == 0 {
        return f64::NAN;
    } else {
        unsafe {
            let slice = std::slice::from_raw_parts(effect_ptr, effect_len);
            match std::str::from_utf8(slice) {
                Ok(s) => s,
                Err(_) => return f64::NAN,
            }
        }
    };

    let op = if op_ptr.is_null() || op_len == 0 {
        return f64::NAN;
    } else {
        unsafe {
            let slice = std::slice::from_raw_parts(op_ptr, op_len);
            match std::str::from_utf8(slice) {
                Ok(s) => s,
                Err(_) => return f64::NAN,
            }
        }
    };

    let args = [arg0, arg1];

    if let Ok(state) = get_jit_effect_state().lock() {
        state.dispatch_effect(effect, op, &args).unwrap_or(f64::NAN)
    } else {
        f64::NAN
    }
}

/// Sample from a probability distribution (Prob.sample)
/// Takes mean and std, returns sampled f64 value from Normal(mean, std)
#[cfg(feature = "jit")]
extern "C" fn runtime_prob_sample(mean: f64, std: f64) -> f64 {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        // Sample from Normal(mean, std) distribution
        let sample = state.rng.next_normal() * std + mean;
        sample
    } else {
        // Fallback: return mean if lock fails
        mean
    }
}

/// Sample from a uniform distribution
#[cfg(feature = "jit")]
extern "C" fn runtime_prob_sample_uniform(low: f64, high: f64) -> f64 {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        low + (high - low) * state.rng.next_f64()
    } else {
        (low + high) / 2.0
    }
}

/// Sample from a Bernoulli distribution (returns 0 or 1)
#[cfg(feature = "jit")]
extern "C" fn runtime_prob_sample_bernoulli(p: f64) -> i64 {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        if state.rng.next_f64() < p { 1 } else { 0 }
    } else {
        if p >= 0.5 { 1 } else { 0 }
    }
}

/// Observe/condition on a value (Prob.observe)
/// Records observation for probabilistic inference
#[cfg(feature = "jit")]
extern "C" fn runtime_prob_observe(mean: f64, std: f64, observed: f64) -> f64 {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        // Compute log probability of observation under Normal(mean, std)
        let z = (observed - mean) / std;
        let log_prob = -0.5 * z * z - std.ln() - 0.5 * std::f64::consts::TAU.ln();
        state.log_prob += log_prob;
        log_prob
    } else {
        0.0
    }
}

// ==================== MATH RUNTIME FUNCTIONS ====================

/// Compute sine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_sin(x: f64) -> f64 {
    x.sin()
}

/// Compute cosine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_cos(x: f64) -> f64 {
    x.cos()
}

/// Compute exponential (e^x) of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_exp(x: f64) -> f64 {
    x.exp()
}

/// Compute natural logarithm of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_log(x: f64) -> f64 {
    x.ln()
}

/// Compute square root of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Compute tangent of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_tan(x: f64) -> f64 {
    x.tan()
}

/// Compute arctangent of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_atan(x: f64) -> f64 {
    x.atan()
}

/// Compute absolute value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_fabs(x: f64) -> f64 {
    x.abs()
}

/// Compute arcsine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_asin(x: f64) -> f64 {
    x.asin()
}

/// Compute arccosine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_acos(x: f64) -> f64 {
    x.acos()
}

/// Compute hyperbolic sine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_sinh(x: f64) -> f64 {
    x.sinh()
}

/// Compute hyperbolic cosine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_cosh(x: f64) -> f64 {
    x.cosh()
}

/// Compute hyperbolic tangent of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_tanh(x: f64) -> f64 {
    x.tanh()
}

/// Compute inverse hyperbolic sine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_asinh(x: f64) -> f64 {
    x.asinh()
}

/// Compute inverse hyperbolic cosine of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_acosh(x: f64) -> f64 {
    x.acosh()
}

/// Compute inverse hyperbolic tangent of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_atanh(x: f64) -> f64 {
    x.atanh()
}

/// Compute log base 2 of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_log2(x: f64) -> f64 {
    x.log2()
}

/// Compute log base 10 of a value
#[cfg(feature = "jit")]
extern "C" fn runtime_math_log10(x: f64) -> f64 {
    x.log10()
}

/// Compute two-argument arctangent
#[cfg(feature = "jit")]
extern "C" fn runtime_math_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Jacobian computation for forward-mode autodiff
/// func_ptr: function pointer fn(*mut Dual_array) -> *mut Dual_array
/// x_ptr: pointer to input f64 array with layout: [length: i64, data: f64...]
/// m: output dimension
/// n is read automatically from array header at x_ptr[0]
/// Returns pointer to flattened result matrix (m rows × n cols, row-major)
#[cfg(feature = "jit")]
extern "C" fn runtime_jacobian(
    func_ptr: extern "C" fn(i64) -> i64,
    x_ptr: *const i64,  // pointer to array with length header
    m: i64,
) -> i64 {
    if x_ptr.is_null() || m <= 0 {
        return 0;
    }

    // Read n from array header at offset 0
    let n = unsafe { *x_ptr } as usize;
    if n == 0 {
        return 0;
    }
    let m = m as usize;

    // Data starts after the length header (offset 8 bytes = 1 i64)
    let data_ptr = unsafe { x_ptr.add(1) } as *const f64;

    // Allocate result matrix (m × n)
    let result_size = m * n * std::mem::size_of::<f64>();
    let result_ptr = unsafe {
        std::alloc::alloc(std::alloc::Layout::from_size_align(result_size, 8).unwrap())
    } as *mut f64;

    if result_ptr.is_null() {
        return 0;
    }

    // Read input values from data (after length header)
    let x: Vec<f64> = unsafe { std::slice::from_raw_parts(data_ptr, n).to_vec() };

    // For each input dimension j, compute column j of Jacobian
    for j in 0..n {
        // Create dual input array with seed at position j
        // Each dual is 16 bytes: [value: f64, derivative: f64]
        let input_size = n * 16;
        let input_ptr = unsafe {
            std::alloc::alloc(std::alloc::Layout::from_size_align(input_size, 8).unwrap())
        } as *mut f64;

        if input_ptr.is_null() {
            continue;
        }

        // Fill input: value = x[i], derivative = 1 if i==j else 0
        for i in 0..n {
            unsafe {
                *input_ptr.add(i * 2) = x[i];           // value
                *input_ptr.add(i * 2 + 1) = if i == j { 1.0 } else { 0.0 }; // derivative
            }
        }

        // Call function
        let output_ptr = func_ptr(input_ptr as i64) as *const f64;

        // Extract derivatives from output (these form column j)
        if !output_ptr.is_null() {
            for i in 0..m {
                let deriv = unsafe { *output_ptr.add(i * 2 + 1) }; // derivative at position i
                unsafe {
                    *result_ptr.add(i * n + j) = deriv; // J[i][j] in row-major
                }
            }
        }

        // Free input
        unsafe {
            std::alloc::dealloc(
                input_ptr as *mut u8,
                std::alloc::Layout::from_size_align(input_size, 8).unwrap(),
            );
        }
    }

    result_ptr as i64
}

/// Hessian computation using finite differences on gradient
/// func_ptr: function pointer fn(*mut Dual_array) -> Dual (scalar)
/// x_ptr: pointer to input f64 array with layout: [length: i64, data: f64...]
/// Returns pointer to flattened result matrix (n × n, row-major)
#[cfg(feature = "jit")]
extern "C" fn runtime_hessian(
    func_ptr: extern "C" fn(i64) -> i64,
    x_ptr: *const i64,  // pointer to array with length header
) -> i64 {
    if x_ptr.is_null() {
        return 0;
    }

    // Read n from array header at offset 0
    let n = unsafe { *x_ptr } as usize;
    if n == 0 {
        return 0;
    }
    let eps = 1e-7_f64; // Finite difference step

    // Data starts after the length header (offset 8 bytes = 1 i64)
    let data_ptr = unsafe { x_ptr.add(1) } as *const f64;

    // Allocate result matrix (n × n)
    let result_size = n * n * std::mem::size_of::<f64>();
    let result_ptr = unsafe {
        std::alloc::alloc(std::alloc::Layout::from_size_align(result_size, 8).unwrap())
    } as *mut f64;

    if result_ptr.is_null() {
        return 0;
    }

    // Read input values from data (after length header)
    let x: Vec<f64> = unsafe { std::slice::from_raw_parts(data_ptr, n).to_vec() };

    // Helper to compute gradient at a point
    let compute_gradient = |point: &[f64]| -> Vec<f64> {
        let mut grad = vec![0.0; n];
        for j in 0..n {
            // Create dual input with seed at j
            let input_size = n * 16;
            let input_ptr = unsafe {
                std::alloc::alloc(std::alloc::Layout::from_size_align(input_size, 8).unwrap())
            } as *mut f64;

            if input_ptr.is_null() {
                continue;
            }

            for i in 0..n {
                unsafe {
                    *input_ptr.add(i * 2) = point[i];
                    *input_ptr.add(i * 2 + 1) = if i == j { 1.0 } else { 0.0 };
                }
            }

            // Call function - returns pointer to single Dual
            let output_ptr = func_ptr(input_ptr as i64) as *const f64;
            if !output_ptr.is_null() {
                grad[j] = unsafe { *output_ptr.add(1) }; // derivative
            }

            unsafe {
                std::alloc::dealloc(
                    input_ptr as *mut u8,
                    std::alloc::Layout::from_size_align(input_size, 8).unwrap(),
                );
            }
        }
        grad
    };

    // Compute Hessian using finite differences: H[i][j] ≈ (∂f/∂x_j(x+εe_i) - ∂f/∂x_j(x-εe_i)) / (2ε)
    let grad_center = compute_gradient(&x);

    for i in 0..n {
        // x + eps * e_i
        let mut x_plus = x.clone();
        x_plus[i] += eps;
        let grad_plus = compute_gradient(&x_plus);

        // x - eps * e_i
        let mut x_minus = x.clone();
        x_minus[i] -= eps;
        let grad_minus = compute_gradient(&x_minus);

        for j in 0..n {
            // Central difference
            let h_ij = (grad_plus[j] - grad_minus[j]) / (2.0 * eps);
            unsafe {
                *result_ptr.add(i * n + j) = h_ij;
            }
        }
    }

    result_ptr as i64
}

/// Do intervention (Causal.do)
/// Records intervention and returns the intervention value
#[cfg(feature = "jit")]
extern "C" fn runtime_causal_do(var_ptr: *const u8, value: f64) -> f64 {
    if var_ptr.is_null() {
        return value;
    }

    if let Ok(mut state) = get_jit_effect_state().lock() {
        // Get variable name from C string
        let var_name = unsafe {
            std::ffi::CStr::from_ptr(var_ptr as *const std::ffi::c_char)
                .to_str()
                .unwrap_or("unknown")
                .to_string()
        };

        // Record intervention
        state.interventions.push((var_name, value));
        value
    } else {
        value
    }
}

/// Reset effect state (for new probabilistic execution)
#[cfg(feature = "jit")]
extern "C" fn runtime_effect_reset() {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.reset();
    }
}

/// Set random seed for reproducibility
#[cfg(feature = "jit")]
extern "C" fn runtime_effect_set_seed(seed: i64) {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.rng = crate::runtime::prob::Rng::new(seed as u64);
    }
}

// ==================== IO Effect Runtime Functions ====================
// These implement IO effect operations for JIT-compiled code

/// Read a line from stdin (IO.read_line)
/// Returns pointer to heap-allocated string data
/// Caller must track length separately (we return null-terminated for simplicity)
#[cfg(feature = "jit")]
extern "C" fn runtime_io_read_line() -> *const u8 {
    use std::io::BufRead;

    let mut line = String::new();
    if let Ok(_) = std::io::stdin().lock().read_line(&mut line) {
        // Trim trailing newline
        if line.ends_with('\n') {
            line.pop();
            if line.ends_with('\r') {
                line.pop();
            }
        }

        // Store in global string storage to keep it alive
        if let Ok(mut storage) = STRING_STORAGE.lock() {
            let cstring = std::ffi::CString::new(line).unwrap_or_default();
            let ptr = cstring.as_ptr() as *const u8;
            storage.push(cstring);
            ptr
        } else {
            std::ptr::null()
        }
    } else {
        std::ptr::null()
    }
}

/// Read entire file contents (IO.read_file)
/// Takes path as null-terminated string, returns pointer to file contents
#[cfg(feature = "jit")]
extern "C" fn runtime_io_read_file(path_ptr: *const u8) -> *const u8 {
    if path_ptr.is_null() {
        return std::ptr::null();
    }

    let path = unsafe {
        std::ffi::CStr::from_ptr(path_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if path.is_empty() {
        return std::ptr::null();
    }

    match std::fs::read_to_string(path) {
        Ok(contents) => {
            if let Ok(mut storage) = STRING_STORAGE.lock() {
                let cstring = std::ffi::CString::new(contents).unwrap_or_default();
                let ptr = cstring.as_ptr() as *const u8;
                storage.push(cstring);
                ptr
            } else {
                std::ptr::null()
            }
        }
        Err(_) => std::ptr::null(),
    }
}

/// Write string to file (IO.write_file)
/// Takes path and data as null-terminated strings
/// Returns 1 on success, 0 on failure
#[cfg(feature = "jit")]
extern "C" fn runtime_io_write_file(path_ptr: *const u8, data_ptr: *const u8) -> i64 {
    if path_ptr.is_null() || data_ptr.is_null() {
        return 0;
    }

    let path = unsafe {
        std::ffi::CStr::from_ptr(path_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    let data = unsafe {
        std::ffi::CStr::from_ptr(data_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if path.is_empty() {
        return 0;
    }

    match std::fs::write(path, data) {
        Ok(()) => 1,
        Err(_) => 0,
    }
}

/// Append string to file (IO.append_file)
/// Takes path and data as null-terminated strings
/// Returns 1 on success, 0 on failure
#[cfg(feature = "jit")]
extern "C" fn runtime_io_append_file(path_ptr: *const u8, data_ptr: *const u8) -> i64 {
    use std::io::Write;

    if path_ptr.is_null() || data_ptr.is_null() {
        return 0;
    }

    let path = unsafe {
        std::ffi::CStr::from_ptr(path_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    let data = unsafe {
        std::ffi::CStr::from_ptr(data_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if path.is_empty() {
        return 0;
    }

    match std::fs::OpenOptions::new().append(true).create(true).open(path) {
        Ok(mut file) => {
            match file.write_all(data.as_bytes()) {
                Ok(()) => 1,
                Err(_) => 0,
            }
        }
        Err(_) => 0,
    }
}

/// Check if file exists (IO.file_exists)
/// Returns 1 if exists, 0 otherwise
#[cfg(feature = "jit")]
extern "C" fn runtime_io_file_exists(path_ptr: *const u8) -> i64 {
    if path_ptr.is_null() {
        return 0;
    }

    let path = unsafe {
        std::ffi::CStr::from_ptr(path_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if std::path::Path::new(path).exists() { 1 } else { 0 }
}

// ==================== Mut Effect Runtime Functions ====================
// These implement mutable state effect operations using a named state store

/// Get a value from mutable state by name (Mut.get)
/// Returns 0.0 if name doesn't exist
#[cfg(feature = "jit")]
extern "C" fn runtime_mut_get(name_ptr: *const u8) -> f64 {
    if name_ptr.is_null() {
        return 0.0;
    }

    let name = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if let Ok(state) = get_jit_effect_state().lock() {
        *state.mutable_state.get(name).unwrap_or(&0.0)
    } else {
        0.0
    }
}

/// Set a value in mutable state by name (Mut.set)
/// Returns 1 on success, 0 on failure
#[cfg(feature = "jit")]
extern "C" fn runtime_mut_set(name_ptr: *const u8, value: f64) -> i64 {
    if name_ptr.is_null() {
        return 0;
    }

    let name = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
            .to_string()
    };

    if name.is_empty() {
        return 0;
    }

    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.mutable_state.insert(name, value);
        1
    } else {
        0
    }
}

/// Modify a value in mutable state by adding delta (Mut.modify)
/// Returns the new value after modification
#[cfg(feature = "jit")]
extern "C" fn runtime_mut_modify(name_ptr: *const u8, delta: f64) -> f64 {
    if name_ptr.is_null() {
        return delta;
    }

    let name = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
            .to_string()
    };

    if name.is_empty() {
        return delta;
    }

    if let Ok(mut state) = get_jit_effect_state().lock() {
        let current = *state.mutable_state.get(&name).unwrap_or(&0.0);
        let new_value = current + delta;
        state.mutable_state.insert(name, new_value);
        new_value
    } else {
        delta
    }
}

/// Clear all mutable state (Mut.clear)
#[cfg(feature = "jit")]
extern "C" fn runtime_mut_clear() {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.mutable_state.clear();
    }
}

/// Check if a name exists in mutable state (Mut.exists)
/// Returns 1 if exists, 0 otherwise
#[cfg(feature = "jit")]
extern "C" fn runtime_mut_exists(name_ptr: *const u8) -> i64 {
    if name_ptr.is_null() {
        return 0;
    }

    let name = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if let Ok(state) = get_jit_effect_state().lock() {
        if state.mutable_state.contains_key(name) { 1 } else { 0 }
    } else {
        0
    }
}

/// Delete a value from mutable state (Mut.delete)
/// Returns the deleted value or 0.0 if not found
#[cfg(feature = "jit")]
extern "C" fn runtime_mut_delete(name_ptr: *const u8) -> f64 {
    if name_ptr.is_null() {
        return 0.0;
    }

    let name = unsafe {
        std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.mutable_state.remove(name).unwrap_or(0.0)
    } else {
        0.0
    }
}

// ==================== Alloc Effect Runtime Functions ====================
// These implement tracked memory allocation for JIT-compiled code

/// Allocate memory of given size (Alloc.alloc)
/// Returns pointer to allocated memory, or 0 on failure
#[cfg(feature = "jit")]
extern "C" fn runtime_alloc(size: i64) -> i64 {
    if size <= 0 {
        return 0;
    }

    let size = size as usize;

    // Use Vec<u8> for allocation to ensure proper alignment and tracking
    let mut buffer: Vec<u8> = Vec::with_capacity(size);
    buffer.resize(size, 0);

    let ptr = buffer.as_ptr() as usize;

    // Prevent deallocation - we'll track it manually
    std::mem::forget(buffer);

    // Track the allocation
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.allocations.insert(ptr, size);
        state.total_allocated += size;
    }

    ptr as i64
}

/// Allocate array of count elements of elem_size bytes (Alloc.alloc_array)
/// Returns pointer to allocated memory, or 0 on failure
#[cfg(feature = "jit")]
extern "C" fn runtime_alloc_array(count: i64, elem_size: i64) -> i64 {
    if count <= 0 || elem_size <= 0 {
        return 0;
    }

    let total_size = (count as usize).saturating_mul(elem_size as usize);
    if total_size == 0 {
        return 0;
    }

    runtime_alloc(total_size as i64)
}

/// Deallocate memory (Alloc.dealloc)
/// Returns 1 on success, 0 if pointer was not tracked
#[cfg(feature = "jit")]
extern "C" fn runtime_dealloc(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }

    let ptr = ptr as usize;

    if let Ok(mut state) = get_jit_effect_state().lock() {
        if let Some(size) = state.allocations.remove(&ptr) {
            // Reconstruct the Vec to properly deallocate
            unsafe {
                let _ = Vec::from_raw_parts(ptr as *mut u8, size, size);
            }
            state.total_allocated = state.total_allocated.saturating_sub(size);
            1
        } else {
            0 // Pointer not tracked
        }
    } else {
        0
    }
}

/// Reallocate memory to new size (Alloc.realloc)
/// Returns new pointer, or 0 on failure
#[cfg(feature = "jit")]
extern "C" fn runtime_realloc(ptr: i64, new_size: i64) -> i64 {
    if new_size <= 0 {
        // Size 0 means dealloc
        runtime_dealloc(ptr);
        return 0;
    }

    if ptr == 0 {
        // Null pointer means just alloc
        return runtime_alloc(new_size);
    }

    let ptr_val = ptr as usize;
    let new_size = new_size as usize;

    if let Ok(mut state) = get_jit_effect_state().lock() {
        if let Some(old_size) = state.allocations.remove(&ptr_val) {
            // Create new allocation
            let mut new_buffer: Vec<u8> = Vec::with_capacity(new_size);
            new_buffer.resize(new_size, 0);

            // Copy old data
            let copy_size = old_size.min(new_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    ptr_val as *const u8,
                    new_buffer.as_mut_ptr(),
                    copy_size,
                );
            }

            let new_ptr = new_buffer.as_ptr() as usize;
            std::mem::forget(new_buffer);

            // Free old memory
            unsafe {
                let _ = Vec::from_raw_parts(ptr_val as *mut u8, old_size, old_size);
            }

            // Update tracking
            state.allocations.insert(new_ptr, new_size);
            state.total_allocated = state.total_allocated.saturating_sub(old_size) + new_size;

            new_ptr as i64
        } else {
            0 // Original pointer not tracked
        }
    } else {
        0
    }
}

/// Get size of allocation (Alloc.size_of)
/// Returns size in bytes, or 0 if pointer not tracked
#[cfg(feature = "jit")]
extern "C" fn runtime_alloc_size_of(ptr: i64) -> i64 {
    if ptr == 0 {
        return 0;
    }

    let ptr = ptr as usize;

    if let Ok(state) = get_jit_effect_state().lock() {
        state.allocations.get(&ptr).copied().unwrap_or(0) as i64
    } else {
        0
    }
}

/// Clear all allocations (Alloc.clear)
/// Frees all tracked memory
#[cfg(feature = "jit")]
extern "C" fn runtime_alloc_clear() {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        // Free all tracked allocations
        for (&ptr, &size) in state.allocations.iter() {
            unsafe {
                let _ = Vec::from_raw_parts(ptr as *mut u8, size, size);
            }
        }
        state.allocations.clear();
        state.total_allocated = 0;
    }
}

/// Get total bytes currently allocated (Alloc.total_allocated)
#[cfg(feature = "jit")]
extern "C" fn runtime_alloc_total() -> i64 {
    if let Ok(state) = get_jit_effect_state().lock() {
        state.total_allocated as i64
    } else {
        0
    }
}

/// Get number of active allocations (Alloc.count)
#[cfg(feature = "jit")]
extern "C" fn runtime_alloc_count() -> i64 {
    if let Ok(state) = get_jit_effect_state().lock() {
        state.allocations.len() as i64
    } else {
        0
    }
}

// ==================== Handler Stack Runtime Functions ====================
// These implement continuation-based effect handler infrastructure

/// Push a handler onto the handler stack
/// effect_ptr: C string pointer to effect name
/// name_ptr: C string pointer to handler name
/// handler_id: predefined handler ID (0 for custom)
/// Returns: handler stack depth after push
#[cfg(feature = "jit")]
extern "C" fn runtime_handler_push(effect_ptr: *const u8, name_ptr: *const u8, handler_id: u32) -> i64 {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        let effect = if effect_ptr.is_null() {
            "unknown".to_string()
        } else {
            unsafe {
                std::ffi::CStr::from_ptr(effect_ptr as *const std::ffi::c_char)
                    .to_str()
                    .unwrap_or("unknown")
                    .to_string()
            }
        };

        let name = if name_ptr.is_null() {
            "anonymous".to_string()
        } else {
            unsafe {
                std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char)
                    .to_str()
                    .unwrap_or("anonymous")
                    .to_string()
            }
        };

        let handler = JitHandler::new(&effect, &name, handler_id);
        state.push_handler(handler);
        state.handler_stack.len() as i64
    } else {
        -1
    }
}

/// Pop a handler from the handler stack
/// Returns: handler stack depth after pop, or -1 if stack was empty
#[cfg(feature = "jit")]
extern "C" fn runtime_handler_pop() -> i64 {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        if state.pop_handler().is_some() {
            state.handler_stack.len() as i64
        } else {
            -1
        }
    } else {
        -1
    }
}

/// Get current handler stack depth
#[cfg(feature = "jit")]
extern "C" fn runtime_handler_depth() -> i64 {
    if let Ok(state) = get_jit_effect_state().lock() {
        state.handler_stack.len() as i64
    } else {
        0
    }
}

/// Check if a handler exists for the given effect
/// Returns: 1 if handler exists, 0 otherwise
#[cfg(feature = "jit")]
extern "C" fn runtime_handler_has(effect_ptr: *const u8) -> i64 {
    if effect_ptr.is_null() {
        return 0;
    }

    if let Ok(state) = get_jit_effect_state().lock() {
        let effect = unsafe {
            std::ffi::CStr::from_ptr(effect_ptr as *const std::ffi::c_char)
                .to_str()
                .unwrap_or("")
        };
        if state.find_handler(effect).is_some() { 1 } else { 0 }
    } else {
        0
    }
}

/// Dispatch effect to handler stack, returning handled value or NaN if not handled
/// effect_ptr: C string pointer to effect name
/// op_ptr: C string pointer to operation name
/// args_ptr: pointer to array of f64 arguments
/// args_len: number of arguments
/// Returns: handled value, or NaN if no handler or handler declined
#[cfg(feature = "jit")]
extern "C" fn runtime_handler_dispatch(
    effect_ptr: *const u8,
    op_ptr: *const u8,
    args_ptr: *const f64,
    args_len: usize,
) -> f64 {
    if effect_ptr.is_null() || op_ptr.is_null() {
        return f64::NAN;
    }

    if let Ok(state) = get_jit_effect_state().lock() {
        let effect = unsafe {
            std::ffi::CStr::from_ptr(effect_ptr as *const std::ffi::c_char)
                .to_str()
                .unwrap_or("")
        };
        let op = unsafe {
            std::ffi::CStr::from_ptr(op_ptr as *const std::ffi::c_char)
                .to_str()
                .unwrap_or("")
        };
        let args: Vec<f64> = if args_ptr.is_null() || args_len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(args_ptr, args_len).to_vec() }
        };

        if let Some(value) = state.dispatch_to_handler(effect, op, &args) {
            value
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

/// Create a continuation object for the current effect operation
/// Returns: continuation ID (for future resume), or 0 on error
#[cfg(feature = "jit")]
extern "C" fn runtime_continuation_create(
    effect_ptr: *const u8,
    op_ptr: *const u8,
    args_ptr: *const f64,
    args_len: usize,
) -> i64 {
    if effect_ptr.is_null() || op_ptr.is_null() {
        return 0;
    }

    if let Ok(mut state) = get_jit_effect_state().lock() {
        let effect = unsafe {
            std::ffi::CStr::from_ptr(effect_ptr as *const std::ffi::c_char)
                .to_str()
                .unwrap_or("")
                .to_string()
        };
        let operation = unsafe {
            std::ffi::CStr::from_ptr(op_ptr as *const std::ffi::c_char)
                .to_str()
                .unwrap_or("")
                .to_string()
        };
        let args: Vec<f64> = if args_ptr.is_null() || args_len == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(args_ptr, args_len).to_vec() }
        };

        state.continuation_data = Some(ContinuationData {
            effect,
            operation,
            args,
            resumed: false,
            resume_value: 0.0,
        });

        1 // Success, continuation ID (we only support one at a time currently)
    } else {
        0
    }
}

/// Resume the current continuation with a value
/// continuation_id: ID from runtime_continuation_create
/// value: value to resume with
/// Returns: 1 on success, 0 on error
#[cfg(feature = "jit")]
#[unsafe(no_mangle)]
extern "C" fn runtime_continuation_resume(continuation_id: i64, value: f64) -> i64 {
    if continuation_id != 1 {
        return 0;
    }

    if let Ok(mut state) = get_jit_effect_state().lock() {
        if let Some(ref mut cont) = state.continuation_data {
            cont.resumed = true;
            cont.resume_value = value;
            1
        } else {
            0
        }
    } else {
        0
    }
}

/// Get the resume value from the current continuation
/// Returns: the resume value, or NaN if no continuation or not resumed
#[cfg(feature = "jit")]
extern "C" fn runtime_continuation_get_value() -> f64 {
    if let Ok(state) = get_jit_effect_state().lock() {
        if let Some(ref cont) = state.continuation_data {
            if cont.resumed {
                cont.resume_value
            } else {
                f64::NAN
            }
        } else {
            f64::NAN
        }
    } else {
        f64::NAN
    }
}

/// Clear the current continuation
#[cfg(feature = "jit")]
extern "C" fn runtime_continuation_clear() {
    if let Ok(mut state) = get_jit_effect_state().lock() {
        state.continuation_data = None;
    }
}

#[cfg(feature = "jit")]
use crate::hlir::{
    BinaryOp, BlockId, HlirBlock, HlirConstant, HlirFunction, HlirTerminator, HlirType, Op,
    UnaryOp, ValueId,
};
use std::collections::HashMap;

#[cfg(feature = "jit")]
use cranelift_codegen::Context;
#[cfg(feature = "jit")]
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, UserFuncName, types};
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{FuncId, Linkage, Module};

/// Compile HLIR module to native code via Cranelift JIT
#[cfg(feature = "jit")]
pub fn compile(module: &HlirModule) -> Result<Vec<u8>, String> {
    let jit = CraneliftJit::new();
    let compiled = jit.compile(module)?;
    // Return empty vec - the compiled code is held in memory by JITModule
    Ok(vec![])
}

#[cfg(not(feature = "jit"))]
pub fn compile(_module: &HlirModule) -> Result<Vec<u8>, String> {
    Err("JIT backend not enabled. Compile with --features jit".to_string())
}

/// Cranelift JIT compiler
pub struct CraneliftJit {
    /// Whether to enable optimization
    optimize: bool,
}

impl CraneliftJit {
    pub fn new() -> Self {
        Self { optimize: false }
    }

    pub fn with_optimization(mut self) -> Self {
        self.optimize = true;
        self
    }

    /// Compile and immediately run the module, returning the result of main()
    #[cfg(feature = "jit")]
    pub fn compile_and_run(&self, module: &HlirModule) -> Result<i64, String> {
        let compiled = self.compile(module)?;
        unsafe { compiled.call_i64("main") }
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile_and_run(&self, _module: &HlirModule) -> Result<i64, String> {
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }

    /// Compile the module and return a handle to the compiled code
    #[cfg(feature = "jit")]
    pub fn compile(&self, module: &HlirModule) -> Result<CompiledModule, String> {
        let mut compiler = JitCompiler::new(self.optimize)?;
        compiler.compile_module(module)?;
        compiler.finalize()
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile(&self, _module: &HlirModule) -> Result<CompiledModule, String> {
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }
}

impl Default for CraneliftJit {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to compiled JIT code
pub struct CompiledModule {
    #[cfg(feature = "jit")]
    #[allow(dead_code)]
    jit_module: JITModule,
    /// Function entry points
    functions: HashMap<String, *const u8>,
}

// SAFETY: The JIT module owns the compiled code and manages its lifetime
unsafe impl Send for CompiledModule {}
unsafe impl Sync for CompiledModule {}

impl CompiledModule {
    /// Create a new compiled module from a JIT module and function pointers
    #[cfg(feature = "jit")]
    pub fn new(jit_module: JITModule, functions: HashMap<String, *const u8>) -> Self {
        Self {
            jit_module,
            functions,
        }
    }

    /// Get a function pointer by name
    pub fn get_function(&self, name: &str) -> Option<*const u8> {
        self.functions.get(name).copied()
    }

    /// Call a function with no arguments returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    pub unsafe fn call_i64(&self, name: &str) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func())
    }

    /// Call a function with no arguments returning void.
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    pub unsafe fn call_void(&self, name: &str) -> Result<(), String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn() = unsafe { std::mem::transmute(ptr) };
        func();
        Ok(())
    }

    /// Call a function with one i64 argument returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    #[allow(dead_code)]
    pub unsafe fn call_i64_i64(&self, name: &str, arg: i64) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func(arg))
    }

    /// Call a function with two i64 arguments returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    #[allow(dead_code)]
    pub unsafe fn call_i64_i64_i64(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func(a, b))
    }
}

/// JIT compilation settings
#[allow(dead_code)]
pub struct JitSettings {
    /// Enable basic optimizations
    pub optimize: bool,
    /// Enable bounds checking
    pub bounds_check: bool,
    /// Enable overflow checking
    pub overflow_check: bool,
    /// Stack size in bytes
    pub stack_size: usize,
}

impl Default for JitSettings {
    fn default() -> Self {
        Self {
            optimize: false,
            bounds_check: true,
            overflow_check: true,
            stack_size: 1024 * 1024, // 1 MB
        }
    }
}

impl JitSettings {
    #[allow(dead_code)]
    pub fn release() -> Self {
        Self {
            optimize: true,
            bounds_check: false,
            overflow_check: false,
            stack_size: 8 * 1024 * 1024, // 8 MB
        }
    }
}

// ==================== JIT Compiler Implementation ====================

#[cfg(feature = "jit")]
struct JitCompiler {
    jit_module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
    /// Map from HLIR function names to Cranelift function IDs
    func_ids: HashMap<String, FuncId>,
    /// Map from function names to their signatures (for calling)
    func_sigs: HashMap<String, Signature>,
    /// Set of exported (user-defined) function names
    exported_funcs: std::collections::HashSet<String>,
}

#[cfg(feature = "jit")]
impl JitCompiler {
    fn new(optimize: bool) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        if optimize {
            flag_builder.set("opt_level", "speed").unwrap();
        } else {
            flag_builder.set("opt_level", "none").unwrap();
        }

        let isa_builder = cranelift_native::builder()
            .map_err(|e| format!("Failed to create ISA builder: {}", e))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("Failed to create ISA: {}", e))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register runtime functions
        jit_builder.symbol("runtime_print_i64", runtime_print_i64 as *const u8);
        jit_builder.symbol("runtime_print_f64", runtime_print_f64 as *const u8);
        jit_builder.symbol("runtime_print_newline", runtime_print_newline as *const u8);
        jit_builder.symbol("runtime_print_str", runtime_print_str as *const u8);
        jit_builder.symbol("runtime_print_cstr", runtime_print_cstr as *const u8);
        jit_builder.symbol("runtime_print_bool", runtime_print_bool as *const u8);
        jit_builder.symbol("runtime_debug_test", runtime_debug_test as *const u8);

        // Register effect runtime functions
        jit_builder.symbol("runtime_prob_sample", runtime_prob_sample as *const u8);
        jit_builder.symbol("runtime_prob_sample_uniform", runtime_prob_sample_uniform as *const u8);
        jit_builder.symbol("runtime_prob_sample_bernoulli", runtime_prob_sample_bernoulli as *const u8);
        jit_builder.symbol("runtime_prob_observe", runtime_prob_observe as *const u8);
        jit_builder.symbol("runtime_causal_do", runtime_causal_do as *const u8);
        jit_builder.symbol("runtime_effect_reset", runtime_effect_reset as *const u8);
        jit_builder.symbol("runtime_effect_set_seed", runtime_effect_set_seed as *const u8);

        // Register generic effect dispatch functions (for registry-based handlers)
        jit_builder.symbol("runtime_effect_dispatch", runtime_effect_dispatch as *const u8);
        jit_builder.symbol("runtime_dispatch_effect_f64", runtime_dispatch_effect_f64 as *const u8);

        // Register math runtime functions
        jit_builder.symbol("runtime_math_sin", runtime_math_sin as *const u8);
        jit_builder.symbol("runtime_math_cos", runtime_math_cos as *const u8);
        jit_builder.symbol("runtime_math_exp", runtime_math_exp as *const u8);
        jit_builder.symbol("runtime_math_log", runtime_math_log as *const u8);
        jit_builder.symbol("runtime_math_sqrt", runtime_math_sqrt as *const u8);
        jit_builder.symbol("runtime_math_tan", runtime_math_tan as *const u8);
        jit_builder.symbol("runtime_math_atan", runtime_math_atan as *const u8);
        jit_builder.symbol("runtime_math_fabs", runtime_math_fabs as *const u8);
        jit_builder.symbol("runtime_math_asin", runtime_math_asin as *const u8);
        jit_builder.symbol("runtime_math_acos", runtime_math_acos as *const u8);
        jit_builder.symbol("runtime_math_sinh", runtime_math_sinh as *const u8);
        jit_builder.symbol("runtime_math_cosh", runtime_math_cosh as *const u8);
        jit_builder.symbol("runtime_math_tanh", runtime_math_tanh as *const u8);
        jit_builder.symbol("runtime_math_asinh", runtime_math_asinh as *const u8);
        jit_builder.symbol("runtime_math_acosh", runtime_math_acosh as *const u8);
        jit_builder.symbol("runtime_math_atanh", runtime_math_atanh as *const u8);
        jit_builder.symbol("runtime_math_log2", runtime_math_log2 as *const u8);
        jit_builder.symbol("runtime_math_log10", runtime_math_log10 as *const u8);
        jit_builder.symbol("runtime_math_atan2", runtime_math_atan2 as *const u8);

        // Register autodiff runtime functions (jacobian, hessian)
        jit_builder.symbol("runtime_jacobian", runtime_jacobian as *const u8);
        jit_builder.symbol("runtime_hessian", runtime_hessian as *const u8);

        // Register IO effect runtime functions
        jit_builder.symbol("runtime_io_read_line", runtime_io_read_line as *const u8);
        jit_builder.symbol("runtime_io_read_file", runtime_io_read_file as *const u8);
        jit_builder.symbol("runtime_io_write_file", runtime_io_write_file as *const u8);
        jit_builder.symbol("runtime_io_append_file", runtime_io_append_file as *const u8);
        jit_builder.symbol("runtime_io_file_exists", runtime_io_file_exists as *const u8);

        // Register Mut effect runtime functions
        jit_builder.symbol("runtime_mut_get", runtime_mut_get as *const u8);
        jit_builder.symbol("runtime_mut_set", runtime_mut_set as *const u8);
        jit_builder.symbol("runtime_mut_modify", runtime_mut_modify as *const u8);
        jit_builder.symbol("runtime_mut_clear", runtime_mut_clear as *const u8);
        jit_builder.symbol("runtime_mut_exists", runtime_mut_exists as *const u8);
        jit_builder.symbol("runtime_mut_delete", runtime_mut_delete as *const u8);

        // Register Alloc effect runtime functions
        jit_builder.symbol("runtime_alloc", runtime_alloc as *const u8);
        jit_builder.symbol("runtime_alloc_array", runtime_alloc_array as *const u8);
        jit_builder.symbol("runtime_dealloc", runtime_dealloc as *const u8);
        jit_builder.symbol("runtime_realloc", runtime_realloc as *const u8);
        jit_builder.symbol("runtime_alloc_size_of", runtime_alloc_size_of as *const u8);
        jit_builder.symbol("runtime_alloc_clear", runtime_alloc_clear as *const u8);
        jit_builder.symbol("runtime_alloc_total", runtime_alloc_total as *const u8);
        jit_builder.symbol("runtime_alloc_count", runtime_alloc_count as *const u8);

        // Register handler stack and continuation runtime functions
        jit_builder.symbol("runtime_handler_push", runtime_handler_push as *const u8);
        jit_builder.symbol("runtime_handler_pop", runtime_handler_pop as *const u8);
        jit_builder.symbol("runtime_handler_depth", runtime_handler_depth as *const u8);
        jit_builder.symbol("runtime_handler_has", runtime_handler_has as *const u8);
        jit_builder.symbol("runtime_handler_dispatch", runtime_handler_dispatch as *const u8);
        jit_builder.symbol("runtime_continuation_create", runtime_continuation_create as *const u8);
        jit_builder.symbol("runtime_continuation_resume", runtime_continuation_resume as *const u8);
        jit_builder.symbol("runtime_continuation_get_value", runtime_continuation_get_value as *const u8);
        jit_builder.symbol("runtime_continuation_clear", runtime_continuation_clear as *const u8);

        // Register __sounio_* functions from runtime handler_stack module
        // These are used by SIR lowering and other backends
        jit_builder.symbol(
            "__sounio_push_handler_io",
            handler_stack::__sounio_push_handler_io as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_mut",
            handler_stack::__sounio_push_handler_mut as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_div",
            handler_stack::__sounio_push_handler_div as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_prob",
            handler_stack::__sounio_push_handler_prob as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_alloc",
            handler_stack::__sounio_push_handler_alloc as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_panic",
            handler_stack::__sounio_push_handler_panic as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_async",
            handler_stack::__sounio_push_handler_async as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_gpu",
            handler_stack::__sounio_push_handler_gpu as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_grad",
            handler_stack::__sounio_push_handler_grad as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_network",
            handler_stack::__sounio_push_handler_network as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_sensor",
            handler_stack::__sounio_push_handler_sensor as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_exn",
            handler_stack::__sounio_push_handler_exn as *const u8,
        );
        jit_builder.symbol(
            "__sounio_push_handler_causal",
            handler_stack::__sounio_push_handler_causal as *const u8,
        );
        jit_builder.symbol(
            "__sounio_pop_handler",
            handler_stack::__sounio_pop_handler as *const u8,
        );
        jit_builder.symbol(
            "__sounio_handler_depth",
            handler_stack::__sounio_handler_depth as *const u8,
        );

        // Register dispatch functions
        jit_builder.symbol(
            "__sounio_dispatch_io_print",
            handler_stack::__sounio_dispatch_io_print as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_io_println",
            handler_stack::__sounio_dispatch_io_println as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_io_read",
            handler_stack::__sounio_dispatch_io_read as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_mut_get",
            handler_stack::__sounio_dispatch_mut_get as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_mut_set",
            handler_stack::__sounio_dispatch_mut_set as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_mut_modify",
            handler_stack::__sounio_dispatch_mut_modify as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_div_div",
            handler_stack::__sounio_dispatch_div_div as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_prob_sample",
            handler_stack::__sounio_dispatch_prob_sample as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_prob_observe",
            handler_stack::__sounio_dispatch_prob_observe as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_alloc_alloc",
            handler_stack::__sounio_dispatch_alloc_alloc as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_alloc_dealloc",
            handler_stack::__sounio_dispatch_alloc_dealloc as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_panic_panic",
            handler_stack::__sounio_dispatch_panic_panic as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_gpu_sync",
            handler_stack::__sounio_dispatch_gpu_sync as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_gpu_launch",
            handler_stack::__sounio_dispatch_gpu_launch as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_async_spawn",
            handler_stack::__sounio_dispatch_async_spawn as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_async_yield",
            handler_stack::__sounio_dispatch_async_yield as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_async_await",
            handler_stack::__sounio_dispatch_async_await as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_grad_forward",
            handler_stack::__sounio_dispatch_grad_forward as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_grad_reverse",
            handler_stack::__sounio_dispatch_grad_reverse as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_causal_do",
            handler_stack::__sounio_dispatch_causal_do as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_causal_observe",
            handler_stack::__sounio_dispatch_causal_observe as *const u8,
        );
        jit_builder.symbol(
            "__sounio_dispatch_generic",
            handler_stack::__sounio_dispatch_generic as *const u8,
        );

        let jit_module = JITModule::new(jit_builder);
        let ctx = jit_module.make_context();

        Ok(Self {
            jit_module,
            ctx,
            func_ctx: FunctionBuilderContext::new(),
            func_ids: HashMap::new(),
            func_sigs: HashMap::new(),
            exported_funcs: std::collections::HashSet::new(),
        })
    }

    fn compile_module(&mut self, module: &HlirModule) -> Result<(), String> {
        // Declare runtime functions first
        self.declare_runtime_functions()?;

        // First pass: declare all user functions
        for func in &module.functions {
            // Declaration-only functions are treated as imports (e.g., `extern { fn ...; }`).
            let is_import = func.blocks.is_empty();
            let sig = self.create_signature(func);
            let symbol_name = func.link_name.as_deref().unwrap_or(&func.name);
            let linkage = if is_import {
                Linkage::Import
            } else {
                Linkage::Export
            };
            let func_id = self
                .jit_module
                .declare_function(symbol_name, linkage, &sig)
                .map_err(|e| format!("Failed to declare function {}: {}", func.name, e))?;
            self.func_ids.insert(func.name.clone(), func_id);
            self.func_sigs.insert(func.name.clone(), sig);
            if !is_import {
                self.exported_funcs.insert(func.name.clone());
            }
        }

        // Second pass: compile all functions
        for func in &module.functions {
            if func.blocks.is_empty() {
                // Import-only declarations have no body to compile/define.
                continue;
            }
            self.compile_function(func)?;
        }

        Ok(())
    }

    fn declare_runtime_functions(&mut self) -> Result<(), String> {
        let call_conv = self.jit_module.isa().default_call_conv();

        // runtime_print_i64(i64) -> void
        let mut sig_print_i64 = Signature::new(call_conv);
        sig_print_i64.params.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_print_i64", Linkage::Import, &sig_print_i64)
            .map_err(|e| format!("Failed to declare runtime_print_i64: {}", e))?;
        self.func_ids.insert("runtime_print_i64".to_string(), id);
        self.func_sigs
            .insert("runtime_print_i64".to_string(), sig_print_i64);

        // runtime_print_f64(f64) -> void
        let mut sig_print_f64 = Signature::new(call_conv);
        sig_print_f64.params.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_print_f64", Linkage::Import, &sig_print_f64)
            .map_err(|e| format!("Failed to declare runtime_print_f64: {}", e))?;
        self.func_ids.insert("runtime_print_f64".to_string(), id);
        self.func_sigs
            .insert("runtime_print_f64".to_string(), sig_print_f64);

        // runtime_print_newline() -> void
        let sig_print_newline = Signature::new(call_conv);
        let id = self
            .jit_module
            .declare_function("runtime_print_newline", Linkage::Import, &sig_print_newline)
            .map_err(|e| format!("Failed to declare runtime_print_newline: {}", e))?;
        self.func_ids
            .insert("runtime_print_newline".to_string(), id);
        self.func_sigs
            .insert("runtime_print_newline".to_string(), sig_print_newline);

        // runtime_print_str(ptr, len) -> void
        let mut sig_print_str = Signature::new(call_conv);
        sig_print_str.params.push(AbiParam::new(types::I64)); // ptr
        sig_print_str.params.push(AbiParam::new(types::I64)); // len
        let id = self
            .jit_module
            .declare_function("runtime_print_str", Linkage::Import, &sig_print_str)
            .map_err(|e| format!("Failed to declare runtime_print_str: {}", e))?;
        self.func_ids.insert("runtime_print_str".to_string(), id);
        self.func_sigs
            .insert("runtime_print_str".to_string(), sig_print_str);

        // runtime_print_cstr(ptr) -> void  (null-terminated C string)
        let mut sig_print_cstr = Signature::new(call_conv);
        sig_print_cstr.params.push(AbiParam::new(types::I64)); // ptr
        let id = self
            .jit_module
            .declare_function("runtime_print_cstr", Linkage::Import, &sig_print_cstr)
            .map_err(|e| format!("Failed to declare runtime_print_cstr: {}", e))?;
        self.func_ids.insert("runtime_print_cstr".to_string(), id);
        self.func_sigs
            .insert("runtime_print_cstr".to_string(), sig_print_cstr);

        // runtime_print_bool(i8) -> void
        let mut sig_print_bool = Signature::new(call_conv);
        sig_print_bool.params.push(AbiParam::new(types::I8));
        let id = self
            .jit_module
            .declare_function("runtime_print_bool", Linkage::Import, &sig_print_bool)
            .map_err(|e| format!("Failed to declare runtime_print_bool: {}", e))?;
        self.func_ids.insert("runtime_print_bool".to_string(), id);
        self.func_sigs
            .insert("runtime_print_bool".to_string(), sig_print_bool);

        // ==================== Effect Runtime Functions ====================

        // runtime_prob_sample(mean: f64, std: f64) -> f64
        let mut sig_prob_sample = Signature::new(call_conv);
        sig_prob_sample.params.push(AbiParam::new(types::F64)); // mean
        sig_prob_sample.params.push(AbiParam::new(types::F64)); // std
        sig_prob_sample.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_prob_sample", Linkage::Import, &sig_prob_sample)
            .map_err(|e| format!("Failed to declare runtime_prob_sample: {}", e))?;
        self.func_ids.insert("runtime_prob_sample".to_string(), id);
        self.func_sigs
            .insert("runtime_prob_sample".to_string(), sig_prob_sample);

        // runtime_prob_sample_uniform(low: f64, high: f64) -> f64
        let mut sig_prob_uniform = Signature::new(call_conv);
        sig_prob_uniform.params.push(AbiParam::new(types::F64)); // low
        sig_prob_uniform.params.push(AbiParam::new(types::F64)); // high
        sig_prob_uniform.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_prob_sample_uniform", Linkage::Import, &sig_prob_uniform)
            .map_err(|e| format!("Failed to declare runtime_prob_sample_uniform: {}", e))?;
        self.func_ids
            .insert("runtime_prob_sample_uniform".to_string(), id);
        self.func_sigs
            .insert("runtime_prob_sample_uniform".to_string(), sig_prob_uniform);

        // runtime_prob_sample_bernoulli(p: f64) -> i64
        let mut sig_prob_bernoulli = Signature::new(call_conv);
        sig_prob_bernoulli.params.push(AbiParam::new(types::F64)); // p
        sig_prob_bernoulli.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_prob_sample_bernoulli", Linkage::Import, &sig_prob_bernoulli)
            .map_err(|e| format!("Failed to declare runtime_prob_sample_bernoulli: {}", e))?;
        self.func_ids
            .insert("runtime_prob_sample_bernoulli".to_string(), id);
        self.func_sigs
            .insert("runtime_prob_sample_bernoulli".to_string(), sig_prob_bernoulli);

        // runtime_prob_observe(mean: f64, std: f64, observed: f64) -> f64
        let mut sig_prob_observe = Signature::new(call_conv);
        sig_prob_observe.params.push(AbiParam::new(types::F64)); // mean
        sig_prob_observe.params.push(AbiParam::new(types::F64)); // std
        sig_prob_observe.params.push(AbiParam::new(types::F64)); // observed
        sig_prob_observe.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_prob_observe", Linkage::Import, &sig_prob_observe)
            .map_err(|e| format!("Failed to declare runtime_prob_observe: {}", e))?;
        self.func_ids
            .insert("runtime_prob_observe".to_string(), id);
        self.func_sigs
            .insert("runtime_prob_observe".to_string(), sig_prob_observe);

        // runtime_causal_do(var_ptr: i64, value: f64) -> f64
        let mut sig_causal_do = Signature::new(call_conv);
        sig_causal_do.params.push(AbiParam::new(types::I64)); // var_ptr
        sig_causal_do.params.push(AbiParam::new(types::F64)); // value
        sig_causal_do.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_causal_do", Linkage::Import, &sig_causal_do)
            .map_err(|e| format!("Failed to declare runtime_causal_do: {}", e))?;
        self.func_ids.insert("runtime_causal_do".to_string(), id);
        self.func_sigs
            .insert("runtime_causal_do".to_string(), sig_causal_do);

        // runtime_effect_reset() -> void
        let sig_effect_reset = Signature::new(call_conv);
        let id = self
            .jit_module
            .declare_function("runtime_effect_reset", Linkage::Import, &sig_effect_reset)
            .map_err(|e| format!("Failed to declare runtime_effect_reset: {}", e))?;
        self.func_ids
            .insert("runtime_effect_reset".to_string(), id);
        self.func_sigs
            .insert("runtime_effect_reset".to_string(), sig_effect_reset);

        // runtime_effect_set_seed(seed: i64) -> void
        let mut sig_effect_seed = Signature::new(call_conv);
        sig_effect_seed.params.push(AbiParam::new(types::I64)); // seed
        let id = self
            .jit_module
            .declare_function("runtime_effect_set_seed", Linkage::Import, &sig_effect_seed)
            .map_err(|e| format!("Failed to declare runtime_effect_set_seed: {}", e))?;
        self.func_ids
            .insert("runtime_effect_set_seed".to_string(), id);
        self.func_sigs
            .insert("runtime_effect_set_seed".to_string(), sig_effect_seed);

        // ==================== Math Runtime Functions ====================

        // runtime_math_sin(x: f64) -> f64
        let mut sig_math_sin = Signature::new(call_conv);
        sig_math_sin.params.push(AbiParam::new(types::F64));
        sig_math_sin.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_sin", Linkage::Import, &sig_math_sin)
            .map_err(|e| format!("Failed to declare runtime_math_sin: {}", e))?;
        self.func_ids.insert("runtime_math_sin".to_string(), id);
        self.func_sigs.insert("runtime_math_sin".to_string(), sig_math_sin);

        // runtime_math_cos(x: f64) -> f64
        let mut sig_math_cos = Signature::new(call_conv);
        sig_math_cos.params.push(AbiParam::new(types::F64));
        sig_math_cos.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_cos", Linkage::Import, &sig_math_cos)
            .map_err(|e| format!("Failed to declare runtime_math_cos: {}", e))?;
        self.func_ids.insert("runtime_math_cos".to_string(), id);
        self.func_sigs.insert("runtime_math_cos".to_string(), sig_math_cos);

        // runtime_math_exp(x: f64) -> f64
        let mut sig_math_exp = Signature::new(call_conv);
        sig_math_exp.params.push(AbiParam::new(types::F64));
        sig_math_exp.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_exp", Linkage::Import, &sig_math_exp)
            .map_err(|e| format!("Failed to declare runtime_math_exp: {}", e))?;
        self.func_ids.insert("runtime_math_exp".to_string(), id);
        self.func_sigs.insert("runtime_math_exp".to_string(), sig_math_exp);

        // runtime_math_log(x: f64) -> f64
        let mut sig_math_log = Signature::new(call_conv);
        sig_math_log.params.push(AbiParam::new(types::F64));
        sig_math_log.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_log", Linkage::Import, &sig_math_log)
            .map_err(|e| format!("Failed to declare runtime_math_log: {}", e))?;
        self.func_ids.insert("runtime_math_log".to_string(), id);
        self.func_sigs.insert("runtime_math_log".to_string(), sig_math_log);

        // runtime_math_sqrt(x: f64) -> f64
        let mut sig_math_sqrt = Signature::new(call_conv);
        sig_math_sqrt.params.push(AbiParam::new(types::F64));
        sig_math_sqrt.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_sqrt", Linkage::Import, &sig_math_sqrt)
            .map_err(|e| format!("Failed to declare runtime_math_sqrt: {}", e))?;
        self.func_ids.insert("runtime_math_sqrt".to_string(), id);
        self.func_sigs.insert("runtime_math_sqrt".to_string(), sig_math_sqrt);

        // runtime_math_tan(x: f64) -> f64
        let mut sig_math_tan = Signature::new(call_conv);
        sig_math_tan.params.push(AbiParam::new(types::F64));
        sig_math_tan.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_tan", Linkage::Import, &sig_math_tan)
            .map_err(|e| format!("Failed to declare runtime_math_tan: {}", e))?;
        self.func_ids.insert("runtime_math_tan".to_string(), id);
        self.func_sigs.insert("runtime_math_tan".to_string(), sig_math_tan);

        // runtime_math_atan(x: f64) -> f64
        let mut sig_math_atan = Signature::new(call_conv);
        sig_math_atan.params.push(AbiParam::new(types::F64));
        sig_math_atan.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_atan", Linkage::Import, &sig_math_atan)
            .map_err(|e| format!("Failed to declare runtime_math_atan: {}", e))?;
        self.func_ids.insert("runtime_math_atan".to_string(), id);
        self.func_sigs.insert("runtime_math_atan".to_string(), sig_math_atan);

        // runtime_math_fabs(x: f64) -> f64
        let mut sig_math_fabs = Signature::new(call_conv);
        sig_math_fabs.params.push(AbiParam::new(types::F64));
        sig_math_fabs.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_fabs", Linkage::Import, &sig_math_fabs)
            .map_err(|e| format!("Failed to declare runtime_math_fabs: {}", e))?;
        self.func_ids.insert("runtime_math_fabs".to_string(), id);
        self.func_sigs.insert("runtime_math_fabs".to_string(), sig_math_fabs);

        // runtime_math_asin(x: f64) -> f64
        let mut sig_math_asin = Signature::new(call_conv);
        sig_math_asin.params.push(AbiParam::new(types::F64));
        sig_math_asin.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_asin", Linkage::Import, &sig_math_asin)
            .map_err(|e| format!("Failed to declare runtime_math_asin: {}", e))?;
        self.func_ids.insert("runtime_math_asin".to_string(), id);
        self.func_sigs.insert("runtime_math_asin".to_string(), sig_math_asin);

        // runtime_math_acos(x: f64) -> f64
        let mut sig_math_acos = Signature::new(call_conv);
        sig_math_acos.params.push(AbiParam::new(types::F64));
        sig_math_acos.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_acos", Linkage::Import, &sig_math_acos)
            .map_err(|e| format!("Failed to declare runtime_math_acos: {}", e))?;
        self.func_ids.insert("runtime_math_acos".to_string(), id);
        self.func_sigs.insert("runtime_math_acos".to_string(), sig_math_acos);

        // runtime_math_sinh(x: f64) -> f64
        let mut sig_math_sinh = Signature::new(call_conv);
        sig_math_sinh.params.push(AbiParam::new(types::F64));
        sig_math_sinh.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_sinh", Linkage::Import, &sig_math_sinh)
            .map_err(|e| format!("Failed to declare runtime_math_sinh: {}", e))?;
        self.func_ids.insert("runtime_math_sinh".to_string(), id);
        self.func_sigs.insert("runtime_math_sinh".to_string(), sig_math_sinh);

        // runtime_math_cosh(x: f64) -> f64
        let mut sig_math_cosh = Signature::new(call_conv);
        sig_math_cosh.params.push(AbiParam::new(types::F64));
        sig_math_cosh.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_cosh", Linkage::Import, &sig_math_cosh)
            .map_err(|e| format!("Failed to declare runtime_math_cosh: {}", e))?;
        self.func_ids.insert("runtime_math_cosh".to_string(), id);
        self.func_sigs.insert("runtime_math_cosh".to_string(), sig_math_cosh);

        // runtime_math_tanh(x: f64) -> f64
        let mut sig_math_tanh = Signature::new(call_conv);
        sig_math_tanh.params.push(AbiParam::new(types::F64));
        sig_math_tanh.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_tanh", Linkage::Import, &sig_math_tanh)
            .map_err(|e| format!("Failed to declare runtime_math_tanh: {}", e))?;
        self.func_ids.insert("runtime_math_tanh".to_string(), id);
        self.func_sigs.insert("runtime_math_tanh".to_string(), sig_math_tanh);

        // runtime_math_asinh(x: f64) -> f64
        let mut sig_math_asinh = Signature::new(call_conv);
        sig_math_asinh.params.push(AbiParam::new(types::F64));
        sig_math_asinh.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_asinh", Linkage::Import, &sig_math_asinh)
            .map_err(|e| format!("Failed to declare runtime_math_asinh: {}", e))?;
        self.func_ids.insert("runtime_math_asinh".to_string(), id);
        self.func_sigs.insert("runtime_math_asinh".to_string(), sig_math_asinh);

        // runtime_math_acosh(x: f64) -> f64
        let mut sig_math_acosh = Signature::new(call_conv);
        sig_math_acosh.params.push(AbiParam::new(types::F64));
        sig_math_acosh.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_acosh", Linkage::Import, &sig_math_acosh)
            .map_err(|e| format!("Failed to declare runtime_math_acosh: {}", e))?;
        self.func_ids.insert("runtime_math_acosh".to_string(), id);
        self.func_sigs.insert("runtime_math_acosh".to_string(), sig_math_acosh);

        // runtime_math_atanh(x: f64) -> f64
        let mut sig_math_atanh = Signature::new(call_conv);
        sig_math_atanh.params.push(AbiParam::new(types::F64));
        sig_math_atanh.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_atanh", Linkage::Import, &sig_math_atanh)
            .map_err(|e| format!("Failed to declare runtime_math_atanh: {}", e))?;
        self.func_ids.insert("runtime_math_atanh".to_string(), id);
        self.func_sigs.insert("runtime_math_atanh".to_string(), sig_math_atanh);

        // runtime_math_log2(x: f64) -> f64
        let mut sig_math_log2 = Signature::new(call_conv);
        sig_math_log2.params.push(AbiParam::new(types::F64));
        sig_math_log2.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_log2", Linkage::Import, &sig_math_log2)
            .map_err(|e| format!("Failed to declare runtime_math_log2: {}", e))?;
        self.func_ids.insert("runtime_math_log2".to_string(), id);
        self.func_sigs.insert("runtime_math_log2".to_string(), sig_math_log2);

        // runtime_math_log10(x: f64) -> f64
        let mut sig_math_log10 = Signature::new(call_conv);
        sig_math_log10.params.push(AbiParam::new(types::F64));
        sig_math_log10.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_log10", Linkage::Import, &sig_math_log10)
            .map_err(|e| format!("Failed to declare runtime_math_log10: {}", e))?;
        self.func_ids.insert("runtime_math_log10".to_string(), id);
        self.func_sigs.insert("runtime_math_log10".to_string(), sig_math_log10);

        // runtime_math_atan2(y: f64, x: f64) -> f64
        let mut sig_math_atan2 = Signature::new(call_conv);
        sig_math_atan2.params.push(AbiParam::new(types::F64));
        sig_math_atan2.params.push(AbiParam::new(types::F64));
        sig_math_atan2.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_math_atan2", Linkage::Import, &sig_math_atan2)
            .map_err(|e| format!("Failed to declare runtime_math_atan2: {}", e))?;
        self.func_ids.insert("runtime_math_atan2".to_string(), id);
        self.func_sigs.insert("runtime_math_atan2".to_string(), sig_math_atan2);

        // ==================== Autodiff Runtime Functions ====================

        // runtime_jacobian(func_ptr: i64, x_ptr: i64, m: i64) -> i64
        // n is read from array header at x_ptr[0]
        let mut sig_jacobian = Signature::new(call_conv);
        sig_jacobian.params.push(AbiParam::new(types::I64)); // func_ptr
        sig_jacobian.params.push(AbiParam::new(types::I64)); // x_ptr (with length header)
        sig_jacobian.params.push(AbiParam::new(types::I64)); // m (output dimension)
        sig_jacobian.returns.push(AbiParam::new(types::I64)); // result matrix ptr
        let id = self
            .jit_module
            .declare_function("runtime_jacobian", Linkage::Import, &sig_jacobian)
            .map_err(|e| format!("Failed to declare runtime_jacobian: {}", e))?;
        self.func_ids.insert("runtime_jacobian".to_string(), id);
        self.func_sigs.insert("runtime_jacobian".to_string(), sig_jacobian);

        // runtime_hessian(func_ptr: i64, x_ptr: i64) -> i64
        // n is read from array header at x_ptr[0]
        let mut sig_hessian = Signature::new(call_conv);
        sig_hessian.params.push(AbiParam::new(types::I64)); // func_ptr
        sig_hessian.params.push(AbiParam::new(types::I64)); // x_ptr (with length header)
        sig_hessian.returns.push(AbiParam::new(types::I64)); // result matrix ptr
        let id = self
            .jit_module
            .declare_function("runtime_hessian", Linkage::Import, &sig_hessian)
            .map_err(|e| format!("Failed to declare runtime_hessian: {}", e))?;
        self.func_ids.insert("runtime_hessian".to_string(), id);
        self.func_sigs.insert("runtime_hessian".to_string(), sig_hessian);

        // ==================== IO Effect Runtime Functions ====================

        // runtime_io_read_line() -> i64 (ptr to string)
        let mut sig_io_read_line = Signature::new(call_conv);
        sig_io_read_line.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_io_read_line", Linkage::Import, &sig_io_read_line)
            .map_err(|e| format!("Failed to declare runtime_io_read_line: {}", e))?;
        self.func_ids
            .insert("runtime_io_read_line".to_string(), id);
        self.func_sigs
            .insert("runtime_io_read_line".to_string(), sig_io_read_line);

        // runtime_io_read_file(path_ptr: i64) -> i64 (ptr to string)
        let mut sig_io_read_file = Signature::new(call_conv);
        sig_io_read_file.params.push(AbiParam::new(types::I64)); // path_ptr
        sig_io_read_file.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_io_read_file", Linkage::Import, &sig_io_read_file)
            .map_err(|e| format!("Failed to declare runtime_io_read_file: {}", e))?;
        self.func_ids
            .insert("runtime_io_read_file".to_string(), id);
        self.func_sigs
            .insert("runtime_io_read_file".to_string(), sig_io_read_file);

        // runtime_io_write_file(path_ptr: i64, data_ptr: i64) -> i64
        let mut sig_io_write_file = Signature::new(call_conv);
        sig_io_write_file.params.push(AbiParam::new(types::I64)); // path_ptr
        sig_io_write_file.params.push(AbiParam::new(types::I64)); // data_ptr
        sig_io_write_file.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_io_write_file", Linkage::Import, &sig_io_write_file)
            .map_err(|e| format!("Failed to declare runtime_io_write_file: {}", e))?;
        self.func_ids
            .insert("runtime_io_write_file".to_string(), id);
        self.func_sigs
            .insert("runtime_io_write_file".to_string(), sig_io_write_file);

        // runtime_io_append_file(path_ptr: i64, data_ptr: i64) -> i64
        let mut sig_io_append_file = Signature::new(call_conv);
        sig_io_append_file.params.push(AbiParam::new(types::I64)); // path_ptr
        sig_io_append_file.params.push(AbiParam::new(types::I64)); // data_ptr
        sig_io_append_file.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_io_append_file", Linkage::Import, &sig_io_append_file)
            .map_err(|e| format!("Failed to declare runtime_io_append_file: {}", e))?;
        self.func_ids
            .insert("runtime_io_append_file".to_string(), id);
        self.func_sigs
            .insert("runtime_io_append_file".to_string(), sig_io_append_file);

        // runtime_io_file_exists(path_ptr: i64) -> i64
        let mut sig_io_file_exists = Signature::new(call_conv);
        sig_io_file_exists.params.push(AbiParam::new(types::I64)); // path_ptr
        sig_io_file_exists.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_io_file_exists", Linkage::Import, &sig_io_file_exists)
            .map_err(|e| format!("Failed to declare runtime_io_file_exists: {}", e))?;
        self.func_ids
            .insert("runtime_io_file_exists".to_string(), id);
        self.func_sigs
            .insert("runtime_io_file_exists".to_string(), sig_io_file_exists);

        // ==================== Mut Effect Runtime Functions ====================

        // runtime_mut_get(name_ptr: i64) -> f64
        let mut sig_mut_get = Signature::new(call_conv);
        sig_mut_get.params.push(AbiParam::new(types::I64)); // name_ptr
        sig_mut_get.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_mut_get", Linkage::Import, &sig_mut_get)
            .map_err(|e| format!("Failed to declare runtime_mut_get: {}", e))?;
        self.func_ids.insert("runtime_mut_get".to_string(), id);
        self.func_sigs
            .insert("runtime_mut_get".to_string(), sig_mut_get);

        // runtime_mut_set(name_ptr: i64, value: f64) -> i64
        let mut sig_mut_set = Signature::new(call_conv);
        sig_mut_set.params.push(AbiParam::new(types::I64)); // name_ptr
        sig_mut_set.params.push(AbiParam::new(types::F64)); // value
        sig_mut_set.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_mut_set", Linkage::Import, &sig_mut_set)
            .map_err(|e| format!("Failed to declare runtime_mut_set: {}", e))?;
        self.func_ids.insert("runtime_mut_set".to_string(), id);
        self.func_sigs
            .insert("runtime_mut_set".to_string(), sig_mut_set);

        // runtime_mut_modify(name_ptr: i64, delta: f64) -> f64
        let mut sig_mut_modify = Signature::new(call_conv);
        sig_mut_modify.params.push(AbiParam::new(types::I64)); // name_ptr
        sig_mut_modify.params.push(AbiParam::new(types::F64)); // delta
        sig_mut_modify.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_mut_modify", Linkage::Import, &sig_mut_modify)
            .map_err(|e| format!("Failed to declare runtime_mut_modify: {}", e))?;
        self.func_ids
            .insert("runtime_mut_modify".to_string(), id);
        self.func_sigs
            .insert("runtime_mut_modify".to_string(), sig_mut_modify);

        // runtime_mut_clear() -> void
        let sig_mut_clear = Signature::new(call_conv);
        let id = self
            .jit_module
            .declare_function("runtime_mut_clear", Linkage::Import, &sig_mut_clear)
            .map_err(|e| format!("Failed to declare runtime_mut_clear: {}", e))?;
        self.func_ids.insert("runtime_mut_clear".to_string(), id);
        self.func_sigs
            .insert("runtime_mut_clear".to_string(), sig_mut_clear);

        // runtime_mut_exists(name_ptr: i64) -> i64
        let mut sig_mut_exists = Signature::new(call_conv);
        sig_mut_exists.params.push(AbiParam::new(types::I64)); // name_ptr
        sig_mut_exists.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_mut_exists", Linkage::Import, &sig_mut_exists)
            .map_err(|e| format!("Failed to declare runtime_mut_exists: {}", e))?;
        self.func_ids
            .insert("runtime_mut_exists".to_string(), id);
        self.func_sigs
            .insert("runtime_mut_exists".to_string(), sig_mut_exists);

        // runtime_mut_delete(name_ptr: i64) -> f64
        let mut sig_mut_delete = Signature::new(call_conv);
        sig_mut_delete.params.push(AbiParam::new(types::I64)); // name_ptr
        sig_mut_delete.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_mut_delete", Linkage::Import, &sig_mut_delete)
            .map_err(|e| format!("Failed to declare runtime_mut_delete: {}", e))?;
        self.func_ids
            .insert("runtime_mut_delete".to_string(), id);
        self.func_sigs
            .insert("runtime_mut_delete".to_string(), sig_mut_delete);

        // ==================== Alloc Effect Runtime Functions ====================

        // runtime_alloc(size: i64) -> i64
        let mut sig_alloc = Signature::new(call_conv);
        sig_alloc.params.push(AbiParam::new(types::I64)); // size
        sig_alloc.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_alloc", Linkage::Import, &sig_alloc)
            .map_err(|e| format!("Failed to declare runtime_alloc: {}", e))?;
        self.func_ids.insert("runtime_alloc".to_string(), id);
        self.func_sigs.insert("runtime_alloc".to_string(), sig_alloc);

        // runtime_alloc_array(count: i64, elem_size: i64) -> i64
        let mut sig_alloc_array = Signature::new(call_conv);
        sig_alloc_array.params.push(AbiParam::new(types::I64)); // count
        sig_alloc_array.params.push(AbiParam::new(types::I64)); // elem_size
        sig_alloc_array.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_alloc_array", Linkage::Import, &sig_alloc_array)
            .map_err(|e| format!("Failed to declare runtime_alloc_array: {}", e))?;
        self.func_ids
            .insert("runtime_alloc_array".to_string(), id);
        self.func_sigs
            .insert("runtime_alloc_array".to_string(), sig_alloc_array);

        // runtime_dealloc(ptr: i64) -> i64
        let mut sig_dealloc = Signature::new(call_conv);
        sig_dealloc.params.push(AbiParam::new(types::I64)); // ptr
        sig_dealloc.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_dealloc", Linkage::Import, &sig_dealloc)
            .map_err(|e| format!("Failed to declare runtime_dealloc: {}", e))?;
        self.func_ids.insert("runtime_dealloc".to_string(), id);
        self.func_sigs
            .insert("runtime_dealloc".to_string(), sig_dealloc);

        // runtime_realloc(ptr: i64, new_size: i64) -> i64
        let mut sig_realloc = Signature::new(call_conv);
        sig_realloc.params.push(AbiParam::new(types::I64)); // ptr
        sig_realloc.params.push(AbiParam::new(types::I64)); // new_size
        sig_realloc.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_realloc", Linkage::Import, &sig_realloc)
            .map_err(|e| format!("Failed to declare runtime_realloc: {}", e))?;
        self.func_ids.insert("runtime_realloc".to_string(), id);
        self.func_sigs
            .insert("runtime_realloc".to_string(), sig_realloc);

        // runtime_alloc_size_of(ptr: i64) -> i64
        let mut sig_alloc_size_of = Signature::new(call_conv);
        sig_alloc_size_of.params.push(AbiParam::new(types::I64)); // ptr
        sig_alloc_size_of.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_alloc_size_of", Linkage::Import, &sig_alloc_size_of)
            .map_err(|e| format!("Failed to declare runtime_alloc_size_of: {}", e))?;
        self.func_ids
            .insert("runtime_alloc_size_of".to_string(), id);
        self.func_sigs
            .insert("runtime_alloc_size_of".to_string(), sig_alloc_size_of);

        // runtime_alloc_clear() -> void
        let sig_alloc_clear = Signature::new(call_conv);
        let id = self
            .jit_module
            .declare_function("runtime_alloc_clear", Linkage::Import, &sig_alloc_clear)
            .map_err(|e| format!("Failed to declare runtime_alloc_clear: {}", e))?;
        self.func_ids
            .insert("runtime_alloc_clear".to_string(), id);
        self.func_sigs
            .insert("runtime_alloc_clear".to_string(), sig_alloc_clear);

        // runtime_alloc_total() -> i64
        let mut sig_alloc_total = Signature::new(call_conv);
        sig_alloc_total.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_alloc_total", Linkage::Import, &sig_alloc_total)
            .map_err(|e| format!("Failed to declare runtime_alloc_total: {}", e))?;
        self.func_ids
            .insert("runtime_alloc_total".to_string(), id);
        self.func_sigs
            .insert("runtime_alloc_total".to_string(), sig_alloc_total);

        // runtime_alloc_count() -> i64
        let mut sig_alloc_count = Signature::new(call_conv);
        sig_alloc_count.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_alloc_count", Linkage::Import, &sig_alloc_count)
            .map_err(|e| format!("Failed to declare runtime_alloc_count: {}", e))?;
        self.func_ids
            .insert("runtime_alloc_count".to_string(), id);
        self.func_sigs
            .insert("runtime_alloc_count".to_string(), sig_alloc_count);

        // ==================== Handler Stack Functions ====================

        // runtime_handler_push(effect_ptr, name_ptr, handler_id) -> i64
        let mut sig_handler_push = Signature::new(call_conv);
        sig_handler_push.params.push(AbiParam::new(types::I64)); // effect_ptr
        sig_handler_push.params.push(AbiParam::new(types::I64)); // name_ptr
        sig_handler_push.params.push(AbiParam::new(types::I32)); // handler_id
        sig_handler_push.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_handler_push", Linkage::Import, &sig_handler_push)
            .map_err(|e| format!("Failed to declare runtime_handler_push: {}", e))?;
        self.func_ids.insert("runtime_handler_push".to_string(), id);
        self.func_sigs.insert("runtime_handler_push".to_string(), sig_handler_push);

        // runtime_handler_pop() -> i64
        let mut sig_handler_pop = Signature::new(call_conv);
        sig_handler_pop.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_handler_pop", Linkage::Import, &sig_handler_pop)
            .map_err(|e| format!("Failed to declare runtime_handler_pop: {}", e))?;
        self.func_ids.insert("runtime_handler_pop".to_string(), id);
        self.func_sigs.insert("runtime_handler_pop".to_string(), sig_handler_pop);

        // runtime_handler_depth() -> i64
        let mut sig_handler_depth = Signature::new(call_conv);
        sig_handler_depth.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_handler_depth", Linkage::Import, &sig_handler_depth)
            .map_err(|e| format!("Failed to declare runtime_handler_depth: {}", e))?;
        self.func_ids.insert("runtime_handler_depth".to_string(), id);
        self.func_sigs.insert("runtime_handler_depth".to_string(), sig_handler_depth);

        // runtime_handler_has(effect_ptr) -> i64
        let mut sig_handler_has = Signature::new(call_conv);
        sig_handler_has.params.push(AbiParam::new(types::I64));
        sig_handler_has.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_handler_has", Linkage::Import, &sig_handler_has)
            .map_err(|e| format!("Failed to declare runtime_handler_has: {}", e))?;
        self.func_ids.insert("runtime_handler_has".to_string(), id);
        self.func_sigs.insert("runtime_handler_has".to_string(), sig_handler_has);

        // runtime_handler_dispatch(effect_ptr, op_ptr, args_ptr, args_len) -> f64
        let mut sig_handler_dispatch = Signature::new(call_conv);
        sig_handler_dispatch.params.push(AbiParam::new(types::I64)); // effect_ptr
        sig_handler_dispatch.params.push(AbiParam::new(types::I64)); // op_ptr
        sig_handler_dispatch.params.push(AbiParam::new(types::I64)); // args_ptr
        sig_handler_dispatch.params.push(AbiParam::new(types::I64)); // args_len
        sig_handler_dispatch.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_handler_dispatch", Linkage::Import, &sig_handler_dispatch)
            .map_err(|e| format!("Failed to declare runtime_handler_dispatch: {}", e))?;
        self.func_ids.insert("runtime_handler_dispatch".to_string(), id);
        self.func_sigs.insert("runtime_handler_dispatch".to_string(), sig_handler_dispatch);

        // ==================== Continuation Functions ====================

        // runtime_continuation_create(effect_ptr, op_ptr, args_ptr, args_len) -> i64
        let mut sig_cont_create = Signature::new(call_conv);
        sig_cont_create.params.push(AbiParam::new(types::I64));
        sig_cont_create.params.push(AbiParam::new(types::I64));
        sig_cont_create.params.push(AbiParam::new(types::I64));
        sig_cont_create.params.push(AbiParam::new(types::I64));
        sig_cont_create.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_continuation_create", Linkage::Import, &sig_cont_create)
            .map_err(|e| format!("Failed to declare runtime_continuation_create: {}", e))?;
        self.func_ids.insert("runtime_continuation_create".to_string(), id);
        self.func_sigs.insert("runtime_continuation_create".to_string(), sig_cont_create);

        // runtime_continuation_resume(continuation_id, value) -> i64
        let mut sig_cont_resume = Signature::new(call_conv);
        sig_cont_resume.params.push(AbiParam::new(types::I64));
        sig_cont_resume.params.push(AbiParam::new(types::F64));
        sig_cont_resume.returns.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_continuation_resume", Linkage::Import, &sig_cont_resume)
            .map_err(|e| format!("Failed to declare runtime_continuation_resume: {}", e))?;
        self.func_ids.insert("runtime_continuation_resume".to_string(), id);
        self.func_sigs.insert("runtime_continuation_resume".to_string(), sig_cont_resume);

        // runtime_continuation_get_value() -> f64
        let mut sig_cont_get = Signature::new(call_conv);
        sig_cont_get.returns.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_continuation_get_value", Linkage::Import, &sig_cont_get)
            .map_err(|e| format!("Failed to declare runtime_continuation_get_value: {}", e))?;
        self.func_ids.insert("runtime_continuation_get_value".to_string(), id);
        self.func_sigs.insert("runtime_continuation_get_value".to_string(), sig_cont_get);

        // runtime_continuation_clear() -> void
        let sig_cont_clear = Signature::new(call_conv);
        let id = self
            .jit_module
            .declare_function("runtime_continuation_clear", Linkage::Import, &sig_cont_clear)
            .map_err(|e| format!("Failed to declare runtime_continuation_clear: {}", e))?;
        self.func_ids.insert("runtime_continuation_clear".to_string(), id);
        self.func_sigs.insert("runtime_continuation_clear".to_string(), sig_cont_clear);

        Ok(())
    }

    fn create_signature(&self, func: &HlirFunction) -> Signature {
        let call_conv = self.jit_module.isa().default_call_conv();
        let mut sig = Signature::new(call_conv);

        for param in &func.params {
            sig.params
                .push(AbiParam::new(hlir_to_cranelift_type(&param.ty)));
        }

        if func.return_type != HlirType::Void {
            sig.returns
                .push(AbiParam::new(hlir_to_cranelift_type(&func.return_type)));
        }

        sig
    }

    fn compile_function(&mut self, func: &HlirFunction) -> Result<(), String> {
        let func_id = self.func_ids[&func.name];

        // Create function signature
        self.ctx.func.signature = self.create_signature(func);
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Build function body
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);

            // Collect func_refs we need to declare
            let mut needed_funcs: Vec<(String, FuncId)> = Vec::new();
            for (name, &id) in &self.func_ids {
                needed_funcs.push((name.clone(), id));
            }

            // Declare functions in this function's namespace
            let mut local_func_refs = HashMap::new();
            for (name, id) in &needed_funcs {
                if let Some(sig) = self.func_sigs.get(name) {
                    let local_ref = self.jit_module.declare_func_in_func(*id, builder.func);
                    local_func_refs.insert(name.clone(), local_ref);
                }
            }

            translate_function(&mut builder, func, &local_func_refs)?;
            builder.finalize();
        }

        // Compile the function
        self.jit_module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define function {}: {}", func.name, e))?;

        self.jit_module.clear_context(&mut self.ctx);

        Ok(())
    }

    fn finalize(mut self) -> Result<CompiledModule, String> {
        self.jit_module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize: {}", e))?;

        let mut functions = HashMap::new();
        // Only get function pointers for exported (user-defined) functions, not imported runtime functions
        for name in &self.exported_funcs {
            if let Some(&func_id) = self.func_ids.get(name) {
                let ptr = self.jit_module.get_finalized_function(func_id);
                functions.insert(name.clone(), ptr);
            }
        }

        Ok(CompiledModule {
            jit_module: self.jit_module,
            functions,
        })
    }
}

/// Convert HLIR type to Cranelift type
#[cfg(feature = "jit")]
fn hlir_to_cranelift_type(ty: &HlirType) -> types::Type {
    match ty {
        HlirType::Void => types::I64, // Use I64 for void to avoid issues
        HlirType::Bool => types::I8,
        HlirType::I8 | HlirType::U8 => types::I8,
        HlirType::I16 | HlirType::U16 => types::I16,
        HlirType::I32 | HlirType::U32 => types::I32,
        HlirType::I64 | HlirType::U64 => types::I64,
        HlirType::I128 | HlirType::U128 => types::I128,
        HlirType::F32 => types::F32,
        HlirType::F64 => types::F64,
        HlirType::Ptr(_) => types::I64,
        HlirType::Array(_, _) => types::I64, // Pointer to array
        HlirType::Struct(_) => types::I64,   // Pointer to struct
        HlirType::Tuple(_) => types::I64,    // Pointer to tuple or packed
        HlirType::Function { .. } => types::I64, // Function pointer
        // Linear algebra types - use SIMD vector types for performance
        // vec2: 2x f32 = 64 bits, but we use F32X4 for alignment
        HlirType::Vec2 => types::F32X4,
        // vec3: 3x f32, padded to 4x f32 for SIMD (128 bits)
        HlirType::Vec3 => types::F32X4,
        // vec4: 4x f32 = 128 bits
        HlirType::Vec4 => types::F32X4,
        // mat2: 2x2 = 4 floats, fits in F32X4
        HlirType::Mat2 => types::F32X4,
        // mat3: 3x3 = 9 floats, use 3x F32X4 (12 floats, 3 wasted)
        // For now, represent as pointer to data
        HlirType::Mat3 => types::I64,
        // mat4: 4x4 = 16 floats = 4x F32X4
        // For now, represent as pointer to data
        HlirType::Mat4 => types::I64,
        // quat: 4x f32 = 128 bits, same as vec4
        HlirType::Quat => types::F32X4,
        // dual: pointer to stack-allocated 16-byte struct (value: f64, derivative: f64)
        // We use a pointer because SIMD F64X2 causes issues with Cranelift
        HlirType::Dual => types::I64,
    }
}

/// Translate an entire HLIR function to Cranelift IR
#[cfg(feature = "jit")]
fn translate_function(
    builder: &mut FunctionBuilder,
    func: &HlirFunction,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
) -> Result<(), String> {
    let mut values: HashMap<ValueId, cranelift_codegen::ir::Value> = HashMap::new();
    let mut blocks: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
    // Track which ValueIds are string constants (for print handling)
    let mut string_values: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

    // Create all blocks first
    for block in &func.blocks {
        let cl_block = builder.create_block();
        blocks.insert(block.id, cl_block);
    }

    // Entry block parameters (function arguments)
    if let Some(entry) = func.blocks.first() {
        let entry_block = blocks[&entry.id];
        builder.switch_to_block(entry_block);

        // Add function parameters
        for param in &func.params {
            let ty = hlir_to_cranelift_type(&param.ty);
            let val = builder.append_block_param(entry_block, ty);
            values.insert(param.value, val);
        }
    }

    // Translate each block (skip entry block switch since we're already there)
    let mut first = true;
    for block in &func.blocks {
        translate_block(
            builder,
            block,
            &blocks,
            &mut values,
            &mut string_values,
            func_refs,
            first,
        )?;
        first = false;
    }

    // Seal all blocks at the end
    builder.seal_all_blocks();

    Ok(())
}

#[cfg(feature = "jit")]
fn translate_block(
    builder: &mut FunctionBuilder,
    block: &HlirBlock,
    blocks: &HashMap<BlockId, cranelift_codegen::ir::Block>,
    values: &mut HashMap<ValueId, cranelift_codegen::ir::Value>,
    string_values: &mut std::collections::HashSet<ValueId>,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
    is_entry: bool,
) -> Result<(), String> {
    let cl_block = blocks[&block.id];

    // Only switch if not already on this block (entry block is already active)
    if !is_entry {
        builder.switch_to_block(cl_block);
    }

    // Translate instructions
    for instr in &block.instructions {
        let result = translate_instruction(builder, instr, values, string_values, func_refs)?;
        if let (Some(res_id), Some(val)) = (instr.result, result) {
            values.insert(res_id, val);
        }
    }

    // Translate terminator
    translate_terminator(builder, &block.terminator, blocks, values)?;

    Ok(())
}

#[cfg(feature = "jit")]
fn translate_instruction(
    builder: &mut FunctionBuilder,
    instr: &crate::hlir::HlirInstr,
    values: &HashMap<ValueId, cranelift_codegen::ir::Value>,
    string_values: &mut std::collections::HashSet<ValueId>,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
) -> Result<Option<cranelift_codegen::ir::Value>, String> {
    let ty = hlir_to_cranelift_type(&instr.ty);

    match &instr.op {
        Op::Const(constant) => {
            // Track string constants for proper print handling
            if let HlirConstant::String(_) = constant {
                if let Some(result_id) = instr.result {
                    string_values.insert(result_id);
                }
            }
            let val = translate_constant(builder, constant, &instr.ty, func_refs)?;
            Ok(Some(val))
        }

        Op::Copy(src) => {
            let src_val = get_value(values, *src)?;
            Ok(Some(src_val))
        }

        Op::Binary { op, left, right } => {
            let lhs = get_value(values, *left)?;
            let rhs = get_value(values, *right)?;
            let result = translate_binary_op(builder, *op, lhs, rhs, &instr.ty)?;
            Ok(Some(result))
        }

        Op::Unary { op, operand } => {
            let val = get_value(values, *operand)?;
            let result = translate_unary_op(builder, *op, val, &instr.ty)?;
            Ok(Some(result))
        }

        Op::CallDirect { name, args } => {
            let arg_vals: Vec<_> = args
                .iter()
                .map(|a| get_value(values, *a))
                .collect::<Result<_, _>>()?;

            // Handle print/println specially by routing to runtime functions
            if name == "print" || name == "println" {
                for (i, arg_val) in arg_vals.iter().enumerate() {
                    let arg_type = builder.func.dfg.value_type(*arg_val);
                    let arg_id = args[i];

                    // Check if this argument is a string constant
                    if string_values.contains(&arg_id) {
                        // Use runtime_print_cstr for string constants
                        if let Some(&func_ref) = func_refs.get("runtime_print_cstr") {
                            builder.ins().call(func_ref, &[*arg_val]);
                        }
                    } else {
                        // Choose runtime function based on argument type
                        let runtime_func = if arg_type == types::F64 || arg_type == types::F32 {
                            "runtime_print_f64"
                        } else if arg_type == types::I8 {
                            "runtime_print_bool"
                        } else {
                            "runtime_print_i64"
                        };

                        if let Some(&func_ref) = func_refs.get(runtime_func) {
                            // Convert argument to expected type if needed
                            let converted_arg = if runtime_func == "runtime_print_f64"
                                && arg_type == types::F32
                            {
                                builder.ins().fpromote(types::F64, *arg_val)
                            } else if runtime_func == "runtime_print_i64" && arg_type != types::I64
                            {
                                if arg_type.is_int() && arg_type.bits() < 64 {
                                    builder.ins().sextend(types::I64, *arg_val)
                                } else {
                                    *arg_val
                                }
                            } else {
                                *arg_val
                            };
                            builder.ins().call(func_ref, &[converted_arg]);
                        }
                    }
                }

                // Add newline for println
                if name == "println" {
                    if let Some(&func_ref) = func_refs.get("runtime_print_newline") {
                        builder.ins().call(func_ref, &[]);
                    }
                }

                return Ok(None);
            }

            // Handle dual number builtins using stack allocation
            // Dual numbers are stored as 16-byte structs: [value: f64, derivative: f64]
            // dual(value, deriv) -> allocates stack slot, stores both, returns pointer
            // dual_value(ptr) -> loads f64 from offset 0
            // dual_deriv(ptr) -> loads f64 from offset 8
            if name == "dual" {
                if arg_vals.len() >= 2 {
                    // Allocate 16-byte stack slot for dual number
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            16, // 2 x f64 = 16 bytes
                            8,  // 8-byte alignment for f64
                        ),
                    );

                    // Get pointer to stack slot
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);

                    // Store value at offset 0
                    builder.ins().store(
                        cranelift_codegen::ir::MemFlags::new(),
                        arg_vals[0],
                        ptr,
                        0,
                    );

                    // Store derivative at offset 8
                    builder.ins().store(
                        cranelift_codegen::ir::MemFlags::new(),
                        arg_vals[1],
                        ptr,
                        8,
                    );

                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            if name == "dual_value" {
                if !arg_vals.is_empty() {
                    // Load value from offset 0 of dual pointer
                    let value = builder.ins().load(
                        types::F64,
                        cranelift_codegen::ir::MemFlags::new(),
                        arg_vals[0],
                        0,
                    );
                    return Ok(Some(value));
                }
                return Ok(None);
            }

            if name == "dual_deriv" {
                if !arg_vals.is_empty() {
                    // Load derivative from offset 8 of dual pointer
                    let deriv = builder.ins().load(
                        types::F64,
                        cranelift_codegen::ir::MemFlags::new(),
                        arg_vals[0],
                        8,
                    );
                    return Ok(Some(deriv));
                }
                let zero = builder.ins().f64const(0.0);
                return Ok(Some(zero));
            }

            // dual_add(a, b) = (a.val + b.val, a.der + b.der)
            if name == "dual_add" {
                if arg_vals.len() >= 2 {
                    let a_ptr = arg_vals[0];
                    let b_ptr = arg_vals[1];

                    // Load values and derivatives
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);
                    let b_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 0);
                    let b_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 8);

                    // Compute result
                    let result_val = builder.ins().fadd(a_val, b_val);
                    let result_der = builder.ins().fadd(a_der, b_der);

                    // Allocate and store result
                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_sub(a, b) = (a.val - b.val, a.der - b.der)
            if name == "dual_sub" {
                if arg_vals.len() >= 2 {
                    let a_ptr = arg_vals[0];
                    let b_ptr = arg_vals[1];

                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);
                    let b_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 0);
                    let b_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 8);

                    let result_val = builder.ins().fsub(a_val, b_val);
                    let result_der = builder.ins().fsub(a_der, b_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_mul(a, b) = (a.val * b.val, a.der * b.val + a.val * b.der) [product rule]
            if name == "dual_mul" {
                if arg_vals.len() >= 2 {
                    let a_ptr = arg_vals[0];
                    let b_ptr = arg_vals[1];

                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);
                    let b_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 0);
                    let b_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 8);

                    // Product rule: (a * b)' = a' * b + a * b'
                    let result_val = builder.ins().fmul(a_val, b_val);
                    let term1 = builder.ins().fmul(a_der, b_val);
                    let term2 = builder.ins().fmul(a_val, b_der);
                    let result_der = builder.ins().fadd(term1, term2);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_div(a, b) = (a.val / b.val, (a.der * b.val - a.val * b.der) / b.val^2) [quotient rule]
            if name == "dual_div" {
                if arg_vals.len() >= 2 {
                    let a_ptr = arg_vals[0];
                    let b_ptr = arg_vals[1];

                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);
                    let b_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 0);
                    let b_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), b_ptr, 8);

                    // Quotient rule: (a / b)' = (a' * b - a * b') / b^2
                    let result_val = builder.ins().fdiv(a_val, b_val);
                    let term1 = builder.ins().fmul(a_der, b_val);
                    let term2 = builder.ins().fmul(a_val, b_der);
                    let numerator = builder.ins().fsub(term1, term2);
                    let b_squared = builder.ins().fmul(b_val, b_val);
                    let result_der = builder.ins().fdiv(numerator, b_squared);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_sin(a) = (sin(a.val), cos(a.val) * a.der) [chain rule]
            if name == "dual_sin" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    // Call runtime_math_sin and runtime_math_cos
                    let sin_func = func_refs.get("runtime_math_sin").ok_or("runtime_math_sin not found")?;
                    let cos_func = func_refs.get("runtime_math_cos").ok_or("runtime_math_cos not found")?;

                    let sin_call = builder.ins().call(*sin_func, &[a_val]);
                    let sin_val = builder.inst_results(sin_call)[0];

                    let cos_call = builder.ins().call(*cos_func, &[a_val]);
                    let cos_val = builder.inst_results(cos_call)[0];

                    // Derivative: cos(a.val) * a.der
                    let result_der = builder.ins().fmul(cos_val, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), sin_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_cos(a) = (cos(a.val), -sin(a.val) * a.der) [chain rule]
            if name == "dual_cos" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let sin_func = func_refs.get("runtime_math_sin").ok_or("runtime_math_sin not found")?;
                    let cos_func = func_refs.get("runtime_math_cos").ok_or("runtime_math_cos not found")?;

                    let cos_call = builder.ins().call(*cos_func, &[a_val]);
                    let cos_val = builder.inst_results(cos_call)[0];

                    let sin_call = builder.ins().call(*sin_func, &[a_val]);
                    let sin_val = builder.inst_results(sin_call)[0];

                    // Derivative: -sin(a.val) * a.der
                    let neg_sin = builder.ins().fneg(sin_val);
                    let result_der = builder.ins().fmul(neg_sin, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), cos_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_exp(a) = (exp(a.val), exp(a.val) * a.der) [chain rule]
            if name == "dual_exp" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let exp_func = func_refs.get("runtime_math_exp").ok_or("runtime_math_exp not found")?;

                    let exp_call = builder.ins().call(*exp_func, &[a_val]);
                    let exp_val = builder.inst_results(exp_call)[0];

                    // Derivative: exp(a.val) * a.der
                    let result_der = builder.ins().fmul(exp_val, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), exp_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_log(a) = (log(a.val), a.der / a.val) [chain rule]
            if name == "dual_log" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let log_func = func_refs.get("runtime_math_log").ok_or("runtime_math_log not found")?;

                    let log_call = builder.ins().call(*log_func, &[a_val]);
                    let log_val = builder.inst_results(log_call)[0];

                    // Derivative: a.der / a.val
                    let result_der = builder.ins().fdiv(a_der, a_val);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), log_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_sqrt(a) = (sqrt(a.val), a.der / (2 * sqrt(a.val))) [chain rule]
            if name == "dual_sqrt" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let sqrt_func = func_refs.get("runtime_math_sqrt").ok_or("runtime_math_sqrt not found")?;

                    let sqrt_call = builder.ins().call(*sqrt_func, &[a_val]);
                    let sqrt_val = builder.inst_results(sqrt_call)[0];

                    // Derivative: a.der / (2 * sqrt(a.val))
                    let two = builder.ins().f64const(2.0);
                    let denom = builder.ins().fmul(two, sqrt_val);
                    let result_der = builder.ins().fdiv(a_der, denom);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), sqrt_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_pow(a, n) = (a.val^n, n * a.val^(n-1) * a.der) [power rule]
            // where n is an f64 constant (not a dual)
            if name == "dual_pow" {
                if arg_vals.len() >= 2 {
                    let a_ptr = arg_vals[0];
                    let n = arg_vals[1]; // n is a plain f64

                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let exp_func = func_refs.get("runtime_math_exp").ok_or("runtime_math_exp not found")?;
                    let log_func = func_refs.get("runtime_math_log").ok_or("runtime_math_log not found")?;

                    // a^n = exp(n * log(a))
                    let log_call = builder.ins().call(*log_func, &[a_val]);
                    let log_a = builder.inst_results(log_call)[0];
                    let n_log_a = builder.ins().fmul(n, log_a);
                    let exp_call = builder.ins().call(*exp_func, &[n_log_a]);
                    let pow_val = builder.inst_results(exp_call)[0];

                    // Derivative: n * a^(n-1) * a.der = n * (a^n / a) * a.der = n * a^n * a.der / a
                    // Simpler: (n * a^n / a) * a.der
                    let n_pow = builder.ins().fmul(n, pow_val);
                    let n_pow_over_a = builder.ins().fdiv(n_pow, a_val);
                    let result_der = builder.ins().fmul(n_pow_over_a, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), pow_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_tan(a) = (tan(a.val), a.der / cos²(a.val)) [chain rule: d/dx tan(x) = sec²(x)]
            if name == "dual_tan" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let tan_func = func_refs.get("runtime_math_tan").ok_or("runtime_math_tan not found")?;
                    let cos_func = func_refs.get("runtime_math_cos").ok_or("runtime_math_cos not found")?;

                    let tan_call = builder.ins().call(*tan_func, &[a_val]);
                    let tan_val = builder.inst_results(tan_call)[0];

                    // Derivative: a.der / cos²(a.val) = a.der * sec²(a.val)
                    let cos_call = builder.ins().call(*cos_func, &[a_val]);
                    let cos_val = builder.inst_results(cos_call)[0];
                    let cos_sq = builder.ins().fmul(cos_val, cos_val);
                    let result_der = builder.ins().fdiv(a_der, cos_sq);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), tan_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_atan(a) = (atan(a.val), a.der / (1 + a.val²)) [chain rule: d/dx atan(x) = 1/(1+x²)]
            if name == "dual_atan" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let atan_func = func_refs.get("runtime_math_atan").ok_or("runtime_math_atan not found")?;

                    let atan_call = builder.ins().call(*atan_func, &[a_val]);
                    let atan_val = builder.inst_results(atan_call)[0];

                    // Derivative: a.der / (1 + a.val²)
                    let one = builder.ins().f64const(1.0);
                    let a_sq = builder.ins().fmul(a_val, a_val);
                    let denom = builder.ins().fadd(one, a_sq);
                    let result_der = builder.ins().fdiv(a_der, denom);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), atan_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_abs(a) = (|a.val|, sign(a.val) * a.der) where sign(x) = x/|x|
            if name == "dual_abs" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let fabs_func = func_refs.get("runtime_math_fabs").ok_or("runtime_math_fabs not found")?;

                    let fabs_call = builder.ins().call(*fabs_func, &[a_val]);
                    let abs_val = builder.inst_results(fabs_call)[0];

                    // Derivative: sign(a.val) * a.der = (a.val / |a.val|) * a.der
                    let sign = builder.ins().fdiv(a_val, abs_val);
                    let result_der = builder.ins().fmul(sign, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), abs_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_asin(a) = (asin(a.val), a.der / sqrt(1 - a.val²)) [chain rule: d/dx asin(x) = 1/sqrt(1-x²)]
            if name == "dual_asin" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let asin_func = func_refs.get("runtime_math_asin").ok_or("runtime_math_asin not found")?;
                    let sqrt_func = func_refs.get("runtime_math_sqrt").ok_or("runtime_math_sqrt not found")?;

                    let asin_call = builder.ins().call(*asin_func, &[a_val]);
                    let asin_val = builder.inst_results(asin_call)[0];

                    // Derivative: a.der / sqrt(1 - a.val²)
                    let one = builder.ins().f64const(1.0);
                    let a_sq = builder.ins().fmul(a_val, a_val);
                    let one_minus_asq = builder.ins().fsub(one, a_sq);
                    let sqrt_call = builder.ins().call(*sqrt_func, &[one_minus_asq]);
                    let sqrt_val = builder.inst_results(sqrt_call)[0];
                    let result_der = builder.ins().fdiv(a_der, sqrt_val);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), asin_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_acos(a) = (acos(a.val), -a.der / sqrt(1 - a.val²)) [chain rule: d/dx acos(x) = -1/sqrt(1-x²)]
            if name == "dual_acos" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let acos_func = func_refs.get("runtime_math_acos").ok_or("runtime_math_acos not found")?;
                    let sqrt_func = func_refs.get("runtime_math_sqrt").ok_or("runtime_math_sqrt not found")?;

                    let acos_call = builder.ins().call(*acos_func, &[a_val]);
                    let acos_val = builder.inst_results(acos_call)[0];

                    // Derivative: -a.der / sqrt(1 - a.val²)
                    let one = builder.ins().f64const(1.0);
                    let a_sq = builder.ins().fmul(a_val, a_val);
                    let one_minus_asq = builder.ins().fsub(one, a_sq);
                    let sqrt_call = builder.ins().call(*sqrt_func, &[one_minus_asq]);
                    let sqrt_val = builder.inst_results(sqrt_call)[0];
                    let neg_a_der = builder.ins().fneg(a_der);
                    let result_der = builder.ins().fdiv(neg_a_der, sqrt_val);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), acos_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_sinh(a) = (sinh(a.val), cosh(a.val) * a.der) [chain rule: d/dx sinh(x) = cosh(x)]
            if name == "dual_sinh" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let sinh_func = func_refs.get("runtime_math_sinh").ok_or("runtime_math_sinh not found")?;
                    let cosh_func = func_refs.get("runtime_math_cosh").ok_or("runtime_math_cosh not found")?;

                    let sinh_call = builder.ins().call(*sinh_func, &[a_val]);
                    let sinh_val = builder.inst_results(sinh_call)[0];

                    // Derivative: cosh(a.val) * a.der
                    let cosh_call = builder.ins().call(*cosh_func, &[a_val]);
                    let cosh_val = builder.inst_results(cosh_call)[0];
                    let result_der = builder.ins().fmul(cosh_val, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), sinh_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_cosh(a) = (cosh(a.val), sinh(a.val) * a.der) [chain rule: d/dx cosh(x) = sinh(x)]
            if name == "dual_cosh" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let sinh_func = func_refs.get("runtime_math_sinh").ok_or("runtime_math_sinh not found")?;
                    let cosh_func = func_refs.get("runtime_math_cosh").ok_or("runtime_math_cosh not found")?;

                    let cosh_call = builder.ins().call(*cosh_func, &[a_val]);
                    let cosh_val = builder.inst_results(cosh_call)[0];

                    // Derivative: sinh(a.val) * a.der
                    let sinh_call = builder.ins().call(*sinh_func, &[a_val]);
                    let sinh_val = builder.inst_results(sinh_call)[0];
                    let result_der = builder.ins().fmul(sinh_val, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), cosh_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_tanh(a) = (tanh(a.val), (1 - tanh²(a.val)) * a.der) [chain rule: d/dx tanh(x) = sech²(x) = 1 - tanh²(x)]
            if name == "dual_tanh" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let tanh_func = func_refs.get("runtime_math_tanh").ok_or("runtime_math_tanh not found")?;

                    let tanh_call = builder.ins().call(*tanh_func, &[a_val]);
                    let tanh_val = builder.inst_results(tanh_call)[0];

                    let one = builder.ins().f64const(1.0);
                    let tanh_sq = builder.ins().fmul(tanh_val, tanh_val);
                    let sech_sq = builder.ins().fsub(one, tanh_sq);
                    let result_der = builder.ins().fmul(sech_sq, a_der);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), tanh_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_asinh(a) = (asinh(a.val), a.der / sqrt(a.val² + 1))
            if name == "dual_asinh" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let asinh_func = func_refs.get("runtime_math_asinh").ok_or("runtime_math_asinh not found")?;
                    let sqrt_func = func_refs.get("runtime_math_sqrt").ok_or("runtime_math_sqrt not found")?;

                    let asinh_call = builder.ins().call(*asinh_func, &[a_val]);
                    let asinh_val = builder.inst_results(asinh_call)[0];

                    let one = builder.ins().f64const(1.0);
                    let a_sq = builder.ins().fmul(a_val, a_val);
                    let a_sq_plus_one = builder.ins().fadd(a_sq, one);
                    let sqrt_call = builder.ins().call(*sqrt_func, &[a_sq_plus_one]);
                    let sqrt_val = builder.inst_results(sqrt_call)[0];
                    let result_der = builder.ins().fdiv(a_der, sqrt_val);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), asinh_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_acosh(a) = (acosh(a.val), a.der / sqrt(a.val² - 1))
            if name == "dual_acosh" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let acosh_func = func_refs.get("runtime_math_acosh").ok_or("runtime_math_acosh not found")?;
                    let sqrt_func = func_refs.get("runtime_math_sqrt").ok_or("runtime_math_sqrt not found")?;

                    let acosh_call = builder.ins().call(*acosh_func, &[a_val]);
                    let acosh_val = builder.inst_results(acosh_call)[0];

                    let one = builder.ins().f64const(1.0);
                    let a_sq = builder.ins().fmul(a_val, a_val);
                    let a_sq_minus_one = builder.ins().fsub(a_sq, one);
                    let sqrt_call = builder.ins().call(*sqrt_func, &[a_sq_minus_one]);
                    let sqrt_val = builder.inst_results(sqrt_call)[0];
                    let result_der = builder.ins().fdiv(a_der, sqrt_val);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), acosh_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_atanh(a) = (atanh(a.val), a.der / (1 - a.val²))
            if name == "dual_atanh" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let atanh_func = func_refs.get("runtime_math_atanh").ok_or("runtime_math_atanh not found")?;

                    let atanh_call = builder.ins().call(*atanh_func, &[a_val]);
                    let atanh_val = builder.inst_results(atanh_call)[0];

                    let one = builder.ins().f64const(1.0);
                    let a_sq = builder.ins().fmul(a_val, a_val);
                    let one_minus_asq = builder.ins().fsub(one, a_sq);
                    let result_der = builder.ins().fdiv(a_der, one_minus_asq);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), atanh_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_log2(a) = (log2(a.val), a.der / (a.val * ln(2)))
            if name == "dual_log2" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let log2_func = func_refs.get("runtime_math_log2").ok_or("runtime_math_log2 not found")?;

                    let log2_call = builder.ins().call(*log2_func, &[a_val]);
                    let log2_val = builder.inst_results(log2_call)[0];

                    // Derivative: a.der / (a.val * ln(2))
                    let ln2 = builder.ins().f64const(std::f64::consts::LN_2);
                    let denom = builder.ins().fmul(a_val, ln2);
                    let result_der = builder.ins().fdiv(a_der, denom);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), log2_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_log10(a) = (log10(a.val), a.der / (a.val * ln(10)))
            if name == "dual_log10" {
                if !arg_vals.is_empty() {
                    let a_ptr = arg_vals[0];
                    let a_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 0);
                    let a_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), a_ptr, 8);

                    let log10_func = func_refs.get("runtime_math_log10").ok_or("runtime_math_log10 not found")?;

                    let log10_call = builder.ins().call(*log10_func, &[a_val]);
                    let log10_val = builder.inst_results(log10_call)[0];

                    // Derivative: a.der / (a.val * ln(10))
                    let ln10 = builder.ins().f64const(std::f64::consts::LN_10);
                    let denom = builder.ins().fmul(a_val, ln10);
                    let result_der = builder.ins().fdiv(a_der, denom);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), log10_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // dual_atan2(y, x) = (atan2(y.val, x.val), (x.val * y.der - y.val * x.der) / (x.val² + y.val²))
            if name == "dual_atan2" {
                if arg_vals.len() >= 2 {
                    let y_ptr = arg_vals[0];
                    let x_ptr = arg_vals[1];

                    let y_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), y_ptr, 0);
                    let y_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), y_ptr, 8);
                    let x_val = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), x_ptr, 0);
                    let x_der = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), x_ptr, 8);

                    let atan2_func = func_refs.get("runtime_math_atan2").ok_or("runtime_math_atan2 not found")?;

                    let atan2_call = builder.ins().call(*atan2_func, &[y_val, x_val]);
                    let atan2_val = builder.inst_results(atan2_call)[0];

                    // Derivative: (x.val * y.der - y.val * x.der) / (x.val² + y.val²)
                    let x_sq = builder.ins().fmul(x_val, x_val);
                    let y_sq = builder.ins().fmul(y_val, y_val);
                    let denom = builder.ins().fadd(x_sq, y_sq);
                    let term1 = builder.ins().fmul(x_val, y_der);
                    let term2 = builder.ins().fmul(y_val, x_der);
                    let numer = builder.ins().fsub(term1, term2);
                    let result_der = builder.ins().fdiv(numer, denom);

                    let slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let ptr = builder.ins().stack_addr(types::I64, slot, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), atan2_val, ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), result_der, ptr, 8);
                    return Ok(Some(ptr));
                }
                return Ok(None);
            }

            // grad(f, x) = f'(x) where f: fn(Dual) -> Dual, x: f64
            // Implementation: create dual(x, 1.0), call f, extract derivative
            if name == "grad" {
                if arg_vals.len() >= 2 {
                    let func_ptr = arg_vals[0];  // fn(Dual) -> Dual
                    let x_val = arg_vals[1];     // f64 value

                    // Create dual(x, 1.0) on stack
                    let input_slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 16, 8,
                        ),
                    );
                    let input_ptr = builder.ins().stack_addr(types::I64, input_slot, 0);
                    let one = builder.ins().f64const(1.0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), x_val, input_ptr, 0);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), one, input_ptr, 8);

                    // Call f(dual) - signature: fn(Dual*) -> Dual*
                    let mut sig = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
                    sig.params.push(AbiParam::new(types::I64));  // Dual* input
                    sig.returns.push(AbiParam::new(types::I64)); // Dual* output

                    let sig_ref = builder.import_signature(sig);
                    let call = builder.ins().call_indirect(sig_ref, func_ptr, &[input_ptr]);
                    let result_ptr = builder.inst_results(call)[0];

                    // Extract derivative (offset 8) from result
                    let derivative = builder.ins().load(types::F64, cranelift_codegen::ir::MemFlags::new(), result_ptr, 8);
                    return Ok(Some(derivative));
                }
                return Ok(None);
            }

            // jacobian(f, x, m) - compute Jacobian matrix
            // f: fn([Dual]) -> [Dual], x: [f64] (with length header), m: output dim
            // n is read automatically from array header at x[0]
            // Returns pointer to m×n matrix (row-major)
            if name == "jacobian" {
                if arg_vals.len() >= 3 {
                    let func_ptr = arg_vals[0];
                    let x_ptr = arg_vals[1];
                    let m = arg_vals[2];

                    let jacobian_func = func_refs.get("runtime_jacobian").ok_or("runtime_jacobian not found")?;
                    let call = builder.ins().call(*jacobian_func, &[func_ptr, x_ptr, m]);
                    let result = builder.inst_results(call)[0];
                    return Ok(Some(result));
                }
                return Ok(None);
            }

            // hessian(f, x) - compute Hessian matrix
            // f: fn([Dual]) -> Dual, x: [f64] (with length header)
            // n is read automatically from array header at x[0]
            // Returns pointer to n×n matrix (row-major)
            if name == "hessian" {
                if arg_vals.len() >= 2 {
                    let func_ptr = arg_vals[0];
                    let x_ptr = arg_vals[1];

                    let hessian_func = func_refs.get("runtime_hessian").ok_or("runtime_hessian not found")?;
                    let call = builder.ins().call(*hessian_func, &[func_ptr, x_ptr]);
                    let result = builder.inst_results(call)[0];
                    return Ok(Some(result));
                }
                return Ok(None);
            }

            // array_len(arr) - get length of array from header
            // Arrays have layout: [length: i64, data...]
            if name == "array_len" {
                if !arg_vals.is_empty() {
                    let arr_ptr = arg_vals[0];
                    // Length is stored at offset 0
                    let len = builder.ins().load(types::I64, cranelift_codegen::ir::MemFlags::new(), arr_ptr, 0);
                    return Ok(Some(len));
                }
                return Ok(None);
            }

            // array_ptr(arr) - get pointer to array data (skip length header)
            // Returns pointer to first element (offset 8 from array start)
            if name == "array_ptr" {
                if !arg_vals.is_empty() {
                    let arr_ptr = arg_vals[0];
                    // Data starts at offset 8 (after length header)
                    let data_ptr = builder.ins().iadd_imm(arr_ptr, 8);
                    return Ok(Some(data_ptr));
                }
                return Ok(None);
            }

            // ptr_load_f64(addr) - load f64 from memory address
            if name == "ptr_load_f64" {
                if !arg_vals.is_empty() {
                    let addr = arg_vals[0];
                    let val = builder.ins().load(types::F64, MemFlags::new(), addr, 0);
                    return Ok(Some(val));
                }
                return Ok(None);
            }

            // ptr_store_f64(addr, value) - store f64 to memory address
            if name == "ptr_store_f64" {
                if arg_vals.len() >= 2 {
                    let addr = arg_vals[0];
                    let val = arg_vals[1];
                    builder.ins().store(MemFlags::new(), val, addr, 0);
                    return Ok(None);
                }
                return Ok(None);
            }

            if let Some(&func_ref) = func_refs.get(name) {
                let call = builder.ins().call(func_ref, &arg_vals);
                let results = builder.inst_results(call);
                if results.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(results[0]))
                }
            } else {
                // Unknown function - return zero
                let zero = builder.ins().iconst(ty, 0);
                Ok(Some(zero))
            }
        }

        Op::Call { func, args } => {
            let func_val = get_value(values, *func)?;
            let arg_vals: Vec<_> = args
                .iter()
                .map(|a| get_value(values, *a))
                .collect::<Result<_, _>>()?;

            // Indirect call - need a signature
            // For now, assume simple i64 -> i64 signature
            let mut sig = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
            for _ in &arg_vals {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));

            let sig_ref = builder.import_signature(sig);
            let call = builder.ins().call_indirect(sig_ref, func_val, &arg_vals);
            let results = builder.inst_results(call);
            if results.is_empty() {
                Ok(None)
            } else {
                Ok(Some(results[0]))
            }
        }

        Op::Load { ptr } => {
            let ptr_val = get_value(values, *ptr)?;
            let loaded = builder.ins().load(ty, MemFlags::new(), ptr_val, 0);
            Ok(Some(loaded))
        }

        Op::Store { ptr, value } => {
            let ptr_val = get_value(values, *ptr)?;
            let val = get_value(values, *value)?;
            builder.ins().store(MemFlags::new(), val, ptr_val, 0);
            Ok(None)
        }

        Op::Alloca { ty: alloc_ty } => {
            let size = alloc_ty.size_bits() / 8;
            let size = if size == 0 { 8 } else { size };
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size as u32,
                0,
            ));
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            Ok(Some(addr))
        }

        Op::GetFieldPtr { base, field } => {
            let base_val = get_value(values, *base)?;
            let offset = (*field * 8) as i64; // Assume 8-byte fields
            let ptr = builder.ins().iadd_imm(base_val, offset);
            Ok(Some(ptr))
        }

        Op::GetElementPtr { base, index } => {
            let base_val = get_value(values, *base)?;
            let idx_val = get_value(values, *index)?;
            let idx_val = match builder.func.dfg.value_type(idx_val) {
                types::I64 => idx_val,
                t if t.is_int() && t.bits() < 64 => builder.ins().sextend(types::I64, idx_val),
                _ => idx_val,
            };
            // Extract element size from pointer type (fixes issue #11)
            let elem_size = match &instr.ty {
                HlirType::Ptr(inner) => {
                    let bits = inner.size_bits();
                    if bits == 0 { 8 } else { bits / 8 }
                }
                _ => 8, // Fallback for non-pointer types
            } as i64;
            let offset = builder.ins().imul_imm(idx_val, elem_size);
            let ptr = builder.ins().iadd(base_val, offset);
            Ok(Some(ptr))
        }

        Op::Cast {
            value,
            source,
            target,
        } => {
            let val = get_value(values, *value)?;
            let target_ty = hlir_to_cranelift_type(target);
            let val_ty = builder.func.dfg.value_type(val);

            if val_ty == target_ty {
                Ok(Some(val))
            } else if val_ty.is_int() && target_ty.is_int() && val_ty.bits() < target_ty.bits() {
                // Integer widening: preserve signedness of the *source* type.
                let extended = if matches!(
                    source,
                    HlirType::I8 | HlirType::I16 | HlirType::I32 | HlirType::I64 | HlirType::I128
                ) {
                    builder.ins().sextend(target_ty, val)
                } else {
                    builder.ins().uextend(target_ty, val)
                };
                Ok(Some(extended))
            } else if !val_ty.is_int() && !target_ty.is_int() && val_ty.bits() < target_ty.bits() {
                Ok(Some(builder.ins().fpromote(target_ty, val)))
            } else {
                // Truncate / demote
                if target_ty.is_int() {
                    Ok(Some(builder.ins().ireduce(target_ty, val)))
                } else {
                    Ok(Some(builder.ins().fdemote(target_ty, val)))
                }
            }
        }

        Op::Phi { incoming } => {
            // Phi nodes should be handled as block parameters in Cranelift
            // For now, return first incoming value if available
            if let Some((_, first_val)) = incoming.first() {
                let val = get_value(values, *first_val)?;
                Ok(Some(val))
            } else {
                let zero = builder.ins().iconst(ty, 0);
                Ok(Some(zero))
            }
        }

        Op::ExtractValue { base, index } => {
            // For tuples/structs stored as aggregates
            let base_val = get_value(values, *base)?;
            // Simplified: treat as field access
            let offset = (*index * 8) as i32;
            let ptr = builder.ins().iadd_imm(base_val, offset as i64);
            let loaded = builder.ins().load(ty, MemFlags::new(), ptr, 0);
            Ok(Some(loaded))
        }

        Op::InsertValue { base, value, index } => {
            let base_val = get_value(values, *base)?;
            let val = get_value(values, *value)?;
            // Simplified: treat as field store
            let offset = (*index * 8) as i32;
            let ptr = builder.ins().iadd_imm(base_val, offset as i64);
            builder.ins().store(MemFlags::new(), val, ptr, 0);
            Ok(Some(base_val))
        }

        Op::Tuple(vals) => {
            // Tuples: no length header, just store values
            let size = (vals.len() * 8) as u32;
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size,
                0,
            ));
            let base = builder.ins().stack_addr(types::I64, slot, 0);

            for (i, v) in vals.iter().enumerate() {
                let val = get_value(values, *v)?;
                let offset = (i * 8) as i32;
                builder.ins().store(MemFlags::new(), val, base, offset);
            }

            Ok(Some(base))
        }

        Op::Array(vals) => {
            // Arrays: include length header at offset 0, data starts at offset 8
            // Layout: [length: i64, elem0, elem1, ...]
            let size = (vals.len() * 8 + 8) as u32; // +8 for length header
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size,
                8, // align to 8 bytes
            ));
            let base = builder.ins().stack_addr(types::I64, slot, 0);

            // Store length at offset 0
            let len_val = builder.ins().iconst(types::I64, vals.len() as i64);
            builder.ins().store(MemFlags::new(), len_val, base, 0);

            // Store elements starting at offset 8
            for (i, v) in vals.iter().enumerate() {
                let val = get_value(values, *v)?;
                let offset = (i * 8 + 8) as i32; // +8 to skip length header
                builder.ins().store(MemFlags::new(), val, base, offset);
            }

            Ok(Some(base))
        }

        Op::Struct { name: _, fields } => {
            // Similar to tuple
            let size = (fields.len() * 8) as u32;
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size,
                0,
            ));
            let base = builder.ins().stack_addr(types::I64, slot, 0);

            for (i, (_, v)) in fields.iter().enumerate() {
                let val = get_value(values, *v)?;
                let offset = (i * 8) as i32;
                builder.ins().store(MemFlags::new(), val, base, offset);
            }

            Ok(Some(base))
        }

        Op::PerformEffect { effect, op, args } => {
            // Dispatch effect operations to runtime functions
            let arg_vals: Vec<_> = args
                .iter()
                .map(|a| get_value(values, *a))
                .collect::<Result<_, _>>()?;

            match (effect.as_str(), op.as_str()) {
                // Prob.sample - sample from Normal distribution
                ("Prob", "sample") => {
                    if let Some(&func_ref) = func_refs.get("runtime_prob_sample") {
                        // Default to Normal(0, 1) if no args provided
                        let mean = if arg_vals.len() > 0 {
                            arg_vals[0]
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        let std = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().f64const(1.0)
                        };
                        let call = builder.ins().call(func_ref, &[mean, std]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().f64const(0.0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().f64const(0.0)))
                    }
                }

                // Prob.observe - condition on observation
                ("Prob", "observe") => {
                    if let Some(&func_ref) = func_refs.get("runtime_prob_observe") {
                        let mean = if arg_vals.len() > 0 {
                            arg_vals[0]
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        let std = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().f64const(1.0)
                        };
                        let observed = if arg_vals.len() > 2 {
                            arg_vals[2]
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        let call = builder.ins().call(func_ref, &[mean, std, observed]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().f64const(0.0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().f64const(0.0)))
                    }
                }

                // Causal.do - intervention
                ("Causal", "do") => {
                    if let Some(&func_ref) = func_refs.get("runtime_causal_do") {
                        let var_ptr = if arg_vals.len() > 0 {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let value = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        let call = builder.ins().call(func_ref, &[var_ptr, value]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().f64const(0.0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        // Just return the value
                        if arg_vals.len() > 1 {
                            Ok(Some(arg_vals[1]))
                        } else {
                            Ok(Some(builder.ins().f64const(0.0)))
                        }
                    }
                }

                // IO.read_line - read a line from stdin
                ("IO", "read_line") => {
                    if let Some(&func_ref) = func_refs.get("runtime_io_read_line") {
                        let call = builder.ins().call(func_ref, &[]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // IO.read_file - read file contents
                ("IO", "read_file") => {
                    if let Some(&func_ref) = func_refs.get("runtime_io_read_file") {
                        let path_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[path_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // IO.write_file - write data to file
                ("IO", "write_file") => {
                    if let Some(&func_ref) = func_refs.get("runtime_io_write_file") {
                        let path_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let data_ptr = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[path_ptr, data_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // IO.append_file - append data to file
                ("IO", "append_file") => {
                    if let Some(&func_ref) = func_refs.get("runtime_io_append_file") {
                        let path_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let data_ptr = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[path_ptr, data_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // IO.file_exists - check if file exists
                ("IO", "file_exists") => {
                    if let Some(&func_ref) = func_refs.get("runtime_io_file_exists") {
                        let path_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[path_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // IO.print - print to stdout (already handled by print functions)
                ("IO", "print") => {
                    // Delegate to runtime_print_cstr for string printing
                    if let Some(&func_ref) = func_refs.get("runtime_print_cstr") {
                        let ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        builder.ins().call(func_ref, &[ptr]);
                    }
                    Ok(Some(builder.ins().iconst(types::I64, 0)))
                }

                // Mut.get - get value from mutable state
                ("Mut", "get") => {
                    if let Some(&func_ref) = func_refs.get("runtime_mut_get") {
                        let name_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[name_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().f64const(0.0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().f64const(0.0)))
                    }
                }

                // Mut.set - set value in mutable state
                ("Mut", "set") => {
                    if let Some(&func_ref) = func_refs.get("runtime_mut_set") {
                        let name_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let value = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        let call = builder.ins().call(func_ref, &[name_ptr, value]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Mut.modify - modify value in mutable state
                ("Mut", "modify") => {
                    if let Some(&func_ref) = func_refs.get("runtime_mut_modify") {
                        let name_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let delta = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        let call = builder.ins().call(func_ref, &[name_ptr, delta]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().f64const(0.0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().f64const(0.0)))
                    }
                }

                // Mut.clear - clear all mutable state
                ("Mut", "clear") => {
                    if let Some(&func_ref) = func_refs.get("runtime_mut_clear") {
                        builder.ins().call(func_ref, &[]);
                    }
                    Ok(Some(builder.ins().iconst(types::I64, 0)))
                }

                // Mut.exists - check if name exists in mutable state
                ("Mut", "exists") => {
                    if let Some(&func_ref) = func_refs.get("runtime_mut_exists") {
                        let name_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[name_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Mut.delete - delete a value from mutable state
                ("Mut", "delete") => {
                    if let Some(&func_ref) = func_refs.get("runtime_mut_delete") {
                        let name_ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[name_ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().f64const(0.0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().f64const(0.0)))
                    }
                }

                // Alloc.alloc - allocate memory
                ("Alloc", "alloc") => {
                    if let Some(&func_ref) = func_refs.get("runtime_alloc") {
                        let size = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[size]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Alloc.alloc_array - allocate array
                ("Alloc", "alloc_array") => {
                    if let Some(&func_ref) = func_refs.get("runtime_alloc_array") {
                        let count = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let elem_size = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().iconst(types::I64, 1)
                        };
                        let call = builder.ins().call(func_ref, &[count, elem_size]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Alloc.dealloc - deallocate memory
                ("Alloc", "dealloc") => {
                    if let Some(&func_ref) = func_refs.get("runtime_dealloc") {
                        let ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Alloc.realloc - reallocate memory
                ("Alloc", "realloc") => {
                    if let Some(&func_ref) = func_refs.get("runtime_realloc") {
                        let ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let new_size = if arg_vals.len() > 1 {
                            arg_vals[1]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[ptr, new_size]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Alloc.size_of - get allocation size
                ("Alloc", "size_of") => {
                    if let Some(&func_ref) = func_refs.get("runtime_alloc_size_of") {
                        let ptr = if !arg_vals.is_empty() {
                            arg_vals[0]
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        let call = builder.ins().call(func_ref, &[ptr]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Alloc.clear - clear all allocations
                ("Alloc", "clear") => {
                    if let Some(&func_ref) = func_refs.get("runtime_alloc_clear") {
                        builder.ins().call(func_ref, &[]);
                    }
                    Ok(Some(builder.ins().iconst(types::I64, 0)))
                }

                // Alloc.total_allocated - get total allocated bytes
                ("Alloc", "total_allocated") => {
                    if let Some(&func_ref) = func_refs.get("runtime_alloc_total") {
                        let call = builder.ins().call(func_ref, &[]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Alloc.count - get number of allocations
                ("Alloc", "count") => {
                    if let Some(&func_ref) = func_refs.get("runtime_alloc_count") {
                        let call = builder.ins().call(func_ref, &[]);
                        let results = builder.inst_results(call);
                        if results.is_empty() {
                            Ok(Some(builder.ins().iconst(types::I64, 0)))
                        } else {
                            Ok(Some(results[0]))
                        }
                    } else {
                        Ok(Some(builder.ins().iconst(types::I64, 0)))
                    }
                }

                // Default: return zero for unhandled effects
                _ => {
                    // Log warning for unhandled effect (in debug builds)
                    #[cfg(debug_assertions)]
                    eprintln!("Warning: Unhandled effect {}.{} in JIT", effect, op);
                    let zero = builder.ins().iconst(ty, 0);
                    Ok(Some(zero))
                }
            }
        }

        Op::PushHandler {
            effect,
            handler_name,
            handler_id,
        } => {
            // Push a handler onto the handler stack
            if let Some(&func_ref) = func_refs.get("runtime_handler_push") {
                // Store effect string in global storage for lifetime
                let effect_cstr = std::ffi::CString::new(effect.as_str())
                    .unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
                let name_cstr = std::ffi::CString::new(handler_name.as_str())
                    .unwrap_or_else(|_| std::ffi::CString::new("").unwrap());

                let effect_ptr = if let Ok(mut storage) = STRING_STORAGE.lock() {
                    storage.push(effect_cstr);
                    storage.last().unwrap().as_ptr() as i64
                } else {
                    0
                };

                let name_ptr = if let Ok(mut storage) = STRING_STORAGE.lock() {
                    storage.push(name_cstr);
                    storage.last().unwrap().as_ptr() as i64
                } else {
                    0
                };

                let effect_val = builder.ins().iconst(types::I64, effect_ptr);
                let name_val = builder.ins().iconst(types::I64, name_ptr);
                let handler_id_val = builder.ins().iconst(types::I32, *handler_id as i64);
                builder.ins().call(func_ref, &[effect_val, name_val, handler_id_val]);
            }
            Ok(Some(builder.ins().iconst(types::I64, 0)))
        }

        Op::PopHandler => {
            // Pop the topmost handler from the stack
            if let Some(&func_ref) = func_refs.get("runtime_handler_pop") {
                builder.ins().call(func_ref, &[]);
            }
            Ok(Some(builder.ins().iconst(types::I64, 0)))
        }

        Op::DispatchEffect { effect, op, args } => {
            // Dispatch effect through handler stack, falling back to registry
            let arg_vals: Vec<_> = args
                .iter()
                .map(|a| get_value(values, *a))
                .collect::<Result<_, _>>()?;

            // Use runtime_handler_dispatch which goes through the handler stack + registry
            if let Some(&dispatch_ref) = func_refs.get("runtime_handler_dispatch") {
                // Store effect and op strings in global storage
                let effect_cstr = std::ffi::CString::new(effect.as_str())
                    .unwrap_or_else(|_| std::ffi::CString::new("").unwrap());
                let op_cstr = std::ffi::CString::new(op.as_str())
                    .unwrap_or_else(|_| std::ffi::CString::new("").unwrap());

                let effect_ptr = if let Ok(mut storage) = STRING_STORAGE.lock() {
                    storage.push(effect_cstr);
                    storage.last().unwrap().as_ptr() as i64
                } else {
                    0
                };

                let op_ptr = if let Ok(mut storage) = STRING_STORAGE.lock() {
                    storage.push(op_cstr);
                    storage.last().unwrap().as_ptr() as i64
                } else {
                    0
                };

                let effect_val = builder.ins().iconst(types::I64, effect_ptr);
                let op_val = builder.ins().iconst(types::I64, op_ptr);

                // Build args array on stack
                if !arg_vals.is_empty() {
                    let args_slot = builder.create_sized_stack_slot(
                        cranelift_codegen::ir::StackSlotData::new(
                            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                            (arg_vals.len() * 8) as u32,
                            8,
                        ),
                    );
                    let args_ptr = builder.ins().stack_addr(types::I64, args_slot, 0);

                    // Store args as f64
                    for (i, &val) in arg_vals.iter().enumerate() {
                        let f64_val = if builder.func.dfg.value_type(val) == types::F64 {
                            val
                        } else if builder.func.dfg.value_type(val).is_int() {
                            builder.ins().fcvt_from_sint(types::F64, val)
                        } else {
                            builder.ins().f64const(0.0)
                        };
                        builder.ins().store(
                            MemFlags::new(),
                            f64_val,
                            args_ptr,
                            (i * 8) as i32,
                        );
                    }

                    let args_len = builder.ins().iconst(types::I64, arg_vals.len() as i64);
                    let dispatch_call = builder.ins().call(
                        dispatch_ref,
                        &[effect_val, op_val, args_ptr, args_len],
                    );
                    let results = builder.inst_results(dispatch_call);
                    if !results.is_empty() {
                        return Ok(Some(results[0]));
                    }
                } else {
                    // No args - create empty args pointer
                    let null_ptr = builder.ins().iconst(types::I64, 0);
                    let zero_len = builder.ins().iconst(types::I64, 0);
                    let dispatch_call = builder.ins().call(
                        dispatch_ref,
                        &[effect_val, op_val, null_ptr, zero_len],
                    );
                    let results = builder.inst_results(dispatch_call);
                    if !results.is_empty() {
                        return Ok(Some(results[0]));
                    }
                }
            }

            // Fall back to default value if no handler dispatch available
            Ok(Some(builder.ins().f64const(0.0)))
        }
    }
}

#[cfg(feature = "jit")]
fn translate_constant(
    builder: &mut FunctionBuilder,
    constant: &HlirConstant,
    ty: &HlirType,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
) -> Result<cranelift_codegen::ir::Value, String> {
    let cl_ty = hlir_to_cranelift_type(ty);

    match constant {
        HlirConstant::Unit => Ok(builder.ins().iconst(types::I64, 0)),
        HlirConstant::Bool(b) => Ok(builder.ins().iconst(types::I8, *b as i64)),
        HlirConstant::Int(i, _) => Ok(builder.ins().iconst(cl_ty, *i)),
        HlirConstant::Float(f, _) => {
            if cl_ty == types::F32 {
                Ok(builder.ins().f32const(*f as f32))
            } else {
                Ok(builder.ins().f64const(*f))
            }
        }
        HlirConstant::String(s) => {
            // Store the string in global storage so it survives during JIT execution
            // Use CString for null-termination so runtime_print_cstr can use it
            let cstring = std::ffi::CString::new(s.as_str())
                .unwrap_or_else(|_| std::ffi::CString::new("").unwrap());

            // Store in global storage to keep alive, then get pointer from stored CString
            if let Ok(mut storage) = STRING_STORAGE.lock() {
                storage.push(cstring);
                // Get pointer from the stored CString (it's now at the end of the vec)
                let stored_ptr = storage.last().unwrap().as_ptr() as i64;
                Ok(builder.ins().iconst(types::I64, stored_ptr))
            } else {
                // Fallback to null if lock fails
                Ok(builder.ins().iconst(types::I64, 0))
            }
        }
        HlirConstant::Null(_) => Ok(builder.ins().iconst(types::I64, 0)),
        HlirConstant::Undef(_) => Ok(builder.ins().iconst(cl_ty, 0)),
        HlirConstant::FunctionRef(name) => {
            if let Some(&func_ref) = func_refs.get(name) {
                Ok(builder.ins().func_addr(types::I64, func_ref))
            } else {
                Ok(builder.ins().iconst(types::I64, 0))
            }
        }
        HlirConstant::GlobalRef(_) => Ok(builder.ins().iconst(types::I64, 0)),
        HlirConstant::Array(_) | HlirConstant::Struct(_) => {
            // Complex constants - return null for now
            Ok(builder.ins().iconst(types::I64, 0))
        }
    }
}

#[cfg(feature = "jit")]
fn translate_binary_op(
    builder: &mut FunctionBuilder,
    op: BinaryOp,
    lhs: cranelift_codegen::ir::Value,
    rhs: cranelift_codegen::ir::Value,
    result_ty: &HlirType,
) -> Result<cranelift_codegen::ir::Value, String> {
    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};

    let result = match op {
        // Integer arithmetic
        BinaryOp::Add => builder.ins().iadd(lhs, rhs),
        BinaryOp::Sub => builder.ins().isub(lhs, rhs),
        BinaryOp::Mul => builder.ins().imul(lhs, rhs),
        BinaryOp::SDiv => builder.ins().sdiv(lhs, rhs),
        BinaryOp::UDiv => builder.ins().udiv(lhs, rhs),
        BinaryOp::SRem => builder.ins().srem(lhs, rhs),
        BinaryOp::URem => builder.ins().urem(lhs, rhs),

        // Float arithmetic
        BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
        BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
        BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
        BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
        BinaryOp::FRem => {
            // Cranelift doesn't have frem, use a workaround
            let div = builder.ins().fdiv(lhs, rhs);
            let trunc = builder.ins().trunc(div);
            let mul = builder.ins().fmul(trunc, rhs);
            builder.ins().fsub(lhs, mul)
        }

        // Bitwise
        BinaryOp::And => builder.ins().band(lhs, rhs),
        BinaryOp::Or => builder.ins().bor(lhs, rhs),
        BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
        BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
        BinaryOp::AShr => builder.ins().sshr(lhs, rhs),
        BinaryOp::LShr => builder.ins().ushr(lhs, rhs),

        // Integer comparison
        BinaryOp::Eq => builder.ins().icmp(IntCC::Equal, lhs, rhs),
        BinaryOp::Ne => builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
        BinaryOp::SLt => builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
        BinaryOp::SLe => builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
        BinaryOp::SGt => builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
        BinaryOp::SGe => builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
        BinaryOp::ULt => builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs),
        BinaryOp::ULe => builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs),
        BinaryOp::UGt => builder.ins().icmp(IntCC::UnsignedGreaterThan, lhs, rhs),
        BinaryOp::UGe => builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs),

        // Float comparison
        BinaryOp::FOEq => builder.ins().fcmp(FloatCC::Equal, lhs, rhs),
        BinaryOp::FONe => builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs),
        BinaryOp::FOLt => builder.ins().fcmp(FloatCC::LessThan, lhs, rhs),
        BinaryOp::FOLe => builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs),
        BinaryOp::FOGt => builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs),
        BinaryOp::FOGe => builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
        BinaryOp::Concat => {
            // String concatenation - not supported in JIT, would need runtime call
            return Err("String concatenation not supported in JIT".to_string());
        }
    };

    // Comparisons return i8, may need to extend for result type
    let result_cl_ty = hlir_to_cranelift_type(result_ty);
    let result_ty_current = builder.func.dfg.value_type(result);

    if result_ty_current != result_cl_ty && result_cl_ty.is_int() {
        if result_ty_current.bits() < result_cl_ty.bits() {
            Ok(builder.ins().uextend(result_cl_ty, result))
        } else {
            Ok(result)
        }
    } else {
        Ok(result)
    }
}

#[cfg(feature = "jit")]
fn translate_unary_op(
    builder: &mut FunctionBuilder,
    op: UnaryOp,
    val: cranelift_codegen::ir::Value,
    ty: &HlirType,
) -> Result<cranelift_codegen::ir::Value, String> {
    match op {
        UnaryOp::Neg => Ok(builder.ins().ineg(val)),
        UnaryOp::FNeg => Ok(builder.ins().fneg(val)),
        UnaryOp::Not => {
            // Logical not: xor with all 1s
            let ones = builder.ins().iconst(hlir_to_cranelift_type(ty), -1);
            Ok(builder.ins().bxor(val, ones))
        }
    }
}

#[cfg(feature = "jit")]
fn translate_terminator(
    builder: &mut FunctionBuilder,
    term: &HlirTerminator,
    blocks: &HashMap<BlockId, cranelift_codegen::ir::Block>,
    values: &HashMap<ValueId, cranelift_codegen::ir::Value>,
) -> Result<(), String> {
    match term {
        HlirTerminator::Return(val) => {
            if let Some(v) = val {
                let ret_val = get_value(values, *v)?;

                // Get expected return type from function signature
                let expected_ret_ty = builder
                    .func
                    .signature
                    .returns
                    .first()
                    .map(|p| p.value_type)
                    .unwrap_or(types::I64);
                let actual_ty = builder.func.dfg.value_type(ret_val);

                // Insert cast if types don't match
                let final_val = if actual_ty != expected_ret_ty {
                    if actual_ty.bits() > expected_ret_ty.bits() {
                        // Truncate (e.g., I64 -> I32)
                        builder.ins().ireduce(expected_ret_ty, ret_val)
                    } else if actual_ty.bits() < expected_ret_ty.bits() {
                        // Extend (e.g., I32 -> I64)
                        builder.ins().sextend(expected_ret_ty, ret_val)
                    } else {
                        ret_val
                    }
                } else {
                    ret_val
                };

                builder.ins().return_(&[final_val]);
            } else {
                builder.ins().return_(&[]);
            }
        }

        HlirTerminator::Branch(target) => {
            let target_block = blocks[target];
            builder.ins().jump(target_block, &[]);
        }

        HlirTerminator::CondBranch {
            condition,
            then_block,
            else_block,
        } => {
            let cond = get_value(values, *condition)?;
            let then_b = blocks[then_block];
            let else_b = blocks[else_block];
            builder.ins().brif(cond, then_b, &[], else_b, &[]);
        }

        HlirTerminator::Switch {
            value,
            default,
            cases,
        } => {
            let val = get_value(values, *value)?;
            let default_block = blocks[default];

            // Build switch using a chain of conditionals
            for (case_val, target) in cases {
                let target_block = blocks[target];
                let case_const = builder.ins().iconst(types::I64, *case_val);
                let cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    val,
                    case_const,
                );

                let next_block = builder.create_block();
                builder.ins().brif(cmp, target_block, &[], next_block, &[]);
                builder.seal_block(next_block);
                builder.switch_to_block(next_block);
            }

            builder.ins().jump(default_block, &[]);
        }

        HlirTerminator::Unreachable => {
            builder
                .ins()
                .trap(cranelift_codegen::ir::TrapCode::unwrap_user(0));
        }
    }

    Ok(())
}

#[cfg(feature = "jit")]
fn get_value(
    values: &HashMap<ValueId, cranelift_codegen::ir::Value>,
    id: ValueId,
) -> Result<cranelift_codegen::ir::Value, String> {
    values
        .get(&id)
        .copied()
        .ok_or_else(|| format!("Value not found: {:?}", id))
}

// Non-JIT placeholder implementation
#[cfg(not(feature = "jit"))]
impl CompiledModule {
    #[allow(dead_code)]
    fn new_stub() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configure_jit_registry_stub() {
        // Test that stub functions don't panic (works for both jit and non-jit builds)
        configure_jit_all_effects();
        reset_jit_effect_state();
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_configure_jit_registry() {
        use crate::effects::handlers::HandlerRegistry;

        let registry = HandlerRegistry::with_defaults();
        configure_jit_registry(registry);

        // Verify state was updated
        if let Ok(state) = get_jit_effect_state().lock() {
            assert!(state.registry.is_some());
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_configure_jit_all_effects() {
        configure_jit_all_effects();

        // Verify 12 handlers available
        if let Ok(state) = get_jit_effect_state().lock() {
            assert!(state.registry.is_some());
            if let Some(ref registry) = state.registry {
                assert_eq!(registry.len(), 12);
            }
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_reset_jit_effect_state() {
        configure_jit_all_effects();

        // Set some state
        if let Ok(mut state) = get_jit_effect_state().lock() {
            state.log_prob = 123.0;
            state.mutable_state.insert("test".to_string(), 42.0);
        }

        // Reset
        reset_jit_effect_state();

        // Verify state was reset but registry preserved
        if let Ok(state) = get_jit_effect_state().lock() {
            assert_eq!(state.log_prob, 0.0);
            assert!(state.mutable_state.is_empty());
            assert!(state.registry.is_some()); // Registry preserved
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_dispatch_via_registry() {
        configure_jit_all_effects();

        if let Ok(state) = get_jit_effect_state().lock() {
            // Test Div.div dispatch
            let result = state.dispatch_via_registry("Div", "div", &[10.0, 2.0]);
            assert!(result.is_some());
            assert!((result.unwrap() - 5.0).abs() < 0.001);
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_dispatch_effect_combined() {
        configure_jit_all_effects();

        if let Ok(state) = get_jit_effect_state().lock() {
            // Test dispatch_effect (tries stack first, then registry)
            let result = state.dispatch_effect("Div", "div", &[20.0, 4.0]);
            assert!(result.is_some());
            assert!((result.unwrap() - 5.0).abs() < 0.001);
        }
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_runtime_effect_dispatch_function() {
        use std::ffi::CString;

        configure_jit_all_effects();

        // Create null-terminated strings
        let effect = CString::new("Div").unwrap();
        let op = CString::new("div").unwrap();
        let args = [10.0f64, 2.0f64];

        let result = runtime_effect_dispatch(
            effect.as_ptr() as *const u8,
            op.as_ptr() as *const u8,
            args.as_ptr(),
            args.len(),
        );

        assert!(!result.is_nan());
        assert!((result - 5.0).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_runtime_effect_dispatch_gpu_sync() {
        use std::ffi::CString;

        configure_jit_all_effects();

        // GPU.sync dispatch (no args required)
        let effect = CString::new("GPU").unwrap();
        let op = CString::new("sync").unwrap();
        let args: [f64; 0] = [];

        let result = runtime_effect_dispatch(
            effect.as_ptr() as *const u8,
            op.as_ptr() as *const u8,
            args.as_ptr(),
            args.len(),
        );

        // GPU.sync returns Unit (0.0)
        assert!(!result.is_nan());
    }

    #[test]
    #[cfg(feature = "jit")]
    fn test_jit_handler_stack_precedence() {
        configure_jit_all_effects();

        // Push a custom handler that returns a fixed value
        if let Ok(mut state) = get_jit_effect_state().lock() {
            let custom_handler = JitHandler::new("Div", "custom_div", 3); // ID 3 = deterministic sampler
            state.push_handler(custom_handler);

            // dispatch_effect should try stack first
            // Since handler ID 3 is a deterministic sampler for "sample" op,
            // "div" op won't be handled by it, so it falls through to registry
            let result = state.dispatch_effect("Div", "div", &[10.0, 2.0]);
            assert!(result.is_some());

            state.pop_handler();
        }
    }

    #[test]
    fn test_cranelift_jit_new() {
        let jit = CraneliftJit::new();
        assert!(!jit.optimize);
    }

    #[test]
    fn test_cranelift_jit_with_optimization() {
        let jit = CraneliftJit::new().with_optimization();
        assert!(jit.optimize);
    }
}
