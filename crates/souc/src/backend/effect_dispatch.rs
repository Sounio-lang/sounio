//! Unified Effect Dispatch System
//!
//! This module provides a common effect dispatch infrastructure that all backends
//! can use. It unifies effect handling across Native, Cranelift (JIT), and LLVM
//! backends by providing:
//!
//! 1. A common `EffectDispatch` trait for backend-agnostic effect operations
//! 2. A `UnifiedEffectRuntime` that wraps the runtime handler stack
//! 3. Standardized effect operation enums for all 13 effects
//!
//! # Architecture
//!
//! ```text
//! Backend Code
//!     ↓
//! EffectDispatch trait
//!     ↓
//! UnifiedEffectRuntime
//!     ↓
//! runtime::handler_stack (C-callable functions)
//! ```
//!
//! # Supported Effects (13 total)
//!
//! 1. **IO** - Input/output operations (print, read, write, etc.)
//! 2. **Mut** - Mutable state (get, set, modify)
//! 3. **Alloc** - Memory allocation (alloc, dealloc)
//! 4. **Panic** - Panic handling
//! 5. **Async** - Asynchronous operations (spawn, yield, await)
//! 6. **GPU** - GPU operations (launch, sync)
//! 7. **Div** - Safe division
//! 8. **Prob** - Probabilistic operations (sample, observe, factor)
//! 9. **Grad** - Automatic differentiation (forward, reverse)
//! 10. **Causal** - Causal inference (do, observe, counterfactual)
//! 11. **Network** - Network operations (ping, send, recv)
//! 12. **Sensor** - Sensor reading (read)
//! 13. **Exn** - Exception handling (throw, catch)

use std::fmt;

/// Trait for backend-agnostic effect dispatch
///
/// All backends (Native, Cranelift, LLVM) should implement this trait
/// to dispatch effects through a unified runtime.
pub trait EffectDispatch {
    /// Dispatch an effect operation
    ///
    /// # Arguments
    /// * `effect` - Effect name (e.g., "IO", "Mut", "Prob")
    /// * `op` - Operation name (e.g., "print", "sample", "get")
    /// * `args` - Operation arguments as f64 values
    ///
    /// # Returns
    /// Result value as f64, or error if dispatch failed
    fn dispatch_effect(&mut self, effect: &str, op: &str, args: &[f64])
    -> Result<f64, EffectError>;

    /// Push a handler onto the effect handler stack
    ///
    /// # Arguments
    /// * `effect` - Effect name to handle
    /// * `handler_id` - Predefined handler ID (0 for default)
    fn push_handler(&mut self, effect: &str, handler_id: u32) -> Result<(), EffectError>;

    /// Pop the topmost handler from the stack
    fn pop_handler(&mut self) -> Result<(), EffectError>;

    /// Get the current handler stack depth
    fn handler_depth(&self) -> usize;
}

/// Effect operation enum for type-safe dispatch
#[derive(Debug, Clone, PartialEq)]
pub enum EffectOp {
    // IO operations
    IoRead,
    IoPrint,
    IoPrintln,
    IoWrite,
    IoFlush,

    // Mut operations
    MutGet,
    MutSet,
    MutModify,

    // Alloc operations
    AllocAlloc,
    AllocDealloc,
    AllocRealloc,

    // Panic operations
    PanicPanic,

    // Async operations
    AsyncSpawn,
    AsyncYield,
    AsyncAwait,

    // GPU operations
    GpuLaunch,
    GpuSync,
    GpuMemcpy,

    // Div operations
    DivDiv,
    DivRem,

    // Prob operations
    ProbSample,
    ProbObserve,
    ProbFactor,
    ProbScore,

    // Grad operations
    GradForward,
    GradReverse,
    GradJvp,
    GradVjp,

    // Causal operations
    CausalDo,
    CausalObserve,
    CausalCounterfactual,
    CausalQuery,
    CausalAte,
    CausalCate,

    // Network operations
    NetworkPing,
    NetworkSend,
    NetworkRecv,

    // Sensor operations
    SensorRead,
    SensorCalibrate,

    // Exn operations
    ExnThrow,
    ExnCatch,

    // Generic operation (string-based)
    Generic(String, String),
}

impl EffectOp {
    /// Parse an effect operation from effect name and operation name
    pub fn parse(effect: &str, op: &str) -> Self {
        let effect_lower = effect.to_lowercase();
        let op_lower = op.to_lowercase();

        match (effect_lower.as_str(), op_lower.as_str()) {
            // IO
            ("io", "read") => EffectOp::IoRead,
            ("io", "print") => EffectOp::IoPrint,
            ("io", "println") => EffectOp::IoPrintln,
            ("io", "write") => EffectOp::IoWrite,
            ("io", "flush") => EffectOp::IoFlush,

            // Mut
            ("mut", "get") => EffectOp::MutGet,
            ("mut", "set") => EffectOp::MutSet,
            ("mut", "modify") => EffectOp::MutModify,

            // Alloc
            ("alloc", "alloc") => EffectOp::AllocAlloc,
            ("alloc", "dealloc") => EffectOp::AllocDealloc,
            ("alloc", "realloc") => EffectOp::AllocRealloc,

            // Panic
            ("panic", "panic") => EffectOp::PanicPanic,

            // Async
            ("async", "spawn") => EffectOp::AsyncSpawn,
            ("async", "yield") => EffectOp::AsyncYield,
            ("async", "await") => EffectOp::AsyncAwait,

            // GPU
            ("gpu", "launch") => EffectOp::GpuLaunch,
            ("gpu", "sync") => EffectOp::GpuSync,
            ("gpu", "memcpy") => EffectOp::GpuMemcpy,

            // Div
            ("div", "div") => EffectOp::DivDiv,
            ("div", "rem") => EffectOp::DivRem,

            // Prob
            ("prob", "sample") => EffectOp::ProbSample,
            ("prob", "observe") => EffectOp::ProbObserve,
            ("prob", "factor") => EffectOp::ProbFactor,
            ("prob", "score") => EffectOp::ProbScore,

            // Grad
            ("grad", "forward") => EffectOp::GradForward,
            ("grad", "reverse") => EffectOp::GradReverse,
            ("grad", "jvp") => EffectOp::GradJvp,
            ("grad", "vjp") => EffectOp::GradVjp,

            // Causal
            ("causal", "do") => EffectOp::CausalDo,
            ("causal", "observe") => EffectOp::CausalObserve,
            ("causal", "counterfactual") => EffectOp::CausalCounterfactual,
            ("causal", "query") => EffectOp::CausalQuery,
            ("causal", "ate") => EffectOp::CausalAte,
            ("causal", "cate") => EffectOp::CausalCate,

            // Network
            ("network", "ping") => EffectOp::NetworkPing,
            ("network", "send") => EffectOp::NetworkSend,
            ("network", "recv") => EffectOp::NetworkRecv,

            // Sensor
            ("sensor", "read") => EffectOp::SensorRead,
            ("sensor", "calibrate") => EffectOp::SensorCalibrate,

            // Exn
            ("exn", "throw") => EffectOp::ExnThrow,
            ("exn", "catch") => EffectOp::ExnCatch,

            // Generic fallback
            _ => EffectOp::Generic(effect.to_string(), op.to_string()),
        }
    }

    /// Get the effect name for this operation
    pub fn effect_name(&self) -> &str {
        match self {
            EffectOp::IoRead
            | EffectOp::IoPrint
            | EffectOp::IoPrintln
            | EffectOp::IoWrite
            | EffectOp::IoFlush => "IO",

            EffectOp::MutGet | EffectOp::MutSet | EffectOp::MutModify => "Mut",

            EffectOp::AllocAlloc | EffectOp::AllocDealloc | EffectOp::AllocRealloc => "Alloc",

            EffectOp::PanicPanic => "Panic",

            EffectOp::AsyncSpawn | EffectOp::AsyncYield | EffectOp::AsyncAwait => "Async",

            EffectOp::GpuLaunch | EffectOp::GpuSync | EffectOp::GpuMemcpy => "GPU",

            EffectOp::DivDiv | EffectOp::DivRem => "Div",

            EffectOp::ProbSample
            | EffectOp::ProbObserve
            | EffectOp::ProbFactor
            | EffectOp::ProbScore => "Prob",

            EffectOp::GradForward
            | EffectOp::GradReverse
            | EffectOp::GradJvp
            | EffectOp::GradVjp => "Grad",

            EffectOp::CausalDo
            | EffectOp::CausalObserve
            | EffectOp::CausalCounterfactual
            | EffectOp::CausalQuery
            | EffectOp::CausalAte
            | EffectOp::CausalCate => "Causal",

            EffectOp::NetworkPing | EffectOp::NetworkSend | EffectOp::NetworkRecv => "Network",

            EffectOp::SensorRead | EffectOp::SensorCalibrate => "Sensor",

            EffectOp::ExnThrow | EffectOp::ExnCatch => "Exn",

            EffectOp::Generic(effect, _) => effect,
        }
    }

    /// Get the operation name for this operation
    pub fn operation_name(&self) -> &str {
        match self {
            EffectOp::IoRead => "read",
            EffectOp::IoPrint => "print",
            EffectOp::IoPrintln => "println",
            EffectOp::IoWrite => "write",
            EffectOp::IoFlush => "flush",

            EffectOp::MutGet => "get",
            EffectOp::MutSet => "set",
            EffectOp::MutModify => "modify",

            EffectOp::AllocAlloc => "alloc",
            EffectOp::AllocDealloc => "dealloc",
            EffectOp::AllocRealloc => "realloc",

            EffectOp::PanicPanic => "panic",

            EffectOp::AsyncSpawn => "spawn",
            EffectOp::AsyncYield => "yield",
            EffectOp::AsyncAwait => "await",

            EffectOp::GpuLaunch => "launch",
            EffectOp::GpuSync => "sync",
            EffectOp::GpuMemcpy => "memcpy",

            EffectOp::DivDiv => "div",
            EffectOp::DivRem => "rem",

            EffectOp::ProbSample => "sample",
            EffectOp::ProbObserve => "observe",
            EffectOp::ProbFactor => "factor",
            EffectOp::ProbScore => "score",

            EffectOp::GradForward => "forward",
            EffectOp::GradReverse => "reverse",
            EffectOp::GradJvp => "jvp",
            EffectOp::GradVjp => "vjp",

            EffectOp::CausalDo => "do",
            EffectOp::CausalObserve => "observe",
            EffectOp::CausalCounterfactual => "counterfactual",
            EffectOp::CausalQuery => "query",
            EffectOp::CausalAte => "ate",
            EffectOp::CausalCate => "cate",

            EffectOp::NetworkPing => "ping",
            EffectOp::NetworkSend => "send",
            EffectOp::NetworkRecv => "recv",

            EffectOp::SensorRead => "read",
            EffectOp::SensorCalibrate => "calibrate",

            EffectOp::ExnThrow => "throw",
            EffectOp::ExnCatch => "catch",

            EffectOp::Generic(_, op) => op,
        }
    }
}

/// Effect dispatch error
#[derive(Debug, Clone, PartialEq)]
pub enum EffectError {
    /// No handler found for the effect
    NoHandler(String),
    /// Handler execution failed
    HandlerFailed(String),
    /// Invalid arguments
    InvalidArgs(String),
    /// Runtime error
    RuntimeError(String),
}

impl fmt::Display for EffectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EffectError::NoHandler(effect) => write!(f, "No handler found for effect: {}", effect),
            EffectError::HandlerFailed(msg) => write!(f, "Handler failed: {}", msg),
            EffectError::InvalidArgs(msg) => write!(f, "Invalid arguments: {}", msg),
            EffectError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for EffectError {}

/// Unified effect runtime that wraps the handler_stack module
///
/// This provides a safe, high-level interface to the C-callable runtime functions
/// in `runtime::handler_stack`. All backends should use this instead of calling
/// the C functions directly.
pub struct UnifiedEffectRuntime;

impl UnifiedEffectRuntime {
    /// Create a new unified effect runtime
    pub fn new() -> Self {
        Self
    }

    /// Reset the runtime state (clear handler stack and state)
    pub fn reset(&mut self) {
        crate::runtime::handler_stack::reset_handler_stack();
    }
}

impl Default for UnifiedEffectRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl EffectDispatch for UnifiedEffectRuntime {
    fn dispatch_effect(
        &mut self,
        effect: &str,
        op: &str,
        args: &[f64],
    ) -> Result<f64, EffectError> {
        // Use the typed EffectOp for better dispatch
        let effect_op = EffectOp::parse(effect, op);

        // Dispatch based on the parsed operation
        let result = match effect_op {
            // IO operations
            EffectOp::IoPrint => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_io_print(args[0])
                } else {
                    0.0
                }
            }
            EffectOp::IoPrintln => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_io_println(args[0])
                } else {
                    crate::runtime::handler_stack::__sounio_dispatch_io_println(0.0)
                }
            }
            EffectOp::IoRead => crate::runtime::handler_stack::__sounio_dispatch_io_read(),

            // Mut operations
            EffectOp::MutGet => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_mut_get(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "MutGet requires key argument".to_string(),
                    ));
                }
            }
            EffectOp::MutSet => {
                if args.len() >= 2 {
                    crate::runtime::handler_stack::__sounio_dispatch_mut_set(args[0], args[1])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "MutSet requires key and value arguments".to_string(),
                    ));
                }
            }
            EffectOp::MutModify => {
                if args.len() >= 2 {
                    crate::runtime::handler_stack::__sounio_dispatch_mut_modify(args[0], args[1])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "MutModify requires key and delta arguments".to_string(),
                    ));
                }
            }

            // Div operations
            EffectOp::DivDiv => {
                if args.len() >= 2 {
                    crate::runtime::handler_stack::__sounio_dispatch_div_div(args[0], args[1])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "DivDiv requires two arguments".to_string(),
                    ));
                }
            }

            // Prob operations
            EffectOp::ProbSample => {
                if args.len() >= 2 {
                    crate::runtime::handler_stack::__sounio_dispatch_prob_sample(args[0], args[1])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "ProbSample requires low and high arguments".to_string(),
                    ));
                }
            }
            EffectOp::ProbObserve => {
                if args.len() >= 3 {
                    crate::runtime::handler_stack::__sounio_dispatch_prob_observe(
                        args[0], args[1], args[2],
                    )
                } else {
                    return Err(EffectError::InvalidArgs(
                        "ProbObserve requires mean, std, and value arguments".to_string(),
                    ));
                }
            }

            // Alloc operations
            EffectOp::AllocAlloc => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_alloc_alloc(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "AllocAlloc requires size argument".to_string(),
                    ));
                }
            }
            EffectOp::AllocDealloc => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_alloc_dealloc(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "AllocDealloc requires pointer argument".to_string(),
                    ));
                }
            }

            // Panic operations
            EffectOp::PanicPanic => crate::runtime::handler_stack::__sounio_dispatch_panic_panic(),

            // GPU operations
            EffectOp::GpuSync => crate::runtime::handler_stack::__sounio_dispatch_gpu_sync(),
            EffectOp::GpuLaunch => {
                if args.len() >= 4 {
                    crate::runtime::handler_stack::__sounio_dispatch_gpu_launch(
                        args[0], args[1], args[2], args[3],
                    )
                } else {
                    return Err(EffectError::InvalidArgs(
                        "GpuLaunch requires kernel_id, grid_x, grid_y, grid_z arguments"
                            .to_string(),
                    ));
                }
            }

            // Async operations
            EffectOp::AsyncSpawn => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_async_spawn(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "AsyncSpawn requires task_id argument".to_string(),
                    ));
                }
            }
            EffectOp::AsyncYield => crate::runtime::handler_stack::__sounio_dispatch_async_yield(),
            EffectOp::AsyncAwait => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_async_await(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "AsyncAwait requires task_id argument".to_string(),
                    ));
                }
            }

            // Grad operations
            EffectOp::GradForward => {
                if args.len() >= 2 {
                    crate::runtime::handler_stack::__sounio_dispatch_grad_forward(args[0], args[1])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "GradForward requires value and derivative arguments".to_string(),
                    ));
                }
            }
            EffectOp::GradReverse => {
                if args.len() >= 2 {
                    crate::runtime::handler_stack::__sounio_dispatch_grad_reverse(args[0], args[1])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "GradReverse requires value and adjoint arguments".to_string(),
                    ));
                }
            }

            // Causal operations
            EffectOp::CausalDo => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_causal_do(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "CausalDo requires value argument".to_string(),
                    ));
                }
            }
            EffectOp::CausalObserve => {
                if !args.is_empty() {
                    crate::runtime::handler_stack::__sounio_dispatch_causal_observe(args[0])
                } else {
                    return Err(EffectError::InvalidArgs(
                        "CausalObserve requires value argument".to_string(),
                    ));
                }
            }

            // Generic dispatch for unhandled operations
            _ => {
                // Use generic dispatch for operations not explicitly handled
                use std::ffi::CString;

                let effect_cstr = CString::new(effect).map_err(|_| {
                    EffectError::RuntimeError("Failed to create effect CString".to_string())
                })?;
                let op_cstr = CString::new(op).map_err(|_| {
                    EffectError::RuntimeError("Failed to create op CString".to_string())
                })?;

                crate::runtime::handler_stack::__sounio_dispatch_generic(
                    effect_cstr.as_ptr() as *const u8,
                    op_cstr.as_ptr() as *const u8,
                    args.as_ptr(),
                    args.len(),
                )
            }
        };

        Ok(result)
    }

    fn push_handler(&mut self, effect: &str, handler_id: u32) -> Result<(), EffectError> {
        // Map effect names to the appropriate push_handler_* function
        match effect.to_lowercase().as_str() {
            "io" => crate::runtime::handler_stack::__sounio_push_handler_io(),
            "mut" => crate::runtime::handler_stack::__sounio_push_handler_mut(),
            "div" => crate::runtime::handler_stack::__sounio_push_handler_div(),
            "prob" => crate::runtime::handler_stack::__sounio_push_handler_prob(),
            "alloc" => crate::runtime::handler_stack::__sounio_push_handler_alloc(),
            "panic" => crate::runtime::handler_stack::__sounio_push_handler_panic(),
            "async" => crate::runtime::handler_stack::__sounio_push_handler_async(),
            "gpu" => crate::runtime::handler_stack::__sounio_push_handler_gpu(),
            "grad" => crate::runtime::handler_stack::__sounio_push_handler_grad(),
            "network" => crate::runtime::handler_stack::__sounio_push_handler_network(),
            "sensor" => crate::runtime::handler_stack::__sounio_push_handler_sensor(),
            "exn" => crate::runtime::handler_stack::__sounio_push_handler_exn(),
            "causal" => crate::runtime::handler_stack::__sounio_push_handler_causal(),
            _ => return Err(EffectError::NoHandler(effect.to_string())),
        }

        // Note: handler_id is currently ignored because the C runtime functions
        // don't take a handler_id parameter. This could be enhanced in the future.
        let _ = handler_id;

        Ok(())
    }

    fn pop_handler(&mut self) -> Result<(), EffectError> {
        crate::runtime::handler_stack::__sounio_pop_handler();
        Ok(())
    }

    fn handler_depth(&self) -> usize {
        crate::runtime::handler_stack::__sounio_handler_depth() as usize
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_op_parse() {
        assert_eq!(EffectOp::parse("IO", "print"), EffectOp::IoPrint);
        assert_eq!(EffectOp::parse("io", "PRINT"), EffectOp::IoPrint);
        assert_eq!(EffectOp::parse("Mut", "get"), EffectOp::MutGet);
        assert_eq!(EffectOp::parse("Prob", "sample"), EffectOp::ProbSample);
        assert_eq!(EffectOp::parse("Div", "div"), EffectOp::DivDiv);
    }

    #[test]
    fn test_effect_op_names() {
        let op = EffectOp::IoPrint;
        assert_eq!(op.effect_name(), "IO");
        assert_eq!(op.operation_name(), "print");

        let op = EffectOp::MutSet;
        assert_eq!(op.effect_name(), "Mut");
        assert_eq!(op.operation_name(), "set");
    }

    #[test]
    #[ignore = "Uses global handler stack, conflicts with parallel tests"]
    fn test_unified_runtime_creation() {
        let runtime = UnifiedEffectRuntime::new();
        assert_eq!(runtime.handler_depth(), 0);
    }

    #[test]
    #[ignore = "Uses global handler stack, conflicts with parallel tests"]
    fn test_unified_runtime_push_pop() {
        let mut runtime = UnifiedEffectRuntime::new();
        runtime.reset();

        assert!(runtime.push_handler("IO", 1).is_ok());
        assert_eq!(runtime.handler_depth(), 1);

        assert!(runtime.push_handler("Mut", 2).is_ok());
        assert_eq!(runtime.handler_depth(), 2);

        assert!(runtime.pop_handler().is_ok());
        assert_eq!(runtime.handler_depth(), 1);

        assert!(runtime.pop_handler().is_ok());
        assert_eq!(runtime.handler_depth(), 0);
    }

    #[test]
    #[ignore = "Uses global handler stack, conflicts with parallel tests"]
    fn test_unified_runtime_dispatch_div() {
        let mut runtime = UnifiedEffectRuntime::new();
        runtime.reset();

        runtime.push_handler("Div", 3).unwrap();

        let result = runtime.dispatch_effect("Div", "div", &[20.0, 4.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 5.0);

        let result = runtime.dispatch_effect("Div", "div", &[10.0, 0.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0.0); // Safe division returns 0 for div by zero
    }

    #[test]
    #[ignore = "Uses global handler stack, conflicts with parallel tests"]
    fn test_unified_runtime_dispatch_mut() {
        let mut runtime = UnifiedEffectRuntime::new();
        runtime.reset();

        runtime.push_handler("Mut", 2).unwrap();

        let result = runtime.dispatch_effect("Mut", "set", &[1.0, 42.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42.0);

        let result = runtime.dispatch_effect("Mut", "get", &[1.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42.0);

        let result = runtime.dispatch_effect("Mut", "modify", &[1.0, 10.0]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 52.0);
    }

    #[test]
    fn test_unified_runtime_invalid_args() {
        let mut runtime = UnifiedEffectRuntime::new();
        runtime.reset();

        runtime.push_handler("Div", 3).unwrap();

        let result = runtime.dispatch_effect("Div", "div", &[20.0]); // Missing second arg
        assert!(result.is_err());
        assert!(matches!(result, Err(EffectError::InvalidArgs(_))));
    }

    #[test]
    fn test_unified_runtime_no_handler() {
        let mut runtime = UnifiedEffectRuntime::new();
        runtime.reset();

        let result = runtime.push_handler("UnknownEffect", 999);
        assert!(result.is_err());
        assert!(matches!(result, Err(EffectError::NoHandler(_))));
    }
}
