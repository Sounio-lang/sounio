//! Runtime Handler Stack for Effect Dispatch
//!
//! This module provides the runtime handler stack used by compiled Sounio code
//! to manage effect handlers. Handlers are pushed onto a stack and effect
//! operations are dispatched through the stack from top to bottom.
//!
//! The module exports C-callable functions with the `__sounio_*` naming
//! convention that are linked into compiled code.

use std::collections::HashMap;
use std::cell::RefCell;
use std::ffi::CStr;

/// A handler entry on the runtime handler stack
#[derive(Debug, Clone)]
pub struct RuntimeHandler {
    /// Effect name this handler handles (e.g., "IO", "Mut", "Prob")
    pub effect: String,
    /// Handler name for debugging
    pub name: String,
    /// Handler ID for predefined handlers (0 = custom)
    pub handler_id: u32,
    /// Custom handler function pointer (if handler_id == 0)
    pub handler_fn: Option<extern "C" fn(*const f64, usize) -> f64>,
}

impl RuntimeHandler {
    /// Create a new handler entry
    pub fn new(effect: &str, name: &str, handler_id: u32) -> Self {
        Self {
            effect: effect.to_string(),
            name: name.to_string(),
            handler_id,
            handler_fn: None,
        }
    }

    /// Create a handler with a custom function
    pub fn with_fn(
        effect: &str,
        name: &str,
        handler_fn: extern "C" fn(*const f64, usize) -> f64,
    ) -> Self {
        Self {
            effect: effect.to_string(),
            name: name.to_string(),
            handler_id: 0,
            handler_fn: Some(handler_fn),
        }
    }
}

/// The runtime handler stack
#[derive(Debug, Default)]
pub struct RuntimeHandlerStack {
    /// Stack of handlers, most recent first
    handlers: Vec<RuntimeHandler>,
    /// Mutable state storage for Mut effect
    mut_state: HashMap<String, f64>,
    /// Last dispatched value (for continuations)
    last_value: f64,
}

impl RuntimeHandlerStack {
    /// Create a new empty handler stack
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a handler onto the stack
    pub fn push(&mut self, handler: RuntimeHandler) {
        self.handlers.push(handler);
    }

    /// Pop the topmost handler
    pub fn pop(&mut self) -> Option<RuntimeHandler> {
        self.handlers.pop()
    }

    /// Get the current stack depth
    pub fn depth(&self) -> usize {
        self.handlers.len()
    }

    /// Find a handler for the given effect
    pub fn find_handler(&self, effect: &str) -> Option<&RuntimeHandler> {
        // Search from top of stack (most recent) to bottom
        self.handlers.iter().rev().find(|h| h.effect == effect)
    }

    /// Check if a handler exists for the effect
    pub fn has_handler(&self, effect: &str) -> bool {
        self.find_handler(effect).is_some()
    }

    /// Dispatch an effect operation
    /// Returns Some(value) if handled, None if no handler
    pub fn dispatch(&mut self, effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        // First check stack handlers
        if let Some(handler) = self.find_handler(effect).cloned() {
            if let Some(handler_fn) = handler.handler_fn {
                // Custom handler function
                let result = handler_fn(args.as_ptr(), args.len());
                self.last_value = result;
                return Some(result);
            }

            // Predefined handlers by ID
            let result = self.dispatch_predefined(effect, op, args, handler.handler_id);
            if let Some(val) = result {
                self.last_value = val;
            }
            return result;
        }

        // Fall back to default behavior
        self.dispatch_default(effect, op, args)
    }

    /// Dispatch to a predefined handler by ID
    fn dispatch_predefined(
        &mut self,
        effect: &str,
        op: &str,
        args: &[f64],
        handler_id: u32,
    ) -> Option<f64> {
        match (effect, op, handler_id) {
            // IO effect
            ("IO" | "io", "print", _) => {
                if !args.is_empty() {
                    println!("{}", args[0]);
                }
                Some(0.0)
            }
            ("IO" | "io", "println", _) => {
                if !args.is_empty() {
                    println!("{}", args[0]);
                } else {
                    println!();
                }
                Some(0.0)
            }

            // Mut effect
            ("Mut" | "mut", "get", _) => {
                // First arg is key hash (we use simplified version here)
                let key = format!("var_{}", args.first().copied().unwrap_or(0.0) as i64);
                Some(*self.mut_state.get(&key).unwrap_or(&0.0))
            }
            ("Mut" | "mut", "set", _) => {
                if args.len() >= 2 {
                    let key = format!("var_{}", args[0] as i64);
                    let value = args[1];
                    self.mut_state.insert(key, value);
                    Some(value)
                } else {
                    Some(0.0)
                }
            }
            ("Mut" | "mut", "modify", _) => {
                if args.len() >= 2 {
                    let key = format!("var_{}", args[0] as i64);
                    let delta = args[1];
                    let current = *self.mut_state.get(&key).unwrap_or(&0.0);
                    let new_val = current + delta;
                    self.mut_state.insert(key, new_val);
                    Some(new_val)
                } else {
                    Some(0.0)
                }
            }

            // Div effect (safe division)
            ("Div" | "div", "div", _) => {
                if args.len() >= 2 {
                    let a = args[0];
                    let b = args[1];
                    if b == 0.0 {
                        Some(0.0) // Safe division returns 0 for div by zero
                    } else {
                        Some(a / b)
                    }
                } else {
                    Some(0.0)
                }
            }

            // Prob effect
            ("Prob" | "prob", "sample", _) => {
                // Simple uniform random (simplified)
                use std::time::{SystemTime, UNIX_EPOCH};
                let seed = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let random = ((seed as f64) % 1000.0) / 1000.0;
                if args.len() >= 2 {
                    let low = args[0];
                    let high = args[1];
                    Some(low + random * (high - low))
                } else {
                    Some(random)
                }
            }

            // Panic effect
            ("Panic" | "panic", "panic", _) => {
                eprintln!("Runtime panic!");
                Some(f64::NAN)
            }

            // GPU effect
            ("GPU" | "gpu", "sync", _) => {
                // GPU sync is a no-op on CPU
                Some(0.0)
            }

            // Alloc effect
            ("Alloc" | "alloc", "alloc", _) => {
                // Simplified: just return a "pointer" as i64 in f64
                let size = args.first().copied().unwrap_or(0.0) as usize;
                let boxed = vec![0u8; size].into_boxed_slice();
                let ptr = Box::into_raw(boxed) as *mut u8 as i64;
                Some(ptr as f64)
            }

            // Network effect
            ("Network" | "network", "ping", _) => {
                // Simplified: return success
                Some(1.0)
            }

            // Async effect
            ("Async" | "async", "spawn", _) | ("Async" | "async", "yield", _) => {
                // Async operations are no-ops in synchronous runtime
                Some(0.0)
            }

            // Grad effect (autodiff)
            ("Grad" | "grad", "forward", _) => {
                // Forward mode gradient (simplified)
                if args.len() >= 2 {
                    Some(args[1]) // Return derivative directly
                } else {
                    Some(0.0)
                }
            }

            // Sensor effect
            ("Sensor" | "sensor", "read", _) => {
                // Simulated sensor reading
                Some(25.0) // e.g., temperature
            }

            // Exn effect (exceptions)
            ("Exn" | "exn", "throw", _) => {
                eprintln!("Exception thrown!");
                Some(f64::NAN)
            }

            // Causal effect
            ("Causal" | "causal", "do", _) => {
                if !args.is_empty() {
                    Some(args[0]) // Return the intervention value
                } else {
                    Some(0.0)
                }
            }

            _ => None,
        }
    }

    /// Default behavior when no handler is found
    fn dispatch_default(&mut self, effect: &str, op: &str, args: &[f64]) -> Option<f64> {
        // For most effects, default is to return 0 or perform basic operation
        self.dispatch_predefined(effect, op, args, 0)
    }

    /// Clear the handler stack
    pub fn clear(&mut self) {
        self.handlers.clear();
    }

    /// Clear mutable state
    pub fn clear_state(&mut self) {
        self.mut_state.clear();
    }

    /// Get the last dispatched value
    pub fn last_value(&self) -> f64 {
        self.last_value
    }
}

// ============================================================================
// Global Handler Stack
// ============================================================================

thread_local! {
    /// Per-thread handler stack.
    ///
    /// This avoids global cross-test interference (Rust tests run in parallel by default)
    /// and matches the typical semantics that handler stacks are thread-local.
    static HANDLER_STACK: RefCell<RuntimeHandlerStack> = RefCell::new(RuntimeHandlerStack::new());
}

fn try_with_handler_stack_mut<R>(f: impl FnOnce(&mut RuntimeHandlerStack) -> R) -> Option<R> {
    HANDLER_STACK.with(|stack| {
        stack
            .try_borrow_mut()
            .ok()
            .map(|mut stack| f(&mut *stack))
    })
}

fn try_with_handler_stack<R>(f: impl FnOnce(&RuntimeHandlerStack) -> R) -> Option<R> {
    HANDLER_STACK.with(|stack| stack.try_borrow().ok().map(|stack| f(&*stack)))
}

/// Reset the thread-local handler stack (for the current thread).
pub fn reset_handler_stack() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.clear();
        stack.clear_state();
    });
}

// ============================================================================
// Extern "C" Functions for Compiled Code
// ============================================================================

/// Push a handler for the IO effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_io() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("IO", "default_io", 1));
    });
}

/// Push a handler for the Mut effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_mut() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Mut", "default_mut", 2));
    });
}

/// Push a handler for the Div effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_div() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Div", "default_div", 3));
    });
}

/// Push a handler for the Prob effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_prob() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Prob", "default_prob", 4));
    });
}

/// Push a handler for the Alloc effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_alloc() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Alloc", "default_alloc", 5));
    });
}

/// Push a handler for the Panic effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_panic() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Panic", "default_panic", 6));
    });
}

/// Push a handler for the Async effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_async() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Async", "default_async", 7));
    });
}

/// Push a handler for the GPU effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_gpu() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("GPU", "default_gpu", 8));
    });
}

/// Push a handler for the Grad effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_grad() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Grad", "default_grad", 9));
    });
}

/// Push a handler for the Network effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_network() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Network", "default_network", 10));
    });
}

/// Push a handler for the Sensor effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_sensor() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Sensor", "default_sensor", 11));
    });
}

/// Push a handler for the Exn effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_exn() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Exn", "default_exn", 12));
    });
}

/// Push a handler for the Causal effect
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_push_handler_causal() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.push(RuntimeHandler::new("Causal", "default_causal", 13));
    });
}

/// Pop the topmost handler from the stack
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_pop_handler() {
    let _ = try_with_handler_stack_mut(|stack| {
        stack.pop();
    });
}

/// Get the current handler stack depth
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_handler_depth() -> i64 {
    try_with_handler_stack(|stack| stack.depth() as i64).unwrap_or(0)
}

// ============================================================================
// Dispatch Functions for Each Effect/Operation Combination
// ============================================================================

// --- IO Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_io_print(value: f64) -> f64 {
    if let Some(v) =
        try_with_handler_stack_mut(|stack| stack.dispatch("IO", "print", &[value]).unwrap_or(0.0))
    {
        v
    } else {
        println!("{}", value);
        0.0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_io_println(value: f64) -> f64 {
    if let Some(v) = try_with_handler_stack_mut(|stack| {
        stack.dispatch("IO", "println", &[value]).unwrap_or(0.0)
    }) {
        v
    } else {
        println!("{}", value);
        0.0
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_io_read() -> f64 {
    // Read a line and parse as f64
    let mut input = String::new();
    if std::io::stdin().read_line(&mut input).is_ok() {
        input.trim().parse().unwrap_or(0.0)
    } else {
        0.0
    }
}

// --- Mut Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_mut_get(key: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Mut", "get", &[key]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_mut_set(key: f64, value: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack.dispatch("Mut", "set", &[key, value]).unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_mut_modify(key: f64, delta: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Mut", "modify", &[key, delta])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

// --- Div Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_div_div(a: f64, b: f64) -> f64 {
    if let Some(v) =
        try_with_handler_stack_mut(|stack| stack.dispatch("Div", "div", &[a, b]).unwrap_or(0.0))
    {
        v
    } else if b == 0.0 {
        0.0
    } else {
        a / b
    }
}

// --- Prob Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_prob_sample(low: f64, high: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Prob", "sample", &[low, high])
            .unwrap_or(0.0)
    })
    .unwrap_or(low) // Fallback to low bound
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_prob_observe(mean: f64, std: f64, value: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Prob", "observe", &[mean, std, value])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

// --- Alloc Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_alloc_alloc(size: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Alloc", "alloc", &[size]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_alloc_dealloc(ptr: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Alloc", "dealloc", &[ptr]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

// --- Panic Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_panic_panic() -> f64 {
    if let Some(v) =
        try_with_handler_stack_mut(|stack| stack.dispatch("Panic", "panic", &[]).unwrap_or(f64::NAN))
    {
        v
    } else {
        eprintln!("Runtime panic!");
        f64::NAN
    }
}

// --- GPU Effect ---

/// Synchronize GPU execution (wait for all kernels to complete)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_sync() -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("GPU", "sync", &[]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Allocate GPU memory (returns buffer_id)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_alloc(size: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("GPU", "alloc", &[size]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Free GPU memory
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_free(buffer_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("GPU", "free", &[buffer_id]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Copy data from host to device
/// host_ptr is encoded as f64 (use transmute carefully)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_copy_htod(buffer_id: f64, host_ptr: f64, size: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("GPU", "copy_htod", &[buffer_id, host_ptr, size])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Copy data from device to host
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_copy_dtoh(buffer_id: f64, host_ptr: f64, size: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("GPU", "copy_dtoh", &[buffer_id, host_ptr, size])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Load PTX kernel (returns kernel_id)
/// name_ptr and ptx_ptr are encoded as f64
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_load_ptx(name_ptr: f64, ptx_ptr: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("GPU", "load_ptx", &[name_ptr, ptx_ptr])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_gpu_launch(
    kernel_id: f64,
    grid_x: f64,
    grid_y: f64,
    grid_z: f64,
) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("GPU", "launch", &[kernel_id, grid_x, grid_y, grid_z])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

// --- Async Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_spawn(task_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Async", "spawn", &[task_id]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_yield() -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Async", "yield", &[]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_await(task_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Async", "await", &[task_id]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Join multiple tasks - waits for all to complete
/// task_ids is encoded as: count followed by task IDs (passed through additional calls)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_join(task_count: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Async", "join", &[task_count]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Add a task to a join set
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_join_add(join_id: f64, task_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "join_add", &[join_id, task_id])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Wait for a join to complete
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_join_await(join_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "join_await", &[join_id])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Select from multiple tasks - waits for any to complete
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_select(task_count: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "select", &[task_count])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Add a task to a select set
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_select_add(select_id: f64, task_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "select_add", &[select_id, task_id])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Wait for a select to complete
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_async_select_await(select_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "select_await", &[select_id])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

// --- Channel Operations ---

/// Create a new unbounded channel, returns channel_id
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_channel_create() -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Async", "channel_create", &[]).unwrap_or(0.0))
        .unwrap_or(0.0)
}

/// Create a new bounded channel with capacity, returns channel_id
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_channel_create_bounded(capacity: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "channel_create_bounded", &[capacity])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Send a value to a channel (returns 1.0 on success, 0.0 on failure)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_channel_send(channel_id: f64, value: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "channel_send", &[channel_id, value])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Receive a value from a channel (returns value or NaN if empty/closed)
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_channel_recv(channel_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "channel_recv", &[channel_id])
            .unwrap_or(f64::NAN)
    })
    .unwrap_or(f64::NAN)
}

/// Close a channel
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_channel_close(channel_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "channel_close", &[channel_id])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

/// Check if a channel is closed
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_channel_is_closed(channel_id: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Async", "channel_is_closed", &[channel_id])
            .unwrap_or(0.0)
    })
    .unwrap_or(0.0)
}

// --- Grad Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_grad_forward(value: f64, derivative: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Grad", "forward", &[value, derivative])
            .unwrap_or(0.0)
    })
    .unwrap_or(derivative)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_grad_reverse(value: f64, adjoint: f64) -> f64 {
    try_with_handler_stack_mut(|stack| {
        stack
            .dispatch("Grad", "reverse", &[value, adjoint])
            .unwrap_or(0.0)
    })
    .unwrap_or(adjoint)
}

// --- Causal Effect ---

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_causal_do(value: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Causal", "do", &[value]).unwrap_or(value))
        .unwrap_or(value)
}

#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_causal_observe(value: f64) -> f64 {
    try_with_handler_stack_mut(|stack| stack.dispatch("Causal", "observe", &[value]).unwrap_or(value))
        .unwrap_or(value)
}

// --- Generic Dispatch ---

/// Generic dispatch function that takes effect and op as C strings
#[unsafe(no_mangle)]
pub extern "C" fn __sounio_dispatch_generic(
    effect_ptr: *const u8,
    op_ptr: *const u8,
    args_ptr: *const f64,
    args_len: usize,
) -> f64 {
    if effect_ptr.is_null() || op_ptr.is_null() {
        return f64::NAN;
    }

    let effect = unsafe {
        CStr::from_ptr(effect_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    let op = unsafe {
        CStr::from_ptr(op_ptr as *const std::ffi::c_char)
            .to_str()
            .unwrap_or("")
    };

    let args: Vec<f64> = if args_ptr.is_null() || args_len == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(args_ptr, args_len).to_vec() }
    };

    try_with_handler_stack_mut(|stack| stack.dispatch(effect, op, &args).unwrap_or(f64::NAN))
        .unwrap_or(f64::NAN)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_stack_push_pop() {
        let mut stack = RuntimeHandlerStack::new();
        assert_eq!(stack.depth(), 0);

        stack.push(RuntimeHandler::new("IO", "test", 1));
        assert_eq!(stack.depth(), 1);

        stack.push(RuntimeHandler::new("Mut", "test2", 2));
        assert_eq!(stack.depth(), 2);

        let handler = stack.pop();
        assert!(handler.is_some());
        assert_eq!(handler.unwrap().effect, "Mut");
        assert_eq!(stack.depth(), 1);

        stack.pop();
        assert_eq!(stack.depth(), 0);
    }

    #[test]
    fn test_handler_stack_find() {
        let mut stack = RuntimeHandlerStack::new();
        stack.push(RuntimeHandler::new("IO", "io_handler", 1));
        stack.push(RuntimeHandler::new("Mut", "mut_handler", 2));
        stack.push(RuntimeHandler::new("IO", "io_handler2", 3));

        // Should find the most recent IO handler
        let handler = stack.find_handler("IO");
        assert!(handler.is_some());
        assert_eq!(handler.unwrap().name, "io_handler2");

        // Should find Mut handler
        let handler = stack.find_handler("Mut");
        assert!(handler.is_some());
        assert_eq!(handler.unwrap().name, "mut_handler");

        // Should not find nonexistent handler
        assert!(stack.find_handler("Prob").is_none());
    }

    #[test]
    fn test_handler_dispatch_io() {
        let mut stack = RuntimeHandlerStack::new();
        stack.push(RuntimeHandler::new("IO", "default", 1));

        let result = stack.dispatch("IO", "print", &[42.0]);
        assert_eq!(result, Some(0.0));
    }

    #[test]
    fn test_handler_dispatch_mut() {
        let mut stack = RuntimeHandlerStack::new();
        stack.push(RuntimeHandler::new("Mut", "default", 2));

        // Set a value
        let result = stack.dispatch("Mut", "set", &[1.0, 42.0]);
        assert_eq!(result, Some(42.0));

        // Get the value
        let result = stack.dispatch("Mut", "get", &[1.0]);
        assert_eq!(result, Some(42.0));

        // Modify the value
        let result = stack.dispatch("Mut", "modify", &[1.0, 10.0]);
        assert_eq!(result, Some(52.0));
    }

    #[test]
    fn test_handler_dispatch_div() {
        let mut stack = RuntimeHandlerStack::new();
        stack.push(RuntimeHandler::new("Div", "default", 3));

        // Normal division
        let result = stack.dispatch("Div", "div", &[10.0, 2.0]);
        assert_eq!(result, Some(5.0));

        // Division by zero
        let result = stack.dispatch("Div", "div", &[10.0, 0.0]);
        assert_eq!(result, Some(0.0));
    }

    #[test]
    fn test_global_handler_stack() {
        reset_handler_stack();

        __sounio_push_handler_io();
        assert_eq!(__sounio_handler_depth(), 1);

        __sounio_push_handler_mut();
        assert_eq!(__sounio_handler_depth(), 2);

        __sounio_pop_handler();
        assert_eq!(__sounio_handler_depth(), 1);

        __sounio_pop_handler();
        assert_eq!(__sounio_handler_depth(), 0);
    }

    #[test]
    fn test_dispatch_functions() {
        reset_handler_stack();
        __sounio_push_handler_div();

        let result = __sounio_dispatch_div_div(20.0, 4.0);
        assert_eq!(result, 5.0);

        let result = __sounio_dispatch_div_div(10.0, 0.0);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_mut_dispatch() {
        reset_handler_stack();
        __sounio_push_handler_mut();

        __sounio_dispatch_mut_set(1.0, 100.0);
        let result = __sounio_dispatch_mut_get(1.0);
        assert_eq!(result, 100.0);

        let result = __sounio_dispatch_mut_modify(1.0, 50.0);
        assert_eq!(result, 150.0);
    }
}
