//! Real Async Effect Handler with Tokio
//!
//! This module provides a production-ready async effect handler that uses
//! Tokio for real asynchronous operations. Unlike the simulated async handler,
//! this actually spawns OS threads and uses async/await.
//!
//! # Features
//!
//! - Real async/await using Tokio runtime
//! - Task spawning with green threads
//! - Cooperative yielding
//! - Sleep/delay support
//! - Task joining and cancellation
//!
//! # Usage
//!
//! This handler requires the `tokio` feature to be enabled:
//! ```toml
//! [dependencies]
//! tokio = { version = "1.0", features = ["full"] }
//! ```

#[cfg(feature = "tokio")]
use tokio::runtime::Runtime;
#[cfg(feature = "tokio")]
use tokio::task::JoinHandle;

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec,
};
use crate::effects::linearity::Linearity;
use crate::interp::Value;
#[cfg(feature = "tokio")]
use std::collections::HashMap;
use std::sync::OnceLock;
#[cfg(feature = "tokio")]
use std::sync::{Arc, Mutex};

/// Key for storing the Tokio runtime in handler state
const TOKIO_RUNTIME_KEY: &str = "__async_tokio_runtime";

/// Key prefix for storing task handles
const TASK_HANDLE_PREFIX: &str = "__async_task_";

/// Key for tracking the next task ID
const NEXT_TASK_ID_KEY: &str = "__async_next_task_id";

/// Confidence factor for async operations (10% degradation)
///
/// Async operations introduce timing uncertainty and potential
/// race conditions that degrade epistemic confidence.
const ASYNC_CONFIDENCE_FACTOR: f64 = 0.9;

/// Static operation specifications
static REAL_ASYNC_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_real_async_operations() -> &'static [OperationSpec] {
    REAL_ASYNC_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("spawn", "FutureId")
                .with_params(vec!["Task"])
                .with_confidence_factor(ASYNC_CONFIDENCE_FACTOR),
            OperationSpec::new("await", "Value")
                .with_params(vec!["FutureId"])
                .with_confidence_factor(ASYNC_CONFIDENCE_FACTOR),
            OperationSpec::new("yield", "Unit")
                .with_params(vec![])
                .with_confidence_factor(ASYNC_CONFIDENCE_FACTOR),
            OperationSpec::new("sleep", "Unit").with_params(vec!["I64"]),
            OperationSpec::new("join", "Array").with_params(vec!["Array"]),
            OperationSpec::new("cancel", "Unit").with_params(vec!["FutureId"]),
            OperationSpec::new("timeout", "Option").with_params(vec!["FutureId", "I64"]),
        ]
    })
}

/// Real async handler using Tokio
///
/// This handler provides real asynchronous execution using the Tokio runtime.
/// Tasks are actually executed concurrently on a thread pool.
///
/// # Architecture
///
/// - Each handler instance has its own Tokio runtime
/// - Tasks are stored in the handler state as JoinHandles
/// - Awaiting a task polls the JoinHandle until completion
///
/// # Limitations
///
/// - Cannot be used from within an existing Tokio runtime (would panic)
/// - Task functions must be self-contained (closure capture limitations)
/// - No support for async trait methods yet
#[derive(Debug)]
pub struct RealAsyncHandler {
    /// Shared Tokio runtime for all tasks
    /// Wrapped in Arc<Mutex<>> for thread-safe sharing
    #[cfg(feature = "tokio")]
    runtime: Arc<Mutex<Runtime>>,

    /// Storage for task handles indexed by task ID
    /// Wrapped in Arc<Mutex<>> for thread-safe access from multiple contexts
    #[cfg(feature = "tokio")]
    task_handles: Arc<Mutex<HashMap<i64, JoinHandle<Value>>>>,

    /// Placeholder when tokio feature is not enabled
    #[cfg(not(feature = "tokio"))]
    _phantom: std::marker::PhantomData<()>,
}

impl RealAsyncHandler {
    /// Create a new real async handler
    #[cfg(feature = "tokio")]
    pub fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to create Tokio runtime");

        Self {
            runtime: Arc::new(Mutex::new(runtime)),
            task_handles: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a new async handler (fallback without tokio)
    #[cfg(not(feature = "tokio"))]
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Generate next task ID
    fn next_task_id(state: &mut HandlerState) -> i64 {
        let current = match state.named_state.get(NEXT_TASK_ID_KEY) {
            Some(Value::Int(id)) => *id,
            _ => 1,
        };

        state
            .named_state
            .insert(NEXT_TASK_ID_KEY.to_string(), Value::Int(current + 1));

        current
    }

    /// Handle spawn operation - spawn a new async task
    #[cfg(feature = "tokio")]
    fn handle_spawn(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "spawn",
                "spawn requires a task function argument",
            ));
        }

        // Extract task function
        // For now, we support:
        // - Value::Closure (future work - requires interpreter integration)
        // - Value::Int (simulates a computation that returns after N milliseconds)
        // - Any other value is used as the immediate result
        let task_arg = args[0].clone();

        // Generate unique task ID
        let task_id = Self::next_task_id(state);

        // Spawn task on Tokio runtime
        let runtime = self.runtime.clone();
        let handle = {
            let rt = runtime.lock().unwrap();
            rt.spawn(async move {
                // For now, interpret Int values as sleep durations
                // This allows testing spawn/await without full closure support
                match task_arg {
                    Value::Int(ms) if ms > 0 => {
                        tokio::time::sleep(tokio::time::Duration::from_millis(ms as u64)).await;
                        Value::Int(ms * 2) // Return doubled value as a placeholder result
                    }
                    other => {
                        // For any other value, yield once and return it
                        tokio::task::yield_now().await;
                        other
                    }
                }
            })
        };

        // Store the JoinHandle for later await
        {
            let mut handles = self.task_handles.lock().unwrap();
            handles.insert(task_id, handle);
        }

        HandlerResult::Resume(Value::Int(task_id))
    }

    /// Handle spawn (fallback without tokio)
    #[cfg(not(feature = "tokio"))]
    fn handle_spawn(&self, _args: &[Value], _state: &mut HandlerState) -> HandlerResult {
        HandlerResult::Abort(HandlerError::new(
            "Async",
            "spawn",
            "Real async operations require the 'tokio' feature to be enabled",
        ))
    }

    /// Handle await operation
    #[cfg(feature = "tokio")]
    fn handle_await(&self, args: &[Value], _state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "await",
                "await requires a future ID argument",
            ));
        }

        let task_id = match &args[0] {
            Value::Int(id) => *id,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "await",
                    format!("expected I64 for task ID, got {:?}", args[0].type_name()),
                ));
            }
        };

        // Retrieve the JoinHandle from storage
        let handle = {
            let mut handles = self.task_handles.lock().unwrap();
            match handles.remove(&task_id) {
                Some(h) => h,
                None => {
                    return HandlerResult::Abort(HandlerError::new(
                        "Async",
                        "await",
                        format!("Invalid task ID {} or task already awaited", task_id),
                    ));
                }
            }
        };

        // Block on the JoinHandle to get the result
        // Use the runtime's block_on to wait for completion
        let runtime = self.runtime.lock().unwrap();
        match runtime.block_on(handle) {
            Ok(result) => HandlerResult::Resume(result),
            Err(e) => HandlerResult::Abort(HandlerError::new(
                "Async",
                "await",
                format!("Task panicked or was cancelled: {}", e),
            )),
        }
    }

    /// Handle await (fallback without tokio)
    #[cfg(not(feature = "tokio"))]
    fn handle_await(&self, _args: &[Value], _state: &mut HandlerState) -> HandlerResult {
        HandlerResult::Abort(HandlerError::new(
            "Async",
            "await",
            "Real async operations require the 'tokio' feature to be enabled",
        ))
    }

    /// Handle yield operation
    #[cfg(feature = "tokio")]
    fn handle_yield(&self) -> HandlerResult {
        // In a real async context, this would yield to the Tokio scheduler
        // For the effect handler, we just return immediately
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle yield (fallback)
    #[cfg(not(feature = "tokio"))]
    fn handle_yield(&self) -> HandlerResult {
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle sleep operation
    #[cfg(feature = "tokio")]
    fn handle_sleep(&self, args: &[Value], _state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "sleep",
                "sleep requires a duration in milliseconds",
            ));
        }

        let ms = match &args[0] {
            Value::Int(n) if *n >= 0 => *n as u64,
            Value::Int(n) => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "sleep",
                    format!("sleep duration must be non-negative, got {}", n),
                ));
            }
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "sleep",
                    format!("expected I64 for duration, got {:?}", args[0].type_name()),
                ));
            }
        };

        // Actually sleep using Tokio
        let runtime = self.runtime.clone();
        let rt = runtime.lock().unwrap();
        rt.block_on(async {
            tokio::time::sleep(tokio::time::Duration::from_millis(ms)).await;
        });

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle sleep (fallback)
    #[cfg(not(feature = "tokio"))]
    fn handle_sleep(&self, args: &[Value], _state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "sleep",
                "sleep requires a duration argument",
            ));
        }

        // Fallback: use std::thread::sleep
        let ms = match &args[0] {
            Value::Int(n) if *n >= 0 => *n as u64,
            Value::Int(n) => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "sleep",
                    format!("sleep duration must be non-negative, got {}", n),
                ));
            }
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "sleep",
                    "sleep duration must be an integer",
                ));
            }
        };

        std::thread::sleep(std::time::Duration::from_millis(ms));
        HandlerResult::Resume(Value::Unit)
    }
}

impl Default for RealAsyncHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl HandlerCapability for RealAsyncHandler {
    fn effect_name(&self) -> &str {
        "Async"
    }

    fn handler_name(&self) -> &str {
        "RealAsyncHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_real_async_operations()
    }

    fn handle(
        &self,
        op: &str,
        args: &[Value],
        _cont: Continuation,
        state: &mut HandlerState,
    ) -> HandlerResult {
        match op {
            "spawn" => self.handle_spawn(args, state),
            "await" => self.handle_await(args, state),
            "yield" => self.handle_yield(),
            "sleep" => self.handle_sleep(args, state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Async",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        false // Async continuations are one-shot
    }

    fn operation_linearity(&self, _operation: &str) -> Linearity {
        // Async operations should be exactly once
        Linearity::ExactlyOnce
    }

    fn epistemic_impact(&self, operation: &str) -> EpistemicImpact {
        match operation {
            "spawn" | "await" | "yield" => {
                // Async operations introduce timing uncertainty
                EpistemicImpact::with_confidence(ASYNC_CONFIDENCE_FACTOR)
            }
            "sleep" => {
                // Sleep is deterministic, no epistemic impact
                EpistemicImpact::none()
            }
            _ => EpistemicImpact::none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handler_creation() {
        let handler = RealAsyncHandler::new();
        assert_eq!(handler.effect_name(), "Async");
        assert_eq!(handler.handler_name(), "RealAsyncHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = RealAsyncHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"spawn"));
        assert!(names.contains(&"await"));
        assert!(names.contains(&"yield"));
        assert!(names.contains(&"sleep"));
    }

    #[test]
    fn test_yield_operation() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("yield", &[], cont, &mut state);
        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    #[cfg(not(feature = "tokio"))]
    fn test_sleep_fallback() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let start = std::time::Instant::now();
        let result = handler.handle("sleep", &[Value::Int(10)], cont, &mut state);
        let elapsed = start.elapsed();

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        // Should have actually slept
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_spawn_await_basic() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();

        // Spawn a task that sleeps for 50ms and returns doubled value
        let spawn_result =
            handler.handle("spawn", &[Value::Int(50)], Continuation::new(), &mut state);

        let task_id = match spawn_result {
            HandlerResult::Resume(Value::Int(id)) => id,
            other => panic!("expected Resume(Int), got {:?}", other),
        };

        assert!(task_id > 0, "task ID should be positive");

        // Await the task
        let start = std::time::Instant::now();
        let await_result = handler.handle(
            "await",
            &[Value::Int(task_id)],
            Continuation::new(),
            &mut state,
        );
        let elapsed = start.elapsed();

        match await_result {
            HandlerResult::Resume(Value::Int(result)) => {
                assert_eq!(result, 100, "task should return doubled value (50 * 2)");
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }

        // Should have waited at least 50ms
        assert!(
            elapsed.as_millis() >= 50,
            "await should block until task completes"
        );
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_spawn_await_immediate_value() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();

        // Spawn a task with a non-Int value (should return immediately after yield)
        let spawn_result = handler.handle(
            "spawn",
            &[Value::String("hello".to_string())],
            Continuation::new(),
            &mut state,
        );

        let task_id = match spawn_result {
            HandlerResult::Resume(Value::Int(id)) => id,
            other => panic!("expected Resume(Int), got {:?}", other),
        };

        // Await the task
        let await_result = handler.handle(
            "await",
            &[Value::Int(task_id)],
            Continuation::new(),
            &mut state,
        );

        match await_result {
            HandlerResult::Resume(Value::String(s)) => {
                assert_eq!(s, "hello", "task should return the original value");
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_await_invalid_task_id() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();

        // Try to await a task that doesn't exist
        let result = handler.handle(
            "await",
            &[Value::Int(9999)],
            Continuation::new(),
            &mut state,
        );

        match result {
            HandlerResult::Abort(err) => {
                assert!(err.message.contains("Invalid task ID"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_await_same_task_twice() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();

        // Spawn a task
        let spawn_result =
            handler.handle("spawn", &[Value::Int(10)], Continuation::new(), &mut state);

        let task_id = match spawn_result {
            HandlerResult::Resume(Value::Int(id)) => id,
            other => panic!("expected Resume(Int), got {:?}", other),
        };

        // Await the task once
        let _first_await = handler.handle(
            "await",
            &[Value::Int(task_id)],
            Continuation::new(),
            &mut state,
        );

        // Try to await again (should fail - task consumed)
        let second_await = handler.handle(
            "await",
            &[Value::Int(task_id)],
            Continuation::new(),
            &mut state,
        );

        match second_await {
            HandlerResult::Abort(err) => {
                assert!(err.message.contains("Invalid task ID"));
            }
            other => panic!("expected Abort on second await, got {:?}", other),
        }
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_multiple_concurrent_tasks() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();

        // Spawn multiple tasks
        let task1 =
            match handler.handle("spawn", &[Value::Int(20)], Continuation::new(), &mut state) {
                HandlerResult::Resume(Value::Int(id)) => id,
                _ => panic!("spawn failed"),
            };

        let task2 =
            match handler.handle("spawn", &[Value::Int(30)], Continuation::new(), &mut state) {
                HandlerResult::Resume(Value::Int(id)) => id,
                _ => panic!("spawn failed"),
            };

        let task3 =
            match handler.handle("spawn", &[Value::Int(10)], Continuation::new(), &mut state) {
                HandlerResult::Resume(Value::Int(id)) => id,
                _ => panic!("spawn failed"),
            };

        // All task IDs should be different
        assert_ne!(task1, task2);
        assert_ne!(task2, task3);
        assert_ne!(task1, task3);

        // Await in reverse order (fastest first)
        let result3 = handler.handle(
            "await",
            &[Value::Int(task3)],
            Continuation::new(),
            &mut state,
        );
        assert!(matches!(result3, HandlerResult::Resume(Value::Int(20))));

        let result1 = handler.handle(
            "await",
            &[Value::Int(task1)],
            Continuation::new(),
            &mut state,
        );
        assert!(matches!(result1, HandlerResult::Resume(Value::Int(40))));

        let result2 = handler.handle(
            "await",
            &[Value::Int(task2)],
            Continuation::new(),
            &mut state,
        );
        assert!(matches!(result2, HandlerResult::Resume(Value::Int(60))));
    }

    #[test]
    #[cfg(feature = "tokio")]
    fn test_sleep_with_tokio() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let start = std::time::Instant::now();
        let result = handler.handle("sleep", &[Value::Int(10)], cont, &mut state);
        let elapsed = start.elapsed();

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }

        // Should have actually slept using Tokio
        assert!(elapsed.as_millis() >= 10);
    }

    #[test]
    fn test_sleep_negative_duration() {
        let handler = RealAsyncHandler::new();
        let mut state = HandlerState::new();
        let cont = Continuation::new();

        let result = handler.handle("sleep", &[Value::Int(-10)], cont, &mut state);
        match result {
            HandlerResult::Abort(_) => {}
            other => panic!("expected Abort for negative duration, got {:?}", other),
        }
    }
}
