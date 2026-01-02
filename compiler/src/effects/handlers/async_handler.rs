//! Async Effect Handler
//!
//! This module implements the `Async` effect handler for asynchronous computation
//! through algebraic effects. The Async effect enables structured concurrency
//! with explicit effect tracking.
//!
//! # Effect Operations
//!
//! | Operation  | Arguments                    | Return Type   | Description                      |
//! |-----------|------------------------------|---------------|----------------------------------|
//! | `spawn`   | `(task: Function)`           | `FutureId`    | Spawn an async task              |
//! | `await`   | `(future: FutureId)`         | `Value`       | Await a future's result          |
//! | `yield`   | `()`                         | `Unit`        | Yield control to scheduler       |
//! | `sleep`   | `(ms: I64)`                  | `Unit`        | Sleep for milliseconds           |
//! | `join`    | `(futures: Array<FutureId>)` | `Array<Value>`| Await multiple futures           |
//! | `select`  | `(futures: Array<FutureId>)` | `(I64, Value)`| Await first completed future     |
//! | `cancel`  | `(future: FutureId)`         | `Bool`        | Cancel a spawned task            |
//! | `is_ready`| `(future: FutureId)`         | `Bool`        | Check if future is complete      |
//!
//! # Epistemic Impact
//!
//! Async operations do not inherently degrade epistemic confidence. The timing
//! of async operations may be non-deterministic, but the values produced are
//! as reliable as their underlying computations.
//!
//! # Execution Model
//!
//! This handler simulates async execution for the interpreter. Futures are
//! tracked by ID, and the handler maintains:
//! - A task queue of pending futures
//! - Results for completed futures
//! - Cancellation status
//!
//! In a real runtime, this would integrate with an async executor (e.g., tokio).
//!
//! # Example
//!
//! ```text
//! // Sounio code using Async effect:
//! fn fetch_data() -> string with Async, IO {
//!     let future1 = spawn(|| read_file("data1.txt"))
//!     let future2 = spawn(|| read_file("data2.txt"))
//!     let results = join([future1, future2])
//!     results[0] ++ results[1]
//! }
//! ```

use crate::effects::handler_capability::{
    Continuation, EpistemicImpact, HandlerCapability, HandlerError, HandlerResult, HandlerState,
    OperationSpec, SuspensionId,
};
use crate::interp::Value;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

/// Key prefix for storing future state in HandlerState
const FUTURE_PREFIX: &str = "__async_future_";

/// Key prefix for future results
const RESULT_PREFIX: &str = "__async_result_";

/// Key prefix for cancelled futures
const CANCELLED_PREFIX: &str = "__async_cancelled_";

/// Key for pending suspension data
const SUSPEND_DATA_PREFIX: &str = "__async_suspend_";

/// Global counter for future IDs (used across handler instances)
static FUTURE_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Global counter for suspension IDs
static SUSPENSION_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Static operation specifications for AsyncHandler
static ASYNC_OPERATIONS: OnceLock<Vec<OperationSpec>> = OnceLock::new();

fn get_async_operations() -> &'static [OperationSpec] {
    ASYNC_OPERATIONS.get_or_init(|| {
        vec![
            OperationSpec::new("spawn", "FutureId")
                .with_params(vec!["Function"])
                .with_continuation(),
            OperationSpec::new("await", "Value")
                .with_params(vec!["FutureId"])
                .with_continuation(),
            OperationSpec::new("yield", "Unit").with_params(vec![]),
            OperationSpec::new("sleep", "Unit").with_params(vec!["I64"]),
            OperationSpec::new("join", "Array")
                .with_params(vec!["Array"])
                .with_continuation(),
            OperationSpec::new("select", "Tuple")
                .with_params(vec!["Array"])
                .with_continuation(),
            OperationSpec::new("cancel", "Bool").with_params(vec!["FutureId"]),
            OperationSpec::new("is_ready", "Bool").with_params(vec!["FutureId"]),
        ]
    })
}

/// Handler for the Async (asynchronous computation) effect
///
/// AsyncHandler provides structured concurrency through algebraic effects.
/// It simulates async execution for the interpreter by tracking futures
/// and their states.
///
/// # Future Management
///
/// Futures are identified by unique IDs and can be:
/// - Spawned: Creates a new pending future
/// - Awaited: Blocks until the future completes
/// - Cancelled: Marks the future for cancellation
/// - Joined: Awaits multiple futures
/// - Selected: Returns the first completing future
///
/// # Thread Safety
///
/// AsyncHandler uses atomic operations for future ID generation,
/// enabling safe use across threads. State is stored in HandlerState.
#[derive(Debug)]
pub struct AsyncHandler {
    _private: (),
}

impl Default for AsyncHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncHandler {
    /// Create a new AsyncHandler
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate a unique future ID
    fn next_future_id() -> i64 {
        FUTURE_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as i64
    }

    /// Generate a unique suspension ID
    fn next_suspension_id() -> SuspensionId {
        SuspensionId(SUSPENSION_ID_COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the state key for a future
    fn future_key(id: i64) -> String {
        format!("{}{}", FUTURE_PREFIX, id)
    }

    /// Get the result key for a future
    fn result_key(id: i64) -> String {
        format!("{}{}", RESULT_PREFIX, id)
    }

    /// Get the cancelled key for a future
    fn cancelled_key(id: i64) -> String {
        format!("{}{}", CANCELLED_PREFIX, id)
    }

    /// Get the suspension data key
    fn suspend_key(id: u64) -> String {
        format!("{}{}", SUSPEND_DATA_PREFIX, id)
    }

    /// Check if a value is a function/closure
    fn is_function(value: &Value) -> bool {
        matches!(value, Value::Function { .. } | Value::Builtin(_))
    }

    /// Handle spawn operation - create a new async task
    fn handle_spawn(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "spawn",
                "spawn requires a task function argument",
            ));
        }

        // Validate that the argument is a function
        if !Self::is_function(&args[0]) {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "spawn",
                format!(
                    "expected Function for task, got {:?}",
                    args[0].type_name()
                ),
            ));
        }

        let task = args[0].clone();
        let future_id = Self::next_future_id();
        let key = Self::future_key(future_id);

        // Store the task as pending
        state.named_state.insert(key, task);

        // Return the future ID
        HandlerResult::Resume(Value::Int(future_id))
    }

    /// Handle await operation - wait for a future to complete
    fn handle_await(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "await",
                "await requires a future ID argument",
            ));
        }

        let future_id = match &args[0] {
            Value::Int(id) => *id,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "await",
                    format!(
                        "expected FutureId (I64) for future, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        // Check if cancelled
        let cancelled_key = Self::cancelled_key(future_id);
        if matches!(state.named_state.get(&cancelled_key), Some(Value::Bool(true))) {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "await",
                format!("future {} was cancelled", future_id),
            ));
        }

        // Check if result is already available
        let result_key = Self::result_key(future_id);
        if let Some(result) = state.named_state.get(&result_key) {
            return HandlerResult::Resume(result.clone());
        }

        // Check if future exists - if so, we need to suspend
        let future_key = Self::future_key(future_id);
        if state.named_state.contains_key(&future_key) {
            // Create a suspension - the interpreter should execute the task
            // and call set_future_result, then resume
            let suspension_id = Self::next_suspension_id();

            // Store suspension data so interpreter knows what to do
            state.named_state.insert(
                Self::suspend_key(suspension_id.0),
                Value::Tuple(vec![
                    Value::String("await".into()),
                    Value::Int(future_id),
                ]),
            );

            HandlerResult::Suspend(suspension_id)
        } else {
            HandlerResult::Abort(HandlerError::new(
                "Async",
                "await",
                format!("future {} not found", future_id),
            ))
        }
    }

    /// Handle yield operation - yield control to scheduler
    fn handle_yield(&self) -> HandlerResult {
        // In simulation, yield is a no-op that returns immediately
        // In a real runtime, this would yield to the async scheduler
        HandlerResult::Resume(Value::Unit)
    }

    /// Handle sleep operation - sleep for specified milliseconds
    fn handle_sleep(&self, args: &[Value]) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "sleep",
                "sleep requires a duration in milliseconds",
            ));
        }

        let ms = match &args[0] {
            Value::Int(n) => *n,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "sleep",
                    format!("expected I64 for duration, got {:?}", args[0].type_name()),
                ));
            }
        };

        if ms < 0 {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "sleep",
                format!("sleep duration must be non-negative, got {}", ms),
            ));
        }

        // In simulation/test, sleep is immediate
        // In a real runtime, this would actually sleep
        #[cfg(not(test))]
        if ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(ms as u64));
        }

        HandlerResult::Resume(Value::Unit)
    }

    /// Handle join operation - await multiple futures
    fn handle_join(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "join",
                "join requires an array of future IDs",
            ));
        }

        let futures_rc = match &args[0] {
            Value::Array(arr) => arr.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "join",
                    format!(
                        "expected Array of FutureIds, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        let futures = futures_rc.borrow();
        let mut results = Vec::with_capacity(futures.len());
        let mut pending = Vec::new();

        for (idx, future_val) in futures.iter().enumerate() {
            let future_id = match future_val {
                Value::Int(id) => *id,
                _ => {
                    return HandlerResult::Abort(HandlerError::new(
                        "Async",
                        "join",
                        format!(
                            "expected FutureId (I64) at index {}, got {:?}",
                            idx,
                            future_val.type_name()
                        ),
                    ));
                }
            };

            // Check for cancellation
            let cancelled_key = Self::cancelled_key(future_id);
            if matches!(state.named_state.get(&cancelled_key), Some(Value::Bool(true))) {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "join",
                    format!("future {} in join was cancelled", future_id),
                ));
            }

            // Check if result is available
            let result_key = Self::result_key(future_id);
            if let Some(result) = state.named_state.get(&result_key) {
                results.push(result.clone());
            } else {
                pending.push(future_id);
            }
        }
        drop(futures);

        if pending.is_empty() {
            // All futures completed
            HandlerResult::Resume(Value::Array(Rc::new(RefCell::new(results))))
        } else {
            // Some futures still pending - suspend
            let suspension_id = Self::next_suspension_id();
            state.named_state.insert(
                Self::suspend_key(suspension_id.0),
                Value::Tuple(vec![
                    Value::String("join".into()),
                    args[0].clone(),
                ]),
            );
            HandlerResult::Suspend(suspension_id)
        }
    }

    /// Handle select operation - await first completing future
    fn handle_select(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "select",
                "select requires an array of future IDs",
            ));
        }

        let futures_rc = match &args[0] {
            Value::Array(arr) => arr.clone(),
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "select",
                    format!(
                        "expected Array of FutureIds, got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        let futures = futures_rc.borrow();

        if futures.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "select",
                "select requires at least one future",
            ));
        }

        // Find first completed future
        for (idx, future_val) in futures.iter().enumerate() {
            let future_id = match future_val {
                Value::Int(id) => *id,
                _ => continue,
            };

            // Skip cancelled futures
            let cancelled_key = Self::cancelled_key(future_id);
            if matches!(state.named_state.get(&cancelled_key), Some(Value::Bool(true))) {
                continue;
            }

            // Check if result is available
            let result_key = Self::result_key(future_id);
            if let Some(result) = state.named_state.get(&result_key) {
                // Return (index, value) tuple
                return HandlerResult::Resume(Value::Tuple(vec![
                    Value::Int(idx as i64),
                    result.clone(),
                ]));
            }
        }
        drop(futures);

        // No future completed yet - suspend
        let suspension_id = Self::next_suspension_id();
        state.named_state.insert(
            Self::suspend_key(suspension_id.0),
            Value::Tuple(vec![
                Value::String("select".into()),
                args[0].clone(),
            ]),
        );
        HandlerResult::Suspend(suspension_id)
    }

    /// Handle cancel operation - cancel a spawned task
    fn handle_cancel(&self, args: &[Value], state: &mut HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "cancel",
                "cancel requires a future ID argument",
            ));
        }

        let future_id = match &args[0] {
            Value::Int(id) => *id,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "cancel",
                    format!(
                        "expected FutureId (I64), got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        let future_key = Self::future_key(future_id);
        let cancelled_key = Self::cancelled_key(future_id);

        // Check if future exists
        if state.named_state.contains_key(&future_key)
            || state.named_state.contains_key(&Self::result_key(future_id))
        {
            // Mark as cancelled
            state
                .named_state
                .insert(cancelled_key, Value::Bool(true));
            HandlerResult::Resume(Value::Bool(true))
        } else {
            // Future doesn't exist
            HandlerResult::Resume(Value::Bool(false))
        }
    }

    /// Handle is_ready operation - check if future is complete
    fn handle_is_ready(&self, args: &[Value], state: &HandlerState) -> HandlerResult {
        if args.is_empty() {
            return HandlerResult::Abort(HandlerError::new(
                "Async",
                "is_ready",
                "is_ready requires a future ID argument",
            ));
        }

        let future_id = match &args[0] {
            Value::Int(id) => *id,
            _ => {
                return HandlerResult::Abort(HandlerError::new(
                    "Async",
                    "is_ready",
                    format!(
                        "expected FutureId (I64), got {:?}",
                        args[0].type_name()
                    ),
                ));
            }
        };

        let result_key = Self::result_key(future_id);
        let is_ready = state.named_state.contains_key(&result_key);

        HandlerResult::Resume(Value::Bool(is_ready))
    }

    /// Set the result of a future (called by interpreter after executing task)
    pub fn set_future_result(state: &mut HandlerState, future_id: i64, result: Value) {
        let result_key = Self::result_key(future_id);
        state.named_state.insert(result_key, result);

        // Remove the pending task
        let future_key = Self::future_key(future_id);
        state.named_state.remove(&future_key);
    }

    /// Get the result of a future if available
    pub fn get_future_result(state: &HandlerState, future_id: i64) -> Option<Value> {
        let result_key = Self::result_key(future_id);
        state.named_state.get(&result_key).cloned()
    }

    /// Get the pending task for a future if available
    pub fn get_pending_task(state: &HandlerState, future_id: i64) -> Option<Value> {
        let future_key = Self::future_key(future_id);
        state.named_state.get(&future_key).cloned()
    }

    /// Check if a future is cancelled
    pub fn is_cancelled(state: &HandlerState, future_id: i64) -> bool {
        let cancelled_key = Self::cancelled_key(future_id);
        matches!(state.named_state.get(&cancelled_key), Some(Value::Bool(true)))
    }

    /// Get suspension data for resumption
    pub fn get_suspension_data(state: &HandlerState, suspension_id: SuspensionId) -> Option<Value> {
        let key = Self::suspend_key(suspension_id.0);
        state.named_state.get(&key).cloned()
    }
}

impl HandlerCapability for AsyncHandler {
    fn effect_name(&self) -> &str {
        "Async"
    }

    fn handler_name(&self) -> &str {
        "AsyncHandler"
    }

    fn operations(&self) -> &[OperationSpec] {
        get_async_operations()
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
            "sleep" => self.handle_sleep(args),
            "join" => self.handle_join(args, state),
            "select" => self.handle_select(args, state),
            "cancel" => self.handle_cancel(args, state),
            "is_ready" => self.handle_is_ready(args, state),
            _ => HandlerResult::Abort(HandlerError::new(
                "Async",
                op,
                format!("unknown operation: {}", op),
            )),
        }
    }

    fn supports_multi_shot(&self) -> bool {
        false
    }

    fn epistemic_impact(&self, _operation: &str) -> EpistemicImpact {
        // Async operations do not degrade confidence
        // Timing may vary but values are deterministic
        EpistemicImpact::none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::HirFn;
    use std::collections::HashMap;

    fn new_state() -> HandlerState {
        HandlerState::new()
    }

    fn cont() -> Continuation {
        Continuation::new()
    }

    fn dummy_function() -> Value {
        // Create a minimal HirFn for testing
        Value::Builtin("test_task".into())
    }

    #[test]
    fn test_handler_identity() {
        let handler = AsyncHandler::new();
        assert_eq!(handler.effect_name(), "Async");
        assert_eq!(handler.handler_name(), "AsyncHandler");
    }

    #[test]
    fn test_operations_list() {
        let handler = AsyncHandler::new();
        let ops = handler.operations();
        let names: Vec<_> = ops.iter().map(|o| o.name.as_str()).collect();
        assert!(names.contains(&"spawn"));
        assert!(names.contains(&"await"));
        assert!(names.contains(&"yield"));
        assert!(names.contains(&"sleep"));
        assert!(names.contains(&"join"));
        assert!(names.contains(&"select"));
        assert!(names.contains(&"cancel"));
        assert!(names.contains(&"is_ready"));
    }

    #[test]
    fn test_does_not_support_multi_shot() {
        let handler = AsyncHandler::new();
        assert!(!handler.supports_multi_shot());
    }

    #[test]
    fn test_epistemic_impact_is_identity() {
        let handler = AsyncHandler::new();

        let spawn_impact = handler.epistemic_impact("spawn");
        assert!((spawn_impact.confidence_factor - 1.0).abs() < 0.001);

        let await_impact = handler.epistemic_impact("await");
        assert!((await_impact.confidence_factor - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spawn_basic() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("spawn", &[dummy_function()], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Int(id)) => {
                assert!(id > 0, "future ID should be positive");
            }
            other => panic!("expected Resume(Int), got {:?}", other),
        }
    }

    #[test]
    fn test_spawn_requires_function() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("spawn", &[Value::Int(42)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("expected Function"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_spawn_requires_argument() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("spawn", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("requires a task"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_await_with_result() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        // Spawn a task
        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        // Simulate completion
        AsyncHandler::set_future_result(&mut state, future_id, Value::String("done".into()));

        // Await should return the result
        let result = handler.handle("await", &[Value::Int(future_id)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::String(s)) => {
                assert_eq!(s, "done");
            }
            other => panic!("expected Resume(String), got {:?}", other),
        }
    }

    #[test]
    fn test_await_pending_suspends() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        // Spawn a task
        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        // Await without result should suspend
        let result = handler.handle("await", &[Value::Int(future_id)], cont(), &mut state);

        match result {
            HandlerResult::Suspend(id) => {
                assert!(id.0 > 0);
                // Check suspension data was stored
                let data = AsyncHandler::get_suspension_data(&state, id);
                assert!(data.is_some());
            }
            other => panic!("expected Suspend, got {:?}", other),
        }
    }

    #[test]
    fn test_await_not_found() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("await", &[Value::Int(99999)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("not found"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_await_cancelled() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        // Spawn and cancel
        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        handler.handle("cancel", &[Value::Int(future_id)], cont(), &mut state);

        // Await should fail
        let result = handler.handle("await", &[Value::Int(future_id)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("cancelled"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_yield_basic() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("yield", &[], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_sleep_basic() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("sleep", &[Value::Int(0)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Unit) => {}
            other => panic!("expected Resume(Unit), got {:?}", other),
        }
    }

    #[test]
    fn test_sleep_negative_fails() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("sleep", &[Value::Int(-100)], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("non-negative"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_join_all_complete() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        // Spawn two tasks
        let id1 = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };
        let id2 = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        // Complete both
        AsyncHandler::set_future_result(&mut state, id1, Value::Int(1));
        AsyncHandler::set_future_result(&mut state, id2, Value::Int(2));

        // Join
        let futures = Value::Array(Rc::new(RefCell::new(vec![Value::Int(id1), Value::Int(id2)])));
        let result = handler.handle("join", &[futures], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Array(results_rc)) => {
                let results = results_rc.borrow();
                assert_eq!(results.len(), 2);
                assert_eq!(results[0], Value::Int(1));
                assert_eq!(results[1], Value::Int(2));
            }
            other => panic!("expected Resume(Array), got {:?}", other),
        }
    }

    #[test]
    fn test_join_pending_suspends() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let id1 = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        // Don't complete the future
        let futures = Value::Array(Rc::new(RefCell::new(vec![Value::Int(id1)])));
        let result = handler.handle("join", &[futures], cont(), &mut state);

        match result {
            HandlerResult::Suspend(id) => {
                assert!(id.0 > 0);
            }
            other => panic!("expected Suspend, got {:?}", other),
        }
    }

    #[test]
    fn test_select_first_complete() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let id1 = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };
        let id2 = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        // Complete only the second
        AsyncHandler::set_future_result(&mut state, id2, Value::String("second".into()));

        let futures = Value::Array(Rc::new(RefCell::new(vec![Value::Int(id1), Value::Int(id2)])));
        let result = handler.handle("select", &[futures], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Tuple(t)) => {
                assert_eq!(t.len(), 2);
                assert_eq!(t[0], Value::Int(1)); // Index of id2
                assert_eq!(t[1], Value::String("second".into()));
            }
            other => panic!("expected Resume(Tuple), got {:?}", other),
        }
    }

    #[test]
    fn test_select_none_ready_suspends() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let id1 = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        let futures = Value::Array(Rc::new(RefCell::new(vec![Value::Int(id1)])));
        let result = handler.handle("select", &[futures], cont(), &mut state);

        match result {
            HandlerResult::Suspend(id) => {
                assert!(id.0 > 0);
            }
            other => panic!("expected Suspend, got {:?}", other),
        }
    }

    #[test]
    fn test_select_empty_fails() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let futures = Value::Array(Rc::new(RefCell::new(vec![])));
        let result = handler.handle("select", &[futures], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("at least one"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_cancel_existing() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        let result = handler.handle("cancel", &[Value::Int(future_id)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }

        assert!(AsyncHandler::is_cancelled(&state, future_id));
    }

    #[test]
    fn test_cancel_nonexistent() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("cancel", &[Value::Int(99999)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_is_ready_true() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        AsyncHandler::set_future_result(&mut state, future_id, Value::Int(42));

        let result = handler.handle("is_ready", &[Value::Int(future_id)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Bool(true)) => {}
            other => panic!("expected Resume(Bool(true)), got {:?}", other),
        }
    }

    #[test]
    fn test_is_ready_false() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        let result = handler.handle("is_ready", &[Value::Int(future_id)], cont(), &mut state);

        match result {
            HandlerResult::Resume(Value::Bool(false)) => {}
            other => panic!("expected Resume(Bool(false)), got {:?}", other),
        }
    }

    #[test]
    fn test_helper_methods() {
        let mut state = new_state();
        let future_id = 12345i64;

        // Initially no result
        assert!(AsyncHandler::get_future_result(&state, future_id).is_none());
        assert!(!AsyncHandler::is_cancelled(&state, future_id));

        // Set result
        AsyncHandler::set_future_result(&mut state, future_id, Value::String("result".into()));
        assert_eq!(
            AsyncHandler::get_future_result(&state, future_id),
            Some(Value::String("result".into()))
        );
    }

    #[test]
    fn test_unknown_operation() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let result = handler.handle("unknown_op", &[], cont(), &mut state);

        match result {
            HandlerResult::Abort(e) => {
                assert!(e.message.contains("unknown operation"));
            }
            other => panic!("expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn test_default_impl() {
        let handler = AsyncHandler::default();
        assert_eq!(handler.effect_name(), "Async");
    }

    #[test]
    fn test_unique_future_ids() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let mut ids = Vec::new();
        for _ in 0..10 {
            let result = handler.handle("spawn", &[dummy_function()], cont(), &mut state);
            if let HandlerResult::Resume(Value::Int(id)) = result {
                assert!(!ids.contains(&id), "future IDs should be unique");
                ids.push(id);
            }
        }
        assert_eq!(ids.len(), 10);
    }

    #[test]
    fn test_get_pending_task() {
        let handler = AsyncHandler::new();
        let mut state = new_state();

        let future_id = match handler.handle("spawn", &[dummy_function()], cont(), &mut state) {
            HandlerResult::Resume(Value::Int(id)) => id,
            _ => panic!("spawn failed"),
        };

        // Task should be pending
        let task = AsyncHandler::get_pending_task(&state, future_id);
        assert!(task.is_some());

        // After setting result, task should be removed
        AsyncHandler::set_future_result(&mut state, future_id, Value::Int(42));
        let task = AsyncHandler::get_pending_task(&state, future_id);
        assert!(task.is_none());
    }
}
