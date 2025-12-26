//! Async/Await Runtime for Sounio
//!
//! This module provides the async runtime infrastructure for Sounio programs.
//! It wraps tokio (when available) and provides a Sounio-compatible Future trait
//! and task management system.
//!
//! # Example (in Sounio)
//! ```d
//! async fn fetch_data(url: string) -> string with Async, IO {
//!     let response = http_get(url).await
//!     response.body
//! }
//!
//! fn main() with Async {
//!     let result = spawn { fetch_data("https://api.example.com") }
//!     println(result.await)
//! }
//! ```

use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future as StdFuture;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// Unique identifier for async tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(u64);

impl TaskId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        TaskId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }
}

/// The state of an async task
#[derive(Debug, Clone)]
pub enum TaskState {
    /// Task is ready to be polled
    Ready,
    /// Task is waiting for an event
    Pending,
    /// Task has completed with a value
    Completed(SounioValue),
    /// Task has failed with an error
    Failed(String),
    /// Task was cancelled
    Cancelled,
}

/// A Sounio-compatible value that can be stored in async tasks
/// This mirrors the interpreter's Value but is Send + Sync safe
#[derive(Debug, Clone)]
pub enum SounioValue {
    Unit,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    /// A future that can be awaited
    Future(TaskId),
    /// Array of values
    Array(Vec<SounioValue>),
    /// Tuple of values
    Tuple(Vec<SounioValue>),
    /// Struct with named fields
    Struct {
        name: String,
        fields: HashMap<String, SounioValue>,
    },
    /// Variant (enum)
    Variant {
        enum_name: String,
        variant_name: String,
        fields: Vec<SounioValue>,
    },
    /// None/null value
    None,
    /// Some value (Option::Some)
    Some(Box<SounioValue>),
    /// Ok value (Result::Ok)
    Ok(Box<SounioValue>),
    /// Err value (Result::Err)
    Err(Box<SounioValue>),
}

impl Default for SounioValue {
    fn default() -> Self {
        SounioValue::Unit
    }
}

/// A Sounio Future - represents a computation that will complete later
pub struct SounioFuture {
    /// The task ID for this future
    pub id: TaskId,
    /// The current state
    state: Arc<Mutex<FutureState>>,
}

/// Internal state of a Sounio future
struct FutureState {
    /// Current value/result
    value: Option<SounioValue>,
    /// Whether the future has completed
    completed: bool,
    /// Wakers to notify when the future completes
    wakers: Vec<Waker>,
}

impl SounioFuture {
    /// Create a new pending future
    pub fn new() -> Self {
        Self {
            id: TaskId::new(),
            state: Arc::new(Mutex::new(FutureState {
                value: None,
                completed: false,
                wakers: Vec::new(),
            })),
        }
    }

    /// Create a future that is already completed with a value
    pub fn ready(value: SounioValue) -> Self {
        Self {
            id: TaskId::new(),
            state: Arc::new(Mutex::new(FutureState {
                value: Some(value),
                completed: true,
                wakers: Vec::new(),
            })),
        }
    }

    /// Complete this future with a value
    pub fn complete(&self, value: SounioValue) {
        let mut state = self.state.lock().unwrap();
        state.value = Some(value);
        state.completed = true;
        // Wake all waiting tasks
        for waker in state.wakers.drain(..) {
            waker.wake();
        }
    }

    /// Get the task ID
    pub fn task_id(&self) -> TaskId {
        self.id
    }

    /// Check if the future is completed
    pub fn is_completed(&self) -> bool {
        self.state.lock().unwrap().completed
    }

    /// Try to get the value if completed
    pub fn try_get(&self) -> Option<SounioValue> {
        let state = self.state.lock().unwrap();
        if state.completed {
            state.value.clone()
        } else {
            None
        }
    }
}

impl Default for SounioFuture {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for SounioFuture {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            state: Arc::clone(&self.state),
        }
    }
}

impl StdFuture for SounioFuture {
    type Output = SounioValue;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap();
        if state.completed {
            Poll::Ready(state.value.clone().unwrap_or(SounioValue::Unit))
        } else {
            // Register the waker to be notified when the future completes
            state.wakers.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// A task handle that can be used to await a spawned task
#[derive(Clone)]
pub struct TaskHandle {
    /// The task ID
    pub id: TaskId,
    /// The future associated with this task
    future: SounioFuture,
}

impl TaskHandle {
    /// Wait for the task to complete and get the result
    pub fn try_get(&self) -> Option<SounioValue> {
        self.future.try_get()
    }

    /// Check if the task is completed
    pub fn is_completed(&self) -> bool {
        self.future.is_completed()
    }

    /// Get the task ID
    pub fn task_id(&self) -> TaskId {
        self.id
    }
}

/// The Sounio async runtime
///
/// This manages async tasks and provides the execution context for
/// async/await in Sounio programs.
pub struct SounioRuntime {
    /// All registered tasks
    tasks: Arc<Mutex<HashMap<TaskId, SounioFuture>>>,
    /// Task execution queue (tasks ready to be polled)
    ready_queue: Arc<Mutex<Vec<TaskId>>>,
    /// Whether the runtime is running
    running: Arc<Mutex<bool>>,
}

impl SounioRuntime {
    /// Create a new Sounio async runtime
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(Mutex::new(HashMap::new())),
            ready_queue: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
        }
    }

    /// Spawn a new async task
    ///
    /// Returns a TaskHandle that can be used to await the result.
    pub fn spawn<F>(&self, f: F) -> TaskHandle
    where
        F: FnOnce() -> SounioValue + 'static,
    {
        let future = SounioFuture::new();
        let id = future.id;

        // Store the task
        self.tasks.lock().unwrap().insert(id, future.clone());

        // Execute the function immediately in this simple implementation
        // In a full implementation, this would be scheduled on a thread pool
        let value = f();
        future.complete(value);

        TaskHandle { id, future }
    }

    /// Spawn an async task that returns a future
    pub fn spawn_future(&self, future: SounioFuture) -> TaskHandle {
        let id = future.id;
        self.tasks.lock().unwrap().insert(id, future.clone());
        self.ready_queue.lock().unwrap().push(id);

        TaskHandle { id, future }
    }

    /// Block on a future until it completes
    ///
    /// This is used to run async code from a synchronous context.
    pub fn block_on<F: FnOnce() -> SounioValue>(&self, f: F) -> SounioValue {
        // In a simple implementation, just execute the function
        f()
    }

    /// Block on a SounioFuture until it completes
    pub fn block_on_future(&self, future: &SounioFuture) -> SounioValue {
        // Busy-wait for completion (simple implementation)
        // In a production implementation, this would use proper async I/O
        loop {
            if let Some(value) = future.try_get() {
                return value;
            }
            // Yield to other tasks
            std::thread::yield_now();
        }
    }

    /// Try to get a task by ID
    pub fn get_task(&self, id: TaskId) -> Option<SounioFuture> {
        self.tasks.lock().unwrap().get(&id).cloned()
    }

    /// Remove a completed task
    pub fn remove_task(&self, id: TaskId) {
        self.tasks.lock().unwrap().remove(&id);
    }

    /// Get the number of active tasks
    pub fn task_count(&self) -> usize {
        self.tasks.lock().unwrap().len()
    }
}

impl Default for SounioRuntime {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// Global runtime instance for convenience
    static RUNTIME: RefCell<Option<SounioRuntime>> = RefCell::new(None);
}

/// Initialize the global runtime
pub fn init_runtime() {
    RUNTIME.with(|r| {
        *r.borrow_mut() = Some(SounioRuntime::new());
    });
}

/// Get the global runtime, initializing it if necessary
pub fn runtime() -> SounioRuntime {
    RUNTIME.with(|r| {
        let mut borrow = r.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(SounioRuntime::new());
        }
        // Clone the runtime (it uses Arc internally so this is cheap)
        SounioRuntime {
            tasks: Arc::clone(&borrow.as_ref().unwrap().tasks),
            ready_queue: Arc::clone(&borrow.as_ref().unwrap().ready_queue),
            running: Arc::clone(&borrow.as_ref().unwrap().running),
        }
    })
}

/// Spawn a task on the global runtime
pub fn spawn<F>(f: F) -> TaskHandle
where
    F: FnOnce() -> SounioValue + 'static,
{
    runtime().spawn(f)
}

/// Block on a function using the global runtime
pub fn block_on<F: FnOnce() -> SounioValue>(f: F) -> SounioValue {
    runtime().block_on(f)
}

/// Async state machine state for transformed async functions
///
/// When an async function is transformed, each await point becomes
/// a state transition. This enum represents those states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AsyncState {
    /// Initial state - function just started
    Start,
    /// Waiting at an await point (index indicates which one)
    Awaiting(u32),
    /// Function has completed
    Completed,
    /// Function has failed
    Failed,
}

/// State machine context for async functions
///
/// This holds the captured variables and current state of an
/// async function execution.
pub struct AsyncContext {
    /// Current state of the state machine
    pub state: AsyncState,
    /// Local variables captured from the async function
    pub locals: HashMap<String, SounioValue>,
    /// The final result when completed
    pub result: Option<SounioValue>,
    /// Error message if failed
    pub error: Option<String>,
}

impl AsyncContext {
    /// Create a new async context
    pub fn new() -> Self {
        Self {
            state: AsyncState::Start,
            locals: HashMap::new(),
            result: None,
            error: None,
        }
    }

    /// Set a local variable
    pub fn set_local(&mut self, name: &str, value: SounioValue) {
        self.locals.insert(name.to_string(), value);
    }

    /// Get a local variable
    pub fn get_local(&self, name: &str) -> Option<&SounioValue> {
        self.locals.get(name)
    }

    /// Transition to the next await point
    pub fn await_at(&mut self, point: u32) {
        self.state = AsyncState::Awaiting(point);
    }

    /// Complete the async function with a result
    pub fn complete(&mut self, value: SounioValue) {
        self.state = AsyncState::Completed;
        self.result = Some(value);
    }

    /// Fail the async function with an error
    pub fn fail(&mut self, error: String) {
        self.state = AsyncState::Failed;
        self.error = Some(error);
    }

    /// Check if the context is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self.state, AsyncState::Completed | AsyncState::Failed)
    }
}

impl Default for AsyncContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sounio_future_ready() {
        let future = SounioFuture::ready(SounioValue::Int(42));
        assert!(future.is_completed());
        assert!(matches!(future.try_get(), Some(SounioValue::Int(42))));
    }

    #[test]
    fn test_sounio_future_complete() {
        let future = SounioFuture::new();
        assert!(!future.is_completed());
        assert!(future.try_get().is_none());

        future.complete(SounioValue::String("done".to_string()));
        assert!(future.is_completed());
        assert!(matches!(
            future.try_get(),
            Some(SounioValue::String(s)) if s == "done"
        ));
    }

    #[test]
    fn test_runtime_spawn() {
        let rt = SounioRuntime::new();
        let handle = rt.spawn(|| SounioValue::Int(100));

        assert!(handle.is_completed());
        assert!(matches!(handle.try_get(), Some(SounioValue::Int(100))));
    }

    #[test]
    fn test_runtime_block_on() {
        let rt = SounioRuntime::new();
        let result = rt.block_on(|| SounioValue::Float(3.14));

        assert!(matches!(result, SounioValue::Float(f) if (f - 3.14).abs() < 0.001));
    }

    #[test]
    fn test_async_context() {
        let mut ctx = AsyncContext::new();
        assert_eq!(ctx.state, AsyncState::Start);

        ctx.set_local("x", SounioValue::Int(10));
        assert!(matches!(ctx.get_local("x"), Some(SounioValue::Int(10))));

        ctx.await_at(1);
        assert_eq!(ctx.state, AsyncState::Awaiting(1));

        ctx.complete(SounioValue::Unit);
        assert!(ctx.is_terminal());
    }

    #[test]
    fn test_global_runtime() {
        init_runtime();
        let handle = spawn(|| SounioValue::Bool(true));
        assert!(handle.is_completed());

        let result = block_on(|| SounioValue::String("test".to_string()));
        assert!(matches!(result, SounioValue::String(s) if s == "test"));
    }
}
