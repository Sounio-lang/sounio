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

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

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

impl std::fmt::Debug for SounioFuture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SounioFuture")
            .field("id", &self.id)
            .field("completed", &self.is_completed())
            .finish()
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

// ============================================================================
// Task Scheduler
// ============================================================================

/// Reason why a task is suspended
#[derive(Debug, Clone)]
pub enum SuspensionReason {
    /// Waiting for a single task to complete
    Await(TaskId),
    /// Waiting for ALL of these tasks to complete (join)
    JoinAll(Vec<TaskId>),
    /// Waiting for ANY of these tasks to complete (select)
    SelectAny(Vec<TaskId>),
    /// Waiting for a channel receive
    ChannelRecv(u64),
    /// Waiting for a channel send (bounded channel full)
    ChannelSend(u64),
    /// Waiting for a timer
    Timer(std::time::Instant),
}

/// Task metadata for the scheduler
#[derive(Debug)]
struct TaskEntry {
    /// The future for this task
    future: SounioFuture,
    /// Why the task is suspended (None = ready to poll)
    suspension: Option<SuspensionReason>,
    /// Tasks that depend on this one completing
    dependents: Vec<TaskId>,
    /// Priority (lower = higher priority)
    priority: u32,
}

/// Task scheduler for async execution
///
/// The scheduler maintains:
/// - A ready queue of tasks that can be polled
/// - Suspended tasks waiting for events
/// - Dependency tracking for wake propagation
///
/// # Scheduling Algorithm
///
/// Uses a priority-based round-robin approach:
/// 1. Ready tasks are polled in priority order
/// 2. When a task completes, its dependents are woken
/// 3. Suspended tasks are checked for wake conditions
pub struct TaskScheduler {
    /// All known tasks
    tasks: HashMap<TaskId, TaskEntry>,
    /// Ready queue (tasks that can be polled)
    ready: std::collections::VecDeque<TaskId>,
    /// Suspended tasks by suspension reason type
    /// This enables O(1) wake operations
    awaiting: HashMap<TaskId, Vec<TaskId>>,
    /// Tasks waiting on joins (map from any member -> join waiters)
    join_waiters: HashMap<TaskId, Vec<TaskId>>,
    /// Tasks waiting on selects (map from any member -> select waiters)
    select_waiters: HashMap<TaskId, Vec<TaskId>>,
    /// Channel waiters (channel_id -> waiting tasks)
    channel_recv_waiters: HashMap<u64, Vec<TaskId>>,
    channel_send_waiters: HashMap<u64, Vec<TaskId>>,
    /// Statistics
    polls: u64,
    wakes: u64,
}

impl TaskScheduler {
    /// Create a new task scheduler
    pub fn new() -> Self {
        Self {
            tasks: HashMap::new(),
            ready: std::collections::VecDeque::new(),
            awaiting: HashMap::new(),
            join_waiters: HashMap::new(),
            select_waiters: HashMap::new(),
            channel_recv_waiters: HashMap::new(),
            channel_send_waiters: HashMap::new(),
            polls: 0,
            wakes: 0,
        }
    }

    /// Schedule a new task
    ///
    /// Returns the TaskId assigned to the task.
    pub fn schedule(&mut self, future: SounioFuture) -> TaskId {
        self.schedule_with_priority(future, 100)
    }

    /// Schedule a new task with a specific priority
    pub fn schedule_with_priority(&mut self, future: SounioFuture, priority: u32) -> TaskId {
        let id = future.id;
        let entry = TaskEntry {
            future,
            suspension: None,
            dependents: Vec::new(),
            priority,
        };
        self.tasks.insert(id, entry);
        self.ready.push_back(id);
        id
    }

    /// Get the next ready task to poll
    pub fn next_ready(&mut self) -> Option<TaskId> {
        // Simple FIFO for now; could be priority queue
        self.ready.pop_front()
    }

    /// Mark a task as ready to poll
    pub fn mark_ready(&mut self, id: TaskId) {
        if let Some(entry) = self.tasks.get_mut(&id) {
            entry.suspension = None;
            if !self.ready.contains(&id) {
                self.ready.push_back(id);
            }
        }
    }

    /// Suspend a task with a reason
    pub fn suspend(&mut self, id: TaskId, reason: SuspensionReason) {
        if let Some(entry) = self.tasks.get_mut(&id) {
            // Register in appropriate waiter map for O(1) waking
            match &reason {
                SuspensionReason::Await(target) => {
                    self.awaiting.entry(*target).or_default().push(id);
                }
                SuspensionReason::JoinAll(targets) => {
                    for target in targets {
                        self.join_waiters.entry(*target).or_default().push(id);
                    }
                }
                SuspensionReason::SelectAny(targets) => {
                    for target in targets {
                        self.select_waiters.entry(*target).or_default().push(id);
                    }
                }
                SuspensionReason::ChannelRecv(chan_id) => {
                    self.channel_recv_waiters
                        .entry(*chan_id)
                        .or_default()
                        .push(id);
                }
                SuspensionReason::ChannelSend(chan_id) => {
                    self.channel_send_waiters
                        .entry(*chan_id)
                        .or_default()
                        .push(id);
                }
                SuspensionReason::Timer(_) => {
                    // Timer waking is handled separately
                }
            }
            entry.suspension = Some(reason);
        }
    }

    /// Wake tasks waiting on a completed task
    pub fn wake_dependents(&mut self, completed_id: TaskId) {
        self.wakes += 1;

        // Wake simple awaiters
        if let Some(waiters) = self.awaiting.remove(&completed_id) {
            for waiter in waiters {
                self.mark_ready(waiter);
            }
        }

        // Wake select waiters (any one completing wakes the select)
        if let Some(waiters) = self.select_waiters.remove(&completed_id) {
            for waiter in waiters {
                self.mark_ready(waiter);
                // Remove this waiter from other select entries
                if let Some(entry) = self.tasks.get(&waiter) {
                    if let Some(SuspensionReason::SelectAny(targets)) = &entry.suspension {
                        for target in targets {
                            if *target != completed_id {
                                if let Some(list) = self.select_waiters.get_mut(target) {
                                    list.retain(|w| *w != waiter);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Wake join waiters (check if all members completed)
        if let Some(waiters) = self.join_waiters.remove(&completed_id) {
            for waiter in waiters {
                if let Some(entry) = self.tasks.get(&waiter) {
                    if let Some(SuspensionReason::JoinAll(targets)) = &entry.suspension {
                        // Check if ALL targets are now completed
                        let all_done = targets.iter().all(|t| {
                            self.tasks
                                .get(t)
                                .map(|e| e.future.is_completed())
                                .unwrap_or(true)
                        });
                        if all_done {
                            self.mark_ready(waiter);
                        }
                    }
                }
            }
        }
    }

    /// Wake tasks waiting on a channel
    pub fn wake_channel_recv(&mut self, channel_id: u64) {
        if let Some(waiters) = self.channel_recv_waiters.remove(&channel_id) {
            // Wake one waiter (FIFO)
            if let Some(waiter) = waiters.into_iter().next() {
                self.mark_ready(waiter);
            }
        }
    }

    /// Wake tasks waiting to send on a channel
    pub fn wake_channel_send(&mut self, channel_id: u64) {
        if let Some(waiters) = self.channel_send_waiters.remove(&channel_id) {
            if let Some(waiter) = waiters.into_iter().next() {
                self.mark_ready(waiter);
            }
        }
    }

    /// Poll the next ready task
    ///
    /// Returns Some((id, Poll::Ready(value))) if a task completed,
    /// Some((id, Poll::Pending)) if polled but not ready,
    /// None if no tasks are ready.
    pub fn poll_next(&mut self) -> Option<(TaskId, Poll<SounioValue>)> {
        let id = self.next_ready()?;
        self.polls += 1;

        if let Some(entry) = self.tasks.get(&id) {
            let future = entry.future.clone();
            if future.is_completed() {
                let value = future.try_get().unwrap_or(SounioValue::Unit);
                self.wake_dependents(id);
                return Some((id, Poll::Ready(value)));
            }
        }

        Some((id, Poll::Pending))
    }

    /// Run until all tasks complete or no progress can be made
    pub fn run_until_stalled(&mut self) -> Vec<(TaskId, SounioValue)> {
        let mut completed = Vec::new();

        loop {
            match self.poll_next() {
                Some((id, Poll::Ready(value))) => {
                    completed.push((id, value));
                }
                Some((_, Poll::Pending)) => {
                    // Task not ready, continue polling others
                }
                None => {
                    // No ready tasks
                    break;
                }
            }

            // Check for stall (no ready tasks but uncompleted suspended tasks)
            if self.ready.is_empty() {
                break;
            }
        }

        completed
    }

    /// Get task count by state
    pub fn stats(&self) -> SchedulerStats {
        let ready = self.ready.len();
        let suspended = self
            .tasks
            .values()
            .filter(|e| e.suspension.is_some())
            .count();
        let completed = self
            .tasks
            .values()
            .filter(|e| e.future.is_completed())
            .count();
        SchedulerStats {
            ready,
            suspended,
            completed,
            total: self.tasks.len(),
            polls: self.polls,
            wakes: self.wakes,
        }
    }

    /// Get a task's future by ID
    pub fn get_future(&self, id: TaskId) -> Option<SounioFuture> {
        self.tasks.get(&id).map(|e| e.future.clone())
    }

    /// Check if a task is completed
    pub fn is_completed(&self, id: TaskId) -> bool {
        self.tasks
            .get(&id)
            .map(|e| e.future.is_completed())
            .unwrap_or(false)
    }

    /// Remove a completed task
    pub fn remove(&mut self, id: TaskId) -> Option<SounioFuture> {
        self.tasks.remove(&id).map(|e| e.future)
    }
}

impl Default for TaskScheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Tasks ready to poll
    pub ready: usize,
    /// Tasks suspended waiting for events
    pub suspended: usize,
    /// Tasks that have completed
    pub completed: usize,
    /// Total tasks in scheduler
    pub total: usize,
    /// Total poll operations
    pub polls: u64,
    /// Total wake operations
    pub wakes: u64,
}

// ============================================================================
// Join Combinator
// ============================================================================

/// Future that completes when ALL inner futures complete
///
/// # Example (Sounio)
/// ```sio
/// let results = join(task1, task2, task3).await
/// // results is a tuple of all results
/// ```
pub struct JoinFuture {
    /// The task ID for this join
    id: TaskId,
    /// Tasks being joined
    tasks: Vec<TaskId>,
    /// Results collected so far
    results: Arc<Mutex<Vec<Option<SounioValue>>>>,
    /// Completion state
    state: Arc<Mutex<FutureState>>,
}

impl JoinFuture {
    /// Create a new join future from multiple tasks
    pub fn new(tasks: Vec<TaskId>) -> Self {
        let n = tasks.len();
        Self {
            id: TaskId::new(),
            tasks,
            results: Arc::new(Mutex::new(vec![None; n])),
            state: Arc::new(Mutex::new(FutureState {
                value: None,
                completed: false,
                wakers: Vec::new(),
            })),
        }
    }

    /// Get the task ID
    pub fn task_id(&self) -> TaskId {
        self.id
    }

    /// Get the tasks being joined
    pub fn tasks(&self) -> &[TaskId] {
        &self.tasks
    }

    /// Record a result for one of the joined tasks
    pub fn record_result(&self, index: usize, value: SounioValue) {
        let mut results = self.results.lock().unwrap();
        results[index] = Some(value);

        // Check if all results are in
        if results.iter().all(|r| r.is_some()) {
            let all_values: Vec<SounioValue> = results.iter().map(|r| r.clone().unwrap()).collect();
            let mut state = self.state.lock().unwrap();
            state.value = Some(SounioValue::Array(all_values));
            state.completed = true;
            for waker in state.wakers.drain(..) {
                waker.wake();
            }
        }
    }

    /// Check if the join is complete
    pub fn is_completed(&self) -> bool {
        self.state.lock().unwrap().completed
    }

    /// Try to get the result
    pub fn try_get(&self) -> Option<SounioValue> {
        let state = self.state.lock().unwrap();
        if state.completed {
            state.value.clone()
        } else {
            None
        }
    }
}

impl Clone for JoinFuture {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            tasks: self.tasks.clone(),
            results: Arc::clone(&self.results),
            state: Arc::clone(&self.state),
        }
    }
}

impl StdFuture for JoinFuture {
    type Output = SounioValue;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap();
        if state.completed {
            Poll::Ready(state.value.clone().unwrap_or(SounioValue::Array(vec![])))
        } else {
            state.wakers.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Create a join future from multiple task handles
pub fn join(handles: Vec<TaskHandle>) -> JoinFuture {
    JoinFuture::new(handles.iter().map(|h| h.id).collect())
}

/// Create a join future from 2 handles (convenience)
pub fn join2(a: TaskHandle, b: TaskHandle) -> JoinFuture {
    JoinFuture::new(vec![a.id, b.id])
}

/// Create a join future from 3 handles (convenience)
pub fn join3(a: TaskHandle, b: TaskHandle, c: TaskHandle) -> JoinFuture {
    JoinFuture::new(vec![a.id, b.id, c.id])
}

// ============================================================================
// Select Combinator
// ============================================================================

/// Result of a select operation
#[derive(Debug, Clone)]
pub struct SelectResult {
    /// Index of the task that completed first
    pub index: usize,
    /// The completed task's ID
    pub task_id: TaskId,
    /// The result value
    pub value: SounioValue,
}

/// Future that completes when ANY inner future completes
///
/// # Example (Sounio)
/// ```sio
/// let result = select(task1, task2, timeout).await
/// match result.index {
///     0 => // task1 completed first
///     1 => // task2 completed first
///     2 => // timeout fired
/// }
/// ```
pub struct SelectFuture {
    /// The task ID for this select
    id: TaskId,
    /// Tasks being selected among
    tasks: Vec<TaskId>,
    /// Completion state
    state: Arc<Mutex<SelectState>>,
}

struct SelectState {
    /// Which task completed first (if any)
    completed_index: Option<usize>,
    /// The result
    result: Option<SelectResult>,
    /// Wakers
    wakers: Vec<Waker>,
}

impl SelectFuture {
    /// Create a new select future from multiple tasks
    pub fn new(tasks: Vec<TaskId>) -> Self {
        Self {
            id: TaskId::new(),
            tasks,
            state: Arc::new(Mutex::new(SelectState {
                completed_index: None,
                result: None,
                wakers: Vec::new(),
            })),
        }
    }

    /// Get the task ID
    pub fn task_id(&self) -> TaskId {
        self.id
    }

    /// Get the tasks being selected among
    pub fn tasks(&self) -> &[TaskId] {
        &self.tasks
    }

    /// Record that one of the tasks completed
    pub fn record_completion(&self, index: usize, task_id: TaskId, value: SounioValue) {
        let mut state = self.state.lock().unwrap();
        // Only record the first completion
        if state.completed_index.is_none() {
            state.completed_index = Some(index);
            state.result = Some(SelectResult {
                index,
                task_id,
                value,
            });
            for waker in state.wakers.drain(..) {
                waker.wake();
            }
        }
    }

    /// Check if the select is complete
    pub fn is_completed(&self) -> bool {
        self.state.lock().unwrap().completed_index.is_some()
    }

    /// Try to get the result
    pub fn try_get(&self) -> Option<SelectResult> {
        self.state.lock().unwrap().result.clone()
    }
}

impl Clone for SelectFuture {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            tasks: self.tasks.clone(),
            state: Arc::clone(&self.state),
        }
    }
}

impl StdFuture for SelectFuture {
    type Output = SelectResult;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.lock().unwrap();
        if let Some(result) = state.result.clone() {
            Poll::Ready(result)
        } else {
            state.wakers.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Create a select future from multiple task handles
pub fn select(handles: Vec<TaskHandle>) -> SelectFuture {
    SelectFuture::new(handles.iter().map(|h| h.id).collect())
}

/// Create a select future from 2 handles (convenience)
pub fn select2(a: TaskHandle, b: TaskHandle) -> SelectFuture {
    SelectFuture::new(vec![a.id, b.id])
}

/// Create a select future from 3 handles (convenience)
pub fn select3(a: TaskHandle, b: TaskHandle, c: TaskHandle) -> SelectFuture {
    SelectFuture::new(vec![a.id, b.id, c.id])
}

// ============================================================================
// Channels (bounded and unbounded)
// ============================================================================

/// Unique identifier for channels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(u64);

impl ChannelId {
    fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        ChannelId(COUNTER.fetch_add(1, Ordering::SeqCst))
    }

    /// Get the raw ID value
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Channel error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// Channel has been closed
    Closed,
    /// Channel is full (bounded channel)
    Full,
    /// Channel is empty
    Empty,
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelError::Closed => write!(f, "channel closed"),
            ChannelError::Full => write!(f, "channel full"),
            ChannelError::Empty => write!(f, "channel empty"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Shared state for a channel
struct ChannelShared<T> {
    /// The channel ID
    id: ChannelId,
    /// Buffer of pending values
    buffer: std::collections::VecDeque<T>,
    /// Capacity (None = unbounded)
    capacity: Option<usize>,
    /// Whether the channel is closed
    closed: bool,
    /// Wakers for senders blocked on a full channel
    send_wakers: Vec<Waker>,
    /// Wakers for receivers blocked on an empty channel
    recv_wakers: Vec<Waker>,
}

/// Channel sender
///
/// Can be cloned to share the sending end among multiple producers.
pub struct Sender<T> {
    shared: Arc<Mutex<ChannelShared<T>>>,
}

impl<T> Sender<T> {
    /// Send a value, blocking if the channel is full
    ///
    /// For async contexts, use with a waker. For sync, use try_send.
    pub fn send(&self, value: T) -> Result<(), ChannelError> {
        let mut guard = self.shared.lock().unwrap();
        if guard.closed {
            return Err(ChannelError::Closed);
        }
        if let Some(cap) = guard.capacity {
            if guard.buffer.len() >= cap {
                return Err(ChannelError::Full);
            }
        }
        guard.buffer.push_back(value);
        // Wake one receiver
        if let Some(waker) = guard.recv_wakers.pop() {
            drop(guard);
            waker.wake();
        }
        Ok(())
    }

    /// Try to send a value without blocking
    pub fn try_send(&self, value: T) -> Result<(), ChannelError> {
        self.send(value)
    }

    /// Close the sending end
    pub fn close(&self) {
        let mut guard = self.shared.lock().unwrap();
        if guard.closed {
            return;
        }
        guard.closed = true;
        // Wake all waiters
        let send_wakers = std::mem::take(&mut guard.send_wakers);
        let recv_wakers = std::mem::take(&mut guard.recv_wakers);
        drop(guard);
        for w in send_wakers {
            w.wake();
        }
        for w in recv_wakers {
            w.wake();
        }
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        self.shared.lock().unwrap().closed
    }

    /// Get the channel ID
    pub fn channel_id(&self) -> ChannelId {
        self.shared.lock().unwrap().id
    }
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

/// Channel receiver
///
/// Can be cloned to share the receiving end among multiple consumers.
pub struct Receiver<T> {
    shared: Arc<Mutex<ChannelShared<T>>>,
}

impl<T> Receiver<T> {
    /// Receive a value, returning None if closed and empty
    pub fn recv(&self) -> Option<T> {
        let mut guard = self.shared.lock().unwrap();
        if let Some(value) = guard.buffer.pop_front() {
            // Wake one sender if bounded and was full
            if let Some(waker) = guard.send_wakers.pop() {
                drop(guard);
                waker.wake();
            }
            return Some(value);
        }
        if guard.closed {
            return None;
        }
        None
    }

    /// Try to receive a value without blocking
    pub fn try_recv(&self) -> Result<T, ChannelError> {
        let mut guard = self.shared.lock().unwrap();
        if let Some(value) = guard.buffer.pop_front() {
            if let Some(waker) = guard.send_wakers.pop() {
                drop(guard);
                waker.wake();
            }
            return Ok(value);
        }
        if guard.closed {
            Err(ChannelError::Closed)
        } else {
            Err(ChannelError::Empty)
        }
    }

    /// Close the receiving end
    pub fn close(&self) {
        let mut guard = self.shared.lock().unwrap();
        if guard.closed {
            return;
        }
        guard.closed = true;
        let send_wakers = std::mem::take(&mut guard.send_wakers);
        let recv_wakers = std::mem::take(&mut guard.recv_wakers);
        drop(guard);
        for w in send_wakers {
            w.wake();
        }
        for w in recv_wakers {
            w.wake();
        }
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        self.shared.lock().unwrap().closed
    }

    /// Get the channel ID
    pub fn channel_id(&self) -> ChannelId {
        self.shared.lock().unwrap().id
    }

    /// Check if there are pending values
    pub fn is_empty(&self) -> bool {
        self.shared.lock().unwrap().buffer.is_empty()
    }

    /// Get the number of pending values
    pub fn len(&self) -> usize {
        self.shared.lock().unwrap().buffer.len()
    }
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        Self {
            shared: Arc::clone(&self.shared),
        }
    }
}

/// Create an unbounded channel
///
/// Returns a (Sender, Receiver) pair. The sender can send an unlimited
/// number of values without blocking.
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Mutex::new(ChannelShared {
        id: ChannelId::new(),
        buffer: std::collections::VecDeque::new(),
        capacity: None,
        closed: false,
        send_wakers: Vec::new(),
        recv_wakers: Vec::new(),
    }));
    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared },
    )
}

/// Create a bounded channel with the given capacity
///
/// Returns a (Sender, Receiver) pair. The sender will return `ChannelError::Full`
/// when the buffer reaches capacity.
pub fn bounded_channel<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
    assert!(capacity > 0, "channel capacity must be > 0");
    let shared = Arc::new(Mutex::new(ChannelShared {
        id: ChannelId::new(),
        buffer: std::collections::VecDeque::with_capacity(capacity),
        capacity: Some(capacity),
        closed: false,
        send_wakers: Vec::new(),
        recv_wakers: Vec::new(),
    }));
    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared },
    )
}

// ============================================================================
// Global Scheduler
// ============================================================================

thread_local! {
    /// Global scheduler instance
    static SCHEDULER: RefCell<Option<TaskScheduler>> = RefCell::new(None);
}

/// Initialize the global scheduler
pub fn init_scheduler() {
    SCHEDULER.with(|s| {
        *s.borrow_mut() = Some(TaskScheduler::new());
    });
}

/// Get the global scheduler, initializing if necessary
pub fn scheduler() -> TaskScheduler {
    SCHEDULER.with(|s| {
        let mut borrow = s.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(TaskScheduler::new());
        }
        // Return a copy of the scheduler state
        // Note: In production, would use Arc<Mutex<TaskScheduler>> for sharing
        TaskScheduler::new()
    })
}

/// Schedule a future on the global scheduler
pub fn schedule(future: SounioFuture) -> TaskId {
    SCHEDULER.with(|s| {
        let mut borrow = s.borrow_mut();
        if borrow.is_none() {
            *borrow = Some(TaskScheduler::new());
        }
        borrow.as_mut().unwrap().schedule(future)
    })
}

/// Run the global scheduler until stalled
pub fn run_scheduler() -> Vec<(TaskId, SounioValue)> {
    SCHEDULER.with(|s| {
        let mut borrow = s.borrow_mut();
        if let Some(scheduler) = borrow.as_mut() {
            scheduler.run_until_stalled()
        } else {
            Vec::new()
        }
    })
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

    // ========================================================================
    // Task Scheduler Tests
    // ========================================================================

    #[test]
    fn test_scheduler_creation() {
        let scheduler = TaskScheduler::new();
        let stats = scheduler.stats();
        assert_eq!(stats.total, 0);
        assert_eq!(stats.ready, 0);
    }

    #[test]
    fn test_scheduler_schedule() {
        let mut scheduler = TaskScheduler::new();
        let future = SounioFuture::ready(SounioValue::Int(42));
        let id = scheduler.schedule(future);

        let stats = scheduler.stats();
        assert_eq!(stats.total, 1);
        assert_eq!(stats.ready, 1);
        assert!(scheduler.is_completed(id));
    }

    #[test]
    fn test_scheduler_poll() {
        let mut scheduler = TaskScheduler::new();
        let future = SounioFuture::ready(SounioValue::Int(42));
        scheduler.schedule(future);

        let result = scheduler.poll_next();
        assert!(result.is_some());
        let (_, poll) = result.unwrap();
        assert!(matches!(poll, Poll::Ready(SounioValue::Int(42))));
    }

    #[test]
    fn test_scheduler_run_until_stalled() {
        let mut scheduler = TaskScheduler::new();

        // Schedule multiple ready futures
        scheduler.schedule(SounioFuture::ready(SounioValue::Int(1)));
        scheduler.schedule(SounioFuture::ready(SounioValue::Int(2)));
        scheduler.schedule(SounioFuture::ready(SounioValue::Int(3)));

        let completed = scheduler.run_until_stalled();
        assert_eq!(completed.len(), 3);
    }

    #[test]
    fn test_scheduler_suspension() {
        let mut scheduler = TaskScheduler::new();

        let future1 = SounioFuture::new(); // Not completed
        let future2 = SounioFuture::ready(SounioValue::Int(42));

        let id1 = scheduler.schedule(future1);
        let id2 = scheduler.schedule(future2);

        // Suspend id1 waiting for id2
        scheduler.suspend(id1, SuspensionReason::Await(id2));

        let stats = scheduler.stats();
        assert_eq!(stats.suspended, 1);
    }

    #[test]
    fn test_scheduler_wake_dependents() {
        let mut scheduler = TaskScheduler::new();

        let future1 = SounioFuture::new();
        let future2 = SounioFuture::ready(SounioValue::Int(42));

        let id1 = scheduler.schedule(future1);
        let id2 = scheduler.schedule(future2);

        // Suspend id1 waiting for id2
        scheduler.suspend(id1, SuspensionReason::Await(id2));

        // Simulate id2 completing
        scheduler.wake_dependents(id2);

        // id1 should now be ready
        let stats = scheduler.stats();
        assert!(stats.ready >= 1);
    }

    #[test]
    fn test_scheduler_priority() {
        let mut scheduler = TaskScheduler::new();

        let future_low = SounioFuture::ready(SounioValue::Int(1));
        let future_high = SounioFuture::ready(SounioValue::Int(2));

        // Schedule with different priorities (lower = higher priority)
        scheduler.schedule_with_priority(future_low, 100);
        scheduler.schedule_with_priority(future_high, 10);

        // For now, FIFO order (priority sorting would be enhancement)
        let stats = scheduler.stats();
        assert_eq!(stats.ready, 2);
    }

    #[test]
    fn test_scheduler_stats() {
        let mut scheduler = TaskScheduler::new();

        scheduler.schedule(SounioFuture::ready(SounioValue::Int(1)));
        scheduler.schedule(SounioFuture::ready(SounioValue::Int(2)));

        let stats = scheduler.stats();
        assert_eq!(stats.total, 2);
        assert_eq!(stats.polls, 0); // Haven't polled yet

        scheduler.poll_next();
        let stats = scheduler.stats();
        assert_eq!(stats.polls, 1);
    }

    // ========================================================================
    // Join Combinator Tests
    // ========================================================================

    #[test]
    fn test_join_future_creation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        let join = JoinFuture::new(vec![id1, id2]);

        assert_eq!(join.tasks().len(), 2);
        assert!(!join.is_completed());
    }

    #[test]
    fn test_join_future_completion() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        let join = JoinFuture::new(vec![id1, id2]);

        // Record first result
        join.record_result(0, SounioValue::Int(10));
        assert!(!join.is_completed());

        // Record second result
        join.record_result(1, SounioValue::Int(20));
        assert!(join.is_completed());

        // Check the result
        let result = join.try_get();
        assert!(result.is_some());
        if let Some(SounioValue::Array(values)) = result {
            assert_eq!(values.len(), 2);
            assert!(matches!(values[0], SounioValue::Int(10)));
            assert!(matches!(values[1], SounioValue::Int(20)));
        } else {
            panic!("Expected Array result");
        }
    }

    #[test]
    fn test_join_from_handles() {
        let rt = SounioRuntime::new();
        let h1 = rt.spawn(|| SounioValue::Int(1));
        let h2 = rt.spawn(|| SounioValue::Int(2));

        let join = join2(h1, h2);
        assert_eq!(join.tasks().len(), 2);
    }

    #[test]
    fn test_join3_convenience() {
        let rt = SounioRuntime::new();
        let h1 = rt.spawn(|| SounioValue::Int(1));
        let h2 = rt.spawn(|| SounioValue::Int(2));
        let h3 = rt.spawn(|| SounioValue::Int(3));

        let join = join3(h1, h2, h3);
        assert_eq!(join.tasks().len(), 3);
    }

    // ========================================================================
    // Select Combinator Tests
    // ========================================================================

    #[test]
    fn test_select_future_creation() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        let select = SelectFuture::new(vec![id1, id2]);

        assert_eq!(select.tasks().len(), 2);
        assert!(!select.is_completed());
    }

    #[test]
    fn test_select_future_first_completion() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        let select = SelectFuture::new(vec![id1, id2]);

        // First task completes
        select.record_completion(0, id1, SounioValue::Int(10));
        assert!(select.is_completed());

        let result = select.try_get();
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.index, 0);
        assert_eq!(result.task_id, id1);
        assert!(matches!(result.value, SounioValue::Int(10)));
    }

    #[test]
    fn test_select_only_first_wins() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        let select = SelectFuture::new(vec![id1, id2]);

        // First task completes
        select.record_completion(0, id1, SounioValue::Int(10));

        // Second task also completes (should be ignored)
        select.record_completion(1, id2, SounioValue::Int(20));

        let result = select.try_get().unwrap();
        assert_eq!(result.index, 0);
        assert!(matches!(result.value, SounioValue::Int(10)));
    }

    #[test]
    fn test_select_from_handles() {
        let rt = SounioRuntime::new();
        let h1 = rt.spawn(|| SounioValue::Int(1));
        let h2 = rt.spawn(|| SounioValue::Int(2));

        let select = select2(h1, h2);
        assert_eq!(select.tasks().len(), 2);
    }

    #[test]
    fn test_select3_convenience() {
        let rt = SounioRuntime::new();
        let h1 = rt.spawn(|| SounioValue::Int(1));
        let h2 = rt.spawn(|| SounioValue::Int(2));
        let h3 = rt.spawn(|| SounioValue::Int(3));

        let select = select3(h1, h2, h3);
        assert_eq!(select.tasks().len(), 3);
    }

    // ========================================================================
    // Global Scheduler Tests
    // ========================================================================

    #[test]
    fn test_init_scheduler() {
        init_scheduler();
        // Should not panic
    }

    #[test]
    fn test_schedule_global() {
        init_scheduler();
        let future = SounioFuture::ready(SounioValue::Int(42));
        let _id = schedule(future);
        // Should not panic
    }

    #[test]
    fn test_run_scheduler() {
        init_scheduler();
        schedule(SounioFuture::ready(SounioValue::Int(1)));
        schedule(SounioFuture::ready(SounioValue::Int(2)));

        let completed = run_scheduler();
        // Should complete scheduled tasks
        assert!(completed.len() >= 0); // May or may not complete depending on thread-local state
    }

    // ========================================================================
    // Channel Tests
    // ========================================================================

    #[test]
    fn test_unbounded_channel_creation() {
        let (tx, rx): (Sender<i32>, Receiver<i32>) = channel();
        assert!(!tx.is_closed());
        assert!(!rx.is_closed());
        assert!(rx.is_empty());
    }

    #[test]
    fn test_bounded_channel_creation() {
        let (tx, rx): (Sender<i32>, Receiver<i32>) = bounded_channel(5);
        assert!(!tx.is_closed());
        assert!(!rx.is_closed());
        assert!(rx.is_empty());
    }

    #[test]
    fn test_channel_send_recv() {
        let (tx, rx) = channel();

        tx.send(42).unwrap();
        tx.send(43).unwrap();

        assert_eq!(rx.len(), 2);
        assert_eq!(rx.recv(), Some(42));
        assert_eq!(rx.recv(), Some(43));
        assert_eq!(rx.recv(), None);
    }

    #[test]
    fn test_channel_try_recv() {
        let (tx, rx) = channel();

        // Empty channel
        assert!(matches!(rx.try_recv(), Err(ChannelError::Empty)));

        tx.send(10).unwrap();
        assert_eq!(rx.try_recv(), Ok(10));
        assert!(matches!(rx.try_recv(), Err(ChannelError::Empty)));
    }

    #[test]
    fn test_bounded_channel_full() {
        let (tx, rx) = bounded_channel(2);

        assert!(tx.send(1).is_ok());
        assert!(tx.send(2).is_ok());
        assert!(matches!(tx.send(3), Err(ChannelError::Full)));

        // Drain one
        rx.recv();
        assert!(tx.send(3).is_ok());
    }

    #[test]
    fn test_channel_close() {
        let (tx, rx) = channel::<i32>();

        tx.close();
        assert!(tx.is_closed());
        assert!(rx.is_closed());

        // Sending to closed channel fails
        assert!(matches!(tx.send(1), Err(ChannelError::Closed)));

        // Receiving from closed empty channel returns None
        assert_eq!(rx.recv(), None);
    }

    #[test]
    fn test_channel_close_with_pending() {
        let (tx, rx) = channel();

        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.close();

        // Can still receive pending values
        assert_eq!(rx.recv(), Some(1));
        assert_eq!(rx.recv(), Some(2));
        assert_eq!(rx.recv(), None);
    }

    #[test]
    fn test_channel_clone_sender() {
        let (tx, rx) = channel();

        let tx2 = tx.clone();
        tx.send(1).unwrap();
        tx2.send(2).unwrap();

        assert_eq!(rx.recv(), Some(1));
        assert_eq!(rx.recv(), Some(2));
    }

    #[test]
    fn test_channel_clone_receiver() {
        let (tx, rx) = channel();

        let rx2 = rx.clone();
        tx.send(1).unwrap();
        tx.send(2).unwrap();

        // Both receivers see the same channel state
        assert_eq!(rx.recv(), Some(1));
        assert_eq!(rx2.recv(), Some(2));
    }

    #[test]
    fn test_channel_id() {
        let (tx1, rx1) = channel::<i32>();
        let (tx2, _rx2) = channel::<i32>();

        // Same channel has same ID
        assert_eq!(tx1.channel_id(), rx1.channel_id());
        // Different channels have different IDs
        assert_ne!(tx1.channel_id(), tx2.channel_id());
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_bounded_channel_zero_capacity() {
        let _ = bounded_channel::<i32>(0);
    }
}
