/// Task and JoinHandle
///
/// Represents a spawned asynchronous task and provides mechanisms
/// for awaiting its completion and retrieving its result.

module async::task

import async::future::{Future, Poll, Context, Waker}

/// Unique identifier for a task
pub struct TaskId {
    id: u64,
}

impl TaskId {
    /// Creates a new task ID
    fn new(id: u64) -> TaskId {
        TaskId { id }
    }

    /// Returns the raw ID value
    pub fn as_u64(self) -> u64 {
        self.id
    }
}

impl Eq for TaskId {
    fn eq(&self, other: &TaskId) -> bool {
        self.id == other.id
    }
}

impl Hash for TaskId {
    fn hash(&self) -> u64 {
        self.id
    }
}

/// The state of a task
pub enum TaskState {
    /// Task is waiting to be polled
    Pending,
    /// Task is currently being polled
    Running,
    /// Task has completed successfully
    Completed,
    /// Task has been cancelled
    Cancelled,
    /// Task panicked during execution
    Panicked,
}

impl TaskState {
    /// Returns true if the task is in a terminal state
    pub fn is_terminal(self) -> bool {
        match self {
            TaskState::Completed => true,
            TaskState::Cancelled => true,
            TaskState::Panicked => true,
            _ => false,
        }
    }
}

/// A handle to an awaitable spawned task
///
/// When a task is spawned, a JoinHandle is returned that can be used
/// to await the task's completion and retrieve its result.
///
/// # Example
/// ```
/// async fn example() {
///     let handle = spawn {
///         compute_something().await
///     }
///
///     // Do other work...
///
///     // Wait for the task to complete
///     let result = handle.await
/// }
/// ```
pub struct JoinHandle<T> {
    /// The task ID
    task_id: TaskId,
    /// Shared state with the task
    state: &TaskSharedState<T>,
}

/// Shared state between task and JoinHandle
struct TaskSharedState<T> {
    /// Current task state
    state: TaskState,
    /// The result when completed
    result: Option<T>,
    /// Waker to notify when complete
    waker: Option<Waker>,
    /// Whether the handle has been dropped
    handle_dropped: bool,
}

impl<T> TaskSharedState<T> {
    fn new() -> TaskSharedState<T> {
        TaskSharedState {
            state: TaskState::Pending,
            result: None,
            waker: None,
            handle_dropped: false,
        }
    }
}

impl<T> JoinHandle<T> {
    /// Creates a new JoinHandle
    pub(crate) fn new(task_id: TaskId, state: &TaskSharedState<T>) -> JoinHandle<T> {
        JoinHandle { task_id, state }
    }

    /// Returns the task's unique identifier
    pub fn id(&self) -> TaskId {
        self.task_id
    }

    /// Returns true if the task has completed
    pub fn is_finished(&self) -> bool {
        self.state.state.is_terminal()
    }

    /// Cancels the task
    ///
    /// This signals that the task should stop execution. The task
    /// may not stop immediately - it will be cancelled at the next
    /// yield point.
    pub fn cancel(&self) {
        if !self.state.state.is_terminal() {
            self.state.state = TaskState::Cancelled;
            // Wake the task so it can observe cancellation
            if let Some(waker) = &self.state.waker {
                waker.wake_by_ref();
            }
        }
    }

    /// Detaches the handle, allowing the task to run in the background
    ///
    /// After calling detach, the task will continue running but its
    /// result will be discarded when it completes.
    pub fn detach(self) {
        self.state.handle_dropped = true;
        // Don't drop - let the task continue
    }

    /// Blocks the current thread until the task completes
    ///
    /// Note: This should only be used from synchronous code.
    /// In async code, use `.await` instead.
    pub fn blocking_join(self) -> T with Panic {
        // Spin wait (in real impl, would use proper synchronization)
        while !self.is_finished() {
            // Yield to scheduler
        }

        match self.state.result.take() {
            Some(v) => v,
            None => panic("Task completed but no result available"),
        }
    }
}

impl<T> Future for JoinHandle<T> {
    type Output = T

    fn poll(&mut self, cx: &mut Context) -> Poll<T> {
        match self.state.state {
            TaskState::Completed => {
                match self.state.result.take() {
                    Some(v) => Poll::Ready(v),
                    None => panic("Task completed but no result"),
                }
            }
            TaskState::Cancelled => {
                panic("Task was cancelled")
            }
            TaskState::Panicked => {
                panic("Task panicked during execution")
            }
            _ => {
                // Register waker for notification
                self.state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

impl<T> Drop for JoinHandle<T> {
    fn drop(&mut self) {
        self.state.handle_dropped = true;
    }
}

/// Result type for tasks that may be cancelled
pub enum JoinResult<T> {
    /// Task completed successfully
    Ok(T),
    /// Task was cancelled
    Cancelled,
    /// Task panicked
    Panicked(string),
}

impl<T> JoinResult<T> {
    /// Returns true if the task completed successfully
    pub fn is_ok(&self) -> bool {
        match self {
            JoinResult::Ok(_) => true,
            _ => false,
        }
    }

    /// Unwraps the successful result, panicking otherwise
    pub fn unwrap(self) -> T with Panic {
        match self {
            JoinResult::Ok(v) => v,
            JoinResult::Cancelled => panic("Task was cancelled"),
            JoinResult::Panicked(msg) => panic(msg),
        }
    }

    /// Returns the value if Ok, or a default otherwise
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            JoinResult::Ok(v) => v,
            _ => default,
        }
    }
}

/// A cancellation-safe JoinHandle that returns JoinResult
pub struct SafeJoinHandle<T> {
    inner: JoinHandle<T>,
}

impl<T> SafeJoinHandle<T> {
    /// Wraps a JoinHandle in a SafeJoinHandle
    pub fn new(handle: JoinHandle<T>) -> SafeJoinHandle<T> {
        SafeJoinHandle { inner: handle }
    }

    /// Returns the task's unique identifier
    pub fn id(&self) -> TaskId {
        self.inner.id()
    }

    /// Cancels the task
    pub fn cancel(&self) {
        self.inner.cancel()
    }
}

impl<T> Future for SafeJoinHandle<T> {
    type Output = JoinResult<T>

    fn poll(&mut self, cx: &mut Context) -> Poll<JoinResult<T>> {
        match self.inner.state.state {
            TaskState::Completed => {
                match self.inner.state.result.take() {
                    Some(v) => Poll::Ready(JoinResult::Ok(v)),
                    None => Poll::Ready(JoinResult::Panicked("No result".to_string())),
                }
            }
            TaskState::Cancelled => {
                Poll::Ready(JoinResult::Cancelled)
            }
            TaskState::Panicked => {
                Poll::Ready(JoinResult::Panicked("Task panicked".to_string()))
            }
            _ => {
                self.inner.state.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// A task that wraps a future for execution by the runtime
pub struct Task<F: Future> {
    /// Unique task identifier
    id: TaskId,
    /// The wrapped future
    future: F,
    /// Shared state with JoinHandle
    shared_state: &TaskSharedState<F::Output>,
}

impl<F: Future> Task<F> {
    /// Creates a new task wrapping the given future
    pub fn new(id: TaskId, future: F, shared_state: &TaskSharedState<F::Output>) -> Task<F> {
        Task { id, future, shared_state }
    }

    /// Returns the task's ID
    pub fn id(&self) -> TaskId {
        self.id
    }

    /// Polls the task
    pub fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        // Check for cancellation
        if matches!(self.shared_state.state, TaskState::Cancelled) {
            return Poll::Ready(());
        }

        // Mark as running
        self.shared_state.state = TaskState::Running;

        // Poll the inner future
        match self.future.poll(cx) {
            Poll::Ready(result) => {
                self.shared_state.result = Some(result);
                self.shared_state.state = TaskState::Completed;

                // Wake the JoinHandle if it's waiting
                if let Some(waker) = self.shared_state.waker.take() {
                    waker.wake();
                }

                Poll::Ready(())
            }
            Poll::Pending => {
                self.shared_state.state = TaskState::Pending;
                Poll::Pending
            }
        }
    }
}

/// Builder for configuring task spawn options
pub struct TaskBuilder {
    name: Option<string>,
    stack_size: Option<usize>,
}

impl TaskBuilder {
    /// Creates a new TaskBuilder with default options
    pub fn new() -> TaskBuilder {
        TaskBuilder {
            name: None,
            stack_size: None,
        }
    }

    /// Sets the name of the task (for debugging)
    pub fn name(mut self, name: string) -> TaskBuilder {
        self.name = Some(name);
        self
    }

    /// Sets the stack size for the task
    pub fn stack_size(mut self, size: usize) -> TaskBuilder {
        self.stack_size = Some(size);
        self
    }

    /// Spawns a task with the configured options
    pub fn spawn<F: Future>(self, future: F) -> JoinHandle<F::Output> with Async {
        // The actual spawn is handled by the runtime
        spawn_with_options(future, self.name, self.stack_size)
    }
}

/// Internal function to spawn with options (implemented by runtime)
fn spawn_with_options<F: Future>(
    future: F,
    name: Option<string>,
    stack_size: Option<usize>
) -> JoinHandle<F::Output> with Async {
    // This would be implemented by the runtime
    // For now, just spawn normally
    spawn future
}

/// Yields execution back to the runtime scheduler
///
/// This allows other tasks to run. Useful in compute-heavy
/// loops to prevent starving other tasks.
pub async fn yield_now() with Async {
    YieldOnce { yielded: false }.await
}

struct YieldOnce {
    yielded: bool,
}

impl Future for YieldOnce {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if self.yielded {
            Poll::Ready(())
        } else {
            self.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

/// Spawns a blocking operation on a dedicated thread pool
///
/// Use this for CPU-intensive or blocking I/O operations that
/// would otherwise block the async executor.
pub async fn spawn_blocking<T, F: FnOnce() -> T>(f: F) -> T with Async {
    // This would be implemented by the runtime to run on a blocking thread pool
    f()
}

/// Returns the current task's ID, if called from within a task
pub fn current_task_id() -> Option<TaskId> {
    // This would be implemented by the runtime using task-local storage
    None
}
