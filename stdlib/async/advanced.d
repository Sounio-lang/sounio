/// Advanced Async Features
///
/// This module provides advanced async patterns including async traits,
/// async drop, cancellation tokens, structured concurrency, and task-local storage.

module async::advanced;

use async::future::*;
use async::task::*;
use async::sync::*;

// =============================================================================
// Async Trait Support
// =============================================================================

/// Marker trait for async traits.
///
/// Async traits allow defining traits with async methods. Unlike regular traits,
/// async traits have special handling for the returned futures.
///
/// # Example
///
/// ```d
/// #[async_trait]
/// pub trait AsyncRead {
///     async fn read(&mut self, buf: &mut [u8]) -> Result<usize, IoError>;
/// }
///
/// // Implementation
/// impl AsyncRead for TcpStream {
///     async fn read(&mut self, buf: &mut [u8]) -> Result<usize, IoError> {
///         // Implementation
///     }
/// }
/// ```
pub trait AsyncTrait {}

/// Type alias for boxed futures returned by async trait methods.
///
/// Since async trait methods return different concrete future types,
/// they need to be boxed for trait object safety.
pub type BoxFuture<'a, T> = Box<dyn Future<Output = T> + Send + 'a>;

/// Type alias for boxed futures that don't need to be Send.
pub type LocalBoxFuture<'a, T> = Box<dyn Future<Output = T> + 'a>;

// =============================================================================
// Cancellation Tokens
// =============================================================================

/// A cancellation token that can be used to cancel async operations.
///
/// Cancellation tokens provide cooperative cancellation - operations must
/// check the token periodically and respond to cancellation requests.
///
/// # Example
///
/// ```d
/// let source = CancellationTokenSource::new();
/// let token = source.token();
///
/// spawn async {
///     loop {
///         if token.is_cancelled() {
///             println("Operation cancelled");
///             break;
///         }
///
///         // Do some work
///         do_work().await;
///     }
/// };
///
/// // Later, cancel the operation
/// source.cancel();
/// ```
pub struct CancellationToken {
    /// Shared cancellation state
    inner: Arc<CancellationInner>,
}

struct CancellationInner {
    /// Whether cancellation has been requested
    cancelled: AtomicBool,
    /// Notify waiters when cancelled
    notify: Notify,
    /// Parent token (for linked cancellation)
    parent: Option<CancellationToken>,
    /// Child tokens
    children: Mutex<Vec<CancellationToken>>,
}

impl CancellationToken {
    /// Create a new cancellation token.
    fn new(parent: Option<CancellationToken>) -> Self with Alloc {
        CancellationToken {
            inner: Arc::new(CancellationInner {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
                parent: parent,
                children: Mutex::new(Vec::new()),
            }),
        }
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    /// Wait until cancellation is requested.
    pub async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }

        self.inner.notify.notified().await;
    }

    /// Create a child token that is cancelled when this token is cancelled.
    pub fn child_token(&self) -> CancellationToken with Alloc {
        let child = CancellationToken::new(Some(self.clone()));

        // Register with parent
        let mut children = self.inner.children.lock().await;
        children.push(child.clone());

        // If parent already cancelled, cancel child immediately
        if self.is_cancelled() {
            child.cancel_internal();
        }

        child
    }

    /// Cancel this token (internal, called by source).
    fn cancel_internal(&self) {
        if self.inner.cancelled.swap(true, Ordering::AcqRel) {
            // Already cancelled
            return;
        }

        // Notify all waiters
        self.inner.notify.notify_all();

        // Cancel all children
        if let Ok(children) = self.inner.children.try_lock() {
            for child in children.iter() {
                child.cancel_internal();
            }
        }
    }
}

impl Clone for CancellationToken {
    fn clone(&self) -> Self {
        CancellationToken {
            inner: self.inner.clone(),
        }
    }
}

/// Source for creating and controlling cancellation tokens.
///
/// The source owns the ability to trigger cancellation.
pub struct CancellationTokenSource {
    /// The token
    token: CancellationToken,
}

impl CancellationTokenSource {
    /// Create a new cancellation token source.
    pub fn new() -> Self with Alloc {
        CancellationTokenSource {
            token: CancellationToken::new(None),
        }
    }

    /// Get a token that can be used to check for cancellation.
    pub fn token(&self) -> CancellationToken {
        self.token.clone()
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.token.cancel_internal();
    }

    /// Check if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.token.is_cancelled()
    }

    /// Cancel after a delay.
    pub fn cancel_after(&self, delay: Duration) with Async {
        let token = self.token.clone();
        spawn async move {
            sleep(delay).await;
            token.cancel_internal();
        };
    }
}

// =============================================================================
// Async Drop
// =============================================================================

/// Trait for types that need async cleanup.
///
/// Unlike regular Drop which runs synchronously, AsyncDrop allows
/// awaiting async operations during cleanup.
///
/// # Example
///
/// ```d
/// struct Connection {
///     handle: ConnectionHandle,
/// }
///
/// impl AsyncDrop for Connection {
///     async fn async_drop(&mut self) {
///         // Send close message and wait for acknowledgment
///         self.handle.close().await;
///     }
/// }
/// ```
pub trait AsyncDrop {
    /// Perform async cleanup.
    async fn async_drop(&mut self);
}

/// Guard that runs async drop when dropped.
///
/// Since the regular Drop trait cannot be async, this wrapper
/// provides a way to ensure async cleanup happens.
pub struct AsyncDropGuard<T>
where
    T: AsyncDrop,
{
    /// The value to drop
    value: Option<T>,
    /// Runtime handle for running async drop
    runtime: RuntimeHandle,
}

impl<T> AsyncDropGuard<T>
where
    T: AsyncDrop,
{
    /// Create a new async drop guard.
    pub fn new(value: T, runtime: RuntimeHandle) -> Self {
        AsyncDropGuard {
            value: Some(value),
            runtime: runtime,
        }
    }

    /// Get a reference to the inner value.
    pub fn get(&self) -> Option<&T> {
        self.value.as_ref()
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        self.value.as_mut()
    }

    /// Take the inner value, preventing async drop.
    pub fn take(&mut self) -> Option<T> {
        self.value.take()
    }

    /// Manually run async drop.
    pub async fn async_drop(&mut self) {
        if let Some(mut value) = self.value.take() {
            value.async_drop().await;
        }
    }
}

impl<T> Drop for AsyncDropGuard<T>
where
    T: AsyncDrop,
{
    fn drop(&mut self) {
        if let Some(mut value) = self.value.take() {
            // Block on async drop
            self.runtime.block_on(async {
                value.async_drop().await;
            });
        }
    }
}

// =============================================================================
// Structured Concurrency
// =============================================================================

/// A scope for structured concurrency.
///
/// Scopes ensure that all spawned tasks complete before the scope exits,
/// preventing "fire and forget" task leaks.
///
/// # Example
///
/// ```d
/// scope(|s| async {
///     s.spawn(async { task1().await });
///     s.spawn(async { task2().await });
///     s.spawn(async { task3().await });
///
///     // All tasks are guaranteed to complete when scope exits
/// }).await;
/// ```
pub struct Scope<'scope> {
    /// Tasks spawned in this scope
    tasks: Mutex<Vec<JoinHandle<()>>>,
    /// Cancellation token for this scope
    cancellation: CancellationToken,
    /// Error from any task
    error: Mutex<Option<Box<dyn Error + Send + 'scope>>>,
}

impl<'scope> Scope<'scope> {
    /// Create a new scope.
    fn new(cancellation: CancellationToken) -> Self with Alloc {
        Scope {
            tasks: Mutex::new(Vec::new()),
            cancellation: cancellation,
            error: Mutex::new(None),
        }
    }

    /// Spawn a task in this scope.
    ///
    /// The task will be cancelled if the scope is cancelled.
    pub fn spawn<F>(&self, future: F) with Async, Alloc
    where
        F: Future<Output = ()> + Send + 'scope,
    {
        let token = self.cancellation.clone();

        let handle = spawn(async move {
            select! {
                _ = token.cancelled() => {
                    // Cancelled
                }
                _ = future => {
                    // Completed
                }
            }
        });

        let mut tasks = self.tasks.lock();
        tasks.push(handle);
    }

    /// Spawn a task that produces a result.
    pub fn spawn_result<F, T>(&self, future: F) -> ScopedJoinHandle<T> with Async, Alloc
    where
        F: Future<Output = T> + Send + 'scope,
        T: Send + 'scope,
    {
        let token = self.cancellation.clone();
        let (tx, rx) = oneshot::channel();

        let handle = spawn(async move {
            select! {
                _ = token.cancelled() => {
                    // Don't send result if cancelled
                }
                result = future => {
                    let _ = tx.send(result);
                }
            }
        });

        let mut tasks = self.tasks.lock();
        tasks.push(handle);

        ScopedJoinHandle { receiver: rx }
    }

    /// Cancel all tasks in the scope.
    pub fn cancel(&self) {
        self.cancellation.cancel_internal();
    }

    /// Wait for all tasks to complete.
    async fn wait(&self) {
        let tasks = {
            let mut guard = self.tasks.lock();
            std::mem::take(&mut *guard)
        };

        for task in tasks {
            let _ = task.await;
        }
    }
}

/// Handle for getting the result of a scoped task.
pub struct ScopedJoinHandle<T> {
    receiver: oneshot::Receiver<T>,
}

impl<T> ScopedJoinHandle<T> {
    /// Wait for the task to complete and get the result.
    pub async fn join(self) -> Option<T> {
        self.receiver.await.ok()
    }
}

/// Run a function with a structured concurrency scope.
///
/// All tasks spawned in the scope are guaranteed to complete
/// before this function returns.
pub async fn scope<'scope, F, R>(f: F) -> R with Async, Alloc
where
    F: FnOnce(&Scope<'scope>) -> R,
{
    let cancellation = CancellationTokenSource::new();
    let scope = Scope::new(cancellation.token());

    let result = f(&scope);

    // Wait for all tasks
    scope.wait().await;

    result
}

/// A task group for managing multiple concurrent tasks.
///
/// Unlike Scope, TaskGroup allows dynamic task addition and
/// provides more control over error handling.
///
/// # Example
///
/// ```d
/// let mut group = TaskGroup::new();
///
/// group.spawn(async { fetch_data("url1").await });
/// group.spawn(async { fetch_data("url2").await });
/// group.spawn(async { fetch_data("url3").await });
///
/// // Wait for all tasks
/// let results = group.join_all().await;
/// ```
pub struct TaskGroup<T>
where
    T: Send + 'static,
{
    /// Task handles
    handles: Vec<JoinHandle<T>>,
    /// Cancellation source
    cancellation: CancellationTokenSource,
    /// Maximum concurrent tasks (0 = unlimited)
    max_concurrent: usize,
    /// Semaphore for limiting concurrency
    semaphore: Option<Arc<Semaphore>>,
}

impl<T> TaskGroup<T>
where
    T: Send + 'static,
{
    /// Create a new task group.
    pub fn new() -> Self with Alloc {
        TaskGroup {
            handles: Vec::new(),
            cancellation: CancellationTokenSource::new(),
            max_concurrent: 0,
            semaphore: None,
        }
    }

    /// Create a task group with limited concurrency.
    pub fn with_max_concurrent(max: usize) -> Self with Alloc {
        TaskGroup {
            handles: Vec::new(),
            cancellation: CancellationTokenSource::new(),
            max_concurrent: max,
            semaphore: if max > 0 { Some(Arc::new(Semaphore::new(max))) } else { None },
        }
    }

    /// Spawn a task in the group.
    pub fn spawn<F>(&mut self, future: F) with Async, Alloc
    where
        F: Future<Output = T> + Send + 'static,
    {
        let token = self.cancellation.token();
        let semaphore = self.semaphore.clone();

        let handle = spawn(async move {
            // Acquire semaphore permit if limited
            let _permit = if let Some(sem) = semaphore {
                Some(sem.acquire().await)
            } else {
                None
            };

            select! {
                _ = token.cancelled() => {
                    panic("task cancelled")
                }
                result = future => {
                    result
                }
            }
        });

        self.handles.push(handle);
    }

    /// Get the number of tasks in the group.
    pub fn len(&self) -> usize {
        self.handles.len()
    }

    /// Check if the group is empty.
    pub fn is_empty(&self) -> bool {
        self.handles.is_empty()
    }

    /// Cancel all tasks.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }

    /// Wait for all tasks to complete and collect results.
    pub async fn join_all(self) -> Vec<Result<T, JoinError>> {
        let mut results = Vec::with_capacity(self.handles.len());

        for handle in self.handles {
            results.push(handle.await);
        }

        results
    }

    /// Wait for the first task to complete.
    pub async fn join_first(&mut self) -> Option<Result<T, JoinError>> {
        if self.handles.is_empty() {
            return None;
        }

        // Race all handles
        let (result, index, remaining) = select_vec(self.handles.drain(..).collect()).await;

        // Put remaining handles back
        self.handles = remaining;

        Some(result)
    }
}

// =============================================================================
// Task-Local Storage
// =============================================================================

/// Key for task-local storage.
///
/// Each key is unique and identifies a slot in task-local storage.
pub struct TaskLocalKey<T: 'static> {
    /// Unique identifier
    id: usize,
    /// Initializer function
    init: fn() -> T,
    /// Phantom data for type
    _marker: PhantomData<T>,
}

impl<T: 'static> TaskLocalKey<T> {
    /// Create a new task-local key.
    pub const fn new(init: fn() -> T) -> Self {
        TaskLocalKey {
            id: __builtin_unique_id(),
            init: init,
            _marker: PhantomData,
        }
    }

    /// Access the task-local value.
    pub fn with<F, R>(&'static self, f: F) -> R
    where
        F: FnOnce(&T) -> R,
    {
        TASK_LOCAL_STORAGE.with(|storage| {
            let value = storage.get_or_insert(self.id, || Box::new((self.init)()));
            let typed_value = value.downcast_ref::<T>().unwrap();
            f(typed_value)
        })
    }

    /// Set the task-local value for the duration of a future.
    pub async fn scope<F, R>(self: &'static Self, value: T, f: F) -> R
    where
        F: Future<Output = R>,
    {
        TASK_LOCAL_STORAGE.with(|storage| {
            storage.set(self.id, Box::new(value));
        });

        let result = f.await;

        TASK_LOCAL_STORAGE.with(|storage| {
            storage.remove(self.id);
        });

        result
    }

    /// Try to get the current value.
    pub fn try_with<F, R>(&'static self, f: F) -> Option<R>
    where
        F: FnOnce(&T) -> R,
    {
        TASK_LOCAL_STORAGE.with(|storage| {
            storage.get(self.id).map(|value| {
                let typed_value = value.downcast_ref::<T>().unwrap();
                f(typed_value)
            })
        })
    }
}

/// Macro for declaring task-local variables.
///
/// # Example
///
/// ```d
/// task_local! {
///     static REQUEST_ID: String = String::new();
///     static TRACE_CONTEXT: TraceContext = TraceContext::default();
/// }
///
/// async fn handler(request: Request) {
///     REQUEST_ID.scope(request.id.clone(), async {
///         // REQUEST_ID is available throughout this async context
///         process_request(request).await;
///     }).await;
/// }
/// ```
#[macro_export]
macro_rules! task_local {
    ($(static $name:ident : $ty:ty = $init:expr ;)*) => {
        $(
            static $name: TaskLocalKey<$ty> = TaskLocalKey::new(|| $init);
        )*
    };
}

// Task-local storage implementation
thread_local! {
    static TASK_LOCAL_STORAGE: TaskLocalStorage = TaskLocalStorage::new();
}

struct TaskLocalStorage {
    values: RefCell<HashMap<usize, Box<dyn Any + Send>>>,
}

impl TaskLocalStorage {
    fn new() -> Self {
        TaskLocalStorage {
            values: RefCell::new(HashMap::new()),
        }
    }

    fn get(&self, id: usize) -> Option<&Box<dyn Any + Send>> {
        self.values.borrow().get(&id)
    }

    fn get_or_insert(&self, id: usize, init: impl FnOnce() -> Box<dyn Any + Send>) -> &Box<dyn Any + Send> {
        self.values.borrow_mut().entry(id).or_insert_with(init);
        self.values.borrow().get(&id).unwrap()
    }

    fn set(&self, id: usize, value: Box<dyn Any + Send>) {
        self.values.borrow_mut().insert(id, value);
    }

    fn remove(&self, id: usize) {
        self.values.borrow_mut().remove(&id);
    }
}

// =============================================================================
// Timeout Utilities
// =============================================================================

/// Run a future with a timeout, supporting cancellation.
///
/// If the timeout expires, the future is cancelled via the cancellation token.
pub async fn with_timeout_cancellable<F, T>(
    duration: Duration,
    token: CancellationToken,
    future: F,
) -> Result<T, TimeoutError>
where
    F: Future<Output = T>,
{
    select! {
        _ = sleep(duration) => {
            Err(TimeoutError::Elapsed)
        }
        _ = token.cancelled() => {
            Err(TimeoutError::Cancelled)
        }
        result = future => {
            Ok(result)
        }
    }
}

/// Timeout error types.
pub enum TimeoutError {
    /// The timeout elapsed.
    Elapsed,
    /// The operation was cancelled.
    Cancelled,
}

// =============================================================================
// Retry Utilities
// =============================================================================

/// Configuration for retry behavior.
pub struct RetryConfig {
    /// Maximum number of attempts (including initial)
    pub max_attempts: usize,
    /// Initial delay between retries
    pub initial_delay: Duration,
    /// Maximum delay between retries
    pub max_delay: Duration,
    /// Multiplier for exponential backoff
    pub backoff_multiplier: f64,
    /// Whether to add jitter to delays
    pub jitter: bool,
}

impl RetryConfig {
    /// Create a default retry configuration.
    pub fn new() -> Self {
        RetryConfig {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Set maximum attempts.
    pub fn max_attempts(mut self, n: usize) -> Self {
        self.max_attempts = n;
        self
    }

    /// Set initial delay.
    pub fn initial_delay(mut self, d: Duration) -> Self {
        self.initial_delay = d;
        self
    }

    /// Set maximum delay.
    pub fn max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }
}

/// Retry a fallible async operation with exponential backoff.
pub async fn retry<F, Fut, T, E>(
    config: RetryConfig,
    mut f: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let mut delay = config.initial_delay;

    for attempt in 0..config.max_attempts {
        match f().await {
            Ok(value) => return Ok(value),
            Err(e) => {
                if attempt + 1 >= config.max_attempts {
                    return Err(e);
                }

                // Calculate next delay
                let mut next_delay = Duration::from_millis(
                    (delay.as_millis() as f64 * config.backoff_multiplier) as u64
                );

                if next_delay > config.max_delay {
                    next_delay = config.max_delay;
                }

                // Add jitter if configured
                if config.jitter {
                    let jitter = Duration::from_millis(
                        (rand::random::<f64>() * delay.as_millis() as f64 * 0.1) as u64
                    );
                    delay = delay + jitter;
                }

                sleep(delay).await;
                delay = next_delay;
            }
        }
    }

    unreachable!()
}

// =============================================================================
// Helper Types
// =============================================================================

/// Atomic boolean for cancellation state.
struct AtomicBool {
    value: std::sync::atomic::AtomicBool,
}

impl AtomicBool {
    fn new(value: bool) -> Self {
        AtomicBool {
            value: std::sync::atomic::AtomicBool::new(value),
        }
    }

    fn load(&self, order: Ordering) -> bool {
        self.value.load(order.into())
    }

    fn swap(&self, value: bool, order: Ordering) -> bool {
        self.value.swap(value, order.into())
    }
}

/// Memory ordering for atomic operations.
pub enum Ordering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Runtime handle for async operations.
pub struct RuntimeHandle {
    // Implementation details
}

impl RuntimeHandle {
    /// Block on a future.
    pub fn block_on<F, T>(&self, future: F) -> T
    where
        F: Future<Output = T>,
    {
        // Implementation provided by runtime
        todo!()
    }
}

/// Join error type.
pub struct JoinError {
    /// Whether the task was cancelled.
    pub cancelled: bool,
    /// Whether the task panicked.
    pub panicked: bool,
}

/// PhantomData marker type.
pub struct PhantomData<T>;

/// Arc type for shared ownership.
pub struct Arc<T>(std::sync::Arc<T>);

impl<T> Arc<T> {
    pub fn new(value: T) -> Self with Alloc {
        Arc(std::sync::Arc::new(value))
    }
}

impl<T> Clone for Arc<T> {
    fn clone(&self) -> Self {
        Arc(self.0.clone())
    }
}

/// Error trait.
pub trait Error {}
