/// Future Trait and Poll Types
///
/// Core abstractions for asynchronous programming in Demetrios.
/// Futures represent values that may not be available yet but will
/// be computed asynchronously.

module async::future

/// Result of polling a future
///
/// When a future is polled, it returns either:
/// - `Ready(T)` if the computation is complete
/// - `Pending` if the computation is still in progress
pub enum Poll<T> {
    /// The future has completed with a value
    Ready(T),
    /// The future is not yet complete
    Pending,
}

impl<T> Poll<T> {
    /// Returns true if the poll result is Ready
    pub fn is_ready(self) -> bool {
        match self {
            Poll::Ready(_) => true,
            Poll::Pending => false,
        }
    }

    /// Returns true if the poll result is Pending
    pub fn is_pending(self) -> bool {
        match self {
            Poll::Ready(_) => false,
            Poll::Pending => true,
        }
    }

    /// Maps a `Poll<T>` to `Poll<U>` by applying a function
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> Poll<U> {
        match self {
            Poll::Ready(t) => Poll::Ready(f(t)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Unwraps the Ready value, panicking if Pending
    pub fn unwrap(self) -> T with Panic {
        match self {
            Poll::Ready(t) => t,
            Poll::Pending => panic("called unwrap on Poll::Pending"),
        }
    }

    /// Returns the Ready value or a default
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            Poll::Ready(t) => t,
            Poll::Pending => default,
        }
    }
}

/// A handle to wake a task when progress can be made
///
/// When a future returns Pending, it should arrange for the waker
/// to be called when it can make progress. This signals the executor
/// to poll the future again.
pub struct Waker {
    /// Internal wake function pointer
    wake_fn: fn(),
    /// Data pointer for the waker
    data: &void,
}

impl Waker {
    /// Creates a new waker from a wake function
    pub fn new(wake_fn: fn(), data: &void) -> Waker {
        Waker { wake_fn, data }
    }

    /// Wake up the task associated with this waker
    pub fn wake(self) {
        (self.wake_fn)()
    }

    /// Wake up the task by reference (doesn't consume the waker)
    pub fn wake_by_ref(&self) {
        (self.wake_fn)()
    }

    /// Returns true if this waker will wake the same task as another
    pub fn will_wake(&self, other: &Waker) -> bool {
        self.data == other.data
    }

    /// Clone the waker
    pub fn clone(&self) -> Waker {
        Waker {
            wake_fn: self.wake_fn,
            data: self.data,
        }
    }
}

/// Context passed to futures when polling
///
/// Contains a reference to the waker that should be used to signal
/// when the future can make progress.
pub struct Context {
    waker: &Waker,
}

impl Context {
    /// Creates a new context with the given waker
    pub fn from_waker(waker: &Waker) -> Context {
        Context { waker }
    }

    /// Returns a reference to the waker
    pub fn waker(&self) -> &Waker {
        self.waker
    }
}

/// The Future trait - core abstraction for async computation
///
/// A future represents an asynchronous computation that produces
/// a value of type `Output` when complete.
///
/// # Example
/// ```
/// struct MyFuture {
///     value: i32,
///     ready: bool,
/// }
///
/// impl Future for MyFuture {
///     type Output = i32
///
///     fn poll(&mut self, cx: &mut Context) -> Poll<i32> {
///         if self.ready {
///             Poll::Ready(self.value)
///         } else {
///             // Register waker to be called later
///             cx.waker().wake_by_ref()
///             Poll::Pending
///         }
///     }
/// }
/// ```
pub trait Future {
    /// The type of value produced on completion
    type Output

    /// Attempt to resolve the future to a final value
    ///
    /// This method should not block. If the future can make progress,
    /// it should do so and return `Ready` or `Pending`.
    fn poll(&mut self, cx: &mut Context) -> Poll<Self::Output>
}

/// Extension methods for Future
impl<F: Future> F {
    /// Maps this future's output to a different type
    pub fn map<U, M: Fn(F::Output) -> U>(self, mapper: M) -> MapFuture<F, M> {
        MapFuture { future: self, mapper }
    }

    /// Chains another future after this one completes
    pub fn and_then<U, G: Future<Output = U>, C: Fn(F::Output) -> G>(
        self,
        continuation: C
    ) -> AndThenFuture<F, C, G> {
        AndThenFuture {
            state: AndThenState::First { future: self, continuation },
        }
    }

    /// Runs this future with a timeout
    pub fn timeout(self, duration: Duration) -> TimeoutFuture<F> {
        TimeoutFuture {
            future: self,
            deadline: Instant::now() + duration,
        }
    }

    /// Boxes this future for dynamic dispatch
    pub fn boxed(self) -> BoxFuture<F::Output> {
        BoxFuture { inner: Box::new(self) }
    }
}

/// A future that maps the output of another future
pub struct MapFuture<F: Future, M> {
    future: F,
    mapper: M,
}

impl<F: Future, U, M: Fn(F::Output) -> U> Future for MapFuture<F, M> {
    type Output = U

    fn poll(&mut self, cx: &mut Context) -> Poll<U> {
        match self.future.poll(cx) {
            Poll::Ready(value) => Poll::Ready((self.mapper)(value)),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// State for AndThen combinator
enum AndThenState<F, C, G> {
    First { future: F, continuation: C },
    Second { future: G },
    Complete,
}

/// A future that chains two futures together
pub struct AndThenFuture<F: Future, C, G: Future> {
    state: AndThenState<F, C, G>,
}

impl<F: Future, U, G: Future<Output = U>, C: Fn(F::Output) -> G> Future for AndThenFuture<F, C, G> {
    type Output = U

    fn poll(&mut self, cx: &mut Context) -> Poll<U> {
        loop {
            match &mut self.state {
                AndThenState::First { future, continuation } => {
                    match future.poll(cx) {
                        Poll::Ready(value) => {
                            let next = continuation(value);
                            self.state = AndThenState::Second { future: next };
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                AndThenState::Second { future } => {
                    let result = future.poll(cx);
                    if result.is_ready() {
                        self.state = AndThenState::Complete;
                    }
                    return result;
                }
                AndThenState::Complete => {
                    panic("AndThenFuture polled after completion")
                }
            }
        }
    }
}

/// A boxed future for dynamic dispatch
pub struct BoxFuture<T> {
    inner: Box<dyn Future<Output = T>>,
}

impl<T> Future for BoxFuture<T> {
    type Output = T

    fn poll(&mut self, cx: &mut Context) -> Poll<T> {
        self.inner.poll(cx)
    }
}

/// A future that is immediately ready with a value
pub struct Ready<T> {
    value: Option<T>,
}

/// Creates a future that is immediately ready with a value
pub fn ready<T>(value: T) -> Ready<T> {
    Ready { value: Some(value) }
}

impl<T> Future for Ready<T> {
    type Output = T

    fn poll(&mut self, _cx: &mut Context) -> Poll<T> {
        match self.value.take() {
            Some(v) => Poll::Ready(v),
            None => panic("Ready future polled after completion"),
        }
    }
}

/// A future that never resolves
pub struct Pending<T> {
    _marker: PhantomData<T>,
}

/// Creates a future that is never ready
pub fn pending<T>() -> Pending<T> {
    Pending { _marker: PhantomData }
}

impl<T> Future for Pending<T> {
    type Output = T

    fn poll(&mut self, _cx: &mut Context) -> Poll<T> {
        Poll::Pending
    }
}

/// A future that yields once before completing
pub struct YieldNow {
    yielded: bool,
}

/// Creates a future that yields control back to the executor once
pub fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

impl Future for YieldNow {
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

/// A future that resolves to an error
pub struct Err<E> {
    error: Option<E>,
}

/// Creates a future that immediately fails with an error
pub fn err<T, E>(error: E) -> ResultFuture<T, E> {
    ResultFuture::Err(Err { error: Some(error) })
}

/// A future that resolves to a Result
pub enum ResultFuture<T, E> {
    Ok(Ready<T>),
    Err(Err<E>),
}

impl<T, E> Future for ResultFuture<T, E> {
    type Output = Result<T, E>

    fn poll(&mut self, cx: &mut Context) -> Poll<Result<T, E>> {
        match self {
            ResultFuture::Ok(ready) => ready.poll(cx).map(|v| Ok(v)),
            ResultFuture::Err(err) => {
                match err.error.take() {
                    Some(e) => Poll::Ready(Err(e)),
                    None => panic("Err future polled after completion"),
                }
            }
        }
    }
}

/// A lazy future that evaluates a closure on first poll
pub struct Lazy<F> {
    func: Option<F>,
}

/// Creates a future that lazily evaluates a closure
pub fn lazy<T, F: FnOnce() -> T>(f: F) -> Lazy<F> {
    Lazy { func: Some(f) }
}

impl<T, F: FnOnce() -> T> Future for Lazy<F> {
    type Output = T

    fn poll(&mut self, _cx: &mut Context) -> Poll<T> {
        match self.func.take() {
            Some(f) => Poll::Ready(f()),
            None => panic("Lazy future polled after completion"),
        }
    }
}

/// A future that polls another future with a fused behavior
/// (returns Pending after completion instead of panicking)
pub struct Fuse<F: Future> {
    inner: Option<F>,
}

impl<F: Future> Fuse<F> {
    /// Creates a new fused future
    pub fn new(future: F) -> Fuse<F> {
        Fuse { inner: Some(future) }
    }

    /// Returns true if the inner future has completed
    pub fn is_done(&self) -> bool {
        self.inner.is_none()
    }
}

impl<F: Future> Future for Fuse<F> {
    type Output = F::Output

    fn poll(&mut self, cx: &mut Context) -> Poll<F::Output> {
        match &mut self.inner {
            Some(future) => {
                match future.poll(cx) {
                    Poll::Ready(value) => {
                        self.inner = None;
                        Poll::Ready(value)
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
            None => Poll::Pending,
        }
    }
}
