/// Async Runtime Module
///
/// Provides comprehensive async/await support for Demetrios:
///
/// - `future`: Core Future trait and Poll types
/// - `task`: Task spawning and JoinHandle
/// - `executor`: Runtime and executors
/// - `io`: Async file I/O
/// - `net`: Async networking
/// - `time`: Timers, sleep, and timeouts
/// - `channel`: Async message passing
/// - `select`: Combinators for concurrent futures
/// - `stream`: Async iterators
/// - `sync`: Async synchronization primitives
///
/// # Quick Start
///
/// ```
/// import async::*
///
/// async fn main() {
///     // Spawn concurrent tasks
///     let task1 = spawn async {
///         sleep(Duration::from_secs(1)).await
///         "hello"
///     }
///
///     let task2 = spawn async {
///         sleep(Duration::from_millis(500)).await
///         "world"
///     }
///
///     // Wait for both
///     let (a, b) = join(task1, task2).await
///     println("{} {}", a, b)
/// }
///
/// fn entry() {
///     block_on(main())
/// }
/// ```

module async

// Core exports
pub use async::future::{
    Future,
    Poll,
    Context,
    Waker,
    ready,
    pending,
    yield_now,
    lazy,
    Fuse,
}

pub use async::task::{
    Task,
    TaskId,
    TaskState,
    JoinHandle,
    SafeJoinHandle,
    JoinResult,
    TaskBuilder,
    spawn_blocking,
    current_task_id,
}

pub use async::executor::{
    Runtime,
    RuntimeBuilder,
    RuntimeConfig,
    LocalExecutor,
    ThreadPoolExecutor,
    block_on,
    spawn,
    Handle,
}

pub use async::io::{
    AsyncRead,
    AsyncWrite,
    AsyncSeek,
    AsyncBufRead,
    File,
    OpenOptions,
    BufReader,
    BufWriter,
    IoResult,
    IoError,
    IoErrorKind,
    SeekFrom,
    read,
    write,
    read_to_string,
    write_string,
    copy,
}

pub use async::net::{
    TcpListener,
    TcpStream,
    UdpSocket,
    IpAddr,
    Ipv4Addr,
    Ipv6Addr,
    SocketAddr,
    ToSocketAddrs,
    lookup_host,
    lookup_addr,
}

pub use async::time::{
    Duration,
    Instant,
    Sleep,
    Timeout,
    Interval,
    sleep,
    sleep_until,
    timeout,
    timeout_at,
    interval,
    interval_at,
    Elapsed,
    Throttle,
    Debounce,
}

pub use async::channel::{
    mpsc,
    oneshot,
    broadcast,
    watch,
    SendError,
    RecvError,
    TryRecvError,
    TrySendError,
}

pub use async::select::{
    select,
    select3,
    select4,
    select_vec,
    SelectResult,
    Select3Result,
    Select4Result,
    join,
    join3,
    join4,
    join_all,
    try_join,
    try_join_all,
    race,
    race_all,
    Either,
}

pub use async::stream::{
    Stream,
    StreamExt,
    BoxStream,
    empty,
    once,
    iter,
    repeat,
    from_fn,
    unfold,
}

pub use async::sync::{
    Mutex,
    MutexGuard,
    RwLock,
    RwLockReadGuard,
    RwLockWriteGuard,
    Semaphore,
    SemaphorePermit,
    Barrier,
    BarrierWaitResult,
    Notify,
    OnceCell,
}

// Advanced async features
pub use async::advanced::{
    // Async traits
    AsyncTrait,
    BoxFuture,
    LocalBoxFuture,

    // Cancellation
    CancellationToken,
    CancellationTokenSource,

    // Async drop
    AsyncDrop,
    AsyncDropGuard,

    // Structured concurrency
    Scope,
    ScopedJoinHandle,
    scope,
    TaskGroup,

    // Task-local storage
    TaskLocalKey,
    task_local,

    // Utilities
    with_timeout_cancellable,
    TimeoutError,
    retry,
    RetryConfig,
}
