/// Async Synchronization Primitives
///
/// Provides async-aware synchronization primitives:
/// - Mutex: Mutual exclusion lock
/// - RwLock: Read-write lock
/// - Semaphore: Counting semaphore
/// - Barrier: Synchronization barrier
/// - Notify: Task notification

module async::sync

import async::future::{Future, Poll, Context, Waker}

// =============================================================================
// Mutex - Async mutual exclusion
// =============================================================================

/// An async mutex for protecting shared data
///
/// Unlike std::sync::Mutex, this mutex is designed for use in async
/// code and will not block the runtime when waiting for the lock.
///
/// # Example
/// ```
/// async fn example() {
///     let mutex = Mutex::new(0)
///
///     // Acquire lock
///     let mut guard = mutex.lock().await
///     *guard += 1
/// }
/// ```
pub struct Mutex<T> {
    /// The protected data
    data: T,
    /// Whether the mutex is currently locked
    locked: bool,
    /// Waiters for the lock
    waiters: Vec<Waker>,
}

impl<T> Mutex<T> {
    /// Creates a new mutex with the given value
    pub fn new(value: T) -> Mutex<T> {
        Mutex {
            data: value,
            locked: false,
            waiters: Vec::new(),
        }
    }

    /// Acquires the lock asynchronously
    pub async fn lock(&self) -> MutexGuard<T> with Async {
        LockFuture { mutex: self }.await
    }

    /// Tries to acquire the lock immediately
    pub fn try_lock(&self) -> Option<MutexGuard<T>> {
        if !self.locked {
            self.locked = true;
            Some(MutexGuard { mutex: self })
        } else {
            None
        }
    }

    /// Returns a reference to the inner value
    ///
    /// This requires unique access and doesn't need locking.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consumes the mutex, returning the inner value
    pub fn into_inner(self) -> T {
        self.data
    }
}

struct LockFuture<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<'a, T> Future for LockFuture<'a, T> {
    type Output = MutexGuard<'a, T>

    fn poll(&mut self, cx: &mut Context) -> Poll<MutexGuard<'a, T>> {
        if !self.mutex.locked {
            self.mutex.locked = true;
            Poll::Ready(MutexGuard { mutex: self.mutex })
        } else {
            self.mutex.waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// RAII guard for a locked mutex
pub struct MutexGuard<'a, T> {
    mutex: &'a Mutex<T>,
}

impl<'a, T> MutexGuard<'a, T> {
    /// Returns a reference to the guarded data
    pub fn get(&self) -> &T {
        &self.mutex.data
    }

    /// Returns a mutable reference to the guarded data
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.mutex.data
    }
}

impl<'a, T> Deref for MutexGuard<'a, T> {
    type Target = T

    fn deref(&self) -> &T {
        &self.mutex.data
    }
}

impl<'a, T> DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.mutex.data
    }
}

impl<'a, T> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        self.mutex.locked = false;
        // Wake one waiter
        if let Some(waker) = self.mutex.waiters.pop() {
            waker.wake();
        }
    }
}

// =============================================================================
// RwLock - Async read-write lock
// =============================================================================

/// An async read-write lock
///
/// Allows multiple concurrent readers or a single writer.
///
/// # Example
/// ```
/// async fn example() {
///     let lock = RwLock::new(vec![1, 2, 3])
///
///     // Multiple readers
///     let read1 = lock.read().await
///     let read2 = lock.read().await
///
///     // Exclusive writer
///     drop(read1)
///     drop(read2)
///     let mut write = lock.write().await
///     write.push(4)
/// }
/// ```
pub struct RwLock<T> {
    data: T,
    /// Number of active readers (negative means writer has lock)
    state: i32,
    /// Waiting readers
    read_waiters: Vec<Waker>,
    /// Waiting writers
    write_waiters: Vec<Waker>,
}

impl<T> RwLock<T> {
    /// Creates a new read-write lock
    pub fn new(value: T) -> RwLock<T> {
        RwLock {
            data: value,
            state: 0,
            read_waiters: Vec::new(),
            write_waiters: Vec::new(),
        }
    }

    /// Acquires a read lock
    pub async fn read(&self) -> RwLockReadGuard<T> with Async {
        ReadLockFuture { lock: self }.await
    }

    /// Acquires a write lock
    pub async fn write(&self) -> RwLockWriteGuard<T> with Async {
        WriteLockFuture { lock: self }.await
    }

    /// Tries to acquire a read lock immediately
    pub fn try_read(&self) -> Option<RwLockReadGuard<T>> {
        if self.state >= 0 {
            self.state += 1;
            Some(RwLockReadGuard { lock: self })
        } else {
            None
        }
    }

    /// Tries to acquire a write lock immediately
    pub fn try_write(&self) -> Option<RwLockWriteGuard<T>> {
        if self.state == 0 {
            self.state = -1;
            Some(RwLockWriteGuard { lock: self })
        } else {
            None
        }
    }

    /// Returns a mutable reference (requires unique access)
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.data
    }

    /// Consumes the lock, returning the inner value
    pub fn into_inner(self) -> T {
        self.data
    }
}

struct ReadLockFuture<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Future for ReadLockFuture<'a, T> {
    type Output = RwLockReadGuard<'a, T>

    fn poll(&mut self, cx: &mut Context) -> Poll<RwLockReadGuard<'a, T>> {
        if self.lock.state >= 0 {
            self.lock.state += 1;
            Poll::Ready(RwLockReadGuard { lock: self.lock })
        } else {
            self.lock.read_waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

struct WriteLockFuture<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Future for WriteLockFuture<'a, T> {
    type Output = RwLockWriteGuard<'a, T>

    fn poll(&mut self, cx: &mut Context) -> Poll<RwLockWriteGuard<'a, T>> {
        if self.lock.state == 0 {
            self.lock.state = -1;
            Poll::Ready(RwLockWriteGuard { lock: self.lock })
        } else {
            self.lock.write_waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Read guard for RwLock
pub struct RwLockReadGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Deref for RwLockReadGuard<'a, T> {
    type Target = T

    fn deref(&self) -> &T {
        &self.lock.data
    }
}

impl<'a, T> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.state -= 1;

        // If no more readers, wake a waiting writer
        if self.lock.state == 0 {
            if let Some(waker) = self.lock.write_waiters.pop() {
                waker.wake();
            }
        }
    }
}

/// Write guard for RwLock
pub struct RwLockWriteGuard<'a, T> {
    lock: &'a RwLock<T>,
}

impl<'a, T> Deref for RwLockWriteGuard<'a, T> {
    type Target = T

    fn deref(&self) -> &T {
        &self.lock.data
    }
}

impl<'a, T> DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.lock.data
    }
}

impl<'a, T> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        self.lock.state = 0;

        // Wake waiting readers or a writer
        if !self.lock.read_waiters.is_empty() {
            for waker in self.lock.read_waiters.drain(..) {
                waker.wake();
            }
        } else if let Some(waker) = self.lock.write_waiters.pop() {
            waker.wake();
        }
    }
}

// =============================================================================
// Semaphore - Counting semaphore
// =============================================================================

/// An async counting semaphore
///
/// Limits the number of concurrent accesses to a resource.
///
/// # Example
/// ```
/// async fn example() {
///     // Allow at most 3 concurrent connections
///     let sem = Semaphore::new(3)
///
///     let permit = sem.acquire().await
///     // Do work...
///     // Permit is released when dropped
/// }
/// ```
pub struct Semaphore {
    /// Number of available permits
    permits: usize,
    /// Maximum permits (for closed semaphore detection)
    max_permits: usize,
    /// Waiters for permits
    waiters: Vec<(usize, Waker)>,  // (permits_needed, waker)
    /// Whether the semaphore is closed
    closed: bool,
}

impl Semaphore {
    /// Creates a new semaphore with the given number of permits
    pub fn new(permits: usize) -> Semaphore {
        Semaphore {
            permits,
            max_permits: permits,
            waiters: Vec::new(),
            closed: false,
        }
    }

    /// Acquires a single permit
    pub async fn acquire(&self) -> SemaphorePermit with Async {
        self.acquire_many(1).await
    }

    /// Acquires multiple permits
    pub async fn acquire_many(&self, permits: usize) -> SemaphorePermit with Async {
        AcquireFuture { semaphore: self, permits }.await
    }

    /// Tries to acquire a permit immediately
    pub fn try_acquire(&self) -> Option<SemaphorePermit> {
        self.try_acquire_many(1)
    }

    /// Tries to acquire multiple permits immediately
    pub fn try_acquire_many(&self, permits: usize) -> Option<SemaphorePermit> {
        if !self.closed && self.permits >= permits {
            self.permits -= permits;
            Some(SemaphorePermit { semaphore: self, permits })
        } else {
            None
        }
    }

    /// Returns the number of available permits
    pub fn available_permits(&self) -> usize {
        self.permits
    }

    /// Adds permits to the semaphore
    pub fn add_permits(&self, n: usize) {
        self.permits += n;
        self.wake_waiters();
    }

    /// Closes the semaphore
    ///
    /// All pending acquires will fail.
    pub fn close(&self) {
        self.closed = true;
        // Wake all waiters so they can observe closure
        for (_, waker) in self.waiters.drain(..) {
            waker.wake();
        }
    }

    /// Returns true if the semaphore is closed
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    fn wake_waiters(&self) {
        // Wake waiters that can now proceed
        let mut i = 0;
        while i < self.waiters.len() {
            let (needed, _) = &self.waiters[i];
            if self.permits >= *needed {
                let (_, waker) = self.waiters.remove(i);
                waker.wake();
            } else {
                i += 1;
            }
        }
    }
}

struct AcquireFuture<'a> {
    semaphore: &'a Semaphore,
    permits: usize,
}

impl<'a> Future for AcquireFuture<'a> {
    type Output = SemaphorePermit<'a>

    fn poll(&mut self, cx: &mut Context) -> Poll<SemaphorePermit<'a>> {
        if self.semaphore.closed {
            panic("Semaphore closed");
        }

        if self.semaphore.permits >= self.permits {
            self.semaphore.permits -= self.permits;
            Poll::Ready(SemaphorePermit {
                semaphore: self.semaphore,
                permits: self.permits,
            })
        } else {
            self.semaphore.waiters.push((self.permits, cx.waker().clone()));
            Poll::Pending
        }
    }
}

/// Permit from a semaphore
pub struct SemaphorePermit<'a> {
    semaphore: &'a Semaphore,
    permits: usize,
}

impl<'a> SemaphorePermit<'a> {
    /// Forgets the permit without releasing it
    pub fn forget(self) {
        std::mem::forget(self);
    }
}

impl<'a> Drop for SemaphorePermit<'a> {
    fn drop(&mut self) {
        self.semaphore.permits += self.permits;
        self.semaphore.wake_waiters();
    }
}

/// Owned permit that doesn't borrow the semaphore
pub struct OwnedSemaphorePermit {
    semaphore: &Semaphore,  // Would be Arc in real impl
    permits: usize,
}

// =============================================================================
// Barrier - Synchronization barrier
// =============================================================================

/// An async barrier for synchronizing multiple tasks
///
/// All tasks must wait at the barrier before any can proceed.
///
/// # Example
/// ```
/// async fn example() {
///     let barrier = Barrier::new(3)
///
///     // Spawn 3 tasks
///     for i in 0..3 {
///         let b = barrier.clone()
///         spawn async move {
///             println("Task {} waiting", i)
///             b.wait().await
///             println("Task {} continuing", i)
///         }
///     }
/// }
/// ```
pub struct Barrier {
    /// Number of parties
    parties: usize,
    /// Current count of waiting tasks
    count: usize,
    /// Generation (increments each time barrier releases)
    generation: u64,
    /// Waiters
    waiters: Vec<Waker>,
}

impl Barrier {
    /// Creates a new barrier for the given number of parties
    pub fn new(parties: usize) -> Barrier {
        Barrier {
            parties,
            count: 0,
            generation: 0,
            waiters: Vec::new(),
        }
    }

    /// Waits at the barrier
    ///
    /// Returns a BarrierWaitResult. The "leader" is the last task to arrive.
    pub async fn wait(&self) -> BarrierWaitResult with Async {
        WaitFuture { barrier: self, generation: self.generation }.await
    }
}

struct WaitFuture<'a> {
    barrier: &'a Barrier,
    generation: u64,
}

impl<'a> Future for WaitFuture<'a> {
    type Output = BarrierWaitResult

    fn poll(&mut self, cx: &mut Context) -> Poll<BarrierWaitResult> {
        // Check if we've already been released
        if self.barrier.generation != self.generation {
            return Poll::Ready(BarrierWaitResult { is_leader: false });
        }

        self.barrier.count += 1;

        if self.barrier.count >= self.barrier.parties {
            // We're the leader - release everyone
            self.barrier.count = 0;
            self.barrier.generation += 1;

            // Wake all waiters
            for waker in self.barrier.waiters.drain(..) {
                waker.wake();
            }

            Poll::Ready(BarrierWaitResult { is_leader: true })
        } else {
            self.barrier.waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// Result from waiting on a barrier
pub struct BarrierWaitResult {
    is_leader: bool,
}

impl BarrierWaitResult {
    /// Returns true if this was the last task to reach the barrier
    pub fn is_leader(&self) -> bool {
        self.is_leader
    }
}

// =============================================================================
// Notify - Task notification
// =============================================================================

/// A notification mechanism for async tasks
///
/// Allows one task to notify others that something has occurred.
///
/// # Example
/// ```
/// async fn example() {
///     let notify = Notify::new()
///
///     // Waiter task
///     let n = notify.clone()
///     spawn async move {
///         n.notified().await
///         println("Notified!")
///     }
///
///     // Notifier
///     notify.notify_one()
/// }
/// ```
pub struct Notify {
    /// Number of pending notifications
    pending: usize,
    /// Waiting tasks
    waiters: Vec<Waker>,
}

impl Notify {
    /// Creates a new Notify
    pub fn new() -> Notify {
        Notify {
            pending: 0,
            waiters: Vec::new(),
        }
    }

    /// Waits for a notification
    pub async fn notified(&self) with Async {
        NotifiedFuture { notify: self }.await
    }

    /// Notifies one waiting task
    pub fn notify_one(&self) {
        if let Some(waker) = self.waiters.pop() {
            waker.wake();
        } else {
            self.pending += 1;
        }
    }

    /// Notifies all waiting tasks
    pub fn notify_all(&self) {
        for waker in self.waiters.drain(..) {
            waker.wake();
        }
    }

    /// Notifies waiters (up to a maximum)
    pub fn notify_waiters(&self, max: usize) {
        let count = max.min(self.waiters.len());
        for waker in self.waiters.drain(..count) {
            waker.wake();
        }
    }
}

struct NotifiedFuture<'a> {
    notify: &'a Notify,
}

impl<'a> Future for NotifiedFuture<'a> {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if self.notify.pending > 0 {
            self.notify.pending -= 1;
            Poll::Ready(())
        } else {
            self.notify.waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

// =============================================================================
// OnceCell - Async lazy initialization
// =============================================================================

/// An async once-cell for lazy initialization
///
/// The value is initialized exactly once, even with concurrent access.
///
/// # Example
/// ```
/// async fn example() {
///     let cell = OnceCell::new()
///
///     let value = cell.get_or_init(|| async {
///         expensive_computation().await
///     }).await
/// }
/// ```
pub struct OnceCell<T> {
    value: Option<T>,
    initialized: bool,
    initializing: bool,
    waiters: Vec<Waker>,
}

impl<T> OnceCell<T> {
    /// Creates a new empty OnceCell
    pub fn new() -> OnceCell<T> {
        OnceCell {
            value: None,
            initialized: false,
            initializing: false,
            waiters: Vec::new(),
        }
    }

    /// Creates a OnceCell with a value already set
    pub fn with_value(value: T) -> OnceCell<T> {
        OnceCell {
            value: Some(value),
            initialized: true,
            initializing: false,
            waiters: Vec::new(),
        }
    }

    /// Gets the value if initialized
    pub fn get(&self) -> Option<&T> {
        if self.initialized {
            self.value.as_ref()
        } else {
            None
        }
    }

    /// Sets the value if not already initialized
    pub fn set(&self, value: T) -> Result<(), T> {
        if self.initialized || self.initializing {
            Err(value)
        } else {
            self.value = Some(value);
            self.initialized = true;
            // Wake waiters
            for waker in self.waiters.drain(..) {
                waker.wake();
            }
            Ok(())
        }
    }

    /// Gets the value, initializing it if necessary
    pub async fn get_or_init<F, Fut>(&self, init: F) -> &T with Async
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = T>,
    {
        if self.initialized {
            return self.value.as_ref().unwrap();
        }

        if self.initializing {
            // Someone else is initializing, wait
            WaitInitFuture { cell: self }.await;
            return self.value.as_ref().unwrap();
        }

        // We'll initialize
        self.initializing = true;
        let value = init().await;
        self.value = Some(value);
        self.initialized = true;
        self.initializing = false;

        // Wake waiters
        for waker in self.waiters.drain(..) {
            waker.wake();
        }

        self.value.as_ref().unwrap()
    }

    /// Gets mutable access to the value
    pub fn get_mut(&mut self) -> Option<&mut T> {
        if self.initialized {
            self.value.as_mut()
        } else {
            None
        }
    }

    /// Takes the value out of the cell
    pub fn take(&mut self) -> Option<T> {
        if self.initialized {
            self.initialized = false;
            self.value.take()
        } else {
            None
        }
    }
}

struct WaitInitFuture<'a, T> {
    cell: &'a OnceCell<T>,
}

impl<'a, T> Future for WaitInitFuture<'a, T> {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        if self.cell.initialized {
            Poll::Ready(())
        } else {
            self.cell.waiters.push(cx.waker().clone());
            Poll::Pending
        }
    }
}

// =============================================================================
// Helper Traits
// =============================================================================

/// Deref trait (placeholder)
trait Deref {
    type Target
    fn deref(&self) -> &Self::Target;
}

/// DerefMut trait (placeholder)
trait DerefMut: Deref {
    fn deref_mut(&mut self) -> &mut Self::Target;
}
