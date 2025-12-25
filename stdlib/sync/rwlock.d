//! Reader-writer lock with RwLock<T>
//!
//! RwLock<T> allows multiple readers or a single writer at a time.

use std::sync::atomic::{AtomicInt, Ordering}
use std::ops::{Deref, DerefMut}
use std::fmt::{Debug, Formatter, FmtError}
use std::default::Default

/// A reader-writer lock.
///
/// This type of lock allows a number of readers or at most one writer
/// at any point in time. The write portion of this lock typically allows
/// modification of the underlying data, and the read portion typically
/// allows for read-only access.
///
/// The priority policy of this lock is unspecified. In particular,
/// a writer that is blocked on a lock will not prevent readers from
/// acquiring the lock.
///
/// # Examples
///
/// ```d
/// let lock = RwLock::new(42)
///
/// // Many readers can hold the lock at once
/// {
///     let r1 = lock.read()
///     let r2 = lock.read()
///     assert_eq(*r1, *r2)
/// }
///
/// // Only one writer can hold the lock
/// {
///     let mut w = lock.write()
///     *w = 100
/// }
/// ```
pub struct RwLock<T> {
    /// State: 0 = unlocked, >0 = number of readers, -1 = writer
    state: AtomicInt,
    value: T,
}

const UNLOCKED: int = 0
const WRITER: int = -1

unsafe impl<T> Send for RwLock<T> where T: Send {}
unsafe impl<T> Sync for RwLock<T> where T: Send + Sync {}

impl<T> RwLock<T> {
    /// Creates a new RwLock in an unlocked state.
    ///
    /// # Examples
    ///
    /// ```d
    /// let lock = RwLock::new(42)
    /// ```
    pub fn new(value: T) -> RwLock<T> {
        RwLock {
            state: AtomicInt::new(UNLOCKED),
            value,
        }
    }

    /// Locks this RwLock with shared read access, blocking the current thread
    /// until it can be acquired.
    ///
    /// The calling thread will be blocked until there are no more writers
    /// holding the lock. There may be other readers currently inside the lock
    /// when this method returns.
    ///
    /// Returns an RAII guard which will release this thread's shared access
    /// once it is dropped.
    pub fn read(self: &RwLock<T>) -> RwLockReadGuard<T> {
        loop {
            let state = self.state.load(Ordering::Relaxed)

            // Can acquire if not write-locked
            if state >= 0 {
                if self.state.compare_exchange_weak(
                    state, state + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed
                ).is_ok() {
                    return RwLockReadGuard { lock: self }
                }
            }

            std::hint::spin_loop()
        }
    }

    /// Attempts to acquire this RwLock with shared read access.
    ///
    /// If the access could not be granted at this time, then None is returned.
    /// Otherwise, an RAII guard is returned which will release the shared access
    /// when it is dropped.
    pub fn try_read(self: &RwLock<T>) -> Option<RwLockReadGuard<T>> {
        let state = self.state.load(Ordering::Relaxed)

        if state >= 0 {
            if self.state.compare_exchange(
                state, state + 1,
                Ordering::Acquire,
                Ordering::Relaxed
            ).is_ok() {
                return Option::Some(RwLockReadGuard { lock: self })
            }
        }

        Option::None
    }

    /// Locks this RwLock with exclusive write access, blocking the current
    /// thread until it can be acquired.
    ///
    /// This function will not return while other writers or readers
    /// currently have access to the lock.
    ///
    /// Returns an RAII guard which will drop the write access of this RwLock
    /// when dropped.
    pub fn write(self: &RwLock<T>) -> RwLockWriteGuard<T> {
        loop {
            if self.state.compare_exchange_weak(
                UNLOCKED, WRITER,
                Ordering::Acquire,
                Ordering::Relaxed
            ).is_ok() {
                return RwLockWriteGuard { lock: self }
            }

            std::hint::spin_loop()
        }
    }

    /// Attempts to lock this RwLock with exclusive write access.
    ///
    /// If the lock could not be acquired at this time, then None is returned.
    /// Otherwise, an RAII guard is returned which will release the lock when
    /// it is dropped.
    pub fn try_write(self: &RwLock<T>) -> Option<RwLockWriteGuard<T>> {
        if self.state.compare_exchange(
            UNLOCKED, WRITER,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_ok() {
            Option::Some(RwLockWriteGuard { lock: self })
        } else {
            Option::None
        }
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this requires &! access to the RwLock, no locking is needed.
    pub fn get_mut(self: &!RwLock<T>) -> &!T {
        &!self.value
    }

    /// Consumes this RwLock, returning the underlying data.
    pub fn into_inner(self: RwLock<T>) -> T {
        self.value
    }

    /// Returns whether the lock is currently held by a writer.
    pub fn is_write_locked(self: &RwLock<T>) -> bool {
        self.state.load(Ordering::Relaxed) == WRITER
    }

    /// Returns whether the lock is currently held by any readers.
    pub fn is_read_locked(self: &RwLock<T>) -> bool {
        self.state.load(Ordering::Relaxed) > 0
    }
}

impl<T> Default for RwLock<T>
where T: Default
{
    fn default() -> RwLock<T> {
        RwLock::new(T::default())
    }
}

impl<T> Debug for RwLock<T>
where T: Debug
{
    fn fmt(self: &RwLock<T>, f: &!Formatter) -> Result<unit, FmtError> {
        match self.try_read() {
            Option::Some(guard) => f.debug_struct("RwLock")
                .field("data", &*guard)
                .finish(),
            Option::None => f.debug_struct("RwLock")
                .field("data", &"<locked>")
                .finish(),
        }
    }
}

impl<T> From<T> for RwLock<T> {
    fn from(value: T) -> RwLock<T> {
        RwLock::new(value)
    }
}

/// RAII structure used to release the shared read access of a lock when dropped.
pub struct RwLockReadGuard<T> {
    lock: &RwLock<T>,
}

impl<T> Deref for RwLockReadGuard<T> {
    type Target = T

    fn deref(self: &RwLockReadGuard<T>) -> &T {
        unsafe { &*(&self.lock.value as *const T) }
    }
}

impl<T> Drop for RwLockReadGuard<T> {
    fn drop(self: &!RwLockReadGuard<T>) {
        self.lock.state.fetch_sub(1, Ordering::Release)
    }
}

impl<T> Debug for RwLockReadGuard<T>
where T: Debug
{
    fn fmt(self: &RwLockReadGuard<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

// Read guards are !Send
impl<T> !Send for RwLockReadGuard<T> {}

/// RAII structure used to release the exclusive write access of a lock when dropped.
pub struct RwLockWriteGuard<T> {
    lock: &RwLock<T>,
}

impl<T> Deref for RwLockWriteGuard<T> {
    type Target = T

    fn deref(self: &RwLockWriteGuard<T>) -> &T {
        unsafe { &*(&self.lock.value as *const T) }
    }
}

impl<T> DerefMut for RwLockWriteGuard<T> {
    fn deref_mut(self: &!RwLockWriteGuard<T>) -> &!T {
        unsafe { &!*(&self.lock.value as *const T as *mut T) }
    }
}

impl<T> Drop for RwLockWriteGuard<T> {
    fn drop(self: &!RwLockWriteGuard<T>) {
        self.lock.state.store(UNLOCKED, Ordering::Release)
    }
}

impl<T> Debug for RwLockWriteGuard<T>
where T: Debug
{
    fn fmt(self: &RwLockWriteGuard<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

// Write guards are !Send
impl<T> !Send for RwLockWriteGuard<T> {}

// Unit tests
#[test]
fn test_rwlock_read() {
    let lock = RwLock::new(42)
    let r1 = lock.read()
    let r2 = lock.read()
    assert_eq(*r1, 42)
    assert_eq(*r2, 42)
}

#[test]
fn test_rwlock_write() {
    let lock = RwLock::new(0)
    {
        let mut w = lock.write()
        *w = 42
    }
    {
        let r = lock.read()
        assert_eq(*r, 42)
    }
}

#[test]
fn test_rwlock_try_read() {
    let lock = RwLock::new(42)

    // Multiple try_reads should succeed
    let r1 = lock.try_read()
    assert(r1.is_some())

    let r2 = lock.try_read()
    assert(r2.is_some())
}

#[test]
fn test_rwlock_try_write_fails_when_reading() {
    let lock = RwLock::new(42)

    let _r = lock.read()

    // try_write should fail while there's a reader
    let w = lock.try_write()
    assert(w.is_none())
}
