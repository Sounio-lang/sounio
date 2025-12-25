//! Mutual exclusion with Mutex<T>
//!
//! Mutex<T> provides mutual exclusion, ensuring only one thread
//! can access the data at a time.

use std::sync::atomic::{AtomicBool, Ordering}
use std::ops::{Deref, DerefMut}
use std::fmt::{Debug, Formatter, FmtError}
use std::default::Default

/// A mutual exclusion primitive useful for protecting shared data.
///
/// This mutex will block threads waiting for the lock to become available.
/// The mutex can be statically initialized or created by the new constructor.
///
/// # Examples
///
/// ```d
/// let mutex = Mutex::new(0)
///
/// {
///     let mut guard = mutex.lock()
///     *guard = 10
/// }
///
/// assert_eq(*mutex.lock(), 10)
/// ```
pub struct Mutex<T> {
    locked: AtomicBool,
    value: T,
}

unsafe impl<T> Send for Mutex<T> where T: Send {}
unsafe impl<T> Sync for Mutex<T> where T: Send {}

impl<T> Mutex<T> {
    /// Creates a new mutex in an unlocked state.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mutex = Mutex::new(42)
    /// ```
    pub fn new(value: T) -> Mutex<T> {
        Mutex {
            locked: AtomicBool::new(false),
            value,
        }
    }

    /// Acquires the mutex, blocking the current thread until it is able to do so.
    ///
    /// This function will block the local thread until it is available to acquire
    /// the mutex. Upon returning, the thread is the only thread with the lock held.
    /// A RAII guard is returned to allow scoped unlock of the lock.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mutex = Mutex::new(42)
    /// let guard = mutex.lock()
    /// println(*guard)
    /// ```
    pub fn lock(self: &Mutex<T>) -> MutexGuard<T> {
        // Spin until we acquire the lock
        while self.locked.compare_exchange_weak(
            false, true,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_err() {
            // Spin hint - tells the CPU we're in a spin loop
            std::hint::spin_loop()
        }

        MutexGuard { mutex: self }
    }

    /// Attempts to acquire the mutex without blocking.
    ///
    /// If the lock could not be acquired at this time, then None is returned.
    /// Otherwise, a RAII guard is returned which will release the lock when dropped.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mutex = Mutex::new(42)
    /// match mutex.try_lock() {
    ///     Option::Some(guard) => println(*guard),
    ///     Option::None => println("Couldn't get lock"),
    /// }
    /// ```
    pub fn try_lock(self: &Mutex<T>) -> Option<MutexGuard<T>> {
        if self.locked.compare_exchange(
            false, true,
            Ordering::Acquire,
            Ordering::Relaxed
        ).is_ok() {
            Option::Some(MutexGuard { mutex: self })
        } else {
            Option::None
        }
    }

    /// Returns whether the mutex is currently locked.
    ///
    /// This method does not provide any synchronization guarantees and
    /// should only be used for heuristics or debugging.
    pub fn is_locked(self: &Mutex<T>) -> bool {
        self.locked.load(Ordering::Relaxed)
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// Since this requires &! access to the Mutex, no locking is needed.
    pub fn get_mut(self: &!Mutex<T>) -> &!T {
        &!self.value
    }

    /// Consumes the mutex, returning the underlying data.
    pub fn into_inner(self: Mutex<T>) -> T {
        self.value
    }
}

impl<T> Default for Mutex<T>
where T: Default
{
    fn default() -> Mutex<T> {
        Mutex::new(T::default())
    }
}

impl<T> Debug for Mutex<T>
where T: Debug
{
    fn fmt(self: &Mutex<T>, f: &!Formatter) -> Result<unit, FmtError> {
        match self.try_lock() {
            Option::Some(guard) => f.debug_struct("Mutex")
                .field("data", &*guard)
                .finish(),
            Option::None => f.debug_struct("Mutex")
                .field("data", &"<locked>")
                .finish(),
        }
    }
}

impl<T> From<T> for Mutex<T> {
    fn from(value: T) -> Mutex<T> {
        Mutex::new(value)
    }
}

/// RAII structure used to release the exclusive lock on a mutex when dropped.
///
/// This structure implements Deref and DerefMut, so you can access the
/// underlying data through it.
pub struct MutexGuard<T> {
    mutex: &Mutex<T>,
}

impl<T> Deref for MutexGuard<T> {
    type Target = T

    fn deref(self: &MutexGuard<T>) -> &T {
        unsafe { &*(&self.mutex.value as *const T) }
    }
}

impl<T> DerefMut for MutexGuard<T> {
    fn deref_mut(self: &!MutexGuard<T>) -> &!T {
        unsafe { &!*(&self.mutex.value as *const T as *mut T) }
    }
}

impl<T> Drop for MutexGuard<T> {
    fn drop(self: &!MutexGuard<T>) {
        self.mutex.locked.store(false, Ordering::Release)
    }
}

impl<T> Debug for MutexGuard<T>
where T: Debug
{
    fn fmt(self: &MutexGuard<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

// Mark MutexGuard as !Send to prevent sending guards between threads
// (the guard must be dropped on the same thread that acquired it)
impl<T> !Send for MutexGuard<T> {}

// Unit tests
#[test]
fn test_mutex_basic() {
    let mutex = Mutex::new(42)
    {
        let guard = mutex.lock()
        assert_eq(*guard, 42)
    }
}

#[test]
fn test_mutex_mutation() {
    let mutex = Mutex::new(0)
    {
        let mut guard = mutex.lock()
        *guard = 42
    }
    {
        let guard = mutex.lock()
        assert_eq(*guard, 42)
    }
}

#[test]
fn test_mutex_try_lock() {
    let mutex = Mutex::new(42)

    // First try_lock should succeed
    let guard = mutex.try_lock()
    assert(guard.is_some())

    // Second try_lock should fail (already locked)
    let guard2 = mutex.try_lock()
    assert(guard2.is_none())

    // After dropping, should be able to lock again
    drop(guard)
    let guard3 = mutex.try_lock()
    assert(guard3.is_some())
}
