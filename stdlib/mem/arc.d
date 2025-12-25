//! Thread-safe reference counting with Arc<T>
//!
//! Arc<T> provides shared ownership of a value of type T,
//! allocated in the heap, with atomic reference counting.

use std::ops::Deref
use std::sync::atomic::{AtomicInt, Ordering}
use std::cmp::{Eq, Ord, Ordering as CmpOrdering}
use std::fmt::{Display, Debug, Formatter, FmtError}
use std::clone::Clone
use std::default::Default

/// A thread-safe reference-counting pointer.
///
/// Arc stands for "Atomically Reference Counted".
///
/// The type Arc<T> provides shared ownership of a value of type T,
/// allocated in the heap. Invoking clone on Arc produces a new Arc
/// instance, which points to the same allocation on the heap as the
/// source Arc, while increasing a reference count.
///
/// When the last Arc pointer to a given allocation is destroyed,
/// the value stored in that allocation is also dropped.
///
/// # Thread Safety
///
/// Unlike Rc<T>, Arc<T> uses atomic operations for its reference
/// counting. This means that it is thread-safe.
///
/// # Examples
///
/// ```d
/// let arc = Arc::new(42)
/// let arc2 = arc.clone()
///
/// // Send to another thread
/// spawn(fn() {
///     println(*arc2)
/// })
/// ```
pub struct Arc<T> {
    ptr: *mut ArcInner<T>,
}

struct ArcInner<T> {
    strong: AtomicInt,
    weak: AtomicInt,
    value: T,
}

// Arc is Send + Sync if T is Send + Sync
unsafe impl<T> Send for Arc<T> where T: Send + Sync {}
unsafe impl<T> Sync for Arc<T> where T: Send + Sync {}

impl<T> Arc<T> {
    /// Constructs a new Arc<T>.
    ///
    /// # Examples
    ///
    /// ```d
    /// let five = Arc::new(5)
    /// ```
    pub fn new(value: T) -> Arc<T> with Alloc {
        let ptr = alloc::alloc::<ArcInner<T>>(1)
        unsafe {
            ptr::write(ptr, ArcInner {
                strong: AtomicInt::new(1),
                weak: AtomicInt::new(1),
                value,
            })
        }
        Arc { ptr }
    }

    /// Gets the number of strong references to this allocation.
    pub fn strong_count(this: &Arc<T>) -> int {
        unsafe { (*this.ptr).strong.load(Ordering::Acquire) }
    }

    /// Gets the number of weak references to this allocation.
    pub fn weak_count(this: &Arc<T>) -> int {
        unsafe { (*this.ptr).weak.load(Ordering::Acquire) - 1 }
    }

    /// Returns a mutable reference if there is exactly one strong reference.
    pub fn get_mut(this: &!Arc<T>) -> Option<&!T> {
        if Arc::strong_count(this) == 1 {
            // Fence to ensure we see all writes before we mutate
            std::sync::atomic::fence(Ordering::Acquire)
            unsafe { Option::Some(&!(*this.ptr).value) }
        } else {
            Option::None
        }
    }

    /// Makes a mutable reference, cloning if necessary.
    ///
    /// If there are other Arc pointers to the same allocation, then
    /// make_mut will clone the inner value to ensure unique ownership.
    pub fn make_mut(this: &!Arc<T>) -> &!T with Alloc
    where T: Clone
    {
        if Arc::strong_count(this) != 1 {
            let new_arc = Arc::new((**this).clone())
            *this = new_arc
        }

        unsafe { &!(*this.ptr).value }
    }

    /// Attempts to unwrap the Arc, returning the inner value if there
    /// is exactly one strong reference.
    pub fn try_unwrap(this: Arc<T>) -> Result<T, Arc<T>> {
        // Try to set strong count to 0
        if unsafe { (*this.ptr).strong.compare_exchange(
            1, 0,
            Ordering::Release,
            Ordering::Relaxed
        ).is_ok() } {
            // Success! We're the only strong reference
            std::sync::atomic::fence(Ordering::Acquire)

            unsafe {
                let val = ptr::read(&(*this.ptr).value)

                // Decrement weak count
                if (*this.ptr).weak.fetch_sub(1, Ordering::Release) == 1 {
                    std::sync::atomic::fence(Ordering::Acquire)
                    alloc::dealloc(this.ptr, 1)
                }

                std::mem::forget(this)
                Result::Ok(val)
            }
        } else {
            Result::Err(this)
        }
    }

    /// Creates a new weak reference.
    pub fn downgrade(this: &Arc<T>) -> Weak<T> {
        unsafe {
            (*this.ptr).weak.fetch_add(1, Ordering::Relaxed)
        }
        Weak { ptr: this.ptr }
    }

    /// Returns true if two Arcs point to the same allocation.
    pub fn ptr_eq(this: &Arc<T>, other: &Arc<T>) -> bool {
        this.ptr == other.ptr
    }
}

impl<T> Deref for Arc<T> {
    type Target = T

    fn deref(self: &Arc<T>) -> &T {
        unsafe { &(*self.ptr).value }
    }
}

impl<T> Clone for Arc<T> {
    fn clone(self: &Arc<T>) -> Arc<T> {
        // Using Relaxed is safe because we're just incrementing the count
        // The actual synchronization happens through the data access
        unsafe {
            (*self.ptr).strong.fetch_add(1, Ordering::Relaxed)
        }
        Arc { ptr: self.ptr }
    }
}

impl<T> Drop for Arc<T> {
    fn drop(self: &!Arc<T>) with Alloc {
        unsafe {
            // Decrement strong count
            if (*self.ptr).strong.fetch_sub(1, Ordering::Release) == 1 {
                // Last strong reference
                std::sync::atomic::fence(Ordering::Acquire)

                // Drop the value
                ptr::drop_in_place(&!(*self.ptr).value)

                // Decrement weak count
                if (*self.ptr).weak.fetch_sub(1, Ordering::Release) == 1 {
                    std::sync::atomic::fence(Ordering::Acquire)
                    alloc::dealloc(self.ptr, 1)
                }
            }
        }
    }
}

impl<T> Eq for Arc<T>
where T: Eq
{
    fn eq(self: &Arc<T>, other: &Arc<T>) -> bool {
        **self == **other
    }
}

impl<T> Ord for Arc<T>
where T: Ord
{
    fn cmp(self: &Arc<T>, other: &Arc<T>) -> CmpOrdering {
        (**self).cmp(&**other)
    }
}

impl<T> Display for Arc<T>
where T: Display
{
    fn fmt(self: &Arc<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Debug for Arc<T>
where T: Debug
{
    fn fmt(self: &Arc<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Default for Arc<T>
where T: Default
{
    fn default() -> Arc<T> with Alloc {
        Arc::new(T::default())
    }
}

impl<T> From<T> for Arc<T> {
    fn from(x: T) -> Arc<T> with Alloc {
        Arc::new(x)
    }
}

/// A weak reference to an Arc-managed allocation.
///
/// Weak pointers do not count towards determining if the inner value
/// should be dropped. A Weak<T> can be upgraded to an Arc<T>, but
/// will return None if the value has already been dropped.
pub struct Weak<T> {
    ptr: *mut ArcInner<T>,
}

unsafe impl<T> Send for Weak<T> where T: Send + Sync {}
unsafe impl<T> Sync for Weak<T> where T: Send + Sync {}

impl<T> Weak<T> {
    /// Creates a new Weak pointer that doesn't point to any allocation.
    pub fn new() -> Weak<T> {
        Weak { ptr: null_mut() }
    }

    /// Attempts to upgrade the Weak pointer to an Arc.
    ///
    /// Returns None if the inner value has since been dropped.
    pub fn upgrade(self: &Weak<T>) -> Option<Arc<T>> {
        if self.ptr == null_mut() {
            return Option::None
        }

        unsafe {
            // Use a CAS loop to increment the strong count
            loop {
                let strong = (*self.ptr).strong.load(Ordering::Relaxed)
                if strong == 0 {
                    return Option::None
                }

                if (*self.ptr).strong.compare_exchange_weak(
                    strong, strong + 1,
                    Ordering::Acquire,
                    Ordering::Relaxed
                ).is_ok() {
                    return Option::Some(Arc { ptr: self.ptr })
                }
            }
        }
    }

    /// Gets the number of strong references pointing to this allocation.
    pub fn strong_count(self: &Weak<T>) -> int {
        if self.ptr == null_mut() { 0 }
        else { unsafe { (*self.ptr).strong.load(Ordering::Relaxed) } }
    }

    /// Gets the number of weak references pointing to this allocation.
    pub fn weak_count(self: &Weak<T>) -> int {
        if self.ptr == null_mut() { 0 }
        else { unsafe { (*self.ptr).weak.load(Ordering::Relaxed) - 1 } }
    }
}

impl<T> Clone for Weak<T> {
    fn clone(self: &Weak<T>) -> Weak<T> {
        if self.ptr != null_mut() {
            unsafe {
                (*self.ptr).weak.fetch_add(1, Ordering::Relaxed)
            }
        }
        Weak { ptr: self.ptr }
    }
}

impl<T> Drop for Weak<T> {
    fn drop(self: &!Weak<T>) with Alloc {
        if self.ptr != null_mut() {
            unsafe {
                if (*self.ptr).weak.fetch_sub(1, Ordering::Release) == 1 {
                    std::sync::atomic::fence(Ordering::Acquire)
                    alloc::dealloc(self.ptr, 1)
                }
            }
        }
    }
}

impl<T> Default for Weak<T> {
    fn default() -> Weak<T> {
        Weak::new()
    }
}

// Unit tests
#[test]
fn test_arc_basic() {
    let arc = Arc::new(42)
    assert_eq(*arc, 42)
    assert_eq(Arc::strong_count(&arc), 1)
}

#[test]
fn test_arc_clone() {
    let arc1 = Arc::new(42)
    let arc2 = arc1.clone()
    assert_eq(Arc::strong_count(&arc1), 2)
    assert(Arc::ptr_eq(&arc1, &arc2))
}

#[test]
fn test_arc_weak() {
    let arc = Arc::new(42)
    let weak = Arc::downgrade(&arc)

    assert(weak.upgrade().is_some())
    drop(arc)
    assert(weak.upgrade().is_none())
}
