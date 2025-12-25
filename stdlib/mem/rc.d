//! Single-threaded reference counting with Rc<T>
//!
//! Rc<T> provides shared ownership of a value of type T,
//! allocated in the heap.

use std::ops::Deref
use std::cmp::{Eq, Ord, Ordering}
use std::fmt::{Display, Debug, Formatter, FmtError}
use std::clone::Clone
use std::default::Default

/// A single-threaded reference-counting pointer.
///
/// The inherent methods of Rc are all associated functions, which means
/// that you have to call them as e.g., Rc::strong_count(&rc) instead
/// of rc.strong_count(). This avoids conflicts with methods of the
/// inner type T.
///
/// # Examples
///
/// ```d
/// let rc = Rc::new(42)
/// let rc2 = rc.clone()
/// assert_eq(Rc::strong_count(&rc), 2)
/// ```
pub struct Rc<T> {
    ptr: *mut RcBox<T>,
}

struct RcBox<T> {
    strong: Cell<int>,
    weak: Cell<int>,
    value: T,
}

impl<T> Rc<T> {
    /// Constructs a new Rc<T>.
    ///
    /// # Examples
    ///
    /// ```d
    /// let five = Rc::new(5)
    /// ```
    pub fn new(value: T) -> Rc<T> with Alloc {
        let ptr = alloc::alloc::<RcBox<T>>(1)
        unsafe {
            ptr::write(ptr, RcBox {
                strong: Cell::new(1),
                weak: Cell::new(1),
                value,
            })
        }
        Rc { ptr }
    }

    /// Gets the number of strong references to this allocation.
    ///
    /// # Examples
    ///
    /// ```d
    /// let rc = Rc::new(42)
    /// let rc2 = rc.clone()
    /// assert_eq(Rc::strong_count(&rc), 2)
    /// ```
    pub fn strong_count(this: &Rc<T>) -> int {
        unsafe { (*this.ptr).strong.get() }
    }

    /// Gets the number of weak references to this allocation.
    pub fn weak_count(this: &Rc<T>) -> int {
        unsafe { (*this.ptr).weak.get() - 1 }
    }

    /// Returns a mutable reference if there is exactly one strong reference.
    ///
    /// Returns None otherwise, because it is not safe to mutate a shared value.
    pub fn get_mut(this: &!Rc<T>) -> Option<&!T> {
        if Rc::strong_count(this) == 1 && Rc::weak_count(this) == 0 {
            unsafe { Option::Some(&!(*this.ptr).value) }
        } else {
            Option::None
        }
    }

    /// Makes a mutable reference into the given Rc.
    ///
    /// If there are other Rc pointers to the same allocation, then
    /// make_mut will clone the inner value to a new allocation to
    /// ensure unique ownership.
    pub fn make_mut(this: &!Rc<T>) -> &!T with Alloc
    where T: Clone
    {
        if Rc::strong_count(this) != 1 {
            // Clone the data
            let new_rc = Rc::new((**this).clone())
            *this = new_rc
        }

        unsafe { &!(*this.ptr).value }
    }

    /// Attempts to unwrap the Rc, returning the inner value if there
    /// is exactly one strong reference.
    ///
    /// # Examples
    ///
    /// ```d
    /// let rc = Rc::new(42)
    /// let result = Rc::try_unwrap(rc)
    /// assert_eq(result, Result::Ok(42))
    /// ```
    pub fn try_unwrap(this: Rc<T>) -> Result<T, Rc<T>> {
        if Rc::strong_count(&this) == 1 {
            unsafe {
                let val = ptr::read(&(*this.ptr).value)

                // Decrement weak count
                let weak = (*this.ptr).weak.get() - 1
                (*this.ptr).weak.set(weak)

                if weak == 0 {
                    alloc::dealloc(this.ptr, 1)
                }

                std::mem::forget(this)
                Result::Ok(val)
            }
        } else {
            Result::Err(this)
        }
    }

    /// Creates a new weak reference to this allocation.
    ///
    /// # Examples
    ///
    /// ```d
    /// let rc = Rc::new(42)
    /// let weak = Rc::downgrade(&rc)
    /// ```
    pub fn downgrade(this: &Rc<T>) -> Weak<T> {
        unsafe {
            let weak = (*this.ptr).weak.get() + 1
            (*this.ptr).weak.set(weak)
        }
        Weak { ptr: this.ptr }
    }

    /// Returns true if two Rcs point to the same allocation.
    pub fn ptr_eq(this: &Rc<T>, other: &Rc<T>) -> bool {
        this.ptr == other.ptr
    }
}

impl<T> Deref for Rc<T> {
    type Target = T

    fn deref(self: &Rc<T>) -> &T {
        unsafe { &(*self.ptr).value }
    }
}

impl<T> Clone for Rc<T> {
    fn clone(self: &Rc<T>) -> Rc<T> {
        unsafe {
            let strong = (*self.ptr).strong.get() + 1
            (*self.ptr).strong.set(strong)
        }
        Rc { ptr: self.ptr }
    }
}

impl<T> Drop for Rc<T> {
    fn drop(self: &!Rc<T>) with Alloc {
        unsafe {
            let strong = (*self.ptr).strong.get() - 1
            (*self.ptr).strong.set(strong)

            if strong == 0 {
                // Drop the value
                ptr::drop_in_place(&!(*self.ptr).value)

                // Decrement weak count
                let weak = (*self.ptr).weak.get() - 1
                (*self.ptr).weak.set(weak)

                if weak == 0 {
                    alloc::dealloc(self.ptr, 1)
                }
            }
        }
    }
}

impl<T> Eq for Rc<T>
where T: Eq
{
    fn eq(self: &Rc<T>, other: &Rc<T>) -> bool {
        **self == **other
    }
}

impl<T> Ord for Rc<T>
where T: Ord
{
    fn cmp(self: &Rc<T>, other: &Rc<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T> Display for Rc<T>
where T: Display
{
    fn fmt(self: &Rc<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Debug for Rc<T>
where T: Debug
{
    fn fmt(self: &Rc<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Default for Rc<T>
where T: Default
{
    fn default() -> Rc<T> with Alloc {
        Rc::new(T::default())
    }
}

impl<T> From<T> for Rc<T> {
    fn from(x: T) -> Rc<T> with Alloc {
        Rc::new(x)
    }
}

/// A weak reference to an Rc-managed allocation.
///
/// Weak references do not count towards determining if the inner value
/// should be dropped.
///
/// # Examples
///
/// ```d
/// let rc = Rc::new(42)
/// let weak = Rc::downgrade(&rc)
///
/// let upgraded = weak.upgrade()
/// assert(upgraded.is_some())
/// ```
pub struct Weak<T> {
    ptr: *mut RcBox<T>,
}

impl<T> Weak<T> {
    /// Creates a new Weak pointer that doesn't point to any allocation.
    pub fn new() -> Weak<T> {
        Weak { ptr: null_mut() }
    }

    /// Attempts to upgrade the Weak pointer to an Rc.
    ///
    /// Returns None if the inner value has since been dropped.
    pub fn upgrade(self: &Weak<T>) -> Option<Rc<T>> {
        if self.ptr == null_mut() {
            return Option::None
        }

        unsafe {
            let strong = (*self.ptr).strong.get()
            if strong == 0 {
                return Option::None
            }

            (*self.ptr).strong.set(strong + 1)
            Option::Some(Rc { ptr: self.ptr })
        }
    }

    /// Gets the number of strong references pointing to this allocation.
    pub fn strong_count(self: &Weak<T>) -> int {
        if self.ptr == null_mut() { 0 }
        else { unsafe { (*self.ptr).strong.get() } }
    }

    /// Gets the number of weak references pointing to this allocation.
    pub fn weak_count(self: &Weak<T>) -> int {
        if self.ptr == null_mut() { 0 }
        else { unsafe { (*self.ptr).weak.get() - 1 } }
    }
}

impl<T> Clone for Weak<T> {
    fn clone(self: &Weak<T>) -> Weak<T> {
        if self.ptr != null_mut() {
            unsafe {
                let weak = (*self.ptr).weak.get() + 1
                (*self.ptr).weak.set(weak)
            }
        }
        Weak { ptr: self.ptr }
    }
}

impl<T> Drop for Weak<T> {
    fn drop(self: &!Weak<T>) with Alloc {
        if self.ptr != null_mut() {
            unsafe {
                let weak = (*self.ptr).weak.get() - 1
                (*self.ptr).weak.set(weak)

                if weak == 0 {
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
fn test_rc_basic() {
    let rc = Rc::new(42)
    assert_eq(*rc, 42)
    assert_eq(Rc::strong_count(&rc), 1)
}

#[test]
fn test_rc_clone() {
    let rc1 = Rc::new(42)
    let rc2 = rc1.clone()
    assert_eq(Rc::strong_count(&rc1), 2)
    assert_eq(Rc::strong_count(&rc2), 2)
    assert(Rc::ptr_eq(&rc1, &rc2))
}

#[test]
fn test_rc_weak() {
    let rc = Rc::new(42)
    let weak = Rc::downgrade(&rc)

    assert_eq(Rc::weak_count(&rc), 1)
    assert(weak.upgrade().is_some())
}
