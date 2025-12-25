//! Heap allocation with Box<T>
//!
//! Box<T> provides ownership of a heap-allocated value.

use std::ops::{Deref, DerefMut}
use std::cmp::{Eq, Ord, Ordering}
use std::fmt::{Display, Debug, Formatter, FmtError}
use std::clone::Clone
use std::default::Default

/// A pointer type that uniquely owns a heap allocation of type T.
///
/// Box<T> allocates memory on the heap and places the value there.
/// When the Box goes out of scope, the value is dropped and the
/// memory is deallocated.
///
/// # Examples
///
/// ```d
/// let b = Box::new(42)
/// println(*b)  // Prints: 42
/// ```
pub struct Box<T> {
    ptr: *mut T,
}

impl<T> Box<T> {
    /// Allocates memory on the heap and places x into it.
    ///
    /// # Examples
    ///
    /// ```d
    /// let x = Box::new(5)
    /// ```
    pub fn new(x: T) -> Box<T> with Alloc {
        let ptr = alloc::alloc::<T>(1)
        unsafe {
            ptr::write(ptr, x)
        }
        Box { ptr }
    }

    /// Constructs a box from a raw pointer.
    ///
    /// # Safety
    ///
    /// The raw pointer must have been previously returned by Box::into_raw.
    /// After calling this function, the raw pointer is owned by the
    /// resulting Box.
    pub unsafe fn from_raw(ptr: *mut T) -> Box<T> {
        Box { ptr }
    }

    /// Consumes the Box, returning a wrapped raw pointer.
    ///
    /// The pointer will be properly aligned and non-null.
    ///
    /// After calling this function, the caller is responsible for the
    /// memory previously managed by the Box. In particular, the caller
    /// should properly destroy T and release the memory.
    pub fn into_raw(b: Box<T>) -> *mut T {
        let ptr = b.ptr
        std::mem::forget(b)
        ptr
    }

    /// Consumes the Box, returning the wrapped value.
    ///
    /// # Examples
    ///
    /// ```d
    /// let b = Box::new(42)
    /// let x = Box::into_inner(b)
    /// assert_eq(x, 42)
    /// ```
    pub fn into_inner(b: Box<T>) -> T {
        unsafe {
            let val = ptr::read(b.ptr)
            alloc::dealloc(b.ptr, 1)
            std::mem::forget(b)
            val
        }
    }

    /// Returns a raw pointer to the boxed value.
    ///
    /// The caller must ensure that the Box outlives the pointer.
    pub fn as_ptr(self: &Box<T>) -> *const T {
        self.ptr
    }

    /// Returns a mutable raw pointer to the boxed value.
    ///
    /// The caller must ensure that the Box outlives the pointer.
    pub fn as_mut_ptr(self: &!Box<T>) -> *mut T {
        self.ptr
    }

    /// Leaks the box, returning a mutable reference with a 'static lifetime.
    ///
    /// This function is mainly useful for data that lives for the
    /// remainder of the program's life. Dropping the returned reference
    /// will cause a memory leak.
    pub fn leak(b: Box<T>) -> &!T {
        unsafe {
            let ptr = Box::into_raw(b)
            &!*ptr
        }
    }
}

impl<T> Deref for Box<T> {
    type Target = T

    fn deref(self: &Box<T>) -> &T {
        unsafe { &*self.ptr }
    }
}

impl<T> DerefMut for Box<T> {
    fn deref_mut(self: &!Box<T>) -> &!T {
        unsafe { &!*self.ptr }
    }
}

impl<T> Drop for Box<T> {
    fn drop(self: &!Box<T>) with Alloc {
        unsafe {
            ptr::drop_in_place(self.ptr)
            alloc::dealloc(self.ptr, 1)
        }
    }
}

impl<T> Clone for Box<T>
where T: Clone
{
    fn clone(self: &Box<T>) -> Box<T> with Alloc {
        Box::new((*self).clone())
    }
}

impl<T> Eq for Box<T>
where T: Eq
{
    fn eq(self: &Box<T>, other: &Box<T>) -> bool {
        **self == **other
    }
}

impl<T> Ord for Box<T>
where T: Ord
{
    fn cmp(self: &Box<T>, other: &Box<T>) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T> Display for Box<T>
where T: Display
{
    fn fmt(self: &Box<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Debug for Box<T>
where T: Debug
{
    fn fmt(self: &Box<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Default for Box<T>
where T: Default
{
    fn default() -> Box<T> with Alloc {
        Box::new(T::default())
    }
}

impl<T> From<T> for Box<T> {
    fn from(x: T) -> Box<T> with Alloc {
        Box::new(x)
    }
}

// Unit tests
#[test]
fn test_box_basic() {
    let b = Box::new(42)
    assert_eq(*b, 42)
}

#[test]
fn test_box_into_inner() {
    let b = Box::new("hello")
    let s = Box::into_inner(b)
    assert_eq(s, "hello")
}

#[test]
fn test_box_clone() {
    let b1 = Box::new(42)
    let b2 = b1.clone()
    assert_eq(*b1, *b2)
}

#[test]
fn test_box_eq() {
    let b1 = Box::new(42)
    let b2 = Box::new(42)
    let b3 = Box::new(43)
    assert(b1 == b2)
    assert(b1 != b3)
}
