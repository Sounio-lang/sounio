//! Interior mutability with Cell and RefCell
//!
//! These types provide interior mutability, allowing mutation
//! through shared references.

use std::cmp::Eq
use std::fmt::{Display, Debug, Formatter, FmtError}
use std::clone::Clone
use std::default::Default
use std::ops::{Deref, DerefMut}

/// A mutable memory location.
///
/// Cell<T> provides interior mutability for Copy types.
/// It enables mutation through a shared reference by moving
/// values in and out of the cell.
///
/// # Examples
///
/// ```d
/// let cell = Cell::new(5)
/// assert_eq(cell.get(), 5)
/// cell.set(10)
/// assert_eq(cell.get(), 10)
/// ```
pub struct Cell<T> {
    value: T,
}

impl<T> Cell<T>
where T: Copy
{
    /// Creates a new Cell containing the given value.
    pub fn new(value: T) -> Cell<T> {
        Cell { value }
    }

    /// Returns a copy of the contained value.
    pub fn get(self: &Cell<T>) -> T {
        self.value
    }

    /// Sets the contained value.
    ///
    /// This method takes a shared reference and still allows
    /// mutation through it, demonstrating interior mutability.
    pub fn set(self: &Cell<T>, value: T) {
        // SAFETY: This is safe because we have & access
        // and T: Copy means we don't need to drop the old value
        unsafe {
            let ptr = &self.value as *const T as *mut T
            ptr::write(ptr, value)
        }
    }

    /// Replaces the contained value, returning the old value.
    pub fn replace(self: &Cell<T>, value: T) -> T {
        let old = self.get()
        self.set(value)
        old
    }

    /// Swaps the values of two Cells.
    pub fn swap(self: &Cell<T>, other: &Cell<T>) {
        let temp = self.get()
        self.set(other.get())
        other.set(temp)
    }

    /// Updates the contained value using a function and returns the new value.
    pub fn update<F>(self: &Cell<T>, f: F) -> T
    where F: fn(T) -> T
    {
        let new_val = f(self.get())
        self.set(new_val)
        new_val
    }

    /// Returns a raw pointer to the underlying data.
    pub fn as_ptr(self: &Cell<T>) -> *mut T {
        &self.value as *const T as *mut T
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// This requires exclusive access to the Cell.
    pub fn get_mut(self: &!Cell<T>) -> &!T {
        &!self.value
    }

    /// Unwraps the value, consuming the Cell.
    pub fn into_inner(self: Cell<T>) -> T {
        self.value
    }
}

impl<T> Clone for Cell<T>
where T: Copy
{
    fn clone(self: &Cell<T>) -> Cell<T> {
        Cell::new(self.get())
    }
}

impl<T> Eq for Cell<T>
where T: Copy + Eq
{
    fn eq(self: &Cell<T>, other: &Cell<T>) -> bool {
        self.get() == other.get()
    }
}

impl<T> Default for Cell<T>
where T: Copy + Default
{
    fn default() -> Cell<T> {
        Cell::new(T::default())
    }
}

impl<T> Debug for Cell<T>
where T: Copy + Debug
{
    fn fmt(self: &Cell<T>, f: &!Formatter) -> Result<unit, FmtError> {
        f.debug_struct("Cell")
            .field("value", &self.get())
            .finish()
    }
}

impl<T> From<T> for Cell<T>
where T: Copy
{
    fn from(value: T) -> Cell<T> {
        Cell::new(value)
    }
}

// Borrow flag values
const UNUSED: int = 0
const WRITING: int = -1

/// A mutable memory location with dynamically checked borrow rules.
///
/// RefCell<T> provides interior mutability with runtime borrow checking.
/// Unlike Cell<T>, it allows borrowing the contents by reference.
///
/// # Panics
///
/// RefCell enforces Rust-like borrow rules at runtime:
/// - Multiple shared borrows are allowed
/// - Only one mutable borrow is allowed
/// - Mutable and shared borrows cannot coexist
///
/// Violating these rules causes a panic.
///
/// # Examples
///
/// ```d
/// let refcell = RefCell::new(vec![1, 2, 3])
///
/// // Immutable borrow
/// {
///     let r = refcell.borrow()
///     println(r.len())  // 3
/// }
///
/// // Mutable borrow
/// {
///     let mut r = refcell.borrow_mut()
///     r.push(4)
/// }
/// ```
pub struct RefCell<T> {
    borrow: Cell<int>,
    value: T,
}

impl<T> RefCell<T> {
    /// Creates a new RefCell containing the given value.
    pub fn new(value: T) -> RefCell<T> {
        RefCell {
            borrow: Cell::new(UNUSED),
            value,
        }
    }

    /// Immutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned Ref exits scope.
    /// Multiple immutable borrows can be taken out at the same time.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently mutably borrowed.
    pub fn borrow(self: &RefCell<T>) -> Ref<T> with Panic {
        match self.try_borrow() {
            Option::Some(r) => r,
            Option::None => panic("RefCell already mutably borrowed"),
        }
    }

    /// Attempts to immutably borrow the wrapped value.
    ///
    /// Returns None if the value is currently mutably borrowed.
    pub fn try_borrow(self: &RefCell<T>) -> Option<Ref<T>> {
        let b = self.borrow.get()

        if b >= 0 {
            self.borrow.set(b + 1)
            Option::Some(Ref { refcell: self })
        } else {
            Option::None
        }
    }

    /// Mutably borrows the wrapped value.
    ///
    /// The borrow lasts until the returned RefMut exits scope.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    pub fn borrow_mut(self: &RefCell<T>) -> RefMut<T> with Panic {
        match self.try_borrow_mut() {
            Option::Some(r) => r,
            Option::None => panic("RefCell already borrowed"),
        }
    }

    /// Attempts to mutably borrow the wrapped value.
    ///
    /// Returns None if the value is currently borrowed.
    pub fn try_borrow_mut(self: &RefCell<T>) -> Option<RefMut<T>> {
        if self.borrow.get() == UNUSED {
            self.borrow.set(WRITING)
            Option::Some(RefMut { refcell: self })
        } else {
            Option::None
        }
    }

    /// Returns a raw pointer to the underlying data.
    pub fn as_ptr(self: &RefCell<T>) -> *mut T {
        &self.value as *const T as *mut T
    }

    /// Returns a mutable reference to the underlying data.
    ///
    /// This method requires exclusive access.
    pub fn get_mut(self: &!RefCell<T>) -> &!T {
        &!self.value
    }

    /// Replaces the wrapped value with a new one.
    ///
    /// # Panics
    ///
    /// Panics if the value is currently borrowed.
    pub fn replace(self: &RefCell<T>, value: T) -> T with Panic {
        std::mem::replace(&!*self.borrow_mut(), value)
    }

    /// Replaces the wrapped value with the result of a function.
    pub fn replace_with<F>(self: &RefCell<T>, f: F) -> T with Panic
    where F: fn(&!T) -> T
    {
        let mut_borrow = &!*self.borrow_mut()
        let replacement = f(mut_borrow)
        std::mem::replace(mut_borrow, replacement)
    }

    /// Swaps the values of two RefCells.
    ///
    /// # Panics
    ///
    /// Panics if either value is currently borrowed.
    pub fn swap(self: &RefCell<T>, other: &RefCell<T>) with Panic {
        std::mem::swap(&!*self.borrow_mut(), &!*other.borrow_mut())
    }

    /// Unwraps the value, consuming the RefCell.
    pub fn into_inner(self: RefCell<T>) -> T {
        self.value
    }
}

impl<T> Clone for RefCell<T>
where T: Clone
{
    fn clone(self: &RefCell<T>) -> RefCell<T> with Panic, Alloc {
        RefCell::new(self.borrow().clone())
    }
}

impl<T> Default for RefCell<T>
where T: Default
{
    fn default() -> RefCell<T> {
        RefCell::new(T::default())
    }
}

impl<T> Debug for RefCell<T>
where T: Debug
{
    fn fmt(self: &RefCell<T>, f: &!Formatter) -> Result<unit, FmtError> {
        match self.try_borrow() {
            Option::Some(b) => f.debug_struct("RefCell")
                .field("value", &*b)
                .finish(),
            Option::None => {
                f.write_str("RefCell { <borrowed> }")
            },
        }
    }
}

impl<T> From<T> for RefCell<T> {
    fn from(value: T) -> RefCell<T> {
        RefCell::new(value)
    }
}

/// A wrapper type for an immutably borrowed value from a RefCell.
///
/// This type implements Deref, so you can use * to access the borrowed value.
pub struct Ref<T> {
    refcell: &RefCell<T>,
}

impl<T> Deref for Ref<T> {
    type Target = T

    fn deref(self: &Ref<T>) -> &T {
        unsafe { &*self.refcell.as_ptr() }
    }
}

impl<T> Drop for Ref<T> {
    fn drop(self: &!Ref<T>) {
        let b = self.refcell.borrow.get() - 1
        self.refcell.borrow.set(b)
    }
}

impl<T> Debug for Ref<T>
where T: Debug
{
    fn fmt(self: &Ref<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Display for Ref<T>
where T: Display
{
    fn fmt(self: &Ref<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

/// A wrapper type for a mutably borrowed value from a RefCell.
///
/// This type implements Deref and DerefMut.
pub struct RefMut<T> {
    refcell: &RefCell<T>,
}

impl<T> Deref for RefMut<T> {
    type Target = T

    fn deref(self: &RefMut<T>) -> &T {
        unsafe { &*self.refcell.as_ptr() }
    }
}

impl<T> DerefMut for RefMut<T> {
    fn deref_mut(self: &!RefMut<T>) -> &!T {
        unsafe { &!*self.refcell.as_ptr() }
    }
}

impl<T> Drop for RefMut<T> {
    fn drop(self: &!RefMut<T>) {
        self.refcell.borrow.set(UNUSED)
    }
}

impl<T> Debug for RefMut<T>
where T: Debug
{
    fn fmt(self: &RefMut<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

impl<T> Display for RefMut<T>
where T: Display
{
    fn fmt(self: &RefMut<T>, f: &!Formatter) -> Result<unit, FmtError> {
        (**self).fmt(f)
    }
}

// Unit tests
#[test]
fn test_cell_basic() {
    let cell = Cell::new(5)
    assert_eq(cell.get(), 5)
    cell.set(10)
    assert_eq(cell.get(), 10)
}

#[test]
fn test_cell_replace() {
    let cell = Cell::new(5)
    let old = cell.replace(10)
    assert_eq(old, 5)
    assert_eq(cell.get(), 10)
}

#[test]
fn test_cell_swap() {
    let a = Cell::new(1)
    let b = Cell::new(2)
    a.swap(&b)
    assert_eq(a.get(), 2)
    assert_eq(b.get(), 1)
}

#[test]
fn test_refcell_borrow() {
    let refcell = RefCell::new(42)
    {
        let r = refcell.borrow()
        assert_eq(*r, 42)
    }
}

#[test]
fn test_refcell_borrow_mut() {
    let refcell = RefCell::new(42)
    {
        let mut r = refcell.borrow_mut()
        *r = 100
    }
    assert_eq(*refcell.borrow(), 100)
}

#[test]
fn test_refcell_multiple_borrows() {
    let refcell = RefCell::new(42)
    let r1 = refcell.borrow()
    let r2 = refcell.borrow()
    assert_eq(*r1, 42)
    assert_eq(*r2, 42)
}
