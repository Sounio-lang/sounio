//! Double-ended queue implementation.
//!
//! Deque<T> is a double-ended queue implemented as a ring buffer.

use std::ops::{Index, IndexMut}
use std::iter::{Iterator, IntoIterator, FromIterator}
use std::cmp::Eq
use std::fmt::{Debug, Formatter, FmtError}
use std::clone::Clone
use std::default::Default

/// A double-ended queue implemented with a growable ring buffer.
///
/// The "default" usage of this type is to use push_back to add to
/// the back and pop_front to remove from the front. This provides
/// FIFO (first-in, first-out) queue behavior.
///
/// Since VecDeque is a ring buffer, its elements are not necessarily
/// contiguous in memory.
///
/// # Examples
///
/// ```d
/// let mut deque = Deque::new()
///
/// deque.push_back(1)
/// deque.push_back(2)
/// deque.push_back(3)
///
/// assert_eq(deque.pop_front(), Option::Some(1))
/// assert_eq(deque.pop_front(), Option::Some(2))
/// ```
pub struct Deque<T> {
    buf: Vec<Option<T>>,
    head: int,
    tail: int,
    len: int,
}

impl<T> Deque<T> {
    /// Creates an empty Deque.
    pub fn new() -> Deque<T> with Alloc {
        Deque::with_capacity(8)
    }

    /// Creates an empty Deque with space for at least `capacity` elements.
    pub fn with_capacity(capacity: int) -> Deque<T> with Alloc {
        let actual_cap = if capacity < 1 { 1 } else { capacity }
        let mut buf = Vec::with_capacity(actual_cap)
        for _ in 0..actual_cap {
            buf.push(Option::None)
        }

        Deque {
            buf,
            head: 0,
            tail: 0,
            len: 0,
        }
    }

    /// Returns the number of elements in the Deque.
    pub fn len(self: &Deque<T>) -> int {
        self.len
    }

    /// Returns true if the Deque is empty.
    pub fn is_empty(self: &Deque<T>) -> bool {
        self.len == 0
    }

    /// Returns the capacity of the Deque.
    pub fn capacity(self: &Deque<T>) -> int {
        self.buf.len()
    }

    /// Clears the Deque, removing all values.
    pub fn clear(self: &!Deque<T>) {
        while self.pop_front().is_some() {}
    }

    /// Adds an element to the front of the Deque.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut deque = Deque::new()
    /// deque.push_front(1)
    /// deque.push_front(2)
    /// assert_eq(deque.front(), Option::Some(&2))
    /// ```
    pub fn push_front(self: &!Deque<T>, value: T) with Alloc {
        if self.len == self.capacity() {
            self.grow()
        }

        self.head = self.wrap_sub(self.head, 1)
        self.buf[self.head] = Option::Some(value)
        self.len = self.len + 1
    }

    /// Adds an element to the back of the Deque.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut deque = Deque::new()
    /// deque.push_back(1)
    /// deque.push_back(2)
    /// assert_eq(deque.back(), Option::Some(&2))
    /// ```
    pub fn push_back(self: &!Deque<T>, value: T) with Alloc {
        if self.len == self.capacity() {
            self.grow()
        }

        self.buf[self.tail] = Option::Some(value)
        self.tail = self.wrap_add(self.tail, 1)
        self.len = self.len + 1
    }

    /// Removes the first element and returns it, or None if empty.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut deque = Deque::new()
    /// deque.push_back(1)
    /// deque.push_back(2)
    /// assert_eq(deque.pop_front(), Option::Some(1))
    /// assert_eq(deque.pop_front(), Option::Some(2))
    /// assert_eq(deque.pop_front(), Option::None)
    /// ```
    pub fn pop_front(self: &!Deque<T>) -> Option<T> {
        if self.is_empty() {
            return Option::None
        }

        let value = self.buf[self.head].take()
        self.head = self.wrap_add(self.head, 1)
        self.len = self.len - 1
        value
    }

    /// Removes the last element and returns it, or None if empty.
    ///
    /// # Examples
    ///
    /// ```d
    /// let mut deque = Deque::new()
    /// deque.push_back(1)
    /// deque.push_back(2)
    /// assert_eq(deque.pop_back(), Option::Some(2))
    /// assert_eq(deque.pop_back(), Option::Some(1))
    /// ```
    pub fn pop_back(self: &!Deque<T>) -> Option<T> {
        if self.is_empty() {
            return Option::None
        }

        self.tail = self.wrap_sub(self.tail, 1)
        self.len = self.len - 1
        self.buf[self.tail].take()
    }

    /// Returns a reference to the front element, or None if empty.
    pub fn front(self: &Deque<T>) -> Option<&T> {
        if self.is_empty() {
            Option::None
        } else {
            self.buf[self.head].as_ref()
        }
    }

    /// Returns a reference to the back element, or None if empty.
    pub fn back(self: &Deque<T>) -> Option<&T> {
        if self.is_empty() {
            Option::None
        } else {
            let idx = self.wrap_sub(self.tail, 1)
            self.buf[idx].as_ref()
        }
    }

    /// Returns a mutable reference to the front element, or None if empty.
    pub fn front_mut(self: &!Deque<T>) -> Option<&!T> {
        if self.is_empty() {
            Option::None
        } else {
            self.buf[self.head].as_mut()
        }
    }

    /// Returns a mutable reference to the back element, or None if empty.
    pub fn back_mut(self: &!Deque<T>) -> Option<&!T> {
        if self.is_empty() {
            Option::None
        } else {
            let idx = self.wrap_sub(self.tail, 1)
            self.buf[idx].as_mut()
        }
    }

    /// Gets a reference to an element by index.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn get(self: &Deque<T>, index: int) -> Option<&T> {
        if index < 0 || index >= self.len {
            return Option::None
        }

        let idx = self.wrap_add(self.head, index)
        self.buf[idx].as_ref()
    }

    /// Gets a mutable reference to an element by index.
    pub fn get_mut(self: &!Deque<T>, index: int) -> Option<&!T> {
        if index < 0 || index >= self.len {
            return Option::None
        }

        let idx = self.wrap_add(self.head, index)
        self.buf[idx].as_mut()
    }

    /// Returns an iterator over the elements.
    pub fn iter(self: &Deque<T>) -> Iter<T> {
        Iter {
            deque: self,
            pos: 0,
        }
    }

    /// Returns a mutable iterator over the elements.
    pub fn iter_mut(self: &!Deque<T>) -> IterMut<T> {
        IterMut {
            deque: self,
            pos: 0,
        }
    }

    /// Swaps elements at indices i and j.
    ///
    /// # Panics
    ///
    /// Panics if either index is out of bounds.
    pub fn swap(self: &!Deque<T>, i: int, j: int) with Panic {
        if i >= self.len || j >= self.len || i < 0 || j < 0 {
            panic("index out of bounds")
        }

        let idx_i = self.wrap_add(self.head, i)
        let idx_j = self.wrap_add(self.head, j)
        self.buf.swap(idx_i, idx_j)
    }

    /// Rotates the deque n positions to the left.
    ///
    /// This is equivalent to calling pop_front and push_back n times.
    pub fn rotate_left(self: &!Deque<T>, n: int) {
        if self.is_empty() || n == 0 {
            return
        }

        let n = ((n % self.len) + self.len) % self.len
        self.head = self.wrap_add(self.head, n)
        self.tail = self.wrap_add(self.tail, n)
    }

    /// Rotates the deque n positions to the right.
    ///
    /// This is equivalent to calling pop_back and push_front n times.
    pub fn rotate_right(self: &!Deque<T>, n: int) {
        if self.is_empty() || n == 0 {
            return
        }

        let n = ((n % self.len) + self.len) % self.len
        self.head = self.wrap_sub(self.head, n)
        self.tail = self.wrap_sub(self.tail, n)
    }

    /// Reserves capacity for at least `additional` more elements.
    pub fn reserve(self: &!Deque<T>, additional: int) with Alloc {
        let needed = self.len + additional
        if needed > self.capacity() {
            self.grow_to(needed)
        }
    }

    /// Returns true if the deque contains the value.
    pub fn contains(self: &Deque<T>, value: &T) -> bool
    where T: Eq
    {
        self.iter().any(|v| v == value)
    }

    // Private helpers

    fn wrap_add(self: &Deque<T>, idx: int, n: int) -> int {
        (idx + n) % self.capacity()
    }

    fn wrap_sub(self: &Deque<T>, idx: int, n: int) -> int {
        (idx - n + self.capacity()) % self.capacity()
    }

    fn grow(self: &!Deque<T>) with Alloc {
        self.grow_to(self.capacity() * 2)
    }

    fn grow_to(self: &!Deque<T>, new_cap: int) with Alloc {
        let old_cap = self.capacity()

        // Expand buffer
        for _ in old_cap..new_cap {
            self.buf.push(Option::None)
        }

        // Rearrange if wrapped
        if self.head > 0 && self.tail <= self.head && self.len > 0 {
            // Move wrapped elements to end
            for i in 0..self.tail {
                self.buf[old_cap + i] = self.buf[i].take()
            }
            self.tail = old_cap + self.tail
        }
    }
}

impl<T> Index<int> for Deque<T> {
    type Output = T

    fn index(self: &Deque<T>, index: int) -> &T with Panic {
        match self.get(index) {
            Option::Some(v) => v,
            Option::None => panic("index out of bounds"),
        }
    }
}

impl<T> IndexMut<int> for Deque<T> {
    fn index_mut(self: &!Deque<T>, index: int) -> &!T with Panic {
        let len = self.len
        match self.get_mut(index) {
            Option::Some(v) => v,
            Option::None => panic("index {} out of bounds for deque of length {}", index, len),
        }
    }
}

impl<T> Default for Deque<T> {
    fn default() -> Deque<T> with Alloc {
        Deque::new()
    }
}

impl<T> Clone for Deque<T>
where T: Clone
{
    fn clone(self: &Deque<T>) -> Deque<T> with Alloc {
        let mut new_deque = Deque::with_capacity(self.capacity())
        for item in self.iter() {
            new_deque.push_back(item.clone())
        }
        new_deque
    }
}

impl<T> Eq for Deque<T>
where T: Eq
{
    fn eq(self: &Deque<T>, other: &Deque<T>) -> bool {
        if self.len != other.len {
            return false
        }

        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<T> Debug for Deque<T>
where T: Debug
{
    fn fmt(self: &Deque<T>, f: &!Formatter) -> Result<unit, FmtError> {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T> FromIterator<T> for Deque<T> {
    fn from_iter<I>(iter: I) -> Deque<T> with Alloc
    where I: IntoIterator<Item = T>
    {
        let mut deque = Deque::new()
        for item in iter {
            deque.push_back(item)
        }
        deque
    }
}

impl<T> IntoIterator for Deque<T> {
    type Item = T
    type IntoIter = IntoIter<T>

    fn into_iter(self: Deque<T>) -> IntoIter<T> {
        IntoIter { deque: self }
    }
}

impl<T> Extend<T> for Deque<T> {
    fn extend<I>(self: &!Deque<T>, iter: I) with Alloc
    where I: IntoIterator<Item = T>
    {
        for item in iter {
            self.push_back(item)
        }
    }
}

/// Iterator over Deque elements
pub struct Iter<T> {
    deque: &Deque<T>,
    pos: int,
}

impl<T> Iterator for Iter<T> {
    type Item = &T

    fn next(self: &!Iter<T>) -> Option<&T> {
        if self.pos >= self.deque.len {
            return Option::None
        }

        let item = self.deque.get(self.pos)
        self.pos = self.pos + 1
        item
    }

    fn size_hint(self: &Iter<T>) -> (int, Option<int>) {
        let remaining = self.deque.len - self.pos
        (remaining, Option::Some(remaining))
    }
}

/// Mutable iterator over Deque elements
pub struct IterMut<T> {
    deque: &!Deque<T>,
    pos: int,
}

impl<T> Iterator for IterMut<T> {
    type Item = &!T

    fn next(self: &!IterMut<T>) -> Option<&!T> {
        if self.pos >= self.deque.len {
            return Option::None
        }

        let item = self.deque.get_mut(self.pos)
        self.pos = self.pos + 1
        item
    }
}

/// Owning iterator over Deque elements
pub struct IntoIter<T> {
    deque: Deque<T>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T

    fn next(self: &!IntoIter<T>) -> Option<T> {
        self.deque.pop_front()
    }

    fn size_hint(self: &IntoIter<T>) -> (int, Option<int>) {
        let len = self.deque.len()
        (len, Option::Some(len))
    }
}

// Unit tests
#[test]
fn test_deque_basic() {
    let mut deque = Deque::new()
    assert(deque.is_empty())

    deque.push_back(1)
    deque.push_back(2)
    deque.push_back(3)

    assert_eq(deque.len(), 3)
    assert_eq(deque.pop_front(), Option::Some(1))
    assert_eq(deque.pop_front(), Option::Some(2))
    assert_eq(deque.pop_front(), Option::Some(3))
    assert_eq(deque.pop_front(), Option::None)
}

#[test]
fn test_deque_push_front() {
    let mut deque = Deque::new()
    deque.push_front(1)
    deque.push_front(2)
    deque.push_front(3)

    assert_eq(deque.pop_front(), Option::Some(3))
    assert_eq(deque.pop_front(), Option::Some(2))
    assert_eq(deque.pop_front(), Option::Some(1))
}

#[test]
fn test_deque_pop_back() {
    let mut deque = Deque::new()
    deque.push_back(1)
    deque.push_back(2)
    deque.push_back(3)

    assert_eq(deque.pop_back(), Option::Some(3))
    assert_eq(deque.pop_back(), Option::Some(2))
    assert_eq(deque.pop_back(), Option::Some(1))
}

#[test]
fn test_deque_index() {
    let mut deque = Deque::new()
    deque.push_back(10)
    deque.push_back(20)
    deque.push_back(30)

    assert_eq(deque[0], 10)
    assert_eq(deque[1], 20)
    assert_eq(deque[2], 30)
}

#[test]
fn test_deque_front_back() {
    let mut deque = Deque::new()
    deque.push_back(1)
    deque.push_back(2)
    deque.push_back(3)

    assert_eq(*deque.front().unwrap(), 1)
    assert_eq(*deque.back().unwrap(), 3)
}

#[test]
fn test_deque_grow() {
    let mut deque = Deque::with_capacity(2)
    deque.push_back(1)
    deque.push_back(2)
    deque.push_back(3)  // Should trigger grow

    assert_eq(deque.len(), 3)
    assert(deque.capacity() >= 3)
}

#[test]
fn test_deque_rotate() {
    let mut deque = Deque::new()
    deque.push_back(1)
    deque.push_back(2)
    deque.push_back(3)

    deque.rotate_left(1)
    assert_eq(*deque.front().unwrap(), 2)

    deque.rotate_right(1)
    assert_eq(*deque.front().unwrap(), 1)
}
