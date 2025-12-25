// stdlib/collections/vec.d - Growable array type
//
// A contiguous growable array with heap-allocated storage.

module std.collections.vec;

import std.core.option;
import std.core.result;
import std.mem.allocator;
import std.iter.iterator;
import std.cmp;
import std.ops;
import std.fmt;
import std.hash;

/// A contiguous growable array type.
///
/// # Examples
/// ```
/// let mut v = Vec.new();
/// v.push(1);
/// v.push(2);
/// v.push(3);
/// assert_eq!(v.len(), 3);
/// assert_eq!(v[0], 1);
/// ```
pub struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}

impl<T> Vec<T> {
    /// Creates a new empty Vec.
    pub fn new() -> Vec<T> {
        Vec {
            ptr: ptr.null_mut(),
            len: 0,
            cap: 0,
        }
    }

    /// Creates a new Vec with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Vec<T> with Alloc {
        if capacity == 0 {
            return Vec.new();
        }

        let ptr = alloc.allocate::<T>(capacity);
        Vec {
            ptr,
            len: 0,
            cap: capacity,
        }
    }

    /// Creates a Vec from raw parts.
    ///
    /// # Safety
    /// - ptr must be allocated by the same allocator
    /// - length must be <= capacity
    /// - the first length elements must be initialized
    pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Vec<T> {
        Vec { ptr, len: length, cap: capacity }
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the Vec is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the capacity.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns a slice of the Vec's contents.
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { slice.from_raw_parts(self.ptr, self.len) }
        }
    }

    /// Returns a mutable slice of the Vec's contents.
    pub fn as_mut_slice(&!self) -> &![T] {
        if self.len == 0 {
            &![]
        } else {
            unsafe { slice.from_raw_parts_mut(self.ptr, self.len) }
        }
    }

    /// Returns a raw pointer to the buffer.
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Returns a mutable raw pointer to the buffer.
    pub fn as_mut_ptr(&!self) -> *mut T {
        self.ptr
    }

    /// Pushes an element to the back.
    pub fn push(&!self, value: T) with Alloc {
        if self.len == self.cap {
            self.grow();
        }

        unsafe {
            ptr.write(self.ptr.add(self.len), value);
        }
        self.len += 1;
    }

    /// Removes and returns the last element.
    pub fn pop(&!self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            unsafe {
                Some(ptr.read(self.ptr.add(self.len)))
            }
        }
    }

    /// Inserts an element at the given index.
    ///
    /// # Panics
    /// Panics if index > len.
    pub fn insert(&!self, index: usize, element: T) with Alloc, Panic {
        assert!(index <= self.len, "index out of bounds");

        if self.len == self.cap {
            self.grow();
        }

        unsafe {
            // Shift elements to make room
            if index < self.len {
                ptr.copy(
                    self.ptr.add(index),
                    self.ptr.add(index + 1),
                    self.len - index
                );
            }
            ptr.write(self.ptr.add(index), element);
        }
        self.len += 1;
    }

    /// Removes and returns the element at the given index.
    ///
    /// # Panics
    /// Panics if index >= len.
    pub fn remove(&!self, index: usize) -> T with Panic {
        assert!(index < self.len, "index out of bounds");

        unsafe {
            let result = ptr.read(self.ptr.add(index));

            // Shift elements
            if index < self.len - 1 {
                ptr.copy(
                    self.ptr.add(index + 1),
                    self.ptr.add(index),
                    self.len - index - 1
                );
            }

            self.len -= 1;
            result
        }
    }

    /// Removes an element by swapping it with the last element.
    /// This is O(1) but doesn't preserve order.
    pub fn swap_remove(&!self, index: usize) -> T with Panic {
        assert!(index < self.len, "index out of bounds");

        self.len -= 1;
        unsafe {
            let last = ptr.read(self.ptr.add(self.len));
            let removed = ptr.read(self.ptr.add(index));
            if index != self.len {
                ptr.write(self.ptr.add(index), last);
            }
            removed
        }
    }

    /// Clears the Vec, removing all elements.
    pub fn clear(&!self) {
        // Drop all elements
        for i in 0..self.len {
            unsafe {
                ptr.drop_in_place(self.ptr.add(i));
            }
        }
        self.len = 0;
    }

    /// Truncates the Vec to the specified length.
    pub fn truncate(&!self, new_len: usize) {
        if new_len >= self.len {
            return;
        }

        // Drop elements beyond new_len
        for i in new_len..self.len {
            unsafe {
                ptr.drop_in_place(self.ptr.add(i));
            }
        }
        self.len = new_len;
    }

    /// Reserves capacity for at least additional more elements.
    pub fn reserve(&!self, additional: usize) with Alloc {
        let required = self.len + additional;
        if required > self.cap {
            self.reserve_exact(required - self.cap);
        }
    }

    /// Reserves exact capacity for additional more elements.
    pub fn reserve_exact(&!self, additional: usize) with Alloc {
        let new_cap = self.len + additional;
        if new_cap <= self.cap {
            return;
        }

        self.reallocate(new_cap);
    }

    /// Shrinks capacity to fit the current length.
    pub fn shrink_to_fit(&!self) with Alloc {
        if self.cap > self.len {
            if self.len == 0 {
                if self.cap > 0 {
                    unsafe {
                        alloc.deallocate(self.ptr, self.cap);
                    }
                    self.ptr = ptr.null_mut();
                    self.cap = 0;
                }
            } else {
                self.reallocate(self.len);
            }
        }
    }

    /// Returns a reference to the element at the index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            unsafe { Some(&*self.ptr.add(index)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the element at the index.
    pub fn get_mut(&!self, index: usize) -> Option<&!T> {
        if index < self.len {
            unsafe { Some(&!*self.ptr.add(index)) }
        } else {
            None
        }
    }

    /// Returns the first element.
    pub fn first(&self) -> Option<&T> {
        self.get(0)
    }

    /// Returns the first element mutably.
    pub fn first_mut(&!self) -> Option<&!T> {
        self.get_mut(0)
    }

    /// Returns the last element.
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            self.get(self.len - 1)
        }
    }

    /// Returns the last element mutably.
    pub fn last_mut(&!self) -> Option<&!T> {
        if self.len == 0 {
            None
        } else {
            self.get_mut(self.len - 1)
        }
    }

    /// Swaps two elements by index.
    pub fn swap(&!self, a: usize, b: usize) with Panic {
        assert!(a < self.len && b < self.len, "index out of bounds");

        if a != b {
            unsafe {
                ptr.swap(self.ptr.add(a), self.ptr.add(b));
            }
        }
    }

    /// Reverses the order of elements.
    pub fn reverse(&!self) {
        let mut i = 0;
        let mut j = self.len.saturating_sub(1);

        while i < j {
            self.swap(i, j);
            i += 1;
            j -= 1;
        }
    }

    /// Returns an iterator over references.
    pub fn iter(&self) -> Iter<T> {
        Iter {
            ptr: self.ptr,
            end: unsafe { self.ptr.add(self.len) },
            _marker: PhantomData,
        }
    }

    /// Returns an iterator over mutable references.
    pub fn iter_mut(&!self) -> IterMut<T> {
        IterMut {
            ptr: self.ptr,
            end: unsafe { self.ptr.add(self.len) },
            _marker: PhantomData,
        }
    }

    /// Retains only elements for which the predicate returns true.
    pub fn retain<F>(&!self, mut predicate: F)
    where
        F: FnMut(&T) -> bool
    {
        let mut write = 0;

        for read in 0..self.len {
            unsafe {
                let elem = &*self.ptr.add(read);
                if predicate(elem) {
                    if read != write {
                        ptr.copy_nonoverlapping(
                            self.ptr.add(read),
                            self.ptr.add(write),
                            1
                        );
                    }
                    write += 1;
                } else {
                    ptr.drop_in_place(self.ptr.add(read));
                }
            }
        }

        self.len = write;
    }

    /// Appends all elements from a slice.
    pub fn extend_from_slice(&!self, slice: &[T]) with Alloc
    where
        T: Clone
    {
        self.reserve(slice.len());

        for item in slice {
            self.push(item.clone());
        }
    }

    /// Appends all elements from another Vec.
    pub fn append(&!self, other: &!Vec<T>) with Alloc {
        self.reserve(other.len);

        unsafe {
            ptr.copy_nonoverlapping(
                other.ptr,
                self.ptr.add(self.len),
                other.len
            );
        }

        self.len += other.len;
        other.len = 0;
    }

    /// Splits the Vec at the given index.
    pub fn split_off(&!self, at: usize) -> Vec<T> with Alloc, Panic {
        assert!(at <= self.len, "index out of bounds");

        let other_len = self.len - at;
        let mut other = Vec.with_capacity(other_len);

        unsafe {
            ptr.copy_nonoverlapping(
                self.ptr.add(at),
                other.ptr,
                other_len
            );
        }

        other.len = other_len;
        self.len = at;
        other
    }

    /// Resizes the Vec to the new length.
    pub fn resize(&!self, new_len: usize, value: T) with Alloc
    where
        T: Clone
    {
        if new_len > self.len {
            self.reserve(new_len - self.len);
            for _ in self.len..new_len {
                self.push(value.clone());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Resizes the Vec using a closure to create new elements.
    pub fn resize_with<F>(&!self, new_len: usize, mut f: F) with Alloc
    where
        F: FnMut() -> T
    {
        if new_len > self.len {
            self.reserve(new_len - self.len);
            for _ in self.len..new_len {
                self.push(f());
            }
        } else {
            self.truncate(new_len);
        }
    }

    /// Deduplicates consecutive equal elements.
    pub fn dedup(&!self)
    where
        T: Eq
    {
        self.dedup_by(|a, b| a == b);
    }

    /// Deduplicates consecutive elements using a predicate.
    pub fn dedup_by<F>(&!self, mut same: F)
    where
        F: FnMut(&T, &T) -> bool
    {
        if self.len <= 1 {
            return;
        }

        let mut write = 1;

        for read in 1..self.len {
            unsafe {
                let prev = &*self.ptr.add(write - 1);
                let curr = &*self.ptr.add(read);

                if !same(prev, curr) {
                    if read != write {
                        ptr.copy_nonoverlapping(
                            self.ptr.add(read),
                            self.ptr.add(write),
                            1
                        );
                    }
                    write += 1;
                } else {
                    ptr.drop_in_place(self.ptr.add(read));
                }
            }
        }

        self.len = write;
    }

    /// Fills the Vec with the given value.
    pub fn fill(&!self, value: T)
    where
        T: Clone
    {
        for i in 0..self.len {
            unsafe {
                ptr.drop_in_place(self.ptr.add(i));
                ptr.write(self.ptr.add(i), value.clone());
            }
        }
    }

    /// Fills the Vec using a closure.
    pub fn fill_with<F>(&!self, mut f: F)
    where
        F: FnMut() -> T
    {
        for i in 0..self.len {
            unsafe {
                ptr.drop_in_place(self.ptr.add(i));
                ptr.write(self.ptr.add(i), f());
            }
        }
    }

    // Internal: Grow capacity
    fn grow(&!self) with Alloc {
        let new_cap = if self.cap == 0 {
            4
        } else {
            self.cap * 2
        };
        self.reallocate(new_cap);
    }

    // Internal: Reallocate to new capacity
    fn reallocate(&!self, new_cap: usize) with Alloc {
        let new_ptr = alloc.allocate::<T>(new_cap);

        if self.len > 0 {
            unsafe {
                ptr.copy_nonoverlapping(self.ptr, new_ptr, self.len);
            }
        }

        if self.cap > 0 {
            unsafe {
                alloc.deallocate(self.ptr, self.cap);
            }
        }

        self.ptr = new_ptr;
        self.cap = new_cap;
    }
}

// ============================================================================
// Iterators
// ============================================================================

/// Iterator over references.
pub struct Iter<T> {
    ptr: *const T,
    end: *const T,
    _marker: PhantomData<&T>,
}

impl<T> Iterator for Iter<T> {
    type Item = &T;

    fn next(&!self) -> Option<&T> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                let item = &*self.ptr;
                self.ptr = self.ptr.add(1);
                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe { self.end.offset_from(self.ptr) as usize };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for Iter<T> {}

impl<T> DoubleEndedIterator for Iter<T> {
    fn next_back(&!self) -> Option<&T> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                self.end = self.end.sub(1);
                Some(&*self.end)
            }
        }
    }
}

/// Iterator over mutable references.
pub struct IterMut<T> {
    ptr: *mut T,
    end: *mut T,
    _marker: PhantomData<&!T>,
}

impl<T> Iterator for IterMut<T> {
    type Item = &!T;

    fn next(&!self) -> Option<&!T> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                let item = &!*self.ptr;
                self.ptr = self.ptr.add(1);
                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe { self.end.offset_from(self.ptr) as usize };
        (len, Some(len))
    }
}

impl<T> ExactSizeIterator for IterMut<T> {}

impl<T> DoubleEndedIterator for IterMut<T> {
    fn next_back(&!self) -> Option<&!T> {
        if self.ptr == self.end {
            None
        } else {
            unsafe {
                self.end = self.end.sub(1);
                Some(&!*self.end)
            }
        }
    }
}

/// Owning iterator.
pub struct IntoIter<T> {
    vec: Vec<T>,
    pos: usize,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        if self.pos >= self.vec.len {
            None
        } else {
            unsafe {
                let item = ptr.read(self.vec.ptr.add(self.pos));
                self.pos += 1;
                Some(item)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.vec.len - self.pos;
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for IntoIter<T> {}

impl<T> Drop for IntoIter<T> {
    fn drop(&!self) {
        // Drop remaining elements
        for i in self.pos..self.vec.len {
            unsafe {
                ptr.drop_in_place(self.vec.ptr.add(i));
            }
        }
        // Don't run Vec's destructor, we handle cleanup
        self.vec.len = 0;
    }
}

// ============================================================================
// Trait Implementations
// ============================================================================

impl<T> Drop for Vec<T> {
    fn drop(&!self) {
        self.clear();
        if self.cap > 0 {
            unsafe {
                alloc.deallocate(self.ptr, self.cap);
            }
        }
    }
}

impl<T: Clone> Clone for Vec<T> {
    fn clone(&self) -> Vec<T> with Alloc {
        let mut new_vec = Vec.with_capacity(self.len);
        for item in self.iter() {
            new_vec.push(item.clone());
        }
        new_vec
    }
}

impl<T> Default for Vec<T> {
    fn default() -> Vec<T> {
        Vec.new()
    }
}

impl<T: Eq> Eq for Vec<T> {
    fn eq(&self, other: &Vec<T>) -> bool {
        if self.len != other.len {
            return false;
        }

        for i in 0..self.len {
            if self[i] != other[i] {
                return false;
            }
        }
        true
    }
}

impl<T: Ord> Ord for Vec<T> {
    fn cmp(&self, other: &Vec<T>) -> Ordering {
        let min_len = if self.len < other.len { self.len } else { other.len };

        for i in 0..min_len {
            match self[i].cmp(&other[i]) {
                Ordering.Equal => continue,
                ord => return ord,
            }
        }

        self.len.cmp(&other.len)
    }
}

impl<T: PartialOrd> PartialOrd for Vec<T> {
    fn partial_cmp(&self, other: &Vec<T>) -> Option<Ordering> {
        let min_len = if self.len < other.len { self.len } else { other.len };

        for i in 0..min_len {
            match self[i].partial_cmp(&other[i]) {
                Some(Ordering.Equal) => continue,
                ord => return ord,
            }
        }

        Some(self.len.cmp(&other.len))
    }
}

impl<T: Hash> Hash for Vec<T> {
    fn hash<H: Hasher>(&self, state: &!H) {
        self.len.hash(state);
        for item in self.iter() {
            item.hash(state);
        }
    }
}

impl<T: Debug> Debug for Vec<T> {
    fn fmt(&self, f: &!Formatter) -> Result<(), Error> {
        write!(f, "[")?;
        let mut first = true;
        for item in self.iter() {
            if !first {
                write!(f, ", ")?;
            }
            item.fmt(f)?;
            first = false;
        }
        write!(f, "]")
    }
}

impl<T> Index<usize> for Vec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T with Panic {
        self.get(index).expect("index out of bounds")
    }
}

impl<T> IndexMut<usize> for Vec<T> {
    fn index_mut(&!self, index: usize) -> &!T with Panic {
        self.get_mut(index).expect("index out of bounds")
    }
}

impl<T> Index<Range<usize>> for Vec<T> {
    type Output = [T];

    fn index(&self, range: Range<usize>) -> &[T] with Panic {
        assert!(range.start <= range.end && range.end <= self.len);
        unsafe {
            slice.from_raw_parts(self.ptr.add(range.start), range.end - range.start)
        }
    }
}

impl<T> IndexMut<Range<usize>> for Vec<T> {
    fn index_mut(&!self, range: Range<usize>) -> &![T] with Panic {
        assert!(range.start <= range.end && range.end <= self.len);
        unsafe {
            slice.from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start)
        }
    }
}

impl<T> IntoIterator for Vec<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> IntoIter<T> {
        let iter = IntoIter { vec: self, pos: 0 };
        // Prevent double-free by forgetting original vec
        mem.forget(self);
        iter
    }
}

impl<T> FromIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Vec<T> with Alloc {
        let mut vec = Vec.new();
        for item in iter {
            vec.push(item);
        }
        vec
    }
}

impl<T> Extend<T> for Vec<T> {
    fn extend<I: IntoIterator<Item = T>>(&!self, iter: I) with Alloc {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T: Clone> From<&[T]> for Vec<T> {
    fn from(slice: &[T]) -> Vec<T> with Alloc {
        let mut vec = Vec.with_capacity(slice.len());
        for item in slice {
            vec.push(item.clone());
        }
        vec
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Creates a Vec from a list of elements.
///
/// This is typically used via the vec![] macro.
pub fn vec_from_array<T, const N: usize>(array: [T; N]) -> Vec<T> with Alloc {
    let mut v = Vec.with_capacity(N);
    for item in array {
        v.push(item);
    }
    v
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let v: Vec<i32> = Vec.new();
        assert!(v.is_empty());
        assert_eq!(v.capacity(), 0);
    }

    #[test]
    fn test_with_capacity() {
        let v: Vec<i32> = Vec.with_capacity(10);
        assert!(v.is_empty());
        assert!(v.capacity() >= 10);
    }

    #[test]
    fn test_push_pop() {
        let mut v = Vec.new();
        v.push(1);
        v.push(2);
        v.push(3);

        assert_eq!(v.len(), 3);
        assert_eq!(v.pop(), Some(3));
        assert_eq!(v.pop(), Some(2));
        assert_eq!(v.pop(), Some(1));
        assert_eq!(v.pop(), None);
    }

    #[test]
    fn test_indexing() {
        let mut v = Vec.new();
        v.push(10);
        v.push(20);
        v.push(30);

        assert_eq!(v[0], 10);
        assert_eq!(v[1], 20);
        assert_eq!(v[2], 30);

        v[1] = 25;
        assert_eq!(v[1], 25);
    }

    #[test]
    fn test_insert_remove() {
        let mut v = Vec.new();
        v.push(1);
        v.push(3);

        v.insert(1, 2);
        assert_eq!(v.as_slice(), &[1, 2, 3]);

        let removed = v.remove(1);
        assert_eq!(removed, 2);
        assert_eq!(v.as_slice(), &[1, 3]);
    }

    #[test]
    fn test_iter() {
        let mut v = Vec.new();
        v.push(1);
        v.push(2);
        v.push(3);

        let sum: i32 = v.iter().sum();
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_clone() {
        let mut v1 = Vec.new();
        v1.push(1);
        v1.push(2);

        let v2 = v1.clone();
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_extend() {
        let mut v = Vec.new();
        v.push(1);
        v.extend([2, 3, 4]);
        assert_eq!(v.as_slice(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_retain() {
        let mut v = Vec.new();
        v.extend([1, 2, 3, 4, 5]);
        v.retain(|x| x % 2 == 0);
        assert_eq!(v.as_slice(), &[2, 4]);
    }

    #[test]
    fn test_dedup() {
        let mut v = Vec.new();
        v.extend([1, 1, 2, 2, 2, 3]);
        v.dedup();
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_reverse() {
        let mut v = Vec.new();
        v.extend([1, 2, 3]);
        v.reverse();
        assert_eq!(v.as_slice(), &[3, 2, 1]);
    }
}
