---
title: Vec<T>
description: Growable array type with heap-allocated storage
---

# Vec<T>

`Vec<T>` is a contiguous growable array type with heap-allocated storage. It is the most commonly used collection in Sounio for storing sequences of elements.

## Type Definition

```sio
pub struct Vec<T> {
    ptr: *mut T,
    len: usize,
    cap: usize,
}
```

## Constructors

### new

```sio
pub fn new() -> Vec<T>
```

Creates a new empty `Vec`.

**Example:**

```sio
let v: Vec<i32> = Vec::new()
```

### with_capacity

```sio
pub fn with_capacity(capacity: usize) -> Vec<T> with Alloc
```

Creates a new `Vec` with the specified capacity. The vector will be able to hold at least `capacity` elements without reallocating.

**Parameters:**
- `capacity` - The minimum number of elements to allocate space for

**Example:**

```sio
let v: Vec<i32> = Vec::with_capacity(100)
// v.len() == 0
// v.capacity() >= 100
```

### from_raw_parts (unsafe)

```sio
pub unsafe fn from_raw_parts(ptr: *mut T, length: usize, capacity: usize) -> Vec<T>
```

Creates a `Vec` from raw parts.

**Safety:**
- `ptr` must be allocated by the same allocator
- `length` must be <= `capacity`
- The first `length` elements must be initialized

## Methods

### Basic Properties

#### len

```sio
pub fn len(&self) -> usize
```

Returns the number of elements in the vector.

**Example:**

```sio
let v = vec![1, 2, 3]
v.len()  // 3
```

#### is_empty

```sio
pub fn is_empty(&self) -> bool
```

Returns `true` if the vector contains no elements.

**Example:**

```sio
let empty: Vec<i32> = Vec::new()
empty.is_empty()  // true
```

#### capacity

```sio
pub fn capacity(&self) -> usize
```

Returns the total number of elements the vector can hold without reallocating.

**Example:**

```sio
let v: Vec<i32> = Vec::with_capacity(10)
v.capacity()  // >= 10
```

### Adding Elements

#### push

```sio
pub fn push(&!self, value: T) with Alloc
```

Appends an element to the back of the vector.

**Parameters:**
- `value` - The element to append

**Example:**

```sio
var v: Vec<i32> = Vec::new()
v.push(1)
v.push(2)
v.push(3)
// v = [1, 2, 3]
```

#### insert

```sio
pub fn insert(&!self, index: usize, element: T) with Alloc, Panic
```

Inserts an element at the given index, shifting all elements after it to the right.

**Parameters:**
- `index` - The position to insert at (must be <= len)
- `element` - The element to insert

**Panics:** If `index > len`.

**Example:**

```sio
var v = vec![1, 3]
v.insert(1, 2)
// v = [1, 2, 3]
```

#### extend_from_slice

```sio
pub fn extend_from_slice(&!self, slice: &[T]) with Alloc
where T: Clone
```

Appends all elements from a slice by cloning.

**Example:**

```sio
var v = vec![1, 2]
v.extend_from_slice(&[3, 4, 5])
// v = [1, 2, 3, 4, 5]
```

#### append

```sio
pub fn append(&!self, other: &!Vec<T>) with Alloc
```

Moves all elements from `other` into `self`, leaving `other` empty.

**Example:**

```sio
var v1 = vec![1, 2]
var v2 = vec![3, 4]
v1.append(&!v2)
// v1 = [1, 2, 3, 4]
// v2 = []
```

### Removing Elements

#### pop

```sio
pub fn pop(&!self) -> Option<T>
```

Removes and returns the last element, or `None` if empty.

**Example:**

```sio
var v = vec![1, 2, 3]
let last = v.pop()  // Some(3)
// v = [1, 2]
```

#### remove

```sio
pub fn remove(&!self, index: usize) -> T with Panic
```

Removes and returns the element at the given index, shifting all elements after it to the left.

**Parameters:**
- `index` - The position to remove from

**Panics:** If `index >= len`.

**Example:**

```sio
var v = vec![1, 2, 3]
let removed = v.remove(1)  // 2
// v = [1, 3]
```

#### swap_remove

```sio
pub fn swap_remove(&!self, index: usize) -> T with Panic
```

Removes an element by swapping it with the last element. This is O(1) but doesn't preserve order.

**Parameters:**
- `index` - The position to remove from

**Panics:** If `index >= len`.

**Example:**

```sio
var v = vec![1, 2, 3, 4]
let removed = v.swap_remove(1)  // 2
// v = [1, 4, 3] (order not preserved)
```

#### clear

```sio
pub fn clear(&!self)
```

Removes all elements from the vector.

**Example:**

```sio
var v = vec![1, 2, 3]
v.clear()
// v = []
```

#### truncate

```sio
pub fn truncate(&!self, new_len: usize)
```

Shortens the vector to the specified length. If `new_len >= len`, this has no effect.

**Example:**

```sio
var v = vec![1, 2, 3, 4, 5]
v.truncate(2)
// v = [1, 2]
```

#### retain

```sio
pub fn retain<F>(&!self, predicate: F)
where F: FnMut(&T) -> bool
```

Retains only the elements for which the predicate returns `true`.

**Example:**

```sio
var v = vec![1, 2, 3, 4, 5]
v.retain(|x| x % 2 == 0)
// v = [2, 4]
```

### Accessing Elements

#### get

```sio
pub fn get(&self, index: usize) -> Option<&T>
```

Returns a reference to the element at the index, or `None` if out of bounds.

**Example:**

```sio
let v = vec![10, 20, 30]
v.get(1)  // Some(&20)
v.get(5)  // None
```

#### get_mut

```sio
pub fn get_mut(&!self, index: usize) -> Option<&!T>
```

Returns a mutable reference to the element at the index, or `None` if out of bounds.

**Example:**

```sio
var v = vec![10, 20, 30]
if let Some(elem) = v.get_mut(1) {
    *elem = 25
}
// v = [10, 25, 30]
```

#### first

```sio
pub fn first(&self) -> Option<&T>
```

Returns a reference to the first element, or `None` if empty.

**Example:**

```sio
let v = vec![1, 2, 3]
v.first()  // Some(&1)
```

#### first_mut

```sio
pub fn first_mut(&!self) -> Option<&!T>
```

Returns a mutable reference to the first element.

#### last

```sio
pub fn last(&self) -> Option<&T>
```

Returns a reference to the last element, or `None` if empty.

**Example:**

```sio
let v = vec![1, 2, 3]
v.last()  // Some(&3)
```

#### last_mut

```sio
pub fn last_mut(&!self) -> Option<&!T>
```

Returns a mutable reference to the last element.

### Index Operations

Vectors support indexing with `[]`:

```sio
let v = vec![10, 20, 30]
let second = v[1]  // 20

var v = vec![10, 20, 30]
v[1] = 25  // v = [10, 25, 30]
```

**Panics:** If the index is out of bounds.

### Slice Operations

#### as_slice

```sio
pub fn as_slice(&self) -> &[T]
```

Returns a slice containing all elements.

**Example:**

```sio
let v = vec![1, 2, 3]
let slice = v.as_slice()
```

#### as_mut_slice

```sio
pub fn as_mut_slice(&!self) -> &![T]
```

Returns a mutable slice containing all elements.

### Capacity Management

#### reserve

```sio
pub fn reserve(&!self, additional: usize) with Alloc
```

Reserves capacity for at least `additional` more elements.

**Example:**

```sio
var v: Vec<i32> = Vec::new()
v.reserve(100)
// Can add 100 elements without reallocating
```

#### reserve_exact

```sio
pub fn reserve_exact(&!self, additional: usize) with Alloc
```

Reserves the exact amount of additional capacity.

#### shrink_to_fit

```sio
pub fn shrink_to_fit(&!self) with Alloc
```

Shrinks the capacity to match the length.

**Example:**

```sio
var v: Vec<i32> = Vec::with_capacity(100)
v.push(1)
v.push(2)
v.shrink_to_fit()
// capacity is now 2
```

### Modification

#### swap

```sio
pub fn swap(&!self, a: usize, b: usize) with Panic
```

Swaps two elements by index.

**Panics:** If either index is out of bounds.

**Example:**

```sio
var v = vec![1, 2, 3]
v.swap(0, 2)
// v = [3, 2, 1]
```

#### reverse

```sio
pub fn reverse(&!self)
```

Reverses the order of elements in place.

**Example:**

```sio
var v = vec![1, 2, 3]
v.reverse()
// v = [3, 2, 1]
```

#### fill

```sio
pub fn fill(&!self, value: T)
where T: Clone
```

Fills the vector with the given value.

**Example:**

```sio
var v = vec![1, 2, 3]
v.fill(0)
// v = [0, 0, 0]
```

#### fill_with

```sio
pub fn fill_with<F>(&!self, f: F)
where F: FnMut() -> T
```

Fills the vector using a closure to generate values.

#### resize

```sio
pub fn resize(&!self, new_len: usize, value: T) with Alloc
where T: Clone
```

Resizes the vector to the new length, filling with `value` if growing.

**Example:**

```sio
var v = vec![1, 2]
v.resize(5, 0)
// v = [1, 2, 0, 0, 0]

v.resize(2, 0)
// v = [1, 2]
```

#### resize_with

```sio
pub fn resize_with<F>(&!self, new_len: usize, f: F) with Alloc
where F: FnMut() -> T
```

Resizes the vector using a closure to create new elements.

#### dedup

```sio
pub fn dedup(&!self)
where T: Eq
```

Removes consecutive duplicate elements.

**Example:**

```sio
var v = vec![1, 1, 2, 2, 2, 3]
v.dedup()
// v = [1, 2, 3]
```

#### dedup_by

```sio
pub fn dedup_by<F>(&!self, same: F)
where F: FnMut(&T, &T) -> bool
```

Removes consecutive elements where the predicate returns `true`.

### Splitting

#### split_off

```sio
pub fn split_off(&!self, at: usize) -> Vec<T> with Alloc, Panic
```

Splits the vector at the given index, returning elements from `at` onwards.

**Panics:** If `at > len`.

**Example:**

```sio
var v = vec![1, 2, 3, 4, 5]
let second_half = v.split_off(3)
// v = [1, 2, 3]
// second_half = [4, 5]
```

### Iteration

#### iter

```sio
pub fn iter(&self) -> Iter<T>
```

Returns an iterator over references to elements.

**Example:**

```sio
let v = vec![1, 2, 3]
for item in v.iter() {
    println(item.to_string())
}
```

#### iter_mut

```sio
pub fn iter_mut(&!self) -> IterMut<T>
```

Returns an iterator over mutable references to elements.

**Example:**

```sio
var v = vec![1, 2, 3]
for item in v.iter_mut() {
    *item = *item * 2
}
// v = [2, 4, 6]
```

### Raw Access

#### as_ptr

```sio
pub fn as_ptr(&self) -> *const T
```

Returns a raw pointer to the vector's buffer.

#### as_mut_ptr

```sio
pub fn as_mut_ptr(&!self) -> *mut T
```

Returns a mutable raw pointer to the vector's buffer.

## Trait Implementations

### Index and IndexMut

```sio
let v = vec![10, 20, 30]
let x = v[1]      // 20
v[1] = 25         // requires &!

// Range indexing
let slice = v[1..3]  // [20, 30]
```

### Clone

```sio
let v1 = vec![1, 2, 3]
let v2 = v1.clone()  // with Alloc
```

### Eq and Ord

```sio
let a = vec![1, 2, 3]
let b = vec![1, 2, 3]
a == b  // true

let c = vec![1, 2, 4]
a < c   // true (lexicographic comparison)
```

### Debug

```sio
let v = vec![1, 2, 3]
// Debug format: [1, 2, 3]
```

### IntoIterator

```sio
let v = vec![1, 2, 3]
for item in v {  // Consumes v
    println(item.to_string())
}
```

### FromIterator

```sio
let v: Vec<i32> = (1..=5).collect()
// v = [1, 2, 3, 4, 5]
```

### Extend

```sio
var v = vec![1, 2]
v.extend([3, 4, 5])
// v = [1, 2, 3, 4, 5]
```

### From<&[T]>

```sio
let arr = [1, 2, 3]
let v = Vec::from(&arr)
```

## Common Patterns

### Building a Vector

```sio
// From literal
let v = vec![1, 2, 3]

// Iteratively
var v: Vec<i32> = Vec::new()
for i in 0..10 {
    v.push(i)
}

// From iterator
let v: Vec<i32> = (0..10).collect()

// With initial capacity
var v: Vec<i32> = Vec::with_capacity(1000)
for i in 0..1000 {
    v.push(i)  // No reallocations
}
```

### Filtering and Mapping

```sio
let numbers = vec![1, 2, 3, 4, 5]

// Filter
let even: Vec<i32> = numbers.iter()
    .filter(|x| *x % 2 == 0)
    .copied()
    .collect()

// Map
let doubled: Vec<i32> = numbers.iter()
    .map(|x| x * 2)
    .collect()

// Filter and map
let result: Vec<i32> = numbers.iter()
    .filter(|x| *x % 2 == 0)
    .map(|x| x * x)
    .collect()
```

### Stack Operations

```sio
var stack: Vec<i32> = Vec::new()

// Push (add to top)
stack.push(1)
stack.push(2)
stack.push(3)

// Pop (remove from top)
while let Some(top) = stack.pop() {
    println(top.to_string())
}
// Prints: 3, 2, 1
```

### Sorting

```sio
var v = vec![3, 1, 4, 1, 5, 9, 2, 6]

// Standard sort
v.sort()
// v = [1, 1, 2, 3, 4, 5, 6, 9]

// Custom comparator
v.sort_by(|a, b| b.cmp(a))  // Descending
```

## Performance Considerations

- `push` is amortized O(1) due to capacity doubling
- `insert` and `remove` are O(n) as they shift elements
- `swap_remove` is O(1) but doesn't preserve order
- Pre-allocating with `with_capacity` avoids reallocations
- Iteration is cache-friendly due to contiguous memory

## See Also

- [Iterator](../iter.md) - For lazy sequence processing
- [HashMap](hashmap.md) - For key-value storage
- [Option](../core/option.md) - For handling missing values
