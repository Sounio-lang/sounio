---
title: Iterator
description: Lazy sequence processing with functional combinators
---

# Iterator

The `iter` module provides lazy sequence processing through the `Iterator` trait and a rich set of adapters and combinators. Iterators are the foundation of functional-style data processing in Sounio.

## Iterator Trait

### Definition

```sio
pub trait Iterator {
    /// The type of elements yielded by this iterator.
    type Item;

    /// Advances the iterator and returns the next value.
    fn next(&!self) -> Option<Self::Item>;

    /// Returns bounds on the remaining length.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}
```

### Basic Usage

```sio
let numbers = vec![1, 2, 3, 4, 5]

// Manual iteration
var iter = numbers.iter()
while let Some(n) = iter.next() {
    println(n.to_string())
}

// For loop (syntactic sugar)
for n in numbers.iter() {
    println(n.to_string())
}
```

## Adapters

Adapters transform iterators into new iterators without consuming elements immediately.

### map

```sio
fn map<B, F>(self, f: F) -> Map<Self, F>
where F: FnMut(Self::Item) -> B
```

Transforms each element using a function.

**Example:**

```sio
let numbers = vec![1, 2, 3]
let doubled: Vec<i32> = numbers.iter()
    .map(|x| x * 2)
    .collect()
// doubled = [2, 4, 6]
```

### filter

```sio
fn filter<P>(self, predicate: P) -> Filter<Self, P>
where P: FnMut(&Self::Item) -> bool
```

Yields only elements that satisfy the predicate.

**Example:**

```sio
let numbers = vec![1, 2, 3, 4, 5, 6]
let even: Vec<i32> = numbers.iter()
    .filter(|x| *x % 2 == 0)
    .collect()
// even = [2, 4, 6]
```

### filter_map

```sio
fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F>
where F: FnMut(Self::Item) -> Option<B>
```

Filters and maps in one operation. Yields only `Some` values.

**Example:**

```sio
let strings = vec!["1", "two", "3", "four", "5"]
let numbers: Vec<i32> = strings.iter()
    .filter_map(|s| s.parse::<i32>().ok())
    .collect()
// numbers = [1, 3, 5]
```

### flat_map

```sio
fn flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F>
where
    F: FnMut(Self::Item) -> U,
    U: IntoIterator
```

Maps then flattens nested iterators.

**Example:**

```sio
let words = vec!["hello", "world"]
let chars: Vec<char> = words.iter()
    .flat_map(|s| s.chars())
    .collect()
// chars = ['h', 'e', 'l', 'l', 'o', 'w', 'o', 'r', 'l', 'd']
```

### flatten

```sio
fn flatten(self) -> Flatten<Self>
where Self::Item: IntoIterator
```

Flattens nested iterators.

**Example:**

```sio
let nested = vec![vec![1, 2], vec![3, 4], vec![5]]
let flat: Vec<i32> = nested.iter()
    .flatten()
    .collect()
// flat = [1, 2, 3, 4, 5]
```

### take

```sio
fn take(self, n: usize) -> Take<Self>
```

Yields at most `n` elements.

**Example:**

```sio
let first_three: Vec<i32> = (0..100).take(3).collect()
// first_three = [0, 1, 2]
```

### take_while

```sio
fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P>
where P: FnMut(&Self::Item) -> bool
```

Yields elements while the predicate is true.

**Example:**

```sio
let result: Vec<i32> = vec![1, 2, 3, 10, 4, 5].iter()
    .take_while(|x| *x < 10)
    .collect()
// result = [1, 2, 3]
```

### skip

```sio
fn skip(self, n: usize) -> Skip<Self>
```

Skips the first `n` elements.

**Example:**

```sio
let rest: Vec<i32> = vec![1, 2, 3, 4, 5].iter()
    .skip(2)
    .collect()
// rest = [3, 4, 5]
```

### skip_while

```sio
fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
where P: FnMut(&Self::Item) -> bool
```

Skips elements while the predicate is true.

**Example:**

```sio
let result: Vec<i32> = vec![1, 2, 3, 10, 4, 5].iter()
    .skip_while(|x| *x < 10)
    .collect()
// result = [10, 4, 5]
```

### enumerate

```sio
fn enumerate(self) -> Enumerate<Self>
```

Yields `(index, element)` pairs.

**Example:**

```sio
let names = vec!["Alice", "Bob", "Carol"]
for (i, name) in names.iter().enumerate() {
    println(i.to_string() ++ ": " ++ name)
}
// 0: Alice
// 1: Bob
// 2: Carol
```

### peekable

```sio
fn peekable(self) -> Peekable<Self>
```

Creates an iterator that can peek at the next element without consuming it.

**Example:**

```sio
var iter = vec![1, 2, 3].iter().peekable()

// Look without consuming
if let Some(first) = iter.peek() {
    println("Next: " ++ first.to_string())
}

// Now consume
let actual = iter.next()  // Some(1)
```

### chain

```sio
fn chain<U>(self, other: U) -> Chain<Self, U::IntoIter>
where U: IntoIterator<Item = Self::Item>
```

Chains two iterators together.

**Example:**

```sio
let combined: Vec<i32> = vec![1, 2]
    .iter()
    .chain(vec![3, 4])
    .collect()
// combined = [1, 2, 3, 4]
```

### zip

```sio
fn zip<U>(self, other: U) -> Zip<Self, U::IntoIter>
where U: IntoIterator
```

Zips two iterators into pairs.

**Example:**

```sio
let names = vec!["Alice", "Bob"]
let scores = vec![100, 85]

let results: Vec<(&str, i32)> = names.iter()
    .zip(scores)
    .collect()
// results = [("Alice", 100), ("Bob", 85)]
```

### inspect

```sio
fn inspect<F>(self, f: F) -> Inspect<Self, F>
where F: FnMut(&Self::Item)
```

Calls a closure on each element for side effects (debugging).

**Example:**

```sio
let result: Vec<i32> = vec![1, 2, 3]
    .iter()
    .inspect(|x| println("Processing: " ++ x.to_string()))
    .map(|x| x * 2)
    .collect()
```

### intersperse

```sio
fn intersperse(self, separator: Self::Item) -> Intersperse<Self>
where Self::Item: Clone
```

Inserts a separator between each element.

**Example:**

```sio
let result: Vec<i32> = vec![1, 2, 3]
    .iter()
    .intersperse(0)
    .collect()
// result = [1, 0, 2, 0, 3]
```

### rev

```sio
fn rev(self) -> Rev<Self>
where Self: DoubleEndedIterator
```

Reverses the iteration direction.

**Example:**

```sio
let reversed: Vec<i32> = vec![1, 2, 3]
    .iter()
    .rev()
    .collect()
// reversed = [3, 2, 1]
```

### cycle

```sio
fn cycle(self) -> Cycle<Self>
where Self: Clone
```

Repeats the iterator endlessly.

**Example:**

```sio
let repeated: Vec<i32> = vec![1, 2, 3]
    .iter()
    .cycle()
    .take(7)
    .collect()
// repeated = [1, 2, 3, 1, 2, 3, 1]
```

### fuse

```sio
fn fuse(self) -> Fuse<Self>
```

Creates an iterator that yields `None` forever after the first `None`.

### step_by

```sio
fn step_by(self, step: usize) -> StepBy<Self>
```

Yields every `step`-th element.

**Example:**

```sio
let every_other: Vec<i32> = (0..10)
    .step_by(2)
    .collect()
// every_other = [0, 2, 4, 6, 8]
```

## Consumers

Consumers execute the iterator chain and produce a final result.

### fold

```sio
fn fold<B, F>(self, init: B, f: F) -> B
where F: FnMut(B, Self::Item) -> B
```

Folds all elements into a single accumulator value.

**Example:**

```sio
let sum = vec![1, 2, 3, 4, 5]
    .iter()
    .fold(0, |acc, x| acc + x)
// sum = 15

let product = vec![1, 2, 3, 4]
    .iter()
    .fold(1, |acc, x| acc * x)
// product = 24
```

### reduce

```sio
fn reduce<F>(self, f: F) -> Option<Self::Item>
where F: FnMut(Self::Item, Self::Item) -> Self::Item
```

Reduces elements to a single value using the first element as initial accumulator.

**Example:**

```sio
let max = vec![3, 1, 4, 1, 5, 9]
    .iter()
    .reduce(|a, b| if a > b { a } else { b })
// max = Some(9)
```

### for_each

```sio
fn for_each<F>(self, f: F)
where F: FnMut(Self::Item)
```

Calls a closure on each element.

**Example:**

```sio
vec![1, 2, 3].iter()
    .for_each(|x| println(x.to_string()))
```

### collect

```sio
fn collect<B: FromIterator<Self::Item>>(self) -> B
```

Collects elements into a collection.

**Example:**

```sio
let vec: Vec<i32> = (1..=5).collect()
let set: HashSet<i32> = (1..=5).collect()
```

### count

```sio
fn count(self) -> usize
```

Counts the number of elements.

**Example:**

```sio
let count = vec![1, 2, 3, 4, 5]
    .iter()
    .filter(|x| *x % 2 == 0)
    .count()
// count = 2
```

### last

```sio
fn last(self) -> Option<Self::Item>
```

Returns the last element.

**Example:**

```sio
let last = vec![1, 2, 3].iter().last()
// last = Some(3)
```

### nth

```sio
fn nth(&!self, n: usize) -> Option<Self::Item>
```

Returns the `n`-th element (0-indexed).

**Example:**

```sio
var iter = vec![10, 20, 30, 40].iter()
let third = iter.nth(2)  // Some(30)
```

### first

```sio
fn first(self) -> Option<Self::Item>
```

Returns the first element.

### find

```sio
fn find<P>(&!self, predicate: P) -> Option<Self::Item>
where P: FnMut(&Self::Item) -> bool
```

Finds the first element satisfying the predicate.

**Example:**

```sio
let numbers = vec![1, 2, 3, 4, 5]
let first_even = numbers.iter()
    .find(|x| *x % 2 == 0)
// first_even = Some(2)
```

### find_map

```sio
fn find_map<B, F>(&!self, f: F) -> Option<B>
where F: FnMut(Self::Item) -> Option<B>
```

Finds and maps the first `Some` result.

### position

```sio
fn position<P>(&!self, predicate: P) -> Option<usize>
where P: FnMut(Self::Item) -> bool
```

Returns the index of the first element satisfying the predicate.

**Example:**

```sio
let pos = vec!['a', 'b', 'c', 'd']
    .iter()
    .position(|c| *c == 'c')
// pos = Some(2)
```

### any

```sio
fn any<P>(&!self, predicate: P) -> bool
where P: FnMut(Self::Item) -> bool
```

Tests if any element satisfies the predicate.

**Example:**

```sio
let has_negative = vec![1, -2, 3]
    .iter()
    .any(|x| *x < 0)
// has_negative = true
```

### all

```sio
fn all<P>(&!self, predicate: P) -> bool
where P: FnMut(Self::Item) -> bool
```

Tests if all elements satisfy the predicate.

**Example:**

```sio
let all_positive = vec![1, 2, 3]
    .iter()
    .all(|x| *x > 0)
// all_positive = true
```

### max / min

```sio
fn max(self) -> Option<Self::Item>
where Self::Item: Ord

fn min(self) -> Option<Self::Item>
where Self::Item: Ord
```

Returns the maximum or minimum element.

**Example:**

```sio
let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6]
let max = numbers.iter().max()  // Some(9)
let min = numbers.iter().min()  // Some(1)
```

### max_by_key / min_by_key

```sio
fn max_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item>
where F: FnMut(&Self::Item) -> B

fn min_by_key<B: Ord, F>(self, f: F) -> Option<Self::Item>
where F: FnMut(&Self::Item) -> B
```

Returns max/min by a key function.

**Example:**

```sio
let words = vec!["apple", "fig", "banana", "cherry"]
let longest = words.iter()
    .max_by_key(|s| s.len())
// longest = Some("banana")
```

### sum / product

```sio
fn sum<S: Sum<Self::Item>>(self) -> S
fn product<P: Product<Self::Item>>(self) -> P
```

Sums or multiplies all elements.

**Example:**

```sio
let sum: i32 = vec![1, 2, 3, 4, 5].iter().sum()
// sum = 15

let product: i32 = vec![1, 2, 3, 4].iter().product()
// product = 24
```

### partition

```sio
fn partition<B, F>(self, f: F) -> (B, B)
where
    B: Default + Extend<Self::Item>,
    F: FnMut(&Self::Item) -> bool
```

Partitions elements into two collections.

**Example:**

```sio
let numbers = vec![1, 2, 3, 4, 5, 6]
let (even, odd): (Vec<i32>, Vec<i32>) = numbers.iter()
    .partition(|x| *x % 2 == 0)
// even = [2, 4, 6]
// odd = [1, 3, 5]
```

### unzip

```sio
fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
where
    Self: Iterator<Item = (A, B)>,
    FromA: Default + Extend<A>,
    FromB: Default + Extend<B>
```

Unzips pairs into two collections.

**Example:**

```sio
let pairs = vec![(1, "a"), (2, "b"), (3, "c")]
let (nums, letters): (Vec<i32>, Vec<&str>) = pairs.iter().unzip()
// nums = [1, 2, 3]
// letters = ["a", "b", "c"]
```

### eq / cmp

```sio
fn eq<I>(self, other: I) -> bool
where I: IntoIterator<Item = Self::Item>, Self::Item: Eq

fn cmp<I>(self, other: I) -> Ordering
where I: IntoIterator<Item = Self::Item>, Self::Item: Ord
```

Compares two iterators element by element.

## Related Traits

### DoubleEndedIterator

```sio
pub trait DoubleEndedIterator: Iterator {
    fn next_back(&!self) -> Option<Self::Item>;
    fn nth_back(&!self, n: usize) -> Option<Self::Item>;
    fn rfold<B, F>(self, init: B, f: F) -> B;
    fn rfind<P>(&!self, predicate: P) -> Option<Self::Item>;
}
```

Allows iteration from both ends.

### ExactSizeIterator

```sio
pub trait ExactSizeIterator: Iterator {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}
```

For iterators with known exact length.

### FusedIterator

```sio
pub trait FusedIterator: Iterator {}
```

Marker trait for iterators that always return `None` after the first `None`.

### IntoIterator

```sio
pub trait IntoIterator {
    type Item;
    type IntoIter: Iterator<Item = Self::Item>;
    fn into_iter(self) -> Self::IntoIter;
}
```

Enables types to be used in `for` loops.

### FromIterator

```sio
pub trait FromIterator<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
}
```

For creating collections from iterators via `collect()`.

### Extend

```sio
pub trait Extend<A> {
    fn extend<T: IntoIterator<Item = A>>(&!self, iter: T);
}
```

For extending collections with iterator elements.

## Utility Functions

### empty

```sio
pub fn empty<T>() -> Empty<T>
```

Creates an iterator that yields nothing.

**Example:**

```sio
let nothing: Vec<i32> = empty().collect()
// nothing = []
```

### once

```sio
pub fn once<T>(value: T) -> Once<T>
```

Creates an iterator that yields exactly one element.

**Example:**

```sio
let single: Vec<i32> = once(42).collect()
// single = [42]
```

### repeat

```sio
pub fn repeat<T: Clone>(value: T) -> Repeat<T>
```

Creates an iterator that yields an element forever.

**Example:**

```sio
let zeros: Vec<i32> = repeat(0).take(5).collect()
// zeros = [0, 0, 0, 0, 0]
```

### repeat_with

```sio
pub fn repeat_with<T, F: FnMut() -> T>(f: F) -> RepeatWith<F>
```

Creates an iterator from a closure called repeatedly.

**Example:**

```sio
var counter = 0
let values: Vec<i32> = repeat_with(|| {
    counter = counter + 1
    counter
}).take(5).collect()
// values = [1, 2, 3, 4, 5]
```

### from_fn

```sio
pub fn from_fn<T, F: FnMut() -> Option<T>>(f: F) -> FromFn<F>
```

Creates an iterator from a closure returning `Option`.

**Example:**

```sio
var n = 0
let fibonacci = from_fn(|| {
    let (curr, next) = (n, n + 1)
    n = next
    Some(curr)
})

let first_ten: Vec<i32> = fibonacci.take(10).collect()
```

### successors

```sio
pub fn successors<T, F: FnMut(&T) -> Option<T>>(first: Option<T>, f: F) -> Successors<T, F>
```

Creates an iterator of successive values.

**Example:**

```sio
let powers_of_two: Vec<i32> = successors(Some(1), |n| {
    let next = n * 2
    if next <= 1024 { Some(next) } else { None }
}).collect()
// powers_of_two = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
```

## Common Patterns

### Processing Data Pipelines

```sio
let result = data
    .iter()
    .filter(|item| item.is_valid())
    .map(|item| transform(item))
    .take(100)
    .collect::<Vec<_>>()
```

### Lazy Evaluation

```sio
// Nothing happens until collect() is called
let pipeline = huge_dataset
    .iter()
    .filter(|x| expensive_check(x))
    .map(|x| expensive_transform(x))

// Now it executes
let results: Vec<_> = pipeline.take(10).collect()
```

### Aggregating Statistics

```sio
fn statistics(data: &[f64]) -> (f64, f64, f64) {
    let sum: f64 = data.iter().sum()
    let count = data.len() as f64
    let mean = sum / count

    let variance: f64 = data.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / count

    let min = data.iter().fold(f64::INFINITY, |a, b| a.min(*b))
    let max = data.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b))

    (mean, variance.sqrt(), max - min)
}
```

### Grouping with fold

```sio
fn group_by_length(words: &[String]) -> HashMap<usize, Vec<String>> {
    words.iter()
        .fold(HashMap::new(), |mut groups, word| {
            groups.entry(word.len())
                .or_insert(Vec::new())
                .push(word.clone());
            groups
        })
}
```

## See Also

- [Vec<T>](collections/vec.md) - Common iterator source
- [HashMap<K, V>](collections/hashmap.md) - Another iterator source
- [Option<T>](core/option.md) - Iterator return type
