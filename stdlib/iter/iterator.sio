// stdlib/iter/iterator.d - Iterator trait and combinators
//
// Lazy sequence processing with functional combinators.

module std.iter.iterator;

import std.core.option;
import std.core.result;
import std.cmp;
import std.ops;

/// The core iterator trait for lazy sequence processing.
///
/// # Examples
/// ```
/// let sum: i32 = [1, 2, 3, 4, 5]
///     .iter()
///     .filter(|x| x % 2 == 0)
///     .map(|x| x * 2)
///     .sum();
/// assert_eq!(sum, 12);  // (2 + 4) * 2
/// ```
pub trait Iterator {
    /// The type of elements yielded by this iterator.
    type Item;

    /// Advances the iterator and returns the next value.
    fn next(&!self) -> Option<Self.Item>;

    /// Returns the bounds on the remaining length.
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }

    // ========================================================================
    // Adapters (return new iterators)
    // ========================================================================

    /// Creates an iterator that yields elements satisfying the predicate.
    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where
        P: FnMut(&Self.Item) -> bool,
        Self: Sized,
    {
        Filter { iter: self, predicate }
    }

    /// Creates an iterator that transforms each element.
    fn map<B, F>(self, f: F) -> Map<Self, F>
    where
        F: FnMut(Self.Item) -> B,
        Self: Sized,
    {
        Map { iter: self, f }
    }

    /// Creates an iterator that both filters and maps.
    fn filter_map<B, F>(self, f: F) -> FilterMap<Self, F>
    where
        F: FnMut(Self.Item) -> Option<B>,
        Self: Sized,
    {
        FilterMap { iter: self, f }
    }

    /// Creates an iterator that flattens nested structure.
    fn flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F>
    where
        F: FnMut(Self.Item) -> U,
        U: IntoIterator,
        Self: Sized,
    {
        FlatMap {
            iter: self,
            f,
            front: None
        }
    }

    /// Flattens an iterator of iterators.
    fn flatten(self) -> Flatten<Self>
    where
        Self.Item: IntoIterator,
        Self: Sized,
    {
        Flatten {
            iter: self,
            front: None
        }
    }

    /// Creates an iterator that yields at most n elements.
    fn take(self, n: usize) -> Take<Self>
    where
        Self: Sized,
    {
        Take { iter: self, remaining: n }
    }

    /// Creates an iterator that yields elements while predicate is true.
    fn take_while<P>(self, predicate: P) -> TakeWhile<Self, P>
    where
        P: FnMut(&Self.Item) -> bool,
        Self: Sized,
    {
        TakeWhile { iter: self, predicate, done: false }
    }

    /// Creates an iterator that skips the first n elements.
    fn skip(self, n: usize) -> Skip<Self>
    where
        Self: Sized,
    {
        Skip { iter: self, remaining: n }
    }

    /// Creates an iterator that skips elements while predicate is true.
    fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
    where
        P: FnMut(&Self.Item) -> bool,
        Self: Sized,
    {
        SkipWhile { iter: self, predicate, done: false }
    }

    /// Creates an iterator that yields (index, element) pairs.
    fn enumerate(self) -> Enumerate<Self>
    where
        Self: Sized,
    {
        Enumerate { iter: self, count: 0 }
    }

    /// Creates an iterator that peeks at the next element.
    fn peekable(self) -> Peekable<Self>
    where
        Self: Sized,
    {
        Peekable { iter: self, peeked: None }
    }

    /// Creates an iterator that chains two iterators.
    fn chain<U>(self, other: U) -> Chain<Self, U.IntoIter>
    where
        U: IntoIterator<Item = Self.Item>,
        Self: Sized,
    {
        Chain {
            first: Some(self),
            second: other.into_iter()
        }
    }

    /// Creates an iterator that zips two iterators.
    fn zip<U>(self, other: U) -> Zip<Self, U.IntoIter>
    where
        U: IntoIterator,
        Self: Sized,
    {
        Zip {
            a: self,
            b: other.into_iter()
        }
    }

    /// Creates an iterator that calls a closure on each element.
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        F: FnMut(&Self.Item),
        Self: Sized,
    {
        Inspect { iter: self, f }
    }

    /// Creates an iterator that repeats each element n times.
    fn intersperse(self, separator: Self.Item) -> Intersperse<Self>
    where
        Self.Item: Clone,
        Self: Sized,
    {
        Intersperse {
            iter: self,
            separator,
            needs_sep: false
        }
    }

    /// Creates a reversible iterator (requires DoubleEndedIterator).
    fn rev(self) -> Rev<Self>
    where
        Self: DoubleEndedIterator + Sized,
    {
        Rev { iter: self }
    }

    /// Creates an iterator that yields borrowed elements.
    fn by_ref(&!self) -> &!Self {
        self
    }

    /// Creates an iterator that cycles forever.
    fn cycle(self) -> Cycle<Self>
    where
        Self: Clone + Sized,
    {
        Cycle {
            orig: self.clone(),
            iter: self
        }
    }

    /// Creates an iterator with elements grouped by key.
    fn fuse(self) -> Fuse<Self>
    where
        Self: Sized,
    {
        Fuse { iter: Some(self) }
    }

    /// Creates a stepping iterator.
    fn step_by(self, step: usize) -> StepBy<Self>
    where
        Self: Sized,
    {
        assert!(step > 0, "step must be positive");
        StepBy { iter: self, step, first: true }
    }

    // ========================================================================
    // Consumers (consume the iterator)
    // ========================================================================

    /// Folds every element into an accumulator.
    fn fold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self.Item) -> B,
        Self: Sized,
    {
        let mut acc = init;
        for item in self {
            acc = f(acc, item);
        }
        acc
    }

    /// Reduces elements to a single value using a closure.
    fn reduce<F>(self, f: F) -> Option<Self.Item>
    where
        F: FnMut(Self.Item, Self.Item) -> Self.Item,
        Self: Sized,
    {
        let mut iter = self;
        let first = iter.next()?;
        Some(iter.fold(first, f))
    }

    /// Calls a closure on each element.
    fn for_each<F>(self, mut f: F)
    where
        F: FnMut(Self.Item),
        Self: Sized,
    {
        for item in self {
            f(item);
        }
    }

    /// Collects elements into a collection.
    fn collect<B: FromIterator<Self.Item>>(self) -> B
    where
        Self: Sized,
    {
        B.from_iter(self)
    }

    /// Counts the number of elements.
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.fold(0, |acc, _| acc + 1)
    }

    /// Returns the last element.
    fn last(self) -> Option<Self.Item>
    where
        Self: Sized,
    {
        self.fold(None, |_, item| Some(item))
    }

    /// Returns the nth element.
    fn nth(&!self, n: usize) -> Option<Self.Item> {
        for _ in 0..n {
            self.next()?;
        }
        self.next()
    }

    /// Returns the first element.
    fn first(self) -> Option<Self.Item>
    where
        Self: Sized,
    {
        let mut iter = self;
        iter.next()
    }

    /// Returns the first element satisfying the predicate.
    fn find<P>(&!self, mut predicate: P) -> Option<Self.Item>
    where
        P: FnMut(&Self.Item) -> bool,
    {
        for item in self {
            if predicate(&item) {
                return Some(item);
            }
        }
        None
    }

    /// Returns the first mapped value that is Some.
    fn find_map<B, F>(&!self, mut f: F) -> Option<B>
    where
        F: FnMut(Self.Item) -> Option<B>,
    {
        for item in self {
            if let Some(b) = f(item) {
                return Some(b);
            }
        }
        None
    }

    /// Returns the position of the first element satisfying the predicate.
    fn position<P>(&!self, mut predicate: P) -> Option<usize>
    where
        P: FnMut(Self.Item) -> bool,
    {
        let mut i = 0;
        for item in self {
            if predicate(item) {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    /// Tests if any element satisfies the predicate.
    fn any<P>(&!self, mut predicate: P) -> bool
    where
        P: FnMut(Self.Item) -> bool,
    {
        for item in self {
            if predicate(item) {
                return true;
            }
        }
        false
    }

    /// Tests if all elements satisfy the predicate.
    fn all<P>(&!self, mut predicate: P) -> bool
    where
        P: FnMut(Self.Item) -> bool,
    {
        for item in self {
            if !predicate(item) {
                return false;
            }
        }
        true
    }

    /// Returns the maximum element.
    fn max(self) -> Option<Self.Item>
    where
        Self.Item: Ord,
        Self: Sized,
    {
        self.reduce(|a, b| if a >= b { a } else { b })
    }

    /// Returns the minimum element.
    fn min(self) -> Option<Self.Item>
    where
        Self.Item: Ord,
        Self: Sized,
    {
        self.reduce(|a, b| if a <= b { a } else { b })
    }

    /// Returns the maximum element by a key function.
    fn max_by_key<B: Ord, F>(self, mut f: F) -> Option<Self.Item>
    where
        F: FnMut(&Self.Item) -> B,
        Self: Sized,
    {
        self.reduce(|a, b| if f(&a) >= f(&b) { a } else { b })
    }

    /// Returns the minimum element by a key function.
    fn min_by_key<B: Ord, F>(self, mut f: F) -> Option<Self.Item>
    where
        F: FnMut(&Self.Item) -> B,
        Self: Sized,
    {
        self.reduce(|a, b| if f(&a) <= f(&b) { a } else { b })
    }

    /// Returns the maximum element by a comparison function.
    fn max_by<F>(self, mut compare: F) -> Option<Self.Item>
    where
        F: FnMut(&Self.Item, &Self.Item) -> Ordering,
        Self: Sized,
    {
        self.reduce(|a, b| {
            match compare(&a, &b) {
                Ordering.Less => b,
                _ => a,
            }
        })
    }

    /// Returns the minimum element by a comparison function.
    fn min_by<F>(self, mut compare: F) -> Option<Self.Item>
    where
        F: FnMut(&Self.Item, &Self.Item) -> Ordering,
        Self: Sized,
    {
        self.reduce(|a, b| {
            match compare(&a, &b) {
                Ordering.Greater => b,
                _ => a,
            }
        })
    }

    /// Sums the elements.
    fn sum<S: Sum<Self.Item>>(self) -> S
    where
        Self: Sized,
    {
        S.sum(self)
    }

    /// Multiplies the elements.
    fn product<P: Product<Self.Item>>(self) -> P
    where
        Self: Sized,
    {
        P.product(self)
    }

    /// Compares two iterators element by element.
    fn cmp<I>(self, other: I) -> Ordering
    where
        I: IntoIterator<Item = Self.Item>,
        Self.Item: Ord,
        Self: Sized,
    {
        let mut other = other.into_iter();

        for item in self {
            match other.next() {
                None => return Ordering.Greater,
                Some(other_item) => {
                    match item.cmp(&other_item) {
                        Ordering.Equal => continue,
                        ord => return ord,
                    }
                }
            }
        }

        if other.next().is_some() {
            Ordering.Less
        } else {
            Ordering.Equal
        }
    }

    /// Tests equality of two iterators.
    fn eq<I>(self, other: I) -> bool
    where
        I: IntoIterator<Item = Self.Item>,
        Self.Item: Eq,
        Self: Sized,
    {
        let mut other = other.into_iter();

        for item in self {
            match other.next() {
                None => return false,
                Some(other_item) if item != other_item => return false,
                _ => continue,
            }
        }

        other.next().is_none()
    }

    /// Partitions elements into two collections.
    fn partition<B, F>(self, mut f: F) -> (B, B)
    where
        B: Default + Extend<Self.Item>,
        F: FnMut(&Self.Item) -> bool,
        Self: Sized,
    {
        let mut left = B.default();
        let mut right = B.default();

        for item in self {
            if f(&item) {
                left.extend(once(item));
            } else {
                right.extend(once(item));
            }
        }

        (left, right)
    }

    /// Unzips an iterator of pairs into two collections.
    fn unzip<A, B, FromA, FromB>(self) -> (FromA, FromB)
    where
        Self: Iterator<Item = (A, B)> + Sized,
        FromA: Default + Extend<A>,
        FromB: Default + Extend<B>,
    {
        let mut a = FromA.default();
        let mut b = FromB.default();

        for (x, y) in self {
            a.extend(once(x));
            b.extend(once(y));
        }

        (a, b)
    }

    /// Copies elements to a slice.
    fn copy_to_slice(&!self, slice: &![Self.Item]) -> usize
    where
        Self.Item: Copy,
    {
        let mut i = 0;
        for item in self {
            if i >= slice.len() {
                break;
            }
            slice[i] = item;
            i += 1;
        }
        i
    }

    /// Advances the iterator by n elements.
    fn advance_by(&!self, n: usize) -> Result<(), usize> {
        for i in 0..n {
            if self.next().is_none() {
                return Err(i);
            }
        }
        Ok(())
    }
}

// ============================================================================
// Double-Ended Iterator
// ============================================================================

/// An iterator that can be traversed from both ends.
pub trait DoubleEndedIterator: Iterator {
    /// Removes and returns an element from the back.
    fn next_back(&!self) -> Option<Self.Item>;

    /// Returns the nth element from the back.
    fn nth_back(&!self, n: usize) -> Option<Self.Item> {
        for _ in 0..n {
            self.next_back()?;
        }
        self.next_back()
    }

    /// Folds from the back.
    fn rfold<B, F>(self, init: B, mut f: F) -> B
    where
        F: FnMut(B, Self.Item) -> B,
        Self: Sized,
    {
        let mut acc = init;
        let mut iter = self;
        while let Some(item) = iter.next_back() {
            acc = f(acc, item);
        }
        acc
    }

    /// Finds the last element satisfying the predicate.
    fn rfind<P>(&!self, mut predicate: P) -> Option<Self.Item>
    where
        P: FnMut(&Self.Item) -> bool,
    {
        while let Some(item) = self.next_back() {
            if predicate(&item) {
                return Some(item);
            }
        }
        None
    }
}

// ============================================================================
// Exact Size Iterator
// ============================================================================

/// An iterator with an exact size.
pub trait ExactSizeIterator: Iterator {
    /// Returns the exact length.
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        debug_assert_eq!(Some(lower), upper);
        lower
    }

    /// Returns true if the iterator is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ============================================================================
// FusedIterator
// ============================================================================

/// An iterator that yields None forever after returning None once.
pub trait FusedIterator: Iterator {}

// ============================================================================
// IntoIterator
// ============================================================================

/// Conversion into an iterator.
pub trait IntoIterator {
    type Item;
    type IntoIter: Iterator<Item = Self.Item>;

    fn into_iter(self) -> Self.IntoIter;
}

// All iterators implement IntoIterator
impl<I: Iterator> IntoIterator for I {
    type Item = I.Item;
    type IntoIter = I;

    fn into_iter(self) -> I {
        self
    }
}

// ============================================================================
// FromIterator
// ============================================================================

/// Creating a collection from an iterator.
pub trait FromIterator<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self;
}

// ============================================================================
// Extend
// ============================================================================

/// Extend a collection with elements from an iterator.
pub trait Extend<A> {
    fn extend<T: IntoIterator<Item = A>>(&!self, iter: T);
}

// ============================================================================
// Sum and Product
// ============================================================================

/// Sum trait for iterator sum().
pub trait Sum<A = Self> {
    fn sum<I: Iterator<Item = A>>(iter: I) -> Self;
}

/// Product trait for iterator product().
pub trait Product<A = Self> {
    fn product<I: Iterator<Item = A>>(iter: I) -> Self;
}

// Implement for numeric types
impl Sum for i32 {
    fn sum<I: Iterator<Item = i32>>(iter: I) -> i32 {
        iter.fold(0, |a, b| a + b)
    }
}

impl Product for i32 {
    fn product<I: Iterator<Item = i32>>(iter: I) -> i32 {
        iter.fold(1, |a, b| a * b)
    }
}

impl Sum for i64 {
    fn sum<I: Iterator<Item = i64>>(iter: I) -> i64 {
        iter.fold(0, |a, b| a + b)
    }
}

impl Product for i64 {
    fn product<I: Iterator<Item = i64>>(iter: I) -> i64 {
        iter.fold(1, |a, b| a * b)
    }
}

impl Sum for f64 {
    fn sum<I: Iterator<Item = f64>>(iter: I) -> f64 {
        iter.fold(0.0, |a, b| a + b)
    }
}

impl Product for f64 {
    fn product<I: Iterator<Item = f64>>(iter: I) -> f64 {
        iter.fold(1.0, |a, b| a * b)
    }
}

// ============================================================================
// Iterator Adapters
// ============================================================================

/// Iterator that filters elements.
pub struct Filter<I, P> {
    iter: I,
    predicate: P,
}

impl<I, P> Iterator for Filter<I, P>
where
    I: Iterator,
    P: FnMut(&I.Item) -> bool,
{
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        loop {
            let item = self.iter.next()?;
            if (self.predicate)(&item) {
                return Some(item);
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, self.iter.size_hint().1)
    }
}

/// Iterator that transforms elements.
pub struct Map<I, F> {
    iter: I,
    f: F,
}

impl<B, I, F> Iterator for Map<I, F>
where
    I: Iterator,
    F: FnMut(I.Item) -> B,
{
    type Item = B;

    fn next(&!self) -> Option<B> {
        self.iter.next().map(&!self.f)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<B, I, F> ExactSizeIterator for Map<I, F>
where
    I: ExactSizeIterator,
    F: FnMut(I.Item) -> B,
{}

impl<B, I, F> DoubleEndedIterator for Map<I, F>
where
    I: DoubleEndedIterator,
    F: FnMut(I.Item) -> B,
{
    fn next_back(&!self) -> Option<B> {
        self.iter.next_back().map(&!self.f)
    }
}

/// Iterator that filters and maps.
pub struct FilterMap<I, F> {
    iter: I,
    f: F,
}

impl<B, I, F> Iterator for FilterMap<I, F>
where
    I: Iterator,
    F: FnMut(I.Item) -> Option<B>,
{
    type Item = B;

    fn next(&!self) -> Option<B> {
        loop {
            let item = self.iter.next()?;
            if let Some(b) = (self.f)(item) {
                return Some(b);
            }
        }
    }
}

/// Iterator that flat maps.
pub struct FlatMap<I, U, F> {
    iter: I,
    f: F,
    front: Option<U.IntoIter>,
}

impl<I, U, F> Iterator for FlatMap<I, U, F>
where
    I: Iterator,
    U: IntoIterator,
    F: FnMut(I.Item) -> U,
{
    type Item = U.Item;

    fn next(&!self) -> Option<U.Item> {
        loop {
            if let Some(ref mut front) = self.front {
                if let Some(item) = front.next() {
                    return Some(item);
                }
            }

            let item = self.iter.next()?;
            self.front = Some((self.f)(item).into_iter());
        }
    }
}

/// Iterator that flattens nested iterators.
pub struct Flatten<I>
where
    I: Iterator,
    I.Item: IntoIterator,
{
    iter: I,
    front: Option<<I.Item as IntoIterator>.IntoIter>,
}

impl<I> Iterator for Flatten<I>
where
    I: Iterator,
    I.Item: IntoIterator,
{
    type Item = <I.Item as IntoIterator>.Item;

    fn next(&!self) -> Option<Self.Item> {
        loop {
            if let Some(ref mut front) = self.front {
                if let Some(item) = front.next() {
                    return Some(item);
                }
            }

            let inner = self.iter.next()?;
            self.front = Some(inner.into_iter());
        }
    }
}

/// Iterator that takes first n elements.
pub struct Take<I> {
    iter: I,
    remaining: usize,
}

impl<I: Iterator> Iterator for Take<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        if self.remaining > 0 {
            self.remaining -= 1;
            self.iter.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let lower = lower.min(self.remaining);
        let upper = upper.map(|u| u.min(self.remaining));
        (lower, upper)
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Take<I> {}

/// Iterator that takes while predicate is true.
pub struct TakeWhile<I, P> {
    iter: I,
    predicate: P,
    done: bool,
}

impl<I, P> Iterator for TakeWhile<I, P>
where
    I: Iterator,
    P: FnMut(&I.Item) -> bool,
{
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        if self.done {
            return None;
        }

        let item = self.iter.next()?;
        if (self.predicate)(&item) {
            Some(item)
        } else {
            self.done = true;
            None
        }
    }
}

/// Iterator that skips first n elements.
pub struct Skip<I> {
    iter: I,
    remaining: usize,
}

impl<I: Iterator> Iterator for Skip<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        while self.remaining > 0 {
            self.remaining -= 1;
            self.iter.next()?;
        }
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let lower = lower.saturating_sub(self.remaining);
        let upper = upper.map(|u| u.saturating_sub(self.remaining));
        (lower, upper)
    }
}

/// Iterator that skips while predicate is true.
pub struct SkipWhile<I, P> {
    iter: I,
    predicate: P,
    done: bool,
}

impl<I, P> Iterator for SkipWhile<I, P>
where
    I: Iterator,
    P: FnMut(&I.Item) -> bool,
{
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        if !self.done {
            loop {
                let item = self.iter.next()?;
                if !(self.predicate)(&item) {
                    self.done = true;
                    return Some(item);
                }
            }
        }
        self.iter.next()
    }
}

/// Iterator that yields (index, element) pairs.
pub struct Enumerate<I> {
    iter: I,
    count: usize,
}

impl<I: Iterator> Iterator for Enumerate<I> {
    type Item = (usize, I.Item);

    fn next(&!self) -> Option<(usize, I.Item)> {
        let item = self.iter.next()?;
        let idx = self.count;
        self.count += 1;
        Some((idx, item))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for Enumerate<I> {}

/// Iterator that can peek at the next element.
pub struct Peekable<I: Iterator> {
    iter: I,
    peeked: Option<Option<I.Item>>,
}

impl<I: Iterator> Peekable<I> {
    /// Peeks at the next element without consuming it.
    pub fn peek(&!self) -> Option<&I.Item> {
        if self.peeked.is_none() {
            self.peeked = Some(self.iter.next());
        }
        self.peeked.as_ref().and_then(|opt| opt.as_ref())
    }

    /// Peeks at the next element mutably.
    pub fn peek_mut(&!self) -> Option<&!I.Item> {
        if self.peeked.is_none() {
            self.peeked = Some(self.iter.next());
        }
        self.peeked.as_mut().and_then(|opt| opt.as_mut())
    }

    /// Consumes the next element if it equals the expected value.
    pub fn next_if(&!self, expected: &I.Item) -> Option<I.Item>
    where
        I.Item: Eq,
    {
        match self.peek() {
            Some(item) if item == expected => self.next(),
            _ => None,
        }
    }

    /// Consumes the next element if the predicate is true.
    pub fn next_if_eq<F>(&!self, func: F) -> Option<I.Item>
    where
        F: FnOnce(&I.Item) -> bool,
    {
        match self.peek() {
            Some(item) if func(item) => self.next(),
            _ => None,
        }
    }
}

impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        match self.peeked.take() {
            Some(peeked) => peeked,
            None => self.iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let add = if self.peeked.is_some() { 1 } else { 0 };
        (lower + add, upper.map(|u| u + add))
    }
}

/// Iterator that chains two iterators.
pub struct Chain<A, B> {
    first: Option<A>,
    second: B,
}

impl<A, B> Iterator for Chain<A, B>
where
    A: Iterator,
    B: Iterator<Item = A.Item>,
{
    type Item = A.Item;

    fn next(&!self) -> Option<A.Item> {
        if let Some(ref mut first) = self.first {
            if let Some(item) = first.next() {
                return Some(item);
            }
            self.first = None;
        }
        self.second.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.first.as_ref()
            .map(|f| f.size_hint())
            .unwrap_or((0, Some(0)));
        let (b_lower, b_upper) = self.second.size_hint();

        let lower = a_lower + b_lower;
        let upper = match (a_upper, b_upper) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        (lower, upper)
    }
}

/// Iterator that zips two iterators.
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A, B> Iterator for Zip<A, B>
where
    A: Iterator,
    B: Iterator,
{
    type Item = (A.Item, B.Item);

    fn next(&!self) -> Option<(A.Item, B.Item)> {
        let a = self.a.next()?;
        let b = self.b.next()?;
        Some((a, b))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (a_lower, a_upper) = self.a.size_hint();
        let (b_lower, b_upper) = self.b.size_hint();

        let lower = a_lower.min(b_lower);
        let upper = match (a_upper, b_upper) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        (lower, upper)
    }
}

impl<A, B> ExactSizeIterator for Zip<A, B>
where
    A: ExactSizeIterator,
    B: ExactSizeIterator,
{}

/// Iterator that calls a closure on each element for side effects.
pub struct Inspect<I, F> {
    iter: I,
    f: F,
}

impl<I, F> Iterator for Inspect<I, F>
where
    I: Iterator,
    F: FnMut(&I.Item),
{
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        let item = self.iter.next()?;
        (self.f)(&item);
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Iterator that intersperses a separator.
pub struct Intersperse<I: Iterator> {
    iter: I,
    separator: I.Item,
    needs_sep: bool,
}

impl<I> Iterator for Intersperse<I>
where
    I: Iterator,
    I.Item: Clone,
{
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        if self.needs_sep {
            self.needs_sep = false;
            Some(self.separator.clone())
        } else {
            let item = self.iter.next()?;
            self.needs_sep = true;
            Some(item)
        }
    }
}

/// Iterator that reverses iteration.
pub struct Rev<I> {
    iter: I,
}

impl<I: DoubleEndedIterator> Iterator for Rev<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        self.iter.next_back()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<I: DoubleEndedIterator> DoubleEndedIterator for Rev<I> {
    fn next_back(&!self) -> Option<I.Item> {
        self.iter.next()
    }
}

impl<I: ExactSizeIterator + DoubleEndedIterator> ExactSizeIterator for Rev<I> {}

/// Iterator that cycles forever.
pub struct Cycle<I> {
    orig: I,
    iter: I,
}

impl<I: Clone + Iterator> Iterator for Cycle<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        match self.iter.next() {
            Some(item) => Some(item),
            None => {
                self.iter = self.orig.clone();
                self.iter.next()
            }
        }
    }
}

/// Iterator that yields None forever after first None.
pub struct Fuse<I> {
    iter: Option<I>,
}

impl<I: Iterator> Iterator for Fuse<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        match self.iter {
            Some(ref mut iter) => {
                match iter.next() {
                    Some(item) => Some(item),
                    None => {
                        self.iter = None;
                        None
                    }
                }
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.iter {
            Some(ref iter) => iter.size_hint(),
            None => (0, Some(0)),
        }
    }
}

impl<I: Iterator> FusedIterator for Fuse<I> {}

/// Iterator that steps by n.
pub struct StepBy<I> {
    iter: I,
    step: usize,
    first: bool,
}

impl<I: Iterator> Iterator for StepBy<I> {
    type Item = I.Item;

    fn next(&!self) -> Option<I.Item> {
        if self.first {
            self.first = false;
            self.iter.next()
        } else {
            self.iter.nth(self.step - 1)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let step = self.step;

        let div_ceil = |n: usize| (n + step - 1) / step;

        (div_ceil(lower), upper.map(div_ceil))
    }
}

// ============================================================================
// Utility Iterators
// ============================================================================

/// An iterator that yields nothing.
pub struct Empty<T> {
    _marker: PhantomData<T>,
}

impl<T> Iterator for Empty<T> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

impl<T> ExactSizeIterator for Empty<T> {}
impl<T> DoubleEndedIterator for Empty<T> {
    fn next_back(&!self) -> Option<T> {
        None
    }
}
impl<T> FusedIterator for Empty<T> {}

/// Creates an empty iterator.
pub fn empty<T>() -> Empty<T> {
    Empty { _marker: PhantomData }
}

/// An iterator that yields exactly one element.
pub struct Once<T> {
    value: Option<T>,
}

impl<T> Iterator for Once<T> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        self.value.take()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = if self.value.is_some() { 1 } else { 0 };
        (n, Some(n))
    }
}

impl<T> ExactSizeIterator for Once<T> {}
impl<T> DoubleEndedIterator for Once<T> {
    fn next_back(&!self) -> Option<T> {
        self.value.take()
    }
}
impl<T> FusedIterator for Once<T> {}

/// Creates an iterator that yields exactly one element.
pub fn once<T>(value: T) -> Once<T> {
    Once { value: Some(value) }
}

/// An iterator that yields elements by calling a closure.
pub struct FromFn<F> {
    f: F,
}

impl<T, F: FnMut() -> Option<T>> Iterator for FromFn<F> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        (self.f)()
    }
}

/// Creates an iterator from a closure.
pub fn from_fn<T, F: FnMut() -> Option<T>>(f: F) -> FromFn<F> {
    FromFn { f }
}

/// An iterator that yields an element forever.
pub struct Repeat<T> {
    value: T,
}

impl<T: Clone> Iterator for Repeat<T> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        Some(self.value.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize.MAX, None)
    }
}

impl<T: Clone> DoubleEndedIterator for Repeat<T> {
    fn next_back(&!self) -> Option<T> {
        Some(self.value.clone())
    }
}

/// Creates an iterator that yields an element forever.
pub fn repeat<T: Clone>(value: T) -> Repeat<T> {
    Repeat { value }
}

/// An iterator that yields elements by repeatedly calling a closure.
pub struct RepeatWith<F> {
    f: F,
}

impl<T, F: FnMut() -> T> Iterator for RepeatWith<F> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        Some((self.f)())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize.MAX, None)
    }
}

/// Creates an iterator that yields elements by repeatedly calling a closure.
pub fn repeat_with<T, F: FnMut() -> T>(f: F) -> RepeatWith<F> {
    RepeatWith { f }
}

/// An iterator that yields successive values.
pub struct Successors<T, F> {
    next: Option<T>,
    f: F,
}

impl<T, F: FnMut(&T) -> Option<T>> Iterator for Successors<T, F> {
    type Item = T;

    fn next(&!self) -> Option<T> {
        let item = self.next.take()?;
        self.next = (self.f)(&item);
        Some(item)
    }
}

impl<T, F: FnMut(&T) -> Option<T>> FusedIterator for Successors<T, F> {}

/// Creates an iterator of successive values.
pub fn successors<T, F: FnMut(&T) -> Option<T>>(first: Option<T>, f: F) -> Successors<T, F> {
    Successors { next: first, f }
}

// ============================================================================
// Range Iterators
// ============================================================================

/// A range iterator.
pub struct Range<T> {
    start: T,
    end: T,
}

impl Iterator for Range<i32> {
    type Item = i32;

    fn next(&!self) -> Option<i32> {
        if self.start < self.end {
            let item = self.start;
            self.start += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = if self.start < self.end {
            (self.end - self.start) as usize
        } else {
            0
        };
        (len, Some(len))
    }
}

impl ExactSizeIterator for Range<i32> {}

impl DoubleEndedIterator for Range<i32> {
    fn next_back(&!self) -> Option<i32> {
        if self.start < self.end {
            self.end -= 1;
            Some(self.end)
        } else {
            None
        }
    }
}

impl Iterator for Range<usize> {
    type Item = usize;

    fn next(&!self) -> Option<usize> {
        if self.start < self.end {
            let item = self.start;
            self.start += 1;
            Some(item)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end.saturating_sub(self.start);
        (len, Some(len))
    }
}

impl ExactSizeIterator for Range<usize> {}

impl DoubleEndedIterator for Range<usize> {
    fn next_back(&!self) -> Option<usize> {
        if self.start < self.end {
            self.end -= 1;
            Some(self.end)
        } else {
            None
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let result: Vec<i32> = [1, 2, 3].iter()
            .map(|x| x * 2)
            .collect();
        assert_eq!(result, [2, 4, 6]);
    }

    #[test]
    fn test_filter() {
        let result: Vec<i32> = [1, 2, 3, 4, 5].iter()
            .filter(|x| x % 2 == 0)
            .collect();
        assert_eq!(result, [2, 4]);
    }

    #[test]
    fn test_filter_map() {
        let result: Vec<i32> = ["1", "two", "3"].iter()
            .filter_map(|s| s.parse::<i32>().ok())
            .collect();
        assert_eq!(result, [1, 3]);
    }

    #[test]
    fn test_fold() {
        let sum = [1, 2, 3, 4, 5].iter().fold(0, |acc, x| acc + x);
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_take() {
        let result: Vec<i32> = [1, 2, 3, 4, 5].iter().take(3).collect();
        assert_eq!(result, [1, 2, 3]);
    }

    #[test]
    fn test_skip() {
        let result: Vec<i32> = [1, 2, 3, 4, 5].iter().skip(2).collect();
        assert_eq!(result, [3, 4, 5]);
    }

    #[test]
    fn test_enumerate() {
        let result: Vec<(usize, i32)> = [10, 20, 30].iter().enumerate().collect();
        assert_eq!(result, [(0, 10), (1, 20), (2, 30)]);
    }

    #[test]
    fn test_chain() {
        let result: Vec<i32> = [1, 2].iter()
            .chain([3, 4])
            .collect();
        assert_eq!(result, [1, 2, 3, 4]);
    }

    #[test]
    fn test_zip() {
        let result: Vec<(i32, &str)> = [1, 2, 3].iter()
            .zip(["a", "b", "c"])
            .collect();
        assert_eq!(result, [(1, "a"), (2, "b"), (3, "c")]);
    }

    #[test]
    fn test_flatten() {
        let result: Vec<i32> = [[1, 2], [3, 4]].iter()
            .flatten()
            .collect();
        assert_eq!(result, [1, 2, 3, 4]);
    }

    #[test]
    fn test_any_all() {
        assert!([1, 2, 3, 4, 5].iter().any(|x| x > 3));
        assert!(![1, 2, 3, 4, 5].iter().any(|x| x > 10));

        assert!([2, 4, 6].iter().all(|x| x % 2 == 0));
        assert!(![1, 2, 3].iter().all(|x| x % 2 == 0));
    }

    #[test]
    fn test_min_max() {
        assert_eq!([3, 1, 4, 1, 5].iter().min(), Some(&1));
        assert_eq!([3, 1, 4, 1, 5].iter().max(), Some(&5));
    }

    #[test]
    fn test_sum_product() {
        assert_eq!([1, 2, 3, 4].iter().sum::<i32>(), 10);
        assert_eq!([1, 2, 3, 4].iter().product::<i32>(), 24);
    }

    #[test]
    fn test_peekable() {
        let mut iter = [1, 2, 3].iter().peekable();

        assert_eq!(iter.peek(), Some(&&1));
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.peek(), Some(&&2));
    }

    #[test]
    fn test_cycle() {
        let result: Vec<i32> = [1, 2].iter().cycle().take(5).collect();
        assert_eq!(result, [1, 2, 1, 2, 1]);
    }

    #[test]
    fn test_step_by() {
        let result: Vec<i32> = [0, 1, 2, 3, 4, 5].iter().step_by(2).collect();
        assert_eq!(result, [0, 2, 4]);
    }

    #[test]
    fn test_rev() {
        let result: Vec<i32> = [1, 2, 3].iter().rev().collect();
        assert_eq!(result, [3, 2, 1]);
    }

    #[test]
    fn test_once_empty() {
        assert_eq!(once(42).collect::<Vec<_>>(), [42]);
        assert_eq!(empty::<i32>().collect::<Vec<_>>(), []);
    }

    #[test]
    fn test_repeat() {
        let result: Vec<i32> = repeat(5).take(3).collect();
        assert_eq!(result, [5, 5, 5]);
    }

    #[test]
    fn test_from_fn() {
        let mut count = 0;
        let result: Vec<i32> = from_fn(|| {
            if count < 3 {
                count += 1;
                Some(count)
            } else {
                None
            }
        }).collect();
        assert_eq!(result, [1, 2, 3]);
    }
}
