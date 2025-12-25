/// Async Stream Trait
///
/// Streams are asynchronous iterators that yield a sequence of values.
/// They're the async equivalent of Iterator.

module async::stream

import async::future::{Future, Poll, Context, Waker}

/// The Stream trait - async iterator
///
/// # Example
/// ```
/// async fn process_stream<S: Stream<Item = i32>>(mut stream: S) {
///     while let Some(value) = stream.next().await {
///         println("Got: {}", value)
///     }
/// }
/// ```
pub trait Stream {
    /// The type of items yielded by this stream
    type Item

    /// Attempts to pull the next value from the stream
    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<Self::Item>>

    /// Returns a hint about the remaining length
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

/// Extension trait for Stream with useful combinators
pub trait StreamExt: Stream {
    /// Returns the next item in the stream
    fn next(&mut self) -> Next<Self> where Self: Sized {
        Next { stream: self }
    }

    /// Maps each item with a function
    fn map<U, F: FnMut(Self::Item) -> U>(self, f: F) -> Map<Self, F> where Self: Sized {
        Map { stream: self, f }
    }

    /// Filters items based on a predicate
    fn filter<P: FnMut(&Self::Item) -> bool>(self, predicate: P) -> Filter<Self, P> where Self: Sized {
        Filter { stream: self, predicate }
    }

    /// Filters and maps simultaneously
    fn filter_map<U, F: FnMut(Self::Item) -> Option<U>>(self, f: F) -> FilterMap<Self, F> where Self: Sized {
        FilterMap { stream: self, f }
    }

    /// Takes only the first n items
    fn take(self, n: usize) -> Take<Self> where Self: Sized {
        Take { stream: self, remaining: n }
    }

    /// Skips the first n items
    fn skip(self, n: usize) -> Skip<Self> where Self: Sized {
        Skip { stream: self, remaining: n }
    }

    /// Takes items while predicate returns true
    fn take_while<P: FnMut(&Self::Item) -> bool>(self, predicate: P) -> TakeWhile<Self, P> where Self: Sized {
        TakeWhile { stream: self, predicate, done: false }
    }

    /// Skips items while predicate returns true
    fn skip_while<P: FnMut(&Self::Item) -> bool>(self, predicate: P) -> SkipWhile<Self, P> where Self: Sized {
        SkipWhile { stream: self, predicate, done: false }
    }

    /// Chains another stream after this one
    fn chain<S: Stream<Item = Self::Item>>(self, other: S) -> Chain<Self, S> where Self: Sized {
        Chain { first: Some(self), second: other }
    }

    /// Zips with another stream
    fn zip<S: Stream>(self, other: S) -> Zip<Self, S> where Self: Sized {
        Zip { a: self, b: other }
    }

    /// Flattens a stream of streams
    fn flatten(self) -> Flatten<Self> where Self: Sized, Self::Item: Stream {
        Flatten { outer: self, inner: None }
    }

    /// Maps then flattens
    fn flat_map<U: Stream, F: FnMut(Self::Item) -> U>(self, f: F) -> FlatMap<Self, U, F> where Self: Sized {
        FlatMap { stream: self.map(f).flatten() }
    }

    /// Folds the stream into a single value
    fn fold<B, F: FnMut(B, Self::Item) -> B>(self, init: B, f: F) -> Fold<Self, B, F> where Self: Sized {
        Fold { stream: self, acc: Some(init), f }
    }

    /// Collects stream items into a collection
    fn collect<C: FromStream<Self::Item>>(self) -> Collect<Self, C> where Self: Sized {
        Collect { stream: self, collection: None }
    }

    /// Runs a closure on each item
    fn for_each<F: FnMut(Self::Item)>(self, f: F) -> ForEach<Self, F> where Self: Sized {
        ForEach { stream: self, f }
    }

    /// Returns true if any item satisfies the predicate
    fn any<P: FnMut(Self::Item) -> bool>(self, predicate: P) -> Any<Self, P> where Self: Sized {
        Any { stream: self, predicate }
    }

    /// Returns true if all items satisfy the predicate
    fn all<P: FnMut(Self::Item) -> bool>(self, predicate: P) -> All<Self, P> where Self: Sized {
        All { stream: self, predicate }
    }

    /// Finds the first item satisfying a predicate
    fn find<P: FnMut(&Self::Item) -> bool>(self, predicate: P) -> Find<Self, P> where Self: Sized {
        Find { stream: self, predicate }
    }

    /// Returns the nth item
    fn nth(self, n: usize) -> Nth<Self> where Self: Sized {
        Nth { stream: self, n }
    }

    /// Counts the number of items
    fn count(self) -> Count<Self> where Self: Sized {
        Count { stream: self, count: 0 }
    }

    /// Enumerates items with their index
    fn enumerate(self) -> Enumerate<Self> where Self: Sized {
        Enumerate { stream: self, index: 0 }
    }

    /// Inspects each item without modifying
    fn inspect<F: FnMut(&Self::Item)>(self, f: F) -> Inspect<Self, F> where Self: Sized {
        Inspect { stream: self, f }
    }

    /// Fuses the stream (returns None forever after first None)
    fn fuse(self) -> Fuse<Self> where Self: Sized {
        Fuse { stream: Some(self) }
    }

    /// Boxes the stream for dynamic dispatch
    fn boxed(self) -> BoxStream<Self::Item> where Self: Sized + Send + 'static {
        BoxStream { inner: Box::new(self) }
    }
}

// Blanket implementation
impl<S: Stream> StreamExt for S {}

// =============================================================================
// Stream Combinators
// =============================================================================

/// Future for getting the next item
pub struct Next<'a, S: Stream> {
    stream: &'a mut S,
}

impl<'a, S: Stream> Future for Next<'a, S> {
    type Output = Option<S::Item>

    fn poll(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        self.stream.poll_next(cx)
    }
}

/// Stream that maps items
pub struct Map<S, F> {
    stream: S,
    f: F,
}

impl<S: Stream, U, F: FnMut(S::Item) -> U> Stream for Map<S, F> {
    type Item = U

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<U>> {
        match self.stream.poll_next(cx) {
            Poll::Ready(Some(item)) => Poll::Ready(Some((self.f)(item))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Stream that filters items
pub struct Filter<S, P> {
    stream: S,
    predicate: P,
}

impl<S: Stream, P: FnMut(&S::Item) -> bool> Stream for Filter<S, P> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if (self.predicate)(&item) {
                        return Poll::Ready(Some(item));
                    }
                    // Continue polling for next item
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Stream that filters and maps
pub struct FilterMap<S, F> {
    stream: S,
    f: F,
}

impl<S: Stream, U, F: FnMut(S::Item) -> Option<U>> Stream for FilterMap<S, F> {
    type Item = U

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<U>> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if let Some(mapped) = (self.f)(item) {
                        return Poll::Ready(Some(mapped));
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Stream that takes first n items
pub struct Take<S> {
    stream: S,
    remaining: usize,
}

impl<S: Stream> Stream for Take<S> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        if self.remaining == 0 {
            return Poll::Ready(None);
        }

        match self.stream.poll_next(cx) {
            Poll::Ready(Some(item)) => {
                self.remaining -= 1;
                Poll::Ready(Some(item))
            }
            other => other,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.stream.size_hint();
        (lower.min(self.remaining), upper.map(|u| u.min(self.remaining)))
    }
}

/// Stream that skips first n items
pub struct Skip<S> {
    stream: S,
    remaining: usize,
}

impl<S: Stream> Stream for Skip<S> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        while self.remaining > 0 {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(_)) => {
                    self.remaining -= 1;
                }
                other => return other,
            }
        }
        self.stream.poll_next(cx)
    }
}

/// Stream that takes while predicate is true
pub struct TakeWhile<S, P> {
    stream: S,
    predicate: P,
    done: bool,
}

impl<S: Stream, P: FnMut(&S::Item) -> bool> Stream for TakeWhile<S, P> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        if self.done {
            return Poll::Ready(None);
        }

        match self.stream.poll_next(cx) {
            Poll::Ready(Some(item)) => {
                if (self.predicate)(&item) {
                    Poll::Ready(Some(item))
                } else {
                    self.done = true;
                    Poll::Ready(None)
                }
            }
            other => other,
        }
    }
}

/// Stream that skips while predicate is true
pub struct SkipWhile<S, P> {
    stream: S,
    predicate: P,
    done: bool,
}

impl<S: Stream, P: FnMut(&S::Item) -> bool> Stream for SkipWhile<S, P> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        if self.done {
            return self.stream.poll_next(cx);
        }

        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if (self.predicate)(&item) {
                        continue;
                    } else {
                        self.done = true;
                        return Poll::Ready(Some(item));
                    }
                }
                other => return other,
            }
        }
    }
}

/// Stream that chains two streams
pub struct Chain<A, B> {
    first: Option<A>,
    second: B,
}

impl<A: Stream, B: Stream<Item = A::Item>> Stream for Chain<A, B> {
    type Item = A::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<A::Item>> {
        if let Some(ref mut first) = self.first {
            match first.poll_next(cx) {
                Poll::Ready(Some(item)) => return Poll::Ready(Some(item)),
                Poll::Ready(None) => self.first = None,
                Poll::Pending => return Poll::Pending,
            }
        }
        self.second.poll_next(cx)
    }
}

/// Stream that zips two streams
pub struct Zip<A, B> {
    a: A,
    b: B,
}

impl<A: Stream, B: Stream> Stream for Zip<A, B> {
    type Item = (A::Item, B::Item)

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<(A::Item, B::Item)>> {
        // This is simplified - real impl would need to handle partial completions
        match (self.a.poll_next(cx), self.b.poll_next(cx)) {
            (Poll::Ready(Some(a)), Poll::Ready(Some(b))) => Poll::Ready(Some((a, b))),
            (Poll::Ready(None), _) | (_, Poll::Ready(None)) => Poll::Ready(None),
            _ => Poll::Pending,
        }
    }
}

/// Stream that flattens nested streams
pub struct Flatten<S: Stream> where S::Item: Stream {
    outer: S,
    inner: Option<S::Item>,
}

impl<S: Stream> Stream for Flatten<S> where S::Item: Stream {
    type Item = <S::Item as Stream>::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<Self::Item>> {
        loop {
            // First, try the inner stream
            if let Some(ref mut inner) = self.inner {
                match inner.poll_next(cx) {
                    Poll::Ready(Some(item)) => return Poll::Ready(Some(item)),
                    Poll::Ready(None) => self.inner = None,
                    Poll::Pending => return Poll::Pending,
                }
            }

            // Then, get next inner stream from outer
            match self.outer.poll_next(cx) {
                Poll::Ready(Some(inner)) => self.inner = Some(inner),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Stream that maps then flattens
pub struct FlatMap<S, U: Stream, F> {
    stream: Flatten<Map<S, F>>,
}

impl<S: Stream, U: Stream, F: FnMut(S::Item) -> U> Stream for FlatMap<S, U, F> {
    type Item = U::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<U::Item>> {
        self.stream.poll_next(cx)
    }
}

/// Stream with index
pub struct Enumerate<S> {
    stream: S,
    index: usize,
}

impl<S: Stream> Stream for Enumerate<S> {
    type Item = (usize, S::Item)

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<(usize, S::Item)>> {
        match self.stream.poll_next(cx) {
            Poll::Ready(Some(item)) => {
                let index = self.index;
                self.index += 1;
                Poll::Ready(Some((index, item)))
            }
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Stream that inspects items
pub struct Inspect<S, F> {
    stream: S,
    f: F,
}

impl<S: Stream, F: FnMut(&S::Item)> Stream for Inspect<S, F> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        match self.stream.poll_next(cx) {
            Poll::Ready(Some(item)) => {
                (self.f)(&item);
                Poll::Ready(Some(item))
            }
            other => other,
        }
    }
}

/// Fused stream
pub struct Fuse<S> {
    stream: Option<S>,
}

impl<S: Stream> Stream for Fuse<S> {
    type Item = S::Item

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        match &mut self.stream {
            Some(stream) => {
                match stream.poll_next(cx) {
                    Poll::Ready(None) => {
                        self.stream = None;
                        Poll::Ready(None)
                    }
                    other => other,
                }
            }
            None => Poll::Ready(None),
        }
    }
}

/// Boxed stream for dynamic dispatch
pub struct BoxStream<T> {
    inner: Box<dyn Stream<Item = T> + Send>,
}

impl<T> Stream for BoxStream<T> {
    type Item = T

    fn poll_next(&mut self, cx: &mut Context) -> Poll<Option<T>> {
        self.inner.poll_next(cx)
    }
}

// =============================================================================
// Terminal Operations (return Futures)
// =============================================================================

/// Future that folds a stream
pub struct Fold<S, B, F> {
    stream: S,
    acc: Option<B>,
    f: F,
}

impl<S: Stream, B, F: FnMut(B, S::Item) -> B> Future for Fold<S, B, F> {
    type Output = B

    fn poll(&mut self, cx: &mut Context) -> Poll<B> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    let acc = self.acc.take().unwrap();
                    self.acc = Some((self.f)(acc, item));
                }
                Poll::Ready(None) => {
                    return Poll::Ready(self.acc.take().unwrap());
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that collects a stream
pub struct Collect<S, C> {
    stream: S,
    collection: Option<C>,
}

/// Trait for collecting from a stream
pub trait FromStream<T> {
    fn initialize() -> Self;
    fn extend(&mut self, item: T);
}

impl<T> FromStream<T> for Vec<T> {
    fn initialize() -> Vec<T> {
        Vec::new()
    }

    fn extend(&mut self, item: T) {
        self.push(item);
    }
}

impl<S: Stream, C: FromStream<S::Item>> Future for Collect<S, C> {
    type Output = C

    fn poll(&mut self, cx: &mut Context) -> Poll<C> {
        if self.collection.is_none() {
            self.collection = Some(C::initialize());
        }

        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    self.collection.as_mut().unwrap().extend(item);
                }
                Poll::Ready(None) => {
                    return Poll::Ready(self.collection.take().unwrap());
                }
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that runs for_each on a stream
pub struct ForEach<S, F> {
    stream: S,
    f: F,
}

impl<S: Stream, F: FnMut(S::Item)> Future for ForEach<S, F> {
    type Output = ()

    fn poll(&mut self, cx: &mut Context) -> Poll<()> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => (self.f)(item),
                Poll::Ready(None) => return Poll::Ready(()),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that checks if any item satisfies predicate
pub struct Any<S, P> {
    stream: S,
    predicate: P,
}

impl<S: Stream, P: FnMut(S::Item) -> bool> Future for Any<S, P> {
    type Output = bool

    fn poll(&mut self, cx: &mut Context) -> Poll<bool> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if (self.predicate)(item) {
                        return Poll::Ready(true);
                    }
                }
                Poll::Ready(None) => return Poll::Ready(false),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that checks if all items satisfy predicate
pub struct All<S, P> {
    stream: S,
    predicate: P,
}

impl<S: Stream, P: FnMut(S::Item) -> bool> Future for All<S, P> {
    type Output = bool

    fn poll(&mut self, cx: &mut Context) -> Poll<bool> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if !(self.predicate)(item) {
                        return Poll::Ready(false);
                    }
                }
                Poll::Ready(None) => return Poll::Ready(true),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that finds first matching item
pub struct Find<S, P> {
    stream: S,
    predicate: P,
}

impl<S: Stream, P: FnMut(&S::Item) -> bool> Future for Find<S, P> {
    type Output = Option<S::Item>

    fn poll(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if (self.predicate)(&item) {
                        return Poll::Ready(Some(item));
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that gets the nth item
pub struct Nth<S> {
    stream: S,
    n: usize,
}

impl<S: Stream> Future for Nth<S> {
    type Output = Option<S::Item>

    fn poll(&mut self, cx: &mut Context) -> Poll<Option<S::Item>> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(item)) => {
                    if self.n == 0 {
                        return Poll::Ready(Some(item));
                    }
                    self.n -= 1;
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

/// Future that counts items
pub struct Count<S> {
    stream: S,
    count: usize,
}

impl<S: Stream> Future for Count<S> {
    type Output = usize

    fn poll(&mut self, cx: &mut Context) -> Poll<usize> {
        loop {
            match self.stream.poll_next(cx) {
                Poll::Ready(Some(_)) => self.count += 1,
                Poll::Ready(None) => return Poll::Ready(self.count),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// =============================================================================
// Stream Constructors
// =============================================================================

/// Creates an empty stream
pub fn empty<T>() -> Empty<T> {
    Empty { _marker: PhantomData }
}

pub struct Empty<T> {
    _marker: PhantomData<T>,
}

impl<T> Stream for Empty<T> {
    type Item = T

    fn poll_next(&mut self, _cx: &mut Context) -> Poll<Option<T>> {
        Poll::Ready(None)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(0))
    }
}

/// Creates a stream that yields a single item
pub fn once<T>(value: T) -> Once<T> {
    Once { value: Some(value) }
}

pub struct Once<T> {
    value: Option<T>,
}

impl<T> Stream for Once<T> {
    type Item = T

    fn poll_next(&mut self, _cx: &mut Context) -> Poll<Option<T>> {
        Poll::Ready(self.value.take())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.value.is_some() {
            (1, Some(1))
        } else {
            (0, Some(0))
        }
    }
}

/// Creates a stream from an iterator
pub fn iter<I: Iterator>(iter: I) -> Iter<I> {
    Iter { iter }
}

pub struct Iter<I> {
    iter: I,
}

impl<I: Iterator> Stream for Iter<I> {
    type Item = I::Item

    fn poll_next(&mut self, _cx: &mut Context) -> Poll<Option<I::Item>> {
        Poll::Ready(self.iter.next())
    }
}

/// Creates a stream that repeats a value forever
pub fn repeat<T: Clone>(value: T) -> Repeat<T> {
    Repeat { value }
}

pub struct Repeat<T> {
    value: T,
}

impl<T: Clone> Stream for Repeat<T> {
    type Item = T

    fn poll_next(&mut self, _cx: &mut Context) -> Poll<Option<T>> {
        Poll::Ready(Some(self.value.clone()))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (usize::MAX, None)
    }
}

/// Creates a stream from a generating function
pub fn from_fn<T, F: FnMut() -> Option<T>>(f: F) -> FromFn<F> {
    FromFn { f }
}

pub struct FromFn<F> {
    f: F,
}

impl<T, F: FnMut() -> Option<T>> Stream for FromFn<F> {
    type Item = T

    fn poll_next(&mut self, _cx: &mut Context) -> Poll<Option<T>> {
        Poll::Ready((self.f)())
    }
}

/// Creates a stream that unfolds from a seed value
pub fn unfold<T, S, F: FnMut(S) -> Option<(T, S)>>(seed: S, f: F) -> Unfold<S, F> {
    Unfold { state: Some(seed), f }
}

pub struct Unfold<S, F> {
    state: Option<S>,
    f: F,
}

impl<T, S, F: FnMut(S) -> Option<(T, S)>> Stream for Unfold<S, F> {
    type Item = T

    fn poll_next(&mut self, _cx: &mut Context) -> Poll<Option<T>> {
        let state = match self.state.take() {
            Some(s) => s,
            None => return Poll::Ready(None),
        };

        match (self.f)(state) {
            Some((item, next_state)) => {
                self.state = Some(next_state);
                Poll::Ready(Some(item))
            }
            None => Poll::Ready(None),
        }
    }
}

// Marker type
struct PhantomData<T> {}
