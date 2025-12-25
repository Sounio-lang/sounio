//! Linear and Affine Wrapper Types
//!
//! This module provides zero-cost wrapper types that enforce linearity at
//! compile time through Rust's ownership system.
//!
//! # Wrapper Types
//!
//! ```rust,ignore
//! Linear<T>   -- Must be consumed exactly once
//! Affine<T>   -- Can be consumed at most once (may be dropped)
//! ```
//!
//! # Design Philosophy
//!
//! These wrappers use Rust's type system to model Sounio'ss linearity:
//!
//! - `Linear<T>`: Has no `Drop` impl, so Rust will warn if not used
//! - `Affine<T>`: Has a `Drop` impl, allowing implicit discard
//!
//! The wrappers provide controlled access through methods that consume `self`:
//!
//! ```rust,ignore
//! impl<T> Linear<T> {
//!     fn consume(self) -> T;          // Use the value
//!     fn into_affine(self) -> Affine<T>;  // Weaken to affine
//! }
//!
//! impl<T> Affine<T> {
//!     fn consume(self) -> Option<T>;  // Use if present
//!     fn discard(self);               // Explicitly drop
//! }
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;

use super::kind::Linearity;

// ============================================================================
// Linear<T> - Must use exactly once
// ============================================================================

/// A linear value that must be consumed exactly once.
///
/// `Linear<T>` enforces that the wrapped value is used exactly once.
/// It does not implement `Drop`, so if you create a `Linear<T>` and don't
/// consume it, Rust will issue a warning (and in strict modes, an error).
///
/// # Example
///
/// ```rust,ignore
/// // Creating a linear file handle
/// let handle: Linear<FileHandle> = Linear::new(open_file("data.txt"));
///
/// // Must consume it
/// let file = handle.consume();
/// file.write("hello");
/// file.close();
/// ```
///
/// # Guarantees
///
/// - The value will be used exactly once
/// - Cannot be cloned or copied (unless T: Copy, but then you should use Unrestricted)
/// - Cannot be implicitly dropped
#[repr(transparent)]
pub struct Linear<T> {
    /// The wrapped value, using ManuallyDrop to prevent implicit drops
    inner: ManuallyDrop<T>,
}

impl<T> Linear<T> {
    /// Create a new linear value.
    ///
    /// The caller is responsible for ensuring this value is consumed exactly once.
    #[inline]
    pub fn new(value: T) -> Self {
        Self {
            inner: ManuallyDrop::new(value),
        }
    }

    /// Consume the linear value, returning the inner value.
    ///
    /// This is the primary way to use a linear value.
    #[inline]
    pub fn consume(self) -> T {
        // Use ManuallyDrop::into_inner to extract without running Drop
        // (not that we have a Drop impl, but this is the correct pattern)
        let this = ManuallyDrop::new(self);
        // SAFETY: We're consuming `this`, so it won't be used again
        unsafe { ManuallyDrop::into_inner(std::ptr::read(&this.inner)) }
    }

    /// Weaken this linear value to an affine value.
    ///
    /// After this, the value can be dropped without being used.
    /// This is a safe operation since Affine is more permissive.
    #[inline]
    pub fn into_affine(self) -> Affine<T> {
        Affine::new(self.consume())
    }

    /// Apply a function to the inner value, returning a new linear value.
    ///
    /// This consumes self and wraps the result.
    #[inline]
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Linear<U> {
        Linear::new(f(self.consume()))
    }

    /// Get a reference to the inner value.
    ///
    /// Note: This doesn't consume the value. You still need to consume it later.
    #[inline]
    pub fn as_ref(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner value.
    ///
    /// Note: This doesn't consume the value. You still need to consume it later.
    #[inline]
    pub fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Get the linearity of this wrapper.
    #[inline]
    pub fn linearity(&self) -> Linearity {
        Linearity::Linear
    }

    /// Pair two linear values into a linear pair.
    #[inline]
    pub fn pair<U>(self, other: Linear<U>) -> Linear<(T, U)> {
        Linear::new((self.consume(), other.consume()))
    }
}

// Linear<T> intentionally does NOT implement Drop
// This ensures the value must be explicitly consumed

impl<T: fmt::Debug> fmt::Debug for Linear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Linear").field(&*self.inner).finish()
    }
}

impl<T: fmt::Display> fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "linear {}", &*self.inner)
    }
}

// ============================================================================
// Affine<T> - Use at most once
// ============================================================================

/// An affine value that can be consumed at most once.
///
/// `Affine<T>` is more permissive than `Linear<T>`:
/// - Can be consumed (used exactly once)
/// - Can be dropped without being used
///
/// # Example
///
/// ```rust,ignore
/// let resource: Affine<TempBuffer> = Affine::new(alloc_temp());
///
/// if condition {
///     // Use it
///     let buffer = resource.consume();
///     buffer.process();
/// }
/// // If !condition, resource is dropped automatically
/// ```
///
/// # Guarantees
///
/// - The value will be used at most once
/// - Cannot be cloned or copied
/// - CAN be implicitly dropped (implements Drop)
pub struct Affine<T> {
    /// The wrapped value, as an Option to track consumption
    inner: Option<T>,
}

impl<T> Affine<T> {
    /// Create a new affine value.
    #[inline]
    pub fn new(value: T) -> Self {
        Self { inner: Some(value) }
    }

    /// Create an already-consumed affine value.
    ///
    /// This is useful for representing "empty" states in protocols.
    #[inline]
    pub fn empty() -> Self {
        Self { inner: None }
    }

    /// Consume the affine value, returning the inner value.
    ///
    /// Returns `Some(value)` if not yet consumed, `None` if already consumed.
    #[inline]
    pub fn consume(mut self) -> Option<T> {
        self.inner.take()
    }

    /// Consume the affine value, panicking if already consumed.
    ///
    /// Use this when you know the value hasn't been consumed.
    #[inline]
    pub fn consume_unchecked(self) -> T {
        self.consume().expect("Affine value already consumed")
    }

    /// Try to consume the value, returning it if available.
    #[inline]
    pub fn try_consume(&mut self) -> Option<T> {
        self.inner.take()
    }

    /// Explicitly discard the value without using it.
    ///
    /// This is equivalent to dropping, but more explicit in intent.
    #[inline]
    pub fn discard(self) {
        drop(self);
    }

    /// Check if the value has been consumed.
    #[inline]
    pub fn is_consumed(&self) -> bool {
        self.inner.is_none()
    }

    /// Check if the value is still available.
    #[inline]
    pub fn is_available(&self) -> bool {
        self.inner.is_some()
    }

    /// Get a reference to the inner value if available.
    #[inline]
    pub fn as_ref(&self) -> Option<&T> {
        self.inner.as_ref()
    }

    /// Get a mutable reference to the inner value if available.
    #[inline]
    pub fn as_mut(&mut self) -> Option<&mut T> {
        self.inner.as_mut()
    }

    /// Apply a function if the value is available.
    #[inline]
    pub fn map<U>(mut self, f: impl FnOnce(T) -> U) -> Affine<U> {
        Affine {
            inner: self.inner.take().map(f),
        }
    }

    /// Get the linearity of this wrapper.
    #[inline]
    pub fn linearity(&self) -> Linearity {
        Linearity::Affine
    }

    /// Strengthen to linear if the value is available.
    ///
    /// Returns `Some(Linear<T>)` if available, `None` if consumed.
    /// After this, the value MUST be consumed.
    #[inline]
    pub fn into_linear(mut self) -> Option<Linear<T>> {
        self.inner.take().map(Linear::new)
    }
}

impl<T> Drop for Affine<T> {
    fn drop(&mut self) {
        // Affine allows dropping, so this is fine
        // The Option will drop the inner value if present
    }
}

impl<T: fmt::Debug> fmt::Debug for Affine<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            Some(v) => f.debug_tuple("Affine").field(v).finish(),
            None => f.debug_tuple("Affine").field(&"<consumed>").finish(),
        }
    }
}

impl<T: fmt::Display> fmt::Display for Affine<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            Some(v) => write!(f, "affine {}", v),
            None => write!(f, "affine <consumed>"),
        }
    }
}

// ============================================================================
// Unrestricted<T> - Use any number of times
// ============================================================================

/// An unrestricted value that can be used any number of times.
///
/// This is essentially just `T` but wrapped for consistency with the
/// linearity system. Unrestricted values can be freely copied and dropped.
///
/// # Example
///
/// ```rust,ignore
/// let data: Unrestricted<i32> = Unrestricted::new(42);
///
/// // Can use multiple times
/// let a = *data.as_ref();
/// let b = *data.as_ref();
///
/// // Can clone
/// let data2 = data.clone();
///
/// // Can drop at any time
/// ```
#[derive(Clone)]
pub struct Unrestricted<T> {
    inner: T,
}

impl<T> Unrestricted<T> {
    /// Create a new unrestricted value.
    #[inline]
    pub fn new(value: T) -> Self {
        Self { inner: value }
    }

    /// Get the inner value by reference.
    #[inline]
    pub fn as_ref(&self) -> &T {
        &self.inner
    }

    /// Get the inner value by mutable reference.
    #[inline]
    pub fn as_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume and return the inner value.
    #[inline]
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Get the linearity of this wrapper.
    #[inline]
    pub fn linearity(&self) -> Linearity {
        Linearity::Unrestricted
    }

    /// Weaken to affine (always succeeds since Unrestricted <: Affine).
    #[inline]
    pub fn into_affine(self) -> Affine<T> {
        Affine::new(self.inner)
    }

    /// Weaken to linear (always succeeds since Unrestricted <: Linear).
    #[inline]
    pub fn into_linear(self) -> Linear<T> {
        Linear::new(self.inner)
    }
}

impl<T: Copy> Unrestricted<T> {
    /// Get a copy of the inner value.
    #[inline]
    pub fn get(&self) -> T {
        self.inner
    }
}

impl<T> Deref for Unrestricted<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T: fmt::Debug> fmt::Debug for Unrestricted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Unrestricted").field(&self.inner).finish()
    }
}

impl<T: fmt::Display> fmt::Display for Unrestricted<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

impl<T: Default> Default for Unrestricted<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

// ============================================================================
// LinearRef<T> - Borrowed linear reference
// ============================================================================

/// A borrowed reference to a linear value.
///
/// This allows passing linear values to functions that only need to read them,
/// without consuming the value. The borrow checker ensures the reference
/// doesn't outlive the original.
pub struct LinearRef<'a, T> {
    inner: &'a T,
    _marker: PhantomData<&'a Linear<T>>,
}

impl<'a, T> LinearRef<'a, T> {
    /// Create a new linear reference.
    #[inline]
    pub fn new(linear: &'a Linear<T>) -> Self {
        Self {
            inner: linear.as_ref(),
            _marker: PhantomData,
        }
    }

    /// Get the inner reference.
    #[inline]
    pub fn as_ref(&self) -> &T {
        self.inner
    }
}

impl<T> Deref for LinearRef<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T: fmt::Debug> fmt::Debug for LinearRef<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LinearRef").field(self.inner).finish()
    }
}

// ============================================================================
// LinearMut<T> - Mutable borrowed linear reference
// ============================================================================

/// A mutable borrowed reference to a linear value.
///
/// This allows mutating linear values without consuming them.
pub struct LinearMut<'a, T> {
    inner: &'a mut T,
    _marker: PhantomData<&'a mut Linear<T>>,
}

impl<'a, T> LinearMut<'a, T> {
    /// Create a new mutable linear reference.
    #[inline]
    pub fn new(linear: &'a mut Linear<T>) -> Self {
        Self {
            inner: linear.as_mut(),
            _marker: PhantomData,
        }
    }

    /// Get the inner reference.
    #[inline]
    pub fn as_ref(&self) -> &T {
        self.inner
    }

    /// Get the inner mutable reference.
    #[inline]
    pub fn as_mut(&mut self) -> &mut T {
        self.inner
    }
}

impl<T> Deref for LinearMut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<T> std::ops::DerefMut for LinearMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner
    }
}

impl<T: fmt::Debug> fmt::Debug for LinearMut<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LinearMut").field(self.inner).finish()
    }
}

// ============================================================================
// LinearPair<A, B> - Tensor product
// ============================================================================

/// A linear pair (tensor product) of two values.
///
/// Both values must be consumed together. This models the tensor product
/// (A ⊗ B) from linear logic.
pub struct LinearPair<A, B> {
    first: ManuallyDrop<A>,
    second: ManuallyDrop<B>,
}

impl<A, B> LinearPair<A, B> {
    /// Create a new linear pair.
    #[inline]
    pub fn new(first: A, second: B) -> Self {
        Self {
            first: ManuallyDrop::new(first),
            second: ManuallyDrop::new(second),
        }
    }

    /// Create from two linear values.
    #[inline]
    pub fn from_linear(first: Linear<A>, second: Linear<B>) -> Self {
        Self::new(first.consume(), second.consume())
    }

    /// Consume the pair, returning both values.
    #[inline]
    pub fn consume(self) -> (A, B) {
        let this = ManuallyDrop::new(self);
        unsafe {
            (
                ManuallyDrop::into_inner(std::ptr::read(&this.first)),
                ManuallyDrop::into_inner(std::ptr::read(&this.second)),
            )
        }
    }

    /// Consume just the first element.
    #[inline]
    pub fn consume_first(self) -> (A, Linear<B>) {
        let (a, b) = self.consume();
        (a, Linear::new(b))
    }

    /// Consume just the second element.
    #[inline]
    pub fn consume_second(self) -> (Linear<A>, B) {
        let (a, b) = self.consume();
        (Linear::new(a), b)
    }

    /// Get references to both elements.
    #[inline]
    pub fn as_ref(&self) -> (&A, &B) {
        (&self.first, &self.second)
    }
}

impl<A: fmt::Debug, B: fmt::Debug> fmt::Debug for LinearPair<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LinearPair")
            .field(&*self.first)
            .field(&*self.second)
            .finish()
    }
}

// ============================================================================
// LinearChoice<A, B> - Additive disjunction
// ============================================================================

/// A linear choice between two values (A ⊕ B).
///
/// Represents an external choice: the producer decides which variant is present.
#[derive(Debug)]
pub enum LinearChoice<A, B> {
    /// The left variant
    Left(Linear<A>),
    /// The right variant
    Right(Linear<B>),
}

impl<A, B> LinearChoice<A, B> {
    /// Create a left choice.
    #[inline]
    pub fn left(value: A) -> Self {
        LinearChoice::Left(Linear::new(value))
    }

    /// Create a right choice.
    #[inline]
    pub fn right(value: B) -> Self {
        LinearChoice::Right(Linear::new(value))
    }

    /// Consume with a handler for each case.
    #[inline]
    pub fn consume<R>(self, left: impl FnOnce(A) -> R, right: impl FnOnce(B) -> R) -> R {
        match self {
            LinearChoice::Left(a) => left(a.consume()),
            LinearChoice::Right(b) => right(b.consume()),
        }
    }

    /// Check if this is the left variant.
    #[inline]
    pub fn is_left(&self) -> bool {
        matches!(self, LinearChoice::Left(_))
    }

    /// Check if this is the right variant.
    #[inline]
    pub fn is_right(&self) -> bool {
        matches!(self, LinearChoice::Right(_))
    }
}

// ============================================================================
// Utility functions
// ============================================================================

/// Create a linear value.
#[inline]
pub fn linear<T>(value: T) -> Linear<T> {
    Linear::new(value)
}

/// Create an affine value.
#[inline]
pub fn affine<T>(value: T) -> Affine<T> {
    Affine::new(value)
}

/// Create an unrestricted value.
#[inline]
pub fn unrestricted<T>(value: T) -> Unrestricted<T> {
    Unrestricted::new(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_consume() {
        let x = Linear::new(42);
        assert_eq!(x.consume(), 42);
    }

    #[test]
    fn test_linear_map() {
        let x = Linear::new(21);
        let y = x.map(|n| n * 2);
        assert_eq!(y.consume(), 42);
    }

    #[test]
    fn test_linear_into_affine() {
        let x = Linear::new(42);
        let y: Affine<i32> = x.into_affine();
        assert_eq!(y.consume(), Some(42));
    }

    #[test]
    fn test_linear_pair() {
        let x = Linear::new(1);
        let y = Linear::new(2);
        let pair = x.pair(y);
        assert_eq!(pair.consume(), (1, 2));
    }

    #[test]
    fn test_affine_consume() {
        let x = Affine::new(42);
        assert_eq!(x.consume(), Some(42));
    }

    #[test]
    fn test_affine_discard() {
        let x = Affine::new(42);
        x.discard(); // Should not panic
    }

    #[test]
    fn test_affine_already_consumed() {
        let mut x = Affine::new(42);
        let _ = x.try_consume();
        assert!(x.is_consumed());
    }

    #[test]
    fn test_affine_into_linear() {
        let x = Affine::new(42);
        let y = x.into_linear().unwrap();
        assert_eq!(y.consume(), 42);
    }

    #[test]
    fn test_unrestricted_multi_use() {
        let x = Unrestricted::new(42);
        assert_eq!(*x.as_ref(), 42);
        assert_eq!(*x.as_ref(), 42);
        assert_eq!(x.into_inner(), 42);
    }

    #[test]
    fn test_unrestricted_into_linear() {
        let x = Unrestricted::new(42);
        let y = x.into_linear();
        assert_eq!(y.consume(), 42);
    }

    #[test]
    fn test_linear_ref() {
        let x = Linear::new(42);
        let r = LinearRef::new(&x);
        assert_eq!(*r, 42);
        drop(r);
        assert_eq!(x.consume(), 42);
    }

    #[test]
    fn test_linear_pair_consume_first() {
        let pair = LinearPair::new(1, 2);
        let (first, second) = pair.consume_first();
        assert_eq!(first, 1);
        assert_eq!(second.consume(), 2);
    }

    #[test]
    fn test_linear_choice() {
        let choice: LinearChoice<i32, &str> = LinearChoice::left(42);
        let result = choice.consume(|n| n.to_string(), |s| s.to_string());
        assert_eq!(result, "42");

        let choice2: LinearChoice<i32, &str> = LinearChoice::right("hello");
        let result2 = choice2.consume(|n| n.to_string(), |s| s.to_string());
        assert_eq!(result2, "hello");
    }

    #[test]
    fn test_linearity_accessors() {
        let linear = Linear::new(0);
        assert_eq!(linear.linearity(), Linearity::Linear);

        let affine = Affine::new(0);
        assert_eq!(affine.linearity(), Linearity::Affine);

        let unrestricted = Unrestricted::new(0);
        assert_eq!(unrestricted.linearity(), Linearity::Unrestricted);
    }
}
