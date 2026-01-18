//! Abstract Domain Trait
//!
//! Defines the core `AbstractDomain` trait that all abstract domains must implement.
//! This forms the foundation of the abstract interpretation framework.
//!
//! # Theory
//!
//! An abstract domain is a complete lattice (D, <=) with:
//! - A top element (most imprecise - all values)
//! - A bottom element (no values - unreachable)
//! - Join operation (least upper bound)
//! - Meet operation (greatest lower bound)
//!
//! # References
//!
//! - Cousot, P., & Cousot, R. (1977). Abstract interpretation: a unified lattice
//!   model for static analysis of programs.
//! - Nielson, F., Nielson, H. R., & Hankin, C. (2015). Principles of program analysis.

use std::fmt::Debug;
use std::hash::Hash;

/// The core trait for abstract domains.
///
/// Abstract domains form the semantic foundation of abstract interpretation.
/// They approximate sets of concrete values with abstract elements that can
/// be efficiently computed.
///
/// # Lattice Requirements
///
/// - `top()` is the greatest element: `forall x. x <= top()`
/// - `bottom()` is the least element: `forall x. bottom() <= x`
/// - `join(a, b)` is the least upper bound: `a <= join(a, b)` and `b <= join(a, b)`
/// - `meet(a, b)` is the greatest lower bound: `meet(a, b) <= a` and `meet(a, b) <= b`
///
/// # Example
///
/// ```ignore
/// use crate::analysis::abstract_::domain::AbstractDomain;
///
/// // Check if a value is unreachable
/// if state.is_bottom() {
///     println!("This code path is unreachable");
/// }
///
/// // Widen to ensure termination at loop heads
/// let widened = old_state.widen(&new_state);
/// ```
pub trait AbstractDomain: Clone + Eq + Debug + Sized {
    /// Create the top element (most imprecise - represents all possible values).
    ///
    /// The top element is used as the initial state for forward analysis
    /// and represents complete uncertainty about the value.
    fn top() -> Self;

    /// Create the bottom element (no values - represents unreachable code).
    ///
    /// Bottom indicates that no concrete execution can reach this point
    /// with this abstract state.
    fn bottom() -> Self;

    /// Compute the least upper bound (join) of two abstract values.
    ///
    /// The join operation merges information at control flow merge points.
    /// It must be sound: `gamma(join(a, b)) >= gamma(a) union gamma(b)`
    /// where `gamma` is the concretization function.
    ///
    /// # Properties
    ///
    /// - Commutative: `a.join(b) == b.join(a)`
    /// - Associative: `a.join(b.join(c)) == a.join(b).join(c)`
    /// - Idempotent: `a.join(a) == a`
    fn join(&self, other: &Self) -> Self;

    /// Compute the greatest lower bound (meet) of two abstract values.
    ///
    /// The meet operation computes the intersection of information,
    /// useful for path-sensitive analysis.
    ///
    /// # Properties
    ///
    /// - Commutative: `a.meet(b) == b.meet(a)`
    /// - Associative: `a.meet(b.meet(c)) == a.meet(b).meet(c)`
    /// - Idempotent: `a.meet(a) == a`
    fn meet(&self, other: &Self) -> Self;

    /// Widen to ensure termination of fixpoint iteration.
    ///
    /// Widening accelerates convergence by extrapolating from the observed
    /// sequence of iterates. It must satisfy: if `x_n` is a sequence with
    /// `x_{n+1} = x_n widen f(x_n)`, then the sequence stabilizes in
    /// finite steps.
    ///
    /// The default implementation just returns join, which may not terminate
    /// for infinite-height domains.
    ///
    /// # Example
    ///
    /// For intervals: `[0, 5] widen [0, 10]` might give `[0, +inf]`
    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }

    /// Narrow to recover precision after widening.
    ///
    /// Narrowing refines an over-approximation after fixpoint is reached.
    /// It must satisfy: `a narrow b <= a` when `b <= a`.
    ///
    /// The default implementation just returns meet.
    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }

    /// Check if this element is less than or equal to another in the lattice order.
    ///
    /// `a <= b` means `a` is more precise (contains fewer concrete values).
    /// Equivalently: `gamma(a) subset gamma(b)`.
    fn leq(&self, other: &Self) -> bool;

    /// Check if this is the bottom element.
    fn is_bottom(&self) -> bool;

    /// Check if this is the top element.
    fn is_top(&self) -> bool;
}

/// A lifted domain that adds undefined/uninitialized value.
///
/// This is useful for tracking variables that may not be initialized
/// on all paths.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Lifted<D: AbstractDomain> {
    /// Value is undefined/uninitialized
    Undefined,
    /// Value is defined with given abstract value
    Defined(D),
}

impl<D: AbstractDomain> AbstractDomain for Lifted<D> {
    fn top() -> Self {
        Lifted::Defined(D::top())
    }

    fn bottom() -> Self {
        Lifted::Undefined
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Lifted::Undefined, Lifted::Undefined) => Lifted::Undefined,
            (Lifted::Defined(d), Lifted::Undefined) | (Lifted::Undefined, Lifted::Defined(d)) => {
                Lifted::Defined(d.clone())
            }
            (Lifted::Defined(a), Lifted::Defined(b)) => Lifted::Defined(a.join(b)),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Lifted::Undefined, _) | (_, Lifted::Undefined) => Lifted::Undefined,
            (Lifted::Defined(a), Lifted::Defined(b)) => Lifted::Defined(a.meet(b)),
        }
    }

    fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (Lifted::Undefined, Lifted::Undefined) => Lifted::Undefined,
            (Lifted::Defined(d), Lifted::Undefined) | (Lifted::Undefined, Lifted::Defined(d)) => {
                Lifted::Defined(d.clone())
            }
            (Lifted::Defined(a), Lifted::Defined(b)) => Lifted::Defined(a.widen(b)),
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        match (self, other) {
            (Lifted::Undefined, _) | (_, Lifted::Undefined) => Lifted::Undefined,
            (Lifted::Defined(a), Lifted::Defined(b)) => Lifted::Defined(a.narrow(b)),
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (Lifted::Undefined, _) => true,
            (_, Lifted::Undefined) => false,
            (Lifted::Defined(a), Lifted::Defined(b)) => a.leq(b),
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Lifted::Undefined)
    }

    fn is_top(&self) -> bool {
        match self {
            Lifted::Undefined => false,
            Lifted::Defined(d) => d.is_top(),
        }
    }
}

/// A flat domain where all concrete values are incomparable.
///
/// The lattice looks like:
/// ```text
///        top
///     /   |   \
///    a    b    c ...
///     \   |   /
///       bottom
/// ```
///
/// Useful for tracking constant values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Flat<T: Clone + Eq + Debug> {
    /// No value (unreachable)
    Bottom,
    /// Single concrete value
    Value(T),
    /// Unknown value (could be any)
    Top,
}

impl<T: Clone + Eq + Debug> AbstractDomain for Flat<T> {
    fn top() -> Self {
        Flat::Top
    }

    fn bottom() -> Self {
        Flat::Bottom
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Flat::Bottom, x) | (x, Flat::Bottom) => x.clone(),
            (Flat::Top, _) | (_, Flat::Top) => Flat::Top,
            (Flat::Value(a), Flat::Value(b)) => {
                if a == b {
                    Flat::Value(a.clone())
                } else {
                    Flat::Top
                }
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Flat::Top, x) | (x, Flat::Top) => x.clone(),
            (Flat::Bottom, _) | (_, Flat::Bottom) => Flat::Bottom,
            (Flat::Value(a), Flat::Value(b)) => {
                if a == b {
                    Flat::Value(a.clone())
                } else {
                    Flat::Bottom
                }
            }
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (Flat::Bottom, _) => true,
            (_, Flat::Top) => true,
            (Flat::Value(a), Flat::Value(b)) => a == b,
            _ => false,
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Flat::Bottom)
    }

    fn is_top(&self) -> bool {
        matches!(self, Flat::Top)
    }
}

/// A powerset domain tracking a set of possible values.
///
/// Warning: Can be exponentially large for large value sets.
/// Use only for small, bounded sets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PowerSet<T: Clone + Eq + Hash + Debug> {
    elements: std::collections::HashSet<T>,
    /// Maximum size before collapsing to top
    max_size: usize,
    /// Whether this represents all values (top)
    is_top: bool,
}

impl<T: Clone + Eq + Hash + Debug> PowerSet<T> {
    /// Create a new powerset domain with given max size
    pub fn new(max_size: usize) -> Self {
        Self {
            elements: std::collections::HashSet::new(),
            max_size,
            is_top: false,
        }
    }

    /// Create a singleton set
    pub fn singleton(value: T, max_size: usize) -> Self {
        let mut elements = std::collections::HashSet::new();
        elements.insert(value);
        Self {
            elements,
            max_size,
            is_top: false,
        }
    }

    /// Get the elements in this set
    pub fn elements(&self) -> &std::collections::HashSet<T> {
        &self.elements
    }

    /// Check if this contains a specific value
    pub fn contains(&self, value: &T) -> bool {
        self.is_top || self.elements.contains(value)
    }
}

impl<T: Clone + Eq + Hash + Debug> AbstractDomain for PowerSet<T> {
    fn top() -> Self {
        Self {
            elements: std::collections::HashSet::new(),
            max_size: 10, // Default max size
            is_top: true,
        }
    }

    fn bottom() -> Self {
        Self {
            elements: std::collections::HashSet::new(),
            max_size: 10,
            is_top: false,
        }
    }

    fn join(&self, other: &Self) -> Self {
        if self.is_top || other.is_top {
            return Self::top();
        }

        let mut elements = self.elements.clone();
        elements.extend(other.elements.clone());

        if elements.len() > self.max_size {
            Self::top()
        } else {
            Self {
                elements,
                max_size: self.max_size,
                is_top: false,
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self.is_top {
            return other.clone();
        }
        if other.is_top {
            return self.clone();
        }

        let elements: std::collections::HashSet<_> =
            self.elements.intersection(&other.elements).cloned().collect();

        Self {
            elements,
            max_size: self.max_size,
            is_top: false,
        }
    }

    fn leq(&self, other: &Self) -> bool {
        if self.is_bottom() {
            return true;
        }
        if other.is_top {
            return true;
        }
        if self.is_top {
            return other.is_top;
        }
        self.elements.is_subset(&other.elements)
    }

    fn is_bottom(&self) -> bool {
        !self.is_top && self.elements.is_empty()
    }

    fn is_top(&self) -> bool {
        self.is_top
    }
}

/// Boolean domain for boolean abstractions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoolDomain {
    /// No value (unreachable)
    Bottom,
    /// Definitely true
    True,
    /// Definitely false
    False,
    /// Could be either true or false
    Top,
}

impl AbstractDomain for BoolDomain {
    fn top() -> Self {
        BoolDomain::Top
    }

    fn bottom() -> Self {
        BoolDomain::Bottom
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolDomain::Bottom, x) | (x, BoolDomain::Bottom) => *x,
            (BoolDomain::Top, _) | (_, BoolDomain::Top) => BoolDomain::Top,
            (BoolDomain::True, BoolDomain::True) => BoolDomain::True,
            (BoolDomain::False, BoolDomain::False) => BoolDomain::False,
            _ => BoolDomain::Top,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolDomain::Top, x) | (x, BoolDomain::Top) => *x,
            (BoolDomain::Bottom, _) | (_, BoolDomain::Bottom) => BoolDomain::Bottom,
            (BoolDomain::True, BoolDomain::True) => BoolDomain::True,
            (BoolDomain::False, BoolDomain::False) => BoolDomain::False,
            _ => BoolDomain::Bottom,
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (BoolDomain::Bottom, _) => true,
            (_, BoolDomain::Top) => true,
            (BoolDomain::True, BoolDomain::True) => true,
            (BoolDomain::False, BoolDomain::False) => true,
            _ => false,
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, BoolDomain::Bottom)
    }

    fn is_top(&self) -> bool {
        matches!(self, BoolDomain::Top)
    }
}

impl BoolDomain {
    /// Create from a concrete boolean
    pub fn from_bool(b: bool) -> Self {
        if b {
            BoolDomain::True
        } else {
            BoolDomain::False
        }
    }

    /// Check if this can be true
    pub fn may_be_true(&self) -> bool {
        matches!(self, BoolDomain::True | BoolDomain::Top)
    }

    /// Check if this can be false
    pub fn may_be_false(&self) -> bool {
        matches!(self, BoolDomain::False | BoolDomain::Top)
    }

    /// Check if this is definitely true
    pub fn is_definitely_true(&self) -> bool {
        matches!(self, BoolDomain::True)
    }

    /// Check if this is definitely false
    pub fn is_definitely_false(&self) -> bool {
        matches!(self, BoolDomain::False)
    }

    /// Negate the boolean
    pub fn not(&self) -> Self {
        match self {
            BoolDomain::Bottom => BoolDomain::Bottom,
            BoolDomain::True => BoolDomain::False,
            BoolDomain::False => BoolDomain::True,
            BoolDomain::Top => BoolDomain::Top,
        }
    }

    /// Logical and
    pub fn and(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolDomain::Bottom, _) | (_, BoolDomain::Bottom) => BoolDomain::Bottom,
            (BoolDomain::False, _) | (_, BoolDomain::False) => BoolDomain::False,
            (BoolDomain::True, BoolDomain::True) => BoolDomain::True,
            _ => BoolDomain::Top,
        }
    }

    /// Logical or
    pub fn or(&self, other: &Self) -> Self {
        match (self, other) {
            (BoolDomain::Bottom, _) | (_, BoolDomain::Bottom) => BoolDomain::Bottom,
            (BoolDomain::True, _) | (_, BoolDomain::True) => BoolDomain::True,
            (BoolDomain::False, BoolDomain::False) => BoolDomain::False,
            _ => BoolDomain::Top,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flat_domain() {
        let a: Flat<i32> = Flat::Value(42);
        let b: Flat<i32> = Flat::Value(42);
        let c: Flat<i32> = Flat::Value(100);

        assert_eq!(a.join(&b), Flat::Value(42));
        assert_eq!(a.join(&c), Flat::Top);
        assert_eq!(a.meet(&b), Flat::Value(42));
        assert_eq!(a.meet(&c), Flat::Bottom);
        assert!(a.leq(&Flat::Top));
        assert!(Flat::<i32>::Bottom.leq(&a));
    }

    #[test]
    fn test_lifted_domain() {
        let a: Lifted<Flat<i32>> = Lifted::Defined(Flat::Value(42));
        let b: Lifted<Flat<i32>> = Lifted::Undefined;

        assert!(!a.is_bottom());
        assert!(b.is_bottom());
        assert!(b.leq(&a));

        let joined = a.join(&b);
        assert!(matches!(joined, Lifted::Defined(Flat::Value(42))));
    }

    #[test]
    fn test_bool_domain() {
        let t = BoolDomain::True;
        let f = BoolDomain::False;

        assert_eq!(t.and(&f), BoolDomain::False);
        assert_eq!(t.or(&f), BoolDomain::True);
        assert_eq!(t.not(), BoolDomain::False);
        assert_eq!(t.join(&f), BoolDomain::Top);
    }

    #[test]
    fn test_powerset_domain() {
        let a = PowerSet::singleton(1, 10);
        let b = PowerSet::singleton(2, 10);

        let joined = a.join(&b);
        assert!(joined.contains(&1));
        assert!(joined.contains(&2));
        assert!(!joined.contains(&3));

        let met = a.meet(&b);
        assert!(met.is_bottom());
    }
}
