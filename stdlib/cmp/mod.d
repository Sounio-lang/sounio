//! Comparison and Ordering Utilities for Demetrios
//!
//! Provides min, max, clamp, and comparison traits.

// =============================================================================
// Ordering Type
// =============================================================================

/// Result of a comparison between two values
pub enum Ordering {
    /// First value is less than second
    Less,
    /// Values are equal
    Equal,
    /// First value is greater than second
    Greater,
}

impl Ordering {
    /// Check if ordering is Less
    pub fn is_lt(self) -> bool {
        match self {
            Ordering::Less => true,
            _ => false,
        }
    }

    /// Check if ordering is Equal
    pub fn is_eq(self) -> bool {
        match self {
            Ordering::Equal => true,
            _ => false,
        }
    }

    /// Check if ordering is Greater
    pub fn is_gt(self) -> bool {
        match self {
            Ordering::Greater => true,
            _ => false,
        }
    }

    /// Check if Less or Equal
    pub fn is_le(self) -> bool {
        match self {
            Ordering::Less | Ordering::Equal => true,
            _ => false,
        }
    }

    /// Check if Greater or Equal
    pub fn is_ge(self) -> bool {
        match self {
            Ordering::Greater | Ordering::Equal => true,
            _ => false,
        }
    }

    /// Reverse the ordering
    pub fn reverse(self) -> Ordering {
        match self {
            Ordering::Less => Ordering::Greater,
            Ordering::Equal => Ordering::Equal,
            Ordering::Greater => Ordering::Less,
        }
    }

    /// Chain comparison with another ordering
    ///
    /// Returns self if not Equal, otherwise returns other.
    pub fn then(self, other: Ordering) -> Ordering {
        match self {
            Ordering::Equal => other,
            _ => self,
        }
    }
}

// =============================================================================
// Comparison Traits
// =============================================================================

/// Trait for types that have a partial ordering
pub trait PartialOrd {
    /// Compare two values, returning None if not comparable
    fn partial_cmp(self: &Self, other: &Self) -> Option<Ordering>;

    /// Check if self < other
    fn lt(self: &Self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Less) => true,
            _ => false,
        }
    }

    /// Check if self <= other
    fn le(self: &Self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Less) | Some(Ordering::Equal) => true,
            _ => false,
        }
    }

    /// Check if self > other
    fn gt(self: &Self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Greater) => true,
            _ => false,
        }
    }

    /// Check if self >= other
    fn ge(self: &Self, other: &Self) -> bool {
        match self.partial_cmp(other) {
            Some(Ordering::Greater) | Some(Ordering::Equal) => true,
            _ => false,
        }
    }
}

/// Trait for types that have a total ordering
pub trait Ord: PartialOrd {
    /// Compare two values
    fn cmp(self: &Self, other: &Self) -> Ordering;

    /// Return maximum of two values
    fn max(self: Self, other: Self) -> Self {
        if self.cmp(&other).is_ge() {
            self
        } else {
            other
        }
    }

    /// Return minimum of two values
    fn min(self: Self, other: Self) -> Self {
        if self.cmp(&other).is_le() {
            self
        } else {
            other
        }
    }

    /// Clamp value to range [min, max]
    fn clamp(self: Self, min_val: Self, max_val: Self) -> Self {
        if self.cmp(&min_val).is_lt() {
            min_val
        } else if self.cmp(&max_val).is_gt() {
            max_val
        } else {
            self
        }
    }
}

// =============================================================================
// Free Functions
// =============================================================================

/// Return the minimum of two values
///
/// # Example
/// ```d
/// let smaller = min(10, 20);  // 10
/// let smaller = min(3.14, 2.71);  // 2.71
/// ```
pub fn min<T: Ord>(a: T, b: T) -> T {
    if a.cmp(&b).is_le() {
        a
    } else {
        b
    }
}

/// Return the maximum of two values
///
/// # Example
/// ```d
/// let larger = max(10, 20);  // 20
/// let larger = max(3.14, 2.71);  // 3.14
/// ```
pub fn max<T: Ord>(a: T, b: T) -> T {
    if a.cmp(&b).is_ge() {
        a
    } else {
        b
    }
}

/// Return the minimum of two values with partial ordering
///
/// Returns None if values are not comparable (e.g., NaN for floats)
pub fn min_partial<T: PartialOrd>(a: T, b: T) -> Option<T> {
    match a.partial_cmp(&b) {
        Some(Ordering::Less) | Some(Ordering::Equal) => Some(a),
        Some(Ordering::Greater) => Some(b),
        None => None,
    }
}

/// Return the maximum of two values with partial ordering
pub fn max_partial<T: PartialOrd>(a: T, b: T) -> Option<T> {
    match a.partial_cmp(&b) {
        Some(Ordering::Greater) | Some(Ordering::Equal) => Some(a),
        Some(Ordering::Less) => Some(b),
        None => None,
    }
}

/// Clamp a value to a range
///
/// # Example
/// ```d
/// let clamped = clamp(15, 0, 10);  // 10
/// let clamped = clamp(-5, 0, 10);  // 0
/// let clamped = clamp(5, 0, 10);   // 5
/// ```
pub fn clamp<T: Ord>(value: T, min_val: T, max_val: T) -> T {
    if value.cmp(&min_val).is_lt() {
        min_val
    } else if value.cmp(&max_val).is_gt() {
        max_val
    } else {
        value
    }
}

/// Find minimum in an iterator
///
/// Returns None for empty iterator
pub fn min_by<T, I, F>(iter: I, compare: F) -> Option<T>
where
    I: Iterator<Item = T>,
    F: fn(&T, &T) -> Ordering
{
    var result: Option<T> = None;
    for item in iter {
        result = match result {
            None => Some(item),
            Some(current) => {
                if compare(&item, &current).is_lt() {
                    Some(item)
                } else {
                    Some(current)
                }
            }
        };
    }
    result
}

/// Find maximum in an iterator
pub fn max_by<T, I, F>(iter: I, compare: F) -> Option<T>
where
    I: Iterator<Item = T>,
    F: fn(&T, &T) -> Ordering
{
    var result: Option<T> = None;
    for item in iter {
        result = match result {
            None => Some(item),
            Some(current) => {
                if compare(&item, &current).is_gt() {
                    Some(item)
                } else {
                    Some(current)
                }
            }
        };
    }
    result
}

// =============================================================================
// Implementations for Built-in Types
// =============================================================================

impl PartialOrd for i32 {
    fn partial_cmp(self: &i32, other: &i32) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for i32 {
    fn cmp(self: &i32, other: &i32) -> Ordering {
        if *self < *other {
            Ordering::Less
        } else if *self > *other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for i64 {
    fn partial_cmp(self: &i64, other: &i64) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for i64 {
    fn cmp(self: &i64, other: &i64) -> Ordering {
        if *self < *other {
            Ordering::Less
        } else if *self > *other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for u32 {
    fn partial_cmp(self: &u32, other: &u32) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for u32 {
    fn cmp(self: &u32, other: &u32) -> Ordering {
        if *self < *other {
            Ordering::Less
        } else if *self > *other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for u64 {
    fn partial_cmp(self: &u64, other: &u64) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for u64 {
    fn cmp(self: &u64, other: &u64) -> Ordering {
        if *self < *other {
            Ordering::Less
        } else if *self > *other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

impl PartialOrd for usize {
    fn partial_cmp(self: &usize, other: &usize) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for usize {
    fn cmp(self: &usize, other: &usize) -> Ordering {
        if *self < *other {
            Ordering::Less
        } else if *self > *other {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

// Float comparison with NaN handling
impl PartialOrd for f32 {
    fn partial_cmp(self: &f32, other: &f32) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else if *self < *other {
            Some(Ordering::Less)
        } else if *self > *other {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl PartialOrd for f64 {
    fn partial_cmp(self: &f64, other: &f64) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else if *self < *other {
            Some(Ordering::Less)
        } else if *self > *other {
            Some(Ordering::Greater)
        } else {
            Some(Ordering::Equal)
        }
    }
}

impl PartialOrd for String {
    fn partial_cmp(self: &String, other: &String) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for String {
    fn cmp(self: &String, other: &String) -> Ordering {
        self.as_str().cmp(other.as_str())
    }
}

impl PartialOrd for &str {
    fn partial_cmp(self: &&str, other: &&str) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for &str {
    fn cmp(self: &&str, other: &&str) -> Ordering {
        let a = *self;
        let b = *other;
        let min_len = min(a.len(), b.len());

        for i in 0..min_len {
            let ca = a.as_bytes()[i];
            let cb = b.as_bytes()[i];
            if ca < cb {
                return Ordering::Less;
            } else if ca > cb {
                return Ordering::Greater;
            }
        }

        // Prefixes match, compare lengths
        if a.len() < b.len() {
            Ordering::Less
        } else if a.len() > b.len() {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

fn test_min_max() {
    assert(min(10, 20) == 10);
    assert(min(20, 10) == 10);
    assert(min(10, 10) == 10);

    assert(max(10, 20) == 20);
    assert(max(20, 10) == 20);
    assert(max(10, 10) == 10);
}

fn test_clamp() {
    assert(clamp(15, 0, 10) == 10);
    assert(clamp(-5, 0, 10) == 0);
    assert(clamp(5, 0, 10) == 5);
}

fn test_ordering() {
    assert(Ordering::Less.is_lt());
    assert(!Ordering::Less.is_eq());
    assert(!Ordering::Less.is_gt());
    assert(Ordering::Less.is_le());
    assert(!Ordering::Less.is_ge());

    assert(Ordering::Less.reverse() == Ordering::Greater);
    assert(Ordering::Equal.reverse() == Ordering::Equal);
}

fn test_partial_ord_float() {
    let nan = 0.0 / 0.0;  // NaN
    assert(min_partial(1.0, 2.0) == Some(1.0));
    assert(min_partial(nan, 1.0) == None);
}
