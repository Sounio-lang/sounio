//! Option type for optional values
//!
//! Option<T> represents an optional value: every Option is either
//! Some(T) containing a value, or None representing absence.

/// A type that represents an optional value.
pub enum Option<T> {
    /// No value
    None,

    /// Some value of type T
    Some(T),
}

impl<T> Option<T> {
    /// Returns true if the option is a Some value.
    pub fn is_some(self: &Option<T>) -> bool {
        match self {
            Option::Some(_) => true,
            Option::None => false,
        }
    }

    /// Returns true if the option is a None value.
    pub fn is_none(self: &Option<T>) -> bool {
        match self {
            Option::Some(_) => false,
            Option::None => true,
        }
    }

    /// Returns the contained Some value, consuming the self value.
    /// Panics if the value is None.
    pub fn unwrap(self: Option<T>) -> T with Panic {
        match self {
            Option::Some(val) => val,
            Option::None => panic("called Option::unwrap() on a None value"),
        }
    }

    /// Returns the contained Some value or a provided default.
    pub fn unwrap_or(self: Option<T>, default: T) -> T {
        match self {
            Option::Some(val) => val,
            Option::None => default,
        }
    }

    /// Returns the contained Some value or computes it from a closure.
    pub fn unwrap_or_else<F>(self: Option<T>, f: F) -> T
    where F: fn() -> T
    {
        match self {
            Option::Some(val) => val,
            Option::None => f(),
        }
    }

    /// Returns the contained Some value or a default.
    pub fn unwrap_or_default(self: Option<T>) -> T
    where T: Default
    {
        match self {
            Option::Some(val) => val,
            Option::None => T::default(),
        }
    }

    /// Maps an Option<T> to Option<U> by applying a function.
    pub fn map<U, F>(self: Option<T>, f: F) -> Option<U>
    where F: fn(T) -> U
    {
        match self {
            Option::Some(val) => Option::Some(f(val)),
            Option::None => Option::None,
        }
    }

    /// Applies a function to the contained value (if Some),
    /// or returns the provided default (if None).
    pub fn map_or<U, F>(self: Option<T>, default: U, f: F) -> U
    where F: fn(T) -> U
    {
        match self {
            Option::Some(val) => f(val),
            Option::None => default,
        }
    }

    /// Applies a function to the contained value (if Some),
    /// or computes a default (if None).
    pub fn map_or_else<U, D, F>(self: Option<T>, default: D, f: F) -> U
    where
        D: fn() -> U,
        F: fn(T) -> U
    {
        match self {
            Option::Some(val) => f(val),
            Option::None => default(),
        }
    }

    /// Returns None if the option is None, otherwise calls predicate
    /// with the wrapped value and returns Some(t) if predicate returns true.
    pub fn filter<P>(self: Option<T>, predicate: P) -> Option<T>
    where P: fn(&T) -> bool
    {
        match self {
            Option::Some(val) => {
                if predicate(&val) {
                    Option::Some(val)
                } else {
                    Option::None
                }
            },
            Option::None => Option::None,
        }
    }

    /// Returns None if the option is None, otherwise calls f with the
    /// wrapped value and returns the result.
    pub fn and_then<U, F>(self: Option<T>, f: F) -> Option<U>
    where F: fn(T) -> Option<U>
    {
        match self {
            Option::Some(val) => f(val),
            Option::None => Option::None,
        }
    }

    /// Returns the option if it contains a value, otherwise returns optb.
    pub fn or(self: Option<T>, optb: Option<T>) -> Option<T> {
        match self {
            Option::Some(_) => self,
            Option::None => optb,
        }
    }

    /// Returns the option if it contains a value, otherwise calls f and
    /// returns the result.
    pub fn or_else<F>(self: Option<T>, f: F) -> Option<T>
    where F: fn() -> Option<T>
    {
        match self {
            Option::Some(_) => self,
            Option::None => f(),
        }
    }

    /// Returns Some if exactly one of self, optb is Some, otherwise returns None.
    pub fn xor(self: Option<T>, optb: Option<T>) -> Option<T> {
        match (self, optb) {
            (Option::Some(a), Option::None) => Option::Some(a),
            (Option::None, Option::Some(b)) => Option::Some(b),
            _ => Option::None,
        }
    }

    /// Takes the value out of the option, leaving a None in its place.
    pub fn take(self: &!Option<T>) -> Option<T> {
        let val = *self
        *self = Option::None
        val
    }

    /// Replaces the actual value in the option with the value given,
    /// returning the old value if present.
    pub fn replace(self: &!Option<T>, value: T) -> Option<T> {
        let old = *self
        *self = Option::Some(value)
        old
    }

    /// Converts from Option<T> to Option<&T>.
    pub fn as_ref(self: &Option<T>) -> Option<&T> {
        match self {
            Option::Some(ref val) => Option::Some(val),
            Option::None => Option::None,
        }
    }

    /// Converts from Option<T> to Option<&!T>.
    pub fn as_mut(self: &!Option<T>) -> Option<&!T> {
        match self {
            Option::Some(ref mut val) => Option::Some(val),
            Option::None => Option::None,
        }
    }

    /// Returns an iterator over the possibly contained value.
    pub fn iter(self: &Option<T>) -> OptionIter<T> {
        OptionIter { opt: self.as_ref() }
    }

    /// Transforms the Option<T> into a Result<T, E>, mapping Some(v) to Ok(v)
    /// and None to Err(err).
    pub fn ok_or<E>(self: Option<T>, err: E) -> Result<T, E> {
        match self {
            Option::Some(val) => Result::Ok(val),
            Option::None => Result::Err(err),
        }
    }

    /// Transforms the Option<T> into a Result<T, E>, mapping Some(v) to Ok(v)
    /// and None to Err(err()).
    pub fn ok_or_else<E, F>(self: Option<T>, err: F) -> Result<T, E>
    where F: fn() -> E
    {
        match self {
            Option::Some(val) => Result::Ok(val),
            Option::None => Result::Err(err()),
        }
    }

    /// Zips self with another Option.
    pub fn zip<U>(self: Option<T>, other: Option<U>) -> Option<(T, U)> {
        match (self, other) {
            (Option::Some(a), Option::Some(b)) => Option::Some((a, b)),
            _ => Option::None,
        }
    }

    /// Zips self and another Option with a function.
    pub fn zip_with<U, R, F>(self: Option<T>, other: Option<U>, f: F) -> Option<R>
    where F: fn(T, U) -> R
    {
        match (self, other) {
            (Option::Some(a), Option::Some(b)) => Option::Some(f(a, b)),
            _ => Option::None,
        }
    }

    /// Unzips an option containing a tuple of two options.
    pub fn unzip<A, B>(self: Option<(A, B)>) -> (Option<A>, Option<B>) {
        match self {
            Option::Some((a, b)) => (Option::Some(a), Option::Some(b)),
            Option::None => (Option::None, Option::None),
        }
    }

    /// Returns true if the option is a Some value containing the given value.
    pub fn contains<U>(self: &Option<T>, x: &U) -> bool
    where T: PartialEq<U>
    {
        match self {
            Option::Some(ref val) => val == x,
            Option::None => false,
        }
    }

    /// Inserts a value computed from f into the option if it is None,
    /// then returns a mutable reference to the contained value.
    pub fn get_or_insert_with<F>(self: &!Option<T>, f: F) -> &!T
    where F: fn() -> T
    {
        if self.is_none() {
            *self = Option::Some(f())
        }

        match self {
            Option::Some(ref mut val) => val,
            Option::None => unreachable(),
        }
    }

    /// Inserts value into the option if it is None, then returns a
    /// mutable reference to the contained value.
    pub fn get_or_insert(self: &!Option<T>, value: T) -> &!T {
        self.get_or_insert_with(|| value)
    }

    /// Flattens Option<Option<T>> to Option<T>.
    pub fn flatten(self: Option<Option<T>>) -> Option<T> {
        match self {
            Option::Some(inner) => inner,
            Option::None => Option::None,
        }
    }

    /// Returns the contained Some value, without checking that the value is not None.
    ///
    /// # Safety
    /// Calling this method on a None value is undefined behavior.
    pub unsafe fn unwrap_unchecked(self: Option<T>) -> T {
        match self {
            Option::Some(val) => val,
            Option::None => unreachable(),
        }
    }

    /// Expects the option to be Some, panics with a custom message if None.
    pub fn expect(self: Option<T>, msg: &str) -> T with Panic {
        match self {
            Option::Some(val) => val,
            Option::None => panic(msg),
        }
    }
}

impl<T> Clone for Option<T>
where T: Clone
{
    fn clone(self: &Option<T>) -> Option<T> {
        match self {
            Option::Some(ref val) => Option::Some(val.clone()),
            Option::None => Option::None,
        }
    }
}

impl<T> Default for Option<T> {
    fn default() -> Option<T> {
        Option::None
    }
}

impl<T> Eq for Option<T>
where T: Eq
{
    fn eq(self: &Option<T>, other: &Option<T>) -> bool {
        match (self, other) {
            (Option::Some(ref a), Option::Some(ref b)) => a == b,
            (Option::None, Option::None) => true,
            _ => false,
        }
    }
}

impl<T> Ord for Option<T>
where T: Ord
{
    fn cmp(self: &Option<T>, other: &Option<T>) -> Ordering {
        match (self, other) {
            (Option::Some(ref a), Option::Some(ref b)) => a.cmp(b),
            (Option::Some(_), Option::None) => Ordering::Greater,
            (Option::None, Option::Some(_)) => Ordering::Less,
            (Option::None, Option::None) => Ordering::Equal,
        }
    }
}

impl<T> Debug for Option<T>
where T: Debug
{
    fn fmt(self: &Option<T>, f: &!Formatter) -> Result<unit, FmtError> {
        match self {
            Option::Some(ref val) => {
                f.write_str("Some(")?
                val.fmt(f)?
                f.write_str(")")
            },
            Option::None => f.write_str("None"),
        }
    }
}

impl<T> Hash for Option<T>
where T: Hash
{
    fn hash<H>(self: &Option<T>, state: &!H)
    where H: Hasher
    {
        match self {
            Option::Some(ref val) => {
                state.write_u8(1)
                val.hash(state)
            },
            Option::None => state.write_u8(0),
        }
    }
}

/// Iterator over an Option.
pub struct OptionIter<T> {
    opt: Option<&T>,
}

impl<T> Iterator for OptionIter<T> {
    type Item = &T

    fn next(self: &!OptionIter<T>) -> Option<&T> {
        self.opt.take()
    }

    fn size_hint(self: &OptionIter<T>) -> (int, Option<int>) {
        let n = if self.opt.is_some() { 1 } else { 0 }
        (n, Option::Some(n))
    }
}

impl<T> IntoIterator for Option<T> {
    type Item = T
    type IntoIter = OptionIntoIter<T>

    fn into_iter(self: Option<T>) -> OptionIntoIter<T> {
        OptionIntoIter { opt: self }
    }
}

/// Owning iterator over an Option.
pub struct OptionIntoIter<T> {
    opt: Option<T>,
}

impl<T> Iterator for OptionIntoIter<T> {
    type Item = T

    fn next(self: &!OptionIntoIter<T>) -> Option<T> {
        self.opt.take()
    }
}

// Convenience functions

/// Creates a Some value.
pub fn Some<T>(value: T) -> Option<T> {
    Option::Some(value)
}

/// Returns the None value.
pub fn None<T>() -> Option<T> {
    Option::None
}
