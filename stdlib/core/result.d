//! Result type for error handling
//!
//! Result<T, E> is the type used for returning and propagating errors.
//! It is an enum with variants Ok(T) and Err(E).

/// A type that represents either success (Ok) or failure (Err).
pub enum Result<T, E> {
    /// Contains the success value
    Ok(T),

    /// Contains the error value
    Err(E),
}

impl<T, E> Result<T, E> {
    /// Returns true if the result is Ok.
    pub fn is_ok(self: &Result<T, E>) -> bool {
        match self {
            Result::Ok(_) => true,
            Result::Err(_) => false,
        }
    }

    /// Returns true if the result is Err.
    pub fn is_err(self: &Result<T, E>) -> bool {
        match self {
            Result::Ok(_) => false,
            Result::Err(_) => true,
        }
    }

    /// Returns true if the result is an Ok value containing the given value.
    pub fn is_ok_and<F>(self: Result<T, E>, f: F) -> bool
    where F: fn(T) -> bool
    {
        match self {
            Result::Ok(val) => f(val),
            Result::Err(_) => false,
        }
    }

    /// Returns true if the result is an Err value containing the given value.
    pub fn is_err_and<F>(self: Result<T, E>, f: F) -> bool
    where F: fn(E) -> bool
    {
        match self {
            Result::Ok(_) => false,
            Result::Err(err) => f(err),
        }
    }

    /// Returns the contained Ok value, consuming the self value.
    /// Panics if the value is an Err.
    pub fn unwrap(self: Result<T, E>) -> T with Panic
    where E: Debug
    {
        match self {
            Result::Ok(val) => val,
            Result::Err(err) => panic("called Result::unwrap() on an Err value: {:?}", err),
        }
    }

    /// Returns the contained Err value, consuming the self value.
    /// Panics if the value is an Ok.
    pub fn unwrap_err(self: Result<T, E>) -> E with Panic
    where T: Debug
    {
        match self {
            Result::Ok(val) => panic("called Result::unwrap_err() on an Ok value: {:?}", val),
            Result::Err(err) => err,
        }
    }

    /// Returns the contained Ok value or a provided default.
    pub fn unwrap_or(self: Result<T, E>, default: T) -> T {
        match self {
            Result::Ok(val) => val,
            Result::Err(_) => default,
        }
    }

    /// Returns the contained Ok value or computes it from a closure.
    pub fn unwrap_or_else<F>(self: Result<T, E>, f: F) -> T
    where F: fn(E) -> T
    {
        match self {
            Result::Ok(val) => val,
            Result::Err(err) => f(err),
        }
    }

    /// Returns the contained Ok value or a default.
    pub fn unwrap_or_default(self: Result<T, E>) -> T
    where T: Default
    {
        match self {
            Result::Ok(val) => val,
            Result::Err(_) => T::default(),
        }
    }

    /// Returns the contained Ok value, consuming the self value,
    /// without checking that the value is not an Err.
    ///
    /// # Safety
    /// Calling this method on an Err is undefined behavior.
    pub unsafe fn unwrap_unchecked(self: Result<T, E>) -> T {
        match self {
            Result::Ok(val) => val,
            Result::Err(_) => unreachable(),
        }
    }

    /// Returns the contained Err value, consuming the self value,
    /// without checking that the value is not an Ok.
    ///
    /// # Safety
    /// Calling this method on an Ok is undefined behavior.
    pub unsafe fn unwrap_err_unchecked(self: Result<T, E>) -> E {
        match self {
            Result::Ok(_) => unreachable(),
            Result::Err(err) => err,
        }
    }

    /// Expects the result to be Ok, panics with a custom message if Err.
    pub fn expect(self: Result<T, E>, msg: &str) -> T with Panic
    where E: Debug
    {
        match self {
            Result::Ok(val) => val,
            Result::Err(err) => panic("{}: {:?}", msg, err),
        }
    }

    /// Expects the result to be Err, panics with a custom message if Ok.
    pub fn expect_err(self: Result<T, E>, msg: &str) -> E with Panic
    where T: Debug
    {
        match self {
            Result::Ok(val) => panic("{}: {:?}", msg, val),
            Result::Err(err) => err,
        }
    }

    /// Maps a Result<T, E> to Result<U, E> by applying a function.
    pub fn map<U, F>(self: Result<T, E>, f: F) -> Result<U, E>
    where F: fn(T) -> U
    {
        match self {
            Result::Ok(val) => Result::Ok(f(val)),
            Result::Err(err) => Result::Err(err),
        }
    }

    /// Maps a Result<T, E> to Result<T, F> by applying a function to the error.
    pub fn map_err<F, O>(self: Result<T, E>, op: O) -> Result<T, F>
    where O: fn(E) -> F
    {
        match self {
            Result::Ok(val) => Result::Ok(val),
            Result::Err(err) => Result::Err(op(err)),
        }
    }

    /// Applies a function to the contained value (if Ok),
    /// or returns the provided default (if Err).
    pub fn map_or<U, F>(self: Result<T, E>, default: U, f: F) -> U
    where F: fn(T) -> U
    {
        match self {
            Result::Ok(val) => f(val),
            Result::Err(_) => default,
        }
    }

    /// Maps a Result<T, E> to U by applying fallback function default to
    /// a contained Err value, or function f to a contained Ok value.
    pub fn map_or_else<U, D, F>(self: Result<T, E>, default: D, f: F) -> U
    where
        D: fn(E) -> U,
        F: fn(T) -> U
    {
        match self {
            Result::Ok(val) => f(val),
            Result::Err(err) => default(err),
        }
    }

    /// Calls op if the result is Ok, otherwise returns the Err value.
    pub fn and_then<U, F>(self: Result<T, E>, op: F) -> Result<U, E>
    where F: fn(T) -> Result<U, E>
    {
        match self {
            Result::Ok(val) => op(val),
            Result::Err(err) => Result::Err(err),
        }
    }

    /// Calls op if the result is Err, otherwise returns the Ok value.
    pub fn or_else<F, O>(self: Result<T, E>, op: O) -> Result<T, F>
    where O: fn(E) -> Result<T, F>
    {
        match self {
            Result::Ok(val) => Result::Ok(val),
            Result::Err(err) => op(err),
        }
    }

    /// Returns res if the result is Ok, otherwise returns the Err value.
    pub fn and<U>(self: Result<T, E>, res: Result<U, E>) -> Result<U, E> {
        match self {
            Result::Ok(_) => res,
            Result::Err(err) => Result::Err(err),
        }
    }

    /// Returns res if the result is Err, otherwise returns the Ok value.
    pub fn or<F>(self: Result<T, E>, res: Result<T, F>) -> Result<T, F> {
        match self {
            Result::Ok(val) => Result::Ok(val),
            Result::Err(_) => res,
        }
    }

    /// Converts from Result<T, E> to Option<T>.
    pub fn ok(self: Result<T, E>) -> Option<T> {
        match self {
            Result::Ok(val) => Option::Some(val),
            Result::Err(_) => Option::None,
        }
    }

    /// Converts from Result<T, E> to Option<E>.
    pub fn err(self: Result<T, E>) -> Option<E> {
        match self {
            Result::Ok(_) => Option::None,
            Result::Err(err) => Option::Some(err),
        }
    }

    /// Converts from &Result<T, E> to Result<&T, &E>.
    pub fn as_ref(self: &Result<T, E>) -> Result<&T, &E> {
        match self {
            Result::Ok(ref val) => Result::Ok(val),
            Result::Err(ref err) => Result::Err(err),
        }
    }

    /// Converts from &!Result<T, E> to Result<&!T, &!E>.
    pub fn as_mut(self: &!Result<T, E>) -> Result<&!T, &!E> {
        match self {
            Result::Ok(ref mut val) => Result::Ok(val),
            Result::Err(ref mut err) => Result::Err(err),
        }
    }

    /// Transposes a Result of an Option into an Option of a Result.
    pub fn transpose(self: Result<Option<T>, E>) -> Option<Result<T, E>> {
        match self {
            Result::Ok(Option::Some(val)) => Option::Some(Result::Ok(val)),
            Result::Ok(Option::None) => Option::None,
            Result::Err(err) => Option::Some(Result::Err(err)),
        }
    }

    /// Returns true if the result is an Ok value containing the given value.
    pub fn contains<U>(self: &Result<T, E>, x: &U) -> bool
    where T: PartialEq<U>
    {
        match self {
            Result::Ok(ref val) => val == x,
            Result::Err(_) => false,
        }
    }

    /// Returns true if the result is an Err value containing the given value.
    pub fn contains_err<F>(self: &Result<T, E>, f: &F) -> bool
    where E: PartialEq<F>
    {
        match self {
            Result::Ok(_) => false,
            Result::Err(ref err) => err == f,
        }
    }

    /// Returns an iterator over the possibly contained value.
    pub fn iter(self: &Result<T, E>) -> ResultIter<T> {
        ResultIter { opt: self.as_ref().ok() }
    }

    /// Flattens Result<Result<T, E>, E> to Result<T, E>.
    pub fn flatten(self: Result<Result<T, E>, E>) -> Result<T, E> {
        match self {
            Result::Ok(inner) => inner,
            Result::Err(err) => Result::Err(err),
        }
    }

    /// Copies the result if the value is Copy.
    pub fn copied(self: Result<&T, E>) -> Result<T, E>
    where T: Copy
    {
        self.map(|val| *val)
    }

    /// Clones the result if the value is Clone.
    pub fn cloned(self: Result<&T, E>) -> Result<T, E>
    where T: Clone
    {
        self.map(|val| val.clone())
    }
}

impl<T, E> Clone for Result<T, E>
where T: Clone, E: Clone
{
    fn clone(self: &Result<T, E>) -> Result<T, E> {
        match self {
            Result::Ok(ref val) => Result::Ok(val.clone()),
            Result::Err(ref err) => Result::Err(err.clone()),
        }
    }
}

impl<T, E> Eq for Result<T, E>
where T: Eq, E: Eq
{
    fn eq(self: &Result<T, E>, other: &Result<T, E>) -> bool {
        match (self, other) {
            (Result::Ok(ref a), Result::Ok(ref b)) => a == b,
            (Result::Err(ref a), Result::Err(ref b)) => a == b,
            _ => false,
        }
    }
}

impl<T, E> Ord for Result<T, E>
where T: Ord, E: Ord
{
    fn cmp(self: &Result<T, E>, other: &Result<T, E>) -> Ordering {
        match (self, other) {
            (Result::Ok(ref a), Result::Ok(ref b)) => a.cmp(b),
            (Result::Err(ref a), Result::Err(ref b)) => a.cmp(b),
            (Result::Ok(_), Result::Err(_)) => Ordering::Less,
            (Result::Err(_), Result::Ok(_)) => Ordering::Greater,
        }
    }
}

impl<T, E> Debug for Result<T, E>
where T: Debug, E: Debug
{
    fn fmt(self: &Result<T, E>, f: &!Formatter) -> Result<unit, FmtError> {
        match self {
            Result::Ok(ref val) => {
                f.write_str("Ok(")?
                val.fmt(f)?
                f.write_str(")")
            },
            Result::Err(ref err) => {
                f.write_str("Err(")?
                err.fmt(f)?
                f.write_str(")")
            },
        }
    }
}

impl<T, E> Hash for Result<T, E>
where T: Hash, E: Hash
{
    fn hash<H>(self: &Result<T, E>, state: &!H)
    where H: Hasher
    {
        match self {
            Result::Ok(ref val) => {
                state.write_u8(0)
                val.hash(state)
            },
            Result::Err(ref err) => {
                state.write_u8(1)
                err.hash(state)
            },
        }
    }
}

/// Iterator over a Result.
pub struct ResultIter<T> {
    opt: Option<&T>,
}

impl<T> Iterator for ResultIter<T> {
    type Item = &T

    fn next(self: &!ResultIter<T>) -> Option<&T> {
        self.opt.take()
    }
}

impl<T, E> IntoIterator for Result<T, E> {
    type Item = T
    type IntoIter = ResultIntoIter<T>

    fn into_iter(self: Result<T, E>) -> ResultIntoIter<T> {
        ResultIntoIter { opt: self.ok() }
    }
}

/// Owning iterator over a Result.
pub struct ResultIntoIter<T> {
    opt: Option<T>,
}

impl<T> Iterator for ResultIntoIter<T> {
    type Item = T

    fn next(self: &!ResultIntoIter<T>) -> Option<T> {
        self.opt.take()
    }
}

impl<A, E> FromIterator<Result<A, E>> for Result<Vec<A>, E> {
    fn from_iter<I>(iter: I) -> Result<Vec<A>, E> with Alloc
    where I: IntoIterator<Item = Result<A, E>>
    {
        let mut vec = Vec::new()

        for item in iter {
            match item {
                Result::Ok(val) => vec.push(val),
                Result::Err(err) => return Result::Err(err),
            }
        }

        Result::Ok(vec)
    }
}

// Convenience functions

/// Creates an Ok value.
pub fn Ok<T, E>(value: T) -> Result<T, E> {
    Result::Ok(value)
}

/// Creates an Err value.
pub fn Err<T, E>(error: E) -> Result<T, E> {
    Result::Err(error)
}

/// The ? operator for Result
/// Propagates the error if Result is Err, otherwise unwraps the Ok value.
#[lang = "try_result"]
pub fn try_result<T, E>(result: Result<T, E>) -> T with Propagate<E> {
    match result {
        Result::Ok(val) => val,
        Result::Err(err) => return Result::Err(err),
    }
}
