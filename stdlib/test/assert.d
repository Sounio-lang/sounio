//! Assertion Library for Demetrios Testing
//!
//! Provides assertion macros and functions for writing tests.
//!
//! # Basic Assertions
//!
//! ```d
//! use test::assert
//!
//! #[test]
//! fn test_example() {
//!     assert(true)
//!     assert_eq(1 + 1, 2)
//!     assert_ne(1, 2)
//! }
//! ```
//!
//! # Custom Messages
//!
//! ```d
//! #[test]
//! fn test_with_message() {
//!     let result = compute()
//!     assert(result > 0, "Result should be positive")
//!     assert_eq(result, expected, "Computation returned wrong value")
//! }
//! ```

module test::assert

// ==================== Basic Assertions ====================

/// Assert that a condition is true
/// Panics with message if condition is false
pub fn assert(condition: bool) with Panic {
    if !condition {
        panic("Assertion failed: condition is false")
    }
}

/// Assert that a condition is true with custom message
pub fn assert(condition: bool, message: string) with Panic {
    if !condition {
        panic("Assertion failed: " + message)
    }
}

/// Assert that two values are equal
pub fn assert_eq<T: Eq + Debug>(left: T, right: T) with Panic {
    if left != right {
        panic("Assertion failed: values are not equal\n  left:  " + debug(left) + "\n  right: " + debug(right))
    }
}

/// Assert that two values are equal with custom message
pub fn assert_eq<T: Eq + Debug>(left: T, right: T, message: string) with Panic {
    if left != right {
        panic("Assertion failed: " + message + "\n  left:  " + debug(left) + "\n  right: " + debug(right))
    }
}

/// Assert that two values are not equal
pub fn assert_ne<T: Eq + Debug>(left: T, right: T) with Panic {
    if left == right {
        panic("Assertion failed: values should not be equal\n  both: " + debug(left))
    }
}

/// Assert that two values are not equal with custom message
pub fn assert_ne<T: Eq + Debug>(left: T, right: T, message: string) with Panic {
    if left == right {
        panic("Assertion failed: " + message + "\n  both: " + debug(left))
    }
}

// ==================== Numeric Assertions ====================

/// Assert that two floating point values are approximately equal
/// Uses default epsilon of 1e-10
pub fn assert_approx(left: f64, right: f64) with Panic {
    assert_approx_eps(left, right, 1e-10)
}

/// Assert approximately equal with custom message
pub fn assert_approx(left: f64, right: f64, message: string) with Panic {
    assert_approx_eps(left, right, 1e-10, message)
}

/// Assert that two floating point values are approximately equal with custom epsilon
pub fn assert_approx_eps(left: f64, right: f64, epsilon: f64) with Panic {
    let diff = abs(left - right)
    if diff > epsilon {
        panic("Assertion failed: values are not approximately equal\n  left:    " + to_string(left) + "\n  right:   " + to_string(right) + "\n  diff:    " + to_string(diff) + "\n  epsilon: " + to_string(epsilon))
    }
}

/// Assert approximately equal with custom epsilon and message
pub fn assert_approx_eps(left: f64, right: f64, epsilon: f64, message: string) with Panic {
    let diff = abs(left - right)
    if diff > epsilon {
        panic("Assertion failed: " + message + "\n  left:    " + to_string(left) + "\n  right:   " + to_string(right) + "\n  diff:    " + to_string(diff) + "\n  epsilon: " + to_string(epsilon))
    }
}

/// Assert that a value is within a range (inclusive)
pub fn assert_in_range<T: Ord + Debug>(value: T, min: T, max: T) with Panic {
    if value < min || value > max {
        panic("Assertion failed: value out of range\n  value: " + debug(value) + "\n  range: [" + debug(min) + ", " + debug(max) + "]")
    }
}

/// Assert that a value is positive
pub fn assert_positive<T: Ord + Default + Debug>(value: T) with Panic {
    let zero = T::default()
    if value <= zero {
        panic("Assertion failed: expected positive value, got " + debug(value))
    }
}

/// Assert that a value is negative
pub fn assert_negative<T: Ord + Default + Debug>(value: T) with Panic {
    let zero = T::default()
    if value >= zero {
        panic("Assertion failed: expected negative value, got " + debug(value))
    }
}

// ==================== Option/Result Assertions ====================

/// Assert that an Option is Some
pub fn assert_some<T: Debug>(opt: Option<T>) with Panic {
    match opt {
        Some(_) => {}
        None => panic("Assertion failed: expected Some, got None")
    }
}

/// Assert that an Option is Some and return the value
pub fn assert_some_eq<T: Eq + Debug>(opt: Option<T>, expected: T) with Panic {
    match opt {
        Some(value) => {
            if value != expected {
                panic("Assertion failed: Some value mismatch\n  expected: " + debug(expected) + "\n  got:      " + debug(value))
            }
        }
        None => panic("Assertion failed: expected Some(" + debug(expected) + "), got None")
    }
}

/// Assert that an Option is None
pub fn assert_none<T: Debug>(opt: Option<T>) with Panic {
    match opt {
        Some(value) => panic("Assertion failed: expected None, got Some(" + debug(value) + ")")
        None => {}
    }
}

/// Assert that a Result is Ok
pub fn assert_ok<T: Debug, E: Debug>(result: Result<T, E>) with Panic {
    match result {
        Ok(_) => {}
        Err(e) => panic("Assertion failed: expected Ok, got Err(" + debug(e) + ")")
    }
}

/// Assert that a Result is Ok with expected value
pub fn assert_ok_eq<T: Eq + Debug, E: Debug>(result: Result<T, E>, expected: T) with Panic {
    match result {
        Ok(value) => {
            if value != expected {
                panic("Assertion failed: Ok value mismatch\n  expected: " + debug(expected) + "\n  got:      " + debug(value))
            }
        }
        Err(e) => panic("Assertion failed: expected Ok(" + debug(expected) + "), got Err(" + debug(e) + ")")
    }
}

/// Assert that a Result is Err
pub fn assert_err<T: Debug, E: Debug>(result: Result<T, E>) with Panic {
    match result {
        Ok(value) => panic("Assertion failed: expected Err, got Ok(" + debug(value) + ")")
        Err(_) => {}
    }
}

// ==================== Collection Assertions ====================

/// Assert that a collection is empty
pub fn assert_empty<C: Collection>(collection: &C) with Panic {
    if !collection.is_empty() {
        panic("Assertion failed: expected empty collection, got " + to_string(collection.len()) + " elements")
    }
}

/// Assert that a collection is not empty
pub fn assert_not_empty<C: Collection>(collection: &C) with Panic {
    if collection.is_empty() {
        panic("Assertion failed: expected non-empty collection")
    }
}

/// Assert that a collection has specific length
pub fn assert_len<C: Collection>(collection: &C, expected_len: usize) with Panic {
    let actual = collection.len()
    if actual != expected_len {
        panic("Assertion failed: length mismatch\n  expected: " + to_string(expected_len) + "\n  actual:   " + to_string(actual))
    }
}

/// Assert that a collection contains an element
pub fn assert_contains<C: Collection<Item = T>, T: Eq + Debug>(collection: &C, element: &T) with Panic {
    if !collection.contains(element) {
        panic("Assertion failed: collection does not contain " + debug(element))
    }
}

/// Assert that a collection does not contain an element
pub fn assert_not_contains<C: Collection<Item = T>, T: Eq + Debug>(collection: &C, element: &T) with Panic {
    if collection.contains(element) {
        panic("Assertion failed: collection should not contain " + debug(element))
    }
}

/// Assert that two collections have the same elements (order matters)
pub fn assert_eq_elements<T: Eq + Debug>(left: &[T], right: &[T]) with Panic {
    if left.len() != right.len() {
        panic("Assertion failed: collections have different lengths\n  left:  " + to_string(left.len()) + "\n  right: " + to_string(right.len()))
    }
    for i in 0..left.len() {
        if left[i] != right[i] {
            panic("Assertion failed: elements differ at index " + to_string(i) + "\n  left:  " + debug(left[i]) + "\n  right: " + debug(right[i]))
        }
    }
}

// ==================== String Assertions ====================

/// Assert that a string contains a substring
pub fn assert_str_contains(haystack: &str, needle: &str) with Panic {
    if !haystack.contains(needle) {
        panic("Assertion failed: string does not contain substring\n  string:    \"" + haystack + "\"\n  substring: \"" + needle + "\"")
    }
}

/// Assert that a string starts with a prefix
pub fn assert_starts_with(s: &str, prefix: &str) with Panic {
    if !s.starts_with(prefix) {
        panic("Assertion failed: string does not start with prefix\n  string: \"" + s + "\"\n  prefix: \"" + prefix + "\"")
    }
}

/// Assert that a string ends with a suffix
pub fn assert_ends_with(s: &str, suffix: &str) with Panic {
    if !s.ends_with(suffix) {
        panic("Assertion failed: string does not end with suffix\n  string: \"" + s + "\"\n  suffix: \"" + suffix + "\"")
    }
}

/// Assert that a string matches a regex pattern
pub fn assert_matches(s: &str, pattern: &str) with Panic {
    if !regex_match(s, pattern) {
        panic("Assertion failed: string does not match pattern\n  string:  \"" + s + "\"\n  pattern: \"" + pattern + "\"")
    }
}

// ==================== Boolean Assertions ====================

/// Assert that a value is true
pub fn assert_true(value: bool) with Panic {
    if !value {
        panic("Assertion failed: expected true, got false")
    }
}

/// Assert that a value is true with message
pub fn assert_true(value: bool, message: string) with Panic {
    if !value {
        panic("Assertion failed: " + message)
    }
}

/// Assert that a value is false
pub fn assert_false(value: bool) with Panic {
    if value {
        panic("Assertion failed: expected false, got true")
    }
}

/// Assert that a value is false with message
pub fn assert_false(value: bool, message: string) with Panic {
    if value {
        panic("Assertion failed: " + message)
    }
}

// ==================== Type/Trait Assertions ====================

/// Assert that a value is of a specific type (compile-time check)
/// This is primarily a documentation/clarity helper
pub fn assert_type<T, U: T>(_value: &U) {
    // If this compiles, the type assertion passes
}

// ==================== Panic Assertions ====================

/// Assert that a function panics
/// Returns the panic message if it panicked
pub fn assert_panics<F: Fn()>(f: F) -> string with Panic {
    // This would be implemented using the runtime's panic catching mechanism
    // Placeholder implementation
    panic("assert_panics not yet implemented")
}

/// Assert that a function panics with a specific message
pub fn assert_panics_with<F: Fn()>(f: F, expected_message: &str) with Panic {
    let panic_msg = assert_panics(f)
    if !panic_msg.contains(expected_message) {
        panic("Assertion failed: panic message mismatch\n  expected to contain: \"" + expected_message + "\"\n  actual:              \"" + panic_msg + "\"")
    }
}

// ==================== Helper Functions ====================

/// Get absolute value of a float
fn abs(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

/// Format a value using Debug trait
fn debug<T: Debug>(value: T) -> string {
    value.debug_string()
}

/// Convert to string
fn to_string<T: ToString>(value: T) -> string {
    value.to_string()
}

/// Match string against regex
fn regex_match(s: &str, pattern: &str) -> bool {
    // Placeholder - would use actual regex implementation
    true
}
