//! Advanced Assertion Library for Demetrios Testing
//!
//! Provides advanced assertion patterns including:
//! - Soft assertions (collect multiple failures)
//! - Assertion chains
//! - Custom matchers
//!
//! # Soft Assertions
//!
//! ```d
//! use test::assert_advanced::*
//!
//! #[test]
//! fn test_multiple_conditions() {
//!     let soft = SoftAssertions::new()
//!
//!     soft.assert_eq(result.name, "expected")
//!     soft.assert_true(result.valid)
//!     soft.assert_in_range(result.score, 0, 100)
//!
//!     soft.assert_all()  // Panics if any assertion failed
//! }
//! ```

module test::assert_advanced

use test::assert::*

// ==================== Soft Assertions ====================

/// A soft assertion that collects failures instead of immediately panicking
struct AssertionFailure {
    message: string,
    location: string,
}

/// Soft assertions collector - allows multiple assertions to be checked
/// and reports all failures at once
pub struct SoftAssertions {
    failures: Vec<AssertionFailure>,
    location_prefix: string,
}

impl SoftAssertions {
    /// Create a new soft assertions collector
    pub fn new() -> Self {
        SoftAssertions {
            failures: Vec::new(),
            location_prefix: "",
        }
    }

    /// Create with a location prefix for better error messages
    pub fn with_context(context: string) -> Self {
        SoftAssertions {
            failures: Vec::new(),
            location_prefix: context + ": ",
        }
    }

    /// Check if all assertions passed
    pub fn all_passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// Get number of failures
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }

    /// Assert all collected assertions passed, panic if any failed
    pub fn assert_all(&self) with Panic {
        if !self.failures.is_empty() {
            let mut message = "Soft assertions failed (" + to_string(self.failures.len()) + " failures):\n"
            for (i, failure) in self.failures.iter().enumerate() {
                message = message + "\n  " + to_string(i + 1) + ". " + failure.message
                if !failure.location.is_empty() {
                    message = message + " at " + failure.location
                }
            }
            panic(message)
        }
    }

    /// Get failures as a report string
    pub fn report(&self) -> string {
        if self.failures.is_empty() {
            return "All soft assertions passed"
        }

        let mut message = "Soft assertion failures:\n"
        for (i, failure) in self.failures.iter().enumerate() {
            message = message + "  " + to_string(i + 1) + ". " + failure.message + "\n"
        }
        message
    }

    // ==================== Soft Assertion Methods ====================

    /// Soft assert condition is true
    pub fn assert(&mut self, condition: bool) {
        if !condition {
            self.add_failure("condition is false")
        }
    }

    /// Soft assert condition is true with message
    pub fn assert(&mut self, condition: bool, message: string) {
        if !condition {
            self.add_failure(message)
        }
    }

    /// Soft assert equality
    pub fn assert_eq<T: Eq + Debug>(&mut self, left: T, right: T) {
        if left != right {
            self.add_failure("values not equal: " + debug(left) + " != " + debug(right))
        }
    }

    /// Soft assert equality with message
    pub fn assert_eq<T: Eq + Debug>(&mut self, left: T, right: T, message: string) {
        if left != right {
            self.add_failure(message + ": " + debug(left) + " != " + debug(right))
        }
    }

    /// Soft assert inequality
    pub fn assert_ne<T: Eq + Debug>(&mut self, left: T, right: T) {
        if left == right {
            self.add_failure("values should not be equal: " + debug(left))
        }
    }

    /// Soft assert true
    pub fn assert_true(&mut self, value: bool) {
        if !value {
            self.add_failure("expected true, got false")
        }
    }

    /// Soft assert false
    pub fn assert_false(&mut self, value: bool) {
        if value {
            self.add_failure("expected false, got true")
        }
    }

    /// Soft assert approximately equal
    pub fn assert_approx(&mut self, left: f64, right: f64, epsilon: f64) {
        let diff = abs(left - right)
        if diff > epsilon {
            self.add_failure("values not approximately equal: " + to_string(left) + " vs " + to_string(right) + " (diff: " + to_string(diff) + ")")
        }
    }

    /// Soft assert value in range
    pub fn assert_in_range<T: Ord + Debug>(&mut self, value: T, min: T, max: T) {
        if value < min || value > max {
            self.add_failure("value " + debug(value) + " not in range [" + debug(min) + ", " + debug(max) + "]")
        }
    }

    /// Soft assert Some
    pub fn assert_some<T: Debug>(&mut self, opt: &Option<T>) {
        if opt.is_none() {
            self.add_failure("expected Some, got None")
        }
    }

    /// Soft assert None
    pub fn assert_none<T: Debug>(&mut self, opt: &Option<T>) {
        if let Some(v) = opt {
            self.add_failure("expected None, got Some(" + debug(v) + ")")
        }
    }

    /// Soft assert Ok
    pub fn assert_ok<T: Debug, E: Debug>(&mut self, result: &Result<T, E>) {
        if let Err(e) = result {
            self.add_failure("expected Ok, got Err(" + debug(e) + ")")
        }
    }

    /// Soft assert Err
    pub fn assert_err<T: Debug, E: Debug>(&mut self, result: &Result<T, E>) {
        if let Ok(v) = result {
            self.add_failure("expected Err, got Ok(" + debug(v) + ")")
        }
    }

    /// Soft assert collection contains element
    pub fn assert_contains<C: Collection<Item = T>, T: Eq + Debug>(&mut self, collection: &C, element: &T) {
        if !collection.contains(element) {
            self.add_failure("collection does not contain " + debug(element))
        }
    }

    /// Soft assert string contains substring
    pub fn assert_str_contains(&mut self, haystack: &str, needle: &str) {
        if !haystack.contains(needle) {
            self.add_failure("\"" + haystack + "\" does not contain \"" + needle + "\"")
        }
    }

    // Internal helper to add a failure
    fn add_failure(&mut self, message: string) {
        self.failures.push(AssertionFailure {
            message: self.location_prefix.clone() + message,
            location: "", // Would capture stack trace in real implementation
        })
    }
}

// ==================== Fluent Assertion API ====================

/// Fluent assertion builder for more readable assertions
pub struct Asserter<T> {
    value: T,
    description: Option<string>,
}

impl<T> Asserter<T> {
    /// Create a new asserter for a value
    pub fn that(value: T) -> Self {
        Asserter {
            value,
            description: None,
        }
    }

    /// Add a description for better error messages
    pub fn described_as(mut self, description: string) -> Self {
        self.description = Some(description)
        self
    }
}

impl<T: Eq + Debug> Asserter<T> {
    /// Assert equals
    pub fn is_equal_to(self, expected: T) with Panic {
        if self.value != expected {
            let msg = match self.description {
                Some(desc) => desc + ": ",
                None => "",
            }
            panic(msg + "expected " + debug(expected) + " but was " + debug(self.value))
        }
    }

    /// Assert not equals
    pub fn is_not_equal_to(self, unexpected: T) with Panic {
        if self.value == unexpected {
            let msg = match self.description {
                Some(desc) => desc + ": ",
                None => "",
            }
            panic(msg + "expected value different from " + debug(unexpected))
        }
    }
}

impl<T: Ord + Debug> Asserter<T> {
    /// Assert greater than
    pub fn is_greater_than(self, other: T) with Panic {
        if self.value <= other {
            panic("expected " + debug(self.value) + " > " + debug(other))
        }
    }

    /// Assert greater than or equal
    pub fn is_greater_than_or_equal_to(self, other: T) with Panic {
        if self.value < other {
            panic("expected " + debug(self.value) + " >= " + debug(other))
        }
    }

    /// Assert less than
    pub fn is_less_than(self, other: T) with Panic {
        if self.value >= other {
            panic("expected " + debug(self.value) + " < " + debug(other))
        }
    }

    /// Assert less than or equal
    pub fn is_less_than_or_equal_to(self, other: T) with Panic {
        if self.value > other {
            panic("expected " + debug(self.value) + " <= " + debug(other))
        }
    }

    /// Assert in range (inclusive)
    pub fn is_between(self, min: T, max: T) with Panic {
        if self.value < min || self.value > max {
            panic("expected " + debug(self.value) + " to be in [" + debug(min) + ", " + debug(max) + "]")
        }
    }
}

impl Asserter<bool> {
    /// Assert is true
    pub fn is_true(self) with Panic {
        if !self.value {
            let msg = self.description.unwrap_or_default()
            panic(if msg.is_empty() { "expected true" } else { msg + ": expected true" })
        }
    }

    /// Assert is false
    pub fn is_false(self) with Panic {
        if self.value {
            let msg = self.description.unwrap_or_default()
            panic(if msg.is_empty() { "expected false" } else { msg + ": expected false" })
        }
    }
}

impl<T: Debug> Asserter<Option<T>> {
    /// Assert is Some
    pub fn is_some(self) with Panic {
        if self.value.is_none() {
            panic("expected Some, got None")
        }
    }

    /// Assert is None
    pub fn is_none(self) with Panic {
        if let Some(v) = self.value {
            panic("expected None, got Some(" + debug(v) + ")")
        }
    }
}

impl<T: Eq + Debug> Asserter<Option<T>> {
    /// Assert is Some with specific value
    pub fn contains(self, expected: T) with Panic {
        match self.value {
            Some(v) if v == expected => {}
            Some(v) => panic("expected Some(" + debug(expected) + "), got Some(" + debug(v) + ")"),
            None => panic("expected Some(" + debug(expected) + "), got None"),
        }
    }
}

impl<T: Debug, E: Debug> Asserter<Result<T, E>> {
    /// Assert is Ok
    pub fn is_ok(self) with Panic {
        if let Err(e) = self.value {
            panic("expected Ok, got Err(" + debug(e) + ")")
        }
    }

    /// Assert is Err
    pub fn is_err(self) with Panic {
        if let Ok(v) = self.value {
            panic("expected Err, got Ok(" + debug(v) + ")")
        }
    }
}

// ==================== Custom Matchers ====================

/// Trait for custom matchers
pub trait Matcher<T> {
    /// Check if the value matches
    fn matches(&self, value: &T) -> bool

    /// Get description of what this matcher expects
    fn description(&self) -> string

    /// Get description of the mismatch
    fn mismatch_description(&self, value: &T) -> string
}

/// Assert using a custom matcher
pub fn assert_that<T: Debug, M: Matcher<T>>(value: T, matcher: M) with Panic {
    if !matcher.matches(&value) {
        panic("Expected: " + matcher.description() + "\n  but: " + matcher.mismatch_description(&value))
    }
}

// ==================== Built-in Matchers ====================

/// Matcher that checks for equality
pub struct EqualsMatcher<T> {
    expected: T,
}

impl<T> EqualsMatcher<T> {
    pub fn new(expected: T) -> Self {
        EqualsMatcher { expected }
    }
}

impl<T: Eq + Debug> Matcher<T> for EqualsMatcher<T> {
    fn matches(&self, value: &T) -> bool {
        value == &self.expected
    }

    fn description(&self) -> string {
        "value equal to " + debug(self.expected)
    }

    fn mismatch_description(&self, value: &T) -> string {
        "was " + debug(value)
    }
}

/// Matcher that checks if a collection contains an element
pub struct ContainsMatcher<T> {
    element: T,
}

impl<T> ContainsMatcher<T> {
    pub fn new(element: T) -> Self {
        ContainsMatcher { element }
    }
}

impl<C: Collection<Item = T>, T: Eq + Debug> Matcher<C> for ContainsMatcher<T> {
    fn matches(&self, collection: &C) -> bool {
        collection.contains(&self.element)
    }

    fn description(&self) -> string {
        "collection containing " + debug(self.element)
    }

    fn mismatch_description(&self, _collection: &C) -> string {
        "element not found"
    }
}

/// Matcher that combines multiple matchers with AND
pub struct AllOfMatcher<T, M: Matcher<T>> {
    matchers: Vec<M>,
    _phantom: PhantomData<T>,
}

impl<T, M: Matcher<T>> AllOfMatcher<T, M> {
    pub fn new(matchers: Vec<M>) -> Self {
        AllOfMatcher { matchers, _phantom: PhantomData }
    }
}

impl<T: Debug, M: Matcher<T>> Matcher<T> for AllOfMatcher<T, M> {
    fn matches(&self, value: &T) -> bool {
        self.matchers.iter().all(|m| m.matches(value))
    }

    fn description(&self) -> string {
        let descs: Vec<string> = self.matchers.iter().map(|m| m.description()).collect()
        "(" + descs.join(" AND ") + ")"
    }

    fn mismatch_description(&self, value: &T) -> string {
        let failed: Vec<string> = self.matchers
            .iter()
            .filter(|m| !m.matches(value))
            .map(|m| m.mismatch_description(value))
            .collect()
        failed.join("; ")
    }
}

/// Matcher that combines multiple matchers with OR
pub struct AnyOfMatcher<T, M: Matcher<T>> {
    matchers: Vec<M>,
    _phantom: PhantomData<T>,
}

impl<T, M: Matcher<T>> AnyOfMatcher<T, M> {
    pub fn new(matchers: Vec<M>) -> Self {
        AnyOfMatcher { matchers, _phantom: PhantomData }
    }
}

impl<T: Debug, M: Matcher<T>> Matcher<T> for AnyOfMatcher<T, M> {
    fn matches(&self, value: &T) -> bool {
        self.matchers.iter().any(|m| m.matches(value))
    }

    fn description(&self) -> string {
        let descs: Vec<string> = self.matchers.iter().map(|m| m.description()).collect()
        "(" + descs.join(" OR ") + ")"
    }

    fn mismatch_description(&self, value: &T) -> string {
        "none of the matchers matched: " + debug(value)
    }
}

// ==================== Convenience Functions ====================

/// Create an equals matcher
pub fn equals<T>(expected: T) -> EqualsMatcher<T> {
    EqualsMatcher::new(expected)
}

/// Create a contains matcher
pub fn contains<T>(element: T) -> ContainsMatcher<T> {
    ContainsMatcher::new(element)
}

/// Helper functions
fn abs(x: f64) -> f64 {
    if x < 0.0 { -x } else { x }
}

fn debug<T: Debug>(value: T) -> string {
    value.debug_string()
}

fn to_string<T: ToString>(value: T) -> string {
    value.to_string()
}
