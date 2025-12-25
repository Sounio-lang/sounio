//! Property-Based Testing for Demetrios
//!
//! Generates random test inputs to find edge cases.
//!
//! # Basic Usage
//!
//! ```d
//! use test::prop::*
//!
//! #[test]
//! fn test_addition_commutative() {
//!     prop_check(|a: i32, b: i32| {
//!         a + b == b + a
//!     })
//! }
//!
//! #[test]
//! fn test_sort_idempotent() {
//!     prop_check(|xs: Vec<i32>| {
//!         let sorted = xs.clone().sorted()
//!         sorted == sorted.clone().sorted()
//!     })
//! }
//! ```
//!
//! # Shrinking
//!
//! When a failing case is found, the framework automatically "shrinks"
//! the input to find a minimal failing case.

module test::prop

use test::assert::*

// ==================== Configuration ====================

/// Configuration for property-based testing
pub struct PropConfig {
    /// Number of test cases to generate
    pub num_tests: usize,
    /// Maximum shrink iterations
    pub max_shrinks: usize,
    /// Random seed (None = random seed)
    pub seed: Option<u64>,
    /// Verbose output
    pub verbose: bool,
    /// Size parameter for generators (grows with test number)
    pub max_size: usize,
}

impl Default for PropConfig {
    fn default() -> Self {
        PropConfig {
            num_tests: 100,
            max_shrinks: 100,
            seed: None,
            verbose: false,
            max_size: 100,
        }
    }
}

impl PropConfig {
    /// Create config for quick testing
    pub fn quick() -> Self {
        PropConfig {
            num_tests: 20,
            max_shrinks: 50,
            ..Default::default()
        }
    }

    /// Create config for thorough testing
    pub fn thorough() -> Self {
        PropConfig {
            num_tests: 1000,
            max_shrinks: 500,
            max_size: 200,
            ..Default::default()
        }
    }

    /// Set number of tests
    pub fn with_tests(mut self, n: usize) -> Self {
        self.num_tests = n
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed)
        self
    }
}

// ==================== Random Number Generator ====================

/// Simple random number generator (xorshift64)
pub struct Rng {
    state: u64,
}

impl Rng {
    /// Create with a seed
    pub fn new(seed: u64) -> Self {
        Rng { state: if seed == 0 { 1 } else { seed } }
    }

    /// Create with current time as seed
    pub fn from_time() -> Self {
        Rng::new(current_time_nanos() as u64)
    }

    /// Generate next u64
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        self.state = x
        x
    }

    /// Generate u64 in range [0, max)
    pub fn next_u64_max(&mut self, max: u64) -> u64 {
        self.next_u64() % max
    }

    /// Generate i64 in range [min, max]
    pub fn next_i64_range(&mut self, min: i64, max: i64) -> i64 {
        let range = (max - min + 1) as u64
        min + (self.next_u64() % range) as i64
    }

    /// Generate f64 in range [0, 1)
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }

    /// Generate f64 in range [min, max)
    pub fn next_f64_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }

    /// Generate bool with given probability of true
    pub fn next_bool(&mut self, prob: f64) -> bool {
        self.next_f64() < prob
    }

    /// Choose random element from slice
    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> Option<&'a T> {
        if slice.is_empty() {
            None
        } else {
            Some(&slice[self.next_u64_max(slice.len() as u64) as usize])
        }
    }

    /// Shuffle slice in place
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.next_u64_max((i + 1) as u64) as usize
            slice.swap(i, j)
        }
    }
}

// ==================== Arbitrary Trait ====================

/// Trait for types that can be randomly generated
pub trait Arbitrary: Sized {
    /// Generate a random value
    fn arbitrary(rng: &mut Rng, size: usize) -> Self

    /// Shrink to simpler values (for finding minimal failing cases)
    fn shrink(&self) -> Vec<Self> {
        Vec::new()  // Default: no shrinking
    }
}

// ==================== Arbitrary Implementations ====================

impl Arbitrary for bool {
    fn arbitrary(rng: &mut Rng, _size: usize) -> Self {
        rng.next_bool(0.5)
    }

    fn shrink(&self) -> Vec<Self> {
        if *self { vec![false] } else { vec![] }
    }
}

impl Arbitrary for i32 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let max = (size as i64).min(i32::MAX as i64)
        rng.next_i64_range(-max, max) as i32
    }

    fn shrink(&self) -> Vec<Self> {
        let n = *self
        if n == 0 {
            vec![]
        } else if n < 0 {
            vec![0, -n, n + 1]
        } else {
            vec![0, n - 1, n / 2]
        }
    }
}

impl Arbitrary for i64 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let max = (size as i64 * 1000).min(i64::MAX / 2)
        rng.next_i64_range(-max, max)
    }

    fn shrink(&self) -> Vec<Self> {
        let n = *self
        if n == 0 {
            vec![]
        } else if n < 0 {
            vec![0, -n, n + 1]
        } else {
            vec![0, n - 1, n / 2]
        }
    }
}

impl Arbitrary for u32 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let max = (size as u64).min(u32::MAX as u64)
        rng.next_u64_max(max + 1) as u32
    }

    fn shrink(&self) -> Vec<Self> {
        let n = *self
        if n == 0 {
            vec![]
        } else {
            vec![0, n - 1, n / 2]
        }
    }
}

impl Arbitrary for u64 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let max = (size as u64 * 1000).min(u64::MAX / 2)
        rng.next_u64_max(max + 1)
    }

    fn shrink(&self) -> Vec<Self> {
        let n = *self
        if n == 0 {
            vec![]
        } else {
            vec![0, n - 1, n / 2]
        }
    }
}

impl Arbitrary for f64 {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let max = size as f64 * 10.0
        rng.next_f64_range(-max, max)
    }

    fn shrink(&self) -> Vec<Self> {
        let n = *self
        if n == 0.0 {
            vec![]
        } else {
            vec![0.0, n.floor(), n.ceil(), n / 2.0]
        }
    }
}

impl Arbitrary for char {
    fn arbitrary(rng: &mut Rng, _size: usize) -> Self {
        // Generate printable ASCII
        (rng.next_u64_max(95) as u8 + 32) as char
    }

    fn shrink(&self) -> Vec<Self> {
        let c = *self
        if c == 'a' {
            vec![]
        } else if c.is_alphabetic() {
            vec!['a']
        } else {
            vec!['a', ' ']
        }
    }
}

impl Arbitrary for string {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let len = rng.next_u64_max((size + 1) as u64) as usize
        (0..len).map(|_| char::arbitrary(rng, size)).collect()
    }

    fn shrink(&self) -> Vec<Self> {
        let s = self.clone()
        if s.is_empty() {
            vec![]
        } else {
            let mut shrinks = vec!["".to_string()]
            // Remove one character at a time
            for i in 0..s.len() {
                let mut shorter = s.clone()
                shorter.remove(i)
                shrinks.push(shorter)
            }
            // Try half length
            if s.len() > 1 {
                shrinks.push(s[..s.len() / 2].to_string())
            }
            shrinks
        }
    }
}

impl<T: Arbitrary> Arbitrary for Vec<T> {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        let len = rng.next_u64_max((size + 1) as u64) as usize
        (0..len).map(|_| T::arbitrary(rng, size)).collect()
    }

    fn shrink(&self) -> Vec<Self> {
        if self.is_empty() {
            return vec![]
        }

        let mut shrinks = vec![vec![]]  // Empty list is simplest

        // Remove one element at a time
        for i in 0..self.len() {
            let mut shorter = self.clone()
            shorter.remove(i)
            shrinks.push(shorter)
        }

        // Shrink individual elements
        for (i, elem) in self.iter().enumerate() {
            for shrunk_elem in elem.shrink() {
                let mut modified = self.clone()
                modified[i] = shrunk_elem
                shrinks.push(modified)
            }
        }

        // Try half length
        if self.len() > 1 {
            shrinks.push(self[..self.len() / 2].to_vec())
        }

        shrinks
    }
}

impl<T: Arbitrary> Arbitrary for Option<T> {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        if rng.next_bool(0.1) {  // 10% chance of None
            None
        } else {
            Some(T::arbitrary(rng, size))
        }
    }

    fn shrink(&self) -> Vec<Self> {
        match self {
            None => vec![],
            Some(v) => {
                let mut shrinks = vec![None]
                for s in v.shrink() {
                    shrinks.push(Some(s))
                }
                shrinks
            }
        }
    }
}

impl<T: Arbitrary, E: Arbitrary> Arbitrary for Result<T, E> {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        if rng.next_bool(0.2) {  // 20% chance of Err
            Err(E::arbitrary(rng, size))
        } else {
            Ok(T::arbitrary(rng, size))
        }
    }
}

impl<A: Arbitrary, B: Arbitrary> Arbitrary for (A, B) {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        (A::arbitrary(rng, size), B::arbitrary(rng, size))
    }

    fn shrink(&self) -> Vec<Self> {
        let (a, b) = self
        let mut shrinks = vec![]
        for sa in a.shrink() {
            shrinks.push((sa, b.clone()))
        }
        for sb in b.shrink() {
            shrinks.push((a.clone(), sb))
        }
        shrinks
    }
}

impl<A: Arbitrary, B: Arbitrary, C: Arbitrary> Arbitrary for (A, B, C) {
    fn arbitrary(rng: &mut Rng, size: usize) -> Self {
        (A::arbitrary(rng, size), B::arbitrary(rng, size), C::arbitrary(rng, size))
    }
}

// ==================== Generators ====================

/// A generator produces random values
pub struct Gen<T> {
    generate: Fn(&mut Rng, usize) -> T,
}

impl<T> Gen<T> {
    /// Create a generator from a function
    pub fn new<F: Fn(&mut Rng, usize) -> T>(f: F) -> Self {
        Gen { generate: f }
    }

    /// Generate a value
    pub fn generate(&self, rng: &mut Rng, size: usize) -> T {
        (self.generate)(rng, size)
    }

    /// Map over generated values
    pub fn map<U, F: Fn(T) -> U>(self, f: F) -> Gen<U> {
        Gen::new(move |rng, size| f(self.generate(rng, size)))
    }

    /// Filter generated values
    pub fn filter<F: Fn(&T) -> bool>(self, predicate: F) -> Gen<T> {
        Gen::new(move |rng, size| {
            loop {
                let value = self.generate(rng, size)
                if predicate(&value) {
                    return value
                }
            }
        })
    }

    /// Flat map
    pub fn flat_map<U, F: Fn(T) -> Gen<U>>(self, f: F) -> Gen<U> {
        Gen::new(move |rng, size| {
            let value = self.generate(rng, size)
            f(value).generate(rng, size)
        })
    }
}

/// Generate integers in a range
pub fn int_range(min: i64, max: i64) -> Gen<i64> {
    Gen::new(move |rng, _| rng.next_i64_range(min, max))
}

/// Generate floats in a range
pub fn float_range(min: f64, max: f64) -> Gen<f64> {
    Gen::new(move |rng, _| rng.next_f64_range(min, max))
}

/// Generate from a list of choices
pub fn one_of<T: Clone>(choices: Vec<T>) -> Gen<T> {
    Gen::new(move |rng, _| {
        choices[rng.next_u64_max(choices.len() as u64) as usize].clone()
    })
}

/// Generate vectors of specific length
pub fn vec_of<T: Arbitrary>(len: usize) -> Gen<Vec<T>> {
    Gen::new(move |rng, size| {
        (0..len).map(|_| T::arbitrary(rng, size)).collect()
    })
}

/// Generate vectors with length in range
pub fn vec_of_range<T: Arbitrary>(min_len: usize, max_len: usize) -> Gen<Vec<T>> {
    Gen::new(move |rng, size| {
        let len = min_len + rng.next_u64_max((max_len - min_len + 1) as u64) as usize
        (0..len).map(|_| T::arbitrary(rng, size)).collect()
    })
}

/// Generate sorted vectors
pub fn sorted_vec<T: Arbitrary + Ord>() -> Gen<Vec<T>> {
    Gen::new(|rng, size| {
        let mut v = Vec::<T>::arbitrary(rng, size)
        v.sort()
        v
    })
}

/// Generate non-empty strings
pub fn non_empty_string() -> Gen<string> {
    Gen::new(|rng, size| {
        let len = 1 + rng.next_u64_max(size as u64) as usize
        (0..len).map(|_| char::arbitrary(rng, size)).collect()
    })
}

/// Generate alphanumeric strings
pub fn alphanumeric() -> Gen<string> {
    let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars().collect()
    Gen::new(move |rng, size| {
        let len = rng.next_u64_max((size + 1) as u64) as usize
        (0..len).map(|_| chars[rng.next_u64_max(chars.len() as u64) as usize]).collect()
    })
}

// ==================== Property Checking ====================

/// Result of a property check
pub enum PropResult {
    /// Property passed all tests
    Passed { num_tests: usize },
    /// Property failed with a counterexample
    Failed {
        counterexample: string,
        shrunk_counterexample: Option<string>,
        num_tests: usize,
        seed: u64,
    },
    /// Property threw an exception
    Error { message: string, seed: u64 },
}

/// Check a property with default config
pub fn prop_check<F: Fn() -> bool>(property: F) with Panic {
    prop_check_config(property, PropConfig::default())
}

/// Check a property with custom config
pub fn prop_check_config<F: Fn() -> bool>(property: F, config: PropConfig) with Panic {
    let seed = config.seed.unwrap_or_else(|| current_time_nanos() as u64)
    let mut rng = Rng::new(seed)

    for i in 0..config.num_tests {
        let size = (i * config.max_size) / config.num_tests + 1

        // The property function would receive generated arguments
        // This is a simplified version
        if !property() {
            panic("Property failed after " + to_string(i + 1) + " tests (seed: " + to_string(seed) + ")")
        }
    }

    if config.verbose {
        println("OK, passed " + to_string(config.num_tests) + " tests")
    }
}

/// Check a property with one generated argument
pub fn prop_check1<A: Arbitrary + Debug, F: Fn(A) -> bool>(property: F) with Panic {
    prop_check1_config(property, PropConfig::default())
}

pub fn prop_check1_config<A: Arbitrary + Debug, F: Fn(A) -> bool>(property: F, config: PropConfig) with Panic {
    let seed = config.seed.unwrap_or_else(|| current_time_nanos() as u64)
    let mut rng = Rng::new(seed)

    for i in 0..config.num_tests {
        let size = (i * config.max_size) / config.num_tests + 1
        let arg = A::arbitrary(&mut rng, size)

        if !property(arg.clone()) {
            // Try to shrink
            let shrunk = shrink_to_minimal(&arg, &property, config.max_shrinks)
            let shrunk_str = shrunk.map(|s| debug(s))

            panic("Property failed!\n  Counterexample: " + debug(arg) +
                  shrunk_str.map_or("".to_string(), |s| "\n  Shrunk to: " + s) +
                  "\n  (seed: " + to_string(seed) + ", test: " + to_string(i + 1) + ")")
        }
    }
}

/// Check a property with two generated arguments
pub fn prop_check2<A: Arbitrary + Debug, B: Arbitrary + Debug, F: Fn(A, B) -> bool>(property: F) with Panic {
    prop_check2_config(property, PropConfig::default())
}

pub fn prop_check2_config<A: Arbitrary + Debug, B: Arbitrary + Debug, F: Fn(A, B) -> bool>(property: F, config: PropConfig) with Panic {
    let seed = config.seed.unwrap_or_else(|| current_time_nanos() as u64)
    let mut rng = Rng::new(seed)

    for i in 0..config.num_tests {
        let size = (i * config.max_size) / config.num_tests + 1
        let a = A::arbitrary(&mut rng, size)
        let b = B::arbitrary(&mut rng, size)

        if !property(a.clone(), b.clone()) {
            panic("Property failed!\n  Counterexample: (" + debug(a) + ", " + debug(b) + ")" +
                  "\n  (seed: " + to_string(seed) + ", test: " + to_string(i + 1) + ")")
        }
    }
}

/// Shrink a value to find minimal failing case
fn shrink_to_minimal<T: Arbitrary + Clone, F: Fn(T) -> bool>(
    value: &T,
    property: &F,
    max_iterations: usize
) -> Option<T> {
    let mut current = value.clone()
    let mut improved = true
    let mut iterations = 0

    while improved && iterations < max_iterations {
        improved = false
        iterations += 1

        for shrunk in current.shrink() {
            if !property(shrunk.clone()) {
                current = shrunk
                improved = true
                break
            }
        }
    }

    if iterations > 0 {
        Some(current)
    } else {
        None
    }
}

// ==================== Helper Functions ====================

fn debug<T: Debug>(value: T) -> string {
    value.debug_string()
}

fn to_string<T: ToString>(value: T) -> string {
    value.to_string()
}

fn current_time_nanos() -> i64 {
    // Would return actual timestamp
    0
}

fn println(s: string) {
    // Would print to stdout
}
