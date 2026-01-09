# Migrating from Rust to Sounio

This guide helps Rust developers understand Sounio's key differences and adopt idiomatic Sounio patterns.

## Quick Reference

| Rust | Sounio | Notes |
|------|--------|-------|
| `&mut T` | `&!T` | Exclusive/mutable reference |
| `let mut x` | `var x` | Mutable binding |
| `#[derive(...)]` | Not supported | No proc macros |
| `assert!()` | `assert()` | No macros, use functions |
| `println!()` | `println()` | No macros |
| `vec![]` | `vec![]` | Same syntax |
| `#[test]` | `//@ run-pass` | Test annotations in comments |
| `unsafe { }` | `trust { }` | Different keyword |
| `impl Trait for Type` | `impl Type: Trait` | Different syntax |
| `dyn Trait` | `dyn Trait` | Same |
| `Box<T>` | `Box<T>` | Same |
| `Option<T>` | `Option<T>` | Same |
| `Result<T, E>` | `Result<T, E>` | Same |

## Mutable References: `&!` not `&mut`

The most visible difference is Sounio's syntax for mutable references.

### Rust

```rust
fn increment(x: &mut i32) {
    *x += 1;
}

fn main() {
    let mut value = 5;
    increment(&mut value);
    println!("{}", value);  // 6
}
```

### Sounio

```sio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn main() -> i32 {
    var value = 5
    increment(&!value)
    println(value)  // 6
    return 0
}
```

### Why `&!`?

The `&!` syntax emphasizes that the reference is *exclusive*: no other reference to this data can exist while `&!` is held. The exclamation mark suggests "exclusive access" rather than the mutation-focused `mut`.

## Mutable Bindings: `var` not `let mut`

### Rust

```rust
let mut counter = 0;
counter += 1;

let immutable = 42;  // Cannot be reassigned
```

### Sounio

```sio
var counter = 0
counter = counter + 1

let immutable = 42  // Cannot be reassigned
```

### Additional: `const` for Compile-Time Constants

```sio
const PI = 3.14159265358979    // Compile-time constant
let tau = PI * 2.0             // Runtime immutable
var radius = 1.0               // Runtime mutable
```

## No Macros

Sounio does not have procedural macros. All Rust macros must be replaced with function calls.

### Rust

```rust
println!("Hello, {}!", name);
assert!(x > 0);
vec![1, 2, 3]
format!("{}: {}", key, value)
```

### Sounio

```sio
println("Hello, " ++ name ++ "!")
assert(x > 0)
vec![1, 2, 3]  // vec! is a special form, not a macro
format("{}: {}", key, value)  // Function, not macro
```

### No Attribute Macros

| Rust | Sounio Equivalent |
|------|-------------------|
| `#[derive(Debug)]` | Implement `Debug` trait manually |
| `#[derive(Clone)]` | Implement `Clone` trait manually |
| `#[test]` | Use `//@ run-pass` annotation |
| `#[cfg(test)]` | Use separate test files |
| `#[inline]` | Compiler handles inlining |

## Effect System Instead of Traits for Side Effects

Sounio uses an algebraic effect system rather than trait bounds for side effects.

### Rust

```rust
// Side effects implicit
fn read_config(path: &Path) -> std::io::Result<Config> {
    let contents = std::fs::read_to_string(path)?;
    // ...
}

// Or with trait bounds
fn process<R: Read>(reader: R) -> Result<Data, Error> {
    // ...
}
```

### Sounio

```sio
// Effects declared in signature
fn read_config(path: string) -> Result<Config, string> with IO {
    let contents = read_file(path)
    // ...
}

// Multiple effects
fn simulate() -> f64 with IO, Prob, Alloc {
    // Can do I/O, use randomness, and allocate memory
}

// Pure function (no effects)
fn calculate(x: f64) -> f64 {
    return x * x + 1.0
}
```

### Common Effects

| Effect | Description |
|--------|-------------|
| `IO` | File I/O, network, console |
| `Mut` | Mutating state |
| `Alloc` | Heap allocation |
| `Panic` | May panic |
| `Async` | Asynchronous operations |
| `Prob` | Probabilistic/random operations |
| `GPU` | GPU kernel execution |
| `Div` | May diverge (infinite loops) |

## Units of Measure

Sounio has built-in support for units of measure, which Rust requires external crates for.

### Rust (with uom crate)

```rust
use uom::si::f64::*;
use uom::si::length::meter;
use uom::si::time::second;

let distance = Length::new::<meter>(100.0);
let time = Time::new::<second>(10.0);
let speed = distance / time;  // Velocity
```

### Sounio

```sio
let distance: m = 100.0
let time: s = 10.0
let speed: m/s = distance / time  // Type-checked!

// Medical units
let dose: mg = 500.0
let volume: mL = 10.0
let concentration: mg/mL = dose / volume

// Compile error: unit mismatch
// let wrong: kg = dose  // Error: expected kg, got mg
```

## Epistemic Types (Sounio-Specific)

Sounio has first-class support for values with uncertainty, which has no direct Rust equivalent.

### Rust (manual)

```rust
struct Measurement {
    value: f64,
    uncertainty: f64,
}

impl std::ops::Add for Measurement {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Measurement {
            value: self.value + other.value,
            uncertainty: (self.uncertainty.powi(2)
                        + other.uncertainty.powi(2)).sqrt(),
        }
    }
}
```

### Sounio

```sio
use epistemic::core::*

// Create epistemic values
let mass = epistemic_std(75.0, 0.5, 0.95)  // value, uncertainty, confidence

// Uncertainty propagates automatically through operations
let volume = epistemic_std(0.25, 0.01, 0.90)
let density = div_epistemic(mass, volume)

// Access components
let value = density.value
let uncertainty = get_std_uncertainty(density)
let confidence = density.conf
```

## Pattern Matching

Pattern matching is similar, but with some syntax differences.

### Rust

```rust
match value {
    Some(x) if x > 0 => println!("Positive: {}", x),
    Some(x) => println!("Non-positive: {}", x),
    None => println!("None"),
}

let (a, b) = tuple;  // Destructuring
```

### Sounio

```sio
match value {
    Some(x) if x > 0 => println("Positive: " ++ x.to_string()),
    Some(x) => println("Non-positive: " ++ x.to_string()),
    None => println("None"),
}

// Note: tuple destructuring is limited in Sounio
let tuple = (1, 2)
let a = tuple.0
let b = tuple.1
```

## Error Handling

Error handling is similar but with effect annotations.

### Rust

```rust
fn divide(a: f64, b: f64) -> Result<f64, String> {
    if b == 0.0 {
        Err("Division by zero".to_string())
    } else {
        Ok(a / b)
    }
}

fn main() -> Result<(), String> {
    let result = divide(10.0, 2.0)?;
    println!("{}", result);
    Ok(())
}
```

### Sounio

```sio
fn divide(a: f64, b: f64) -> Result<f64, string> {
    if b == 0.0 {
        return Err("Division by zero")
    }
    return Ok(a / b)
}

fn main() -> i32 with Panic {
    match divide(10.0, 2.0) {
        Ok(result) => println(result),
        Err(msg) => panic(msg),
    }
    return 0
}
```

## Loops

Loop syntax is similar with minor differences.

### Rust

```rust
for i in 0..10 {
    println!("{}", i);
}

for item in items.iter() {
    process(item);
}

let mut i = 0;
while i < 10 {
    i += 1;
}

loop {
    if condition { break; }
}
```

### Sounio

```sio
for i in 0..10 {
    println(i)
}

for item in items {
    process(item)
}

var i = 0
while i < 10 {
    i = i + 1
}

loop {
    if condition { break }
}
```

## Structs and Impl

### Rust

```rust
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Point { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

impl Default for Point {
    fn default() -> Self {
        Point { x: 0.0, y: 0.0 }
    }
}
```

### Sounio

```sio
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Point {
        return Point { x: x, y: y }
    }

    fn distance(self: &Point, other: &Point) -> f64 {
        let dx = self.x - other.x
        let dy = self.y - other.y
        return sqrt_f64(dx * dx + dy * dy)
    }
}

impl Point: Default {
    fn default() -> Point {
        return Point { x: 0.0, y: 0.0 }
    }
}
```

## Linear Types

Sounio has first-class linear and affine types for resource management.

### Rust

```rust
// Ownership is always affine (use at most once)
let file = File::open("data.txt")?;
// file is moved/consumed when used
```

### Sounio

```sio
// Affine type: can be used at most once
affine struct Buffer {
    ptr: *u8,
    len: usize,
}

// Linear type: must be used exactly once
linear struct FileHandle {
    fd: i32,
}

// Compiler ensures FileHandle is closed
fn process_file() with IO {
    let handle = open_file("data.txt")
    // ...
    close_file(handle)  // Required: linear type must be consumed
}
```

## Testing

### Rust

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        assert_eq!(add(2, 2), 4);
    }

    #[test]
    #[should_panic]
    fn test_panic() {
        divide(1.0, 0.0).unwrap();
    }
}
```

### Sounio

Test files use annotation comments:

```sio
//@ run-pass

fn test_addition() -> bool {
    return add(2, 2) == 4
}

fn main() -> i32 {
    if !test_addition() { return 1 }
    return 0
}
```

For expected failures:

```sio
//@ compile-fail
//@ error-pattern: type mismatch

fn main() -> i32 {
    let x: i32 = "not an integer"  // Error expected
    return 0
}
```

## Common Patterns Migration

### Builder Pattern

```rust
// Rust
Config::builder()
    .name("test")
    .count(5)
    .build()
```

```sio
// Sounio - use struct literal instead
Config {
    name: "test",
    count: 5,
    ..Config::default()
}
```

### Iterator Chains

```rust
// Rust
items.iter()
    .filter(|x| x > &0)
    .map(|x| x * 2)
    .collect()
```

```sio
// Sounio
var result: Vec<i32> = vec![]
for x in items {
    if x > 0 {
        result.push(x * 2)
    }
}
result
```

### RAII / Drop

```rust
// Rust - Drop trait
impl Drop for Resource {
    fn drop(&mut self) {
        cleanup();
    }
}
```

```sio
// Sounio - use linear types + explicit cleanup
linear struct Resource { ... }

fn use_resource() {
    let r = acquire_resource()
    // ... use r ...
    release_resource(r)  // Must be called (linear type)
}
```

## Key Mindset Shifts

1. **Effects are explicit**: Functions declare what side effects they may have
2. **Uncertainty is first-class**: Use `Knowledge<T>` for uncertain values
3. **Units are enforced**: Physical quantities have compile-time unit checking
4. **Linear types for resources**: Explicit resource management via type system
5. **No macros**: Use functions and special forms instead
6. **Scientific computing focus**: Built-in support for scientific domains

## Further Reading

- [Sounio Language Guide](../language/index.md)
- [Epistemic Types](../epistemic/core.md)
- [Effect System](../language/effects.md)
- [Units of Measure](../language/units.md)
