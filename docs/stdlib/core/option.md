---
title: Option<T>
description: Optional values for representing presence or absence
---

# Option<T>

`Option<T>` represents an optional value: every `Option` is either `Some(T)` containing a value, or `None` representing absence. This type is fundamental for handling nullable values safely without null pointer exceptions.

## Type Definition

```sio
pub enum Option<T> {
    /// No value
    None,

    /// Some value of type T
    Some(T),
}
```

## Constructors

### Some

Creates an `Option` containing a value.

```sio
pub fn Some<T>(value: T) -> Option<T>
```

**Example:**

```sio
let x = Some(42)
let name = Some("Alice")
```

### None

Returns the `None` value representing absence.

```sio
pub fn None<T>() -> Option<T>
```

**Example:**

```sio
let empty: Option<i32> = None()
```

## Methods

### is_some

```sio
pub fn is_some(self: &Option<T>) -> bool
```

Returns `true` if the option is a `Some` value.

**Example:**

```sio
let x = Some(42)
let y: Option<i32> = None()

if x.is_some() {
    println("x has a value")
}
// y.is_some() returns false
```

### is_none

```sio
pub fn is_none(self: &Option<T>) -> bool
```

Returns `true` if the option is a `None` value.

**Example:**

```sio
let x: Option<i32> = None()
if x.is_none() {
    println("x is empty")
}
```

### unwrap

```sio
pub fn unwrap(self: Option<T>) -> T with Panic
```

Returns the contained `Some` value, consuming the `self` value.

**Panics:** If the value is `None`.

**Example:**

```sio
let x = Some(42)
let value = x.unwrap()  // value = 42

let y: Option<i32> = None()
// y.unwrap() would panic!
```

### unwrap_or

```sio
pub fn unwrap_or(self: Option<T>, default: T) -> T
```

Returns the contained `Some` value or a provided default.

**Parameters:**
- `default` - The value to return if `None`

**Example:**

```sio
let x = Some(42)
let y: Option<i32> = None()

let a = x.unwrap_or(0)  // a = 42
let b = y.unwrap_or(0)  // b = 0
```

### unwrap_or_else

```sio
pub fn unwrap_or_else<F>(self: Option<T>, f: F) -> T
where F: fn() -> T
```

Returns the contained `Some` value or computes it from a closure.

**Parameters:**
- `f` - Closure that produces the default value

**Example:**

```sio
let x: Option<i32> = None()
let value = x.unwrap_or_else(|| expensive_computation())
```

### unwrap_or_default

```sio
pub fn unwrap_or_default(self: Option<T>) -> T
where T: Default
```

Returns the contained `Some` value or the default for the type.

**Example:**

```sio
let x: Option<i32> = None()
let value = x.unwrap_or_default()  // value = 0 (default for i32)
```

### expect

```sio
pub fn expect(self: Option<T>, msg: &str) -> T with Panic
```

Returns the contained `Some` value, panicking with a custom message if `None`.

**Parameters:**
- `msg` - Custom panic message

**Example:**

```sio
let config = get_config().expect("Config file must exist")
```

### map

```sio
pub fn map<U, F>(self: Option<T>, f: F) -> Option<U>
where F: fn(T) -> U
```

Maps an `Option<T>` to `Option<U>` by applying a function to the contained value.

**Parameters:**
- `f` - Function to apply to the value

**Returns:** `Some(f(value))` if `Some`, `None` if `None`

**Example:**

```sio
let x = Some(5)
let doubled = x.map(|n| n * 2)  // Some(10)

let y: Option<i32> = None()
let still_none = y.map(|n| n * 2)  // None
```

### map_or

```sio
pub fn map_or<U, F>(self: Option<T>, default: U, f: F) -> U
where F: fn(T) -> U
```

Applies a function to the contained value (if `Some`), or returns the provided default (if `None`).

**Parameters:**
- `default` - Default value to return if `None`
- `f` - Function to apply to the value

**Example:**

```sio
let x = Some("hello")
let len = x.map_or(0, |s| s.len())  // len = 5

let y: Option<String> = None()
let len = y.map_or(0, |s| s.len())  // len = 0
```

### and_then

```sio
pub fn and_then<U, F>(self: Option<T>, f: F) -> Option<U>
where F: fn(T) -> Option<U>
```

Returns `None` if the option is `None`, otherwise calls `f` with the wrapped value and returns the result. Also known as "flatMap" in other languages.

**Parameters:**
- `f` - Function that returns an `Option`

**Example:**

```sio
fn parse_port(s: &str) -> Option<u16> {
    // Returns Some(port) if valid, None otherwise
}

let port_str = Some("8080")
let port = port_str.and_then(|s| parse_port(s))  // Some(8080)
```

### filter

```sio
pub fn filter<P>(self: Option<T>, predicate: P) -> Option<T>
where P: fn(&T) -> bool
```

Returns `None` if the option is `None`, otherwise calls predicate with the wrapped value and returns `Some(t)` if predicate returns `true`.

**Parameters:**
- `predicate` - Function to test the value

**Example:**

```sio
let x = Some(42)
let even = x.filter(|n| n % 2 == 0)  // Some(42)
let odd = x.filter(|n| n % 2 == 1)   // None
```

### or

```sio
pub fn or(self: Option<T>, optb: Option<T>) -> Option<T>
```

Returns the option if it contains a value, otherwise returns `optb`.

**Example:**

```sio
let x = Some(2)
let y: Option<i32> = None()

let a = x.or(y)        // Some(2)
let b = y.or(Some(5))  // Some(5)
```

### or_else

```sio
pub fn or_else<F>(self: Option<T>, f: F) -> Option<T>
where F: fn() -> Option<T>
```

Returns the option if it contains a value, otherwise calls `f` and returns the result.

**Example:**

```sio
let x: Option<i32> = None()
let value = x.or_else(|| fetch_default())
```

### xor

```sio
pub fn xor(self: Option<T>, optb: Option<T>) -> Option<T>
```

Returns `Some` if exactly one of `self`, `optb` is `Some`, otherwise returns `None`.

**Example:**

```sio
let x = Some(2)
let y: Option<i32> = None()

let a = x.xor(None())   // Some(2)
let b = x.xor(Some(3))  // None (both are Some)
let c = y.xor(None())   // None (both are None)
```

### take

```sio
pub fn take(self: &!Option<T>) -> Option<T>
```

Takes the value out of the option, leaving a `None` in its place.

**Example:**

```sio
var x = Some(42)
let taken = x.take()  // taken = Some(42), x = None
```

### replace

```sio
pub fn replace(self: &!Option<T>, value: T) -> Option<T>
```

Replaces the actual value in the option with the value given, returning the old value if present.

**Example:**

```sio
var x = Some(2)
let old = x.replace(5)  // old = Some(2), x = Some(5)

var y: Option<i32> = None()
let old = y.replace(3)  // old = None, y = Some(3)
```

### as_ref

```sio
pub fn as_ref(self: &Option<T>) -> Option<&T>
```

Converts from `&Option<T>` to `Option<&T>`.

**Example:**

```sio
let text = Some("Hello")
let length = text.as_ref().map(|s| s.len())  // Some(5)
```

### as_mut

```sio
pub fn as_mut(self: &!Option<T>) -> Option<&!T>
```

Converts from `&!Option<T>` to `Option<&!T>`.

**Example:**

```sio
var x = Some(vec![1, 2, 3])
if let Some(v) = x.as_mut() {
    v.push(4)
}
```

### ok_or

```sio
pub fn ok_or<E>(self: Option<T>, err: E) -> Result<T, E>
```

Transforms the `Option<T>` into a `Result<T, E>`, mapping `Some(v)` to `Ok(v)` and `None` to `Err(err)`.

**Example:**

```sio
let x = Some(42)
let result = x.ok_or("No value")  // Ok(42)

let y: Option<i32> = None()
let result = y.ok_or("No value")  // Err("No value")
```

### zip

```sio
pub fn zip<U>(self: Option<T>, other: Option<U>) -> Option<(T, U)>
```

Zips `self` with another `Option`.

**Returns:** `Some((a, b))` if both are `Some`, otherwise `None`

**Example:**

```sio
let x = Some(1)
let y = Some("hi")
let z: Option<i32> = None()

let a = x.zip(y)  // Some((1, "hi"))
let b = x.zip(z)  // None
```

### flatten

```sio
pub fn flatten(self: Option<Option<T>>) -> Option<T>
```

Flattens `Option<Option<T>>` to `Option<T>`.

**Example:**

```sio
let nested = Some(Some(42))
let flat = nested.flatten()  // Some(42)

let outer = Some(None())
let flat = outer.flatten()   // None
```

### contains

```sio
pub fn contains<U>(self: &Option<T>, x: &U) -> bool
where T: PartialEq<U>
```

Returns `true` if the option is a `Some` value containing the given value.

**Example:**

```sio
let x = Some(42)
let has_42 = x.contains(&42)  // true
let has_10 = x.contains(&10)  // false
```

### iter

```sio
pub fn iter(self: &Option<T>) -> OptionIter<T>
```

Returns an iterator over the possibly contained value.

**Example:**

```sio
let x = Some(42)
for value in x.iter() {
    println(value.to_string())
}
```

## Pattern Matching

The idiomatic way to handle `Option` values is with pattern matching:

```sio
fn process(opt: Option<i32>) {
    match opt {
        Some(value) => {
            println("Got value: " ++ value.to_string())
        },
        None => {
            println("No value")
        },
    }
}
```

## Common Patterns

### Safe Division

```sio
fn safe_divide(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        return None()
    }
    Some(a / b)
}

let result = safe_divide(10.0, 2.0)  // Some(5.0)
let error = safe_divide(10.0, 0.0)   // None
```

### Chaining Operations

```sio
fn get_user_email(user_id: i32) -> Option<String> {
    get_user(user_id)
        .and_then(|user| user.email)
        .filter(|email| email.contains("@"))
}
```

### Providing Defaults

```sio
let config = read_config("app.toml")
    .unwrap_or_else(|| Config::default())

let name = user.nickname
    .or(user.username)
    .unwrap_or("Anonymous")
```

## Trait Implementations

`Option<T>` implements the following traits when `T` meets the requirements:

- `Clone` when `T: Clone`
- `Default` (returns `None`)
- `Eq` when `T: Eq`
- `Ord` when `T: Ord`
- `Debug` when `T: Debug`
- `Hash` when `T: Hash`
- `IntoIterator`

## See Also

- [Result<T, E>](result.md) - For operations that can fail with an error
- [Iterator](../iter.md) - For sequence processing
