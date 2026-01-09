---
title: Result<T, E>
description: Error handling with success and failure variants
---

# Result<T, E>

`Result<T, E>` is the type used for returning and propagating errors. It is an enum with variants `Ok(T)` representing success and containing a value, and `Err(E)` representing failure and containing an error value.

## Type Definition

```sio
pub enum Result<T, E> {
    /// Contains the success value
    Ok(T),

    /// Contains the error value
    Err(E),
}
```

## Constructors

### Ok

Creates a `Result` containing a success value.

```sio
pub fn Ok<T, E>(value: T) -> Result<T, E>
```

**Example:**

```sio
let success: Result<i32, String> = Ok(42)
```

### Err

Creates a `Result` containing an error value.

```sio
pub fn Err<T, E>(error: E) -> Result<T, E>
```

**Example:**

```sio
let failure: Result<i32, String> = Err("Something went wrong")
```

## Methods

### is_ok

```sio
pub fn is_ok(self: &Result<T, E>) -> bool
```

Returns `true` if the result is `Ok`.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let y: Result<i32, &str> = Err("error")

x.is_ok()  // true
y.is_ok()  // false
```

### is_err

```sio
pub fn is_err(self: &Result<T, E>) -> bool
```

Returns `true` if the result is `Err`.

**Example:**

```sio
let x: Result<i32, &str> = Err("error")
x.is_err()  // true
```

### is_ok_and

```sio
pub fn is_ok_and<F>(self: Result<T, E>, f: F) -> bool
where F: fn(T) -> bool
```

Returns `true` if the result is `Ok` and the value satisfies the predicate.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let is_positive = x.is_ok_and(|n| n > 0)  // true
```

### is_err_and

```sio
pub fn is_err_and<F>(self: Result<T, E>, f: F) -> bool
where F: fn(E) -> bool
```

Returns `true` if the result is `Err` and the error satisfies the predicate.

**Example:**

```sio
let x: Result<i32, &str> = Err("not found")
let is_not_found = x.is_err_and(|e| e.contains("not found"))  // true
```

### unwrap

```sio
pub fn unwrap(self: Result<T, E>) -> T with Panic
where E: Debug
```

Returns the contained `Ok` value, consuming the `self` value.

**Panics:** If the value is an `Err`, with a panic message including the error.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let value = x.unwrap()  // value = 42

let y: Result<i32, &str> = Err("error")
// y.unwrap() would panic with "called Result::unwrap() on an Err value: error"
```

### unwrap_err

```sio
pub fn unwrap_err(self: Result<T, E>) -> E with Panic
where T: Debug
```

Returns the contained `Err` value, consuming the `self` value.

**Panics:** If the value is an `Ok`.

**Example:**

```sio
let x: Result<i32, &str> = Err("error")
let err = x.unwrap_err()  // err = "error"
```

### unwrap_or

```sio
pub fn unwrap_or(self: Result<T, E>, default: T) -> T
```

Returns the contained `Ok` value or a provided default.

**Parameters:**
- `default` - The value to return if `Err`

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let y: Result<i32, &str> = Err("error")

x.unwrap_or(0)  // 42
y.unwrap_or(0)  // 0
```

### unwrap_or_else

```sio
pub fn unwrap_or_else<F>(self: Result<T, E>, f: F) -> T
where F: fn(E) -> T
```

Returns the contained `Ok` value or computes it from a closure.

**Parameters:**
- `f` - Closure that takes the error and produces a default value

**Example:**

```sio
let x: Result<i32, &str> = Err("error")
let value = x.unwrap_or_else(|e| {
    log_error(e)
    0
})
```

### unwrap_or_default

```sio
pub fn unwrap_or_default(self: Result<T, E>) -> T
where T: Default
```

Returns the contained `Ok` value or the default for the type.

**Example:**

```sio
let x: Result<Vec<i32>, &str> = Err("error")
let value = x.unwrap_or_default()  // empty Vec
```

### expect

```sio
pub fn expect(self: Result<T, E>, msg: &str) -> T with Panic
where E: Debug
```

Returns the contained `Ok` value, panicking with a custom message if `Err`.

**Parameters:**
- `msg` - Custom panic message

**Example:**

```sio
let config = read_config("app.toml").expect("Config file must exist")
```

### expect_err

```sio
pub fn expect_err(self: Result<T, E>, msg: &str) -> E with Panic
where T: Debug
```

Returns the contained `Err` value, panicking with a custom message if `Ok`.

**Example:**

```sio
let err = result.expect_err("Expected an error")
```

### map

```sio
pub fn map<U, F>(self: Result<T, E>, f: F) -> Result<U, E>
where F: fn(T) -> U
```

Maps a `Result<T, E>` to `Result<U, E>` by applying a function to the `Ok` value.

**Parameters:**
- `f` - Function to apply to the success value

**Example:**

```sio
let x: Result<i32, &str> = Ok(5)
let doubled = x.map(|n| n * 2)  // Ok(10)

let y: Result<i32, &str> = Err("error")
let still_err = y.map(|n| n * 2)  // Err("error")
```

### map_err

```sio
pub fn map_err<F, O>(self: Result<T, E>, op: O) -> Result<T, F>
where O: fn(E) -> F
```

Maps a `Result<T, E>` to `Result<T, F>` by applying a function to the error value.

**Parameters:**
- `op` - Function to apply to the error

**Example:**

```sio
let x: Result<i32, i32> = Err(13)
let stringified = x.map_err(|e| e.to_string())  // Err("13")
```

### map_or

```sio
pub fn map_or<U, F>(self: Result<T, E>, default: U, f: F) -> U
where F: fn(T) -> U
```

Applies a function to the contained value (if `Ok`), or returns the provided default (if `Err`).

**Parameters:**
- `default` - Default value to return if `Err`
- `f` - Function to apply to the success value

**Example:**

```sio
let x: Result<String, &str> = Ok("hello")
let len = x.map_or(0, |s| s.len())  // 5

let y: Result<String, &str> = Err("error")
let len = y.map_or(0, |s| s.len())  // 0
```

### map_or_else

```sio
pub fn map_or_else<U, D, F>(self: Result<T, E>, default: D, f: F) -> U
where
    D: fn(E) -> U,
    F: fn(T) -> U
```

Maps a `Result<T, E>` to `U` by applying fallback function to the error, or function `f` to the success value.

**Example:**

```sio
let x: Result<i32, &str> = Err("error")
let value = x.map_or_else(
    |e| e.len() as i32,  // fallback: error length
    |v| v * 2            // success: double the value
)
```

### and_then

```sio
pub fn and_then<U, F>(self: Result<T, E>, op: F) -> Result<U, E>
where F: fn(T) -> Result<U, E>
```

Calls `op` if the result is `Ok`, otherwise returns the `Err` value. Also known as "flatMap".

**Parameters:**
- `op` - Function that returns a `Result`

**Example:**

```sio
fn parse_number(s: &str) -> Result<i32, ParseError> {
    // parsing logic
}

fn square_if_small(n: i32) -> Result<i32, ParseError> {
    if n < 100 {
        Ok(n * n)
    } else {
        Err(ParseError::new("Number too large"))
    }
}

let result = parse_number("10")
    .and_then(|n| square_if_small(n))  // Ok(100)
```

### or_else

```sio
pub fn or_else<F, O>(self: Result<T, E>, op: O) -> Result<T, F>
where O: fn(E) -> Result<T, F>
```

Calls `op` if the result is `Err`, otherwise returns the `Ok` value.

**Example:**

```sio
fn retry_with_backup() -> Result<Data, Error> {
    fetch_primary()
        .or_else(|_| fetch_backup())
}
```

### and

```sio
pub fn and<U>(self: Result<T, E>, res: Result<U, E>) -> Result<U, E>
```

Returns `res` if the result is `Ok`, otherwise returns the `Err` value.

**Example:**

```sio
let x: Result<i32, &str> = Ok(2)
let y: Result<&str, &str> = Err("late error")

x.and(y)  // Err("late error")
```

### or

```sio
pub fn or<F>(self: Result<T, E>, res: Result<T, F>) -> Result<T, F>
```

Returns `res` if the result is `Err`, otherwise returns the `Ok` value.

**Example:**

```sio
let x: Result<i32, &str> = Err("early error")
let y: Result<i32, &str> = Ok(2)

x.or(y)  // Ok(2)
```

### ok

```sio
pub fn ok(self: Result<T, E>) -> Option<T>
```

Converts from `Result<T, E>` to `Option<T>`, discarding the error.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let y: Result<i32, &str> = Err("error")

x.ok()  // Some(42)
y.ok()  // None
```

### err

```sio
pub fn err(self: Result<T, E>) -> Option<E>
```

Converts from `Result<T, E>` to `Option<E>`, discarding the success value.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let y: Result<i32, &str> = Err("error")

x.err()  // None
y.err()  // Some("error")
```

### as_ref

```sio
pub fn as_ref(self: &Result<T, E>) -> Result<&T, &E>
```

Converts from `&Result<T, E>` to `Result<&T, &E>`.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
let ref_result = x.as_ref()  // Ok(&42)
```

### as_mut

```sio
pub fn as_mut(self: &!Result<T, E>) -> Result<&!T, &!E>
```

Converts from `&!Result<T, E>` to `Result<&!T, &!E>`.

**Example:**

```sio
var x: Result<i32, &str> = Ok(42)
if let Ok(v) = x.as_mut() {
    *v = 100
}
```

### transpose

```sio
pub fn transpose(self: Result<Option<T>, E>) -> Option<Result<T, E>>
```

Transposes a `Result` of an `Option` into an `Option` of a `Result`.

**Example:**

```sio
let x: Result<Option<i32>, &str> = Ok(Some(42))
let transposed = x.transpose()  // Some(Ok(42))

let y: Result<Option<i32>, &str> = Ok(None())
let transposed = y.transpose()  // None
```

### flatten

```sio
pub fn flatten(self: Result<Result<T, E>, E>) -> Result<T, E>
```

Flattens `Result<Result<T, E>, E>` to `Result<T, E>`.

**Example:**

```sio
let nested: Result<Result<i32, &str>, &str> = Ok(Ok(42))
let flat = nested.flatten()  // Ok(42)
```

### contains

```sio
pub fn contains<U>(self: &Result<T, E>, x: &U) -> bool
where T: PartialEq<U>
```

Returns `true` if the result is an `Ok` value containing the given value.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
x.contains(&42)  // true
x.contains(&10)  // false
```

### contains_err

```sio
pub fn contains_err<F>(self: &Result<T, E>, f: &F) -> bool
where E: PartialEq<F>
```

Returns `true` if the result is an `Err` value containing the given error.

**Example:**

```sio
let x: Result<i32, &str> = Err("error")
x.contains_err(&"error")  // true
```

### copied

```sio
pub fn copied(self: Result<&T, E>) -> Result<T, E>
where T: Copy
```

Copies the `Ok` value if `T` is `Copy`.

**Example:**

```sio
let value = 42
let result: Result<&i32, &str> = Ok(&value)
let copied = result.copied()  // Ok(42)
```

### cloned

```sio
pub fn cloned(self: Result<&T, E>) -> Result<T, E>
where T: Clone
```

Clones the `Ok` value if `T` is `Clone`.

**Example:**

```sio
let s = "hello".to_string()
let result: Result<&String, &str> = Ok(&s)
let cloned = result.cloned()  // Ok("hello")
```

### iter

```sio
pub fn iter(self: &Result<T, E>) -> ResultIter<T>
```

Returns an iterator over the possibly contained value.

**Example:**

```sio
let x: Result<i32, &str> = Ok(42)
for value in x.iter() {
    println(value.to_string())
}
```

## Pattern Matching

The idiomatic way to handle `Result` values is with pattern matching:

```sio
fn process(result: Result<i32, &str>) {
    match result {
        Ok(value) => {
            println("Success: " ++ value.to_string())
        },
        Err(error) => {
            println("Error: " ++ error)
        },
    }
}
```

## Error Propagation

Sounio supports error propagation through the effect system. Functions that can fail should declare their error effects:

```sio
fn read_and_parse(path: &str) -> Result<Config, IoError> with IO {
    let content = read_file(path)?  // Propagates error if Err
    parse_config(content)
}
```

## Common Patterns

### Chaining Fallible Operations

```sio
fn process_user_data(user_id: i32) -> Result<Report, Error> {
    get_user(user_id)
        .and_then(|user| fetch_orders(user.id))
        .and_then(|orders| generate_report(orders))
        .map_err(|e| Error::Processing(e))
}
```

### Converting Option to Result

```sio
fn find_or_error(id: i32) -> Result<User, &str> {
    find_user(id).ok_or("User not found")
}
```

### Collecting Results

```sio
fn parse_all(strings: Vec<&str>) -> Result<Vec<i32>, ParseError> {
    strings
        .iter()
        .map(|s| s.parse::<i32>())
        .collect()
}
```

### Handling Multiple Error Types

```sio
enum AppError {
    Io(IoError),
    Parse(ParseError),
    Network(NetworkError),
}

fn load_config() -> Result<Config, AppError> with IO {
    let content = read_file("config.toml")
        .map_err(|e| AppError::Io(e))?

    parse_toml(content)
        .map_err(|e| AppError::Parse(e))
}
```

## Trait Implementations

`Result<T, E>` implements the following traits when `T` and `E` meet the requirements:

- `Clone` when `T: Clone` and `E: Clone`
- `Eq` when `T: Eq` and `E: Eq`
- `Ord` when `T: Ord` and `E: Ord`
- `Debug` when `T: Debug` and `E: Debug`
- `Hash` when `T: Hash` and `E: Hash`
- `IntoIterator`
- `FromIterator<Result<A, E>>` for `Result<Vec<A>, E>`

## See Also

- [Option<T>](option.md) - For optional values without error information
- [I/O Module](../io/files.md) - Uses `Result` for file operations
