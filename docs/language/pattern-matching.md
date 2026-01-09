---
title: Pattern Matching
description: Comprehensive guide to pattern matching in Sounio
prerequisites:
  - /docs/getting-started.md
  - /docs/language/types.md
reading_time: 10 minutes
---

# Pattern Matching

Pattern matching is a powerful control flow construct in Sounio that allows you to match values against patterns and destructure data. It provides a concise and exhaustive way to handle different cases.

## Match Expressions

The `match` expression compares a value against a series of patterns:

```sio
fn describe_number(n: i32) -> string {
    match n {
        0 => "zero",
        1 => "one",
        2 => "two",
        _ => "many",
    }
}
```

### Basic Syntax

```sio
match scrutinee {
    pattern1 => expression1,
    pattern2 => expression2,
    pattern3 => {
        // Block expression for multiple statements
        let x = compute()
        x + 1
    },
    _ => default_expression,
}
```

The match expression:
1. Evaluates the scrutinee (the value being matched)
2. Tests each pattern in order from top to bottom
3. Executes the expression for the first matching pattern
4. Returns the result of that expression

## Pattern Types

### Literal Patterns

Match against specific constant values:

```sio
fn day_name(day: i32) -> string {
    match day {
        1 => "Monday",
        2 => "Tuesday",
        3 => "Wednesday",
        4 => "Thursday",
        5 => "Friday",
        6 => "Saturday",
        7 => "Sunday",
        _ => "Invalid day",
    }
}

fn check_char(c: char) -> string {
    match c {
        'a' => "lowercase a",
        'A' => "uppercase A",
        '0' => "digit zero",
        _ => "other character",
    }
}

fn check_bool(b: bool) -> string {
    match b {
        true => "it's true",
        false => "it's false",
    }
}
```

### Variable Binding

Bind the matched value to a new variable:

```sio
fn process(opt: Option<i32>) -> i32 {
    match opt {
        Some(value) => value * 2,  // value is bound to the inner i32
        None => 0,
    }
}

fn describe(result: Result<string, Error>) -> string {
    match result {
        Ok(message) => message,        // message is bound to the string
        Err(error) => error.to_string(),  // error is bound to the Error
    }
}
```

### Wildcard Pattern

The underscore `_` matches anything and discards the value:

```sio
fn is_zero(n: i32) -> bool {
    match n {
        0 => true,
        _ => false,  // Matches any other value
    }
}

fn get_first(pair: (i32, i32)) -> i32 {
    match pair {
        (first, _) => first,  // Ignore second element
    }
}
```

### Struct Patterns

Match and destructure struct fields:

```sio
struct Point {
    x: f64,
    y: f64,
}

fn describe_point(p: Point) -> string {
    match p {
        Point { x: 0.0, y: 0.0 } => "origin",
        Point { x: 0.0, y } => format("on y-axis at {}", y),
        Point { x, y: 0.0 } => format("on x-axis at {}", x),
        Point { x, y } => format("at ({}, {})", x, y),
    }
}

// Shorthand when variable name matches field name
fn point_info(p: Point) -> string {
    match p {
        Point { x, y } => format("x={}, y={}", x, y),
    }
}
```

### Enum Variant Patterns

Match against enum variants:

```sio
enum Color {
    Red,
    Green,
    Blue,
    Rgb(u8, u8, u8),
    Named(string),
}

fn color_to_hex(color: Color) -> string {
    match color {
        Color::Red => "#FF0000",
        Color::Green => "#00FF00",
        Color::Blue => "#0000FF",
        Color::Rgb(r, g, b) => format("#{:02X}{:02X}{:02X}", r, g, b),
        Color::Named(name) => lookup_color(name),
    }
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(string),
    ChangeColor(Color),
}

fn handle_message(msg: Message) {
    match msg {
        Message::Quit => quit_application(),
        Message::Move { x, y } => move_cursor(x, y),
        Message::Write(text) => print(text),
        Message::ChangeColor(color) => set_color(color),
    }
}
```

### Or Patterns

Match multiple patterns with `|`:

```sio
fn is_vowel(c: char) -> bool {
    match c {
        'a' | 'e' | 'i' | 'o' | 'u' => true,
        'A' | 'E' | 'I' | 'O' | 'U' => true,
        _ => false,
    }
}

fn categorize(n: i32) -> string {
    match n {
        0 => "zero",
        1 | 2 | 3 => "small",
        4 | 5 | 6 => "medium",
        7 | 8 | 9 => "large",
        _ => "out of range",
    }
}
```

## Guards

Pattern guards add additional conditions using `if`:

```sio
fn describe_number(n: i32) -> string {
    match n {
        0 => "zero",
        n if n > 0 => "positive",
        n if n < 0 => "negative",
        _ => "unreachable",  // Needed for exhaustiveness, but never reached
    }
}

fn classify_score(score: i32) -> string {
    match score {
        s if s >= 90 => "A",
        s if s >= 80 => "B",
        s if s >= 70 => "C",
        s if s >= 60 => "D",
        _ => "F",
    }
}

fn process_option(opt: Option<i32>) -> string {
    match opt {
        Some(n) if n > 100 => "large value",
        Some(n) if n > 0 => "positive value",
        Some(n) if n < 0 => "negative value",
        Some(0) => "zero",
        None => "no value",
    }
}
```

### Complex Guards

Guards can use any boolean expression:

```sio
fn check_range(x: i32, min: i32, max: i32) -> string {
    match x {
        n if n < min => "below range",
        n if n > max => "above range",
        _ => "in range",
    }
}

fn validate_point(p: Point, bounds: Rect) -> bool {
    match p {
        Point { x, y } if x >= bounds.left && x <= bounds.right
                      && y >= bounds.top && y <= bounds.bottom => true,
        _ => false,
    }
}
```

## Exhaustiveness Checking

The compiler ensures all possible values are covered:

```sio
enum Direction {
    North,
    South,
    East,
    West,
}

fn to_degrees(dir: Direction) -> i32 {
    match dir {
        Direction::North => 0,
        Direction::East => 90,
        Direction::South => 180,
        // ERROR: non-exhaustive patterns: `Direction::West` not covered
    }
}

// Correct: all variants covered
fn to_degrees_complete(dir: Direction) -> i32 {
    match dir {
        Direction::North => 0,
        Direction::East => 90,
        Direction::South => 180,
        Direction::West => 270,
    }
}

// Also correct: use wildcard for "everything else"
fn is_horizontal(dir: Direction) -> bool {
    match dir {
        Direction::East | Direction::West => true,
        _ => false,
    }
}
```

### Boolean Exhaustiveness

```sio
// Must cover both true and false
fn not(b: bool) -> bool {
    match b {
        true => false,
        false => true,
    }
}
```

### Option and Result Exhaustiveness

```sio
fn unwrap_or_default(opt: Option<i32>) -> i32 {
    match opt {
        Some(value) => value,
        None => 0,
    }
}

fn handle_result(result: Result<Data, Error>) {
    match result {
        Ok(data) => process(data),
        Err(e) => log_error(e),
    }
}
```

## Nested Patterns

Patterns can be nested to match complex structures:

```sio
fn describe_nested(opt: Option<Option<i32>>) -> string {
    match opt {
        Some(Some(n)) => format("nested value: {}", n),
        Some(None) => "outer Some, inner None",
        None => "outer None",
    }
}

struct Person {
    name: string,
    address: Option<Address>,
}

struct Address {
    city: string,
    country: string,
}

fn get_country(person: Person) -> Option<string> {
    match person {
        Person { address: Some(Address { country, .. }), .. } => Some(country),
        _ => None,
    }
}

enum Tree<T> {
    Leaf(T),
    Node(Box<Tree<T>>, Box<Tree<T>>),
}

fn count_leaves<T>(tree: Tree<T>) -> i32 {
    match tree {
        Tree::Leaf(_) => 1,
        Tree::Node(left, right) => count_leaves(*left) + count_leaves(*right),
    }
}
```

## Match in Function Arguments

Pattern matching can be used directly in function parameters (in some contexts):

```sio
// Destructuring in function body is preferred
fn process_point(p: Point) -> f64 {
    match p {
        Point { x, y } => x * x + y * y,
    }
}

// Alternative: use field access
fn process_point_alt(p: Point) -> f64 {
    return p.x * p.x + p.y * p.y
}
```

## What Does NOT Work

Sounio has specific limitations on pattern matching that differ from some other languages.

### No Tuple Destructuring in Let

```sio
// WRONG - does not work in Sounio
// let (a, b) = get_pair()

// CORRECT - use field access
let pair = get_pair()
let a = pair.0
let b = pair.1

// OR use a struct with named fields
struct Pair<T, U> {
    first: T,
    second: U,
}

let pair = get_pair()
let a = pair.first
let b = pair.second
```

### No Destructuring in Closure Parameters

```sio
// WRONG - tuple destructuring in closure
// let sum = pairs.map(|(x, y)| x + y)

// CORRECT - use explicit indexing
let sum = pairs.map(|pair| pair.0 + pair.1)

// OR use a method if working with structs
let sum = points.map(|p| p.x + p.y)
```

### No Irrefutable Pattern in Let

```sio
// WRONG - enum pattern in let (refutable)
// let Some(x) = optional_value

// CORRECT - use match
let x = match optional_value {
    Some(v) => v,
    None => default_value,
}

// OR use if-let style (when supported)
match optional_value {
    Some(x) => {
        // use x here
    },
    None => {
        // handle missing case
    },
}
```

## Pattern Matching with Effects

Pattern matching on refutable patterns may have the `Panic` effect:

```sio
fn risky_match(opt: Option<i32>) -> i32 with Panic {
    // If opt is None, this will panic
    match opt {
        Some(n) => n,
        // No None case - will panic if None
    }
}

// Better: handle all cases
fn safe_match(opt: Option<i32>) -> i32 {
    match opt {
        Some(n) => n,
        None => 0,  // Explicit handling
    }
}
```

## Best Practices

### 1. Prefer Exhaustive Matching

```sio
// Good: all cases explicit
fn handle(msg: Message) {
    match msg {
        Message::A => handle_a(),
        Message::B => handle_b(),
        Message::C => handle_c(),
    }
}

// Less ideal: wildcard hides new variants
fn handle_unsafe(msg: Message) {
    match msg {
        Message::A => handle_a(),
        _ => default_handler(),  // Might accidentally match new variants
    }
}
```

### 2. Use Guards for Range Checks

```sio
// Good: clear intent
match age {
    a if a < 0 => panic("invalid age"),
    a if a < 13 => "child",
    a if a < 20 => "teenager",
    a if a < 65 => "adult",
    _ => "senior",
}
```

### 3. Destructure Only What You Need

```sio
// Good: only extract needed fields
fn get_name(person: Person) -> string {
    match person {
        Person { name, .. } => name,  // Ignore other fields
    }
}
```

### 4. Order Patterns from Specific to General

```sio
match value {
    0 => "zero",           // Most specific first
    n if n < 0 => "negative",
    n if n < 100 => "small positive",
    _ => "large positive",  // Most general last
}
```

## See Also

- [Control Flow](/docs/language/control-flow.md) - Other control flow constructs
- [Enums](/docs/language/enums.md) - Defining enum types
- [Algebraic Effects](/docs/language/effects.md) - Effects and pattern matching
- [LLM Programming Guide](/docs/LLM_PROGRAMMING_GUIDE.md) - Complete syntax reference
