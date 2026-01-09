---
title: Standard Library
description: Overview of the Sounio standard library
---

# Sounio Standard Library

The Sounio standard library provides a comprehensive set of modules for building systems and scientific applications. It includes core types, collections, I/O operations, iteration utilities, and domain-specific modules for scientific computing.

## Module Categories

### Core Types

Fundamental types that form the foundation of Sounio programs.

| Module | Description |
|--------|-------------|
| [`core/option`](core/option.md) | Optional values with `Option<T>` |
| [`core/result`](core/result.md) | Error handling with `Result<T, E>` |

### Collections

Data structures for storing and organizing data.

| Module | Description |
|--------|-------------|
| [`collections/vec`](collections/vec.md) | Growable arrays with `Vec<T>` |
| [`collections/hashmap`](collections/hashmap.md) | Hash-based key-value maps with `HashMap<K, V>` |
| `collections/hashset` | Hash-based sets with `HashSet<T>` |
| `collections/deque` | Double-ended queues with `Deque<T>` |

### Input/Output

File system operations and standard streams.

| Module | Description |
|--------|-------------|
| [`io/files`](io/files.md) | File reading, writing, and path operations |
| `io/env` | Environment variables and process control |

### Iteration

Lazy sequence processing and functional combinators.

| Module | Description |
|--------|-------------|
| [`iter`](iter.md) | Iterator trait and adapters |

### Scientific Computing

Specialized modules for scientific and epistemic computing.

| Module | Description |
|--------|-------------|
| `epistemic` | Knowledge types with uncertainty tracking |
| `prob` | Probability distributions and sampling |
| `stats` | Statistical functions and analysis |
| `linalg` | Linear algebra operations |
| `ode` | Ordinary differential equation solvers |
| `nn` | Neural network building blocks |

### Testing

Test framework and utilities.

| Module | Description |
|--------|-------------|
| `test` | Test assertions and benchmarking |
| `test/prop` | Property-based testing |
| `test/mock` | Mocking framework |

## Quick Reference

### Importing Modules

```sio
// Import entire module
import std::collections::vec

// Import specific items
import std::core::option::{Option, Some, None}

// Import with alias
import std::collections::hashmap as hm
```

### Common Patterns

**Option for nullable values:**

```sio
fn find_user(id: i32) -> Option<User> {
    // Returns Some(user) if found, None otherwise
}

// Usage
match find_user(42) {
    Some(user) => println("Found: " ++ user.name),
    None => println("User not found"),
}
```

**Result for error handling:**

```sio
fn read_config(path: &str) -> Result<Config, IoError> with IO {
    let content = read_file(path)?
    parse_config(content)
}

// Usage
match read_config("config.toml") {
    Ok(config) => use_config(config),
    Err(err) => eprintln("Error: " ++ err.message()),
}
```

**Vec for dynamic arrays:**

```sio
var items: Vec<i32> = Vec::new()
items.push(1)
items.push(2)
items.push(3)

for item in items.iter() {
    println(item.to_string())
}
```

**HashMap for key-value storage:**

```sio
var scores: HashMap<String, i32> = HashMap::new()
scores.insert("Alice", 100)
scores.insert("Bob", 85)

match scores.get(&"Alice") {
    Some(score) => println("Alice's score: " ++ score.to_string()),
    None => println("Alice not found"),
}
```

**Iterator chains:**

```sio
let result = numbers
    .iter()
    .filter(|x| *x > 0)
    .map(|x| x * 2)
    .take(5)
    .collect::<Vec<i32>>()
```

## Effect System Integration

Many stdlib functions require effect annotations:

| Effect | Used By |
|--------|---------|
| `IO` | File operations, environment access, printing |
| `Alloc` | Collection growth, cloning |
| `Panic` | Operations that may fail (unwrap, indexing) |

Example with effects:

```sio
fn process_data(path: &str) -> Vec<i32> with IO, Alloc, Panic {
    let content = read_file(path).unwrap()
    let lines = content.lines()

    var result: Vec<i32> = Vec::new()
    for line in lines {
        let num = line.parse::<i32>().unwrap()
        result.push(num)
    }

    result
}
```

## Memory Safety

Sounio uses exclusive references (`&!T`) instead of mutable references to ensure memory safety:

```sio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn process_vec(v: &!Vec<i32>) with Alloc {
    v.push(42)
    v.reverse()
}
```

## See Also

- [LLM Programming Guide](/docs/LLM_PROGRAMMING_GUIDE.md) - Comprehensive syntax reference
- [Getting Started](/docs/getting-started.md) - Introduction to Sounio
