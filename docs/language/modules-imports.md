# Modules and Imports

This chapter covers Sounio's module system, including how to organize code into modules, control visibility, and manage dependencies between source files.

## Overview

Sounio's module system provides:

- **Namespacing**: Organize code into logical units to avoid name collisions
- **Encapsulation**: Control visibility of items with the `pub` modifier
- **Code organization**: Split large programs across multiple files
- **Dependency management**: Import functionality from other modules and packages

## Module Declaration

### Inline Modules

Use the `mod` keyword to declare an inline module:

```sio
mod geometry {
    pub struct Point {
        x: f64,
        y: f64,
    }

    pub fn distance(a: &Point, b: &Point) -> f64 {
        let dx = a.x - b.x
        let dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    // Private helper - not visible outside this module
    fn normalize(v: f64) -> f64 {
        if v < 0.0 { return -v }
        return v
    }
}

fn main() -> i32 {
    let p1 = geometry::Point { x: 0.0, y: 0.0 }
    let p2 = geometry::Point { x: 3.0, y: 4.0 }
    let d = geometry::distance(&p1, &p2)
    return 0
}
```

### File-Based Modules

For larger projects, modules can be organized across files. The module system follows these conventions:

**Single-file module:**
```
project/
  main.sio
  math.sio        # Imported as `math`
```

**Directory module:**
```
project/
  main.sio
  geometry/
    mod.sio       # Module root (required)
    point.sio     # Submodule
    vector.sio    # Submodule
```

The `mod.sio` file (or alternatively `lib.sio`) serves as the entry point for a directory-based module.

### Module Declaration Syntax

At the top of a file, declare the module name:

```sio
module geometry::point

pub struct Point {
    x: f64,
    y: f64,
}
```

## Imports

### Basic Import

Use `import` or `use` to bring items into scope:

```sio
// Import a module
import std::io

// Alternative syntax (equivalent)
use std::io

// Both :: and . work as path separators
import std.collections.HashMap
use std::collections::HashMap
```

### Selective Import

Import specific items from a module:

```sio
// Import specific items
import std::io::{read_file, write_file}

// Import multiple items
use std::collections::{Vec, HashMap, HashSet}
```

### Wildcard Import

Import all public items from a module:

```sio
import std::io::*

// Now read_file, write_file, etc. are available directly
let content = read_file("data.txt")
```

**Note**: Use wildcard imports sparingly as they can make it unclear where names come from.

### Aliased Import

Rename imports to avoid conflicts or for convenience:

```sio
import std::collections::HashMap as Map

let data: Map<string, i32> = Map::new()
```

### Path Separators

Sounio accepts both `::` and `.` as path separators:

```sio
// These are equivalent
import std::io::read_file
import std.io.read_file

// In expressions too
let result = geometry::distance(p1, p2)
let result = geometry.distance(p1, p2)
```

## Visibility

### Public Items

Use `pub` to make items visible outside their module:

```sio
mod math {
    // Public - accessible from outside
    pub const PI: f64 = 3.14159265359

    pub fn square(x: f64) -> f64 {
        return x * x
    }

    // Private - only accessible within this module
    fn internal_helper() -> f64 {
        return 0.0
    }
}
```

### Struct Field Visibility

By default, struct fields are private. Use `pub` on individual fields:

```sio
pub struct Config {
    pub name: string,        // Public field
    pub version: string,     // Public field
    api_key: string,         // Private field
}

impl Config {
    // Constructor can access private fields
    pub fn new(name: string, version: string, key: string) -> Config {
        return Config {
            name: name,
            version: version,
            api_key: key,
        }
    }

    // Public method to access private data
    pub fn has_api_key(&self) -> bool {
        return !self.api_key.is_empty()
    }
}
```

### Re-exports

Use `pub use` to re-export items from other modules:

```sio
// In geometry/mod.sio
mod point;
mod vector;

// Re-export for convenient access
pub use geometry::point::Point
pub use geometry::vector::Vector

// Users can now write:
// import geometry::Point
// instead of:
// import geometry::point::Point
```

## Standard Library Structure

The Sounio standard library is organized into these major modules:

| Module | Description |
|--------|-------------|
| `std::io` | File I/O, console I/O, paths |
| `std::collections` | Vec, HashMap, HashSet, Deque |
| `std::string` | String manipulation |
| `std::cmp` | Comparison, ordering, min/max |
| `std::json` | JSON parsing and serialization |
| `std::math` | Mathematical functions |
| `std::async` | Async runtime, futures, channels |
| `std::ffi` | Foreign function interface |
| `std::mem` | Memory management |
| `std::epistemic` | Knowledge types, uncertainty |

### Importing Standard Library

```sio
// Common patterns
import std::io::*
import std::collections::{Vec, HashMap}
import std::json::{parse_json, JsonValue}

// The std prefix is resolved to the stdlib directory
// automatically by the compiler
```

## Package Configuration

### Sounio.toml

Every Sounio package has a `Sounio.toml` manifest file:

```toml
[package]
name = "my-project"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
description = "A Sounio project"

[dependencies]
http = "1.0"
json = "2.1"
crypto = { version = "1.0", features = ["sha256"] }

# Local dependency
my-lib = { path = "../my-lib" }

# Git dependency
utils = { git = "https://github.com/user/utils" }
```

### Project Structure

A typical Sounio project:

```
my-project/
  Sounio.toml           # Package manifest
  Sounio.lock           # Lock file (auto-generated)
  src/
    main.sio            # Entry point for binaries
    lib.sio             # Entry point for libraries
    utils/
      mod.sio
      helper.sio
  tests/
    integration.sio
  examples/
    demo.sio
```

### Creating a New Package

```bash
# Initialize a new project
sou pkg init my-project

# Add a dependency
sou pkg add http

# Build the project
sou pkg build

# Run tests
sou pkg test
```

## Module Resolution

### Resolution Order

When resolving an import path like `import math::trig`, the compiler searches:

1. **Local modules**: Files in the same directory as the importing file
2. **Package modules**: Files relative to the package root
3. **Dependencies**: Packages listed in `Sounio.toml`
4. **Standard library**: The `std` prefix maps to the stdlib

### File Resolution Rules

For `import foo::bar`:

1. Look for `foo/bar.sio`
2. Look for `foo/bar/mod.sio`
3. Look for `foo/bar/lib.sio`
4. Try lowercase: `foo/bar.sio` (case-insensitive fallback)

## Circular Dependencies

Sounio detects and reports circular dependencies at compile time:

```sio
// a.sio
import b
pub fn foo() -> i32 { return b::bar() }

// b.sio
import a  // ERROR: Circular import detected
pub fn bar() -> i32 { return a::foo() }
```

### Resolving Circular Dependencies

**Strategy 1: Extract shared code**

```sio
// shared.sio - no dependencies on a or b
pub struct Data { value: i32 }

// a.sio
import shared
pub fn foo(d: &shared::Data) -> i32 { return d.value }

// b.sio
import shared
pub fn bar(d: &shared::Data) -> i32 { return d.value * 2 }
```

**Strategy 2: Use traits for abstraction**

```sio
// traits.sio
pub trait Processor {
    fn process(&self) -> i32
}

// a.sio
import traits::Processor
pub fn use_processor<T: Processor>(p: &T) -> i32 {
    return p.process()
}

// b.sio
import traits::Processor
pub struct MyProcessor {}
impl Processor for MyProcessor {
    fn process(&self) -> i32 { return 42 }
}
```

## Workspaces

For multi-package projects, use a workspace:

```toml
# Root Sounio.toml
[workspace]
members = [
    "core",
    "cli",
    "plugins/*"
]

[workspace.dependencies]
# Shared dependencies across all packages
serde = "1.0"
```

```toml
# core/Sounio.toml
[package]
name = "my-core"
version = "0.1.0"

[dependencies]
serde.workspace = true  # Inherit from workspace
```

## Best Practices

1. **Keep modules focused**: Each module should have a single responsibility

2. **Minimize public API**: Only expose what consumers need

3. **Use re-exports**: Create a clean public API through `pub use`

4. **Prefer explicit imports**: Avoid `*` imports in library code

5. **Document public items**: Add doc comments to public functions and types

6. **Avoid deep nesting**: Keep module hierarchies shallow (2-3 levels)

7. **Use meaningful names**: Module names should describe their contents

```sio
// Good
mod pharmacokinetics {
    pub mod absorption;
    pub mod distribution;
    pub mod metabolism;
    pub mod excretion;
}

// Less clear
mod pk {
    pub mod a;
    pub mod d;
    pub mod m;
    pub mod e;
}
```
