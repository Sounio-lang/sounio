---
title: Project Structure
description: How to organize Sounio projects with modules, manifests, and imports
prerequisites: hello-world.md
reading_time: 10 minutes
---

# Project Structure

This tutorial explains how to organize Sounio projects beyond single-file programs. You will learn about the module system, project manifests, and how to import from the standard library.

## Directory Layout

A typical Sounio project has this structure:

```
my_project/
├── sounio.toml         # Project manifest
├── src/
│   ├── main.sio        # Entry point for binaries
│   └── lib.sio         # Library root
├── tests/
│   └── test_main.sio   # Test files
├── examples/
│   └── demo.sio        # Example programs
└── .gitignore
```

### Minimal Project

For small projects, you only need:

```
my_project/
├── sounio.toml
└── src/
    └── main.sio
```

## The Project Manifest: sounio.toml

Every Sounio project has a `sounio.toml` file at its root:

```toml
[package]
name = "my-project"
version = "0.1.0"
authors = ["Your Name <you@example.com>"]
description = "A Sounio project"

[dependencies]
# External dependencies go here
```

### Package Section

The `[package]` section defines project metadata:

```toml
[package]
name = "scientific-analysis"
version = "1.2.3"
authors = [
    "Alice Smith <alice@example.com>",
    "Bob Jones <bob@example.com>"
]
description = "Tools for scientific data analysis with uncertainty"
license = "MIT"
repository = "https://github.com/user/scientific-analysis"
```

### Dependencies Section

Add dependencies from the package registry:

```toml
[dependencies]
# Version from registry
json-parser = "1.0"

# Specific version
http-client = { version = "2.1.3" }

# Git dependency
my-lib = { git = "https://github.com/user/my-lib" }

# Local path (for development)
local-lib = { path = "../local-lib" }

# With feature flags
plotting = { version = "1.0", features = ["svg", "png"] }
```

## The Module System

### Declaring a Module

Each `.sio` file can declare its module name:

```sio
// In src/math/stats.sio
module math.stats

pub fn mean(data: &[f64]) -> f64 {
    var sum = 0.0
    for x in data {
        sum = sum + *x
    }
    return sum / len(data)
}

pub fn variance(data: &[f64]) -> f64 {
    let m = mean(data)
    var sum_sq = 0.0
    for x in data {
        let diff = *x - m
        sum_sq = sum_sq + diff * diff
    }
    return sum_sq / len(data)
}
```

### Public and Private

By default, functions and types are private to their module. Use `pub` to export:

```sio
module my_module

// Public - accessible from other modules
pub fn exported_function() -> i32 {
    return helper()
}

// Private - only accessible within this module
fn helper() -> i32 {
    return 42
}

// Public struct
pub struct Point {
    pub x: f64,    // Public field
    pub y: f64,    // Public field
}

// Private struct
struct InternalState {
    data: [f64],
}
```

### Importing Modules

Use `import` or `use` (they are equivalent) to bring modules into scope:

```sio
// Import entire module
import std::io

// Import specific items
import std::math::{sqrt, sin, cos}

// Both syntaxes work for paths
import std.collections.HashMap
use std::collections::HashSet

// Wildcard import (use sparingly)
import stdlib.epistemic.core::*
```

### Path Separators

Sounio accepts both `::` and `.` as path separators:

```sio
// These are equivalent
import std::io::read_file
import std.io.read_file
```

## Organizing Larger Projects

### Source Directory Structure

For a library with multiple modules:

```
my_library/
├── sounio.toml
└── src/
    ├── lib.sio           # Library root, re-exports public API
    ├── core/
    │   ├── mod.sio       # Module definition
    │   ├── types.sio
    │   └── utils.sio
    ├── analysis/
    │   ├── mod.sio
    │   ├── stats.sio
    │   └── regression.sio
    └── io/
        ├── mod.sio
        ├── csv.sio
        └── json.sio
```

### The lib.sio File

The library root re-exports the public API:

```sio
// src/lib.sio
module my_library

// Re-export submodules
pub use core::*
pub use analysis::*
pub use io::*

// Library-wide constants
pub const VERSION: string = "1.0.0"
```

### Module Definition Files

Each directory needs a `mod.sio`:

```sio
// src/core/mod.sio
module my_library.core

pub use types::*
pub use utils::*
```

## Importing from the Standard Library

Sounio has a rich standard library (151,000+ lines). Here are the main modules:

### Epistemic Types

```sio
import stdlib.epistemic.core::*

let measurement = epistemic_std(100.0, 2.5, 0.95)
```

### Input/Output

```sio
import std.io::*

let content = read_file("data.txt")
write_file("output.txt", result)
```

### Collections

```sio
import std.collections::*

var map: HashMap<string, i32> = HashMap::new()
map.insert("key", 42)

var set: HashSet<i32> = HashSet::new()
set.insert(1)
```

### Mathematics

```sio
import std.math::*

let result = sqrt(x) + sin(angle)
```

### JSON

```sio
import std.json::*

let data = parse_json(json_string)
let value = data["key"].as_i64()
```

### Scientific Computing

```sio
// Linear algebra
import stdlib.linalg::*

// ODE solving
import stdlib.ode::*

// Signal processing
import stdlib.signal::*

// Optimization
import stdlib.optimize::*
```

### Domain-Specific

```sio
// Pharmacokinetics/pharmacodynamics
import stdlib.medlang::*

// Neuroimaging
import stdlib.fmri::*

// Causal inference
import stdlib.causal::*

// Connectivity analysis
import stdlib.connectivity::*
```

## Tests Directory

Sounio looks for tests in the `tests/` directory:

```
my_project/
├── sounio.toml
├── src/
│   └── lib.sio
└── tests/
    ├── test_core.sio
    ├── test_analysis.sio
    └── integration/
        └── test_full_pipeline.sio
```

### Test File Format

Test files use special annotations:

```sio
//@ run-pass

import my_library::*

fn test_mean() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0]
    let result = mean(&data)
    assert(abs(result - 3.0) < 0.001)
}

fn test_variance() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0]
    let result = variance(&data)
    assert(abs(result - 2.0) < 0.001)
}

fn main() -> i32 {
    test_mean()
    test_variance()
    0
}
```

### Test Annotations

- `//@ run-pass` - Test should compile and run successfully
- `//@ compile-fail` - Test should fail to compile
- `//@ error-pattern: <text>` - Expected error message

## Examples Directory

Provide usage examples in `examples/`:

```
my_project/
├── sounio.toml
└── examples/
    ├── basic_usage.sio
    ├── advanced_analysis.sio
    └── integration_demo.sio
```

Run examples with:

```bash
souc run examples/basic_usage.sio
```

## Building and Running

### Development Workflow

```bash
# Check for errors without running
souc check src/main.sio

# Run the main program
souc run src/main.sio

# Run with verbose output
souc run src/main.sio --show-types
```

### Package Manager Commands

When using `sou pkg` (requires `--features pkg`):

```bash
# Initialize a new project
sou pkg init my-project

# Add a dependency
sou pkg add json-parser

# Install dependencies
sou pkg install

# Build the project
sou pkg build

# Run the project
sou pkg run

# Run tests
sou pkg test
```

## Best Practices

### Module Organization

1. **One responsibility per module** - Keep modules focused
2. **Minimize public exports** - Only expose what is needed
3. **Use descriptive names** - `analysis::regression` not `ar`
4. **Group related functionality** - Keep coupled code together

### File Naming

- Use lowercase with underscores: `my_module.sio`
- Match module name to file name: `module utils` in `utils.sio`
- Use `mod.sio` for directory modules

### Import Hygiene

```sio
// Prefer specific imports
import std::io::{read_file, write_file}

// Avoid wildcard imports except for common libraries
import stdlib.epistemic.core::*  // OK - frequently used types

// Do not use wildcards for large modules
// import std::*  // BAD - pollutes namespace
```

### Documentation

Document public items:

```sio
/// Calculate the arithmetic mean of a dataset.
///
/// # Arguments
/// * `data` - A slice of f64 values
///
/// # Returns
/// The arithmetic mean
///
/// # Example
/// ```
/// let values = [1.0, 2.0, 3.0]
/// let avg = mean(&values)  // Returns 2.0
/// ```
pub fn mean(data: &[f64]) -> f64 {
    // ...
}
```

## Next Steps

- [Editor Setup](./editor-setup.md) - Configure your development environment
- [Language Reference](../LLM_PROGRAMMING_GUIDE.md) - Complete syntax guide

## See Also

- [Installation](./installation.md) - Package manager setup
- [Your First Uncertainty](./your-first-uncertainty.md) - Core epistemic features
