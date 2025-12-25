//! mdBook integration for D documentation
//!
//! Generates mdBook-compatible documentation structure.

use std::fs;
use std::path::{Path, PathBuf};

use crate::doc::model::*;

/// mdBook generator
pub struct BookGenerator {
    /// Output directory
    output_dir: PathBuf,

    /// Book title
    title: String,

    /// Book description
    description: String,

    /// Author
    author: String,
}

impl BookGenerator {
    /// Create a new book generator
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            title: "Documentation".to_string(),
            description: String::new(),
            author: String::new(),
        }
    }

    /// Set the book title
    pub fn with_title(mut self, title: &str) -> Self {
        self.title = title.to_string();
        self
    }

    /// Set the book description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    /// Set the book author
    pub fn with_author(mut self, author: &str) -> Self {
        self.author = author.to_string();
        self
    }

    /// Generate mdBook structure
    pub fn generate(&self, crate_doc: &CrateDoc) -> Result<(), std::io::Error> {
        let book_dir = self.output_dir.join("book");
        let src_dir = book_dir.join("src");

        fs::create_dir_all(&src_dir)?;

        // Generate book.toml
        self.generate_book_toml(&book_dir)?;

        // Generate SUMMARY.md
        self.generate_summary(&src_dir, crate_doc)?;

        // Generate chapter files
        self.generate_chapters(&src_dir, crate_doc)?;

        Ok(())
    }

    /// Generate book.toml
    fn generate_book_toml(&self, book_dir: &Path) -> Result<(), std::io::Error> {
        let content = format!(
            r#"[book]
title = "{}"
description = "{}"
authors = ["{}"]
language = "en"

[build]
build-dir = "book"

[output.html]
default-theme = "light"
preferred-dark-theme = "navy"
git-repository-url = ""
edit-url-template = ""

[output.html.search]
enable = true
limit-results = 30
teaser-word-count = 30
use-boolean-and = true
boost-title = 2
boost-hierarchy = 1
boost-paragraph = 1
expand = true
heading-split-level = 3
"#,
            self.title, self.description, self.author
        );

        fs::write(book_dir.join("book.toml"), content)
    }

    /// Generate SUMMARY.md
    fn generate_summary(&self, src_dir: &Path, crate_doc: &CrateDoc) -> Result<(), std::io::Error> {
        let mut content = String::from("# Summary\n\n");

        // Introduction
        content.push_str("[Introduction](./introduction.md)\n\n");

        // Getting Started
        content.push_str("# Getting Started\n\n");
        content.push_str("- [Installation](./getting-started/installation.md)\n");
        content.push_str("- [Hello World](./getting-started/hello-world.md)\n");
        content.push_str("- [Basic Concepts](./getting-started/concepts.md)\n\n");

        // Language Reference
        content.push_str("# Language Reference\n\n");
        content.push_str("- [Types](./reference/types.md)\n");
        content.push_str("  - [Primitive Types](./reference/types/primitives.md)\n");
        content.push_str("  - [Compound Types](./reference/types/compounds.md)\n");
        content.push_str("  - [Generic Types](./reference/types/generics.md)\n");
        content.push_str("- [Functions](./reference/functions.md)\n");
        content.push_str("- [Control Flow](./reference/control-flow.md)\n");
        content.push_str("- [Ownership](./reference/ownership.md)\n");
        content.push_str("- [Effects](./reference/effects.md)\n");
        content.push_str("- [Traits](./reference/traits.md)\n");
        content.push_str("- [Modules](./reference/modules.md)\n\n");

        // Standard Library
        content.push_str("# Standard Library\n\n");
        self.add_module_to_summary(&crate_doc.root_module, &mut content, 0, &crate_doc.name)?;

        // Appendices
        content.push_str("\n# Appendices\n\n");
        content.push_str("- [Appendix A: Keywords](./appendix/keywords.md)\n");
        content.push_str("- [Appendix B: Operators](./appendix/operators.md)\n");
        content.push_str("- [Appendix C: Glossary](./appendix/glossary.md)\n");

        fs::write(src_dir.join("SUMMARY.md"), content)
    }

    /// Add module to SUMMARY.md
    fn add_module_to_summary(
        &self,
        module: &ModuleDoc,
        content: &mut String,
        depth: usize,
        crate_name: &str,
    ) -> Result<(), std::io::Error> {
        let indent = "  ".repeat(depth);

        // Module link
        let path = module.path.replace("::", "/");
        content.push_str(&format!(
            "{}- [{}](./api/{}/index.md)\n",
            indent,
            module.name,
            if module.path == crate_name {
                crate_name.to_string()
            } else {
                path
            }
        ));

        // Submodules
        for submodule in &module.modules {
            self.add_module_to_summary(submodule, content, depth + 1, crate_name)?;
        }

        Ok(())
    }

    /// Generate chapter markdown files
    fn generate_chapters(
        &self,
        src_dir: &Path,
        crate_doc: &CrateDoc,
    ) -> Result<(), std::io::Error> {
        // Introduction
        let intro = format!(
            r#"# Introduction

Welcome to the **{}** documentation!

{}

## Quick Links

- [Getting Started](./getting-started/installation.md)
- [API Reference](./api/{}/index.md)

## What is {}?

{} is a novel systems and scientific programming language featuring:

- **Algebraic Effects** - First-class effect system with handlers
- **Linear/Affine Types** - Safe resource management
- **Units of Measure** - Compile-time dimensional analysis
- **Refinement Types** - SMT-backed verification
- **GPU-Native** - First-class GPU computation support
"#,
            self.title,
            crate_doc.doc.as_deref().unwrap_or(""),
            crate_doc.name,
            self.title,
            self.title
        );
        fs::write(src_dir.join("introduction.md"), intro)?;

        // Create directory structure
        let dirs = [
            "getting-started",
            "reference",
            "reference/types",
            "api",
            "appendix",
        ];

        for dir in dirs {
            fs::create_dir_all(src_dir.join(dir))?;
        }

        // Generate API documentation
        self.generate_api_chapters(src_dir, &crate_doc.root_module, &crate_doc.name)?;

        // Generate reference chapters
        self.generate_reference_chapters(src_dir)?;

        // Generate getting started chapters
        self.generate_getting_started_chapters(src_dir)?;

        // Generate appendix chapters
        self.generate_appendix_chapters(src_dir)?;

        Ok(())
    }

    /// Generate API chapter for a module
    fn generate_api_chapters(
        &self,
        src_dir: &Path,
        module: &ModuleDoc,
        crate_name: &str,
    ) -> Result<(), std::io::Error> {
        let path = module.path.replace("::", "/");
        let module_dir = src_dir.join("api").join(if module.path == crate_name {
            crate_name
        } else {
            &path
        });
        fs::create_dir_all(&module_dir)?;

        let mut content = format!("# Module `{}`\n\n", module.path);

        // Module documentation
        if let Some(ref doc) = module.doc {
            content.push_str(doc);
            content.push_str("\n\n");
        }

        // Structs
        let structs: Vec<_> = module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::Struct)
            .collect();
        if !structs.is_empty() {
            content.push_str("## Structs\n\n");
            for ty in structs {
                content.push_str(&format!("### `{}`\n\n", ty.name));
                if let Some(ref doc) = ty.doc {
                    content.push_str(doc);
                }
                content.push_str("\n\n");

                // Fields
                if !ty.fields.is_empty() {
                    content.push_str("**Fields:**\n\n");
                    for field in &ty.fields {
                        content.push_str(&format!("- `{}`: `{}`", field.name, field.ty.display));
                        if let Some(ref doc) = field.doc {
                            content.push_str(&format!(" - {}", doc));
                        }
                        content.push('\n');
                    }
                    content.push('\n');
                }
            }
        }

        // Enums
        let enums: Vec<_> = module
            .types
            .iter()
            .filter(|t| t.kind == TypeKind::Enum)
            .collect();
        if !enums.is_empty() {
            content.push_str("## Enums\n\n");
            for ty in enums {
                content.push_str(&format!("### `{}`\n\n", ty.name));
                if let Some(ref doc) = ty.doc {
                    content.push_str(doc);
                }
                content.push_str("\n\n");

                // Variants
                if !ty.variants.is_empty() {
                    content.push_str("**Variants:**\n\n");
                    for variant in &ty.variants {
                        content.push_str(&format!("- `{}`", variant.name));
                        if let Some(ref doc) = variant.doc {
                            content.push_str(&format!(" - {}", doc));
                        }
                        content.push('\n');
                    }
                    content.push('\n');
                }
            }
        }

        // Traits
        if !module.traits.is_empty() {
            content.push_str("## Traits\n\n");
            for t in &module.traits {
                content.push_str(&format!("### `{}`\n\n", t.name));
                if let Some(ref doc) = t.doc {
                    content.push_str(doc);
                }
                content.push_str("\n\n");

                // Required methods
                if !t.required_methods.is_empty() {
                    content.push_str("**Required Methods:**\n\n");
                    for method in &t.required_methods {
                        content.push_str(&format!("```d\n{}\n```\n\n", method.signature));
                    }
                }
            }
        }

        // Functions
        if !module.functions.is_empty() {
            content.push_str("## Functions\n\n");
            for f in &module.functions {
                content.push_str(&format!("### `{}`\n\n", f.name));
                content.push_str(&format!("```d\n{}\n```\n\n", f.signature));
                if let Some(ref doc) = f.doc {
                    content.push_str(doc);
                }
                content.push_str("\n\n");
            }
        }

        // Constants
        if !module.constants.is_empty() {
            content.push_str("## Constants\n\n");
            for c in &module.constants {
                content.push_str(&format!("### `{}`\n\n", c.name));
                content.push_str(&format!(
                    "```d\nconst {}: {} = {};\n```\n\n",
                    c.name,
                    c.ty.display,
                    c.value.as_deref().unwrap_or("...")
                ));
                if let Some(ref doc) = c.doc {
                    content.push_str(doc);
                }
                content.push_str("\n\n");
            }
        }

        fs::write(module_dir.join("index.md"), content)?;

        // Recursively generate submodules
        for submodule in &module.modules {
            self.generate_api_chapters(src_dir, submodule, crate_name)?;
        }

        Ok(())
    }

    /// Generate reference chapters
    fn generate_reference_chapters(&self, src_dir: &Path) -> Result<(), std::io::Error> {
        // Types chapter
        let types_content = r#"# Types

D has a rich type system designed for safety and expressiveness.

## Overview

- **Primitive types**: `int`, `f64`, `bool`, `char`, `unit`
- **Compound types**: `struct`, `enum`, `tuple`
- **Generic types**: `Vec<T>`, `Option<T>`, `Result<T, E>`
- **Linear types**: Types that must be used exactly once
- **Affine types**: Types that can be used at most once
- **Refinement types**: Types with predicate constraints

## Type Annotations

Type annotations use the `:` syntax:

```d
let x: int = 42
let name: String = "hello"
let items: Vec<int> = Vec::new()
```

## Type Inference

D has powerful type inference:

```d
let x = 42        // Inferred as int
let y = 3.14      // Inferred as f64
let z = x + 1     // Inferred from context
```
"#;
        fs::write(src_dir.join("reference/types.md"), types_content)?;

        // Primitives
        let primitives_content = r#"# Primitive Types

## Integer Types

| Type | Size | Range |
|------|------|-------|
| `i8` | 8-bit | -128 to 127 |
| `i16` | 16-bit | -32,768 to 32,767 |
| `i32` | 32-bit | -2,147,483,648 to 2,147,483,647 |
| `i64` | 64-bit | -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807 |
| `int` | Platform | Platform-dependent |

## Unsigned Integer Types

| Type | Size | Range |
|------|------|-------|
| `u8` | 8-bit | 0 to 255 |
| `u16` | 16-bit | 0 to 65,535 |
| `u32` | 32-bit | 0 to 4,294,967,295 |
| `u64` | 64-bit | 0 to 18,446,744,073,709,551,615 |

## Floating Point Types

| Type | Size | Precision |
|------|------|-----------|
| `f32` | 32-bit | ~6-7 decimal digits |
| `f64` | 64-bit | ~15-16 decimal digits |

## Other Primitives

- `bool`: Boolean (`true` or `false`)
- `char`: Unicode scalar value
- `unit`: The unit type `()`
"#;
        fs::write(
            src_dir.join("reference/types/primitives.md"),
            primitives_content,
        )?;

        // Compounds
        let compounds_content = r#"# Compound Types

## Structs

Define custom data types with named fields:

```d
struct Point {
    x: f64,
    y: f64,
}

let p = Point { x: 1.0, y: 2.0 }
```

## Enums

Define types with a fixed set of variants:

```d
enum Color {
    Red,
    Green,
    Blue,
    RGB(u8, u8, u8),
}

let c = Color::RGB(255, 128, 0)
```

## Tuples

Group multiple values:

```d
let pair: (int, String) = (42, "hello")
let (x, y) = pair
```
"#;
        fs::write(
            src_dir.join("reference/types/compounds.md"),
            compounds_content,
        )?;

        // Generics
        let generics_content = r#"# Generic Types

## Type Parameters

Define types and functions that work with any type:

```d
struct Container<T> {
    value: T,
}

fn identity<T>(x: T) -> T {
    x
}
```

## Trait Bounds

Constrain type parameters:

```d
fn print_debug<T: Debug>(x: T) {
    println(x.debug_string())
}
```

## Where Clauses

Complex constraints:

```d
fn complex<T, U>(x: T, y: U) -> T
where
    T: Clone + Debug,
    U: Into<T>,
{
    // ...
}
```
"#;
        fs::write(
            src_dir.join("reference/types/generics.md"),
            generics_content,
        )?;

        // Functions chapter
        let functions_content = r#"# Functions

Functions in D are declared with the `fn` keyword.

## Basic Functions

```d
fn add(a: int, b: int) -> int {
    a + b
}
```

## Effects

Functions can declare effects they perform:

```d
fn read_file(path: &str) -> String with IO, Panic {
    // ...
}

fn simulate() -> f64 with Prob, Alloc {
    // ...
}
```

## Async Functions

```d
async fn fetch_data(url: &str) -> Response with IO {
    // ...
}
```

## Kernel Functions (GPU)

```d
kernel fn vector_add(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) {
    let idx = thread_idx()
    c[idx] = a[idx] + b[idx]
}
```
"#;
        fs::write(src_dir.join("reference/functions.md"), functions_content)?;

        // Control flow
        let control_flow_content = r#"# Control Flow

## If Expressions

```d
let x = if condition { 1 } else { 2 }
```

## Match Expressions

```d
match value {
    0 => println("zero"),
    1 | 2 => println("one or two"),
    n if n < 10 => println("small"),
    _ => println("other"),
}
```

## Loops

```d
// Infinite loop
loop {
    if condition { break }
}

// While loop
while condition {
    // ...
}

// For loop
for item in collection {
    // ...
}
```
"#;
        fs::write(
            src_dir.join("reference/control-flow.md"),
            control_flow_content,
        )?;

        // Ownership
        let ownership_content = r#"# Ownership

D uses ownership and borrowing for memory safety.

## Ownership Rules

1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Values can be borrowed (shared or exclusive)

## Borrowing

```d
fn read_only(x: &int) { }     // Shared borrow
fn modify(x: &!int) { }       // Exclusive borrow (mutable)
```

## Linear Types

Linear types must be used exactly once:

```d
linear struct FileHandle {
    fd: int,
}

// Must be used - cannot be dropped implicitly
let file = open("data.txt")
file.close()  // Must explicitly close
```

## Affine Types

Affine types can be used at most once:

```d
affine struct Connection {
    socket: Socket,
}
```
"#;
        fs::write(src_dir.join("reference/ownership.md"), ownership_content)?;

        // Effects
        let effects_content = r#"# Effects

D has a powerful effect system for tracking computational effects.

## Built-in Effects

- `IO`: Input/output operations
- `Mut`: Mutable state
- `Alloc`: Memory allocation
- `Panic`: Potential panics
- `Async`: Asynchronous operations
- `GPU`: GPU computation
- `Prob`: Probabilistic operations
- `Div`: Division (potential division by zero)

## Declaring Effects

```d
fn risky_operation() -> int with IO, Panic {
    // ...
}
```

## Effect Handlers

```d
handler LogIO for IO {
    fn print(msg: &str) {
        log(msg)
        resume()
    }
}

handle io_operation() with LogIO
```

## Custom Effects

```d
effect State<T> {
    fn get() -> T,
    fn set(value: T),
}
```
"#;
        fs::write(src_dir.join("reference/effects.md"), effects_content)?;

        // Traits
        let traits_content = r#"# Traits

Traits define shared behavior.

## Defining Traits

```d
trait Printable {
    fn to_string(&self) -> String
}
```

## Implementing Traits

```d
impl Printable for Point {
    fn to_string(&self) -> String {
        format("({}, {})", self.x, self.y)
    }
}
```

## Associated Types

```d
trait Iterator {
    type Item

    fn next(&mut self) -> Option<Self::Item>
}
```

## Default Implementations

```d
trait Greet {
    fn greet(&self) -> String {
        "Hello!"
    }
}
```
"#;
        fs::write(src_dir.join("reference/traits.md"), traits_content)?;

        // Modules
        let modules_content = r#"# Modules

D uses modules for code organization.

## Module Declaration

```d
module myproject::utils

pub fn helper() { }
```

## Imports

```d
use std::collections::Vec
use std::io::{read, write}
```

## Visibility

- `pub`: Public (visible everywhere)
- `pub(crate)`: Visible within the crate
- `pub(super)`: Visible in parent module
- Private (default): Visible in current module
"#;
        fs::write(src_dir.join("reference/modules.md"), modules_content)?;

        Ok(())
    }

    /// Generate getting started chapters
    fn generate_getting_started_chapters(&self, src_dir: &Path) -> Result<(), std::io::Error> {
        // Installation
        let install_content = r#"# Installation

## Prerequisites

- A modern operating system (Linux, macOS, or Windows)
- Git (for building from source)

## Installation Methods

### Quick Install (Recommended)

```bash
curl -sSf https://sounio-lang.org/install.sh | sh
```

### Building from Source

```bash
git clone https://github.com/sounio-lang/sounio
cd sounio/compiler
cargo build --release
```

### Package Managers

**Homebrew (macOS):**
```bash
brew install sounio
```

**Cargo:**
```bash
cargo install sounio
```

## Verifying Installation

```bash
d --version
```
"#;
        fs::write(
            src_dir.join("getting-started/installation.md"),
            install_content,
        )?;

        // Hello World
        let hello_content = r#"# Hello World

Let's write your first D program!

## Create a Project

```bash
d new hello_world
cd hello_world
```

## The Code

Open `src/main.sio`:

```d
fn main() {
    println("Hello, World!")
}
```

## Run It

```bash
d run
```

You should see:

```
Hello, World!
```

## Understanding the Code

- `fn main()` - The entry point of every D program
- `println(...)` - Prints text followed by a newline

## Next Steps

Try modifying the program:

```d
fn main() {
    let name = "Sounio"
    println("Hello, " + name + "!")
}
```
"#;
        fs::write(
            src_dir.join("getting-started/hello-world.md"),
            hello_content,
        )?;

        // Concepts
        let concepts_content = r#"# Basic Concepts

## Variables

```d
let x = 5           // Immutable
let mut y = 10      // Mutable
const PI = 3.14159  // Compile-time constant
```

## Functions

```d
fn greet(name: String) -> String {
    "Hello, " + name + "!"
}
```

## Structs

```d
struct Person {
    name: String,
    age: int,
}
```

## Enums

```d
enum Status {
    Active,
    Inactive,
    Pending,
}
```

## Control Flow

```d
if condition {
    // ...
} else {
    // ...
}

match value {
    1 => println("one"),
    2 => println("two"),
    _ => println("other"),
}
```

## Collections

```d
let vec = Vec::new()
vec.push(1)
vec.push(2)

let map = HashMap::new()
map.insert("key", "value")
```
"#;
        fs::write(
            src_dir.join("getting-started/concepts.md"),
            concepts_content,
        )?;

        Ok(())
    }

    /// Generate appendix chapters
    fn generate_appendix_chapters(&self, src_dir: &Path) -> Result<(), std::io::Error> {
        // Keywords
        let keywords_content = r#"# Appendix A: Keywords

## Reserved Keywords

| Keyword | Description |
|---------|-------------|
| `as` | Type casting |
| `async` | Asynchronous function |
| `await` | Await async result |
| `break` | Exit loop |
| `const` | Compile-time constant |
| `continue` | Skip to next iteration |
| `else` | Alternative branch |
| `enum` | Enumeration type |
| `effect` | Effect declaration |
| `extern` | External linkage |
| `false` | Boolean false |
| `fn` | Function declaration |
| `for` | For loop |
| `handler` | Effect handler |
| `if` | Conditional |
| `impl` | Implementation block |
| `import` | Import items |
| `in` | Iterator binding |
| `kernel` | GPU kernel function |
| `let` | Variable binding |
| `linear` | Linear type modifier |
| `loop` | Infinite loop |
| `match` | Pattern matching |
| `mod` | Module declaration |
| `move` | Move ownership |
| `mut` | Mutable binding |
| `pub` | Public visibility |
| `return` | Return from function |
| `self` | Self reference |
| `Self` | Self type |
| `struct` | Structure type |
| `trait` | Trait declaration |
| `true` | Boolean true |
| `type` | Type alias |
| `unsafe` | Unsafe block |
| `use` | Use declaration |
| `where` | Where clause |
| `while` | While loop |
| `with` | Effect annotation |
"#;
        fs::write(src_dir.join("appendix/keywords.md"), keywords_content)?;

        // Operators
        let operators_content = r#"# Appendix B: Operators

## Arithmetic Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `a + b` |
| `-` | Subtraction | `a - b` |
| `*` | Multiplication | `a * b` |
| `/` | Division | `a / b` |
| `%` | Remainder | `a % b` |

## Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `==` | Equal | `a == b` |
| `!=` | Not equal | `a != b` |
| `<` | Less than | `a < b` |
| `>` | Greater than | `a > b` |
| `<=` | Less or equal | `a <= b` |
| `>=` | Greater or equal | `a >= b` |

## Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&&` | Logical AND | `a && b` |
| `\|\|` | Logical OR | `a \|\| b` |
| `!` | Logical NOT | `!a` |

## Reference Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `&` | Shared reference | `&x` |
| `&!` | Exclusive reference | `&!x` |
| `*` | Dereference | `*ptr` |
"#;
        fs::write(src_dir.join("appendix/operators.md"), operators_content)?;

        // Glossary
        let glossary_content = r#"# Appendix C: Glossary

## A

**Affine Type**: A type that can be used at most once.

**Algebraic Effect**: A structured way to represent and handle computational effects.

## B

**Borrow**: A reference to a value without taking ownership.

## E

**Effect**: A computational side effect like I/O, mutation, or allocation.

**Effect Handler**: Code that handles and interprets effects.

## L

**Linear Type**: A type that must be used exactly once.

**Lifetime**: The scope during which a reference is valid.

## O

**Ownership**: The exclusive right to use and dispose of a value.

## R

**Refinement Type**: A type with additional predicate constraints verified by SMT solver.

## T

**Trait**: A collection of methods that types can implement.

## U

**Unit of Measure**: Compile-time dimensional analysis for numeric values.
"#;
        fs::write(src_dir.join("appendix/glossary.md"), glossary_content)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_book_generator_creation() {
        let generator = BookGenerator::new(PathBuf::from("/tmp/test"))
            .with_title("Test Book")
            .with_description("A test book")
            .with_author("Test Author");

        assert_eq!(generator.title, "Test Book");
        assert_eq!(generator.description, "A test book");
        assert_eq!(generator.author, "Test Author");
    }
}
