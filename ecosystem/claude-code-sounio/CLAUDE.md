# Sounio Language Support

**Sounio** is an L0 systems + scientific programming language for epistemic computing.
It is **NOT** Rust, Julia, or ML. Own syntax, semantics, philosophy.

## Critical Syntax — What You MUST Know

| Wrong (Rust) | Right (Sounio) | Why |
|---|---|---|
| `let x = 5;` | `let x = 5` | No semicolons ever |
| `let mut x = 5` | `var x = 5` | `var` for mutable bindings |
| `&mut x` | `&!x` | `&!T` = exclusive reference |
| `assert!(cond)` | `assert(cond)` | No Rust macros |
| `println!("hi")` | `println("hi")` | No Rust macros |
| `-42` | `0 - 42` | No unary minus operator |
| `x >> 4` | `x >> 4u8` | Shift operand must be `u8` |
| `\|x\| x + 1` | named fn ref: `let f = add_one` | No closure literals |
| `#[test]` / `#[derive]` | `//@ run-pass` | No attributes |
| `Vec<T>` / `&[T]` | `[T; N]` fixed-size only | No dynamic arrays or slices |
| `Result<T,E>` | `(T, i32)` tuple | Error codes, not sum types |

## Effects System (MANDATORY)

Every function must declare its side effects. Missing effects = **compile error**.

| Effect | When Required |
|---|---|
| `IO` | `println()`, `print()`, file ops |
| `Mut` | Mutating `&!` refs, `var` reassignment |
| `Div` | Division `/`, modulo `%` |
| `Panic` | Array indexing, `assert()`, `as` casts |
| `Alloc` | Heap allocation |

```sio
fn pure(a: i64, b: i64) -> i64 { a + b }

fn greet(name: &[i8; 64]) with IO { println("hello") }

fn process(buf: &![i64; 100]) with IO, Mut, Panic, Div {
    buf[0] = buf[1] / buf[2]
    println("done")
}
```

## Language Reference

### Variables
```sio
let x = 5                    // immutable
let y: i64 = 10              // with type annotation
var z = 0                    // mutable — can reassign
z = 42
```

### Types
```sio
// Primitives: i8, i16, i32, i64, u8, u16, u32, u64, f32, f64, bool
// Fixed arrays (size required)
let arr: [i64; 4] = [1, 2, 3, 4]
let zeros = [0; 256]         // 256 zeros
let combined = a ++ b        // concatenation

// Structs
struct Point { x: f64, y: f64 }
let p = Point { x: 1.0, y: 2.0 }

// Enums
enum Color { Red, Green, Blue }

// Linear types (consumed on use)
linear struct Handle { fd: i32 }

// Refinement types
type Probability = { p: f64 | p >= 0.0 && p <= 1.0 }

// Units of measure
let dose: mg = 500.0

// Epistemic types
let k: Knowledge<mg> = measure(500.0, uncertainty: 2.5)
```

### Functions
```sio
fn add(a: i64, b: i64) -> i64 { a + b }

fn divide(a: f64, b: f64) -> f64 with Div, Panic { a / b }

// impl blocks — explicit self
impl Point {
    fn magnitude(self: &Point) -> f64 with Div, Panic {
        sqrt(self.x * self.x + self.y * self.y)
    }
    fn set_x(self: &!Point, v: f64) with Mut { self.x = v }
}

// Function references (NOT closures)
fn square(x: i64) -> i64 { x * x }
let f = square
let result = f(7)            // 49

// Higher-order
fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
```

### Control Flow
```sio
// if/else (expression)
let v = if x > 0 { 1 } else { 0 }

// while
var i = 0
while i < 10 { i = i + 1 }

// for-in
for i in 0..10 { }           // exclusive range
for i in 0..=10 { }          // inclusive range
for x in arr { }             // array iteration

// match
match color {
    Color::Red => 1
    Color::Green => 2
    _ => 0
}
```

### References
```sio
let r: &i64 = &x             // shared (read-only)
let m: &!i64 = &!y           // exclusive (mutable)
*m = 99                       // deref to mutate
```

**Known JIT bug**: Bare `&![T; N]` array mutations don't propagate. Wrap in struct:
```sio
struct Buffer { data: [i64; 100] }
fn mutate(buf: &!Buffer) with Mut { buf.data[0] = 99 }
```

### Modules & Imports
```sio
use my_module::{my_fn, MyStruct}
pub fn exported() -> i64 { 42 }
```

### Strings & I/O
```sio
println("Hello, World!")
print("no newline")
// String concatenation: "a" ++ "b"
```

### FFI (math functions only)
```sio
extern "C" {
    fn sqrt(x: f64) -> f64
    fn sin(x: f64) -> f64
}
```

**Integer FFI is broken** in JIT — only `f64` math functions work.

### Testing
```sio
//@ run-pass
fn main() -> i32 with IO, Mut, Panic {
    var passed = 0
    assert(1 + 1 == 2)
    passed = passed + 1
    println("ALL PASS")
    0
}
```

## Compiler Commands

```bash
# Set compiler path
export SOUC=/path/to/souc-linux-x86_64-jit

$SOUC check file.sio             # type-check only
$SOUC check file.sio --show-ast  # dump AST
$SOUC check file.sio --show-types # dump types
$SOUC run file.sio               # JIT execute
```

## Slash Commands

- `/sounio-check [file]` — Type-check a .sio file
- `/sounio-run [file]` — JIT-execute a .sio file
- `/sounio-lint [file]` — Scan for Rust-isms and fix them
