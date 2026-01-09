---
title: Hello World in Sounio
description: Write and run your first Sounio program
prerequisites: installation.md
reading_time: 8 minutes
---

# Hello World in Sounio

This tutorial introduces Sounio's basic syntax by walking through a simple program. By the end, you will understand functions, variables, types, and how to run Sounio code.

## Your First Program

Create a file called `hello.sio`:

```sio
fn main() -> i32 {
    print("Hello, Sounio!")
    println()
    0
}
```

Run it with:

```bash
souc run hello.sio
```

You should see:

```
Hello, Sounio!
```

Let us break down what this code does.

## Anatomy of a Sounio Program

### The main Function

Every Sounio program starts with a `main` function:

```sio
fn main() -> i32 {
    // Your code here
    0
}
```

- `fn` declares a function
- `main` is the entry point (required)
- `-> i32` specifies the return type (a 32-bit integer)
- The function body is enclosed in `{ }`
- `0` is the return value (0 typically means success)

### Console Output

Sounio provides simple functions for console output:

```sio
print("Hello")      // Print without newline
println()           // Print a newline
print("World")
```

Output:

```
Hello
World
```

Note: Unlike Rust, Sounio does not use macros. There is no `println!()` or `print!()` - just plain function calls.

## Variables

### Immutable Bindings with `let`

By default, variables in Sounio are immutable:

```sio
let x = 5
let name = "Sounio"
let pi = 3.14159

// This would be an error:
// x = 6  // Cannot reassign immutable variable
```

### Mutable Bindings with `var`

Use `var` when you need to reassign a variable:

```sio
var counter = 0
counter = counter + 1
counter = counter + 1
print(counter)  // Output: 2
```

### Compile-Time Constants

Use `const` for values known at compile time:

```sio
const MAX_SIZE: i32 = 1024
const EPSILON: f64 = 0.00001
```

## Basic Types

Sounio is statically typed. The compiler infers most types, but you can be explicit:

```sio
// Integers
let a: i32 = 42           // 32-bit signed
let b: i64 = 1000000000   // 64-bit signed
let c: u8 = 255           // 8-bit unsigned

// Floating point
let x: f64 = 3.14159      // 64-bit float (default for decimals)
let y: f32 = 2.71828      // 32-bit float

// Boolean
let flag: bool = true
let done: bool = false

// Strings
let message: string = "Hello, world!"

// Unit type (no value)
let nothing: () = ()
```

## Functions

### Basic Functions

```sio
fn add(a: i32, b: i32) -> i32 {
    return a + b
}

fn greet(name: string) {
    print("Hello, ")
    print(name)
    println()
}

fn main() -> i32 {
    let sum = add(3, 4)
    print(sum)  // Output: 7
    println()

    greet("Sounio")  // Output: Hello, Sounio

    0
}
```

### Implicit Returns

The last expression in a function can be the return value:

```sio
fn square(x: i32) -> i32 {
    x * x  // No semicolon, this is the return value
}
```

Using explicit `return` is also valid and often clearer:

```sio
fn square(x: i32) -> i32 {
    return x * x
}
```

## Control Flow

### If-Else

```sio
fn check_age(age: i32) {
    if age >= 18 {
        print("Adult")
    } else if age >= 13 {
        print("Teenager")
    } else {
        print("Child")
    }
    println()
}
```

### Match Expressions

Pattern matching is powerful in Sounio:

```sio
fn describe_number(n: i32) {
    match n {
        0 => print("zero"),
        1 => print("one"),
        2 | 3 => print("two or three"),
        _ => print("something else"),
    }
    println()
}
```

### Loops

```sio
// For loop with range (exclusive end)
for i in 0..5 {
    print(i)    // Prints 0, 1, 2, 3, 4
    print(" ")
}
println()

// Inclusive range
for i in 0..=5 {
    print(i)    // Prints 0, 1, 2, 3, 4, 5
    print(" ")
}
println()

// While loop
var count = 0
while count < 3 {
    print(count)
    count = count + 1
}
println()

// Infinite loop with break
var x = 0
loop {
    x = x + 1
    if x >= 5 {
        break
    }
}
```

## References

Sounio has two kinds of references:

### Shared References (`&T`)

Allow reading but not modifying:

```sio
fn print_value(x: &i32) {
    print(*x)  // Dereference to get the value
    println()
}

fn main() -> i32 {
    let value = 42
    print_value(&value)  // Pass a reference
    0
}
```

### Exclusive References (`&!T`)

Allow reading and modifying. **Note: Sounio uses `&!`, not `&mut` like Rust.**

```sio
fn increment(x: &!i32) {
    *x = *x + 1
}

fn main() -> i32 {
    var value = 10
    increment(&!value)  // Pass an exclusive reference
    print(value)        // Output: 11
    println()
    0
}
```

## A Complete Example

Here is a program that demonstrates multiple concepts:

```sio
// Calculate factorial using recursion
fn factorial(n: i32) -> i32 {
    if n <= 1 {
        return 1
    }
    return n * factorial(n - 1)
}

// Calculate factorial using iteration
fn factorial_iter(n: i32) -> i32 {
    var result = 1
    for i in 2..=n {
        result = result * i
    }
    return result
}

fn main() -> i32 {
    let n = 5

    print("Factorial of ")
    print(n)
    print(" is ")
    print(factorial(n))
    println()

    print("Iterative: ")
    print(factorial_iter(n))
    println()

    0
}
```

Output:

```
Factorial of 5 is 120
Iterative: 120
```

## Running and Checking Programs

### Type Checking Only

To check your code without running it:

```bash
souc check hello.sio
```

If there are no errors, this produces no output. Errors are reported with line numbers and explanations.

### Viewing the AST

For debugging, you can see the parsed structure:

```bash
souc check hello.sio --show-ast
```

### Viewing Inferred Types

See what types the compiler inferred:

```bash
souc check hello.sio --show-types
```

### Interactive REPL

For experimentation, use the REPL:

```bash
souc repl
```

Then type expressions to evaluate them immediately.

## Common Mistakes

### Using `&mut` instead of `&!`

```sio
// WRONG - Sounio does not have &mut
fn wrong(x: &mut i32) { }

// CORRECT - Use &! for exclusive references
fn correct(x: &!i32) { }
```

### Using Rust Macros

```sio
// WRONG - No macros in Sounio
println!("Hello")
assert!(x > 0)

// CORRECT - Use regular functions
print("Hello")
println()
if x <= 0 {
    panic("x must be positive")
}
```

### Forgetting `var` for Mutable Variables

```sio
// WRONG - let bindings are immutable
let count = 0
count = count + 1  // Error!

// CORRECT - Use var for mutable bindings
var count = 0
count = count + 1  // OK
```

## Next Steps

Now that you understand the basics, move on to Sounio's unique feature:

- [Your First Uncertainty](./your-first-uncertainty.md) - Learn how Sounio tracks uncertainty in computations

## See Also

- [Language Reference](../LLM_PROGRAMMING_GUIDE.md) - Complete syntax guide
- [Project Structure](./project-structure.md) - Organizing Sounio projects
- [Installation](./installation.md) - Compiler build options and features
