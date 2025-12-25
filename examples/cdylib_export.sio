//! Example: Exporting functions for use from C/Python/other languages
//!
//! Compile with: dc build --cdylib examples/cdylib_export.d
//! This will create a shared library (libcdylib_export.so/.dylib/.dll)

/// Simple add function exported with C ABI
pub extern "C" fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Multiply two numbers - also exported
pub extern "C" fn multiply(x: i32, y: i32) -> i32 {
    x * y
}

/// Compute factorial - demonstrates recursion in exported function
pub extern "C" fn factorial(n: i32) -> i32 {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}

/// Initialize the library (called when loaded)
pub extern "C" fn init() -> i32 {
    0
}

/// Cleanup the library (called when unloaded)
pub extern "C" fn cleanup() {
    // Nothing to clean up
}

// Internal helper - NOT exported (no extern "C")
fn internal_helper(x: i32) -> i32 {
    x * 2
}

// Main function for testing standalone
fn main() with IO {
    let result = add(10, 20)
    println(result)

    let fact = factorial(5)
    println(fact)
}
