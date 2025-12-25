// Watch Mode Demo
//
// This example demonstrates the watch mode and hot reload features.
// Run with: dc watch examples --test
//
// The watch system supports:
// - Automatic rebuilds on file changes
// - Hot reload for running applications
// - Build hooks for custom actions
// - Development server with live reload

module watch_demo

// A simple counter that can be hot-reloaded
var counter: int = 0

// This function can be patched at runtime during hot reload
fn increment() -> int {
    counter = counter + 1
    return counter
}

// Reset the counter
fn reset() {
    counter = 0
}

// Get current value
fn get_value() -> int {
    return counter
}

// Main entry point
fn main() -> int with IO {
    println("Watch Mode Demo")
    println("===============")
    println("")

    // Demonstrate basic functionality
    println("Initial value: " + str(get_value()))

    increment()
    increment()
    increment()

    println("After 3 increments: " + str(get_value()))

    reset()
    println("After reset: " + str(get_value()))

    // Hot reload information
    println("")
    println("Hot Reload Support:")
    println("  - Functions can be patched at runtime")
    println("  - State is preserved across reloads")
    println("  - Rollback available on errors")

    return 0
}

// Example build hook configuration (d.toml format):
//
// [[hooks]]
// name = "format-check"
// points = ["pre-build"]
// command = "dc"
// args = ["fmt", "--check"]
// priority = 10
//
// [[hooks]]
// name = "run-tests"
// points = ["post-build"]
// command = "dc"
// args = ["test"]
// continue_on_failure = false

// Example build.d script:
//
// fn main() with IO {
//     // Rerun if this file changes
//     println("cargo:rerun-if-changed=build.d")
//
//     // Set a feature flag
//     println("cargo:rustc-cfg=feature=\"demo\"")
// }
