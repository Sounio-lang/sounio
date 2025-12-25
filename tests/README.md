# Sounio Test Suite

This directory contains the language test suite for the Sounio compiler.

## Test Categories

### `ui/` - UI Tests
Tests for compiler diagnostics and error messages. Each test verifies that the compiler produces the expected error output.

### `run-pass/` - Run-Pass Tests  
Tests that should compile successfully and run without errors. These verify correct code generation and runtime behavior.

### `compile-fail/` - Compile-Fail Tests
Tests that should fail to compile. These verify that the compiler correctly rejects invalid programs.

## Running Tests

```bash
# Run all language tests
cd compiler && cargo test

# Run specific test category
cargo test ui::
cargo test run_pass::
cargo test compile_fail::
```

## Writing Tests

### UI Test Format

```d
// tests/ui/error_name.d
//@ error-pattern: expected error message

fn main() {
    // code that triggers the error
}
```

### Run-Pass Test Format

```d
// tests/run-pass/feature_name.d
//@ run-pass

fn main() {
    // code that should work
    assert(1 + 1 == 2)
}
```

### Compile-Fail Test Format

```d
// tests/compile-fail/invalid_syntax.d
//@ compile-fail
//@ error-pattern: type mismatch

fn main() {
    let x: int = "not an int"  // should fail
}
```

## Test Annotations

- `//@ run-pass` - Test should compile and run successfully
- `//@ compile-fail` - Test should fail to compile
- `//@ error-pattern: <text>` - Expected error message substring
- `//@ ignore` - Skip this test
- `//@ ignore-platform: <platform>` - Skip on specific platform
