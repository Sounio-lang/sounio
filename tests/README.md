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
# Run the Rust compiler test suite (includes the language suite)
cargo test -p souc

# Run only the `.sio` language fixtures under `tests/`
cargo test -p souc --test language_suite

# Fast preflight (drift scan + key tests)
./scripts/fast_gate.sh
```

## Writing Tests

### UI Test Format

```sio
// tests/ui/error_name.sio
//@ error-pattern: expected error message

fn main() {
    // code that triggers the error
}
```

### Run-Pass Test Format

```sio
// tests/run-pass/feature_name.sio
//@ run-pass

fn main() {
    // code that should work
    assert(1 + 1 == 2)
}
```

### Compile-Fail Test Format

```sio
// tests/compile-fail/invalid_syntax.sio
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
- `//@ check-only` - For `run-pass` tests, only run `souc check` (skip `souc run`)
- `//@ timeout-ms: <n>` - Override timeout for `check` and `run` (milliseconds)
- `//@ run-timeout-ms: <n>` - Override timeout for `run` only (milliseconds)
