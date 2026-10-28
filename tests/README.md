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
# Run the .sio language suite
bash scripts/run_sio_test_suite.sh

# Fast preflight
bash scripts/dev/fast_gate.sh
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

A file the suite selects needs one of `run-pass`, `compile-fail`, `check-only` or `typecheck-fail` in the leading header. Without one of those, the runner skips it (`no-annotation`) and does not compile it. The header is the leading run of `//@ ` lines, `//` comments and blank lines.

These are the keys `scripts/dev/run_sio_test_suite_v2.sh` reads:

- `//@ run-pass` - Compile and run; expect exit 0
- `//@ compile-fail` - Expect a compiler diagnostic, not a timeout or signal
- `//@ typecheck-fail` - Expect type-checking to fail
- `//@ check-only` - Compile only (`souc check`); do not execute
- `//@ ignore` - Skip this test
- `//@ error-pattern: <text>` - Diagnostics must contain `<text>` (`compile-fail` / `typecheck-fail`)
- `//@ expect-stdout: <text>` - Stdout must contain `<text>` (`run-pass`)
- `//@ expect-stdout-contains: <text>` - Stdout must contain `<text>` (`run-pass`)
- `//@ known-failure: <reason>` - Documented accepted failure
- `//@ xfail-expect: <text>` - The known failure's output must contain `<text>` (repeatable)
- `//@ xfail-exit: <n>` - The known failure's souc exit code must be `<n>` (0–255; 124 = timeout)
- `//@ skip-if: <condition>` - Conditional skip: `no-gpu`, `no-llvm` or `ci-only`
- `//@ requires: <feature>` - Feature gate: `gpu`, `llvm`, `madaros`, `lean_single`, `slow` or `vendor-material`
- `//@ flaky` - Known flaky test
- `//@ timeout: <seconds>` - Override the default 30s timeout for `check` and `run`. The integer is seconds.

An `expect-*` or `expected-*` key other than `expect-stdout` and `expect-stdout-contains` fails the test.
