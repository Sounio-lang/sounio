# Stdlib Test Suite

This directory contains correctness tests for Sounio standard library modules.

## Organization

Test files mirror the stdlib structure:

```
tests/stdlib/
├── causal/
├── graph/
├── collections/
├── ode/
├── autodiff/
├── compiler/
├── ffi/
├── epistemic/
└── ...
```

## Running Tests

```bash
# The suite picks up tests/stdlib/*/test_*.sio
bash scripts/run_sio_test_suite.sh

# Fast preflight
bash scripts/dev/fast_gate.sh
```

## Writing New Tests

When adding new stdlib features, include comprehensive tests. Follow these guidelines:

1. **File naming**: `test_*.sio` for test files
2. **Location**: Match the stdlib structure (e.g., `stdlib/mymodule/` → `tests/stdlib/mymodule/test_mymodule.sio`)
3. **Coverage**: Test:
   - Basic functionality
   - Edge cases
   - Error conditions (if applicable)
4. **Output**: Print clear pass/fail indicators

Example test structure:

```sio
module test_myfeature

use std::mymodule::*

fn test_basic() -> bool {
    let result = my_function()
    if result == expected {
        print("✓ Basic test passed\n")
        return true
    } else {
        print("✗ Basic test failed\n")
        return false
    }
}

fn main() -> i32 {
    var passed = 0
    var total = 1

    if test_basic() { passed = passed + 1 }

    print(passed, " / ", total, " tests passed\n")

    if passed == total { 0 } else { 1 }
}
```

## Test Annotations

Tests can use annotations to control behavior:

```sio
//@ run-pass        - Should compile and run successfully
//@ compile-fail    - Should fail to compile
//@ ignore          - Skip this test
```

## Related

- **Examples**: [../../examples/README.md](../../examples/README.md)
- **Stdlib**: [../../stdlib/README.md](../../stdlib/README.md)
