<!-- docs:meta
topic_id: repo.docs.archived.getting-started-duplicates.check-sounio-guide
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.getting-started-duplicates.check-sounio-guide
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sounio Code Validator Guide

## Overview

The `check_sounio.sh` script validates Sounio code files for common errors and Rust-isms that shouldn't appear in Sounio code. It helps ensure code follows Sounio conventions and idioms.

## What It Checks

### 1. Rust Method Calls ❌
Sounio doesn't use Rust-style method calls like `.len()`, `.push()`, `.iter()`, etc.
- **Error**: `.first()`, `.last()`, `.iter().sum()`
- **Allowed**: `.ε`, `.value`, `println()`, `print()`, `assert()`

### 2. Mutable References ❌
Sounio uses `&!` for mutable references, not Rust's `&mut`.
- **Error**: `&mut x`
- **Correct**: `&!x`

### 3. Vec Type ❌
Sounio uses fixed-size arrays instead of Rust's `Vec`.
- **Error**: `Vec<i32>`, `Vec::new()`
- **Correct**: `[i32; 10]`, `[0; 10]`

### 4. Unnecessary Semicolons ❌
Sounio generally doesn't use semicolons (except in array literals).
- **Error**: `let x = 10;`
- **Allowed**: `[0; 10]` (array literal)
- **Correct**: `let x = 10`

### 5. Missing Function Effects ⚠️
Functions with certain operations need explicit effects.
- **Warning**: Function with `&!` parameter but no `with Mut` effect
- **Warning**: Function with division but no `with Div` effect

### 6. Missing Return Statements ⚠️
Sounio requires explicit return statements for non-void functions.
- **Warning**: Function without `return` statement

### 7. Knowledge Type Patterns ℹ️
Checks for proper usage of Sounio's epistemic types.
- **Info**: `Knowledge[f64]` type references
- **Info**: `Knowledge()` constructor usage
- **Info**: `.ε` field access for confidence

## Usage

```bash
# Check a single file
./check_sounio.sh examples/hello.sio

# Check multiple files
./check_sounio.sh file1.sio file2.sio

# Check all .sio files in current directory
./check_sounio.sh *.sio

# Check all .sio files recursively
find . -name "*.sio" -exec ./check_sounio.sh {} \;
```

## Exit Codes

- `0`: All files passed validation (may have warnings)
- `1`: Critical errors found in one or more files

## Examples

### Bad Sounio Code (will fail):
```sounio
fn bad_example(&mut x: i32) -> i32 {
    let v: Vec<i32> = Vec::new();
    v.push(10);
    let len = v.len();
    return len;
}
```

### Good Sounio Code (will pass):
```sounio
fn good_example(x: &!i32) -> i32 with Mut {
    let arr: [i32; 5] = [1, 2, 3, 4, 5]
    var sum = 0
    for i in 0..5 {
        sum = sum + arr[i]
    }
    return sum
}
```

## Testing the Validator

Test files are available:
- `test_rusty_sounio.sio` - Contains intentional errors for testing
- `examples/hello.sio` - Simple valid Sounio code
- `tests/run-pass/` - More complex valid Sounio examples

## Implementation Notes

The script uses `grep` patterns to identify issues. It's designed to be:
- **Fast**: Uses simple pattern matching
- **Conservative**: May produce false warnings but not false errors
- **Educational**: Provides clear error messages with suggestions

## Limitations

1. **Simple pattern matching**: Doesn't fully parse Sounio syntax
2. **Function body analysis**: Only checks function signatures for effects
3. **Array literal detection**: Basic regex for `[x; y]` patterns
4. **Comments**: Handles simple `//` comments but not block comments

## Future Improvements

1. Integrate with Sounio compiler for more accurate checking
2. Add support for more Sounio-specific patterns
3. Improve function body analysis
4. Add configuration options
5. Support for CI/CD integration
