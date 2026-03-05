# Sprint 1: Critical Bug Fixes

## Goal: Fix security and correctness issues in core compiler

### Bug 1: Bounds Checking in char_from_i64

**File:** `self-hosted/compiler/main.sio`
**Function:** `char_from_i64`

**Current Code:**
```sio
fn char_from_i64(n: i64) -> string with Mut, Panic {
    str_slice("0123456789", n, n + 1)
}
```

**Problem:**
- No validation that `n` is between 0-9
- Will crash or produce garbage for invalid input
- Used by `int_to_string` which could pass invalid values

**Specification for Fix:**
```
Create a fixed version of char_from_i64 with:

Requirements:
1. Validate input: n must be 0 <= n <= 9
2. If invalid, panic with clear error message
3. Maintain same function signature
4. Performance: O(1) time, no allocation

Error message format:
"char_from_i64: digit must be 0-9, got {n}"

Test cases to verify:
- n=0 → "0"
- n=5 → "5"
- n=9 → "9"
- n=-1 → panic with error
- n=10 → panic with error

Generate the fixed implementation.
```

### Bug 2: Bounds Checking in arg_list_get

**File:** `self-hosted/compiler/main.sio`
**Function:** `arg_list_get`

**Current Code:**
```sio
fn arg_list_get(lst: ArgList, i: i64) -> string {
    lst.items[i as usize]
}
```

**Problem:**
- No bounds checking on array access
- Could read out of bounds memory
- No validation that i < lst.len

**Specification for Fix:**
```
Create a fixed version of arg_list_get with:

Requirements:
1. Check bounds: 0 <= i < lst.len
2. If out of bounds, panic with clear error
3. Include index and length in error message
4. Maintain same function signature (add Panic effect)

Error message format:
"arg_list_get: index {i} out of bounds (length {lst.len})"

Test cases:
- Valid index → returns item
- i = -1 → panic
- i = lst.len → panic
- i = lst.len + 1 → panic

Generate the fixed implementation.
```

### Bug 3: O(n²) String Concatenation

**File:** `self-hosted/compiler/main.sio`
**Function:** `int_to_string`

**Current Code:**
```sio
result = char_from_i64(ch) + result  // Prepending is O(n²)
```

**Problem:**
- String prepending creates new string each iteration
- For n-digit number: O(n²) time complexity
- Memory allocation overhead

**Specification for Fix:**
```
Create an optimized version of int_to_string:

Requirements:
1. Time complexity: O(n) where n = number of digits
2. Space complexity: O(n) temporary storage
3. Algorithm:
   a. Count digits first
   b. Allocate array of appropriate size
   c. Fill array from right to left
   d. Convert array to string
4. Handle all cases: zero, negative, large numbers
5. Use existing char_from_i64 (fixed version)

Performance targets:
- 1,000,000 conversions of random 64-bit integers in < 1 second
- Memory: < 100 bytes per conversion

Edge cases:
- n = 0 → "0"
- n = -0 → "0" (should not happen but handle)
- n = i64::MIN → special handling
- Very large numbers (close to i64 limits)

Generate the optimized implementation.
```

### Bug 4: Missing Error Capture in Compilation

**File:** `self-hosted/compiler/main.sio`
**Function:** `compile`

**Current Code:**
```sio
fn compile(opts: CompilerOptions) -> CompileResult with IO, Mut, Panic, Div, Alloc {
    let exit_code = compile_multimodule_native(...)
    CompileResult {
        success: exit_code == 0,
        errors: compile_errors_new(),  // Always empty!
        output: if exit_code == 0 { opts.output_file } else { "" },
    }
}
```

**Problem:**
- Errors from `compile_multimodule_native` not captured
- User gets "Compilation failed!" with no details
- No way to debug compilation issues

**Specification for Fix:**
```
We need to modify the compilation pipeline to capture errors.
Two options:

Option A: Modify compile_multimodule_native to return errors
Option B: Capture stderr output from compilation

Since we can't easily modify compile_multimodule_native, implement Option B:

Create a wrapper function that:
1. Redirects stderr to a temporary file
2. Calls compile_multimodule_native
3. Reads stderr output
4. Parses into error messages
5. Returns structured errors

Requirements:
1. Capture all error output
2. Parse into individual error messages
3. Preserve line numbers, file names
4. Format for user display
5. Clean up temporary files

Error format example:
"error: at foo.sio:10:5 - type mismatch, expected i32 got f64"

Generate the error capture implementation.
```

## Implementation Order

1. **Fix char_from_i64** (easiest, foundational)
2. **Fix arg_list_get** (similar pattern)
3. **Optimize int_to_string** (performance critical)
4. **Implement error capture** (most complex)

## Testing Strategy

For each fix, create:
1. Unit tests verifying correct behavior
2. Edge case tests
3. Performance benchmarks (for int_to_string)

## Success Criteria

- All bounds checking bugs fixed
- int_to_string 10x faster for large numbers
- Compilation errors displayed to user
- Zero regressions in existing functionality

## Time Estimate: 3-5 days

## Next Sprint: Testing Framework Implementation


