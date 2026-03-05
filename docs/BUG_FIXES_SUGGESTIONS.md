# Bug Fixes & Improvements for Sounio

## Issues Found During Code Review

### 1. Potential Buffer Overflow in `char_from_i64`
**File:** `self-hosted/compiler/main.sio`
**Line:** ~25

**Current code:**
```sio
fn char_from_i64(n: i64) -> string with Mut, Panic {
    str_slice("0123456789", n, n + 1)
}
```

**Problem:**
- No bounds checking on `n`
- If `n < 0` or `n > 9`, will slice out of bounds
- Could cause runtime panic or memory corruption

**Solution:**
```sio
fn char_from_i64(n: i64) -> string with Mut, Panic {
    if n < 0 || n > 9 {
        panic("char_from_i64: n must be between 0-9, got " + int_to_string(n))
    }
    str_slice("0123456789", n, n + 1)
}
```

### 2. Inefficient String Concatenation in `int_to_string`
**File:** `self-hosted/compiler/main.sio`
**Line:** ~30

**Current code:** Repeated string concatenation in a loop:
```sio
result = char_from_i64(ch) + result  // Prepending is O(n²)
```

**Problem:**
- String prepending creates new strings each iteration
- O(n²) time complexity for n digits
- Memory allocation overhead

**Solution:** Build string in array, then convert:
```sio
fn int_to_string(n: i64) -> string with Mut, Panic, Div, Alloc {
    if n == 0 {
        return "0"
    }
    
    let negative = n < 0
    var num = if negative { 0 - n } else { n }
    
    // Count digits first
    var digit_count = 0
    var temp = num
    while temp > 0 {
        digit_count = digit_count + 1
        temp = temp / 10
    }
    
    // Allocate array for digits
    var digits: [i8; 20]  // Max 64-bit int has 19 digits + sign
    var idx = if negative { 1 } else { 0 }
    
    // Fill from end
    var pos = digit_count - 1
    while pos >= 0 {
        let digit = num % 10
        digits[idx + pos] = (48 + digit) as i8
        num = num / 10
        pos = pos - 1
    }
    
    // Add negative sign if needed
    if negative {
        digits[0] = 45  // '-'
        idx = idx + 1
    }
    
    // Convert to string
    // (Need string_from_bytes function)
}
```

### 3. Fixed-Size Arrays Without Bounds Checking
**File:** `self-hosted/compiler/main.sio`
**Lines:** ~45, ~65

**Problem:** `ArgList` and `CompileErrors` use fixed-size arrays:
```sio
struct ArgList {
    items: [string; 64],
    len: i64,
}
```

But functions like `arg_list_get` don't check bounds:
```sio
fn arg_list_get(lst: ArgList, i: i64) -> string {
    lst.items[i as usize]  // No bounds check!
}
```

**Solution:** Add bounds checking:
```sio
fn arg_list_get(lst: ArgList, i: i64) -> string with Panic {
    if i < 0 || i >= lst.len {
        panic("arg_list_get: index " + int_to_string(i) + " out of bounds")
    }
    lst.items[i as usize]
}
```

### 4. Missing Error Messages in Compilation Pipeline
**File:** `self-hosted/compiler/main.sio`
**Line:** ~200

**Current:**
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

**Problem:** Errors from `compile_multimodule_native` aren't captured.

**Solution:** Need to modify `compile_multimodule_native` to return error messages, or capture stderr.

### 5. Hardcoded Array Initialization
**File:** `self-hosted/compiler/main.sio`
**Lines:** ~70-130

**Problem:** Massive hardcoded array initializers are error-prone:
```sio
items: ["", "", "", "", "", ... 256 times ...]
```

**Solution:** Create helper function:
```sio
fn create_empty_string_array(size: i64) -> [string; size] with Alloc {
    var arr: [string; size]
    var i: i64 = 0
    while i < size {
        arr[i] = ""
        i = i + 1
    }
    arr
}
```

## Performance Improvements

### 1. String Building
**Issue:** Current string operations are O(n²) in many places.

**Solution:** Implement `StringBuilder` type:
```sio
struct StringBuilder {
    buffer: [i8; 4096],
    length: i64,
}

fn string_builder_append(sb: StringBuilder, s: string) -> StringBuilder with Mut {
    // Copy characters from s to buffer
    // Resize if needed
}

fn string_builder_to_string(sb: StringBuilder) -> string with Alloc {
    // Create string from buffer[0..length]
}
```

### 2. Argument Parsing
**Current:** Linear scan through arguments O(n).

**Optimization:** Could use hash table for known options O(1) lookup.

## Security Improvements

### 1. Path Validation
**File:** `self-hosted/compiler/main.sio`
**Line:** ~150

**Problem:** No validation of `input_file` path.

**Solution:**
```sio
fn validate_file_path(path: string) -> bool with IO {
    // Check if file exists
    // Check for path traversal (../)
    // Check permissions
}
```

### 2. Output File Safety
**Problem:** Could overwrite important files.

**Solution:**
```sio
fn safe_output_path(path: string) -> string with IO {
    if file_exists(path) {
        // Ask for confirmation or generate unique name
    }
    path
}
```

## Testing Suggestions

### 1. Unit Tests Needed
```sio
// test_int_to_string.sio
fn test_int_to_string() -> bool {
    let cases = [
        (0, "0"),
        (42, "42"),
        (-42, "-42"),
        (999999, "999999"),
    ]
    
    for (input, expected) in cases {
        let result = int_to_string(input)
        if result != expected {
            println("FAIL: int_to_string(" + int_to_string(input) + ") = " + result + ", expected " + expected)
            return false
        }
    }
    true
}
```

### 2. Fuzz Testing
Create fuzzer for argument parsing:
```sio
fn fuzz_argument_parsing() {
    var random_args: [string; 100]
    // Generate random arguments
    let opts = parse_options(ArgList { items: random_args, len: 100 })
    // Should not crash
}
```

## Next Steps

1. **Priority 1**: Fix bounds checking issues
2. **Priority 2**: Improve string performance
3. **Priority 3**: Add proper error handling
4. **Priority 4**: Write comprehensive tests

## How to Apply These Fixes

Use your AI workflow with these specifications:

**Prompt template for fixes:**
```
I need to fix [PROBLEM] in [FILE].

Current code:
[PASTE CURRENT CODE]

Problems:
1. [ISSUE 1]
2. [ISSUE 2]

Requirements for fix:
1. [REQUIREMENT 1]
2. [REQUIREMENT 2]

Example of correct behavior:
[EXAMPLE]

Please generate the fixed code.
```
