# Sounio Gotchas & Common Mistakes

These are the mistakes LLMs (and I) have made writing Sounio. Learn them.

## 1. SEMICOLONS - The #1 Mistake

### The Mistake
```sio
// ❌ COMPLETELY WRONG
let x = 5;
let y = 10;
fn foo() -> i32 {
    let result = x + y;
    result;
}
```

### Why It's Wrong
- Sounio expressions don't end with semicolons
- The parser treats `;` as statement separator, not terminator
- Result type becomes `()` if you add `;`

### The Fix
```sio
// ✅ CORRECT
let x = 5
let y = 10
fn foo() -> i32 {
    let result = x + y
    result
}
```

---

## 2. DYNAMIC ARRAYS / VECTORS

### The Mistake
```sio
// ❌ WRONG - Rust-ism
var data: Vec<u8> = vec![0, 1, 2, 3]
let slice: &[u8] = &data[..]
let len = data.len()
```

### Why It's Wrong
- Sounio has NO heap allocation in core language
- No `Vec<T>`, no slices `&[T]`, no `.len()` methods
- Only fixed-size arrays: `[Type; Size]`

### The Fix
```sio
// ✅ CORRECT - fixed array with explicit size and length variable
var data: [u8; 256] = [0; 256]  // Fixed capacity
var len: i32 = 4  // Track actual length yourself
data[0] = 0
data[1] = 1
data[2] = 2
data[3] = 3
// Now len = 4, data[0..4] is valid
```

---

## 3. MUTABLE REFERENCES: `&!` vs `&mut`

### The Mistake
```sio
// ❌ WRONG - Rust syntax
fn increment(x: &mut i32) with Mut {
    *x = *x + 1
}

var counter: i32 = 0
increment(&mut counter)
```

### Why It's Wrong
- Sounio uses `&!` (two tokens: `&` then `!`), not `&mut`
- This signals Sounio's different borrowing system

### The Fix
```sio
// ✅ CORRECT
fn increment(x: &!i32) with Mut {
    *x = *x + 1
}

var counter: i32 = 0
increment(&!counter)
```

---

## 4. MISSING EFFECTS

### The Mistake
```sio
// ❌ WRONG - missing Mut effect
fn set_value(x: &!i32) {
    *x = 42  // ERROR: mutation without 'with Mut'
}

// ❌ WRONG - missing Div effect
fn divide(a: f64, b: f64) -> f64 {
    a / b  // ERROR: division without 'with Div'
}

// ❌ WRONG - missing IO effect
fn say_hello() {
    print("Hello")  // ERROR: IO without 'with IO'
}
```

### Why It's Wrong
- Sounio tracks effects explicitly
- Division REQUIRES `with Div` (can panic on infinity)
- Mutation REQUIRES `with Mut`
- All I/O REQUIRES `with IO`

### The Fix
```sio
// ✅ CORRECT
fn set_value(x: &!i32) with Mut {
    *x = 42
}

fn divide(a: f64, b: f64) -> f64 with Div {
    a / b
}

fn say_hello() with IO {
    print("Hello")
}
```

---

## 5. RUST METHODS DON'T EXIST

### The Mistake
```sio
// ❌ WRONG - no methods
let text = "hello"
let len = text.len()          // No .len() method!
let upper = text.to_uppercase() // No .to_uppercase()!

let arr = [1, 2, 3]
let first = arr.first()       // No .first()!
let doubled = arr.iter().map(|x| x * 2).collect()  // No .iter()!
```

### Why It's Wrong
- Sounio is NOT object-oriented
- No method dispatch (no `.` operator for methods)
- Arrays have no built-in methods
- Strings are just `[i8; N]` arrays

### The Fix
```sio
// ✅ CORRECT - manual loops and functions
fn count_bytes(s: &[i8; 64]) -> i32 {
    var len = 0
    var i = 0
    while i < 64 {
        if s[i] == 0i8 { break }
        len = len + 1
        i = i + 1
    }
    len
}

// Process array
var arr: [i32; 10] = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
var sum = 0
var i = 0
while i < 5 {
    sum = sum + arr[i]
    i = i + 1
}
```

---

## 6. NEGATIVE NUMBERS & UNARY MINUS

### The Mistake
```sio
// ❌ WRONG - unary minus doesn't exist
let neg = -42
let result = -x
let value = a - -b
```

### Why It's Wrong
- Sounio has no unary minus operator
- Must use `0 - x` instead

### The Fix
```sio
// ✅ CORRECT
let neg = 0 - 42
let result = 0 - x
let value = a - (0 - b)  // = a + b
```

---

## 7. BIT SHIFT REQUIRES u8 OPERAND

### The Mistake
```sio
// ❌ WRONG - shift amount must be u8
let shifted = byte >> 4       // ERROR: 4 is i32!
let masked = byte & 255       // ERROR: 255 is i32!
```

### Why It's Wrong
- Shift operators require `u8` for the shift amount
- Bitwise AND requires both operands to match type

### The Fix
```sio
// ✅ CORRECT
let shifted = byte >> 4u8
let masked = byte & 255u8
let high = (byte >> 4u8) & 15u8
```

---

## 8. ARRAY SIZE MISMATCHES

### The Mistake
```sio
// ❌ WRONG - initialization size doesn't match type
var small_buffer: [u8; 10] = [0; 256]  // ERROR: 256 != 10!

// ❌ WRONG - accessing past bounds
var arr: [i32; 5] = [1, 2, 3, 4, 5]
let x = arr[100]  // ERROR: bounds check fails!
```

### Why It's Wrong
- Array type and initialization must match exactly
- Index must be statically provable to be in bounds (mostly)

### The Fix
```sio
// ✅ CORRECT
var buffer: [u8; 256] = [0; 256]

var arr: [i32; 100] = [0; 100]  // Safe to access 0..99
var i: i32 = 5
if i >= 0 && i < 100 {
    let x = arr[i as usize]  // Safe
}
```

---

## 9. TYPE MISMATCHES IN TUPLES/RETURNS

### The Mistake
```sio
// ❌ WRONG - missing return statement
fn compute(x: i32) -> i32 {
    let result = x * 2
    // Oops, forgot to return result!
}

// ❌ WRONG - wrong tuple type
fn divide_safe(a: f64, b: f64) -> (f64, i32) with Div {
    if b == 0.0 { return 0.0 }  // ERROR: should return (f64, i32) not f64!
}
```

### Why It's Wrong
- Function must return declared type
- Tuple returns require all elements

### The Fix
```sio
// ✅ CORRECT
fn compute(x: i32) -> i32 {
    let result = x * 2
    result
}

fn divide_safe(a: f64, b: f64) -> (f64, i32) with Div {
    if b == 0.0 { return (0.0, 1) }  // Error code
    (a / b, 0)  // Success
}
```

---

## 10. TYPE CASTING WITH `as`

### The Mistake
```sio
// ❌ WRONG - missing casts
let i: i32 = 5
let u: u8 = i        // ERROR: type mismatch!
let idx = 5          // Is it i32 or what?
let arr: [u8; 256] = [0; 256]
let val = arr[i]     // ERROR: arr[i32] but needs [usize]
```

### Why It's Wrong
- Type conversions require explicit `as` casts
- Array indexing requires `usize` (usually)

### The Fix
```sio
// ✅ CORRECT
let i: i32 = 5
let u: u8 = i as u8
let idx: usize = 5
let arr: [u8; 256] = [0; 256]
let val = arr[i as usize]
```

---

## 11. CLOSURES & HIGHER-ORDER FUNCTIONS

### The Mistake
```sio
// ❌ WRONG - Rust style, doesn't exist in Sounio
let numbers = [1, 2, 3, 4, 5]
let doubled = numbers.iter().map(|x| x * 2).collect()

let callback = |x| { x + 1 }
apply_callback(numbers, callback)
```

### Why It's Wrong
- Sounio has NO closures
- No lambda/anonymous functions
- No higher-order function dispatch

### The Fix
```sio
// ✅ CORRECT - named functions only
fn double(x: i32) -> i32 { x * 2 }

var numbers: [i32; 10] = [1, 2, 3, 4, 5, 0, 0, 0, 0, 0]
var result: [i32; 10] = [0; 10]
var i = 0
while i < 5 {
    result[i] = double(numbers[i])
    i = i + 1
}
```

---

## 12. EXCEPTION HANDLING DOESN'T EXIST

### The Mistake
```sio
// ❌ WRONG - Rust/Python style, doesn't exist
try {
    let x = risky_operation()
} catch (error) {
    print("Error!")
}

fn divide(a: f64, b: f64) -> f64 {
    if b == 0.0 { panic!("divide by zero") }
    a / b
}
```

### Why It's Wrong
- Sounio has no try/catch
- No exceptions (only effects + error codes)

### The Fix
```sio
// ✅ CORRECT - return error codes
fn divide(a: f64, b: f64) -> (f64, i32) with Div {
    if b == 0.0 {
        (0.0, 1)  // Error code 1 = divide by zero
    } else {
        (a / b, 0)  // Error code 0 = success
    }
}

// Caller:
let (result, err) = divide(10.0, 0.0)
if err != 0 {
    print("Error in division\n")
} else {
    print(result)
}
```

---

## 13. STRING MANIPULATION

### The Mistake
```sio
// ❌ WRONG - strings are not dynamic
let greeting = "Hello"
greeting.push_str(" World")   // No method!
let upper = greeting.to_uppercase()  // No method!
let split = greeting.split(' ')  // No method!
```

### Why It's Wrong
- Strings in Sounio are fixed-size char arrays `[i8; N]`
- No `.push_str()`, `.to_uppercase()`, etc.

### The Fix
```sio
// ✅ CORRECT - manual array manipulation
var greeting: [i8; 64] = [0; 64]
var len = 5
greeting[0] = 72i8   // 'H'
greeting[1] = 101i8  // 'e'
greeting[2] = 108i8  // 'l'
greeting[3] = 108i8  // 'l'
greeting[4] = 111i8  // 'o'

// To append: track length and add
if len + 6 <= 64 {
    greeting[len] = 32i8      // ' '
    greeting[len + 1] = 87i8  // 'W'
    greeting[len + 2] = 111i8 // 'o'
    greeting[len + 3] = 114i8 // 'r'
    greeting[len + 4] = 108i8 // 'l'
    greeting[len + 5] = 100i8 // 'd'
    len = len + 6
}
```

---

## 14. FOR LOOPS WITH RANGES

### The Mistake
```sio
// ❌ WRONG - syntax might work, but loop var is i64?
for i in 0..10 {
    print(i)
}

// ❌ WRONG - no step/stride syntax
for i in 0..100 by 2 {  // Doesn't exist!
    process(i)
}
```

### Why It's Wrong
- For-range loop var type might not be what you expect
- No built-in stride syntax

### The Fix
```sio
// ✅ CORRECT - while loop for control
var i: i32 = 0
while i < 10 {
    print(i)
    i = i + 1
}

// For stride:
var j: i32 = 0
while j < 100 {
    process(j)
    j = j + 2
}
```

---

## 15. STRUCT MUTATION PATTERN (JIT BUG WORKAROUND)

### The Mistake
```sio
// ❌ WRONG - JIT &! bug causes invisible mutations
struct Counter { value: i32 }

fn increment(c: &!Counter) with Mut {
    c.value = c.value + 1
}

var c = Counter { value: 0 }
increment(&!c)
print(c.value)  // Still 0! (JIT bug)
```

### Why It's Wrong
- Sounio JIT has a bug: `&!` mutations don't propagate to caller
- Affects mutable reference patterns

### The Fix
```sio
// ✅ CORRECT - by-value return pattern
struct Counter { value: i32 }

fn increment(c: Counter) -> Counter {
    Counter { value: c.value + 1 }
}

var c = Counter { value: 0 }
c = increment(c)  // Reassign to propagate change
print(c.value)  // Now 1! ✓
```

---

## Checklist Before Submitting

- [ ] No semicolons in function bodies
- [ ] All arrays are fixed-size `[Type; Size]`
- [ ] Mutable refs use `&!` not `&mut`
- [ ] All effects declared (`with Mut, Div, Panic, IO`)
- [ ] No Rust methods (`.len()`, `.push()`, `.iter()`)
- [ ] No unary minus (use `0 - x`)
- [ ] Bit shifts use `u8`: `x >> 4u8`
- [ ] Array sizes match in init: `[u8; 256] = [0; 256]`
- [ ] All functions return declared type
- [ ] Casts explicit: `i as u8`
- [ ] No closures, just named functions
- [ ] Error codes not exceptions
- [ ] String manipulation is manual array work
- [ ] No methods on built-in types
- [ ] Struct mutations use by-value return pattern

---

**TL;DR**: Sounio ≠ Rust. Study the existing code. When you're tempted to write Rust, **Stop. Check actual `.sio` files instead.**
