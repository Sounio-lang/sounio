<!-- docs:meta
topic_id: repo.docs.tutorials.higher-order-functions
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.tutorials.higher-order-functions
-->

# Higher-Order Functions in Sounio Without Closures

Sounio supports first-class function references: you can store named functions in variables, pass them as arguments, and return them from other functions. Closure *literals* (`|x| x + 1`) are currently blocked, but named function references cover the most important higher-order patterns: map, filter, fold, any, all, and sort-by-comparator.

This tutorial shows verified patterns from the test suite. The source files are:

- [`tests/run-pass/closure_fn_ref.sio`](../../tests/run-pass/closure_fn_ref.sio) -- function references, `apply`, `select_op`
- [`tests/run-pass/closure_higher_order.sio`](../../tests/run-pass/closure_higher_order.sio) -- map, fold, any, all

## Prerequisites

```bash
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
$SOUC run tests/run-pass/closure_fn_ref.sio
$SOUC run tests/run-pass/closure_higher_order.sio
```

## 1. Function References as Values

A named function can be stored in a variable or passed as an argument. The type is written `fn(ParamTypes) -> ReturnType`.

```sio
fn square(x: i64) -> i64 { x * x }
fn add_one(x: i64) -> i64 { x + 1 }

fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }

fn main() -> i32 with IO, Mut, Panic, Div {
    // Store in variable
    let f = square
    let r = f(7)              // 49
    assert(r == 49)

    // Pass directly
    let r2 = apply(add_one, 5)  // 6
    assert(r2 == 6)

    print(0)
    0
}
```

Key syntax: `fn(i64) -> i64` is the type of any function that takes one `i64` and returns an `i64`. You call through the variable with normal call syntax: `f(7)`.

## 2. Returning Functions from Functions

Functions can return function references, enabling dynamic dispatch based on runtime values.

```sio
fn add_one(x: i64) -> i64 { x + 1 }
fn sub_one(x: i64) -> i64 { x - 1 }
fn negate(x: i64) -> i64 { 0 - x }

fn select_op(which: i64) -> fn(i64) -> i64 with Mut, Panic, Div {
    if which == 0 { add_one }
    else if which == 1 { sub_one }
    else { negate }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let op = select_op(0)
    assert(op(10) == 11)

    let op2 = select_op(2)
    assert(op2(10) == 0 - 10)

    // Dynamic dispatch in a loop
    var total: i64 = 0
    var i: i64 = 0
    while i < 3 {
        let op = select_op(i)
        total = total + op(10)
        i = i + 1
    }
    // add_one(10) + sub_one(10) + negate(10) = 11 + 9 + (-10) = 10
    assert(total == 10)

    print(0)
    0
}
```

This is how you build strategy patterns, plugin systems, or configurable pipelines without closures.

## 3. Map: Transform Every Element

The `map` pattern applies a function to each element of a fixed-size array and returns a new array.

```sio
fn dbl(x: i64) -> i64 { x * 2 }
fn sq(x: i64) -> i64 { x * x }

fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 {
        out[i] = f(arr[i])
        i = i + 1
    }
    out
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let data: [i64; 4] = [1, 2, 3, 4]

    let doubled = map4(data, dbl)
    assert(doubled[0] == 2)
    assert(doubled[1] == 4)
    assert(doubled[2] == 6)
    assert(doubled[3] == 8)

    let squared = map4(data, sq)
    assert(squared[0] == 1)
    assert(squared[1] == 4)
    assert(squared[2] == 9)
    assert(squared[3] == 16)

    print(0)
    0
}
```

Because Sounio uses fixed-size arrays, `map4` is hardcoded to 4 elements. Write `map8`, `map16`, etc. for other sizes. This is explicit but avoids dynamic allocation.

## 4. Fold: Reduce to a Single Value

Fold (also called reduce) combines all elements using a binary function and an initial accumulator.

```sio
fn add(a: i64, b: i64) -> i64 { a + b }
fn mul(a: i64, b: i64) -> i64 { a * b }
fn max2(a: i64, b: i64) -> i64 { if a > b { a } else { b } }

fn fold4(arr: [i64; 4], init: i64, f: fn(i64, i64) -> i64) -> i64 with Mut, Panic, Div {
    var acc = init
    var i: i64 = 0
    while i < 4 {
        acc = f(acc, arr[i])
        i = i + 1
    }
    acc
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let data: [i64; 4] = [1, 2, 3, 4]

    let sum = fold4(data, 0, add)          // 10
    assert(sum == 10)

    let product = fold4(data, 1, mul)      // 24
    assert(product == 24)

    let max_val = fold4(data, 0, max2)     // 4
    assert(max_val == 4)

    print(0)
    0
}
```

The type of the combining function is `fn(i64, i64) -> i64`. The first argument is the accumulator, the second is the current element.

## 5. Any and All: Predicate Testing

Test whether any or all elements satisfy a predicate. Because Sounio does not have a dedicated `bool` return for predicates over collections, these use `i64` where 0 means false and nonzero means true.

```sio
fn is_positive(x: i64) -> i64 { if x > 0 { 1 } else { 0 } }
fn is_gt3(x: i64) -> i64 { if x > 3 { 1 } else { 0 } }
fn is_gt10(x: i64) -> i64 { if x > 10 { 1 } else { 0 } }
fn is_gt2(x: i64) -> i64 { if x > 2 { 1 } else { 0 } }

fn any4(arr: [i64; 4], pred: fn(i64) -> i64) -> i64 with Mut, Panic, Div {
    var i: i64 = 0
    while i < 4 {
        if pred(arr[i]) != 0 { return 1 }
        i = i + 1
    }
    0
}

fn all4(arr: [i64; 4], pred: fn(i64) -> i64) -> i64 with Mut, Panic, Div {
    var i: i64 = 0
    while i < 4 {
        if pred(arr[i]) == 0 { return 0 }
        i = i + 1
    }
    1
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let data: [i64; 4] = [1, 2, 3, 4]

    assert(any4(data, is_gt3) == 1)     // yes, 4 > 3
    assert(any4(data, is_gt10) == 0)    // no element > 10
    assert(all4(data, is_positive) == 1) // all > 0
    assert(all4(data, is_gt2) == 0)     // 1 and 2 are not > 2

    print(0)
    0
}
```

Each predicate is a standalone named function. Without closure literals, you write `is_gt3` and `is_gt10` as separate functions. This is more verbose but perfectly explicit.

## 6. Chaining: Map Then Fold

Higher-order functions compose by passing the output of one as input to another.

```sio
fn sq(x: i64) -> i64 { x * x }
fn add(a: i64, b: i64) -> i64 { a + b }

fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 {
        out[i] = f(arr[i])
        i = i + 1
    }
    out
}

fn fold4(arr: [i64; 4], init: i64, f: fn(i64, i64) -> i64) -> i64 with Mut, Panic, Div {
    var acc = init
    var i: i64 = 0
    while i < 4 {
        acc = f(acc, arr[i])
        i = i + 1
    }
    acc
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let data: [i64; 4] = [1, 2, 3, 4]

    // Sum of squares: 1 + 4 + 9 + 16 = 30
    let sum_sq = fold4(map4(data, sq), 0, add)
    assert(sum_sq == 30)

    print(0)
    0
}
```

The expression `fold4(map4(data, sq), 0, add)` reads inside-out: square each element, then sum.

## 7. Sort by Comparator

Sorting with a custom comparator is the classic higher-order pattern. Sounio uses the struct wrapper pattern for mutable array parameters.

```sio
struct SortBuf { data: [i64; 16] }

fn sort_by(buf: &!SortBuf, n: i64, cmp: fn(i64, i64) -> i64) with Mut, Panic, Div {
    // Insertion sort with custom comparator
    // cmp(a, b) returns negative if a < b, 0 if equal, positive if a > b
    var i: i64 = 1
    while i < n {
        let key = buf.data[i]
        var j: i64 = i - 1
        while j >= 0 && cmp(buf.data[j], key) > 0 {
            buf.data[j + 1] = buf.data[j]
            j = j - 1
        }
        buf.data[j + 1] = key
        i = i + 1
    }
}

// Ascending order
fn cmp_asc(a: i64, b: i64) -> i64 { a - b }

// Descending order
fn cmp_desc(a: i64, b: i64) -> i64 { b - a }

fn main() -> i32 with IO, Mut, Panic, Div {
    var buf = SortBuf { data: [0; 16] }
    buf.data[0] = 3
    buf.data[1] = 1
    buf.data[2] = 4
    buf.data[3] = 1
    buf.data[4] = 5

    sort_by(&!buf, 5, cmp_asc)
    assert(buf.data[0] == 1)
    assert(buf.data[1] == 1)
    assert(buf.data[2] == 3)
    assert(buf.data[3] == 4)
    assert(buf.data[4] == 5)

    sort_by(&!buf, 5, cmp_desc)
    assert(buf.data[0] == 5)
    assert(buf.data[1] == 4)
    assert(buf.data[2] == 3)
    assert(buf.data[3] == 1)
    assert(buf.data[4] == 1)

    println("sort by comparator OK")
    0
}
```

Note the `SortBuf` struct wrapper -- this ensures `&!` mutations propagate correctly through the JIT. The comparator `cmp: fn(i64, i64) -> i64` follows the C convention: negative means less-than, positive means greater-than.

## 8. Filter: Collecting Matching Elements

Filtering requires a mutable output index since the result may be shorter than the input.

```sio
struct FilterResult {
    data: [i64; 16],
    len:  i64,
}

fn is_even(x: i64) -> i64 with Div, Panic { if x % 2 == 0 { 1 } else { 0 } }
fn is_positive(x: i64) -> i64 { if x > 0 { 1 } else { 0 } }

fn filter(arr: [i64; 8], n: i64, pred: fn(i64) -> i64) -> FilterResult with Mut, Panic, Div {
    var result = FilterResult { data: [0; 16], len: 0 }
    var i: i64 = 0
    while i < n {
        if pred(arr[i]) != 0 {
            result.data[result.len] = arr[i]
            result.len = result.len + 1
        }
        i = i + 1
    }
    result
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let data: [i64; 8] = [1, 2, 3, 4, 5, 6, 7, 8]

    let evens = filter(data, 8, is_even)
    assert(evens.len == 4)
    assert(evens.data[0] == 2)
    assert(evens.data[1] == 4)
    assert(evens.data[2] == 6)
    assert(evens.data[3] == 8)

    println("filter OK")
    0
}
```

The `FilterResult` struct wraps the output array and its length, since filtered results have variable length.

## 9. What About Closures?

Sounio's compiler supports function references through `IrLoadFnRef` and `IrCallIndirect` opcodes. Full closure literals (`|x| x + 1`) are architecturally possible but currently blocked pending a compiler rebuild. The workaround is always the same: define a named function.

```sio
// BLOCKED -- does not compile
// let f = |x| x + 1

// WORKS -- named function reference
fn add_one(x: i64) -> i64 { x + 1 }
let f = add_one
```

When closures need to capture state, use a struct parameter instead:

```sio
struct Config { threshold: i64 }

fn above_threshold(cfg: &Config, x: i64) -> i64 {
    if x > cfg.threshold { 1 } else { 0 }
}

// Pass config explicitly rather than closing over it
fn count_above(arr: [i64; 4], cfg: &Config) -> i64 with Mut, Panic {
    var count: i64 = 0
    var i: i64 = 0
    while i < 4 {
        if above_threshold(cfg, arr[i]) != 0 {
            count = count + 1
        }
        i = i + 1
    }
    count
}
```

This is the standard pattern: pass captured state as an explicit parameter.

## Summary

| Pattern | Function Type | Example |
|---------|--------------|---------|
| Apply | `fn(T) -> U` | `apply(square, 5)` |
| Map | `fn(T) -> T` | `map4(data, dbl)` |
| Fold | `fn(T, T) -> T` | `fold4(data, 0, add)` |
| Any | `fn(T) -> i64` | `any4(data, is_positive)` |
| All | `fn(T) -> i64` | `all4(data, is_positive)` |
| Sort | `fn(T, T) -> i64` | `sort_by(&!buf, n, cmp_asc)` |
| Filter | `fn(T) -> i64` | `filter(data, n, is_even)` |
| Select | `fn(i64) -> fn(T)->T` | `select_op(0)` |

All patterns use named function references. No closures, no dynamic allocation, no hidden state -- just explicit function pointers with effect-tracked types.
