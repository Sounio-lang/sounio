<!-- docs:meta
topic_id: repo.docs.cookbook
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.cookbook
-->

> **Status**: Production | **Last validated**: 2026-04-12 | **Source**: `tests/run-pass/`

# Sounio Cookbook

Task-oriented recipes. Every example is derived from or verified against `tests/run-pass/`.

---

## Hello World

```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

---

## Variables

```sio
let x = 42
var y = 10
y = y + 1
let pi: f64 = 3.14159
```

---

## Mutable References

```sio
fn increment(x: &!i32) with Mut { *x = *x + 1 }

var counter: i32 = 0
increment(&!counter)
```

---

## Error Codes (instead of Result/Option)

```sio
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { (0.0, 1) }
    else { (a / b, 0) }
}

let (result, err) = safe_divide(10.0, 0.0)
```

---

## Fixed-Size Arrays

```sio
var buffer: [u8; 256] = [0; 256]
let data: [i64; 4] = [1, 2, 3, 4]
let first = data[0]
```

---

## Mutable Array References

Source: `tests/run-pass/array_mut_ref.sio`

```sio
fn fill(arr: &![i64; 8]) with Mut, Panic {
    (*arr)[0] = 99
    (*arr)[1] = 42
}

fn main() -> i64 with IO, Mut, Panic {
    var buf: [i64; 8] = [0, 0, 0, 0, 0, 0, 0, 0]
    fill(&! buf)
    if buf[0] == 99 {
        return 0
    }
    return 1
}
```

---

## Structs and Methods

Source: `tests/run-pass/impl_inherent_method.sio`

```sio
struct Point { x: f64, y: f64 }

impl Point {
    fn get_x(self: Point) -> f64 { self.x }
    fn get_y(self: Point) -> f64 { self.y }
}

fn main() -> i64 with IO {
    let p = Point { x: 3.0, y: 4.0 }
    let px = p.get_x()
    println(px)
    0
}
```

---

## Struct Mutation (var + reassign)

Source: `tests/run-pass/while_struct_mutation_minimal.sio`

```sio
struct State { x: f64 }

var s = State { x: 0.0 }
s = State { x: s.x + 1.0 }
```

---

## Enums and Pattern Matching

Source: `tests/run-pass/native_enum_basic.sio`

```sio
enum Color { Red, Green, Blue }

fn value(c: Color) -> i64 {
    match c {
        Color::Red => 10
        Color::Green => 20
        _ => 0
    }
}
```

---

## For-In Loops

Source: `tests/run-pass/for_in_loops.sio`

```sio
for i in 0..5 { /* 0, 1, 2, 3, 4 */ }
for i in 0..=5 { /* 0, 1, 2, 3, 4, 5 */ }

var arr: [i64; 4] = [10, 20, 30, 40]
for x in arr {
    println(x)
}
```

---

## While Loops

```sio
var i = 0
while i < 10 {
    println(i)
    i = i + 1
}
```

---

## Higher-Order Functions (map, fold)

Source: `tests/run-pass/closure_higher_order.sio`

```sio
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

fn square(x: i64) -> i64 { x * x }
fn add(a: i64, b: i64) -> i64 { a + b }

let data: [i64; 4] = [1, 2, 3, 4]
let doubled = map4(data, square)
let sum = fold4(doubled, 0, add)
```

---

## Named Function References

Source: `tests/run-pass/closure_fn_ref.sio`

```sio
fn square(x: i64) -> i64 { x * x }

let f = square
let r = f(7)  // 49

fn apply(f: fn(i64) -> i64, x: i64) -> i64 { f(x) }
let r2 = apply(square, 5)  // 25
```

---

## Tuple Destructuring

Source: `tests/run-pass/tuple_destructure_let.sio`

```sio
let (a, b) = (1, 2)
let (x, (y, z)) = (10, (20, 30))
let (first, _) = (5, 10)
```

---

## Imports

Source: `tests/run-pass/import_basic_main.sio`

```sio
use my_module::{imported_add}

fn main() with IO {
    let result = imported_add(3, 4)
    println(result)
}
```

---

## Custom Effects

Source: `tests/run-pass/effect_handler_basic.sio`

```sio
effect Choice {
    fn pick() -> bool
}

fn coin_flip() with Choice {
    // uses Choice effect
}

fn main() with Choice {
    coin_flip()
}
```

---

## Sort with Comparator

Source: `tests/run-pass/closure_sort_by.sio`

```sio
fn cmp_asc(a: i64, b: i64) -> i64 {
    if a < b { 0 - 1 } else if a > b { 1 } else { 0 }
}

fn bubble_sort(arr: &![i64; 8], n: i64, cmp: fn(i64, i64) -> i64) with Mut, Panic {
    var swapped = 1
    while swapped == 1 {
        swapped = 0
        var i: i64 = 1
        while i < n {
            if cmp(arr[i], arr[i - 1]) < 0 {
                let tmp = arr[i]
                (*arr)[i] = arr[i - 1]
                (*arr)[i - 1] = tmp
                swapped = 1
            }
            i = i + 1
        }
    }
}
```

---

## String Concatenation

Source: `tests/run-pass/wave_d_concat.sio`

```sio
let greeting = "Hello" ++ ", " ++ "World!"
```

---

## Units of Measure

Source: `tests/run-pass/unit_decl_keyword.sio`

```sio
unit dosage

fn main() with IO {
    let x: dosage = 42.0
    let y: dosage = x * 2.0
    println(y)
}
```

---

## GPU Kernel

Source: `tests/run-pass/gpu_vec_add.sio`

```sio
kernel fn vec_add(a: &[f64], b: &[f64], out: &![f64], n: i32) {
    let idx = gpu_thread_id_x()
    if idx < n {
        out[idx] = a[idx] + b[idx]
    }
}
```

---

## Epistemic Values with GUM Propagation

Source: `tests/run-pass/vancomycin_propagation.sio`

```sio
struct KnowledgeF64 {
    value: f64,
    uncertainty: f64,
    confidence: f64,
}

fn gum_mul(a: KnowledgeF64, b: KnowledgeF64) -> KnowledgeF64 {
    let val = a.value * b.value
    let unc = val * (((a.uncertainty / a.value) * (a.uncertainty / a.value)) + ((b.uncertainty / b.value) * (b.uncertainty / b.uncertainty))) * 0.5
    KnowledgeF64 { value: val, uncertainty: unc, confidence: 0.0 }
}
```

---

## Linear Types

Source: `tests/run-pass/closure_linear.sio`

```sio
linear struct FileHandle { fd: i32 }

fn open_file(path: &str) -> FileHandle with IO {
    FileHandle { fd: 0 }
}

fn close_file(h: FileHandle) with IO {
    // consumes h — cannot be used again
}
```

---

## Trait and Impl

Source: `tests/run-pass/trait_basic.sio`

```sio
trait Shape {
    fn area(self: Shape) -> f64
}

struct Circle { radius: f64 }

impl Shape for Circle {
    fn area(self: Shape) -> f64 {
        3.14159 * self.radius * self.radius
    }
}

let c = Circle { radius: 5.0 }
let a = c.area()
```

---

## Nested Loops (Matrix Iteration)

```sio
var matrix: [f64; 9] = [0.0; 9]
let rows = 3
let cols = 3
var i = 0
while i < rows {
    var j = 0
    while j < cols {
        matrix[i * cols + j] = (i + j) as f64
        j = j + 1
    }
    i = i + 1
}
```

---

## Effect Combinations

```sio
fn process(arr: &![i64; 8], n: i64) with IO, Mut, Panic, Div {
    var i: i64 = 0
    while i < n {
        let val = arr[i]
        if val > 100 {
            (*arr)[i] = val / 2
            println(arr[i])
        }
        i = i + 1
    }
}
```
