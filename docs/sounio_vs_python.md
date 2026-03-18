# Sounio vs Python: Side-by-Side Comparison

A practical comparison showing how common programming tasks look in Python and Sounio. Each example highlights the key differences in syntax, type systems, and safety guarantees.

---

## 1. Hello World

**Python:**
```python
print("Hello, World!")
```

**Sounio:**
```sio
fn main() with IO {
    println("Hello, Sounio!")
}
```

**Differences:**
- Sounio requires a `main()` function entry point.
- The `with IO` effect annotation declares that this function performs I/O. In Python, side effects are implicit; in Sounio, they are tracked in the type system.
- `println()` is a regular function, not a macro (no `!` suffix).

---

## 2. Fibonacci

**Python:**
```python
def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

**Sounio:**
```sio
fn fib(n: i64) -> i64 with Mut, Panic {
    if n <= 1 { return n }
    var a: i64 = 0
    var b: i64 = 1
    var i: i64 = 2
    while i <= n {
        let temp = b
        b = a + b
        a = temp
        i = i + 1
    }
    b
}
```

**Differences:**
- Sounio requires explicit types on function parameters and return values (`n: i64`, `-> i64`).
- Mutable variables use `var`, not `let` or Python's implicit mutability.
- No semicolons at the end of statements.
- The `with Mut, Panic` effect declares that this function mutates variables and may panic on bounds checks.
- No tuple destructuring assignment (`a, b = b, a + b`); use explicit temporary variables.
- No `for _ in range(...)` syntax; Sounio uses `while` loops or `for i in 2..=n`.

---

## 3. Sorting (Insertion Sort)

**Python:**
```python
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
```

**Sounio:**
```sio
struct SortBuf { data: [i64; 1024], len: i64 }

fn insertion_sort(buf: &! SortBuf) with Mut, Panic {
    var i: i64 = 1
    while i < buf.len {
        let key = buf.data[i]
        var j: i64 = i - 1
        while j >= 0 && buf.data[j] > key {
            buf.data[j + 1] = buf.data[j]
            j = j - 1
        }
        buf.data[j + 1] = key
        i = i + 1
    }
}
```

**Differences:**
- Sounio uses fixed-size arrays (`[i64; 1024]`), not dynamically-sized lists. The array is wrapped in a struct to work around a known JIT bug with bare `&!` array mutation.
- `&! SortBuf` is an exclusive (mutable) reference; Sounio uses `&!` where Rust uses `&mut`.
- No `j -= 1`; Sounio uses `j = j - 1` (no compound assignment operators).
- The `with Mut, Panic` effect is required because the function mutates through `&!` and accesses array indices.

---

## 4. Binary Search

**Python:**
```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

**Sounio:**
```sio
fn binary_search(arr: &[i64; 1024], len: i64, target: i64) -> i64 with Mut, Panic, Div {
    var lo: i64 = 0
    var hi: i64 = len - 1
    while lo <= hi {
        let mid = (lo + hi) / 2
        if arr[mid] == target { return mid }
        else if arr[mid] < target { lo = mid + 1 }
        else { hi = mid - 1 }
    }
    0 - 1
}
```

**Differences:**
- Sounio passes arrays by reference (`&[i64; 1024]`) with an explicit length parameter, since arrays are fixed-size.
- Division requires the `Div` effect (always paired with `Panic`).
- Returning `-1` is written as `0 - 1` because Sounio has no unary minus operator.
- No tuple destructuring for `lo, hi = 0, len(arr) - 1`; each variable is declared separately.

---

## 5. Matrix Multiply (3x3)

**Python:**
```python
def matmul(a, b):
    n = len(a)
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] += a[i][k] * b[k][j]
    return result
```

**Sounio:**
```sio
fn matmul3(a: &[f64; 9], b: &[f64; 9], out: &![f64; 9]) with Mut, Panic {
    var i: i64 = 0
    while i < 3 {
        var j: i64 = 0
        while j < 3 {
            var sum = 0.0
            var k: i64 = 0
            while k < 3 {
                sum = sum + a[i * 3 + k] * b[k * 3 + j]
                k = k + 1
            }
            (*out)[i * 3 + j] = sum
            j = j + 1
        }
        i = i + 1
    }
}
```

**Differences:**
- Sounio stores matrices as flat 1D arrays with manual index arithmetic (`i * 3 + j`). Python uses nested lists.
- The output is passed as `&![f64; 9]` (exclusive reference) and requires explicit dereference `(*out)[idx]`.
- No `+=` operator; use `sum = sum + ...`.
- The `with Mut, Panic` effect covers both array mutation and bounds-checked indexing.

---

## 6. Error Handling

**Python:**
```python
def safe_divide(a, b):
    try:
        return a / b, None
    except ZeroDivisionError:
        return 0.0, "division by zero"
```

**Sounio:**
```sio
fn safe_divide(a: f64, b: f64) -> (f64, i32) with Div, Panic {
    if b == 0.0 { return (0.0, 1) }
    (a / b, 0)
}

fn main() with IO, Mut, Panic, Div {
    let (result, err) = safe_divide(10.0, 3.0)
    if err != 0 {
        println("Division by zero")
    } else {
        print("Result = ")
        print(result)
        println("")
    }
}
```

**Differences:**
- Sounio has no exceptions or try/catch. Error handling is done through return values: tuples with error codes, or monomorphic result types from the stdlib.
- The caller checks the error code explicitly, making the error path visible in the code.
- The `Div` effect is required even though the function guards against zero division, because the `/` operator is present.

---

## 7. Higher-Order Functions

**Python:**
```python
data = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, data))
evens = list(filter(lambda x: x % 2 == 0, data))
total = sum(data)
```

**Sounio:**
```sio
fn double(x: i64) -> i64 { x * 2 }
fn is_positive(x: i64) -> bool { x > 0 }
fn add(a: i64, b: i64) -> i64 { a + b }

fn map4(arr: [i64; 4], f: fn(i64) -> i64) -> [i64; 4] with Mut, Panic, Div {
    var out: [i64; 4] = [0; 4]
    var i: i64 = 0
    while i < 4 { out[i] = f(arr[i]); i = i + 1 }
    out
}

fn fold4(arr: [i64; 4], init: i64, f: fn(i64, i64) -> i64) -> i64 with Mut, Panic, Div {
    var acc = init
    var i: i64 = 0
    while i < 4 { acc = f(acc, arr[i]); i = i + 1 }
    acc
}

fn main() -> i64 with IO, Mut, Panic, Div {
    let data: [i64; 4] = [1, 2, 3, 4]
    let doubled = map4(data, double)
    let total = fold4(data, 0, add)
    0
}
```

**Differences:**
- Sounio does not support lambda/closure literals (`lambda x: x * 2`). Instead, define named functions and pass them as references.
- Function types are explicit: `f: fn(i64) -> i64`.
- Higher-order functions like `map` and `fold` must be defined for specific array sizes (e.g., `map4` for arrays of length 4) because Sounio uses fixed-size arrays.
- The operations chain through named function references: `let doubled = map4(data, double)`.

---

## 8. Statistics: Mean and Variance

**Python:**
```python
import statistics

data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
mean = statistics.mean(data)
variance = statistics.variance(data)
```

**Sounio:**
```sio
fn mean(data: &[f64; 256], len: i64) -> f64 with Mut, Panic, Div {
    var sum = 0.0
    var i: i64 = 0
    while i < len {
        sum = sum + data[i]
        i = i + 1
    }
    sum / (len as f64)
}

fn variance(data: &[f64; 256], len: i64) -> f64 with Mut, Panic, Div {
    let m = mean(data, len)
    var ss = 0.0
    var i: i64 = 0
    while i < len {
        let d = data[i] - m
        ss = ss + d * d
        i = i + 1
    }
    ss / (len as f64)
}

fn main() -> i64 with IO, Mut, Panic, Div {
    var data: [f64; 256] = [0.0; 256]
    data[0] = 2.0
    data[1] = 4.0
    data[2] = 4.0
    data[3] = 4.0
    data[4] = 5.0
    data[5] = 5.0
    data[6] = 7.0
    data[7] = 9.0
    let m = mean(&data, 8)
    let v = variance(&data, 8)
    0
}
```

**Differences:**
- Python uses a standard library import (`statistics.mean`). Sounio implements these from scratch using explicit loops.
- Sounio uses fixed-size arrays with an explicit length parameter. Data is populated element-by-element.
- The `&[f64; 256]` shared reference lets the function read the array without copying.
- Effects `Div, Panic` are required for the division in mean calculation and for array indexing.
- Type casts are explicit: `len as f64`.

---

## When to Use Sounio

Sounio is designed for domains where safety, correctness, and epistemic honesty matter:

- **Scientific computing** -- Fixed-size arrays, explicit numeric types, and no hidden allocations make numeric code predictable and fast.
- **Epistemic computing** -- The `Knowledge<T>` type tracks measurement uncertainty through computations following GUM (Guide to Uncertainty in Measurement) standards. No other language has this built in.
- **Uncertainty-aware numeric code** -- Every arithmetic operation can propagate uncertainty bounds, so you know how confident your results are.
- **Pharmacokinetics and biomedical modeling** -- The stdlib includes ODE solvers, PBPK models, and the medlang DSL for clinical pharmacology.
- **Effect-safe systems** -- The effect system (`IO`, `Mut`, `Div`, `Panic`) makes side effects visible in the type system. Pure functions are guaranteed pure.
- **Dimensional analysis** -- Units of measure (`kg`, `mg`, `mL`) are checked at compile time, preventing unit mismatch errors that have caused real-world disasters.

Python excels at rapid prototyping, scripting, data exploration, web development, and leveraging its massive ecosystem. Sounio excels when you need compile-time safety guarantees, effect tracking, uncertainty propagation, or systems-level performance with scientific correctness.
