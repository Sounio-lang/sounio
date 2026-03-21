# Newton-Raphson Root Finding

## Python
```python
import math

def newton(f, x0, tol=1e-12, max_iter=100):
    x = x0
    h = 1e-8
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = (f(x + h) - f(x - h)) / (2 * h)
        if abs(dfx) < 1e-16:
            return x
        x = x - fx / dfx
    return x

# Find root of sin(x) near 3.0 → should be pi
root = newton(math.sin, 3.0)
print(f"Root: {root:.10f}")  # 3.1415926536
```

## Sounio
```sio
fn tc_abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn tc_sin(x: f64) -> f64 with Mut, Panic, Div {
    let pi = 3.14159265358979323846
    let twopi = 2.0 * pi
    var t = x
    while t > pi { t = t - twopi }
    while t < 0.0 - pi { t = t + twopi }
    var sum = t
    var term = t
    var i: i64 = 1
    while i < 12 {
        let n2 = (2 * i) as f64
        term = 0.0 - term * t * t / (n2 * (n2 + 1.0))
        sum = sum + term
        i = i + 1
    }
    sum
}

fn deriv(f: fn(f64) -> f64 with Mut, Panic, Div, x: f64) -> f64 with Mut, Panic, Div {
    let h = 0.00000001
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn newton(f: fn(f64) -> f64 with Mut, Panic, Div, x0: f64) -> f64 with Mut, Panic, Div {
    var x = x0
    var i: i64 = 0
    while i < 100 {
        let fx = f(x)
        if tc_abs(fx) < 0.000000000001 { return x }
        let dfx = deriv(f, x)
        if tc_abs(dfx) < 0.0000000000000001 { return x }
        x = x - fx / dfx
        i = i + 1
    }
    x
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let root = newton(tc_sin, 3.0)
    print("Root of sin near 3.0: ")
    print(root)
    println("")
    0
}
```

## Key Differences
- **No `import math`** — Sounio has no stdlib imports for math; write your own `sin` (Taylor series)
- **Function references**: `newton(tc_sin, 3.0)` passes named function, not lambda
- **No f-strings** — use `print()` for each value
- **No unary minus** — `0.0 - x` instead of `-x`
- **Effects propagate**: `fn(f64) -> f64 with Mut, Panic, Div` in function pointer type
- **Explicit tolerance** — `0.000000000001` instead of `1e-12` (scientific notation works in literals but spelling out is clearer)
