<!-- docs:meta
topic_id: repo.docs.tutorials.scientific-computing
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.tutorials.scientific-computing
-->

# Scientific Computing with Sounio: Numerical Methods Using First-Class Functions

Sounio's first-class function references let you write generic numerical algorithms that work on *any* function -- differentiation, integration, root-finding, and optimization -- with compile-time effect tracking that prevents silent numerical bugs.

This tutorial builds complete, runnable examples. The production implementations live in [`stdlib/functional/calculus.sio`](../../stdlib/functional/calculus.sio).

## Prerequisites

The programs below are built up inside this tutorial; there is no ready-made
example file to run. Paste a snippet into a file of your own and run it with the
prebuilt compiler:

```bash
SOUC=./bin/souc
$SOUC run my_science.sio      # a file you create from the snippets below
```

## 1. Passing Functions to Numerical Algorithms

In Sounio, named functions are first-class values. You pass them by name -- no closure syntax needed.

```sio
// Define a function you want to analyze
fn my_poly(x: f64) -> f64 {
    x * x * x - 2.0 * x - 5.0
}

// A generic numerical derivative accepts fn(f64) -> f64
fn deriv(f: fn(f64) -> f64, x: f64) -> f64 with Div, Panic {
    let h = 0.00000001
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn main() with IO, Div, Panic {
    let slope = deriv(my_poly, 1.0)
    println("slope at x=1")
}
```

The key line is `f: fn(f64) -> f64` -- this declares a parameter that accepts any function with matching signature. You call it just like a regular function: `f(x + h)`.

## 2. Numerical Differentiation

Central differences approximate derivatives without symbolic math. The stdlib provides several variants.

```sio
// Central difference: f'(x) = (f(x+h) - f(x-h)) / 2h
fn deriv(f: fn(f64) -> f64, x: f64) -> f64 with Div, Panic {
    let h = 0.00000001
    let fph = f(x + h)
    let fmh = f(x - h)
    (fph - fmh) / (2.0 * h)
}

// Second derivative: f''(x) = (f(x+h) - 2f(x) + f(x-h)) / h^2
fn deriv2(f: fn(f64) -> f64, x: f64) -> f64 with Div, Panic {
    let h = 0.0001
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

// Test function: f(x) = x^2, f'(x) = 2x, f''(x) = 2
fn x_squared(x: f64) -> f64 { x * x }

fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn main() with IO, Div, Panic {
    let d1 = deriv(x_squared, 3.0)    // expect ~6.0
    let d2 = deriv2(x_squared, 3.0)   // expect ~2.0

    assert(abs(d1 - 6.0) < 0.001)
    assert(abs(d2 - 2.0) < 0.01)
    println("derivatives OK")
}
```

Notice the effect annotations: `with Div, Panic` because we divide by `h`. The compiler enforces this -- if you forget `Div`, you get a compile error, not a silent runtime division.

## 3. Numerical Integration (Simpson's Rule)

Composite Simpson's 1/3 rule integrates any `fn(f64) -> f64` over an interval.

```sio
fn integrate(f: fn(f64) -> f64, a: f64, b: f64) -> f64 with Mut, Div, Panic {
    let n: i64 = 1000
    let h = (b - a) / (n as f64)
    var sum = f(a) + f(b)
    var i: i64 = 1
    while i < n {
        let x = a + (i as f64) * h
        if i % 2 == 0 {
            sum = sum + 2.0 * f(x)
        } else {
            sum = sum + 4.0 * f(x)
        }
        i = i + 1
    }
    sum * h / 3.0
}

// Integrate x^2 from 0 to 3: exact answer = 9.0
fn x_squared(x: f64) -> f64 { x * x }

fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn main() with IO, Mut, Div, Panic {
    let area = integrate(x_squared, 0.0, 3.0)
    assert(abs(area - 9.0) < 0.0001)
    println("integral of x^2 from 0 to 3 OK")
}
```

The `with Mut` effect is required because `var sum` is a mutable binding. Sounio tracks mutation at the type level -- pure functions cannot accidentally accumulate state.

## 4. Root Finding (Newton-Raphson)

Newton-Raphson finds zeros of a function using its derivative. Because `deriv` itself takes `fn(f64) -> f64`, Newton's method composes naturally.

```sio
fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn deriv(f: fn(f64) -> f64, x: f64) -> f64 with Mut, Div, Panic {
    let h = 0.00000001
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn newton(f: fn(f64) -> f64, x0: f64) -> f64 with Mut, Div, Panic {
    var x = x0
    var i: i64 = 0
    while i < 100 {
        let fx = f(x)
        if abs(fx) < 0.000000000001 { return x }
        let dfx = deriv(f, x)
        if abs(dfx) < 0.0000000000000001 { return x }
        x = x - fx / dfx
        i = i + 1
    }
    x
}

// Find root of x^3 - 2x - 5 = 0 (real root near 2.0946)
fn cubic(x: f64) -> f64 {
    x * x * x - 2.0 * x - 5.0
}

fn main() with IO, Mut, Div, Panic {
    let root = newton(cubic, 2.0)
    // Verify: f(root) should be ~0
    assert(abs(cubic(root)) < 0.00001)
    println("root found")
}
```

Note: `0.0 - x` instead of `-x`. Sounio has no unary minus operator.

## 5. Optimization (Golden Section Search)

Golden-section search finds the minimum of a unimodal function on an interval. Same pattern: pass any `fn(f64) -> f64`.

```sio
fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn minimize(f: fn(f64) -> f64, a: f64, b: f64) -> f64 with Mut, Div, Panic {
    let phi = 1.6180339887
    let resphi = 2.0 - phi
    var lo = a
    var hi = b
    var x1 = lo + resphi * (hi - lo)
    var x2 = hi - resphi * (hi - lo)
    var f1 = f(x1)
    var f2 = f(x2)
    var i: i64 = 0
    while i < 100 {
        if abs(hi - lo) < 0.0000000001 { return 0.5 * (lo + hi) }
        if f1 < f2 {
            hi = x2
            x2 = x1
            f2 = f1
            x1 = lo + resphi * (hi - lo)
            f1 = f(x1)
        } else {
            lo = x1
            x1 = x2
            f1 = f2
            x2 = hi - resphi * (hi - lo)
            f2 = f(x2)
        }
        i = i + 1
    }
    0.5 * (lo + hi)
}

// Minimize (x - 3)^2 on [0, 10]: minimum at x = 3
fn parabola(x: f64) -> f64 {
    (x - 3.0) * (x - 3.0)
}

fn main() with IO, Mut, Div, Panic {
    let xmin = minimize(parabola, 0.0, 10.0)
    assert(abs(xmin - 3.0) < 0.0001)
    println("minimum found at x=3")
}
```

## 6. Why Effects Make Scientific Code Safer

In most languages, a numerical function can silently perform I/O, mutate global state, or panic on division by zero. Sounio's effect system makes these properties visible in the type signature.

```sio
// This function is PURE -- no effects. The compiler guarantees it cannot
// print, mutate external state, or divide by zero.
fn quadratic(x: f64) -> f64 {
    x * x - 4.0 * x + 3.0
}

// This function MUST declare Div because it divides.
// Forgetting "with Div" is a compile error, not a runtime surprise.
fn safe_ratio(a: f64, b: f64) -> f64 with Div, Panic {
    a / b
}

// This function mutates a mutable binding (var). The Mut effect
// tells callers that internal state changes happen.
fn running_mean(values: [f64; 8], n: i64) -> f64 with Mut, Div, Panic {
    var sum: f64 = 0.0
    var i: i64 = 0
    while i < n {
        sum = sum + values[i]
        i = i + 1
    }
    sum / (n as f64)
}
```

Effect subsetting means a caller with `with IO, Mut, Div, Panic` can call any function with fewer effects. Effects propagate upward through the call stack, so `main` typically declares all effects used by its callees.

## 7. Composing Algorithms

Because every algorithm takes `fn(f64) -> f64`, you can combine them freely.

```sio
fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn deriv(f: fn(f64) -> f64, x: f64) -> f64 with Mut, Div, Panic {
    let h = 0.00000001
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn newton(f: fn(f64) -> f64, x0: f64) -> f64 with Mut, Div, Panic {
    var x = x0
    var i: i64 = 0
    while i < 100 {
        let fx = f(x)
        if abs(fx) < 0.000000000001 { return x }
        let dfx = deriv(f, x)
        if abs(dfx) < 0.0000000000000001 { return x }
        x = x - fx / dfx
        i = i + 1
    }
    x
}

fn integrate(f: fn(f64) -> f64, a: f64, b: f64) -> f64 with Mut, Div, Panic {
    let n: i64 = 1000
    let h = (b - a) / (n as f64)
    var sum = f(a) + f(b)
    var i: i64 = 1
    while i < n {
        let x = a + (i as f64) * h
        if i % 2 == 0 {
            sum = sum + 2.0 * f(x)
        } else {
            sum = sum + 4.0 * f(x)
        }
        i = i + 1
    }
    sum * h / 3.0
}

// Problem: find the root of x^3 - x - 1 = 0, then integrate x^2 from 0 to that root
fn poly(x: f64) -> f64 { x * x * x - x - 1.0 }
fn x_squared(x: f64) -> f64 { x * x }

fn main() with IO, Mut, Div, Panic {
    let root = newton(poly, 1.5)          // find where poly = 0
    let area = integrate(x_squared, 0.0, root)  // integrate up to that root
    println("composed root-finding + integration OK")
}
```

## Production Reference

The complete, tested implementations of all algorithms above live in:

- **`stdlib/functional/calculus.sio`** -- `deriv`, `deriv2`, `deriv4`, `newton`, `bisect`, `secant`, `integrate`, `integrate_n`, `trapezoid`, `minimize`, `maximize`, `fixed_point`

Import with:
```sio
use functional::calculus::{deriv, newton, integrate, minimize}
```

## Summary

| Algorithm | Function Signature | Use Case |
|-----------|-------------------|----------|
| `deriv` | `fn(fn(f64)->f64, f64) -> f64` | Slopes, sensitivities |
| `integrate` | `fn(fn(f64)->f64, f64, f64) -> f64` | Areas, cumulative quantities |
| `newton` | `fn(fn(f64)->f64, f64) -> f64` | Solving f(x) = 0 |
| `bisect` | `fn(fn(f64)->f64, f64, f64) -> f64` | Guaranteed-convergence root finding |
| `minimize` | `fn(fn(f64)->f64, f64, f64) -> f64` | Parameter optimization |

All algorithms share the pattern `fn(fn(f64)->f64, ...) -> f64` -- pass any named function reference that matches the signature.
