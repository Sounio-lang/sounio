# Automatic Differentiation API Reference

The `autodiff` module provides automatic differentiation (AD) capabilities for computing exact derivatives of functions. Sounio supports both forward-mode (dual numbers) and reverse-mode (tape-based) AD.

## Overview

| Mode | Method | Best For | Complexity |
|------|--------|----------|------------|
| Forward | Dual numbers | Few inputs, many outputs | O(n) per input |
| Reverse | Wengert tape | Many inputs, few outputs | O(1) per output |

**Reference:** Baydin et al., "Automatic Differentiation in Machine Learning: a Survey", JMLR 2018

---

## Forward-Mode AD (Dual Numbers)

Forward-mode AD computes derivatives by propagating tangent values alongside primal values using dual numbers.

### Theory

A dual number has the form:
```
x + epsilon * x'
```

where epsilon^2 = 0. This algebraic property enables automatic derivative computation:

```
f(x + epsilon) = f(x) + epsilon * f'(x)
```

### Type Definition

```sio
/// Dual number: val + dot*epsilon where epsilon^2 = 0
struct Dual {
    val: f64,  // Primal (function value)
    dot: f64   // Tangent (derivative)
}
```

### Constructors

#### `dual_const`

Create a constant (derivative = 0).

```sio
fn dual_const(val: f64) -> Dual
```

**Returns:** `Dual { val: val, dot: 0.0 }`

**Use:** For fixed parameters that don't vary.

#### `dual_var`

Create a variable (derivative = 1).

```sio
fn dual_var(val: f64) -> Dual
```

**Returns:** `Dual { val: val, dot: 1.0 }`

**Use:** Mark the input you're differentiating with respect to.

#### `dual_new`

Create with explicit derivative.

```sio
fn dual_new(val: f64, dot: f64) -> Dual
```

### Arithmetic Operations

#### `dual_add`

Addition of dual numbers.

```sio
fn dual_add(a: Dual, b: Dual) -> Dual
```

**Derivative rule:** `(f + g)' = f' + g'`

**Example:**
```sio
let a = dual_var(2.0)    // d/da = 1
let b = dual_const(3.0)  // d/da = 0
let c = dual_add(a, b)   // c.val = 5.0, c.dot = 1.0
```

#### `dual_sub`

Subtraction.

```sio
fn dual_sub(a: Dual, b: Dual) -> Dual
```

**Derivative rule:** `(f - g)' = f' - g'`

#### `dual_mul`

Multiplication.

```sio
fn dual_mul(a: Dual, b: Dual) -> Dual
```

**Derivative rule (product rule):** `(f * g)' = f' * g + f * g'`

**Implementation:**
```sio
return Dual {
    val: a.val * b.val,
    dot: a.dot * b.val + a.val * b.dot
}
```

#### `dual_div`

Division.

```sio
fn dual_div(a: Dual, b: Dual) -> Dual
```

**Derivative rule (quotient rule):** `(f/g)' = (f'*g - f*g') / g^2`

#### `dual_neg`

Negation.

```sio
fn dual_neg(a: Dual) -> Dual
```

#### `dual_scale`

Scalar multiplication.

```sio
fn dual_scale(a: Dual, s: f64) -> Dual
```

### Mathematical Functions

#### `dual_sqrt`

Square root.

```sio
fn dual_sqrt(a: Dual) -> Dual
```

**Derivative:** `d/dx sqrt(x) = 1 / (2 * sqrt(x))`

#### `dual_exp`

Exponential.

```sio
fn dual_exp(a: Dual) -> Dual
```

**Derivative:** `d/dx exp(x) = exp(x)`

#### `dual_ln`

Natural logarithm.

```sio
fn dual_ln(a: Dual) -> Dual
```

**Derivative:** `d/dx ln(x) = 1/x`

#### `dual_pow`

Power function.

```sio
fn dual_pow(a: Dual, n: f64) -> Dual
```

**Derivative:** `d/dx x^n = n * x^(n-1)`

#### `dual_sin`

Sine.

```sio
fn dual_sin(a: Dual) -> Dual
```

**Derivative:** `d/dx sin(x) = cos(x)`

#### `dual_cos`

Cosine.

```sio
fn dual_cos(a: Dual) -> Dual
```

**Derivative:** `d/dx cos(x) = -sin(x)`

#### `dual_tan`

Tangent (not in base implementation, can be derived).

```sio
fn dual_tan(a: Dual) -> Dual {
    return dual_div(dual_sin(a), dual_cos(a))
}
```

#### `dual_tanh`

Hyperbolic tangent.

```sio
fn dual_tanh(a: Dual) -> Dual
```

**Derivative:** `d/dx tanh(x) = 1 - tanh(x)^2`

#### `dual_sigmoid`

Sigmoid (logistic) function.

```sio
fn dual_sigmoid(a: Dual) -> Dual
```

**Formula:** `sigma(x) = 1 / (1 + exp(-x))`
**Derivative:** `sigma'(x) = sigma(x) * (1 - sigma(x))`

#### `dual_relu`

Rectified Linear Unit.

```sio
fn dual_relu(a: Dual) -> Dual
```

**Formula:** `relu(x) = max(0, x)`
**Derivative:** `relu'(x) = 1 if x > 0, else 0`

---

## Gradient Computation

### Result Types

#### `Grad2`

Gradient result for 2D input.

```sio
struct Grad2 {
    fx: f64,   // Function value
    dx: f64,   // df/dx
    dy: f64    // df/dy
}
```

#### `Grad3`

Gradient result for 3D input.

```sio
struct Grad3 {
    fx: f64,   // Function value
    dx: f64,   // df/dx
    dy: f64,   // df/dy
    dz: f64    // df/dz
}
```

#### `Grad4`

Gradient result for 4D input.

```sio
struct Grad4 {
    fx: f64,
    d1: f64,
    d2: f64,
    d3: f64,
    d4: f64
}
```

### Computing Gradients

To compute a gradient, evaluate the function once per input variable, setting each as the active variable in turn.

**Example: Gradient of Rosenbrock function**

```sio
/// Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
fn rosenbrock_dual(x: Dual, y: Dual) -> Dual {
    let one = dual_const(1.0)
    let hundred = dual_const(100.0)
    let term1 = dual_sub(one, x)
    let term1_sq = dual_mul(term1, term1)
    let x_sq = dual_mul(x, x)
    let term2 = dual_sub(y, x_sq)
    let term2_sq = dual_mul(term2, term2)
    return dual_add(term1_sq, dual_mul(hundred, term2_sq))
}

/// Compute gradient at (x, y)
fn grad_rosenbrock(x: f64, y: f64) -> Grad2 {
    // Compute df/dx by setting x as variable
    let fx_dx = rosenbrock_dual(dual_var(x), dual_const(y))

    // Compute df/dy by setting y as variable
    let fx_dy = rosenbrock_dual(dual_const(x), dual_var(y))

    return Grad2 {
        fx: fx_dx.val,
        dx: fx_dx.dot,
        dy: fx_dy.dot
    }
}
```

**Usage:**
```sio
let g = grad_rosenbrock(1.0, 1.0)
// At the minimum (1,1): f=0, grad=(0,0)
println("f(1,1) = ", g.fx)    // 0.0
println("df/dx = ", g.dx)     // 0.0
println("df/dy = ", g.dy)     // 0.0
```

---

## Jacobian Computation

### Result Types

#### `Jacobian2x2`

Jacobian for f: R^2 -> R^2.

```sio
struct Jacobian2x2 {
    df1_dx: f64,
    df1_dy: f64,
    df2_dx: f64,
    df2_dy: f64
}
```

#### `Jacobian3x3`

Jacobian for f: R^3 -> R^3.

```sio
struct Jacobian3x3 {
    j11: f64, j12: f64, j13: f64,
    j21: f64, j22: f64, j23: f64,
    j31: f64, j32: f64, j33: f64
}
```

### Computing Jacobians

**Example: Polar to Cartesian transformation**

```sio
/// f1(r, theta) = r * cos(theta)
fn polar_to_cart_f1(r: Dual, theta: Dual) -> Dual {
    return dual_mul(r, dual_cos(theta))
}

/// f2(r, theta) = r * sin(theta)
fn polar_to_cart_f2(r: Dual, theta: Dual) -> Dual {
    return dual_mul(r, dual_sin(theta))
}

/// Compute Jacobian of polar to Cartesian
fn jacobian_polar_to_cart(r: f64, theta: f64) -> Jacobian2x2 {
    // df1/dr
    let f1_dr = polar_to_cart_f1(dual_var(r), dual_const(theta))
    // df1/dtheta
    let f1_dtheta = polar_to_cart_f1(dual_const(r), dual_var(theta))
    // df2/dr
    let f2_dr = polar_to_cart_f2(dual_var(r), dual_const(theta))
    // df2/dtheta
    let f2_dtheta = polar_to_cart_f2(dual_const(r), dual_var(theta))

    return Jacobian2x2 {
        df1_dx: f1_dr.dot,
        df1_dy: f1_dtheta.dot,
        df2_dx: f2_dr.dot,
        df2_dy: f2_dtheta.dot
    }
}
```

**Expected result for (r=2, theta=pi/4):**
```
J = [ cos(pi/4)  -r*sin(pi/4) ]  =  [ 0.707  -1.414 ]
    [ sin(pi/4)   r*cos(pi/4) ]     [ 0.707   1.414 ]
```

---

## Hessian Computation

### Diagonal Hessian

For second derivatives, use finite differences on the gradient.

```sio
struct HessianDiag2 {
    d2x: f64,  // d^2f/dx^2
    d2y: f64   // d^2f/dy^2
}

/// Compute Hessian diagonal using finite differences
fn hessian_diag_rosenbrock(x: f64, y: f64) -> HessianDiag2 {
    let h = 0.00001

    // d^2f/dx^2 = (df/dx(x+h) - df/dx(x-h)) / (2h)
    let g_plus_x = grad_rosenbrock(x + h, y)
    let g_minus_x = grad_rosenbrock(x - h, y)
    let d2x = (g_plus_x.dx - g_minus_x.dx) / (2.0 * h)

    // d^2f/dy^2 = (df/dy(y+h) - df/dy(y-h)) / (2h)
    let g_plus_y = grad_rosenbrock(x, y + h)
    let g_minus_y = grad_rosenbrock(x, y - h)
    let d2y = (g_plus_y.dy - g_minus_y.dy) / (2.0 * h)

    return HessianDiag2 { d2x: d2x, d2y: d2y }
}
```

---

## Gradient Descent Example

```sio
struct GDState {
    x: f64,
    y: f64,
    fx: f64,
    iter: i64
}

/// Gradient descent on Rosenbrock function
fn gradient_descent_rosenbrock(x0: f64, y0: f64, lr: f64, max_iter: i64) -> GDState {
    var x = x0
    var y = y0
    var i: i64 = 0

    while i < max_iter {
        let g = grad_rosenbrock(x, y)

        // Check convergence
        let grad_norm = sqrt(g.dx * g.dx + g.dy * g.dy)
        if grad_norm < 0.000001 {
            return GDState { x: x, y: y, fx: g.fx, iter: i }
        }

        // Update
        x = x - lr * g.dx
        y = y - lr * g.dy
        i = i + 1
    }

    let g_final = grad_rosenbrock(x, y)
    return GDState { x: x, y: y, fx: g_final.fx, iter: max_iter }
}
```

**Usage:**
```sio
let result = gradient_descent_rosenbrock(-1.0, 1.0, 0.001, 10000)
println("Minimum at: (", result.x, ", ", result.y, ")")
println("f(x,y) = ", result.fx)
println("Iterations: ", result.iter)
```

---

## Reverse-Mode AD (Tape-Based)

For functions with many inputs and few outputs, reverse-mode is more efficient.

### Theory

Reverse-mode AD records operations in a computational graph (tape), then propagates adjoints backward from outputs to inputs.

### Type Definitions

```sio
/// Node in the computational graph
struct TapeNode {
    value: f64,    // Primal value
    adjoint: f64,  // Accumulated gradient
    op: i64,       // Operation code
    parent1: i64,  // First input node
    parent2: i64,  // Second input node
    aux: f64       // Auxiliary data (e.g., constant)
}

/// Computational tape (fixed size)
struct Tape {
    nodes: [TapeNode; 20],  // Pre-allocated nodes
    count: i64              // Current node count
}

/// Variable handle
struct Var {
    idx: i64,   // Index in tape
    val: f64    // Value
}
```

### Operations

```sio
/// Create new tape
fn new_tape() -> Tape

/// Create input variable
fn tape_new_var(tape: &!Tape, val: f64) -> Var

/// Create constant
fn tape_const(tape: &!Tape, val: f64) -> Var

/// Arithmetic
fn tape_add(tape: &!Tape, a: Var, b: Var) -> Var
fn tape_mul(tape: &!Tape, a: Var, b: Var) -> Var
fn tape_sub(tape: &!Tape, a: Var, b: Var) -> Var
fn tape_div(tape: &!Tape, a: Var, b: Var) -> Var
fn tape_neg(tape: &!Tape, a: Var) -> Var
fn tape_scale(tape: &!Tape, a: Var, s: f64) -> Var

/// Math functions
fn tape_exp(tape: &!Tape, a: Var) -> Var
fn tape_ln(tape: &!Tape, a: Var) -> Var
fn tape_sin(tape: &!Tape, a: Var) -> Var
fn tape_cos(tape: &!Tape, a: Var) -> Var
fn tape_sqrt(tape: &!Tape, a: Var) -> Var
fn tape_pow(tape: &!Tape, a: Var, n: f64) -> Var
```

### Backward Pass

```sio
/// Compute gradients by backpropagation
fn backward(tape: &!Tape, output: Var)

/// Get gradient of output with respect to a variable
fn get_grad(tape: &Tape, v: Var) -> f64
```

### Example

```sio
// f(x, y) = x^2 + x*y + y^2
var tape = new_tape()
let x = tape_new_var(&!tape, 3.0)
let y = tape_new_var(&!tape, 4.0)

let x_sq = tape_mul(&!tape, x, x)
let xy = tape_mul(&!tape, x, y)
let y_sq = tape_mul(&!tape, y, y)
let sum1 = tape_add(&!tape, x_sq, xy)
let f = tape_add(&!tape, sum1, y_sq)

// Backward pass
backward(&!tape, f)

// Get gradients
let df_dx = get_grad(&tape, x)  // 2x + y = 10
let df_dy = get_grad(&tape, y)  // x + 2y = 11
```

---

## Choosing Forward vs Reverse Mode

| Criterion | Forward (Dual) | Reverse (Tape) |
|-----------|----------------|----------------|
| n inputs, 1 output | O(n) | O(1) |
| 1 input, m outputs | O(1) | O(m) |
| Memory | O(1) | O(operations) |
| Implementation | Simple | Complex |
| Jacobians | Column by column | Row by row |

**Guidelines:**
- **Forward**: Gradients, small Jacobians, few inputs
- **Reverse**: Neural networks, many parameters, loss functions

---

## Complete Example: Neural Network Loss

```sio
/// Neural network loss: L = (sigmoid(w*x + b) - target)^2
fn nn_loss_dual(w: Dual, x: Dual, b: Dual, target: f64) -> Dual {
    let wx = dual_mul(w, x)
    let wxb = dual_add(wx, b)
    let pred = dual_sigmoid(wxb)
    let t = dual_const(target)
    let diff = dual_sub(pred, t)
    return dual_mul(diff, diff)
}

/// Compute gradient w.r.t. (w, b)
fn grad_nn_loss(w: f64, x: f64, b: f64, target: f64) -> Grad2 {
    let L_dw = nn_loss_dual(dual_var(w), dual_const(x), dual_const(b), target)
    let L_db = nn_loss_dual(dual_const(w), dual_const(x), dual_var(b), target)

    return Grad2 {
        fx: L_dw.val,
        dx: L_dw.dot,  // dL/dw
        dy: L_db.dot   // dL/db
    }
}
```

---

## Performance Notes

1. **Memory**: Dual numbers add 2x storage; tape adds O(n) per operation
2. **Precision**: Exact derivatives (to machine precision)
3. **Composability**: AD functions compose naturally
4. **Higher derivatives**: Forward mode can be nested

---

## See Also

- [Optimization](../optimization/index.md)
- [ODE Solvers](../ode/solvers.md) (uses AD for Jacobians)
- [Linear Algebra](../linalg/matrices.md)
