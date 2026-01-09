# Automatic Differentiation

Automatic differentiation (AD) computes exact derivatives of functions by systematically applying the chain rule through program execution. Unlike numerical differentiation (finite differences), AD produces machine-precision derivatives without truncation error.

**Reference:** Baydin et al. (2018), "Automatic Differentiation in Machine Learning: a Survey", JMLR

## Module Overview

| Module | Mode | Best For | Memory |
|--------|------|----------|--------|
| `autodiff::dual` | Forward | Few inputs, many outputs | O(n) |
| `autodiff::tape` | Reverse | Many inputs, few outputs | O(ops) |

## Forward Mode (Dual Numbers)

Forward-mode AD uses **dual numbers**: extensions of real numbers with an infinitesimal component.

### Mathematical Foundation

A dual number has the form `a + b*epsilon` where `epsilon^2 = 0`.

For any analytic function f:
```
f(a + b*epsilon) = f(a) + b*f'(a)*epsilon
```

This property enables derivative computation through arithmetic:

| Operation | Dual Result |
|-----------|-------------|
| (a + b*eps) + (c + d*eps) | (a+c) + (b+d)*eps |
| (a + b*eps) * (c + d*eps) | a*c + (b*c + a*d)*eps |
| sin(a + b*eps) | sin(a) + b*cos(a)*eps |
| exp(a + b*eps) | exp(a) + b*exp(a)*eps |

### Dual Number Type

```sio
use autodiff::dual::*

// Dual number: val + dot*epsilon
struct Dual {
    val: f64,  // Primal value (function value)
    dot: f64   // Tangent (derivative)
}

// Create a constant (derivative = 0)
fn dual_const(val: f64) -> Dual {
    return Dual { val: val, dot: 0.0 }
}

// Create a variable (derivative = 1)
fn dual_var(val: f64) -> Dual {
    return Dual { val: val, dot: 1.0 }
}

// Create with explicit derivative
fn dual_new(val: f64, dot: f64) -> Dual {
    return Dual { val: val, dot: dot }
}
```

### Arithmetic Operations

```sio
use autodiff::dual::*

fn main() -> i32 {
    let x = dual_var(3.0)    // x = 3, dx/dx = 1
    let c = dual_const(2.0)  // c = 2, dc/dx = 0

    // Addition: (f + g)' = f' + g'
    let sum = dual_add(x, c)  // val=5, dot=1

    // Subtraction: (f - g)' = f' - g'
    let diff = dual_sub(x, c)  // val=1, dot=1

    // Multiplication (product rule): (f*g)' = f'*g + f*g'
    let prod = dual_mul(x, c)  // val=6, dot=2

    // Division (quotient rule): (f/g)' = (f'*g - f*g') / g^2
    let quot = dual_div(x, c)  // val=1.5, dot=0.5

    // Negation
    let neg = dual_neg(x)  // val=-3, dot=-1

    // Scale by constant
    let scaled = dual_scale(x, 5.0)  // val=15, dot=5

    return 0
}
```

### Mathematical Functions

```sio
use autodiff::dual::*

fn main() -> i32 {
    let x = dual_var(1.0)

    // Square root: d/dx sqrt(x) = 1/(2*sqrt(x))
    let sqrt_x = dual_sqrt(x)

    // Exponential: d/dx exp(x) = exp(x)
    let exp_x = dual_exp(x)

    // Natural log: d/dx ln(x) = 1/x
    let ln_x = dual_ln(x)

    // Power: d/dx x^n = n * x^(n-1)
    let pow_x = dual_pow(x, 3.0)

    // Trigonometric
    let sin_x = dual_sin(x)   // d/dx sin(x) = cos(x)
    let cos_x = dual_cos(x)   // d/dx cos(x) = -sin(x)
    let tan_x = dual_tan(x)   // d/dx tan(x) = sec^2(x)

    // Hyperbolic
    let tanh_x = dual_tanh(x)  // d/dx tanh(x) = 1 - tanh^2(x)

    // Neural network activations
    let sigmoid_x = dual_sigmoid(x)  // d/dx sigmoid(x) = sigmoid(x)*(1-sigmoid(x))
    let relu_x = dual_relu(x)        // d/dx relu(x) = 1 if x > 0, else 0

    // Absolute value
    let abs_x = dual_abs(x)  // d/dx |x| = sign(x)

    return 0
}
```

### Computing Derivatives

**Single derivative:**

```sio
use autodiff::dual::*

// f(x) = x^2 + sin(x)
fn f(x: Dual) -> Dual {
    let x_sq = dual_mul(x, x)
    let sin_x = dual_sin(x)
    return dual_add(x_sq, sin_x)
}

fn main() -> i32 {
    // Compute f(2) and f'(2)
    let x = dual_var(2.0)
    let result = f(x)

    println("f(2) = ", result.val)    // 4 + sin(2) = 4.909
    println("f'(2) = ", result.dot)   // 2*2 + cos(2) = 3.584

    return 0
}
```

**Chain rule automatically applied:**

```sio
use autodiff::dual::*

// f(x) = exp(x^2)
// f'(x) = 2x * exp(x^2)
fn f(x: Dual) -> Dual {
    return dual_exp(dual_mul(x, x))
}

fn main() -> i32 {
    let x = dual_var(1.0)
    let result = f(x)

    println("f(1) = ", result.val)    // e^1 = 2.718
    println("f'(1) = ", result.dot)   // 2*1*e^1 = 5.436

    return 0
}
```

### Gradients (Multi-Variable Functions)

For functions f: R^n -> R, compute the gradient by running forward mode n times:

```sio
use autodiff::dual::*

// DualVec3 for 3D gradient computation
struct DualVec3 {
    x: Dual,
    y: Dual,
    z: Dual
}

// Create with i-th component as the variable
fn dualvec3_var_i(x: f64, y: f64, z: f64, i: i64) -> DualVec3 {
    if i == 0 {
        return DualVec3 { x: dual_var(x), y: dual_const(y), z: dual_const(z) }
    }
    if i == 1 {
        return DualVec3 { x: dual_const(x), y: dual_var(y), z: dual_const(z) }
    }
    return DualVec3 { x: dual_const(x), y: dual_const(y), z: dual_var(z) }
}

// f(x,y,z) = x^2 + x*y + z^3
fn f(v: DualVec3) -> Dual {
    let x_sq = dual_mul(v.x, v.x)       // x^2
    let xy = dual_mul(v.x, v.y)         // x*y
    let z_cubed = dual_pow(v.z, 3.0)    // z^3
    return dual_add(dual_add(x_sq, xy), z_cubed)
}

fn main() -> i32 {
    let point = (1.0, 2.0, 3.0)

    // df/dx at (1,2,3)
    let v_x = dualvec3_var_i(1.0, 2.0, 3.0, 0)
    let fx = f(v_x)
    println("df/dx = ", fx.dot)  // 2x + y = 2 + 2 = 4

    // df/dy at (1,2,3)
    let v_y = dualvec3_var_i(1.0, 2.0, 3.0, 1)
    let fy = f(v_y)
    println("df/dy = ", fy.dot)  // x = 1

    // df/dz at (1,2,3)
    let v_z = dualvec3_var_i(1.0, 2.0, 3.0, 2)
    let fz = f(v_z)
    println("df/dz = ", fz.dot)  // 3z^2 = 27

    // Gradient = (4, 1, 27)
    return 0
}
```

### DualVec3 Operations

```sio
use autodiff::dual::*

fn main() -> i32 {
    let a = dualvec3_new(dual_var(1.0), dual_const(2.0), dual_const(3.0))
    let b = dualvec3_const(4.0, 5.0, 6.0)

    // Vector addition
    let sum = dualvec3_add(a, b)

    // Scalar multiplication
    let scaled = dualvec3_scale(a, dual_const(2.0))

    // Dot product
    let dot = dualvec3_dot(a, b)

    // Norm
    let norm = dualvec3_norm(a)

    return 0
}
```

## Reverse Mode (Tape-Based)

Reverse-mode AD records operations on a **tape** (Wengert list), then propagates adjoints backward.

### When to Use Reverse Mode

| Scenario | Forward Mode | Reverse Mode |
|----------|--------------|--------------|
| f: R -> R^m | 1 pass | m passes |
| f: R^n -> R | n passes | 1 pass |
| f: R^n -> R^m | n passes | m passes |

For functions with many inputs and few outputs (typical in ML), reverse mode is dramatically more efficient.

### Tape Structure

```sio
use autodiff::tape::*

// Node on the computation tape
struct TapeNode {
    value: f64,        // Forward value
    adjoint: f64,      // Backward adjoint (accumulated gradient)
    op: i64,           // Operation code
    parent1: i64,      // First parent index
    parent2: i64       // Second parent index (-1 if none)
}

// Fixed-size tape (20 nodes)
struct Tape {
    nodes: [TapeNode; 20],
    len: i64
}
```

### Basic Operations

```sio
use autodiff::tape::*

fn main() -> i32 {
    // Create tape
    var tape = tape_new()

    // Create variables (inputs)
    let x = tape_var(&!tape, 2.0)  // x = 2
    let y = tape_var(&!tape, 3.0)  // y = 3

    // Build computation graph
    let x_sq = tape_mul(&!tape, x, x)     // x^2
    let xy = tape_mul(&!tape, x, y)       // x*y
    let sum = tape_add(&!tape, x_sq, xy)  // x^2 + x*y

    // Forward pass (already computed)
    println("f(2,3) = ", tape_value(&tape, sum))  // 4 + 6 = 10

    // Backward pass
    tape_backward(&!tape, sum)

    // Get gradients
    let dx = tape_grad(&tape, x)  // df/dx = 2x + y = 7
    let dy = tape_grad(&tape, y)  // df/dy = x = 2

    println("df/dx = ", dx)
    println("df/dy = ", dy)

    return 0
}
```

### Tape Operations

```sio
use autodiff::tape::*

fn main() -> i32 {
    var tape = tape_new()

    let a = tape_var(&!tape, 1.0)
    let b = tape_var(&!tape, 2.0)

    // Arithmetic
    let sum = tape_add(&!tape, a, b)
    let diff = tape_sub(&!tape, a, b)
    let prod = tape_mul(&!tape, a, b)
    let quot = tape_div(&!tape, a, b)
    let neg = tape_neg(&!tape, a)

    // Mathematical functions
    let sqrt_a = tape_sqrt(&!tape, a)
    let exp_a = tape_exp(&!tape, a)
    let ln_a = tape_ln(&!tape, a)
    let sin_a = tape_sin(&!tape, a)
    let cos_a = tape_cos(&!tape, a)

    // Scale by constant
    let scaled = tape_scale(&!tape, a, 3.0)

    // Add constant
    let shifted = tape_add_const(&!tape, a, 5.0)

    return 0
}
```

### Example: Neural Network Layer

```sio
use autodiff::tape::*

// Simple MLP layer: y = sigmoid(W*x + b)
fn mlp_layer(tape: &!Tape, W: [i64; 4], x: [i64; 2], b: [i64; 2]) -> [i64; 2] {
    // W is 2x2 weight matrix (stored as indices)
    // x is 2-element input (stored as indices)
    // b is 2-element bias (stored as indices)

    // y[0] = sigmoid(W[0,0]*x[0] + W[0,1]*x[1] + b[0])
    let wx00 = tape_mul(tape, W[0], x[0])
    let wx01 = tape_mul(tape, W[1], x[1])
    let z0 = tape_add(tape, tape_add(tape, wx00, wx01), b[0])
    let y0 = tape_sigmoid(tape, z0)

    // y[1] = sigmoid(W[1,0]*x[0] + W[1,1]*x[1] + b[1])
    let wx10 = tape_mul(tape, W[2], x[0])
    let wx11 = tape_mul(tape, W[3], x[1])
    let z1 = tape_add(tape, tape_add(tape, wx10, wx11), b[1])
    let y1 = tape_sigmoid(tape, z1)

    return [y0, y1]
}

fn main() -> i32 {
    var tape = tape_new()

    // Initialize weights
    let W: [i64; 4] = [
        tape_var(&!tape, 0.5),   // W[0,0]
        tape_var(&!tape, -0.3),  // W[0,1]
        tape_var(&!tape, 0.2),   // W[1,0]
        tape_var(&!tape, 0.8)    // W[1,1]
    ]

    // Input
    let x: [i64; 2] = [
        tape_var(&!tape, 1.0),
        tape_var(&!tape, 0.5)
    ]

    // Bias
    let b: [i64; 2] = [
        tape_var(&!tape, 0.1),
        tape_var(&!tape, -0.1)
    ]

    // Forward pass
    let y = mlp_layer(&!tape, W, x, b)

    // Loss: sum of outputs (for demo)
    let loss = tape_add(&!tape, y[0], y[1])

    println("Loss: ", tape_value(&tape, loss))

    // Backward pass
    tape_backward(&!tape, loss)

    // Get weight gradients
    println("dL/dW[0,0] = ", tape_grad(&tape, W[0]))
    println("dL/dW[0,1] = ", tape_grad(&tape, W[1]))
    println("dL/dW[1,0] = ", tape_grad(&tape, W[2]))
    println("dL/dW[1,1] = ", tape_grad(&tape, W[3]))

    return 0
}
```

## Comparison: Forward vs Reverse

### Forward Mode Advantages
- Simple implementation
- O(n) memory for n inputs
- Streaming-friendly (no tape storage)
- Good for Jacobian-vector products

### Reverse Mode Advantages
- O(1) passes for scalar output (typical loss functions)
- Efficient for high-dimensional inputs
- Standard in deep learning frameworks

### Complexity Analysis

For f: R^n -> R^m:

| Mode | Forward Passes | Backward Passes | Memory |
|------|----------------|-----------------|--------|
| Forward | n | 0 | O(n) |
| Reverse | 1 | m | O(operations) |

For neural networks with millions of parameters (n >> 1) and scalar loss (m = 1), reverse mode requires ~10^6x fewer passes.

## Higher-Order Derivatives

### Second Derivatives with Forward Mode

Apply forward mode twice using nested dual numbers:

```sio
use autodiff::dual::*

// HyperDual for second derivatives
struct HyperDual {
    val: f64,   // f(x)
    d1: f64,    // f'(x)
    d2: f64,    // f''(x) contribution
    d12: f64    // Mixed partial
}

// For f(x) = x^3, compute f, f', f''
fn hyperdual_cubic(x: f64) -> (f64, f64, f64) {
    // f(x) = x^3
    // f'(x) = 3x^2
    // f''(x) = 6x
    return (x*x*x, 3.0*x*x, 6.0*x)
}
```

### Hessian-Vector Products

For optimization (Newton's method), we often need Hv (Hessian times vector):

```sio
// Compute Hessian-vector product using forward-over-reverse
fn hessian_vector_product(
    f: fn(&[Dual]) -> Dual,
    x: &[f64],
    v: &[f64],
    hv: &![f64]
) {
    // Forward pass with dual perturbation
    // Reverse pass to get gradient of gradient
    // Result: Hv = d/dt grad(f)(x + t*v) at t=0
}
```

## Applications

### Sensitivity Analysis

```sio
use autodiff::dual::*

// How sensitive is drug concentration to absorption rate?
fn pk_sensitivity(ka: f64, ke: f64, t: f64) -> f64 {
    // C(t) = (ka/(ka-ke)) * (exp(-ke*t) - exp(-ka*t))
    // Compute dC/d(ka)

    let ka_dual = dual_var(ka)
    let ke_dual = dual_const(ke)

    let diff = dual_sub(ka_dual, ke_dual)
    let ratio = dual_div(ka_dual, diff)

    let exp_ke = dual_exp(dual_scale(ke_dual, -t))
    let exp_ka = dual_exp(dual_scale(ka_dual, -t))

    let C = dual_mul(ratio, dual_sub(exp_ke, exp_ka))

    return C.dot  // dC/d(ka)
}
```

### Optimization (Gradient Descent)

```sio
use autodiff::tape::*

fn gradient_descent(x0: f64, y0: f64, learning_rate: f64, iterations: i64) -> (f64, f64) {
    var x = x0
    var y = y0

    var i: i64 = 0
    while i < iterations {
        // Build fresh tape each iteration
        var tape = tape_new()

        let x_node = tape_var(&!tape, x)
        let y_node = tape_var(&!tape, y)

        // Loss: (x-1)^2 + (y-2)^2 (minimum at (1, 2))
        let dx = tape_add_const(&!tape, x_node, -1.0)
        let dy = tape_add_const(&!tape, y_node, -2.0)
        let loss = tape_add(&!tape,
            tape_mul(&!tape, dx, dx),
            tape_mul(&!tape, dy, dy)
        )

        // Backward pass
        tape_backward(&!tape, loss)

        // Update
        x = x - learning_rate * tape_grad(&tape, x_node)
        y = y - learning_rate * tape_grad(&tape, y_node)

        if i % 100 == 0 {
            println("Iteration ", i, ": loss = ", tape_value(&tape, loss))
        }

        i = i + 1
    }

    return (x, y)
}

fn main() -> i32 {
    let (x_opt, y_opt) = gradient_descent(0.0, 0.0, 0.1, 1000)
    println("Optimum: (", x_opt, ", ", y_opt, ")")
    // Should be close to (1, 2)

    return 0
}
```

### Jacobian Computation

```sio
use autodiff::dual::*

// f: R^2 -> R^2
// f(x, y) = (x^2 + y, x*y^2)
// Jacobian: [[2x, 1], [y^2, 2xy]]

fn compute_jacobian(x: f64, y: f64) -> [[f64; 2]; 2] {
    var J: [[f64; 2]; 2] = [[0.0; 2]; 2]

    // Column 0: df/dx
    let x_d = dual_var(x)
    let y_c = dual_const(y)
    let f0_dx = dual_add(dual_mul(x_d, x_d), y_c)  // x^2 + y
    let f1_dx = dual_mul(x_d, dual_mul(y_c, y_c))  // x*y^2
    J[0][0] = f0_dx.dot  // df0/dx = 2x
    J[1][0] = f1_dx.dot  // df1/dx = y^2

    // Column 1: df/dy
    let x_c = dual_const(x)
    let y_d = dual_var(y)
    let f0_dy = dual_add(dual_mul(x_c, x_c), y_d)  // x^2 + y
    let f1_dy = dual_mul(x_c, dual_mul(y_d, y_d))  // x*y^2
    J[0][1] = f0_dy.dot  // df0/dy = 1
    J[1][1] = f1_dy.dot  // df1/dy = 2xy

    return J
}

fn main() -> i32 {
    let J = compute_jacobian(2.0, 3.0)

    println("Jacobian at (2, 3):")
    println("  [", J[0][0], ", ", J[0][1], "]")  // [4, 1]
    println("  [", J[1][0], ", ", J[1][1], "]")  // [9, 12]

    return 0
}
```

## Best Practices

### 1. Choose the Right Mode

- **Few inputs (n < 10):** Forward mode is simpler
- **Scalar output (m = 1):** Reverse mode is optimal
- **Dense Jacobian needed:** Forward mode, n passes
- **Jacobian-vector product:** Forward mode, 1 pass
- **Vector-Jacobian product:** Reverse mode, 1 pass

### 2. Avoid Tape Overflow

The fixed-size tape (20 nodes) limits computation complexity:

```sio
// BAD: Long computation chain
var result = tape_var(&!tape, x)
var i: i64 = 0
while i < 100 {
    result = tape_mul(&!tape, result, result)  // Tape overflow!
    i = i + 1
}

// GOOD: Checkpoint or restart tape
fn power_n(x: f64, n: i64) -> (f64, f64) {
    // Compute x^n and d/dx(x^n) manually
    return (pow(x, n as f64), n as f64 * pow(x, (n-1) as f64))
}
```

### 3. Verify with Finite Differences

```sio
fn verify_gradient(f: fn(f64) -> f64, x: f64, df_dx_ad: f64) -> bool {
    let eps = 1e-7
    let df_dx_fd = (f(x + eps) - f(x - eps)) / (2.0 * eps)

    let rel_err = abs_f64(df_dx_ad - df_dx_fd) / (abs_f64(df_dx_fd) + 1e-10)

    if rel_err > 1e-5 {
        println("WARNING: AD and FD disagree")
        println("  AD: ", df_dx_ad)
        println("  FD: ", df_dx_fd)
        println("  Relative error: ", rel_err)
        return false
    }
    return true
}
```
