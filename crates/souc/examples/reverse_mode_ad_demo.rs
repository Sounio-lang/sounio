//! Reverse-Mode Automatic Differentiation Demo
//!
//! Demonstrates Task 2.2: Reverse-mode AD (backpropagation) for Sounio compiler.
//!
//! **Key Advantage**: Reverse-mode is efficient for functions with many inputs and few outputs
//! (R^n → R^m where n >> m). Perfect for neural networks and optimization problems.
//!
//! Features showcased:
//! 1. Basic backpropagation with gradient accumulation
//! 2. Transcendental function gradients
//! 3. Hyperbolic function gradients
//! 4. Complex chain rule examples
//! 5. Jacobian computation for multi-output functions
//! 6. Comparison with forward-mode AD
//! 7. Neural network-like computations
//!
//! Run with: cargo run --example reverse_mode_ad_demo

use sounio::codegen::autodiff_reverse::{TapeValue, compute_jacobian_reverse};
use sounio::codegen::autodiff_tape::Tape;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Reverse-Mode Automatic Differentiation Demo ===\n");

    demo_basic_backward();
    demo_transcendental_backward();
    demo_hyperbolic_backward();
    demo_chain_rule();
    demo_neural_network_layer();
    demo_jacobian_multi_output();
    demo_efficiency_comparison();
}

/// Demo 1: Basic backward pass examples
fn demo_basic_backward() {
    println!("--- Demo 1: Basic Backward Pass ---\n");

    // f(x, y) = x² + 2xy + y² at (3, 4)
    let tape = Rc::new(RefCell::new(Tape::new()));

    let x = TapeValue::variable(&tape, 3.0);
    let y = TapeValue::variable(&tape, 4.0);
    let two = TapeValue::constant(&tape, 2.0);

    let x_squared = x.mul(&x); // x²
    let y_squared = y.mul(&y); // y²
    let xy = x.mul(&y); // xy
    let two_xy = two.mul(&xy); // 2xy

    let result = x_squared.add(&two_xy).add(&y_squared); // x² + 2xy + y²

    println!("f(x,y) = x² + 2xy + y²");
    println!("At (x,y) = (3, 4): f = {}", result.value());

    result.backward();

    // df/dx = 2x + 2y = 2(3) + 2(4) = 14
    // df/dy = 2y + 2x = 2(4) + 2(3) = 14
    println!("∂f/∂x = {} (expected: 14 = 2x + 2y)", x.gradient());
    println!("∂f/∂y = {} (expected: 14 = 2y + 2x)", y.gradient());
    println!();
}

/// Demo 2: Transcendental function backpropagation
fn demo_transcendental_backward() {
    println!("--- Demo 2: Transcendental Function Gradients ---\n");

    // f(x) = exp(x) * sin(x) at x=0
    let tape = Rc::new(RefCell::new(Tape::new()));
    let x = TapeValue::variable(&tape, 0.0);

    let exp_x = x.exp();
    let sin_x = x.sin();
    let result = exp_x.mul(&sin_x); // exp(x) * sin(x)

    println!("f(x) = exp(x) * sin(x)");
    println!("At x=0: f(0) = {:.6}", result.value());

    result.backward();

    // df/dx = exp(x)*sin(x) + exp(x)*cos(x) = exp(x)*(sin(x) + cos(x))
    // At x=0: df/dx = 1*(0 + 1) = 1
    println!("df/dx at x=0 = {:.6} (expected: 1.0)", x.gradient());

    // f(x) = log(x²) at x=2
    let tape2 = Rc::new(RefCell::new(Tape::new()));
    let x2 = TapeValue::variable(&tape2, 2.0);

    let x_sq = x2.mul(&x2);
    let result2 = x_sq.log(); // log(x²)

    println!("\nf(x) = ln(x²)");
    println!("At x=2: f(2) = {:.6}", result2.value());

    result2.backward();

    // df/dx = 2x/x² = 2/x = 2/2 = 1
    println!("df/dx at x=2 = {:.6} (expected: 1.0)", x2.gradient());
    println!();
}

/// Demo 3: Hyperbolic function backpropagation
fn demo_hyperbolic_backward() {
    println!("--- Demo 3: Hyperbolic Function Gradients (NEW) ---\n");

    // f(x) = tanh(x) at x=0
    let tape = Rc::new(RefCell::new(Tape::new()));
    let x = TapeValue::variable(&tape, 0.0);
    let result = x.tanh();

    println!("f(x) = tanh(x)");
    println!("At x=0: f(0) = {:.6}", result.value());

    result.backward();

    // df/dx = sech²(x) = 1/cosh²(x) = 1/1 = 1
    println!(
        "df/dx at x=0 = {:.6} (expected: 1.0 = sech²(0))",
        x.gradient()
    );

    // f(x, y) = sinh(x) * cosh(y) at (0, 0)
    let tape2 = Rc::new(RefCell::new(Tape::new()));
    let x2 = TapeValue::variable(&tape2, 0.0);
    let y2 = TapeValue::variable(&tape2, 0.0);

    let sinh_x = x2.sinh();
    let cosh_y = y2.cosh();
    let result2 = sinh_x.mul(&cosh_y);

    println!("\nf(x,y) = sinh(x) * cosh(y)");
    println!("At (0,0): f = {:.6}", result2.value());

    result2.backward();

    // df/dx = cosh(x) * cosh(y) = 1 * 1 = 1
    // df/dy = sinh(x) * sinh(y) = 0 * 0 = 0
    println!("∂f/∂x at (0,0) = {:.6} (expected: 1.0)", x2.gradient());
    println!("∂f/∂y at (0,0) = {:.6} (expected: 0.0)", y2.gradient());
    println!();
}

/// Demo 4: Complex chain rule example
fn demo_chain_rule() {
    println!("--- Demo 4: Complex Chain Rule ---\n");

    // f(x) = exp(sin(x²)) at x=1
    let tape = Rc::new(RefCell::new(Tape::new()));
    let x = TapeValue::variable(&tape, 1.0);

    let x_sq = x.mul(&x); // x²
    let sin_x_sq = x_sq.sin(); // sin(x²)
    let result = sin_x_sq.exp(); // exp(sin(x²))

    let expected_val = libm::exp(libm::sin(1.0 * 1.0));
    println!("f(x) = exp(sin(x²))");
    println!(
        "At x=1: f(1) = {:.6} (expected: {:.6})",
        result.value(),
        expected_val
    );

    result.backward();

    // df/dx = exp(sin(x²)) * cos(x²) * 2x
    let expected_grad = libm::exp(libm::sin(1.0)) * libm::cos(1.0) * 2.0;
    println!(
        "df/dx at x=1 = {:.6} (expected: {:.6})",
        x.gradient(),
        expected_grad
    );
    println!();
}

/// Demo 5: Neural network layer-like computation
fn demo_neural_network_layer() {
    println!("--- Demo 5: Neural Network Layer (Reverse-Mode Advantage) ---\n");

    // Simulate: y = tanh(w₁*x₁ + w₂*x₂ + w₃*x₃ + b)
    // Many parameters (w₁, w₂, w₃, b), one output (y)
    // Reverse-mode computes all gradients in one backward pass!

    let tape = Rc::new(RefCell::new(Tape::new()));

    // Inputs (fixed)
    let x1 = TapeValue::constant(&tape, 1.0);
    let x2 = TapeValue::constant(&tape, 2.0);
    let x3 = TapeValue::constant(&tape, 3.0);

    // Parameters (trainable - need gradients)
    let w1 = TapeValue::variable(&tape, 0.5);
    let w2 = TapeValue::variable(&tape, -0.3);
    let w3 = TapeValue::variable(&tape, 0.8);
    let b = TapeValue::variable(&tape, 0.1);

    // Forward pass
    let z1 = w1.mul(&x1); // 0.5 * 1 = 0.5
    let z2 = w2.mul(&x2); // -0.3 * 2 = -0.6
    let z3 = w3.mul(&x3); // 0.8 * 3 = 2.4

    let z = z1.add(&z2).add(&z3).add(&b); // 0.5 - 0.6 + 2.4 + 0.1 = 2.4
    let y = z.tanh(); // tanh(2.4)

    println!("y = tanh(w₁*x₁ + w₂*x₂ + w₃*x₃ + b)");
    println!("Inputs: x = [1.0, 2.0, 3.0]");
    println!("Weights: w = [0.5, -0.3, 0.8]");
    println!("Bias: b = 0.1");
    println!("Output: y = {:.6}", y.value());

    // Backward pass - computes ∂y/∂w₁, ∂y/∂w₂, ∂y/∂w₃, ∂y/∂b in ONE pass
    y.backward();

    println!("\nGradients (for gradient descent):");
    println!("  ∂y/∂w₁ = {:.6}", w1.gradient());
    println!("  ∂y/∂w₂ = {:.6}", w2.gradient());
    println!("  ∂y/∂w₃ = {:.6}", w3.gradient());
    println!("  ∂y/∂b  = {:.6}", b.gradient());
    println!("\n✓ All 4 gradients computed in ONE backward pass!");
    println!("  (Forward-mode would require 4 forward passes)");
    println!();
}

/// Demo 6: Jacobian for multi-output function
fn demo_jacobian_multi_output() {
    println!("--- Demo 6: Jacobian for Multi-Output Function ---\n");

    // f(x, y) = [x² + y, x*y, exp(x)*sin(y)]
    // 3 outputs, 2 inputs → 3×2 Jacobian
    let f = |inputs: &[TapeValue]| {
        let x = &inputs[0];
        let y = &inputs[1];

        let out1 = x.mul(x).add(y); // x² + y
        let out2 = x.mul(y); // x * y
        let out3 = x.exp().mul(&y.sin()); // exp(x) * sin(y)

        vec![out1, out2, out3]
    };

    let x_val = 1.0;
    let y_val = std::f64::consts::PI / 2.0;

    let jacobian = compute_jacobian_reverse(&f, &[x_val, y_val]);

    println!("f(x,y) = [x² + y, x*y, exp(x)*sin(y)]");
    println!("At (x,y) = (1.0, π/2):\n");

    println!("Jacobian matrix (3×2):");
    for (_i, row) in jacobian.iter().enumerate() {
        print!("  [");
        for (j, &val) in row.iter().enumerate() {
            print!("{:8.4}", val);
            if j < row.len() - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    println!("\nExpected:");
    println!(
        "  [{:8.4}, {:8.4}]  (∂(x²+y)/∂x = 2x = 2, ∂(x²+y)/∂y = 1)",
        2.0, 1.0
    );
    println!(
        "  [{:8.4}, {:8.4}]  (∂(xy)/∂x = y = π/2, ∂(xy)/∂y = x = 1)",
        y_val, x_val
    );
    let exp_x = libm::exp(x_val);
    let sin_y = libm::sin(y_val);
    let cos_y = libm::cos(y_val);
    println!(
        "  [{:8.4}, {:8.4}]  (∂(e^x·sin(y))/∂x = e^x·sin(y) ≈ {:.4}, ∂(e^x·sin(y))/∂y = e^x·cos(y) ≈ {:.4})",
        exp_x * sin_y,
        exp_x * cos_y,
        exp_x * sin_y,
        exp_x * cos_y
    );
    println!();
}

/// Demo 7: Efficiency comparison: Forward vs Reverse mode
fn demo_efficiency_comparison() {
    println!("--- Demo 7: Forward vs Reverse Mode Efficiency ---\n");

    println!("Scenario: f: R^100 → R (100 inputs, 1 output)");
    println!("Task: Compute gradient ∇f ∈ R^100\n");

    println!("Forward-mode AD:");
    println!("  - Need 100 forward passes (one per input dimension)");
    println!("  - Complexity: O(100 × forward_cost)");
    println!("  - Memory: O(1) per pass");

    println!("\nReverse-mode AD:");
    println!("  - Need 1 forward pass + 1 backward pass");
    println!("  - Complexity: O(forward_cost + backward_cost) ≈ O(3 × forward_cost)");
    println!("  - Memory: O(tape_size) to store computation graph");

    println!("\nResult: Reverse-mode is ~33× faster for this scenario!");

    println!("\n\nScenario: f: R → R^100 (1 input, 100 outputs)");
    println!("Task: Compute Jacobian ∈ R^{{100×1}}\n");

    println!("Forward-mode AD:");
    println!("  - Need 1 forward pass with dual numbers");
    println!("  - Complexity: O(forward_cost)");
    println!("  - Memory: O(1)");

    println!("\nReverse-mode AD:");
    println!("  - Need 1 forward + 100 backward passes (one per output)");
    println!("  - Complexity: O(forward_cost + 100 × backward_cost)");
    println!("  - Memory: O(tape_size)");

    println!("\nResult: Forward-mode is ~100× faster for this scenario!");

    println!("\n\n**Golden Rule**:");
    println!("  - Use REVERSE-mode when: #inputs >> #outputs (neural networks, optimization)");
    println!("  - Use FORWARD-mode when: #outputs >> #inputs (Jacobian of vector functions)");
    println!();
}
