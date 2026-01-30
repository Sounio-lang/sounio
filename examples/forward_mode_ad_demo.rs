//! Forward-Mode Automatic Differentiation Demo
//!
//! Demonstrates Task 2.1: Complete Forward-Mode AD with libm integration.
//!
//! This example shows:
//! - Dual number arithmetic for automatic differentiation
//! - Computing derivatives symbolically (no numerical errors)
//! - Composing functions with the chain rule
//! - Integration with pure-Rust libm for mathematical functions

use sounio::codegen::autodiff::Dual;

fn main() {
    println!("=== Forward-Mode Automatic Differentiation Demo ===\n");

    // Example 1: Simple polynomial
    demo_polynomial();

    // Example 2: Trigonometric functions
    demo_trig();

    // Example 3: Exponential and logarithmic
    demo_exp_log();

    // Example 4: Chain rule composition
    demo_chain_rule();

    // Example 5: Scientific computing use case
    demo_scientific();
}

fn demo_polynomial() {
    println!("--- Example 1: Polynomial Differentiation ---");
    println!("f(x) = 3x² + 2x + 1");
    println!("f'(x) = 6x + 2\n");

    let x = Dual::variable(2.0);

    // f(x) = 3x² + 2x + 1
    let three_x_sq = Dual::constant(3.0).mul(x.mul(x));
    let two_x = Dual::constant(2.0).mul(x);
    let one = Dual::constant(1.0);
    let result = three_x_sq.add(two_x).add(one);

    println!("At x = 2.0:");
    println!("  f(2) = {}", result.val); // 3*4 + 2*2 + 1 = 17
    println!("  f'(2) = {}", result.deriv); // 6*2 + 2 = 14
    println!();
}

fn demo_trig() {
    println!("--- Example 2: Trigonometric Functions ---");
    println!("f(x) = sin(x)");
    println!("f'(x) = cos(x)\n");

    let x = Dual::variable(0.0);
    let result = x.sin();

    println!("At x = 0:");
    println!("  sin(0) = {:.6}", result.val);
    println!("  d/dx sin(0) = cos(0) = {:.6}", result.deriv);
    println!();
}

fn demo_exp_log() {
    println!("--- Example 3: Exponential and Logarithmic ---");
    println!("f(x) = exp(x)");
    println!("f'(x) = exp(x)\n");

    let x = Dual::variable(0.0);
    let result = x.exp();

    println!("At x = 0:");
    println!("  exp(0) = {:.6}", result.val);
    println!("  d/dx exp(0) = {:.6}", result.deriv);
    println!();

    println!("g(x) = log(x)");
    println!("g'(x) = 1/x\n");

    let y = Dual::variable(std::f64::consts::E);
    let result2 = y.log();

    println!("At x = e:");
    println!("  log(e) = {:.6}", result2.val);
    println!("  d/dx log(e) = 1/e = {:.6}", result2.deriv);
    println!();
}

fn demo_chain_rule() {
    println!("--- Example 4: Chain Rule Composition ---");
    println!("f(x) = sin(x²)");
    println!("f'(x) = 2x * cos(x²)\n");

    let x = Dual::variable(1.0);
    let x_squared = x.mul(x);
    let result = x_squared.sin();

    println!("At x = 1.0:");
    println!("  f(1) = sin(1) = {:.6}", result.val);
    println!("  f'(1) = 2 * cos(1) = {:.6}", result.deriv);
    println!(
        "  (Expected: 2 * {:.6} = {:.6})",
        libm::cos(1.0),
        2.0 * libm::cos(1.0)
    );
    println!();
}

fn demo_scientific() {
    println!("--- Example 5: Scientific Computing Use Case ---");
    println!("Gaussian probability density function:");
    println!("f(x) = exp(-x²/2) / sqrt(2π)");
    println!("f'(x) = -x * exp(-x²/2) / sqrt(2π)\n");

    let x = Dual::variable(0.0);

    // Compute -x²/2
    let x_squared = x.mul(x);
    let neg_half_x_sq = x_squared.mul(Dual::constant(-0.5));

    // exp(-x²/2)
    let exp_term = neg_half_x_sq.exp();

    // Normalization constant
    let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
    let norm = Dual::constant(1.0 / sqrt_2pi);

    // Full PDF
    let pdf = exp_term.mul(norm);

    println!("At x = 0 (peak of Gaussian):");
    println!("  PDF value: {:.6}", pdf.val);
    println!("  PDF derivative: {:.6}", pdf.deriv);
    println!("  (Derivative should be 0 at the peak)");
}
