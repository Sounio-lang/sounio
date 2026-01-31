//! Forward-Mode Automatic Differentiation Demo
//!
//! Demonstrates Task 2.1: Complete forward-mode AD for Sounio compiler.
//!
//! Features showcased:
//! 1. Basic differentiation with Dual numbers
//! 2. Transcendental functions (exp, log, sin, cos, tan)
//! 3. Hyperbolic functions (sinh, cosh, tanh)
//! 4. General power function
//! 5. Jacobian computation for multi-dimensional functions
//!
//! Run with: cargo run --example forward_mode_ad_demo

use sounio::codegen::autodiff::{compute_jacobian, Dual};

fn main() {
    println!("=== Forward-Mode Automatic Differentiation Demo ===\n");

    demo_basic_derivatives();
    demo_transcendental_functions();
    demo_hyperbolic_functions();
    demo_general_power();
    demo_chain_rule();
    demo_jacobian_2d();
    demo_jacobian_nonlinear();
    demo_jacobian_3d();
}

/// Demo 1: Basic derivative rules
fn demo_basic_derivatives() {
    println!("--- Demo 1: Basic Derivative Rules ---\n");

    // f(x) = x² at x=3
    let x = Dual::variable(3.0);
    let y = x.mul(x);
    println!("f(x) = x²");
    println!(
        "  At x=3: f(3) = {}, f'(3) = {} (expected: 9, 6)",
        y.val, y.deriv
    );

    // f(x) = 2x + 5 at x=4
    let x = Dual::variable(4.0);
    let two = Dual::constant(2.0);
    let five = Dual::constant(5.0);
    let y = two.mul(x).add(five);
    println!("\nf(x) = 2x + 5");
    println!(
        "  At x=4: f(4) = {}, f'(4) = {} (expected: 13, 2)",
        y.val, y.deriv
    );

    // f(x) = 1/x at x=2
    let x = Dual::variable(2.0);
    let one = Dual::constant(1.0);
    let y = one.div(x);
    println!("\nf(x) = 1/x");
    println!(
        "  At x=2: f(2) = {}, f'(2) = {} (expected: 0.5, -0.25)",
        y.val, y.deriv
    );

    println!();
}

/// Demo 2: Transcendental functions
fn demo_transcendental_functions() {
    println!("--- Demo 2: Transcendental Functions ---\n");

    // f(x) = exp(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.exp();
    println!("f(x) = exp(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 1, 1)",
        y.val, y.deriv
    );

    // f(x) = log(x) at x=1
    let x = Dual::variable(1.0);
    let y = x.log();
    println!("\nf(x) = ln(x)");
    println!(
        "  At x=1: f(1) = {:.6}, f'(1) = {:.6} (expected: 0, 1)",
        y.val, y.deriv
    );

    // f(x) = sin(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.sin();
    println!("\nf(x) = sin(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 0, 1)",
        y.val, y.deriv
    );

    // f(x) = cos(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.cos();
    println!("\nf(x) = cos(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 1, 0)",
        y.val, y.deriv
    );

    // f(x) = tan(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.tan();
    println!("\nf(x) = tan(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 0, 1)",
        y.val, y.deriv
    );

    println!();
}

/// Demo 3: Hyperbolic functions (NEW in Task 2.1)
fn demo_hyperbolic_functions() {
    println!("--- Demo 3: Hyperbolic Functions (NEW) ---\n");

    // f(x) = sinh(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.sinh();
    println!("f(x) = sinh(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 0, 1 = cosh(0))",
        y.val, y.deriv
    );

    // f(x) = cosh(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.cosh();
    println!("\nf(x) = cosh(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 1, 0 = sinh(0))",
        y.val, y.deriv
    );

    // f(x) = tanh(x) at x=0
    let x = Dual::variable(0.0);
    let y = x.tanh();
    println!("\nf(x) = tanh(x)");
    println!(
        "  At x=0: f(0) = {:.6}, f'(0) = {:.6} (expected: 0, 1 = sech²(0))",
        y.val, y.deriv
    );

    // f(x) = sinh(x) at x=1
    let x = Dual::variable(1.0);
    let y = x.sinh();
    let expected_val = libm::sinh(1.0);
    let expected_deriv = libm::cosh(1.0);
    println!("\nf(x) = sinh(x)");
    println!(
        "  At x=1: f(1) = {:.6}, f'(1) = {:.6} (expected: {:.6}, {:.6})",
        y.val, y.deriv, expected_val, expected_deriv
    );

    println!();
}

/// Demo 4: General power function (NEW in Task 2.1)
fn demo_general_power() {
    println!("--- Demo 4: General Power Function (NEW) ---\n");

    // f(x) = x^3 using general pow
    let x = Dual::variable(2.0);
    let exp = Dual::constant(3.0);
    let y = x.pow(exp);
    println!("f(x) = x³ (using general pow)");
    println!(
        "  At x=2: f(2) = {:.6}, f'(2) = {:.6} (expected: 8, 12 = 3*2²)",
        y.val, y.deriv
    );

    // f(x) = 2^x (exponential with variable exponent)
    let base = Dual::constant(2.0);
    let x = Dual::variable(3.0);
    let y = base.pow(x);
    let expected_deriv = 8.0 * libm::log(2.0);
    println!("\nf(x) = 2^x");
    println!(
        "  At x=3: f(3) = {:.6}, f'(3) = {:.6} (expected: 8, {:.6} = 8*ln(2))",
        y.val, y.deriv, expected_deriv
    );

    // f(x, y) = x^y with both variables active
    let x = Dual::new(2.0, 1.0); // dx/dx = 1
    let y = Dual::new(3.0, 0.0); // dy/dx = 0
    let result = x.pow(y);
    println!("\nf(x,y) = x^y with respect to x");
    println!(
        "  At (x,y)=(2,3): ∂f/∂x = {:.6} (expected: 12 = 3*2²)",
        result.deriv
    );

    println!();
}

/// Demo 5: Complex chain rule example
fn demo_chain_rule() {
    println!("--- Demo 5: Complex Chain Rule ---\n");

    // f(x) = exp(sin(x²))
    // f'(x) = exp(sin(x²)) * cos(x²) * 2x
    let x = Dual::variable(1.0);
    let x_squared = x.mul(x);
    let sin_x_squared = x_squared.sin();
    let result = sin_x_squared.exp();

    let x_val = 1.0;
    let expected_val = libm::exp(libm::sin(x_val * x_val));
    let expected_deriv =
        libm::exp(libm::sin(x_val * x_val)) * libm::cos(x_val * x_val) * (2.0 * x_val);

    println!("f(x) = exp(sin(x²))");
    println!(
        "  At x=1: f(1) = {:.6}, f'(1) = {:.6}",
        result.val, result.deriv
    );
    println!(
        "  Expected: f(1) = {:.6}, f'(1) = {:.6}",
        expected_val, expected_deriv
    );

    println!();
}

/// Demo 6: Jacobian for 2D function (NEW in Task 2.1)
fn demo_jacobian_2d() {
    println!("--- Demo 6: Jacobian Matrix (2D Function) ---\n");

    // f(x, y) = [x² + y, x*y]
    // J = [[2x, 1  ],
    //      [y,  x  ]]
    let f = |inputs: &[Dual]| {
        vec![
            inputs[0].pow_const(2.0).add(inputs[1]), // x² + y
            inputs[0].mul(inputs[1]),                // x * y
        ]
    };

    let jacobian = compute_jacobian(&f, &[2.0, 3.0]);

    println!("f(x,y) = [x² + y, x*y]");
    println!("At point (x,y) = (2, 3):\n");
    println!("Jacobian matrix:");
    println!("  [[{:6.2}, {:6.2}],", jacobian[0][0], jacobian[0][1]);
    println!("   [{:6.2}, {:6.2}]]", jacobian[1][0], jacobian[1][1]);
    println!("\nExpected:");
    println!("  [[{:6.2}, {:6.2}],  (2x=4, 1)", 4.0, 1.0);
    println!("   [{:6.2}, {:6.2}]]   (y=3, x=2)", 3.0, 2.0);

    println!();
}

/// Demo 7: Jacobian for nonlinear function
fn demo_jacobian_nonlinear() {
    println!("--- Demo 7: Jacobian for Nonlinear System ---\n");

    // f(x, y) = [exp(x) * sin(y), x² * cos(y)]
    let f = |inputs: &[Dual]| {
        vec![
            inputs[0].exp().mul(inputs[1].sin()),          // exp(x) * sin(y)
            inputs[0].pow_const(2.0).mul(inputs[1].cos()), // x² * cos(y)
        ]
    };

    let x_val = 0.0;
    let y_val = std::f64::consts::PI / 2.0;

    let jacobian = compute_jacobian(&f, &[x_val, y_val]);

    println!("f(x,y) = [exp(x)*sin(y), x²*cos(y)]");
    println!("At point (x,y) = (0, π/2):\n");
    println!("Jacobian matrix:");
    println!("  [[{:8.4}, {:8.4}],", jacobian[0][0], jacobian[0][1]);
    println!("   [{:8.4}, {:8.4}]]", jacobian[1][0], jacobian[1][1]);

    // Expected at (0, π/2):
    // ∂(exp(x)*sin(y))/∂x = exp(x)*sin(y) = exp(0)*1 = 1
    // ∂(exp(x)*sin(y))/∂y = exp(x)*cos(y) = 1*0 = 0
    // ∂(x²*cos(y))/∂x = 2x*cos(y) = 0*0 = 0
    // ∂(x²*cos(y))/∂y = -x²*sin(y) = 0*1 = 0
    println!("\nExpected:");
    println!("  [[{:8.4}, {:8.4}],", 1.0, 0.0);
    println!("   [{:8.4}, {:8.4}]]", 0.0, 0.0);

    println!();
}

/// Demo 8: Jacobian for 3D function
fn demo_jacobian_3d() {
    println!("--- Demo 8: Jacobian for 3D System ---\n");

    // f(x, y, z) = [x + y + z, x*y*z, x² + y² + z²]
    let f = |inputs: &[Dual]| {
        vec![
            inputs[0].add(inputs[1]).add(inputs[2]), // x + y + z
            inputs[0].mul(inputs[1]).mul(inputs[2]), // x * y * z
            inputs[0]
                .pow_const(2.0)
                .add(inputs[1].pow_const(2.0))
                .add(inputs[2].pow_const(2.0)), // x² + y² + z²
        ]
    };

    let jacobian = compute_jacobian(&f, &[1.0, 2.0, 3.0]);

    println!("f(x,y,z) = [x+y+z, x*y*z, x²+y²+z²]");
    println!("At point (x,y,z) = (1, 2, 3):\n");
    println!("Jacobian matrix (3×3):");
    for (i, row) in jacobian.iter().enumerate() {
        print!("  [");
        for (j, &val) in row.iter().enumerate() {
            print!("{:6.2}", val);
            if j < row.len() - 1 {
                print!(", ");
            }
        }
        println!("]");
    }

    println!("\nExpected:");
    println!(
        "  [{:6.2}, {:6.2}, {:6.2}]  (∂(x+y+z) = [1,1,1])",
        1.0, 1.0, 1.0
    );
    println!(
        "  [{:6.2}, {:6.2}, {:6.2}]  (∂(xyz) = [yz, xz, xy] = [6,3,2])",
        6.0, 3.0, 2.0
    );
    println!(
        "  [{:6.2}, {:6.2}, {:6.2}]  (∂(x²+y²+z²) = [2x,2y,2z] = [2,4,6])",
        2.0, 4.0, 6.0
    );

    println!();
}
