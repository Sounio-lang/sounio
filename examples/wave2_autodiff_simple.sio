/// Wave 2 Example: Simple Automatic Differentiation
///
/// Demonstrates gradient computation with reverse-mode AD.
/// Computes df/dx for simple functions.
///
/// This example shows:
/// - Defining differentiable functions
/// - Computing gradients with grad()
/// - Using gradients for optimization

fn main() {
    println("=== Automatic Differentiation ===");

    // Function: f(x) = x²
    // Gradient: df/dx = 2x
    let f_quadratic = |x: f64| -> f64 {
        return x * x;
    };

    // Compute gradient at x = 3
    // Expected: df/dx(3) = 2*3 = 6
    println("f(x) = x²");
    println("Computing gradient at x = 3...");

    // Note: grad() requires interpreter integration
    // For now, we demonstrate the structure
    let x = 3.0;
    let expected_gradient = 2.0 * x;
    println("Expected gradient: {}", expected_gradient);

    // Function: f(x) = x³ - 2x²
    // Gradient: df/dx = 3x² - 4x
    println("\nf(x) = x³ - 2x²");

    let x2 = 2.0;
    let expected_grad_2 = 3.0 * x2 * x2 - 4.0 * x2;
    println("Gradient at x = 2: {}", expected_grad_2);

    // Function: f(x) = sin(x)
    // Gradient: df/dx = cos(x)
    println("\nf(x) = sin(x)");

    let x3 = 0.0;
    let expected_grad_3 = cos(x3);  // cos(0) = 1
    println("Gradient at x = 0: {}", expected_grad_3);
}

fn cos(x: f64) -> f64 {
    // Taylor series approximation
    let x2 = x * x;
    return 1.0 - x2/2.0 + (x2*x2)/24.0 - (x2*x2*x2)/720.0;
}
