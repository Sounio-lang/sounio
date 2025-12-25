// Automatic Differentiation Example
//
// This example demonstrates forward-mode automatic differentiation using dual numbers.
// Dual numbers are pairs (value, derivative) that propagate derivatives through computations.

// A dual number represents both a value and its derivative with respect to some variable.
// When we seed the derivative with 1.0, the final derivative gives us df/dx.

fn main() -> i32 {
    // Example 1: Derivative of f(x) = x^2 at x = 3
    // f'(x) = 2x, so f'(3) = 6
    let x = dual(3.0, 1.0);  // Create dual number: value=3.0, derivative=1.0 (seed)
    let f_x = x * x;         // x^2 using dual multiplication (product rule)

    let value = dual_value(f_x);    // Should be 9.0
    let deriv = dual_deriv(f_x);    // Should be 6.0 (the derivative at x=3)

    println("f(x) = x^2 at x = 3:");
    println("  value = ", value);    // 9.0
    println("  derivative = ", deriv); // 6.0

    // Example 2: Derivative of g(x) = x^3 at x = 2
    // g'(x) = 3x^2, so g'(2) = 12
    let y = dual(2.0, 1.0);
    let g_y = y * y * y;  // y^3

    println("g(x) = x^3 at x = 2:");
    println("  value = ", dual_value(g_y));     // 8.0
    println("  derivative = ", dual_deriv(g_y)); // 12.0

    // Example 3: Derivative of h(x) = 1/x at x = 2
    // h'(x) = -1/x^2, so h'(2) = -0.25
    let z = dual(2.0, 1.0);
    let one = dual(1.0, 0.0);  // Constant 1 (derivative = 0)
    let h_z = one / z;

    println("h(x) = 1/x at x = 2:");
    println("  value = ", dual_value(h_z));     // 0.5
    println("  derivative = ", dual_deriv(h_z)); // -0.25

    // Example 4: Chain rule - derivative of f(g(x)) where f(u) = u^2, g(x) = 2x + 1
    // At x = 1: g(1) = 3, f(g(1)) = 9
    // f'(g(x)) * g'(x) = 2*g(x) * 2 = 4*g(x) = 12 at x = 1
    let x4 = dual(1.0, 1.0);
    let two = dual(2.0, 0.0);
    let g_x4 = two * x4 + one;  // g(x) = 2x + 1
    let f_g_x4 = g_x4 * g_x4;   // f(g(x)) = (2x + 1)^2

    println("f(g(x)) = (2x + 1)^2 at x = 1:");
    println("  value = ", dual_value(f_g_x4));     // 9.0
    println("  derivative = ", dual_deriv(f_g_x4)); // 12.0

    // Example 5: Using grad() builtin for cleaner syntax
    // Define a function that works with dual numbers
    fn square(d: dual) -> dual {
        d * d
    }

    // grad(f, x) computes df/dx at x
    let gradient = grad(square, 3.0);
    println("grad(x^2, 3.0) = ", gradient);  // 6.0

    0
}

// Mathematical operations propagate derivatives automatically:
// - Addition: (a, a') + (b, b') = (a + b, a' + b')
// - Subtraction: (a, a') - (b, b') = (a - b, a' - b')
// - Multiplication: (a, a') * (b, b') = (a*b, a'*b + a*b')  [Product rule]
// - Division: (a, a') / (b, b') = (a/b, (a'*b - a*b') / b^2)  [Quotient rule]
// - sqrt: sqrt(a, a') = (sqrt(a), a' / (2*sqrt(a)))
// - exp: exp(a, a') = (exp(a), exp(a) * a')
// - log: log(a, a') = (log(a), a' / a)
// - sin: sin(a, a') = (sin(a), cos(a) * a')
// - cos: cos(a, a') = (cos(a), -sin(a) * a')
