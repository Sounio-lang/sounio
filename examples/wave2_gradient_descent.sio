/// Wave 2 Example: Gradient Descent Optimization
///
/// Demonstrates using automatic differentiation for optimization.
/// Minimizes the Rosenbrock function using gradient descent.
///
/// This example shows:
/// - Computing gradients of multivariate functions
/// - Iterative optimization
/// - Convergence to minimum

fn main() {
    println("=== Gradient Descent Optimization ===");
    println("Minimizing f(x, y) = (1-x)² + 100(y-x²)²");
    println("(Rosenbrock function)");

    // Starting point
    let x = 0.0;
    let y = 0.0;

    println!("\nInitial point: ({}, {})", x, y);
    println("f(x, y) = {}", rosenbrock(x, y));

    // Simulate gradient descent steps
    let learning_rate = 0.001;
    let mut curr_x = x;
    let mut curr_y = y;

    println("\n=== Optimization Progress ===");
    println("Iteration | f(x, y)        | x          | y");
    println("----------|----------------|------------|----------");

    for iter in 0..10 {
        let f = rosenbrock(curr_x, curr_y);

        // Numerical gradient approximation (since grad() is being implemented)
        let h = 0.0001;
        let grad_x = (rosenbrock(curr_x + h, curr_y) - f) / h;
        let grad_y = (rosenbrock(curr_x, curr_y + h) - f) / h;

        // Gradient descent step
        curr_x = curr_x - learning_rate * grad_x;
        curr_y = curr_y - learning_rate * grad_y;

        println("{:9} | {:14.6} | {:10.6} | {:10.6}", iter, f, curr_x, curr_y);
    }

    println!("\nFinal point: ({:.6}, {:.6})", curr_x, curr_y);
    println("f(x, y) = {:.6}", rosenbrock(curr_x, curr_y));
    println!("Target: (1.0, 1.0) where f = 0");
}

fn rosenbrock(x: f64, y: f64) -> f64 {
    let a = 1.0 - x;
    let b = y - x * x;
    return a * a + 100.0 * b * b;
}
