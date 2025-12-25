/// Wave 1 Example: Exponential Decay ODE
///
/// Demonstrates solving a simple ODE: dy/dt = -k*y
/// Solution: y(t) = y0 * exp(-k*t)
///
/// This example shows:
/// - Defining and solving ODEs
/// - Working with ODE solutions
/// - Basic scientific computing

fn main() {
    println("=== Exponential Decay Example ===");

    // Define the ODE: dy/dt = -0.5 * y
    let k = 0.5;
    let decay = |t: f64, y: [f64]| -> [f64] {
        let dydt = [-k * y[0]];
        return dydt;
    };

    // Initial condition: y(0) = 1.0
    let y0 = [1.0];

    // Solve from t=0 to t=10
    let solution = solve_ode(decay, y0, (0.0, 10.0));

    println!("Solution obtained!");
    println!("Time points: {} values", solution.t.len());
    println!("Number of trajectories: {} dimensions", solution.y.len());

    // Display some results
    println("\nResults:");
    for i in 0..min(5, solution.t.len()) {
        println!("t = {:.2}, y = {:.4}", solution.t[i], solution.y[i][0]);
    }
}

fn min(a: i64, b: i64) -> i64 {
    if a < b { a } else { b }
}
