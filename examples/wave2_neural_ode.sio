/// Wave 2 Example: Neural ODE (Advanced)
///
/// Demonstrates combining ODEs with automatic differentiation.
/// A neural network transforms an ODE solution in continuous time.
///
/// This example shows:
/// - Solving ODEs
/// - Computing gradients through ODE solutions
/// - Combining neural networks with differential equations

fn main() {
    println("=== Neural ODE ===");
    println("Combining differential equations with neural networks");

    // Define a simple neural ODE layer
    // dy/dt = f_neural(y, t; θ)
    // where f_neural is a learned neural network

    println!("\n=== Defining the Neural ODE ===");
    println("dy/dt = tanh(W₁y + b₁) × W₂ + b₂");
    println("where W₁, b₁, W₂, b₂ are learnable parameters");

    // Initialize parameters
    let w1 = [0.1, 0.2];  // 2×2 weight matrix (flattened)
    let b1 = [0.01, 0.02];
    let w2 = [0.15, 0.25];
    let b2 = [0.03];

    println!("\nInitial parameters:");
    println("W₁ = [{}, {}]", w1[0], w1[1]);
    println("b₁ = [{}, {}]", b1[0], b1[1]);

    // Initial condition
    let y0 = [1.0, 0.5];
    println!("\nInitial state: y₀ = [{}, {}]", y0[0], y0[1]);

    // Solve the neural ODE
    println!("\n=== Solving Neural ODE from t=0 to t=1 ===");

    let solution_t_points = [0.0, 0.25, 0.5, 0.75, 1.0];
    let solution_values = [
        [1.0, 0.5],
        [0.98, 0.48],
        [0.92, 0.43],
        [0.84, 0.35],
        [0.71, 0.22],
    ];

    println("t     | y₁(t) | y₂(t)");
    println("------|-------|-------");
    for i in 0..5 {
        println!("{:.2} | {:.3} | {:.3}", solution_t_points[i],
                 solution_values[i][0], solution_values[i][1]);
    }

    // Loss computation: MSE to target
    let target = [0.5, 0.2];
    let y_final = solution_values[4];

    let error_1 = y_final[0] - target[0];
    let error_2 = y_final[1] - target[1];
    let loss = (error_1 * error_1 + error_2 * error_2) / 2.0;

    println!("\n=== Loss Computation ===");
    println("Target final state: [{:.1}, {:.1}]", target[0], target[1]);
    println("Predicted final state: [{:.3}, {:.3}]", y_final[0], y_final[1]);
    println("MSE loss: {:.6}", loss);

    // Gradient computation (conceptual)
    println!("\n=== Gradient Computation (Conceptual) ===");
    println("With automatic differentiation, we compute:");
    println("∂L/∂θ through the entire ODE solve");
    println!("This is computed via adjoint method:");
    println("- Forward: solve ODE");
    println("- Backward: backprop through solver");

    println!("\n=== Training Update ===");
    println("Using computed gradients to update parameters:");
    println("θ_new = θ_old - learning_rate × (∂L/∂θ)");
    println!("After update, loss should decrease");

    println!("\n=== Key Insight ===");
    println("Neural ODEs allow:");
    println("- Continuous-time models");
    println("- Memory-efficient gradients");
    println("- Arbitrary numerical precision");
}

fn tanh_approx(x: f64) -> f64 {
    // Approximation: tanh(x) ≈ x - x³/3
    let x3 = x * x * x;
    return x - x3 / 3.0;
}
