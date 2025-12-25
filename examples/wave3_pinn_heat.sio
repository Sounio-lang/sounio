/// Wave 3 Example: Physics-Informed Neural Network (PINN)
///
/// Uses a hybrid model to solve the 1D heat equation:
/// ∂u/∂t - ∂²u/∂x² = 0
///
/// The symbolic component encodes the PDE constraint,
/// and the neural component approximates the solution.

fn main() {
    println!("=== Physics-Informed Neural Network (PINN) ===");
    println!("Solving heat equation: ∂u/∂t = ∂²u/∂x² \n");

    println!("=== Problem Setup ===");
    println!("Domain: x ∈ [0, 1], t ∈ [0, 1]");
    println!("Initial condition: u(x, 0) = sin(π*x)");
    println!("Boundary conditions: u(0, t) = 0, u(1, t) = 0\n");

    println!("Analytical solution:");
    println!("u(x, t) = sin(π*x) * exp(-π²*t)\n");

    println!("=== PINN Strategy ===");
    println!("1. Use neural network to approximate u(x, t)");
    println!("2. Compute ∂u/∂t and ∂²u/∂x² symbolically (autodiff)");
    println!("3. Enforce PDE residual: r = ∂u/∂t - ∂²u/∂x²");
    println!("4. Loss = data_loss + physics_loss\n");

    println!("Data points (observations):");
    let observations = vec![
        // (x, t, u_observed)
        (0.5, 0.0, 1.0),           // Initial: u(0.5, 0) = sin(π*0.5) = 1.0
        (0.5, 0.1, 0.070),         // u(0.5, 0.1) ≈ exp(-π²*0.1) ≈ 0.070
        (0.25, 0.0, 0.707),        // u(0.25, 0) = sin(π*0.25) ≈ 0.707
        (0.75, 0.0, 0.707),        // u(0.75, 0) = sin(π*0.75) ≈ 0.707
    ];

    println!("x\t| t\t| u_observed");
    for (x, t, u) in &observations {
        println!("{:.2}\t| {:.2}\t| {:.3}", x, t, u);
    }
    println!();

    println!("Physics-informed collocation points:");
    let collocation = vec![
        (0.3, 0.05),
        (0.5, 0.05),
        (0.7, 0.05),
        (0.3, 0.1),
        (0.5, 0.1),
        (0.7, 0.1),
    ];
    println!("({} interior points for PDE constraint)\n", collocation.len());

    println!("=== Training Loop ===");
    let epochs = 10;
    for epoch in 0..epochs {
        println!("Epoch {}/{}:", epoch + 1, epochs);

        println!("  Data loss:");
        let mut data_loss = 0.0;
        for (x, t, u_true) in &observations {
            // u_pred would come from neural network
            // For illustration: use exact solution
            let u_pred = (std::f64::consts::PI * x).sin() * (-std::f64::consts::PI * std::f64::consts::PI * t).exp();
            let error = u_pred - u_true;
            data_loss += error * error;
            println!("    ({:.2}, {:.2}): error = {:.6}", x, t, error.abs());
        }
        data_loss /= observations.len() as f64;

        println!("  Physics loss (PDE residual):");
        let mut physics_loss = 0.0;
        for (x, t) in &collocation {
            // For exact solution: residual should be ~0
            // r = ∂u/∂t - ∂²u/∂x²
            // u = sin(πx) * exp(-π²t)
            // ∂u/∂t = -π² * sin(πx) * exp(-π²t)
            // ∂²u/∂x² = -π² * sin(πx) * exp(-π²t)
            // r = -π² * sin(πx) * exp(-π²t) - (-π² * sin(πx) * exp(-π²t)) = 0
            let residual = 0.0;  // Would be computed from neural network output
            physics_loss += residual * residual;
        }
        physics_loss /= collocation.len() as f64;

        println!("    Physics residual norm: {:.6}", physics_loss);

        let total_loss = data_loss + physics_loss;
        println!("  Total loss: {:.6}\n", total_loss);
    }

    println!("=== Solution Visualization ===");
    println!("t=0.0:");
    for x in (0..=10).map(|i| i as f64 / 10.0) {
        let u = (std::f64::consts::PI * x).sin();
        let bar_len = (u * 40.0) as usize;
        println!("{:.1} | {}  {:.3}", x, "█".repeat(bar_len), u);
    }

    println!("\nt=0.1:");
    for x in (0..=10).map(|i| i as f64 / 10.0) {
        let u = (std::f64::consts::PI * x).sin() * (-std::f64::consts::PI * std::f64::consts::PI * 0.1).exp();
        let bar_len = (u * 40.0) as usize;
        println!("{:.1} | {}  {:.3}", x, "█".repeat(bar_len), u);
    }

    println!("\n=== Key Insights ===");
    println!("✓ PINNs encode physics as soft constraints (loss term)");
    println!("✓ No need for mesh generation or numerical discretization");
    println!("✓ Handles complex boundary conditions naturally");
    println!("✓ Can work with sparse data + strong physics");
    println!("✓ Symbolic derivatives ensure conservation laws");
}
