/// Wave 3 Example: Hybrid Neural-Symbolic Regression
///
/// Learns a quadratic function f(x) = ax² + bx + c using a hybrid model:
/// - Symbolic part: ax² + bx + c (structural form is symbolic)
/// - Neural part: learns parameters a, b, c via gradient descent
/// - Fusion: direct evaluation (combine parameters with structure)
///
/// This demonstrates how symbolic structure can guide neural learning.

fn main() {
    println!("=== Hybrid Neural-Symbolic Regression ===");
    println!("Learning f(x) = ax² + bx + c from data\n");

    // Simulated training data from f(x) = 2x² - 3x + 1
    let true_model = "2x² - 3x + 1";
    println!("True model: f(x) = {}\n", true_model);

    println!("=== Data Points ===");
    let data = [
        (-2.0, 15.0),   // 2(4) - 3(-2) + 1 = 8 + 6 + 1 = 15
        (-1.0, 6.0),    // 2(1) - 3(-1) + 1 = 2 + 3 + 1 = 6
        (0.0, 1.0),     // 2(0) - 3(0) + 1 = 1
        (1.0, 0.0),     // 2(1) - 3(1) + 1 = 2 - 3 + 1 = 0
        (2.0, 3.0),     // 2(4) - 3(2) + 1 = 8 - 6 + 1 = 3
        (3.0, 10.0),    // 2(9) - 3(3) + 1 = 18 - 9 + 1 = 10
    ];

    println!("x\t| y (observed)");
    println!("-----+----------");
    for (x, y) in &data {
        println!("{:.1}\t| {:.1}", x, y);
    }
    println!();

    println!("=== Model Structure (Symbolic) ===");
    println!("f(x) = a*x^2 + b*x + c");
    println!("where a, b, c are learnable parameters\n");

    println!("=== Initial Parameters (Random) ===");
    let mut a = 0.1;
    let mut b = 0.2;
    let mut c = 0.5;
    println!("a = {}, b = {}, c = {}\n", a, b, c);

    println!("=== Training Phase ===");
    println!("Using gradient descent to minimize MSE loss:\n");

    let learning_rate = 0.01;
    let epochs = 5;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;

        println!("Epoch {} / {}:", epoch + 1, epochs);
        println!("  a={:.4}, b={:.4}, c={:.4}", a, b, c);

        for (x, y_true) in &data {
            // Forward pass: evaluate model
            let y_pred = a * x * x + b * x + c;
            let error = y_pred - y_true;
            let loss = error * error;
            total_loss += loss;

            // Symbolic gradients (exact, not numeric)
            // ∂(y-ŷ)²/∂a = 2(y-ŷ) * ∂ŷ/∂a = 2(y-ŷ) * x²
            // ∂(y-ŷ)²/∂b = 2(y-ŷ) * x
            // ∂(y-ŷ)²/∂c = 2(y-ŷ)
            let grad_a = 2.0 * error * x * x;
            let grad_b = 2.0 * error * x;
            let grad_c = 2.0 * error;

            // Update parameters
            a -= learning_rate * grad_a;
            b -= learning_rate * grad_b;
            c -= learning_rate * grad_c;
        }

        let avg_loss = total_loss / data.len() as f64;
        println!("  Loss: {:.6}", avg_loss);
    }

    println!("\n=== Final Parameters (Learned) ===");
    println!("a = {:.4} (target: 2.0)", a);
    println!("b = {:.4} (target: -3.0)", b);
    println!("c = {:.4} (target: 1.0)", c);
    println!();

    println!("=== Predictions vs Ground Truth ===");
    println!("x\t| y_true\t| y_pred\t| error");
    println!("-----+----------+----------+--------");
    let mut total_error = 0.0;
    for (x, y_true) in &data {
        let y_pred = a * x * x + b * x + c;
        let error = (y_pred - y_true).abs();
        total_error += error;
        println!("{:.1}\t| {:.2}\t| {:.2}\t| {:.4}", x, y_true, y_pred, error);
    }
    println!("Mean Absolute Error: {:.4}\n", total_error / data.len() as f64);

    println!("=== Key Advantage ===");
    println!("✓ Symbolic structure ax² + bx + c is interpretable");
    println!("✓ Neural learning optimizes parameters precisely");
    println!("✓ Exact gradients from symbolic form");
    println!("✓ No black-box: we know the learned formula");
}
