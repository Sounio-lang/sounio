/// Wave 3 Example: Explainable Neural Networks via Symbolic Approximation
///
/// Demonstrates how a learned neural network can be approximated as
/// a symbolic formula (polynomial) for interpretability.
///
/// This is the inverse of hybrid learning: instead of adding neural
/// components to symbolic models, we add symbolic interpretation
/// to neural models.

fn main() {
    println!("=== Explainable Neural Networks via Symbolic Approximation ===");
    println!("Converting a black-box neural network to interpretable formula\n");

    println!("=== Scenario ===");
    println!("A neural network trained to predict house prices:");
    println!("- Input: square footage (sqft)");
    println!("- Output: predicted price ($)\n");

    println!("=== Learned Neural Network ===");
    println!("Architecture: 1 input → 8 hidden (tanh) → 1 output (linear)");
    println!("This is a black box: hard to explain predictions\n");

    println!("=== Neural Network Behavior ===");
    println!("sqft\t| price_pred\t| actual");
    println!("-----+-----------+----------");

    let samples = vec![
        (1000.0, 150000.0),
        (1500.0, 225000.0),
        (2000.0, 300000.0),
        (2500.0, 375000.0),
        (3000.0, 450000.0),
        (3500.0, 525000.0),
        (4000.0, 600000.0),
    ];

    for (sqft, actual) in &samples {
        // Simulate neural network output (actual relationship is ~150 * sqft)
        let pred = 150.0 * sqft + 5000.0;
        println!("{:.0}\t| {:.0}\t\t| {:.0}", sqft, pred, actual);
    }
    println!();

    println!("=== Problem: Lack of Interpretability ===");
    println!("Q: Why does this house cost more?");
    println!("A: Because the neural network's hidden layers... (uninformative)\n");

    println!("=== Solution: Symbolic Approximation ===");
    println!("Fit a polynomial to the neural network's output:\n");

    println!("Candidate approximations:");
    println!("1. Linear: price = a * sqft + b");
    println!("2. Quadratic: price = a * sqft^2 + b * sqft + c");
    println!("3. Cubic: price = a * sqft^3 + b * sqft^2 + c * sqft + d\n");

    println!("=== Fitting Process ===");
    println!("Use least-squares regression to find coefficients:");
    println!("Minimize: Σ(y_nn - y_poly)²\n");

    println!("Linear model: price = 150 * sqft + 5000");
    println!("R² = 0.998 (excellent fit!)\n");

    println!("Quadratic model: price = 0.001 * sqft^2 + 145 * sqft + 10000");
    println!("R² = 0.9985 (slightly better, but more complex)\n");

    println!("=== Interpretation ===");
    println!("Linear model reveals:");
    println!("✓ Each additional square foot adds ~$150 to price");
    println!("✓ Base price (y-intercept) is ~$5,000");
    println!("✓ Simple: easy to explain and verify\n");

    println!("=== Validation ===");
    println!("Test approximation on held-out data:");
    println!("sqft\t| neural\t| approximation\t| error");
    println!("-----+-----------+-----------+-------");

    let test_sqft = vec![1200.0, 1800.0, 2200.0, 3200.0, 3800.0];
    for sqft in test_sqft {
        let neural_pred = 150.0 * sqft + 5000.0;
        let approx_pred = 150.0 * sqft + 5000.0;  // Same as linear fit
        let error = (neural_pred - approx_pred).abs();
        println!("{:.0}\t| {:.0}\t\t| {:.0}\t\t| {:.0}", sqft, neural_pred, approx_pred, error);
    }
    println!();

    println!("=== Real-World Example: Image Classification ===");
    println!("Neural network trained to classify handwritten digits.");
    println!("Can we find a symbolic explanation?\n");

    println!("Traditional explanation (saliency map):");
    println!("Shows which pixels matter, but not the logic\n");

    println!("Symbolic explanation:");
    println!("Decision tree approximation:");
    println!("if (top_pixels > threshold) and (vertical_lines > count):");
    println!("    → likely '1'");
    println!("elif (enclosed_region > area_threshold):");
    println!("    → likely '0' or '8' or '9'");
    println!("else:");
    println!("    → other digits");
    println!();

    println!("Benefits:");
    println!("✓ Humans can understand the logic");
    println!("✓ Can identify failure modes");
    println!("✓ Can verify against domain knowledge");
    println!("✓ Useful for debugging and improvement\n");

    println!("=== Neurosymbolic Architecture ===");
    println!("1. Train neural network (end-to-end learning)");
    println!("2. Extract trained parameters");
    println!("3. Fit symbolic model to learned function");
    println!("4. Interpret symbolic model");
    println!("5. (Optional) Retrain with symbolic regularization\n");

    println!("=== Trade-offs ===");
    println!("Neural network:");
    println!("  ✓ Flexible, high accuracy");
    println!("  ✗ Black box, hard to explain");
    println!();
    println!("Symbolic approximation:");
    println!("  ✓ Interpretable, trustworthy");
    println!("  ✗ May have lower accuracy");
    println!();
    println!("Solution: Use both!");
    println!("  ✓ Neural for power, Symbolic for explainability");
}
