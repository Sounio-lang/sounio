/// Wave 1 Example: Bayesian Coin Flip Inference
///
/// Demonstrates probabilistic inference with distributions.
/// We observe coin flips and update our belief about fairness.
///
/// This example shows:
/// - Working with probability distributions
/// - Sampling from distributions
/// - Bayesian reasoning

fn main() {
    println("=== Bayesian Coin Flip Inference ===");

    // Prior belief: the coin is fair-ish (mean 0.5, std 0.1)
    let prior = Distribution::Normal(0.5, 0.1);

    println!("Prior belief about coin bias:");
    println!("  Mean: 0.5");
    println!("  Std Dev: 0.1");

    // Simulate some coin flips
    let num_flips = 100;
    let true_bias = 0.6;  // The coin is actually biased towards heads
    let heads_count = count_heads(true_bias, num_flips);

    println!("\nObserved {} heads in {} flips ({:.1}%)",
             heads_count, num_flips, 100.0 * heads_count as f64 / num_flips as f64);

    // Posterior distribution would be Beta distributed
    // For now, simulate with samples from the prior
    let posterior = Distribution::Beta(
        0.5 + heads_count as f64,      // alpha
        0.5 + (num_flips - heads_count) as f64  // beta
    );

    println!("\nPosterior distribution (Beta):");
    println!("  alpha: {}", 0.5 + heads_count as f64);
    println!("  beta: {}", 0.5 + (num_flips - heads_count) as f64);

    // Sample from posterior
    let posterior_mean = sample(posterior);
    println!("\nEstimated coin bias: {:.3}", posterior_mean);
    println!("True bias was: {:.1}", true_bias);
}

fn count_heads(bias: f64, n: i64) -> i64 {
    // Simple simulation: count how many times a random number < bias
    let mut count = 0;
    for i in 0..n {
        // Pseudo-random: use sin(i) to generate values
        let random = (sin(i as f64 * 3.7) + 1.0) / 2.0;  // Normalize to [0, 1]
        if random < bias {
            count = count + 1;
        }
    }
    return count;
}

fn sin(x: f64) -> f64 {
    // TODO: Use built-in sin
    // For now, use a simple approximation
    return x - (x * x * x) / 6.0;  // Taylor series approximation
}

// Distribution type for example
struct Distribution {
    type_name: string,
    param1: f64,
    param2: f64,
}

fn Distribution::Normal(mean: f64, std: f64) -> Distribution {
    return Distribution {
        type_name: "Normal",
        param1: mean,
        param2: std,
    };
}

fn Distribution::Beta(alpha: f64, beta: f64) -> Distribution {
    return Distribution {
        type_name: "Beta",
        param1: alpha,
        param2: beta,
    };
}
