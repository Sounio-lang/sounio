// PUCT Epistemic em L0 Puro
// Manual recursion, linear ownership, no high-level loops
// Novelty: epistemic bonus = sqrt(variance) * c_ignorance
// Explores branches where "I don't know" (high variance = high ignorance)

module epistemic_mcts

// Basic type aliases (semicolons required)
type Visits = i64;
type Confidence = f64;   // mean of Beta posterior
type Variance = f64;     // epistemic uncertainty measure
type Action = i64;       // action index

// Beta update result (since tuple destructuring may not work)
struct BetaResult {
    mean: f64,
    variance: f64
}

// MCTS Node with epistemic state
struct MCTSNode {
    prior_mean: f64,       // P from policy network
    value_mean: f64,       // Q mean
    value_variance: f64,   // ignorance measure
    visits: f64,           // use f64 to avoid cast issues
    parent_visits: f64     // cached for speed
}

// Core epistemic PUCT formula
fn puct_epistemic(node: MCTSNode, c_exploration: f64, c_ignorance: f64) -> f64 {
    // Q term (exploitation)
    let q = node.value_mean

    // U term classic (PUCT exploration)
    let parent_sqrt = sqrt(node.parent_visits)
    let visit_factor = 1.0 + node.visits
    let u = node.prior_mean * parent_sqrt / visit_factor
    let exploitation_bonus = c_exploration * u

    // Epistemic bonus: sqrt(variance) for active inference
    // High variance = high ignorance = explore more
    let safe_variance = max(node.value_variance, 0.0001)  // avoid sqrt(0)
    let epistemic_bonus = c_ignorance * sqrt(safe_variance)

    // Total score: exploit + explore + epistemic
    q + exploitation_bonus + epistemic_bonus
}

// Helper: max of two f64
fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// Helper: sqrt approximation (Newton-Raphson, 5 iterations)
fn sqrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0
    }
    // Initial guess
    let mut guess = x / 2.0
    // Newton-Raphson iterations (manual unroll for L0)
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess
}

// Beta distribution update after observing outcome
fn beta_update(old_mean: f64, old_variance: f64, outcome: f64, weight: f64) -> BetaResult {
    // Approximate Beta update via moment matching
    // outcome in [0,1], weight controls learning rate

    // Current effective sample size (from variance)
    let safe_var = max(old_variance, 0.0001)
    let n_eff = old_mean * (1.0 - old_mean) / safe_var - 1.0
    let n_eff_safe = max(n_eff, 1.0)

    // Update mean (weighted average)
    let new_mean = (old_mean * n_eff_safe + outcome * weight) / (n_eff_safe + weight)

    // Update variance (shrinks with more observations)
    let new_n = n_eff_safe + weight
    let new_variance = new_mean * (1.0 - new_mean) / (new_n + 1.0)

    BetaResult { mean: new_mean, variance: new_variance }
}

// Backpropagate outcome through path (recursive, L0 style)
fn backprop_node(node: MCTSNode, outcome: f64) -> MCTSNode {
    // Update visits
    let new_visits = node.visits + 1.0

    // Update value with Beta update
    let result = beta_update(
        node.value_mean,
        node.value_variance,
        outcome,
        1.0  // weight = 1 per visit
    )

    // Return updated node
    MCTSNode {
        prior_mean: node.prior_mean,
        value_mean: result.mean,
        value_variance: result.variance,
        visits: new_visits,
        parent_visits: node.parent_visits
    }
}

// Test: Create a toy tree and compute PUCT scores
fn test_puct_scores() -> f64 {
    // Node with high confidence (low variance) - should have low epistemic bonus
    let confident_node = MCTSNode {
        prior_mean: 0.3,
        value_mean: 0.7,
        value_variance: 0.01,   // low variance = confident
        visits: 10.0,
        parent_visits: 100.0
    }

    // Node with low confidence (high variance) - should have high epistemic bonus
    let uncertain_node = MCTSNode {
        prior_mean: 0.3,
        value_mean: 0.5,
        value_variance: 0.25,   // high variance = uncertain
        visits: 10.0,
        parent_visits: 100.0
    }

    // Standard params
    let c_exploration = 1.4   // standard PUCT
    let c_ignorance = 0.8     // epistemic weight

    let score_confident = puct_epistemic(confident_node, c_exploration, c_ignorance)
    let score_uncertain = puct_epistemic(uncertain_node, c_exploration, c_ignorance)

    // The uncertain node should have higher score due to epistemic bonus
    // even though its value_mean is lower
    let epistemic_advantage = score_uncertain - score_confident

    epistemic_advantage
}

// Test: Beta update preserves reasonable bounds
fn test_beta_update() -> f64 {
    let initial_mean = 0.5
    let initial_variance = 0.25  // maximum variance for Beta

    // Simulate wins (chain of updates)
    let r1 = beta_update(initial_mean, initial_variance, 1.0, 1.0)
    let r2 = beta_update(r1.mean, r1.variance, 1.0, 1.0)
    let r3 = beta_update(r2.mean, r2.variance, 1.0, 1.0)

    // After 3 wins, mean should increase, variance should decrease
    let mean_increased = r3.mean - initial_mean
    let variance_decreased = initial_variance - r3.variance

    // Return combined metric (both should be positive)
    mean_increased + variance_decreased
}

// Main test entry point
fn main() -> f64 {
    // Test PUCT epistemic scoring
    let puct_test = test_puct_scores()
    print("Epistemic PUCT advantage (uncertain vs confident): ")
    print(puct_test)

    // Test Beta update
    let beta_test = test_beta_update()
    print("Beta update quality metric: ")
    print(beta_test)

    // Combined result
    let total = puct_test + beta_test
    print("Total test score (should be positive): ")
    print(total)

    total
}
