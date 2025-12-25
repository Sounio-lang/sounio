// Full Epistemic MCTS with Self-Play Loop
// L0 Pure: manual recursion, linear ownership, no high-level loops
// Features:
//   - Epistemic PUCT (exploration via ignorance)
//   - Beta posterior updates (Bayesian value estimates)
//   - Tree structure with children
//   - Full select -> expand -> evaluate -> backprop cycle
//   - Self-play game simulation

module epistemic_mcts_full

// =============================================================================
// Core Types
// =============================================================================

struct BetaResult {
    mean: f64,
    variance: f64
}

// MCTS Node with epistemic state
struct MCTSNode {
    prior: f64,            // P from policy network
    value_mean: f64,       // Q mean (Beta posterior)
    value_variance: f64,   // epistemic uncertainty
    visits: f64,
    // For simplicity, we use fixed-size children (actions 0-3)
    // In real impl, would use dynamic Map
    child0_visits: f64,
    child1_visits: f64,
    child2_visits: f64,
    child3_visits: f64,
    child0_value: f64,
    child1_value: f64,
    child2_value: f64,
    child3_value: f64,
    child0_var: f64,
    child1_var: f64,
    child2_var: f64,
    child3_var: f64,
    is_terminal: f64,      // 1.0 if terminal, 0.0 otherwise
    terminal_value: f64    // outcome if terminal
}

// Game state (simple 1D position game for demo)
struct Gst {
    position: f64,         // position on line [-10, 10]
    turn: f64,             // 0 = player 1, 1 = player 2
    moves_left: f64        // max moves before draw
}

// =============================================================================
// Math Helpers
// =============================================================================

fn max(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn sqrt(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut guess = x / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess = (guess + x / guess) / 2.0
    guess
}

// =============================================================================
// Beta Distribution Updates
// =============================================================================

fn beta_update(old_mean: f64, old_variance: f64, outcome: f64, weight: f64) -> BetaResult {
    let safe_var = max(old_variance, 0.0001)
    let n_eff = old_mean * (1.0 - old_mean) / safe_var - 1.0
    let n_eff_safe = max(n_eff, 1.0)

    let new_mean = (old_mean * n_eff_safe + outcome * weight) / (n_eff_safe + weight)
    let new_n = n_eff_safe + weight
    let new_variance = new_mean * (1.0 - new_mean) / (new_n + 1.0)

    BetaResult { mean: new_mean, variance: new_variance }
}

// =============================================================================
// PUCT with Epistemic Bonus
// =============================================================================

fn puct_score(
    child_value: f64,
    child_variance: f64,
    child_visits: f64,
    parent_visits: f64,
    prior: f64,
    c_puct: f64,
    c_epistemic: f64
) -> f64 {
    // Q term
    let q = child_value

    // U term (exploration)
    let u = prior * sqrt(parent_visits) / (1.0 + child_visits)

    // Epistemic bonus (active inference)
    let epistemic = sqrt(max(child_variance, 0.0001))

    q + c_puct * u + c_epistemic * epistemic
}

// Select best child action (0-3)
fn select_action(node: MCTSNode, c_puct: f64, c_epistemic: f64) -> f64 {
    let parent_visits = node.visits
    let prior = 0.25  // uniform prior for simplicity

    let score0 = puct_score(node.child0_value, node.child0_var, node.child0_visits, parent_visits, prior, c_puct, c_epistemic)
    let score1 = puct_score(node.child1_value, node.child1_var, node.child1_visits, parent_visits, prior, c_puct, c_epistemic)
    let score2 = puct_score(node.child2_value, node.child2_var, node.child2_visits, parent_visits, prior, c_puct, c_epistemic)
    let score3 = puct_score(node.child3_value, node.child3_var, node.child3_visits, parent_visits, prior, c_puct, c_epistemic)

    // Find max (manual comparison)
    let mut best_action = 0.0
    let mut best_score = score0

    if score1 > best_score {
        best_score = score1
        best_action = 1.0
    }
    if score2 > best_score {
        best_score = score2
        best_action = 2.0
    }
    if score3 > best_score {
        best_action = 3.0
    }

    best_action
}

// =============================================================================
// Game Logic
// =============================================================================

fn make_init_st() -> Gst {
    Gst {
        position: 0.0,
        turn: 0.0,
        moves_left: 10.0
    }
}

fn apply_action(st: Gst, action: f64) -> Gst {
    // Actions: 0 = left 2, 1 = left 1, 2 = right 1, 3 = right 2
    let delta = if action < 0.5 {
        0.0 - 2.0
    } else if action < 1.5 {
        0.0 - 1.0
    } else if action < 2.5 {
        1.0
    } else {
        2.0
    }

    // Player 2 moves in opposite direction (zero-sum)
    let actual_delta = if st.turn > 0.5 { 0.0 - delta } else { delta }

    let new_position = st.position + actual_delta
    let clamped = max(0.0 - 10.0, min(10.0, new_position))

    Gst {
        position: clamped,
        turn: 1.0 - st.turn,  // flip turn
        moves_left: st.moves_left - 1.0
    }
}

fn is_terminal(st: Gst) -> f64 {
    // Terminal if position at boundary or no moves left
    if abs(st.position) > 9.5 { 1.0 }
    else if st.moves_left < 0.5 { 1.0 }
    else { 0.0 }
}

fn get_value(st: Gst) -> f64 {
    // Value from player 1's perspective
    // Win if position > 9, lose if < -9, else draw
    if st.position > 9.5 { 1.0 }
    else if st.position < 0.0 - 9.5 { 0.0 }
    else { 0.5 }  // draw
}

// =============================================================================
// MCTS Operations
// =============================================================================

fn make_init_node() -> MCTSNode {
    MCTSNode {
        prior: 0.25,
        value_mean: 0.5,
        value_variance: 0.25,  // max uncertainty
        visits: 0.0,
        child0_visits: 0.0,
        child1_visits: 0.0,
        child2_visits: 0.0,
        child3_visits: 0.0,
        child0_value: 0.5,
        child1_value: 0.5,
        child2_value: 0.5,
        child3_value: 0.5,
        child0_var: 0.25,
        child1_var: 0.25,
        child2_var: 0.25,
        child3_var: 0.25,
        is_terminal: 0.0,
        terminal_value: 0.5
    }
}

// Update a specific child's stats
fn update_child(node: MCTSNode, action: f64, outcome: f64) -> MCTSNode {
    if action < 0.5 {
        let r = beta_update(node.child0_value, node.child0_var, outcome, 1.0)
        MCTSNode {
            prior: node.prior,
            value_mean: node.value_mean,
            value_variance: node.value_variance,
            visits: node.visits,
            child0_visits: node.child0_visits + 1.0,
            child1_visits: node.child1_visits,
            child2_visits: node.child2_visits,
            child3_visits: node.child3_visits,
            child0_value: r.mean,
            child1_value: node.child1_value,
            child2_value: node.child2_value,
            child3_value: node.child3_value,
            child0_var: r.variance,
            child1_var: node.child1_var,
            child2_var: node.child2_var,
            child3_var: node.child3_var,
            is_terminal: node.is_terminal,
            terminal_value: node.terminal_value
        }
    } else if action < 1.5 {
        let r = beta_update(node.child1_value, node.child1_var, outcome, 1.0)
        MCTSNode {
            prior: node.prior,
            value_mean: node.value_mean,
            value_variance: node.value_variance,
            visits: node.visits,
            child0_visits: node.child0_visits,
            child1_visits: node.child1_visits + 1.0,
            child2_visits: node.child2_visits,
            child3_visits: node.child3_visits,
            child0_value: node.child0_value,
            child1_value: r.mean,
            child2_value: node.child2_value,
            child3_value: node.child3_value,
            child0_var: node.child0_var,
            child1_var: r.variance,
            child2_var: node.child2_var,
            child3_var: node.child3_var,
            is_terminal: node.is_terminal,
            terminal_value: node.terminal_value
        }
    } else if action < 2.5 {
        let r = beta_update(node.child2_value, node.child2_var, outcome, 1.0)
        MCTSNode {
            prior: node.prior,
            value_mean: node.value_mean,
            value_variance: node.value_variance,
            visits: node.visits,
            child0_visits: node.child0_visits,
            child1_visits: node.child1_visits,
            child2_visits: node.child2_visits + 1.0,
            child3_visits: node.child3_visits,
            child0_value: node.child0_value,
            child1_value: node.child1_value,
            child2_value: r.mean,
            child3_value: node.child3_value,
            child0_var: node.child0_var,
            child1_var: node.child1_var,
            child2_var: r.variance,
            child3_var: node.child3_var,
            is_terminal: node.is_terminal,
            terminal_value: node.terminal_value
        }
    } else {
        let r = beta_update(node.child3_value, node.child3_var, outcome, 1.0)
        MCTSNode {
            prior: node.prior,
            value_mean: node.value_mean,
            value_variance: node.value_variance,
            visits: node.visits,
            child0_visits: node.child0_visits,
            child1_visits: node.child1_visits,
            child2_visits: node.child2_visits,
            child3_visits: node.child3_visits + 1.0,
            child0_value: node.child0_value,
            child1_value: node.child1_value,
            child2_value: node.child2_value,
            child3_value: r.mean,
            child0_var: node.child0_var,
            child1_var: node.child1_var,
            child2_var: node.child2_var,
            child3_var: r.variance,
            is_terminal: node.is_terminal,
            terminal_value: node.terminal_value
        }
    }
}

// Update root node stats
fn update_root(node: MCTSNode, outcome: f64) -> MCTSNode {
    let r = beta_update(node.value_mean, node.value_variance, outcome, 1.0)
    MCTSNode {
        prior: node.prior,
        value_mean: r.mean,
        value_variance: r.variance,
        visits: node.visits + 1.0,
        child0_visits: node.child0_visits,
        child1_visits: node.child1_visits,
        child2_visits: node.child2_visits,
        child3_visits: node.child3_visits,
        child0_value: node.child0_value,
        child1_value: node.child1_value,
        child2_value: node.child2_value,
        child3_value: node.child3_value,
        child0_var: node.child0_var,
        child1_var: node.child1_var,
        child2_var: node.child2_var,
        child3_var: node.child3_var,
        is_terminal: node.is_terminal,
        terminal_value: node.terminal_value
    }
}

// =============================================================================
// Single MCTS Simulation (1-ply for simplicity)
// =============================================================================

fn mcts_simulate(node: MCTSNode, st: Gst, c_puct: f64, c_epistemic: f64) -> MCTSNode {
    // Select action using epistemic PUCT
    let action = select_action(node, c_puct, c_epistemic)

    // Apply action to get new state
    let nst = apply_action(st, action)

    // Evaluate (simple: use terminal value or heuristic)
    let value = if is_terminal(nst) > 0.5 {
        get_value(nst)
    } else {
        // Heuristic: position normalized to [0,1]
        (nst.position + 10.0) / 20.0
    }

    // Flip value for player 2's perspective
    let adjusted_value = if st.turn > 0.5 { 1.0 - value } else { value }

    // Backpropagate: update child and root
    let node1 = update_child(node, action, adjusted_value)
    let node2 = update_root(node1, adjusted_value)

    node2
}

// Run N simulations (recursive, L0 style)
fn run_simulations_rec(node: MCTSNode, st: Gst, n: f64, c_puct: f64, c_epistemic: f64) -> MCTSNode {
    if n < 0.5 {
        return node
    }
    let updated = mcts_simulate(node, st, c_puct, c_epistemic)
    run_simulations_rec(updated, st, n - 1.0, c_puct, c_epistemic)
}

fn run_simulations(node: MCTSNode, st: Gst, n: f64) -> MCTSNode {
    let c_puct = 1.4
    let c_epistemic = 0.8
    run_simulations_rec(node, st, n, c_puct, c_epistemic)
}

// =============================================================================
// Get Best Action (most visited)
// =============================================================================

fn get_best_action(node: MCTSNode) -> f64 {
    let mut best = 0.0
    let mut best_visits = node.child0_visits

    if node.child1_visits > best_visits {
        best = 1.0
        best_visits = node.child1_visits
    }
    if node.child2_visits > best_visits {
        best = 2.0
        best_visits = node.child2_visits
    }
    if node.child3_visits > best_visits {
        best = 3.0
    }

    best
}

// =============================================================================
// Self-Play Game (recursive)
// =============================================================================

fn self_play_step(st: Gst, step: f64) -> f64 {
    if is_terminal(st) > 0.5 {
        return get_value(st)
    }
    if step > 20.0 {
        return 0.5  // draw by max steps
    }

    // Run MCTS from current state
    let root = make_init_node()
    let searched = run_simulations(root, st, 50.0)  // 50 sims per move

    // Select best action
    let action = get_best_action(searched)

    // Apply and continue
    let nst = apply_action(st, action)

    // Recurse
    self_play_step(nst, step + 1.0)
}

fn self_play() -> f64 {
    let init_st = make_init_st()
    self_play_step(init_st, 0.0)
}

// =============================================================================
// Tests
// =============================================================================

fn test_puct_epistemic() -> f64 {
    // High variance should get higher score
    let score_low_var = puct_score(0.5, 0.01, 5.0, 100.0, 0.25, 1.4, 0.8)
    let score_high_var = puct_score(0.5, 0.25, 5.0, 100.0, 0.25, 1.4, 0.8)

    // High variance should win
    score_high_var - score_low_var
}

fn test_mcts_reduces_variance() -> f64 {
    let init_node = make_init_node()
    let st = make_init_st()

    // Initial variance
    let var_before = init_node.value_variance

    // Run some simulations
    let after = run_simulations(init_node, st, 20.0)

    // Variance should decrease
    let var_after = after.value_variance

    var_before - var_after  // should be positive
}

fn test_game_mechanics() -> f64 {
    let st = make_init_st()

    // Action 3 = right 2
    let s1 = apply_action(st, 3.0)

    // Position should be 2.0
    let correct_pos = abs(s1.position - 2.0) < 0.1

    // Turn should flip
    let correct_turn = abs(s1.turn - 1.0) < 0.1

    if correct_pos { 1.0 } else { 0.0 } + if correct_turn { 1.0 } else { 0.0 }
}

// =============================================================================
// Main
// =============================================================================

fn main() -> f64 {
    print("=== Epistemic MCTS Full Test ===")
    print("")

    // Test 1: PUCT prefers uncertainty
    let puct_test = test_puct_epistemic()
    print("PUCT epistemic advantage: ")
    print(puct_test)

    // Test 2: MCTS reduces variance
    let variance_test = test_mcts_reduces_variance()
    print("Variance reduction: ")
    print(variance_test)

    // Test 3: Game mechanics
    let game_test = test_game_mechanics()
    print("Game mechanics score (should be 2): ")
    print(game_test)

    // Test 4: Self-play game
    print("")
    print("Running self-play game...")
    let game_result = self_play()
    print("Game result (1=P1 win, 0=P2 win, 0.5=draw): ")
    print(game_result)

    // Summary
    print("")
    print("=== All Tests Complete ===")

    puct_test + variance_test + game_test
}
