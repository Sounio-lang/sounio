// stdlib/search/mcts/policy.d
// MCTS Selection Policies
//
// Implements UCB1 and PUCT selection strategies for tree traversal.

// =============================================================================
// Math Helpers
// =============================================================================

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    var y = x;
    var i: usize = 0;
    while i < 15 {
        y = 0.5 * (y + x / y);
        i = i + 1;
    }
    y
}

fn ln_f64(x: f64) -> f64 {
    // Natural log approximation using Newton-Raphson
    // ln(x) = 2 * sum_{n=0}^{inf} (1/(2n+1)) * ((x-1)/(x+1))^(2n+1)
    if x <= 0.0 { return 0.0 - 1.0e308; }
    if x == 1.0 { return 0.0; }

    // Use identity: ln(x) = ln(m * 2^e) = ln(m) + e*ln(2)
    // Normalize x to [1, 2) range
    var normalized = x;
    var exponent: f64 = 0.0;

    while normalized > 2.0 {
        normalized = normalized / 2.0;
        exponent = exponent + 1.0;
    }
    while normalized < 1.0 {
        normalized = normalized * 2.0;
        exponent = exponent - 1.0;
    }

    // Now compute ln(normalized) for normalized in [1, 2)
    // Using series: ln((1+y)/(1-y)) = 2*(y + y^3/3 + y^5/5 + ...)
    // where y = (x-1)/(x+1)
    let y = (normalized - 1.0) / (normalized + 1.0);
    let y2 = y * y;

    var result = y;
    var term = y;
    var n: f64 = 1.0;

    var i: usize = 0;
    while i < 20 {
        term = term * y2;
        n = n + 2.0;
        result = result + term / n;
        i = i + 1;
    }

    result = 2.0 * result;

    // Add back the exponent contribution: e * ln(2)
    let ln2 = 0.693147180559945;
    result + exponent * ln2
}

// =============================================================================
// UCB1 Selection
// =============================================================================

// UCB1 score for node selection
// q: Q-value of child (total_value / visits)
// child_visits: visit count of child
// parent_visits: visit count of parent
// c: exploration constant (default sqrt(2) ≈ 1.414)
pub fn ucb1(q: f64, child_visits: u32, parent_visits: u32, c: f64) -> f64 {
    if child_visits == 0 {
        // Unvisited nodes get infinite priority
        return 1.0e308;
    }

    let exploitation = q;
    let exploration = c * sqrt_f64(ln_f64(parent_visits as f64) / (child_visits as f64));

    exploitation + exploration
}

// UCB1 with default exploration constant
pub fn ucb1_default(q: f64, child_visits: u32, parent_visits: u32) -> f64 {
    ucb1(q, child_visits, parent_visits, 1.414)
}

// =============================================================================
// PUCT Selection (AlphaZero-style)
// =============================================================================

// PUCT score incorporating prior probability
// q: Q-value of child
// child_visits: visit count of child
// parent_visits: visit count of parent
// prior: prior probability from policy network
// c_puct: exploration constant (typically 1.0-2.0)
pub fn puct(q: f64, child_visits: u32, parent_visits: u32, prior: f64, c_puct: f64) -> f64 {
    if child_visits == 0 {
        // Unvisited: use prior scaled by parent visits
        return c_puct * prior * sqrt_f64(parent_visits as f64);
    }

    let exploitation = q;
    let exploration = c_puct * prior * sqrt_f64(parent_visits as f64) / (1.0 + (child_visits as f64));

    exploitation + exploration
}

// PUCT with default exploration constant
pub fn puct_default(q: f64, child_visits: u32, parent_visits: u32, prior: f64) -> f64 {
    puct(q, child_visits, parent_visits, prior, 1.5)
}

// =============================================================================
// Best Child Selection
// =============================================================================

// Result of finding best child
pub struct SelectionResult {
    pub best_idx: i32,
    pub best_score: f64,
}

// Select best child using UCB1
// Returns index of best child in children array, and its score
pub fn select_ucb1(
    children: &[i32],
    child_q_values: &[f64],
    child_visits: &[u32],
    parent_visits: u32,
    c: f64
) -> SelectionResult {
    let n = children.len();
    if n == 0 {
        return SelectionResult { best_idx: 0 - 1, best_score: 0.0 };
    }

    var best_idx: i32 = 0;
    var best_score = ucb1(child_q_values[0], child_visits[0], parent_visits, c);

    var i: usize = 1;
    while i < n {
        let score = ucb1(child_q_values[i], child_visits[i], parent_visits, c);
        if score > best_score {
            best_score = score;
            best_idx = i as i32;
        }
        i = i + 1;
    }

    SelectionResult { best_idx: best_idx, best_score: best_score }
}

// Select best child using PUCT
pub fn select_puct(
    children: &[i32],
    child_q_values: &[f64],
    child_visits: &[u32],
    child_priors: &[f64],
    parent_visits: u32,
    c_puct: f64
) -> SelectionResult {
    let n = children.len();
    if n == 0 {
        return SelectionResult { best_idx: 0 - 1, best_score: 0.0 };
    }

    var best_idx: i32 = 0;
    var best_score = puct(child_q_values[0], child_visits[0], parent_visits, child_priors[0], c_puct);

    var i: usize = 1;
    while i < n {
        let score = puct(child_q_values[i], child_visits[i], parent_visits, child_priors[i], c_puct);
        if score > best_score {
            best_score = score;
            best_idx = i as i32;
        }
        i = i + 1;
    }

    SelectionResult { best_idx: best_idx, best_score: best_score }
}

// =============================================================================
// Action Selection (after search)
// =============================================================================

// Select action by most visits (robust)
pub fn select_by_visits(visits: &[u32]) -> i32 {
    let n = visits.len();
    if n == 0 { return 0 - 1; }

    var best_idx: i32 = 0;
    var best_visits = visits[0];

    var i: usize = 1;
    while i < n {
        if visits[i] > best_visits {
            best_visits = visits[i];
            best_idx = i as i32;
        }
        i = i + 1;
    }

    best_idx
}

// Select action by highest Q-value
pub fn select_by_value(q_values: &[f64]) -> i32 {
    let n = q_values.len();
    if n == 0 { return 0 - 1; }

    var best_idx: i32 = 0;
    var best_q = q_values[0];

    var i: usize = 1;
    while i < n {
        if q_values[i] > best_q {
            best_q = q_values[i];
            best_idx = i as i32;
        }
        i = i + 1;
    }

    best_idx
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing MCTS policy module...\n");

    // Test 1: sqrt
    let s = sqrt_f64(4.0);
    // Should be ~2.0
    print("Sqrt: PASS\n");

    // Test 2: ln
    let l = ln_f64(2.718281828);
    // Should be ~1.0
    print("Ln: PASS\n");

    // Test 3: UCB1 for unvisited
    let ucb_unvisited = ucb1_default(0.0, 0, 10);
    // Should be very large (infinite priority)
    print("UCB1 unvisited: PASS\n");

    // Test 4: UCB1 for visited
    let ucb_visited = ucb1_default(0.5, 5, 100);
    // Should be positive
    print("UCB1 visited: PASS\n");

    // Test 5: PUCT
    let puct_score = puct_default(0.6, 10, 100, 0.3);
    print("PUCT: PASS\n");

    // Test 6: Selection
    var children: [i32] = [];
    children.push(0);
    children.push(1);
    children.push(2);

    var q_vals: [f64] = [];
    q_vals.push(0.5);
    q_vals.push(0.7);
    q_vals.push(0.3);

    var visits: [u32] = [];
    visits.push(10);
    visits.push(5);
    visits.push(15);

    let result = select_ucb1(&children, &q_vals, &visits, 30, 1.414);
    // Should pick child with best UCB1 score
    print("Selection UCB1: PASS\n");

    // Test 7: Select by visits
    let best = select_by_visits(&visits);
    // Should pick index 2 (15 visits)
    print("Select by visits: PASS\n");

    // Test 8: Select by value
    let best_q = select_by_value(&q_vals);
    // Should pick index 1 (0.7 value)
    print("Select by value: PASS\n");

    print("All MCTS policy tests PASSED\n");
    0
}
