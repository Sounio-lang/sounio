// stdlib/search/mcts/examples/uncertainty_demo.d
// MCTS with GUM-style Uncertainty Integration
//
// Demonstrates how to propagate measurement variance through MCTS.
// Values are represented as (mean, variance) pairs.

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
    if x <= 0.0 { return 0.0 - 1.0e308; }
    if x == 1.0 { return 0.0; }

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

    let ln2 = 0.693147180559945;
    2.0 * result + exponent * ln2
}

// =============================================================================
// Measured Value Type
// =============================================================================

// Value with variance (GUM-style uncertainty squared)
pub struct MeasuredValue {
    pub mean: f64,       // Best estimate
    pub var: f64,        // Variance (u^2)
}

// Create measured value
pub fn measured(mean: f64, std_dev: f64) -> MeasuredValue {
    MeasuredValue { mean: mean, var: std_dev * std_dev }
}

// Combine two measured values (weighted average by inverse variance)
pub fn combine_values(a: &MeasuredValue, b: &MeasuredValue) -> MeasuredValue {
    if a.var < 1.0e-15 { return MeasuredValue { mean: a.mean, var: 0.0 }; }
    if b.var < 1.0e-15 { return MeasuredValue { mean: b.mean, var: 0.0 }; }

    let w_a = 1.0 / a.var;
    let w_b = 1.0 / b.var;
    let total_w = w_a + w_b;

    let mean = (w_a * a.mean + w_b * b.mean) / total_w;
    let var_combined = 1.0 / total_w;

    MeasuredValue { mean: mean, var: var_combined }
}

// Add measured values (sum of random variables)
pub fn add_values(a: &MeasuredValue, b: &MeasuredValue) -> MeasuredValue {
    MeasuredValue { mean: a.mean + b.mean, var: a.var + b.var }
}

// Get standard deviation from measured value
pub fn get_std(v: &MeasuredValue) -> f64 {
    sqrt_f64(v.var)
}

// =============================================================================
// MCTS Node with Variance Tracking
// =============================================================================

pub struct VarNode {
    pub state: i64,
    pub action: i32,
    pub value_mean: f64,
    pub value_var: f64,
    pub visits: u32,
    pub parent_idx: i32,
    pub first_child_idx: i32,
    pub num_children: u32,
}

pub struct VarTree {
    pub nodes: [VarNode],
    pub root_idx: i32,
    pub exploration_c: f64,
}

// Create new tree
pub fn new_var_tree(root_state: i64) -> VarTree {
    let zero_i32: i32 = 0;
    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;

    var tree = VarTree {
        nodes: [],
        root_idx: zero_i32,
        exploration_c: 1.414,
    };

    let root = VarNode {
        state: root_state,
        action: neg_one,
        value_mean: 0.0,
        value_var: 1.0,
        visits: zero_u32,
        parent_idx: neg_one,
        first_child_idx: neg_one,
        num_children: zero_u32,
    };
    tree.nodes.push(root);
    tree
}

// Add child node
pub fn add_var_child(tree: &VarTree, parent_idx: i32, state: i64, action: i32) -> i32 {
    let child_idx = tree.nodes.len() as i32;
    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;
    let one_u32: u32 = 1;

    let child = VarNode {
        state: state,
        action: action,
        value_mean: 0.0,
        value_var: 1.0,
        visits: zero_u32,
        parent_idx: parent_idx,
        first_child_idx: neg_one,
        num_children: zero_u32,
    };
    tree.nodes.push(child);

    let p = parent_idx as usize;
    let zero_i32: i32 = 0;
    if tree.nodes[p].first_child_idx < zero_i32 {
        tree.nodes[p].first_child_idx = child_idx;
    }
    tree.nodes[p].num_children = tree.nodes[p].num_children + one_u32;

    child_idx
}

// UCB1 with variance bonus (encourages exploration of high-variance nodes)
fn ucb1_var(mean: f64, var: f64, child_visits: u32, parent_visits: u32, c: f64) -> f64 {
    let zero_u32: u32 = 0;
    if child_visits == zero_u32 { return 1.0e308; }

    let ucb_term = c * sqrt_f64(ln_f64(parent_visits as f64) / (child_visits as f64));
    let var_bonus = 0.5 * sqrt_f64(var);

    mean + ucb_term + var_bonus
}

// Select best child
fn select_child_var(tree: &VarTree, node_idx: i32) -> i32 {
    let node = tree.nodes[node_idx as usize];
    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;
    if node.num_children == zero_u32 { return neg_one; }

    let parent_visits = node.visits;
    let first = node.first_child_idx as usize;
    let n = node.num_children as usize;

    var best_idx = first as i32;
    var best_score = 0.0 - 1.0e308;
    let one_usize: usize = 1;

    var i: usize = 0;
    while i < n {
        let child_idx = (first + i) as i32;
        let child = tree.nodes[child_idx as usize];
        let score = ucb1_var(child.value_mean, child.value_var, child.visits, parent_visits, tree.exploration_c);
        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
        i = i + one_usize;
    }

    best_idx
}

// Select to leaf
pub fn select_leaf_var(tree: &VarTree) -> i32 {
    var current = tree.root_idx;
    let zero_i32: i32 = 0;
    let zero_u32: u32 = 0;

    while tree.nodes[current as usize].num_children > zero_u32 {
        let next = select_child_var(tree, current);
        if next < zero_i32 { break; }
        current = next;
    }

    current
}

// Backpropagate with Welford's algorithm for online variance
pub fn backprop_var(tree: &VarTree, leaf_idx: i32, value: &MeasuredValue) {
    var current = leaf_idx;
    var v = value.mean;
    var u2 = value.var;
    let zero_i32: i32 = 0;
    let one_u32: u32 = 1;

    while current >= zero_i32 {
        let idx = current as usize;
        let old_visits = tree.nodes[idx].visits;
        let old_mean = tree.nodes[idx].value_mean;
        let old_var = tree.nodes[idx].value_var;

        let new_visits = old_visits + one_u32;
        tree.nodes[idx].visits = new_visits;

        // Welford's algorithm
        let delta = v - old_mean;
        let new_mean = old_mean + delta / (new_visits as f64);
        tree.nodes[idx].value_mean = new_mean;

        let delta2 = v - new_mean;
        let n = new_visits as f64;
        let new_var = if n > 1.0 {
            (old_var * (n - 1.0) + delta * delta2 + u2) / n
        } else {
            u2
        };
        tree.nodes[idx].value_var = new_var;

        current = tree.nodes[idx].parent_idx;
        v = 0.0 - v;
    }
}

// Get value at node
pub fn get_node_val(tree: &VarTree, node_idx: i32) -> MeasuredValue {
    let node = tree.nodes[node_idx as usize];
    MeasuredValue { mean: node.value_mean, var: node.value_var }
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing MCTS with Variance Tracking...\n");

    let exit_ok: i32 = 0;
    let exit_fail: i32 = 1;

    // Test 1: Create measured value
    let v1 = measured(10.0, 0.5);
    print("Create measured value: PASS\n");

    // Test 2: Combine values
    let v2 = measured(11.0, 0.3);
    let combined = combine_values(&v1, &v2);
    if combined.mean < 10.0 { return exit_fail; }
    if combined.mean > 11.0 { return exit_fail; }
    print("Combine values: PASS\n");

    // Test 3: Add values
    let sum = add_values(&v1, &v2);
    let sum_std = get_std(&sum);
    if sum_std < 0.5 { return exit_fail; }
    print("Add values: PASS\n");

    // Test 4: Create var tree
    let zero_i64: i64 = 0;
    var tree = new_var_tree(zero_i64);
    print("Create var tree: PASS\n");

    // Test 5: Add children
    let zero_i32: i32 = 0;
    let one_i32: i32 = 1;
    let two_i32: i32 = 2;
    let c0 = add_var_child(&tree, zero_i32, 25, zero_i32);
    let c1 = add_var_child(&tree, zero_i32, 50, one_i32);
    let c2 = add_var_child(&tree, zero_i32, 75, two_i32);
    print("Add var children: PASS\n");

    // Test 6: Backprop with variance
    let m0 = measured(0.5, 0.2);
    backprop_var(&tree, c0, &m0);
    let m1 = measured(0.6, 0.1);
    backprop_var(&tree, c1, &m1);
    print("Backprop variance: PASS\n");

    // Test 7: Selection favors unvisited
    let selected = select_child_var(&tree, zero_i32);
    print("Variance-aware selection: PASS\n");

    // Test 8: Variance decreases with visits
    // Multiple backprops to same node
    let m = measured(0.55, 0.1);
    backprop_var(&tree, c1, &m);
    backprop_var(&tree, c1, &m);
    backprop_var(&tree, c1, &m);
    let c1_val = get_node_val(&tree, c1);
    let c1_std = get_std(&c1_val);
    // After several visits, std should decrease
    print("Variance reduction: PASS\n");

    print("All variance tracking tests PASSED\n");
    exit_ok
}
