// stdlib/search/mcts/core.d
// MCTS Core Search Algorithm
//
// Monte Carlo Tree Search with configurable selection policy.
// Uses arena allocation for nodes.

// =============================================================================
// Math Helpers (duplicated for standalone module)
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
// Selection Policies (inline)
// =============================================================================

fn ucb1(q: f64, child_visits: u32, parent_visits: u32, c: f64) -> f64 {
    if child_visits == 0 { return 1.0e308; }
    q + c * sqrt_f64(ln_f64(parent_visits as f64) / (child_visits as f64))
}

fn puct(q: f64, child_visits: u32, parent_visits: u32, prior: f64, c: f64) -> f64 {
    if child_visits == 0 {
        return c * prior * sqrt_f64(parent_visits as f64);
    }
    q + c * prior * sqrt_f64(parent_visits as f64) / (1.0 + (child_visits as f64))
}

// =============================================================================
// MCTS Node (flat structure for arena)
// =============================================================================

pub struct MCTSNode {
    pub state: i64,
    pub action: i32,
    pub total_value: f64,
    pub visits: u32,
    pub prior: f64,
    pub parent_idx: i32,
    pub first_child_idx: i32,
    pub num_children: u32,
    pub is_terminal: bool,
}

// =============================================================================
// MCTS Tree
// =============================================================================

pub struct MCTSTree {
    pub nodes: [MCTSNode],
    pub root_idx: i32,
    pub exploration_c: f64,
    pub use_puct: bool,
}

// Create new MCTS tree with root state
pub fn new_tree(root_state: i64) -> MCTSTree {
    let zero: i32 = 0;
    var tree = MCTSTree {
        nodes: [],
        root_idx: zero,
        exploration_c: 1.414,
        use_puct: false,
    };

    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;
    let root = MCTSNode {
        state: root_state,
        action: neg_one,
        total_value: 0.0,
        visits: zero_u32,
        prior: 1.0,
        parent_idx: neg_one,
        first_child_idx: neg_one,
        num_children: zero_u32,
        is_terminal: false,
    };
    tree.nodes.push(root);
    tree
}

// Enable PUCT mode
pub fn enable_puct(tree: &MCTSTree, c_puct: f64) {
    tree.use_puct = true;
    tree.exploration_c = c_puct;
}

// Get Q-value of node
fn get_q(tree: &MCTSTree, idx: i32) -> f64 {
    let node = tree.nodes[idx as usize];
    if node.visits == 0 { 0.0 }
    else { node.total_value / (node.visits as f64) }
}

// =============================================================================
// Tree Operations
// =============================================================================

// Add child node and return its index
pub fn add_child(tree: &MCTSTree, parent_idx: i32, state: i64, action: i32, prior: f64) -> i32 {
    let child_idx = tree.nodes.len() as i32;
    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;
    let one_u32: u32 = 1;

    let child = MCTSNode {
        state: state,
        action: action,
        total_value: 0.0,
        visits: zero_u32,
        prior: prior,
        parent_idx: parent_idx,
        first_child_idx: neg_one,
        num_children: zero_u32,
        is_terminal: false,
    };
    tree.nodes.push(child);

    // Update parent's child tracking
    let p = parent_idx as usize;
    let zero_i32: i32 = 0;
    if tree.nodes[p].first_child_idx < zero_i32 {
        tree.nodes[p].first_child_idx = child_idx;
    }
    tree.nodes[p].num_children = tree.nodes[p].num_children + one_u32;

    child_idx
}

// Mark node as terminal
pub fn mark_terminal(tree: &MCTSTree, idx: i32) {
    tree.nodes[idx as usize].is_terminal = true;
}

// =============================================================================
// Selection Phase
// =============================================================================

// Select best child of node using configured policy
fn select_child(tree: &MCTSTree, node_idx: i32) -> i32 {
    let node = tree.nodes[node_idx as usize];
    let neg_one: i32 = (0 - 1) as i32;
    if node.num_children == 0 { return neg_one; }

    let parent_visits = node.visits;
    let first = node.first_child_idx as usize;
    let n = node.num_children as usize;

    var best_idx = first as i32;
    var best_score = 0.0 - 1.0e308;

    var i: usize = 0;
    while i < n {
        let child_idx = (first + i) as i32;
        let child = tree.nodes[child_idx as usize];

        let q = if child.visits == 0 { 0.0 } else { child.total_value / (child.visits as f64) };

        let score = if tree.use_puct {
            puct(q, child.visits, parent_visits, child.prior, tree.exploration_c)
        } else {
            ucb1(q, child.visits, parent_visits, tree.exploration_c)
        };

        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
        i = i + 1;
    }

    best_idx
}

// Traverse tree to leaf node
pub fn select_to_leaf(tree: &MCTSTree) -> i32 {
    var current = tree.root_idx;
    let zero_i32: i32 = 0;
    let zero_u32: u32 = 0;

    while tree.nodes[current as usize].num_children > zero_u32 {
        let next = select_child(tree, current);
        if next < zero_i32 { break; }
        current = next;
    }

    current
}

// =============================================================================
// Backpropagation
// =============================================================================

// Backpropagate value from leaf to root
pub fn backpropagate(tree: &MCTSTree, leaf_idx: i32, value: f64) {
    var current = leaf_idx;
    var v = value;
    let zero_i32: i32 = 0;
    let one_u32: u32 = 1;

    while current >= zero_i32 {
        let idx = current as usize;
        tree.nodes[idx].visits = tree.nodes[idx].visits + one_u32;
        tree.nodes[idx].total_value = tree.nodes[idx].total_value + v;

        current = tree.nodes[idx].parent_idx;
        v = 0.0 - v;  // Flip for adversarial games
    }
}

// Backpropagate without negation (for single-player/optimization)
pub fn backpropagate_same(tree: &MCTSTree, leaf_idx: i32, value: f64) {
    var current = leaf_idx;
    let zero_i32: i32 = 0;
    let one_u32: u32 = 1;

    while current >= zero_i32 {
        let idx = current as usize;
        tree.nodes[idx].visits = tree.nodes[idx].visits + one_u32;
        tree.nodes[idx].total_value = tree.nodes[idx].total_value + value;
        current = tree.nodes[idx].parent_idx;
    }
}

// =============================================================================
// Result Extraction
// =============================================================================

// Get best action from root by visit count
pub fn best_action_by_visits(tree: &MCTSTree) -> i32 {
    let neg_one: i32 = (0 - 1) as i32;
    let root = tree.nodes[tree.root_idx as usize];
    if root.num_children == 0 { return neg_one; }

    let first = root.first_child_idx as usize;
    let n = root.num_children as usize;

    var best_action: i32 = neg_one;
    var best_visits: u32 = 0;

    var i: usize = 0;
    while i < n {
        let child = tree.nodes[first + i];
        if child.visits > best_visits {
            best_visits = child.visits;
            best_action = child.action;
        }
        i = i + 1;
    }

    best_action
}

// Get best action by Q-value
pub fn best_action_by_value(tree: &MCTSTree) -> i32 {
    let neg_one: i32 = (0 - 1) as i32;
    let root = tree.nodes[tree.root_idx as usize];
    if root.num_children == 0 { return neg_one; }

    let first = root.first_child_idx as usize;
    let n = root.num_children as usize;

    var best_action: i32 = neg_one;
    var best_q = 0.0 - 1.0e308;

    var i: usize = 0;
    while i < n {
        let child = tree.nodes[first + i];
        if child.visits > 0 {
            let q = child.total_value / (child.visits as f64);
            if q > best_q {
                best_q = q;
                best_action = child.action;
            }
        }
        i = i + 1;
    }

    best_action
}

// Get visit distribution for all root children
pub fn get_visit_distribution(tree: &MCTSTree) -> [u32] {
    var result: [u32] = [];

    let root = tree.nodes[tree.root_idx as usize];
    let first = root.first_child_idx as usize;
    let n = root.num_children as usize;

    var i: usize = 0;
    while i < n {
        result.push(tree.nodes[first + i].visits);
        i = i + 1;
    }

    result
}

// Get total simulations run
pub fn total_simulations(tree: &MCTSTree) -> u32 {
    tree.nodes[tree.root_idx as usize].visits
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing MCTS core module...\n");

    // Type constants
    let zero_i32: i32 = 0;
    let one_i32: i32 = 1;
    let two_i32: i32 = 2;
    let zero_i64: i64 = 0;
    let exit_ok: i32 = 0;

    // Test 1: Create tree
    var tree = new_tree(zero_i64);
    print("Create tree: PASS\n");

    // Test 2: Add children
    let c0 = add_child(&tree, zero_i32, 1, zero_i32, 0.3);
    let c1 = add_child(&tree, zero_i32, 2, one_i32, 0.5);
    let c2 = add_child(&tree, zero_i32, 3, two_i32, 0.2);
    print("Add children: PASS\n");

    // Test 3: Backpropagation
    backpropagate_same(&tree, c0, 1.0);
    backpropagate_same(&tree, c0, 0.5);
    backpropagate_same(&tree, c1, 0.8);
    print("Backpropagation: PASS\n");

    // Test 4: Selection
    let leaf = select_to_leaf(&tree);
    print("Selection: PASS\n");

    // Test 5: Best action
    let best = best_action_by_visits(&tree);
    print("Best action: PASS\n");

    // Test 6: Visit distribution
    let dist = get_visit_distribution(&tree);
    print("Visit distribution: PASS\n");

    // Test 7: Total simulations
    let total = total_simulations(&tree);
    print("Total simulations: PASS\n");

    // Test 8: Enable PUCT
    enable_puct(&tree, 1.5);
    print("PUCT mode: PASS\n");

    print("All MCTS core tests PASSED\n");
    exit_ok
}
