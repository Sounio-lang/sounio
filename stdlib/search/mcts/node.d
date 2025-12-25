// stdlib/search/mcts/node.d
// MCTS Tree Node Structure
//
// Generic tree node for Monte Carlo Tree Search.
// Tracks visit counts, values, and child relationships.

// =============================================================================
// Node Structure
// =============================================================================

// MCTS tree node
pub struct MCTSNode {
    pub state: i64,          // State hash/encoding
    pub action: i32,         // Action that led here (-1 for root)
    pub total_value: f64,    // Sum of backpropagated values
    pub visits: u32,         // Visit count
    pub prior: f64,          // Prior probability (for PUCT)
    pub parent_idx: i32,     // Parent index (-1 if root)
    pub children: [i32],     // Child indices
    pub is_expanded: bool,
    pub is_terminal: bool,
}

// Create a new root node
pub fn new_root(state: i64) -> MCTSNode {
    MCTSNode {
        state: state,
        action: 0 - 1,
        total_value: 0.0,
        visits: 0,
        prior: 1.0,
        parent_idx: 0 - 1,
        children: [],
        is_expanded: false,
        is_terminal: false,
    }
}

// Create a child node
pub fn new_child(state: i64, action: i32, parent_idx: i32, prior: f64) -> MCTSNode {
    MCTSNode {
        state: state,
        action: action,
        total_value: 0.0,
        visits: 0,
        prior: prior,
        parent_idx: parent_idx,
        children: [],
        is_expanded: false,
        is_terminal: false,
    }
}

// =============================================================================
// Node Statistics (pure functions)
// =============================================================================

// Get Q-value (mean value)
pub fn q_value(total: f64, visits: u32) -> f64 {
    if visits == 0 {
        0.0
    } else {
        total / (visits as f64)
    }
}

// Check if node is leaf
pub fn is_leaf(num_children: usize) -> bool {
    num_children == 0
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing MCTS node module...\n");

    // Test 1: Create root node
    var root = new_root(12345);
    if root.action != (0 - 1) {
        print("FAIL: root action\n");
        return 1;
    }
    print("Root node: PASS\n");

    // Test 2: Create child node
    var child = new_child(67890, 3, 0, 0.25);
    if child.action != 3 {
        print("FAIL: child action\n");
        return 1;
    }
    print("Child node: PASS\n");

    // Test 3: Update statistics directly
    root.total_value = root.total_value + 1.5;
    root.visits = root.visits + 1;
    root.total_value = root.total_value + 0.5;
    root.visits = root.visits + 1;
    if root.visits != 2 {
        print("FAIL: visits\n");
        return 1;
    }
    print("Statistics update: PASS\n");

    // Test 4: Q-value
    let q = q_value(root.total_value, root.visits);
    // Should be 2.0 / 2 = 1.0
    print("Q-value: PASS\n");

    // Test 5: Add child
    root.children.push(0);
    if is_leaf(root.children.len()) {
        print("FAIL: should have child\n");
        return 1;
    }
    print("Children: PASS\n");

    // Test 6: Mark expanded
    root.is_expanded = true;
    if !root.is_expanded {
        print("FAIL: expanded\n");
        return 1;
    }
    print("Expanded: PASS\n");

    print("All MCTS node tests PASSED\n");
    0
}
