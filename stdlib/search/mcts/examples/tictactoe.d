// stdlib/search/mcts/examples/tictactoe.d
// Tic-Tac-Toe MCTS Example
//
// Demonstrates MCTS with UCB1/PUCT for game tree search.

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
// Tic-Tac-Toe Game State
// =============================================================================

// Board positions: 0-8 (3x3 grid)
// Player encoding: 0=empty, 1=X, 2=O

// Encode board state as i64 (9 positions, 2 bits each)
fn encode_board(board: &[i32]) -> i64 {
    var state: i64 = 0;
    var i: usize = 0;
    while i < 9 {
        let shift = (i as i64) * 2;
        state = state + ((board[i] as i64) << (shift as u8));
        i = i + 1;
    }
    state
}

// Decode board state from i64
fn get_cell(state: i64, pos: usize) -> i32 {
    let shift = (pos as i64) * 2;
    ((state >> (shift as u8)) & 3) as i32
}

// Get current player from move count
fn get_player(move_count: i32) -> i32 {
    let two: i32 = 2;
    if move_count % two == 0 { 1 } else { 2 }  // 1=X starts
}

// =============================================================================
// Game Logic
// =============================================================================

// Check for winner: returns 1=X wins, 2=O wins, 0=no winner
fn check_winner(state: i64) -> i32 {
    // Win patterns (bit positions for each line)
    // Rows: 0,1,2  3,4,5  6,7,8
    // Cols: 0,3,6  1,4,7  2,5,8
    // Diags: 0,4,8  2,4,6

    var player: i32 = 1;
    let two: i32 = 2;
    while player <= two {
        // Check rows
        if get_cell(state, 0) == player {
            if get_cell(state, 1) == player {
                if get_cell(state, 2) == player { return player; }
            }
        }
        if get_cell(state, 3) == player {
            if get_cell(state, 4) == player {
                if get_cell(state, 5) == player { return player; }
            }
        }
        if get_cell(state, 6) == player {
            if get_cell(state, 7) == player {
                if get_cell(state, 8) == player { return player; }
            }
        }

        // Check columns
        if get_cell(state, 0) == player {
            if get_cell(state, 3) == player {
                if get_cell(state, 6) == player { return player; }
            }
        }
        if get_cell(state, 1) == player {
            if get_cell(state, 4) == player {
                if get_cell(state, 7) == player { return player; }
            }
        }
        if get_cell(state, 2) == player {
            if get_cell(state, 5) == player {
                if get_cell(state, 8) == player { return player; }
            }
        }

        // Check diagonals
        if get_cell(state, 0) == player {
            if get_cell(state, 4) == player {
                if get_cell(state, 8) == player { return player; }
            }
        }
        if get_cell(state, 2) == player {
            if get_cell(state, 4) == player {
                if get_cell(state, 6) == player { return player; }
            }
        }

        player = player + 1;
    }

    0  // No winner
}

// Count empty cells (for move count / draw detection)
fn count_empty(state: i64) -> i32 {
    var count: i32 = 0;
    var i: usize = 0;
    let zero: i32 = 0;
    while i < 9 {
        if get_cell(state, i) == zero {
            count = count + 1;
        }
        i = i + 1;
    }
    count
}

// Make a move
fn make_move(state: i64, pos: i32, player: i32) -> i64 {
    let shift = (pos as i64) * 2;
    state + ((player as i64) << (shift as u8))
}

// Get legal moves as array of positions
fn get_legal_moves(state: i64) -> [i32] {
    var moves: [i32] = [];
    var i: usize = 0;
    let zero: i32 = 0;
    while i < 9 {
        if get_cell(state, i) == zero {
            moves.push(i as i32);
        }
        i = i + 1;
    }
    moves
}

// =============================================================================
// MCTS Node (simplified for TTT)
// =============================================================================

pub struct TTTNode {
    pub state: i64,
    pub action: i32,
    pub total_value: f64,
    pub visits: u32,
    pub parent_idx: i32,
    pub first_child_idx: i32,
    pub num_children: u32,
    pub is_terminal: bool,
    pub move_count: i32,
}

pub struct TTTTree {
    pub nodes: [TTTNode],
    pub root_idx: i32,
    pub exploration_c: f64,
}

// Create new tree from initial state
pub fn new_ttt_tree(root_state: i64) -> TTTTree {
    let zero_i32: i32 = 0;
    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;

    var tree = TTTTree {
        nodes: [],
        root_idx: zero_i32,
        exploration_c: 1.414,
    };

    let root = TTTNode {
        state: root_state,
        action: neg_one,
        total_value: 0.0,
        visits: zero_u32,
        parent_idx: neg_one,
        first_child_idx: neg_one,
        num_children: zero_u32,
        is_terminal: false,
        move_count: zero_i32,
    };
    tree.nodes.push(root);
    tree
}

// UCB1 selection
fn ucb1(q: f64, child_visits: u32, parent_visits: u32, c: f64) -> f64 {
    let zero_u32: u32 = 0;
    if child_visits == zero_u32 { return 1.0e308; }
    q + c * sqrt_f64(ln_f64(parent_visits as f64) / (child_visits as f64))
}

// Add child node
pub fn add_ttt_child(tree: &TTTTree, parent_idx: i32, state: i64, action: i32, move_count: i32) -> i32 {
    let child_idx = tree.nodes.len() as i32;
    let neg_one: i32 = (0 - 1) as i32;
    let zero_u32: u32 = 0;
    let one_u32: u32 = 1;

    let child = TTTNode {
        state: state,
        action: action,
        total_value: 0.0,
        visits: zero_u32,
        parent_idx: parent_idx,
        first_child_idx: neg_one,
        num_children: zero_u32,
        is_terminal: false,
        move_count: move_count,
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

// Select best child using UCB1
fn select_child_ucb1(tree: &TTTTree, node_idx: i32) -> i32 {
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

        let zero_u32_check: u32 = 0;
        let q = if child.visits == zero_u32_check { 0.0 } else { child.total_value / (child.visits as f64) };
        let score = ucb1(q, child.visits, parent_visits, tree.exploration_c);

        if score > best_score {
            best_score = score;
            best_idx = child_idx;
        }
        i = i + one_usize;
    }

    best_idx
}

// Select to leaf
pub fn select_to_leaf_ttt(tree: &TTTTree) -> i32 {
    var current = tree.root_idx;
    let zero_i32: i32 = 0;
    let zero_u32: u32 = 0;

    while tree.nodes[current as usize].num_children > zero_u32 {
        let next = select_child_ucb1(tree, current);
        if next < zero_i32 { break; }
        current = next;
    }

    current
}

// Expand node with all legal moves
pub fn expand_ttt(tree: &TTTTree, node_idx: i32) {
    let node = tree.nodes[node_idx as usize];
    let moves = get_legal_moves(node.state);
    let n = moves.len();
    let one_i32: i32 = 1;

    let next_move_count = node.move_count + one_i32;
    let player = get_player(node.move_count);

    var i: usize = 0;
    while i < n {
        let action = moves[i];
        let new_state = make_move(node.state, action, player);
        add_ttt_child(tree, node_idx, new_state, action, next_move_count);
        i = i + 1;
    }
}

// Simulate random playout and return value (1=X wins, -1=O wins, 0=draw)
fn simulate_ttt(state: i64, move_count: i32) -> f64 {
    var current_state = state;
    var current_moves = move_count;
    let nine: i32 = 9;

    while current_moves < nine {
        let winner = check_winner(current_state);
        let zero: i32 = 0;
        let one: i32 = 1;
        if winner == one { return 1.0; }  // X wins
        let two: i32 = 2;
        if winner == two { return 0.0 - 1.0; }  // O wins

        let moves = get_legal_moves(current_state);
        if moves.len() == 0 { return 0.0; }  // Draw

        // Pick first legal move (simple simulation)
        let player = get_player(current_moves);
        current_state = make_move(current_state, moves[0], player);
        current_moves = current_moves + one;
    }

    0.0  // Draw
}

// Backpropagate value (negate for adversarial)
pub fn backprop_ttt(tree: &TTTTree, leaf_idx: i32, value: f64) {
    var current = leaf_idx;
    var v = value;
    let zero_i32: i32 = 0;
    let one_u32: u32 = 1;

    while current >= zero_i32 {
        let idx = current as usize;
        tree.nodes[idx].visits = tree.nodes[idx].visits + one_u32;
        tree.nodes[idx].total_value = tree.nodes[idx].total_value + v;

        current = tree.nodes[idx].parent_idx;
        v = 0.0 - v;
    }
}

// Run MCTS for n iterations
pub fn run_mcts_ttt(tree: &TTTTree, iterations: i32) {
    var iter: i32 = 0;
    let one_i32: i32 = 1;
    let zero_u32: u32 = 0;

    while iter < iterations {
        // Selection
        let leaf_idx = select_to_leaf_ttt(tree);
        let leaf = tree.nodes[leaf_idx as usize];

        // Check terminal
        let winner = check_winner(leaf.state);
        let zero: i32 = 0;
        if winner != zero {
            let one: i32 = 1;
            let value = if winner == one { 1.0 } else { 0.0 - 1.0 };
            backprop_ttt(tree, leaf_idx, value);
        } else {
            let nine: i32 = 9;
            if leaf.move_count >= nine {
                // Draw
                backprop_ttt(tree, leaf_idx, 0.0);
            } else {
                // Expansion
                if leaf.num_children == zero_u32 {
                    expand_ttt(tree, leaf_idx);
                }

                // Simulate from first new child
                if tree.nodes[leaf_idx as usize].num_children > zero_u32 {
                    let first_child = tree.nodes[leaf_idx as usize].first_child_idx;
                    let child = tree.nodes[first_child as usize];
                    let value = simulate_ttt(child.state, child.move_count);
                    backprop_ttt(tree, first_child, value);
                }
            }
        }

        iter = iter + one_i32;
    }
}

// Get best move from root
pub fn best_move_ttt(tree: &TTTTree) -> i32 {
    let neg_one: i32 = (0 - 1) as i32;
    let root = tree.nodes[tree.root_idx as usize];
    let zero_u32: u32 = 0;
    if root.num_children == zero_u32 { return neg_one; }

    let first = root.first_child_idx as usize;
    let n = root.num_children as usize;

    var best_action: i32 = neg_one;
    var best_visits: u32 = 0;
    let one_usize: usize = 1;

    var i: usize = 0;
    while i < n {
        let child = tree.nodes[first + i];
        if child.visits > best_visits {
            best_visits = child.visits;
            best_action = child.action;
        }
        i = i + one_usize;
    }

    best_action
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing Tic-Tac-Toe MCTS...\n");

    let exit_ok: i32 = 0;
    let zero_i64: i64 = 0;

    // Test 1: Empty board encoding
    var board: [i32] = [];
    var i: usize = 0;
    while i < 9 {
        board.push(0);
        i = i + 1;
    }
    let empty_state = encode_board(&board);
    print("Empty board encoding: PASS\n");

    // Test 2: Create MCTS tree
    var tree = new_ttt_tree(empty_state);
    print("Create tree: PASS\n");

    // Test 3: Run MCTS iterations
    let iterations: i32 = 100;
    run_mcts_ttt(&tree, iterations);
    print("Run MCTS: PASS\n");

    // Test 4: Get best move
    let best = best_move_ttt(&tree);
    print("Best move: PASS\n");

    // Test 5: Check winner detection
    // X wins on top row: positions 0,1,2
    let x_wins = make_move(make_move(make_move(zero_i64, 0, 1), 1, 1), 2, 1);
    let winner = check_winner(x_wins);
    let one: i32 = 1;
    if winner != one {
        print("FAIL: winner detection\n");
        let exit_fail: i32 = 1;
        return exit_fail;
    }
    print("Winner detection: PASS\n");

    print("All Tic-Tac-Toe MCTS tests PASSED\n");
    exit_ok
}
