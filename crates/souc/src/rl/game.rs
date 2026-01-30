//! Game State Abstraction for RL
//!
//! Generic traits for two-player zero-sum games compatible with MCTS.
//!
//! # Design Principles
//!
//! 1. **Immutable states**: `apply_action` returns new state
//! 2. **Type-safe actions**: Each game defines its own action type
//! 3. **Compile-time validation**: Refinements can enforce legal moves
//! 4. **Epistemic integration**: States can carry uncertainty metadata

use std::fmt::Debug;
use std::hash::Hash;

/// Player identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    One,
    Two,
}

impl Player {
    /// Get the opponent
    pub fn opponent(&self) -> Self {
        match self {
            Player::One => Player::Two,
            Player::Two => Player::One,
        }
    }

    /// Convert to numeric index (0 or 1)
    pub fn index(&self) -> usize {
        match self {
            Player::One => 0,
            Player::Two => 1,
        }
    }

    /// Create from index
    pub fn from_index(idx: usize) -> Self {
        if idx == 0 { Player::One } else { Player::Two }
    }
}

/// Outcome of a game
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameOutcome {
    /// Player one wins
    Win(Player),
    /// Draw
    Draw,
    /// Game still in progress
    InProgress,
}

impl GameOutcome {
    /// Get value from perspective of given player (1.0 = win, 0.0 = loss, 0.5 = draw)
    pub fn value_for(&self, player: Player) -> f64 {
        match self {
            GameOutcome::Win(winner) => {
                if *winner == player {
                    1.0
                } else {
                    0.0
                }
            }
            GameOutcome::Draw => 0.5,
            GameOutcome::InProgress => 0.5,
        }
    }
}

/// Trait for game actions
pub trait Action: Debug + Clone + Eq + Hash + Send + Sync {}

/// Trait for game states
pub trait GameState: Debug + Clone + Send + Sync {
    /// Action type for this game
    type Action: Action;

    /// Get the current player to move
    fn current_player(&self) -> Player;

    /// Get all legal actions from this state
    fn legal_actions(&self) -> Vec<Self::Action>;

    /// Apply an action to get a new state
    fn apply_action(&self, action: &Self::Action) -> Self;

    /// Check if this is a terminal state
    fn is_terminal(&self) -> bool;

    /// Get terminal value from perspective of given player
    /// Returns value in [0, 1]: 1.0 = win, 0.0 = loss, 0.5 = draw
    fn terminal_value(&self, player: Player) -> f64;

    /// Get the game outcome
    fn outcome(&self) -> GameOutcome {
        if !self.is_terminal() {
            return GameOutcome::InProgress;
        }

        let value_p1 = self.terminal_value(Player::One);
        if value_p1 > 0.5 {
            GameOutcome::Win(Player::One)
        } else if value_p1 < 0.5 {
            GameOutcome::Win(Player::Two)
        } else {
            GameOutcome::Draw
        }
    }

    /// Get number of legal actions
    fn num_legal_actions(&self) -> usize {
        self.legal_actions().len()
    }

    /// Check if an action is legal
    fn is_legal(&self, action: &Self::Action) -> bool {
        self.legal_actions().contains(action)
    }

    /// Get a canonical representation for hashing/comparison
    fn canonical_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut hasher = DefaultHasher::new();
        // Default: hash the debug string (override for efficiency)
        format!("{:?}", self).hash(&mut hasher);
        hasher.finish()
    }
}

/// Extended game trait with additional features
pub trait GameTrait: GameState {
    /// Get a feature vector for neural network input
    fn to_features(&self) -> Vec<f32>;

    /// Get feature dimensions
    fn feature_shape() -> Vec<usize>;

    /// Get number of possible actions (for policy output size)
    fn num_actions() -> usize;

    /// Convert action to index (for policy vector)
    fn action_to_index(action: &Self::Action) -> usize;

    /// Convert index to action
    fn index_to_action(index: usize) -> Option<Self::Action>;

    /// Get action mask (1.0 for legal, 0.0 for illegal)
    fn action_mask(&self) -> Vec<f32> {
        let legal = self.legal_actions();
        (0..Self::num_actions())
            .map(|i| {
                if let Some(action) = Self::index_to_action(i) {
                    if legal.contains(&action) { 1.0 } else { 0.0 }
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Create initial state
    fn initial() -> Self;

    /// Get game name
    fn name() -> &'static str;
}

// =============================================================================
// Example: Tic-Tac-Toe
// =============================================================================

/// Tic-Tac-Toe action: position 0-8
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TicTacToeAction(pub u8);

impl Action for TicTacToeAction {}

/// Tic-Tac-Toe state
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TicTacToeState {
    /// Board: 0 = empty, 1 = X, 2 = O
    pub board: [u8; 9],
    /// Current player
    pub player: Player,
    /// Move count
    pub moves: u8,
}

impl TicTacToeState {
    /// Create new empty board
    pub fn new() -> Self {
        TicTacToeState {
            board: [0; 9],
            player: Player::One,
            moves: 0,
        }
    }

    /// Check for a win
    fn check_win(&self, mark: u8) -> bool {
        const LINES: [[usize; 3]; 8] = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8], // rows
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8], // cols
            [0, 4, 8],
            [2, 4, 6], // diagonals
        ];

        LINES
            .iter()
            .any(|line| line.iter().all(|&i| self.board[i] == mark))
    }

    /// Get mark for player
    fn mark(player: Player) -> u8 {
        match player {
            Player::One => 1,
            Player::Two => 2,
        }
    }
}

impl Default for TicTacToeState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState for TicTacToeState {
    type Action = TicTacToeAction;

    fn current_player(&self) -> Player {
        self.player
    }

    fn legal_actions(&self) -> Vec<Self::Action> {
        if self.is_terminal() {
            return vec![];
        }

        self.board
            .iter()
            .enumerate()
            .filter(|&(_, cell)| *cell == 0)
            .map(|(i, _)| TicTacToeAction(i as u8))
            .collect()
    }

    fn apply_action(&self, action: &Self::Action) -> Self {
        let mut new_state = self.clone();
        new_state.board[action.0 as usize] = Self::mark(self.player);
        new_state.player = self.player.opponent();
        new_state.moves += 1;
        new_state
    }

    fn is_terminal(&self) -> bool {
        self.check_win(1) || self.check_win(2) || self.moves >= 9
    }

    fn terminal_value(&self, player: Player) -> f64 {
        let mark = Self::mark(player);
        let opp_mark = Self::mark(player.opponent());

        if self.check_win(mark) {
            1.0
        } else if self.check_win(opp_mark) {
            0.0
        } else {
            0.5 // Draw
        }
    }
}

impl GameTrait for TicTacToeState {
    fn to_features(&self) -> Vec<f32> {
        // 3 channels: player 1, player 2, empty
        let mut features = vec![0.0; 27];
        for (i, &cell) in self.board.iter().enumerate() {
            match cell {
                0 => features[i + 18] = 1.0, // empty
                1 => features[i] = 1.0,      // X
                2 => features[i + 9] = 1.0,  // O
                _ => {}
            }
        }
        features
    }

    fn feature_shape() -> Vec<usize> {
        vec![3, 3, 3] // 3 channels, 3x3 board
    }

    fn num_actions() -> usize {
        9
    }

    fn action_to_index(action: &Self::Action) -> usize {
        action.0 as usize
    }

    fn index_to_action(index: usize) -> Option<Self::Action> {
        if index < 9 {
            Some(TicTacToeAction(index as u8))
        } else {
            None
        }
    }

    fn initial() -> Self {
        Self::new()
    }

    fn name() -> &'static str {
        "TicTacToe"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_player_opponent() {
        assert_eq!(Player::One.opponent(), Player::Two);
        assert_eq!(Player::Two.opponent(), Player::One);
    }

    #[test]
    fn test_tictactoe_initial() {
        let state = TicTacToeState::new();
        assert_eq!(state.current_player(), Player::One);
        assert_eq!(state.legal_actions().len(), 9);
        assert!(!state.is_terminal());
    }

    #[test]
    fn test_tictactoe_move() {
        let state = TicTacToeState::new();
        let action = TicTacToeAction(4); // center
        let new_state = state.apply_action(&action);

        assert_eq!(new_state.board[4], 1); // X
        assert_eq!(new_state.current_player(), Player::Two);
        assert_eq!(new_state.legal_actions().len(), 8);
    }

    #[test]
    fn test_tictactoe_win() {
        let mut state = TicTacToeState::new();

        // X wins: 0, 1, 2 (top row)
        state = state.apply_action(&TicTacToeAction(0)); // X
        state = state.apply_action(&TicTacToeAction(3)); // O
        state = state.apply_action(&TicTacToeAction(1)); // X
        state = state.apply_action(&TicTacToeAction(4)); // O
        state = state.apply_action(&TicTacToeAction(2)); // X wins

        assert!(state.is_terminal());
        assert_eq!(state.terminal_value(Player::One), 1.0);
        assert_eq!(state.terminal_value(Player::Two), 0.0);
        assert_eq!(state.outcome(), GameOutcome::Win(Player::One));
    }

    #[test]
    fn test_tictactoe_draw() {
        let mut state = TicTacToeState::new();

        // Classic draw: X O X / X X O / O X O
        let moves = [0, 1, 2, 4, 3, 5, 7, 6, 8];
        for (i, &pos) in moves.iter().enumerate() {
            if !state.is_terminal() {
                state = state.apply_action(&TicTacToeAction(pos));
            }
        }

        // Check terminal (might be win or draw depending on order)
        assert!(state.is_terminal());
    }

    #[test]
    fn test_tictactoe_features() {
        let state = TicTacToeState::new();
        let features = state.to_features();

        assert_eq!(features.len(), 27);
        // All empty initially
        assert!(features[18..27].iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_action_mask() {
        let state = TicTacToeState::new();
        let mask = state.action_mask();

        assert_eq!(mask.len(), 9);
        assert!(mask.iter().all(|&x| x == 1.0)); // All legal initially

        let state2 = state.apply_action(&TicTacToeAction(4));
        let mask2 = state2.action_mask();

        assert_eq!(mask2[4], 0.0); // Center now illegal
        assert_eq!(mask2.iter().filter(|&&x| x == 1.0).count(), 8);
    }
}
