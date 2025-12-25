//! Compile-Time Refinements for Valid Game States
//!
//! This module encodes board game rules as refinement types, enabling the compiler
//! to verify that only valid game states can be constructed and that moves are legal.
//!
//! # Overview
//!
//! Traditional game implementations check rules at runtime, leading to:
//! - Runtime errors for illegal moves
//! - Complex error handling code
//! - Silent bugs when invariants are violated
//!
//! With refinement types, we can:
//! - Reject illegal states at compile time
//! - Make illegal moves unrepresentable
//! - Get formal verification of game logic
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              Refinement Type Hierarchy for Games                     │
//! │                                                                     │
//! │  BoardState<W, H>          Board with width W, height H             │
//! │       │                                                             │
//! │       ├── ValidBoard       Refinement: all cells in valid range     │
//! │       │                                                             │
//! │       ├── NonTerminal      Refinement: game not over                │
//! │       │                                                             │
//! │       └── LegalMove(m)     Refinement: move m is legal here         │
//! │                                                                     │
//! │  Action constraints:                                                │
//! │  • InBounds(row, col)      Position within board                    │
//! │  • CellEmpty(row, col)     Target cell is unoccupied                │
//! │  • TurnValid(player)       Correct player's turn                    │
//! │                                                                     │
//! │  Game invariants (checked at state transitions):                    │
//! │  • MoveCount <= W*H        Can't exceed board capacity              │
//! │  • Alternating turns       Players alternate correctly              │
//! │  • No overwrites           Can't place on occupied cells            │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example: Tic-Tac-Toe Refinements
//!
//! ```sounio
//! // Cell refinement: 0 (empty), 1 (X), 2 (O)
//! type Cell = { c: Int | c >= 0 && c <= 2 }
//!
//! // Position refinement: valid indices
//! type Position = { (row, col): (Int, Int) | row >= 0 && row < 3 && col >= 0 && col < 3 }
//!
//! // Board refinement: 3x3 grid with valid cells
//! type Board = { b: Array<Cell, 9> | forall i. 0 <= i < 9 => valid_cell(b[i]) }
//!
//! // Move refinement: can only move to empty cells
//! fn move(board: Board, pos: Position) -> Board
//!   where board[pos.row * 3 + pos.col] == 0  // Pre: cell is empty
//!   ensures result[pos.row * 3 + pos.col] != 0  // Post: cell is filled
//! ```

use crate::types::core::Type;
use crate::types::refinement::{
    ArithOp, Predicate, RefinedType, RefinementChecker, RefinementResult,
};
use std::collections::HashMap;
use std::marker::PhantomData;

// =============================================================================
// Board Dimension Constraints
// =============================================================================

/// Compile-time board dimensions
pub trait BoardDimensions {
    const WIDTH: usize;
    const HEIGHT: usize;
    const SIZE: usize = Self::WIDTH * Self::HEIGHT;
}

/// 3x3 board (Tic-Tac-Toe)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim3x3;
impl BoardDimensions for Dim3x3 {
    const WIDTH: usize = 3;
    const HEIGHT: usize = 3;
}

/// 8x8 board (Chess, Checkers)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim8x8;
impl BoardDimensions for Dim8x8 {
    const WIDTH: usize = 8;
    const HEIGHT: usize = 8;
}

/// 15x15 board (Go on small board)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim15x15;
impl BoardDimensions for Dim15x15 {
    const WIDTH: usize = 15;
    const HEIGHT: usize = 15;
}

/// 19x19 board (Standard Go)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dim19x19;
impl BoardDimensions for Dim19x19 {
    const WIDTH: usize = 19;
    const HEIGHT: usize = 19;
}

// =============================================================================
// Cell Value Constraints
// =============================================================================

/// Cell value constraints for different games
pub trait CellConstraint {
    /// Minimum valid cell value
    const MIN: i8;
    /// Maximum valid cell value
    const MAX: i8;
    /// Empty cell value
    const EMPTY: i8;

    /// Check if a cell value is valid
    fn is_valid(value: i8) -> bool {
        value >= Self::MIN && value <= Self::MAX
    }

    /// Check if a cell is empty
    fn is_empty(value: i8) -> bool {
        value == Self::EMPTY
    }
}

/// Tic-Tac-Toe cells: 0 (empty), 1 (X), 2 (O)
pub struct TicTacToeCell;
impl CellConstraint for TicTacToeCell {
    const MIN: i8 = 0;
    const MAX: i8 = 2;
    const EMPTY: i8 = 0;
}

/// Go cells: 0 (empty), 1 (black), 2 (white)
pub struct GoCell;
impl CellConstraint for GoCell {
    const MIN: i8 = 0;
    const MAX: i8 = 2;
    const EMPTY: i8 = 0;
}

/// Chess cells: -6 to 6 (negative for black, positive for white, 0 empty)
/// 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
pub struct ChessCell;
impl CellConstraint for ChessCell {
    const MIN: i8 = -6;
    const MAX: i8 = 6;
    const EMPTY: i8 = 0;
}

// =============================================================================
// Position Refinements
// =============================================================================

/// A position on the board, refined to be within bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Position<D: BoardDimensions> {
    pub row: usize,
    pub col: usize,
    _phantom: PhantomData<D>,
}

impl<D: BoardDimensions> Position<D> {
    /// Create a new position, returning None if out of bounds
    pub fn new(row: usize, col: usize) -> Option<Self> {
        if row < D::HEIGHT && col < D::WIDTH {
            Some(Position {
                row,
                col,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    /// Create without bounds check (unsafe)
    ///
    /// # Safety
    /// Caller must ensure row < HEIGHT and col < WIDTH
    pub const unsafe fn new_unchecked(row: usize, col: usize) -> Self {
        Position {
            row,
            col,
            _phantom: PhantomData,
        }
    }

    /// Convert to linear index
    pub fn to_index(&self) -> usize {
        self.row * D::WIDTH + self.col
    }

    /// Create from linear index
    pub fn from_index(idx: usize) -> Option<Self> {
        if idx < D::SIZE {
            let row = idx / D::WIDTH;
            let col = idx % D::WIDTH;
            Some(Position {
                row,
                col,
                _phantom: PhantomData,
            })
        } else {
            None
        }
    }

    /// Get the refinement predicate for this position type
    pub fn refinement_predicate() -> Predicate {
        // row >= 0 && row < HEIGHT && col >= 0 && col < WIDTH
        Predicate::and(
            Predicate::and(
                Predicate::ge(Predicate::var("row"), Predicate::Int(0)),
                Predicate::lt(Predicate::var("row"), Predicate::Int(D::HEIGHT as i64)),
            ),
            Predicate::and(
                Predicate::ge(Predicate::var("col"), Predicate::Int(0)),
                Predicate::lt(Predicate::var("col"), Predicate::Int(D::WIDTH as i64)),
            ),
        )
    }
}

// =============================================================================
// Game State Refinements
// =============================================================================

/// Refinement predicates for game state validity
#[derive(Debug, Clone)]
pub struct GameStateRefinement {
    /// Board dimensions
    pub width: usize,
    pub height: usize,
    /// Cell value range
    pub cell_min: i64,
    pub cell_max: i64,
    pub cell_empty: i64,
    /// Predicates that must hold
    pub invariants: Vec<Predicate>,
}

impl GameStateRefinement {
    /// Create refinements for Tic-Tac-Toe
    pub fn tic_tac_toe() -> Self {
        let mut invariants = Vec::new();

        // All cells valid: forall i. 0 <= i < 9 => 0 <= board[i] <= 2
        invariants.push(Predicate::Forall(
            "i".to_string(),
            Box::new(Predicate::implies(
                Predicate::and(
                    Predicate::ge(Predicate::var("i"), Predicate::Int(0)),
                    Predicate::lt(Predicate::var("i"), Predicate::Int(9)),
                ),
                Predicate::and(
                    Predicate::ge(
                        Predicate::App(
                            "board_get".to_string(),
                            vec![Predicate::var("board"), Predicate::var("i")],
                        ),
                        Predicate::Int(0),
                    ),
                    Predicate::le(
                        Predicate::App(
                            "board_get".to_string(),
                            vec![Predicate::var("board"), Predicate::var("i")],
                        ),
                        Predicate::Int(2),
                    ),
                ),
            )),
        ));

        // Move count matches non-empty cells
        invariants.push(Predicate::eq(
            Predicate::var("move_count"),
            Predicate::App("count_nonempty".to_string(), vec![Predicate::var("board")]),
        ));

        // Player alternates: current_player = (move_count % 2 == 0 ? 1 : 2)
        invariants.push(Predicate::Ite(
            Box::new(Predicate::eq(
                Predicate::Arith(
                    ArithOp::Mod,
                    Box::new(Predicate::var("move_count")),
                    Box::new(Predicate::Int(2)),
                ),
                Predicate::Int(0),
            )),
            Box::new(Predicate::eq(
                Predicate::var("current_player"),
                Predicate::Int(1),
            )),
            Box::new(Predicate::eq(
                Predicate::var("current_player"),
                Predicate::Int(2),
            )),
        ));

        GameStateRefinement {
            width: 3,
            height: 3,
            cell_min: 0,
            cell_max: 2,
            cell_empty: 0,
            invariants,
        }
    }

    /// Create refinements for Go
    pub fn go(size: usize) -> Self {
        let mut invariants = Vec::new();
        let board_size = size * size;

        // All cells valid
        invariants.push(Predicate::Forall(
            "i".to_string(),
            Box::new(Predicate::implies(
                Predicate::and(
                    Predicate::ge(Predicate::var("i"), Predicate::Int(0)),
                    Predicate::lt(Predicate::var("i"), Predicate::Int(board_size as i64)),
                ),
                Predicate::and(
                    Predicate::ge(
                        Predicate::App(
                            "board_get".to_string(),
                            vec![Predicate::var("board"), Predicate::var("i")],
                        ),
                        Predicate::Int(0),
                    ),
                    Predicate::le(
                        Predicate::App(
                            "board_get".to_string(),
                            vec![Predicate::var("board"), Predicate::var("i")],
                        ),
                        Predicate::Int(2),
                    ),
                ),
            )),
        ));

        // Ko rule: can't repeat previous board state
        // (simplified - full rule needs hash comparison)
        invariants.push(Predicate::ne(
            Predicate::App("board_hash".to_string(), vec![Predicate::var("board")]),
            Predicate::var("ko_hash"),
        ));

        GameStateRefinement {
            width: size,
            height: size,
            cell_min: 0,
            cell_max: 2,
            cell_empty: 0,
            invariants,
        }
    }

    /// Get the combined invariant predicate
    pub fn combined_invariant(&self) -> Predicate {
        self.invariants
            .iter()
            .cloned()
            .fold(Predicate::true_pred(), Predicate::and)
    }

    /// Get refined type for board
    pub fn board_type(&self) -> RefinedType {
        let base = Type::Array {
            element: Box::new(Type::I64),
            size: Some(self.width * self.height),
        };
        RefinedType::new(base, "board", Some(self.combined_invariant()))
    }
}

// =============================================================================
// Move Refinements
// =============================================================================

/// Preconditions for a move to be legal
#[derive(Debug, Clone)]
pub struct MoveRefinement {
    /// The position must be in bounds
    pub in_bounds: Predicate,
    /// The cell must be empty (for placement games)
    pub cell_empty: Predicate,
    /// It must be the correct player's turn
    pub correct_turn: Predicate,
    /// Game must not be terminal
    pub non_terminal: Predicate,
    /// Additional game-specific constraints
    pub custom: Vec<Predicate>,
}

impl MoveRefinement {
    /// Create move refinements for Tic-Tac-Toe
    pub fn tic_tac_toe(row: &str, col: &str) -> Self {
        // In bounds: 0 <= row < 3 && 0 <= col < 3
        let in_bounds = Predicate::and(
            Predicate::and(
                Predicate::ge(Predicate::var(row), Predicate::Int(0)),
                Predicate::lt(Predicate::var(row), Predicate::Int(3)),
            ),
            Predicate::and(
                Predicate::ge(Predicate::var(col), Predicate::Int(0)),
                Predicate::lt(Predicate::var(col), Predicate::Int(3)),
            ),
        );

        // Cell empty: board[row * 3 + col] == 0
        let idx = Predicate::Arith(
            ArithOp::Add,
            Box::new(Predicate::Arith(
                ArithOp::Mul,
                Box::new(Predicate::var(row)),
                Box::new(Predicate::Int(3)),
            )),
            Box::new(Predicate::var(col)),
        );
        let cell_empty = Predicate::eq(
            Predicate::App("board_get".to_string(), vec![Predicate::var("board"), idx]),
            Predicate::Int(0),
        );

        // Not terminal: no winner and board not full
        let non_terminal = Predicate::and(
            Predicate::eq(
                Predicate::App("winner".to_string(), vec![Predicate::var("board")]),
                Predicate::Int(0),
            ),
            Predicate::lt(Predicate::var("move_count"), Predicate::Int(9)),
        );

        MoveRefinement {
            in_bounds,
            cell_empty,
            correct_turn: Predicate::true_pred(), // Implicit in state
            non_terminal,
            custom: vec![],
        }
    }

    /// Create move refinements for Go
    pub fn go(row: &str, col: &str, board_size: usize) -> Self {
        let size = board_size as i64;

        let in_bounds = Predicate::and(
            Predicate::and(
                Predicate::ge(Predicate::var(row), Predicate::Int(0)),
                Predicate::lt(Predicate::var(row), Predicate::Int(size)),
            ),
            Predicate::and(
                Predicate::ge(Predicate::var(col), Predicate::Int(0)),
                Predicate::lt(Predicate::var(col), Predicate::Int(size)),
            ),
        );

        let idx = Predicate::Arith(
            ArithOp::Add,
            Box::new(Predicate::Arith(
                ArithOp::Mul,
                Box::new(Predicate::var(row)),
                Box::new(Predicate::Int(size)),
            )),
            Box::new(Predicate::var(col)),
        );
        let cell_empty = Predicate::eq(
            Predicate::App("board_get".to_string(), vec![Predicate::var("board"), idx]),
            Predicate::Int(0),
        );

        // Not suicide: move must leave at least one liberty (simplified)
        let not_suicide = Predicate::App(
            "has_liberty_after".to_string(),
            vec![
                Predicate::var("board"),
                Predicate::var(row),
                Predicate::var(col),
            ],
        );

        // Not ko: result board hash must differ from ko hash
        let not_ko = Predicate::ne(
            Predicate::App(
                "result_hash".to_string(),
                vec![
                    Predicate::var("board"),
                    Predicate::var(row),
                    Predicate::var(col),
                ],
            ),
            Predicate::var("ko_hash"),
        );

        MoveRefinement {
            in_bounds,
            cell_empty,
            correct_turn: Predicate::true_pred(),
            non_terminal: Predicate::true_pred(),
            custom: vec![not_suicide, not_ko],
        }
    }

    /// Get the combined precondition
    pub fn precondition(&self) -> Predicate {
        let base = Predicate::and(
            Predicate::and(self.in_bounds.clone(), self.cell_empty.clone()),
            Predicate::and(self.correct_turn.clone(), self.non_terminal.clone()),
        );

        self.custom.iter().cloned().fold(base, Predicate::and)
    }
}

// =============================================================================
// Postconditions and State Transitions
// =============================================================================

/// Postconditions that must hold after a move
#[derive(Debug, Clone)]
pub struct TransitionRefinement {
    /// The target cell is now occupied
    pub cell_filled: Predicate,
    /// Move count increased by 1
    pub move_count_incremented: Predicate,
    /// Player switched
    pub player_switched: Predicate,
    /// All other cells unchanged
    pub frame_preserved: Predicate,
}

impl TransitionRefinement {
    /// Create transition refinements for placement games
    pub fn placement_game(row: &str, col: &str, width: usize) -> Self {
        // Cell filled: result[idx] != 0
        let idx = Predicate::Arith(
            ArithOp::Add,
            Box::new(Predicate::Arith(
                ArithOp::Mul,
                Box::new(Predicate::var(row)),
                Box::new(Predicate::Int(width as i64)),
            )),
            Box::new(Predicate::var(col)),
        );
        let cell_filled = Predicate::ne(
            Predicate::App(
                "board_get".to_string(),
                vec![Predicate::var("result"), idx.clone()],
            ),
            Predicate::Int(0),
        );

        // Move count incremented
        let move_count_incremented = Predicate::eq(
            Predicate::var("result_move_count"),
            Predicate::Arith(
                ArithOp::Add,
                Box::new(Predicate::var("move_count")),
                Box::new(Predicate::Int(1)),
            ),
        );

        // Player switched
        let player_switched = Predicate::ne(
            Predicate::var("result_player"),
            Predicate::var("current_player"),
        );

        // Frame preservation: all cells except target unchanged
        let board_size = width * width; // Assuming square board
        let frame_preserved = Predicate::Forall(
            "i".to_string(),
            Box::new(Predicate::implies(
                Predicate::and(
                    Predicate::and(
                        Predicate::ge(Predicate::var("i"), Predicate::Int(0)),
                        Predicate::lt(Predicate::var("i"), Predicate::Int(board_size as i64)),
                    ),
                    Predicate::ne(Predicate::var("i"), idx),
                ),
                Predicate::eq(
                    Predicate::App(
                        "board_get".to_string(),
                        vec![Predicate::var("result"), Predicate::var("i")],
                    ),
                    Predicate::App(
                        "board_get".to_string(),
                        vec![Predicate::var("board"), Predicate::var("i")],
                    ),
                ),
            )),
        );

        TransitionRefinement {
            cell_filled,
            move_count_incremented,
            player_switched,
            frame_preserved,
        }
    }

    /// Get combined postcondition
    pub fn postcondition(&self) -> Predicate {
        Predicate::and(
            Predicate::and(
                self.cell_filled.clone(),
                self.move_count_incremented.clone(),
            ),
            Predicate::and(self.player_switched.clone(), self.frame_preserved.clone()),
        )
    }
}

// =============================================================================
// Game Rule Verifier
// =============================================================================

/// Verifier for game state and move legality
pub struct GameRuleVerifier {
    checker: RefinementChecker,
    state_refinement: GameStateRefinement,
    /// Cache of verified moves (state_hash, move) -> result
    verified_cache: HashMap<(u64, usize), bool>,
}

impl GameRuleVerifier {
    /// Create a new verifier for a game
    pub fn new(state_refinement: GameStateRefinement) -> Self {
        GameRuleVerifier {
            checker: RefinementChecker::new(),
            state_refinement,
            verified_cache: HashMap::new(),
        }
    }

    /// Create for Tic-Tac-Toe
    pub fn tic_tac_toe() -> Self {
        Self::new(GameStateRefinement::tic_tac_toe())
    }

    /// Create for Go
    pub fn go(size: usize) -> Self {
        Self::new(GameStateRefinement::go(size))
    }

    /// Verify that a state satisfies all invariants
    pub fn verify_state(&mut self, state_repr: &StateRepr) -> RefinementResult {
        // Assume the state representation as context
        for (name, value) in &state_repr.bindings {
            self.checker
                .assume(Predicate::eq(Predicate::var(name), Predicate::Int(*value)));
        }

        // Check combined invariant
        self.checker
            .check(&self.state_refinement.combined_invariant())
    }

    /// Verify that a move is legal in the current state
    pub fn verify_move(&mut self, row: usize, col: usize) -> RefinementResult {
        let move_ref = MoveRefinement::tic_tac_toe("row", "col");

        // Bind move coordinates
        self.checker.assume(Predicate::eq(
            Predicate::var("row"),
            Predicate::Int(row as i64),
        ));
        self.checker.assume(Predicate::eq(
            Predicate::var("col"),
            Predicate::Int(col as i64),
        ));

        // Check precondition
        self.checker.check(&move_ref.precondition())
    }

    /// Verify state transition (pre -> post)
    pub fn verify_transition(&mut self, row: usize, col: usize) -> RefinementResult {
        let transition =
            TransitionRefinement::placement_game("row", "col", self.state_refinement.width);

        // Assume move coordinates
        self.checker.assume(Predicate::eq(
            Predicate::var("row"),
            Predicate::Int(row as i64),
        ));
        self.checker.assume(Predicate::eq(
            Predicate::var("col"),
            Predicate::Int(col as i64),
        ));

        // Check postcondition
        self.checker.check(&transition.postcondition())
    }
}

/// State representation for verification
#[derive(Debug, Clone)]
pub struct StateRepr {
    pub bindings: HashMap<String, i64>,
}

impl StateRepr {
    pub fn new() -> Self {
        StateRepr {
            bindings: HashMap::new(),
        }
    }

    pub fn bind(&mut self, name: &str, value: i64) {
        self.bindings.insert(name.to_string(), value);
    }
}

impl Default for StateRepr {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_bounds() {
        // Valid positions
        assert!(Position::<Dim3x3>::new(0, 0).is_some());
        assert!(Position::<Dim3x3>::new(2, 2).is_some());

        // Invalid positions
        assert!(Position::<Dim3x3>::new(3, 0).is_none());
        assert!(Position::<Dim3x3>::new(0, 3).is_none());
    }

    #[test]
    fn test_position_index_conversion() {
        let pos = Position::<Dim3x3>::new(1, 2).unwrap();
        assert_eq!(pos.to_index(), 5); // 1*3 + 2 = 5

        let pos2 = Position::<Dim3x3>::from_index(5).unwrap();
        assert_eq!(pos, pos2);
    }

    #[test]
    fn test_cell_constraints() {
        assert!(TicTacToeCell::is_valid(0));
        assert!(TicTacToeCell::is_valid(1));
        assert!(TicTacToeCell::is_valid(2));
        assert!(!TicTacToeCell::is_valid(3));
        assert!(!TicTacToeCell::is_valid(-1));

        assert!(TicTacToeCell::is_empty(0));
        assert!(!TicTacToeCell::is_empty(1));
    }

    #[test]
    fn test_chess_cell_constraints() {
        assert!(ChessCell::is_valid(-6)); // Black king
        assert!(ChessCell::is_valid(6)); // White king
        assert!(ChessCell::is_valid(0)); // Empty
        assert!(!ChessCell::is_valid(7)); // Invalid
    }

    #[test]
    fn test_game_state_refinement() {
        let refinement = GameStateRefinement::tic_tac_toe();
        assert_eq!(refinement.width, 3);
        assert_eq!(refinement.height, 3);
        assert_eq!(refinement.cell_empty, 0);

        let board_type = refinement.board_type();
        assert!(board_type.is_refined());
    }

    #[test]
    fn test_move_refinement() {
        let move_ref = MoveRefinement::tic_tac_toe("row", "col");
        let precond = move_ref.precondition();

        // The precondition should be an And of multiple conditions
        match precond {
            Predicate::And(_, _) => (), // Expected
            _ => panic!("Expected And predicate"),
        }
    }

    #[test]
    fn test_position_refinement_predicate() {
        let pred = Position::<Dim3x3>::refinement_predicate();

        // Should encode bounds checking
        match pred {
            Predicate::And(_, _) => (), // Expected
            _ => panic!("Expected And predicate"),
        }
    }

    #[test]
    fn test_go_refinement() {
        let refinement = GameStateRefinement::go(19);
        assert_eq!(refinement.width, 19);
        assert_eq!(refinement.height, 19);

        let move_ref = MoveRefinement::go("row", "col", 19);
        assert_eq!(move_ref.custom.len(), 2); // not_suicide, not_ko
    }

    #[test]
    fn test_verifier_creation() {
        let verifier = GameRuleVerifier::tic_tac_toe();
        assert_eq!(verifier.state_refinement.width, 3);

        let go_verifier = GameRuleVerifier::go(19);
        assert_eq!(go_verifier.state_refinement.width, 19);
    }

    #[test]
    fn test_state_repr() {
        let mut repr = StateRepr::new();
        repr.bind("move_count", 3);
        repr.bind("current_player", 2);

        assert_eq!(repr.bindings.get("move_count"), Some(&3));
        assert_eq!(repr.bindings.get("current_player"), Some(&2));
    }

    #[test]
    fn test_transition_refinement() {
        let transition = TransitionRefinement::placement_game("row", "col", 3);
        let postcond = transition.postcondition();

        // Should combine all postconditions
        match postcond {
            Predicate::And(_, _) => (),
            _ => panic!("Expected And predicate"),
        }
    }
}
