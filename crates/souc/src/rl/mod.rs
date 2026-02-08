//! Reinforcement Learning Module for Sounio
//!
//! Native epistemic RL inspired by AlphaZero/MuZero, with key innovations:
//!
//! 1. **Epistemic MCTS**: Q-values as Beta distributions (mean + variance)
//! 2. **Uncertainty-guided exploration**: PUCT bonus for high-variance nodes
//! 3. **Provenance trees**: Merkle DAGs for explainable trajectories
//! 4. **Type-safe game states**: Compile-time refinements for valid moves
//! 5. **Neural effects**: Differentiable policy/value with gradient flow
//! 6. **MuZero latent dynamics**: Learned world model with epistemic uncertainty
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     AlphaSounio                               │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │ MCTSTree     │  │ NeuralNet    │  │ SelfPlay     │          │
//! │  │ (epistemic   │  │ (policy +    │  │ (training    │          │
//! │  │  nodes)      │  │  value)      │  │  loop)       │          │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
//! │         │                 │                 │                   │
//! │         └─────────────────┼─────────────────┘                   │
//! │                           │                                     │
//! │                    ┌──────▼───────┐                             │
//! │                    │ GameState    │                             │
//! │                    │ (typed +     │                             │
//! │                    │  epistemic)  │                             │
//! │                    └──────────────┘                             │
//! │                           │                                     │
//! │  ┌────────────────────────┼────────────────────────┐           │
//! │  │              MuZero Latent Dynamics              │           │
//! │  │  h(o)→s    f(s)→(π,v)    g(s,a)→(s',r)          │           │
//! │  │  represent   predict      dynamics               │           │
//! │  │  + epistemic uncertainty propagation             │           │
//! │  └─────────────────────────────────────────────────┘           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Innovation: Epistemic Q-Values
//!
//! Traditional RL uses scalar Q-values. AlphaSounio uses Beta distributions:
//!
//! ```text
//! Q_traditional = 0.75
//! Q_epistemic = Beta(α=15, β=5) → mean=0.75, variance=0.009
//!
//! High variance = "I don't know" → explore more
//! Low variance = "I'm confident" → exploit
//! ```
//!
//! This enables principled exploration via epistemic uncertainty (active inference).
//!
//! # MuZero with Epistemic States
//!
//! Our MuZero variant adds epistemic uncertainty to latent states:
//!
//! ```text
//! Standard MuZero:  s = [f1, f2, ..., fn]  (point estimate)
//! Epistemic MuZero: s = [(f1,σ1), (f2,σ2), ..., (fn,σn)]  (mean + variance)
//! ```
//!
//! Benefits:
//! - Model knows when it doesn't know (OOD detection)
//! - MCTS explores high-uncertainty model predictions
//! - Training penalizes overconfident wrong predictions

pub mod game;
pub mod mcts;
pub mod muzero;
pub mod neural;
pub mod refinements;
pub mod selfplay;

pub use game::{Action, GameOutcome, GameState, GameTrait, Player};
pub use mcts::{MCTSConfig, MCTSNode, MCTSTree, SearchResult};
pub use muzero::{
    DummyMuZeroModel, DynamicsResult, EpistemicPrediction, LatentMCTSConfig, LatentMCTSNode,
    LatentMCTSTree, LatentSearchStats, LatentState, MuZeroModel, MuZeroTarget,
};
pub use neural::{NeuralConfig, NeuralEval, PolicyValue, TrainingExample};
pub use refinements::{
    BoardDimensions, CellConstraint, Dim3x3, Dim8x8, Dim15x15, Dim19x19, GameRuleVerifier,
    GameStateRefinement, MoveRefinement, Position, StateRepr, TransitionRefinement,
};
pub use selfplay::{SelfPlayConfig, Trajectory};
