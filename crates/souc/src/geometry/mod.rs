//! Geometry Symbolic Engine for Sounio
//!
//! Native neuro-symbolic geometry reasoning inspired by AlphaGeometry.
//! This module provides:
//!
//! - Geometric primitives (Point, Line, Circle) with epistemic semantics
//! - Predicate graph (proof state with confidence propagation)
//! - Forward-chaining deduction (DD) with refinement checking
//! - Algebraic reasoning (AR) with unit validation
//! - Integration with effects system for NeSy loop
//! - Epistemic pruning with uncertainty-based branch control
//! - GeometryReasoning effect for algebraic effect integration
//!
//! # Key Innovation
//!
//! Every predicate is `Knowledge<Predicate>` - deductions automatically
//! propagate confidence and provenance. Rules are refinement-checked
//! (Z3 proves application). Low-confidence branches trigger neural
//! suggestions via effects.

pub mod algebraic;
pub mod alpha_geo_zero;
pub mod engine;
pub mod final_integration;
pub mod geo_game;
pub mod geo_training;
pub mod imo_benchmark;
pub mod imo_parser;
pub mod predicates;
pub mod primitives;
pub mod proof_state;
pub mod reasoning_effect;
pub mod rules;
pub mod self_play;
pub mod showcase;
pub mod synthetic;

pub use algebraic::{AlgebraicReasoner, Expression};
pub use engine::{DeductionResult, EngineConfig, EpistemicPruner, PruningDecision, SymbolicEngine};
pub use predicates::{Predicate, PredicateKind, PredicatePattern};
pub use primitives::{Angle, Circle, GeometryPrimitive, Line, Point, Segment};
pub use proof_state::{ProofState, ProofStep, ProvenanceNode};
pub use reasoning_effect::{
    geometry_reasoning_effect, GeometryReasoningHandler, MerkleProof, NeSyHandler, NeuralSuggester,
    PureSymbolicHandler,
};
pub use rules::{GeometryRule, RuleDatabase, RuleMatch};

// Geometry Game for AlphaGeometry-style self-play
pub use geo_game::{
    generate_proof_game, generate_proof_game_random, isoceles_perpendicular, midpoint_theorem,
    triangle_congruence_sas, GeoAction, GeoConstruction, GeoGameConfig, GeoProofGame,
    ProofGameEpisode, TrajectoryStep,
};

// Geometry Self-Play Training
pub use geo_training::{
    test_midpoint_theorem, test_triangle_congruence, train_geometry_prover, GeoNetworkEvaluator,
    GeoNeuralNetwork, GeoProblem, GeoSelfPlayTrainer, GeoTrainingConfig, GeoTrainingExample,
    IterationResult, ProblemGenerator, ProblemType, TrainingStats, UniformGeoNetwork,
};

// IMO Benchmark
pub use imo_benchmark::{
    generate_synthetic_training, get_problem, get_problems_by_difficulty, get_problems_by_year,
    imo_ag_30, run_baseline_benchmark, run_benchmark, run_benchmark_on_problems, BenchmarkConfig,
    BenchmarkResult, BenchmarkStats,
};
pub use imo_parser::{parse_ag_problem, parse_imo_problem, AGParser, IMOProblem, ParseError};

// AlphaGeoZero: Full Self-Play Training Loop with Epistemic MCTS
pub use alpha_geo_zero::{
    AlphaGeoZeroConfig, AlphaGeoZeroLoop, AlphaGeoZeroTrainer, BufferStats, EpistemicMCTSNode,
    EpistemicMCTSTree, LoopConfig, LoopStats, MCTSStats, ProblemCurriculum, SelfPlayGenerator,
    SelfPlayStats, TrainingConfig, TrainingResult, VariancePriorityBuffer,
};

// Synthetic Problem Generation
pub use synthetic::{
    generate_batch, generate_curriculum_batch, GeneratorConfig, GeneratorStats, ProblemTemplate,
    ProblemVariation, SyntheticProblem, SyntheticProblemGenerator,
};

// Full Self-Play Loop
pub use self_play::{
    run_default_self_play, AlphaGeoZeroSelfPlay, BenchmarkSnapshot, SelfPlayConfig,
    SelfPlayStats as FullSelfPlayStats, TemplateStats, TrainingStepResult, VarianceReplayBuffer,
};

// Final Integration: Complete AlphaGeoZero System
pub use final_integration::{
    main_alphageozero, run_full_training, run_quick_test, run_with_config, AlphaGeoZeroFull,
    AlphaGeoZeroFullConfig, BenchmarkSnapshot as FinalBenchmarkSnapshot, FullTrainingStats,
    TemplatePerfStats,
};
