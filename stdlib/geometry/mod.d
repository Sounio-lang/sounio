// Demetrios Standard Library: Geometry Module
//
// Neuro-symbolic geometry reasoning inspired by AlphaGeometry.
// Provides native Demetrios support for:
//
// - Epistemic geometric types (Point, Line, Circle with Knowledge semantics)
// - First-order predicates (collinear, parallel, concyclic, etc.)
// - Forward-chaining deduction (DD) with algebraic verification (AR)
// - Algebraic effects for composable proof search
// - Neural construction suggestion interface
//
// This module demonstrates Demetrios as a native neuro-symbolic language:
// - Types carry confidence and provenance
// - Refinements enforce geometric validity at compile time
// - Effects enable clean separation of deduction/construction/search
// - GPU kernels can accelerate neural components

pub mod types;
pub mod predicates;
pub mod engine;
pub mod symbolic_engine;

// Re-exports for convenience
pub use types::{
    Point, Line, Circle, Segment, Triangle, AngleABC,
    ValidTriangle, EquilateralTriangle, IsoscelesTriangle, RightTriangle,
    ParallelLines, PerpendicularLines, CollinearPoints, ConcyclicPoints,
};

pub use predicates::{
    Predicate, Collinear, Concyclic, Parallel, Perpendicular,
    EqualLength, EqualAngle, RightAngle, OnLine, OnCircle,
    Midpoint, AngleBisector, PerpBisector, CircleCenter, Tangent,
    Similar, Congruent, Construction, ConstructionType,
};

pub use engine::{
    ProofState, ProofStep, DeductionRule, DeductionEngine,
    AlgebraicReasoner, DDAREngine, SearchResult,
    geometry_search_handler, generate_proof_text,
};

pub use symbolic_engine::{
    PredicateEpistemic, PredicateKind, EpistemicPruner, PruningDecision,
    forward_chain, algebraic_simplify, combine_parent_betas,
};
