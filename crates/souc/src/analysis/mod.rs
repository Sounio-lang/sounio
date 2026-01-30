//! Program Analysis Module for Sounio
//!
//! Provides comprehensive static analysis capabilities including:
//!
//! - **Abstract Interpretation**: Precise invariant discovery using
//!   interval, sign, congruence, and octagon domains
//! - **Dead Code Detection**: Finding unreachable and unused code
//! - **Code Metrics**: LOC, complexity, and other measurements
//!
//! # Abstract Interpretation
//!
//! The abstract interpretation framework enables:
//! - Array bounds checking
//! - Loop invariant discovery
//! - Automatic precondition inference
//! - Integration with refinement types
//!
//! See the [`r#abstract`] module for details.

pub mod r#abstract;

// Re-export abstract interpretation types
pub use r#abstract::{
    AbstractDomain, AbstractInterpreter, AbstractState, AnalysisConfig, AnalysisResult,
    ArrayBoundsAnalysis, BoolDomain, Bound, Congruence, FixpointConfig, FixpointEngine,
    FixpointResult, Flat, Interval, IntervalCongruence, IntervalSign, IntervalTransfer, Invariant,
    Lifted, LoopInvariant, NumericDomain, NumericTransfer, Octagon, PowerSet, Product,
    RefinementIntegration, SafetyResult, SccDecomposition, Sign, ThresholdWidening, VarId, Weight,
    analyze_function, simple_fixpoint,
};
