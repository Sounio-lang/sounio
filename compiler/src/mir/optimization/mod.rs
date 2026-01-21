//! MIR Optimization Framework
//!
//! This module provides optimization passes for MIR,
//! implementing proven algorithms from academic literature.

pub mod pass_manager;
pub mod constant_propagation;
pub mod dead_code_elimination;
pub mod common_subexpression_elimination;
pub mod licm;

#[cfg(feature = "glm")]
pub mod glm_integration;

#[cfg(feature = "glm")]
pub mod ml_guided_optimization;

// Re-export commonly used optimization types
pub use pass_manager::{
    MIRPass, AnalysisPass, PassManager, OptimizationLevel,
    create_default_pass_manager
};
pub use dead_code_elimination::DeadCodeElimination;

// Re-export optimization passes
pub use constant_propagation::{ConstantPropagation, LatticeValue};
pub use common_subexpression_elimination::{
    CommonSubexpressionElimination, Expression, AvailableExpressions
};
pub use licm::{LoopInvariantCodeMotion, LoopInfo};

#[cfg(feature = "glm")]
pub use ml_guided_optimization::MLGuidedOptimization;