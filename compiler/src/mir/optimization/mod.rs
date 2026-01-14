//! MIR Optimization Framework
//!
//! This module provides optimization passes for MIR,
//! implementing proven algorithms from academic literature.

pub mod pass_manager;
pub mod constant_propagation;

// Re-export commonly used optimization types
pub use pass_manager::{
    MIRPass, AnalysisPass, PassManager, OptimizationLevel,
    DeadCodeElimination, CommonSubexpressionElimination,
    LoopInvariantCodeMotion, create_default_pass_manager
};

// Re-export the full SCCP implementation
pub use constant_propagation::{ConstantPropagation, LatticeValue};