//! MIR Analysis Framework
//!
//! This module provides analysis passes for MIR optimization,
//! including dominance analysis, data flow analysis, and SSA validation.

pub mod alias;
pub mod dominators;
pub mod loops;
pub mod ssa_validator;

// Re-export commonly used analysis types
pub use alias::{AliasAnalysis, AliasResult, MemoryLocation, ModuleAliasAnalysis, PointsToSet};
pub use dominators::DominatorAnalysis;
pub use loops::{DominatorTree, LoopAnalysis, LoopOptimizer, NaturalLoop};
pub use ssa_validator::{SSAValidationError, SSAValidationResult, SSAValidationWarning, SSAValidator};
