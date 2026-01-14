//! MIR Analysis Framework
//!
//! This module provides analysis passes for MIR optimization,
//! including dominance analysis, data flow analysis, and SSA validation.

pub mod dominators;
pub mod ssa_validator;

// Re-export commonly used analysis types
pub use dominators::DominatorAnalysis;
pub use ssa_validator::{SSAValidator, SSAError, Location};