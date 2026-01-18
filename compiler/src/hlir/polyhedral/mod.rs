//! Polyhedral Optimization Infrastructure for Sounio
//!
//! This module provides polyhedral analysis and transformation infrastructure
//! for optimizing scientific kernels with nested loops. The polyhedral model
//! represents loop nests as integer polyhedra, enabling powerful loop transformations.
//!
//! # Background
//!
//! The polyhedral model is based on:
//! - Feautrier (1991) "Dataflow analysis of array and scalar references"
//! - Pugh (1991) "The Omega Test: a fast and practical integer programming algorithm"
//! - Bondhugula et al. (2008) "A practical automatic polyhedral parallelizer and locality optimizer"
//! - Verdoolaege (2010) "isl: An Integer Set Library for the Polyhedral Model"
//!
//! # Key Concepts
//!
//! - **Iteration Space**: The set of all iterations of a loop nest, represented as an integer polyhedron
//! - **Access Function**: An affine map from iteration space to array indices
//! - **Dependence Polyhedron**: Captures data dependencies between statement instances
//! - **Schedule**: A mapping from statement instances to execution time
//!
//! # Module Structure
//!
//! - `domain.rs`: Iteration domains and integer sets
//! - `affine.rs`: Affine expressions and maps
//! - `analysis.rs`: Dependence analysis and loop detection
//! - `transform.rs`: Loop transformations (tiling, interchange, fusion)
//! - `codegen.rs`: Generate transformed HLIR from polyhedral representation

pub mod domain;
pub mod affine;
pub mod analysis;
pub mod transform;
pub mod codegen;

// Re-export main types
pub use domain::{IntegerSet, Constraint, IterationDomain};
pub use affine::{AffineExpr, AffineMap, AccessRelation};
pub use analysis::{
    PolyhedralAnalysis, LoopNest, ScopInfo, StatementInfo,
    DependenceInfo, DependenceType,
};
pub use transform::{
    PolyhedralTransform, TileTransform, InterchangeTransform,
    FusionTransform, Schedule, ScheduleNode,
};
pub use codegen::PolyhedralCodegen;

use crate::hlir::HlirFunction;

/// Result type for polyhedral operations
pub type PolyResult<T> = Result<T, PolyError>;

/// Errors that can occur during polyhedral analysis or transformation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolyError {
    /// Loop is not affine (bounds or accesses are not affine functions)
    NonAffineLoop { reason: String },
    /// Cannot compute dependence (e.g., due to unknown bounds)
    DependenceAnalysisFailed { reason: String },
    /// Transformation is not legal (would violate dependencies)
    IllegalTransformation { reason: String },
    /// Code generation failed
    CodegenFailed { reason: String },
    /// Internal error
    Internal { reason: String },
}

impl std::fmt::Display for PolyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolyError::NonAffineLoop { reason } => {
                write!(f, "Non-affine loop: {}", reason)
            }
            PolyError::DependenceAnalysisFailed { reason } => {
                write!(f, "Dependence analysis failed: {}", reason)
            }
            PolyError::IllegalTransformation { reason } => {
                write!(f, "Illegal transformation: {}", reason)
            }
            PolyError::CodegenFailed { reason } => {
                write!(f, "Code generation failed: {}", reason)
            }
            PolyError::Internal { reason } => {
                write!(f, "Internal error: {}", reason)
            }
        }
    }
}

impl std::error::Error for PolyError {}

/// Main entry point for polyhedral optimization
///
/// Analyzes a function for polyhedral opportunities and applies
/// transformations to optimize loop nests.
pub fn optimize_function(func: &mut HlirFunction) -> PolyResult<bool> {
    // Step 1: Analyze function to find Static Control Parts (SCOPs)
    let analysis = PolyhedralAnalysis::new();
    let scops = analysis.find_scops(func)?;

    if scops.is_empty() {
        return Ok(false);
    }

    let mut modified = false;

    // Step 2: For each SCOP, analyze dependencies and apply transformations
    for scop in scops {
        // Compute dependence polyhedra
        let deps = analysis.compute_dependencies(&scop)?;

        // Apply transformations (tiling is the most important for performance)
        let transform = PolyhedralTransform::new(&scop, &deps);

        // Try to apply automatic tiling for locality
        if let Ok(new_schedule) = transform.auto_tile() {
            // Generate transformed code
            let codegen = PolyhedralCodegen::new();
            codegen.apply_schedule(func, &scop, &new_schedule)?;
            modified = true;
        }
    }

    Ok(modified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poly_error_display() {
        let err = PolyError::NonAffineLoop {
            reason: "index is quadratic".to_string(),
        };
        assert!(err.to_string().contains("Non-affine loop"));
        assert!(err.to_string().contains("quadratic"));
    }
}
