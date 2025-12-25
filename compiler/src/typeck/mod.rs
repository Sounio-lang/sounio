//! Type Checking with Semantic Distance
//!
//! This module extends traditional Hindley-Milner type checking
//! with semantic distance for ontological types.
//!
//! Key features:
//! - Distance-aware unification
//! - Configurable thresholds (#[compat(...)])
//! - Rich diagnostics with distance breakdown
//! - "Did you mean?" suggestions
//! - Automatic coercion insertion

pub mod coercion_insert;
pub mod diagnostics;
pub mod hooks;
pub mod suggestions;
pub mod threshold;
pub mod unify_distance;

pub use coercion_insert::CoercionInserter;
pub use diagnostics::CompatibilityDiagnostic;
pub use hooks::SemanticTypeChecker;
pub use suggestions::{ScoredSuggestion, SuggestionEngine};
pub use threshold::{ResolvedThreshold, ThresholdLevel, ThresholdResolver};
pub use unify_distance::{CoercionKind, CoercionSite, UnificationContext, UnificationError};

use crate::ontology::alignment::AlignmentIndex;
use crate::ontology::distance::SemanticDistanceIndex;
use std::sync::Arc;

/// Create a fully configured semantic type checker
pub fn create_semantic_type_checker(
    distance_index: Arc<SemanticDistanceIndex>,
    alignment_index: Arc<AlignmentIndex>,
) -> SemanticTypeChecker {
    let suggestion_engine = Arc::new(SuggestionEngine::new(Arc::clone(&distance_index)));

    SemanticTypeChecker::new(distance_index, alignment_index, suggestion_engine)
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_module_creation() {
        // Basic smoke test - more comprehensive tests in submodules
    }
}
