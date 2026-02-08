//! Layout Synthesis Module - Days 38-39
//!
//! Uses semantic distance from the ontology to inform memory layout decisions.
//! The hypothesis: concepts that are semantically close should be physically
//! close in memory to improve cache performance.
//!
//! # Algorithm
//!
//! 1. **Extract** concepts used in HIR (Knowledge[T, ...] types)
//! 2. **Build** distance matrix using ontology hierarchy
//! 3. **Cluster** concepts by semantic proximity + co-occurrence
//! 4. **Generate** layout plan assigning clusters to memory regions
//! 5. **Measure** cache performance to validate the hypothesis
//!
//! # The Hypothesis (Day 38)
//!
//! ```text
//! If concepts A and B are semantically close (low ontology distance),
//! and they are accessed together in code,
//! then placing them physically close in memory will improve cache hit rate.
//! ```
//!
//! Day 38 validated this hypothesis: +15.9% improvement with semantic clustering.
//!
//! # Participatory Compilation (Day 39)
//!
//! Developers can influence layout decisions through annotations:
//! - `#[colocate(a, b)]` - concepts should be in the same cluster
//! - `#[separate(a, b)]` - concepts should NOT be colocated
//! - `#[hot]` / `#[cold]` - force region assignment
//! - `#[explain_layout]` - request detailed layout explanation

pub mod cluster;
pub mod constraint;
pub mod diagnostics;
pub mod distance;
pub mod extract;
pub mod instrument;
pub mod plan;
pub mod report;
pub mod solver;
pub mod visualize;

pub use cluster::{Cluster, ClusteringResult, cluster_concepts};
pub use constraint::{ConstraintSet, ConstraintSource, ForcedRegion, LayoutConstraint};
pub use diagnostics::{
    DiagnosticLevel, LayoutDiagnostic, format_diagnostics, generate_solver_diagnostics,
    layout_summary_diagnostic, validate_constraints_diagnostic,
};
pub use distance::DistanceMatrix;
pub use extract::{ConceptUsage, extract_concepts_from_hir, extract_concepts_from_types};
pub use instrument::{CacheInstrumentation, CacheStats, LayoutComparison, compare_layouts};
pub use plan::{LayoutConfig, LayoutPlan, MemoryRegion, generate_layout};
pub use report::generate_report;
pub use solver::{SolverResult, solve_constraints};
pub use visualize::{generate_ascii, generate_mermaid, generate_summary, generate_table};

use crate::ontology::native::NativeOntology;

/// Main entry point for layout synthesis
pub struct LayoutSynthesizer<'a> {
    /// Reference to the ontology
    ontology: &'a NativeOntology,
    /// Configuration
    config: LayoutConfig,
}

impl<'a> LayoutSynthesizer<'a> {
    /// Create a new layout synthesizer
    pub fn new(ontology: &'a NativeOntology, config: LayoutConfig) -> Self {
        Self { ontology, config }
    }

    /// Synthesize a layout plan from concept usage
    pub fn synthesize(&self, usage: &ConceptUsage) -> LayoutPlan {
        if usage.concepts.is_empty() {
            return LayoutPlan::empty();
        }

        // Build distance matrix
        let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
        let distances = DistanceMatrix::build(&concepts, self.ontology);

        // Cluster by semantic proximity + co-occurrence
        let clustering = cluster_concepts(usage, &distances, self.config.max_clusters);

        // Generate layout plan
        generate_layout(clustering, self.config.clone())
    }

    /// Synthesize a layout plan with developer constraints (Day 39)
    ///
    /// This is "participatory compilation" - the developer can influence
    /// layout decisions through annotations like #[colocate], #[separate],
    /// #[hot], and #[cold].
    pub fn synthesize_with_constraints(
        &self,
        usage: &ConceptUsage,
        constraints: &ConstraintSet,
    ) -> SolverResult {
        if usage.concepts.is_empty() {
            let empty_clustering = ClusteringResult::empty();
            return SolverResult {
                layout: LayoutPlan::empty(),
                satisfied: Vec::new(),
                conflicts: Vec::new(),
                warnings: Vec::new(),
                original_clustering: empty_clustering.clone(),
                modified_clustering: empty_clustering,
            };
        }

        // Build distance matrix
        let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
        let distances = DistanceMatrix::build(&concepts, self.ontology);

        // Cluster by semantic proximity + co-occurrence
        let clustering = cluster_concepts(usage, &distances, self.config.max_clusters);

        // Apply constraints and generate layout
        solve_constraints(clustering, constraints, self.config.clone())
    }

    /// Synthesize and measure cache effectiveness
    pub fn synthesize_and_measure(
        &self,
        usage: &ConceptUsage,
        access_pattern: &[String],
    ) -> (LayoutPlan, LayoutComparison) {
        let plan = self.synthesize(usage);

        // Convert access pattern to concept accesses
        let accesses: Vec<_> = access_pattern
            .iter()
            .filter(|s| usage.concepts.contains(*s))
            .cloned()
            .collect();

        // Measure baseline vs optimized
        let comparison = instrument::compare_layouts(&accesses, &plan, self.config.cache_size);

        (plan, comparison)
    }

    /// Synthesize with constraints and measure cache effectiveness
    pub fn synthesize_with_constraints_and_measure(
        &self,
        usage: &ConceptUsage,
        constraints: &ConstraintSet,
        access_pattern: &[String],
    ) -> (SolverResult, LayoutComparison) {
        let result = self.synthesize_with_constraints(usage, constraints);

        // Convert access pattern to concept accesses
        let accesses: Vec<_> = access_pattern
            .iter()
            .filter(|s| usage.concepts.contains(*s))
            .cloned()
            .collect();

        // Measure baseline vs optimized
        let comparison =
            instrument::compare_layouts(&accesses, &result.layout, self.config.cache_size);

        (result, comparison)
    }
}

/// Layout hint for HIR nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayoutHint {
    /// Allocate on stack (hot data, L1/L2 friendly)
    Stack,
    /// Allocate in bump arena (warm data, L2/L3)
    Arena,
    /// Allocate on heap (cold data, RAM)
    Heap,
}

impl From<MemoryRegion> for LayoutHint {
    fn from(region: MemoryRegion) -> Self {
        match region {
            MemoryRegion::Hot => LayoutHint::Stack,
            MemoryRegion::Warm => LayoutHint::Arena,
            MemoryRegion::Cold => LayoutHint::Heap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_hint_from_region() {
        assert_eq!(LayoutHint::from(MemoryRegion::Hot), LayoutHint::Stack);
        assert_eq!(LayoutHint::from(MemoryRegion::Warm), LayoutHint::Arena);
        assert_eq!(LayoutHint::from(MemoryRegion::Cold), LayoutHint::Heap);
    }
}
