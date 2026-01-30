//! Constraint Solver - Day 39
//!
//! Integrates developer constraints with automatic clustering.
//! Handles conflicts, applies forced regions, and generates diagnostics.

use std::collections::{HashMap, HashSet};

use super::cluster::{Cluster, ClusteringResult};
use super::constraint::{ConstraintSet, ForcedRegion, LayoutConstraint};
use super::plan::{LayoutConfig, LayoutPlan, MemoryRegion, generate_layout};

/// Result of constraint solving
#[derive(Debug)]
pub struct SolverResult {
    /// The adjusted layout plan
    pub layout: LayoutPlan,
    /// Constraints that were satisfied
    pub satisfied: Vec<SatisfiedConstraint>,
    /// Constraints that could not be satisfied (conflicts)
    pub conflicts: Vec<ConstraintConflict>,
    /// Warnings (soft constraint violations)
    pub warnings: Vec<ConstraintWarning>,
    /// Original clustering (for comparison)
    pub original_clustering: ClusteringResult,
    /// Modified clustering after constraint application
    pub modified_clustering: ClusteringResult,
}

impl SolverResult {
    /// Check if all constraints were satisfied without conflicts
    pub fn is_success(&self) -> bool {
        self.conflicts.is_empty()
    }

    /// Get total number of issues (conflicts + warnings)
    pub fn issue_count(&self) -> usize {
        self.conflicts.len() + self.warnings.len()
    }
}

/// A constraint that was successfully satisfied
#[derive(Debug, Clone)]
pub struct SatisfiedConstraint {
    pub constraint: LayoutConstraint,
    pub impact: String,
}

/// A conflict between constraints
#[derive(Debug, Clone)]
pub struct ConstraintConflict {
    pub constraint_a: LayoutConstraint,
    pub constraint_b: LayoutConstraint,
    pub reason: String,
}

/// A warning about a constraint
#[derive(Debug, Clone)]
pub struct ConstraintWarning {
    pub constraint: LayoutConstraint,
    pub message: String,
}

/// Solve constraints and adjust layout
pub fn solve_constraints(
    clustering: ClusteringResult,
    constraints: &ConstraintSet,
    config: LayoutConfig,
) -> SolverResult {
    let original_clustering = clustering.clone();
    let mut modified_clustering = clustering;
    let mut satisfied = Vec::new();
    let mut conflicts = Vec::new();
    let mut warnings = Vec::new();

    // Phase 1: Check for conflicts between constraints
    for (i, c1) in constraints.constraints.iter().enumerate() {
        for c2 in constraints.constraints.iter().skip(i + 1) {
            if let Some(conflict) = check_conflict(c1, c2) {
                conflicts.push(conflict);
            }
        }
    }

    // Phase 2: Process colocate constraints (merge clusters)
    for constraint in constraints.colocate_constraints() {
        if let LayoutConstraint::Colocate { concepts, source } = constraint {
            match merge_for_colocate(&mut modified_clustering, concepts) {
                Ok(impact) => {
                    satisfied.push(SatisfiedConstraint {
                        constraint: constraint.clone(),
                        impact,
                    });
                }
                Err(e) => {
                    warnings.push(ConstraintWarning {
                        constraint: constraint.clone(),
                        message: e,
                    });
                }
            }
        }
    }

    // Phase 3: Process separate constraints (split clusters)
    for constraint in constraints.separate_constraints() {
        if let LayoutConstraint::Separate { concepts, source } = constraint {
            match split_for_separate(&mut modified_clustering, concepts) {
                Ok(impact) => {
                    satisfied.push(SatisfiedConstraint {
                        constraint: constraint.clone(),
                        impact,
                    });
                }
                Err(e) => {
                    warnings.push(ConstraintWarning {
                        constraint: constraint.clone(),
                        message: e,
                    });
                }
            }
        }
    }

    // Phase 4: Generate base layout from modified clustering
    let mut layout = generate_layout(modified_clustering.clone(), config);

    // Phase 5: Apply forced regions (override automatic assignment)
    for constraint in constraints.force_region_constraints() {
        if let LayoutConstraint::ForceRegion {
            concept,
            region,
            source,
        } = constraint
        {
            let target_region = match region {
                ForcedRegion::Hot => MemoryRegion::Hot,
                ForcedRegion::Warm => MemoryRegion::Warm,
                ForcedRegion::Cold => MemoryRegion::Cold,
            };

            // Find the concept in layout and update its region
            if let Some(concept_layout) = layout.by_concept.get_mut(concept) {
                let old_region = concept_layout.region;
                concept_layout.region = target_region;

                // Update by_region maps
                if let Some(concepts) = layout.by_region.get_mut(&old_region) {
                    concepts.retain(|c| c != concept);
                }
                layout
                    .by_region
                    .entry(target_region)
                    .or_default()
                    .push(concept.clone());

                satisfied.push(SatisfiedConstraint {
                    constraint: constraint.clone(),
                    impact: format!(
                        "Forced {} from {:?} to {:?}",
                        concept, old_region, target_region
                    ),
                });
            } else {
                warnings.push(ConstraintWarning {
                    constraint: constraint.clone(),
                    message: format!("Concept '{}' not found in layout", concept),
                });
            }
        }
    }

    SolverResult {
        layout,
        satisfied,
        conflicts,
        warnings,
        original_clustering,
        modified_clustering,
    }
}

/// Merge clusters to satisfy a colocate constraint
fn merge_for_colocate(
    clustering: &mut ClusteringResult,
    concepts: &[String],
) -> Result<String, String> {
    // Find which clusters contain these concepts
    let mut cluster_ids: HashSet<usize> = HashSet::new();
    let mut found_concepts: Vec<&String> = Vec::new();

    for concept in concepts {
        for (idx, cluster) in clustering.clusters.iter().enumerate() {
            if cluster.concepts.contains(concept) {
                cluster_ids.insert(idx);
                found_concepts.push(concept);
                break;
            }
        }
    }

    if found_concepts.len() < 2 {
        return Err(format!(
            "Only {} of {} concepts found in clusters",
            found_concepts.len(),
            concepts.len()
        ));
    }

    if cluster_ids.len() <= 1 {
        return Ok("Concepts already colocated in same cluster".to_string());
    }

    // Merge all clusters into the first one
    let cluster_ids: Vec<usize> = cluster_ids.into_iter().collect();
    let target_idx = cluster_ids[0];
    let clusters_to_merge: Vec<usize> = cluster_ids[1..].to_vec();

    // Collect concepts from clusters to merge
    let mut concepts_to_move: Vec<String> = Vec::new();
    let mut total_accesses: u32 = 0;

    for &idx in &clusters_to_merge {
        if let Some(cluster) = clustering.clusters.get(idx) {
            concepts_to_move.extend(cluster.concepts.clone());
            total_accesses += cluster.total_accesses;
        }
    }

    // Add to target cluster
    if let Some(target) = clustering.clusters.get_mut(target_idx) {
        target.concepts.extend(concepts_to_move);
        target.total_accesses += total_accesses;
    }

    // Remove merged clusters (in reverse order to preserve indices)
    let mut sorted_indices = clusters_to_merge.clone();
    sorted_indices.sort_by(|a, b| b.cmp(a));
    for idx in sorted_indices {
        if idx < clustering.clusters.len() {
            clustering.clusters.remove(idx);
        }
    }

    Ok(format!(
        "Merged {} clusters to colocate {} concepts",
        clusters_to_merge.len() + 1,
        concepts.len()
    ))
}

/// Split clusters to satisfy a separate constraint
fn split_for_separate(
    clustering: &mut ClusteringResult,
    concepts: &[String],
) -> Result<String, String> {
    // Find concepts that are in the same cluster
    let mut concept_to_cluster: HashMap<&String, usize> = HashMap::new();

    for concept in concepts {
        for (idx, cluster) in clustering.clusters.iter().enumerate() {
            if cluster.concepts.contains(concept) {
                concept_to_cluster.insert(concept, idx);
                break;
            }
        }
    }

    // Group by cluster
    let mut by_cluster: HashMap<usize, Vec<&String>> = HashMap::new();
    for (concept, cluster_idx) in &concept_to_cluster {
        by_cluster.entry(*cluster_idx).or_default().push(*concept);
    }

    // Find clusters with multiple concepts that need separating
    let mut splits = 0;
    let mut new_clusters: Vec<Cluster> = Vec::new();

    for (cluster_idx, concepts_in_cluster) in by_cluster {
        if concepts_in_cluster.len() > 1 {
            // Keep first concept, move rest to new clusters
            for concept in concepts_in_cluster.into_iter().skip(1) {
                // Remove from original cluster
                if let Some(cluster) = clustering.clusters.get_mut(cluster_idx)
                    && let Some(pos) = cluster.concepts.iter().position(|c| c == concept)
                {
                    cluster.concepts.remove(pos);
                    // Estimate access split (simple division)
                    let moved_accesses = cluster.total_accesses / 2;
                    cluster.total_accesses -= moved_accesses;

                    // Create new singleton cluster
                    let new_id = clustering.clusters.len() + new_clusters.len();
                    new_clusters.push(Cluster {
                        id: new_id,
                        concepts: vec![concept.clone()],
                        avg_distance: 0.0,
                        total_accesses: moved_accesses,
                    });
                    splits += 1;
                }
            }
        }
    }

    clustering.clusters.extend(new_clusters);

    if splits == 0 {
        Ok("Concepts already in separate clusters".to_string())
    } else {
        Ok(format!("Split {} concepts to separate clusters", splits))
    }
}

/// Check if two constraints conflict with each other
fn check_conflict(c1: &LayoutConstraint, c2: &LayoutConstraint) -> Option<ConstraintConflict> {
    match (c1, c2) {
        // Colocate vs Separate conflict
        (
            LayoutConstraint::Colocate {
                concepts: colocate, ..
            },
            LayoutConstraint::Separate {
                concepts: separate, ..
            },
        )
        | (
            LayoutConstraint::Separate {
                concepts: separate, ..
            },
            LayoutConstraint::Colocate {
                concepts: colocate, ..
            },
        ) => {
            // Conflict if two or more concepts appear in both
            let colocate_set: HashSet<&String> = colocate.iter().collect();
            let separate_set: HashSet<&String> = separate.iter().collect();
            let overlap: Vec<&&String> = colocate_set.intersection(&separate_set).collect();

            if overlap.len() >= 2 {
                return Some(ConstraintConflict {
                    constraint_a: c1.clone(),
                    constraint_b: c2.clone(),
                    reason: format!(
                        "Cannot both colocate and separate concepts: {:?}",
                        overlap.iter().map(|s| s.as_str()).collect::<Vec<_>>()
                    ),
                });
            }
        }

        // Hot vs Cold on same concept
        (
            LayoutConstraint::ForceRegion {
                concept: c1_concept,
                region: r1,
                ..
            },
            LayoutConstraint::ForceRegion {
                concept: c2_concept,
                region: r2,
                ..
            },
        ) => {
            if c1_concept == c2_concept && r1 != r2 {
                return Some(ConstraintConflict {
                    constraint_a: c1.clone(),
                    constraint_b: c2.clone(),
                    reason: format!(
                        "Concept '{}' cannot be both {:?} and {:?}",
                        c1_concept, r1, r2
                    ),
                });
            }
        }

        _ => {}
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::constraint::ConstraintSource;

    fn test_source() -> ConstraintSource {
        ConstraintSource::new("test.sio", 1, 1, "test")
    }

    fn simple_clustering() -> ClusteringResult {
        ClusteringResult {
            clusters: vec![
                Cluster {
                    id: 0,
                    concepts: vec!["A".to_string(), "B".to_string()],
                    avg_distance: 1.0,
                    total_accesses: 100,
                },
                Cluster {
                    id: 1,
                    concepts: vec!["C".to_string(), "D".to_string()],
                    avg_distance: 1.5,
                    total_accesses: 50,
                },
            ],
            dendrogram: Vec::new(),
        }
    }

    #[test]
    fn test_colocate_merges_clusters() {
        let clustering = simple_clustering();
        let mut constraints = ConstraintSet::new();

        // Colocate A and C (in different clusters)
        constraints.add(LayoutConstraint::Colocate {
            concepts: vec!["A".to_string(), "C".to_string()],
            source: test_source(),
        });

        let result = solve_constraints(clustering, &constraints, LayoutConfig::default());

        assert!(result.is_success());
        assert!(!result.satisfied.is_empty());

        // Check that A and C are now in the same cluster
        let a_cluster = result
            .modified_clustering
            .clusters
            .iter()
            .find(|c| c.concepts.contains(&"A".to_string()));
        assert!(a_cluster.is_some());
        assert!(a_cluster.unwrap().concepts.contains(&"C".to_string()));
    }

    #[test]
    fn test_separate_splits_clusters() {
        let clustering = simple_clustering();
        let mut constraints = ConstraintSet::new();

        // Separate A and B (in same cluster)
        constraints.add(LayoutConstraint::Separate {
            concepts: vec!["A".to_string(), "B".to_string()],
            source: test_source(),
        });

        let result = solve_constraints(clustering, &constraints, LayoutConfig::default());

        assert!(result.is_success());

        // A and B should now be in different clusters
        let a_cluster = result
            .modified_clustering
            .clusters
            .iter()
            .find(|c| c.concepts.contains(&"A".to_string()));
        let b_cluster = result
            .modified_clustering
            .clusters
            .iter()
            .find(|c| c.concepts.contains(&"B".to_string()));

        assert!(a_cluster.is_some());
        assert!(b_cluster.is_some());
        assert_ne!(a_cluster.unwrap().id, b_cluster.unwrap().id);
    }

    #[test]
    fn test_conflict_detection() {
        let clustering = simple_clustering();
        let mut constraints = ConstraintSet::new();

        // Conflicting constraints
        constraints.add(LayoutConstraint::Colocate {
            concepts: vec!["A".to_string(), "B".to_string()],
            source: test_source(),
        });
        constraints.add(LayoutConstraint::Separate {
            concepts: vec!["A".to_string(), "B".to_string()],
            source: test_source(),
        });

        let result = solve_constraints(clustering, &constraints, LayoutConfig::default());

        assert!(!result.is_success());
        assert!(!result.conflicts.is_empty());
    }

    #[test]
    fn test_forced_region() {
        let clustering = ClusteringResult {
            clusters: vec![Cluster {
                id: 0,
                concepts: vec!["hot_data".to_string()],
                avg_distance: 0.0,
                total_accesses: 1, // Low access count would normally make this cold
            }],
            dendrogram: Vec::new(),
        };

        let mut constraints = ConstraintSet::new();
        constraints.add(LayoutConstraint::ForceRegion {
            concept: "hot_data".to_string(),
            region: ForcedRegion::Hot,
            source: test_source(),
        });

        let result = solve_constraints(clustering, &constraints, LayoutConfig::default());

        assert!(result.is_success());

        // Check that the concept is in the hot region
        let concept_layout = result.layout.get("hot_data");
        assert!(concept_layout.is_some());
        assert_eq!(concept_layout.unwrap().region, MemoryRegion::Hot);
    }

    #[test]
    fn test_region_conflict() {
        let clustering = simple_clustering();
        let mut constraints = ConstraintSet::new();

        // Same concept forced to different regions
        constraints.add(LayoutConstraint::ForceRegion {
            concept: "A".to_string(),
            region: ForcedRegion::Hot,
            source: test_source(),
        });
        constraints.add(LayoutConstraint::ForceRegion {
            concept: "A".to_string(),
            region: ForcedRegion::Cold,
            source: test_source(),
        });

        let result = solve_constraints(clustering, &constraints, LayoutConfig::default());

        assert!(!result.is_success());
        assert!(!result.conflicts.is_empty());
    }
}
