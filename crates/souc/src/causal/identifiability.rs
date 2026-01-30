//! Causal Identifiability Analysis
//!
//! This module implements algorithms for determining whether causal effects
//! can be identified from observational data, including:
//!
//! - Backdoor path detection and blocking
//! - Frontdoor criterion checking
//! - d-separation testing (generalizing conditional independence)
//! - Minimal adjustment set finding
//!
//! # Backdoor Criterion
//!
//! A set Z satisfies the backdoor criterion for (X, Y) if:
//! 1. No node in Z is a descendant of X
//! 2. Z blocks all backdoor paths from X to Y
//!
//! # Frontdoor Criterion
//!
//! A set M satisfies the frontdoor criterion for (X, Y) if:
//! 1. M intercepts all directed paths from X to Y
//! 2. There is no backdoor path from X to M
//! 3. All backdoor paths from M to Y are blocked by X

use std::collections::{HashSet, VecDeque};

use super::dag::CausalDAG;

/// Result of backdoor path analysis
#[derive(Clone, Debug)]
pub struct BackdoorPathAnalysis {
    /// All backdoor paths found
    pub paths: Vec<BackdoorPath>,
    /// Whether any backdoor paths exist
    pub has_backdoor_paths: bool,
    /// Minimal sets that block all backdoor paths
    pub minimal_adjustment_sets: Vec<HashSet<String>>,
}

/// A backdoor path from treatment to outcome
#[derive(Clone, Debug, PartialEq)]
pub struct BackdoorPath {
    /// Sequence of nodes in the path
    pub nodes: Vec<String>,
    /// Whether this path is blocked by current conditioning set
    pub blocked: bool,
    /// Variables that could block this path
    pub blocking_vars: HashSet<String>,
}

impl BackdoorPath {
    /// Create a new backdoor path
    pub fn new(nodes: Vec<String>) -> Self {
        BackdoorPath {
            nodes,
            blocked: false,
            blocking_vars: HashSet::new(),
        }
    }

    /// Check if this path contains a collider
    pub fn has_collider(&self, dag: &CausalDAG) -> bool {
        if self.nodes.len() < 3 {
            return false;
        }

        for i in 1..self.nodes.len() - 1 {
            let prev = &self.nodes[i - 1];
            let curr = &self.nodes[i];
            let next = &self.nodes[i + 1];

            // Check if curr is a collider (both edges point into it)
            let parents = dag.parents(curr);
            if parents.contains(&prev.as_str()) && parents.contains(&next.as_str()) {
                return true;
            }
        }

        false
    }
}

/// Find all backdoor paths from treatment to outcome
///
/// A backdoor path is a path that:
/// - Starts with an edge into the treatment (←)
/// - Ends at the outcome
pub fn find_backdoor_paths(dag: &CausalDAG, treatment: &str, outcome: &str) -> Vec<BackdoorPath> {
    let mut paths = Vec::new();

    // Start from parents of treatment (backdoor = starts with ←)
    for parent in dag.parents(treatment) {
        let mut current_path = vec![treatment.to_string(), parent.to_string()];
        // Mark treatment as visited to prevent paths that go through treatment again
        let mut visited = HashSet::new();
        visited.insert(treatment.to_string());
        explore_paths(
            dag,
            parent,
            outcome,
            &mut current_path,
            &mut visited,
            &mut paths,
        );
    }

    paths
}

/// Recursively explore paths in the graph
fn explore_paths(
    dag: &CausalDAG,
    current: &str,
    target: &str,
    path: &mut Vec<String>,
    visited: &mut HashSet<String>,
    found_paths: &mut Vec<BackdoorPath>,
) {
    if current == target {
        // Found a complete backdoor path
        found_paths.push(BackdoorPath::new(path.clone()));
        return;
    }

    if visited.contains(current) {
        return;
    }

    visited.insert(current.to_string());

    // Explore children (→)
    for child in dag.children(current) {
        if !path.contains(&child.to_string()) {
            path.push(child.to_string());
            explore_paths(dag, child, target, path, visited, found_paths);
            path.pop();
        }
    }

    // Explore parents (←) for backdoor paths
    for parent in dag.parents(current) {
        if !path.contains(&parent.to_string()) {
            path.push(parent.to_string());
            explore_paths(dag, parent, target, path, visited, found_paths);
            path.pop();
        }
    }

    visited.remove(current);
}

/// Find minimal adjustment sets that satisfy backdoor criterion
///
/// Returns all minimal sets of variables that block all backdoor paths
pub fn find_valid_adjustment_sets(
    dag: &CausalDAG,
    treatment: &str,
    outcome: &str,
) -> Vec<HashSet<String>> {
    let backdoor_paths = find_backdoor_paths(dag, treatment, outcome);

    if backdoor_paths.is_empty() {
        // No backdoor paths = empty set is valid
        return vec![HashSet::new()];
    }

    // Collect all potential blocking variables
    let mut candidates = HashSet::new();
    for path in &backdoor_paths {
        for node in &path.nodes {
            if node != treatment && node != outcome {
                candidates.insert(node.clone());
            }
        }
    }

    // Remove descendants of treatment (backdoor criterion rule 1)
    let descendants = descendants_of(dag, treatment);
    candidates.retain(|v| !descendants.contains(v));

    // Find minimal hitting sets (sets that hit every path)
    find_minimal_hitting_sets(&backdoor_paths, &candidates)
}

/// Find minimal hitting sets - sets that contain at least one blocking variable from each path
fn find_minimal_hitting_sets(
    paths: &[BackdoorPath],
    candidates: &HashSet<String>,
) -> Vec<HashSet<String>> {
    let mut minimal_sets = Vec::new();

    // Try single variables
    for var in candidates {
        let mut set = HashSet::new();
        set.insert(var.clone());
        if blocks_all_paths(paths, &set) && is_minimal(&set, paths) {
            minimal_sets.push(set);
        }
    }

    // Try pairs (limited combinatorial search)
    let cand_vec: Vec<_> = candidates.iter().collect();
    for i in 0..cand_vec.len().min(10) {
        for j in (i + 1)..cand_vec.len().min(10) {
            let mut set = HashSet::new();
            set.insert(cand_vec[i].clone());
            set.insert(cand_vec[j].clone());
            if blocks_all_paths(paths, &set) && is_minimal(&set, paths) {
                minimal_sets.push(set);
            }
        }
    }

    minimal_sets
}

/// Check if a set blocks all backdoor paths
fn blocks_all_paths(paths: &[BackdoorPath], adjustment_set: &HashSet<String>) -> bool {
    paths
        .iter()
        .all(|path| path_is_blocked(path, adjustment_set))
}

/// Check if a path is blocked by the adjustment set
fn path_is_blocked(path: &BackdoorPath, adjustment_set: &HashSet<String>) -> bool {
    // A path is blocked if any non-collider node on the path is in the adjustment set
    // (simplified - full version needs to check for colliders)
    for node in &path.nodes {
        if adjustment_set.contains(node) {
            return true;
        }
    }
    false
}

/// Check if an adjustment set is minimal (no proper subset blocks all paths)
fn is_minimal(set: &HashSet<String>, paths: &[BackdoorPath]) -> bool {
    for var in set {
        let mut subset = set.clone();
        subset.remove(var);
        if blocks_all_paths(paths, &subset) {
            return false; // Found smaller set that works
        }
    }
    true
}

/// Check if a path is blocked by a collider
///
/// A collider is a node where both adjacent edges point INTO it.
/// A path is blocked by a collider unless the collider or one of its
/// descendants is in the conditioning set.
fn path_blocked_by_collider(
    dag: &CausalDAG,
    path: &BackdoorPath,
    conditioning: &HashSet<String>,
) -> bool {
    if path.nodes.len() < 3 {
        return false;
    }

    for i in 1..path.nodes.len() - 1 {
        let prev = &path.nodes[i - 1];
        let curr = &path.nodes[i];
        let next = &path.nodes[i + 1];

        // Check if curr is a collider (both edges point into it)
        // Edge direction: prev -> curr or prev <- curr?
        let parents_of_curr = dag.parents(curr);
        let is_prev_parent = parents_of_curr.iter().any(|p| p == prev);
        let is_next_parent = parents_of_curr.iter().any(|p| p == next);

        if is_prev_parent && is_next_parent {
            // curr is a collider - check if it's conditioned on or has conditioned descendant
            if conditioning.contains(curr) {
                continue; // Collider opened by conditioning
            }

            // Check if any descendant of collider is in conditioning set
            let descendants = descendants_of(dag, curr);
            let has_conditioned_descendant = descendants.iter().any(|d| conditioning.contains(d));

            if !has_conditioned_descendant {
                return true; // Path is blocked by this collider
            }
        }
    }

    false
}

/// Check if frontdoor criterion is satisfied
///
/// M satisfies frontdoor for (X, Y) if:
/// 1. M intercepts all directed paths X → Y
/// 2. No unblocked backdoor path X ← → M
/// 3. All backdoor paths M ← → Y blocked by X
pub fn check_frontdoor_criterion(
    dag: &CausalDAG,
    treatment: &str,
    outcome: &str,
    mediators: &HashSet<String>,
) -> bool {
    // Check 1: M intercepts all directed paths
    if !intercepts_all_directed_paths(dag, treatment, outcome, mediators) {
        return false;
    }

    // Check 2: No unblocked backdoor path from treatment to any mediator
    // Paths blocked by colliders are considered blocked
    for mediator in mediators {
        let backdoor_paths = find_backdoor_paths(dag, treatment, mediator);
        let empty_conditioning: HashSet<String> = HashSet::new();
        let unblocked_paths: Vec<_> = backdoor_paths
            .into_iter()
            .filter(|p| !path_blocked_by_collider(dag, p, &empty_conditioning))
            .collect();
        if !unblocked_paths.is_empty() {
            return false;
        }
    }

    // Check 3: All backdoor paths from mediators to outcome blocked by treatment
    let mut treatment_set = HashSet::new();
    treatment_set.insert(treatment.to_string());

    for mediator in mediators {
        let backdoor_paths = find_backdoor_paths(dag, mediator, outcome);
        // Filter out paths blocked by colliders, then check if treatment blocks the rest
        let unblocked_paths: Vec<_> = backdoor_paths
            .into_iter()
            .filter(|p| !path_blocked_by_collider(dag, p, &treatment_set))
            .collect();
        if !unblocked_paths.is_empty() && !blocks_all_paths(&unblocked_paths, &treatment_set) {
            return false;
        }
    }

    true
}

/// Check if mediators intercept all directed paths from source to target
fn intercepts_all_directed_paths(
    dag: &CausalDAG,
    source: &str,
    target: &str,
    mediators: &HashSet<String>,
) -> bool {
    // Find all directed paths and check if each goes through at least one mediator
    let directed_paths = find_directed_paths(dag, source, target);

    for path in directed_paths {
        let mut has_mediator = false;
        for node in &path {
            if mediators.contains(node) {
                has_mediator = true;
                break;
            }
        }
        if !has_mediator {
            return false; // Found path not through mediators
        }
    }

    true
}

/// Find all directed paths from source to target (only following →)
fn find_directed_paths(dag: &CausalDAG, source: &str, target: &str) -> Vec<Vec<String>> {
    let mut paths = Vec::new();
    let mut current_path = vec![source.to_string()];
    let mut visited = HashSet::new();

    explore_directed_paths(
        dag,
        source,
        target,
        &mut current_path,
        &mut visited,
        &mut paths,
    );

    paths
}

/// Explore directed paths recursively
fn explore_directed_paths(
    dag: &CausalDAG,
    current: &str,
    target: &str,
    path: &mut Vec<String>,
    visited: &mut HashSet<String>,
    found_paths: &mut Vec<Vec<String>>,
) {
    if current == target {
        found_paths.push(path.clone());
        return;
    }

    if visited.contains(current) {
        return;
    }

    visited.insert(current.to_string());

    // Only follow children (directed paths only)
    for child in dag.children(current) {
        if !path.contains(&child.to_string()) {
            path.push(child.to_string());
            explore_directed_paths(dag, child, target, path, visited, found_paths);
            path.pop();
        }
    }

    visited.remove(current);
}

/// Test d-separation between two sets of nodes
///
/// Returns true if X and Y are d-separated given Z in the graph
pub fn d_separation(
    dag: &CausalDAG,
    x: &HashSet<String>,
    y: &HashSet<String>,
    z: &HashSet<String>,
) -> bool {
    // Check if every pair (xi, yj) is d-separated given Z
    for xi in x {
        for yj in y {
            if !dag.d_separated(xi, yj, z) {
                return false;
            }
        }
    }
    true
}

/// Get all descendants of a node
fn descendants_of(dag: &CausalDAG, node: &str) -> HashSet<String> {
    let mut descendants = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(node.to_string());

    while let Some(current) = queue.pop_front() {
        for child in dag.children(&current) {
            if descendants.insert(child.to_string()) {
                queue.push_back(child.to_string());
            }
        }
    }

    descendants
}

/// Get all ancestors of a node
fn ancestors_of(dag: &CausalDAG, node: &str) -> HashSet<String> {
    let mut ancestors = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(node.to_string());

    while let Some(current) = queue.pop_front() {
        for parent in dag.parents(&current) {
            if ancestors.insert(parent.to_string()) {
                queue.push_back(parent.to_string());
            }
        }
    }

    ancestors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::dag::{EffectEstimate, EpistemicCausalNode, UncertainCausalEdge};
    use crate::epistemic::BetaConfidence;

    fn simple_confounded_dag() -> CausalDAG {
        // U -> X -> Y, U -> Y
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::confounder("U"));
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge1 = UncertainCausalEdge::direct(
            "U",
            "X",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge2 = UncertainCausalEdge::direct(
            "U",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge3 = UncertainCausalEdge::direct(
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.3),
        );

        dag.add_edge(edge1).unwrap();
        dag.add_edge(edge2).unwrap();
        dag.add_edge(edge3).unwrap();

        dag
    }

    #[test]
    fn test_find_backdoor_paths() {
        let dag = simple_confounded_dag();

        let paths = find_backdoor_paths(&dag, "X", "Y");

        // Should find: X ← U → Y
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_find_valid_adjustment_sets() {
        let dag = simple_confounded_dag();

        let sets = find_valid_adjustment_sets(&dag, "X", "Y");

        // Should find {U} as valid adjustment set
        assert!(!sets.is_empty());
        assert!(sets.iter().any(|s| s.contains("U")));
    }

    #[test]
    fn test_check_frontdoor_criterion() {
        // Create frontdoor graph: X -> M -> Y, U -> X, U -> Y
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::latent("U"));
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::mediator("M"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge1 = UncertainCausalEdge::direct(
            "U",
            "X",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge2 = UncertainCausalEdge::direct(
            "U",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge3 = UncertainCausalEdge::direct(
            "X",
            "M",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.4),
        );
        let edge4 = UncertainCausalEdge::direct(
            "M",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.6),
        );

        dag.add_edge(edge1).unwrap();
        dag.add_edge(edge2).unwrap();
        dag.add_edge(edge3).unwrap();
        dag.add_edge(edge4).unwrap();

        let mut mediators = HashSet::new();
        mediators.insert("M".to_string());

        // M should satisfy frontdoor criterion
        assert!(check_frontdoor_criterion(&dag, "X", "Y", &mediators));
    }

    #[test]
    fn test_descendants_of() {
        let dag = simple_confounded_dag();

        let desc = descendants_of(&dag, "U");
        assert!(desc.contains("X"));
        assert!(desc.contains("Y"));
    }

    #[test]
    fn test_ancestors_of() {
        let dag = simple_confounded_dag();

        let anc = ancestors_of(&dag, "Y");
        assert!(anc.contains("X"));
        assert!(anc.contains("U"));
    }

    #[test]
    fn test_d_separation() {
        let dag = simple_confounded_dag();

        let mut x = HashSet::new();
        x.insert("X".to_string());

        let mut y = HashSet::new();
        y.insert("Y".to_string());

        let mut z = HashSet::new();
        z.insert("U".to_string());

        // X and Y are NOT d-separated given U because the direct path X → Y
        // remains open. Conditioning on U only blocks the path X ← U → Y.
        assert!(!d_separation(&dag, &x, &y, &z));

        // Test that U and Y are NOT d-separated given empty set
        // (there's a direct path U → Y)
        let mut u_set = HashSet::new();
        u_set.insert("U".to_string());
        let empty: HashSet<String> = HashSet::new();
        assert!(!d_separation(&dag, &u_set, &y, &empty));
    }
}
