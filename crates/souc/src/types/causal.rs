//! Causal type system for Sounio
//!
//! This module implements causal inference types and validation:
//! - `Causal<G>` type for referencing declared causal graphs
//! - DAG validation (cycle detection)
//! - d-separation for conditional independence
//! - Causal assumption tracking for epistemic confidence

use std::collections::{HashMap, HashSet, VecDeque};

use crate::ast::CausalModelDef;
use crate::common::Span;

/// A validated causal graph definition ready for type checking
#[derive(Debug, Clone)]
pub struct CausalGraphDef {
    /// Name of the causal model/graph
    pub name: String,
    /// Node names with optional types
    pub nodes: Vec<CausalNodeDef>,
    /// Directed edges (from -> to)
    pub edges: Vec<CausalEdgeDef>,
    /// Topological order (if DAG)
    pub topological_order: Option<Vec<String>>,
    /// Source span for error reporting
    pub span: Span,
}

/// A node in the causal graph
#[derive(Debug, Clone)]
pub struct CausalNodeDef {
    pub name: String,
    pub span: Span,
}

/// A directed edge in the causal graph
#[derive(Debug, Clone)]
pub struct CausalEdgeDef {
    pub from: String,
    pub to: String,
    pub span: Span,
}

/// Errors from causal graph validation
#[derive(Debug, Clone)]
pub enum CausalError {
    /// Graph contains a cycle (not a DAG)
    CyclicGraph {
        /// The cycle path
        cycle: Vec<String>,
        span: Span,
    },
    /// Reference to unknown node
    UnknownNode {
        node: String,
        graph: String,
        span: Span,
    },
    /// Invalid intervention target
    InvalidIntervention {
        target: String,
        reason: String,
        span: Span,
    },
    /// Unblocked confounding path
    UnblockedConfounder {
        from: String,
        to: String,
        confounders: Vec<String>,
        span: Span,
    },
    /// Reference to undeclared causal graph
    UndeclaredGraph { name: String, span: Span },
}

impl std::fmt::Display for CausalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CausalError::CyclicGraph { cycle, .. } => {
                write!(f, "causal graph contains cycle: {}", cycle.join(" -> "))
            }
            CausalError::UnknownNode { node, graph, .. } => {
                write!(f, "unknown node '{}' in causal graph '{}'", node, graph)
            }
            CausalError::InvalidIntervention { target, reason, .. } => {
                write!(f, "invalid intervention on '{}': {}", target, reason)
            }
            CausalError::UnblockedConfounder {
                from,
                to,
                confounders,
                ..
            } => {
                write!(
                    f,
                    "unblocked confounding path from '{}' to '{}' via: {}",
                    from,
                    to,
                    confounders.join(", ")
                )
            }
            CausalError::UndeclaredGraph { name, .. } => {
                write!(f, "undeclared causal graph '{}'", name)
            }
        }
    }
}

impl std::error::Error for CausalError {}

/// Result type for causal operations
pub type CausalResult<T> = Result<T, CausalError>;

/// Registry of declared causal graphs
#[derive(Debug, Default)]
pub struct CausalGraphRegistry {
    graphs: HashMap<String, CausalGraphDef>,
}

impl CausalGraphRegistry {
    pub fn new() -> Self {
        Self {
            graphs: HashMap::new(),
        }
    }

    /// Register a causal model from AST, validating it's a DAG
    pub fn register(&mut self, model: &CausalModelDef) -> CausalResult<()> {
        let nodes: Vec<CausalNodeDef> = model
            .nodes
            .iter()
            .map(|n| CausalNodeDef {
                name: n.name.clone(),
                span: n.span,
            })
            .collect();

        let edges: Vec<CausalEdgeDef> = model
            .edges
            .iter()
            .map(|e| CausalEdgeDef {
                from: e.from.clone(),
                to: e.to.clone(),
                span: e.span,
            })
            .collect();

        // Validate nodes are defined
        let node_names: HashSet<_> = nodes.iter().map(|n| n.name.as_str()).collect();
        for edge in &edges {
            if !node_names.contains(edge.from.as_str()) {
                return Err(CausalError::UnknownNode {
                    node: edge.from.clone(),
                    graph: model.name.clone(),
                    span: edge.span,
                });
            }
            if !node_names.contains(edge.to.as_str()) {
                return Err(CausalError::UnknownNode {
                    node: edge.to.clone(),
                    graph: model.name.clone(),
                    span: edge.span,
                });
            }
        }

        // Validate DAG (detect cycles)
        let topo_order = validate_dag(&nodes, &edges, model.span)?;

        let graph_def = CausalGraphDef {
            name: model.name.clone(),
            nodes,
            edges,
            topological_order: Some(topo_order),
            span: model.span,
        };

        self.graphs.insert(model.name.clone(), graph_def);
        Ok(())
    }

    /// Look up a causal graph by name
    pub fn get(&self, name: &str) -> Option<&CausalGraphDef> {
        self.graphs.get(name)
    }

    /// Check if a graph is registered
    pub fn contains(&self, name: &str) -> bool {
        self.graphs.contains_key(name)
    }

    /// Return the number of registered graphs
    pub fn len(&self) -> usize {
        self.graphs.len()
    }

    /// Check if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.graphs.is_empty()
    }

    /// Return the names of all registered graphs
    pub fn names(&self) -> Vec<&str> {
        self.graphs.keys().map(|s| s.as_str()).collect()
    }
}

/// Validate that edges form a DAG and return topological order
pub fn validate_dag(
    nodes: &[CausalNodeDef],
    edges: &[CausalEdgeDef],
    span: Span,
) -> CausalResult<Vec<String>> {
    // Build adjacency list
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut in_degree: HashMap<&str, usize> = HashMap::new();

    for node in nodes {
        adj.entry(node.name.as_str()).or_default();
        in_degree.entry(node.name.as_str()).or_insert(0);
    }

    for edge in edges {
        adj.entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
        *in_degree.entry(edge.to.as_str()).or_insert(0) += 1;
    }

    // Kahn's algorithm for topological sort
    let mut queue: VecDeque<&str> = VecDeque::new();
    for (node, &degree) in &in_degree {
        if degree == 0 {
            queue.push_back(node);
        }
    }

    let mut topo_order = Vec::new();
    while let Some(node) = queue.pop_front() {
        topo_order.push(node.to_string());

        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if let Some(degree) = in_degree.get_mut(neighbor) {
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }
    }

    // If not all nodes are in topo_order, there's a cycle
    if topo_order.len() != nodes.len() {
        // Find cycle for error message
        let cycle = find_cycle(nodes, edges);
        return Err(CausalError::CyclicGraph { cycle, span });
    }

    Ok(topo_order)
}

/// Find a cycle in the graph (for error reporting)
fn find_cycle(nodes: &[CausalNodeDef], edges: &[CausalEdgeDef]) -> Vec<String> {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for edge in edges {
        adj.entry(edge.from.as_str())
            .or_default()
            .push(edge.to.as_str());
    }

    let mut visited: HashSet<&str> = HashSet::new();
    let mut rec_stack: HashSet<&str> = HashSet::new();
    let mut path: Vec<&str> = Vec::new();

    fn dfs<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashSet<&'a str>,
        rec_stack: &mut HashSet<&'a str>,
        path: &mut Vec<&'a str>,
    ) -> Option<Vec<String>> {
        visited.insert(node);
        rec_stack.insert(node);
        path.push(node);

        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if let Some(cycle) = dfs(neighbor, adj, visited, rec_stack, path) {
                        return Some(cycle);
                    }
                } else if rec_stack.contains(neighbor) {
                    // Found cycle
                    let cycle_start = path.iter().position(|&n| n == neighbor).unwrap();
                    let mut cycle: Vec<String> =
                        path[cycle_start..].iter().map(|s| s.to_string()).collect();
                    cycle.push(neighbor.to_string()); // Complete the cycle
                    return Some(cycle);
                }
            }
        }

        path.pop();
        rec_stack.remove(node);
        None
    }

    for node in nodes {
        if !visited.contains(node.name.as_str()) {
            if let Some(cycle) = dfs(
                node.name.as_str(),
                &adj,
                &mut visited,
                &mut rec_stack,
                &mut path,
            ) {
                return cycle;
            }
        }
    }

    vec!["unknown cycle".to_string()]
}

/// Check d-separation between two nodes given a conditioning set
///
/// Two nodes X and Y are d-separated by Z if all paths between X and Y are blocked by Z.
/// A path is blocked if:
/// 1. It contains a chain (A -> B -> C) or fork (A <- B -> C) where B is in Z
/// 2. It contains a collider (A -> B <- C) where B and its descendants are not in Z
pub fn check_d_separation(
    graph: &CausalGraphDef,
    x: &str,
    y: &str,
    conditioning: &HashSet<String>,
) -> bool {
    // Build adjacency structures
    let mut parents: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut children: HashMap<&str, HashSet<&str>> = HashMap::new();

    for edge in &graph.edges {
        parents
            .entry(edge.to.as_str())
            .or_default()
            .insert(edge.from.as_str());
        children
            .entry(edge.from.as_str())
            .or_default()
            .insert(edge.to.as_str());
    }

    // Get all descendants of conditioning set
    let descendants = get_all_descendants(&children, conditioning);

    // BFS to find active paths (d-connected paths)
    // State: (current_node, came_from_child)
    // came_from_child is true if we arrived via a child->parent edge
    let mut visited: HashSet<(&str, bool)> = HashSet::new();
    let mut queue: VecDeque<(&str, bool)> = VecDeque::new();

    // Start from x, going both directions
    queue.push_back((x, true));
    queue.push_back((x, false));

    while let Some((current, from_child)) = queue.pop_front() {
        if visited.contains(&(current, from_child)) {
            continue;
        }
        visited.insert((current, from_child));

        // Check if we reached y
        if current == y {
            return false; // Path found, not d-separated
        }

        let in_z = conditioning.contains(current);
        let descendant_in_z = descendants.contains(current);

        // Explore parents (going against edge direction)
        if let Some(node_parents) = parents.get(current) {
            for &parent in node_parents {
                // Can go to parent if:
                // - We came from a child AND current is not in Z (chain/fork case)
                // - We came from a parent AND current is a collider in Z or has descendant in Z
                let can_traverse = if from_child {
                    !in_z // Chain or fork: blocked if middle node in Z
                } else {
                    in_z || descendant_in_z // Collider: open if collider or descendant in Z
                };

                if can_traverse {
                    queue.push_back((parent, false));
                }
            }
        }

        // Explore children (going with edge direction)
        if let Some(node_children) = children.get(current) {
            for &child in node_children {
                // Can go to child if:
                // - We came from a parent AND current is not in Z (chain case)
                // - We came from a child AND current is not in Z (fork case)
                let can_traverse = !in_z;

                if can_traverse {
                    queue.push_back((child, true));
                }
            }
        }
    }

    true // No active path found, d-separated
}

/// Get all descendants of a set of nodes
fn get_all_descendants(
    children: &HashMap<&str, HashSet<&str>>,
    nodes: &HashSet<String>,
) -> HashSet<String> {
    let mut descendants = HashSet::new();
    let mut queue: VecDeque<&str> = nodes.iter().map(|s| s.as_str()).collect();

    while let Some(node) = queue.pop_front() {
        if let Some(node_children) = children.get(node) {
            for &child in node_children {
                if !descendants.contains(child) {
                    descendants.insert(child.to_string());
                    queue.push_back(child);
                }
            }
        }
    }

    descendants
}

/// Causal assumptions that affect epistemic confidence
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CausalAssumption {
    /// Faithfulness: statistical independencies imply causal structure
    Faithfulness,
    /// No unmeasured confounders between treatment and outcome
    NoUnmeasuredConfounders,
    /// Positivity: all treatment levels have positive probability
    Positivity,
    /// SUTVA: stable unit treatment value assumption
    SUTVA,
    /// Causal Markov: nodes are independent of non-descendants given parents
    CausalMarkov,
}

impl CausalAssumption {
    /// Get the confidence factor for this assumption
    /// Lower values mean more confidence loss when assuming
    pub fn confidence_factor(&self) -> f64 {
        match self {
            CausalAssumption::Faithfulness => 0.95,
            CausalAssumption::NoUnmeasuredConfounders => 0.85, // High risk
            CausalAssumption::Positivity => 0.98,
            CausalAssumption::SUTVA => 0.95,
            CausalAssumption::CausalMarkov => 0.99, // Usually satisfied
        }
    }
}

/// Calculate combined confidence factor for a set of assumptions
pub fn causal_confidence_factor(assumptions: &[CausalAssumption]) -> f64 {
    assumptions.iter().map(|a| a.confidence_factor()).product()
}

/// Find the backdoor adjustment set for causal effect of X on Y
///
/// Returns a set of variables to condition on to block all backdoor paths
/// while not blocking any causal paths from X to Y.
pub fn find_backdoor_adjustment(
    graph: &CausalGraphDef,
    treatment: &str,
    outcome: &str,
) -> Option<HashSet<String>> {
    // Simple heuristic: condition on all parents of treatment that are not
    // descendants of treatment
    let mut parents: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut children: HashMap<&str, HashSet<&str>> = HashMap::new();

    for edge in &graph.edges {
        parents
            .entry(edge.to.as_str())
            .or_default()
            .insert(edge.from.as_str());
        children
            .entry(edge.from.as_str())
            .or_default()
            .insert(edge.to.as_str());
    }

    // Get descendants of treatment
    let treatment_descendants = get_all_descendants(&children, &[treatment.to_string()].into());

    // Get parents of treatment that are not descendants
    let mut adjustment_set = HashSet::new();
    if let Some(treatment_parents) = parents.get(treatment) {
        for &parent in treatment_parents {
            if !treatment_descendants.contains(parent) {
                adjustment_set.insert(parent.to_string());
            }
        }
    }

    // Verify d-separation with this adjustment set
    // (In a full implementation, we'd verify all backdoor paths are blocked)
    Some(adjustment_set)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span() -> Span {
        Span::new(0, 0)
    }

    #[test]
    fn test_dag_validation_simple() {
        let nodes = vec![
            CausalNodeDef {
                name: "A".to_string(),
                span: make_span(),
            },
            CausalNodeDef {
                name: "B".to_string(),
                span: make_span(),
            },
            CausalNodeDef {
                name: "C".to_string(),
                span: make_span(),
            },
        ];
        let edges = vec![
            CausalEdgeDef {
                from: "A".to_string(),
                to: "B".to_string(),
                span: make_span(),
            },
            CausalEdgeDef {
                from: "B".to_string(),
                to: "C".to_string(),
                span: make_span(),
            },
        ];

        let result = validate_dag(&nodes, &edges, make_span());
        assert!(result.is_ok());
        let order = result.unwrap();
        assert_eq!(order.len(), 3);
        // A must come before B, B must come before C
        assert!(order.iter().position(|x| x == "A") < order.iter().position(|x| x == "B"));
        assert!(order.iter().position(|x| x == "B") < order.iter().position(|x| x == "C"));
    }

    #[test]
    fn test_dag_validation_cycle() {
        let nodes = vec![
            CausalNodeDef {
                name: "A".to_string(),
                span: make_span(),
            },
            CausalNodeDef {
                name: "B".to_string(),
                span: make_span(),
            },
            CausalNodeDef {
                name: "C".to_string(),
                span: make_span(),
            },
        ];
        let edges = vec![
            CausalEdgeDef {
                from: "A".to_string(),
                to: "B".to_string(),
                span: make_span(),
            },
            CausalEdgeDef {
                from: "B".to_string(),
                to: "C".to_string(),
                span: make_span(),
            },
            CausalEdgeDef {
                from: "C".to_string(),
                to: "A".to_string(),
                span: make_span(),
            },
        ];

        let result = validate_dag(&nodes, &edges, make_span());
        assert!(result.is_err());
        match result {
            Err(CausalError::CyclicGraph { cycle, .. }) => {
                assert!(cycle.len() >= 3);
            }
            _ => panic!("Expected CyclicGraph error"),
        }
    }

    #[test]
    fn test_d_separation_chain() {
        // A -> B -> C
        // A and C are d-separated by B
        let graph = CausalGraphDef {
            name: "test".to_string(),
            nodes: vec![
                CausalNodeDef {
                    name: "A".to_string(),
                    span: make_span(),
                },
                CausalNodeDef {
                    name: "B".to_string(),
                    span: make_span(),
                },
                CausalNodeDef {
                    name: "C".to_string(),
                    span: make_span(),
                },
            ],
            edges: vec![
                CausalEdgeDef {
                    from: "A".to_string(),
                    to: "B".to_string(),
                    span: make_span(),
                },
                CausalEdgeDef {
                    from: "B".to_string(),
                    to: "C".to_string(),
                    span: make_span(),
                },
            ],
            topological_order: Some(vec!["A".to_string(), "B".to_string(), "C".to_string()]),
            span: make_span(),
        };

        // Without conditioning, A and C are d-connected
        assert!(!check_d_separation(&graph, "A", "C", &HashSet::new()));

        // Conditioning on B blocks the path
        let conditioning: HashSet<String> = ["B".to_string()].into();
        assert!(check_d_separation(&graph, "A", "C", &conditioning));
    }

    #[test]
    fn test_d_separation_fork() {
        // A <- B -> C
        // A and C are d-separated by B
        let graph = CausalGraphDef {
            name: "test".to_string(),
            nodes: vec![
                CausalNodeDef {
                    name: "A".to_string(),
                    span: make_span(),
                },
                CausalNodeDef {
                    name: "B".to_string(),
                    span: make_span(),
                },
                CausalNodeDef {
                    name: "C".to_string(),
                    span: make_span(),
                },
            ],
            edges: vec![
                CausalEdgeDef {
                    from: "B".to_string(),
                    to: "A".to_string(),
                    span: make_span(),
                },
                CausalEdgeDef {
                    from: "B".to_string(),
                    to: "C".to_string(),
                    span: make_span(),
                },
            ],
            topological_order: Some(vec!["B".to_string(), "A".to_string(), "C".to_string()]),
            span: make_span(),
        };

        // Without conditioning, A and C are d-connected (through B)
        assert!(!check_d_separation(&graph, "A", "C", &HashSet::new()));

        // Conditioning on B blocks the path
        let conditioning: HashSet<String> = ["B".to_string()].into();
        assert!(check_d_separation(&graph, "A", "C", &conditioning));
    }

    #[test]
    fn test_causal_confidence_factor() {
        let assumptions = vec![
            CausalAssumption::Faithfulness,
            CausalAssumption::NoUnmeasuredConfounders,
        ];

        let factor = causal_confidence_factor(&assumptions);
        // 0.95 * 0.85 = 0.8075
        assert!((factor - 0.8075).abs() < 0.0001);
    }
}
