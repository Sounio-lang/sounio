//! Causal Graph: DAG with d-separation algorithm
//!
//! Implements causal directed acyclic graphs with:
//! - Node types (observed, latent, treatment, outcome, mediator)
//! - Edge types (direct causal, bidirected/latent confounder)
//! - d-separation algorithm (Bayes-Ball)
//! - Graph manipulation for interventions (G_X̄, G_X_)

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Node in causal graph
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CausalNode {
    /// Node identifier
    pub name: String,
    /// Type of node in causal structure
    pub node_type: NodeType,
    /// Value domain
    pub domain: ValueDomain,
}

impl CausalNode {
    /// Create a new causal node
    pub fn new(name: impl Into<String>, node_type: NodeType, domain: ValueDomain) -> Self {
        CausalNode {
            name: name.into(),
            node_type,
            domain,
        }
    }

    /// Create an observed variable node
    pub fn observed(name: impl Into<String>) -> Self {
        CausalNode {
            name: name.into(),
            node_type: NodeType::Observed,
            domain: ValueDomain::Continuous {
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            },
        }
    }

    /// Create a treatment node
    pub fn treatment(name: impl Into<String>) -> Self {
        CausalNode {
            name: name.into(),
            node_type: NodeType::Treatment,
            domain: ValueDomain::Continuous {
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            },
        }
    }

    /// Create an outcome node
    pub fn outcome(name: impl Into<String>) -> Self {
        CausalNode {
            name: name.into(),
            node_type: NodeType::Outcome,
            domain: ValueDomain::Continuous {
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            },
        }
    }

    /// Create a latent/unobserved node
    pub fn latent(name: impl Into<String>) -> Self {
        CausalNode {
            name: name.into(),
            node_type: NodeType::Latent,
            domain: ValueDomain::Continuous {
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            },
        }
    }

    /// Create a mediator node
    pub fn mediator(name: impl Into<String>) -> Self {
        CausalNode {
            name: name.into(),
            node_type: NodeType::Mediator,
            domain: ValueDomain::Continuous {
                min: f64::NEG_INFINITY,
                max: f64::INFINITY,
            },
        }
    }

    /// Create a binary variable node
    pub fn binary(name: impl Into<String>, node_type: NodeType) -> Self {
        CausalNode {
            name: name.into(),
            node_type,
            domain: ValueDomain::Binary,
        }
    }
}

/// Type of node in causal structure
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NodeType {
    /// Observed variable
    Observed,
    /// Latent/unobserved variable
    Latent,
    /// Treatment/intervention variable
    Treatment,
    /// Outcome variable
    Outcome,
    /// Mediator between treatment and outcome
    Mediator,
}

/// Value domain for a variable
#[derive(Clone, Debug, PartialEq)]
pub enum ValueDomain {
    /// Binary {0, 1}
    Binary,
    /// Categorical values
    Categorical(Vec<String>),
    /// Continuous range
    Continuous { min: f64, max: f64 },
    /// Non-negative integer count
    Count,
}

impl Eq for ValueDomain {}

impl std::hash::Hash for ValueDomain {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            ValueDomain::Binary => 0u8.hash(state),
            ValueDomain::Categorical(v) => {
                1u8.hash(state);
                v.hash(state);
            }
            ValueDomain::Continuous { min, max } => {
                2u8.hash(state);
                min.to_bits().hash(state);
                max.to_bits().hash(state);
            }
            ValueDomain::Count => 3u8.hash(state),
        }
    }
}

/// Edge types in causal graph
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EdgeType {
    /// Direct causal edge: X → Y
    Direct,
    /// Bidirected edge (latent confounder): X ↔ Y
    Bidirected,
}

/// Causal Directed Acyclic Graph
#[derive(Clone, Debug)]
pub struct CausalGraph {
    /// Nodes in the graph
    nodes: HashMap<String, CausalNode>,
    /// Edges: (from, to, type)
    edges: HashSet<(String, String, EdgeType)>,
    /// Parent mapping for direct edges
    parents: HashMap<String, HashSet<String>>,
    /// Children mapping for direct edges
    children: HashMap<String, HashSet<String>>,
    /// Bidirected neighbors
    bidirected: HashMap<String, HashSet<String>>,
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalGraph {
    /// Create a new empty causal graph
    pub fn new() -> Self {
        CausalGraph {
            nodes: HashMap::new(),
            edges: HashSet::new(),
            parents: HashMap::new(),
            children: HashMap::new(),
            bidirected: HashMap::new(),
        }
    }

    /// Add a node to the graph
    pub fn add_node(&mut self, node: CausalNode) {
        let name = node.name.clone();
        self.nodes.insert(name.clone(), node);
        self.parents.entry(name.clone()).or_default();
        self.children.entry(name.clone()).or_default();
        self.bidirected.entry(name).or_default();
    }

    /// Check if graph contains a node
    pub fn contains_node(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Get a node by name
    pub fn get_node(&self, name: &str) -> Option<&CausalNode> {
        self.nodes.get(name)
    }

    /// Get all node names
    pub fn node_names(&self) -> impl Iterator<Item = &String> {
        self.nodes.keys()
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Add an edge to the graph
    pub fn add_edge(
        &mut self,
        from: &str,
        to: &str,
        edge_type: EdgeType,
    ) -> Result<(), GraphError> {
        if !self.nodes.contains_key(from) {
            return Err(GraphError::NodeNotFound(from.to_string()));
        }
        if !self.nodes.contains_key(to) {
            return Err(GraphError::NodeNotFound(to.to_string()));
        }

        if edge_type == EdgeType::Direct && self.would_create_cycle(from, to) {
            return Err(GraphError::CycleDetected {
                from: from.to_string(),
                to: to.to_string(),
            });
        }

        self.edges
            .insert((from.to_string(), to.to_string(), edge_type.clone()));

        match edge_type {
            EdgeType::Direct => {
                self.parents.get_mut(to).unwrap().insert(from.to_string());
                self.children.get_mut(from).unwrap().insert(to.to_string());
            }
            EdgeType::Bidirected => {
                self.bidirected
                    .get_mut(from)
                    .unwrap()
                    .insert(to.to_string());
                self.bidirected
                    .get_mut(to)
                    .unwrap()
                    .insert(from.to_string());
            }
        }

        Ok(())
    }

    /// Get parents of a node (direct edges only)
    pub fn parents(&self, node: &str) -> Option<&HashSet<String>> {
        self.parents.get(node)
    }

    /// Get children of a node (direct edges only)
    pub fn children(&self, node: &str) -> Option<&HashSet<String>> {
        self.children.get(node)
    }

    /// G_X̄: Graph with incoming edges to X removed (intervention graph)
    pub fn graph_do(&self, x: &str) -> CausalGraph {
        let mut g_do = self.clone();

        // Remove direct edges pointing to X
        g_do.edges
            .retain(|(_, to, et)| !(to == x && *et == EdgeType::Direct));

        // Clear parents of X
        if let Some(parents) = g_do.parents.get_mut(x) {
            // Also remove X from children sets of its former parents
            for parent in parents.iter() {
                if let Some(children) = g_do.children.get_mut(parent) {
                    children.remove(x);
                }
            }
            parents.clear();
        }

        g_do
    }

    /// G_X_: Graph with outgoing edges from X removed
    pub fn graph_no_out(&self, x: &str) -> CausalGraph {
        let mut g_no = self.clone();

        // Remove direct edges from X
        g_no.edges
            .retain(|(from, _, et)| !(from == x && *et == EdgeType::Direct));

        // Clear children of X
        if let Some(children) = g_no.children.get_mut(x) {
            // Also remove X from parents sets of its former children
            for child in children.iter() {
                if let Some(parents) = g_no.parents.get_mut(child) {
                    parents.remove(x);
                }
            }
            children.clear();
        }

        g_no
    }

    /// Check d-separation: (X ⊥⊥ Y | Z)_G
    ///
    /// Two nodes are d-separated given conditioning set Z if there is no
    /// active path between them.
    pub fn d_separated(&self, x: &str, y: &str, z: &HashSet<String>) -> bool {
        !self.d_connected(x, y, z)
    }

    /// Check d-connection (negation of d-separation)
    ///
    /// Uses Bayes-Ball algorithm to find active paths
    pub fn d_connected(&self, x: &str, y: &str, z: &HashSet<String>) -> bool {
        if x == y {
            return true;
        }

        // Bayes-Ball algorithm
        // Track visited states: (node, direction) where direction = true means going up
        let mut visited_up: HashSet<String> = HashSet::new();
        let mut visited_down: HashSet<String> = HashSet::new();

        // Queue: (node, going_up)
        let mut queue: VecDeque<(String, bool)> = VecDeque::new();
        queue.push_back((x.to_string(), true)); // Start going up from X
        queue.push_back((x.to_string(), false)); // Also try going down

        while let Some((node, going_up)) = queue.pop_front() {
            if node == y {
                return true;
            }

            let in_z = z.contains(&node);
            let has_desc_in_z = self.has_descendant_in(z, &node);

            if going_up {
                // Coming from a child
                if !visited_up.contains(&node) {
                    visited_up.insert(node.clone());

                    // If not conditioned, can go to parents
                    if !in_z {
                        for parent in self.parents.get(&node).unwrap_or(&HashSet::new()) {
                            queue.push_back((parent.clone(), true));
                        }
                        // Can also go through bidirected edges
                        for bidir in self.bidirected.get(&node).unwrap_or(&HashSet::new()) {
                            queue.push_back((bidir.clone(), true));
                        }
                    }

                    // Can always pass through to children (chain or fork)
                    if !in_z {
                        for child in self.children.get(&node).unwrap_or(&HashSet::new()) {
                            queue.push_back((child.clone(), false));
                        }
                    }
                }
            } else {
                // Coming from a parent (going down)
                if !visited_down.contains(&node) {
                    visited_down.insert(node.clone());

                    // If not conditioned, continue to children
                    if !in_z {
                        for child in self.children.get(&node).unwrap_or(&HashSet::new()) {
                            queue.push_back((child.clone(), false));
                        }
                    }

                    // Collider: only pass if conditioned or has descendant conditioned
                    if in_z || has_desc_in_z {
                        for parent in self.parents.get(&node).unwrap_or(&HashSet::new()) {
                            queue.push_back((parent.clone(), true));
                        }
                        for bidir in self.bidirected.get(&node).unwrap_or(&HashSet::new()) {
                            queue.push_back((bidir.clone(), true));
                        }
                    }
                }
            }
        }

        false
    }

    /// Get all descendants of a node
    pub fn descendants_of(&self, node: &str) -> HashSet<String> {
        let mut desc = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node.to_string());

        while let Some(n) = queue.pop_front() {
            for child in self.children.get(&n).unwrap_or(&HashSet::new()) {
                if desc.insert(child.clone()) {
                    queue.push_back(child.clone());
                }
            }
        }

        desc
    }

    /// Get all ancestors of a node
    pub fn ancestors_of(&self, node: &str) -> HashSet<String> {
        let mut anc = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node.to_string());

        while let Some(n) = queue.pop_front() {
            for parent in self.parents.get(&n).unwrap_or(&HashSet::new()) {
                if anc.insert(parent.clone()) {
                    queue.push_back(parent.clone());
                }
            }
        }

        anc
    }

    /// Check if node has any descendant in the given set
    fn has_descendant_in(&self, set: &HashSet<String>, node: &str) -> bool {
        self.descendants_of(node).iter().any(|d| set.contains(d))
    }

    /// Check if adding edge would create a cycle
    fn would_create_cycle(&self, from: &str, to: &str) -> bool {
        // Adding from->to creates cycle if 'to' can already reach 'from'
        self.ancestors_of(from).contains(to) || from == to
    }

    /// Compute topological order of nodes
    pub fn topological_order(&self) -> Vec<String> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();

        fn visit(
            node: &str,
            graph: &CausalGraph,
            visited: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) {
            if visited.contains(node) {
                return;
            }
            visited.insert(node.to_string());

            for child in graph.children.get(node).unwrap_or(&HashSet::new()) {
                visit(child, graph, visited, order);
            }

            order.push(node.to_string());
        }

        for node in self.nodes.keys() {
            visit(node, self, &mut visited, &mut order);
        }

        order.reverse();
        order
    }

    /// Find all nodes on directed paths from X to Y
    pub fn find_mediators(&self, x: &str, y: &str) -> Vec<String> {
        let desc_x = self.descendants_of(x);
        let anc_y = self.ancestors_of(y);

        self.nodes
            .keys()
            .filter(|n| *n != x && *n != y && desc_x.contains(*n) && anc_y.contains(*n))
            .cloned()
            .collect()
    }

    /// Check if two graphs are compatible (no conflicting edges)
    pub fn compatible_with(&self, other: &CausalGraph) -> bool {
        for (from, to, et) in &self.edges {
            if let Some((_, _, other_et)) =
                other.edges.iter().find(|(f, t, _)| f == from && t == to)
                && et != other_et
            {
                return false;
            }
        }
        true
    }

    /// Check structural equivalence
    pub fn structurally_equivalent(&self, other: &CausalGraph) -> bool {
        self.edges == other.edges
            && self.nodes.keys().collect::<HashSet<_>>()
                == other.nodes.keys().collect::<HashSet<_>>()
    }

    /// Merge two compatible graphs
    pub fn merge(&self, other: &CausalGraph) -> CausalGraph {
        let mut merged = self.clone();

        for (name, node) in &other.nodes {
            if !merged.nodes.contains_key(name) {
                merged.add_node(node.clone());
            }
        }

        for (from, to, et) in &other.edges {
            let _ = merged.add_edge(from, to, et.clone());
        }

        merged
    }

    /// Check if removing nodes in M disconnects X from Y via directed paths
    pub fn intercepts_all_paths(&self, x: &str, y: &str, m: &HashSet<String>) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(x.to_string());

        while let Some(node) = queue.pop_front() {
            if node == y {
                return false; // Found path not through M
            }

            if visited.contains(&node) || m.contains(&node) {
                continue;
            }
            visited.insert(node.clone());

            for child in self.children.get(&node).unwrap_or(&HashSet::new()) {
                queue.push_back(child.clone());
            }
        }

        true
    }
}

/// Errors that can occur in causal graph operations
#[derive(Debug, Clone)]
pub enum GraphError {
    /// Node not found in graph
    NodeNotFound(String),
    /// Adding edge would create a cycle
    CycleDetected { from: String, to: String },
    /// Graph is not a DAG
    NotDAG,
    /// Invalid operation
    InvalidOperation(String),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::NodeNotFound(name) => write!(f, "Node '{}' not found in graph", name),
            GraphError::CycleDetected { from, to } => {
                write!(f, "Adding edge {} -> {} would create a cycle", from, to)
            }
            GraphError::NotDAG => write!(f, "Graph is not a directed acyclic graph"),
            GraphError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for GraphError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::treatment("X"));
        g.add_node(CausalNode::mediator("M"));
        g.add_node(CausalNode::outcome("Y"));
        g.add_edge("X", "M", EdgeType::Direct).unwrap();
        g.add_edge("M", "Y", EdgeType::Direct).unwrap();
        g
    }

    fn confounded_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::treatment("X"));
        g.add_node(CausalNode::outcome("Y"));
        g.add_node(CausalNode::latent("U"));
        g.add_edge("X", "Y", EdgeType::Direct).unwrap();
        g.add_edge("U", "X", EdgeType::Direct).unwrap();
        g.add_edge("U", "Y", EdgeType::Direct).unwrap();
        g
    }

    #[test]
    fn test_create_graph() {
        let g = simple_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::observed("A"));
        g.add_node(CausalNode::observed("B"));
        g.add_node(CausalNode::observed("C"));

        g.add_edge("A", "B", EdgeType::Direct).unwrap();
        g.add_edge("B", "C", EdgeType::Direct).unwrap();

        // This should fail - would create cycle
        let result = g.add_edge("C", "A", EdgeType::Direct);
        assert!(matches!(result, Err(GraphError::CycleDetected { .. })));
    }

    #[test]
    fn test_descendants() {
        let g = simple_graph();
        let desc = g.descendants_of("X");
        assert!(desc.contains("M"));
        assert!(desc.contains("Y"));
        assert!(!desc.contains("X"));
    }

    #[test]
    fn test_ancestors() {
        let g = simple_graph();
        let anc = g.ancestors_of("Y");
        assert!(anc.contains("M"));
        assert!(anc.contains("X"));
        assert!(!anc.contains("Y"));
    }

    #[test]
    fn test_d_separation_chain() {
        let g = simple_graph();

        // X -> M -> Y: X and Y are d-connected when M is not conditioned
        assert!(g.d_connected("X", "Y", &HashSet::new()));

        // X -> M -> Y: X and Y are d-separated when M is conditioned
        let z: HashSet<String> = ["M".to_string()].into_iter().collect();
        assert!(g.d_separated("X", "Y", &z));
    }

    #[test]
    fn test_d_separation_fork() {
        let g = confounded_graph();

        // X <- U -> Y: X and Y are d-connected when U is not conditioned
        assert!(g.d_connected("X", "Y", &HashSet::new()));

        // X <- U -> Y: X and Y are d-separated when U is conditioned
        let z: HashSet<String> = ["U".to_string()].into_iter().collect();
        // But they're still connected via X->Y direct edge
        assert!(g.d_connected("X", "Y", &z));
    }

    #[test]
    fn test_d_separation_collider() {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::observed("X"));
        g.add_node(CausalNode::observed("Y"));
        g.add_node(CausalNode::observed("C"));

        g.add_edge("X", "C", EdgeType::Direct).unwrap();
        g.add_edge("Y", "C", EdgeType::Direct).unwrap();

        // X -> C <- Y: X and Y are d-separated when C is NOT conditioned
        assert!(g.d_separated("X", "Y", &HashSet::new()));

        // X -> C <- Y: X and Y are d-connected when C IS conditioned
        let z: HashSet<String> = ["C".to_string()].into_iter().collect();
        assert!(g.d_connected("X", "Y", &z));
    }

    #[test]
    fn test_graph_do() {
        let g = confounded_graph();
        let g_do = g.graph_do("X");

        // X should have no parents after do(X)
        assert!(g_do.parents("X").unwrap().is_empty());

        // U -> Y edge should remain
        assert!(g_do.parents("Y").unwrap().contains("U"));
    }

    #[test]
    fn test_topological_order() {
        let g = simple_graph();
        let order = g.topological_order();

        let x_pos = order.iter().position(|n| n == "X").unwrap();
        let m_pos = order.iter().position(|n| n == "M").unwrap();
        let y_pos = order.iter().position(|n| n == "Y").unwrap();

        assert!(x_pos < m_pos);
        assert!(m_pos < y_pos);
    }

    #[test]
    fn test_find_mediators() {
        let g = simple_graph();
        let mediators = g.find_mediators("X", "Y");
        assert_eq!(mediators, vec!["M"]);
    }
}
