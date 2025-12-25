// Sounio Compiler - Ontology Hierarchy Graph Operations
// Path-based semantic distance calculations using graph structures

use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet, VecDeque};

use crate::ontology::loader::{IRI, LoadedTerm};

/// Hierarchy graph for computing path-based semantic distances
///
/// Represents ontology subsumption relations (is_a) as a directed acyclic graph.
/// Edges point from child to parent (child is_a parent).
pub struct HierarchyGraph {
    /// Directed graph where nodes are terms and edges are is_a relations
    graph: DiGraph<IRI, EdgeWeight>,

    /// Map from IRI to node index for fast lookup
    iri_to_node: HashMap<IRI, NodeIndex>,

    /// Cached depths from root for each node
    depth_cache: HashMap<NodeIndex, u32>,

    /// Root nodes (terms with no parents)
    roots: HashSet<NodeIndex>,

    /// Leaf nodes (terms with no children)
    leaves: HashSet<NodeIndex>,
}

/// Edge weight for hierarchy relations
#[derive(Debug, Clone, Copy)]
pub struct EdgeWeight {
    /// Base weight (typically 1.0 for direct is_a)
    pub weight: f64,

    /// Whether this is a direct assertion vs inferred
    pub is_asserted: bool,
}

impl Default for EdgeWeight {
    fn default() -> Self {
        Self {
            weight: 1.0,
            is_asserted: true,
        }
    }
}

/// Result of path computation between two terms
#[derive(Debug, Clone)]
pub struct PathResult {
    /// Sequence of IRIs from source to target (empty if no path)
    pub path: Vec<IRI>,

    /// Total path length (number of edges)
    pub length: u32,

    /// Total weighted distance
    pub weighted_distance: f64,

    /// Whether path goes through LCA (up then down)
    pub via_lca: bool,
}

/// Result of LCA (Lowest Common Ancestor) computation
#[derive(Debug, Clone)]
pub struct LCAResult {
    /// The LCA term (or None if no common ancestor)
    pub ancestor: Option<IRI>,

    /// Distance from first term to LCA
    pub dist_a: u32,

    /// Distance from second term to LCA
    pub dist_b: u32,

    /// Depth of LCA from root (higher = more specific)
    pub lca_depth: u32,
}

impl HierarchyGraph {
    /// Create empty hierarchy graph
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            iri_to_node: HashMap::new(),
            depth_cache: HashMap::new(),
            roots: HashSet::new(),
            leaves: HashSet::new(),
        }
    }

    /// Create hierarchy graph with estimated capacity
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: DiGraph::with_capacity(nodes, edges),
            iri_to_node: HashMap::with_capacity(nodes),
            depth_cache: HashMap::with_capacity(nodes),
            roots: HashSet::new(),
            leaves: HashSet::new(),
        }
    }

    /// Add a term to the graph (internal, returns NodeIndex)
    fn add_term_internal(&mut self, iri: IRI) -> NodeIndex {
        if let Some(&idx) = self.iri_to_node.get(&iri) {
            return idx;
        }

        let idx = self.graph.add_node(iri.clone());
        self.iri_to_node.insert(iri, idx);

        // Initially both a root and leaf until edges are added
        self.roots.insert(idx);
        self.leaves.insert(idx);

        idx
    }

    /// Add a term with its superclasses (parents)
    /// This is the main interface used by SemanticDistanceIndex
    pub fn add_term(&mut self, iri: &IRI, superclasses: &[IRI]) {
        let child_idx = self.add_term_internal(iri.clone());

        for parent_iri in superclasses {
            let parent_idx = self.add_term_internal(parent_iri.clone());

            // Edge direction: child -> parent (child is_a parent)
            if !self.graph.contains_edge(child_idx, parent_idx) {
                self.graph
                    .add_edge(child_idx, parent_idx, EdgeWeight::default());

                // Child is no longer a root (has parent)
                self.roots.remove(&child_idx);

                // Parent is no longer a leaf (has child)
                self.leaves.remove(&parent_idx);
            }
        }

        // Invalidate depth cache when graph changes
        if !superclasses.is_empty() {
            self.depth_cache.clear();
        }
    }

    /// Add a single term without parents
    pub fn add_single_term(&mut self, iri: IRI) -> NodeIndex {
        self.add_term_internal(iri)
    }

    /// Add an is_a relation (child is_a parent)
    pub fn add_is_a(&mut self, child: &IRI, parent: &IRI) {
        self.add_is_a_weighted(child, parent, EdgeWeight::default());
    }

    /// Add an is_a relation with custom weight
    pub fn add_is_a_weighted(&mut self, child: &IRI, parent: &IRI, weight: EdgeWeight) {
        let child_idx = self.add_term_internal(child.clone());
        let parent_idx = self.add_term_internal(parent.clone());

        // Edge direction: child -> parent (child is_a parent)
        self.graph.add_edge(child_idx, parent_idx, weight);

        // Child is no longer a root (has parent)
        self.roots.remove(&child_idx);

        // Parent is no longer a leaf (has child)
        self.leaves.remove(&parent_idx);

        // Invalidate depth cache
        self.depth_cache.clear();
    }

    /// Build hierarchy from loaded terms
    pub fn build_from_terms(&mut self, terms: &[LoadedTerm]) {
        // First pass: add all terms
        for term in terms {
            self.add_term_internal(term.iri.clone());
        }

        // Second pass: add relations
        for term in terms {
            for parent_iri in &term.superclasses {
                self.add_is_a(&term.iri, parent_iri);
            }
        }
    }

    /// Check if term exists in graph
    pub fn contains(&self, iri: &IRI) -> bool {
        self.iri_to_node.contains_key(iri)
    }

    /// Get node index for IRI
    pub fn get_node(&self, iri: &IRI) -> Option<NodeIndex> {
        self.iri_to_node.get(iri).copied()
    }

    /// Get IRI for node index
    pub fn get_iri(&self, idx: NodeIndex) -> Option<&IRI> {
        self.graph.node_weight(idx)
    }

    /// Check if `ancestor` is an ancestor of `descendant`
    pub fn is_ancestor(&self, ancestor: &IRI, descendant: &IRI) -> bool {
        let Some(ancestor_idx) = self.get_node(ancestor) else {
            return false;
        };
        let Some(descendant_idx) = self.get_node(descendant) else {
            return false;
        };

        self.is_ancestor_idx(ancestor_idx, descendant_idx)
    }

    /// Check ancestry by node indices
    fn is_ancestor_idx(&self, ancestor: NodeIndex, descendant: NodeIndex) -> bool {
        if ancestor == descendant {
            return true;
        }

        // BFS upward from descendant
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(descendant);

        while let Some(current) = queue.pop_front() {
            if !visited.insert(current) {
                continue;
            }

            for parent in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if parent == ancestor {
                    return true;
                }
                queue.push_back(parent);
            }
        }

        false
    }

    /// Compute shortest path length between two terms
    /// Returns None if no path exists
    pub fn path_length(&self, from: &IRI, to: &IRI) -> Option<u32> {
        let from_idx = self.get_node(from)?;
        let to_idx = self.get_node(to)?;

        // Try direct path first (via ancestry)
        if let Some(len) = self.directed_path_length(from_idx, to_idx) {
            return Some(len);
        }

        // Try reverse path
        if let Some(len) = self.directed_path_length(to_idx, from_idx) {
            return Some(len);
        }

        // Path via LCA
        let lca = self.lowest_common_ancestor_idx(from_idx, to_idx)?;
        Some(lca.dist_a + lca.dist_b)
    }

    /// Directed path length (following is_a edges upward)
    fn directed_path_length(&self, from: NodeIndex, to: NodeIndex) -> Option<u32> {
        if from == to {
            return Some(0);
        }

        // BFS upward from 'from'
        let mut visited = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back((from, 0u32));

        while let Some((current, dist)) = queue.pop_front() {
            if visited.contains_key(&current) {
                continue;
            }
            visited.insert(current, dist);

            for parent in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if parent == to {
                    return Some(dist + 1);
                }
                if !visited.contains_key(&parent) {
                    queue.push_back((parent, dist + 1));
                }
            }
        }

        None
    }

    /// Compute Lowest Common Ancestor (LCA) of two terms
    /// Returns just the LCA IRI for simple usage
    pub fn lowest_common_ancestor(&self, a: &IRI, b: &IRI) -> Option<IRI> {
        let a_idx = self.get_node(a)?;
        let b_idx = self.get_node(b)?;
        self.lowest_common_ancestor_idx(a_idx, b_idx)
            .and_then(|result| result.ancestor)
    }

    /// Compute LCA with full result details (distances, depth)
    pub fn lowest_common_ancestor_full(&self, a: &IRI, b: &IRI) -> Option<LCAResult> {
        let a_idx = self.get_node(a)?;
        let b_idx = self.get_node(b)?;
        self.lowest_common_ancestor_idx(a_idx, b_idx)
    }

    /// LCA computation by node indices
    fn lowest_common_ancestor_idx(&self, a: NodeIndex, b: NodeIndex) -> Option<LCAResult> {
        if a == b {
            let depth = self.compute_depth(a);
            return Some(LCAResult {
                ancestor: self.get_iri(a).cloned(),
                dist_a: 0,
                dist_b: 0,
                lca_depth: depth,
            });
        }

        // Get all ancestors of both terms with their distances
        let ancestors_a = self.get_ancestors_with_dist(a);
        let ancestors_b = self.get_ancestors_with_dist(b);

        // Include self as ancestor with distance 0
        let mut ancestors_a = ancestors_a;
        let mut ancestors_b = ancestors_b;
        ancestors_a.insert(a, 0);
        ancestors_b.insert(b, 0);

        // Find common ancestors with minimum total distance
        let mut best_lca: Option<(NodeIndex, u32, u32)> = None;
        let mut best_depth = 0u32;

        for (ancestor, dist_a) in &ancestors_a {
            if let Some(&dist_b) = ancestors_b.get(ancestor) {
                let total_dist = dist_a + dist_b;
                let depth = self.compute_depth(*ancestor);

                // Prefer deeper LCA (more specific common ancestor)
                // Break ties by total distance
                let is_better = match best_lca {
                    None => true,
                    Some((_, best_dist_a, best_dist_b)) => {
                        let best_total = best_dist_a + best_dist_b;
                        depth > best_depth || (depth == best_depth && total_dist < best_total)
                    }
                };

                if is_better {
                    best_lca = Some((*ancestor, *dist_a, dist_b));
                    best_depth = depth;
                }
            }
        }

        best_lca.map(|(ancestor, dist_a, dist_b)| LCAResult {
            ancestor: self.get_iri(ancestor).cloned(),
            dist_a,
            dist_b,
            lca_depth: best_depth,
        })
    }

    /// Get all ancestors of a term
    pub fn get_ancestors(&self, iri: &IRI) -> HashSet<IRI> {
        let Some(idx) = self.get_node(iri) else {
            return HashSet::new();
        };

        self.get_ancestors_with_dist(idx)
            .keys()
            .filter_map(|&idx| self.get_iri(idx).cloned())
            .collect()
    }

    /// Get ancestors with their distances
    fn get_ancestors_with_dist(&self, start: NodeIndex) -> HashMap<NodeIndex, u32> {
        let mut ancestors = HashMap::new();
        let mut queue = VecDeque::new();

        // Start with immediate parents
        for parent in self.graph.neighbors_directed(start, Direction::Outgoing) {
            queue.push_back((parent, 1u32));
        }

        while let Some((current, dist)) = queue.pop_front() {
            if ancestors.contains_key(&current) {
                continue;
            }
            ancestors.insert(current, dist);

            for parent in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if !ancestors.contains_key(&parent) {
                    queue.push_back((parent, dist + 1));
                }
            }
        }

        ancestors
    }

    /// Get all descendants of a term
    pub fn get_descendants(&self, iri: &IRI) -> HashSet<IRI> {
        let Some(idx) = self.get_node(iri) else {
            return HashSet::new();
        };

        let mut descendants = HashSet::new();
        let mut queue = VecDeque::new();

        // Start with immediate children
        for child in self.graph.neighbors_directed(idx, Direction::Incoming) {
            queue.push_back(child);
        }

        while let Some(current) = queue.pop_front() {
            if let Some(iri) = self.get_iri(current)
                && descendants.insert(iri.clone())
            {
                for child in self.graph.neighbors_directed(current, Direction::Incoming) {
                    queue.push_back(child);
                }
            }
        }

        descendants
    }

    /// Get immediate parents (direct is_a relations)
    pub fn get_parents(&self, iri: &IRI) -> Vec<IRI> {
        let Some(idx) = self.get_node(iri) else {
            return Vec::new();
        };

        self.graph
            .neighbors_directed(idx, Direction::Outgoing)
            .filter_map(|parent| self.get_iri(parent).cloned())
            .collect()
    }

    /// Get immediate children (direct subclasses)
    pub fn get_children(&self, iri: &IRI) -> Vec<IRI> {
        let Some(idx) = self.get_node(iri) else {
            return Vec::new();
        };

        self.graph
            .neighbors_directed(idx, Direction::Incoming)
            .filter_map(|child| self.get_iri(child).cloned())
            .collect()
    }

    /// Compute depth of a node from root
    /// Root nodes have depth 0
    pub fn compute_depth(&self, node: NodeIndex) -> u32 {
        // Check cache
        if let Some(&depth) = self.depth_cache.get(&node) {
            return depth;
        }

        if self.roots.contains(&node) {
            return 0;
        }

        // Find minimum depth path to any root
        let mut min_depth = u32::MAX;
        for parent in self.graph.neighbors_directed(node, Direction::Outgoing) {
            let parent_depth = self.compute_depth(parent);
            min_depth = min_depth.min(parent_depth + 1);
        }

        if min_depth == u32::MAX {
            min_depth = 0; // Orphan node treated as root
        }

        min_depth
    }

    /// Get depth for term by IRI
    pub fn depth(&self, iri: &IRI) -> Option<u32> {
        let idx = self.get_node(iri)?;
        Some(self.compute_depth(idx))
    }

    /// Find the complete path between two terms (returns just IRI sequence)
    pub fn find_path(&self, from: &IRI, to: &IRI) -> Option<Vec<IRI>> {
        self.find_path_full(from, to).map(|result| result.path)
    }

    /// Find the complete path between two terms with full details
    pub fn find_path_full(&self, from: &IRI, to: &IRI) -> Option<PathResult> {
        let from_idx = self.get_node(from)?;
        let to_idx = self.get_node(to)?;

        // Same term
        if from_idx == to_idx {
            return Some(PathResult {
                path: vec![from.clone()],
                length: 0,
                weighted_distance: 0.0,
                via_lca: false,
            });
        }

        // Try upward path (from is descendant of to)
        if let Some(path) = self.find_upward_path(from_idx, to_idx) {
            return Some(path);
        }

        // Try downward path (from is ancestor of to)
        if let Some(mut path) = self.find_upward_path(to_idx, from_idx) {
            path.path.reverse();
            path.via_lca = false;
            return Some(path);
        }

        // Path via LCA
        self.find_path_via_lca(from_idx, to_idx)
    }

    /// Find upward path (following is_a edges)
    fn find_upward_path(&self, from: NodeIndex, to: NodeIndex) -> Option<PathResult> {
        // BFS to find shortest path
        let mut visited = HashMap::new();
        let mut parent_map: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(from);
        visited.insert(from, 0u32);

        while let Some(current) = queue.pop_front() {
            let current_dist = visited[&current];

            for parent in self.graph.neighbors_directed(current, Direction::Outgoing) {
                if parent == to {
                    // Found target, reconstruct path
                    parent_map.insert(parent, current);
                    let path = self.reconstruct_path(&parent_map, from, to);
                    return Some(PathResult {
                        path,
                        length: current_dist + 1,
                        weighted_distance: (current_dist + 1) as f64,
                        via_lca: false,
                    });
                }

                if let std::collections::hash_map::Entry::Vacant(e) = visited.entry(parent) {
                    e.insert(current_dist + 1);
                    parent_map.insert(parent, current);
                    queue.push_back(parent);
                }
            }
        }

        None
    }

    /// Find path via LCA for siblings
    fn find_path_via_lca(&self, from: NodeIndex, to: NodeIndex) -> Option<PathResult> {
        let lca = self.lowest_common_ancestor_idx(from, to)?;
        let lca_idx = self.get_node(lca.ancestor.as_ref()?)?;

        // Get path from 'from' to LCA
        let path_up = self.find_upward_path(from, lca_idx)?;

        // Get path from 'to' to LCA (then reverse)
        let path_down = self.find_upward_path(to, lca_idx)?;

        // Combine paths
        let mut combined = path_up.path;
        let mut down_path: Vec<_> = path_down.path.into_iter().rev().skip(1).collect();
        combined.append(&mut down_path);

        Some(PathResult {
            path: combined,
            length: lca.dist_a + lca.dist_b,
            weighted_distance: (lca.dist_a + lca.dist_b) as f64,
            via_lca: true,
        })
    }

    /// Reconstruct path from parent map
    fn reconstruct_path(
        &self,
        parent_map: &HashMap<NodeIndex, NodeIndex>,
        from: NodeIndex,
        to: NodeIndex,
    ) -> Vec<IRI> {
        let mut path = Vec::new();
        let mut current = to;

        while current != from {
            if let Some(iri) = self.get_iri(current) {
                path.push(iri.clone());
            }
            current = parent_map[&current];
        }

        if let Some(iri) = self.get_iri(from) {
            path.push(iri.clone());
        }

        path.reverse();
        path
    }

    /// Compute Wu-Palmer similarity between two terms
    /// Returns value in [0, 1] where 1 means identical
    pub fn wu_palmer_similarity(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let a_idx = self.get_node(a)?;
        let b_idx = self.get_node(b)?;

        if a_idx == b_idx {
            return Some(1.0);
        }

        let lca = self.lowest_common_ancestor_idx(a_idx, b_idx)?;

        // Wu-Palmer: 2 * depth(LCA) / (depth(a) + depth(b))
        let depth_a = self.compute_depth(a_idx);
        let depth_b = self.compute_depth(b_idx);
        let depth_lca = lca.lca_depth;

        let denominator = depth_a + depth_b;
        if denominator == 0 {
            return Some(1.0); // Both are roots
        }

        Some((2 * depth_lca) as f64 / denominator as f64)
    }

    /// Compute path-based distance (Rada et al.)
    /// Normalized to [0, 1] where 0 means identical
    pub fn path_distance(&self, a: &IRI, b: &IRI) -> Option<f64> {
        let length = self.path_length(a, b)?;

        // Normalize by maximum possible depth
        let max_depth = self.max_depth();
        if max_depth == 0 {
            return Some(0.0);
        }

        // Path of 2*max_depth would connect two deepest leaves via root
        let max_path = 2 * max_depth;
        Some(length as f64 / max_path as f64)
    }

    /// Get maximum depth in the graph
    pub fn max_depth(&self) -> u32 {
        self.leaves
            .iter()
            .map(|&leaf| self.compute_depth(leaf))
            .max()
            .unwrap_or(0)
    }

    /// Get graph statistics
    pub fn stats(&self) -> HierarchyStats {
        HierarchyStats {
            num_terms: self.graph.node_count(),
            num_relations: self.graph.edge_count(),
            num_roots: self.roots.len(),
            num_leaves: self.leaves.len(),
            max_depth: self.max_depth(),
        }
    }
}

impl Default for HierarchyGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the hierarchy graph
#[derive(Debug, Clone)]
pub struct HierarchyStats {
    pub num_terms: usize,
    pub num_relations: usize,
    pub num_roots: usize,
    pub num_leaves: usize,
    pub max_depth: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_graph() -> HierarchyGraph {
        // Create a simple hierarchy:
        //        Thing
        //       /     \
        //    Animal   Plant
        //    /    \
        //  Dog    Cat

        let mut graph = HierarchyGraph::new();

        let thing = IRI::new("http://example.org/Thing");
        let animal = IRI::new("http://example.org/Animal");
        let plant = IRI::new("http://example.org/Plant");
        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        graph.add_is_a(&animal, &thing);
        graph.add_is_a(&plant, &thing);
        graph.add_is_a(&dog, &animal);
        graph.add_is_a(&cat, &animal);

        graph
    }

    #[test]
    fn test_ancestry() {
        let graph = make_test_graph();

        let thing = IRI::new("http://example.org/Thing");
        let animal = IRI::new("http://example.org/Animal");
        let dog = IRI::new("http://example.org/Dog");
        let plant = IRI::new("http://example.org/Plant");

        assert!(graph.is_ancestor(&thing, &dog));
        assert!(graph.is_ancestor(&animal, &dog));
        assert!(!graph.is_ancestor(&plant, &dog));
        assert!(!graph.is_ancestor(&dog, &animal));
    }

    #[test]
    fn test_path_length() {
        let graph = make_test_graph();

        let thing = IRI::new("http://example.org/Thing");
        let animal = IRI::new("http://example.org/Animal");
        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        assert_eq!(graph.path_length(&dog, &dog), Some(0));
        assert_eq!(graph.path_length(&dog, &animal), Some(1));
        assert_eq!(graph.path_length(&dog, &thing), Some(2));
        assert_eq!(graph.path_length(&dog, &cat), Some(2)); // via Animal
    }

    #[test]
    fn test_lca() {
        let graph = make_test_graph();

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");
        let plant = IRI::new("http://example.org/Plant");

        // LCA of Dog and Cat should be Animal
        let lca = graph.lowest_common_ancestor(&dog, &cat).unwrap();
        assert_eq!(lca.as_str(), "http://example.org/Animal");

        // Use full version for detailed checks
        let lca_full = graph.lowest_common_ancestor_full(&dog, &cat).unwrap();
        assert_eq!(lca_full.dist_a, 1);
        assert_eq!(lca_full.dist_b, 1);

        // LCA of Dog and Plant should be Thing
        let lca = graph.lowest_common_ancestor(&dog, &plant).unwrap();
        assert_eq!(lca.as_str(), "http://example.org/Thing");
    }

    #[test]
    fn test_depth() {
        let graph = make_test_graph();

        let thing = IRI::new("http://example.org/Thing");
        let animal = IRI::new("http://example.org/Animal");
        let dog = IRI::new("http://example.org/Dog");

        assert_eq!(graph.depth(&thing), Some(0));
        assert_eq!(graph.depth(&animal), Some(1));
        assert_eq!(graph.depth(&dog), Some(2));
    }

    #[test]
    fn test_wu_palmer() {
        let graph = make_test_graph();

        let dog = IRI::new("http://example.org/Dog");
        let cat = IRI::new("http://example.org/Cat");

        let sim = graph.wu_palmer_similarity(&dog, &cat).unwrap();
        // depth(LCA=Animal) = 1, depth(Dog) = 2, depth(Cat) = 2
        // WP = 2*1 / (2+2) = 0.5
        assert!((sim - 0.5).abs() < 0.001);
    }
}
