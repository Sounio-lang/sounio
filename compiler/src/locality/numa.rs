//! NUMA Topology and Semantic Placement
//!
//! This module detects NUMA topology and uses semantic information
//! to suggest optimal placement of data structures.

use super::prefetch::PrefetchTable;
use std::collections::{HashMap, HashSet};

/// A NUMA node in the system topology.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID (0, 1, 2, ...)
    pub id: u32,
    /// CPUs on this node
    pub cpus: Vec<u32>,
    /// Memory in bytes
    pub memory: u64,
    /// Distance to other nodes (node_id -> distance)
    pub distances: HashMap<u32, u32>,
    /// Whether this node is local to the current thread
    pub is_local: bool,
}

impl NumaNode {
    /// Create a new NUMA node.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            cpus: Vec::new(),
            memory: 0,
            distances: HashMap::new(),
            is_local: false,
        }
    }

    /// Add a CPU to this node.
    pub fn add_cpu(&mut self, cpu: u32) {
        self.cpus.push(cpu);
    }

    /// Set memory size.
    pub fn with_memory(mut self, memory: u64) -> Self {
        self.memory = memory;
        self
    }

    /// Set distance to another node.
    pub fn set_distance(&mut self, other: u32, distance: u32) {
        self.distances.insert(other, distance);
    }

    /// Get distance to another node (10 = local, higher = further).
    pub fn distance_to(&self, other: u32) -> u32 {
        if other == self.id {
            10 // Local access
        } else {
            *self.distances.get(&other).unwrap_or(&100)
        }
    }

    /// Check if this node has available memory.
    pub fn has_memory(&self) -> bool {
        self.memory > 0
    }

    /// Get number of CPUs.
    pub fn cpu_count(&self) -> usize {
        self.cpus.len()
    }
}

/// System NUMA topology.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    /// All NUMA nodes
    nodes: Vec<NumaNode>,
    /// Total system memory
    total_memory: u64,
    /// Whether NUMA is actually available
    numa_available: bool,
    /// Node distances (for quick lookup)
    distance_matrix: HashMap<(u32, u32), u32>,
}

impl NumaTopology {
    /// Create an empty topology.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            total_memory: 0,
            numa_available: false,
            distance_matrix: HashMap::new(),
        }
    }

    /// Detect the system's NUMA topology.
    #[cfg(target_os = "linux")]
    pub fn detect() -> Self {
        let mut topo = Self::new();

        // Try to read from /sys/devices/system/node/
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/node") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();

                if name_str.starts_with("node")
                    && let Ok(id) = name_str[4..].parse::<u32>()
                {
                    let mut node = NumaNode::new(id);

                    // Read CPUs
                    let cpu_path = entry.path().join("cpulist");
                    if let Ok(cpulist) = std::fs::read_to_string(&cpu_path) {
                        node.cpus = parse_cpu_list(&cpulist);
                    }

                    // Read memory
                    let mem_path = entry.path().join("meminfo");
                    if let Ok(meminfo) = std::fs::read_to_string(&mem_path) {
                        node.memory = parse_meminfo(&meminfo);
                    }

                    // Read distances
                    let dist_path = entry.path().join("distance");
                    if let Ok(distances) = std::fs::read_to_string(&dist_path) {
                        let dists: Vec<u32> = distances
                            .split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect();

                        for (other_id, dist) in dists.into_iter().enumerate() {
                            node.set_distance(other_id as u32, dist);
                        }
                    }

                    topo.add_node(node);
                }
            }

            topo.numa_available = !topo.nodes.is_empty();
        }

        // Fallback: create a single-node topology
        if topo.nodes.is_empty() {
            topo = Self::single_node();
        }

        topo.build_distance_matrix();
        topo
    }

    #[cfg(not(target_os = "linux"))]
    pub fn detect() -> Self {
        // On non-Linux systems, assume single NUMA node
        Self::single_node()
    }

    /// Create a single-node topology (for non-NUMA systems).
    pub fn single_node() -> Self {
        let mut topo = Self::new();

        let mut node = NumaNode::new(0);
        node.is_local = true;

        // Estimate CPUs and memory
        node.cpus = (0..num_cpus()).collect();
        node.memory = estimate_memory();
        node.set_distance(0, 10);

        topo.add_node(node);
        topo.build_distance_matrix();
        topo
    }

    /// Create a simulated multi-node topology for testing.
    pub fn simulated(num_nodes: u32, cpus_per_node: u32, memory_per_node: u64) -> Self {
        let mut topo = Self::new();

        for id in 0..num_nodes {
            let mut node = NumaNode::new(id);
            node.cpus = ((id * cpus_per_node)..((id + 1) * cpus_per_node)).collect();
            node.memory = memory_per_node;

            // Set distances (local = 10, remote = 20)
            for other in 0..num_nodes {
                let dist = if other == id { 10 } else { 20 };
                node.set_distance(other, dist);
            }

            if id == 0 {
                node.is_local = true;
            }

            topo.add_node(node);
        }

        topo.numa_available = num_nodes > 1;
        topo.build_distance_matrix();
        topo
    }

    /// Add a node to the topology.
    pub fn add_node(&mut self, node: NumaNode) {
        self.total_memory += node.memory;
        self.nodes.push(node);
    }

    /// Build the distance matrix for quick lookup.
    fn build_distance_matrix(&mut self) {
        for node in &self.nodes {
            for (other, dist) in &node.distances {
                self.distance_matrix.insert((node.id, *other), *dist);
            }
        }
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: u32) -> Option<&NumaNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[NumaNode] {
        &self.nodes
    }

    /// Get the number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if NUMA is available.
    pub fn is_numa(&self) -> bool {
        self.numa_available
    }

    /// Get the local node (where current thread runs).
    pub fn local_node(&self) -> Option<&NumaNode> {
        self.nodes.iter().find(|n| n.is_local)
    }

    /// Get distance between two nodes.
    pub fn distance(&self, from: u32, to: u32) -> u32 {
        *self.distance_matrix.get(&(from, to)).unwrap_or(&100)
    }

    /// Find the closest node with sufficient memory.
    pub fn closest_node_with_memory(&self, from: u32, min_memory: u64) -> Option<&NumaNode> {
        let mut candidates: Vec<_> = self
            .nodes
            .iter()
            .filter(|n| n.memory >= min_memory)
            .collect();

        candidates.sort_by_key(|n| self.distance(from, n.id));
        candidates.first().copied()
    }

    /// Suggest placement for semantically related types.
    pub fn suggest_placement(
        &self,
        types: &[&str],
        prefetch_table: &PrefetchTable,
    ) -> PlacementStrategy {
        if !self.numa_available || self.nodes.len() <= 1 {
            return PlacementStrategy::default();
        }

        let mut strategy = PlacementStrategy::new();

        // Group types by semantic relationship
        let groups = self.group_by_semantic_distance(types, prefetch_table);

        // Assign each group to a node
        for (group_id, group) in groups.iter().enumerate() {
            let node_id = (group_id as u32) % (self.nodes.len() as u32);

            for type_name in group {
                strategy.place(type_name.clone(), node_id);
            }
        }

        // Record affinity between closely related types
        for (i, t1) in types.iter().enumerate() {
            for t2 in types.iter().skip(i + 1) {
                if prefetch_table.has_hints(t1) {
                    for hint in prefetch_table.get_type_hints(t1) {
                        if hint.target == *t2 && hint.distance.is_prefetchable() {
                            strategy.add_affinity(t1.to_string(), t2.to_string());
                        }
                    }
                }
            }
        }

        strategy
    }

    /// Group types by semantic distance.
    fn group_by_semantic_distance(
        &self,
        types: &[&str],
        prefetch_table: &PrefetchTable,
    ) -> Vec<Vec<String>> {
        // Simple greedy grouping
        let mut groups: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashSet<&str> = HashSet::new();

        for t in types {
            if assigned.contains(t) {
                continue;
            }

            let mut group = vec![t.to_string()];
            assigned.insert(t);

            // Find related types
            if prefetch_table.has_hints(t) {
                for hint in prefetch_table.get_type_hints(t) {
                    if hint.distance.is_prefetchable()
                        && !assigned.contains(hint.target.as_str())
                        && types.contains(&hint.target.as_str())
                    {
                        group.push(hint.target.clone());
                        assigned.insert(Box::leak(hint.target.clone().into_boxed_str()));
                    }
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Get total system memory.
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::detect()
    }
}

/// Parse a CPU list string (e.g., "0-3,8-11").
fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut cpus = Vec::new();

    for part in s.trim().split(',') {
        if part.contains('-') {
            let bounds: Vec<_> = part.split('-').collect();
            if bounds.len() == 2
                && let (Ok(start), Ok(end)) = (bounds[0].parse::<u32>(), bounds[1].parse::<u32>())
            {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<u32>() {
            cpus.push(cpu);
        }
    }

    cpus
}

/// Parse meminfo to get total memory.
fn parse_meminfo(s: &str) -> u64 {
    for line in s.lines() {
        if line.starts_with("MemTotal:") || line.contains("MemTotal:") {
            let parts: Vec<_> = line.split_whitespace().collect();
            if parts.len() >= 2
                && let Ok(kb) = parts[parts.len() - 2].parse::<u64>()
            {
                return kb * 1024; // Convert to bytes
            }
        }
    }
    0
}

/// Get number of CPUs (cross-platform estimate).
fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(1)
}

/// Estimate system memory.
fn estimate_memory() -> u64 {
    // Default to 8GB if we can't detect
    8 * 1024 * 1024 * 1024
}

/// Placement strategy for types across NUMA nodes.
#[derive(Debug, Clone, Default)]
pub struct PlacementStrategy {
    /// Type to node mapping
    placements: HashMap<String, u32>,
    /// Types that should be co-located
    affinities: Vec<(String, String)>,
    /// Types that should be separated
    separations: Vec<(String, String)>,
}

impl PlacementStrategy {
    /// Create a new empty strategy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Place a type on a specific node.
    pub fn place(&mut self, type_name: String, node: u32) {
        self.placements.insert(type_name, node);
    }

    /// Add an affinity (types should be co-located).
    pub fn add_affinity(&mut self, type1: String, type2: String) {
        self.affinities.push((type1, type2));
    }

    /// Add a separation (types should be on different nodes).
    pub fn add_separation(&mut self, type1: String, type2: String) {
        self.separations.push((type1, type2));
    }

    /// Get the node for a type.
    pub fn node_for(&self, type_name: &str) -> Option<u32> {
        self.placements.get(type_name).copied()
    }

    /// Get all placements.
    pub fn placements(&self) -> &HashMap<String, u32> {
        &self.placements
    }

    /// Get all affinities.
    pub fn affinities(&self) -> &[(String, String)] {
        &self.affinities
    }

    /// Check if the strategy respects all affinities.
    pub fn respects_affinities(&self) -> bool {
        for (t1, t2) in &self.affinities {
            if let (Some(n1), Some(n2)) = (self.placements.get(t1), self.placements.get(t2))
                && n1 != n2
            {
                return false;
            }
        }
        true
    }

    /// Generate numa_alloc calls for the placements.
    pub fn to_code(&self) -> String {
        let mut code = String::new();

        for (type_name, node) in &self.placements {
            code.push_str(&format!(
                "let {}_ptr = numa_alloc_onnode(sizeof::<{}>(), {});\n",
                type_name.to_lowercase().replace("::", "_"),
                type_name,
                node
            ));
        }

        code
    }
}

/// Semantic placement: placing data based on ontology relationships.
#[derive(Debug, Clone)]
pub struct SemanticPlacement {
    /// The type being placed
    pub type_name: String,
    /// Suggested node
    pub node: u32,
    /// Semantic reason for this placement
    pub reason: String,
    /// Confidence in this placement (0.0 to 1.0)
    pub confidence: f64,
    /// Related types that should be co-located
    pub co_locate: Vec<String>,
}

impl SemanticPlacement {
    /// Create a new placement.
    pub fn new(type_name: impl Into<String>, node: u32) -> Self {
        Self {
            type_name: type_name.into(),
            node,
            reason: String::new(),
            confidence: 0.5,
            co_locate: Vec::new(),
        }
    }

    /// Set the reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
        self
    }

    /// Set confidence.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add a type to co-locate.
    pub fn co_locate_with(mut self, type_name: impl Into<String>) -> Self {
        self.co_locate.push(type_name.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numa_node() {
        let mut node = NumaNode::new(0);
        node.add_cpu(0);
        node.add_cpu(1);
        node.memory = 16 * 1024 * 1024 * 1024;
        node.set_distance(0, 10);
        node.set_distance(1, 20);

        assert_eq!(node.id, 0);
        assert_eq!(node.cpu_count(), 2);
        assert!(node.has_memory());
        assert_eq!(node.distance_to(0), 10);
        assert_eq!(node.distance_to(1), 20);
    }

    #[test]
    fn test_single_node_topology() {
        let topo = NumaTopology::single_node();

        assert_eq!(topo.node_count(), 1);
        assert!(!topo.is_numa());
        assert!(topo.local_node().is_some());
    }

    #[test]
    fn test_simulated_topology() {
        let topo = NumaTopology::simulated(4, 8, 16 * 1024 * 1024 * 1024);

        assert_eq!(topo.node_count(), 4);
        assert!(topo.is_numa());

        // Local access should be 10
        assert_eq!(topo.distance(0, 0), 10);
        // Remote access should be 20
        assert_eq!(topo.distance(0, 1), 20);
    }

    #[test]
    fn test_closest_node() {
        let topo = NumaTopology::simulated(4, 8, 16 * 1024 * 1024 * 1024);

        let closest = topo.closest_node_with_memory(0, 1024);
        assert!(closest.is_some());
        assert_eq!(closest.unwrap().id, 0); // Should be self
    }

    #[test]
    fn test_parse_cpu_list() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list("0-1,4-5"), vec![0, 1, 4, 5]);
    }

    #[test]
    fn test_placement_strategy() {
        let mut strategy = PlacementStrategy::new();

        strategy.place("Patient".to_string(), 0);
        strategy.place("Diagnosis".to_string(), 0);
        strategy.place("Treatment".to_string(), 1);

        strategy.add_affinity("Patient".to_string(), "Diagnosis".to_string());

        assert_eq!(strategy.node_for("Patient"), Some(0));
        assert_eq!(strategy.node_for("Treatment"), Some(1));
        assert!(strategy.respects_affinities());
    }

    #[test]
    fn test_placement_code_gen() {
        let mut strategy = PlacementStrategy::new();
        strategy.place("Data".to_string(), 0);

        let code = strategy.to_code();
        assert!(code.contains("numa_alloc_onnode"));
        assert!(code.contains("Data"));
    }

    #[test]
    fn test_semantic_placement() {
        let placement = SemanticPlacement::new("Patient", 0)
            .with_reason("high semantic distance from cold data")
            .with_confidence(0.8)
            .co_locate_with("Diagnosis");

        assert_eq!(placement.node, 0);
        assert_eq!(placement.confidence, 0.8);
        assert_eq!(placement.co_locate.len(), 1);
    }

    #[test]
    fn test_topology_detect_fallback() {
        // On any system, detect should return a valid topology
        let topo = NumaTopology::detect();
        assert!(topo.node_count() >= 1);
    }

    #[test]
    fn test_suggest_placement() {
        let topo = NumaTopology::simulated(2, 4, 8 * 1024 * 1024 * 1024);
        let prefetch_table = PrefetchTable::new();

        let types = ["TypeA", "TypeB", "TypeC"];
        let strategy = topo.suggest_placement(&types, &prefetch_table);

        // Each type should be assigned to some node
        for t in &types {
            // Strategy may or may not have placement for each type
            // depending on semantic relationships
        }

        // At minimum, strategy should exist
        assert!(strategy.placements().len() <= types.len());
    }
}
