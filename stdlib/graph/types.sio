// Graph Types Module - Core data structures for network geometry analysis

module graph::types

// Type Aliases
pub type NodeId = usize;
pub type Degree = usize;
pub type Count = usize;
pub type Curvature = f64;
pub type Probability = f64;
pub type Distance = usize;

// Geometry Classification
pub enum Geometry {
    Hyperbolic,  // Negative curvature
    Euclidean,   // Zero curvature
    Spherical,   // Positive curvature
}

impl Geometry {
    pub fn from_curvature(kappa: Curvature) -> Geometry {
        if kappa < -0.05 {
            Geometry::Hyperbolic
        } else if kappa > 0.05 {
            Geometry::Spherical
        } else {
            Geometry::Euclidean
        }
    }
}

// Edge Type
pub struct Edge {
    pub u: NodeId,
    pub v: NodeId,
}

impl Edge {
    pub fn new(u: NodeId, v: NodeId) -> Edge {
        if u <= v { Edge { u, v } } else { Edge { u: v, v: u } }
    }

    pub fn other(&self, node: NodeId) -> NodeId {
        if self.u == node { self.v } else { self.u }
    }
}

// Graph Type - adjacency list representation
pub struct Graph {
    n: Count,
    adj: [[NodeId]],
    m: Count,
}

impl Graph {
    pub fn new(n: Count) -> Graph {
        var adj: [[NodeId]] = []
        var i: usize = 0
        while i < n {
            adj.push([])
            i = i + 1
        }
        Graph { n: n, adj: adj, m: 0 }
    }

    pub fn num_nodes(&self) -> Count { self.n }
    pub fn num_edges(&self) -> Count { self.m }

    pub fn add_edge(&mut self, u: NodeId, v: NodeId) {
        if u < self.n && v < self.n && u != v {
            self.adj[u].push(v)
            self.adj[v].push(u)
            self.m = self.m + 1
        }
    }

    pub fn degree(&self, u: NodeId) -> Degree {
        if u < self.n { self.adj[u].len() } else { 0 }
    }

    pub fn neighbors(&self, u: NodeId) -> &[NodeId] {
        &self.adj[u]
    }

    pub fn has_edge(&self, u: NodeId, v: NodeId) -> bool {
        if u >= self.n || v >= self.n { return false }
        let neighbors = &self.adj[u]
        var i: usize = 0
        while i < neighbors.len() {
            if neighbors[i] == v { return true }
            i = i + 1
        }
        false
    }

    pub fn edges(&self) -> [Edge] {
        var result: [Edge] = []
        var u: NodeId = 0
        while u < self.n {
            let neighbors = &self.adj[u]
            var i: usize = 0
            while i < neighbors.len() {
                let v = neighbors[i]
                if u < v { result.push(Edge::new(u, v)) }
                i = i + 1
            }
            u = u + 1
        }
        result
    }

    pub fn mean_degree(&self) -> f64 {
        if self.n == 0 { return 0.0 }
        (2.0 * (self.m as f64)) / (self.n as f64)
    }

    pub fn sparsity_ratio(&self) -> f64 {
        let k = self.mean_degree()
        k * k / (self.n as f64)
    }
}

// Network Statistics
pub struct NetworkStats {
    pub num_nodes: Count,
    pub num_edges: Count,
    pub mean_degree: f64,
    pub sparsity_ratio: f64,
    pub predicted_geometry: Geometry,
}

impl NetworkStats {
    pub fn from_graph(g: &Graph) -> NetworkStats {
        let ratio = g.sparsity_ratio()
        let geometry = if ratio < 2.0 {
            Geometry::Hyperbolic
        } else if ratio > 3.5 {
            Geometry::Spherical
        } else {
            Geometry::Euclidean
        }
        NetworkStats {
            num_nodes: g.num_nodes(),
            num_edges: g.num_edges(),
            mean_degree: g.mean_degree(),
            sparsity_ratio: ratio,
            predicted_geometry: geometry,
        }
    }
}

// Probability Measure for optimal transport
pub struct ProbabilityMeasure {
    pub nodes: [NodeId],
    pub probs: [Probability],
}

impl ProbabilityMeasure {
    pub fn empty() -> ProbabilityMeasure {
        ProbabilityMeasure { nodes: [], probs: [] }
    }

    pub fn uniform(nodes: [NodeId]) -> ProbabilityMeasure {
        let n = nodes.len()
        if n == 0 { return ProbabilityMeasure::empty() }
        let p = 1.0 / (n as f64)
        var probs: [Probability] = []
        var i: usize = 0
        while i < n {
            probs.push(p)
            i = i + 1
        }
        ProbabilityMeasure { nodes, probs }
    }

    pub fn total(&self) -> Probability {
        var sum: f64 = 0.0
        var i: usize = 0
        while i < self.probs.len() {
            sum = sum + self.probs[i]
            i = i + 1
        }
        sum
    }

    pub fn support_size(&self) -> usize { self.nodes.len() }
}

// Tests
pub fn test_graph_creation() -> bool {
    var g = Graph::new(5)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(4, 0)
    g.num_nodes() == 5 && g.num_edges() == 5
}

pub fn test_degree() -> bool {
    var g = Graph::new(4)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.degree(0) == 3 && g.degree(1) == 1
}

pub fn run_types_tests() -> bool {
    test_graph_creation() && test_degree()
}
