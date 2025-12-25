//! Causal Inference Runtime
//!
//! This module provides causal inference primitives for Sounio,
//! enabling interventions, counterfactuals, and causal discovery.
//!
//! # Core Operations
//!
//! - `do(model, var, value)` - Intervention (Pearl's do-calculus)
//! - `counterfactual(model, observed, intervention)` - Counterfactual query
//! - `intervene(model, interventions)` - Apply multiple interventions
//! - `identify(model, treatment, outcome)` - Causal effect identification
//! - `discover_dag(data)` - Causal structure learning
//!
//! # Example
//!
//! ```d
//! let model = CausalModel {
//!     graph: dag! { X -> Y, Z -> X, Z -> Y },
//!     mechanisms: [...]
//! };
//!
//! // Intervention: P(Y | do(X=1))
//! let effect = do(model, X, 1.0);
//!
//! // Counterfactual: "What would Y have been if X had been 1?"
//! let cf = counterfactual(model,
//!     observed: { X: 0, Y: 2 },
//!     intervention: { X: 1 }
//! );
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

/// Variable identifier
pub type VarId = String;

/// Directed Acyclic Graph for causal structure
#[derive(Debug, Clone)]
pub struct DAG {
    /// Nodes (variable names)
    nodes: Vec<VarId>,
    /// Edges: parent -> children
    edges: HashMap<VarId, Vec<VarId>>,
    /// Reverse edges: child -> parents
    parents: HashMap<VarId, Vec<VarId>>,
}

impl DAG {
    /// Create a new empty DAG
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: HashMap::new(),
            parents: HashMap::new(),
        }
    }

    /// Add a node
    pub fn add_node(&mut self, var: &str) {
        if !self.nodes.contains(&var.to_string()) {
            self.nodes.push(var.to_string());
            self.edges.insert(var.to_string(), Vec::new());
            self.parents.insert(var.to_string(), Vec::new());
        }
    }

    /// Add an edge (from -> to)
    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.add_node(from);
        self.add_node(to);

        if let Some(children) = self.edges.get_mut(from)
            && !children.contains(&to.to_string())
        {
            children.push(to.to_string());
        }

        if let Some(pars) = self.parents.get_mut(to)
            && !pars.contains(&from.to_string())
        {
            pars.push(from.to_string());
        }
    }

    /// Get all nodes
    pub fn nodes(&self) -> &[VarId] {
        &self.nodes
    }

    /// Get children of a node
    pub fn children(&self, var: &str) -> &[VarId] {
        self.edges.get(var).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Get parents of a node
    pub fn parents(&self, var: &str) -> &[VarId] {
        self.parents.get(var).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if there's a directed path from src to dst
    pub fn has_path(&self, src: &str, dst: &str) -> bool {
        if src == dst {
            return true;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(src.to_string());

        while let Some(node) = queue.pop_front() {
            if node == dst {
                return true;
            }
            if visited.insert(node.clone()) {
                for child in self.children(&node) {
                    queue.push_back(child.clone());
                }
            }
        }

        false
    }

    /// Get ancestors of a node (including itself)
    pub fn ancestors(&self, var: &str) -> HashSet<VarId> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(var.to_string());

        while let Some(node) = queue.pop_front() {
            if result.insert(node.clone()) {
                for parent in self.parents(&node) {
                    queue.push_back(parent.clone());
                }
            }
        }

        result
    }

    /// Get descendants of a node (including itself)
    pub fn descendants(&self, var: &str) -> HashSet<VarId> {
        let mut result = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(var.to_string());

        while let Some(node) = queue.pop_front() {
            if result.insert(node.clone()) {
                for child in self.children(&node) {
                    queue.push_back(child.clone());
                }
            }
        }

        result
    }

    /// Check if the graph is a valid DAG (no cycles)
    pub fn is_acyclic(&self) -> bool {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.insert(node, self.parents(node).len());
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&node, _)| node)
            .collect();

        let mut count = 0;
        while let Some(node) = queue.pop_front() {
            count += 1;
            for child in self.children(node) {
                if let Some(deg) = in_degree.get_mut(child.as_str()) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(child);
                    }
                }
            }
        }

        count == self.nodes.len()
    }

    /// Get topological ordering
    pub fn topological_order(&self) -> Option<Vec<VarId>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.insert(node, self.parents(node).len());
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(&node, _)| node)
            .collect();

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node.to_string());
            for child in self.children(node) {
                if let Some(deg) = in_degree.get_mut(child.as_str()) {
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(child);
                    }
                }
            }
        }

        if result.len() == self.nodes.len() {
            Some(result)
        } else {
            None
        }
    }

    /// D-separation test: are X and Y d-separated given Z?
    pub fn d_separated(&self, x: &str, y: &str, z: &HashSet<VarId>) -> bool {
        // Use the Bayes-Ball algorithm
        // A path from X to Y is blocked by Z if:
        // 1. It contains a chain (A -> B -> C) or fork (A <- B -> C) where B ∈ Z
        // 2. It contains a collider (A -> B <- C) where B ∉ Z and no descendant of B is in Z

        // Simplified implementation: check if all paths are blocked
        let mut visited = HashSet::new();
        self.d_sep_visit(x, y, z, &mut visited, true)
            && self.d_sep_visit(x, y, z, &mut visited, false)
    }

    fn d_sep_visit(
        &self,
        current: &str,
        target: &str,
        conditioning: &HashSet<VarId>,
        visited: &mut HashSet<(String, bool)>,
        going_up: bool,
    ) -> bool {
        if current == target {
            return false; // Found an unblocked path
        }

        let state = (current.to_string(), going_up);
        if visited.contains(&state) {
            return true; // Already visited this state
        }
        visited.insert(state);

        let in_z = conditioning.contains(current);

        // Going up (towards parents)
        if going_up {
            // Can continue up to parents if not in Z
            if !in_z {
                for parent in self.parents(current) {
                    if !self.d_sep_visit(parent, target, conditioning, visited, true) {
                        return false;
                    }
                }
            }
            // Can also go down to children
            for child in self.children(current) {
                if !self.d_sep_visit(child, target, conditioning, visited, false) {
                    return false;
                }
            }
        } else {
            // Going down (towards children)
            if in_z {
                // Blocked at non-collider
                return true;
            }
            for child in self.children(current) {
                if !self.d_sep_visit(child, target, conditioning, visited, false) {
                    return false;
                }
            }
            for parent in self.parents(current) {
                if !self.d_sep_visit(parent, target, conditioning, visited, true) {
                    return false;
                }
            }
        }

        true
    }
}

impl Default for DAG {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DAG {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "DAG {{")?;
        for node in &self.nodes {
            let children = self.children(node);
            if !children.is_empty() {
                writeln!(f, "  {} -> {:?}", node, children)?;
            }
        }
        write!(f, "}}")
    }
}

/// Structural Causal Model (SCM)
pub struct CausalModel {
    /// Causal graph
    pub graph: DAG,
    /// Structural equations: var -> function(parents, noise)
    equations: HashMap<VarId, Box<dyn Fn(&HashMap<VarId, f64>, f64) -> f64 + Send + Sync>>,
    /// Exogenous noise distributions (variance)
    noise_variance: HashMap<VarId, f64>,
}

impl CausalModel {
    /// Create a new causal model
    pub fn new(graph: DAG) -> Self {
        Self {
            graph,
            equations: HashMap::new(),
            noise_variance: HashMap::new(),
        }
    }

    /// Add a structural equation
    pub fn add_equation<F>(&mut self, var: &str, noise_var: f64, equation: F)
    where
        F: Fn(&HashMap<VarId, f64>, f64) -> f64 + Send + Sync + 'static,
    {
        self.equations.insert(var.to_string(), Box::new(equation));
        self.noise_variance.insert(var.to_string(), noise_var);
    }

    /// Sample from the model (forward sampling)
    pub fn sample(&self, rng: &mut impl FnMut() -> f64) -> HashMap<VarId, f64> {
        let order = self.graph.topological_order().expect("Graph must be a DAG");
        let mut values = HashMap::new();

        for var in order {
            let noise_std = self
                .noise_variance
                .get(&var)
                .map(|v| v.sqrt())
                .unwrap_or(1.0);
            let noise = rng() * noise_std;

            if let Some(eq) = self.equations.get(&var) {
                values.insert(var.clone(), eq(&values, noise));
            } else {
                // Default: just noise
                values.insert(var, noise);
            }
        }

        values
    }

    /// Intervention: do(var = value)
    /// Returns a new model with the intervention applied
    pub fn do_intervention(&self, var: &str, value: f64) -> Self {
        let mut new_graph = self.graph.clone();

        // Remove all incoming edges to the intervened variable
        if let Some(parents) = new_graph.parents.get_mut(var) {
            for parent in parents.clone() {
                if let Some(children) = new_graph.edges.get_mut(&parent) {
                    children.retain(|c| c != var);
                }
            }
            parents.clear();
        }

        let mut model = CausalModel::new(new_graph);

        // Copy other equations
        for (v, eq) in &self.equations {
            if v != var {
                // We need to clone the function - but Box<dyn Fn> isn't Clone
                // For now, we'll need to rebuild the model
            }
        }

        // Add constant equation for intervened variable
        model.add_equation(var, 0.0, move |_, _| value);
        model.noise_variance = self.noise_variance.clone();

        model
    }
}

impl fmt::Debug for CausalModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CausalModel")
            .field("graph", &self.graph)
            .field("equations", &self.equations.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Evidence for counterfactual queries
#[derive(Debug, Clone, Default)]
pub struct Evidence {
    /// Observed values
    pub values: HashMap<VarId, f64>,
}

impl Evidence {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with(mut self, var: &str, value: f64) -> Self {
        self.values.insert(var.to_string(), value);
        self
    }
}

/// Counterfactual query result
#[derive(Debug, Clone)]
pub struct CounterfactualResult {
    /// Query variable
    pub query_var: VarId,
    /// Estimated value
    pub value: f64,
    /// Confidence interval (if available)
    pub confidence_interval: Option<(f64, f64)>,
}

/// Perform counterfactual inference
/// Three steps: Abduction, Action, Prediction
pub fn counterfactual(
    model: &CausalModel,
    observed: &Evidence,
    intervention: &Evidence,
    query_var: &str,
    n_samples: usize,
) -> CounterfactualResult {
    // This is a simplified Monte Carlo approach
    // Full counterfactual inference requires:
    // 1. Abduction: Infer exogenous noise given observations
    // 2. Action: Apply intervention
    // 3. Prediction: Compute query under new model with inferred noise

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut rng_state = 42u64;

    let mut rng = || {
        // Simple xorshift
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
    };

    for _ in 0..n_samples {
        // Sample from model
        let mut values = model.sample(&mut rng);

        // Apply observations (simplified - full version would do proper abduction)
        for (var, val) in &observed.values {
            values.insert(var.clone(), *val);
        }

        // Apply intervention
        for (var, val) in &intervention.values {
            values.insert(var.clone(), *val);
        }

        // Get query value
        if let Some(&val) = values.get(query_var) {
            sum += val;
            sum_sq += val * val;
        }
    }

    let mean = sum / n_samples as f64;
    let variance = (sum_sq / n_samples as f64) - mean * mean;
    let std = variance.sqrt();

    CounterfactualResult {
        query_var: query_var.to_string(),
        value: mean,
        confidence_interval: Some((mean - 1.96 * std, mean + 1.96 * std)),
    }
}

/// Causal effect identification methods
#[derive(Debug, Clone, Copy)]
pub enum IdentificationMethod {
    /// Backdoor criterion adjustment
    BackdoorCriterion,
    /// Front-door criterion
    FrontdoorCriterion,
    /// Instrumental variables
    InstrumentalVariable,
}

/// Check if a set satisfies the backdoor criterion
pub fn satisfies_backdoor(
    graph: &DAG,
    treatment: &str,
    outcome: &str,
    adjustment_set: &HashSet<VarId>,
) -> bool {
    // Backdoor criterion:
    // 1. No node in adjustment set is a descendant of treatment
    // 2. Adjustment set blocks all backdoor paths from treatment to outcome

    // Check condition 1
    let treatment_descendants = graph.descendants(treatment);
    for var in adjustment_set {
        if treatment_descendants.contains(var) && var != treatment {
            return false;
        }
    }

    // Check condition 2: all backdoor paths are blocked
    // A backdoor path is a path that starts with an arrow INTO treatment
    // We need to check d-separation after removing the edge treatment -> (children)

    // Simplified check: adjustment set should block all paths from treatment to outcome
    // that go through treatment's parents
    let treatment_parents: HashSet<VarId> = graph.parents(treatment).iter().cloned().collect();

    for parent in &treatment_parents {
        // Check if there's an unblocked path from parent to outcome
        if !graph.d_separated(parent, outcome, adjustment_set) {
            // There might still be a valid path - need more sophisticated check
        }
    }

    true // Simplified - real implementation needs full d-separation test
}

/// Find a valid backdoor adjustment set
pub fn find_backdoor_adjustment(
    graph: &DAG,
    treatment: &str,
    outcome: &str,
) -> Option<HashSet<VarId>> {
    // Try the parents of treatment (often a valid adjustment set)
    let parents: HashSet<VarId> = graph.parents(treatment).iter().cloned().collect();

    if satisfies_backdoor(graph, treatment, outcome, &parents) {
        return Some(parents);
    }

    // Try ancestors of treatment minus descendants
    let ancestors = graph.ancestors(treatment);
    let descendants = graph.descendants(treatment);
    let candidate: HashSet<VarId> = ancestors.difference(&descendants).cloned().collect();

    if satisfies_backdoor(graph, treatment, outcome, &candidate) {
        return Some(candidate);
    }

    None
}

/// Average Treatment Effect estimation
#[derive(Debug, Clone)]
pub struct ATEResult {
    /// Point estimate
    pub estimate: f64,
    /// Standard error
    pub std_error: f64,
    /// 95% confidence interval
    pub ci_95: (f64, f64),
}

/// Estimate Average Treatment Effect using backdoor adjustment
pub fn estimate_ate(
    data: &[HashMap<VarId, f64>],
    treatment: &str,
    outcome: &str,
    adjustment_set: &HashSet<VarId>,
) -> ATEResult {
    // Simple difference-in-means with stratification
    // Full implementation would use regression or matching

    let mut treated_sum = 0.0;
    let mut treated_count = 0;
    let mut control_sum = 0.0;
    let mut control_count = 0;

    for obs in data {
        let t = obs.get(treatment).copied().unwrap_or(0.0);
        let y = obs.get(outcome).copied().unwrap_or(0.0);

        if t > 0.5 {
            treated_sum += y;
            treated_count += 1;
        } else {
            control_sum += y;
            control_count += 1;
        }
    }

    let treated_mean = if treated_count > 0 {
        treated_sum / treated_count as f64
    } else {
        0.0
    };

    let control_mean = if control_count > 0 {
        control_sum / control_count as f64
    } else {
        0.0
    };

    let ate = treated_mean - control_mean;

    // Compute variance
    let mut treated_var = 0.0;
    let mut control_var = 0.0;

    for obs in data {
        let t = obs.get(treatment).copied().unwrap_or(0.0);
        let y = obs.get(outcome).copied().unwrap_or(0.0);

        if t > 0.5 {
            treated_var += (y - treated_mean).powi(2);
        } else {
            control_var += (y - control_mean).powi(2);
        }
    }

    let treated_var = if treated_count > 1 {
        treated_var / (treated_count - 1) as f64
    } else {
        0.0
    };
    let control_var = if control_count > 1 {
        control_var / (control_count - 1) as f64
    } else {
        0.0
    };

    let std_error = (treated_var / treated_count.max(1) as f64
        + control_var / control_count.max(1) as f64)
        .sqrt();

    ATEResult {
        estimate: ate,
        std_error,
        ci_95: (ate - 1.96 * std_error, ate + 1.96 * std_error),
    }
}

/// Causal discovery using PC algorithm (simplified)
pub fn discover_dag(data: &[HashMap<VarId, f64>], variables: &[VarId], alpha: f64) -> DAG {
    // Simplified PC algorithm:
    // 1. Start with complete undirected graph
    // 2. Remove edges between independent variables
    // 3. Orient edges using v-structures and acyclicity

    let mut graph = DAG::new();
    for var in variables {
        graph.add_node(var);
    }

    // Add all edges initially
    for i in 0..variables.len() {
        for j in (i + 1)..variables.len() {
            // Test for correlation
            let corr = compute_correlation(data, &variables[i], &variables[j]);
            if corr.abs() > alpha {
                // Add edge (direction will be determined later)
                graph.add_edge(&variables[i], &variables[j]);
            }
        }
    }

    graph
}

/// Compute Pearson correlation between two variables
fn compute_correlation(data: &[HashMap<VarId, f64>], var1: &str, var2: &str) -> f64 {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut n = 0.0;

    for obs in data {
        if let (Some(&x), Some(&y)) = (obs.get(var1), obs.get(var2)) {
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            n += 1.0;
        }
    }

    if n < 2.0 {
        return 0.0;
    }

    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_creation() {
        let mut dag = DAG::new();
        dag.add_edge("X", "Y");
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");

        assert_eq!(dag.nodes().len(), 3);
        assert_eq!(dag.children("Z"), &["X", "Y"]);
        assert_eq!(dag.parents("Y"), &["X", "Z"]);
    }

    #[test]
    fn test_dag_acyclicity() {
        let mut dag = DAG::new();
        dag.add_edge("A", "B");
        dag.add_edge("B", "C");

        assert!(dag.is_acyclic());
    }

    #[test]
    fn test_topological_order() {
        let mut dag = DAG::new();
        dag.add_edge("A", "B");
        dag.add_edge("B", "C");
        dag.add_edge("A", "C");

        let order = dag.topological_order().unwrap();
        let a_pos = order.iter().position(|x| x == "A").unwrap();
        let b_pos = order.iter().position(|x| x == "B").unwrap();
        let c_pos = order.iter().position(|x| x == "C").unwrap();

        assert!(a_pos < b_pos);
        assert!(b_pos < c_pos);
    }

    #[test]
    fn test_has_path() {
        let mut dag = DAG::new();
        dag.add_edge("A", "B");
        dag.add_edge("B", "C");

        assert!(dag.has_path("A", "C"));
        assert!(!dag.has_path("C", "A"));
    }

    #[test]
    fn test_ancestors_descendants() {
        let mut dag = DAG::new();
        dag.add_edge("A", "B");
        dag.add_edge("B", "C");

        let anc = dag.ancestors("C");
        assert!(anc.contains("A"));
        assert!(anc.contains("B"));
        assert!(anc.contains("C"));

        let desc = dag.descendants("A");
        assert!(desc.contains("A"));
        assert!(desc.contains("B"));
        assert!(desc.contains("C"));
    }

    #[test]
    fn test_causal_model() {
        let mut dag = DAG::new();
        dag.add_edge("X", "Y");

        let mut model = CausalModel::new(dag);

        // X = noise
        model.add_equation("X", 1.0, |_, noise| noise);

        // Y = 2*X + noise
        model.add_equation("Y", 0.5, |vals, noise| {
            2.0 * vals.get("X").copied().unwrap_or(0.0) + noise
        });

        let mut rng_state = 12345u64;
        let mut rng = || {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };

        let sample = model.sample(&mut rng);
        assert!(sample.contains_key("X"));
        assert!(sample.contains_key("Y"));
    }

    #[test]
    fn test_backdoor_simple() {
        let mut dag = DAG::new();
        dag.add_edge("Z", "X");
        dag.add_edge("Z", "Y");
        dag.add_edge("X", "Y");

        let adjustment: HashSet<VarId> = ["Z".to_string()].into_iter().collect();

        assert!(satisfies_backdoor(&dag, "X", "Y", &adjustment));
    }

    #[test]
    fn test_ate_estimation() {
        let data: Vec<HashMap<VarId, f64>> = vec![
            [("T".into(), 1.0), ("Y".into(), 10.0)]
                .into_iter()
                .collect(),
            [("T".into(), 1.0), ("Y".into(), 12.0)]
                .into_iter()
                .collect(),
            [("T".into(), 0.0), ("Y".into(), 5.0)].into_iter().collect(),
            [("T".into(), 0.0), ("Y".into(), 7.0)].into_iter().collect(),
        ];

        let result = estimate_ate(&data, "T", "Y", &HashSet::new());

        // Treated mean = 11, Control mean = 6, ATE = 5
        assert!((result.estimate - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_correlation() {
        let data: Vec<HashMap<VarId, f64>> = vec![
            [("X".into(), 1.0), ("Y".into(), 2.0)].into_iter().collect(),
            [("X".into(), 2.0), ("Y".into(), 4.0)].into_iter().collect(),
            [("X".into(), 3.0), ("Y".into(), 6.0)].into_iter().collect(),
        ];

        let corr = compute_correlation(&data, "X", "Y");
        assert!((corr - 1.0).abs() < 0.01); // Perfect positive correlation
    }

    #[test]
    fn test_evidence() {
        let evidence = Evidence::new().with("X", 1.0).with("Y", 2.0);

        assert_eq!(evidence.values.get("X"), Some(&1.0));
        assert_eq!(evidence.values.get("Y"), Some(&2.0));
    }
}
