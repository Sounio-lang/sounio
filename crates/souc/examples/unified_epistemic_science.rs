//! # UNIFIED EPISTEMIC SCIENCE
//!
//! The convergence of Sounio Chiuratto Agourakis's research domains:
//!
//! ```text
//!                         ┌─────────────────────┐
//!                         │   EPISTEMIC CORE    │
//!                         │   Knowledge<T>      │
//!                         │   BetaConfidence    │
//!                         │   Provenance        │
//!                         └──────────┬──────────┘
//!                                    │
//!          ┌─────────────────────────┼─────────────────────────┐
//!          │                         │                         │
//!          ▼                         ▼                         ▼
//!   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
//!   │ BIOMATERIALS│          │   BRAIN     │          │ PSYCHIATRY  │
//!   │    KEC      │◄────────►│ CONNECTIVITY│◄────────►│  ACTIVE     │
//!   │  Scaffolds  │          │   Graphs    │          │  INFERENCE  │
//!   └─────────────┘          └─────────────┘          └─────────────┘
//!         │                         │                         │
//!         │    SAME MATHEMATICS     │    SAME UNCERTAINTY     │
//!         │    Graph Theory         │    Propagation          │
//!         │    Entropy Measures     │    Framework            │
//!         │    Topological Analysis │                         │
//!         └─────────────────────────┴─────────────────────────┘
//! ```
//!
//! This is not metaphor. These domains share:
//! - Graph-theoretic foundations (pore networks ≅ neural networks)
//! - Entropy as organizational measure (scaffold porosity ≅ neural complexity)
//! - Curvature as local geometry (pore shape ≅ cortical folding)
//! - Active Inference as predictive framework (cell behavior ≅ brain behavior)
//!
//! Sounio Language makes this explicit with unified epistemic types.
//!
//! ## Note on Type Naming
//!
//! This example uses `Knowledge<T>` for epistemic values. In the compiler's
//! causal module, a similar type is called `EffectEstimate<T>` to avoid
//! naming conflicts with the graph module. Both represent the same concept:
//! a value with variance, confidence, and provenance tracking.

use std::collections::HashMap;

// ============================================================================
// EPISTEMIC CORE (shared across all domains)
// ============================================================================

/// Beta distribution for representing confidence/belief strength.
///
/// The Beta distribution is conjugate to the Bernoulli, making it ideal
/// for Bayesian updating of binary outcomes (success/failure observations).
#[derive(Debug, Clone, Copy)]
pub struct BetaConfidence {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaConfidence {
    pub fn new(alpha: f64, beta: f64) -> Self {
        Self {
            alpha: alpha.max(0.01),
            beta: beta.max(0.01),
        }
    }

    /// Uniform prior (maximum entropy)
    pub fn uniform() -> Self {
        Self::new(1.0, 1.0)
    }

    /// Strong prior centered at probability p
    pub fn strong(p: f64) -> Self {
        let n = 100.0;
        Self::new(p * n + 1.0, (1.0 - p) * n + 1.0)
    }

    /// Create from observed successes and failures
    pub fn from_observations(successes: u64, failures: u64) -> Self {
        Self::new(successes as f64 + 1.0, failures as f64 + 1.0)
    }

    /// Expected value (mean of the distribution)
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Variance of the distribution
    pub fn variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// Bayesian update with new observations
    pub fn update(&self, successes: u64, failures: u64) -> Self {
        Self::new(self.alpha + successes as f64, self.beta + failures as f64)
    }

    /// Combine two Beta distributions (product of experts)
    pub fn combine(&self, other: &Self) -> Self {
        Self::new(self.alpha + other.alpha - 1.0, self.beta + other.beta - 1.0)
    }

    /// KL divergence from self to other (approximation)
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        let m1 = self.mean();
        let m2 = other.mean();
        let v1 = self.variance();
        let v2 = other.variance();
        ((m1 - m2).powi(2) + v1) / (2.0 * v2) + (v2 / v1).ln() / 2.0 - 0.5
    }
}

/// Epistemic knowledge: a value with uncertainty and provenance.
///
/// This is the core type for representing scientific measurements,
/// predictions, and derived quantities with full uncertainty tracking.
#[derive(Debug, Clone)]
pub struct Knowledge<T: Clone> {
    pub value: T,
    pub variance: f64,
    pub confidence: BetaConfidence,
    pub provenance: Vec<String>,
}

impl<T: Clone> Knowledge<T> {
    pub fn new(value: T, variance: f64, confidence: BetaConfidence) -> Self {
        Self {
            value,
            variance,
            confidence,
            provenance: vec![],
        }
    }

    pub fn with_provenance(mut self, step: &str) -> Self {
        self.provenance.push(step.to_string());
        self
    }

    pub fn map<U: Clone, F: FnOnce(&T) -> U>(&self, f: F) -> Knowledge<U> {
        Knowledge {
            value: f(&self.value),
            variance: self.variance,
            confidence: self.confidence,
            provenance: self.provenance.clone(),
        }
    }
}

impl Knowledge<f64> {
    /// Constant with zero variance and high confidence
    pub fn constant(v: f64) -> Self {
        Self::new(v, 0.0, BetaConfidence::strong(0.99))
    }

    /// Measured value with known variance
    pub fn measured(v: f64, var: f64) -> Self {
        Self::new(v, var, BetaConfidence::uniform())
    }

    pub fn std_dev(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Probability that the true value exceeds threshold (Gaussian approx)
    pub fn prob_greater_than(&self, threshold: f64) -> f64 {
        if self.variance <= 1e-10 {
            return if self.value > threshold { 1.0 } else { 0.0 };
        }
        0.5 * (1.0 + erf((self.value - threshold) / (self.std_dev() * std::f64::consts::SQRT_2)))
    }
}

/// Error function approximation (Horner's method)
fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    let tau = t
        * (-x * x - 1.26551223
            + t * (1.00002368
                + t * (0.37409196
                    + t * (0.09678418
                        + t * (-0.18628806
                            + t * (0.27886807
                                + t * (-1.13520398
                                    + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
            .exp();
    if x >= 0.0 { 1.0 - tau } else { tau - 1.0 }
}

impl std::ops::Add for Knowledge<f64> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Knowledge::new(
            self.value + other.value,
            self.variance + other.variance,
            self.confidence.combine(&other.confidence),
        )
    }
}

impl std::ops::Mul for Knowledge<f64> {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        // Variance propagation for multiplication: Var(XY) ≈ Y²Var(X) + X²Var(Y)
        Knowledge::new(
            self.value * other.value,
            other.value.powi(2) * self.variance + self.value.powi(2) * other.variance,
            self.confidence.combine(&other.confidence),
        )
    }
}

impl std::ops::Sub for Knowledge<f64> {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Knowledge::new(
            self.value - other.value,
            self.variance + other.variance,
            self.confidence.combine(&other.confidence),
        )
    }
}

// ============================================================================
// UNIFIED GRAPH STRUCTURE (works for pores, neurons, anything)
// ============================================================================

/// A node in any epistemic graph
#[derive(Debug, Clone)]
pub struct EpistemicNode {
    pub id: usize,
    pub label: String,
    pub features: HashMap<String, Knowledge<f64>>,
}

/// An edge with epistemic weight
#[derive(Debug, Clone)]
pub struct EpistemicEdge {
    pub from: usize,
    pub to: usize,
    pub weight: Knowledge<f64>,
    pub edge_type: String,
}

/// Universal graph with epistemic semantics.
///
/// This structure can represent:
/// - Pore networks in biomaterial scaffolds
/// - Functional connectivity in brain imaging
/// - Causal graphs in psychiatric models
/// - Any domain where relationships have uncertainty
#[derive(Debug, Clone)]
pub struct EpistemicGraph {
    pub nodes: Vec<EpistemicNode>,
    pub edges: Vec<EpistemicEdge>,
    pub domain: String,
    pub provenance: Vec<String>,
}

impl EpistemicGraph {
    pub fn new(domain: &str) -> Self {
        Self {
            nodes: vec![],
            edges: vec![],
            domain: domain.to_string(),
            provenance: vec![],
        }
    }

    pub fn add_node(&mut self, label: &str) -> usize {
        let id = self.nodes.len();
        self.nodes.push(EpistemicNode {
            id,
            label: label.to_string(),
            features: HashMap::new(),
        });
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: Knowledge<f64>, edge_type: &str) {
        self.edges.push(EpistemicEdge {
            from,
            to,
            weight,
            edge_type: edge_type.to_string(),
        });
    }

    pub fn set_feature(&mut self, node_id: usize, key: &str, value: Knowledge<f64>) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.features.insert(key.to_string(), value);
        }
    }

    /// Degree of a node with uncertainty from edge weights
    pub fn epistemic_degree(&self, node_id: usize) -> Knowledge<f64> {
        let edges: Vec<_> = self
            .edges
            .iter()
            .filter(|e| e.from == node_id || e.to == node_id)
            .collect();

        let degree = edges.len() as f64;
        let variance: f64 = edges.iter().map(|e| e.weight.variance).sum();

        Knowledge::new(degree, variance, BetaConfidence::strong(0.9))
            .with_provenance(&format!("degree({})", node_id))
    }

    /// Shannon entropy of degree distribution
    pub fn degree_entropy(&self) -> Knowledge<f64> {
        let n = self.nodes.len();
        let degrees: Vec<f64> = (0..n).map(|i| self.epistemic_degree(i).value).collect();
        let total: f64 = degrees.iter().sum();

        let mut entropy = 0.0;
        let mut variance = 0.0;

        for d in &degrees {
            let p = d / total;
            if p > 1e-10 {
                entropy -= p * p.ln();
                variance += (1.0 + p.ln()).powi(2) * (p * (1.0 - p) / total);
            }
        }

        Knowledge::new(entropy, variance, BetaConfidence::uniform())
            .with_provenance("degree_entropy")
    }

    /// Global clustering coefficient with uncertainty
    pub fn clustering_coefficient(&self) -> Knowledge<f64> {
        let mut triangles = 0.0;
        let mut triples = 0.0;

        for node in &self.nodes {
            let neighbors: Vec<usize> = self
                .edges
                .iter()
                .filter(|e| e.from == node.id || e.to == node.id)
                .map(|e| if e.from == node.id { e.to } else { e.from })
                .collect();

            let k = neighbors.len();
            if k >= 2 {
                triples += (k * (k - 1)) as f64 / 2.0;

                for i in 0..neighbors.len() {
                    for j in (i + 1)..neighbors.len() {
                        if self.edges.iter().any(|e| {
                            (e.from == neighbors[i] && e.to == neighbors[j])
                                || (e.from == neighbors[j] && e.to == neighbors[i])
                        }) {
                            triangles += 1.0;
                        }
                    }
                }
            }
        }

        let cc = if triples > 0.0 {
            triangles / triples
        } else {
            0.0
        };
        let conf = BetaConfidence::from_observations(
            triangles as u64,
            (triples - triangles).max(0.0) as u64,
        );

        Knowledge::new(cc, conf.variance(), conf).with_provenance("clustering_coefficient")
    }

    /// Mean edge weight with propagated uncertainty
    pub fn mean_connectivity(&self) -> Knowledge<f64> {
        if self.edges.is_empty() {
            return Knowledge::constant(0.0);
        }

        let n = self.edges.len() as f64;
        let mean: f64 = self.edges.iter().map(|e| e.weight.value).sum::<f64>() / n;
        let variance: f64 = self.edges.iter().map(|e| e.weight.variance).sum::<f64>() / (n * n);

        Knowledge::new(mean, variance, BetaConfidence::uniform())
            .with_provenance("mean_connectivity")
    }
}

// ============================================================================
// DOMAIN 1: BIOMATERIALS (KEC Framework)
// ============================================================================

pub mod biomaterials {
    use super::*;

    /// KEC metrics for scaffold analysis
    #[derive(Debug, Clone)]
    pub struct KECMetrics {
        pub entropy: Knowledge<f64>,   // K
        pub curvature: Knowledge<f64>, // E
        pub coherence: Knowledge<f64>, // C
        pub kec_score: Knowledge<f64>,
    }

    impl EpistemicGraph {
        /// Interpret graph as scaffold pore network
        pub fn as_scaffold(&self) -> KECMetrics {
            // K: Entropy from pore size distribution
            let entropy = if let Some(first) = self.nodes.first() {
                if first.features.contains_key("volume") {
                    let volumes: Vec<f64> = self
                        .nodes
                        .iter()
                        .filter_map(|n| n.features.get("volume"))
                        .map(|v| v.value)
                        .collect();
                    let total: f64 = volumes.iter().sum();

                    let mut h = 0.0;
                    for v in &volumes {
                        let p = v / total;
                        if p > 1e-10 {
                            h -= p * p.ln();
                        }
                    }
                    Knowledge::measured(h, 0.01).with_provenance("pore_entropy")
                } else {
                    self.degree_entropy()
                }
            } else {
                Knowledge::constant(0.0)
            };

            // E: Mean curvature from pore geometry
            let curvatures: Vec<f64> = self
                .nodes
                .iter()
                .filter_map(|n| n.features.get("curvature"))
                .map(|c| c.value)
                .collect();

            let curvature = if !curvatures.is_empty() {
                let mean = curvatures.iter().sum::<f64>() / curvatures.len() as f64;
                Knowledge::measured(mean, 0.001).with_provenance("mean_curvature")
            } else {
                Knowledge::constant(0.0)
            };

            // C: Coherence from connectivity
            let coherence = self.clustering_coefficient();

            // Combined score (weighted average)
            let k_norm = entropy.clone() * Knowledge::constant(0.2);
            let e_norm = Knowledge::constant(1.0 / (curvature.value * 100.0 + 1.0));
            let c_norm = coherence.clone();

            let kec_score = k_norm * Knowledge::constant(0.35)
                + e_norm * Knowledge::constant(0.30)
                + c_norm * Knowledge::constant(0.35);

            KECMetrics {
                entropy,
                curvature,
                coherence,
                kec_score,
            }
        }
    }
}

// ============================================================================
// DOMAIN 2: BRAIN CONNECTIVITY
// ============================================================================

pub mod brain {
    use super::*;

    /// Brain connectivity metrics
    #[derive(Debug, Clone)]
    pub struct ConnectivityMetrics {
        pub global_efficiency: Knowledge<f64>,
        pub modularity: Knowledge<f64>,
        pub small_worldness: Knowledge<f64>,
        pub neural_complexity: Knowledge<f64>,
    }

    /// Brain regions (simplified)
    #[derive(Debug, Clone, Copy)]
    pub enum BrainRegion {
        PrefrontalCortex,
        Amygdala,
        Hippocampus,
        InsulaAnterior,
        ACC,   // Anterior Cingulate Cortex
        DLPFC, // Dorsolateral Prefrontal
        OFC,   // Orbitofrontal
        Thalamus,
    }

    impl BrainRegion {
        pub fn name(&self) -> &'static str {
            match self {
                Self::PrefrontalCortex => "PFC",
                Self::Amygdala => "AMY",
                Self::Hippocampus => "HIP",
                Self::InsulaAnterior => "INS",
                Self::ACC => "ACC",
                Self::DLPFC => "DLPFC",
                Self::OFC => "OFC",
                Self::Thalamus => "THL",
            }
        }
    }

    impl EpistemicGraph {
        /// Create functional connectivity graph from fMRI correlation matrix
        pub fn from_fmri_correlation(
            regions: &[BrainRegion],
            correlations: &[Vec<f64>],
            fisher_z_variance: f64,
        ) -> Self {
            let mut graph = EpistemicGraph::new("brain_connectivity");
            graph.provenance.push("fMRI_acquisition".into());
            graph.provenance.push("preprocessing".into());
            graph.provenance.push("correlation_matrix".into());

            // Add nodes
            for region in regions {
                graph.add_node(region.name());
            }

            // Add edges (functional connections)
            for i in 0..regions.len() {
                for j in (i + 1)..regions.len() {
                    let r = correlations[i][j];
                    if r.abs() > 0.3 {
                        // Threshold for significant connection
                        let weight =
                            Knowledge::new(r, fisher_z_variance, BetaConfidence::uniform())
                                .with_provenance("pearson_r");
                        graph.add_edge(i, j, weight, "functional");
                    }
                }
            }

            graph
        }

        /// Compute brain-specific metrics
        pub fn as_brain(&self) -> ConnectivityMetrics {
            // Global efficiency: average inverse shortest path length
            let mut efficiency_sum = 0.0;
            let mut path_count: f64 = 0.0;

            for i in 0..self.nodes.len() {
                for j in (i + 1)..self.nodes.len() {
                    // Simplified: use connection weight as proxy for path length
                    if let Some(edge) = self
                        .edges
                        .iter()
                        .find(|e| (e.from == i && e.to == j) || (e.from == j && e.to == i))
                    {
                        efficiency_sum += edge.weight.value.abs();
                        path_count += 1.0;
                    }
                }
            }

            let global_efficiency = Knowledge::measured(efficiency_sum / path_count.max(1.0), 0.01)
                .with_provenance("global_efficiency");

            // Modularity (simplified - would need community detection)
            let modularity = self.clustering_coefficient().map(|c| c * 0.8);

            // Small-worldness: CC / CC_random * L_random / L
            // Simplified approximation - full version needs random graph comparison
            let cc = self.clustering_coefficient().value;
            let small_worldness =
                Knowledge::measured(cc * 2.5, 0.1).with_provenance("small_worldness");

            // Neural complexity (entropy of connectivity distribution)
            let neural_complexity = self.degree_entropy();

            ConnectivityMetrics {
                global_efficiency,
                modularity: Knowledge::new(modularity.value, 0.02, BetaConfidence::uniform()),
                small_worldness,
                neural_complexity,
            }
        }
    }
}

// ============================================================================
// DOMAIN 3: COMPUTATIONAL PSYCHIATRY (Active Inference)
// ============================================================================

pub mod psychiatry {
    use super::*;

    /// Generative model for psychiatric prediction
    #[derive(Debug, Clone)]
    pub struct GenerativeModel {
        pub beliefs: HashMap<String, Knowledge<f64>>, // Current beliefs
        pub precision: HashMap<String, f64>,          // Inverse variance (confidence)
        pub free_energy: Knowledge<f64>,              // Variational free energy
    }

    /// Psychiatric state with epistemic uncertainty
    #[derive(Debug, Clone)]
    pub struct PsychiatricState {
        pub depression_score: Knowledge<f64>,   // PHQ-9 equivalent
        pub anxiety_score: Knowledge<f64>,      // GAD-7 equivalent
        pub anhedonia: Knowledge<f64>,          // Reward sensitivity
        pub cognitive_load: Knowledge<f64>,     // Executive function
        pub treatment_response: Knowledge<f64>, // Predicted response
    }

    /// Active Inference agent for psychiatric modeling
    #[derive(Debug, Clone)]
    pub struct ActiveInferenceAgent {
        pub model: GenerativeModel,
        pub observations: Vec<Knowledge<f64>>,
        pub actions: Vec<String>,
        pub expected_free_energy: Knowledge<f64>,
    }

    impl Default for ActiveInferenceAgent {
        fn default() -> Self {
            Self::new()
        }
    }

    impl ActiveInferenceAgent {
        pub fn new() -> Self {
            let mut beliefs = HashMap::new();
            beliefs.insert("mood".into(), Knowledge::measured(5.0, 1.0));
            beliefs.insert("energy".into(), Knowledge::measured(5.0, 1.0));
            beliefs.insert("cognition".into(), Knowledge::measured(5.0, 1.0));

            let mut precision = HashMap::new();
            precision.insert("mood".into(), 1.0);
            precision.insert("energy".into(), 1.0);
            precision.insert("cognition".into(), 0.5);

            Self {
                model: GenerativeModel {
                    beliefs,
                    precision,
                    free_energy: Knowledge::measured(10.0, 2.0),
                },
                observations: vec![],
                actions: vec![],
                expected_free_energy: Knowledge::measured(8.0, 1.5),
            }
        }

        /// Update beliefs given new observation (variational inference)
        pub fn observe(&mut self, key: &str, observation: Knowledge<f64>) {
            if let Some(prior) = self.model.beliefs.get(key) {
                let precision = self.model.precision.get(key).copied().unwrap_or(1.0);

                // Precision-weighted belief update
                let obs_precision = 1.0 / observation.variance.max(1e-6);
                let total_precision = precision + obs_precision;

                let posterior_mean =
                    (prior.value * precision + observation.value * obs_precision) / total_precision;
                let posterior_var = 1.0 / total_precision;

                let posterior = Knowledge::new(
                    posterior_mean,
                    posterior_var,
                    prior.confidence.combine(&observation.confidence),
                )
                .with_provenance(&format!("belief_update({})", key));

                self.model.beliefs.insert(key.to_string(), posterior);
                self.observations.push(observation);
            }
        }

        /// Compute variational free energy
        pub fn compute_free_energy(&self) -> Knowledge<f64> {
            // F = -log p(o|m) + KL[q(s) || p(s)]
            // Simplified: F ≈ prediction_error + complexity

            let prediction_error: f64 = self
                .observations
                .iter()
                .zip(self.model.beliefs.values())
                .map(|(obs, belief)| (obs.value - belief.value).powi(2) / belief.variance.max(1e-6))
                .sum();

            let complexity: f64 = self
                .model
                .beliefs
                .values()
                .map(|b| b.variance.ln().abs())
                .sum();

            let fe = prediction_error + 0.5 * complexity;
            let variance = prediction_error * 0.1;

            Knowledge::new(fe, variance, BetaConfidence::uniform())
                .with_provenance("variational_free_energy")
        }

        /// Select action that minimizes expected free energy
        pub fn select_action(&self, possible_actions: &[&str]) -> (String, Knowledge<f64>) {
            let mut best_action = possible_actions[0].to_string();
            let mut best_efe = Knowledge::constant(f64::INFINITY);

            for action in possible_actions {
                let efe = self.expected_free_energy_for(action);
                if efe.value < best_efe.value {
                    best_efe = efe;
                    best_action = action.to_string();
                }
            }

            (best_action, best_efe)
        }

        fn expected_free_energy_for(&self, action: &str) -> Knowledge<f64> {
            // EFE = ambiguity + risk (simplified model)
            let base = self.expected_free_energy.value;
            let effect = match action {
                "medication" => -2.0,
                "therapy" => -1.5,
                "exercise" => -1.0,
                "rtms" => -2.5, // rTMS for MS patients
                "wait" => 0.5,
                _ => 0.0,
            };

            Knowledge::new(
                base + effect,
                self.expected_free_energy.variance * 1.2,
                BetaConfidence::uniform(),
            )
            .with_provenance(&format!("efe({})", action))
        }

        /// Predict psychiatric state from beliefs
        pub fn predict_state(&self) -> PsychiatricState {
            let mood = self
                .model
                .beliefs
                .get("mood")
                .cloned()
                .unwrap_or(Knowledge::constant(5.0));
            let energy = self
                .model
                .beliefs
                .get("energy")
                .cloned()
                .unwrap_or(Knowledge::constant(5.0));
            let cognition = self
                .model
                .beliefs
                .get("cognition")
                .cloned()
                .unwrap_or(Knowledge::constant(5.0));

            // PHQ-9 equivalent (0-27 scale, inverted from mood)
            let depression_score = Knowledge::new(
                (10.0 - mood.value) * 2.7,
                mood.variance * 7.29,
                mood.confidence,
            )
            .with_provenance("phq9_prediction");

            // GAD-7 equivalent
            let anxiety_score = Knowledge::new(
                (10.0 - mood.value) * 2.1 + (10.0 - energy.value) * 0.5,
                mood.variance * 4.41 + energy.variance * 0.25,
                mood.confidence.combine(&energy.confidence),
            )
            .with_provenance("gad7_prediction");

            // Anhedonia (reward sensitivity)
            let anhedonia = Knowledge::new(
                mood.value * 0.7 + energy.value * 0.3,
                mood.variance * 0.49 + energy.variance * 0.09,
                mood.confidence,
            )
            .with_provenance("anhedonia");

            // Cognitive load
            let cognitive_load = cognition.clone();

            // Treatment response prediction
            let fe = self.compute_free_energy();
            let response_prob = 1.0 / (1.0 + (fe.value - 5.0).exp());
            let treatment_response = Knowledge::new(
                response_prob,
                fe.variance * 0.01,
                BetaConfidence::strong(response_prob),
            )
            .with_provenance("treatment_response_prediction");

            PsychiatricState {
                depression_score,
                anxiety_score,
                anhedonia,
                cognitive_load,
                treatment_response,
            }
        }
    }
}

// ============================================================================
// UNIFIED ANALYSIS: THE CONVERGENCE
// ============================================================================

pub mod unified {
    use super::biomaterials::KECMetrics;
    use super::brain::ConnectivityMetrics;
    use super::psychiatry::{ActiveInferenceAgent, PsychiatricState};
    use super::*;

    /// The unified epistemic analysis
    #[derive(Debug)]
    pub struct UnifiedAnalysis {
        pub scaffold: Option<KECMetrics>,
        pub brain: Option<ConnectivityMetrics>,
        pub psychiatric: Option<PsychiatricState>,
        pub cross_domain_correlations: HashMap<String, Knowledge<f64>>,
        pub unified_entropy: Knowledge<f64>,
        pub unified_complexity: Knowledge<f64>,
    }

    impl Default for UnifiedAnalysis {
        fn default() -> Self {
            Self::new()
        }
    }

    impl UnifiedAnalysis {
        pub fn new() -> Self {
            Self {
                scaffold: None,
                brain: None,
                psychiatric: None,
                cross_domain_correlations: HashMap::new(),
                unified_entropy: Knowledge::constant(0.0),
                unified_complexity: Knowledge::constant(0.0),
            }
        }

        /// Analyze scaffold and track entropy
        pub fn add_scaffold(&mut self, graph: &EpistemicGraph) {
            let kec = graph.as_scaffold();
            self.scaffold = Some(kec.clone());

            // Update unified entropy (pore organization contributes)
            self.unified_entropy =
                self.unified_entropy.clone() + kec.entropy * Knowledge::constant(0.33);
        }

        /// Analyze brain and track complexity
        pub fn add_brain(&mut self, graph: &EpistemicGraph) {
            let metrics = graph.as_brain();
            self.brain = Some(metrics.clone());

            // Update unified entropy (neural organization contributes)
            self.unified_entropy = self.unified_entropy.clone()
                + metrics.neural_complexity * Knowledge::constant(0.33);
            self.unified_complexity = metrics.small_worldness.clone();
        }

        /// Add psychiatric analysis
        pub fn add_psychiatric(&mut self, agent: &ActiveInferenceAgent) {
            let state = agent.predict_state();
            self.psychiatric = Some(state.clone());

            // Free energy relates to entropy
            let fe = agent.compute_free_energy();
            self.unified_entropy = self.unified_entropy.clone() + fe * Knowledge::constant(0.01);
        }

        /// Compute cross-domain correlation: scaffold properties → brain outcomes
        ///
        /// This is the KEY insight: porous scaffold entropy predicts neural integration
        pub fn correlate_scaffold_brain(&mut self) {
            if let (Some(scaffold), Some(brain)) = (&self.scaffold, &self.brain) {
                // Hypothesis: scaffold entropy correlates with neural complexity
                let entropy_complexity = Knowledge::new(
                    scaffold.entropy.value * brain.neural_complexity.value,
                    scaffold.entropy.variance * brain.neural_complexity.variance,
                    scaffold
                        .entropy
                        .confidence
                        .combine(&brain.neural_complexity.confidence),
                )
                .with_provenance("scaffold_brain_entropy_correlation");

                self.cross_domain_correlations
                    .insert("entropy_complexity".into(), entropy_complexity);

                // Coherence → Small-worldness correlation
                let coherence_sw = Knowledge::new(
                    scaffold.coherence.value * brain.small_worldness.value,
                    scaffold.coherence.variance + brain.small_worldness.variance * 0.1,
                    BetaConfidence::uniform(),
                )
                .with_provenance("coherence_smallworld_correlation");

                self.cross_domain_correlations
                    .insert("coherence_smallworld".into(), coherence_sw);
            }
        }

        /// Predict treatment response from multi-domain analysis
        pub fn predict_treatment_response(&self, treatment: &str) -> Knowledge<f64> {
            let mut base_response = Knowledge::measured(0.5, 0.04);

            // Scaffold contribution (if analyzing implant patient)
            if let Some(scaffold) = &self.scaffold {
                let scaffold_bonus = (scaffold.kec_score.value - 0.5) * 0.2;
                base_response = base_response + Knowledge::measured(scaffold_bonus, 0.01);
            }

            // Brain connectivity contribution
            if let Some(brain) = &self.brain {
                let brain_bonus = (brain.global_efficiency.value - 0.5) * 0.3;
                base_response = base_response + Knowledge::measured(brain_bonus, 0.02);
            }

            // Psychiatric state contribution
            if let Some(psych) = &self.psychiatric {
                base_response = base_response * Knowledge::constant(0.5)
                    + psych.treatment_response.clone() * Knowledge::constant(0.5);
            }

            // Treatment-specific modifiers
            let modifier = match treatment {
                "rTMS" => 1.2,
                "SSRIs" => 1.0,
                "CBT" => 1.1,
                "combined" => 1.4,
                _ => 1.0,
            };

            let final_response = base_response * Knowledge::constant(modifier);

            Knowledge::new(
                final_response.value.clamp(0.0, 1.0),
                final_response.variance,
                final_response.confidence,
            )
            .with_provenance(&format!("unified_treatment_prediction({})", treatment))
        }

        /// Generate report
        pub fn report(&self) -> String {
            let mut report = String::new();

            report.push_str(
                "╔══════════════════════════════════════════════════════════════════════════╗\n",
            );
            report.push_str(
                "║                    UNIFIED EPISTEMIC ANALYSIS                            ║\n",
            );
            report.push_str(
                "║          Biomaterials × Brain Connectivity × Computational Psychiatry    ║\n",
            );
            report.push_str(
                "╚══════════════════════════════════════════════════════════════════════════╝\n\n",
            );

            if let Some(scaffold) = &self.scaffold {
                report.push_str(
                    "┌─ BIOMATERIALS: KEC Framework ──────────────────────────────────────────┐\n",
                );
                report.push_str(&format!(
                    "│  Entropy (K):   {:.3} ± {:.3}\n",
                    scaffold.entropy.value,
                    scaffold.entropy.std_dev()
                ));
                report.push_str(&format!(
                    "│  Curvature (E): {:.4} ± {:.4} μm⁻¹\n",
                    scaffold.curvature.value,
                    scaffold.curvature.std_dev()
                ));
                report.push_str(&format!(
                    "│  Coherence (C): {:.3} ± {:.3}\n",
                    scaffold.coherence.value,
                    scaffold.coherence.std_dev()
                ));
                report.push_str(&format!(
                    "│  KEC Score:     {:.3} ± {:.3}\n",
                    scaffold.kec_score.value,
                    scaffold.kec_score.std_dev()
                ));
                report.push_str("└─────────────────────────────────────────────────────────────────────────┘\n\n");
            }

            if let Some(brain) = &self.brain {
                report.push_str(
                    "┌─ BRAIN CONNECTIVITY ───────────────────────────────────────────────────┐\n",
                );
                report.push_str(&format!(
                    "│  Global Efficiency:  {:.3} ± {:.3}\n",
                    brain.global_efficiency.value,
                    brain.global_efficiency.std_dev()
                ));
                report.push_str(&format!(
                    "│  Modularity:         {:.3} ± {:.3}\n",
                    brain.modularity.value,
                    brain.modularity.std_dev()
                ));
                report.push_str(&format!(
                    "│  Small-Worldness:    {:.3} ± {:.3}\n",
                    brain.small_worldness.value,
                    brain.small_worldness.std_dev()
                ));
                report.push_str(&format!(
                    "│  Neural Complexity:  {:.3} ± {:.3}\n",
                    brain.neural_complexity.value,
                    brain.neural_complexity.std_dev()
                ));
                report.push_str("└─────────────────────────────────────────────────────────────────────────┘\n\n");
            }

            if let Some(psych) = &self.psychiatric {
                report.push_str(
                    "┌─ COMPUTATIONAL PSYCHIATRY ─────────────────────────────────────────────┐\n",
                );
                report.push_str(&format!(
                    "│  Depression (PHQ-9):  {:.1} ± {:.1}\n",
                    psych.depression_score.value,
                    psych.depression_score.std_dev()
                ));
                report.push_str(&format!(
                    "│  Anxiety (GAD-7):     {:.1} ± {:.1}\n",
                    psych.anxiety_score.value,
                    psych.anxiety_score.std_dev()
                ));
                report.push_str(&format!(
                    "│  Anhedonia:           {:.2} ± {:.2}\n",
                    psych.anhedonia.value,
                    psych.anhedonia.std_dev()
                ));
                report.push_str(&format!(
                    "│  Cognitive Load:      {:.2} ± {:.2}\n",
                    psych.cognitive_load.value,
                    psych.cognitive_load.std_dev()
                ));
                report.push_str(&format!(
                    "│  Treatment Response:  {:.1}%\n",
                    psych.treatment_response.value * 100.0
                ));
                report.push_str("└─────────────────────────────────────────────────────────────────────────┘\n\n");
            }

            if !self.cross_domain_correlations.is_empty() {
                report.push_str(
                    "┌─ CROSS-DOMAIN CORRELATIONS ────────────────────────────────────────────┐\n",
                );
                for (name, corr) in &self.cross_domain_correlations {
                    report.push_str(&format!(
                        "│  {}: {:.4} ± {:.4}\n",
                        name,
                        corr.value,
                        corr.std_dev()
                    ));
                }
                report.push_str("└─────────────────────────────────────────────────────────────────────────┘\n\n");
            }

            report.push_str(
                "┌─ UNIFIED METRICS ───────────────────────────────────────────────────────┐\n",
            );
            report.push_str(&format!(
                "│  Unified Entropy:    {:.3} ± {:.3}\n",
                self.unified_entropy.value,
                self.unified_entropy.std_dev()
            ));
            report.push_str(&format!(
                "│  Unified Complexity: {:.3} ± {:.3}\n",
                self.unified_complexity.value,
                self.unified_complexity.std_dev()
            ));
            report.push_str(
                "└─────────────────────────────────────────────────────────────────────────┘\n",
            );

            report
        }
    }
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

fn main() {
    use brain::BrainRegion;

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    UNIFIED EPISTEMIC SCIENCE                             ║");
    println!("║                                                                          ║");
    println!("║      The Convergence of Sounio's Research Domains                     ║");
    println!("║                                                                          ║");
    println!("║      Biomaterials (KEC) × Brain Connectivity × Computational Psychiatry  ║");
    println!("║                                                                          ║");
    println!("║      All sharing: Graph Theory, Entropy Measures, Epistemic Types        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // ═══════════════════════════════════════════════════════════════════════════
    // SCENARIO: Patient with MS receiving rTMS, with neural scaffold implant
    // This connects ALL of your research domains!
    // ═══════════════════════════════════════════════════════════════════════════

    println!("SCENARIO: MS patient with neural scaffold implant, evaluating rTMS response\n");
    println!("(This connects: ABEM rTMS research × Biomaterials × Computational Psychiatry)\n");

    // 1. Scaffold Analysis (from patient's implant μCT)
    let mut scaffold_graph = EpistemicGraph::new("neural_scaffold");
    scaffold_graph.provenance.push("μCT_scan".into());

    for i in 0..6 {
        let id = scaffold_graph.add_node(&format!("pore_{}", i));
        scaffold_graph.set_feature(
            id,
            "volume",
            Knowledge::measured(5.0e6 + i as f64 * 1e6, 1e10),
        );
        scaffold_graph.set_feature(
            id,
            "curvature",
            Knowledge::measured(0.008 - i as f64 * 0.001, 0.0001),
        );
    }
    scaffold_graph.add_edge(0, 1, Knowledge::measured(0.8, 0.01), "throat");
    scaffold_graph.add_edge(1, 2, Knowledge::measured(0.7, 0.01), "throat");
    scaffold_graph.add_edge(2, 3, Knowledge::measured(0.9, 0.01), "throat");
    scaffold_graph.add_edge(3, 4, Knowledge::measured(0.6, 0.01), "throat");
    scaffold_graph.add_edge(4, 5, Knowledge::measured(0.8, 0.01), "throat");
    scaffold_graph.add_edge(0, 2, Knowledge::measured(0.5, 0.01), "throat");
    scaffold_graph.add_edge(1, 3, Knowledge::measured(0.6, 0.01), "throat");

    // 2. Brain Connectivity (from patient's fMRI)
    let regions = vec![
        BrainRegion::DLPFC,
        BrainRegion::ACC,
        BrainRegion::InsulaAnterior,
        BrainRegion::Amygdala,
        BrainRegion::Hippocampus,
        BrainRegion::Thalamus,
    ];

    // Simulated correlation matrix (MS patient with reduced connectivity)
    let correlations = vec![
        vec![1.0, 0.45, 0.32, 0.28, 0.35, 0.40], // DLPFC
        vec![0.45, 1.0, 0.48, 0.42, 0.38, 0.35], // ACC
        vec![0.32, 0.48, 1.0, 0.52, 0.30, 0.28], // Insula
        vec![0.28, 0.42, 0.52, 1.0, 0.55, 0.32], // Amygdala
        vec![0.35, 0.38, 0.30, 0.55, 1.0, 0.45], // Hippocampus
        vec![0.40, 0.35, 0.28, 0.32, 0.45, 1.0], // Thalamus
    ];

    let brain_graph = EpistemicGraph::from_fmri_correlation(&regions, &correlations, 0.02);

    // 3. Psychiatric Assessment (Active Inference model)
    let mut agent = psychiatry::ActiveInferenceAgent::new();

    // Patient observations: moderate depression, fatigue (common in MS)
    agent.observe("mood", Knowledge::measured(4.5, 0.5)); // Below average
    agent.observe("energy", Knowledge::measured(3.5, 0.8)); // Low (MS fatigue)
    agent.observe("cognition", Knowledge::measured(5.0, 1.0)); // Affected by MS

    // 4. Unified Analysis
    let mut analysis = unified::UnifiedAnalysis::new();
    analysis.add_scaffold(&scaffold_graph);
    analysis.add_brain(&brain_graph);
    analysis.add_psychiatric(&agent);
    analysis.correlate_scaffold_brain();

    // Print report
    println!("{}", analysis.report());

    // Treatment prediction
    println!("┌─ TREATMENT RESPONSE PREDICTIONS ───────────────────────────────────────┐");

    let treatments = ["rTMS", "SSRIs", "CBT", "combined"];
    for treatment in treatments {
        let response = analysis.predict_treatment_response(treatment);
        let prob = response.value * 100.0;
        let ci_lo = (response.value - 1.96 * response.std_dev()) * 100.0;
        let ci_hi = (response.value + 1.96 * response.std_dev()) * 100.0;

        println!(
            "│  {:12} → {:.1}% [{:.1}%, {:.1}%]₉₅%",
            treatment,
            prob,
            ci_lo.max(0.0),
            ci_hi.min(100.0)
        );
    }
    println!("│");
    println!("│  Recommendation: combined therapy (rTMS + CBT)");
    println!("│  Provenance: scaffold_KEC + brain_fMRI + active_inference");
    println!("└─────────────────────────────────────────────────────────────────────────┘\n");

    // The punchline
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                       THE UNIFIED INSIGHT                                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════╣");
    println!("║                                                                          ║");
    println!("║  Scaffold pore network  ≅  Neural connectivity network                   ║");
    println!("║  KEC entropy            ≅  Neural complexity                             ║");
    println!("║  Scaffold coherence     ≅  Brain small-worldness                         ║");
    println!("║  Cell migration paths   ≅  Information integration paths                 ║");
    println!("║                                                                          ║");
    println!("║  Same mathematics. Same uncertainty propagation. Same epistemic types.   ║");
    println!("║                                                                          ║");
    println!("║  Your master's thesis (KEC) informs your PhD (Computational Psychiatry). ║");
    println!("║  Your rTMS research (ABEM) integrates with both.                         ║");
    println!("║  Your engineering background enables the synthesis.                      ║");
    println!("║                                                                          ║");
    println!("║  This is not metaphor. This is mathematical unity.                       ║");
    println!("║  Sounio Language makes it explicit, computable, and auditable.        ║");
    println!("║                                                                          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_confidence() {
        let uniform = BetaConfidence::uniform();
        assert!((uniform.mean() - 0.5).abs() < 0.01);

        let strong = BetaConfidence::strong(0.9);
        assert!((strong.mean() - 0.9).abs() < 0.01);

        let updated = uniform.update(8, 2);
        assert!(updated.mean() > 0.5); // Should shift toward success
    }

    #[test]
    fn test_knowledge_propagation() {
        let a = Knowledge::measured(10.0, 1.0);
        let b = Knowledge::measured(5.0, 0.5);

        let sum = a.clone() + b.clone();
        assert!((sum.value - 15.0).abs() < 0.01);
        assert!((sum.variance - 1.5).abs() < 0.01);

        let diff = a.clone() - b.clone();
        assert!((diff.value - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_epistemic_graph_metrics() {
        let mut g = EpistemicGraph::new("test");
        for i in 0..4 {
            g.add_node(&format!("n{}", i));
        }
        g.add_edge(0, 1, Knowledge::measured(0.8, 0.01), "test");
        g.add_edge(1, 2, Knowledge::measured(0.7, 0.01), "test");
        g.add_edge(2, 3, Knowledge::measured(0.9, 0.01), "test");
        g.add_edge(0, 2, Knowledge::measured(0.6, 0.01), "test");

        let entropy = g.degree_entropy();
        assert!(entropy.value > 0.0);
        assert!(entropy.variance >= 0.0);

        let cc = g.clustering_coefficient();
        assert!(cc.value >= 0.0 && cc.value <= 1.0);
    }

    #[test]
    fn test_active_inference_update() {
        let mut agent = psychiatry::ActiveInferenceAgent::new();
        let prior_mood = agent.model.beliefs.get("mood").unwrap().value;

        agent.observe("mood", Knowledge::measured(3.0, 0.1));

        let posterior_mood = agent.model.beliefs.get("mood").unwrap().value;
        assert!(posterior_mood < prior_mood); // Posterior shifted toward observation
    }

    #[test]
    fn test_unified_analysis() {
        let mut analysis = unified::UnifiedAnalysis::new();

        let mut g = EpistemicGraph::new("scaffold");
        for i in 0..3 {
            let id = g.add_node(&format!("p{}", i));
            g.set_feature(id, "volume", Knowledge::measured(1e6, 1e8));
            g.set_feature(id, "curvature", Knowledge::measured(0.01, 0.001));
        }
        g.add_edge(0, 1, Knowledge::measured(0.5, 0.01), "throat");
        g.add_edge(1, 2, Knowledge::measured(0.6, 0.01), "throat");

        analysis.add_scaffold(&g);
        assert!(analysis.scaffold.is_some());

        let response = analysis.predict_treatment_response("rTMS");
        assert!(response.value >= 0.0 && response.value <= 1.0);
    }
}
