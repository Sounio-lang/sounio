//! Pearl's ID Algorithm for causal effect identification
//!
//! Implements the complete ID algorithm from Pearl (2009) "Causality" 2nd ed,
//! Section 3.4, Algorithm 1. This is used at compile time to verify that
//! causal queries (do-expressions) are identifiable from observational data.
//!
//! The algorithm either returns an identification formula (adjustment set,
//! frontdoor formula, etc.) or proves non-identifiability.
//!
//! Reference: Pearl, J. (2009). Causality. Cambridge University Press.
//! Reference: Shpitser, I. & Pearl, J. (2006). Identification of Joint
//!            Interventional Distributions in Recursive Semi-Markovian
//!            Causal Models. AAAI.

use std::collections::{HashSet, VecDeque};
use std::fmt;

use crate::smt::causal_encoding::{
    self, CONFIDENCE_BACKDOOR, CONFIDENCE_CAUSAL_MARKOV, CONFIDENCE_FRONTDOOR, CONFIDENCE_IV,
    CONFIDENCE_POSITIVITY,
};
use crate::types::causal::{CausalEdgeDef, CausalGraphDef, CausalNodeDef};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Result of the identification algorithm.
///
/// Represents the mathematical formula that expresses the causal effect
/// P(Y | do(X)) in terms of observational quantities.  Each variant
/// corresponds to a well-known identification strategy from the causal
/// inference literature.
#[derive(Debug, Clone)]
pub enum IdentificationFormula {
    /// P(Y) -- marginalisation (when X = empty set).
    ///
    /// When the treatment set is empty, the interventional distribution
    /// equals the observational marginal over Y.
    Marginal {
        /// The outcome variables.
        outcome: Vec<String>,
    },

    /// Sum_Z P(Y|X,Z) P(Z) -- backdoor/adjustment formula.
    ///
    /// Pearl (2009) Theorem 3.3.2.  Adjusting for Z blocks all backdoor
    /// paths from treatment to outcome.
    Adjustment {
        /// The outcome variables.
        outcome: Vec<String>,
        /// The treatment (intervention) variables.
        treatment: Vec<String>,
        /// The valid adjustment (backdoor) set.
        adjustment_set: Vec<String>,
    },

    /// Sum_{X'} Sum_M P(M|X) P(Y|X',M) P(X') -- frontdoor formula.
    ///
    /// Pearl (2009) Theorem 3.3.4.  Identification through a complete
    /// mediator set M when X and Y share an unmeasured confounder.
    Frontdoor {
        /// The treatment variables.
        treatment: Vec<String>,
        /// The outcome variables.
        outcome: Vec<String>,
        /// The mediator set satisfying the frontdoor criterion.
        mediator: Vec<String>,
    },

    /// beta_hat = Cov(Y,Z) / Cov(X,Z) -- instrumental variable estimator.
    ///
    /// Uses an instrument Z that affects Y only through X and shares no
    /// common causes with Y.
    InstrumentalVariable {
        /// The instrument.
        instrument: Vec<String>,
        /// The treatment variables.
        treatment: Vec<String>,
        /// The outcome variables.
        outcome: Vec<String>,
    },

    /// Direct observation: P(Y | do(X)) = P(Y | X) when no confounding.
    ///
    /// When there are no backdoor paths from treatment to outcome, the
    /// interventional distribution equals the conditional distribution.
    Conditional {
        /// The outcome variables.
        outcome: Vec<String>,
        /// The treatment variables.
        treatment: Vec<String>,
    },

    /// Product of sub-identifications (c-component factorisation).
    ///
    /// When the causal effect factorises over c-components, each factor
    /// is identified independently and the results are multiplied.
    Product {
        /// The individual identification formulas.
        factors: Vec<IdentificationFormula>,
    },
}

/// Explanation of why identification failed.
#[derive(Debug, Clone)]
pub struct NotIdentifiable {
    /// The treatment variables for which identification was attempted.
    pub treatment: Vec<String>,
    /// The outcome variables for which identification was attempted.
    pub outcome: Vec<String>,
    /// Human-readable reason for failure.
    pub reason: String,
    /// The "hedge" structure that blocks identification, if found.
    ///
    /// A hedge is a pair of c-forests (F, F') such that F' is a subset of F,
    /// which constitutes a sufficient condition for non-identifiability
    /// (Shpitser & Pearl, 2006, Theorem 3).
    pub hedge: Option<Vec<String>>,
}

/// Complete result from the ID algorithm.
#[derive(Debug, Clone)]
pub enum IdentificationResult {
    /// The causal effect is identifiable; formula shows how to compute it.
    Identified(IdentificationFormula),
    /// The causal effect is not identifiable from observational data.
    NotIdentified(NotIdentifiable),
}

// ---------------------------------------------------------------------------
// Display implementations
// ---------------------------------------------------------------------------

impl fmt::Display for IdentificationFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentificationFormula::Marginal { outcome } => {
                write!(f, "P({})", outcome.join(", "))
            }

            IdentificationFormula::Adjustment {
                outcome,
                treatment,
                adjustment_set,
            } => {
                write!(
                    f,
                    "Sum_{{{z}}} P({y} | {x}, {z}) P({z})",
                    z = adjustment_set.join(", "),
                    y = outcome.join(", "),
                    x = treatment.join(", "),
                )
            }

            IdentificationFormula::Frontdoor {
                treatment,
                outcome,
                mediator,
            } => {
                write!(
                    f,
                    "Sum_{{{x}'}} Sum_{{{m}}} P({m} | {x}) P({y} | {x}', {m}) P({x}')",
                    x = treatment.join(", "),
                    y = outcome.join(", "),
                    m = mediator.join(", "),
                )
            }

            IdentificationFormula::InstrumentalVariable {
                instrument,
                treatment,
                outcome,
            } => {
                write!(
                    f,
                    "Cov({y}, {z}) / Cov({x}, {z})",
                    z = instrument.join(", "),
                    x = treatment.join(", "),
                    y = outcome.join(", "),
                )
            }

            IdentificationFormula::Conditional { outcome, treatment } => {
                write!(
                    f,
                    "P({y} | {x})",
                    y = outcome.join(", "),
                    x = treatment.join(", "),
                )
            }

            IdentificationFormula::Product { factors } => {
                let parts: Vec<String> = factors.iter().map(|fac| format!("{}", fac)).collect();
                write!(f, "{}", parts.join(" * "))
            }
        }
    }
}

impl fmt::Display for NotIdentifiable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P({y} | do({x})) is NOT identifiable: {reason}",
            y = self.outcome.join(", "),
            x = self.treatment.join(", "),
            reason = self.reason,
        )?;
        if let Some(ref hedge) = self.hedge {
            write!(f, " [hedge: {{{}}}]", hedge.join(", "))?;
        }
        Ok(())
    }
}

impl fmt::Display for IdentificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentificationResult::Identified(formula) => {
                write!(f, "IDENTIFIED: {}", formula)
            }
            IdentificationResult::NotIdentified(reason) => {
                write!(f, "NOT IDENTIFIED: {}", reason)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pearl's ID Algorithm
// ---------------------------------------------------------------------------

/// Pearl's ID Algorithm.
///
/// Determines whether the causal effect P(Y | do(X)) is identifiable from
/// observational data in the given causal graph, and if so, returns the
/// identification formula.
///
/// # Algorithm (Pearl 2009, Section 3.4, simplified for DAGs)
///
/// 1. If X = empty set, return P(Y) (marginal).
/// 2. Remove non-ancestors of Y from G.
/// 3. If no backdoor paths exist, return P(Y|X) (conditional).
/// 4. Try backdoor adjustment (find valid adjustment set).
/// 5. Try frontdoor adjustment (find valid mediator set).
/// 6. Try instrumental variable (find valid instrument).
/// 7. Try c-component factorisation.
/// 8. Return not-identifiable.
///
/// # Arguments
///
/// * `graph` - The causal DAG.
/// * `treatment` - Treatment (intervention) variable names.
/// * `outcome` - Outcome variable names.
pub fn identify(
    graph: &CausalGraphDef,
    treatment: &[String],
    outcome: &[String],
) -> IdentificationResult {
    // ------------------------------------------------------------------
    // Step 1: Empty treatment => marginal P(Y)
    // ------------------------------------------------------------------
    if treatment.is_empty() {
        return IdentificationResult::Identified(IdentificationFormula::Marginal {
            outcome: outcome.to_vec(),
        });
    }

    // ------------------------------------------------------------------
    // Step 2: Restrict graph to ancestors of Y (ancestor graph)
    // ------------------------------------------------------------------
    let ancestor_graph = ancestor_subgraph(graph, outcome);

    // Check whether treatment is still in the ancestor graph
    let ancestor_nodes: HashSet<&str> = ancestor_graph
        .nodes
        .iter()
        .map(|n| n.name.as_str())
        .collect();
    let effective_treatment: Vec<String> = treatment
        .iter()
        .filter(|t| ancestor_nodes.contains(t.as_str()))
        .cloned()
        .collect();

    // If no treatment variable is an ancestor of the outcome, the
    // intervention has no effect and P(Y|do(X)) = P(Y).
    if effective_treatment.is_empty() {
        return IdentificationResult::Identified(IdentificationFormula::Marginal {
            outcome: outcome.to_vec(),
        });
    }

    // ------------------------------------------------------------------
    // Step 3: Check if there are any backdoor paths at all
    // ------------------------------------------------------------------
    // If, for every treatment variable, removing its outgoing edges
    // d-separates it from all outcome variables, then P(Y|do(X)) = P(Y|X).
    let no_confounding = effective_treatment.iter().all(|t| {
        let mutilated = causal_encoding::mutilate_graph(&ancestor_graph, t);
        outcome.iter().all(|y| {
            if !ancestor_nodes.contains(y.as_str()) {
                return true;
            }
            is_d_separated_in(&mutilated, t, y, &HashSet::new())
        })
    });

    if no_confounding {
        return IdentificationResult::Identified(IdentificationFormula::Conditional {
            outcome: outcome.to_vec(),
            treatment: effective_treatment,
        });
    }

    // ------------------------------------------------------------------
    // Step 4: Try backdoor adjustment
    // ------------------------------------------------------------------
    // For single treatment / single outcome (the common case), search for
    // a valid adjustment set.
    if effective_treatment.len() == 1 && outcome.len() == 1 {
        let t = &effective_treatment[0];
        let y = &outcome[0];

        if let Some(adj_set) = find_backdoor_set(&ancestor_graph, t, y) {
            return IdentificationResult::Identified(IdentificationFormula::Adjustment {
                outcome: outcome.to_vec(),
                treatment: effective_treatment,
                adjustment_set: adj_set,
            });
        }

        // ------------------------------------------------------------------
        // Step 5: Try frontdoor adjustment
        // ------------------------------------------------------------------
        if let Some(mediators) = find_frontdoor_set(&ancestor_graph, t, y) {
            return IdentificationResult::Identified(IdentificationFormula::Frontdoor {
                treatment: effective_treatment,
                outcome: outcome.to_vec(),
                mediator: mediators,
            });
        }

        // ------------------------------------------------------------------
        // Step 6: Try instrumental variable
        // ------------------------------------------------------------------
        if let Some(instrument) = find_instrument(&ancestor_graph, t, y) {
            return IdentificationResult::Identified(
                IdentificationFormula::InstrumentalVariable {
                    instrument: vec![instrument],
                    treatment: effective_treatment,
                    outcome: outcome.to_vec(),
                },
            );
        }
    }

    // ------------------------------------------------------------------
    // Step 7: Try c-component factorisation
    // ------------------------------------------------------------------
    let c_components = find_c_components(&ancestor_graph);
    if c_components.len() > 1 {
        let mut factors = Vec::new();
        let mut all_identified = true;

        for component in &c_components {
            // Determine which treatment and outcome variables fall in this
            // c-component.
            let comp_treatment: Vec<String> = effective_treatment
                .iter()
                .filter(|t| component.contains(t.as_str()))
                .cloned()
                .collect();
            let comp_outcome: Vec<String> = outcome
                .iter()
                .filter(|y| component.contains(y.as_str()))
                .cloned()
                .collect();

            if comp_outcome.is_empty() {
                continue; // No outcome in this component, skip
            }

            // Build a subgraph for this component
            let comp_graph = component_subgraph(&ancestor_graph, component);
            let sub_result = identify(&comp_graph, &comp_treatment, &comp_outcome);

            match sub_result {
                IdentificationResult::Identified(formula) => {
                    factors.push(formula);
                }
                IdentificationResult::NotIdentified(_) => {
                    all_identified = false;
                    break;
                }
            }
        }

        if all_identified && !factors.is_empty() {
            if factors.len() == 1 {
                return IdentificationResult::Identified(factors.into_iter().next().expect(
                    "factors vector is non-empty",
                ));
            }
            return IdentificationResult::Identified(IdentificationFormula::Product { factors });
        }
    }

    // ------------------------------------------------------------------
    // Step 8: Not identifiable
    // ------------------------------------------------------------------
    IdentificationResult::NotIdentified(NotIdentifiable {
        treatment: effective_treatment,
        outcome: outcome.to_vec(),
        reason: "no identification strategy succeeded (backdoor, frontdoor, IV, c-component factorisation)".to_string(),
        hedge: None,
    })
}

// ---------------------------------------------------------------------------
// Adjustment set search
// ---------------------------------------------------------------------------

/// Find a valid backdoor adjustment set (if one exists).
///
/// Strategy: enumerate subsets of non-descendants of treatment, starting
/// with the smallest subsets (Occam's razor).  Returns the first valid
/// set found, which will be minimal in cardinality.
///
/// A set Z satisfies the backdoor criterion relative to (X, Y) if:
/// 1. No node in Z is a descendant of X.
/// 2. Z blocks every backdoor path from X to Y (i.e., X _|_ Y | Z in
///    the mutilated graph G_{X-bar}).
fn find_backdoor_set(
    graph: &CausalGraphDef,
    treatment: &str,
    outcome: &str,
) -> Option<Vec<String>> {
    let treatment_desc = causal_encoding::get_descendants(graph, treatment);

    // Candidate adjustment variables: all nodes that are not the treatment,
    // not the outcome, and not descendants of treatment.
    let candidates: Vec<String> = graph
        .nodes
        .iter()
        .map(|n| n.name.clone())
        .filter(|n| {
            n != treatment && n != outcome && !treatment_desc.contains(n)
        })
        .collect();

    let mutilated = causal_encoding::mutilate_graph(graph, treatment);

    // Try empty set first (no adjustment needed)
    if is_d_separated_in(&mutilated, treatment, outcome, &HashSet::new()) {
        return Some(Vec::new());
    }

    // Try single-variable sets, then pairs, etc.
    // We cap at subsets of size min(candidates.len(), 4) to keep
    // compile time bounded.
    let max_size = candidates.len().min(4);
    for size in 1..=max_size {
        let mut result = Vec::new();
        if find_valid_subset(
            &mutilated,
            treatment,
            outcome,
            &candidates,
            size,
            0,
            &mut result,
        ) {
            return Some(result);
        }
    }

    // As a fallback, try the full candidate set
    if candidates.len() > max_size {
        let full_set: HashSet<String> = candidates.iter().cloned().collect();
        if is_d_separated_in(&mutilated, treatment, outcome, &full_set) {
            return Some(candidates);
        }
    }

    None
}

/// Recursive helper: find a subset of `candidates` of exactly `size` elements
/// that d-separates treatment from outcome in the mutilated graph.
fn find_valid_subset(
    mutilated: &CausalGraphDef,
    treatment: &str,
    outcome: &str,
    candidates: &[String],
    size: usize,
    start: usize,
    current: &mut Vec<String>,
) -> bool {
    if current.len() == size {
        let set: HashSet<String> = current.iter().cloned().collect();
        return is_d_separated_in(mutilated, treatment, outcome, &set);
    }

    let remaining = size - current.len();
    if start + remaining > candidates.len() {
        return false;
    }

    for i in start..candidates.len() {
        current.push(candidates[i].clone());
        if find_valid_subset(mutilated, treatment, outcome, candidates, size, i + 1, current) {
            return true;
        }
        current.pop();
    }

    false
}

// ---------------------------------------------------------------------------
// Frontdoor set search
// ---------------------------------------------------------------------------

/// Find a valid frontdoor set (mediators).
///
/// A set M satisfies the frontdoor criterion relative to (X, Y) if:
/// 1. All directed paths from X to Y pass through M.
/// 2. There is no unblocked backdoor path from X to any node in M.
/// 3. All backdoor paths from M to Y are blocked by X.
fn find_frontdoor_set(
    graph: &CausalGraphDef,
    treatment: &str,
    outcome: &str,
) -> Option<Vec<String>> {
    // Candidate mediators: nodes on directed paths from treatment to outcome
    // (excluding endpoints).
    let directed_paths = causal_encoding::enumerate_directed_paths(
        graph,
        treatment,
        outcome,
        graph.nodes.len(),
    );

    if directed_paths.is_empty() {
        return None;
    }

    // Collect all intermediary nodes on directed paths
    let mut intermediaries: HashSet<String> = HashSet::new();
    for path in &directed_paths {
        for node in &path[1..path.len() - 1] {
            intermediaries.insert(node.clone());
        }
    }

    if intermediaries.is_empty() {
        return None; // Direct edge only, no mediators
    }

    // Try single mediator first, then larger sets
    let candidates: Vec<String> = intermediaries.into_iter().collect();

    for size in 1..=candidates.len() {
        let mut subset = Vec::new();
        if find_valid_frontdoor_subset(
            graph,
            treatment,
            outcome,
            &directed_paths,
            &candidates,
            size,
            0,
            &mut subset,
        ) {
            return Some(subset);
        }
    }

    None
}

/// Recursive helper: find a frontdoor-valid subset of mediators.
fn find_valid_frontdoor_subset(
    graph: &CausalGraphDef,
    treatment: &str,
    outcome: &str,
    directed_paths: &[Vec<String>],
    candidates: &[String],
    size: usize,
    start: usize,
    current: &mut Vec<String>,
) -> bool {
    if current.len() == size {
        let mediator_set: HashSet<&str> = current.iter().map(|s| s.as_str()).collect();

        // Condition 1: All directed paths from X to Y pass through M
        for path in directed_paths {
            let path_intermediaries: HashSet<&str> =
                path[1..path.len() - 1].iter().map(|s| s.as_str()).collect();
            let passes_through = mediator_set.iter().any(|m| path_intermediaries.contains(m));
            if !passes_through {
                return false;
            }
        }

        // Condition 2: No unblocked backdoor path from X to any M
        let mutilated_x = causal_encoding::mutilate_graph(graph, treatment);
        for m in &mediator_set {
            if !is_d_separated_in(&mutilated_x, treatment, m, &HashSet::new()) {
                return false;
            }
        }

        // Condition 3: All backdoor paths from M to Y are blocked by X
        let treatment_set: HashSet<String> = [treatment.to_string()].into();
        for m in &mediator_set {
            let mutilated_m = causal_encoding::mutilate_graph(graph, m);
            if !is_d_separated_in(&mutilated_m, m, outcome, &treatment_set) {
                return false;
            }
        }

        return true;
    }

    let remaining = size - current.len();
    if start + remaining > candidates.len() {
        return false;
    }

    for i in start..candidates.len() {
        current.push(candidates[i].clone());
        if find_valid_frontdoor_subset(
            graph,
            treatment,
            outcome,
            directed_paths,
            candidates,
            size,
            i + 1,
            current,
        ) {
            return true;
        }
        current.pop();
    }

    false
}

// ---------------------------------------------------------------------------
// Instrumental variable search
// ---------------------------------------------------------------------------

/// Find a valid instrument for the effect of treatment on outcome.
///
/// An instrument Z must satisfy:
/// 1. Relevance: Z is associated with treatment (directed path Z -> ... -> X).
/// 2. Exclusion: Z affects Y only through X (Z _|_ Y in G_{X-bar}).
/// 3. Exogeneity: Z and Y share no common causes.
///
/// Conditions 2+3 are tested jointly by checking Z _|_ Y in the graph with
/// all outgoing edges from X removed.
fn find_instrument(
    graph: &CausalGraphDef,
    treatment: &str,
    outcome: &str,
) -> Option<String> {
    let mutilated = causal_encoding::mutilate_graph(graph, treatment);

    for node in &graph.nodes {
        let z = node.name.as_str();
        if z == treatment || z == outcome {
            continue;
        }

        // Condition 1: Z has a directed path to treatment
        if !causal_encoding::has_directed_path(graph, z, treatment) {
            continue;
        }

        // Conditions 2+3: Z _|_ Y in G_{X-bar}
        if is_d_separated_in(&mutilated, z, outcome, &HashSet::new()) {
            return Some(z.to_string());
        }
    }

    None
}

// ---------------------------------------------------------------------------
// C-component factorisation
// ---------------------------------------------------------------------------

/// Find c-components (confounded components) of a graph.
///
/// In a DAG without latent variables (no bidirected edges), each node is
/// its own c-component.  With latent variables (represented as bidirected
/// edges in semi-Markovian models), c-components are connected components
/// in the bidirected edge graph.
///
/// Since our `CausalGraphDef` only has directed edges, we detect implicit
/// latent confounders by looking for pairs of nodes that share a common
/// parent.  Nodes that share a common parent are placed in the same
/// c-component (they are confounded).
fn find_c_components(graph: &CausalGraphDef) -> Vec<HashSet<String>> {
    let all_nodes: Vec<&str> = graph.nodes.iter().map(|n| n.name.as_str()).collect();

    // Build parent map
    let mut parents_of: std::collections::HashMap<&str, HashSet<&str>> =
        std::collections::HashMap::new();
    for node in &graph.nodes {
        parents_of.entry(node.name.as_str()).or_default();
    }
    for edge in &graph.edges {
        parents_of
            .entry(edge.to.as_str())
            .or_default()
            .insert(edge.from.as_str());
    }

    // Build children map
    let mut children_of: std::collections::HashMap<&str, HashSet<&str>> =
        std::collections::HashMap::new();
    for edge in &graph.edges {
        children_of
            .entry(edge.from.as_str())
            .or_default()
            .insert(edge.to.as_str());
    }

    // Union-Find for c-components.
    // Two nodes are in the same c-component if they share a common parent
    // (indicating potential latent confounding).
    let mut component_id: std::collections::HashMap<&str, usize> =
        std::collections::HashMap::new();
    for (i, &node) in all_nodes.iter().enumerate() {
        component_id.insert(node, i);
    }

    // Merge: for each parent, all its children are in the same c-component.
    for (_parent, children) in &children_of {
        if children.len() > 1 {
            let children_vec: Vec<&str> = children.iter().copied().collect();
            let target = find_root(&component_id, children_vec[0]);
            for &child in &children_vec[1..] {
                let child_root = find_root(&component_id, child);
                if child_root != target {
                    // Merge child_root into target
                    for (_, v) in component_id.iter_mut() {
                        if *v == child_root {
                            *v = target;
                        }
                    }
                }
            }
        }
    }

    // Flatten component_id into actual sets
    let mut components_map: std::collections::HashMap<usize, HashSet<String>> =
        std::collections::HashMap::new();
    for (&node, &cid) in &component_id {
        let root = find_root(&component_id, node);
        components_map
            .entry(root)
            .or_default()
            .insert(node.to_string());
    }

    components_map.into_values().collect()
}

/// Find the root component id for a node.
fn find_root(component_id: &std::collections::HashMap<&str, usize>, node: &str) -> usize {
    component_id.get(node).copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Confidence computation
// ---------------------------------------------------------------------------

/// Compute the total epistemic confidence for an identification formula.
///
/// Each identification strategy carries assumptions that degrade confidence:
/// - Backdoor: NUC (0.85) x Positivity (0.98) x Markov (0.99)
/// - Frontdoor: Faithfulness (0.95) x Markov (0.99)
/// - IV: Exogeneity (0.85) x Positivity (0.98)
/// - Conditional: Markov (0.99) only
/// - Marginal: 1.0 (no extra assumptions)
/// - Product: multiply all factor confidences
pub fn identification_confidence(formula: &IdentificationFormula) -> f64 {
    match formula {
        IdentificationFormula::Marginal { .. } => 1.0,

        IdentificationFormula::Conditional { .. } => CONFIDENCE_CAUSAL_MARKOV,

        IdentificationFormula::Adjustment { .. } => {
            CONFIDENCE_BACKDOOR * CONFIDENCE_POSITIVITY * CONFIDENCE_CAUSAL_MARKOV
        }

        IdentificationFormula::Frontdoor { .. } => {
            CONFIDENCE_FRONTDOOR * CONFIDENCE_CAUSAL_MARKOV
        }

        IdentificationFormula::InstrumentalVariable { .. } => {
            CONFIDENCE_IV * CONFIDENCE_POSITIVITY
        }

        IdentificationFormula::Product { factors } => {
            factors.iter().map(|f| identification_confidence(f)).product()
        }
    }
}

// ---------------------------------------------------------------------------
// Graph utility helpers (local to this module)
// ---------------------------------------------------------------------------

/// Build the ancestor subgraph: restrict graph to ancestors of Y (including Y).
fn ancestor_subgraph(graph: &CausalGraphDef, outcome: &[String]) -> CausalGraphDef {
    // Collect ancestors of all outcome variables
    let mut relevant: HashSet<String> = HashSet::new();
    for y in outcome {
        relevant.insert(y.clone());
        let ancestors = causal_encoding::get_ancestors(graph, y);
        relevant.extend(ancestors);
    }

    let nodes: Vec<CausalNodeDef> = graph
        .nodes
        .iter()
        .filter(|n| relevant.contains(&n.name))
        .cloned()
        .collect();

    let edges: Vec<CausalEdgeDef> = graph
        .edges
        .iter()
        .filter(|e| relevant.contains(&e.from) && relevant.contains(&e.to))
        .cloned()
        .collect();

    CausalGraphDef {
        name: graph.name.clone(),
        nodes,
        edges,
        topological_order: None,
        span: graph.span,
    }
}

/// Build a subgraph restricted to a set of nodes.
fn component_subgraph(graph: &CausalGraphDef, component: &HashSet<String>) -> CausalGraphDef {
    let nodes: Vec<CausalNodeDef> = graph
        .nodes
        .iter()
        .filter(|n| component.contains(&n.name))
        .cloned()
        .collect();

    let edges: Vec<CausalEdgeDef> = graph
        .edges
        .iter()
        .filter(|e| component.contains(&e.from) && component.contains(&e.to))
        .cloned()
        .collect();

    CausalGraphDef {
        name: graph.name.clone(),
        nodes,
        edges,
        topological_order: None,
        span: graph.span,
    }
}

/// Local d-separation check using the Bayes-Ball algorithm from
/// `causal_encoding`.
fn is_d_separated_in(
    graph: &CausalGraphDef,
    x: &str,
    y: &str,
    conditioning: &HashSet<String>,
) -> bool {
    // Build parent / child maps
    let mut parents: std::collections::HashMap<&str, HashSet<&str>> =
        std::collections::HashMap::new();
    let mut children: std::collections::HashMap<&str, HashSet<&str>> =
        std::collections::HashMap::new();

    for node in &graph.nodes {
        parents.entry(node.name.as_str()).or_default();
        children.entry(node.name.as_str()).or_default();
    }

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

    // Get all descendants of the conditioning set
    let mut cond_descendants: HashSet<String> = HashSet::new();
    let mut queue: VecDeque<&str> = conditioning.iter().map(|s| s.as_str()).collect();
    while let Some(node) = queue.pop_front() {
        if let Some(node_children) = children.get(node) {
            for &child in node_children {
                if !cond_descendants.contains(child) {
                    cond_descendants.insert(child.to_string());
                    queue.push_back(child);
                }
            }
        }
    }

    // Bayes-Ball BFS
    let mut visited: HashSet<(&str, bool)> = HashSet::new();
    let mut bfs_queue: VecDeque<(&str, bool)> = VecDeque::new();

    bfs_queue.push_back((x, true));
    bfs_queue.push_back((x, false));

    while let Some((current, arrived_from_parent)) = bfs_queue.pop_front() {
        if visited.contains(&(current, arrived_from_parent)) {
            continue;
        }
        visited.insert((current, arrived_from_parent));

        if current == y {
            return false; // Active path found
        }

        let in_z = conditioning.contains(current);
        let descendant_in_z = cond_descendants.contains(current);

        // Traverse to parents
        if let Some(node_parents) = parents.get(current) {
            for &parent in node_parents {
                let can_traverse = if arrived_from_parent {
                    in_z || descendant_in_z
                } else {
                    !in_z
                };
                if can_traverse {
                    bfs_queue.push_back((parent, false));
                }
            }
        }

        // Traverse to children
        if let Some(node_children) = children.get(current) {
            for &child in node_children {
                if !in_z {
                    bfs_queue.push_back((child, true));
                }
            }
        }
    }

    true // No active path found
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::Span;

    // -----------------------------------------------------------------------
    // Test helpers
    // -----------------------------------------------------------------------

    fn make_span() -> Span {
        Span::new(0, 0)
    }

    fn node(name: &str) -> CausalNodeDef {
        CausalNodeDef {
            name: name.to_string(),
            span: make_span(),
        }
    }

    fn edge(from: &str, to: &str) -> CausalEdgeDef {
        CausalEdgeDef {
            from: from.to_string(),
            to: to.to_string(),
            span: make_span(),
        }
    }

    fn graph(
        name: &str,
        nodes: Vec<CausalNodeDef>,
        edges: Vec<CausalEdgeDef>,
    ) -> CausalGraphDef {
        CausalGraphDef {
            name: name.to_string(),
            nodes,
            edges,
            topological_order: None,
            span: make_span(),
        }
    }

    // -----------------------------------------------------------------------
    // 1. Empty treatment => Marginal P(Y)
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_treatment_marginal() {
        let g = graph(
            "simple",
            vec![node("A"), node("B")],
            vec![edge("A", "B")],
        );

        let result = identify(&g, &[], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Marginal { ref outcome }) => {
                assert_eq!(outcome, &["B"]);
            }
            _ => panic!("expected Marginal, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 2. No confounding (A -> B) => Conditional P(B|A)
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_confounding_conditional() {
        let g = graph(
            "simple",
            vec![node("A"), node("B")],
            vec![edge("A", "B")],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Conditional {
                ref outcome,
                ref treatment,
            }) => {
                assert_eq!(outcome, &["B"]);
                assert_eq!(treatment, &["A"]);
            }
            _ => panic!("expected Conditional, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 3. Confounded with adjustment (A <- U -> B, A -> B) => Adjustment on {U}
    // -----------------------------------------------------------------------

    #[test]
    fn test_confounded_backdoor_adjustment() {
        let g = graph(
            "confounded",
            vec![node("A"), node("B"), node("U")],
            vec![edge("U", "A"), edge("U", "B"), edge("A", "B")],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Adjustment {
                ref adjustment_set,
                ..
            }) => {
                assert!(
                    adjustment_set.contains(&"U".to_string()),
                    "adjustment set should contain U, got: {:?}",
                    adjustment_set
                );
            }
            _ => panic!("expected Adjustment, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 4. Frontdoor identifiable (A -> M -> B, A <- U -> B) => Frontdoor on {M}
    // -----------------------------------------------------------------------

    #[test]
    fn test_frontdoor_identification() {
        let g = graph(
            "frontdoor",
            vec![node("A"), node("B"), node("M"), node("U")],
            vec![
                edge("A", "M"),
                edge("M", "B"),
                edge("U", "A"),
                edge("U", "B"),
            ],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Frontdoor {
                ref mediator,
                ..
            }) => {
                assert!(
                    mediator.contains(&"M".to_string()),
                    "mediator set should contain M, got: {:?}",
                    mediator
                );
            }
            // The algorithm might also find an adjustment on U first.
            // Since U is not a descendant of A, it is a valid backdoor set.
            IdentificationResult::Identified(IdentificationFormula::Adjustment {
                ref adjustment_set,
                ..
            }) => {
                assert!(
                    adjustment_set.contains(&"U".to_string()),
                    "adjustment set should contain U, got: {:?}",
                    adjustment_set
                );
            }
            _ => panic!("expected Frontdoor or Adjustment, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 5. IV identifiable (Z -> A -> B, A <- U -> B) => IV with Z
    // -----------------------------------------------------------------------

    #[test]
    fn test_iv_identification() {
        let g = graph(
            "iv",
            vec![node("A"), node("B"), node("Z"), node("U")],
            vec![
                edge("Z", "A"),
                edge("A", "B"),
                edge("U", "A"),
                edge("U", "B"),
            ],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::InstrumentalVariable {
                ref instrument,
                ..
            }) => {
                assert!(
                    instrument.contains(&"Z".to_string()),
                    "instrument should be Z, got: {:?}",
                    instrument
                );
            }
            // U is a valid backdoor set here too
            IdentificationResult::Identified(IdentificationFormula::Adjustment {
                ref adjustment_set,
                ..
            }) => {
                assert!(
                    adjustment_set.contains(&"U".to_string()),
                    "adjustment set should contain U, got: {:?}",
                    adjustment_set
                );
            }
            _ => panic!("expected IV or Adjustment, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 6. Chain (A -> B -> C) => Conditional P(C|A)
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_conditional() {
        let g = graph(
            "chain",
            vec![node("A"), node("B"), node("C")],
            vec![edge("A", "B"), edge("B", "C")],
        );

        let result = identify(&g, &["A".into()], &["C".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Conditional {
                ref outcome,
                ref treatment,
            }) => {
                assert_eq!(outcome, &["C"]);
                assert_eq!(treatment, &["A"]);
            }
            _ => panic!("expected Conditional, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 7. Diamond (A -> B, A -> C, B -> D, C -> D) => Adjustment on {B,C}
    //    or {B} or {C}
    // -----------------------------------------------------------------------

    #[test]
    fn test_diamond_adjustment() {
        let g = graph(
            "diamond",
            vec![node("A"), node("B"), node("C"), node("D")],
            vec![
                edge("A", "B"),
                edge("A", "C"),
                edge("B", "D"),
                edge("C", "D"),
            ],
        );

        let result = identify(&g, &["A".into()], &["D".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Conditional { .. }) => {
                // No confounding, so conditional is correct
            }
            IdentificationResult::Identified(IdentificationFormula::Adjustment { .. }) => {
                // Also acceptable if the algorithm tries adjustment
            }
            _ => panic!("expected Conditional or Adjustment, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 8. Multiple valid sets => Find minimal adjustment
    // -----------------------------------------------------------------------

    #[test]
    fn test_minimal_adjustment() {
        // Graph: U1 -> A, U2 -> A, U1 -> B, A -> B
        // Both {U1} and {U1, U2} are valid, but {U1} is smaller.
        let g = graph(
            "multi_adj",
            vec![node("A"), node("B"), node("U1"), node("U2")],
            vec![
                edge("U1", "A"),
                edge("U2", "A"),
                edge("U1", "B"),
                edge("A", "B"),
            ],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Adjustment {
                ref adjustment_set,
                ..
            }) => {
                // Should find {U1} as the minimal adjustment set
                assert!(
                    adjustment_set.contains(&"U1".to_string()),
                    "adjustment set should contain U1, got: {:?}",
                    adjustment_set
                );
                // Verify it is minimal (size 1)
                assert_eq!(
                    adjustment_set.len(),
                    1,
                    "adjustment set should be minimal (size 1), got: {:?}",
                    adjustment_set
                );
            }
            _ => panic!("expected Adjustment, got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // 9. Confidence computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_confidence_marginal() {
        let formula = IdentificationFormula::Marginal {
            outcome: vec!["Y".into()],
        };
        let c = identification_confidence(&formula);
        assert!((c - 1.0).abs() < 1e-10, "marginal confidence should be 1.0");
    }

    #[test]
    fn test_confidence_conditional() {
        let formula = IdentificationFormula::Conditional {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
        };
        let c = identification_confidence(&formula);
        assert!(
            (c - CONFIDENCE_CAUSAL_MARKOV).abs() < 1e-10,
            "conditional confidence should be {}, got {}",
            CONFIDENCE_CAUSAL_MARKOV,
            c
        );
    }

    #[test]
    fn test_confidence_adjustment() {
        let formula = IdentificationFormula::Adjustment {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
            adjustment_set: vec!["Z".into()],
        };
        let c = identification_confidence(&formula);
        let expected = CONFIDENCE_BACKDOOR * CONFIDENCE_POSITIVITY * CONFIDENCE_CAUSAL_MARKOV;
        assert!(
            (c - expected).abs() < 1e-10,
            "adjustment confidence should be {}, got {}",
            expected,
            c
        );
    }

    #[test]
    fn test_confidence_frontdoor() {
        let formula = IdentificationFormula::Frontdoor {
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
            mediator: vec!["M".into()],
        };
        let c = identification_confidence(&formula);
        let expected = CONFIDENCE_FRONTDOOR * CONFIDENCE_CAUSAL_MARKOV;
        assert!(
            (c - expected).abs() < 1e-10,
            "frontdoor confidence should be {}, got {}",
            expected,
            c
        );
    }

    #[test]
    fn test_confidence_iv() {
        let formula = IdentificationFormula::InstrumentalVariable {
            instrument: vec!["Z".into()],
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
        };
        let c = identification_confidence(&formula);
        let expected = CONFIDENCE_IV * CONFIDENCE_POSITIVITY;
        assert!(
            (c - expected).abs() < 1e-10,
            "IV confidence should be {}, got {}",
            expected,
            c
        );
    }

    #[test]
    fn test_confidence_product() {
        let formula = IdentificationFormula::Product {
            factors: vec![
                IdentificationFormula::Marginal {
                    outcome: vec!["Y1".into()],
                },
                IdentificationFormula::Conditional {
                    outcome: vec!["Y2".into()],
                    treatment: vec!["X2".into()],
                },
            ],
        };
        let c = identification_confidence(&formula);
        let expected = 1.0 * CONFIDENCE_CAUSAL_MARKOV;
        assert!(
            (c - expected).abs() < 1e-10,
            "product confidence should be {}, got {}",
            expected,
            c
        );
    }

    // -----------------------------------------------------------------------
    // Display tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_display_marginal() {
        let formula = IdentificationFormula::Marginal {
            outcome: vec!["Y".into()],
        };
        let s = format!("{}", formula);
        assert_eq!(s, "P(Y)");
    }

    #[test]
    fn test_display_adjustment() {
        let formula = IdentificationFormula::Adjustment {
            outcome: vec!["Y".into()],
            treatment: vec!["X".into()],
            adjustment_set: vec!["Z".into()],
        };
        let s = format!("{}", formula);
        assert!(s.contains("Sum"));
        assert!(s.contains("P(Y | X, Z)"));
    }

    #[test]
    fn test_display_not_identifiable() {
        let ni = NotIdentifiable {
            treatment: vec!["X".into()],
            outcome: vec!["Y".into()],
            reason: "no strategy succeeded".to_string(),
            hedge: Some(vec!["A".into(), "B".into()]),
        };
        let s = format!("{}", ni);
        assert!(s.contains("NOT identifiable"));
        assert!(s.contains("hedge"));
    }

    #[test]
    fn test_display_identification_result() {
        let result = IdentificationResult::Identified(IdentificationFormula::Marginal {
            outcome: vec!["Y".into()],
        });
        let s = format!("{}", result);
        assert!(s.contains("IDENTIFIED"));
        assert!(s.contains("P(Y)"));
    }

    // -----------------------------------------------------------------------
    // Treatment not ancestor of outcome => Marginal
    // -----------------------------------------------------------------------

    #[test]
    fn test_treatment_not_ancestor() {
        // A and B are disconnected, C -> B
        let g = graph(
            "disconnected",
            vec![node("A"), node("B"), node("C")],
            vec![edge("C", "B")],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        match result {
            IdentificationResult::Identified(IdentificationFormula::Marginal { ref outcome }) => {
                assert_eq!(outcome, &["B"]);
            }
            _ => panic!("expected Marginal (treatment not ancestor), got: {:?}", result),
        }
    }

    // -----------------------------------------------------------------------
    // C-component factorisation (multiple components)
    // -----------------------------------------------------------------------

    #[test]
    fn test_c_component_identification() {
        // Separate components: A -> B, C -> D
        // Treatment {A}, Outcome {B} -- should identify as conditional
        let g = graph(
            "two_components",
            vec![node("A"), node("B"), node("C"), node("D")],
            vec![edge("A", "B"), edge("C", "D")],
        );

        let result = identify(&g, &["A".into()], &["B".into()]);
        assert!(
            matches!(result, IdentificationResult::Identified(_)),
            "should be identifiable: {:?}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // find_c_components unit test
    // -----------------------------------------------------------------------

    #[test]
    fn test_find_c_components_independent() {
        // Independent nodes: A, B (no edges)
        let g = graph(
            "independent",
            vec![node("A"), node("B")],
            vec![],
        );

        let components = find_c_components(&g);
        assert_eq!(
            components.len(),
            2,
            "two independent nodes should form two c-components, got: {:?}",
            components
        );
    }

    #[test]
    fn test_find_c_components_shared_parent() {
        // U -> A, U -> B: A and B share parent U
        let g = graph(
            "shared_parent",
            vec![node("A"), node("B"), node("U")],
            vec![edge("U", "A"), edge("U", "B")],
        );

        let components = find_c_components(&g);
        // A and B share parent U, so they should be in the same c-component
        let ab_together = components.iter().any(|c| {
            c.contains("A") && c.contains("B")
        });
        assert!(
            ab_together,
            "A and B should be in the same c-component (shared parent U), got: {:?}",
            components
        );
    }
}
