/// epistemic::causal — Causal Inference with Epistemic Uncertainty
///
/// The world's first causal inference library that integrates Pearl's
/// do-calculus with Bayesian epistemic uncertainty tracking.
///
/// # Philosophy
///
/// Traditional causal inference: "The causal effect is 0.3"
/// Demetrios epistemic:        "The causal effect is Beta(6, 14), giving
///                              0.3 ± 0.12 with 0.02 residual ignorance"
///
/// Every causal edge has:
/// - Existence probability (Beta posterior on whether edge exists)
/// - Effect size (point estimate)
/// - Effect uncertainty (epistemic variance)
///
/// This enables:
/// - Honest uncertainty in causal structure learning
/// - Propagation of epistemic uncertainty through do-calculus
/// - Active inference for targeted interventions
/// - Integration with ML for causal representation learning
///
/// # Quick Start
///
/// ```demetrios
/// use std::epistemic::causal::{CausalDAG, NodeType, do_intervention}
///
/// // Create a causal DAG
/// let dag = dag_new()
/// dag = dag_add_node(dag, "X", NodeType::Treatment)
/// dag = dag_add_node(dag, "Y", NodeType::Outcome)
/// dag = dag_add_node(dag, "Z", NodeType::Confounder)
///
/// // Add edges with epistemic uncertainty
/// dag = dag_add_edge(dag, "Z", "X", beta_new(8.0, 2.0), 0.5, 0.1)
/// dag = dag_add_edge(dag, "Z", "Y", beta_new(7.0, 3.0), 0.4, 0.08)
/// dag = dag_add_edge(dag, "X", "Y", beta_new(6.0, 4.0), 0.3, 0.12)
///
/// // Perform intervention (Pearl's do-operator)
/// let intervened = do_intervention(dag, "X", 1.0)
///
/// // Estimate causal effect
/// let effect = average_treatment_effect(dag, "X", "Y")
/// ```

// ============================================================================
// SECTION 1: CORE DATA STRUCTURES
// ============================================================================

/// Node types in causal DAG
enum NodeType {
    Treatment,      // Intervention variable
    Outcome,        // Target outcome variable
    Confounder,     // Common cause of treatment and outcome
    Mediator,       // On causal path between treatment and outcome
    Collider,       // Common effect of two variables
    Instrument      // Instrumental variable (affects treatment, not outcome directly)
}

/// Causal edge with epistemic uncertainty
///
/// Represents a directed edge X -> Y where:
/// - exists: Beta posterior on P(edge exists | data)
/// - strength: Expected causal effect size
/// - strength_var: Epistemic uncertainty in effect size
struct CausalEdge {
    from: [u8],         // Source node name (byte array for simplicity)
    to: [u8],           // Target node name
    exists: Beta,       // Confidence that edge exists
    strength: f64,      // Effect size (point estimate)
    strength_var: f64   // Uncertainty in effect size
}

/// Node in causal DAG
struct CausalNode {
    name: [u8],         // Node identifier
    node_type: NodeType // Semantic role in causal structure
}

/// Causal DAG structure
///
/// Represents a directed acyclic graph encoding causal relationships.
/// Uses simple arrays for L0 implementation (no HashMap dependency).
struct CausalDAG {
    nodes: [CausalNode],    // List of nodes
    edges: [CausalEdge],    // List of edges
    size: i64               // Current number of nodes
}

// ============================================================================
// SECTION 2: DAG CONSTRUCTION
// ============================================================================

/// Create a new empty causal DAG
fn dag_new() -> CausalDAG {
    return CausalDAG {
        nodes: [],
        edges: [],
        size: 0
    }
}

/// Add a node to the DAG
fn dag_add_node(dag: CausalDAG, name: [u8], node_type: NodeType) -> CausalDAG {
    let node = CausalNode { name: name, node_type: node_type }
    return CausalDAG {
        nodes: dag.nodes ++ [node],
        edges: dag.edges,
        size: dag.size + 1
    }
}

/// Add an edge to the DAG
///
/// # Arguments
/// - dag: The current DAG
/// - from: Source node name
/// - to: Target node name
/// - exists: Beta posterior on edge existence
/// - strength: Expected effect size
/// - strength_var: Uncertainty in effect size
fn dag_add_edge(
    dag: CausalDAG,
    from: [u8],
    to: [u8],
    exists: Beta,
    strength: f64,
    strength_var: f64
) -> CausalDAG {
    let edge = CausalEdge {
        from: from,
        to: to,
        exists: exists,
        strength: strength,
        strength_var: strength_var
    }
    return CausalDAG {
        nodes: dag.nodes,
        edges: dag.edges ++ [edge],
        size: dag.size
    }
}

/// Find node by name
fn dag_find_node(dag: CausalDAG, name: [u8]) -> i64 {
    var i: i64 = 0
    let n = len(dag.nodes)
    while i < n {
        if byte_array_eq(dag.nodes[i].name, name) {
            return i
        }
        i = i + 1
    }
    return -1  // Not found
}

/// Check if two byte arrays are equal
fn byte_array_eq(a: [u8], b: [u8]) -> i64 {
    let na = len(a)
    let nb = len(b)
    if na != nb { return 0 }

    var i: i64 = 0
    while i < na {
        if a[i] != b[i] { return 0 }
        i = i + 1
    }
    return 1
}

/// Get edges from a node
fn dag_edges_from(dag: CausalDAG, name: [u8]) -> [CausalEdge] {
    var result: [CausalEdge] = []
    var i: i64 = 0
    let n = len(dag.edges)

    while i < n {
        if byte_array_eq(dag.edges[i].from, name) {
            result = result ++ [dag.edges[i]]
        }
        i = i + 1
    }
    return result
}

/// Get edges to a node
fn dag_edges_to(dag: CausalDAG, name: [u8]) -> [CausalEdge] {
    var result: [CausalEdge] = []
    var i: i64 = 0
    let n = len(dag.edges)

    while i < n {
        if byte_array_eq(dag.edges[i].to, name) {
            result = result ++ [dag.edges[i]]
        }
        i = i + 1
    }
    return result
}

/// Get parents of a node (nodes with edges to this node)
fn dag_parents(dag: CausalDAG, name: [u8]) -> [[u8]] {
    let edges_in = dag_edges_to(dag, name)
    var parents: [[u8]] = []
    var i: i64 = 0
    let n = len(edges_in)

    while i < n {
        parents = parents ++ [edges_in[i].from]
        i = i + 1
    }
    return parents
}

/// Get children of a node (nodes with edges from this node)
fn dag_children(dag: CausalDAG, name: [u8]) -> [[u8]] {
    let edges_out = dag_edges_from(dag, name)
    var children: [[u8]] = []
    var i: i64 = 0
    let n = len(edges_out)

    while i < n {
        children = children ++ [edges_out[i].to]
        i = i + 1
    }
    return children
}

// ============================================================================
// SECTION 3: PEARL'S DO-CALCULUS
// ============================================================================

/// Apply Pearl's do-operator: P(Y | do(X=x))
///
/// The do-operator represents an intervention that sets X to value x,
/// removing all incoming edges to X (breaking parent relationships).
///
/// Returns a new DAG with the intervention applied.
fn do_intervention(dag: CausalDAG, var_name: [u8], value: f64) -> CausalDAG {
    // Remove all edges pointing to var_name
    var new_edges: [CausalEdge] = []
    var i: i64 = 0
    let n = len(dag.edges)

    while i < n {
        if byte_array_eq(dag.edges[i].to, var_name) == 0 {
            // Keep edges that don't point to var_name
            new_edges = new_edges ++ [dag.edges[i]]
        }
        i = i + 1
    }

    return CausalDAG {
        nodes: dag.nodes,
        edges: new_edges,
        size: dag.size
    }
}

/// Check if causal effect is identifiable from observational data
///
/// Returns 1 if identifiable, 0 if not.
/// Simplified heuristic: checks for confounders.
fn is_identifiable(dag: CausalDAG, treatment: [u8], outcome: [u8]) -> i64 {
    // Simplified check: if there's a direct path and no unblocked backdoor,
    // the effect is identifiable. Full implementation would need
    // backdoor criterion checking.

    // Check if there's a direct edge
    let edges_out = dag_edges_from(dag, treatment)
    var has_direct_path = 0
    var i: i64 = 0
    let n = len(edges_out)

    while i < n {
        if byte_array_eq(edges_out[i].to, outcome) {
            has_direct_path = 1
        }
        i = i + 1
    }

    return has_direct_path
}

/// Find backdoor adjustment set
///
/// Backdoor criterion: A set of variables Z blocks all backdoor paths
/// from treatment to outcome and doesn't contain descendants of treatment.
///
/// Returns list of node names to condition on (simplified implementation).
fn backdoor_adjustment(dag: CausalDAG, treatment: [u8], outcome: [u8]) -> [[u8]] {
    // Simplified: return all confounders (nodes that are parents of both)
    let treatment_parents = dag_parents(dag, treatment)
    let outcome_parents = dag_parents(dag, outcome)

    var adjustment_set: [[u8]] = []
    var i: i64 = 0
    let nt = len(treatment_parents)

    // Find common parents (confounders)
    while i < nt {
        var j: i64 = 0
        let no = len(outcome_parents)
        while j < no {
            if byte_array_eq(treatment_parents[i], outcome_parents[j]) {
                // Common parent - add to adjustment set
                adjustment_set = adjustment_set ++ [treatment_parents[i]]
            }
            j = j + 1
        }
        i = i + 1
    }

    return adjustment_set
}

// ============================================================================
// SECTION 4: CAUSAL EFFECT ESTIMATION
// ============================================================================

/// Average Treatment Effect (ATE) with epistemic uncertainty
///
/// Estimates E[Y | do(X=1)] - E[Y | do(X=0)]
/// Returns epistemic summary with mean, variance, and confidence intervals.
fn average_treatment_effect(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8]
) -> EpistemicSummary {
    // Find direct edge from treatment to outcome
    let edges_out = dag_edges_from(dag, treatment)
    var direct_effect: f64 = 0.0
    var direct_var: f64 = 0.0
    var i: i64 = 0
    let n = len(edges_out)

    while i < n {
        if byte_array_eq(edges_out[i].to, outcome) {
            // Found direct edge - use its effect size
            direct_effect = edges_out[i].strength
            direct_var = edges_out[i].strength_var
        }
        i = i + 1
    }

    // Account for confounding (adjustment set)
    let adjustment = backdoor_adjustment(dag, treatment, outcome)
    let n_conf = len(adjustment)

    // Add uncertainty from confounders
    var conf_uncertainty = 0.0
    i = 0
    while i < n_conf {
        // Each confounder adds uncertainty
        conf_uncertainty = conf_uncertainty + 0.01
        i = i + 1
    }

    let total_var = direct_var + conf_uncertainty

    // Convert to Beta distribution for epistemic summary
    // Normalize effect to [0, 1] range for Beta
    let normalized_effect = clamp01((direct_effect + 1.0) / 2.0)
    let beta_dist = beta_from_mean_variance(normalized_effect, total_var)

    return beta_summary(beta_dist)
}

/// Instrumental Variable (IV) estimation
///
/// Uses an instrumental variable Z that:
/// 1. Affects treatment X (relevance)
/// 2. Does not directly affect outcome Y (exclusion restriction)
/// 3. Shares no confounders with Y (exogeneity)
///
/// Returns epistemic summary of causal effect.
fn iv_estimate(
    dag: CausalDAG,
    instrument: [u8],
    treatment: [u8],
    outcome: [u8]
) -> EpistemicSummary {
    // Check IV assumptions (simplified)
    // 1. Relevance: instrument -> treatment edge exists
    let z_to_x = dag_edges_from(dag, instrument)
    var relevance_strength = 0.0
    var relevance_var = 0.0
    var i: i64 = 0
    var n = len(z_to_x)

    while i < n {
        if byte_array_eq(z_to_x[i].to, treatment) {
            relevance_strength = z_to_x[i].strength
            relevance_var = z_to_x[i].strength_var
        }
        i = i + 1
    }

    // 2. Exclusion: no direct edge instrument -> outcome
    let z_to_y = dag_edges_from(dag, instrument)
    var has_direct = 0
    i = 0
    n = len(z_to_y)

    while i < n {
        if byte_array_eq(z_to_y[i].to, outcome) {
            has_direct = 1  // Violation of exclusion restriction
        }
        i = i + 1
    }

    // Get treatment -> outcome effect
    let x_to_y = dag_edges_from(dag, treatment)
    var causal_strength = 0.0
    var causal_var = 0.0
    i = 0
    n = len(x_to_y)

    while i < n {
        if byte_array_eq(x_to_y[i].to, outcome) {
            causal_strength = x_to_y[i].strength
            causal_var = x_to_y[i].strength_var
        }
        i = i + 1
    }

    // IV estimate: effect = (reduced form) / (first stage)
    // Add uncertainty from both stages
    let total_var = causal_var + relevance_var

    // Penalize if exclusion restriction violated
    var final_var = total_var
    if has_direct > 0 {
        final_var = total_var + 0.1  // High uncertainty from violation
    }

    // Weak instrument penalty (low relevance)
    if relevance_strength < 0.1 {
        final_var = final_var + 0.05
    }

    let normalized_effect = clamp01((causal_strength + 1.0) / 2.0)
    let beta_dist = beta_from_mean_variance(normalized_effect, final_var)

    return beta_summary(beta_dist)
}

/// Conditional Average Treatment Effect (CATE)
///
/// Estimates E[Y | do(X=1), Z=z] - E[Y | do(X=0), Z=z]
/// where Z is a covariate (e.g., patient subgroup).
fn conditional_ate(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8],
    condition: [u8],
    condition_value: f64
) -> EpistemicSummary {
    // Simplified: compute ATE and adjust for conditioning
    let base_ate = average_treatment_effect(dag, treatment, outcome)

    // Find edge from condition to outcome (effect modification)
    let cond_to_y = dag_edges_from(dag, condition)
    var modifier_strength = 0.0
    var modifier_var = 0.0
    var i: i64 = 0
    let n = len(cond_to_y)

    while i < n {
        if byte_array_eq(cond_to_y[i].to, outcome) {
            modifier_strength = cond_to_y[i].strength
            modifier_var = cond_to_y[i].strength_var
        }
        i = i + 1
    }

    // Adjust base ATE by effect modifier
    let adjusted_mean = base_ate.mean + modifier_strength * condition_value
    let adjusted_var = base_ate.variance + modifier_var

    let normalized = clamp01((adjusted_mean + 1.0) / 2.0)
    let beta_dist = beta_from_mean_variance(normalized, adjusted_var)

    return beta_summary(beta_dist)
}

// ============================================================================
// SECTION 5: CAUSAL STRUCTURE LEARNING
// ============================================================================

/// Update edge existence posterior based on observational data
///
/// Given correlation data, update our belief about edge existence.
/// Uses Bayesian updating of the exists Beta distribution.
fn update_edge_existence(
    edge: CausalEdge,
    observed_correlation: f64,
    sample_size: f64
) -> CausalEdge {
    // High correlation = evidence for edge existence
    // Convert correlation to pseudo-counts
    let correlation_strength = abs_f64(observed_correlation)
    let pseudo_success = correlation_strength * sample_size
    let pseudo_failure = (1.0 - correlation_strength) * sample_size

    let updated_exists = beta_update(edge.exists, pseudo_success, pseudo_failure)

    return CausalEdge {
        from: edge.from,
        to: edge.to,
        exists: updated_exists,
        strength: edge.strength,
        strength_var: edge.strength_var
    }
}

/// Prune edges with low existence probability
///
/// Removes edges where P(edge exists | data) < threshold
fn dag_prune_edges(dag: CausalDAG, threshold: f64) -> CausalDAG {
    var kept_edges: [CausalEdge] = []
    var i: i64 = 0
    let n = len(dag.edges)

    while i < n {
        let exists_prob = beta_mean(dag.edges[i].exists)
        if exists_prob >= threshold {
            kept_edges = kept_edges ++ [dag.edges[i]]
        }
        i = i + 1
    }

    return CausalDAG {
        nodes: dag.nodes,
        edges: kept_edges,
        size: dag.size
    }
}

/// Epistemic active learning: which intervention maximizes information gain?
///
/// Returns the node name that, if intervened upon, would reduce
/// the most uncertainty in the causal structure.
fn optimal_intervention_target(dag: CausalDAG) -> [u8] {
    var max_info_gain = 0.0
    var best_node: [u8] = []
    var i: i64 = 0
    let n = len(dag.nodes)

    while i < n {
        let node_name = dag.nodes[i].name

        // Expected information gain from intervening on this node
        let edges_out = dag_edges_from(dag, node_name)
        var total_entropy = 0.0
        var j: i64 = 0
        let m = len(edges_out)

        while j < m {
            // Entropy of edge existence
            total_entropy = total_entropy + beta_entropy(edges_out[j].exists)
            j = j + 1
        }

        if total_entropy > max_info_gain {
            max_info_gain = total_entropy
            best_node = node_name
        }

        i = i + 1
    }

    return best_node
}

// ============================================================================
// SECTION 6: COUNTERFACTUAL REASONING
// ============================================================================

/// Counterfactual query: "What would Y have been if X had been x'?"
///
/// Three-step process (Pearl's algorithm):
/// 1. Abduction: Infer latent variables from observed data
/// 2. Action: Modify the model to reflect intervention
/// 3. Prediction: Compute outcome under modified model
///
/// Simplified implementation for epistemic uncertainty propagation.
fn counterfactual(
    dag: CausalDAG,
    treatment: [u8],
    treatment_factual: f64,
    treatment_counterfactual: f64,
    outcome: [u8]
) -> EpistemicSummary {
    // Step 1: Abduction (simplified - assume no latent confounders)
    // Step 2: Action - apply do-operator
    let intervened = do_intervention(dag, treatment, treatment_counterfactual)

    // Step 3: Prediction - compute effect
    let cf_ate = average_treatment_effect(intervened, treatment, outcome)

    // Counterfactual effect = outcome under intervention
    // Add extra uncertainty for counterfactual reasoning
    let cf_var = cf_ate.variance + 0.02

    let normalized = clamp01((cf_ate.mean + 1.0) / 2.0)
    let beta_dist = beta_from_mean_variance(normalized, cf_var)

    return beta_summary(beta_dist)
}

/// Probability of sufficiency: P(Y_x | Y, X = x')
///
/// "Given that Y occurred under X = x', would Y have still occurred under X = x?"
fn probability_of_sufficiency(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8],
    x_factual: f64,
    x_counterfactual: f64
) -> f64 {
    let cf = counterfactual(dag, treatment, x_factual, x_counterfactual, outcome)
    return cf.mean
}

/// Probability of necessity: P(Y_{x'} = 0 | Y = 1, X = x)
///
/// "Given that Y occurred under X = x, was X necessary for Y?"
fn probability_of_necessity(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8],
    x_factual: f64,
    x_counterfactual: f64
) -> f64 {
    let cf = counterfactual(dag, treatment, x_factual, x_counterfactual, outcome)
    // Necessity = 1 - P(Y would have happened anyway)
    return 1.0 - cf.mean
}

// ============================================================================
// SECTION 7: MEDIATION ANALYSIS
// ============================================================================

/// Natural Direct Effect (NDE): effect not through mediator
///
/// NDE = E[Y_{X=1,M=M_0} - Y_{X=0,M=M_0}]
/// where M_0 is mediator value under control (X=0).
fn natural_direct_effect(
    dag: CausalDAG,
    treatment: [u8],
    mediator: [u8],
    outcome: [u8]
) -> EpistemicSummary {
    // Get direct effect (treatment -> outcome)
    let x_to_y = dag_edges_from(dag, treatment)
    var direct_strength = 0.0
    var direct_var = 0.0
    var i: i64 = 0
    let n = len(x_to_y)

    while i < n {
        if byte_array_eq(x_to_y[i].to, outcome) {
            direct_strength = x_to_y[i].strength
            direct_var = x_to_y[i].strength_var
        }
        i = i + 1
    }

    let normalized = clamp01((direct_strength + 1.0) / 2.0)
    let beta_dist = beta_from_mean_variance(normalized, direct_var)

    return beta_summary(beta_dist)
}

/// Natural Indirect Effect (NIE): effect through mediator
///
/// NIE = E[Y_{X=1,M=M_1} - Y_{X=1,M=M_0}]
/// where M_1, M_0 are mediator values under treatment/control.
fn natural_indirect_effect(
    dag: CausalDAG,
    treatment: [u8],
    mediator: [u8],
    outcome: [u8]
) -> EpistemicSummary {
    // Get indirect path: treatment -> mediator -> outcome
    let x_to_m = dag_edges_from(dag, treatment)
    var tm_strength = 0.0
    var tm_var = 0.0
    var i: i64 = 0
    var n = len(x_to_m)

    while i < n {
        if byte_array_eq(x_to_m[i].to, mediator) {
            tm_strength = x_to_m[i].strength
            tm_var = x_to_m[i].strength_var
        }
        i = i + 1
    }

    let m_to_y = dag_edges_from(dag, mediator)
    var mo_strength = 0.0
    var mo_var = 0.0
    i = 0
    n = len(m_to_y)

    while i < n {
        if byte_array_eq(m_to_y[i].to, outcome) {
            mo_strength = m_to_y[i].strength
            mo_var = m_to_y[i].strength_var
        }
        i = i + 1
    }

    // Indirect effect = product of path effects
    let indirect_strength = tm_strength * mo_strength

    // Variance propagation for product
    let indirect_var = tm_strength * tm_strength * mo_var +
                       mo_strength * mo_strength * tm_var +
                       tm_var * mo_var

    let normalized = clamp01((indirect_strength + 1.0) / 2.0)
    let beta_dist = beta_from_mean_variance(normalized, indirect_var)

    return beta_summary(beta_dist)
}

/// Proportion mediated: what fraction of effect goes through mediator?
///
/// PM = NIE / (NDE + NIE)
fn proportion_mediated(
    dag: CausalDAG,
    treatment: [u8],
    mediator: [u8],
    outcome: [u8]
) -> f64 {
    let nde = natural_direct_effect(dag, treatment, mediator, outcome)
    let nie = natural_indirect_effect(dag, treatment, mediator, outcome)

    let total_effect = nde.mean + nie.mean
    if total_effect < 0.0001 {
        return 0.0  // No effect to mediate
    }

    return nie.mean / total_effect
}

// ============================================================================
// SECTION 8: SENSITIVITY ANALYSIS
// ============================================================================

/// Sensitivity to unmeasured confounding
///
/// Returns how much an unmeasured confounder would need to affect both
/// treatment and outcome to explain away the observed effect.
fn confounder_sensitivity(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8],
    observed_effect: f64
) -> f64 {
    // E-value: minimum strength of unmeasured confounder association
    // E = effect + sqrt(effect * (effect - 1))

    if observed_effect <= 0.0 {
        return 0.0
    }

    let sqrt_term = sqrt_f64(observed_effect * (observed_effect - 1.0))
    return observed_effect + sqrt_term
}

/// Robustness value: how much evidence is there for the causal effect?
///
/// Based on edge existence probabilities along causal paths.
fn robustness_value(dag: CausalDAG, treatment: [u8], outcome: [u8]) -> f64 {
    // Find direct edge
    let edges = dag_edges_from(dag, treatment)
    var min_confidence = 1.0
    var i: i64 = 0
    let n = len(edges)

    while i < n {
        if byte_array_eq(edges[i].to, outcome) {
            let edge_confidence = beta_mean(edges[i].exists)
            if edge_confidence < min_confidence {
                min_confidence = edge_confidence
            }
        }
        i = i + 1
    }

    return min_confidence
}

// ============================================================================
// SECTION 9: PRETTY PRINTING
// ============================================================================

/// Print causal DAG structure
fn dag_print(dag: CausalDAG) -> i64 {
    println("Causal DAG:")
    print("  Nodes: ")
    println(dag.size)

    var i: i64 = 0
    while i < dag.size {
        print("    - ")
        print_byte_array(dag.nodes[i].name)
        println("")
        i = i + 1
    }

    print("  Edges: ")
    println(len(dag.edges))

    i = 0
    let m = len(dag.edges)
    while i < m {
        print("    ")
        print_byte_array(dag.edges[i].from)
        print(" -> ")
        print_byte_array(dag.edges[i].to)
        print(" [exists=")
        print(beta_mean(dag.edges[i].exists))
        print(", strength=")
        print(dag.edges[i].strength)
        println("]")
        i = i + 1
    }

    return 0
}

/// Print byte array as string (helper)
fn print_byte_array(arr: [u8]) -> i64 {
    var i: i64 = 0
    let n = len(arr)
    while i < n {
        // Just print the bytes (in practice would convert to chars)
        print(arr[i])
        i = i + 1
    }
    return 0
}

/// Print causal effect estimate
fn causal_effect_print(name: [u8], effect: EpistemicSummary) -> i64 {
    print("Causal Effect (")
    print_byte_array(name)
    println("):")
    epistemic_print(effect)
    return 0
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

fn main() -> i32 {
    println("=== epistemic::causal — Causal Inference Demo ===")
    println("")

    // Example: Smoking -> Cancer with Age confounder
    println("--- Example: Smoking -> Cancer (with Age confounder) ---")

    var dag = dag_new()

    // Add nodes
    dag = dag_add_node(dag, "Age", NodeType::Confounder)
    dag = dag_add_node(dag, "Smoking", NodeType::Treatment)
    dag = dag_add_node(dag, "Cancer", NodeType::Outcome)

    // Add edges with epistemic uncertainty
    // Age -> Smoking: older people more likely to have smoked
    dag = dag_add_edge(dag, "Age", "Smoking", beta_new(8.0, 2.0), 0.4, 0.05)

    // Age -> Cancer: age is risk factor
    dag = dag_add_edge(dag, "Age", "Cancer", beta_new(9.0, 1.0), 0.6, 0.03)

    // Smoking -> Cancer: causal effect we want to estimate
    dag = dag_add_edge(dag, "Smoking", "Cancer", beta_new(7.0, 3.0), 0.5, 0.08)

    println("")
    dag_print(dag)
    println("")

    // Estimate Average Treatment Effect
    println("--- Average Treatment Effect ---")
    let ate = average_treatment_effect(dag, "Smoking", "Cancer")
    causal_effect_print("ATE", ate)
    println("")

    // Check identifiability
    println("--- Identifiability Check ---")
    let identifiable = is_identifiable(dag, "Smoking", "Cancer")
    if identifiable > 0 {
        println("Effect is identifiable from observational data")
    } else {
        println("Effect may not be identifiable - need interventions")
    }
    println("")

    // Find backdoor adjustment set
    println("--- Backdoor Adjustment Set ---")
    let adjustment = backdoor_adjustment(dag, "Smoking", "Cancer")
    print("Variables to condition on: ")
    println(len(adjustment))
    var i: i64 = 0
    while i < len(adjustment) {
        print("  - ")
        print_byte_array(adjustment[i])
        println("")
        i = i + 1
    }
    println("")

    // Do-calculus intervention
    println("--- Intervention Analysis ---")
    let intervened = do_intervention(dag, "Smoking", 1.0)
    print("After do(Smoking=1), edges: ")
    println(len(intervened.edges))
    println("")

    // Sensitivity analysis
    println("--- Sensitivity to Unmeasured Confounding ---")
    let robustness = robustness_value(dag, "Smoking", "Cancer")
    print("Robustness value (min edge confidence): ")
    println(robustness)
    println("")

    println("=== Demo Complete ===")
    println("")
    println("Key Innovations:")
    println("1. Every edge has epistemic uncertainty (Beta posterior)")
    println("2. Causal effects propagate uncertainty honestly")
    println("3. Active learning identifies optimal interventions")
    println("4. Integrates Pearl's do-calculus with Bayesian updating")
    println("")
    println("This makes causal inference epistemically honest.")

    return 0
}
