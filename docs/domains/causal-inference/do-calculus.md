# Do-Calculus: Pearl's Rules for Causal Inference

Do-calculus is a complete system for deriving causal effects from observational data. It provides the mathematical foundation for answering interventional questions using non-experimental data.

## Pearl's Do-Operator

The `do()` operator represents an intervention that sets a variable to a specific value, breaking all causal influences on that variable.

### Observational vs Interventional

```sio
// Observational: P(Y | X=1)
// "What is Y among cases where X happens to be 1?"
let observational = conditional_probability(Y, X == 1)

// Interventional: P(Y | do(X=1))
// "What is Y if we SET X to 1?"
let interventional = do_intervention(dag, "X", 1.0)
```

The key difference: observational data includes selection bias and confounding, while interventional data represents the true causal effect.

### Graphical Interpretation

The do-operator modifies the causal graph by removing all incoming edges to the intervened variable:

```sio
use std::causal::{CausalDAG, do_intervention, NodeType}

// Original graph:
//   U
//  / \
// v   v
// X   Y
//  \ /
//   v

var dag = dag_new()
dag = dag_add_node(dag, "U", NodeType::Confounder)
dag = dag_add_node(dag, "X", NodeType::Treatment)
dag = dag_add_node(dag, "Y", NodeType::Outcome)
dag = dag_add_edge(dag, "U", "X", beta_new(8.0, 2.0), 0.5, 0.1)
dag = dag_add_edge(dag, "U", "Y", beta_new(8.0, 2.0), 0.3, 0.1)
dag = dag_add_edge(dag, "X", "Y", beta_new(7.0, 3.0), 0.4, 0.1)

// After do(X=1):
//   U
//    \
//     v
// X   Y  (U->X edge is removed!)
//  \ /
//   v

let dag_intervened = do_intervention(dag, "X", 1.0)
// X now has no parents - it's been "set" to 1
```

## The Three Rules of Do-Calculus

Do-calculus consists of three rules that, when applicable, allow manipulating expressions containing the do-operator. Together, they are complete: any identifiable causal effect can be derived using these rules.

### Rule 1: Insertion/Deletion of Observations

**Statement**: P(Y | do(X), Z, W) = P(Y | do(X), W) if (Y perp Z | X, W) in G_X_bar

This rule says we can add or remove an observation Z if Y is independent of Z given (X, W) in the graph where incoming edges to X are removed.

```sio
// G_X_bar = graph with incoming edges to X removed

// If Y is d-separated from Z given X and W in G_X_bar,
// then conditioning on Z doesn't change the interventional distribution

fn can_apply_rule_1(
    dag: CausalDAG,
    y: [u8],
    x: [u8],
    z: [u8],
    w: [[u8]]
) -> bool {
    // Remove incoming edges to X
    let g_x_bar = remove_incoming_edges(dag, x)

    // Check d-separation: Y perp Z | {X, W}
    var conditioning = w ++ [[x]]
    return d_separated(g_x_bar, y, z, conditioning)
}
```

### Rule 2: Action/Observation Exchange

**Statement**: P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if (Y perp Z | X, W) in G_X_bar_Z

This rule allows converting an intervention do(Z) to an observation Z if Y is independent of Z given (X, W) in the graph where incoming edges to X are removed AND outgoing edges from Z are removed.

```sio
// G_X_bar_Z = G with incoming edges to X removed
//             AND outgoing edges from Z removed

fn can_apply_rule_2(
    dag: CausalDAG,
    y: [u8],
    x: [u8],
    z: [u8],
    w: [[u8]]
) -> bool {
    // Remove incoming edges to X
    let g1 = remove_incoming_edges(dag, x)
    // Remove outgoing edges from Z
    let g2 = remove_outgoing_edges(g1, z)

    // Check d-separation: Y perp Z | {X, W}
    var conditioning = w ++ [[x]]
    return d_separated(g2, y, z, conditioning)
}
```

### Rule 3: Insertion/Deletion of Actions

**Statement**: P(Y | do(X), do(Z), W) = P(Y | do(X), W) if (Y perp Z | X, W) in G_X_bar_Z_bar(W)

This rule allows removing an intervention do(Z) entirely if Y is independent of Z given (X, W) in a specific modified graph.

```sio
// G_X_bar_Z_bar(W) = G with:
//   - Incoming edges to X removed
//   - Incoming edges to Z removed (except those through W ancestors)

fn can_apply_rule_3(
    dag: CausalDAG,
    y: [u8],
    x: [u8],
    z: [u8],
    w: [[u8]]
) -> bool {
    // Complex graph manipulation
    let g_modified = build_rule3_graph(dag, x, z, w)

    // Check d-separation
    var conditioning = w ++ [[x]]
    return d_separated(g_modified, y, z, conditioning)
}
```

## The Backdoor Criterion

The backdoor criterion is a special case that identifies when we can estimate P(Y | do(X)) by adjusting for confounders.

### Definition

A set of variables Z satisfies the backdoor criterion relative to (X, Y) if:
1. No node in Z is a descendant of X
2. Z blocks every path between X and Y that contains an arrow into X

### Backdoor Adjustment Formula

When Z satisfies the backdoor criterion:

P(Y | do(X)) = sum_z P(Y | X, Z) * P(Z)

```sio
use std::causal::{backdoor_adjustment, average_treatment_effect}

fn estimate_with_backdoor(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8]
) -> EpistemicSummary {
    // Find valid adjustment set
    let adjustment_set = backdoor_adjustment(dag, treatment, outcome)

    if adjustment_set.len() > 0 {
        // Adjustment set found - effect is identifiable
        print("Adjusting for confounders: ")
        print_adjustment_set(adjustment_set)

        // Compute adjusted effect estimate
        return average_treatment_effect(dag, treatment, outcome)
    } else {
        // No adjustment needed (no confounding)
        return average_treatment_effect(dag, treatment, outcome)
    }
}
```

### Example: Smoking and Lung Cancer

```sio
// Classic example: Does smoking cause lung cancer?
//
//   Genetics
//   /      \
//  v        v
// Smoking  Lung Cancer
//   |          ^
//   +----------+

var dag = dag_new()
dag = dag_add_node(dag, "genetics", NodeType::Confounder)
dag = dag_add_node(dag, "smoking", NodeType::Treatment)
dag = dag_add_node(dag, "cancer", NodeType::Outcome)

// Genetics affects both smoking and cancer
dag = dag_add_edge(dag, "genetics", "smoking", beta_new(7.0, 3.0), 0.3, 0.1)
dag = dag_add_edge(dag, "genetics", "cancer", beta_new(6.0, 4.0), 0.2, 0.1)
// Direct causal effect of smoking
dag = dag_add_edge(dag, "smoking", "cancer", beta_new(8.0, 2.0), 0.5, 0.15)

// Backdoor path: smoking <- genetics -> cancer
// Adjustment set: {genetics}
let adjustment = backdoor_adjustment(dag, "smoking", "cancer")
// Returns: ["genetics"]

// After adjusting for genetics, we can identify the causal effect
let causal_effect = average_treatment_effect(dag, "smoking", "cancer")
```

## The Frontdoor Criterion

When there are unmeasured confounders, the frontdoor criterion can sometimes identify causal effects through mediating variables.

### Definition

A set of variables M satisfies the frontdoor criterion relative to (X, Y) if:
1. M intercepts all directed paths from X to Y
2. There is no unblocked backdoor path from X to M
3. All backdoor paths from M to Y are blocked by X

### Frontdoor Adjustment Formula

When M satisfies the frontdoor criterion:

P(Y | do(X)) = sum_m P(M=m | X) * sum_x' P(Y | M=m, X=x') * P(X=x')

```sio
// Example: Effect of smoking on cancer via tar deposits
//
//     U (unmeasured genetic factor)
//   /   \
//  v     v
// Smoking -> Tar -> Cancer
//
// We cannot measure U, but we CAN measure Tar

var dag = dag_new()
dag = dag_add_node(dag, "U", NodeType::Confounder)
dag = dag_add_node(dag, "smoking", NodeType::Treatment)
dag = dag_add_node(dag, "tar", NodeType::Mediator)
dag = dag_add_node(dag, "cancer", NodeType::Outcome)

dag = dag_add_edge(dag, "U", "smoking", beta_new(7.0, 3.0), 0.3, 0.1)
dag = dag_add_edge(dag, "U", "cancer", beta_new(6.0, 4.0), 0.2, 0.1)
dag = dag_add_edge(dag, "smoking", "tar", beta_new(9.0, 1.0), 0.8, 0.05)
dag = dag_add_edge(dag, "tar", "cancer", beta_new(8.0, 2.0), 0.6, 0.1)

// Tar satisfies frontdoor criterion
// We can estimate the effect even without measuring U!
```

## Implementing Do-Calculus in Sounio

### Complete Identification Algorithm

```sio
use std::causal::{
    CausalDAG,
    is_identifiable,
    backdoor_adjustment,
    average_treatment_effect,
    EpistemicSummary
}

/// Attempt to identify causal effect P(Y | do(X))
fn identify_causal_effect(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8]
) -> EpistemicSummary with Prob {
    // Check if effect is identifiable
    if is_identifiable(dag, treatment, outcome) == 0 {
        // Effect not identifiable - return high uncertainty
        return EpistemicSummary {
            mean: 0.0,
            variance: 1.0,
            confidence: 0.0,
            lower: -1.0,
            upper: 1.0,
        }
    }

    // Try backdoor adjustment first
    let backdoor_set = backdoor_adjustment(dag, treatment, outcome)
    if backdoor_set.len() > 0 || check_no_confounding(dag, treatment, outcome) {
        return average_treatment_effect(dag, treatment, outcome)
    }

    // Try instrumental variable
    let instruments = find_instruments(dag, treatment, outcome)
    if instruments.len() > 0 {
        return iv_estimate(dag, instruments[0], treatment, outcome)
    }

    // Effect identified but no simple formula - use general algorithm
    return general_identification(dag, treatment, outcome)
}

fn check_no_confounding(dag: CausalDAG, treatment: [u8], outcome: [u8]) -> bool {
    // Check if there are any backdoor paths
    let backdoor_set = backdoor_adjustment(dag, treatment, outcome)
    return backdoor_set.len() == 0
}
```

### Handling Non-Identifiability

When effects are not identifiable, Sounio returns results with high uncertainty:

```sio
// Unobserved confounder case
//
//     U (unobserved)
//   /   \
//  v     v
//  X     Y
//   \   /
//    \ /
//     v

// Without measuring U, P(Y|do(X)) is NOT identifiable
let effect = identify_causal_effect(dag, "X", "Y")

// effect.confidence will be very low
// effect.variance will be very high
if effect.confidence < 0.5 {
    print("Warning: Causal effect may not be identifiable")
    print("Consider: collecting data on confounders")
    print("Or: using sensitivity analysis")
}
```

## Sensitivity Analysis

When identification is uncertain, sensitivity analysis bounds the possible causal effect:

```sio
use std::causal::{confounder_sensitivity, robustness_value}

fn analyze_sensitivity(
    dag: CausalDAG,
    treatment: [u8],
    outcome: [u8],
    observed_effect: f64
) -> f64 {
    // E-value: How strong would confounding need to be to explain away the effect?
    let e_value = confounder_sensitivity(dag, treatment, outcome, observed_effect)

    print("E-value: ")
    print(e_value)
    print("\n")

    if e_value > 2.0 {
        print("Effect is robust: confounding would need to be very strong\n")
    } else {
        print("Effect may be sensitive to unmeasured confounding\n")
    }

    // Robustness value: minimum confidence in causal path
    let rv = robustness_value(dag, treatment, outcome)
    print("Robustness value: ")
    print(rv)
    print("\n")

    return e_value
}
```

## Practical Example: Medical Treatment Effect

```sio
use std::causal::{
    CausalDAG, NodeType, dag_new, dag_add_node, dag_add_edge,
    beta_new, backdoor_adjustment, average_treatment_effect,
    confounder_sensitivity, epistemic_print
}

fn analyze_drug_effect() -> i32 {
    // Build causal model for drug effect study
    //
    //   Age  Severity
    //    \   /  \
    //     v v    v
    //     Drug   Recovery
    //       \    ^
    //        \  /
    //         v/
    //       (effect)

    var dag = dag_new()

    // Add nodes
    dag = dag_add_node(dag, "age", NodeType::Confounder)
    dag = dag_add_node(dag, "severity", NodeType::Confounder)
    dag = dag_add_node(dag, "drug", NodeType::Treatment)
    dag = dag_add_node(dag, "recovery", NodeType::Outcome)

    // Add edges with uncertainty
    // Age affects drug prescription
    dag = dag_add_edge(dag, "age", "drug", beta_new(7.0, 3.0), 0.3, 0.08)
    // Severity affects drug and recovery
    dag = dag_add_edge(dag, "severity", "drug", beta_new(8.0, 2.0), 0.5, 0.1)
    dag = dag_add_edge(dag, "severity", "recovery", beta_new(9.0, 1.0), -0.4, 0.05)
    // Drug effect on recovery (what we want to estimate)
    dag = dag_add_edge(dag, "drug", "recovery", beta_new(6.0, 4.0), 0.35, 0.12)

    print("=== Drug Effect Analysis ===\n\n")

    // Find adjustment set
    let adjustment = backdoor_adjustment(dag, "drug", "recovery")
    print("Adjustment set: ")
    print_names(adjustment)
    print("\n")

    // Estimate causal effect
    let effect = average_treatment_effect(dag, "drug", "recovery")
    print("\nCausal effect (ATE):\n")
    epistemic_print(effect)

    // Interpret
    if effect.lower > 0.0 {
        print("\nConclusion: Drug has POSITIVE causal effect on recovery\n")
    } else if effect.upper < 0.0 {
        print("\nConclusion: Drug has NEGATIVE causal effect on recovery\n")
    } else {
        print("\nConclusion: Cannot determine sign of effect (CI includes 0)\n")
    }

    // Sensitivity analysis
    print("\nSensitivity Analysis:\n")
    let e_val = confounder_sensitivity(dag, "drug", "recovery", effect.mean)
    print("E-value: ")
    print(e_val)
    print("\n")

    return 0
}

fn print_names(names: [[u8]]) -> i32 {
    print("[")
    var i: i64 = 0
    while i < names.len() {
        if i > 0 { print(", ") }
        print_byte_array(names[i])
        i = i + 1
    }
    print("]")
    return 0
}

fn main() -> i32 {
    return analyze_drug_effect()
}
```

## Summary

Do-calculus provides a complete logical system for causal inference:

| Rule | Purpose | Graph Condition |
|------|---------|-----------------|
| Rule 1 | Add/remove observations | d-separation in G_X_bar |
| Rule 2 | Convert do() to observation | d-separation in G_X_bar_Z_ |
| Rule 3 | Remove do() entirely | d-separation in G_X_bar_Z_bar(W) |

| Criterion | When to Use | Formula |
|-----------|-------------|---------|
| Backdoor | Confounders observed | sum_z P(Y\|X,Z) P(Z) |
| Frontdoor | Mediator observed, confounder not | Frontdoor formula |
| IV | Instrument available | Wald estimator |

Sounio integrates do-calculus with epistemic uncertainty, ensuring that every causal claim carries quantified confidence in both the existence and strength of causal effects.
