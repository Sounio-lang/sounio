# Causal Inference in Sounio

Sounio provides first-class support for causal inference, integrating Pearl's do-calculus with epistemic uncertainty tracking. This enables rigorous causal reasoning where every causal claim carries quantified confidence.

## What is Causal Inference?

Causal inference is the science of determining cause-and-effect relationships from data. Unlike correlation (which only measures association), causal inference answers questions like:

- **Intervention**: What happens if we do X? (P(Y | do(X)))
- **Attribution**: Did X cause Y in this specific case?
- **Counterfactual**: What would have happened if X had been different?

### The Fundamental Problem

Correlation does not imply causation. Consider:

```sio
// Correlation: ice cream sales and drowning deaths are correlated
let correlation = pearson_correlation(ice_cream_sales, drowning_deaths)
// correlation ~ 0.8

// But ice cream does NOT cause drowning!
// Both are caused by a confounding variable: hot weather
```

Causal inference provides tools to distinguish genuine causal relationships from spurious correlations.

## The Ladder of Causation

Judea Pearl's causal hierarchy defines three levels of causal reasoning:

### Level 1: Association (Seeing)

Questions about what we observe:
- Query: P(Y | X) - "What is Y given that we see X?"
- Example: "Do patients who take the drug have better outcomes?"

```sio
// Level 1: Observational data
let observed = conditional_probability(Y, X)
// This tells us P(Y|X) but NOT P(Y|do(X))
```

### Level 2: Intervention (Doing)

Questions about the effects of actions:
- Query: P(Y | do(X)) - "What is Y if we set X to a value?"
- Example: "What would happen if we give the drug to this patient?"

```sio
use std::causal::{CausalDAG, do_intervention}

// Level 2: Intervention
let dag = build_medical_dag()
let intervened = do_intervention(dag, "treatment", 1.0)
let causal_effect = average_treatment_effect(dag, "treatment", "outcome")
```

### Level 3: Counterfactual (Imagining)

Questions about alternative worlds:
- Query: P(Y_x | X=x', Y=y) - "What would Y have been if X had been different?"
- Example: "Would this patient have recovered without the drug?"

```sio
use std::causal::{what_if, probability_of_necessity}

// Level 3: Counterfactual
let cf_result = what_if(
    dag,
    "treatment",        // intervention variable
    1.0,                // factual value
    0.0,                // counterfactual value
    "recovery"          // outcome
)

// Was the treatment necessary for recovery?
let pn = probability_of_necessity(dag, "treatment", "recovery", 1.0, 0.0)
```

## Sounio's Causal Computing Capabilities

### Epistemic Causal Edges

Every causal relationship in Sounio carries uncertainty about both its existence and its strength:

```sio
use std::causal::{CausalDAG, CausalEdge, Beta}

// Traditional approach: "X causes Y with effect 0.3"
// Sounio approach: "X causes Y with effect 0.3 +/- 0.12,
//                   and we're 85% confident this edge exists"

var dag = dag_new()
dag = dag_add_node(dag, "X", NodeType::Treatment)
dag = dag_add_node(dag, "Y", NodeType::Outcome)

// Edge with epistemic uncertainty
dag = dag_add_edge(
    dag,
    "X",              // from
    "Y",              // to
    beta_new(6.0, 4.0),  // 60% confidence edge exists (Beta posterior)
    0.3,              // effect size point estimate
    0.12              // uncertainty in effect size
)
```

### Automatic Uncertainty Propagation

Causal effects propagate uncertainty through the graph:

```sio
// Treatment effect carries uncertainty from:
// 1. Edge existence uncertainty
// 2. Effect size uncertainty
// 3. Confounding adjustment uncertainty

let effect = average_treatment_effect(dag, "treatment", "outcome")
// effect.mean = 0.35
// effect.variance = 0.08
// effect.confidence = 0.82
// effect.lower = 0.12  (95% CI lower bound)
// effect.upper = 0.58  (95% CI upper bound)

epistemic_print(effect)
// Output: Mean: 0.35 Var: 0.08 [0.12, 0.58]
```

### Confidence-Gated Decisions

Sounio enables confidence-aware causal reasoning:

```sio
let effect = average_treatment_effect(dag, "drug", "recovery")

if effect.confidence > 0.90 && effect.lower > 0.0 {
    // Strong evidence of positive effect
    recommend_treatment()
} else if effect.confidence > 0.80 {
    // Moderate evidence - consider clinical context
    recommend_with_caution()
} else {
    // Insufficient evidence
    recommend_further_study()
}
```

## Key Concepts

### Directed Acyclic Graphs (DAGs)

Causal DAGs represent causal structure:

```sio
// Classic confounding example
//     U (Confounder)
//    / \
//   v   v
//   X   Y
//    \  ^
//     \ |
//      v|
//     (direct effect)

var dag = dag_new()
dag = dag_add_node(dag, "U", NodeType::Confounder)
dag = dag_add_node(dag, "X", NodeType::Treatment)
dag = dag_add_node(dag, "Y", NodeType::Outcome)

dag = dag_add_edge(dag, "U", "X", beta_new(8.0, 2.0), 0.4, 0.05)
dag = dag_add_edge(dag, "U", "Y", beta_new(9.0, 1.0), 0.6, 0.03)
dag = dag_add_edge(dag, "X", "Y", beta_new(7.0, 3.0), 0.3, 0.08)
```

### Interventions

The do-operator removes incoming edges to a variable:

```sio
// Before intervention: X depends on U
// After do(X=1): X is set to 1, breaking U -> X edge

let intervened_dag = do_intervention(dag, "X", 1.0)
// Now we can estimate P(Y | do(X=1)) free of confounding
```

### Adjustment Sets

Variables to control for when estimating causal effects:

```sio
let adjustment = backdoor_adjustment(dag, "X", "Y")
// Returns: ["U"] - control for confounder U to identify effect
```

### Counterfactuals

What-if reasoning about alternative scenarios:

```sio
let cf = what_if(
    dag,
    "treatment",     // variable
    1.0,             // factual value (what actually happened)
    0.0,             // counterfactual value (what if this instead?)
    "outcome"        // target variable
)
// "What would the outcome have been if treatment had been 0 instead of 1?"
```

## Learning Path

### Beginner

1. [Do-Calculus](do-calculus.md) - Pearl's rules for identifying causal effects
2. Understanding DAGs and d-separation
3. Backdoor and frontdoor adjustment

### Intermediate

4. [Causal Discovery](causal-discovery.md) - Learning causal structure from data
5. Mediation analysis
6. Sensitivity analysis for unmeasured confounding

### Advanced

7. [Counterfactual Reasoning](counterfactuals.md) - Level 3 causal queries
8. Probability of causation (PN, PS, PNS)
9. Causal transportability and external validity

## Module Reference

```sio
use std::causal::{
    // Core types
    CausalDAG,
    CausalNode,
    CausalEdge,
    NodeType,
    Beta,
    EpistemicSummary,

    // DAG operations
    dag_new,
    dag_add_node,
    dag_add_edge,
    dag_parents,
    dag_children,

    // Do-calculus
    do_intervention,
    is_identifiable,
    backdoor_adjustment,

    // Effect estimation
    average_treatment_effect,
    conditional_ate,
    iv_estimate,

    // Mediation
    natural_direct_effect,
    natural_indirect_effect,
    proportion_mediated,

    // Counterfactuals
    what_if,
    probability_of_necessity,
    probability_of_sufficiency,

    // Sensitivity
    confounder_sensitivity,
    robustness_value,
}
```

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
- Pearl, J., Glymour, M., & Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*. Wiley.
- Hernan, M. A., & Robins, J. M. (2020). *Causal Inference: What If*. Chapman & Hall/CRC.
- Peters, J., Janzing, D., & Scholkopf, B. (2017). *Elements of Causal Inference*. MIT Press.
