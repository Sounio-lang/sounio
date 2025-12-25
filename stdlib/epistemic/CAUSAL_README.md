# Causal Inference with Epistemic Uncertainty

This module implements Pearl's causal inference framework with integrated Bayesian epistemic uncertainty tracking.

## Key Innovation

**Traditional causal inference:**
```
"The causal effect is 0.5"
```

**Demetrios epistemic causal inference:**
```
"The causal effect is Beta(6, 4), giving 0.6 ± 0.15 with epistemic uncertainty"
```

Every causal relationship carries:
- **Existence probability**: Beta posterior on P(edge exists | data)
- **Effect size**: Point estimate of causal effect magnitude
- **Effect uncertainty**: Epistemic variance in the effect size

## Core Concepts

### 1. Causal DAG with Uncertainty

```d
var dag = dag_new()
dag = dag_add_node(dag, "Treatment", NodeType::Treatment)
dag = dag_add_node(dag, "Outcome", NodeType::Outcome)

// Add edge with epistemic uncertainty
dag = dag_add_edge(dag, "Treatment", "Outcome",
                   beta_new(7.0, 3.0),  // Existence: Beta(7, 3)
                   0.5,                  // Effect size: 0.5
                   0.08)                 // Effect uncertainty: 0.08
```

### 2. Do-Calculus (Pearl's Intervention)

```d
// Apply intervention: do(X = 1)
let intervened_dag = do_intervention(dag, "X", 1.0)

// Estimate causal effect
let effect = average_treatment_effect(dag, "X", "Y")
// Returns EpistemicSummary with mean, variance, CI
```

### 3. Backdoor Adjustment

```d
// Find confounders to adjust for
let adjustment_set = backdoor_adjustment(dag, "Treatment", "Outcome")

// Check if effect is identifiable
let identifiable = is_identifiable(dag, "Treatment", "Outcome")
```

### 4. Mediation Analysis

```d
// Decompose effects into direct and indirect paths
let nde = natural_direct_effect(dag, "X", "M", "Y")
let nie = natural_indirect_effect(dag, "X", "M", "Y")
let prop_mediated = proportion_mediated(dag, "X", "M", "Y")
```

### 5. Instrumental Variables

```d
// Use instrument Z to estimate effect of X on Y
let iv_estimate = iv_estimate(dag, "Z", "X", "Y")
// Automatically checks IV assumptions
```

### 6. Counterfactual Reasoning

```d
// "What would Y have been if X had been x'?"
let cf = counterfactual(dag, "X", x_factual, x_counterfactual, "Y")

// Probability of necessity and sufficiency
let pn = probability_of_necessity(dag, "X", "Y", 1.0, 0.0)
let ps = probability_of_sufficiency(dag, "X", "Y", 0.0, 1.0)
```

### 7. Causal Structure Learning

```d
// Update edge existence based on observed correlation
let updated_edge = update_edge_existence(edge, correlation, sample_size)

// Prune edges below confidence threshold
let pruned_dag = dag_prune_edges(dag, threshold: 0.7)
```

### 8. Active Learning

```d
// Which intervention maximizes information gain?
let best_target = optimal_intervention_target(dag)
// Returns node with highest epistemic uncertainty
```

## Node Types

- **Treatment**: Intervention variable (X)
- **Outcome**: Target outcome (Y)
- **Confounder**: Common cause of treatment and outcome (Z)
- **Mediator**: On causal path between treatment and outcome (M)
- **Collider**: Common effect of two variables
- **Instrument**: Affects treatment but not outcome directly (IV)

## Example: Drug Efficacy with Confounding

```d
var dag = dag_new()

// Nodes
dag = dag_add_node(dag, "Age", NodeType::Confounder)
dag = dag_add_node(dag, "Drug", NodeType::Treatment)
dag = dag_add_node(dag, "Recovery", NodeType::Outcome)

// Edges with epistemic uncertainty
dag = dag_add_edge(dag, "Age", "Drug",
                   beta_new(9.0, 1.0), 0.3, 0.02)
dag = dag_add_edge(dag, "Age", "Recovery",
                   beta_new(15.0, 1.0), -0.4, 0.01)
dag = dag_add_edge(dag, "Drug", "Recovery",
                   beta_new(6.0, 4.0), 0.5, 0.10)

// Analysis
let ate = average_treatment_effect(dag, "Drug", "Recovery")
// ate.mean ≈ 0.6 (positive effect)
// ate.variance ≈ 0.02 (epistemic uncertainty)
```

## Sensitivity Analysis

```d
// How robust is the effect to unmeasured confounding?
let e_value = confounder_sensitivity(dag, "X", "Y", observed_effect)

// How confident are we in the causal structure?
let robustness = robustness_value(dag, "X", "Y")
```

## Integration with ML

The epistemic framework enables:

1. **Variance-aware training**: Penalize predictions with high uncertainty
2. **Active learning**: Target data collection where uncertainty is highest
3. **Causal representation learning**: Learn DAG structure from data
4. **Uncertainty propagation**: Track how epistemic uncertainty flows through models

## Comparison to Traditional Approaches

| Aspect | Traditional | Demetrios Epistemic |
|--------|-------------|---------------------|
| Effect estimate | Point estimate | Full Beta posterior |
| Edge existence | Binary (yes/no) | Probabilistic (Beta) |
| Uncertainty | Confidence intervals | Epistemic variance |
| Structure learning | Hypothesis testing | Bayesian updating |
| Active learning | Ad-hoc | Information-theoretic |

## Mathematical Foundation

### Do-Operator
```
P(Y | do(X = x)) ≠ P(Y | X = x)
```
Intervention removes incoming edges to X.

### Backdoor Criterion
A set Z satisfies backdoor criterion if:
1. No node in Z is a descendant of X
2. Z blocks all backdoor paths from X to Y

### Mediation Formulas
```
NDE = E[Y_{X=1,M=M_0} - Y_{X=0,M=M_0}]
NIE = E[Y_{X=1,M=M_1} - Y_{X=1,M=M_0}]
Total Effect = NDE + NIE
```

## Files

- `causal.d` - Core causal inference primitives
- `test_causal.d` - Comprehensive test suite
- `epistemic_ml_demo.d` - Practical ML integration examples

## References

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*
2. Pearl, J. (2018). *The Book of Why*
3. Hernán, M. A., & Robins, J. M. (2020). *Causal Inference: What If*

## Future Enhancements

- [ ] Front-door criterion implementation
- [ ] Time-varying treatments (causal forests)
- [ ] Bayesian networks integration
- [ ] GPU-accelerated structure learning
- [ ] Integration with differential privacy
