# Counterfactual Reasoning in Sounio

Counterfactual reasoning answers "What if?" questions - the highest level (Level 3) of Pearl's causal hierarchy. Sounio provides first-class support for counterfactual computation with full epistemic uncertainty tracking.

## What Are Counterfactuals?

A counterfactual question asks what would have happened in an alternative scenario:

- "Would the patient have recovered without the treatment?"
- "Would the accident have occurred if the driver had been sober?"
- "What would this customer's purchase have been with a different recommendation?"

Unlike interventional questions (Level 2), counterfactuals reason about a specific unit given what actually happened.

### Notation

The notation Y_{x}(u) represents:
- Y: the outcome variable
- x: the intervention value
- u: the exogenous factors for a specific unit

**Example**: Y_{drug=0}(patient_42) = "What would patient 42's outcome have been without the drug, given everything else about patient 42?"

## The Three-Step Counterfactual Algorithm

Computing counterfactuals requires three steps:

### Step 1: Abduction

Infer the exogenous (background) variables U from the observed evidence.

```sio
use std::causal::{StructuralCausalModel, what_if}

// Given evidence: X=1, Y=1
// Infer: What values of U are consistent with this evidence?

fn abduct(
    model: StructuralCausalModel,
    evidence: [(String, f64)]
) -> ExogenousDistribution with Prob {
    // Compute P(U | evidence)
    // This is often done via rejection sampling or MCMC

    var posterior_samples: [[f64]] = []
    let n_samples: i64 = 10000

    var accepted: i64 = 0
    var proposed: i64 = 0

    while accepted < n_samples && proposed < n_samples * 100 {
        // Sample U from prior
        let u_sample = model.sample_exogenous()
        proposed = proposed + 1

        // Evaluate model with this U
        let values = model.evaluate(u_sample)

        // Check if consistent with evidence
        if matches_evidence(values, evidence, 0.1) {
            posterior_samples = posterior_samples ++ [u_sample]
            accepted = accepted + 1
        }
    }

    return ExogenousDistribution { samples: posterior_samples }
}
```

### Step 2: Action

Modify the structural causal model by intervening on the counterfactual variable.

```sio
fn action(
    model: StructuralCausalModel,
    intervention_var: [u8],
    intervention_value: f64
) -> StructuralCausalModel {
    // Create modified model where intervention_var is set to intervention_value
    // This removes all equations for intervention_var and replaces with constant

    return model.intervene(intervention_var, intervention_value)
}
```

### Step 3: Prediction

Compute the outcome in the modified model using the posterior U from abduction.

```sio
fn predict(
    modified_model: StructuralCausalModel,
    posterior_u: ExogenousDistribution
) -> CounterfactualDistribution with Prob, Alloc {
    var cf_values: [f64] = []

    var i: i64 = 0
    while i < posterior_u.samples.len() {
        let u = posterior_u.samples[i]
        let values = modified_model.evaluate(u)
        cf_values = cf_values ++ [values.target]
        i = i + 1
    }

    return CounterfactualDistribution { samples: cf_values }
}
```

### Complete Algorithm

```sio
use std::causal::{
    StructuralCausalModel, what_if, EpistemicSummary,
    abduct, action, predict
}

/// Compute counterfactual Y_{x'} given evidence (X=x, Y=y)
fn compute_counterfactual(
    model: StructuralCausalModel,
    target: [u8],
    intervention_var: [u8],
    intervention_value: f64,
    evidence: [(String, f64)]
) -> EpistemicSummary with Prob, Alloc {
    // Step 1: ABDUCTION
    let posterior_u = abduct(model, evidence)

    // Step 2: ACTION
    let modified_model = action(model, intervention_var, intervention_value)

    // Step 3: PREDICTION
    let cf_dist = predict(modified_model, posterior_u)

    // Summarize with epistemic uncertainty
    return summarize_counterfactual(cf_dist, posterior_u)
}
```

## Structural Equation Models

Counterfactual reasoning requires a structural causal model (SCM):

```sio
/// Structural Equation for a variable
struct StructuralEquation {
    parents: [[u8]],         // Parent variables
    coefficients: [f64],     // Coefficients for each parent
    intercept: f64,          // Constant term
    exogenous: [u8],         // Name of exogenous variable (U)
}

/// Full Structural Causal Model
struct StructuralCausalModel {
    variables: [[u8]],
    equations: [StructuralEquation],
    exogenous_distributions: [Distribution],
}

// Example: Simple linear SCM
//   X = U_X
//   Y = 0.5*X + U_Y
//
// Where U_X ~ N(0, 1) and U_Y ~ N(0, 0.5)

fn build_simple_scm() -> StructuralCausalModel {
    StructuralCausalModel {
        variables: [[88], [89]],  // ["X", "Y"]
        equations: [
            StructuralEquation {
                parents: [],
                coefficients: [],
                intercept: 0.0,
                exogenous: [85, 95, 88],  // "U_X"
            },
            StructuralEquation {
                parents: [[88]],  // ["X"]
                coefficients: [0.5],
                intercept: 0.0,
                exogenous: [85, 95, 89],  // "U_Y"
            }
        ],
        exogenous_distributions: [
            Distribution::Normal { mean: 0.0, std: 1.0 },
            Distribution::Normal { mean: 0.0, std: 0.5 },
        ],
    }
}
```

## Practical Counterfactual Example

### Medical Treatment Counterfactual

```sio
use std::causal::{
    StructuralCausalModel, what_if, probability_of_necessity,
    probability_of_sufficiency, EpistemicSummary
}

fn analyze_treatment_counterfactual() -> i32 with Prob, IO {
    print("=== Treatment Counterfactual Analysis ===\n\n")

    // Build SCM for treatment effect
    //   Severity = U_S
    //   Treatment = f(Severity) + U_T
    //   Recovery = 0.4*Treatment - 0.3*Severity + U_R
    let model = build_treatment_scm()

    // Observed: Patient received treatment (T=1) and recovered (R=1)
    // Severity was measured at S=0.6
    let evidence = [
        ("Severity", 0.6),
        ("Treatment", 1.0),
        ("Recovery", 1.0),
    ]

    print("Evidence: Severity=0.6, Treatment=1, Recovery=1\n\n")

    // Counterfactual question: Would patient have recovered WITHOUT treatment?
    let cf_no_treatment = what_if(
        model,
        "Treatment",    // intervention variable
        1.0,            // factual value
        0.0,            // counterfactual value
        "Recovery"      // target
    )

    print("Counterfactual: Recovery if Treatment had been 0?\n")
    print("  Mean: ")
    print(cf_no_treatment.mean)
    print("\n  95% CI: [")
    print(cf_no_treatment.lower)
    print(", ")
    print(cf_no_treatment.upper)
    print("]\n\n")

    // Interpret
    if cf_no_treatment.mean < 0.5 {
        print("Conclusion: Patient likely would NOT have recovered without treatment\n")
        print("  Treatment was likely NECESSARY for recovery\n")
    } else if cf_no_treatment.mean > 0.5 {
        print("Conclusion: Patient likely WOULD have recovered without treatment\n")
        print("  Treatment may not have been necessary\n")
    } else {
        print("Conclusion: Uncertain whether treatment was necessary\n")
    }

    return 0
}
```

## Probability of Causation

Counterfactuals enable computing probabilities of causation, which answer legal and attribution questions.

### Probability of Necessity (PN)

PN = P(Y_{x'} = 0 | X=x, Y=1)

"Given that Y occurred with treatment X=x, would Y NOT have occurred if X had been x' instead?"

```sio
/// Compute Probability of Necessity
fn probability_of_necessity(
    model: StructuralCausalModel,
    treatment: [u8],
    treatment_factual: f64,
    treatment_counterfactual: f64,
    outcome: [u8],
    n_samples: i64
) -> ProbabilityOfCausation with Prob {
    var necessary_count: i64 = 0

    var i: i64 = 0
    while i < n_samples {
        // Sample from P(U | X=x, Y=1)
        let u = sample_conditioned_u(model, treatment, treatment_factual, outcome, 1.0)

        // Compute Y in counterfactual world
        let model_cf = model.intervene(treatment, treatment_counterfactual)
        let values_cf = model_cf.evaluate(u)

        // Check if Y_{x'} = 0
        if values_cf.get(outcome) < 0.5 {
            necessary_count = necessary_count + 1
        }

        i = i + 1
    }

    let pn = (necessary_count as f64) / (n_samples as f64)

    return ProbabilityOfCausation {
        causation_type: CausationType::Necessity,
        probability: pn,
        confidence: compute_binomial_confidence(necessary_count, n_samples),
        interpretation: format_pn_interpretation(treatment, outcome, pn),
    }
}
```

### Probability of Sufficiency (PS)

PS = P(Y_x = 1 | X=x', Y=0)

"Given that Y did NOT occur with X=x', would Y have occurred if X had been x?"

```sio
/// Compute Probability of Sufficiency
fn probability_of_sufficiency(
    model: StructuralCausalModel,
    treatment: [u8],
    treatment_factual: f64,
    treatment_counterfactual: f64,
    outcome: [u8],
    n_samples: i64
) -> ProbabilityOfCausation with Prob {
    var sufficient_count: i64 = 0

    var i: i64 = 0
    while i < n_samples {
        // Sample from P(U | X=x', Y=0)
        let u = sample_conditioned_u(model, treatment, treatment_factual, outcome, 0.0)

        // Compute Y in counterfactual world
        let model_cf = model.intervene(treatment, treatment_counterfactual)
        let values_cf = model_cf.evaluate(u)

        // Check if Y_x = 1
        if values_cf.get(outcome) >= 0.5 {
            sufficient_count = sufficient_count + 1
        }

        i = i + 1
    }

    let ps = (sufficient_count as f64) / (n_samples as f64)

    return ProbabilityOfCausation {
        causation_type: CausationType::Sufficiency,
        probability: ps,
        confidence: compute_binomial_confidence(sufficient_count, n_samples),
        interpretation: format_ps_interpretation(treatment, outcome, ps),
    }
}
```

### Probability of Necessity and Sufficiency (PNS)

PNS = P(Y_x = 1, Y_{x'} = 0)

"X is both necessary AND sufficient for Y."

```sio
/// Compute Probability of Necessity and Sufficiency
fn probability_of_necessity_and_sufficiency(
    model: StructuralCausalModel,
    treatment: [u8],
    treatment_value: f64,
    treatment_baseline: f64,
    outcome: [u8],
    n_samples: i64
) -> ProbabilityOfCausation with Prob {
    var pns_count: i64 = 0

    var i: i64 = 0
    while i < n_samples {
        // Sample U from prior (marginal)
        let u = model.sample_exogenous()

        // Y_x (with treatment)
        let model_x = model.intervene(treatment, treatment_value)
        let y_x = model_x.evaluate(u).get(outcome)

        // Y_{x'} (without treatment)
        let model_xp = model.intervene(treatment, treatment_baseline)
        let y_xp = model_xp.evaluate(u).get(outcome)

        // PNS: Y_x = 1 AND Y_{x'} = 0
        if y_x >= 0.5 && y_xp < 0.5 {
            pns_count = pns_count + 1
        }

        i = i + 1
    }

    let pns = (pns_count as f64) / (n_samples as f64)

    return ProbabilityOfCausation {
        causation_type: CausationType::NecessityAndSufficiency,
        probability: pns,
        confidence: compute_binomial_confidence(pns_count, n_samples),
        interpretation: format_pns_interpretation(treatment, outcome, pns),
    }
}
```

### Complete Attribution Analysis

```sio
fn full_attribution_analysis(
    model: StructuralCausalModel,
    treatment: [u8],
    outcome: [u8]
) -> i32 with Prob, IO {
    print("=== Causal Attribution Analysis ===\n\n")

    let n_samples: i64 = 10000

    // Compute PN
    let pn = probability_of_necessity(
        model, treatment, 1.0, 0.0, outcome, n_samples
    )
    print("Probability of Necessity (PN): ")
    print(pn.probability * 100.0)
    print("%\n")
    print("  ")
    print(pn.interpretation)
    print("\n\n")

    // Compute PS
    let ps = probability_of_sufficiency(
        model, treatment, 1.0, 0.0, outcome, n_samples
    )
    print("Probability of Sufficiency (PS): ")
    print(ps.probability * 100.0)
    print("%\n")
    print("  ")
    print(ps.interpretation)
    print("\n\n")

    // Compute PNS
    let pns = probability_of_necessity_and_sufficiency(
        model, treatment, 1.0, 0.0, outcome, n_samples
    )
    print("Probability of Necessity and Sufficiency (PNS): ")
    print(pns.probability * 100.0)
    print("%\n")
    print("  ")
    print(pns.interpretation)
    print("\n\n")

    // Relationships
    print("Theoretical bounds:\n")
    print("  PNS <= min(PN, PS)\n")
    print("  PN >= ATE / P(Y=1|X=1)\n")
    print("  PS >= ATE / P(Y=0|X=0)\n")

    return 0
}
```

## Uncertainty in Counterfactual Claims

Counterfactual conclusions carry multiple sources of uncertainty:

### 1. Model Uncertainty

The structural equations may be misspecified:

```sio
/// Counterfactual with model uncertainty
struct CounterfactualWithModelUncertainty {
    value: EpistemicSummary,
    model_confidence: f64,
    structural_assumptions: [[u8]],  // List of assumptions
}

fn counterfactual_with_model_uncertainty(
    models: [StructuralCausalModel],  // Ensemble of models
    model_weights: [f64],             // Posterior weights
    target: [u8],
    intervention_var: [u8],
    intervention_value: f64,
    evidence: [(String, f64)]
) -> EpistemicSummary with Prob, Alloc {
    // Bayesian model averaging over SCMs
    var weighted_mean = 0.0
    var weighted_var = 0.0

    var m: i64 = 0
    while m < models.len() {
        let cf = compute_counterfactual(
            models[m], target, intervention_var, intervention_value, evidence
        )
        weighted_mean = weighted_mean + model_weights[m] * cf.mean
        weighted_var = weighted_var + model_weights[m] * (cf.variance + cf.mean * cf.mean)
        m = m + 1
    }

    weighted_var = weighted_var - weighted_mean * weighted_mean

    return EpistemicSummary {
        mean: weighted_mean,
        variance: weighted_var,
        confidence: compute_ensemble_confidence(model_weights),
        lower: weighted_mean - 1.96 * sqrt_f64(weighted_var),
        upper: weighted_mean + 1.96 * sqrt_f64(weighted_var),
    }
}
```

### 2. Abduction Uncertainty

Limited evidence means posterior U is uncertain:

```sio
/// Track abduction quality
struct AbductionQuality {
    effective_sample_size: f64,
    acceptance_rate: f64,
    evidence_coverage: f64,  // Fraction of model covered by evidence
}

fn assess_abduction_quality(
    n_proposed: i64,
    n_accepted: i64,
    evidence: [(String, f64)],
    model: StructuralCausalModel
) -> AbductionQuality {
    let acceptance_rate = (n_accepted as f64) / (n_proposed as f64)
    let coverage = (evidence.len() as f64) / (model.variables.len() as f64)

    // ESS accounting for correlation in samples
    let ess = (n_accepted as f64) * acceptance_rate

    return AbductionQuality {
        effective_sample_size: ess,
        acceptance_rate: acceptance_rate,
        evidence_coverage: coverage,
    }
}
```

### 3. Fundamental Untestability

Counterfactuals are fundamentally untestable - we can never observe both X=1 and X=0 for the same unit:

```sio
/// Warn about counterfactual limitations
fn counterfactual_limitations_warning() with IO {
    print("=== Important Limitations ===\n\n")

    print("Counterfactuals are fundamentally UNTESTABLE:\n")
    print("- We cannot observe both treated and untreated outcomes\n")
    print("- Results depend on UNTESTABLE structural assumptions\n")
    print("- Different SCMs can give different counterfactual answers\n\n")

    print("Confidence levels reflect:\n")
    print("- Statistical uncertainty in posterior U\n")
    print("- BUT NOT uncertainty about model correctness\n\n")

    print("Always perform sensitivity analysis.\n")
}
```

## Sensitivity Analysis for Counterfactuals

```sio
/// Sensitivity analysis for counterfactual conclusions
fn counterfactual_sensitivity(
    model: StructuralCausalModel,
    target: [u8],
    intervention_var: [u8],
    intervention_value: f64,
    evidence: [(String, f64)],
    perturbation_range: f64
) -> SensitivityReport with Prob, Alloc {
    var results: [EpistemicSummary] = []

    // Vary structural equation coefficients
    let base_cf = compute_counterfactual(
        model, target, intervention_var, intervention_value, evidence
    )
    results = results ++ [base_cf]

    // Perturb each coefficient
    for eq_idx in 0..model.equations.len() {
        for coef_idx in 0..model.equations[eq_idx].coefficients.len() {
            let perturbed_model = perturb_coefficient(
                model, eq_idx, coef_idx, perturbation_range
            )
            let perturbed_cf = compute_counterfactual(
                perturbed_model, target, intervention_var, intervention_value, evidence
            )
            results = results ++ [perturbed_cf]
        }
    }

    // Summarize sensitivity
    return SensitivityReport {
        base_result: base_cf,
        perturbed_results: results,
        max_deviation: compute_max_deviation(results, base_cf),
        robust: check_sign_robustness(results),
    }
}
```

## Summary

| Concept | Description | Formula |
|---------|-------------|---------|
| Counterfactual | What would Y be if X had been x? | Y_{x}(u) |
| PN | Was X necessary for Y? | P(Y_{x'}=0 \| X=x, Y=1) |
| PS | Would X be sufficient for Y? | P(Y_x=1 \| X=x', Y=0) |
| PNS | Is X both necessary and sufficient? | P(Y_x=1, Y_{x'}=0) |

Key points:

1. **Counterfactuals require SCMs** - You need full structural equations, not just a DAG
2. **Three-step algorithm**: Abduction, Action, Prediction
3. **Uncertainty everywhere** - Model uncertainty, abduction uncertainty, fundamental untestability
4. **Sensitivity analysis is essential** - Results may depend strongly on model assumptions
5. **Sounio tracks epistemic uncertainty** - Every counterfactual conclusion carries confidence

Counterfactual reasoning is powerful but must be used with appropriate humility about its limitations.
