# Meta-Analysis API

When you have multiple `Knowledge` values from different sources (experiments, measurements, studies), this module combines them using principled statistical methods.

## Overview

Meta-analysis answers the question: "What do multiple independent studies tell us together about some effect?"

Sounio's meta-analysis module provides:

- **Fixed-effects pooling**: Inverse-variance weighted combination
- **Random-effects pooling**: DerSimonian-Laird method
- **Bayesian hierarchical pooling**: With informative priors
- **Heterogeneity statistics**: Q, I-squared, tau-squared

## Quick Start

```sio
use epistemic::{Knowledge, meta}

// Results from 3 clinical trials
let trial1 = Knowledge::measured(0.35, 0.04, "RCT_2021")
let trial2 = Knowledge::measured(0.42, 0.06, "RCT_2022")
let trial3 = Knowledge::measured(0.38, 0.03, "RCT_2023")

// Pool the evidence
let pooled = meta::random_effects([trial1, trial2, trial3])

// Report results
println("Pooled effect: {} +/- {}", pooled.pooled.get(), pooled.pooled.std())
println("Heterogeneity I-squared: {}%", pooled.heterogeneity.i_squared * 100.0)
println("Interpretation: {}", pooled.heterogeneity.interpretation())
```

## Heterogeneity Statistics

### `Heterogeneity`

Heterogeneity metrics quantify how much studies differ beyond sampling error.

```sio
pub struct Heterogeneity {
    /// Cochran's Q statistic (chi-squared test for heterogeneity)
    q: f64,

    /// Degrees of freedom (k - 1 where k = number of studies)
    df: i64,

    /// I-squared statistic: percentage of variability due to heterogeneity
    /// 0% = no heterogeneity, 25% = low, 50% = moderate, 75% = high
    i_squared: f64,

    /// tau-squared: between-study variance
    tau_squared: f64,

    /// p-value for Q statistic
    p_value: f64,
}
```

#### `Heterogeneity::from_q`

Create heterogeneity statistics from Q and degrees of freedom.

```sio
pub fn from_q(q: f64, df: i64) -> Heterogeneity
```

#### `is_significant`

Test if heterogeneity is statistically significant.

```sio
pub fn is_significant(self: &Heterogeneity, alpha: f64) -> bool
```

**Example:**
```sio
if result.heterogeneity.is_significant(0.10) {
    println("Significant heterogeneity detected!")
}
```

#### `interpretation`

Get qualitative interpretation of I-squared.

```sio
pub fn interpretation(self: &Heterogeneity) -> string
```

**Returns one of:**
- `"low heterogeneity"` (I-squared < 25%)
- `"moderate heterogeneity"` (25% <= I-squared < 50%)
- `"substantial heterogeneity"` (50% <= I-squared < 75%)
- `"considerable heterogeneity"` (I-squared >= 75%)

### Understanding the Statistics

| Statistic | Meaning |
|-----------|---------|
| **Q** | Chi-squared test statistic. Sensitive to number of studies. |
| **I-squared** | Percentage of variance due to heterogeneity (not chance). |
| **tau-squared** | Absolute between-study variance. |
| **p-value** | Probability Q arose by chance if true heterogeneity is zero. |

**Interpretation Guidelines (Higgins & Thompson, 2002):**
- I-squared < 25%: Use fixed-effects
- I-squared 25-50%: Moderate heterogeneity, consider random-effects
- I-squared 50-75%: Substantial heterogeneity, use random-effects
- I-squared > 75%: Considerable heterogeneity, investigate sources

## MetaResult

Result of meta-analysis combining multiple `Knowledge` values.

```sio
pub struct MetaResult {
    /// Pooled effect estimate
    pooled: Knowledge<f64>,

    /// Heterogeneity statistics
    heterogeneity: Heterogeneity,

    /// Number of studies combined
    k: i64,

    /// Method used
    method: string,

    /// Individual study weights
    weights: Vec<f64>,
}
```

### Methods

#### `get`

Get the pooled `Knowledge` value.

```sio
pub fn get(self: &MetaResult) -> &Knowledge<f64>
```

#### `het`

Get heterogeneity statistics.

```sio
pub fn het(self: &MetaResult) -> &Heterogeneity
```

#### `needs_random_effects`

Should we use random-effects instead of fixed?

```sio
pub fn needs_random_effects(self: &MetaResult) -> bool
```

Returns `true` if significant heterogeneity detected (p < 0.10 or I-squared > 50%).

## Fixed-Effects Meta-Analysis

### `fixed_effects`

Fixed-effects meta-analysis using inverse-variance weighting.

```sio
pub fn fixed_effects(studies: &[Knowledge<f64>]) -> MetaResult
```

**Assumptions:**
- All studies estimate the **same underlying effect**
- Differences between studies are due to sampling error only
- Appropriate when heterogeneity is low (I-squared < 25%)

**Formula:**

```
theta_hat = sum(w_i * theta_i) / sum(w_i)

where w_i = 1 / Var(theta_i)

Var(theta_hat) = 1 / sum(w_i)
```

**Example:**
```sio
let studies = [
    Knowledge::measured(0.35, 0.04, "study_A"),
    Knowledge::measured(0.40, 0.05, "study_B"),
    Knowledge::measured(0.38, 0.03, "study_C"),
]

let result = meta::fixed_effects(&studies)

println("Pooled effect: {} +/- {}", result.pooled.get(), result.pooled.std())
println("Weights: {:?}", result.weights)
```

**When to Use:**
- Low heterogeneity (I-squared < 25%)
- Studies from similar populations with similar protocols
- When you believe there is one "true" effect

## Random-Effects Meta-Analysis

### `random_effects`

Random-effects meta-analysis using DerSimonian-Laird method.

```sio
pub fn random_effects(studies: &[Knowledge<f64>]) -> MetaResult
```

**Assumptions:**
- Studies estimate **different but related effects**
- There is a distribution of true effects
- Accounts for between-study variance (tau-squared)
- Appropriate when heterogeneity exists (I-squared > 25%)

**Formula:**

```
w*_i = 1 / (Var(theta_i) + tau^2)

theta_hat = sum(w*_i * theta_i) / sum(w*_i)

tau^2 = max(0, (Q - df) / C)
```

Where tau-squared is the between-study variance estimated using the DerSimonian-Laird method.

**Example:**
```sio
let trials = [
    Knowledge::measured(0.25, 0.10, "RCT_2020"),
    Knowledge::measured(0.45, 0.08, "RCT_2021"),
    Knowledge::measured(0.35, 0.12, "RCT_2022"),
]

let result = meta::random_effects(&trials)

if result.heterogeneity.i_squared > 0.5 {
    println("Substantial heterogeneity (I^2 = {}%)", result.heterogeneity.i_squared * 100.0)
    println("tau^2 = {}", result.heterogeneity.tau_squared)
}
```

**When to Use:**
- Moderate to high heterogeneity (I-squared >= 25%)
- Studies from different populations or protocols
- When you expect true effects to vary across studies

## Bayesian Hierarchical Meta-Analysis

### `bayesian_pool`

Bayesian hierarchical pooling with informative prior.

```sio
pub fn bayesian_pool(
    studies: &[Knowledge<f64>],
    prior_mean: f64,
    prior_variance: f64,
) -> MetaResult
```

**Parameters:**
- `studies`: Array of `Knowledge` values to combine
- `prior_mean`: Prior belief about the pooled effect
- `prior_variance`: Uncertainty in prior belief

**Formula:**

```
Posterior precision = Prior precision + sum(Study precisions)
Posterior mean = (Prior precision * Prior mean + sum(Study precision * Study effect)) / Posterior precision
```

**Example:**
```sio
// Historical evidence suggests effect around 0.4
let prior_mean = 0.40
let prior_variance = 0.10  // Moderate prior uncertainty

let studies = [
    Knowledge::measured(0.35, 0.05, "new_study_1"),
    Knowledge::measured(0.42, 0.06, "new_study_2"),
]

let result = meta::bayesian_pool(&studies, prior_mean, prior_variance)

println("Posterior: {} +/- {}", result.pooled.get(), result.pooled.std())
```

**When to Use:**
- Strong domain knowledge about plausible effects
- Small number of studies (prior helps stabilize estimates)
- Want to incorporate prior evidence formally

### `bayesian_pool_flat`

Bayesian pooling with non-informative (flat) prior.

```sio
pub fn bayesian_pool_flat(studies: &[Knowledge<f64>]) -> MetaResult
```

Uses very wide prior (variance = 1,000,000) so data dominates.

## Convenience Functions

### `auto_pool`

Automatically choose fixed or random effects based on heterogeneity.

```sio
pub fn auto_pool(studies: &[Knowledge<f64>]) -> MetaResult
```

**Algorithm:**
1. Compute fixed-effects result
2. If `needs_random_effects()` is true, switch to random-effects
3. Return appropriate result

**Example:**
```sio
let result = meta::auto_pool(&studies)
println("Method used: {}", result.method)
```

### `sensitivity_analysis`

Pool with both methods for comparison.

```sio
pub fn sensitivity_analysis(studies: &[Knowledge<f64>]) -> SensitivityResult
```

**Returns:**

```sio
pub struct SensitivityResult {
    fixed: MetaResult,
    random: MetaResult,
    recommendation: string,
}
```

**Example:**
```sio
let sens = meta::sensitivity_analysis(&studies)

println("Fixed-effects: {} +/- {}", sens.fixed.pooled.get(), sens.fixed.pooled.std())
println("Random-effects: {} +/- {}", sens.random.pooled.get(), sens.random.pooled.std())
println("Recommendation: {}", sens.recommendation)
```

## Complete Example: Clinical Trial Meta-Analysis

```sio
use epistemic::{Knowledge, meta}

fn main() {
    // Effect sizes from 5 randomized controlled trials
    // Effect is hazard ratio, variance is SE^2
    let trials = [
        Knowledge::measured(0.72, 0.0225, "PIONEER"),     // HR=0.72, SE=0.15
        Knowledge::measured(0.65, 0.0196, "SUSTAIN"),     // HR=0.65, SE=0.14
        Knowledge::measured(0.78, 0.0289, "LEADER"),      // HR=0.78, SE=0.17
        Knowledge::measured(0.70, 0.0256, "REWIND"),      // HR=0.70, SE=0.16
        Knowledge::measured(0.74, 0.0324, "AMPLITUDE"),   // HR=0.74, SE=0.18
    ]

    // Fixed-effects analysis
    let fe = meta::fixed_effects(&trials)
    println("=== Fixed-Effects ===")
    println("Pooled HR: {:.2} (95% CI: {:.2}-{:.2})",
        fe.pooled.get(),
        fe.pooled.ci95().0,
        fe.pooled.ci95().1)

    // Check heterogeneity
    println("\n=== Heterogeneity ===")
    println("Q = {:.2}, df = {}, p = {:.3}",
        fe.heterogeneity.q,
        fe.heterogeneity.df,
        fe.heterogeneity.p_value)
    println("I^2 = {:.1}%", fe.heterogeneity.i_squared * 100.0)
    println("Interpretation: {}", fe.heterogeneity.interpretation())

    // Random-effects if needed
    if fe.needs_random_effects() {
        let re = meta::random_effects(&trials)
        println("\n=== Random-Effects ===")
        println("Pooled HR: {:.2} (95% CI: {:.2}-{:.2})",
            re.pooled.get(),
            re.pooled.ci95().0,
            re.pooled.ci95().1)
        println("tau^2 = {:.4}", re.heterogeneity.tau_squared)
    }

    // Bayesian with prior from mechanism-of-action studies
    let prior_hr = 0.75
    let prior_se = 0.10
    let bayes = meta::bayesian_pool(&trials, prior_hr, prior_se * prior_se)

    println("\n=== Bayesian (Informative Prior) ===")
    println("Prior: HR = {:.2} +/- {:.2}", prior_hr, prior_se)
    println("Posterior HR: {:.2} (95% CI: {:.2}-{:.2})",
        bayes.pooled.get(),
        bayes.pooled.ci95().0,
        bayes.pooled.ci95().1)

    // Probability of benefit
    let prob_benefit = bayes.pooled.prob_lt(1.0)
    println("P(HR < 1.0) = {:.1}%", prob_benefit * 100.0)
}
```

**Example Output:**
```
=== Fixed-Effects ===
Pooled HR: 0.72 (95% CI: 0.64-0.80)

=== Heterogeneity ===
Q = 3.21, df = 4, p = 0.523
I^2 = 0.0%
Interpretation: low heterogeneity

=== Bayesian (Informative Prior) ===
Prior: HR = 0.75 +/- 0.10
Posterior HR: 0.72 (95% CI: 0.65-0.79)
P(HR < 1.0) = 99.8%
```

## Forest Plot Data

The `weights` field in `MetaResult` can be used to generate forest plots:

```sio
let result = meta::random_effects(&studies)

// Generate forest plot data
for i in 0..result.k {
    let study = studies[i]
    let weight = result.weights[i] / result.weights.sum() * 100.0

    println("Study {}: effect = {:.2}, weight = {:.1}%",
        i + 1,
        study.get(),
        weight)
}

println("Pooled: effect = {:.2}", result.pooled.get())
```

## References

- Borenstein, M., et al. "Introduction to Meta-Analysis" (2009)
- DerSimonian, R. & Laird, N. "Meta-analysis in clinical trials" (1986)
- Higgins, J.P.T. & Thompson, S.G. "Quantifying heterogeneity in a meta-analysis" (2002)
- Cochrane Handbook for Systematic Reviews of Interventions

## See Also

- [Knowledge<T> API Reference](knowledge.md)
- [MCMC Sampling](mcmc.md)
- [Variance Propagation](propagate.md)
