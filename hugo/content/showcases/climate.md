---
title: "Climate Modeling"
date: 2024-01-28
domain: "climate"
---

# Climate Modeling: Epistemic Uncertainty for Multi-Model Ensembles

## The Problem

Climate projections require combining **dozens of global climate models (GCMs)** into ensembles. Each model carries:

- **Structural uncertainty**: Different physical parameterizations
- **Parameter uncertainty**: Calibration ranges within each model
- **Initial condition uncertainty**: Chaotic sensitivity
- **Scenario uncertainty**: Future emission pathways (SSP1-SSP5)

The **IPCC AR6** (2021) mandates transparent uncertainty quantification using calibrated language: *"likely" (66-100%), "very likely" (90-100%), "virtually certain" (99-100%)*.

Traditional tools (Python/xarray, R) compute uncertainty **post-hoc** with no compile-time guarantees.

---

## Sounio's Solution: Knowledge<T> for Climate Science

### Ensemble Averaging with Uncertainty

```sio
use epistemic::Knowledge
use units::{Celsius, mm, year}

fn ensemble_mean(
    models: &[Knowledge<Celsius>]
) -> Knowledge<Celsius> {
    let n = models.len() as f64
    let mean_value = models.iter().map(|m| m.value).sum() / n

    // Combined uncertainty (GUM method)
    let combined_uncertainty = sqrt(
        models.iter()
            .map(|m| m.std_uncertainty.powi(2))
            .sum()
    ) / n

    Knowledge::new(
        value: mean_value,
        std_uncertainty: combined_uncertainty,
        confidence: 0.90  // IPCC "very likely" range
    )
}
```

### IPCC Confidence Levels

Sounio's type system encodes IPCC calibrated language:

```sio
enum IPCCConfidence {
    VirtuallyCertain,  // 99-100% probability
    VeryLikely,        // 90-100%
    Likely,            // 66-100%
    AboutAsLikelyAsNot, // 33-66%
    Unlikely,          // 0-33%
    VeryUnlikely,      // 0-10%
    ExceptionallyUnlikely // 0-1%
}

fn classify_confidence(k: Knowledge<Celsius>) -> IPCCConfidence {
    match k.confidence {
        c if c >= 0.99 => IPCCConfidence::VirtuallyCertain,
        c if c >= 0.90 => IPCCConfidence::VeryLikely,
        c if c >= 0.66 => IPCCConfidence::Likely,
        c if c >= 0.33 => IPCCConfidence::AboutAsLikelyAsNot,
        c if c >= 0.10 => IPCCConfidence::Unlikely,
        c if c >= 0.01 => IPCCConfidence::VeryUnlikely,
        _ => IPCCConfidence::ExceptionallyUnlikely
    }
}
```

### Sea Level Rise Projection

```sio
use units::{mm, year, cm}

struct SeaLevelProjection {
    thermal_expansion: Knowledge<mm/year>,
    glacier_melt: Knowledge<mm/year>,
    ice_sheet_dynamics: Knowledge<mm/year>,
    land_water_storage: Knowledge<mm/year>
}

fn total_sea_level_rise(
    proj: SeaLevelProjection,
    years: f64
) -> Knowledge<cm> {
    // Sum components with uncertainty propagation
    let annual_rate: Knowledge<mm/year> =
        proj.thermal_expansion +
        proj.glacier_melt +
        proj.ice_sheet_dynamics +
        proj.land_water_storage

    // Convert mm to cm and project forward
    let total: Knowledge<cm> = annual_rate.to_cm() * years

    // Provenance tracks each source model
    total
}

// SSP2-4.5 scenario (medium emissions)
let projection = SeaLevelProjection {
    thermal_expansion: Knowledge::new(1.5, 0.3, confidence: 0.90),
    glacier_melt: Knowledge::new(0.8, 0.2, confidence: 0.90),
    ice_sheet_dynamics: Knowledge::new(0.5, 0.4, confidence: 0.66),
    land_water_storage: Knowledge::new(-0.1, 0.1, confidence: 0.90)
}

let rise_2100 = total_sea_level_rise(projection, years: 76.0)
// Result: 20.5 cm ± 4.2 cm (likely range, 2024-2100)
// IPCC classification: "Likely" (66-100% confidence)
```

---

## Bayesian Model Averaging

### Problem

Given **N climate models** with different skill scores, compute weighted ensemble:

**P(Δ|data) = Σᵢ wᵢ · P(Δ|Mᵢ, data)**

where wᵢ are posterior model weights.

### Sounio Implementation

```sio
use stats::{bayesian, posterior}

struct ClimateModel {
    name: string,
    prediction: Knowledge<Celsius>,
    skill_score: f64  // Historical validation metric
}

fn bayesian_model_average(
    models: &[ClimateModel],
    observations: &[Knowledge<Celsius>]
) -> Knowledge<Celsius> with Prob {
    // Compute posterior model weights
    let weights = bayesian::model_weights(
        models.iter().map(|m| m.skill_score).collect(),
        prior: bayesian::uniform_prior(models.len())
    )

    // Weighted ensemble mean
    let weighted_mean = models.iter()
        .zip(weights.iter())
        .map(|(m, w)| m.prediction.value * w)
        .sum()

    // Combined uncertainty (between-model + within-model)
    let between_model_var = models.iter()
        .zip(weights.iter())
        .map(|(m, w)| w * (m.prediction.value - weighted_mean).powi(2))
        .sum()

    let within_model_var = models.iter()
        .zip(weights.iter())
        .map(|(m, w)| w * m.prediction.std_uncertainty.powi(2))
        .sum()

    Knowledge::new(
        value: weighted_mean,
        std_uncertainty: sqrt(between_model_var + within_model_var),
        confidence: 0.90
    )
}
```

---

## Provenance for Reproducibility

### CMIP6 Data Lineage

Every climate calculation in Sounio maintains **full provenance**:

```sio
let temperature_anomaly = Knowledge::new(
    value: 1.5,
    std_uncertainty: 0.2,
    confidence: 0.90,
    source: ProvenanceSource {
        model: "CESM2",
        experiment: "ssp245",
        variant: "r1i1p1f1",
        grid: "gn",
        version: "v20200130",
        institution: "NCAR"
    }
)

// After ensemble averaging, provenance shows ALL contributing models
let ensemble = ensemble_mean(&[cesm2, gfdl_esm4, ukesm1, miroc6])
println(ensemble.provenance().sources())
// Output: ["CESM2/NCAR", "GFDL-ESM4/GFDL", "UKESM1/MOHC", "MIROC6/MIROC"]
```

### Audit Trail for Policy Reports

```sio
fn generate_policy_report(
    projection: Knowledge<Celsius>,
    target: Celsius
) -> PolicyReport with IO {
    let risk = if projection.value > target {
        let exceedance_prob = projection.exceedance_probability(target)
        format("{}% probability of exceeding {} target",
            exceedance_prob * 100.0, target)
    } else {
        "Target likely met under current scenario"
    }

    PolicyReport {
        finding: risk,
        confidence: classify_confidence(projection),
        provenance: projection.provenance(),
        methodology: "Bayesian Model Averaging (GUM-compliant)",
        citation: "IPCC AR6 WG1 Chapter 4"
    }
}
```

---

## References

1. **IPCC** (2021). *Climate Change 2021: The Physical Science Basis. Contribution of Working Group I to the Sixth Assessment Report*. Cambridge University Press. [DOI: 10.1017/9781009157896](https://doi.org/10.1017/9781009157896)

2. **Eyring, V., et al.** (2016). *Overview of the Coupled Model Intercomparison Project Phase 6 (CMIP6) experimental design and organization*. Geoscientific Model Development, 9(5), 1937-1958.

3. **Tebaldi, C., Knutti, R.** (2007). *The use of the multi-model ensemble in probabilistic climate projections*. Philosophical Transactions of the Royal Society A, 365(1857), 2053-2075.

4. **JCGM 100:2008** (2008). *Guide to the expression of uncertainty in measurement (GUM)*. Joint Committee for Guides in Metrology.

---

*For climate science collaborations, contact: demetrios@sounio-lang.org*
