# epistemic-core

**First-order uncertainty propagation for uncorrelated inputs**

Core epistemic types for Sounio: `KCoreKnowledge`, first-order uncertainty propagation, confidence gates, and source metadata.

## Overview

`epistemic-core` implements measurement values with standard uncertainty, explicit confidence, and lightweight source metadata. Its arithmetic helpers implement the first-order law of propagation of uncertainty for uncorrelated inputs described by **JCGM 100:2008**.

The package propagates uncertainty through its supported arithmetic helper functions. Correlated-input and Monte Carlo propagation are not implemented in this release.

## Install

```toml
# sounio.toml
[dependencies]
epistemic-core = "0.1.0"
```

## Quick Start

```sio
// Pharmaceutical measurement example
let dose   = measure(500.0, 25.0, "HPLC_2025")     // 500 ± 25 mg
let volume = measure(10.0,  0.2,  "pipette_A")      // 10  ± 0.2 mL

// GUM division: relative uncertainties combine in quadrature
let conc = knowledge_div(&dose, &volume)             // 50 ± 2.69 mg/mL

// Epistemic safety gate
confidence_gate(&conc, 0.90)                         // panics if conf < 0.90

println_knowledge(&conc)
// -> 50.0000 +/- 2.6926 [conf=1.00]
```

## API

### Constructors

| Function | Description |
|----------|-------------|
| `measure(value, uncertainty, source)` | Standard GUM measurement |
| `measure_with_confidence(v, u, conf, source)` | With explicit epistemic confidence |

### Arithmetic (GUM first-order)

| Function | GUM formula |
|----------|-------------|
| `knowledge_add(a, b)` | u = √(u_a² + u_b²) |
| `knowledge_sub(a, b)` | u = √(u_a² + u_b²) |
| `knowledge_mul(a, b)` | u(ab) = √((b·u_a)² + (a·u_b)²) |
| `knowledge_div(a, b)` | u(a/b) = √((u_a/b)² + (a·u_b/b²)²), b != 0 |
| `knowledge_scale(k, f)` | u_out = f * u_in |

### Multi-input Propagation

```sio
// f(a, b) = a * exp(-b*t)
// ∂f/∂a = exp(-b*t), ∂f/∂b = -a*t*exp(-b*t)
let result = gum_propagate_2(f_val, df_da, &a, df_db, &b)

// 3-input
let result = gum_propagate_3(f_val, df_da, &a, df_db, &b, df_dc, &c)
```

### Epistemic Gates

```sio
confidence_gate(&k, 0.95)         // panics if confidence < 0.95
let ok = confidence_check(&k, 0.90)  // non-panicking variant
```

### Introspection

```sio
knowledge_value(&k)             // central value
knowledge_uncertainty(&k)       // standard uncertainty u(x)
knowledge_confidence(&k)        // epistemic confidence [0,1]
knowledge_rel_uncertainty(&k)   // u / |v|
knowledge_expanded(&k, 2.0)     // expanded uncertainty U = k·u
```

## GUM Scope

This package implements the **JCGM 100:2008** first-order propagation formula for uncorrelated inputs:

- Additive: u_c(y) = √(Σ(∂f/∂xᵢ · u(xᵢ))²)
- Confidence propagation: minimum of input confidence values (a package policy, not a GUM rule)
- Correlations, effective degrees of freedom, coverage intervals, and conformity assessment are outside the current scope

## Provenance

`Knowledge` values carry lightweight source metadata. Full label storage, chain relations, and W3C PROV-DM export are not implemented in this release.

## Roadmap

- `v0.2.0`: Monte Carlo uncertainty propagation (GUM Supplement 1)
- `v0.3.0`: Correlated inputs (correlation matrix support)
- `v0.4.0`: W3C PROV-DM export to JSON-LD
- `v1.0.0`: ISO 17025 / FDA 21 CFR Part 11 compliance kit
