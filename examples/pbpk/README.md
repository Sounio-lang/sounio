# Darwin PBPK in Sounio

Comprehensive PBPK (Physiologically-Based Pharmacokinetic) modeling in Sounio.

## Overview

This directory contains the Darwin PBPK platform ported to Sounio, showcasing:

- **Unit Safety**: Compile-time dimensional analysis (mg, L, h)
- **Epistemic Types**: Track confidence through computations
- **Refinement Types**: Physiological constraint validation
- **Effect System**: Explicit side effects (IO, Prob, Alloc)

## Modules

| File | Description |
|------|-------------|
| `darwin_pbpk_14comp.sio` | Core 14-compartment PBPK with Rodgers-Rowland Kp |
| `darwin_full_pbpk.sio` | Complete PBPK with ODE solver |
| `mechanistic_ddi.sio` | DDI with IVIVE and Monte Carlo UQ |
| `neural_ode_pbpk.sio` | Neural ODE for learning PK dynamics |
| `rbc_dynamics.sio` | Closed-loop erythropoiesis with EPO feedback |
| `darwin_validation_1232.sio` | 1,232-drug validation showcase |
| `ode_solver.sio` | ODE integration (Euler, RK4) |
| `caffeine_model.sio` | Caffeine PBPK |
| `metformin_model.sio` | Metformin PBPK |

## Validation Metrics (Darwin Platform)

```
GMFE: 1.64 (target < 2.0) ✓
78.1% within 2-fold (target > 70%) ✓
R²: 0.755
Correlation: 0.879
```

## Usage

```bash
cd compiler && cargo build --release
./target/release/souc check ../examples/pbpk/darwin_pbpk_14comp.sio --show-types
```

## Key Sounio Features for PBPK

```sounio
// Unit safety
let dose: mg = 100.0;
let volume: L = 5.0;
let conc: mg_per_L = dose / volume;  // Compile-time verified!

// Epistemic types
let cl: Knowledge[L_per_h, epsilon >= 0.80] = predict_clearance(drug);

// Refinement types
type PhysioVd = { vd: L | vd > 0.0 && vd < 2000.0 };
```

## References

- Darwin PBPK Platform: github.com/chiuratto-ai/darwin-pbpk-platform
- FDA DDI Guidance (2020)
- Chen et al. (2018) - Neural ODEs
