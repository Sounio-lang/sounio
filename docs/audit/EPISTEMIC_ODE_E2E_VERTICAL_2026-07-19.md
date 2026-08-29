<!-- docs:meta
topic_id: repo.docs.audit.epistemic-ode-e2e-vertical-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-ode-e2e-vertical-2026-07-19
-->

# Epistemic multi-compartment ODE E2E — 2026-07-19

## Scope

| | |
|---|---|
| Branch | `feat/epistemic-ode-e2e` |
| Driver | `tests/stdlib/ode/test_epistemic_ode_e2e.sio` (self-contained) |
| Gate | `bash scripts/epistemic_ode_e2e_gate.sh` → `EPISTEMIC_ODE_E2E_GATE_OK` |
| Engine | **`lean_single` only** |
| Oracle | closed-form 1-cmt GUM (`scripts/epistemic_ode_e2e_oracle.py`) |

## What is proven

1. **1-cmt IV** analytic `C(t)` with GUM linearisation on `(CL, V)` and ISO-style **budget fractions** (CL vs V).
2. **2-cmt linear IV** RK4 on amounts with step-wise state-uncertainty propagation; mass elimination + positive `C1` + `u_c1 > 0`.
3. **2-state harmonic** multi-state smoke at `t=π/2`.

## Package-kill claim (honest)

Structural **slice** joint of:

- DiffEq.jl-class **integration of multi-state linear systems**, and  
- Measurements.jl-class **standard uncertainty + parameter budget on the endpoint**.

Not full SciML. Not API parity.

## claims_not_made

- full DiffEq.jl / SciML events / DAE / stiff zoo  
- Madaros multi-module import of epistemic ODE stdlib  
- bedside dosing / NONMEM FOCE  
- NumPy / sklearn  

## How to run

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/epistemic_ode_e2e_gate.sh
```
