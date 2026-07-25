# Deep four lanes — 2026-07-25

Branch: `research/particle-exp123-20260725`  
Package: `experiments/deep_four/`, receipts in `results/deep_four/`

## L1 — β on Madaros

| Test | Madaros (default) | lean_single β¹¹ |
|---|---|---|
| `Var(a·b−a·b)=0` | often PASS | PASS |
| Fano FO `Var(4a₁²)=64σ²` | **FAIL_HONEST** (garbage value / zero var) | **PASS** |

**Verdict:** `MADAROS_GUM_GAP`. β¹⁰/β¹¹ live on lean_single seed only.  
**Next port:** IR variance binding in `self-hosted/ir/lower.sio` + native codegen, not a lean_single-only shadow.

## L2 — Multi-component octonion associator

Pure f64 path (engine-independent): 8-component associator for Fano triple,  
FD GUM, component chain FO, MC. Expect `PASS` with `A0=4`, `d7=2`, `Var=64σ²`.

## L3 — FAERS demography residual

Triple-level join DrugBank × demographics (case-level openFDA **not in repo**).  
OLS `asym ~ age + frac_female`, residual Fano contrast.  
**Observation:** residual diff N−F is **negative** after age control (H1 not supported).  
Pearson asym~age ≈ −0.61 (age strongly confounds raw asymmetry).

## L4 — lean_single fixed point

Self-compile: seed → genA → genB; require `md5(genA)==md5(genB)`.  
Optional: `DEEP_FOUR_FIXEDPOINT=1 bash scripts/ci/deep_four_lane_gate.sh`.  
Not the full `make build` JIT tower.

## Gate

```bash
bash scripts/ci/deep_four_lane_gate.sh
DEEP_FOUR_FIXEDPOINT=1 bash scripts/ci/deep_four_lane_gate.sh  # + L4
```
