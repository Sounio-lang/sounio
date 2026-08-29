<!-- docs:meta
topic_id: repo.docs.handoff.deep-four-lanes-2026-07-25
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.deep-four-lanes-2026-07-25
-->

# Deep four lanes — 2026-07-25

Branch: `research/particle-exp123-20260725`  
Package: `experiments/deep_four/`, receipts in `results/deep_four/`

## L1 — β on Madaros

| Test | Madaros (default) pre-fix | Madaros post `is_float` fix | lean_single β¹¹ |
|---|---|---|---|
| `a1=k.value; a1*a1` value | **aa=0** (int imul on IEEE bits) | **aa=1, A=4** | PASS |
| `Var(a·b−a·b)=0` | often PASS | PASS | PASS |
| Fano FO `Var(4a₁²)=64σ²` | FAIL (garbage A) | **FAIL_HONEST** (A=4 ok, **var still 0**) | **PASS** |

**Root cause (value):** `ir_register_knowledge_layout` had `is_float: 0` on value/variance/confidence.  
**Root cause (variance):** measure used int `unc*unc`; shadows preferred over Knowledge field-1; const×x double-counted; same-ident mul FO broken.

**Fix shipped (commit `4fbc89610`):** is_float=1; `ir_binop_typed` for σ²; field-1 variance load; let-RHS FO bind; same-ident `Var(x*x)=4x²Var`; const×x = c²Var(x); sub cancel on shared variance slot.

**Verdict:** `MADAROS_GUM_FO_CLOSED` — default `bin/souc` PASSes Fano FO 64σ² and product cancel. Rebuild: `SOUC_BIN=./bin/souc-lean-single-x86_64 make build-madaros`.

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
