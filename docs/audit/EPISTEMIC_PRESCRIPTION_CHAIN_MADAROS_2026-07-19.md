<!-- docs:meta
topic_id: repo.docs.audit.epistemic-prescription-chain-madaros-2026-07-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-prescription-chain-madaros-2026-07-19
-->

# Epistemic Prescription Chain + Madaros D1 GUM fix — 2026-07-19

## Ousadia claim

> Uncertainty reaches the dose **decision**, expanded coverage uses a **correct**
> Student-t factor under **default Madaros multi-module import**, and over-confident
> prescribe paths remain **compile-fail** witnesses.

## What landed

### 1. D1 GUM-site fix (`stdlib/epistemic/gum.sio` — `dof_to_i64`)

Two defects stacked on the same cast:

| Layer | Failure | Fix |
|---|---|---|
| Madaros D1 (#983) | bare `dof as i64` on an f64 **parameter** is a bitcast → k95 stuck at 1.960 | arithmetic-source before cast |
| Numerical | Welch–Satterthwaite often yields ν_eff = 3.999999 for ideal ν=4; trunc → t95(3)=3.182 | round half-up (`dof + 0.5`) then cast |

```sio
fn dof_to_i64(dof: f64) -> i64 with Div, Panic {
    if dof > 1.0e8 { return 10000 }
    let d = dof + 0.5
    return d as i64
}
```

Verified under `./bin/souc` multi-module import:

- Constant Type-A smoke: n=5 → ν=4 → **k95≈2.776**
- Clinical path (`0.15 * q_std` Type-A, zero Type-B): ν_eff prints 3.999999 → rounded table lookup → **k95≈2.776**

Does **not** land the global param `scalar_kind=2` fix (#983 full) — that unmasks D5 (#986). This is the safe stdlib GUM-critical workaround from the escalation doc.

### 2. Prescription chain E2E

`tests/stdlib/clinical/test_prescription_chain_e2e.sio` +
`scripts/epistemic_prescription_chain_e2e_gate.sh`

- Engine: **default Madaros** (no `SOUNIO_SOUC_ENGINE=lean_single`)
- Imports `epistemic::gum` (multi-module native)
- Type-A-dominant budget (n=5, Type-B=0) so ν_eff stays finite — proves Student-t k95, not normal 1.96
- Vanco AUC/MIC + **GUM U95 band decision** + Knightian CL prior (merge max severity; renal CrCl=20 → REFUSE)
- Compile-fail witnesses present
- Avoids `print(i32)` under multi-module (observed SIGSEGV on fail path)

### 3. Explicit boundary

`epistemic::knowledge` import still **blocked/segfaults** under Madaros (D3). Gate documents
this; does not claim Knowledge&lt;T&gt; native import.

## Gate

```bash
# NO lean_single pin
bash scripts/epistemic_prescription_chain_e2e_gate.sh
# → EPISTEMIC_PRESCRIPTION_CHAIN_E2E_GATE_OK
```

Measured (this worktree, default Madaros):

```
RX_CHAIN q_std=532.967869 u_c=35.752571 k95=2.775999 u95=99.249138 dof=3.999999 decide=ADJUST
RX_CHAIN q_ren=2006.880733 decide=REFUSE
EPISTEMIC_PRESCRIPTION_CHAIN_E2E_OK
```

## claims_not_made

- Knowledge&lt;T&gt; import under Madaros  
- Full D1/#983 without D5/#986  
- Bedside product / NONMEM FOCE  
- numpy/sklearn  

## Priority next (compiler)

D3 Knowledge multi-module segfault; then D5+D1 general param float support.
