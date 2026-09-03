<!-- docs:meta
topic_id: repo.docs.audit.sigsegv-pbpk28-private-struct-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.sigsegv-pbpk28-private-struct-2026-06-02
-->

# PBPK28 SIGSEGV — private struct field access across bundle imports

**Date:** 2026-06-02  
**Symptom:** `tests/run-pass/pbpk28_m5_gum_4th_order.sio` SIGSEGV after lognormal/AUC checks, at `m5_pbpk28_convergence_budget()`.  
**Verdict:** Not stack overflow; **compiler emitted broken code** for cross-file access to private struct fields.

## Bisection

| Step | Result |
|------|--------|
| `extract_auc_inverse_derivatives` only | exit 0 (no ODE) |
| `extract_pbpk28_derivatives(..., t_end=0)` | SIGSEGV before fix |
| `extract_pbpk28_derivatives(..., 168)` | SIGSEGV before fix |
| `m5_variance_zero()` return | exit 0 |
| After `pub struct PBPKParams28` / `PBPKState28` | extract + full E2E exit 0 |

GDB (pre-fix): `rip≈0x409759`, `r8=0x4065000000000000` (garbage f64), useless symbols (stripped ELF).

## Root cause

`stdlib/darwin_pbpk/core/pbpk28_params.sio` declared:

```sio
struct PBPKState28 { ... }
struct PBPKParams28 { ... }
```

`stdlib/darwin_pbpk/cumulants.sio` and `stdlib/darwin_pbpk/tsit5_pbpk28.sio` import these types and read/write fields (`p.cl_central`, `state.cv[i]`, …). The bundled compile places each import in a **distinct file region** (`TK==77` markers). `st_private_access_denied` flags violations but the compiler **still emitted** stores/loads; runtime touched invalid addresses → SIGSEGV.

Compile log (when epistemic modules pulled in) showed hundreds of `error: private struct field access [struct=PBPKParams28]` — same class as the ODE epistemic fix (`pub struct EpistemicDual`).

## Fix

```sio
pub struct PBPKState28 { ... }
pub struct PBPKParams28 { ... }
```

No `lean_single.sio` change required for this bug class.

## Post-fix E2E note

`pbpk28_m5_gum_4th_order.sio` runs to completion (exit 0) but may print `PBPK28 budget failed` / `M5_GUM_FOURTH_ORDER_CUMULANT_BUDGET_OUTPUT` — **numerical acceptance**, not a crash. Tune thresholds or MC reference separately if full `PASS` banner is required.

## Repro commands

```bash
export SOUNIO_SOUC_BIN=artifacts/self-hosted/souc-self-hosted-x86_64
bash scripts/ci/souc-native-wrapper.sh run tests/run-pass/pbpk28_extract_only.sio
bash scripts/ci/souc-native-wrapper.sh run tests/run-pass/pbpk28_m5_gum_4th_order.sio
```

Forensic fixtures: `tests/run-pass/pbpk28_extract_only.sio`, `pbpk28_deriv_zero.sio`, `pbpk28_budget_bisect.sio`.
