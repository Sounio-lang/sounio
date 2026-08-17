<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-pbpk28-cn-sigsegv-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: cursor-agent
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-pbpk28-cn-sigsegv-2026-08-16
-->

# Madaros imported PBPK28 CN: native run SIGSEGV (+ silent zeros)

**Date:** 2026-08-16  
**Scope:** `bin/souc` default Madaros engine — **native run** of a multi-module
program that imports `darwin_pbpk::tsit5_pbpk28::pbpk28_full_cn_step`.  
**Status:** MITIGATED in stdlib (numerically verified under Madaros). Compiler
root causes remain OPEN.  
**Related:** `MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`,
`MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md`,
epistemic trust / imported-module native path escalation (2026-07-14).

## Two bugs, one smoke

| # | Symptom under Madaros | lean_single | Stdlib mitigation |
|---|---|---|---|
| A | Call imported CN, then load a field of the returned `PBPKState28` → **SIGSEGV** | PASS | In-place `pbpk28_full_cn_step_mut` + thin by-value wrapper that `return`s a local copy |
| B | Same-frame second `while` reading local `a_v[i]` after Schur fill → **all-zero** reconstruct (silent wrong science) | PASS | Move reconstruct to `pbpk28_cn_apply_schur(..., &a_v, &b_v, &a_t, &b_t)` |

Bug A alone (mut + same-frame reconstruct loop) stops the crash but still ships
zeros. Both mitigations are required for Madaros `souc run` parity with lean.

## Bisect highlights (workspace control pod, 2026-08-16)

| Case | Madaros |
|---|---|
| Import `pbpk28_state_zero` + print | PASS |
| Call CN, discard return | PASS (likely DCE) |
| Call CN, then `st2.cv[0]` | **SIGSEGV** (bug A) |
| Local large-struct return / large by-value args | PASS |
| `return PBPKState28 { cv, ct }` from CN frame | still **SIGSEGV** |
| Exclusive-ref stores / local-array fill probes | PASS |
| Schur math + stash scalars with constant indices | PASS (correct `cv1≈0.351`) |
| Same Schur + same-frame reconstruct `while` on `a_v[i]` | zeros (bug B) |
| Schur + `pbpk28_cn_apply_schur` in another fn | **PASS** `cv1=0.351214` (lean `0.351215`) |
| Thin by-value wrapper after mut+apply | **PASS** |

## Mitigation (stdlib)

`stdlib/darwin_pbpk/tsit5_pbpk28.sio`:

- `pbpk28_cn_apply_schur` — reconstruct behind `&[f64;14]` (defined before mut)
- `pbpk28_full_cn_step_mut` — Schur fill, then apply
- `pbpk28_full_cn_step` — `var st = state; mut(&!st); return st`

Acceptance: [`docs/audit/repro/smoke_pbpk28_cn_imported.sio`](repro/smoke_pbpk28_cn_imported.sio)
plus numerical gate `cv1 ∈ (0.35, 0.36)` under **default Madaros**.

## Classification (blocker contract)

| Field | Value |
|---|---|
| Class | compiler / imported-module native (SRET + same-frame dynamic local-array reload) |
| Severity | S2 mitigated for PBPK28 CN callers; compiler bugs still open |
| Evidence | bisect table; Madaros/lean cv1 parity after mitigation |
| Owner | Madaros native lane (compiler); stdlib mitigation darwin_pbpk |
| Acceptance (stdlib) | smoke + cv1 gate under default Madaros |
| Acceptance (compiler) | pre-mitigation reconstruct-into-`out` + same-frame loop also PASS |
| Next | keep mitigation; file/keep Madaros dispatches for A and B separately |

## Non-goals

- Do not rename `tsit5_pbpk28.sio` in this dispatch.
- Do not claim general Madaros SRET / local-array families are closed.
- Sibling may still prefer `lean_single` until broader multimodule trust gates expand.
