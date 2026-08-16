<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-pbpk28-cn-segfault-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-08-16
validated_by: cursor-agent
source_of_truth: docs/audit/MADAROS_IMPORTED_PBPK28_CN_SIGSEGV_2026-08-16.md
-->

# Madaros imported PBPK28 CN: native run SIGSEGV

**Date:** 2026-08-16  
**Scope:** `bin/souc` default Madaros engine — **native run** of a multi-module
program that imports `darwin_pbpk::tsit5_pbpk28::pbpk28_full_cn_step` (Crank–
Nicolson stepper; filename is historical).  
**Status:** OPEN — workaround mandatory for sedation-weaning sibling package.  
**Related:** `MADAROS_MULTIMODULE_FALLBACK_SEGFAULT_2026-06-30.md`,
`MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md`,
epistemic trust / imported-module native path escalation (2026-07-14).

## Symptom

`souc check` of a caller that `use`s `pbpk28_full_cn_step` **passes**.  
`souc run` under default Madaros (native-v2) **SIGSEGV** even for a minimal
smoke that builds Params28 locally and takes one CN step.

`SOUNIO_SOUC_ENGINE=lean_single souc run …` completes with correct CL_obs gates
(sibling `scripts/check.sh` parity morphine / clonidine / fentanyl / weaning_e2e).

## Workaround (production for sibling)

Sibling repo `sounio-pbpk-sedation-weaning`:

```bash
export SOUNIO_STDLIB_PATH=/path/to/sounio/stdlib
(cd sibling && SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run src/scenarios/parity_fentanyl.sio)
```

Do **not** treat Madaros native run green as a gate until this dispatch closes.

## Minimal repro sketch (forensic next step)

Keep both files under one CWD with stdlib resolution; prefer a **local** Params28
literal (no drug module) + one imported CN call:

```sounio
// smoke_pbpk28_cn.sio  (illustrative — flesh out with pbpk28_state_zero + params)
use darwin_pbpk::core::pbpk28_params::{pbpk28_state_zero, PBPKParams28}
use darwin_pbpk::tsit5_pbpk28::{pbpk28_full_cn_step}

fn main() -> i64 with Mut, Div, IO, Panic {
    let pa = /* minimal PBPKParams28 with positive v,q,cl */
    var st = pbpk28_state_zero()
    let st2 = pbpk28_full_cn_step(st, pa, 0.0, 0.05)
    println(st2.cv[0])
    return 0
}
```

Bisect order (do not skip):

1. Same smoke **without** `use` — inline a trivial local stub of `pbpk28_full_cn_step`
   (if crash disappears → imported-module native path).
2. Import only `pbpk28_state_zero` (no CN) — isolate Params28 / array returns.
3. Confirm `with Mut` on `main` and on CN (prior D3 exclusive-ref fragility).
4. Compare `souc compile -o` ELF under Madaros vs lean_single; `nm`/`objdump` only
   after crash is reproducible under a locked Madaros build
   (`scripts/dev/souc-build-lock.sh` if rebuilding).

## Classification (blocker contract)

| Field | Value |
|---|---|
| Class | compiler / imported-module native path |
| Severity | S2 (science gates blocked on default engine; lean_single is correct) |
| Evidence | sibling check.sh lean_single ALL PASSED; Madaros run SIGSEGV on same sources |
| Owner | Madaros multimodule native lane |
| Acceptance | minimal multimodule PBPK28 CN smoke `souc run` exit 0 under default Madaros |
| Next | land minimal repro under `tests/run-pass/` or `docs/audit/repro/` + CI note |

## Non-goals

- Do not rename `tsit5_pbpk28.sio` in this dispatch (historical name; CN-only).
- Do not re-inline CN copies into the sibling (architecture already corrected).
- Do not claim Tsit5-14 fix closes this path (separate segfault family).
