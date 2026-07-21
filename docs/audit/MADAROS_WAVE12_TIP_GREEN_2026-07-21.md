<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave12-tip-green-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave12-tip-green-2026-07-21
-->

# Madaros Wave12 tip-green — #1382 imported f64 BSS multi-mod lock

**Date:** 2026-07-21  
**Role:** Wave12 Agent E (implementer)  
**Branch:** `fix/madaros-wave12e-imported-f64-bss-gate`  
**Tip measured:** `origin/main` @ `dbdf1029b` (Madaros v0.80.0)  
**Engine:** default `bin/souc` → Madaros (no lean_single pin)

## Mission

1. Measure `origin/main` for residuals **not** owned by A (`cd_exact` e2e) or C (`dep_begin`)
2. If a bold residual is open and fixable, ship it; else ship the regression gate for
   imported f64 BSS multi-mod (#1382)
3. PR required

## Measurement on origin/main (`dbdf1029b`)

| Gate / probe | Result under **stock prebuilt** | Result under **#1382-capable prebuilt** |
|--------------|----------------------------------|----------------------------------------|
| Wave11 tip-green (8 gates) | **GREEN** (prebuilt already carried Wave10/11 science) | **GREEN** |
| `madaros_imported_f64_const_gate` Defect A (leaf+pad) | **GREEN** (source #1380 already in prebuilt lineage for GLOBAL_VAR_INIT path, or pad-only wipe not hit) | **GREEN** |
| `madaros_imported_f64_const_gate` Defect A′ multi-mod BSS | **RED** — both A/B bits = `4612811918334230528` (2.5; last-init-wins) | **GREEN** — A=`4609434218613702656` (1.5), B=`4612811918334230528` (2.5) |
| lognormal science vertical | **GREEN** | **GREEN** |
| Bare `use m::{CONST}` Ident from main | **RED** (reads 0) — **not** owned this wave; explicit non-claim | still **RED** (helper path is the science gate) |

**Bold residual that was live on tip:** prebuilt lag after #1382. Source on `main`
remapped multi-mod BSS offsets (`ir_merge_place_and_remap_function` /
`ir_merge_modules_into`), but `bin/madaros-linux-x86_64` still emitted the collision.
That is an operational RED on the default `bin/souc` path — not a source gap and not
A/C ownership.

A/C remain open on deeper tracks (`cd_exact` e2e, `dep_begin` memory wall) and are
**not** claimed here.

## Ship

| Artefact | Role |
|----------|------|
| `bin/madaros-linux-x86_64` | Prebuilt refresh carrying #1380+#1382 multi-mod BSS remap (sha256 `824f687df58e56d062806b2791061ba8bf15cc3ee0c6231b531b06ee21c8eebe`) |
| `scripts/dev/madaros_wave12_tip_green_gate.sh` | 9-gate orchestrator + tip receipt writer |
| `artifacts/compiler/madaros_wave12_tip_green_receipt.v1.json` | machine-readable green tip receipt |
| this audit note | measurement + claim boundary |

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/ci/madaros_imported_f64_const_gate.sh
# MADAROS_IMPORTED_F64_CONST_GATE_OK

bash scripts/dev/madaros_wave12_tip_green_gate.sh
# MADAROS_WAVE12_TIP_GREEN_GATE_OK
# receipt: artifacts/compiler/madaros_wave12_tip_green_receipt.v1.json
```

### Gates locked by Wave12

1–8. Wave11 tip-green set (dual, order_spread, knowledge_method, global_array init,
    named_path, unsplit_oct, epistemic_trust/k95, global_array_ref)  
9. **imported_f64** — Defect A + Defect A′ multi-mod BSS (leaf+pad, lognormal,
   A_CONST≠B_CONST bits) — **new**

Wave11 tip-green remains as a historical subset
(`scripts/dev/madaros_wave11_tip_green_gate.sh`). Wave12 supersedes it for tip
regression lock.

## Claims

- All Wave11 tip-green claims (dual, order_spread, knowledge method, global array
  init + ref mutation, named-path print, unsplit oct_mul, k95i=2776)
- **Imported-module f64 constants survive multi-module parse** (Defect A /
  GLOBAL_VAR_INIT accumulate) under default Madaros
- **Multi-mod BSS offsets remapped on merge** (Defect A′): two imported modules with
  scalar f64 BSS read distinct values (`1.5` and `2.5`), not last-init-wins
- **lognormal_pdf science vertical** does not collapse through wiped dens constants

## Explicit non-claims

- `cd_exact_generic_i64` ELF end-to-end (Agent A)
- multi-module `dep_begin` body-lowering memory wall (Agent C)
- bare cross-module `use m::{CONST}` Ident from main without a same-module helper
  (still reads 0; science path uses helpers / dens functions)
- all stdlib dual pairs / full linalg native parity / full Root-2 census

## Why prebuilt, not source-only

#1382 already merged the BSS remap **source**. Without refreshing
`bin/madaros-linux-x86_64`, default `bin/souc` still demonstrated the A′ collision
on this tip. Tip-green that only runs under a private `MADAROS_RAW_BIN` rebuild is
not a public lock. Wave12e therefore ships the prebuilt + the 9th gate together.

## Re-run matrix

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE MADAROS_RAW_BIN SOUNIO_MADAROS_BIN
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave11_tip_green_gate.sh
bash scripts/ci/madaros_imported_f64_const_gate.sh
bash scripts/dev/madaros_wave12_tip_green_gate.sh
```
