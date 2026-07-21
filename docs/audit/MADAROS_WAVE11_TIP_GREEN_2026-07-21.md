<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave11-tip-green-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave11-tip-green-2026-07-21
-->

# Madaros Wave11 tip-green measurement + regression lock

**Date:** 2026-07-21  
**Role:** Wave11 Agent F (implementer)  
**Branch:** `fix/madaros-wave11-tip-green`  
**Tip measured:** `origin/main` @ `bbb4a84aa` (Madaros v0.80.0)  
**Engine:** default `bin/souc` → Madaros (no lean_single pin)

## Mission

1. Run Wave10 tip-green + epistemic trust + dual on `origin/main`
2. Document RED if any
3. Ship a fix for a red residual **or** extend tip-green with Wave10/11 locks (k95, global ref, tip receipt)
4. PR `fix/madaros-wave11-tip-green`

## Measurement on origin/main (`bbb4a84aa`)

| Gate | Script | Result | Notes |
|------|--------|--------|-------|
| dual | `scripts/madaros_dual_import_gate.sh` | **GREEN** | `MADAROS_DUAL_IMPORT_GATE_OK` |
| wave10 tip-green (6-in-1) | `scripts/dev/madaros_wave10_tip_green_gate.sh` | **GREEN** | all six sub-gates pass |
| epistemic trust | `scripts/epistemic_trust_gate.sh` | **GREEN** | Section A + **k95i=2776** |
| global array ref | `scripts/ci/madaros_global_array_ref_gate.sh` | **GREEN** | Defect B / wave10e |

**RED count: 0.** No science/compiler residual in this Agent F scope that is not already owned by deeper open tracks (`claims_not_made` below).

### Epistemic trust detail (k95)

Finite-dof coverage factor on Type-A-dominant budget:

- **PASS:** `k95i=2776` (= `t95(4)`)
- Collapse value that would reintroduce D1 bitcast (`k95i=1960`) was **not** observed

Wave10 tip-green still lists `gum_k95_f64_i64_cast_fixed` under `claims_not_made` even though Section A of the trust gate already gates it. Wave11 promotes that claim into the tip-green orchestrator.

### Global array ref detail (Defect B)

`tests/run-pass/global_array_ref_mut.sio` → `GLOBAL_ARRAY_REF_MUT_OK` under default Madaros. Fixed wave10e (#1364) but not previously locked inside tip-green (Wave10 only locked **init**, not **ref mutation**).

## Ship (because tip was green)

| Artefact | Role |
|----------|------|
| `scripts/dev/madaros_wave11_tip_green_gate.sh` | 8-gate orchestrator + tip receipt writer |
| `scripts/ci/madaros_global_array_ref_gate.sh` | emit `MADAROS_GLOBAL_ARRAY_REF_GATE_OK` sentinel |
| `artifacts/compiler/madaros_wave11_tip_green_receipt.v1.json` | machine-readable green tip receipt |
| this audit note | measurement + claim boundary |

```bash
bash scripts/dev/madaros_wave11_tip_green_gate.sh
# MADAROS_WAVE11_TIP_GREEN_GATE_OK
# receipt: artifacts/compiler/madaros_wave11_tip_green_receipt.v1.json
```

### Gates locked by Wave11

1. dual  
2. order_spread  
3. knowledge_method  
4. global_array (init)  
5. named_path  
6. unsplit_oct  
7. **epistemic_trust** (includes k95i=2776) — **new**  
8. **global_array_ref** (Defect B) — **new**

Wave10 tip-green remains as a historical subset (`scripts/dev/madaros_wave10_tip_green_gate.sh`). Wave11 supersedes it for tip regression lock.

## Claims

- dual gum+knowledge import native
- order_spread N=4 CPC native
- knowledge method form parity
- global array init (i64/f64/i8)
- named path import `print_f64` default and `-O`
- unsplit `oct_mul` no re-entry
- **gum k95 f64→i64 cast fixed** (promoted from Wave10 non-claim)
- **epistemic trust Section A native**
- **global array `&!`/`&` BSS ref mutation** (Defect B)

## Explicit non-claims

- all stdlib dual pairs
- language-level `Knowledge<T>` generic import
- multi-module IrModule memory wall closed
- `cd_exact_generic_i64` ELF
- full Root-2 census closed
- f64-param bitcast free for *all* call shapes beyond the gated witnesses
- full linalg native parity

## Re-run commands

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/madaros_dual_import_gate.sh
bash scripts/epistemic_trust_gate.sh
bash scripts/dev/madaros_wave10_tip_green_gate.sh
bash scripts/ci/madaros_global_array_ref_gate.sh
bash scripts/dev/madaros_wave11_tip_green_gate.sh
```
