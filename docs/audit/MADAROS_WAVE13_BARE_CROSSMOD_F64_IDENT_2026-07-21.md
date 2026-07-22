<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave13-bare-crossmod-f64-ident-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave13-bare-crossmod-f64-ident-2026-07-21
-->

# Madaros Wave13 — bare cross-module f64 Ident

**Date:** 2026-07-21  
**Role:** Wave13 Agent D (implementer)  
**Branch:** `fix/madaros-wave13-bare-crossmod-f64-ident`  
**Tip measured:** `origin/main` post-#1392 (`94bd1dc48` / `0b6809f4d` merge of cd_exact e2e)  
**Engine:** default `bin/souc` → Madaros (no lean_single pin)

## Mission

1. Measure `origin/main` after #1392 for residuals **not** owned by A (`into-acc` / #1393) or B (spec DCE)
2. Ship **one bold residual** with PR
3. Candidates from dispatch: bare cross-mod f64 Ident, multi-stmt args global list, tip-gate reds

## Measurement on tip (pre-fix)

| Probe | Result | Owner |
|-------|--------|-------|
| Wave12 tip-green / imported_f64 helper path | GREEN | closed Wave11e/#1380 + A′/#1382 + Wave12e prebuilt |
| `cd_exact` e2e | GREEN after #1392 | was A-track; landed |
| into-acc lower scaffolding | open PR #1393 | **Agent A — not claimed** |
| Bare `use m::{CONST}` Ident from main | **RED** — `ident_bits 0`, helper `1.5` bits | **this ship** |
| Multi-stmt pure global list `[multi(),20,30]` | GREEN (#1387) | closed |
| Multi-stmt **with args** `[with_arg(9),…]` | RED (all zeros) | residual left open (not this PR) |

Repro (tip, default Madaros):

```sounio
use imported_module_f64_const_a::{A_CONST, get_a}
// get_a() → bits 4609434218613702656 (1.5)
// A_CONST  → bits 0
```

Root cause: multi-mod seed lower runs **before** dep merge. External preseed registered
structs + free-fn signatures only (`lowerer_preseed_external_struct_items_mut`);
module-level BSS (`ItemFn` with `fn_def = None`) was skipped. Seed body Ident of
`A_CONST` missed every local/BSS slot → `report_error`/`emit_unit` → runtime 0.
Same-module helpers lower inside the dep where the BSS slot exists, so they stayed green.

## Ship

| Artefact | Role |
|----------|------|
| `self-hosted/ir/lower.sio` | `lowerer_preseed_external_bss_globals_mut` + seed-only call after own items in `with_externs` |
| `self-hosted/compiler/module_frontend.sio` | BSS-by-name DEDUP on merge; name-based `IrLoadGlobal`/`IrStoreGlobal` remap; non-deduped BSS growth |
| `tests/run-pass/imported_module_f64_const_bare_ident.sio` | bare Ident + helper parity witness |
| `scripts/ci/madaros_imported_f64_const_gate.sh` | 4th arm: bare Ident |
| `bin/madaros-linux-x86_64` | prebuilt refresh so default `bin/souc` carries the fix (sha256 `6d2def5226417bf6276c96cfc7e02d9a98d21ead479b87b63f73b2ad17556b8c`) |
| this audit | claim boundary |

## Claims

- Bare `use m::{CONST}` of an imported **scalar f64** module global from main loads the
  same BSS value as a same-module helper (`get_*`), under default Madaros multi-mod.
- Seed BSS layout remains merge-compatible: own globals first, then deps 1..n.
- Preseeded BSS slots DEDUP on merge (no double-init / double-offset collision).
- Prior Defect A / A′ / helper-path gates stay green.

## Explicit non-claims

- Agent A into-acc production wiring (#1393)
- Agent B spec DCE
- Multi-stmt **argument-bearing** pure calls in global element-list init (still red)
- Bare Ident of imported **aggregate** BSS / arrays from main (not measured)
- Full stdlib dens-constant bare-Ident census

## Re-run

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE MADAROS_RAW_BIN SOUNIO_MADAROS_BIN
ulimit -s unlimited 2>/dev/null || true

bash scripts/ci/madaros_imported_f64_const_gate.sh
# MADAROS_IMPORTED_F64_CONST_GATE_OK

./bin/souc run tests/run-pass/imported_module_f64_const_bare_ident.sio
# BARE_CROSSMOD_F64_IDENT_OK
```
