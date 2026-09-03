<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-f64-bss-arith-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-f64-bss-arith-2026-07-22
-->

# Madaros — imported-module f64 BSS arithmetic (Wave15 D)

**Date:** 2026-07-22  
**Lane:** Wave15 Agent D — science multi-mod surface  
**Status:** **FIXED** (source + rebuilt Madaros)  
**Severity:** high for science path — `stats::densities::lognormal_pdf` under multi-mod

## Claim

> Under **default Madaros multi-module** (into-acc lower), a module-level `let K: f64`
> used in **same-module** arithmetic (`K + 1.0`, `0.0 - K`, `K * 2.0`) produces IEEE
> float results — not `cvtsi2sd` of the bit pattern. Science vertical:
> `stats::densities::lognormal_pdf(1,0,1) ≈ 0.3989422804014327 = 1/√(2π)`.

## Symptom (stock tip pre-fix)

| Program | Result |
|---|---|
| Minimal imported f64 const bits (Defect A) | **GREEN** |
| Multi-mod BSS remap A/B (Defect A′) | **GREEN** |
| Bare cross-mod Ident from main (Wave13) | **GREEN** |
| `normal_pdf` / `exponential_pdf` multi-mod | **GREEN** (no BSS f64 arith) |
| **`lognormal_pdf(1,0,1)` multi-mod** | **RED** → `~1e-300` (not 1.0) |
| Imported leaf `DE + 1.0` | **RED** → `~4.606e18` |

`f64_to_bits(DE)` and `return DE` / `de_exp(DE)` stayed correct — only **arithmetic
on the loaded global** mis-typed.

Decoded wrong class: `cvtsi2sd` of IEEE bits of `0.9189…`
(`4606452282016710325` → `~4.606e18`). Then `de_exp(0.0 - garbage)` hits the
`>709 → 1e300` clamp → `1/1e300 ≈ 1e-300`.

## Root cause

Wave13 seed external BSS preseed allocates dep module-level f64 slots **into the
seed lowerer** (and records `global_types` there). Dep bodies are lowered later via
**into-acc**:

1. `lowerer_from_acc_module` starts with **empty** `global_types`
2. `lowerer_preseed_dep_items_into_acc_mut` skipped type re-record when the BSS
   name already existed (`existing >= 0`)
3. Ident load of `DE` omitted `ir_mark_float_reg`
4. Codegen `IrLoadGlobal` marks the temp **INT**; float binops then `cvtsi2sd`

Orthogonal to Defect A (GLOBAL_VAR_INIT wipe → zero const → pdf=1.0) and Defect A′
(BSS offset collision). Const **init** was correct; **typing of arithmetic** was not.

## Fix

`self-hosted/ir/lower.sio` — `lowerer_preseed_dep_items_into_acc_mut` BSS branch:
when the slot already exists as `IR_STRATEGY_BSS_GLOBAL`, still call
`lowerer_record_global_type_mut` so into-acc body lower emits float markers.

## Gate

```bash
bash scripts/madaros_imported_f64_bss_arith_gate.sh
# → MADAROS_IMPORTED_F64_BSS_ARITH_GATE_OK
```

Arms:

1. `tests/run-pass/imported_f64_bss_arith_main.sio` — micro add/sub/mul
2. `tests/run-pass/imported_f64_lognormal_science.sio` — densities science
3. dual import + `cd_exact` non-regression

Receipt: `artifacts/compiler/madaros_imported_f64_bss_arith_receipt.v1.json`

Also recovers `scripts/ci/madaros_imported_f64_const_gate.sh` lognormal arm under
rebuilt Madaros.

## Claim boundary

**Claims:**

- Same-module f64 BSS arithmetic inside imported modules under into-acc multi-mod
- `lognormal_pdf(1,0,1)` multi-mod ≈ `1/√(2π)` under default Madaros
- dual + cd_exact stay green

**Does not claim:**

- Full `stats::regression` / OLS multi-mod method residual
- `print_f64` display path
- Wave15 A parser multi-stmt / B print_f64 / C large multi-mod capacity
- lean_single substitution for the full stats suite

## AI disclosure

Localisation and fix under human direction. GAIDeT-ICMJE 2025.
