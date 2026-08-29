<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave15-showcase-2026-07-22
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave15-showcase-2026-07-22
-->

# Madaros Wave15 — public showcase (Wave13–14 locks + honest residual)

**Date:** 2026-07-22  
**Role:** Wave15 Agent F (showcase packaging)  
**Branch:** `test/madaros-wave15-showcase`  
**Engine:** default `bin/souc` → **Madaros v0.80.0** (no lean_single pin)  
**Audience:** public / external readers — “what the tip can prove today, without inventing green”

## Mission

Waves 13–14 closed science tip locks and residual gates on `main`. Wave15 **packages** that surface as a single public orchestrator + receipt + claim boundary.

1. Orchestrator `scripts/dev/madaros_wave15_showcase_gate.sh`
   - Wave13 science pillars (dual, order_spread, k95, **required** `cd_exact` + e2e)
   - Wave13 tip locks that remain green on tip (knowledge method, global_array incl **Wave13e** call-list args, named_path, unsplit oct, global_array_ref, imported f64 **core** = minimal + BSS + bare Ident)
   - Wave14 locks: bare_float_arith, Root-2 method + multimodule **chain**, #913 array by-value, #921 thinlink residual
   - Honest residual probe: denser imported f64 **lognormal science** vertical (red on tip after #1405 prebuilt)
2. Machine receipt `artifacts/compiler/madaros_wave15_showcase_receipt.v1.json`
3. PR `test/madaros-wave15-showcase`

## One-command public proof

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave15_showcase_gate.sh
# MADAROS_WAVE15_SHOWCASE_GATE_OK
# receipt: artifacts/compiler/madaros_wave15_showcase_receipt.v1.json
# showcase_verdict: pass_with_imported_f64_science_residual  (tip after #1405)
#   or pass_full when denser lognormal science is green
```

Promote denser lognormal science to required (only when tip proves it):

```bash
REQUIRE_IMPORTED_F64_SCIENCE=1 bash scripts/dev/madaros_wave15_showcase_gate.sh
```

Optional Wave13 showcase probe (records; not required — reds on tip via lognormal inside wave12 tip-green):

```bash
RUN_WAVE13_SHOWCASE_PROBE=1 bash scripts/dev/madaros_wave15_showcase_gate.sh
```

## Prebuilt note — tip honesty

Stock tip prebuilt is the #1405 Wave13e rebuild (`bin/madaros-linux-x86_64`). Measured 2026-07-22:

| Prebuilt ancestor | Wave13e `call_list_args` | Lognormal science |
|-------------------|--------------------------|-------------------|
| post-#1404 (`3d0932795`) | RED (`0 0 0`) | **GREEN** |
| tip #1405 (`3e7ed9f52`) | **GREEN** (`30 1 2`) | **RED** (~1e-300 bits) |

Wave15 **does not invent green** for the denser lognormal residual. It locks Wave13e (via `global_array_init`) and imported f64 **core** (minimal + BSS + bare Ident), and records lognormal as `honest_probe_residual`.

Multi-stmt paramful global list fold remains residual fail-closed inside `madaros_global_array_init_gate.sh` (expects BSS zeros). Not a free claim.

If a later source fix restores lognormal without losing Wave13e, rebuild and re-run with `REQUIRE_IMPORTED_F64_SCIENCE=1`:

```bash
scripts/dev/souc-build-lock.sh make build-madaros
MADAROS_RAW_BIN=artifacts/self-hosted/madaros REQUIRE_IMPORTED_F64_SCIENCE=1 \
  bash scripts/dev/madaros_wave15_showcase_gate.sh
```

## What the tip proves today (default Madaros)

| Pillar | Gate | Public claim when green |
|--------|------|-------------------------|
| Dual import | `scripts/madaros_dual_import_gate.sh` | `gum` + `knowledge` import + native run |
| Order spread | `scripts/madaros_order_spread_native_gate.sh` | CPC N=4 exact spread ≈ `2.044226` |
| GUM k95 | `scripts/epistemic_trust_gate.sh` | Finite-dof **k95i=2776** (= t95(4)), not D1 collapse 1960 |
| CD exact (required) | `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | Generic sedenion ZD over i64 — **ZD PROVED** |
| CD exact e2e (required) | `scripts/madaros_cd_exact_e2e_gate.sh` | specialized_collapse path |
| Global array init | `scripts/dev/madaros_global_array_init_gate.sh` | Wave6–13e folds incl paramful single-stmt call args |
| Imported f64 core | inline (minimal + BSS + bare Ident) | Defect A / A′ / Wave13 bare cross-mod Ident |
| Bare float arith | `scripts/madaros_bare_float_arith_gate.sh` | bare cos/sin/sqrt/exp participate in f64 mul/add |
| Root-2 method | `scripts/madaros_root2_method_gate.sh` | associated + same/multi-module instance methods |
| Root-2 multimodule chain | `scripts/madaros_root2_multimodule_method_gate.sh` | imported Epistemic method **chain** |
| #913 by-value array | `scripts/ci/madaros_imported_array_byvalue_gate.sh` | imported `[f64;N]` by-value payload preserved |
| #921 thinlink | `scripts/madaros_thinlink_921_residual_gate.sh` | default multi-mod thin-link fail class closed |

## Measurement (this packaging run)

Recorded in `artifacts/compiler/madaros_wave15_showcase_receipt.v1.json` (regenerate with the gate).

| Gate | Class | Expected on tip #1405 |
|------|-------|------------------------|
| dual … thinlink_921 (required set) | required | **GREEN** |
| imported_f64_lognormal_science | honest residual | **RED** unless fixed |
| multi-stmt paramful (inside global_array) | residual fail-closed | expects `0 0` (not claimed fixed) |

- **Engine:** Madaros v0.80.0
- **showcase_verdict:** `pass_with_imported_f64_science_residual` on tip stock prebuilt
- **overall:** `pass` → `MADAROS_WAVE15_SHOWCASE_GATE_OK` when all required gates green

## Waves 1–15 — headline merges (residual campaign)

| Wave | Headline merge(s) | What landed |
|------|-------------------|-------------|
| **1** | Early dual / free-fn import surface (#1245 family; D3 knowledge #1203) | Checker prefers defining-module free-fns; dual `gum`+`knowledge` import becomes check-clean |
| **2** | Integer print routing (#1261); CD exact A-track land (#667 historic) | Computed integers print via `print_int`; ExactRing / primitive CD dispatch substrate |
| **3** | Thin-link kind-9 parity (#1271) | Compact-path size+emit parity for multi-module thin-link |
| **4** | Large-frame stack probe (#1283) | Page-by-page stack probe — unblocks deep native frames (oct / algebra) |
| **5** | Bare `var` binding (#1289) | Parser accepts `var` as binding name — unblocks knightian / epistemic leaves |
| **6** | Global array-repeat init (#1305) + f64 global init (#1325) | Honour `[V; N]` BSS init; science globals stop reading as zero/garbage |
| **7** | MIR_MAX_INSTRS 1024→4096 (#1317) | Fail-closed MIR capacity lift for larger multi-module bodies |
| **8** | Float-slot table + i8 BSS (#1333, #1337) | `is_float_slot` 256→2048; signed i8 BSS load + element-list fold |
| **9** | Global-init ship + len-1 array (#1353, #1350); knowledge method residual (#1344) | Cast/ident fold, fail-closed list init, length-1 IndexGet; Epistemic free/method parity gate |
| **10** | Tip-green lock (#1355); gum k95 Section B→A (#1357); global array ref Defect B (#1364); cd_exact memory wall slim (#1361) | Six-gate tip lock; **k95i=2776** trustworthy; BSS `&!`/`&` mutation; multi-mod merge copies slimmed |
| **11** | Tip-green promotion (#1379); imported f64 const Defect A (#1380); multi-mod BSS offset remap (#1382) | Eight-gate tip lock; imported f64 module constants preserved (minimal + science at that tip) |
| **12** | Public residual showcase (#1384); tip-green imported f64 BSS (#1385) | Honest `cd_exact` probe + science tip pedestal |
| **12e** | `cd_exact` e2e under default Madaros (**#1392**) | specialized_collapse + i64 mono markers → **ZD PROVED** |
| **13** | Full-green showcase packaging (#1396) | `REQUIRE_CD_EXACT` default on; public claim of `cd_exact` green with receipt |
| **13e** | Pure paramful single-stmt global element-list fold (**#1405**) | `[add2(10,20),1,2]` → `30 1 2`; multi-stmt paramful residual fail-closed |
| **14** | #913 imported array by-value (#1398); #921 thinlink residual docs/gate (#1399); Root-2 method chain (#1401); bare float arith (#1404); bare cross-mod f64 Ident (#1400) | Wave14 residual closeouts locked by dedicated gates |
| **15** | **This PR — public showcase packaging** | Orchestrates Wave13 science + Wave14 locks; honest residual for denser lognormal science on tip prebuilt |

Supporting science leaves (not always wave-numbered, but tip depends on them): dual native run (#1257), product_nonassoc leaf (#1266), propagate native (#1287), oct_mul exclusive-ref split (#1274), unsplit oct re-entry (#1304), into-acc dep lower (#1402), specialized-list DCE (#1397).

## Claim boundary (read this before quoting)

### Claims (when the Wave15 showcase gate is green)

- Default Madaros dual-imports `gum` and `knowledge` and **runs** them
- CPC N=4 `order_spread4` exact spread is native under Madaros
- Finite-dof GUM coverage factor is **Student-t** (`k95i=2776`), not the historical D1 bitcast collapse (`1960`)
- **`cd_exact_generic_i64` ELF end-to-end** — **GREEN** under default Madaros after [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392): stdout includes `ZD PROVED`, `SQ PASS`, `NONZERO PASS`, and 16× `COMP i 0`
- Global array init path including **Wave13e** pure paramful **single-stmt** call args in element lists
- Imported f64 **minimal** leaf, multi-mod **BSS** distinct A/B constants, and **bare cross-mod Ident** (#1400)
- Bare float intrinsic results participate in subsequent f64 arithmetic (#1404)
- Root-2 associated + multi-module instance methods, including **inline method chains** on imported Epistemic
- Imported `[f64;N]` **by value** payload preserved (#913 / #1398)
- Default multi-module **#921 thin-link** fail class closed (#1399 residual gate)
- Public machine proof exists: `madaros_wave15_showcase_receipt.v1.json`

### Explicit non-claims / honest residuals

- **Denser imported f64 lognormal science** (`stats::densities::lognormal_pdf` / `DE_LN_SQRT_2PI`) — residual on tip stock prebuilt after #1405 (was green on Wave13-era prebuilt). Not claimed until `REQUIRE_IMPORTED_F64_SCIENCE=1` and green
- **Multi-stmt paramful** pure callees in global element-list init — residual fail-closed BSS zeros (KIND 3 intentionally no-op; see Wave13e audit)
- All stdlib dual pairs beyond the gated witnesses
- Language-level generic `Knowledge<T>` import (distinct from `epistemic::knowledge::Epistemic`)
- Multi-module IrModule memory wall fully closed for every corpus
- Full Root-2 method census closed
- Compact imported IR completeness (opt-in path may still emit_failed then fallback)
- f64-param bitcast free for *all* call shapes beyond gated witnesses
- Full linalg native parity
- “All Madaros residuals closed”

## Artefacts

| Path | Role |
|------|------|
| `scripts/dev/madaros_wave15_showcase_gate.sh` | Public orchestrator + receipt writer |
| `artifacts/compiler/madaros_wave15_showcase_receipt.v1.json` | Machine-readable showcase receipt |
| this audit | Human-readable claim boundary + waves 1–15 ledger |
| `scripts/dev/madaros_wave13_showcase_gate.sh` | Prior full-green surface (superseded for public quotes by Wave15 packaging) |

## Re-run matrix

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/madaros_dual_import_gate.sh
bash scripts/madaros_order_spread_native_gate.sh
bash scripts/epistemic_trust_gate.sh
bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh
bash scripts/madaros_cd_exact_e2e_gate.sh
bash scripts/dev/madaros_global_array_init_gate.sh
bash scripts/madaros_bare_float_arith_gate.sh
bash scripts/madaros_root2_method_gate.sh
bash scripts/madaros_root2_multimodule_method_gate.sh
bash scripts/ci/madaros_imported_array_byvalue_gate.sh
bash scripts/madaros_thinlink_921_residual_gate.sh
bash scripts/dev/madaros_wave15_showcase_gate.sh
```

## Why Wave15 is packaging, not a silent rewrite of science

Wave13 put `cd_exact` on the public pedestal. Wave14 closed Root-2 chain, #913, #921, bare float arith, and bare Ident. Wave15’s job is to **compose those locks under one orchestrator and refuse to invent green** for the denser lognormal residual that tip prebuilt shows after the Wave13e rebuild. Quote the receipt’s `showcase_verdict` and `claims` / `claims_not_made` — not a hand-waved “all residuals closed.”
