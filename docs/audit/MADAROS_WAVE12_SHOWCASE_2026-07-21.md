<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave12-showcase-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave12-showcase-2026-07-21
-->

# Madaros Wave12 — public residual showcase

**Date:** 2026-07-21  
**Role:** Wave12 Agent D (implementer) — SHOWCASE packaging  
**Branch:** `test/madaros-wave12-showcase`  
**Engine:** default `bin/souc` → **Madaros v0.80.0** (no lean_single pin)  
**Audience:** public / external readers — “show the world what the tip can prove”

> **Wave13 update (2026-07-21):** [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392) closed
> `cd_exact_generic_i64` e2e (**ZD PROVED**). The Wave12 orchestrator now defaults
> `REQUIRE_CD_EXACT=1` (required green, not honest residual only). Prefer the
> full-green public surface: [`MADAROS_WAVE13_SHOWCASE_2026-07-21.md`](MADAROS_WAVE13_SHOWCASE_2026-07-21.md)
> and `bash scripts/dev/madaros_wave13_showcase_gate.sh`. Set `REQUIRE_CD_EXACT=0`
> only for legacy residual-only packaging. If stock prebuilt lags #1392, rebuild
> (`scripts/dev/souc-build-lock.sh make build-madaros`).

## Mission

Ship a **public-facing** packaging of the Madaros residual campaign:

1. Orchestrator `scripts/dev/madaros_wave12_showcase_gate.sh`
   - Wave11 tip-green
   - dual + order_spread + k95 (spotlight science pillars)
   - `cd_exact` probe — **required by default after #1392** (was honest residual at Wave12 ship)
2. This audit — waves 1–12 headline merges + claim boundaries
3. Machine receipt `artifacts/compiler/madaros_wave12_showcase_receipt.v1.json`
4. PR `test/madaros-wave12-showcase`

## One-command public proof

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave12_showcase_gate.sh
# MADAROS_WAVE12_SHOWCASE_GATE_OK
# receipt: artifacts/compiler/madaros_wave12_showcase_receipt.v1.json
# require_cd_exact=1 (default after Wave13 / #1392)
```

Legacy residual-only mode (fails open on red `cd_exact` — **not** the public claim):

```bash
REQUIRE_CD_EXACT=0 bash scripts/dev/madaros_wave12_showcase_gate.sh
```

## What the tip proves today (default Madaros)

| Pillar | Gate | Public claim when green |
|--------|------|-------------------------|
| Wave11 tip lock | `scripts/dev/madaros_wave11_tip_green_gate.sh` | Eight residual locks superseding Wave10 |
| Dual import | `scripts/madaros_dual_import_gate.sh` | `gum` + `knowledge` import + native run |
| Order spread | `scripts/madaros_order_spread_native_gate.sh` | CPC N=4 exact spread ≈ `2.044226` |
| GUM k95 | `scripts/epistemic_trust_gate.sh` | Finite-dof **k95i=2776** (= t95(4)), not D1 collapse 1960 |
| CD exact (required after #1392) | `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | Generic sedenion ZD over i64 — **ZD PROVED** when green |

Wave11 sub-locks (included by the tip-green orchestrator): dual, order_spread, knowledge method parity, global array init, named-path `print_f64`, unsplit `oct_mul`, epistemic trust (k95), global array ref mutation (Defect B).

## Measurement (this packaging run)

### Original Wave12 ship (pre-#1392 residual packaging)

Recorded historically in `artifacts/compiler/madaros_wave12_showcase_receipt.v1.json` at Wave12 merge.

| Gate | Class | Result | Duration | Notes |
|------|-------|--------|---------:|-------|
| wave11_tip_green | required | **GREEN** | 110s | `MADAROS_WAVE11_TIP_GREEN_GATE_OK` |
| dual | required | **GREEN** | 14s | `MADAROS_DUAL_IMPORT_GATE_OK` |
| order_spread | required | **GREEN** | 12s | `MADAROS_ORDER_SPREAD_NATIVE_GATE_OK` |
| k95 | required | **GREEN** | 40s | `EPISTEMIC_TRUST_GATE_OK` + k95i=2776 |
| cd_exact | honest probe (then) | **RED** | 3s | compile fail: 9×E011 + 1×E008 — residual preserved |

- **Tip SHA measured (Wave12 ship):** `dbdf1029b`
- **showcase_verdict (Wave12 ship):** `pass_with_cd_exact_residual`

### Post-#1392 / Wave13 promotion

After [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392), default packaging requires `cd_exact` green. Re-measure with:

```bash
bash scripts/dev/madaros_wave12_showcase_gate.sh
# or full-green public surface:
bash scripts/dev/madaros_wave13_showcase_gate.sh
```

Expected: `showcase_verdict=pass_full`, claim `cd_exact_zd_proved_pr1392`, sentinel `MADAROS_WAVE12_SHOWCASE_GATE_OK` / `MADAROS_WAVE13_SHOWCASE_GATE_OK`.

## Waves 1–12 — headline merges (residual campaign)

These are the **Madaros residual / science-path** wave headlines that led to this showcase. PR numbers are the public merge anchors on `main`.

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
| **11** | Tip-green promotion (#1379); imported f64 const Defect A (#1380); multi-mod BSS offset remap (#1382) | Eight-gate tip lock (adds epistemic_trust + global_array_ref); imported f64 module constants preserved |
| **12** | **This PR — public showcase packaging** | Wave11 + dual + order_spread + k95 + honest cd_exact probe + public receipt/audit |
| **12e** | cd_exact e2e (#1392) | specialized_collapse + mono markers → **ZD PROVED** under default Madaros |
| **13** | Full-green showcase promotion | `REQUIRE_CD_EXACT` default on; see Wave13 audit |

Supporting science leaves (not always wave-numbered, but tip-green depends on them): dual native run (#1257), product_nonassoc leaf (#1266), propagate native (#1287), oct_mul exclusive-ref split (#1274), unsplit oct re-entry (#1304).

## Claim boundary (read this before quoting)

### Claims (when the showcase gate is green *and* `cd_exact` is green)

- Default Madaros dual-imports `gum` and `knowledge` and **runs** them
- CPC N=4 `order_spread4` exact spread is native under Madaros
- Finite-dof GUM coverage factor is **Student-t** (`k95i=2776`), not the historical D1 bitcast collapse (`1960`)
- Wave11 tip-green locks remain green (knowledge method form, global array init + ref mutation, named-path print, unsplit oct_mul, …)
- **`cd_exact_generic_i64` ELF end-to-end** — **GREEN** after [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392) (`ZD PROVED` / `SQ PASS` / `NONZERO PASS` / 16× `COMP i 0`) when `require_cd_exact` is true (default)
- Public machine proof exists: `madaros_wave12_showcase_receipt.v1.json` (and Wave13 superseding receipt)

### Explicit non-claims

- All stdlib dual pairs beyond the gated witnesses
- Language-level generic `Knowledge<T>` import (distinct from `epistemic::knowledge::Epistemic`)
- Multi-module IrModule memory wall fully closed for every corpus
- Full Root-2 method census closed
- f64-param bitcast free for *all* call shapes beyond gated witnesses
- Full linalg native parity
- “All Madaros residuals closed”
- Historical Wave12-ship residual packaging (`pass_with_cd_exact_residual`) — superseded for public quotes by Wave13 / #1392

## Artefacts

| Path | Role |
|------|------|
| `scripts/dev/madaros_wave12_showcase_gate.sh` | Public orchestrator + receipt writer (`REQUIRE_CD_EXACT` default on) |
| `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | cd_exact gate (required after #1392) |
| `artifacts/compiler/madaros_wave12_showcase_receipt.v1.json` | Machine-readable showcase receipt |
| `scripts/dev/madaros_wave11_tip_green_gate.sh` | Required science tip lock (supersedes Wave10) |
| `scripts/dev/madaros_wave13_showcase_gate.sh` | Full-green superseding public surface |
| this audit | Human-readable claim boundary + wave ledger |

## Re-run matrix

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave11_tip_green_gate.sh
bash scripts/madaros_dual_import_gate.sh
bash scripts/madaros_order_spread_native_gate.sh
bash scripts/epistemic_trust_gate.sh
bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh   # must be GREEN after #1392
bash scripts/dev/madaros_wave12_showcase_gate.sh
bash scripts/dev/madaros_wave13_showcase_gate.sh        # preferred public surface
```

## Why Wave12 was packaging, not a silent “all green”

At Wave12 ship, the residual campaign had earned a **trustworthy science tip** (dual, order spread, k95, knowledge method, global BSS) while `cd_exact_generic_i64` was still honestly red under the public prebuilt. Wave12 put the green tip on a pedestal **without inventing green**. After #1392, Wave13 promotes `cd_exact` to required green — re-run the gate and quote the receipt, including `cd_exact_zd_proved_pr1392`.
