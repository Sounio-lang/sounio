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

## Mission

Ship a **public-facing** packaging of the Madaros residual campaign:

1. Orchestrator `scripts/dev/madaros_wave12_showcase_gate.sh`
   - Wave11 tip-green
   - dual + order_spread + k95 (spotlight science pillars)
   - honest `cd_exact` probe (new gate; **does not invent green**)
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
```

Optional strict mode (fails the showcase if `cd_exact` is still red):

```bash
REQUIRE_CD_EXACT=1 bash scripts/dev/madaros_wave12_showcase_gate.sh
```

## What the tip proves today (default Madaros)

| Pillar | Gate | Public claim when green |
|--------|------|-------------------------|
| Wave11 tip lock | `scripts/dev/madaros_wave11_tip_green_gate.sh` | Eight residual locks superseding Wave10 |
| Dual import | `scripts/madaros_dual_import_gate.sh` | `gum` + `knowledge` import + native run |
| Order spread | `scripts/madaros_order_spread_native_gate.sh` | CPC N=4 exact spread ≈ `2.044226` |
| GUM k95 | `scripts/epistemic_trust_gate.sh` | Finite-dof **k95i=2776** (= t95(4)), not D1 collapse 1960 |
| CD exact (residual) | `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | Generic sedenion ZD over i64 **only if green** |

Wave11 sub-locks (included by the tip-green orchestrator): dual, order_spread, knowledge method parity, global array init, named-path `print_f64`, unsplit `oct_mul`, epistemic trust (k95), global array ref mutation (Defect B).

## Measurement (this packaging run)

Recorded in `artifacts/compiler/madaros_wave12_showcase_receipt.v1.json`.

| Gate | Class | Result | Duration | Notes |
|------|-------|--------|---------:|-------|
| wave11_tip_green | required | **GREEN** | 110s | `MADAROS_WAVE11_TIP_GREEN_GATE_OK` |
| dual | required | **GREEN** | 14s | `MADAROS_DUAL_IMPORT_GATE_OK` |
| order_spread | required | **GREEN** | 12s | `MADAROS_ORDER_SPREAD_NATIVE_GATE_OK` |
| k95 | required | **GREEN** | 40s | `EPISTEMIC_TRUST_GATE_OK` + k95i=2776 |
| cd_exact | honest probe | **RED** | 3s | compile fail: 9×E011 + 1×E008 — **not invented green** |

- **Tip SHA measured:** `dbdf1029b` (`origin/main` at ship)
- **Engine:** Madaros v0.80.0 — `raw_elf_sha256=263a14a0e1fb566856ad3d20511bb905565da2363870c49e294719fe64d93ca6`
- **showcase_verdict:** `pass_with_cd_exact_residual`
- **overall:** `pass` → `MADAROS_WAVE12_SHOWCASE_GATE_OK`
- **required_red_count:** 0

Overall exit 0 prints `MADAROS_WAVE12_SHOWCASE_GATE_OK` so public demos can lock the green science surface without lying about the residual.

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

Supporting science leaves (not always wave-numbered, but tip-green depends on them): dual native run (#1257), product_nonassoc leaf (#1266), propagate native (#1287), oct_mul exclusive-ref split (#1274), unsplit oct re-entry (#1304).

## Claim boundary (read this before quoting)

### Claims (when the showcase gate is green)

- Default Madaros dual-imports `gum` and `knowledge` and **runs** them
- CPC N=4 `order_spread4` exact spread is native under Madaros
- Finite-dof GUM coverage factor is **Student-t** (`k95i=2776`), not the historical D1 bitcast collapse (`1960`)
- Wave11 tip-green locks remain green (knowledge method form, global array init + ref mutation, named-path print, unsplit oct_mul, …)
- Public machine proof exists: `madaros_wave12_showcase_receipt.v1.json`

### Explicit non-claims

- **`cd_exact_generic_i64` ELF end-to-end** — still residual while the honest probe is red (multi-module E011 method resolution under the current prebuilt; source fixes such as #1383 may not yet be in the shipped ELF)
- All stdlib dual pairs beyond the gated witnesses
- Language-level generic `Knowledge<T>` import (distinct from `epistemic::knowledge::Epistemic`)
- Multi-module IrModule memory wall fully closed for every corpus
- Full Root-2 method census closed
- f64-param bitcast free for *all* call shapes beyond gated witnesses
- Full linalg native parity
- “All Madaros residuals closed”

## Artefacts

| Path | Role |
|------|------|
| `scripts/dev/madaros_wave12_showcase_gate.sh` | Public orchestrator + receipt writer |
| `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | Honest cd_exact residual gate |
| `artifacts/compiler/madaros_wave12_showcase_receipt.v1.json` | Machine-readable showcase receipt |
| `scripts/dev/madaros_wave11_tip_green_gate.sh` | Required science tip lock (supersedes Wave10) |
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
bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh   # may be RED — honest
bash scripts/dev/madaros_wave12_showcase_gate.sh
```

## Why Wave12 is packaging, not a silent “all green”

The residual campaign earned a **trustworthy science tip** (dual, order spread, k95, knowledge method, global BSS). The remaining `cd_exact_generic_i64` wall is still real under the public prebuilt. Wave12’s job is to **put the green tip on a pedestal and keep the residual honest** — so external readers can re-run one script and quote only what the receipt claims.
