<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave13-showcase-2026-07-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave13-showcase-2026-07-21
-->

# Madaros Wave13 — public full-green showcase (cd_exact required)

**Date:** 2026-07-21  
**Role:** Wave13 Agent C (implementer) — SHOWCASE promotion after #1392  
**Branch:** `test/madaros-wave13-showcase`  
**Engine:** default `bin/souc` → **Madaros v0.80.0** (no lean_single pin)  
**Audience:** public / external readers — “show the world what the tip can prove”

## Mission

#1392 **MERGED**: `cd_exact_generic_i64` e2e **ZD PROVED** under default Madaros.  
Wave12 packaged a **trustworthy science tip** with an honest residual probe for `cd_exact`.  
Wave13 **promotes that residual to required green** and ships a public full-green receipt.

1. Orchestrator `scripts/dev/madaros_wave13_showcase_gate.sh`
   - Wave12 tip-green
   - dual + order_spread + k95 (spotlight science pillars)
   - **REQUIRED** `cd_exact` (`madaros_cd_exact_generic_i64_gate.sh`)
   - **REQUIRED** `cd_exact_e2e` (`madaros_cd_exact_e2e_gate.sh`, specialized_collapse path)
2. Tip-green lock `scripts/dev/madaros_wave13_tip_green_gate.sh` (Wave12 nine locks + required `cd_exact`)
3. Wave12 showcase updated: `REQUIRE_CD_EXACT` **defaults ON**
4. Machine receipt `artifacts/compiler/madaros_wave13_showcase_receipt.v1.json`
5. PR `test/madaros-wave13-showcase`

## One-command public proof

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave13_showcase_gate.sh
# MADAROS_WAVE13_SHOWCASE_GATE_OK
# receipt: artifacts/compiler/madaros_wave13_showcase_receipt.v1.json
```

`REQUIRE_CD_EXACT` defaults to **1**. Legacy residual-only packaging (not public claim):

```bash
REQUIRE_CD_EXACT=0 bash scripts/dev/madaros_wave13_showcase_gate.sh
```

Wave12 orchestrator now matches the same default:

```bash
bash scripts/dev/madaros_wave12_showcase_gate.sh   # REQUIRE_CD_EXACT=1 by default
```

## Prebuilt lag — rebuild if stock fails check

PR #1392 ships `bin/madaros-linux-x86_64` with the e2e closeout.  
If a checkout still has an **older** stock ELF, `cd_exact` may be RED even though source on `main` is fixed. **Do not invent green** — rebuild:

```bash
scripts/dev/souc-build-lock.sh make build-madaros
# then either install the rebuilt ELF or:
MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash scripts/dev/madaros_wave13_showcase_gate.sh
```

Also useful: `bash scripts/madaros_cd_exact_e2e_gate.sh` (prefers rebuilt RAW over lagging prebuilt candidates when present).

## What the tip proves today (default Madaros)

| Pillar | Gate | Public claim when green |
|--------|------|-------------------------|
| Wave12 tip lock | `scripts/dev/madaros_wave12_tip_green_gate.sh` | Nine residual locks (Wave11 + imported f64 BSS) |
| Dual import | `scripts/madaros_dual_import_gate.sh` | `gum` + `knowledge` import + native run |
| Order spread | `scripts/madaros_order_spread_native_gate.sh` | CPC N=4 exact spread ≈ `2.044226` |
| GUM k95 | `scripts/epistemic_trust_gate.sh` | Finite-dof **k95i=2776** (= t95(4)), not D1 collapse 1960 |
| CD exact (required) | `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | Generic sedenion ZD over i64 — **ZD PROVED** |
| CD exact e2e (required) | `scripts/madaros_cd_exact_e2e_gate.sh` | Same science tokens via specialized_collapse RAW path |

## Measurement (this packaging run)

Recorded in `artifacts/compiler/madaros_wave13_showcase_receipt.v1.json` (regenerate with the gate).

| Gate | Class | Result | Notes |
|------|-------|--------|-------|
| wave12_tip_green | required | **GREEN** | `MADAROS_WAVE12_TIP_GREEN_GATE_OK` |
| dual | required | **GREEN** | `MADAROS_DUAL_IMPORT_GATE_OK` |
| order_spread | required | **GREEN** | `MADAROS_ORDER_SPREAD_NATIVE_GATE_OK` |
| k95 | required | **GREEN** | `EPISTEMIC_TRUST_GATE_OK` + k95i=2776 |
| cd_exact | required | **GREEN** | `MADAROS_CD_EXACT_GENERIC_I64_GATE_OK` — ZD PROVED |
| cd_exact_e2e | required | **GREEN** | `MADAROS_CD_EXACT_E2E_GATE_OK` — specialized_collapse |

- **Anchor merge:** [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392) — `fix/madaros-cd-exact-e2e` → `main`
- **Engine:** Madaros v0.80.0
- **showcase_verdict:** `pass_full` (no residual packaging)
- **overall:** `pass` → `MADAROS_WAVE13_SHOWCASE_GATE_OK`
- **require_cd_exact:** `true`

## Waves 1–13 — headline merges (residual campaign)

| Wave | Headline merge(s) | What landed |
|------|-------------------|-------------|
| **1–11** | See Wave12 showcase ledger | Dual, k95, BSS, tip locks, … |
| **12** | Public residual showcase (#1384); tip-green imported f64 BSS (#1385) | Honest `cd_exact` probe + science tip pedestal |
| **12e** | `cd_exact` e2e under default Madaros (**#1392**) | specialized_collapse + i64 mono markers → **ZD PROVED** |
| **13** | **This PR — full-green showcase packaging** | `REQUIRE_CD_EXACT` default on; public claim of `cd_exact` green with receipt |

## Claim boundary (read this before quoting)

### Claims (when the Wave13 showcase gate is green)

- Default Madaros dual-imports `gum` and `knowledge` and **runs** them
- CPC N=4 `order_spread4` exact spread is native under Madaros
- Finite-dof GUM coverage factor is **Student-t** (`k95i=2776`), not the historical D1 bitcast collapse (`1960`)
- Wave12 tip-green locks remain green
- **`cd_exact_generic_i64` ELF end-to-end** — **GREEN** under default Madaros after [PR #1392](https://github.com/Sounio-lang/sounio/pull/1392): stdout includes `ZD PROVED`, `SQ PASS`, `NONZERO PASS`, and 16× `COMP i 0`
- Public machine proof exists: `madaros_wave13_showcase_receipt.v1.json`

### Explicit non-claims

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
| `scripts/dev/madaros_wave13_showcase_gate.sh` | Public full-green orchestrator + receipt writer |
| `scripts/dev/madaros_wave13_tip_green_gate.sh` | Tip-green lock with required `cd_exact` |
| `scripts/dev/madaros_wave12_showcase_gate.sh` | Updated: `REQUIRE_CD_EXACT` default **on** |
| `scripts/dev/madaros_cd_exact_generic_i64_gate.sh` | Default-souc `cd_exact` gate |
| `scripts/madaros_cd_exact_e2e_gate.sh` | RAW-ELF e2e gate from #1392 |
| `artifacts/compiler/madaros_wave13_showcase_receipt.v1.json` | Machine-readable showcase receipt |
| this audit | Human-readable claim boundary + wave ledger |

## Re-run matrix

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
unset SOUNIO_SOUC_ENGINE
ulimit -s unlimited 2>/dev/null || true

bash scripts/dev/madaros_wave12_tip_green_gate.sh
bash scripts/dev/madaros_wave13_tip_green_gate.sh
bash scripts/madaros_dual_import_gate.sh
bash scripts/madaros_order_spread_native_gate.sh
bash scripts/epistemic_trust_gate.sh
bash scripts/dev/madaros_cd_exact_generic_i64_gate.sh   # must be GREEN
bash scripts/madaros_cd_exact_e2e_gate.sh                # must be GREEN
bash scripts/dev/madaros_wave12_showcase_gate.sh         # REQUIRE_CD_EXACT=1 default
bash scripts/dev/madaros_wave13_showcase_gate.sh
```

## Why Wave13 is packaging, not a silent rewrite of science

The residual campaign closed the science tip first (Wave12). PR #1392 then closed the generic exact Cayley–Dickson e2e that Wave12 honestly kept red. Wave13’s job is to **put the full green tip on a pedestal without inventing green** — re-run one script; quote only what the receipt claims, including `cd_exact_zd_proved_pr1392`.
