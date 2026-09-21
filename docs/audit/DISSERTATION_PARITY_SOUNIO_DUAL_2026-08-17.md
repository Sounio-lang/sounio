<!-- docs:meta
topic_id: repo.docs.audit.dissertation-parity-sounio-dual-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-parity-sounio-dual-2026-08-17
-->

# Route B — Sounio dual-path parity (no Node hard path)

**Date:** 2026-08-17  
**Decision:** **Route B** for the hard gates that remove UNKNOWN on Slurm.  
**Principle:** CLAUDE.md §4 — science in Sounio; ESM `.mjs` as sole hard reference is drift.

## Cost argument

| Item | Cost | Affordable? |
|---|---|---|
| Frontend dual (CN dt=0.01 vs 0.005, same model) | ~1 new 229-line `.sio` + gate restructure | **Yes** (done) |
| PBPK28 dual (CN dt=0.001 vs 0.0005) | ~1 alt of rapamycin ref + gate restructure | **Yes** (done) |
| Full independent second *algorithm* (Euler vs CN, separate TMDD/PD in two pure-Sounio stacks) | Multi-week reimplementation of Node TMDD/PD stack | **No** for this dispatch |
| Route A only (install Node on cpu-ops) | Ops image change | Still useful for *product* arm (website core) |

Half-step dual is **numerical self-consistency**, not a second theory of PK. It is enough to stop treating the gates as permanently UNKNOWN on Node-less compute: they now **PASS or FAIL** without Node.

Website/Node agreement remains an **optional product arm** when Node ≥ 18 is present — does not gate Slurm.

## Hard path (always)

1. **frontend_parity:** Sounio REF ↔ Sounio ALT, 1% RMSE / peak, 14 organs × 12 times.  
2. **pbpk28_parity:** Sounio REF ↔ Sounio ALT (rapamycin), 1% RMSE on cavg; then mass conservation (case 4). Cases 5–10 Node-only product/multi-drug arms omit cleanly without Node.

## Measured

| Gate | No Node | With Node |
|---|---|---|
| frontend_parity | **PASS** dual | PASS dual + PRODUCT_ARM_PASS vs website |
| pbpk28_parity | **PASS** dual + mass | PASS dual + PRODUCT_ARM_PASS; cases 7+ still FAIL on known E175/E008 when Node present |

## Files

- `tests/run-pass/dissertation_frontend_parity_alt.sio`
- `tests/run-pass/dissertation_pbpk28_parity_alt_rapamycin.sio`
- `scripts/ci/dissertation_frontend_parity_gate.sh`
- `scripts/ci/dissertation_pbpk28_parity_gate.sh`
