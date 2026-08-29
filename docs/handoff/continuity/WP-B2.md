<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-b2
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-b2
-->

# WP-B2 — EISA: gate refresh + full-suite verification [Haiku] (dep: WP-B1 merged)

## Goal
Prove EISA default-lane parity end-to-end and refresh the conformance artifacts. NO compiler edits in this WP.

## Environment
EISA lane worktree: `/workspace/sounio-eisa` (branch `gpu/epistemic-tensor-core-next`, owner cursor/grok — read its CLAIM state in `artifacts/omega/agent_handoff.log.md` before writing anything there). Its bundled `bin/madaros-linux-x86_64` is STALE — always pair the default lane with the canonical main binary: `SOUNIO_MADAROS_BIN=/workspace/sounio/bin/madaros-linux-x86_64` (or a fresh build from post-B1 main).

## Steps
1. Regenerate the conformance artifacts (only 16 of the gate's 21 `.eisax.elf` programs exist in `artifacts/eisa/`): `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tools/eisa/eisa_bridge_emit.sio`, then `bash scripts/ci/eisa_bridge_conformance_gate.sh` → expect 21/21 (differential EVM-vs-native + tamper + anti-vacuity lanes).
2. Full EISA suite on the LEAN lane (control): all 13 tests in `tests/stdlib/eisa/` via `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run <t>` → record per-test PASS/FAIL + key output lines.
3. Full EISA suite on the DEFAULT lane (the parity target): same 13 tests via `SOUNIO_MADAROS_BIN=<canonical> ./bin/souc run <t>` → expect 13/13 PASS after WP-B1. Any failure: verbatim error → new-gap ledger, status per-test in your report.
4. Do NOT attempt to merge `gpu/epistemic-tensor-core-next` into main (~90 conflicts; that integration belongs to Lane 6 / cursor-grok). Instead: document in SCOREBOARD.md exactly which EISA-worktree commits are not yet on main (`git cherry -v origin/main` in the worktree, annotated).
5. Update the EISA scoreboard entry in `artifacts/omega/agent_handoff.log.md` (append a fresh entry with the 13-test table + gate result + lane state) and SCOREBOARD.md.

## Done criteria
Conformance 21/21; 13/13 default-lane table published in the handoff log; unmerged-commit inventory documented; scoreboard updated.
