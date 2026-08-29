<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-a5
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-a5
-->

# WP-A5 — Convergence: cd_exact green on BOTH engines + phase-2 PR [Opus, Haiku for sweeps] (dep: A1+A2+A3+A4 all DONE)

## Goal
The master-prompt acceptance (docs/handoff/compiler_generic_F_engine_unblock_prompt.md §2) on the Madaros engine: `cd_exact_generic_i64` runs correctly, matching the lean_single gold standard that merged in PR #650.

## Steps
1. Fresh branch off main (which now has A1–A4 merged). Build: `bash scripts/ci/build_modular_madaros.sh /tmp/madaros-a5`.
2. THE TEST: `MADAROS_RAW_BIN=/tmp/madaros-a5 ./bin/madaros compile tests/run-pass/cd_exact_generic_i64.sio -o /tmp/cd.elf && chmod +x /tmp/cd.elf && /tmp/cd.elf` → stdout MUST be exactly: `ZD PROVED`, `SQ PASS`, `NONZERO PASS`, then 16 lines `COMP <i> 0` (i=0..15). Any deviation = a residual gap: classify verbatim, add to the new-gap ledger, status BLOCKED — do not improvise.
3. Cross-engine check: run the same test with a fresh lean_single stage2 — outputs must be identical line-for-line. Also run `tests/run-pass/cd_exact_generic_vs_concrete.sio` under Madaros (expect 3× MATCH + BYTECOMPARE PASS) — note it may hit the [F;2048]-scale SRET stretch goal from WP-A4; record honestly.
4. Full battery under the integrated Madaros: the 10 generic/trait run-pass tests (turbofish, generic_struct_return{,_structf}, impl_trait_for_type{,_multi}, trait_bounded_dispatch{,_struct,_multi_call}, trait_decl_bodyless_methods, cd_exact_generic_i64), compile-fail E010 still rejected, umbrella zero new reds.
5. [Haiku] Run-pass sweep vs the recorded baseline (see SCOREBOARD.md baseline row): zero regressions.
6. Phase-2 PR with the full witness table; squash-merge on green.
7. Wrap-up: RELEASE all M-track claims in `artifacts/omega/agent_handoff.log.md`; ping the exact-algebra consumer (engine parity achieved — `CDElementExact<F>` is now default-engine-safe; F=Rational next); leave a note in SCOREBOARD.md for fable5 to refresh its memory file (`project_fable5_generic_f.md`) on return.

## Done criteria
cd_exact output-verified identical on both engines; suite + umbrella + sweep green; phase-2 PR merged; claims released.
