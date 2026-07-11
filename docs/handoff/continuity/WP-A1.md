<!-- docs:meta
topic_id: repo.docs.handoff.continuity.wp-a1
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.wp-a1
-->

# WP-A1 — Skeleton E035 effect annotations [Haiku] (dep: WP-A0 merged)

## Problem
Under Madaros, compiling `tests/run-pass/cd_exact_generic_i64.sio` (which imports `stdlib/algebra/cayley_dickson_exact.sio`) emits `error[E035] effect not declared in function signature (missing: Mut, Div, Panic)` ×3, at source spans ~4329/4387/4445 of the imported module. Three fns in the skeleton perform Mut/Div/Panic work without declaring those effects. lean_single does not enforce this; Madaros does.

## Steps
1. Fresh worktree/branch off post-A0 main: `fix/cd-exact-skeleton-effects`.
2. Build Madaros: `bash scripts/ci/build_modular_madaros.sh /tmp/madaros-a1`.
3. Reproduce: `MADAROS_RAW_BIN=/tmp/madaros-a1 ./bin/madaros compile tests/run-pass/cd_exact_generic_i64.sio -o /tmp/x.elf` → see the 3 E035 spans. Map spans to fns by opening `stdlib/algebra/cayley_dickson_exact.sio` (byte offsets ≈ spans; candidates are among `cd_basis_exact`/`cd_associator_exact`/`zd_exact`-family fns whose `with ...` clause is missing `Mut`/`Div`/`Panic`).
4. Add the missing effects to exactly those fn signatures (e.g. `with Mut, Panic` → `with Mut, Panic, Div`). Touch NOTHING else in the file.
5. Validate Madaros: recompile cd_exact — the 3 E035 must be gone (other error classes like E008/E011 may remain; they belong to WP-A2/A3, ignore).
6. Validate lean_single unaffected (the skeleton is shared): build stage2 (`./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/ls1; chmod +x /tmp/ls1; /tmp/ls1 self-hosted/compiler/lean_single.sio /tmp/ls2; chmod +x /tmp/ls2`), then compile+RUN with `/tmp/ls2` (interface `<src> <out>`): `tests/run-pass/cd_exact_generic_i64.sio` (expect ZD PROVED + SQ PASS + NONZERO PASS + 16× `COMP <i> 0`) and `tests/run-pass/cd_exact_generic_vs_concrete.sio` (expect 3× MATCH + BYTECOMPARE PASS). VERIFY ACTUAL STDOUT.
7. Commit, push, PR (small, docs-style body with the witness outputs), squash-merge on green. Update scoreboard + handoff log.

## Done criteria
E035×3 gone under Madaros; both lean_single cd_exact tests still output-verified green; PR merged.
