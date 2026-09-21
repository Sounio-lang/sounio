<!-- docs:meta
topic_id: repo.docs.audit.imported-native-139-residual-census-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.imported-native-139-residual-census-2026-08-18
-->

# Residual imported/native 139 tags after #1853

**Date:** 2026-08-18  
**Lane:** `lane/grok-cli3/imported-139-untag-20260818` (measurement was #1878, squash-merged)  
**Base:** `origin/main` at `3afa7427c5` (#1878 measurement + #1881 E230-patch warning already on main)  
**Question:** of the ~248 imported 139 tags #1853 left untouched, how many are stale like the 168, and how many name a real crash?

## Instrument (validated before the census)

Source-built `artifacts/self-hosted/madaros` (99 964 767 B, 2026-08-17 17:01) via `MADAROS_RAW_BIN`, `ulimit -s 524288`, `SOUNIO_STDLIB_PATH` this worktree. `rc` from `souc run` itself, never a pipe. Per-test wall 90 s.

| Control | Expected | Observed |
|---|---|---|
| `bash -c 'kill -SEGV $$'` | 139 | **139** |
| `tests/run-pass/hello.sio` | 0 | **0** (0.6 s) |
| Untagged portfolio importer (`solver_portfolio_erdos_scope_v30_imported.sio`) | 0 | **0** (7 s) |
| Honest residual v168 | 1 | **1** (41 s) |

What would refute “these 139 tags are stale”: a file in the set exiting **139**. That did not happen.

The Madaros ELF is dated 2026-08-17, not rebuilt from `d3ea284caf`. It is the same binary that measured the #1853 set. Default `bin/souc` is prebuilt; this run forced the source-built ELF.

## How many

#1853’s “~248” was a leftover subtraction, not a recount. On `origin/main` after that merge:

| Query | n |
|---|--:|
| `//@ known-failure` anywhere under `tests/` | 307 |
| Tag text mentions `139` or `segfault` | **259** |
| of which v168 (tag *denies* 139) | 1 |
| Tags that *claim* a 139/segfault | **258** |

Machine table: `docs/audit/IMPORTED_NATIVE_139_RESIDUAL_CENSUS_2026-08-18.tsv`.

## Result — every claiming tag, sequential `souc run`

| class | n | meaning |
|---|--:|---|
| PASS `rc=0` | **246** | tag claims crash; process returned 0 |
| FAIL1 `rc=1` | 7 | compiles and runs; `main` returns 1 |
| OTHER `rc∈{3,4,9}` | 6 | runs or exits nonzero; **not** 139 |
| FAIL139 | **0** | — |
| TIMEOUT | **0** | — |

Elapsed ~33 minutes, one process.

### Split of the 246 zeros

| bucket | n | stale like the 168? |
|---|--:|---|
| `tests/run-pass` imported/native 139 | **239** | **Yes** — default Madaros `souc run` is the path the tag names |
| `zero_event_stdlib_native_v2_probe.sio` | 1 | **Yes** — tag says *default* Madaros segfaults; it returned 0 |
| E-KAN `tests/known_failures/*native_v2*` | **6** | **Not claimed.** Default `souc run` returned 0. The tag names a **native-v2** build. That backend was not re-invoked. |

### The 13 nonzero — real failure, wrong 139 story

v168 is already honest (`rc=1`, tag says fingerprint, not 139). The other **12** still wear a 139/segfault sentence:

| file | rc | tag claims |
|---|--:|---|
| `lorenz_i256_cover_child1_obligation_seed_imported.sio` | 1 | imported/native 139 |
| `lorenz_i256_step1_taylor2_center_artifact_imported.sio` | 1 | imported/native 139 |
| `lorenz_i256_step1_taylor2_local_flowpipe_seed_imported.sio` | 1 | imported/native 139 |
| `lorenz_i256_step1_taylor2_point_time_slab_containment_imported.sio` | 1 | imported/native 139 |
| `pb_kernel_trace_propagation_imported.sio` | 1 | imported/native 139 |
| `lorenz_i256_fixed_step.sio` | 4 | source-level segfault |
| `lorenz_i256_fixed_step_1e6.sio` | 1 | source-level segfault |
| `lorenz_i256_product_smoke.sio` | 3 | source-level segfault |
| `lorenz_i256_smallscale_step.sio` | 3 | source-level segfault |
| `lorenz_i256_xyz_step_bounded_bridge_imported.sio` | 3 | imported/native 139 |
| `lorenz_i256_y_step_bounded_bridge_imported.sio` | 4 | imported/native 139 |
| `solver_proof_profile_dispatch_envelope_imported.sio` | 9 | imported/native 139 |

These should keep a known-failure (they do fail) but the sentence must name the observed `rc`, not 139. Leaving “exits 139” on a process that returns 1 is the same class of lie as the 168.

## Two-engine sample (not a full un-tag)

Five of the stale PASSes, forced `SOUNIO_SOUC_ENGINE=lean_single`: all **rc=0**. `pb_kernel_trace_propagation_imported.sio` is **rc=1** on both engines. This is *not* enough to drop `requires: madaros` on the 110 PASSes that lack it. #1853 taught that the seed suite will execute unannotated files.

Of the 246 PASSes: 136 already have `//@ requires: madaros`, 110 do not.

## Answer

| Question | Answer |
|---|---|
| How many residual 139-claiming tags? | **258** (the “~248” was close) |
| Stale like the 168 (measured path, `rc=0`)? | **240** (239 run-pass + zero_event) |
| Real crash 139? | **0** |
| Real failure, 139 sentence is false? | **12** |
| Already honest residual? | **1** (v168) |
| Named a different backend, default path green? | **6** (E-KAN native-v2) |

A wrong known-failure tag is worse than a red test: 240 of these never run in the suite, and under Madaros they would have returned 0.

## lean_single control of the 110 (criterion written first)

The 139 tag is **not** decided here. lean_single answers only: after the 139 sentence is dropped, will CI Full Test Suite (`souc-stage2`) execute the file and stay green?

**Stopping rule (fixed before the first lean_single `souc run` on this set):**

1. The set is the population. **N = 110.** The only sample that supports “these 110 do not need `requires: madaros`” is 110/110 `rc=0`.
2. Stop when the list is exhausted.
3. Early stop **only** if the instrument is broken: lean_single `souc run tests/run-pass/hello.sio` is not 0, or lean_single `souc run` of a known #1853 seed-fail is not 1.
4. Do not stop at the first failure. Do not stop because time is short. Incomplete rows are `not_run`, not zeros.

110/110 lean_single zeros would **not** have licensed dropping the 139 tag. That tag is about Madaros.

**Instrument (this control only):** `SOUNIO_SOUC_ENGINE=lean_single`, `./bin/souc run <file>`, never `souc <file> -o <out>` (that is rc=0, writes a file named `-o`, and silently uses lean_single). `rc` from the process. `ulimit -s 524288`. `MADAROS_RAW_BIN` unset. No source-built E230 ELF.

Controls fired first: `hello.sio` rc=0; `solver_portfolio_v16_coverage_imported.sio` rc=1. Then the 110, sequential, one process, ~31 s. Zero 139. Zero timeout. Zero `not_run`.

| lean_single class | n | of which |
|---|--:|---|
| PASS `rc=0` | **54** | 47 run-pass + 6 E-KAN + zero_event |
| FAIL1 `rc=1` | **56** | all run-pass; all Madaros PASS |
| FAIL139 | **0** | — |

The earlier sample of 5 was the passing cluster. Machine table: `docs/audit/IMPORTED_NATIVE_139_LEAN_SINGLE_110_2026-08-18.tsv`.

## Un-tag applied (same day, after the 110)

Madaros decides the 139 sentence. lean_single decides `requires: madaros`.

| action | n | why |
|---|--:|---|
| Drop `//@ known-failure` 139 | **239** run-pass + **zero_event** | default Madaros `souc run` returned 0 |
| Add `//@ requires: madaros` | **56** | lean_single FAIL1; seed CI would go red without it |
| Omit `requires` | **47** | pass both engines |
| Keep existing `requires` | **136** | already annotated; not re-measured on lean_single |
| Rewrite 139 sentence to observed `rc` | **12** | real failure, wrong crash story |
| Leave tagged | **6** E-KAN | tag names native-v2; that backend was not re-invoked |
| Leave as-is | **1** v168 | already honest (`rc=1`, tag denies 139) |

Census elapsed on the 239 was ≤16 s (145 of them 10–16 s). No `timeout: 90`. Do not raise the global 30 s harness wall.

### Instrument warnings that can invert a 139 census

1. **Do not use an E230-patched source-built Madaros.** That ELF SIGSEGVs any aggregate of 3+ slots; `[f64;2]` already exits 139. A 139 from that compiler is the patch, not the test (`#1881`, `docs/audit/MADAROS_TRIVIAL_3I64_STRUCT_ALLOC_139_DISPATCH_2026-08-18.md`). This work used `artifacts/self-hosted/madaros` (99 964 767 B, 2026-08-17 17:01).
2. **Do not use `souc <file> -o <out>` without a subcommand.** It returns rc=0, writes a file named `-o`, and routes to lean_single. Use `souc run` / `souc compile`.

Those two lies point opposite ways: the patch invents 139s; the bare `-o` hides them by testing the wrong engine.

### Not done

- E-KAN native-v2 not re-measured on that backend.
- CAP / token table / handle table / global 30 s not raised.
- E175 / E137 / pbpk_suite not touched.
- This PR is not merged by the measuring lane.
