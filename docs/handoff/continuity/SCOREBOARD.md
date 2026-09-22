<!-- docs:meta
topic_id: repo.docs.handoff.continuity.scoreboard
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.continuity.scoreboard
-->

# Continuity Scoreboard — Madaros M-track phase 2 + EISA default-lane

**Single source of truth for the Opus/Haiku continuity campaign.** Every session MUST update its row when it finishes (status + evidence + commit). Protocol: read this file → claim the first TODO work packet compatible with your model → post a CLAIM entry in `artifacts/omega/agent_handoff.log.md` → execute the WP brief to the letter → update this table → RELEASE in the handoff log. See `BOOTSTRAP.md`.

Statuses: `TODO` | `CLAIMED(<session>)` | `IN-PROGRESS` | `DONE(<commit/PR>)` | `BLOCKED(<reason>)`

| WP | Title | Model | Deps | Status | Evidence (command → expected) |
|---|---|---|---|---|---|
| A0 | Phase-1 PR merge | Haiku | — | DONE(91ee4de77, PR #654) | merged 2026-07-07; A1/A2/A3 unblocked |
| A1 | Skeleton E035 effects | Haiku | A0 | DONE(PR #667 `a9f7165b85`, 2026-07-06) | Merged together with A2/A6 in the wave-2 A-track PR. Re-verified on clean `main` (`8fc294b7`, worktree `/tmp/repro-e259`): `cd_exact_generic_i64.sio` → `check: OK` verdict=0, zero E035; `cd_exact_generic_vs_concrete.sio` → `check: OK` + `BYTECOMPARE PASS`. Identical with `SOUNIO_EFFECT_INFER=0`, so the fix is in the checker, not the inference gate. |
| A2 | Primitive-receiver dispatch (E019/E011) | Opus | A0 | DONE(PR #667 `a9f7165b85`, 2026-07-06) | Merged in the same wave-2 A-track PR. Re-verified on clean `main`: brief W1 `rmix` primitive receiver → rc=23 (expected 23); W2 two-impl disambiguation `g::<Wrap>` → rc=60 (expected 60); `trait_bounded_dispatch.sio` → `5`/`spike PASS`, rc=0. **Residual (separate family, do not fold here):** `trait_bounded_dispatch_multi_call.sio` still refused — generic fn instantiated with >1 type-argument list in one compilation unit where a non-primitive arg is involved (monomorphization keeps one instantiation per template). Already baselined at `tests/madaros_corpus_baseline.txt:246` and `LEAN-ONLY` at `tests/engine_parity_baseline.txt:301`. |
| A3 | Specializer in multi-module lane | Opus | A0 | DONE(fix/madaros-specializer-multimodule, draft PR #658) | cd_exact compile E008 `CDElementExact__T`: baseline=1 → fixed=0 (Slurm differential). W2 wrap::<i64>→W<i64> rc=9 (collapse=1); W3 imported `struct F` rc=7 (collapse=0, no misfire); 8-test multi-module battery byte-identical baseline↔fixed (collapse=0); turbofish 3/3, E010 still rejected. Residual E035×3 (A1) + E019×8/E007 (A2) still gate cd_exact. |
| A4 | SRET struct-by-value return | Opus | — (own branch/PR) | DONE(PR #659 `a08a0a737` + #662 + #678 `f935dbd38` + #1392) | Closed by a chain of PRs, not one: #659 `println of i64 array element routes to print_int (unblocks generic_struct_return) [WP-A4]` (2026-07-06), #662 witness rc=0 convention (2026-07-06), #678 cross-module large-struct SRET forwarding (A8, 2026-07-07), #1392 cd_exact e2e (2026-07-19). **Re-verified on clean `main` (`8fc294b7`, worktree `/tmp/repro-e259`, binary built Sep 18):** ladder L0 `sret_8_field_return` rc=0 `OK`; L1/L2 `struct{c:[i64;4],bits:i64}` → `20`/`4`; L3 generic `<i64>` same shape; L4 `generic_struct_return` → `6`/`spike PASS` rc=0. Stretch beyond the brief: 3-arg SRET `combine(11,22,33)` → `11`/`22`/`33`/`66`/`3` (args shifted to rsi+ correctly); 2 KB `[i64;256]` and 16 KB `[i64;2048]` both exact (`100`/`355`/`256`, `7`/`2054`); `[F;2048]` generic at the cd_exact archetype scale → `9`/`9`/`104`/`MAX2048 DONE`. Consumer: `cd_exact_generic_i64` → `ZD PROVED`/`SQ PASS`/`NONZERO PASS`/16×`COMP i 0`; `cd_exact_generic_vs_concrete` → `BYTECOMPARE PASS`; `sret_forwarding_cross_module_min` → `CROSS_SRET_MIN_OK`. 17/17 regression battery rc=0. `docs/compiler/KNOWN_LIMITATIONS.md` no longer carries any SRET entry. |
| A5 | Convergence + phase-2 PR | Opus+Haiku | A1–A4 | DONE(fix/madaros-byval-param-nested-loop, `d87279926`) | **cd_exact_generic_i64 GREEN on Madaros** (Slurm, actual rc): `ZD PROVED`/`SQ PASS`/`NONZERO PASS`/16×`COMP i 0`, rc=0. Root cause of the final runtime SIGSEGV: the generic `cd_mul_exact` called `cd_sigma` via `use algebra::cayley_dickson::{cd_sigma}` — a call into a **transitively-imported** module is DROPPED by the Madaros imported-lane merge (IrCall elided; result slot writes vreg 0, aliasing param `a` → `a.c[i]` null-handle deref → SIGSEGV in the nested loop). Same-module calls (cd_zero_exact, er_*) resolve fine. Fix = same-module `cd_sigma_x` (mirrors the concrete sibling `cayley_dickson_exact_i64.sio`, which already inlines cd_sigma for this exact "HARD BLOCKER"). Stdlib-only; compiler untouched (8MB budget intact). W4: sret_8_field_return OK, generic_struct_return "6"/"spike PASS", sret_forwarding_cross_module_min OK, turbofish 3/3, for_in/for_range/... all rc=0. W3: lean_single can't build the 4-module generic closure (`unresolved cd_zero_exact`, anticipated); Madaros output = the mathematically-proven ZD result = gold standard. Residual COMPILER gap (transitive cross-module call drop) filed below — NOT fixed here. |
| A6 | Imported-lane effect-annotation carry (E035) | Opus | A1 | DONE(fix/madaros-imported-effect-carry, PR TBD) | Root cause: imported/builtin impl-target types lower to empty-name TypeEntry, so name-only `fn_sig_table_find_method` resolved `Rational::er_add`'s body against the effect-empty `i64::er_add` sig → spurious E035. Fix: `checker_check_impl_method_inplace` resolves the method sig by full TypeEntry (kind+name) via `fn_sig_table_find_method_semantic`. Slurm-verified: W1 cd_exact E035 3→0; W2 eff_inherent/eff_trait rc=6; W3 3-module imported witness E035 1→0 (typecheck; runtime blocked by separate cross-module native-lowering segfault); W4 10 impl/trait/multimodule tests EXACT rc+error parity base↔fixed (zero regression). |
| A7 | Integer-width coercion i32<->i64 (E007 branch-join) | Opus | A0 | DONE(fix/madaros-checker-intwidth, PR TBD) | `types_compatible` now widens narrow ints within a signedness family (i8/i32/i64, u8/u32/u64); wide i128/u128 keep exact bit-width. cd_exact `error[E007]` 1→0 → `imported_compile: typecheck ok` (E007=0, error_lines=0); ELF still NOT produced due to the PRE-EXISTING cross-module `lower_array: dep_begin 1` segfault (same gap A6 logged; rc=139, not this fix). W2 probe/intwidth.sio rc=0 OUT=3; minimal generic-struct-return + i32/i64 if-join compiles+runs OUT=3. W3 6 run-pass (turbofish/generic_struct_basic/sret_8_field_return/i32_implicit_return/generic_struct_return/generics_multi_param) all rc=0 unchanged. W4 turbofish_type_arg_arity still REJECTED (E010, no ELF). |
| B1 | EISA `str_from_bytes` dep-closure (ud2/SIGILL) | Opus | — | DONE(fix(native) PR #710 `f26bdd2981`, 2026-07-07) | **Premise falsified then fixed elsewhere.** PR #664 CLOSED proved the WP text wrong: the module dep-closure already works, `str::lib` IS loaded transitively (minimal repro reports `loaded 4 modules`), **no `ud2` is ever emitted and no ELF is produced** — so there was no runtime SIGILL to reach. The real defect was a missing *builtin* body, and it lives in codegen, not `module_loader.sio`/`module_frontend.sio`: PR #710 `fix(native): emit builtins referenced only transitively via imported functions` (`ir_module_ensure_builtin_call_targets` at the end of `ir_module_finalize_merged_calls`, rebinding `IrCall`/`IrCallSret` whose name is a builtin to an appended named stub) + added the missing `str_from_bytes` (id 22) case to `native_v2_builtin_id_for_name`. PR #714 ported the merged-checker fixes. **Re-verified on clean `main` (`8fc294b7`, worktree `/tmp/repro-e259`, binary 2026-09-18):** W1 author-written 3-module repro (main imports only mod_a → mod_a uses mod_b → mod_b uses `str::lib`) rc=0 printing `7`/`W1 REACHED`; W2 `test_eisa_isa` → `ALL PASS: eisa isa P1 P2 P3 P4 P5`, rc=0, no SIGILL, no ud2; W3 `test_eisa_evm` → `ALL PASS: eisa evm V1 V2 V3 V4 V5`, rc=0 under a default 16 GB vmem guard, no SIGILL. |
| B2 | EISA gate refresh + suite | Haiku | B1 | DONE(no compiler edit needed) | **Conformance gate: 25/25 PASS** on the canonical main binary — `eisa_bridge_conformance` with `tamper-sensitivity` and `anti-vacuity` both PASS (the brief's "21/21" is stale; the program list grew to 25, and 31 `artifacts/eisa/*.eisax.elf` now exist, so step 1's regeneration is unnecessary). **Default lane 13/14 PASS, lean lane 14/14** (the brief's "13 tests" is stale — the directory holds 14). Only `test_eisax_format` differs, and it is a **pre-existing, already-documented engine split**, not a B1/B2 red: header says `validated_lane: lean_single`; lean lane prints `ALL PASS: eisax F1..F7` while default lane compiles then exits rc=1. Diagnosed precisely rather than left vague: instrumented per-section counters put the 9 total failures at **F3 = 8 and F5 = 1** (F1/F2/F4 clean). Already recorded in `docs/audit/EISA_ORIGIN_GUM_2026-08-20.md` ("Pre-existing engine split; not this lane"). **Step 4 revised:** `/workspace/sounio-eisa` no longer exists; measuring `origin/gpu/epistemic-tensor-core-next` against `origin/main` shows **zero files present only on the branch** in `stdlib/eisa/`, `tools/eisa/` or `tests/stdlib/eisa/` — main is a strict superset, so there is no unmerged EISA content to inventory or integrate. |

## New-gap ledger (candidates for new WPs — do NOT chase inside an existing WP)

| Found by | Description | Evidence |
|---|---|---|
| fable5 | `println(<annotated computed local>)` segfaults on Madaros (e.g. `let y: i64 = x+11; println(y)`) — pre-existing, distinct from the fixed call/field cases | rc=139 on baseline madaros-m0 too |
| fable5 | `method_receiver_correct.sio` + `generic_struct_instantiate.sio` rc=139 on baseline — families per `docs/audit/MADAROS_METHOD_CALL_SIGSEGV_2026-06-20.md` / `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md` | pre-existing |
| A6 | Imported multi-module native lowering segfaults at `lower_array: dep_begin 1` when a program imports a module that itself `use`s a struct from a third module (chain import with a cross-module struct type). Distinct from A6's type-check fix (which passes clean, E035=0). Blocks W3 runtime of `tests/probe/a6_ring_main.sio`. | rc=139 during `imported_compile: lower_begin` on both baseline and A6-fixed madaros |
| A14 | **FIXED (2026-07-07, PR base main, branch `fix/madaros-transitive-call-drop`) — transitive cross-module call drop.** The A5 residual is root-caused + fixed in `module_frontend.sio`; the `cd_sigma_x` stdlib workaround is REVERTED (cd_mul_exact again calls transitive `cd_sigma`). Root cause: merged-IR call-target canonicalization (`ir_module_compact_duplicate_fn_refs` + finalize resolve loop) rebound `fn_id` via a whole-`IrInstr` writeback; `IrInstr` carries a `Box` → lean_single zeroes the slot on copy (call elided). Nested 3-level scalar stores are ALSO dropped by lean_single; fix rebinds per WHOLE FUNCTION (merge-append idiom). Net +231 bytes (8MB budget intact). | Slurm, actual rc: a13 ctrlA/ctrlC 139→0, ctrlE 0; a14_transitive_min 0→115; cd_exact_generic_i64 139→0 (transitive cd_sigma) ZD PROVED/SQ PASS/NONZERO PASS/16×COMP; cd_exact_generic_vs_concrete 139→0; sret_min/cdmul/sret8/turbofish/a8_diag_* green. See `docs/compiler/KNOWN_LIMITATIONS.md`. |
| A10 | **PARTIAL FIX of the `dep_begin 1` crash** (PR base main, branch `fix/madaros-generic-dep-summary-lowering`): the *summary*-lowering SIGSEGV for an imported module that declares an `impl` is FIXED — `lowerer_preseed_fn_signature_mut` used the nested-lvalue aggregate store `.param_regs=[..;64]` (lean_single two-level-nested-store miscompile); rewritten to the extract-to-local idiom. cd_exact now clears module-1 summary lowering. Minimal repro `gen_dep_summary_min2.sio` green. **Remaining (new gap):** cd_exact still no-ELF — `IrModule`~250 MB × many by-value copies → ~18 GB VM peak → alloc-fail SIGSEGV under `ulimit -v 16G` mid module-1 body lowering (`cd_sub_exact`); with no ulimit it advances to a genuine SIGSEGV at `lower_array: dep_begin 2` (module-2 summary). Same body-lowering wall hits `algebra_g2_invariants_import`/`associator_field_octonion` on base (no impls → independent of the impl fix). | W1 genmin/genmin2 rc0; W4 sret_min/sret_cdmul/sret8/genret/turbofish all green base↔fixed; g2imp/octo rc=139 identical base↔fixed (no regression); cd_exact BUILD rc=139 (past module-1 summary). See `docs/compiler/KNOWN_LIMITATIONS.md`. |

| B2 | `test_eisax_format.sio` v0 on the DEFAULT lane: rc=1, no stdout verdict; 9 asserts fail (F3 block = 8, F5 = 1; F1/F2/F4 clean). Compiles and lowers clean (`Merged IR: 116 functions`, ELF produced), so this is wrong-code, not a SIGILL and not a refusal. lean lane is `ALL PASS: eisax F1 F2 F3 F4 F5 F6 F7`. Header declares `validated_lane: lean_single`, and `docs/audit/EISA_ORIGIN_GUM_2026-08-20.md` already records it as a pre-existing engine split. Left for the engine-parity campaign, not chased here. | `./bin/madaros run tests/stdlib/eisa/test_eisax_format.sio` → ELF rc=1; per-section counter prints: boundary F2=0, F3=0, F4=8, F5=8, F6=9 (cumulative, total 9) |

## Stale-status audit (2026-09-18)

The A-track rows below were re-measured on clean `main` (`8fc294b7`) and three of
them were marked `TODO` while the work had already landed. Do not re-derive A1,
A2 or A4 — read the row evidence.

| WP | Was | Is | Closed by |
|---|---|---|---|
| A1 | TODO | DONE | PR #667 (`a9f7165b85`, 2026-07-06) |
| A2 | TODO | DONE | PR #667 (`a9f7165b85`, 2026-07-06) |
| A4 | TODO | DONE | PR #659 (`a08a0a737`) + #662 + #678 + #1392 |
| A6 | DONE | DONE (row said `PR TBD`) | PR #667 (`a9f7165b85`, 2026-07-06) |
| B1 | TODO | DONE | PR #710 (`f26bdd2981`, 2026-07-07) after #664 falsified the premise |
| B2 | TODO | DONE (no compiler edit needed) | PR #710 + #714; gate 25/25 |

Every row of this campaign is now closed. There is **no remaining work packet** —
before opening a new one, re-measure the tree: two of the six "open" rows above
described work that had already landed, and B1's diagnosis was wrong about both the
mechanism (dep-closure → missing builtin body) and the file (compiler/module_* →
native codegen).

**Measurement surface warning (cost two sessions).** `/workspace/sounio` is NOT on
`main`: it sits on `lane/cursor-1/20260826` (`3a50ccec92`) with a rewritten
`self-hosted/ir/lower.sio` (~9943 insertions / 6347 deletions uncommitted), and the
workspace `bin/madaros-linux-x86_64` was built from that dirty tree. Measuring the
ladder there answers a question about someone else's branch. Use a clean-main
worktree instead — `/tmp/repro-e259` is at `8fc294b7` with only `check.sio` dirty and
a binary dated 2026-09-18. Also: the committed `bin/madaros-linux-x86_64` is
**BEHIND** `self-hosted/` (3 of 5 drift probes: `keyword_on_is_identifier_capable`,
`drift_gate_second_order_mean`, `drift_gate_hessian_of` all rc=126) — that is a
pre-existing packaging lag, not a source defect.

## Fixed reference state (do not re-derive)

- lean_single generic `<F>` engine: MERGED main PR #650 (`2adb8f061`). Gold-standard outputs for cd_exact live in `tests/run-pass/cd_exact_generic_i64.sio` header.
- Madaros phase 1: branch `coord/fable5-madaros-generic-f` @ `d15915f58`, PR #654.
- EISA: lean lane **14/14 GREEN**, default lane **13/14** (re-measured 2026-09-18 on `8fc294b7`; the "12/12" and "13 tests" counts below were stale — the directory holds 14 tests). `test_eisax_format` v0 fixed (`639094a00`) on the lean lane; evm arena fixed (`a096d1c4b`, 25.8→6.5GB). The one default-lane difference is `test_eisax_format` (rc=1, 9 asserts: F3×8 + F5×1), a pre-existing engine split documented in `docs/audit/EISA_ORIGIN_GUM_2026-08-20.md` — the file declares `validated_lane: lean_single`.
- EISA lane worktree `/workspace/sounio-eisa` **no longer exists** (checked 2026-09-18). Branch `gpu/epistemic-tensor-core-next` still exists and is 247 commits ahead of main, but carries **no EISA file that main lacks** in `stdlib/eisa/`, `tools/eisa/` or `tests/stdlib/eisa/` — main is a strict superset, so there is nothing to integrate. Do not re-open that integration question without re-measuring with `git diff --name-status origin/main origin/gpu/epistemic-tensor-core-next`.
- Pre-existing umbrella-gate reds (row-identical across phase 1; not yours to fix): driver_self_compile, science_spine, f64_ladder, gum_primitives, semantic_hardening, lean_single_fixed_point, imported_closure×2 (rc=139), struct_orchestrator, phase_j_conf_gate, kretikos_kaxi_meta, dissertation_pbpk_suite.

---

## Session final (2026-09-18) — fugu-max

- PR #2542 (Madaros Phase 1 effects+sandbox): merged `8fc294b7a6`
- PR #2544 (A-track stale audit): merged `8641c3aa`
- PR #2545 (B-track audit): merged `a427bb1d`
- A1/A2/A4/B1/B2: DONE (work was already merged; measured rather than rebuilt)
- A2 residual: `trait_bounded_dispatch_multi_call.sio` stays baselined (monomorphizer, file `tests/madaros_corpus_baseline.txt:246`); subagent `aa3b94cf-483d...` analyzing `check.sio` `current_impl_type` path to confirm whether it is checker or monomorphizer.
- B2 residual: `test_eisax_format` (lean PASS / default rc=1, 9 asserts F3=8+F5=1) — pre-existing engine split, already in `EISA_ORIGIN_GUM_2026-08-20.md`; file header `validated_lane: lean_single`.
- Measurement trap documented: `/workspace/sounio` is NOT `main` (`lane/cursor-1/20260826`, dirty `lower.sio`); canonical clean surface is `main` `8fc294b7` (worktree `/tmp/repro-e259`).
- No `tests/run-pass/*.sio` source edited in either audit; docs registry gate passed both times; `artifacts/omega/agent_handoff.log.md` has CLAIM/RELEASE entries for A4, A2-analysis-delegation, B1/B2.
