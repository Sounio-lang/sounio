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
| A1 | Skeleton E035 effects | Haiku | A0 | TODO | Madaros compile of cd_exact: E035×3 gone; lean_single BYTECOMPARE PASS intact |
| A2 | Primitive-receiver dispatch (E019/E011) | Opus | A0 | TODO | dispatch witnesses rc=23/74 and rc=23/60; `trait_bounded_dispatch{,_multi_call}.sio` green on Madaros |
| A3 | Specializer in multi-module lane | Opus | A0 | DONE(fix/madaros-specializer-multimodule, draft PR #658) | cd_exact compile E008 `CDElementExact__T`: baseline=1 → fixed=0 (Slurm differential). W2 wrap::<i64>→W<i64> rc=9 (collapse=1); W3 imported `struct F` rc=7 (collapse=0, no misfire); 8-test multi-module battery byte-identical baseline↔fixed (collapse=0); turbofish 3/3, E010 still rejected. Residual E035×3 (A1) + E019×8/E007 (A2) still gate cd_exact. |
| A4 | SRET struct-by-value return | Opus | — (own branch/PR) | TODO | bisect ladder L1–L4 green; `generic_struct_return.sio` runs rc=0 "6"/"spike PASS" |
| A5 | Convergence + phase-2 PR | Opus+Haiku | A1–A4 | TODO | cd_exact on Madaros: ZD PROVED + SQ PASS + NONZERO PASS + 16×COMP 0 |
| A6 | Imported-lane effect-annotation carry (E035) | Opus | A1 | DONE(fix/madaros-imported-effect-carry, PR TBD) | Root cause: imported/builtin impl-target types lower to empty-name TypeEntry, so name-only `fn_sig_table_find_method` resolved `Rational::er_add`'s body against the effect-empty `i64::er_add` sig → spurious E035. Fix: `checker_check_impl_method_inplace` resolves the method sig by full TypeEntry (kind+name) via `fn_sig_table_find_method_semantic`. Slurm-verified: W1 cd_exact E035 3→0; W2 eff_inherent/eff_trait rc=6; W3 3-module imported witness E035 1→0 (typecheck; runtime blocked by separate cross-module native-lowering segfault); W4 10 impl/trait/multimodule tests EXACT rc+error parity base↔fixed (zero regression). |
| A7 | Integer-width coercion i32<->i64 (E007 branch-join) | Opus | A0 | DONE(fix/madaros-checker-intwidth, PR TBD) | `types_compatible` now widens narrow ints within a signedness family (i8/i32/i64, u8/u32/u64); wide i128/u128 keep exact bit-width. cd_exact `error[E007]` 1→0 → `imported_compile: typecheck ok` (E007=0, error_lines=0); ELF still NOT produced due to the PRE-EXISTING cross-module `lower_array: dep_begin 1` segfault (same gap A6 logged; rc=139, not this fix). W2 probe/intwidth.sio rc=0 OUT=3; minimal generic-struct-return + i32/i64 if-join compiles+runs OUT=3. W3 6 run-pass (turbofish/generic_struct_basic/sret_8_field_return/i32_implicit_return/generic_struct_return/generics_multi_param) all rc=0 unchanged. W4 turbofish_type_arg_arity still REJECTED (E010, no ELF). |
| B1 | EISA `str_from_bytes` dep-closure (ud2/SIGILL) | Opus | — | TODO | `test_eisa_isa` + `test_eisa_evm` PASS on default lane, no SIGILL |
| B2 | EISA gate refresh + suite | Haiku | B1 | TODO | conformance gate 21/21; 13-test suite green on default lane |

## New-gap ledger (candidates for new WPs — do NOT chase inside an existing WP)

| Found by | Description | Evidence |
|---|---|---|
| fable5 | `println(<annotated computed local>)` segfaults on Madaros (e.g. `let y: i64 = x+11; println(y)`) — pre-existing, distinct from the fixed call/field cases | rc=139 on baseline madaros-m0 too |
| fable5 | `method_receiver_correct.sio` + `generic_struct_instantiate.sio` rc=139 on baseline — families per `docs/audit/MADAROS_METHOD_CALL_SIGSEGV_2026-06-20.md` / `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md` | pre-existing |
| A6 | Imported multi-module native lowering segfaults at `lower_array: dep_begin 1` when a program imports a module that itself `use`s a struct from a third module (chain import with a cross-module struct type). Distinct from A6's type-check fix (which passes clean, E035=0). Blocks W3 runtime of `tests/probe/a6_ring_main.sio`. | rc=139 during `imported_compile: lower_begin` on both baseline and A6-fixed madaros |

## Fixed reference state (do not re-derive)

- lean_single generic `<F>` engine: MERGED main PR #650 (`2adb8f061`). Gold-standard outputs for cd_exact live in `tests/run-pass/cd_exact_generic_i64.sio` header.
- Madaros phase 1: branch `coord/fable5-madaros-generic-f` @ `d15915f58`, PR #654.
- EISA: 12/12 lean-lane GREEN; `test_eisax_format` v0 fixed (`639094a00`); evm arena fixed (`a096d1c4b`, 25.8→6.5GB). Lane worktree `/workspace/sounio-eisa` (branch `gpu/epistemic-tensor-core-next`, owner cursor/grok, lock-open); its bundled `bin/madaros-linux-x86_64` is STALE — pair with `/workspace/sounio/bin/madaros-linux-x86_64`.
- Pre-existing umbrella-gate reds (row-identical across phase 1; not yours to fix): driver_self_compile, science_spine, f64_ladder, gum_primitives, semantic_hardening, lean_single_fixed_point, imported_closure×2 (rc=139), struct_orchestrator, phase_j_conf_gate, kretikos_kaxi_meta, dissertation_pbpk_suite.
