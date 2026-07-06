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
| A2 | Primitive-receiver dispatch (E019/E011) | Opus | A0 | DONE(fix/madaros-primitive-dispatch, PR pending) | W1 use2::<i64> rmix rc=23/74; W2 g::<i64> ea rc=23 + g::<Wrap>.v rc=60; W3 `trait_bounded_dispatch.sio` "5"/"spike PASS" rc=0. Fix: primitive-receiver branch in check.sio method-call gate (in-place 5757 + BOTH value fns) via `is_primitive_receiver_type` + fn_sig_table_find_method_semantic (kind-disambiguated); lower.sio `lower_method_recv_type` ExprIndex→i64/f64 fallback (imported `a.c[i].er_add`) + let-binding scalar_kind for i64-returning Call/MethodCall (W3 println segfault). Regressions green: struct dispatch 10/PASS, impl_trait_for_type{,_multi}, turbofish 3/3, generic_struct_basic; compile-fail turbofish_type_arg_arity still E010-REJECTED. BONUS: cd_exact E019×8→0. RESIDUAL: W4 `trait_bounded_dispatch_multi_call.sio` blocked on pre-existing specializer transitive-instantiation gap (E009/E008, see new-gap ledger), NOT primitive dispatch. |
| A3 | Specializer in multi-module lane | Opus | A0 | DONE(fix/madaros-specializer-multimodule, draft PR #658) | cd_exact compile E008 `CDElementExact__T`: baseline=1 → fixed=0 (Slurm differential). W2 wrap::<i64>→W<i64> rc=9 (collapse=1); W3 imported `struct F` rc=7 (collapse=0, no misfire); 8-test multi-module battery byte-identical baseline↔fixed (collapse=0); turbofish 3/3, E010 still rejected. Residual E035×3 (A1) + E019×8/E007 (A2) still gate cd_exact. |
| A4 | SRET struct-by-value return | Opus | — (own branch/PR) | TODO | bisect ladder L1–L4 green; `generic_struct_return.sio` runs rc=0 "6"/"spike PASS" |
| A5 | Convergence + phase-2 PR | Opus+Haiku | A1–A4 | TODO | cd_exact on Madaros: ZD PROVED + SQ PASS + NONZERO PASS + 16×COMP 0 |
| B1 | EISA `str_from_bytes` dep-closure (ud2/SIGILL) | Opus | — | TODO | `test_eisa_isa` + `test_eisa_evm` PASS on default lane, no SIGILL |
| B2 | EISA gate refresh + suite | Haiku | B1 | TODO | conformance gate 21/21; 13-test suite green on default lane |

## New-gap ledger (candidates for new WPs — do NOT chase inside an existing WP)

| Found by | Description | Evidence |
|---|---|---|
| fable5 | `println(<annotated computed local>)` segfaults on Madaros (e.g. `let y: i64 = x+11; println(y)`) — pre-existing, distinct from the fixed call/field cases | rc=139 on baseline madaros-m0 too |
| fable5 | `method_receiver_correct.sio` + `generic_struct_instantiate.sio` rc=139 on baseline — families per `docs/audit/MADAROS_METHOD_CALL_SIGSEGV_2026-06-20.md` / `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md` | pre-existing |
| A2 | Specializer transitive-instantiation: a generic fn passing its OWN type param as a nested turbofish arg (`fn combine<F>{ compute::<F>(a,b) }`) leaves `F` unsubstituted in the nested call → E009×3 "expected i64 found F" + E008 "expected F found i64". Blocks `trait_bounded_dispatch_multi_call.sio` (both i64 AND Pair paths); revealed only after A2 removed the E019 wall. Specializer domain (A3/A5), NOT method dispatch. | Madaros build of `trait_bounded_dispatch_multi_call.sio` rc=1 (E009/E008); Slurm job 5490 |

## Fixed reference state (do not re-derive)

- lean_single generic `<F>` engine: MERGED main PR #650 (`2adb8f061`). Gold-standard outputs for cd_exact live in `tests/run-pass/cd_exact_generic_i64.sio` header.
- Madaros phase 1: branch `coord/fable5-madaros-generic-f` @ `d15915f58`, PR #654.
- EISA: 12/12 lean-lane GREEN; `test_eisax_format` v0 fixed (`639094a00`); evm arena fixed (`a096d1c4b`, 25.8→6.5GB). Lane worktree `/workspace/sounio-eisa` (branch `gpu/epistemic-tensor-core-next`, owner cursor/grok, lock-open); its bundled `bin/madaros-linux-x86_64` is STALE — pair with `/workspace/sounio/bin/madaros-linux-x86_64`.
- Pre-existing umbrella-gate reds (row-identical across phase 1; not yours to fix): driver_self_compile, science_spine, f64_ladder, gum_primitives, semantic_hardening, lean_single_fixed_point, imported_closure×2 (rc=139), struct_orchestrator, phase_j_conf_gate, kretikos_kaxi_meta, dissertation_pbpk_suite.
