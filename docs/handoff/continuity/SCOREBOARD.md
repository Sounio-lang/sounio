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
| A3 | Specializer in multi-module lane | Opus | A0 | TODO | cd_exact compile: no `E008 ... CDElementExact__T` |
| A4 | SRET struct-by-value return | Opus | — (own branch/PR) | DONE(fix/madaros-struct-return-runtime, PR pending) | ladder L0–L4 green on Slurm-built madaros: L0 sret_8 rc=0"OK"; L2 [i64;4] rc=13; L3 generic<F>@i64 rc=13; ARGS 3-arg wide rc=20; `generic_struct_return.sio` rc=0 "6"/"spike PASS". Root cause was NOT SRET (struct-by-value array-field return already works — handle-based) but `println` dispatch missing `ExprIndex` in `expr_result_scalar_kind_ref`; i64 array element `r.c[0]` routed to char* printer → SIGSEGV. |
| A5 | Convergence + phase-2 PR | Opus+Haiku | A1–A4 | TODO | cd_exact on Madaros: ZD PROVED + SQ PASS + NONZERO PASS + 16×COMP 0 |
| B1 | EISA `str_from_bytes` dep-closure (ud2/SIGILL) | Opus | — | TODO | `test_eisa_isa` + `test_eisa_evm` PASS on default lane, no SIGILL |
| B2 | EISA gate refresh + suite | Haiku | B1 | TODO | conformance gate 21/21; 13-test suite green on default lane |

## New-gap ledger (candidates for new WPs — do NOT chase inside an existing WP)

| Found by | Description | Evidence |
|---|---|---|
| fable5 | `println(<annotated computed local>)` segfaults on Madaros (e.g. `let y: i64 = x+11; println(y)`) — pre-existing, distinct from the fixed call/field cases | rc=139 on baseline madaros-m0 too |
| fable5 | `method_receiver_correct.sio` + `generic_struct_instantiate.sio` rc=139 on baseline — families per `docs/audit/MADAROS_METHOD_CALL_SIGSEGV_2026-06-20.md` / `MADAROS_BOXNEW_SIGSEGV_2026-06-19.md` | pre-existing |
| A4 | Madaros default lane REJECTS several run-pass tests at type-checking preflight (build rc=1 "type checking preflight failed"): `hof_mut_struct_min.sio` (fn-pointer HOF param), `bdf_stiff.sio`, `ode_generic_solver.sio`, `linear_return_value.sio` — pre-existing checker gaps, not a lowering/codegen bug | Observed on WP-A4 Slurm build; the failure is at the checker phase, which runs BEFORE the WP-A4 lowering-pass edit (`expr_result_scalar_kind_ref`), so the edit cannot be the cause |

## Fixed reference state (do not re-derive)

- lean_single generic `<F>` engine: MERGED main PR #650 (`2adb8f061`). Gold-standard outputs for cd_exact live in `tests/run-pass/cd_exact_generic_i64.sio` header.
- Madaros phase 1: branch `coord/fable5-madaros-generic-f` @ `d15915f58`, PR #654.
- EISA: 12/12 lean-lane GREEN; `test_eisax_format` v0 fixed (`639094a00`); evm arena fixed (`a096d1c4b`, 25.8→6.5GB). Lane worktree `/workspace/sounio-eisa` (branch `gpu/epistemic-tensor-core-next`, owner cursor/grok, lock-open); its bundled `bin/madaros-linux-x86_64` is STALE — pair with `/workspace/sounio/bin/madaros-linux-x86_64`.
- Pre-existing umbrella-gate reds (row-identical across phase 1; not yours to fix): driver_self_compile, science_spine, f64_ladder, gum_primitives, semantic_hardening, lean_single_fixed_point, imported_closure×2 (rc=139), struct_orchestrator, phase_j_conf_gate, kretikos_kaxi_meta, dissertation_pbpk_suite.
