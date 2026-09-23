<!-- docs:meta
topic_id: repo.docs.internal.concepts.madaros-effects-inference-sandbox
authority: lane_draft_pending_founder_review
audience: agents
last_validated: 2026-09-23
validated_by: pending (acceptance-gate receipts; see Acceptance-Gate below)
source_of_truth: docs/internal/concepts/madaros-effects-inference-sandbox.contract.md
-->

# Semantic Lane Declaration — Madaros Effects Inference + Sandbox Codegen (Phase 1)

Status: draft (proposed; founder review pending)
Registry: proposed new Concept-ID — not yet a row in `docs/internal/concepts/registry.tsv`
Provenance: worktree `~/dev/sounio-worktrees/madaros-phase1`, branch
`feat/madaros-phase1-effects-inference-20260923` (base `045474a1c` = `origin/main`)

```text
Semantic-Lane-ID: lane-phase1-effects-inference-sandbox
Owner: agourakis82 (founder) via agent cherry-1 (session 9884caf9-d052-4b2b-946b-0f43a72bb8cd)
Concept-IDs: SOUNIO-MADAROS-EFFECTS-INFERENCE-SANDBOX (PROPOSED; draft contract = this file). No existing registry row covers the compiler-side effect system (verified 2026-09-23: registry.tsv effect-* rows are Pireus/policy-scoped).
Intent-Preserved: effects remain explicit, named and honestly diagnosed; the frozen effect-name closed list only shrinks (founder ruling 2026-08-19); unknown effect names are refused with a diagnostic, never silently aliased (holes #1953 and #1993); effect annotation != physical mechanism
Transformation: Madaros (Stage1) gains (1) consolidation of the three unmerged orphan effects-sandbox fixes (36ca3ba59 "use named sandbox execution result"; c32077140 and d1a5e83a1 tuple-borrow SIGSEGV under lean_single) — by merge, preserving authorship; (2) effect-set inference for closures/HOFs where sets are currently defaulted; (3) the effects-sandbox execution path lowered in the Madaros native x86_64 backend. No new effect names; no IR semantic-field redefinition; no founder-authorized concept redefined.
Types-Changed: none
Effects-Changed: coverage only — inference derives EXISTING effect sets; the frozen closed list untouched (none planned; it may only shrink)
IR-Changed: none
Claims-Introduced: IF the Acceptance-Gate passes THEN: closure/HOF effect sets are inferred rather than defaulted to IO; the effects-sandbox tests pass under the Madaros check+run path; #1953 (unknown name silently accepted) and #1993 (handle<Unknown> runs as handle<IO>) become diagnosed refusals with a file:line
Claims-Forbidden: that inference proves physical side-effect freedom (effect annotation != physical mechanism); that compile success implies runtime parity (compile success != runtime parity — measured 2026-09-23: souc-CLI artifacts are ELF-only on this host, and the committed Mach-Os speak a different CLI or die SIGSYS); that the orphan fixes are harmless-by-inspection (they must measure green first); that the closed list grew (it did not; total=2842 unchanged)
Assumptions: (a) the three orphan branches are unmerged (verified 2026-09-23: none is an ancestor of origin/main 045474a1c; their union delta = witness receipt + effects_sandbox_handlers.sio + stdlib/compiler/effects/sandbox.sio); (b) local execution on the Apple host requires a Darwin bootstrap — measured 2026-09-23: bin/souc wraps madaros-linux ELF, exits 126; artifacts/self-hosted/souc-self-hosted-arm64-macos is the "mini_native" compile-only CLI (unsigned -> SIGKILL 137; ad-hoc re-signed COPY runs; --version -> usage, rc=1); artifacts/self-hosted/souc-self-hosted-x86_64-macos re-signed dies "Bad system call" (rc=140, SIGSYS); the harness default engine resolves to the lean_single ELF (md5 0cb08380); the harness reports honest-empty when nothing matches ("WARNING: no test files matched the active filter -- this run measured nothing"); (c) no coordination lease conflicts (bin/sounio-coord status 2026-09-23T01:45Z: active_claims 0, conflicts 0)
Write-Set: stdlib/compiler/effects/sandbox.sio; tests/run-pass/{closure_effect_infer,closure_effect_infer_auto,effect_superset_ok,effect_chain_propagation,effects_sandbox_handlers,effects_sandbox_integration}.sio; artifacts/gates/witness_declares_its_sabotage.json (regenerated only, never hand-edited); self-hosted/check/effects.sio and effects_row.sio (inference, later); self-hosted/native/* (lowering, later); this contract file; artifacts/omega/agent_handoff.log.md
Read-Set: the three orphan branches (fix/effects-sandbox-baseline, fix/effects-sandbox-stage2-segfault-20260920, fix/effects-sandbox-handlers-segv-20260921); scripts/ci/{effect_name_closed_list_gate.sh,correlated_effect_gate.sh,effect_archaeology_gate.sh,effects_handler_interaction_lean_gate.sh}; scripts/dev/run_sio_test_suite_v2.sh; scripts/ci/build_modular_madaros.sh; docs/internal/concepts/*; tests/ and scripts/ resolvers
Positive-Witness: scripts/dev/run_sio_test_suite_v2.sh green for closure_effect_infer_auto.sio (expect-stdout: "inferred IO PASS"), effects_sandbox_integration.sio, effects_sandbox_handlers.sio, effect_superset_ok.sio, effect_chain_propagation.sio; the effects_sandbox_* tests are THEMSELVES confidence-gated (econf: total=1396, platinum=1396, gate=950 at 95.0%, knightian=0) — acceptance includes that verdict, not merely exit 0; scripts/ci/effect_name_closed_list_gate.sh, correlated_effect_gate.sh and effects_handler_interaction_lean_gate.sh rc=0; make check rc=0
Negative-Witness: tests/compile-fail covering unknown-effect refusal (file to be added by this lane; its name recorded here at implementation); effect_known_names_regression.sio must not regress; the accusation repro docs/audit/repro/effect_unknown_name.sio may flip effect_name_closed_list_gate to green ONLY via a diagnosed refusal, never by weakening the frozen list
Acceptance-Gate: bash scripts/dev/run_sio_test_suite_v2.sh --filter effect (rc=0) AND bash scripts/ci/effect_name_closed_list_gate.sh (rc=0) AND bash scripts/ci/correlated_effect_gate.sh (rc=0) AND bash scripts/ci/effects_handler_interaction_lean_gate.sh (rc=0) AND make check (rc=0) — each receipted with its exact command, exit code and econf verdict line
Integration-Target: origin/main via PR titled "[phase1] effects inference + sandbox codegen (Madaros)"; the orphan-branch fixes enter with their authors (merge; their histories not squashed away)
Authoritative-Only-If: every quantitative statement cites its exact command and exit (house rule: measure before claiming); Madaros-native EXECUTION receipts (not check-only) once the Darwin bootstrap exists; conflicts with the orphan branches are resolved by measurement, not seniority
```

## Stop-conditions honored (SEMANTIC_LANE_CONTRACT)

- (2) founder intent weakened: none; the 2026-08-19 "may only shrink" ruling untouched.
- (5) concurrent writers: none — `bin/sounio-coord status` 2026-09-23T01:45Z reports
  active_claims 0, conflicts 0; the orphan-branch commits are consolidation targets,
  not competing writers (their owners hold no leases).
- (6) falsifiability: stated in Claims-Forbidden and Acceptance-Gate.

## Integration Receipt

Pending — to be appended at review-ready state, per the contract. Until then this
lane's claims are exactly: assumption (a) ancestry check, assumption (c) coord
census, and the environment measurements cited under Assumptions (b) — nothing more.
