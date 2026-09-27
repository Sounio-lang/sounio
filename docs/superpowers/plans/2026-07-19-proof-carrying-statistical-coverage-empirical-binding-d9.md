<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-19-proof-carrying-statistical-coverage-empirical-binding-d9
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-19-proof-carrying-statistical-coverage-empirical-binding-d9
-->

# D9 Proof-Carrying Statistical Coverage and Empirical Binding Plan

Spec:
`docs/superpowers/specs/2026-07-19-proof-carrying-statistical-coverage-empirical-binding-d9-design.md`.

Execution surface: isolated worktree
`/tmp/sounio-psychiatric-d9-20260719`, branch
`codex/psychiatric-d9-statistical-binding-20260719`.

Compiler owner remains codex-2. Imported multimodule execution remains
check-only under `BLK-20260718-D6-MULTIMODULE-RUNTIME`.

## Task 1: Freeze The Semantic Contract

- [x] Verify PR #1155, D8 head, canonical Madaros, dirty primary checkout, and
  semantic-lane status.
- [x] Create an isolated stacked D9 worktree without changing PR #1155.
- [x] Complete primary-literature research across partial identification,
  conditional and selective coverage, transport, positivity, calibration,
  provenance, abstention, and clinical utility.
- [x] Write this design and implementation plan.
- [x] Add the D9 concept contract, registry row, and binding manifest.

## Task 2: Build The Exact Finite Kernel

- [x] Add
  `stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio`.
- [x] Import and revalidate the frozen D8 exact `AB` set as the coverage target.
- [x] Implement design A and B, whole-set and memberwise coverage, exact
  fraction receipts, and `3/4` adequacy/refusal.
- [x] Implement distinct marginal, subgroup, selection-conditioned, and
  predictive coverage types.
- [x] Implement provenance, population, window, calibration, positivity,
  instrument-population compatibility, declared context binding, and
  mandatory abstention receipts.
- [x] Reserve external custody, sealed validation, empirical binding, patient
  state, and clinical action authority without positive constructors.
- [x] Add observational-equivalence action ambiguity without causal or clinical
  authority.

## Task 3: Add Independent Runtime Evidence

- [x] Add the standalone native scalar witness and exact W0-Wn receipts.
- [x] Add the exhaustive independent oracle over all 455 designs.
- [x] Freeze the UCI candidate fixture and predeclared protocol with SHA-256
  manifests and explicit non-clinical claim boundaries.
- [x] Verify the positive-design histogram, threshold count, support histogram,
  validation truth table, selection control, provenance collision, and action
  ambiguity.
- [x] Add the imported reusable API witness as `//@ check-only`.

## Task 4: Add Ontology And Negative Evidence

- [x] Add the parallel nominal ontology and its run-pass witness.
- [x] Add compile-fail tests for identified/confidence/predictive separation.
- [x] Add compile-fail tests for whole-set/memberwise and conditioning-scope
  separation.
- [x] Add compile-fail tests for positivity, calibration, instrument,
  provenance, target, and transport failures.
- [x] Add compile-fail tests preventing declared binding from becoming external
  binding, patient state, or clinical authority.
- [x] Add compile-fail tests preventing abstention from becoming a negative
  prediction, escalation, or binding.

## Task 5: Wire The Acceptance Gate

- [x] Add
  `scripts/ci/proof_carrying_statistical_coverage_empirical_binding_gate.sh`.
- [x] Require canonical Madaros and reject a requested legacy engine.
- [x] Check the kernel, ontology, and imported witness.
- [x] Execute the native witness, ontology witness, and independent oracle.
- [x] Verify every compile-fail expected/found nominal pair plus three private
  authority-constructor refusals.
- [x] Verify registry, bindings, docs, and semantic coordination.
- [x] Recursively keep D8-D0 green after the final stacked-base rewrite.
- [x] Run focused ontology validation through both default and rebuilt
  current-source paths and compare their receipts.

## Task 6: Review And Publish

- [x] Run mandatory xAI and Z.AI math review with
  `bin/llm-offload -t math-review` and log the result.
- [x] Run a hostile clinical-authority review without PHI.
- [x] Resolve or document every BLOCKER/MAJOR finding.
- [x] Run `node scripts/docs/sync_governance_metadata.mjs` and re-run docs gates.
- [x] Commit in narrow phases, push the stacked branch, and open a D9 PR with
  base `codex/psychiatric-mainline-d0-d2-20260717` while PR #1155 is open.
- [ ] Post-D9 integration follow-up: after PR #1155 merges, retarget D9 to
  `main` and revalidate before merge. This is not part of the bounded D9
  implementation closure.
  Retarget acceptance is currently blocked by
  `BLK-20260719-D9-D4-CURRENT-MAIN-AST-CLOSURE`, owned by codex-2.

## Completion Audit

- [x] The same D8 identified set is exercised under two designs with different
  exact coverage.
- [x] The same realized numeric region retains incompatible design and
  provenance identities.
- [x] Calibration, positivity, and instrument-population failure each force
  abstention.
- [x] Exact identified set, confidence region, predictive set, declared and
  external empirical binding, patient state, and clinical action authority are
  nominally non-substitutable.
- [x] No real empirical, patient, causal, or clinical authority claim is made.
- [x] No compiler/resolver or existing D0-D8 semantic file is changed.
- [x] All required local gates, ontology paths, offloads, and remote CI evidence
  are green.
