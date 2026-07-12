<!-- docs:meta
topic_id: repo.docs.internal.coordination.semantic-hotspot-triage-2026-07-12
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.semantic-hotspot-triage-2026-07-12
-->

# Semantic Hotspot Triage: 2026-07-12

Snapshot commit: `ecc5c4018`

This is an observational coordination receipt. It does not grant ownership,
merge branches, remove worktrees, or declare old residue obsolete.

## Classification Rules

- `INTEGRATED`: the inspected branch tip is an ancestor of `origin/main`.
- `REVIEW_READY`: focused work is committed, its worktree is clean, and the
  branch is not known to be integrated.
- `ACTIVE`: recent committed or working-tree activity remains in progress.
- `STALE_WITH_RESIDUE`: an old worktree retains a nonempty patch whose unique
  value has not been extracted or disproved.
- `SCRATCH_COPY`: untracked copies exist without a committed authoritative
  lane; they are evidence inputs, not integration candidates.
- `UNCLASSIFIED`: evidence is insufficient for a stronger state.

## Current Lanes

| Lane | Surface | State | Exact evidence | Next action |
|---|---|---|---|---|
| `feat-checker-e213-tuple-arity` | checker | `INTEGRATED` | tip `b480d839c` is an ancestor of `origin/main`; worktree removed | retain commit as receipt; no recovery action |
| `work/madaros-v2-e3d-multipred-scalar-memory-ssa-full-codex` | ENIR/MIR SSA | `REVIEW_READY` | clean worktree; tip `ce2f94407`; synchronized with branch remote; not contained in `origin/main` | review semantic contract and validate against its actual integration base |
| `gpu/epistemic-tensor-core-next` | IR/lowering/EISA | `ACTIVE` | tip `23e8af265`; current uncommitted change in `self-hosted/ir/lower.sio`; not contained in `origin/main` | preserve owner lane; request receipt when focused gate is complete |
| `neurodyn-docs-registry-batch` | EISA zero flags | `ACTIVE` | `core_v2.sio` and `evm.sio` contain 38 uncommitted inserted lines | keep bound to ZeroEvent gates; do not absorb unrelated EISA scratch copies |

The large divergence counts for the ENIR and EISA branches are base-topology
signals, not semantic failure evidence. They prohibit a casual direct merge but
do not demote the committed work by themselves.

## Residue Queue

These entries require extraction or comparison before archive decisions.

| Surface | Lane/worktree | Patch size | State | Required proof |
|---|---|---:|---|---|
| `check.sio` | `recover/green-first-phase03-step5` | `+356/-13` | `STALE_WITH_RESIDUE` | compare diagnostics and variance behavior with current checker |
| `check.sio` | `codex/project-spine-madaros` | `+116/-19` | `STALE_WITH_RESIDUE` | extract any unique project-spine witness before retirement |
| checker/IR/codegen | `codex/madaros-retire-lean-single-20260628` | `+482/-254` | `STALE_WITH_RESIDUE` | split bootstrap retirement from semantic/compiler fixes |
| IR/codegen | `fix/madaros-singlemodule-fncount-lowering-139` | `+873/-174` | `STALE_WITH_RESIDUE` | isolate function-count lowering witness and compare with current native path |
| IR/codegen | detached lower-known-test worktree | `+184/-117` | `STALE_WITH_RESIDUE` | identify owning commit or export a minimal patch receipt |
| codegen | `codex/madaros-plan-mainline-20260702` | `+182/-45` | `STALE_WITH_RESIDUE` | compare with current mainline emitter before extracting |
| codegen | `fix/bdf64-bridge` | `+52/-6` | `STALE_WITH_RESIDUE` | rerun bounded bdf64 bridge witness on current compiler |
| codegen | `work/madaros-s-next-codex` | `+166/-15` | `STALE_WITH_RESIDUE` | identify unique S-next behavior and its negative witness |
| codegen | `research/solver-ts3-parallel-20260704` | `+28/-63` | `STALE_WITH_RESIDUE` | keep research assumptions separate from production emitter changes |
| codegen | `work/madaros-greenline-codex` | `+27/-7` | `STALE_WITH_RESIDUE` | extract minimal green-line regression witness |
| codegen | `codex/abide-madaros-singlemodule-20260630` | `+31/-29`, staged | `STALE_WITH_RESIDUE` | compare staged patch and record why it was never committed |

No row above is authorized for deletion. `STALE_WITH_RESIDUE` means preserved
but non-authoritative until its unique behavior is demonstrated.

## Scratch Copies

Six Claude scratch worktrees expose untracked copies of
`stdlib/eisa/core_v2.sio` and `stdlib/eisa/evm.sio`. They are classified
`SCRATCH_COPY`: their files are not diffs against a tracked base, and their
presence must not be counted as six semantic writers. Compare content only if
an active lane reports missing evidence.

## Hotspot Integration Order

1. Accept E213 as integrated evidence; do not reopen it without a regression.
2. Review ENIR E3D against the semantic interface contract and the branch's
   real base before choosing cherry-pick, rebase, or successor-lane extraction.
3. Let the active EISA lane finish `lower.sio`; do not concurrently edit its
   IR surface.
4. Extract bounded witnesses from stale checker/codegen residue before any
   archive or worktree removal.
5. Re-run the read-only scanner after each integration receipt.

## Known Coordination Blocker

`BLK-20260712-SEMANTIC-DOCS-REGISTRY` remains a `B2 state-sync` blocker at
evidence level `E2`: global docs-registry regeneration also sees unrelated
untracked CPC/ORC documents in the shared checkout. Owner: CPC/ORC docs lane.
Acceptance gate: `bash scripts/dev/check_docs_registry.sh`. Next action: land or
isolate those documents, then regenerate governance metadata without absorbing
them into this lane.

