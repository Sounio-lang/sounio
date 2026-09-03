<!-- docs:meta
topic_id: repo.docs.handoff.pending-changes-triage-2026-07-11
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.pending-changes-triage-2026-07-11
-->

# Pending working-tree changes — triage (2026-07-11)

Snapshot of the uncommitted working tree on branch `cpc2026-ossm-native-run` at
the time the Madaros/CPC2026 reconciliation (PR #745) was pushed. **None of the
items below belong to that PR.** They are in-flight work from other agents /
campaigns on this shared checkout, plus regenerable artefacts and root scratch.

Recorded so the owners (or a later session) can commit each cluster cleanly
instead of re-deriving what is pending. Repo hygiene is *not* automated on this
checkout (concurrent agents; a past branch-prune auto-closed 15 PRs), so nothing
here was committed or deleted on anyone's behalf.

Regenerate this view: `git status --porcelain`.

## Cluster A — neurodyn / computational-psychiatry documentation batch

Coherent unit: the governance registry edits **register** the handoff/prereg
docs (each carries a `docs:meta authority:repo_only` block). Commit together.

- Modified (tracked): `docs/governance/topic-registry.v1.json` (+385),
  `docs/governance/DOCS_AUTHORITY_MATRIX.md` (+17),
  `docs/governance/DOCS_ACCEPTANCE_REPORT.md`.
- Untracked handoffs: `docs/handoff/neurodyn_*` (oct-mul sign fix ×2, algebra-B
  null-retrain, algebra-C opus coordination + critique, ab-fixed octmul
  re-audit, adhd200 pcp pilot24, computational-psychiatry framework audit) and
  `docs/handoff/compiler_generic_struct_return_*` /
  `compiler_generic_F_engine_unblock_prompt.md` /
  `spike_generic_struct_return.sio` / `docs/handoff/continuity/repro_eisa_f64_sret/`.
- Untracked preregs: `docs/research/neurodyn_*` (abide dynamic-FC switching,
  algebra-B attribution, algebra-C continuous associator, ossm adhd dimensional,
  ossm SOTA deep-research) + `docs/research/gpu-swarm-coordination-2026-07-05.md`.
- Probable owner: the neurodyn campaign agent.

## Cluster B — native O-SSM / ekan CI gates

- Modified (tracked): `examples/brain_ossm_abide.sio` (+92).
- Untracked: `scripts/ci/brain_ossm_sigmoid_polarity_gate.sh`,
  `scripts/ci/ekan_native_*` (6 gates/probes),
  `tests/run-pass/brain_ossm_sigmoid_polarity.sio`,
  `tests/run-pass/int_arg_narrow_nonliteral.sio`,
  `tests/known_failures/ekan_fixed_point_*_native_v2_*.sio` (12 blocker probes),
  `examples/epistemic_kan_fixed_point.sio`, `tests/test_op.sio`,
  `tests/test_op_gen.sio`.
- Probable owner: the ekan / native-v2 O-SSM lane agent.

## Cluster C — GPU / ABIDE research drivers

- Modified (tracked): `scripts/gpu/prepare_abide_campaign_snapshot.sh` (+2).
- Untracked: `scripts/research/` (~68 files) — `abide_*` (dynamic-FC gates,
  runners, checkpoint persist, trained ROI associator), `adhd200_*` (data-access
  audit, dimensional pilot gates, s3 bootstrap, pcp pilot24 reproduce),
  `neurodyn_*` (~60: associator/orientation/fano/temporal-arrow probes, decision
  gates, evidence bundle, null envelopes, readout traces).
- Probable owner: the GPU/ABIDE campaign agent.

## Regenerable artefacts — leave as-is

`artifacts/{eisa,gpu,posters,research,self-hosted}/*`,
`examples/cognitive_ossm/results/*.json`. Generated outputs / gate receipts;
regenerable, not source. `.claude/llm_offload_log.md` (+24) is a shared
append-only log, not a discrete commit.

## Root scratch strays — now gitignored (2026-07-11)

`count.py`, `package.json` (an `@augmentcode/auggie` dep, not a Sounio file),
`test_op.sio`, `test_struct_func.sio`, `test_struct_ret.sio`, `test_probe.asm`,
`test_probe.elf`. Anchored `/`-prefixed entries added to `.gitignore` so they
stop polluting `git status` without deleting them (the two `test_struct_*`
probes may still be live repro for the EISA f64-sret / generic-struct-return
work in Cluster A). Remove the ignore lines and the files once that work lands.
