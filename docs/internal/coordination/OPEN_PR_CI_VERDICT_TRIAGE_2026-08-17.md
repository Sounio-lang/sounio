<!-- docs:meta
topic_id: repo.docs.internal.coordination.open-pr-ci-verdict-triage-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: codex-2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.open-pr-ci-verdict-triage-2026-08-17
-->

# Open PR CI Verdict Triage - 2026-08-17

## Scope

Dispatch asked for the open-PR backlog to be driven toward a state where each
open PR is either clean with a real CI verdict, or closed with a written reason.
The specific requested PRs were #1604, #1605, #1766, #1769, #1770, #1771, #1774,
#1775, #1777, and #1778.

This report is the measured state from an isolated worktree based on
`origin/main` at `dca2775061f0`. No merge was performed. One duplicate PR was
closed with a written reason.

Important measurement boundary: during this audit the live GitHub backlog moved.
After closing #1604, GitHub reported 64 open PRs, not the approximately 20 in the
dispatch. Several PRs also pushed new head SHAs while checks were being read, so
this document freezes the observed evidence rather than claiming the entire
moving queue is now stable.

## Trust Rule Used

Raw check count is not a verdict in this repository. The trusted signal for a PR
is:

- `CI Decision` ran on the PR head SHA and completed `success`, or
- `CI Decision` ran and completed `failure`, which is a real red verdict, or
- `CI Decision` is absent/pending, in which case GitHub's rollup is not enough.

`mergeStateStatus=DIRTY` or `mergeable=CONFLICTING` makes an old green head-SHA
rollup especially misleading: the suite may have run before the branch became
unmergeable, but the current merge candidate was not evaluated.

## Requested PRs

| PR | Merge state observed | Head SHA observed | Did selected CI run? | Verdict | Action |
| --- | --- | --- | --- | --- | --- |
| #1604 `feat(wasm): source-fresh Madaros backend closure` | `DIRTY` / `CONFLICTING` before closure | `32bf57e880d5a0bc64d39edff98491a6c7c6101d` | Historical `CI Decision` success on 2026-08-02, but current PR was conflicting | Not a current merge verdict | Closed as verified duplicate of #1605; branch not deleted |
| #1605 `feat(wasm): source-fresh Madaros backend closure` | `DIRTY` / `CONFLICTING` | `32bf57e880d5a0bc64d39edff98491a6c7c6101d` | Historical `CI Decision` success on 2026-08-02 | Not a current merge verdict | Kept as surviving draft; needs owner rebase/revalidation |
| #1766 `docs(audit): HLIR re-verify...` | `DIRTY` / `CONFLICTING` | `6bbd7efc53396d545e374aa354691c7e1264fa83` | No `CI Decision` row on the observed head; only PR Triage plus skipped issue jobs | Suite absent for mergeability purposes | Not rebased by codex-2; owner coordination required |
| #1769 `formal: SounioSedenionBipartite...` | `CLEAN` / `MERGEABLE` | `9ad355da0e7bdd5b51813cd16bd6e571ab9df508` | Yes: `CI Decision` success, plus Contracts and Lean Proofs success | Clean real verdict | No action needed from triage |
| #1770 `WS-A: Madaros status refresh...` then retitled `fix(stdlib): close E175...` | Initially `DIRTY` / `CONFLICTING`; later refreshed to `UNSTABLE` / `MERGEABLE` with new head `b56adc4846...` | Old observed SHA `e5f3d1cdad6ee96f5bf6f778f4b9a1307f662938`; later SHA `b56adc4846719047c9fa65b87ec1f59145514c6d` | Old SHA had full `CI Decision` success; new SHA was pending when refreshed | Active owner lane, not settled in this audit | Did not rebase or modify; notified minimax-cli2 on the bus |
| #1771 `docs(dissertation): honest STATUS_AUDIT...` | `DIRTY` / `CONFLICTING` | `73912ac5a3b9eb9fae13c82a743bb18e454d3c8c` | No `CI Decision` row on the observed head; only PR Triage plus skipped issue jobs | Suite absent for mergeability purposes | Did not rebase or modify; minimax-cli3 already owns |
| #1774 `Audit context-dependent red CI gates` | `CLEAN` / `MERGEABLE` | `db7459abb16aef1a37d3a96feef376e751bf4840` | Yes: `CI Decision` success; Impact and Contracts success; compiler jobs deliberately skipped by path selection | Clean real verdict for selected set | No action needed from triage |
| #1775 `test(ws-g): V0-C wire/limb ladder gate...` | `UNSTABLE` / `MERGEABLE` | `c59c8f27bfc2b44cd2f70c32957f98a10da5dfc6` | Yes: `CI Decision` success despite `PR Triage` failure | Real selected-set success, with non-blocking PR Triage red needing owner attention | No rebase; report PR Triage failure separately from CI Decision |
| #1777 `test(ws-g): V0-D softfloat gate...` | `CLEAN` / `MERGEABLE` | `f55d620cc06cf9b6124c157e055064c03750718c` | Yes: `CI Decision` success with full compiler matrix success | Clean real verdict | No action needed from triage |
| #1778 `Define the CI trust contract` | `CLEAN` / `MERGEABLE` | `de58dd2e6f3477a01ec4c906cd563b4e29f2fb43` | Yes: `CI Decision` success with full compiler matrix success | Clean real verdict | No action needed from triage |

## Duplicate Closure Evidence

#1604 and #1605 were not merely similar. They had:

- the same title;
- the same base branch, `main`;
- the same head branch, `codex/madaros-wasm-deontic-v3-20260802`;
- the same head SHA, `32bf57e880d5a0bc64d39edff98491a6c7c6101d`;
- identical PR body content, aside from final newline/JSON escaping;
- identical diff file list:

```text
bin/madaros
bin/souc
scripts/ci/bootstrap_chain_gate.sh
scripts/ci/build_modular_madaros.sh
scripts/ci/item_kind_dispatch_gate.mjs
scripts/ci/madaros_version_json_gate.mjs
scripts/ci/madaros_wasm_backend_gate.mjs
self-hosted/check/check.sio
self-hosted/compiler/module_native_driver.sio
self-hosted/io/file_write.sio
self-hosted/ir/lower.sio
self-hosted/resolve/imports.sio
self-hosted/wasm/lower.sio
tests/wasm/deontic_transport_finite_wasm_v0_3.sio
tests/wasm/deontic_transport_finite_wasm_v0_4.sio
tests/wasm/madaros_wasm_i64_fold.sio
```

Action taken: #1604 was closed with the written reason "verified duplicate of
#1605". #1605 remains open as the review surface.

## Newly Moving PRs Seen During Audit

These were not in the original requested list, but changed the headline backlog
while the audit was running:

| PR | Observed state | Selected CI state |
| --- | --- | --- |
| #1776 | moved to `CLEAN` / `MERGEABLE` at head `048e8100...` | rollup success after refresh; not expanded in this narrow audit |
| #1781 | `UNSTABLE` / `MERGEABLE` at head `cc636846...` | only PR Triage failure observed; no `CI Decision` row in the REST check-run read |
| #1782 | pushed new head `4571f5d0...` after an earlier failing SHA | pending after refresh |
| #1783 | `UNSTABLE` / `MERGEABLE` at head `402dc43b...` | real partial run with Contracts failure and Madaros Witness in progress at read time |
| #1784 | `UNSTABLE` / `MERGEABLE` at head `9fe0944a...` | real run in progress with Contracts and macOS self-host failure at read time |
| #1785 | `CLEAN` / `MERGEABLE` at head `ff0e1bb3...` | later rollup success after earlier in-progress checks |
| #1786 | `UNSTABLE` / `MERGEABLE` at head `21f8720c...` | rollup failure |
| #1788 | `UNSTABLE` / `MERGEABLE` at head `e2f99153...` | pending |

## Coordination

Bus notes were sent so this census composes with the two adjacent efforts:

- grok-cli5: this report covers suite-absent/conflict-aborted PR verdicts, not
  empty-input gates.
- minimax-cli2: this report does not duplicate the missing-fixture census, and
  #1770 was not modified under its ownership.

## Residual Work

The queue was not fully normalised because the backlog was actively changing
and because several problematic PRs are owned by other lanes. The remaining
unsafe states from the requested list are:

- #1605: duplicate survivor, still `DIRTY` / `CONFLICTING`; needs rebase and
  fresh `CI Decision`.
- #1766: `DIRTY` / `CONFLICTING`; no selected-set verdict on the observed head.
- #1770: active owner lane pushed a new head during this audit; wait for its new
  `CI Decision`.
- #1771: `DIRTY` / `CONFLICTING`; no selected-set verdict on the observed head.

No Slurm or heavy local validation was run.
