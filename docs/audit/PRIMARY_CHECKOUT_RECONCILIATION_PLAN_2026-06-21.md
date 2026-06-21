<!-- docs:meta
topic_id: repo.docs.audit.primary-checkout-reconciliation-plan-2026-06-21
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.primary-checkout-reconciliation-plan-2026-06-21
-->

# Primary Checkout Reconciliation Plan — 2026-06-21

## Scope

This plan turns the preserved dirty primary-checkout archive into an ordered
resolution queue. The archive is:

- `/workspace/sounio-worktree-archives/primary-main-dirty-20260620/`

The goal is to reduce ambiguity without replaying stale or compiler-adjacent
work into `origin/main`. Each bucket must end as one of:

- a small PR with a named validation gate
- a blocker under `.claude/PARALLEL_BLOCKER_CONTRACT.md`
- an archived/discarded local artifact with the reason recorded

No step in this plan requires cleaning, resetting, rebasing, or switching the
protected `/workspace/sounio` checkout.

## Current Baseline

- Current `origin/main`: `32ed3d3d71a94a6b328d2fff71395f8af7ed8c19`
  (`Merge pull request #347 from Sounio-lang/codex/agents-ctx7-contract`).
- The primary archive was created from `/workspace/sounio` without destructive
  cleanup.
- The MCP patch bucket from the primary archive is already represented on
  `origin/main` by `81fa4976e` (`fix(mcp): align check json invocation and test
  paths`).
- The `AGENTS.md` bucket is resolved by #347, which adds the Codex-facing `ctx7`
  documentation contract.
- The worktree governance gate is resolved by #346 and runs in the `Contracts`
  CI job.

## Resolution Queue

### Bucket A — Already Resolved

These items require no replay from the archive:

| Item | Disposition | Evidence |
|---|---|---|
| `tools/mcp/sounio_mcp/check.py` | already applied | `81fa4976e` |
| `tools/mcp/sounio_mcp/test.py` | already applied | `81fa4976e` |
| `tools/mcp/tests/test_loop.py` | already applied | `81fa4976e` |
| `tools/mcp/tests/test_tools.py` | already applied | `81fa4976e` |
| `AGENTS.md` | resolved | #347 |
| `scripts/dev/worktree_branch_audit.sh` | superseded | #346 adds check mode and CI wiring |

Exit criterion: no PR. Keep the archive as provenance only.

### Bucket B — Safe To Review, Not Safe To Auto-Promote

These items are small, but they still need normal review because the archived
state is local workspace state rather than a proven branch:

| Item | Initial finding | Required gate |
|---|---|---|
| `examples/hello.sio` | changes hello from return-only to `println("Hello, Sounio")` with `with IO` | `env -u SOUC_BIN -u SOUNIO_STDLIB_PATH -u MADAROS_BIN -u SOUNIO_MADAROS_BIN bash scripts/run_sio_test_suite.sh hello --verbose` |
| `examples/erdos/reproducer_madaros_codegen_2026-06-16g.sio` | likely a debugging reproducer, not a general example | classify against current Madaros gates before adding |
| `docs/audit/MADAROS_*.md` from archive | useful forensic notes, but several contradict newer fixed state | promote only after refresh against current `origin/main` and current binary SHA |

Exit criterion: one narrow PR per item or a written discard reason.

### Bucket C — Local Operational Artifacts

These items should not be committed to `origin/main` unless a separate repo
contract exists for them:

| Item | Reason |
|---|---|
| `.beagle/context/**` | workspace hydration state, not source |
| `artifacts/omega/agent_handoff.log.md` | local agent log; preserve in archive, do not replay |
| `data/processed/expansion/**` | large generated data; inventoried but intentionally not copied |
| `bin/madaros.bak-20260619` | local backup binary/wrapper; never promote as source |

Exit criterion: archive only, unless a future task explicitly creates a source
contract for one of these surfaces.

### Bucket D — Scripts Requiring Hardening Before Promotion

The archived helper scripts are not production-ready as-is:

| Item | Finding | Required before PR |
|---|---|---|
| `scripts/coverage_report.sh` | contains a malformed `basename " $f" .mdx` line and lacks a CI contract | fix shell issues, add `shellcheck`/dry-run style validation |
| `scripts/validate_mdx.sh` | uses `TOTAL`/`FAILURES` under `set -u` without initialization and installs packages at runtime | initialize state, avoid implicit dependency mutation, validate under `website` workflow expectations |
| `scripts/translate_all_locales.sh` | hardcodes an internal Beagle service and writes logs in repo root | parameterize endpoint/model/output path and document that it is operator-only |
| `slurm-jobs/madaros-frame-fix/*.sh` | operational Slurm helpers tied to a specific Madaros frame-fix lane | refresh against current foundry/Slurm contract before adding |

Exit criterion: do not commit raw scripts. Promote only via a dedicated tooling
PR with shell validation and an operator-facing contract.

### Bucket E — Compiler-Adjacent High Risk

These items must not be replayed directly while Claude owns an active compiler
lane:

| Item | Risk | Required gate |
|---|---|---|
| `bin/madaros` | launcher/default compiler behavior can contaminate all gates | inspect against `scripts/lib/resolve_madaros.sh`; run source-bootstrap and focused Madaros gates |
| `self-hosted/native/machine_ir.sio` | compiler IR surface; may overlap native/codegen work | reproducer first, then focused native gate; no edit if it overlaps Claude's active files |

Exit criterion: either a clean isolated compiler PR or a blocker record with
owner, severity, evidence level, acceptance gate, and next action.

## Claude Coordination Rule

The active Claude compiler lane currently owns compiler changes. Codex must not
edit the same files in parallel. Codex may:

- inspect the lane
- run read-only diffs or gates
- prepare independent governance/tooling PRs
- review the result after Claude's edit completes

Codex must not write to Claude-owned compiler files during the same phase.

## Operating Order

1. Keep `origin/main` green and use it as the only production baseline.
2. Resolve Bucket A by recording it as already done.
3. Review Bucket B one item at a time, starting with `examples/hello.sio`.
4. Mark Bucket C as archive-only unless a future source contract is created.
5. Harden or discard Bucket D scripts; never commit them raw.
6. Treat Bucket E as compiler work requiring a reproducer, a focused gate, and
   explicit non-overlap with Claude's lane.
7. After every PR, wait for CI and remove the temporary worktree/branch.

## Stop Rule

If a bucket cannot be classified with current evidence, do not guess and do not
apply the archive patch. Record the missing evidence and convert it into a
blocker or a focused probe.
