<!-- docs:meta
topic_id: repo.docs.internal.coordination.compiler-lane-contract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.compiler-lane-contract
-->

# Compiler Lane Contract

Sounio separates compiler work by evidence state rather than by the number of
branches or worktrees that happen to exist.

- `origin/main` is the integrated product surface.
- `origin/canon/madaros-v2-sota` is the ENIR/MIR compiler frontier.
- recent dirty worktrees are active benches and retain exclusive write
  authority over the compiler paths they touch.
- old dirty worktrees are preserved residue, not active work and not deletion
  candidates.

`scripts/dev/compiler_lane_status.sh` provides the read-only snapshot. Its
classifications mean:

- `INTEGRATED`: the clean worktree tip is contained in the configured main ref.
- `CONTENT_INTEGRATED`: Git topology differs, commonly after a squash merge,
  but every committed compiler path is byte-identical to main.
- `FRONTIER`: the canonical ENIR/MIR frontier worktree.
- `FRONTIER_INTEGRATED`: the clean worktree tip is contained in the configured
  frontier ref rather than awaiting review against main.
- `REVIEW_READY`: the worktree is clean but its tip is not contained in main.
- `ACTIVE`: a compiler path has a recent working-tree modification.
- `STALE_WITH_RESIDUE`: compiler changes remain, but no recent file activity was
  observed.
- `SCRATCH_COPY`: compiler files occur in a scratchpad worktree; they are inputs
  for comparison, not an authoritative lane.
- `UNCLASSIFIED`: the scanner lacks enough Git evidence.

These states are observations. They never assign an owner, authorize a merge,
or prove semantic correctness. Promotion still requires the lane's positive
witness, negative witness, acceptance gate, correct integration base, and any
review mandated by `.claude/AGENT_OFFLOAD_POLICY.md`.

Run:

```bash
bash scripts/dev/compiler_lane_status.sh --verbose
bash scripts/ci/compiler_lane_status_gate.sh
```

An active lane must not edit a compiler path already reported as active in
another worktree. A stale lane must be reduced to a bounded witness before any
code is recovered. An integrated lane may be removed only through an explicit
cleanup decision; the scanner itself is deliberately incapable of doing so.
