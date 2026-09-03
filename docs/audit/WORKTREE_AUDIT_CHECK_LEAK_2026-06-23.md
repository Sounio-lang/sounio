<!-- docs:meta
topic_id: repo.docs.audit.worktree-audit-check-leak-2026-06-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.worktree-audit-check-leak-2026-06-23
-->

# Worktree Audit `--check` Output Leak

Date: 2026-06-23
Status: closed (root cause already fixed; residue cleaned; regression test added)
Evidence level: E3 (gate-bound — `scripts/dev/check_audit_out_invariant.sh`)

## Symptom

An untracked 69 KB file literally named `--check` appeared at the repository root
(`/workspace/sounio/--check`). Its contents were the full TSV worktree inventory
produced by `scripts/dev/worktree_branch_audit.sh` (header
`path\tbranch\thead\tupstream\tstate\tdirty_count\t...`), with none of the
script's stdout markers (`audit_tsv=`, `total=`, `critical_dirty_worktrees:`).

The presence of only the TSV body (no stdout) proves the file was the script's
`$OUT` target, not a stdout redirect.

## Root cause

Commit `bc0513783~1` parsed the output path positionally:

```sh
OUT="${1:-}"
```

so an invocation of the form

```sh
scripts/dev/worktree_branch_audit.sh --check
```

silently assigned `OUT="--check"`. Because `$OUT` is a relative path resolved
against the repo root (the script does `cd "$ROOT_DIR"`), the inventory was
written to `./--check`.

## Fix (already landed)

`bc0513783` (2026-06-21, on `origin/main`) replaced the positional capture with
a real argument loop:

- `--check)` sets `CHECK_MODE=1`,
- `-*` rejects unknown leading-dash tokens,
- `*)` is the only path that can set `OUT`.

With this parser there is no way for a leading-dash token to become `OUT`, so
the script cannot regenerate `./--check`. The single live caller
(`scripts/dev/madaros_readiness_status.sh`) already uses the safe form
`worktree_branch_audit.sh --check "$audit_out"` with a `/tmp` mktemp path.

## Residue + action

The `./--check` file was stale output from a run against the pre-fix version and
was never cleaned up. It has been removed from the working tree. No tracked file
depended on it (legitimate inventories are written under `/tmp/...tsv`).

## Regression guard

`scripts/dev/check_outpath_invariant.sh` locks the invariant for the whole
family of `[--check] [OUT.tsv]` output-path dev scripts — currently
`scripts/dev/worktree_branch_audit.sh` and `scripts/dev/madaros_pr_resolution_queue.sh`
(the latter already followed the safe parser pattern):

1. the vulnerable `OUT="${1:-}"` form must not return;
2. `--check` must remain a dedicated flag case;
3. unknown `-*` tokens must be rejected;
4. an unknown flag exits non-zero during arg parsing (fast, no network or full
   scan, not flaky against global worktree state).

Verified: the guard passes on `origin/main` and fails when either covered script
is regressed (e.g. `bc0513783~1:scripts/dev/worktree_branch_audit.sh`, or removing
the `--check)` case from the PR-resolution queue).
