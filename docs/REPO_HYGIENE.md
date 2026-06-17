# Repository hygiene

Conventions that keep Sounio reviewable as it scales across many contributors
and automated agents. This is policy, not just cleanup — please follow it.

## Branches

- **Naming:** `type/slug` — `feat/`, `fix/`, `chore/`, `docs/`, `ci/`,
  `refactor/`, `perf/`, `research/`. Keep slugs short and descriptive.
- **Lifecycle:** branch from `main`, open a PR, **squash-merge**, then **delete
  the branch** (GitHub does this automatically; `--delete-branch` locally).
- **Don't let branches accumulate.** A merged PR's branch is dead — delete it.
  Stale unmerged branches should be revived into a PR or dropped.
- Cross-reference *merged PRs*, not just `git branch --merged`, when pruning:
  a squash-merged branch's commits are **not** ancestors of `main`, so it shows
  as "not merged" even though it is fully integrated.

## Worktrees

Parallel work (multiple agents/features) uses `git worktree`, not branch
switching in a shared checkout.

- **Never `git checkout` a different branch in a checkout another worker uses.**
  Switching the shared tree pulls files out from under in-flight work. Add a
  dedicated worktree instead: `git worktree add ../sounio-<slug> <branch>`.
- **Clean up when done:** `git worktree remove <path>` (no `--force`; it refuses
  if dirty, which is the point). Then `git worktree prune`.
- **Hygiene passes use git's safe guards only** — `git worktree prune`,
  `git branch -d` (never `-D`), `git worktree remove` without `--force`. Force
  is the line between safe and destructive; don't cross it in a sweep. Removing
  even a clean worktree *directory* can break an agent `cd`'d into it, so live
  directory removal is owner-approved, not automatic.

## Files that must not be committed

Binaries and large/generated artifacts do not belong in source history.
`.gitignore` enforces this for new files; the categories:

- Build output: `target/`, `build/`, `logs/`, `dist/`.
- Local seed/rebuild scratch: `bin/*.new`, `bin/*.prev*`, `bin/*.backup.*`,
  any `*.backup.*`.
- Generated data products: `data/processed/`, `artifacts/` (large, regenerable).

If you genuinely need a checked-in binary (e.g. a bootstrap seed compiler), it
belongs in **Git LFS**, not the main object store.

## Known debt (tracked separately)

- **Large binaries already in history** (e.g. a ~90 MB `bin/` seed, multi-MB
  `artifacts/` blobs). Removing these is a *coordinated, scheduled* operation
  (`git lfs migrate` / history rewrite + force-push across all branches and
  worktrees) — **not** an unsupervised hygiene change. Do not `filter-repo` ad
  hoc; it breaks every outstanding branch and worktree. Full coordinated
  procedure: **[REPO_LFS_MIGRATION.md](REPO_LFS_MIGRATION.md)**.
- **CI:** keep `main` green. Checks failing on `main` itself are tracked as
  issues; a PR that does not *add* failures relative to `main` is not blocked by
  pre-existing red, but the goal is zero red.

## PRs

- One logical change per PR; scope the diff (don't sweep unrelated dirty files).
- Describe what was verified. If CI is red, state whether it's pre-existing on
  `main` or introduced by the PR.
- Co-authored/automated commits: keep trailers; squash to a clean message.
