<!-- docs:meta
topic_id: repo.docs.governance.branch-policy
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.governance.branch-policy
-->

# Branch Policy

> **Status**: Beta | **Last validated**: 2026-07-21 | **Source**: this session's branch audit (`git for-each-ref`, `gh pr list`) + `scripts/dev/worktree_branch_audit.sh`

Sounio is a self-hosted programming language under heavy multi-agent development.
That workflow produces a large volume of short-lived branches — probes, per-lane
worktrees, rescue branches. Left ungoverned it becomes sprawl: on 2026-07-21 the
repository carried **772 local** and **479 remote** branches. Sprawl is not just
untidy — it makes real work indistinguishable from abandoned experiments and it
has already caused an incident (see §1).

This document is the forward-looking **policy**. For the historical incident-driven
inventory see [`docs/audit/WORKTREE_BRANCH_GOVERNANCE_AUDIT_2026-06-20.md`](../audit/WORKTREE_BRANCH_GOVERNANCE_AUDIT_2026-06-20.md)
(`GOV-WORKTREE-SPRAWL-20260620`). For worktree/agent coordination see
[`CLAUDE.md`](../../CLAUDE.md) §4 and `bin/sounio-coord`.

---

## 1. The one safety rule (read this first)

**Deleting a *local* branch ref is safe. Deleting or pushing the deletion of a
*remote* branch can close an open pull request.**

- `git branch -d` / `git branch -D` only removes a local ref. It never contacts
  the server, so it **cannot** close a PR. A PR tracks `origin/<branch>`, not your
  local ref.
- `git push origin --delete <branch>`, `git push origin :<branch>`, a `--mirror`
  push, or a server-side bulk "delete branches" **removes the remote head** and
  will close/detach the PR built on it.

On **2026-06-17** a branch-prune flow auto-closed **15 open PRs** in this
repository (recovered). The cause class is remote-side deletion. Therefore:

> **Never automate remote branch deletion.** Any tool, script, or agent action
> that could push a deletion is out of scope for routine hygiene and requires
> explicit human confirmation, one branch at a time, after verifying the PR is
> already merged or closed.

Before any batch local deletion, verify nothing silently propagates a local ref
change to the remote:

```bash
git config --get core.hooksPath          # where hooks live
ls "$(git config --get core.hooksPath || echo .git/hooks)"
# there must be NO pre-push / reference-transaction / post-checkout hook, and no
# CI mirror, that pushes branch-state to origin. If one exists, "local-only" is a
# myth and deletion is unsafe until it is understood.
```

As of 2026-07-21 the only active hooks are `post-merge` (governance-doc regen)
and `pre-commit` (offload-policy check) — neither touches the remote.

---

## 2. Branch naming taxonomy

Use one of the canonical prefixes below. The prefix declares intent and drives
retention (§4). Names are `prefix/short-kebab-description`, optionally suffixed
with an ISO date `-YYYYMMDD` and/or a round `-rNN`.

| Prefix | Meaning | Retention |
|---|---|---|
| `feat/` | New capability (language, stdlib, tooling) | Until merged, then delete |
| `fix/` | Bug fix | Until merged, then delete |
| `docs/` | Documentation only | Until merged, then delete |
| `chore/` | Maintenance, deps, config | Until merged, then delete |
| `research/` | Exploratory / experiment, may never merge | TTL 30 days (§4) |
| `codex/` | Codex-owned automated lane | TTL 14 days after last commit |
| `agent/` | Named-agent automated lane | TTL 14 days after last commit |
| `salvage/` | Rescued dangling/consolidated work (machine) | Keep until triaged, then delete |
| `integration/` | Long-lived integration base | Permanent (protected) |

**Retire these near-synonyms** — pick the canonical form and stop minting the
others:

- `recover/`, `recovery/` → use `salvage/`
- `work/` → use `feat/` or `research/` by intent
- `debug/`, `probe/*` → use `research/`
- no-prefix branches → always add a prefix

Machine-generated names (`worktree-agent-*`, `worktree-wf_*`) are created by the
harness, not humans (see §6).

---

## 3. Probe / round branches

Compiler and IR work legitimately produces iteration branches
(`codex/<lane>-r17`, `-r18`, … `-rNN`) and dated variants
(`<lane>-20260719`). These are the largest source of sprawl and of exact
duplicates — on 2026-07-21 eleven branches pointed at a single commit.

Rules:

1. Only the **latest round** of a lane is a landing candidate. Earlier rounds are
   history, and must not be kept as separate live branches once superseded.
2. When several rounds sit at the **same commit**, keep one and delete the local
   aliases (§5). They carry no unique work.
3. Probe branches are `research/`- or `codex/`-class and inherit that TTL (§4).

---

## 4. Lifecycle & retention

**On merge.** A merged branch's work is on `main`, so the branch is redundant.

- **Enable the GitHub repository setting "Automatically delete head branches."**
  This self-cleans the *remote* on every future merge — it only affects branches
  whose PR just merged, so it carries none of the §1 risk and removes the need to
  ever run a bulk remote delete for merged work.
- Local merged refs are cleaned by the §5 procedure.

**Time-to-live for unmerged lanes.** A `research/`/`codex/`/`agent/` branch with
no commits for longer than its TTL and no open PR is a cleanup candidate — but
only ever locally, or on the remote by explicit human decision per §1.

**Never auto-expire:** anything with an open PR, `integration/*`, `salvage/*`
awaiting triage, or a branch checked out in a live worktree.

---

## 5. Safe local cleanup procedure

This is the exact, audited procedure. It only ever removes **local** refs.

**Safe tier** = a branch that is **all** of:

1. merged into `main` (`git merge-base --is-ancestor <b> main`), **and**
2. local-only — no `refs/remotes/origin/<b>`, **and**
3. not an open-PR head (`gh pr list --state open`), **and**
4. not checked out in any worktree (excluded by the `+` marker in `git branch`).

Procedure:

```bash
# 1. Build the exclude sets
gh pr list --state open --limit 400 --json headRefName \
  --jq '.[].headRefName' > /tmp/pr_heads.txt

# 2. For each merged, not-checked-out branch (strip '*' and '+' markers):
git branch --merged main | grep -vE '^\s*[*+]' | sed 's/^\s*//' | while read -r b; do
  git show-ref --verify -q "refs/remotes/origin/$b" && continue   # has remote → skip
  grep -qxF "$b" /tmp/pr_heads.txt && continue                    # PR head → skip
  sha=$(git rev-parse "$b")
  git merge-base --is-ancestor "$b" main || continue              # RE-CHECK right before
  echo "DELETE $b @ $sha"                                         # log SHA for recreate
  git branch -D "$b"
done
# Recreate any deletion with:  git branch <name> <sha>
```

Non-negotiables:

- **Re-verify `merge-base` immediately before each delete.** The shared checkout's
  `HEAD` moves under concurrent agents, so a stale list is not evidence. A stale
  local `main` only makes this more conservative, never a false delete.
- **Log every `branch @ sha`.** Recreation is then one command.
- **Never `push`.** Not even `--dry-run` deletions to the server.

---

## 6. Machine-generated branches

- `worktree-agent-*`, `worktree-wf_*` — created by the harness for per-agent and
  per-workflow worktrees. They are byte-duplicates of `main` plus uncommitted
  work. When their worktree is reclaimed and the work is committed or discarded,
  their local ref is a §5 candidate. Do not hand-mint these names.
- `salvage/*` — created when reclaiming a worktree that held a **dangling**
  detached-HEAD commit (unreachable from any branch). Convention:
  `salvage/<basename>-<shortsha>`. Removing a worktree drops its per-worktree
  `HEAD`/reflog, so a dangling commit must be branched to `salvage/*` **before**
  the worktree is removed, or it becomes recoverable only via `git fsck`. Triage
  `salvage/*` promptly (promote to a PR, cherry-pick, or delete) and do not let them
  accumulate.

---

## 7. Coordination discipline

Multi-agent work shares one `.git`. Branch and worktree state is repository-global,
so one agent's cleanup is visible to all.

- One worktree per agent. Never run a second agent directly on `/workspace/sounio`
  (CLAUDE.md §4).
- Claim write-bearing branch/worktree operations through `bin/sounio-coord`.
- Re-check `git status` / branch existence immediately before staging or deleting,
  another agent may have moved the shared `HEAD` or committed on a branch you hold.

---

## 8. Enforcement

- `scripts/dev/worktree_branch_audit.sh [--check]` produces the worktree/branch
  inventory and, in `--check` mode, fails on prunable worktree records and
  unallowed dirty critical worktrees. The CI `Contracts` job runs it in strict
  mode.
- This policy governs *branches*, the audit gate governs *worktrees*. They are
  complementary. A future gate extension could warn on: exact-duplicate branch
  tips, no-prefix branches, and probe branches past TTL — as **warnings only**,
  never automated deletion (§1).

---

## 9. Recommended one-time actions

Ordered, lowest-risk first:

1. **Turn on GitHub "Automatically delete head branches."** Zero risk to history —
   stops future remote merged-branch accumulation at the source.
2. Run the §5 local cleanup to clear merged local-only refs.
3. Triage outstanding `salvage/*` (§6).
4. For the remote merged-no-PR backlog, delete **per branch, by hand**, after
   confirming each PR is merged — never in bulk, never scripted (§1).
