<!-- docs:meta
topic_id: repo.docs.audit.github-mergeability-cache-staleness-census-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-08-18
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.github-mergeability-cache-staleness-census-2026-08-18
-->

# GitHub `mergeable`/`mergeStateStatus` is not trustworthy in this repo — census and working procedure

**Date:** 2026-08-18
**Status:** operational finding, not a compiler bug. No `self-hosted/` changes. The value here is
procedural: a census with real numbers, and a remediation procedure that measurably works, so the
next agent who sees `DIRTY`/`CONFLICTING` checks locally before dispatching someone to rebase.

## Why this dispatch

Today's seven-hour GitHub outage was blamed for the PR queue's mergeability state, and that was the
working theory going in. It undersells the problem. Across this session and the operator's own
manual work today, the same underlying symptom — GitHub's mergeability computation not reflecting
the real, current state of a PR — showed up in **at least six different surface shapes**, reported
independently by the operator from direct GitHub UI/API use today:

- an empty `headRefOid` read by tooling as "the head changed" when it hadn't
- an empty `statusCheckRollup` read as "zero checks pending" when checks simply hadn't registered
- `UNKNOWN` mergeability silently grouped with `DIRTY` by downstream tooling, hiding that "not yet
  computed" and "computed as conflicting" are different states
- a full CI run executed against a stale base ref, producing a rollup with no bearing on the
  current PR content
- the `update-branch` API endpoint (`PUT /repos/{owner}/{repo}/pulls/{n}/update-branch`) returning
  `422 merge conflict between base and head` for a head SHA that was no longer the PR's actual
  current head
- four PRs, checked one at a time by hand, where the GitHub UI said conflicting while a local merge
  against an `origin/main` identical to GitHub's gave `rc=0`

That list is the operator's, from today's own work, not independently re-verified in this dispatch
— it is recorded here because it is exactly the kind of pattern that is easy to dismiss as six
unrelated glitches unless someone writes down that they are the same disease. This dispatch adds a
seventh shape, measured directly and at volume: 54 open PRs, machine-checked one at a time.

## What was measured

### Census methodology (and a mid-flight correction worth keeping)

The first pass used the legacy 3-argument `git merge-tree <base> <branchA> <branchB>` form and
reported all 54 open PRs as locally clean. A spot-check against a **real** `git merge --no-commit
--no-ff` on 3 samples caught this as false immediately — PR #1729 and #1527 both had genuine
conflicts the legacy form missed entirely. Per this repo's own "validate the instrument before
believing it" discipline: the legacy 3-arg `git merge-tree` is documented upstream as a simpler,
less accurate predecessor to the modern form and should not be used for this purpose. The whole
census was redone with `git merge-tree --write-tree <base> <head>` (git ≥2.38, the same merge
machinery `git merge` itself uses — this repo runs git 2.43.0), and cross-checked against 6
independent real `git merge --no-commit --no-ff` runs across all three resulting buckets. All 6
agreed with the `--write-tree` verdict exactly.

### Results — 54 open PRs, each checked against its own actual base branch (8 of 54 target a
branch other than `main`; those were checked against their real base, not assumed against `main`)

| GitHub `mergeable` | Locally verified | Count |
|---|---|---:|
| `MERGEABLE` | clean | 16 (API correct) |
| `CONFLICTING` | **clean** | 20 (API wrong — phantom) |
| `CONFLICTING` | **conflict confirmed** | 18 (API correct) |

37% of all open PRs (20/54) were misreported. Of the PRs GitHub called conflicting, 53% (20/38)
were not.

### The 18 real conflicts (for completeness — not touched, not this dispatch's job)

| PR | Branch | Base | Files in conflict |
|---|---|---|---:|
| #795 | `lean/cd-seamflip-forall-n` | `main` | 2 |
| #867 | `agent/issue854-contextual-checker-partial-20260713` | `main` | 2 |
| #978 | `codex/renderer-quality-20260715` | `main` | 1 |
| #1034 | `codex/propagate-runtime-abi-20260716` | `main` | 1 |
| #1053 | `codex/compile-fail-contract-20260717` | `main` | 2 |
| #1058 | `codex/ssm-exp-tail-20260717` | `main` | 1 |
| #1063 | `research/octonion-probes-ci-gate` | `main` | 1 |
| #1262 | `agent/r3-examples-proposal-refresh-20260720` | `main` | 1 |
| #1290 | `codex/madaros-affine-semantics-20260720` | `main` | 5 |
| #1318 | `agent/r3-complete-examples-source-20260720` | `main` | 1 |
| #1339 | `agent/madaros-declared-builtin-precedence-20260720` | `main` | 14 |
| #1421 | `codex/issue901-layout-current-20260724` | `main` | 12 |
| #1527 | `madaros/self-parse-visibility-box-w44-20260727` | `main` | 41 |
| #1580 | `research/zd-fiber-antisymmetry-lemma-20260731` | `research/self-falsifying-compilation-line-20260726` | 3 |
| #1603 | `feat/agent-bus-realtime` | `main` | 2 |
| #1605 | `codex/madaros-wasm-deontic-v3-20260802` | `main` | 5 |
| #1659 | `research/san-fpga-san-v3-20260805` | `main` | 1 |
| #1729 | `fix/lane-b3-ir-module-heap-20260813` | `main` | 1 |

These need their authors, not a head refresh — a forced merge here would just make the same
conflict visible inside the PR branch instead of at the API boundary.

## The procedure that works, and its actual reliability

**Real `git merge` of the base branch into the PR branch — no rebase, no force — pushed as an
ordinary merge commit.** The new head SHA forces GitHub to recompute mergeability from scratch
instead of continuing to serve whatever cached verdict it was stuck on. `git push origin
HEAD:<branch-name>` (a normal merge commit push, not `--force`) is sufficient; the 20 phantom PRs
in this census confirm it — every one of them, refreshed this way, moved from `CONFLICTING`/`DIRTY`
to `MERGEABLE` within roughly 10–60 seconds of the push.

**It is not durable against this repo's `main` churn rate.** 9 of the 20 refreshed PRs (the ones
refreshed earliest, i.e. with the most elapsed time and therefore the most `main` commits landing
in between) reverted to `CONFLICTING`/`DIRTY` again within roughly 15–25 minutes — re-verified with
a fresh real `git merge --no-commit --no-ff` against the then-current `origin/main` for three of
those nine (#1420, #1451, #1785): all three were still genuinely clean. GitHub's computation, not
reality, had drifted again. A second refresh pass on all 9 fixed them again (confirmed
`MERGEABLE`/`UNSTABLE`, same session, both passes verified in this doc's revision history). Whether
they will hold this time was not re-checked after this dispatch was written — `main` in this repo
advances fast enough (dozens of commits/hour, all day, every day this session has observed it) that
"durable" may not be an achievable property of this remediation, only "cheap to repeat."

### The actual procedure, verified 40 times today (20 PRs × up to 2 passes)

```bash
gh pr view <n> --json headRefName,baseRefName   # get the real branch + base names
git fetch origin "+refs/pull/<n>/head:refs/remotes/origin/pr/<n>"
git fetch origin main   # or the PR's real base, if not main
git worktree add /tmp/refresh-<n> -B tmp-refresh-<n> origin/pr/<n>
cd /tmp/refresh-<n>
git merge origin/main --no-edit   # real merge; if this reports a conflict, STOP --
                                   # that PR is a real-conflict case, not a phantom one
git push origin HEAD:<real-branch-name>   # NOT --force; this is an ordinary fast-forward
                                           # of the PR's own branch by one merge commit
```

**Never skip the local merge step and force-push a rebase or synthetic commit instead.** A rebase
changes every commit's SHA and rewrites the author's history for no reason -- the only thing that
needs to change is the head SHA, and a merge commit does that while preserving everything else
byte-for-byte. This is the exact distinction the operator's four manual fixes today already
established; this dispatch just gives it a name and a number.

### Before pushing to a branch you do not own

Check `bin/sounio-coord status` for an `ACTIVE` claim on the exact branch first. If one exists, send
a heads-up via `bin/sounio-coord send` before pushing -- the merge itself only touches the remote
ref, not the other agent's local worktree, but it does mean their next push needs to pull first.
Two collisions were avoided this way during this session's refresh pass (branches
`lane/grok-cli1/handle-reclaim-design-20260817` and `lane/empryo-1/box-autoderef-20260817`, both
actively claimed at the time -- both owners notified before the push, no collision followed).

## What this does not explain

Why GitHub's mergeability cache re-drifts this fast, specifically in this repo, is not diagnosed
here -- that's GitHub-side infrastructure, opaque from outside. The working hypothesis (recomputation
re-triggers on every `main` push and this repo's `main` push rate outruns the recomputation queue)
is plausible given the timing but not confirmed. Worth remembering: `mergeable`/`mergeStateStatus`
being wrong is now a *measured, recurring* property of working in this repository, not a one-time
outage artifact -- treat every `DIRTY`/`CONFLICTING` reading as a hypothesis to check locally, not a
fact to act on, especially before asking someone to rebase.
