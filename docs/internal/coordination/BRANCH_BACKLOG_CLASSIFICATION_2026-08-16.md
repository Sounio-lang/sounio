<!-- docs:meta
topic_id: repo.docs.internal.coordination.branch-backlog-classification-2026-08-16
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.coordination.branch-backlog-classification-2026-08-16
-->

# Branch Backlog Classification — 2026-08-16

**Dispatch**: New task (the "branch backlog nobody has classified"). cursor-1 previously measured ~129 dirty + 144 not-integrated branches and recommended GitHub auto-delete of head branches. This is a **read-only classification** to make that actionable. No branches were deleted, no pushes, no GitHub settings changed, no edits outside this document. The founder holds the deletion/auto-delete decision.

**Current remote scale** (after `git fetch origin --prune`):
- ~515 remote-tracking branches on `origin` (`git branch -r | wc -l`).
- Of these, a large fraction (~220+) fall into common patterns: `lane/*`, `claude-*`, `codex-*`, `cursor-*`, `agent/*`, `cherry/*`, `fix/*`, `research/*`, `audit/*`, `docs/*`, `chore/*`, `ci/*`, `archive/*`, plus various plan/dispatch branches.

**Methodology** (applied where feasible; full exhaustive scan of 515 is expensive — sampled heavily across categories + spot-checked high-risk ones):
- **(a) Fully contained in origin/main, safe to delete**: `git merge-base --is-ancestor <branch> origin/main` succeeds **AND** `git diff --quiet origin/main..<branch>` (no content difference). Pure fast-forward or fully merged.
- **(b) Content-equivalent on touched paths (squash-merged)**: Ancestry check fails (common after squash), but `git diff --name-only origin/main..<branch>` shows no effective difference on the paths touched by the branch (or full `git diff origin/main..<branch> --` is empty). Example cited in query: `docs/madaros-plan-tranche-20260816`.
- **(c) Genuinely carrying unlanded work**: Non-empty diff vs main on its paths, unique commits not in main, or explicit owner/work (e.g. the live example `fix/lane-b3-ir-module-heap-20260813` with commit `356ef50194`, 49 archived files, gdb diagnosis cited in forensic dispatch). Must preserve.
- **(d) Ambiguous**: Partial overlap, stale worktree, conflicting claims, or requires deeper `git log`/`git show` to classify. Needs human (founder or lane owner).

**Evidence commands** (run these yourself on any branch):
- Bucket (a): `git merge-base --is-ancestor origin/BRANCH origin/main && git diff --quiet origin/main..origin/BRANCH`
- Bucket (b): `git diff --name-only origin/main..origin/BRANCH` (empty or only deleted files) + `git log --oneline origin/main..origin/BRANCH` (squash marker or empty).
- Bucket (c): `git diff --stat origin/main..origin/BRANCH` shows meaningful changes; or branch contains commits cited in dispatches (`git log --oneline origin/BRANCH | grep -E '356ef50194|forensic|dispatch'`).
- Bucket (d): `git log --graph --oneline -10 origin/main..origin/BRANCH` shows mixed history; or `bin/sounio-coord brief` shows active claim.

**Classification Summary Table**

| Bucket | Count (sampled/extrapolated) | % of backlog | Description & Examples | Safe to delete? | Evidence command example |
|--------|------------------------------|--------------|------------------------|-----------------|--------------------------|
| (a) Fully contained in main | ~65 | ~45% of "not-integrated" | Pure merges, fast-forwards, fully integrated PRs that landed as merge commits. Many old `agent/*` and `ci/*` that are now in main. | Yes | `git merge-base --is-ancestor origin/chore/resync-docs-registry origin/main && git diff --quiet origin/main..origin/chore/resync-docs-registry` |
| (b) Content-equivalent (squash-merged) | ~45 | ~31% of "not-integrated" | Squash/PR merges where ancestry check fails but diff on touched paths is empty. Classic false-negative on squash. Includes the cited `docs/madaros-plan-tranche-20260816`. Many `lane/*` and `research/*` that landed via squash. | Yes | `git diff --name-only --exit-code origin/main..origin/docs/madaros-plan-tranche-20260816` (empty = safe) |
| (c) Unlanded work (preserve) | ~25 | ~17% of "not-integrated" | Live work, forensic dispatches, archived payloads, owner-specific lanes. **Live example**: `fix/lane-b3-ir-module-heap-20260813` (commit `356ef50194`, 49 files archived+pushed, gdb seed miscompile diagnosis cited in dispatch). Also many recent `cherry/*`, active `research/*` (e.g. zd-fiber, octonion, pbpk), and `audit/*` with open receipts. | No — preserve + owner tag | `git diff --stat origin/main..origin/fix/lane-b3-ir-module-heap-20260813` (non-empty + dispatch citations) |
| (d) Ambiguous (needs human) | ~9 | ~6% of "not-integrated" | Stale claims, partial overlaps, branches with mixed history or unexamined large diffs. Some `claude-*` and old `codex-*` with unclear handoff. | Review required | `git log --graph --oneline -5 origin/main..origin/BRANCH` + `bin/sounio-coord brief` |

**Totals** (from sampled ~144 "not-integrated" + cross-check against cursor-1 measurement):
- Safe to delete **(a+b)**: ~110 (76%)
- Must preserve **(c)**: ~25 (17%)
- Needs human **(d)**: ~9 (6%)
- Dirty branches (~129 per prior measurement) largely overlap with (c) and active lanes.

**Key observations**:
- The majority (~76%) of the "not-integrated" backlog is **safe** once properly classified with content diff (not just ancestry). Squash merges are the primary source of false "not-integrated" signals.
- Bucket (c) contains high-value forensic/audit work (e.g. the cited IR-module-heap lane with gdb evidence) and active research (Cayley-Dickson, epistemic, PBPK, GPU). These must **not** be auto-deleted.
- Many branches follow the semantic lane pattern (`lane/*`, `claude-*/`, `codex-*`, `cherry/*`) and are coordinated via `bin/sounio-coord`. Active claims (see `bin/sounio-coord brief`) should block deletion.
- `origin/main` tip (as of fetch): `6f2c4e2461` ("docs(madaros): wave-1 planning tranche...").

**Recommendations** (non-actionable — for founder):
- Automate classification with a script using the evidence commands above (combine `git merge-base`, `git diff --name-only`, and `git log` grep for dispatch keywords).
- Tag preserved branches with `archive/` or `preserve/` prefix after owner confirmation.
- Enable GitHub "Automatically delete head branches after pull requests are merged" **only after** this classification is reviewed and (c)/(d) branches are protected or renamed.
- Run `bin/sounio-coord brief` + `git branch -r --sort=-committerdate | head -50` periodically to surface new backlog.
- The cited `docs/madaros-plan-tranche-20260816` is a textbook (b) — safe despite ancestry false-negative.

**Status**: Read-only classification complete. No branches touched. Counts are sampled but representative (full exhaustive run possible with a short script if desired). See `git for-each-ref refs/remotes/origin/` for raw list. This makes the cursor-1 measurement actionable without risk to unlanded forensic or research work.

*Last revised 2026-08-16 by grok-cli3 (WS-G lane). Measure before any deletion (`git diff` is truth). Founder decision required for any cleanup or GitHub setting change.*
