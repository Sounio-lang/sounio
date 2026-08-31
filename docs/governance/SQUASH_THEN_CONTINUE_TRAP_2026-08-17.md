<!-- docs:meta
topic_id: repo.docs.governance.squash-then-continue-trap-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: grok-cli2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.governance.squash-then-continue-trap-2026-08-17
-->

# Squash-then-continue: the conflict class a twenty-agent fleet invents

**Date:** 2026-08-17  
**Status:** diagnosis + proposed policy (founder sign-off required before changing
GitHub merge defaults or BRANCH_POLICY binding text)  
**Evidence anchors:** PR **#1759** resolution commit `5b0fbec1f8`; PR **#1754**
squash onto main as `453b2e6e2f` (single parent); repo settings
`allow_squash_merge=true`, `delete_branch_on_merge=false`; open PR count **54**
(measured 2026-08-17); fleet warning on `lane/*/20260814` branches **600+**
commits ahead of main.  
**Scope:** docs only this tranche. Does not flip GitHub settings.

---

## 0. Executive claim

Most “DIRTY” PRs in this fleet are not random merge unluckiness. They are
**topology products**:

1. **Squash-merge a multi-commit lane into `main`**, then  
2. **Keep committing on the same branch name** for the next stage, then  
3. **Merge or rebase that branch onto a main that now contains a squash** of
   paths the branch still “adds.”

Git then reports **add/add** on files that are *the same work twice under
unrelated histories*. A human who only sees the PR UI thinks content diverged.
It did not: **the pair of parents that would have made the histories identical
was deleted by the squash.**

That class is **structural**. Changing merge discipline removes it; better
conflict resolution skill does not.

A second, coarser class is the **600+ commit checkpoint branch** (`lane/*/20260814`
and cousins such as long-lived research tips). That class produces megabyte PRs
and vacuous CI, not only add/add. It is related (never cut a fresh tip from
`origin/main`) but not identical to squash-then-continue.

---

## 1. What a squash does to history (the unpairable commit)

A normal merge of branch `L` into `main` creates a commit with **two parents**:

```text
main:   … — M0 — M1 — M2
                    \
L:                   L1 — L2 — L3
                         ↘
main':  … — M0 — M1 — M2 — M_merge(L3, M2)
```

When Git later merges `L` again (after more commits `L4…`), it can find a
**merge-base** that is an ancestor of both sides. Shared history pairs renames,
additions, and deletions.

A **squash merge** of the same `L` into `main` creates a commit with **one
parent** — a new tree, no parent edge to `L3`:

```text
main:   … — M0 — M1 — M2 — S(#1754)     ← S has parent M2 only
L:                   L1 — L2 — L3 — L4 — L5
                     ^^^^^^^^^^^^^^^^
                     still “adds” ir.sio etc. from L1
```

On disk after squash, `main` contains `self-hosted/mli/ir.sio` (etc.) as if
**someone new added those paths on main**. Branch `L` also contains those paths
as **first-parent history from L1**. There is **no common ancestor commit that
introduced the file on both sides**. Three-way merge therefore classifies the
file as:

```text
add/add  — both sides create the path; no base blob to diff against
```

even when the blobs are **byte-identical** (as proven for #1759 by sha256 in
`5b0fbec1f8`).

That is the trap: **squash erases the identity of the landing commits; the
lane branch still believes it is the author of the paths.**

---

## 2. Measured instance — #1754 → #1759

| Event | What happened |
|---|---|
| MLI **S1** | Developed on `lane/cursor-2/mli-s1-20260816` |
| **#1754** lands on main | **Squash** → `453b2e6e2f` (single parent `8999e0fdff…`) |
| Same branch continues | S2a, S2b, S3 commits still parented on pre-squash lane history |
| **#1759** opens / updates | GitHub: DIRTY; **add/add** on `ir.sio`, `dump.sio`, `verify.sio`, dispatch doc |
| CI | Conflicting PR: **no full check set** (vacuous “green” risk — few checks run) |
| Resolution | `5b0fbec1f8` merges main into the lane; **ours** for the four files after proving main’s blobs equal the branch’s pre-evolution tips |

Quote from that resolution (primary source, not reconstruction):

> Root cause: MLI S1 landed on main as a SQUASH (#1754 → 453b2e6e2f), so main
> carries ir.sio, dump.sio, verify.sio and the nested-aggregate dispatch doc as
> **independent additions of paths this branch also creates** → four add/add
> conflicts, and GitHub runs NO CI on a conflicting PR …

Also measured on `origin/main` (2026-08-17): recent PR landings are overwhelmingly
**single-parent** commits whose subjects end in `(#NNNN)` — i.e. default
**squash** style — while `delete_branch_on_merge` is **false**, so the remote
lane tip is **encouraged to survive** and accept more commits.

That combination is the factory.

---

## 3. Why a twenty-agent fleet makes this dominant

| Fleet fact | Effect on topology |
|---|---|
| **~20 agents**, many with long-lived `lane/<agent>/…` names | Branches outlive a single PR; “stage 2 of the same epic” reuses the name |
| **Squash is allowed and habitual** | Every multi-commit stage lands as a new orphan tree on main |
| **`delete_branch_on_merge = false`** | Remote tip remains; natural to `git push` more commits |
| **Checkpoint branches 600+ ahead of main** | Entire pre-isolation histories share the same name as “the lane”; PRs drag `.beagle/`, `.claude/`, unrelated science |
| **Governance-regen trio** | Separate class: *expected* textual conflict; driver not available on GitHub — **must not** be counted as this trap (see BRANCH_POLICY / `.gitattributes`) |
| **Coordination bus ≠ git history** | Claims prevent write overlap; they do **not** repair unpairable parents after squash |

No single agent is “bad at git.” The **defaults produce add/add** whenever a staged epic lands mid-flight.

Ordinary open-source (one PR branch, squash, **delete branch**, next PR from fresh `main`) rarely sees this class. The unusual configuration is **long-lived multi-stage branches under squash**.

---

## 4. Conflict classes (do not collapse them)

| Class | Symptom | Cause | Counts as “real”? |
|---|---|---|---|
| **A. Squash-then-continue** | add/add on files the lane invented; often byte-identical | Squash on main + continue same first-parent chain | **Yes — workflow defect** |
| **B. True concurrent edit** | content conflict on lines both sides changed | Two landings touched same region after a shared base | **Yes — need human** |
| **C. Governance-regen** | conflict on `topic-registry` / acceptance / authority matrix | Derived files; `merge=governance-regen` | **No — regenerate** |
| **D. Append-only log** | conflict on `.claude/llm_offload_log.md` | Parallel audit rows | **Noise — concat** |
| **E. Stale checkpoint PR** | PR is hundreds of commits / millions of lines | Branch never recut from main | **Not a merge conflict — wrong base** |

Wave-3 DIRTY triage already saw C/D often. **#1759 is class A.** Cursor-1’s multi-megabyte PR is class **E**.

---

## 5. Minimal change that removes class A

Not “ban squash forever” (squash is fine for **throwaway** single-shot PRs).  
The minimal rule that kills the factory:

### Policy rule (proposed, one sentence)

> **After a squash lands on `main`, never put another commit on that same branch tip. Cut a new branch from `origin/main` and cherry-pick only commits that are not already in the squash.**

Equivalently:

| After merge | Allowed | Forbidden |
|---|---|---|
| Squash of PR *N* from branch `B` | New branch `B-stage-k` from `origin/main`; cherry-pick only post-N work | `git commit` / `git push` on `B` |
| Merge commit (two parents) of `B` | Continue on `B` **or** recut — history still pairs | — |
| Single-shot fix PR | Squash + **delete remote branch** | Reuse deleted name for unrelated work without recutting from main |

### Why this is minimal

- Does **not** require turning off squash repo-wide (single-shot PRs stay tidy).  
- Does **not** require merge commits for every PR (optional upgrade).  
- **Does** force a **new first-parent chain** after history identity was destroyed — which is exactly what three-way merge needs.  
- Matches the fleet warning already issued for 600+ checkpoint branches: **cherry-pick onto fresh main, do not rebase the hundreds.**

### Optional accelerators (not required for class A death)

1. Enable **delete branch on merge** for short-lived PR branches (not for protected long names without training).  
2. Prefer **merge commits** for multi-stage epics (MLI S1/S2/S3, WS-C PR1/PR2) so continue-on-branch remains safe.  
3. PR template checkbox: “This branch previously squash-landed: [ ] recut from main.”  
4. CI bot: if PR head is >N commits ahead **and** shares path-adds with a recent squash author set, fail with “squash-then-continue suspected.”

---

## 6. Concrete workflow (agent / human checklist)

### Single-shot change (one PR, done)

```text
1. branch from origin/main
2. commit, open PR
3. squash-merge OK
4. delete remote branch (or abandon the name)
5. next task → new branch from origin/main
```

### Multi-stage epic (S1 then S2 on same *topic*)

```text
1. branch epic/s1 from origin/main → PR → land (squash OK)
2. STOP. Do not commit on epic/s1 again.
3. git fetch origin main
4. git switch -c epic/s2 origin/main
5. cherry-pick only commits that implement S2 (not S1)
6. open new PR from epic/s2
```

### If you already squash-continued (recovery)

```text
# On the polluted branch tip:
git fetch origin main
git log --oneline origin/main..HEAD   # identify commits unique to post-squash work
git switch -c epic/s2-rescue origin/main
git cherry-pick <only those SHAs>
# resolve any true content conflicts; add/add on S1 files should vanish
# open PR from epic/s2-rescue; abandon the old tip for landing
```

### Stale checkpoint (`lane/*/20260814`, 600+ ahead)

```text
# NOT: rebase onto main
# YES:
git fetch origin main   # pin e.g. c66014fda9 / current main
git switch -c lane/<id>/topic-YYYYMMDD origin/main
git cherry-pick <only your commits>
# verify: git log --oneline origin/main..HEAD | wc -l   → small integer
```

---

## 7. What not to do

| Anti-pattern | Why it fails |
|---|---|
| “Always merge main into the lane” after every squash | Still pays add/add tax every stage; wastes CI and human hours |
| Rebase 600 checkpoint commits onto main | Replays foreign history; produces false PR size |
| Treat add/add as proof the other side rewrote your files | Often byte-identical; measure sha256 before editing |
| Count governance-regen conflicts as squash-trap | Different class; regenerate |
| Open PR from `lane/*/20260814` “to save work” | Class E disaster (millions of lines, empty CI) |

---

## 8. Proposed amendments (for founder sign-off)

1. **BRANCH_POLICY.md** — add section “Squash-then-continue is forbidden for multi-stage branches” with the one-sentence rule in §5 and the recovery recipe in §6.  
2. **GitHub** — keep squash for single-shot; for epic labels, prefer merge commits **or** enforce recut via review checklist.  
3. **delete_branch_on_merge** — enable for default PR branches; document exception for shared long-lived names that must be retired by policy instead.  
4. **Fleet onboarding** — one diagram (this doc §1) in AGENT_HANDOFF / attention brief so agents stop “fixing” class A by hand.

Until sign-off, agents should still **follow §5–§6 as practice**: the diagnosis is measured; the settings flip is political.

---

## 9. Relation to founder “beyond SOTA” calibration

Squash-then-continue is not a language feature — it is **epistemic hygiene of the fleet**. A project that can refuse fabricated zeros (E219) should not **fabricate a green PR** under vacuous CI caused by structural DIRTY state. Cleaning the topology is how multi-stage firsts (MLI S3 kind survival, f256 limbs, CD kinds) land without drowning in add/add theatre.

---

## 10. Receipts

```bash
# #1754 squash = single parent
git log -1 --format='%H parents=%P %s' 453b2e6e2f

# #1759 landing also single-parent squash style
git log -1 --format='%H parents=%P %s' b7949edcda

# primary write-up of add/add root cause
git show 5b0fbec1f8 --format=fuller | head -60

# repo merge knobs
gh api repos/Sounio-lang/sounio --jq '{squash:.allow_squash_merge, merge:.allow_merge_commit, rebase:.allow_rebase_merge, delete_branch:.delete_branch_on_merge}'
```

---

## 11. Second demonstration — main velocity turns UNKNOWN → DIRTY (#1770)

While this diagnosis was open, **#1770** (`minimax-cli2/audit-gates-2026-08-17`,
WS-A status refresh cherry-picked from a stale lane) went from **UNKNOWN** to
**DIRTY/CONFLICTING** within about an hour because **main moved multiple times**
under the open PR. That is a *second* daily factory:

| Mechanism | What the UI shows | Fix |
|---|---|---|
| **A. Squash-then-continue** (#1759) | add/add on paths the lane invented | Recut after every squash (§5) |
| **F. Main thrash under open PR** (#1770) | DIRTY after unrelated landings | Rebase/recut onto fresh `origin/main` before merge; do not leave PRs open across a multi-landing hour without a recut |

Both are topology, not content quality. A twenty-agent repo that squash-lands
constantly will **convert every long-lived tip into DIRTY** unless agents recut.

---

## 12. Paste-ready policy block (for `AGENTS.md` or parallel-blocker contract)

Copy below the horizontal rule into the active agent contract when founder
approves.

---

### Squash-then-continue (binding practice)

**Problem.** Squash-merge rewrites a multi-commit branch as a **single-parent**
commit on `main`. That commit *adds* paths with no parent edge back to the
lane. If the same branch tip keeps committing (S2 after S1, stage *k* after
stage *k−1*), Git later classifies those paths as **add/add** even when the
blobs are byte-identical. Measured: #1754 (MLI S1 squash) → continue on
`lane/cursor-2/mli-s1-20260816` → #1759 add/add on `ir.sio`/`dump.sio`/`verify.sio`
(resolution `5b0fbec1f8`, sha256-proved identical). GitHub runs **almost no CI**
on CONFLICTING PRs — vacuous green is a lie.

**Rule (one sentence).** After a squash lands on `main`, **never put another
commit on that branch tip.** Cut a new branch from `origin/main` and
**cherry-pick only commits not already in the squash.**

**Also forbidden.** Opening a PR from a tip that is **hundreds of commits**
ahead of `origin/main` (`git log --oneline origin/main..HEAD | wc -l` in the
hundreds). Those are checkpoint branches, not landing branches — cherry-pick
your own commits onto a fresh tip.

**Not this rule.** Conflicts only on
`docs/governance/{DOCS_ACCEPTANCE_REPORT,DOCS_AUTHORITY_MATRIX,topic-registry.v1.json}`
are **governance-regen** (derived files; driver not available on GitHub) —
regenerate, do not treat as content wars.

**Recovery.**

```bash
git fetch origin main
git switch -c <topic>-stage-N origin/main
git cherry-pick <only post-squash SHAs>
# open PR from <topic>-stage-N; abandon the old tip for landing
```

**Full diagnosis:** `docs/governance/SQUASH_THEN_CONTINUE_TRAP_2026-08-17.md`

---

## Document control

| Date | Change |
|---|---|
| 2026-08-17 | Diagnosis of squash-then-continue; #1759 evidence; proposed one-sentence policy + recovery. Docs only. |
| 2026-08-17 | §11 #1770 UNKNOWN→DIRTY under main thrash; §12 paste-ready AGENTS/blocker block. |
