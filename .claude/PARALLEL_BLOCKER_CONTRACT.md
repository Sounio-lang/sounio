# Parallel Blocker Contract

Status: active
Authority: required for Claude Code, Codex, Cursor agents, subagents, and LLM-offload reviewers when parallel work touches the Sounio repository.
Canonical path: `.claude/PARALLEL_BLOCKER_CONTRACT.md`

This contract defines what an agent may call a blocker, how blocker ownership is assigned, and what evidence is required before work can merge or be handed off.

## Core Rule

A blocker is not a mood, an intuition, or a broad TODO. A blocker is a typed claim with:

1. one owner
2. one blocked lane
3. one reproducible command or explicit missing resource
4. one acceptance condition
5. one evidence artifact or log path

If any of those five fields are missing, report it as an investigation item, not a blocker.

## Evidence Levels

Use the strongest level actually reached.

| Level | Name | Meaning |
|---|---|---|
| E0 | observation | Symptom seen once; not yet isolated. |
| E1 | reproduction | Command, input, branch, and worktree reproduce the symptom. |
| E2 | classified | Failure class is known and unrelated noise is separated. |
| E3 | gate-bound | A specific acceptance gate fails or passes. |
| E4 | remote-confirmed | CI, release gate, external host, GPU host, or offload review confirms the result. |

Do not close a serious blocker below E3 unless the closure reason is `invalid`, `duplicate`, `waived`, or `resource-unavailable`.

## Severity

| Severity | Name | Merge rule |
|---|---|---|
| B0 | stop-the-line | No commit, merge, push, or publication until resolved or explicitly waived by the human author. |
| B1 | lane-blocking | The owning lane cannot merge; unrelated lanes may continue if file ownership is disjoint. |
| B2 | integration-blocking | Local lane is green, but merge/sync/remote validation is not green. |
| B3 | evidence-blocking | Work may be correct, but proof requires unavailable device, credential, dataset, remote host, or external review. |
| B4 | non-blocking noise | Track separately; do not block merge. |

Default to the lower severity if the failure is not reproduced. Raise severity only after evidence improves.

## Failure Classes

Every blocker report must use one primary class.

| Class | Use for |
|---|---|
| `state-sync` | Branch drift, stale main, wrong workspace, dirty worktree, or remote mismatch. |
| `ownership-conflict` | Two agents need to edit the same file, especially serialized surfaces. |
| `gate-regression` | A previously green required gate now fails. |
| `harness-routing` | Wrong compiler, stale wrapper, wrong environment, wrong script, or path confusion. |
| `compiler-semantics` | Parser, checker, lowering, IR, ABI, optimizer, or codegen semantics are wrong. |
| `bootstrap-runtime` | Self-host, stage replay, executable artifact, runtime exit, or fixed-point failure. |
| `platform-resource` | Missing GPU, SSH host, OS surface, driver, cluster quota, or hardware resource. |
| `dependency-secret` | Missing API key, credential, package, network dependency, or external service. |
| `evidence-gap` | Claim may be true but lacks artifact, log, source citation, or acceptance gate. |
| `governance-offload` | LLM-offload, mandatory review, serialized window, or policy compliance failure. |
| `doc-claim` | Documentation overclaims, contradicts executable reality, or references stale paths. |
| `security-privacy` | Credential leak, destructive command risk, PHI, secret-bearing logs, or unsafe publication. |

Do not collapse classes. For example, `harness-routing` can mask `compiler-semantics`; classify routing first, then re-run the semantic gate.

## Required Blocker Record

Agents must use this shape in handoffs, issue comments, PR summaries, and final reports when a blocker remains.

```text
Blocker-ID: BLK-YYYYMMDD-<lane>-<slug>
Status: proposed | reproduced | classified | owned | fixing | review-ready | merge-ready | closed | waived
Severity: B0 | B1 | B2 | B3 | B4
Class: state-sync | ownership-conflict | gate-regression | harness-routing | compiler-semantics | bootstrap-runtime | platform-resource | dependency-secret | evidence-gap | governance-offload | doc-claim | security-privacy
Owner: <agent/person>
Lane: <feature/gate/research lane>
Worktree: <absolute path>
Branch: <branch or detached SHA>
Files-Owned: <exact write set>
Files-Read-Only: <optional read-only surfaces>
Do-Not-Touch: <serialized or high-risk files>
Repro: <single command, or explicit missing resource>
Observed: <actual result>
Expected: <expected result>
Acceptance-Gate: <command or remote workflow>
Evidence-Level: E0 | E1 | E2 | E3 | E4
Evidence: <log/artifact/CI URL/offload log path>
Fallback-Path: none | <name and why>
Legacy-Kept: yes | no | n/a
LLM-Offload: not-required | required-pending | logged:<path>
Next-Action: <one concrete action>
```

If the record cannot be filled, the agent owns an investigation task, not a blocker.

## Ownership Rules

1. One active writer per file.
2. One integration shepherd per merge to `main`.
3. High-risk shared files are serialized according to `AGENTS.md`.
4. `self-hosted/check/check.sio` uses its declared integration window.
5. Agents may inspect another lane's files but must not write them without an ownership transfer.
6. Ownership transfer requires a handoff with current branch, commit, dirty status, files owned, last green gate, and open blocker records.

If two agents need the same file, the second agent must either wait, narrow scope, or open an `ownership-conflict` blocker.

## Parallel Lane Contract

Before starting a parallel lane, declare:

```text
Lane:
Owner:
Base:
Worktree:
Branch:
Write-Set:
Read-Set:
Required-Gates:
Merge-Target:
Known-Blockers:
```

The write set must be disjoint from every other active lane unless the human author explicitly opens a serialized merge window.

## Merge Contract

A lane may merge only when all are true:

1. The worktree is clean except for the lane's intended diff.
2. `origin/main` has been fetched and the merge base is understood.
3. Every B0 and B1 blocker in the lane is closed, waived, or explicitly moved to a different owner.
4. Required local gates are green and named in the final report.
5. Required remote gates are green or the merge is explicitly marked local-only.
6. Any mandatory LLM-offload review is logged before commit.
7. Fallback paths are named; silent fallback is a merge blocker.
8. Legacy paths kept for safety are named; accidental deletion of unproven legacy paths is a merge blocker.

After merge to `main`, the integration shepherd must verify:

```bash
git fetch --prune
git status -sb
git rev-parse HEAD main origin/main
git branch --list
git worktree list
gh run list --branch main --limit 4 --json databaseId,workflowName,headSha,status,conclusion,event,createdAt,url
gh pr list --state open --limit 20 --json number,title,headRefName,baseRefName,isDraft,mergeable,url
```

Delete only branches that are already merged and owned by the current lane.

## Handoff Contract

When handing work to another agent, include:

```text
Current-SHA:
Current-Branch:
Current-Worktree:
Dirty-Status:
Owned-Files:
Do-Not-Touch:
Last-Green-Gates:
Failing-Gates:
Open-Blockers:
Artifacts:
Next-Command:
```

A handoff without `Next-Command` is not actionable.

## Claim Discipline

Agents must separate:

1. what passed locally
2. what passed remotely
3. what was not run
4. what used a fallback
5. what remains a narrow witness rather than a general feature

Do not turn a green witness into a broad language claim. Do not turn a red optional probe into a launch blocker.

## Mandatory Stop Conditions

Stop and escalate as B0 if any of these occur:

1. credential, API key, token, PHI, or private dataset exposure
2. force-push, hard reset, destructive clean, or branch deletion not clearly owned by the lane
3. external-facing math, clinical, or publication artifact without mandatory offload review
4. CI/release gate red on `main` after merge
5. two agents writing the same high-risk file
6. any uncertainty about whether the active workspace is `/workspace/sounio`

## Allowed Waivers

Only the human author can waive B0. An agent may waive B1-B3 only by recording:

1. why the blocker no longer blocks the current lane
2. where the residual risk moved
3. what command or artifact future agents should use to re-open it

Waivers are not deletion. They are traceable deferrals.
