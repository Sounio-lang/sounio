# CPU36 source qualification awaiting job assignment

Blocker-ID: BLK-20260910-pireus-cpu36-admission
Status: owned
Severity: B1
Class: evidence-gap
Owner: codex-pireus
Lane: continuity-20260906
Worktree: /workspace/.wt/pireus-integration-20260906
Branch: codex/pireus-inkling-cycle-20260906
Files-Owned: tools/pireus/continuity/**
Do-Not-Touch: shared /workspace/sounio; frozen execution-v1 resources and permissions
Repro: gh api repos/Sounio-lang/sounio/actions/jobs/102919731180
Observed: At 2026-09-10T15:10:26Z, job queued with runner_id=0; group 3 runner 6140 online and idle.
Expected: Assignment of the existing frozen job and complete source qualification receipts.
Acceptance-Gate: Run 34491511493 Current-Source on runner 6140, all required gates and CI Decision success for e91a7a17ea47fc3df5fbd7e9eea0e27e85ffeda7; source custody checks separately required.
Evidence-Level: E4
Evidence: https://github.com/Sounio-lang/sounio/actions/runs/34491511493/job/102919731180
Fallback-Path: none
Legacy-Kept: yes
LLM-Offload: not-required
Next-Action: Inspect the existing job and admission-watch.jsonl; on a terminal result preserve logs and artifacts before cleaning up owned one-shot resources.

## Scope and continuation

This is a missing qualification receipt, not a confirmed GitHub outage or an
identified routing defect. GitHub's public status reported Actions operational.
Repository Actions are enabled, pending_deployments is empty, and group access
and runner labels match the declared configuration. These observations do not
prove successful scheduler admission.

The same queue dependency persisted across successive goal turns. The source
coverage audit is complete: the nine journal orchestration blobs match e91,
whereas the old source 342 lacks the adapter. Neither result replaces source CI.
All further source-dependent Inkling submission remains pending.

The existing Kubernetes Job and read-only tmux observer remain active. Pausing
the agent goal does not cancel, restart, or extend either process. The observer
records once per minute until completion or 2026-09-10T17:46:15Z and does not
automatically resume the agent. Its private directory is:
/workspace/.cache/pireus-continuity/ci-runner-execute-v1-20260910

The remaining lifetime only accommodates the complete 150-minute job budget
until about 15:16 UTC. A later assignment must preserve its reduced headroom
in the receipt; a queue delay is not a compiler failure and must not trigger
an unrecorded lifetime extension or another dispatch.

Do not push this branch while the frozen run remains active: ci.yml groups
concurrency by branch with cancel-in-progress enabled. Local remote-worktree
commits ac92562aac, deeed87810 and ed0fef1219 are retained. Published source is
still e91a7a17ea47fc3df5fbd7e9eea0e27e85ffeda7.

A successful CPU36 source result still requires a newly source-bound journal
protocol and fresh exclusive Spark preflight before one diagnostic attempt.
The 33 GiB early stop, 32 GiB floor, no automatic model retry, and all broader
M0-M6 acceptance obligations remain unchanged.
