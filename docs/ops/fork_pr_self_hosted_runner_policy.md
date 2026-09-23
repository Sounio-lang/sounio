<!-- docs:meta
topic_id: repo.docs.ops.fork-pr-self-hosted-runner-policy
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.fork-pr-self-hosted-runner-policy
-->

# Fork-PR self-hosted runner policy

## The rule

A workflow job whose `runs-on:` names a self-hosted runner (the literal
`self-hosted` label, or a `${{ vars.*RUNNER* }}` pool-selector expression such
as the `kaxi-ptxas-accept` / `ci-fabric-pool-smoke` pattern) must never be
reachable, unguarded, from a bare `pull_request:` trigger.

`pull_request` runs against the **merge** of a fork's proposed changes. A
self-hosted runner executing that code has whatever access this repository's
environment carries — for Sounio, potentially a path into the project's own
Kubernetes/Slurm cluster (see `.claude/skills/cluster-gpu-jobs/SKILL.md` and
`slurm-jobs/`). A long-lived, non-ephemeral runner Deployment (the shape every
self-hosted runner in this repo currently is — see
`slurm-jobs/kaxi-ptxas-accept/RUNNER_SETUP.md`) makes this worse: a compromised
job's effects can persist into the next job that lands on the same pod.

The required guard on the job's own `if:`:

```yaml
if: (github.event_name != 'pull_request' || github.event.pull_request.head.repo.full_name == github.repository)
```

combined with whatever else already gates the job (an on/off repo variable,
an impact classification, etc. — the guard is `&&`-ed in alongside those, not
a replacement for them).

## Enforcement

`scripts/dev/check_self_hosted_runner_fork_exposure.sh`, wired into `ci.yml`'s
`contracts` job, scans every workflow for this shape and fails the build if it
finds one unguarded. Run it locally with `--selftest` to see its own
positive/negative controls.

It is a text scan of the workflow YAML (the same idiom
`scripts/ci/impact_ci_selftest.sh` already uses to check `ci-decision`'s needs
list against `evaluate_ci_decision.py`), not a full YAML parse — deliberately
matching this repo's existing convention (see
`scripts/ci/gate_workflow_reference_ratchet.sh`) rather than adding a new
YAML-parsing dependency to CI.

## Why this gate exists

No such guard existed anywhere in the repo before it was added. A live
instance was found and fixed in the same change that added the gate:
`.github/workflows/gpu-research.yml`'s `validate-on-gpu` job ran
`runs-on: [self-hosted, gpu, cuda]` under a bare `pull_request:` trigger,
gated only by an on/off repo variable (`vars.SOUNIO_ENABLE_GPU_SELF_HOSTED`)
— no same-repo-origin check at all. Because that job's own gate (`--selftest`)
proved the detector correctly ignores it once the guard is added, the fix
landed in the same PR as the gate: the gate is green from its first commit,
not merged red and fixed later.

## What is out of scope

`pull_request_target` is a **separate, out-of-scope** risk class from what
this gate checks. It runs with base-repo context (secrets, write permissions)
against a fork's code by design — a different and, in most configurations, a
*worse* trust boundary than a self-hosted runner under plain `pull_request`.
The only `pull_request_target` workflow in this repo today,
`.github/workflows/issue-pr-automation.yml`, runs on a GH-hosted
`ubuntu-24.04` runner, not self-hosted infrastructure, which is why it is a
known, separate, lower-priority gap rather than something this gate flags.
Auditing `pull_request_target` isolation is future work, not covered here.

## Future self-hosted pools

`slurm-jobs/sounio-ci-fabric/` (groundwork only — not deployed; see its own
`RUNNER_SETUP.md`) is built against this policy from day one: its smoke
workflow is `workflow_dispatch`-only and carries no `pull_request` trigger at
all, so this gate's check is satisfied trivially rather than by a guard that
could later be edited away. Any future workflow that *does* want to run
self-hosted jobs on `pull_request` (e.g. a trusted-PR fast lane) must add the
guard above and will be caught by this gate if it does not.
