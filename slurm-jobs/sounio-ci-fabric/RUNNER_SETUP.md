# Sounio CI Fabric — first runner pool setup (NOT YET DEPLOYED)

## Read this first: t560 is the control-plane node

`cpuops-t560-proxmox` (t560) is simultaneously a Slurm CPU tier node **and**
the Kubernetes control-plane node for this cluster (etcd + kube-apiserver).
`scripts/dev/souc-build-remote.sh` refuses to schedule heavy Slurm builds
there by name-matching `*t560*`; nothing analogous exists for ordinary
Kubernetes pod scheduling, because a workload's own manifest cannot express
"never land on the control plane" without an explicit node affinity rule
pinned by whoever deploys it.

There is also circumstantial evidence (matching core counts: t560 is a
64-core node, and the interactive workspace pod's own CPU-saturation eviction
on 2026-05-29 peaked at a 15-minute load average of ~153 on a 64-core host)
that the interactive dev pod already shares a node with t560. See CLAUDE.md's
"Concurrency discipline" section.

**Do not `kubectl apply -f runner/ci-fabric-deployment.yaml` until you have:**

1. Confirmed with whoever administers this cluster which node(s) the
   `slurm-pilot` namespace's pods actually land on today, and whether that
   includes t560.
2. Filled in the commented-out `nodeAffinity` block in
   `runner/ci-fabric-deployment.yaml` (or an equivalent taint/toleration
   scheme) to keep this runner off that node.
3. Confirmed with them that a long-lived CI runner pod is an acceptable
   resident of the target node at all, given its CPU/memory request
   (`4 CPU / 8 GiB` request, `8 CPU / 16 GiB` limit in the current manifest --
   adjust to whatever the target node can actually spare).

This is the same discipline `scripts/dev/souc-build-remote.sh` already applies
to Slurm submissions; this runner pool is the Kubernetes-pod equivalent of
that same hazard.

## What this is

The first of the runner pools described in the Sounio CI Fabric proposal
(`ci-fast-x64` / `ci-build-x64` / `ci-test-x64` / `ci-science-x64` /
`ci-arm64` / `ci-gpu`). Phase 1 of that proposal ships exactly **one**
representative pool as groundwork -- RBAC, a Deployment manifest, and a
permanently-dormant smoke workflow -- not a live deployment and not the full
six-pool topology. Standing up additional pools, wiring GitHub's Actions
Runner Controller (ARC) instead of this plain-Deployment shape, and building
the Slurm-backed Qualification plane are explicitly out of scope for what a
coding session can author and verify without cluster-admin or GitHub-org-admin
access -- see the CI Fabric Phase 1 plan for the full list of what is
deliberately excluded and why.

## Why a plain Deployment, not ARC

No GitHub Actions Runner Controller (ARC) setup exists anywhere in this repo.
The one proven, working self-hosted-runner pattern already in the tree is
`slurm-jobs/kaxi-ptxas-accept/`, which uses the `myoung34/github-runner`
community image as a bare Kubernetes `Deployment` -- not ephemeral
per-job, not autoscaled, no runner-scale-set. This pool mirrors that exact,
already-proven shape rather than introducing ARC untested. Migrating to ARC
(ephemeral, autoscaled runners; the shape GitHub itself recommends for
fork-exposed or scale-sensitive repos) is real future work, not something
this groundwork should attempt speculatively.

## Fork-PR isolation, from day one

`.github/workflows/ci-fabric-pool-smoke.yml` (the only workflow that would use
this pool today) is `workflow_dispatch`-only -- it carries no `pull_request`
trigger at all, so `scripts/dev/check_self_hosted_runner_fork_exposure.sh`
(see `docs/ops/fork_pr_self_hosted_runner_policy.md`) is satisfied trivially,
not by a guard that a later edit could accidentally remove. Any future
workflow that runs pool jobs on `pull_request` must add the same-repo-origin
guard that gate requires, or the gate refuses the build.

## Setup steps (once placement is confirmed safe)

1. Apply the RBAC:
   ```bash
   kubectl apply -f slurm-jobs/sounio-ci-fabric/runner/ci-fabric-rbac.yaml
   ```
2. Create a GitHub PAT scoped to manage self-hosted runners on this repo, and
   store it as a namespaced secret (never as a GitHub Actions secret -- it
   never needs to leave the cluster):
   ```bash
   kubectl -n slurm-pilot create secret generic ci-fabric-runner-gh \
     --from-literal=ACCESS_TOKEN='<github-pat>'
   ```
3. Build and push a runner image. The kaxi lane's
   `slurm-jobs/kaxi-ptxas-accept/runner/Dockerfile.kaxi-runner` is a usable
   starting point; this pool does not need its `kubectl`/`build-essential`
   additions unless a future job submits Slurm work from it (see
   `runner/ci-fabric-rbac.yaml`'s own note on why it ships as a bare
   ServiceAccount with no Role/RoleBinding, and no API access at all, until
   that need is real). Tag and reference it in
   `runner/ci-fabric-deployment.yaml`'s `image:` field.
4. Fill in the `nodeAffinity` block per the warning above.
5. Apply the Deployment:
   ```bash
   kubectl apply -f slurm-jobs/sounio-ci-fabric/runner/ci-fabric-deployment.yaml
   kubectl -n slurm-pilot logs deploy/ci-fabric-runner -f   # watch it register
   ```
6. Set the repo variable that makes the dormant smoke workflow live:
   ```
   SOUNIO_ENABLE_CI_FABRIC_POOL=1
   SOUNIO_CI_FABRIC_RUNNER_LABELS=ci-fabric
   ```
7. Run `.github/workflows/ci-fabric-pool-smoke.yml` via `workflow_dispatch` and
   confirm it lands on the new runner.

## Teardown

Unlike the kaxi lane, this pool's ServiceAccount carries no Role/RoleBinding
(see `runner/ci-fabric-rbac.yaml`'s own note on why pods/exec is not granted
pre-emptively) -- so anyone who can dispatch workflows against this repo
gains repo-scoped Actions access to the runner, but not a bridge into the
cluster through it. Treat the PAT as sensitive regardless. Revoke it and
delete the Deployment/ServiceAccount together; do not leave either applied
with nothing using it. If a later change adds a Role/RoleBinding for a
Slurm-submitting job, this note's blast-radius statement no longer holds --
update it then.
