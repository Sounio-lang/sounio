<!-- docs:meta
topic_id: repo.docs.ops.foundry-slurm-handoff
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.ops.foundry-slurm-handoff
-->

# Foundry/Slurm Handoff Pack

> **Status**: Beta | **Last validated**: 2026-05-21 | **Source**: `ONBOARDING.md`, `./sounio-whereami --quick`, `scripts/gpu/`

This pack is the handoff contract for heavy validation. It does not replace
`ONBOARDING.md`; it starts where onboarding stops.

Use this when a workspace/pod agent has a branch, commit, or gate that needs the
Sounio Compiler Foundry, Slurm, OrangeFS, GPU ownership, or long-running batch
execution. The workspace remains an interactive editing surface.

## First Orientation

In the workspace pod:

```bash
cd /workspace/sounio
./sounio-whereami --quick
git status --short --branch
git rev-parse --short HEAD
```

On the host/control plane, when that surface is actually available:

```bash
/home/devsounio/projects/sounio/sounio-whereami --quick
```

If `/home/devsounio` is absent, you are in the pod. Do not assume host paths,
OrangeFS mounts, or Slurm login visibility from the workspace filesystem.

## Slurm Visibility

From the workspace, Slurm may only be visible through the login pod:

```bash
kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- sinfo
kubectl -n slurm-pilot exec deploy/slurm-pilot-login-slinky -- squeue -a
```

A timeout or missing login pod from the workspace is an environment/wiring
observation, not proof that the cluster itself is down. Record it in the handoff
and ask a host/control-plane agent to run the gate.

## Local vs Heavy Work

Keep these local in `/workspace/sounio`:

- editing and review
- docs and small script changes
- `bin/souc info`
- targeted compiler checks
- small `scripts/run_sio_test_suite.sh` smoke runs

Hand these off to Foundry/Slurm:

- full compiler stress
- GPU campaigns
- Slurm jobs
- OrangeFS-backed runs
- long batch validation
- high-volume artifact generation

## Handoff Request Template

To generate the template from the current checkout without submitting a job:

```bash
bash scripts/dev/foundry_slurm_handoff.sh
```

Use `--out <path>` only for small handoff notes that belong in the repo or an
artifact packet. Do not use the workspace as scratch for batch output.

```text
Heavy Validation Handoff
requested_by:
source_surface: workspace pod | host/control-plane
repo_path_seen:
branch:
commit:
dirty_state:
gate_requested:
command_or_foundry_target:
reason:
gpu_requirement:
slurm_requirement:
expected_artifact_root:
acceptance_criterion:
known_blockers:
return_payload:
```

Fill the branch and commit from the checked-out worktree that produced the
request. Do not describe "latest" or "current" without the concrete SHA.

## Foundry Request Shape

If the host/control plane is available, request a Foundry submission by branch
or commit:

```bash
/home/devsounio/projects/sounio/sounio-forge submit full-compiler --source <branch-or-commit> --gpu auto
```

If the control plane is not available from your session, leave the filled
handoff for a host/control-plane agent. Do not invent Kubernetes or Slurm YAML.
For custom gates, name the checked-in script and the branch/commit to run
against; do not transcribe large workspace WIP into the request.

## Artifact Return Contract

The host/control-plane agent returns:

- artifact root
- branch and commit tested
- command or Foundry target used
- status summary
- first failing test and log path, when failed
- failure class: build/bootstrap-path, harness-routing, semantic/kernel, or infra/cluster
- acceptance result against the requested criterion

If the run is blocked before execution, return the same shape with the blocker
as `infra/cluster` or `harness-routing`, whichever is more precise.

## Hard No

- Do not run heavy stress in `/workspace/sounio`.
- Do not use `/workspace/sounio` as batch scratch space.
- Do not write manual Kubernetes or Slurm YAML for Foundry jobs.
- Do not rely on stale VM-era paths.
- Do not switch, reset, clean, or rebase the active branch just to make a handoff easier.
- Do not stage unrelated workspace WIP into the validation request.

## Example

```text
Heavy Validation Handoff
requested_by: codex
source_surface: workspace pod
repo_path_seen: /workspace/sounio
branch: codex/example-heavy-gate
commit: 1234abc
dirty_state: clean
gate_requested: full compiler stress
command_or_foundry_target: sounio-forge submit full-compiler --source 1234abc --gpu auto
reason: confirm bootstrap/runtime behavior after compiler-path patch
gpu_requirement: auto, GPU acceptable but not required unless Foundry selects it
slurm_requirement: yes
expected_artifact_root: artifacts/foundry/<run-id>/
acceptance_criterion: full-compiler target completes or returns first classified blocker
known_blockers: none
return_payload: summary, artifact root, first failing log if any, failure class
```
