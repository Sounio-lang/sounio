---
name: cluster-gpu-jobs
description: Bootstrap context for running GPU or HPC jobs on the AI/HPC cluster control plane at /home/devsounio/beagle/k8s/hpc-sota. Use before submitting cluster jobs, launching GPU work, or reaching for sbatch/kubectl in this repo.
---

# Cluster GPU jobs

The AI/HPC cluster control plane is at `/home/devsounio/beagle/k8s/hpc-sota`. Before GPU work, read:

1. `/home/devsounio/beagle/k8s/hpc-sota/AGENT_BOOTSTRAP.md`
2. `/home/devsounio/beagle/k8s/hpc-sota/DEV_WORKFLOW.md`

Prefer proven wrappers from `ops/lab-ops.sh` over ad hoc `sbatch` or `kubectl`.
