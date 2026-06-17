#!/usr/bin/env bash
# Gate for the RTX 8000 chi6 search-only Kubernetes job manifest generator.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

MAKER="$ROOT/examples/erdos/make_chi6_rtx8000_gpu_job.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$MAKER"
mkdir -p "$WORK"

echo "chi6_rtx8000_gpu_job_manifest_gate: workdir=$WORK"
python3 "$MAKER" \
  --namespace beagle \
  --job-name chi6-rtx8000-test \
  --pvc-name solver-scratch-test \
  --image ttl.sh/beagle-sounio-test:24h \
  --candidate-prefix rtxtest \
  --max-den-list 1,5 \
  --layers-list 1 \
  --max-points-list 16 \
  --split-depth-list 1 \
  --min-split-degree-list 2 \
  --dsatur-node-limit-list 1 \
  --preflight-limit 1 \
  --max-cubes 64 \
  --sample-hard-cubes 4 \
  --refute-limit 1 \
  --timeout-seconds 180 \
  -o "$WORK/job.yaml" > "$WORK/maker.out"

rg -q '^chi6_rtx8000_gpu_job_yaml=' "$WORK/maker.out"
rg -q '^# Search-only chi6 RTX 8000 job' "$WORK/job.yaml"
rg -q '^# Cluster contract: Darwin r740-proxmox RTX 8000 lane with OrangeFS scratch' "$WORK/job.yaml"
rg -q 'storageClassName: orangefs-memory-lab' "$WORK/job.yaml"
rg -q 'namespace: beagle' "$WORK/job.yaml"
rg -q 'name: chi6-rtx8000-test' "$WORK/job.yaml"
rg -q 'nodeSelector:' "$WORK/job.yaml"
rg -q 'kubernetes.io/hostname: r740-proxmox' "$WORK/job.yaml"
rg -q 'runtimeClassName: nvidia' "$WORK/job.yaml"
rg -q 'GPU-1a84782e-5b22-def2-2575-0abf3f971016' "$WORK/job.yaml"
rg -q 'mountPath: /scratch' "$WORK/job.yaml"
rg -q 'trust_boundary=backend_untrusted__drat_lrat_lean_verified_required' "$WORK/job.yaml"
rg -q 'examples/erdos/chi6_frontier_refute_sweep.py' "$WORK/job.yaml"
rg -q -- '--candidate-prefix rtxtest' "$WORK/job.yaml"
rg -q -- '--max-den-list 1,5' "$WORK/job.yaml"
rg -q -- '--max-cubes 64' "$WORK/job.yaml"
rg -q 'SOUNIO_REPO does not contain examples/erdos' "$WORK/job.yaml"
rg -q 'nvidia-smi --query-gpu=name,compute_cap,memory.total,uuid' "$WORK/job.yaml"

if rg -qi 'vllm|triton|llm serving|serve-vllm' "$WORK/job.yaml"; then
  echo "error: RTX 8000 compute job mentions LLM serving" >&2
  exit 1
fi
if rg -q 'emptyDir.*scratch|mountPath: /tmp' "$WORK/job.yaml"; then
  echo "error: job appears to use node-local scratch" >&2
  exit 1
fi

if python3 "$MAKER" --image ttl.sh/beagle-sounio-test:24h --job-name Bad_Name > "$WORK/bad-name.out" 2>&1; then
  echo "error: maker accepted invalid Kubernetes job name" >&2
  exit 1
fi
rg -q 'job-name must be a Kubernetes DNS label' "$WORK/bad-name.out"

if python3 "$MAKER" --image ttl.sh/beagle-sounio-test:24h --sample-hard-cubes 8 --max-cubes 4 > "$WORK/bad-sample.out" 2>&1; then
  echo "error: maker accepted sample-hard-cubes > max-cubes" >&2
  exit 1
fi
rg -q -- '--sample-hard-cubes cannot exceed --max-cubes' "$WORK/bad-sample.out"

if python3 "$MAKER" --job-name chi6-missing-image > "$WORK/missing-image.out" 2>&1; then
  echo "error: maker accepted a missing cluster image" >&2
  exit 1
fi
rg -q 'the following arguments are required: --image' "$WORK/missing-image.out"

if python3 "$MAKER" --image beagle-sounio --job-name chi6-bad-image > "$WORK/bad-image.out" 2>&1; then
  echo "error: maker accepted an unqualified cluster image" >&2
  exit 1
fi
rg -q -- '--image must include a registry host and image path' "$WORK/bad-image.out"

echo "chi6_rtx8000_gpu_job_manifest_gate: PASS"
