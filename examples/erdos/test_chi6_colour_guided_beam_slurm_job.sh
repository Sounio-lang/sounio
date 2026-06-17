#!/usr/bin/env bash
# Gate for the Slurm RTX 4000 Ada colour-guided beam campaign launcher.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

MAKER="$ROOT/examples/erdos/make_chi6_colour_guided_beam_slurm_job.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$MAKER"
mkdir -p "$WORK"

echo "chi6_colour_guided_beam_slurm_job_gate: workdir=$WORK"
python3 "$MAKER" \
  --job-name chi6-beam-ada-test \
  --array-count 4 \
  --coords-csv /orangefs/training/chi6-payloads/latest/seeds/coords.csv \
  --colourings-file /orangefs/training/chi6-payloads/latest/seeds/colourings.txt \
  --candidate-prefix cgadatest \
  --generations-list 1,2 \
  --beam-width-list 1,2 \
  --branch-width-list 1,2 \
  --mutation-max-den-list 5 \
  --mutation-top-points-list 4,8 \
  --dsatur-node-limit-list 1,100000 \
  --mutation-emit-mutations 2 \
  --mutation-add-points 1 \
  --mutation-min-neighbor-count 2 \
  --mutation-edge-gain-pool-points 64 \
  --mutation-edge-gain-max-combinations 12345 \
  --mutation-edge-gain-combination-offset 678 \
  --mutation-edge-gain-combination-stride 9 \
  --mutation-edge-gain-emit-mutations 3 \
  --max-cubes 100 \
  --sample-hard-cubes 2 \
  --run-refute-ready \
  --refute-limit 1 \
  --max-carried-colourings 16 \
  -o "$WORK/beam-campaign.sbatch" \
  > "$WORK/maker.out"

rg -q '^chi6_colour_guided_beam_slurm_job v1$' "$WORK/maker.out"
rg -q '^sbatch_script=' "$WORK/maker.out"
rg -q '^partition=gpu-orangefs$' "$WORK/maker.out"
rg -q '^constraint=rtx4000ada$' "$WORK/maker.out"
rg -q '^gres=gpu:1$' "$WORK/maker.out"
rg -q '^array_count=4$' "$WORK/maker.out"
rg -q '^claim_scope=colour_guided_beam_slurm_launcher_only$' "$WORK/maker.out"
rg -q '^sat_claim=none$' "$WORK/maker.out"
rg -q '^chromatic_claim=none$' "$WORK/maker.out"
rg -q '^global_unsat_claim=none$' "$WORK/maker.out"
rg -q '^verified_claim=none$' "$WORK/maker.out"
rg -q '^promotable=0$' "$WORK/maker.out"
rg -q '^status=COLOUR_GUIDED_BEAM_SLURM_JOB_READY$' "$WORK/maker.out"

rg -q '^#SBATCH --job-name=chi6-beam-ada-test$' "$WORK/beam-campaign.sbatch"
rg -q '^#SBATCH --partition=gpu-orangefs$' "$WORK/beam-campaign.sbatch"
rg -q '^#SBATCH --constraint=rtx4000ada$' "$WORK/beam-campaign.sbatch"
rg -q '^#SBATCH --gres=gpu:1$' "$WORK/beam-campaign.sbatch"
rg -q '^#SBATCH --array=0-3$' "$WORK/beam-campaign.sbatch"
rg -q '^#SBATCH --output=/orangefs/training/chi6-colour-guided-beam/slurm-%x-%A-%a.out$' \
  "$WORK/beam-campaign.sbatch"
rg -F -q 'SOUNIO_REPO="${SOUNIO_REPO:-/orangefs/training/chi6-payloads/latest/sounio}"' \
  "$WORK/beam-campaign.sbatch"
rg -F -q 'SCRATCH_ROOT="${CHI6_SCRATCH_ROOT:-/orangefs/training/chi6-colour-guided-beam}"' \
  "$WORK/beam-campaign.sbatch"
rg -q 'CHI6_SCRATCH_ROOT must stay under /orangefs' "$WORK/beam-campaign.sbatch"
rg -q 'trust_boundary=worker_untrusted__drat_lrat_lean_verified_required' \
  "$WORK/beam-campaign.sbatch"
rg -q 'examples/erdos/chi6_colour_guided_beam_campaign.py' "$WORK/beam-campaign.sbatch"
rg -q -- '--coords-csv /orangefs/training/chi6-payloads/latest/seeds/coords.csv' \
  "$WORK/beam-campaign.sbatch"
rg -q -- '--colourings-file /orangefs/training/chi6-payloads/latest/seeds/colourings.txt' \
  "$WORK/beam-campaign.sbatch"
rg -q -- '--shard-index \$SHARD_INDEX' "$WORK/beam-campaign.sbatch"
rg -q -- '--shard-count 4' "$WORK/beam-campaign.sbatch"
rg -q -- '--resume' "$WORK/beam-campaign.sbatch"
rg -q -- '--run-refute-ready' "$WORK/beam-campaign.sbatch"
rg -q -- '--mutation-min-neighbor-count 2' "$WORK/beam-campaign.sbatch"
rg -q -- '--mutation-edge-gain-pool-points 64' "$WORK/beam-campaign.sbatch"
rg -q -- '--mutation-edge-gain-max-combinations 12345' "$WORK/beam-campaign.sbatch"
rg -q -- '--mutation-edge-gain-combination-offset 678' "$WORK/beam-campaign.sbatch"
rg -q -- '--mutation-edge-gain-combination-stride 9' "$WORK/beam-campaign.sbatch"
rg -q -- '--mutation-edge-gain-emit-mutations 3' "$WORK/beam-campaign.sbatch"
rg -q -- '--max-carried-colourings 16' "$WORK/beam-campaign.sbatch"
rg -q 'nvidia-smi --query-gpu=name,compute_cap,memory.total,uuid' "$WORK/beam-campaign.sbatch"

if rg -q '/tmp|emptyDir|kubectl apply' "$WORK/beam-campaign.sbatch"; then
  echo "error: Slurm launcher mentions node-local or Kubernetes scratch" >&2
  exit 1
fi

if python3 "$MAKER" \
    --coords-csv /orangefs/training/coords.csv \
    --colourings-file /orangefs/training/colourings.txt \
    --scratch-root /tmp/chi6 \
    -o "$WORK/bad-scratch.sbatch" \
    > "$WORK/bad-scratch.out" 2>&1; then
  echo "error: maker accepted node-local scratch" >&2
  exit 1
fi
rg -q -- '--scratch-root must be under /orangefs' "$WORK/bad-scratch.out"

if python3 "$MAKER" \
    --scratch-root /orangefs/training/chi6 \
    -o "$WORK/missing-input.sbatch" \
    > "$WORK/missing-input.out" 2>&1; then
  echo "error: maker accepted missing campaign input" >&2
  exit 1
fi
rg -q 'provide --satfanout-json or both --coords-csv and --colourings-file' \
  "$WORK/missing-input.out"

if python3 "$MAKER" \
    --coords-csv /orangefs/training/coords.csv \
    --colourings-file /orangefs/training/colourings.txt \
    --branch-width-list 3 \
    --mutation-emit-mutations 2 \
    -o "$WORK/bad-branch.sbatch" \
    > "$WORK/bad-branch.out" 2>&1; then
  echo "error: maker accepted branch-width above emitted mutations" >&2
  exit 1
fi
rg -q -- '--branch-width-list cannot exceed --mutation-emit-mutations' "$WORK/bad-branch.out"

if python3 "$MAKER" \
    --coords-csv /orangefs/training/coords.csv \
    --colourings-file /orangefs/training/colourings.txt \
    --mutation-min-neighbor-count 0 \
    -o "$WORK/bad-neighbors.sbatch" \
    > "$WORK/bad-neighbors.out" 2>&1; then
  echo "error: maker accepted non-positive mutation-min-neighbor-count" >&2
  exit 1
fi
rg -q -- '--mutation-min-neighbor-count must be positive' "$WORK/bad-neighbors.out"

if python3 "$MAKER" \
    --coords-csv /orangefs/training/coords.csv \
    --colourings-file /orangefs/training/colourings.txt \
    --mutation-edge-gain-pool-points -1 \
    -o "$WORK/bad-edge-gain.sbatch" \
    > "$WORK/bad-edge-gain.out" 2>&1; then
  echo "error: maker accepted negative mutation-edge-gain-pool-points" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-pool-points must be non-negative' "$WORK/bad-edge-gain.out"

if python3 "$MAKER" \
    --coords-csv /orangefs/training/coords.csv \
    --colourings-file /orangefs/training/colourings.txt \
    --mutation-edge-gain-emit-mutations -1 \
    -o "$WORK/bad-edge-gain-count.sbatch" \
    > "$WORK/bad-edge-gain-count.out" 2>&1; then
  echo "error: maker accepted negative mutation-edge-gain-emit-mutations" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-emit-mutations must be non-negative' \
  "$WORK/bad-edge-gain-count.out"

if python3 "$MAKER" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --mutation-edge-gain-combination-offset -1 \
    -o "$WORK/bad-edge-gain-offset.sbatch" \
    > "$WORK/bad-edge-gain-offset.out" 2>&1; then
  echo "error: maker accepted negative mutation-edge-gain-combination-offset" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-offset must be non-negative' \
  "$WORK/bad-edge-gain-offset.out"

if python3 "$MAKER" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --mutation-edge-gain-combination-stride 0 \
    -o "$WORK/bad-edge-gain-stride.sbatch" \
    > "$WORK/bad-edge-gain-stride.out" 2>&1; then
  echo "error: maker accepted zero mutation-edge-gain-combination-stride" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-stride must be positive' \
  "$WORK/bad-edge-gain-stride.out"

echo "chi6_colour_guided_beam_slurm_job_gate: PASS"
