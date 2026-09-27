#!/bin/bash
#SBATCH --job-name=ont-frontiers-sparse-N100k
#SBATCH --partition=cpu-ops
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=artifacts/ontology-frontiers/real-data/scale/ont_sparse_N100k_%j.out
#SBATCH --error=artifacts/ontology-frontiers/real-data/scale/ont_sparse_N100k_%j.err

set -euo pipefail
cd /workspace/sounio
echo "node: $(hostname)"
echo "commit: $(git rev-parse --short HEAD)"
echo "probe: probe_sparse_loop_star_100000.sio"

time ./bin/souc run \
  artifacts/ontology-frontiers/real-data/scale/probe_sparse_loop_star_100000.sio \
  2>&1 | tee artifacts/ontology-frontiers/real-data/scale/ont_sparse_N100k_%j.run.log
