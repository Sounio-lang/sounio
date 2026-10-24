#!/bin/bash
#SBATCH --job-name=stdlib-scale-correctness
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:45:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/combined_owner_viz02/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/combined_owner_viz02/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/combined_owner_viz02
cd "$r/source"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ownerfix01/artifacts/madaros.elf
export STDLIB_DEEP_OUT="$r/artifacts" TMPDIR="$r/tmp" SOUNIO_MADAROS_CACHE="$r/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
exec bash scripts/stdlib/deep_science_gate.sh
