#!/bin/bash
#SBATCH --job-name=stdlib-final-cache-gate
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:40:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/combined_final_cache01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/combined_final_cache01/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/combined_final_cache01
compiler_run=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01
[[ $(cat "$compiler_run/guard_retry01/STATUS") == COMPLETE ]]
cd "$r/source"
export MADAROS_RAW_BIN="$compiler_run/artifacts/madaros.elf"
export STDLIB_DEEP_OUT="$r/artifacts" TMPDIR="$r/tmp" SOUNIO_MADAROS_CACHE="$r/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
exec /bin/bash scripts/stdlib/deep_science_gate.sh
