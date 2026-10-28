#!/usr/bin/env bash
#SBATCH --job-name=SIO172-final22
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -uo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
p=/scratch/sounio-stdlib-inference-scale-20261024/SIO172/final22
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
export PHONON_GATE_OUTPUT="$p/artifacts"
cd "$p/source"
rc=0;bash scripts/stdlib/phonon_factor_gate.sh > "$p/gate.stdout" 2> "$p/gate.stderr"||rc=$?
echo "$rc">"$p/gate.status"
