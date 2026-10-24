#!/bin/bash
#SBATCH --job-name=stdlib-ownerfix-f64-gate
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/ownerfix_gate02/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/ownerfix_gate02/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/ownerfix_gate02
cd "$r/source"
export SOUNIO_MADAROS_F64_LOWERING_GATE_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ownerfix01/artifacts/madaros.elf
export SOUNIO_MADAROS_F64_LOWERING_GATE_KEEP=1 SOUNIO_MADAROS_F64_LOWERING_GATE_DIR="$r/functional"
export TMPDIR="$r/tmp" SOUNIO_MADAROS_CACHE="$r/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s 524288
find -L bin scripts self-hosted stdlib tests examples -type f -print0 | sort -z | xargs -0 sha256sum > "$r/INPUTS.sha256"
sha256sum "$SOUNIO_MADAROS_F64_LOWERING_GATE_BIN" > "$r/COMPILER.sha256"
rc=0
/usr/bin/time -v -o "$r/resource.txt" bash scripts/ci/madaros_current_source_f64_lowering_gate.sh > "$r/gate.stdout" 2> "$r/gate.stderr" || rc=$?
printf '%s\n' "$rc" > "$r/STATUS"
sha256sum -c "$r/INPUTS.sha256" > "$r/INPUTS.verify.txt"
sha256sum -c "$r/COMPILER.sha256" > "$r/COMPILER.verify.txt"
exit "$rc"
