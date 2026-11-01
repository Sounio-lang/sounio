#!/bin/bash
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --job-name=call-arg-stack-map
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/call-arg-release-stack-20261101/map-%j.out
#SBATCH --error=/scratch/call-arg-release-stack-20261101/map-%j.err
set -euo pipefail
r=/scratch/call-arg-release-stack-20261101
cd "$r/source"
ulimit -s 524288
ulimit -c 0
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_LOOP_REGIONS=0
trap 'rc=$?; echo "$rc" > "$r/results/MAP_BUILD_STATUS"; exit "$rc"' EXIT
timeout 6600 ./-o self-hosted/compiler/main.sio "$r/results/madaros-mapped" --debug-fn-map > "$r/results/mapped.stdout" 2> "$r/results/mapped.stderr"
chmod +x "$r/results/madaros-mapped"
sha256sum "$r/results/madaros-mapped" > "$r/results/MAPPED_COMPILER.sha256"
export MADAROS_RAW_BIN="$r/results/madaros-mapped"
export BASELINE_RAW_BIN=/scratch/raw-cast-fix-20261101/results/madaros
export STACK_GATE_OUT="$r/results/gate"
bash scripts/ci/call_arg_release_stack_gate.sh
