#!/usr/bin/env bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --job-name=SIO586-main-baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/SIO586-main-a232a6b1/baseline-%j.out
#SBATCH --error=/scratch/SIO586-main-a232a6b1/baseline-%j.err
set -euo pipefail
lane=/scratch/SIO586-main-a232a6b1
cd "$lane/source"
out="$lane/results"
mkdir -p "$out"
finish(){ rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$out/STATUS"; date -u +%Y-%m-%dT%H:%M:%SZ > "$out/ENDED_UTC"; exit "$rc"; }
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/STARTED_UTC"
printf '%s\n' a232a6b1d5b7dc8205aabe9064887700d7033ec2 > "$out/SOURCE_COMMIT"
find self-hosted -type f -print0 | sort -z | xargs -0 sha256sum > "$out/SOURCE.sha256"
sha256sum bin/souc-linux-x86_64 bin/souc-lean-single-x86_64 scripts/ci/build_modular_madaros.sh > "$out/BOOTSTRAP_INPUTS.sha256"
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
export SOUNIO_BUILD_LOCK="$out/build.lock" SOUNIO_MADAROS_CACHE="$out/cache" SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
ulimit -c 0
timeout 6600 bash scripts/ci/build_modular_madaros.sh "$out/madaros" > "$out/build.stdout" 2> "$out/build.stderr"
chmod +x "$out/madaros"
sha256sum "$out/madaros" > "$out/COMPILER.sha256"
printf '0\n' > "$out/BUILD_STATUS"
export MADAROS_RAW_BIN="$out/madaros" STACK_GATE_OUT="$out/stack-gate"
bash scripts/ci/call_arg_release_stack_gate.sh
sha256sum -c "$out/SOURCE.sha256" > "$out/SOURCE.final.verify"
