#!/bin/bash
#SBATCH --nodelist=gpuorangefs-multi-r740-proxmox
#SBATCH --job-name=call-arg-stack-build
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/call-arg-release-stack-20261101/build-%j.out
#SBATCH --error=/scratch/call-arg-release-stack-20261101/build-%j.err
set -euo pipefail
r=/scratch/call-arg-release-stack-20261101
cd "$r/source"
export SOUNIO_STDLIB_PATH="$PWD/stdlib" SOUNIO_LOOP_REGIONS=0
ulimit -s 524288
ulimit -c 0
trap 'rc=$?; echo "$rc" > "$r/results/BUILD_STATUS"; exit "$rc"' EXIT
find self-hosted -type f -print0 | sort -z | xargs -0 sha256sum > "$r/results/SOURCE.sha256"
sha256sum ./-o /scratch/raw-cast-fix-20261101/results/madaros > "$r/results/SEEDS.sha256"
/scratch/raw-cast-fix-20261101/results/madaros check self-hosted/check/check.sio > "$r/results/check.log" 2>&1
printf '%s\n' './-o self-hosted/compiler/main.sio /scratch/call-arg-release-stack-20261101/results/madaros' > "$r/results/BUILD_COMMAND"
timeout 6600 ./-o self-hosted/compiler/main.sio "$r/results/madaros" > "$r/results/build.stdout" 2> "$r/results/build.stderr"
chmod +x "$r/results/madaros"
sha256sum "$r/results/madaros" > "$r/results/COMPILER.sha256"
