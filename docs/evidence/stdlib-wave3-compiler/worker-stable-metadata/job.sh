#!/bin/bash
#SBATCH --job-name=stdlib-worker-stable-metadata
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-stable-metadata04/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-stable-metadata04/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/worker-stable-metadata04
D=$R/results
S=$R/source
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated01/results/madaros
trap 'r=$?;echo "$r" > "$D/STATUS";exit "$r"' EXIT
ulimit -s 524288
ulimit -c 0
cp -a "$B/worker-ir-cfg-v5/source" "$S"
cat "$R/inputs/production.sio" "$R/inputs/component.sio" > "$S/self-hosted/native/worker_proof.sio"
cd "$S"
sha256sum "$RAW" bin/souc-lean-single-x86_64 > "$D/TOOLS.sha256"
find self-hosted tests/fixtures/wave4_worker_capability -type f -print0|sort -z|xargs -0 sha256sum > "$D/SOURCE.sha256"
export SOUNIO_STDLIB_PATH="$S/self-hosted"
timeout 100 "$RAW" check tests/fixtures/wave4_worker_capability/worker_ir_component_driver.sio > "$D/check.stdout" 2>&1
timeout 120 bin/souc-lean-single-x86_64 tests/fixtures/wave4_worker_capability/worker_ir_component_driver.sio "$D/component.elf" > "$D/compile.stdout" 2>&1
! grep -Eq '^error:|^error\[|would have dropped|cannot compile an assignment' "$D/compile.stdout"
chmod +x "$D/component.elf"
timeout 60 "$D/component.elf" > "$D/native.stdout" 2>&1
grep -q '^WORKER_RETIREMENT_OK ' "$D/native.stdout"
grep -q '^WORKER_EXTERN_NAMESPACE_OK ' "$D/native.stdout"
sha256sum -c "$D/SOURCE.sha256" > "$D/SOURCE.final.verify"
sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
sha256sum "$D/component.elf" > "$D/ELF.sha256"
echo WORKER_STABLE_METADATA_COMPONENT_OK_NOT_BFS
