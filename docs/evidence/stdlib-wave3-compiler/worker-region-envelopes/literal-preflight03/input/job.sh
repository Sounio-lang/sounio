#!/bin/bash
#SBATCH --job-name=stdlib-worker-literal-preflight
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:22:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-literal-preflight03/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/worker-literal-preflight03/results/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin
ulimit -s 524288
ulimit -c 0
B=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed
R=$B/worker-literal-preflight03
S=$R/source
SEED=$B/worker-portable-gate-v3/input/bin/souc-lean-single-x86_64
RAW=/scratch/sounio-stdlib-wave3-20261027/root-integrated02/results/madaros
TARGET=/scratch/sounio-stdlib-wave3-20261027/root-integrated02-loader/gate
trap 'r=$?;echo "$r" > "$R/results/STATUS";exit "$r"' EXIT
cp -a /scratch/sounio-stdlib-wave3-20261027/root-integrated02/source "$S"
cp "$R/input/worker_proof.sio" "$S/self-hosted/native/worker_proof.sio"
printf '\nuse native::worker_proof::{worker_private_failure_dump}\n' >> "$S/self-hosted/native/codegen_x86_linux.sio"
sed -i '/if !worker_prepare_clones(\&!(\*copy)) {/a\        worker_private_failure_dump(\&(*copy))' "$S/self-hosted/native/codegen_x86_linux.sio"
cp "$R/input/driver.sio" "$S/private_literal_driver.sio"
find "$S" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/SOURCE.before.sha256"
sha256sum "$SEED" "$RAW" > "$R/results/SEEDS.sha256"
cd "$S"
export SOUNIO_STDLIB_PATH=$S/self-hosted
timeout 120 "$RAW" check "$S/self-hosted/native/worker_proof.sio" > "$R/results/private-proof.check.stdout" 2>&1
set +e
timeout 1000 "$SEED" "$S/private_literal_driver.sio" "$R/results/driver.elf" > "$R/results/driver.compile.stdout" 2> "$R/results/driver.compile.stderr"
c=$?
set -e
echo "$c" > "$R/results/driver.compile.status"
if [ "$c" -ne 0 ];then exit "$c";fi
if grep -E 'error\[E|Error:|field initializer mismatch' "$R/results/driver.compile.stdout" "$R/results/driver.compile.stderr";then exit 31;fi
test -s "$R/results/driver.elf"
chmod +x "$R/results/driver.elf"
export SOUNIO_STDLIB_PATH=$TARGET/source/stdlib
export LD_LIBRARY_PATH=$TARGET
export SOUNIO_STDLIB_PLATFORM_LIB=$TARGET/libstdlib_platform.so
set +e
timeout 240 "$R/results/driver.elf" > "$R/results/preflight.stdout" 2> "$R/results/preflight.stderr"
c=$?
set -e
echo "$c" > "$R/results/preflight.status"
sha256sum -c "$R/results/SOURCE.before.sha256" > "$R/results/SOURCE.final.verify"
sha256sum -c "$R/results/SEEDS.sha256" > "$R/results/SEEDS.final.verify"
exit "$c"
