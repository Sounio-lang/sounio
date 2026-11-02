#!/bin/bash
#SBATCH -J sounio-bytewidth-active
#SBATCH -p cpu-ops
#SBATCH -w cpuops-t560-proxmox
#SBATCH -n1
#SBATCH -c4
#SBATCH --mem=12G
#SBATCH -t 00:15:00
#SBATCH -o /scratch/sounio-bytewidth-20261102/active-%j.log
set -euo pipefail
ulimit -s 1048576
cd /scratch/sounio-bytewidth-20261102
run=$PWD/run-$SLURM_JOB_ID
mkdir -p "$run"
cd source
export SOUNIO_BUILD_LOCK="$run/build.lock"
export SOUNIO_MADAROS_CACHE_DIR="$run/cache"
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_BIN SOUNIO_MADAROS_BIN || true
sha256sum self-hosted/compiler/main.sio self-hosted/native/codegen_x86_linux.sio self-hosted/native/machine_ir.sio bin/souc-lean-single-x86_64
if ! timeout --kill-after=10s 600s bash scripts/ci/build_modular_madaros.sh "$run/madaros" > "$run/build.log" 2>&1;then tail -25 "$run/build.log";exit 2;fi
test -x "$run/madaros"
sha256sum "$run/madaros"
export SOUNIO_MADAROS_BIN="$run/madaros"
# Use the repository's public resolver with an explicit freshly built compiler.
timeout --kill-after=5s 90s bin/souc compile ../witness.sio -o "$run/witness.elf" > "$run/compile.log" 2>&1
chmod +x "$run/witness.elf"
timeout --kill-after=5s 30s "$run/witness.elf" > "$run/witness.log"
cat "$run/witness.log"
grep -Fx 'BYTEWIDTH_ACTIVE_PASS' "$run/witness.log"
echo CURRENT_SOURCE_PUBLIC_RESOLVER_BYTEWIDTH_PASS
