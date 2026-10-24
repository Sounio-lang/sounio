#!/bin/bash
#SBATCH --job-name=effectbridge-fresh-selfcheck
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=16384M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/selfcheck_retry01/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/selfcheck_retry01/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01
[[ $(id -u) == 1000 && $(cat "$r/guard_retry01/STATUS") == COMPLETE ]]
cd "$r/source"
trap 'echo "$?" >"$r/selfcheck_retry01/STATUS"' EXIT
ulimit -s 524288
export TMPDIR="$r/tmp" SOUNIO_STDLIB_PATH="$r/source/stdlib"
sha256sum -c "$r/artifacts/COMPILER.sha256" >"$r/selfcheck_retry01/COMPILER.before.verify"
sha256sum -c "$r/INPUTS.sha256" >"$r/selfcheck_retry01/INPUTS.before.verify"
/usr/bin/time -v -o "$r/selfcheck_retry01/resource.txt" timeout 840 "$r/artifacts/madaros.elf" check self-hosted/compiler/main.sio >"$r/selfcheck_retry01/native.stdout" 2>"$r/selfcheck_retry01/native.stderr"
grep -Fq 'check: OK' "$r/selfcheck_retry01/native.stdout"
sha256sum -c "$r/artifacts/COMPILER.sha256" >"$r/selfcheck_retry01/COMPILER.after.verify"
sha256sum -c "$r/INPUTS.sha256" >"$r/selfcheck_retry01/INPUTS.after.verify"
