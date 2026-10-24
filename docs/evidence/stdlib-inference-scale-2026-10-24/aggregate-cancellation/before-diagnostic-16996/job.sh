#!/bin/bash
#SBATCH --job-name=aggregate-cancellation-audit
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation01/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/aggregate_cancellation01
cd "$r/source";trap 'echo "$?" >"$r/STATUS"' EXIT
ulimit -s 524288
export TMPDIR="$r/tmp" SOUNIO_STDLIB_PATH="$r/source/stdlib"
find . -type f -print0|sort -z|xargs -0 sha256sum >"$r/INPUTS.sha256"
g++ -std=c++23 -O2 reference.cpp -o "$r/artifacts/reference" >"$r/artifacts/cpp.compile.stdout" 2>"$r/artifacts/cpp.compile.stderr"
for variant in ownerfix effectbridge;do
 compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ownerfix01/artifacts/madaros.elf
 [[ $variant != effectbridge ]] || compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/artifacts/madaros.elf
 sha256sum "$compiler" >"$r/artifacts/$variant.COMPILER.sha256"
 "$compiler" check diagnostic.sio >"$r/artifacts/$variant.check.stdout" 2>"$r/artifacts/$variant.check.stderr"
 /usr/bin/time -v -o "$r/artifacts/$variant.compile.resource" "$compiler" build diagnostic.sio --target native -o "$r/artifacts/$variant.elf" >"$r/artifacts/$variant.compile.stdout" 2>"$r/artifacts/$variant.compile.stderr"
 chmod +x "$r/artifacts/$variant.elf"
 sha256sum "$r/artifacts/$variant.elf" >"$r/artifacts/$variant.ELF.sha256"
 rc=0
 /usr/bin/time -v -o "$r/artifacts/$variant.native.resource" "$r/artifacts/$variant.elf" >"$r/artifacts/$variant.native.stdout" 2>"$r/artifacts/$variant.native.stderr" || rc=$?
 echo "$rc" >"$r/artifacts/$variant.native.status";[[ $rc == 0 ]]
 "$r/artifacts/reference" "$r/artifacts/$variant.native.stdout" >"$r/artifacts/$variant.REPORT.tsv" 2>"$r/artifacts/$variant.REPORT.stderr"
 sha256sum -c "$r/artifacts/$variant.COMPILER.sha256" >"$r/artifacts/$variant.COMPILER.after.verify"
done
sha256sum -c "$r/INPUTS.sha256" >"$r/INPUTS.after.verify"
(cd "$r" && find artifacts -type f ! -name '*.elf' ! -name reference -print0|sort -z|xargs -0 sha256sum)>"$r/OUTPUTS.sha256"
