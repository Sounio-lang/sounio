#!/bin/bash
#SBATCH --job-name=effectbridge-guards-retry
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/guard_retry01/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01/guard_retry01/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/compiler_effectbridge01
g="$r/guard_retry01"
trap 'rc=$?; if [[ $rc != 0 ]];then echo "FAILED $rc" >"$g/STATUS";fi' EXIT
[[ $(id -u) == 1000 ]];cd "$r/source"
ulimit -s 524288
export TMPDIR="$r/tmp" SOUNIO_BUILD_LOCK="$r/build.lock" SOUNIO_MADAROS_CACHE="$r/cache" SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_STDLIB_PATH="$r/source/stdlib" SOUNIO_SOUC_ENGINE=madaros MADAROS_RAW_BIN="$r/artifacts/madaros.elf"
sha256sum -c "$r/artifacts/COMPILER.sha256" >"$g/COMPILER.before.verify"
grep -q '^d74fbebe0ff442e8c77bec024d6d92325b83a98d3f1487caa1db59ef171a41e6 ' "$r/artifacts/COMPILER.sha256"
sha256sum -c "$r/INPUTS.sha256" >"$g/INPUTS.before.verify"
printf 'kind\tsource\texit_status\n' >"$g/RESULTS.tsv"
for src in tests/run-pass/stdlib_deep_linear_borrow.sio tests/run-pass/stdlib_deep_callback_binding.sio;do
 name=$(basename "$src" .sio)
 bin/souc check "$src" >"$g/$name.check.log" 2>&1
 printf 'check\t%s\t0\n' "$src" >>"$g/RESULTS.tsv"
 bin/souc compile "$src" -o "$g/$name.elf" >"$g/$name.compile.log" 2>&1
 chmod +x "$g/$name.elf"
 printf 'compile\t%s\t0\n' "$src" >>"$g/RESULTS.tsv"
 /usr/bin/time -v -o "$g/$name.resource" "$g/$name.elf" >"$g/$name.stdout" 2>"$g/$name.stderr"
 printf 'native\t%s\t0\n' "$src" >>"$g/RESULTS.tsv"
 sha256sum "$g/$name.elf" >>"$g/ELFS.sha256"
done
for src in tests/compile-fail/stdlib_deep_callback_private*.sio tests/compile-fail/stdlib_deep_buffer*.sio tests/compile-fail/madaros_module_qualified_call_private_fn.sio;do
 name=$(basename "$src" .sio);rc=0
 bin/souc check "$src" >"$g/$name.check.log" 2>&1 || rc=$?
 printf 'negative\t%s\t%s\n' "$src" "$rc" >>"$g/RESULTS.tsv"
 [[ $rc == 1 ]]
 if [[ $name == *callback* || $name == madaros_module* ]];then grep -q E175 "$g/$name.check.log";fi
done
sha256sum -c "$r/artifacts/COMPILER.sha256" >"$g/COMPILER.after.verify"
sha256sum -c "$r/INPUTS.sha256" >"$g/INPUTS.after.verify"
echo COMPLETE >"$g/STATUS"
