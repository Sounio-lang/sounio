#!/bin/bash
#SBATCH --job-name=stdlib-profile-pilot31
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-inference-scale-20261024/profile_pilot02/logs/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-inference-scale-20261024/profile_pilot02/logs/slurm-%j.err
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
r=/scratch/sounio-stdlib-inference-scale-20261024/profile_pilot02
compiler=/scratch/sounio-stdlib-inference-scale-20261024/compiler/artifacts/madaros.elf
finish(){ local rc=$?; printf '%s\n' "$rc" > "$r/STATUS"; }
trap finish EXIT
[[ $(id -u) == 1000 ]]
[[ $(cat /scratch/sounio-stdlib-advanced-20261023/nested_profile_smoke01/artifacts/STATUS) == 0 ]]
ulimit -s 524288
export TMPDIR="$r/tmp"
sha256sum "$compiler" > "$r/COMPILER.sha256"
g++ -std=c++23 -O2 -Wall -Wextra "$r/tools/count_profile_benchmark_reference.cpp" -o "$r/tools/analyzer" > "$r/tools/compile.stdout" 2> "$r/tools/compile.stderr"
for v in before after; do
 cd "$r/source-$v"
 export SOUNIO_STDLIB_PATH="$r/source-$v/stdlib"
 /usr/bin/time -v -o "$r/artifacts/$v.compile.resource" timeout 240 "$compiler" build workload.sio --target native -o "$r/artifacts/$v.elf" > "$r/artifacts/$v.compile.stdout" 2> "$r/artifacts/$v.compile.stderr"
 chmod +x "$r/artifacts/$v.elf"
done
printf 'variant\ttrial\tstdout\tresource\tstatus\telf\tsource_sha256\telf_sha256\tcompiler_sha256\tinput_manifest\tinput_manifest_sha256\n' > "$r/RUNS.tsv"
for v in before after; do
 cd "$r/source-$v"
 rc=0
 /usr/bin/time -v -o "$r/runs/$v.resource" timeout --kill-after=10 600 env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C COUNT_PROFILE_CALIBRATION_CASE=0 "$r/artifacts/$v.elf" > "$r/runs/$v.stdout" 2> "$r/runs/$v.stderr" || rc=$?
 printf '%s\n' "$rc" > "$r/runs/$v.status"
 [[ $rc == 0 ]] || exit "$rc"
 srcsha=$(sha256sum workload.sio);elfsha=$(sha256sum "$r/artifacts/$v.elf");csha=$(sha256sum "$compiler");msha=$(sha256sum "$r/INPUTS.$v.sha256")
 printf '%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$v" "$r/runs/$v.stdout" "$r/runs/$v.resource" "$r/runs/$v.status" "$r/artifacts/$v.elf" "${srcsha%% *}" "${elfsha%% *}" "${csha%% *}" "$r/INPUTS.$v.sha256" "${msha%% *}" >> "$r/RUNS.tsv"
 sha256sum -c "$r/INPUTS.$v.sha256" > "$r/INPUTS.$v.verify"
done
"$r/tools/analyzer" --pilot "$r/source-before/workload.sio" "$r/source-after/workload.sio" "$r/RUNS.tsv" "$compiler" > "$r/REPORT.tsv" 2> "$r/REPORT.stderr"
sha256sum -c "$r/COMPILER.sha256" > "$r/COMPILER.verify"
