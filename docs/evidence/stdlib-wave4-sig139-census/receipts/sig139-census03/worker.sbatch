#!/bin/bash
#SBATCH --job-name=sig139-discriminators
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-census03/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/compiler-resumed/sig139-census03/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
D="$B/compiler-resumed/sig139-census03/results"
S="$B/compiler-resumed/sig139-census03/source"
I="$B/compiler-resumed/sig139-census03/input"
R="$B/root-integrated04"
source "$R/tools-env-v2.sh"
export SOUNIO_STDLIB_PATH="$S/stdlib" SOUNIO_LOOP_REGIONS=1 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
RAW="$B/root-integrated03/results/madaros"
TIME=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027/tools/time
ulimit -s 524288
ulimit -c 0
trap 'r=$?;printf "%s\n" "$r" > "$D/WORKER_STATUS";date -u > "$D/ENDED_UTC";exit "$r"' EXIT
cd "$S"
printf 'e0279ff62e97e861415cd8b105b854ae645aa61865b33b1e2b91184d2c28bf54  %s\n' "$RAW" > "$D/RAW.sha256"
sha256sum -c "$D/RAW.sha256" > "$D/RAW.before.verify"
sha256sum "$I/trace.cpp" "$I/signal-control.cpp" "$I/CASES.txt" "$R/tools-env-v2.sh" "$CXX" "$TIME" > "$D/TOOLS.sha256"
find stdlib self-hosted tests scripts -type f -print0 | sort -z | xargs -0 sha256sum > "$D/SOURCE.sha256"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$I/trace.cpp" -o "$D/trace" > "$D/tracer-build.stdout" 2> "$D/tracer-build.stderr"
"$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$I/signal-control.cpp" -o "$D/signal-control" > "$D/signal-build.stdout" 2> "$D/signal-build.stderr"
status=0
timeout 15 "$D/trace" "$D/signal-control" > "$D/signal-control.stdout" 2>&1 || status=$?
test "$status" = 139
grep -q 'fault_signal=11' "$D/signal-control.stdout"
timeout 15 "$D/trace" /bin/true > "$D/positive-control.stdout" 2>&1
grep -Fxq 'exit=0' "$D/positive-control.stdout"
while IFS= read -r test; do
 name=${test##*/}
 name=${name%.sio}
 status=0
 "$TIME" -v timeout 1200 "$D/trace" --raw03-state "$RAW" compile "$test" -o "$D/$name.elf" > "$D/$name.stdout" 2> "$D/$name.stderr" || status=$?
 printf '%s\t%s\n' "$test" "$status" >> "$D/RESULTS.tsv"
 # Diagnostic capture accepts and records all compiler outcomes; it does not
 # relabel a non-crash, timeout or failure as the historical SIG139.
 if test "$status" = 0;then test -s "$D/$name.elf";sha256sum "$D/$name.elf" >> "$D/PRODUCED_ELFS.sha256";fi
done < "$I/CASES.txt"
sha256sum -c "$D/RAW.sha256" > "$D/RAW.final.verify"
sha256sum -c "$D/TOOLS.sha256" > "$D/TOOLS.final.verify"
sha256sum -c "$D/SOURCE.sha256" > "$D/SOURCE.final.verify"
sha256sum "$D/trace" "$D/signal-control" > "$D/TRACER_ELFS.sha256"
printf 'Segment/disassembly inspection is a separate read-only artifact; readelf absent on worker.\n' > "$D/SEGMENT_SCOPE.txt"
printf 'SIG139_DIAGNOSTICS_CAPTURED_NOT_A_FIX\n' > "$D/STATUS"
