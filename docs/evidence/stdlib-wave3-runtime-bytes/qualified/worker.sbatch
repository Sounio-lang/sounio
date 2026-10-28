#!/bin/bash
#SBATCH --job-name=stdlib-runtime-bytes-cpp23
#SBATCH --partition=all
#SBATCH --nodelist=dl380-dl380-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --export=NIL
#SBATCH --output=/scratch/sounio-stdlib-wave3-20261027/core/runtime-bytes03/results/slurm-%j.out
#SBATCH --error=/scratch/sounio-stdlib-wave3-20261027/core/runtime-bytes03/results/slurm-%j.err
set -euo pipefail
B=/scratch/sounio-stdlib-wave3-20261027
R="$B/core/runtime-bytes03"
S="$R/source"
P=/scratch/sounio-stdlib-inference-scale-20261024/stdlib-next-20261027
T="$P/tooling"
export PATH="$P/tools:$T/root/usr/bin:/usr/bin:/bin"
export LD_LIBRARY_PATH="$T/root/usr/lib/x86_64-linux-gnu"
export MAGIC="$T/root/usr/lib/file/magic.mgc"
ulimit -s524288
ulimit -c0
trap 'r=$?;echo "$r" > "$R/results/WORKER_STATUS";date -u > "$R/results/ENDED_UTC";exit "$r"' EXIT
date -u > "$R/results/STARTED_UTC"
cp -a "$B/whole-integrated01/source" "$S"
cp -a "$R/input/stdlib/." "$S/stdlib/"
cp -a "$R/input/tests/stdlib/runtime_regression/." "$S/tests/stdlib/runtime_regression/"
cp -a "$R/input/tests/fixtures/runtime_bytes_contract" "$S/tests/fixtures/"
cp "$R/input/tests/reference/runtime_bytes_oracle.cpp" "$S/tests/reference/"
cp "$R/input/scripts/stdlib/runtime_bytes_gate.sh" "$S/scripts/stdlib/"
cd "$S"
find stdlib -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/STDLIB.actual.sha256"
cmp "$R/input/STDLIB.sha256" "$R/results/STDLIB.actual.sha256"
sha256sum -c "$R/input/STDLIB.sha256" > "$R/results/STDLIB.before.verify"
find "$R/input" -type f -print0|sort -z|xargs -0 sha256sum > "$R/results/INPUTS.sha256"
sha256sum "$R/worker.sbatch" > "$R/results/WORKER.sha256"
/usr/bin/time -v g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation tests/reference/runtime_bytes_oracle.cpp -o "$R/results/runtime_bytes_oracle" > "$R/results/oracle-compile.stdout" 2> "$R/results/oracle-compile.stderr"
raw1="$B/root-integrated01/results/madaros"
raw2="$B/root-integrated02/results/madaros"
printf '11332ed8a6cf7ccb85f0a31bf63bed1fcc10659054d68639d6dba233a6014ff4  %s\n8abe891a66c71bdbeb784a9200d1ea395d4d5a55f0036a0daa51ad3cf0614932  %s\n' "$raw1" "$raw2" > "$R/results/COMPILERS.sha256"
sha256sum -c "$R/results/COMPILERS.sha256" > "$R/results/COMPILERS.before.verify"
status=0
for label in predecessor converged;do
 raw="$raw1";if test "$label" = converged;then raw="$raw2";fi
 result=0
 bash scripts/stdlib/runtime_bytes_gate.sh "$raw" "$S" "$R/results/runtime_bytes_oracle" "$B/services-tools-recovered/platform-abi-112/libstdlib_platform.so" "$R/results/$label" > "$R/results/$label.stdout" 2> "$R/results/$label.stderr" || result=$?
 echo "$result" > "$R/results/$label-STATUS"
 if test "$result" != 0;then status=1;fi
done
sha256sum -c "$R/results/COMPILERS.sha256" > "$R/results/COMPILERS.final.verify"
sha256sum -c "$R/results/INPUTS.sha256" > "$R/results/INPUTS.final.verify"
sha256sum -c "$R/results/WORKER.sha256" > "$R/results/WORKER.final.verify"
sha256sum -c "$R/input/STDLIB.sha256" > "$R/results/STDLIB.final.verify"
test "$status" = 0
echo RUNTIME_BYTES_TWO_RAW_CPP23_ONLY_OK > "$R/results/STATUS"
