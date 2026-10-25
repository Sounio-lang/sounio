#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=2048M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
BASE=/scratch/sounio-stdlib-inference-scale-20261024/SIO169
OUT="$BASE/memory_trace14"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
sha256sum "$OUT/census.cpp" "$MADAROS_RAW_BIN" "$BASE/memory14/results/n100000.generated.sio" "$BASE/source13/stdlib/stats/checked_survival.sio" "$BASE/source14/stdlib/stats/checked_survival.sio" > "$OUT/inputs.sha256"
g++ -std=c++23 -O2 -Wall -Wextra "$OUT/census.cpp" -o "$OUT/census" > "$OUT/census.compile.stdout" 2> "$OUT/census.compile.stderr"
for version in before after;do
 if [[ "$version" == before ]];then cd "$BASE/source13";else cd "$BASE/source14";fi
 export SOUNIO_STDLIB_PATH="$PWD/stdlib"
 env SOUNIO_SOUC_ENGINE=madaros SOUNIO_NV2_IR_TRACE=1 bin/souc compile "$BASE/memory14/results/n100000.generated.sio" -o "$OUT/$version.elf" > "$OUT/$version.compile.stdout" 2> "$OUT/$version.compile.stderr"
 if grep -Eq '^error:|error\[E' "$OUT/$version.compile.stdout" "$OUT/$version.compile.stderr";then exit 1;fi
 sha256sum "$OUT/$version.elf" > "$OUT/$version.elf.sha256"
 "$OUT/census" "$OUT/$version.compile.stdout" > "$OUT/$version.census.stdout" 2> "$OUT/$version.census.stderr"
done
sha256sum -c "$OUT/inputs.sha256" > "$OUT/inputs.verify.stdout"
echo ALLOCATION_CENSUS_CAPTURE_PASS > "$OUT/gate.result"
