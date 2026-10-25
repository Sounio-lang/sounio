#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:15:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
BASE=/scratch/sounio-stdlib-inference-scale-20261024/SIO169
OUT="$BASE/legacy_baseline02"
cd "$BASE/source01"
export SOUNIO_STDLIB_PATH="$PWD/stdlib" MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
sha256sum stdlib/stats/logrank.sio "$BASE/phase09/results/reference" "$BASE/source09/tests/reference/survival_checked_reference.cpp" "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$OUT/inputs.sha256"
"$BASE/phase09/results/reference" --generate-legacy > "$OUT/legacy.generated.sio"
for engine in lean_single madaros;do
 env SOUNIO_SOUC_ENGINE="$engine" bin/souc check "$OUT/legacy.generated.sio" > "$OUT/$engine.check.stdout" 2> "$OUT/$engine.check.stderr"
 if grep -Eq '^error:|error\[E' "$OUT/$engine.check.stdout" "$OUT/$engine.check.stderr";then exit 1;fi
 /usr/bin/time -v env SOUNIO_SOUC_ENGINE="$engine" bin/souc compile "$OUT/legacy.generated.sio" -o "$OUT/$engine.elf" > "$OUT/$engine.compile.stdout" 2> "$OUT/$engine.compile.stderr"
 chmod +x "$OUT/$engine.elf";sha256sum "$OUT/$engine.elf" > "$OUT/$engine.elf.sha256"
 /usr/bin/time -v "$OUT/$engine.elf" > "$OUT/$engine.native.stdout" 2> "$OUT/$engine.native.stderr"
 set +e
 "$BASE/phase09/results/reference" --legacy "$OUT/$engine.native.stdout" > "$OUT/$engine.verify.stdout" 2> "$OUT/$engine.verify.stderr"
 code=$?;echo "$code" > "$OUT/$engine.verify.exit"
 set -e
 [[ "$code" == 1 ]]
 grep -q LEGACY_LOGRANK_DIRECT_TAIL_MISMATCH "$OUT/$engine.verify.stderr"
done
sha256sum -c "$OUT/inputs.sha256" > "$OUT/inputs.verify.stdout"
echo LEGACY_BASELINE_EXPECTED_FAILURE_PASS > "$OUT/gate.result"
