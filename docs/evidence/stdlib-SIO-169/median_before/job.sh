#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:30:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
cd /scratch/sounio-stdlib-inference-scale-20261024/SIO169/source02
OUT=/scratch/sounio-stdlib-inference-scale-20261024/SIO169/phase02
sha256sum -c SIO169_PHASE02.sha256 > "$OUT/inputs.verify.stdout"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
printf 'a292cc263f3734b1ad6f17a2e6179b8a7874ef3e76d8f5fdfea6859290baf5d3  %s\n' "$MADAROS_RAW_BIN" | sha256sum -c -
printf '8956946def36004d11ec4eca0d20c75c1e7d3d0b17ee0f0a1fdda9cd6ec0e4cc  bin/souc-lean-single-x86_64\n' | sha256sum -c -
g++ -std=c++23 -O2 -Wall -Wextra tests/reference/survival_checked_reference.cpp -o "$OUT/reference" > "$OUT/reference.compile.stdout" 2> "$OUT/reference.compile.stderr"
"$OUT/reference" --generate > "$OUT/small.generated.sio"
"$OUT/reference" --generate-scale > "$OUT/scale.generated.sio"
"$OUT/reference" --generate-legacy > "$OUT/legacy.generated.sio"
run_fixture() {
 local src="$1" name="$2" engine="$3"
 export SOUNIO_STDLIB_PATH="$PWD/stdlib"
 env SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$src" > "$OUT/$name.$engine.check.stdout" 2> "$OUT/$name.$engine.check.stderr"
 if grep -Eq '^error:|error\[E' "$OUT/$name.$engine.check.stdout" "$OUT/$name.$engine.check.stderr";then exit 1;fi
 /usr/bin/time -v env SOUNIO_SOUC_ENGINE="$engine" bin/souc compile "$src" -o "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.compile.stdout" 2> "$OUT/$name.$engine.compile.stderr"
 if grep -Eq '^error:|error\[E' "$OUT/$name.$engine.compile.stdout" "$OUT/$name.$engine.compile.stderr";then exit 1;fi
 chmod +x "$OUT/$name.$engine.elf";sha256sum "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.elf.sha256"
 /usr/bin/time -v timeout 900 "$OUT/$name.$engine.elf" > "$OUT/$name.$engine.native.stdout" 2> "$OUT/$name.$engine.native.stderr"
}
for engine in lean_single madaros;do
 run_fixture "$OUT/small.generated.sio" small "$engine"
 "$OUT/reference" "$OUT/small.$engine.native.stdout" > "$OUT/small.$engine.verify.stdout" 2> "$OUT/small.$engine.verify.stderr"
 run_fixture "$OUT/scale.generated.sio" scale "$engine"
 "$OUT/reference" --scale "$OUT/scale.$engine.native.stdout" > "$OUT/scale.$engine.verify.stdout" 2> "$OUT/scale.$engine.verify.stderr"
 run_fixture "$OUT/legacy.generated.sio" legacy_after "$engine"
 "$OUT/reference" --legacy "$OUT/legacy_after.$engine.native.stdout" > "$OUT/legacy_after.$engine.verify.stdout" 2> "$OUT/legacy_after.$engine.verify.stderr"
 cd /scratch/sounio-stdlib-inference-scale-20261024/SIO169/source01
 sha256sum stdlib/stats/logrank.sio > "$OUT/original_logrank.sha256"
 run_fixture "$OUT/legacy.generated.sio" legacy_before "$engine"
 set +e
 "$OUT/reference" --legacy "$OUT/legacy_before.$engine.native.stdout" > "$OUT/legacy_before.$engine.verify.stdout" 2> "$OUT/legacy_before.$engine.verify.stderr"
 code=$?;echo "$code" > "$OUT/legacy_before.$engine.verify.exit"
 set -e
 [[ "$code" == 1 ]] || exit 1
 grep -q LEGACY_LOGRANK_DIRECT_TAIL_MISMATCH "$OUT/legacy_before.$engine.verify.stderr"
 cd /scratch/sounio-stdlib-inference-scale-20261024/SIO169/source02
done
echo SURVIVAL_PHASE02_PASS > "$OUT/gate.result"
