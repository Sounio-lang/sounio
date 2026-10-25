#!/bin/bash
#SBATCH --partition=cpu-ops
#SBATCH --nodelist=cpuops-t560-proxmox
#SBATCH --cpus-per-task=1
#SBATCH --mem=4096M
#SBATCH --time=00:20:00
#SBATCH --export=NIL
set -euo pipefail
export PATH=/usr/bin:/bin LANG=C LC_ALL=C
ulimit -s 524288
base=/scratch/sounio-stdlib-inference-scale-20261024/SIO168
out="$base/phase05"
cd "$base/source05"
export MADAROS_RAW_BIN=/scratch/sounio-stdlib-inference-scale-20261024/compiler_ci_c21/madaros.elf.ready
export SOUNIO_STDLIB_PATH="$PWD/stdlib"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 "$base/phase04/reference" > "$out/compiler_reference.sha256"
sha256sum stdlib/stats/wilcoxon.sio stdlib/stats/checked_wilcoxon.sio tests/reference/wilcoxon_checked_reference.cpp tests/run-pass/stdlib_wilcoxon_checked.sio > "$out/inputs.sha256"
run_fixture() {
 local src="$1" name="$2" expected="$3"
 SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$src" > "$out/$name.seed.check.stdout" 2> "$out/$name.seed.check.stderr"
 if grep -Eq '^error:|error\[E' "$out/$name.seed.check.stdout" "$out/$name.seed.check.stderr";then return 1;fi
 /usr/bin/time -v env SOUNIO_SOUC_ENGINE=lean_single bin/souc compile "$src" -o "$out/$name.seed.elf" > "$out/$name.seed.compile.stdout" 2> "$out/$name.seed.compile.stderr"
 if grep -Eq '^error:|error\[E' "$out/$name.seed.compile.stdout" "$out/$name.seed.compile.stderr";then return 1;fi
 chmod +x "$out/$name.seed.elf";sha256sum "$out/$name.seed.elf" > "$out/$name.seed.elf.sha256"
 set +e
 /usr/bin/time -v timeout 180 "$out/$name.seed.elf" > "$out/$name.seed.native.stdout" 2> "$out/$name.seed.native.stderr"
 local code=$?;echo "$code" > "$out/$name.seed.native.exit"
 set -e
 [[ "$code" == "$expected" ]]
 /usr/bin/time -v bin/souc compile "$src" -o "$out/$name.elf" > "$out/$name.compile.stdout" 2> "$out/$name.compile.stderr"
 if grep -Eq '^error:|error\[E' "$out/$name.compile.stdout" "$out/$name.compile.stderr";then return 1;fi
 chmod +x "$out/$name.elf";sha256sum "$out/$name.elf" > "$out/$name.elf.sha256"
 set +e
 /usr/bin/time -v timeout 180 "$out/$name.elf" > "$out/$name.native.stdout" 2> "$out/$name.native.stderr"
 code=$?;echo "$code" > "$out/$name.native.exit"
 set -e
 [[ "$code" == "$expected" ]]
}
reference="$base/phase04/reference"
"$reference" --generate-legacy > "$out/legacy.generated.sio"
run_fixture "$out/legacy.generated.sio" legacy_after 0
"$reference" --legacy "$out/legacy_after.seed.native.stdout" > "$out/legacy.after.seed.verify.stdout" 2> "$out/legacy.after.seed.verify.stderr"
"$reference" --legacy "$out/legacy_after.native.stdout" > "$out/legacy.after.verify.stdout" 2> "$out/legacy.after.verify.stderr"
cd "$base/source01";export SOUNIO_STDLIB_PATH="$PWD/stdlib"
run_fixture "$out/legacy.generated.sio" legacy_before 0
set +e
"$reference" --legacy "$out/legacy_before.native.stdout" > "$out/legacy.before.verify.stdout" 2> "$out/legacy.before.verify.stderr"
code=$?;echo "$code" > "$out/legacy.before.verify.exit"
set -e
[[ "$code" == 1 ]]
cd "$base/source05";export SOUNIO_STDLIB_PATH="$PWD/stdlib"
run_fixture tests/run-pass/stdlib_wilcoxon_checked.sio smoke 0
grep -qx STDLIB_WILCOXON_CHECKED_PASS "$out/smoke.native.stdout"
run_fixture stdlib/stats/wilcoxon.sio legacy_module 0
grep -qx 'ALL PASS' "$out/legacy_module.native.stdout"
run_fixture tests/stdlib/stats/wilcoxon_invalid_probe.sio invalid 1
run_fixture tests/stdlib/stats/wilcoxon_nonfinite_probe.sio nonfinite 1
run_fixture examples/stats/full_analysis_report.sio full_analysis 0
run_fixture examples/stats/epistemic_suite_demo.sio epistemic_suite 0
echo WILCOXON_PHASE05_PASS > "$out/gate.result"
