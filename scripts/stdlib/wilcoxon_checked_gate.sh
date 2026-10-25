#!/usr/bin/env bash
# Allocated Linux CPU qualification; both declared compiler engines execute.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: wilcoxon_checked_gate.sh OUTPUT_DIRECTORY}"
: "${MADAROS_RAW_BIN:?explicit source-built compiler required}"
[[ -z "${SOUNIO_SOUC_BIN:-}" ]] || { echo 'raw SOUNIO_SOUC_BIN override would bypass declared engine routing' >&2;exit 64; }
mkdir -p "$OUT";OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s 524288
id > "$OUT/identity.txt"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$OUT/compiler.sha256"
sha256sum stdlib/stats/{wilcoxon,checked_wilcoxon,checked_distributions}.sio \
 stdlib/collections/{numeric_index_sort,numeric_buffer,integer_buffer}.sio \
 stdlib/math/{pure,logarithmic}.sio stdlib/special/checked_gamma_beta.sio \
 tests/reference/{wilcoxon_checked_reference,numeric_index_sort_reference}.cpp \
 tests/run-pass/stdlib_wilcoxon_checked.sio tests/stdlib/stats/wilcoxon_{invalid,nonfinite}_probe.sio \
 examples/stats/{full_analysis_report,epistemic_suite_demo}.sio \
 scripts/stdlib/wilcoxon_checked_gate.sh > "$OUT/inputs.sha256"
g++ --version > "$OUT/reference.compiler.txt"
for name in wilcoxon_checked numeric_index_sort;do
 /usr/bin/time -v g++ -std=c++23 -O2 -Wall -Wextra "tests/reference/${name}_reference.cpp" -o "$OUT/$name.reference" > "$OUT/$name.reference.compile.stdout" 2> "$OUT/$name.reference.compile.stderr"
 sha256sum "$OUT/$name.reference" >> "$OUT/reference.sha256"
done
no_diagnostics() {
 if grep -Eq '^error:|error\[E' "$1" "$2";then echo "compiler diagnostics: $1" >&2;return 1;fi
}
run_fixture() {
 local src="$1" name="$2" expected="$3"
 sha256sum "$src" > "$OUT/$name.source.sha256"
 SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$src" > "$OUT/$name.seed.check.stdout" 2> "$OUT/$name.seed.check.stderr"
 no_diagnostics "$OUT/$name.seed.check.stdout" "$OUT/$name.seed.check.stderr"
 /usr/bin/time -v env SOUNIO_SOUC_ENGINE=lean_single bin/souc compile "$src" -o "$OUT/$name.seed.elf" > "$OUT/$name.seed.compile.stdout" 2> "$OUT/$name.seed.compile.stderr"
 no_diagnostics "$OUT/$name.seed.compile.stdout" "$OUT/$name.seed.compile.stderr"
 chmod +x "$OUT/$name.seed.elf";sha256sum "$OUT/$name.seed.elf" > "$OUT/$name.seed.elf.sha256"
 set +e
 /usr/bin/time -v timeout 600 "$OUT/$name.seed.elf" > "$OUT/$name.seed.native.stdout" 2> "$OUT/$name.seed.native.stderr"
 local code=$?;echo "$code" > "$OUT/$name.seed.native.exit"
 set -e
 [[ "$code" == "$expected" ]]
 /usr/bin/time -v env SOUNIO_SOUC_ENGINE=madaros bin/souc compile "$src" -o "$OUT/$name.elf" > "$OUT/$name.compile.stdout" 2> "$OUT/$name.compile.stderr"
 no_diagnostics "$OUT/$name.compile.stdout" "$OUT/$name.compile.stderr"
 chmod +x "$OUT/$name.elf";sha256sum "$OUT/$name.elf" > "$OUT/$name.elf.sha256"
 set +e
 /usr/bin/time -v timeout 600 "$OUT/$name.elf" > "$OUT/$name.native.stdout" 2> "$OUT/$name.native.stderr"
 code=$?;echo "$code" > "$OUT/$name.native.exit"
 set -e
 [[ "$code" == "$expected" ]]
 sha256sum -c "$OUT/$name.source.sha256" > "$OUT/$name.source.verify.stdout"
}
run_fixture tests/run-pass/stdlib_wilcoxon_checked.sio smoke 0
grep -qx STDLIB_WILCOXON_CHECKED_PASS "$OUT/smoke.native.stdout"
"$OUT/wilcoxon_checked.reference" --generate > "$OUT/modern.generated.sio"
run_fixture "$OUT/modern.generated.sio" modern 0
for engine in seed.native native;do
 "$OUT/wilcoxon_checked.reference" "$OUT/modern.$engine.stdout" > "$OUT/modern.$engine.verify.stdout" 2> "$OUT/modern.$engine.verify.stderr"
done
"$OUT/wilcoxon_checked.reference" --generate-legacy > "$OUT/legacy.generated.sio"
run_fixture "$OUT/legacy.generated.sio" legacy 0
for engine in seed.native native;do
 "$OUT/wilcoxon_checked.reference" --legacy "$OUT/legacy.$engine.stdout" > "$OUT/legacy.$engine.verify.stdout" 2> "$OUT/legacy.$engine.verify.stderr"
done
"$OUT/wilcoxon_checked.reference" --generate-scale > "$OUT/scale.generated.sio"
run_fixture "$OUT/scale.generated.sio" scale 0
for engine in seed.native native;do
 "$OUT/wilcoxon_checked.reference" --scale "$OUT/scale.$engine.stdout" > "$OUT/scale.$engine.verify.stdout" 2> "$OUT/scale.$engine.verify.stderr"
done
"$OUT/numeric_index_sort.reference" --generate > "$OUT/sort.generated.sio"
run_fixture "$OUT/sort.generated.sio" sort 0
for engine in seed.native native;do
 "$OUT/numeric_index_sort.reference" "$OUT/sort.$engine.stdout" > "$OUT/sort.$engine.verify.stdout" 2> "$OUT/sort.$engine.verify.stderr"
done
run_fixture stdlib/stats/wilcoxon.sio legacy_module 0
grep -qx 'ALL PASS' "$OUT/legacy_module.native.stdout"
run_fixture tests/stdlib/stats/wilcoxon_invalid_probe.sio invalid 1
run_fixture tests/stdlib/stats/wilcoxon_nonfinite_probe.sio nonfinite 1
run_fixture examples/stats/full_analysis_report.sio full_analysis 0
run_fixture examples/stats/epistemic_suite_demo.sio epistemic_suite 0
# Corrupt actual native records. Integer counts and representation/metadata
# semantics must be checked, even when an absolute numerical tolerance is loose.
awk '$1=="DIGIT"&&$2==0&&$3==0&&$4==0&&$5==0 {$6=0} {print}' "$OUT/modern.native.stdout" > "$OUT/mutant_numerator.stdout"
awk '$1=="DIGIT"&&$2==23&&$3==0&&$4==0&&$5==8 {$6=0} {print}' "$OUT/modern.native.stdout" > "$OUT/mutant_high_limb.stdout"
awk '$1=="EXACT"&&$2==26&&$3==0&&$4==2 {$6=1;$9=0} {print}' "$OUT/modern.native.stdout" > "$OUT/mutant_underflow.stdout"
awk '$1=="EXACT"&&$2==25&&$3==0&&$4==2 {$10=0} {print}' "$OUT/modern.native.stdout" > "$OUT/mutant_logtail.stdout"
awk '$1=="EXACT"&&$2==0&&$3==0&&$4==0 {$18=2} {print}' "$OUT/modern.native.stdout" > "$OUT/mutant_method.stdout"
awk '$1=="EXACT"&&$2==0&&$3==0&&$4==0 {$12=1} {print}' "$OUT/modern.native.stdout" > "$OUT/mutant_denominator.stdout"
sed '/^WILCOXON_ROWS_COMPLETE /d' "$OUT/modern.native.stdout" > "$OUT/mutant_missing.stdout"
for mutant in numerator high_limb underflow logtail method denominator missing;do
 if "$OUT/wilcoxon_checked.reference" "$OUT/mutant_$mutant.stdout" > "$OUT/mutant_$mutant.verify.stdout" 2> "$OUT/mutant_$mutant.verify.stderr";then
  echo "oracle accepted $mutant mutant" >&2;exit 1
 fi
done
awk '$1=="SORT"&&$2==1000 {$6=0} {print}' "$OUT/sort.native.stdout" > "$OUT/mutant_sort.stdout"
if "$OUT/numeric_index_sort.reference" "$OUT/mutant_sort.stdout" > "$OUT/mutant_sort.verify.stdout" 2> "$OUT/mutant_sort.verify.stderr";then
 echo 'oracle accepted sort mutant' >&2;exit 1
fi
sha256sum -c "$OUT/inputs.sha256" > "$OUT/inputs.verify.stdout"
sha256sum -c "$OUT/compiler.sha256" > "$OUT/compiler.verify.stdout"
sha256sum -c "$OUT/reference.sha256" > "$OUT/reference.verify.stdout"
printf 'WILCOXON_CHECKED_GATE_PASS\n' | tee "$OUT/gate.result"
