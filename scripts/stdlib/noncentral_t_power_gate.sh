#!/usr/bin/env bash
# Run only in an allocated Linux CPU job; explicit current-source compiler.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: noncentral_t_power_gate.sh OUTPUT_DIRECTORY}"
: "${MADAROS_RAW_BIN:?explicit compiler required}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s 524288
id > "$OUT/identity.txt"
sha256sum "$MADAROS_RAW_BIN" bin/souc-lean-single-x86_64 > "$OUT/compiler.sha256"
sha256sum stdlib/stats/{noncentral_t,checked_power,power,checked_distributions}.sio \
 stdlib/special/checked_gamma_beta.sio stdlib/math/{pure,logarithmic,dd64}.sio \
 tests/reference/{noncentral_t_reference,noncentral_t_power_reference,gamma_log_prefix_reference}.cpp \
 tests/run-pass/stdlib_noncentral_t_checked.sio tests/stdlib/stats/power_{invalid,unreachable}_probe.sio \
 scripts/stdlib/noncentral_t_power_gate.sh > "$OUT/inputs.sha256"
g++ --version > "$OUT/reference.compiler.txt"
for name in noncentral_t noncentral_t_power gamma_log_prefix; do
 /usr/bin/time -v g++ -std=c++23 -O2 -Wall -Wextra "tests/reference/${name}_reference.cpp" -o "$OUT/$name.reference" > "$OUT/$name.reference.compile.stdout" 2> "$OUT/$name.reference.compile.stderr"
 sha256sum "$OUT/$name.reference" >> "$OUT/reference.sha256"
 sha256sum "$OUT/$name.reference" >> "$OUT/ELFS.sha256"
 "$OUT/$name.reference" --generate > "$OUT/$name.generated.sio"
done
no_diagnostics() {
 if grep -Eq '^error:|error\[E' "$1" "$2";then echo "compiler diagnostics despite exit status: $1" >&2;return 1;fi
}
run_fixture() {
 local source="$1" name="$2" expected="$3"
 SOUNIO_SOUC_ENGINE=lean_single bin/souc check "$source" > "$OUT/$name.seed.stdout" 2> "$OUT/$name.seed.stderr"
 no_diagnostics "$OUT/$name.seed.stdout" "$OUT/$name.seed.stderr"
 bin/souc check "$source" > "$OUT/$name.check.stdout" 2> "$OUT/$name.check.stderr"
 no_diagnostics "$OUT/$name.check.stdout" "$OUT/$name.check.stderr"
 /usr/bin/time -v bin/souc compile "$source" -o "$OUT/$name.elf" > "$OUT/$name.compile.stdout" 2> "$OUT/$name.compile.stderr"
 no_diagnostics "$OUT/$name.compile.stdout" "$OUT/$name.compile.stderr"
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" > "$OUT/$name.elf.sha256"
 cat "$OUT/$name.elf.sha256" >> "$OUT/ELFS.sha256"
 set +e
 /usr/bin/time -v timeout 180 "$OUT/$name.elf" > "$OUT/$name.native.stdout" 2> "$OUT/$name.native.stderr"
 local code=$?
 set -e
 echo "$code" > "$OUT/$name.native.exit"
 [[ "$code" == "$expected" ]]
}
run_fixture tests/run-pass/stdlib_noncentral_t_checked.sio smoke 0
grep -qx STDLIB_NONCENTRAL_T_CHECKED_PASS "$OUT/smoke.native.stdout"
for name in noncentral_t noncentral_t_power gamma_log_prefix; do
 run_fixture "$OUT/$name.generated.sio" "$name" 0
 /usr/bin/time -v "$OUT/$name.reference" "$OUT/$name.native.stdout" > "$OUT/$name.verify.stdout" 2> "$OUT/$name.verify.stderr"
done
"$OUT/noncentral_t_power.reference" --generate-legacy > "$OUT/legacy.generated.sio"
run_fixture "$OUT/legacy.generated.sio" legacy 0
"$OUT/noncentral_t_power.reference" --legacy "$OUT/legacy.native.stdout" > "$OUT/legacy.verify.stdout" 2> "$OUT/legacy.verify.stderr"
run_fixture tests/stdlib/stats/power_invalid_probe.sio invalid 1
run_fixture tests/stdlib/stats/power_unreachable_probe.sio unreachable 1
# These corrupt actual native output; each must be rejected by the same oracle.
awk '$1=="ROW"&&$2==0 {$3=5} {print}' "$OUT/noncentral_t.native.stdout" > "$OUT/mutant_status.stdout"
awk '$1=="ROW"&&$2==0 {$10=0} {print}' "$OUT/noncentral_t.native.stdout" > "$OUT/mutant_logtail.stdout"
awk '$1=="ROW"&&$2==469 {$5=1;$9=0} {print}' "$OUT/noncentral_t.native.stdout" > "$OUT/mutant_underflow.stdout"
sed '/^NCT_ROWS_COMPLETE /d' "$OUT/noncentral_t.native.stdout" > "$OUT/mutant_missing.stdout"
for mutant in status logtail underflow missing; do
 if "$OUT/noncentral_t.reference" "$OUT/mutant_$mutant.stdout" > "$OUT/mutant_$mutant.verify.stdout" 2> "$OUT/mutant_$mutant.verify.stderr";then
  echo "oracle accepted $mutant mutant" >&2;exit 1
 fi
done
awk '$1=="SROW"&&$2==0 {$4=2} {print}' "$OUT/noncentral_t_power.native.stdout" > "$OUT/mutant_minimum.stdout"
awk '$1=="QROW"&&$2==0 {$9=0} {print}' "$OUT/noncentral_t_power.native.stdout" > "$OUT/mutant_forward.stdout"
for mutant in minimum forward; do
 if "$OUT/noncentral_t_power.reference" "$OUT/mutant_$mutant.stdout" > "$OUT/mutant_$mutant.verify.stdout" 2> "$OUT/mutant_$mutant.verify.stderr";then
  echo "oracle accepted $mutant mutant" >&2;exit 1
 fi
done
sha256sum -c "$OUT/ELFS.sha256" > "$OUT/ELFS.verify.txt"
printf 'NONCENTRAL_T_POWER_GATE_PASS\n' | tee "$OUT/gate.result"
