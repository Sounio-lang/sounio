#!/usr/bin/env bash
# Allocated Linux CPU worker, UID1000, stack 524288 KiB; explicit compiler only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:?usage: checked_distributions_gate.sh OUTPUT_DIRECTORY}"
: "${MADAROS_RAW_BIN:?explicit compiler required; no fallback qualification}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
ulimit -s 524288
id > "$OUT/identity.txt"
sha256sum "$MADAROS_RAW_BIN" > "$OUT/compiler.sha256"
sha256sum stdlib/stats/{distributions,checked_distributions}.sio \
 stdlib/special/{checked_gamma_beta,gamma}.sio stdlib/math/{pure,logarithmic,dd64}.sio \
 tests/reference/checked_distributions_reference.cpp \
 tests/run-pass/stdlib_{checked_distributions,beta_uniform_baseline}.sio \
 tests/stdlib/stats/distributions_{invalid,underflow}_probe.sio \
 scripts/stdlib/checked_distributions_gate.sh > "$OUT/inputs.sha256"
g++ --version > "$OUT/reference.compiler.txt"
g++ -std=c++23 -O2 -Wall -Wextra tests/reference/checked_distributions_reference.cpp -o "$OUT/reference" > "$OUT/reference.compile.stdout" 2> "$OUT/reference.compile.stderr"
sha256sum "$OUT/reference" > "$OUT/reference.sha256"
"$OUT/reference" --generate > "$OUT/generated.sio"
run_fixture() {
 local source="$1" name="$2" expected="$3"
 bin/souc check "$source" > "$OUT/$name.check.stdout" 2> "$OUT/$name.check.stderr"
 /usr/bin/time -v bin/souc compile "$source" -o "$OUT/$name.elf" > "$OUT/$name.compile.stdout" 2> "$OUT/$name.compile.stderr"
 chmod +x "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" > "$OUT/$name.elf.sha256"
 set +e
 /usr/bin/time -v timeout 120 "$OUT/$name.elf" > "$OUT/$name.native.stdout" 2> "$OUT/$name.native.stderr"
 local code=$?
 set -e
 echo "$code" > "$OUT/$name.native.exit"
 if [[ "$expected" == success ]]; then [[ "$code" == 0 ]];else [[ "$code" == 1 ]];fi
}
SOUNIO_SOUC_ENGINE=lean_single bin/souc check tests/run-pass/stdlib_checked_distributions.sio > "$OUT/seed.check.stdout" 2> "$OUT/seed.check.stderr"
run_fixture tests/run-pass/stdlib_beta_uniform_baseline.sio beta_uniform success
run_fixture tests/run-pass/stdlib_checked_distributions.sio smoke success
run_fixture "$OUT/generated.sio" concordance success
"$OUT/reference" "$OUT/concordance.native.stdout" > "$OUT/reference.verify.stdout" 2> "$OUT/reference.verify.stderr"
run_fixture tests/stdlib/stats/distributions_invalid_probe.sio invalid failure
run_fixture tests/stdlib/stats/distributions_underflow_probe.sio underflow failure
sed '0,/^ROW /s/^ROW [0-9]* [0-9]*/ROW 0 5/' "$OUT/concordance.native.stdout" > "$OUT/mutant_status.stdout"
sed '/^ROWS_COMPLETE /d' "$OUT/concordance.native.stdout" > "$OUT/mutant_missing.stdout"
{ cat "$OUT/concordance.native.stdout"; sed -n '/^ROW 0 /p' "$OUT/concordance.native.stdout"; } > "$OUT/mutant_duplicate.stdout"
awk '$1=="ROW" && $2==10 {$6=0} {print}' "$OUT/concordance.native.stdout" > "$OUT/mutant_value.stdout"
for mutant in status missing duplicate value; do
 if "$OUT/reference" "$OUT/mutant_$mutant.stdout" > "$OUT/mutant_$mutant.verify.stdout" 2> "$OUT/mutant_$mutant.verify.stderr";then
  echo "oracle accepted $mutant mutant" >&2;exit 1
 fi
done
printf 'CHECKED_DISTRIBUTIONS_GATE_PASS\n' | tee "$OUT/gate.result"
