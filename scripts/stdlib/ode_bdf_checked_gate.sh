#!/usr/bin/env bash
# Focused native/C++ BDF gate. Native Linux execution; heavy work belongs on Slurm.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute output directory required}
before=${2:-}
[[ "$out" == /* && ! -e "$out" ]]
[[ -n ${MADAROS_RAW_BIN:-} && -x "$MADAROS_RAW_BIN" ]]
mkdir -p "$out"
trap 'rc=$?; printf "%s\n" "$rc" > "$out/STATUS"' EXIT
cd "$root"
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
sha256sum stdlib/linalg/dynamic_factorize.sio tests/reference/deep_factor_reference.cpp tests/stdlib/bdf/compare_factor_memory.sh stdlib/ode/bdf64.sio stdlib/ode/bdf_checked.sio tests/run-pass/ode_bdf_checked.sio tests/stdlib/bdf/*.sio tests/reference/ode_bdf_reference.cpp scripts/stdlib/ode_bdf_checked_gate.sh > "$out/INPUTS.sha256"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum >> "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
build(){ step "$1.check" timeout 180 bin/souc check "$2";step "$1.compile" timeout 180 bin/souc compile "$2" -o "$out/$1.elf";chmod +x "$out/$1.elf";sha256sum "$out/$1.elf" >> "$out/ELFS.sha256"; }
if [[ -n ${SEED_RAW_BIN:-} ]];then
 sha256sum "$SEED_RAW_BIN" > "$out/SEED.sha256"
 step strict-seed-check env SOUC_NATIVE_BIN="$SEED_RAW_BIN" scripts/ci/souc-native-wrapper.sh check tests/run-pass/ode_bdf_checked.sio
fi
build checked tests/run-pass/ode_bdf_checked.sio
step checked.native /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/checked.resource" timeout 180 "$out/checked.elf"
step cpp-build g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/ode_bdf_reference.cpp -lquadmath -o "$out/reference"
sha256sum "$out/reference" >> "$out/ELFS.sha256"
step concordance "$out/reference" "$out/checked.native.stdout"
sed '1s/^BDF 0 1 0 /BDF 0 1 1 /' "$out/checked.native.stdout" > "$out/wrong.stdout"
rc=0;"$out/reference" "$out/wrong.stdout" > "$out/wrong.check.stdout" 2> "$out/wrong.check.stderr" || rc=$?
[[ $rc == 6 ]];printf 'wrong_status_expected6_actual\t%s\n' "$rc" > "$out/CONTROLS.tsv"
awk 'NR==1 {$6=0} {print}' "$out/checked.native.stdout" > "$out/wrong-value.stdout"
rc=0;"$out/reference" "$out/wrong-value.stdout" > "$out/wrong-value.check.stdout" 2> "$out/wrong-value.check.stderr" || rc=$?
[[ $rc == 14 ]];printf 'wrong_value_expected14_actual\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
grep -v '^BDF_CHECKED_PASS$' "$out/checked.native.stdout" > "$out/missing.stdout"
rc=0;"$out/reference" "$out/missing.stdout" > "$out/missing.check.stdout" 2> "$out/missing.check.stderr" || rc=$?
[[ $rc == 27 ]];printf 'missing_completion_expected27_actual\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
cat stdlib/ode/bdf64.sio tests/stdlib/bdf/legacy_regressions_main.sio > "$out/legacy-after.sio"
build legacy-after "$out/legacy-after.sio"
step legacy-after.native timeout 90 "$out/legacy-after.elf"
step legacy-after.marker grep -Fx BDF_LEGACY_PASS "$out/legacy-after.native.stdout"
step legacy-concordance "$out/reference" --legacy "$out/legacy-after.native.stdout"
build scale128 tests/stdlib/bdf/scale128.sio
step scale128.native /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/scale128.resource" timeout 180 "$out/scale128.elf"
step scale128.concordance "$out/reference" --scale "$out/scale128.native.stdout"
build factor-layout tests/stdlib/bdf/factor_layout_refusal.sio
step factor-layout.native timeout 90 "$out/factor-layout.elf"
step factor-layout.marker grep -Fx FACTOR_LAYOUT_REFUSAL_PASS "$out/factor-layout.native.stdout"
step deep-factor.cpp g++ -std=c++23 -O2 -Wall -Wextra tests/reference/deep_factor_reference.cpp -lquadmath -o "$out/factor-reference"
sha256sum "$out/factor-reference" >> "$out/ELFS.sha256"
step deep-factor.emit "$out/factor-reference"
cp "$out/deep-factor.emit.stdout" "$out/factor-generated.sio"
build deep-factor "$out/factor-generated.sio"
step deep-factor.native timeout 180 "$out/deep-factor.elf"
if [[ -n "$before" ]];then
 sha256sum "$before" > "$out/BASELINE.sha256"
 cat "$before" tests/stdlib/bdf/legacy_regressions_main.sio > "$out/legacy-before.sio"
 build legacy-before "$out/legacy-before.sio"
 rc=0;step legacy-before.native timeout 90 "$out/legacy-before.elf" || rc=$?
 [[ $rc == 1 ]]
 grep -F 'LEGACY 0 0 0 0 ' "$out/legacy-before.native.stdout"
fi
step source-integrity sha256sum -c "$out/INPUTS.sha256"
step compiler-integrity sha256sum -c "$out/COMPILER.sha256"
step executable-integrity sha256sum -c "$out/ELFS.sha256"
echo BDF_GATE_PASS
