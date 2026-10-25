#!/usr/bin/env bash
# Native Linux/Slurm replay with an explicitly supplied Madaros raw executable.
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${1:?fresh absolute output required}
baseline=${2:-}
[[ "$out" == /* && ! -e "$out" ]] || exit 64
[[ -n ${MADAROS_RAW_BIN:-} && -x "$MADAROS_RAW_BIN" ]] || exit 78
[[ -z "$baseline" || -f "$baseline" ]] || exit 66
mkdir -p "$out"/{tmp,cache}
trap 'rc=$?; printf "%s\n" "$rc" > "$out/STATUS"' EXIT
cd "$root"
# The canonical resolver's explicit executable override avoids unused prebuilt
# materialization; compiler identity below is the executable actually selected.
export SOUC_BIN="$MADAROS_RAW_BIN" SOUNIO_STDLIB_PATH="$root/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$out/cache" TMPDIR="$out/tmp"
source scripts/lib/resolve_souc.sh
sounio_require_souc
ulimit -s 524288
sha256sum "$SOUC_BIN" > "$out/COMPILER.sha256"
sha256sum stdlib/math/numdiff.sio stdlib/math/numdiff_checked.sio tests/run-pass/math_numdiff_checked.sio tests/reference/math_numdiff_reference.cpp tests/stdlib/numdiff/*.sio scripts/stdlib/math_numdiff_checked_gate.sh > "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
build(){ local name=$1 src=$2;step "$name.check" timeout 180 "$SOUC_BIN" check "$src";step "$name.compile" timeout 180 "$SOUC_BIN" compile "$src" -o "$out/$name.elf";chmod +x "$out/$name.elf";sha256sum "$out/$name.elf" >> "$out/ELFS.sha256"; }
run(){ local name=$1 elf=$2;step "$name.native" /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 180 "$elf"; }
step library.check timeout 180 "$SOUC_BIN" check stdlib/math/numdiff_checked.sio
build checked tests/run-pass/math_numdiff_checked.sio
run checked "$out/checked.elf"
step checked.marker grep -Fx NUMDIFF_CHECKED_PASS "$out/checked.native.stdout"
cat stdlib/math/numdiff.sio tests/stdlib/numdiff/legacy_roots_main.sio > "$out/roots-after.sio"
build roots-after "$out/roots-after.sio"
run roots-after "$out/roots-after.elf"
for name in legacy_negative_h legacy_negative_count;do
 cat stdlib/math/numdiff.sio "tests/stdlib/numdiff/$name.sio" > "$out/$name.sio"
 build "$name" "$out/$name.sio"
 rc=0;run "$name" "$out/$name.elf" || rc=$?
 [[ $rc == 1 ]]
 ! grep -Fq NUMDIFF_REJECTION_MISSING "$out/$name.native.stdout"
 printf '%s.explicit-rejection\t0\n' "$name" >> "$out/RESULTS.tsv"
done
step cpp.toolchain "${CXX:-g++}" --version
step cpp.compile timeout 180 "${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/math_numdiff_reference.cpp -o "$out/reference"
sha256sum "$out/reference" > "$out/REFERENCE.sha256"
step concordance "$out/reference" "$out/checked.native.stdout"
step roots.concordance "$out/reference" --roots "$out/roots-after.native.stdout"
build nonconvergence tests/stdlib/numdiff/vector_nonconvergence.sio
run nonconvergence "$out/nonconvergence.elf"
step nonconvergence.concordance "$out/reference" --nonconvergence "$out/nonconvergence.native.stdout"
sed '2s/.*/999/' "$out/checked.native.stdout" > "$out/wrong-value.stdout"
rc=0;"$out/reference" "$out/wrong-value.stdout" > "$out/wrong-value.check.stdout" 2> "$out/wrong-value.check.stderr" || rc=$?
printf 'wrong_value_expected11_actual\t%s\n' "$rc" > "$out/CONTROLS.tsv"
[[ $rc == 11 ]]
grep -v '^NUMDIFF_CHECKED_PASS$' "$out/checked.native.stdout" > "$out/missing-completion.stdout"
rc=0;"$out/reference" "$out/missing-completion.stdout" > "$out/missing-completion.check.stdout" 2> "$out/missing-completion.check.stderr" || rc=$?
printf 'missing_completion_expected13_actual\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ $rc == 13 ]]
if [[ -n "$baseline" ]];then
 sha256sum "$baseline" > "$out/BASELINE.sha256"
 cat "$baseline" tests/stdlib/numdiff/legacy_roots_main.sio > "$out/roots-before.sio"
 build roots-before "$out/roots-before.sio"
 rc=0;run roots-before "$out/roots-before.elf" || rc=$?
 [[ $rc == 1 ]]
 printf 'roots-before.failure-witness\t0\n' >> "$out/RESULTS.tsv"
 cat "$baseline" tests/stdlib/numdiff/legacy_negative_h.sio > "$out/h-before.sio"
 build h-before "$out/h-before.sio"
 rc=0;run h-before "$out/h-before.elf" || rc=$?
 [[ $rc == 42 ]]
 grep -Fxq NUMDIFF_REJECTION_MISSING "$out/h-before.native.stdout"
 printf 'h-before.missing-rejection-witness\t0\n' >> "$out/RESULTS.tsv"
fi
if [[ -n ${NUMDIFF_SEED_BIN:-} ]];then
 [[ -x $NUMDIFF_SEED_BIN ]] || exit 78
 sha256sum "$NUMDIFF_SEED_BIN" > "$out/SEED_COMPILER.sha256"
 # This seed uses raw SRC OUT CLI; runtime is independently checked below.
 step seed.caller-check timeout 180 "$NUMDIFF_SEED_BIN" tests/run-pass/math_numdiff_checked.sio "$out/seed-check.elf"
 chmod +x "$out/seed-check.elf"
 sha256sum "$out/seed-check.elf" > "$out/SEED_ELF.sha256"
 run strict-seed "$out/seed-check.elf"
 step strict-seed.concordance "$out/reference" "$out/strict-seed.native.stdout"
fi
step compiler_integrity sha256sum -c "$out/COMPILER.sha256"
step input_integrity sha256sum -c "$out/INPUTS.sha256"
printf 'NUMDIFF_GATE_PASS\n'
