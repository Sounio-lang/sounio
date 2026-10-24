#!/usr/bin/env bash
# Native Linux/Slurm replay. Optional second arg: frozen pre-repair fft.sio.
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
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$out/cache" TMPDIR="$out/tmp"
source scripts/lib/resolve_souc.sh
sounio_require_souc
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
sha256sum stdlib/math/fft.sio stdlib/math/fft_checked.sio tests/run-pass/math_fft_checked.sio tests/reference/math_fft_reference.cpp tests/stdlib/fft/*.sio scripts/stdlib/math_fft_checked_gate.sh > "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
step(){ local name=$1;shift;local rc=0;"$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?;printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv";return "$rc"; }
build(){ local name=$1 src=$2;step "$name.check" timeout 180 "$SOUC_BIN" check "$src";step "$name.compile" timeout 180 "$SOUC_BIN" compile "$src" -o "$out/$name.elf";sha256sum "$out/$name.elf" >> "$out/ELFS.sha256"; }
run(){ local name=$1 elf=$2;step "$name.native" /usr/bin/time -f 'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kib=%M\nexit_code=%x' -o "$out/$name.resource" timeout 180 "$elf"; }
step library.check timeout 180 "$SOUC_BIN" check stdlib/math/fft_checked.sio
build checked tests/run-pass/math_fft_checked.sio
run checked "$out/checked.elf"
step checked.marker grep -Fx FFT_CHECKED_PASS "$out/checked.native.stdout"
cat stdlib/math/fft.sio tests/stdlib/fft/legacy_bounds_main.sio > "$out/legacy-after.sio"
build legacy-after "$out/legacy-after.sio"
rc=0;run legacy-after "$out/legacy-after.elf" || rc=$?
[[ $rc == 1 ]]
grep -Fxq FFT_LEGACY_CAPACITY_REQUEST "$out/legacy-after.native.stdout"
! grep -Fq FFT_LEGACY_REJECTION_MISSING "$out/legacy-after.native.stdout"
printf 'legacy-after.explicit-rejection\t0\n' >> "$out/RESULTS.tsv"
cat stdlib/math/fft.sio tests/stdlib/fft/legacy_valid_main.sio > "$out/legacy-valid.sio"
build legacy-valid "$out/legacy-valid.sio"
run legacy-valid "$out/legacy-valid.elf"
step legacy-valid.marker grep -Fx FFT_LEGACY_VALID_PASS "$out/legacy-valid.native.stdout"
step cpp.compile timeout 180 "${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/math_fft_reference.cpp -o "$out/reference"
step concordance "$out/reference" "$out/checked.native.stdout"
sed '2s/.*/999/' "$out/checked.native.stdout" > "$out/mutated-value.stdout"
rc=0;"$out/reference" "$out/mutated-value.stdout" > "$out/mutated-value.check.stdout" 2> "$out/mutated-value.check.stderr" || rc=$?
printf 'wrong_value_expected11_actual\t%s\n' "$rc" > "$out/CONTROLS.tsv"
[[ $rc == 11 ]]
grep -v '^FFT_CHECKED_PASS$' "$out/checked.native.stdout" > "$out/missing-completion.stdout"
rc=0;"$out/reference" "$out/missing-completion.stdout" > "$out/missing-completion.check.stdout" 2> "$out/missing-completion.check.stderr" || rc=$?
printf 'missing_completion_expected13_actual\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ $rc == 13 ]]
if [[ -n "$baseline" ]];then
 sha256sum "$baseline" > "$out/BASELINE.sha256"
 cat "$baseline" tests/stdlib/fft/legacy_bounds_main.sio > "$out/legacy-before.sio"
 build legacy-before "$out/legacy-before.sio"
 rc=0;run legacy-before "$out/legacy-before.elf" || rc=$?
 [[ $rc == 42 ]]
 grep -Fxq FFT_LEGACY_REJECTION_MISSING "$out/legacy-before.native.stdout"
 printf 'legacy-before.missing-rejection-witness\t0\n' >> "$out/RESULTS.tsv"
 cat "$baseline" tests/stdlib/fft/bench_legacy_main.sio > "$out/bench-before.sio"
 build bench-before "$out/bench-before.sio"
 build bench-after tests/stdlib/fft/bench_checked.sio
 printf 'pair\tposition\tvariant\n' > "$out/PAIR_ORDER.tsv"
 for pair in 1 2 3;do
  variants="before after"
  if [[ $pair == 2 ]];then variants="after before";fi
  position=0
  for variant in $variants;do
   position=$((position+1))
   printf '%s\t%s\t%s\n' "$pair" "$position" "$variant" >> "$out/PAIR_ORDER.tsv"
   run "pair$pair-$variant" "$out/bench-$variant.elf"
   step "pair$pair-$variant.concordance" "$out/reference" --bench "$out/pair$pair-$variant.native.stdout"
  done
 done
fi
step input_integrity sha256sum -c "$out/INPUTS.sha256"
printf 'FFT_GATE_PASS\n'
