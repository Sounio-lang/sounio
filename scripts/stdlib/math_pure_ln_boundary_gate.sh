#!/usr/bin/env bash
# Native Linux execution only. Source-built compiler identity required.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
out=${LN_BOUNDARY_OUT:?dedicated absolute output directory}
[[ "$out" == /* && -f "${MADAROS_RAW_BIN:?source-built compiler required}" ]]
mkdir -p "$out"
[[ -z $(find "$out" -mindepth 1 -maxdepth 1 -print -quit) ]]
export PATH=/usr/bin:/bin LANG=C LC_ALL=C SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_STDLIB_PATH="$repo/stdlib" SOUNIO_MADAROS_NOCACHE=1
ulimit -s 524288
cd "$repo"
finish(){ printf '%s\n' "$?" > "$out/STATUS"; }
trap finish EXIT
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INPUTS.sha256"
sha256sum tests/run-pass/math_pure_ln_pos_inf_terminates.sio tests/reference/math_pure_ln_boundary_reference.cpp scripts/stdlib/math_pure_ln_boundary_gate.sh >> "$out/INPUTS.sha256"
g++ -std=c++23 -O2 -Wall -Wextra -Werror tests/reference/math_pure_ln_boundary_reference.cpp -lquadmath -o "$out/reference" > "$out/reference.build.stdout" 2> "$out/reference.build.stderr"
src=tests/run-pass/math_pure_ln_pos_inf_terminates.sio
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
timeout 120 bin/souc check "$src" > "$out/check.stdout" 2> "$out/check.stderr"
printf 'check\t0\n' >> "$out/RESULTS.tsv"
timeout 180 bin/souc compile "$src" -o "$out/current.elf" > "$out/compile.stdout" 2> "$out/compile.stderr"
printf 'compile\t0\n' >> "$out/RESULTS.tsv"
chmod +x "$out/current.elf"
/usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/current.resource" timeout 15 "$out/current.elf" > "$out/current.stdout" 2> "$out/current.stderr"
printf 'native-current-main\t0\n' >> "$out/RESULTS.tsv"
"$out/reference" "$out/current.stdout" > "$out/reference.stdout" 2> "$out/reference.stderr"
printf 'cpp-boundary-reference\t0\n' >> "$out/RESULTS.tsv"
# Discriminating negative control, not a claim that current main still hangs.
mkdir "$out/control"
cp -a stdlib "$out/control/stdlib"
[[ $(grep -Fc '    if x / 2.0 == x { return x }' stdlib/math/pure.sio) == 1 ]]
sed '\|^    if x / 2.0 == x { return x }$|d' stdlib/math/pure.sio > "$out/control/stdlib/math/pure.sio"
SOUNIO_STDLIB_PATH="$out/control/stdlib" timeout 180 bin/souc compile "$src" -o "$out/guard-removed.elf" > "$out/control.compile.stdout" 2> "$out/control.compile.stderr"
chmod +x "$out/guard-removed.elf"
rc=0
timeout --kill-after=2 3 "$out/guard-removed.elf" > "$out/control.stdout" 2> "$out/control.stderr" || rc=$?
printf 'guard-removed-negative-control\t%s\n' "$rc" >> "$out/RESULTS.tsv"
[[ "$rc" == 124 ]]
sha256sum "$out/current.elf" "$out/guard-removed.elf" "$out/reference" > "$out/EXECUTABLES.sha256"
sha256sum -c "$out/INPUTS.sha256" > "$out/input-verify.stdout"
printf 'source-inputs-unchanged\t0\n' >> "$out/RESULTS.tsv"
echo MATH_PURE_LN_BOUNDARY_GATE_PASS
