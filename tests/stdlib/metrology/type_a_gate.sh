#!/usr/bin/env bash
# Focused Type-A replay; execute on an authorized native Linux/Slurm lane.
# MADAROS_RAW_BIN=/absolute/fresh/compiler bash tests/stdlib/metrology/type_a_gate.sh /fresh/output
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
out=${1:?fresh output directory required}
[[ "$out" == /* && ! -e "$out" ]] || { echo 'output must be fresh and absolute' >&2; exit 64; }
[[ -n ${MADAROS_RAW_BIN:-} && -x "$MADAROS_RAW_BIN" ]] || { echo 'explicit executable MADAROS_RAW_BIN required' >&2; exit 78; }
mkdir -p "$out"/{tmp,cache}
trap 'rc=$?; printf "%s\n" "$rc" > "$out/STATUS"' EXIT
cd "$root"
export SOUNIO_STDLIB_PATH="$root/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1 SOUNIO_MADAROS_CACHE="$out/cache" TMPDIR="$out/tmp"
source scripts/lib/resolve_souc.sh
sounio_require_souc
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$out/COMPILER.sha256"
git rev-parse HEAD > "$out/SOURCE_HEAD" 2>/dev/null || printf 'snapshot-without-git\n' > "$out/SOURCE_HEAD"
files=(stdlib/metrology/calibration.sio stdlib/metrology/mod.sio tests/stdlib/metrology/test_type_a_boundary.sio tests/stdlib/metrology/type_a_reference.cpp tests/stdlib/metrology/type_a_gate.sh)
sha256sum "${files[@]}" > "$out/INPUTS.sha256"
printf 'phase\texit_status\n' > "$out/RESULTS.tsv"
step() { local name=$1; shift; local rc=0; "$@" > "$out/$name.stdout" 2> "$out/$name.stderr" || rc=$?; printf '%s\t%s\n' "$name" "$rc" >> "$out/RESULTS.tsv"; return "$rc"; }
step check timeout 180 "$SOUC_BIN" check tests/stdlib/metrology/test_type_a_boundary.sio
step compile timeout 180 "$SOUC_BIN" compile tests/stdlib/metrology/test_type_a_boundary.sio -o "$out/type_a.elf"
sha256sum "$out/type_a.elf" > "$out/ELF.sha256"
step native /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$out/native.resource" timeout 30 "$out/type_a.elf"
step marker grep -Fx TYPE_A_BOUNDARY_OK "$out/native.stdout"
step cpp_compile timeout 180 "${CXX:-g++}" -std=c++23 -O2 -Wall -Wextra -Werror tests/stdlib/metrology/type_a_reference.cpp -o "$out/type_a_reference"
step concordance "$out/type_a_reference" "$out/native.stdout"
# Negative copies verify that agreement and completion are both enforced.
sed '2s/.*/123/' "$out/native.stdout" > "$out/wrong-value.stdout"
rc=0; "$out/type_a_reference" "$out/wrong-value.stdout" > "$out/wrong-value.check.stdout" 2> "$out/wrong-value.check.stderr" || rc=$?
printf 'wrong_value_rejected_7\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ $rc == 7 ]]
grep -v '^TYPE_A_BOUNDARY_OK$' "$out/native.stdout" > "$out/missing-marker.stdout"
rc=0; "$out/type_a_reference" "$out/missing-marker.stdout" > "$out/missing-marker.check.stdout" 2> "$out/missing-marker.check.stderr" || rc=$?
printf 'missing_marker_rejected_8\t%s\n' "$rc" >> "$out/CONTROLS.tsv"
[[ $rc == 8 ]]
step input_integrity sha256sum -c "$out/INPUTS.sha256"
printf 'TYPE_A_GATE_PASS\n'
