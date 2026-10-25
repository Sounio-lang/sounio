#!/usr/bin/env bash
# Current-source native contour preparation/concordance. Run heavy work on Slurm.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=${1:?usage: prepared_contour_gate.sh NEW_OUTPUT_DIRECTORY}
[[ ! -e $OUT ]] || { echo 'output already exists' >&2;exit 64; }
mkdir -p "$OUT";OUT=$(cd "$OUT" && pwd)
cd "$ROOT"
: "${MADAROS_RAW_BIN:?explicit pinned raw compiler path required}"
[[ -x $MADAROS_RAW_BIN ]] || { echo "raw compiler not executable" >&2;exit 64; }
export MADAROS_RAW_BIN SOUC_BIN="$MADAROS_RAW_BIN"
source scripts/lib/resolve_souc.sh
sounio_require_souc
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
export SOUNIO_MADAROS_CACHE="$OUT/cache" TMPDIR="$OUT/tmp"
mkdir -p "$TMPDIR" "$SOUNIO_MADAROS_CACHE"
ulimit -s 524288
sha256sum "$MADAROS_RAW_BIN" > "$OUT/COMPILER.sha256"
sha256sum "$SOUC_BIN" > "$OUT/LAUNCHER.sha256"
find stdlib -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/INPUTS.sha256"
sha256sum scripts/stdlib/prepared_contour_gate.sh tests/run-pass/viz_prepared_contour.sio tests/run-pass/stdlib_deep_viz_dynamic_contour.sio tests/run-pass/stdlib_deep_viz_contour_predicate.sio tests/reference/prepared_contour_reference.cpp tests/reference/deep_viz_dynamic_contour.cpp tests/reference/deep_viz_contour_predicate.cpp >> "$OUT/INPUTS.sha256"
printf 'phase\tsource\texit_status\n' > "$OUT/RESULTS.tsv"
phase(){ local label=$1;shift;local rc=0;"$@" > "$OUT/$label.stdout" 2> "$OUT/$label.stderr" || rc=$?;printf '%s\t%s\t%s\n' "$label" "$*" "$rc" >> "$OUT/RESULTS.tsv";return "$rc"; }
finish(){ local rc=$?;trap - EXIT;printf '%s\n' "$rc" > "$OUT/STATUS";exit "$rc"; };trap finish EXIT
for lib in contour_predicate clip uncertainty dynamic_contour;do phase "$lib.check" "$SOUC_BIN" check "stdlib/viz/$lib.sio";done
for src in stdlib_deep_viz_dynamic_contour stdlib_deep_viz_contour_predicate viz_prepared_contour;do
 phase "$src.check" "$SOUC_BIN" check "tests/run-pass/$src.sio"
 phase "$src.compile" "$SOUC_BIN" compile "tests/run-pass/$src.sio" -o "$OUT/$src.elf"
 chmod +x "$OUT/$src.elf";sha256sum "$OUT/$src.elf" > "$OUT/$src.elf.sha256"
 phase "$src.native" /usr/bin/time -f 'elapsed_seconds=%e\nmax_rss_kib=%M\nexit_code=%x' -o "$OUT/$src.resource" timeout 120 "$OUT/$src.elf"
done
for ref in prepared_contour_reference deep_viz_dynamic_contour deep_viz_contour_predicate;do
 phase "$ref.build" g++ -std=c++23 -O2 -Wall -Wextra "tests/reference/$ref.cpp" -o "$OUT/$ref"
 case "$ref" in prepared_contour_reference) native=viz_prepared_contour;;deep_viz_dynamic_contour)native=stdlib_deep_viz_dynamic_contour;;*)native=stdlib_deep_viz_contour_predicate;;esac
 if [[ $ref == prepared_contour_reference ]];then phase "$ref.concordance" "$OUT/$ref" "$OUT/$native.native.stdout" "$OUT/prepared-canvas.ppm";else phase "$ref.concordance" "$OUT/$ref" "$OUT/$native.native.stdout";fi
done
# One-pixel perturbation and lost completion are negative copies, never native.
sed '0,/x1="50"/s/x1="50"/x1="51"/' "$OUT/viz_prepared_contour.native.stdout" > "$OUT/geometry-mutant.input.stdout"
[[ $(sha256sum "$OUT/geometry-mutant.input.stdout" | cut -d' ' -f1) != $(sha256sum "$OUT/viz_prepared_contour.native.stdout" | cut -d' ' -f1) ]]
if phase geometry-mutant "$OUT/prepared_contour_reference" "$OUT/geometry-mutant.input.stdout";then echo 'geometry mutant accepted' >&2;exit 1;fi
sed '/PREPARED_CONTOUR_PASS/d' "$OUT/viz_prepared_contour.native.stdout" > "$OUT/completion-mutant.input.stdout"
if phase completion-mutant "$OUT/prepared_contour_reference" "$OUT/completion-mutant.input.stdout";then echo 'completion mutant accepted' >&2;exit 1;fi
sed 's/data-contour-level="0.00000000000000e+00"/data-contour-level="1.00000000000000e+00"/g' "$OUT/viz_prepared_contour.native.stdout" > "$OUT/level-mutant.input.stdout"
if phase level-mutant "$OUT/prepared_contour_reference" "$OUT/level-mutant.input.stdout";then echo 'level provenance mutant accepted' >&2;exit 1;fi
for kind in geometry completion level;do
 [[ -s $OUT/$kind-mutant.input.stdout ]]
 if cmp -s "$OUT/$kind-mutant.input.stdout" "$OUT/viz_prepared_contour.native.stdout";then echo "vacuous $kind mutation" >&2;exit 1;fi
 case "$kind" in geometry)reason='exact saddle/flat-edge/T/L/ownership geometry';;completion)reason='completion';;level)reason='captured contour level changed after input reuse';;esac
 phase "$kind-mutant-diagnostic" grep -Fxq "$reason" "$OUT/$kind-mutant.stderr"
 sha256sum "$OUT/$kind-mutant.input.stdout" > "$OUT/$kind-mutant.input.sha256"
done
if [[ -n ${SOUNIO_SEED_COMPILER:-} ]];then
 phase strict-seed "$SOUNIO_SEED_COMPILER" tests/run-pass/viz_prepared_contour.sio "$OUT/strict-seed.elf"
 sha256sum "$SOUNIO_SEED_COMPILER" > "$OUT/SEED_COMPILER.sha256"
fi
phase inputs-unchanged sha256sum -c "$OUT/INPUTS.sha256"
echo PREPARED_CONTOUR_GATE_PASS
