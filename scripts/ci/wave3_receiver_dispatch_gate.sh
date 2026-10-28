#!/usr/bin/env bash
# Actual production specializer; no visibility edits or mock type resolver.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
: "${RECEIVER_SEED_BIN:?pin the legacy positional compiler}"
: "${RECEIVER_SEED_SHA256:?pin its SHA256}"
: "${RECEIVER_TYPECHECK_BIN:?pin a qualified raw Madaros}"
: "${RECEIVER_TYPECHECK_SHA256:?pin its SHA256}"
: "${RECEIVER_BASE_SOURCE_MANIFEST:?pin the qualified baseline manifest}"
: "${RECEIVER_AFTER_SHA256:?pin the production specializer}"
: "${RECEIVER_HEAP_SHA256:?pin the common four-effect legacy heap module}"
test "$#" = 4 || { echo "usage: $0 QUALIFIED_SOURCE AFTER_SPECIALIZER COMMON_HEAP FRESH_ABSOLUTE_OUT" >&2;exit 64; }
BASE=$(cd "$1" && pwd)
AFTER=$(realpath "$2")
HEAP=$(realpath "$3")
case "$4" in /*) ;;*) exit 64;;esac
OUT=$(realpath -m "$4")
case "$OUT" in "$BASE"|"$BASE"/*) exit 64;;esac
test ! -e "$OUT";mkdir -p "$OUT"
trap 'r=$?;echo "$r" > "$OUT/STATUS";exit "$r"' EXIT
SELF=$(realpath "$0")
ROOT=$(cd "$(dirname "$SELF")/../.." && pwd)
FIXTURES="$ROOT/tests/fixtures/wave3_receiver_dispatch"
test -s "$FIXTURES/driver.sio"
ulimit -s524288;ulimit -c0
printf '%s  %s\n' "$RECEIVER_SEED_SHA256" "$RECEIVER_SEED_BIN" "$RECEIVER_TYPECHECK_SHA256" "$RECEIVER_TYPECHECK_BIN" > "$OUT/COMPILERS.sha256"
printf '%s  %s\n' "$RECEIVER_AFTER_SHA256" "$AFTER" "$RECEIVER_HEAP_SHA256" "$HEAP" > "$OUT/PRODUCTION.sha256"
sha256sum -c "$OUT/COMPILERS.sha256" > "$OUT/COMPILERS.before.verify"
sha256sum -c "$OUT/PRODUCTION.sha256" > "$OUT/PRODUCTION.before.verify"
(cd "$BASE";sha256sum -c "$RECEIVER_BASE_SOURCE_MANIFEST") > "$OUT/BASE_SOURCE.before.verify"
printf 'e497a3b45039f2b7facc2dc6b0594e42f4c23c554ba2c8172c686bfaed7482b9  %s\n' "$BASE/self-hosted/check/specializer.sio" | sha256sum -c - > "$OUT/BEFORE_PREIMAGE.verify"
sha256sum "$SELF" > "$OUT/GATE.sha256"
find "$FIXTURES" -type f -print0|sort -z|xargs -0 sha256sum >> "$OUT/GATE.sha256"
cmp "$ROOT/tests/stdlib/collections/test_heap_vec_methods.sio" "$BASE/tests/stdlib/collections/test_heap_vec_methods.sio" > "$OUT/ORIGINAL_WITNESS_SOURCE_IDENTICAL.verify"
printf 'variant\tphase\texit_status\n' > "$OUT/RESULTS.tsv"
total=0
for variant in before after;do
 mkdir "$OUT/$variant"
 cp -a "$BASE" "$OUT/$variant/source"
 S="$OUT/$variant/source"
 cp "$HEAP" "$S/stdlib/collections/heap_vec.sio"
 if test "$variant" = after;then cp "$AFTER" "$S/self-hosted/check/specializer.sio";fi
 mkdir "$S/fixtures"
 cp "$FIXTURES/"* "$S/fixtures/"
 cp "$S/tests/stdlib/collections/test_heap_vec_methods.sio" "$S/fixtures/original_heap_methods.sio"
 cp "$HEAP" "$S/fixtures/original_heap_module.sio"
 cp "$S/fixtures/driver.sio" "$S/receiver_driver.sio"
 cmp "$S/tests/stdlib/collections/test_heap_vec_methods.sio" "$S/fixtures/original_heap_methods.sio" > "$OUT/$variant/ORIGINAL_WITNESS_IDENTICAL.verify"
 cd "$S"
 export SOUNIO_STDLIB_PATH="$S/self-hosted"
 find self-hosted stdlib fixtures -type f -print0|sort -z|xargs -0 sha256sum > "$OUT/$variant/SOURCE.sha256"
 sha256sum receiver_driver.sio >> "$OUT/$variant/SOURCE.sha256"
 if test "$variant" = after;then
  cmp "$AFTER" self-hosted/check/specializer.sio > "$OUT/PRODUCTION_IDENTICAL.verify"
  status=0
  timeout 1500 "$RECEIVER_TYPECHECK_BIN" --check receiver_driver.sio > "$OUT/typecheck.stdout" 2> "$OUT/typecheck.stderr" || status=$?
  printf '%s\ttypecheck\t%s\n' "$variant" "$status" >> "$OUT/RESULTS.tsv"
  test "$status" = 0
  grep -Fx 'run_check_mode: verdict=0' "$OUT/typecheck.stdout" > "$OUT/TYPECHECK_VERDICT.verify"
 fi
 status=0
 timeout 900 "$RECEIVER_SEED_BIN" receiver_driver.sio "$OUT/$variant/component.elf" > "$OUT/$variant/compile.stdout" 2> "$OUT/$variant/compile.stderr" || status=$?
 printf '%s\tcompile\t%s\n' "$variant" "$status" >> "$OUT/RESULTS.tsv"
 test "$status" = 0
 chmod +x "$OUT/$variant/component.elf"
 sha256sum "$OUT/$variant/component.elf" > "$OUT/$variant/ELF.sha256"
 status=0
 timeout 180 "$OUT/$variant/component.elf" > "$OUT/$variant/native.stdout" 2> "$OUT/$variant/native.stderr" || status=$?
 printf '%s\tnative\t%s\n' "$variant" "$status" >> "$OUT/RESULTS.tsv"
 if test "$variant" = before;then
  test "$status" = 1
  grep -F 'FAILURES ' "$OUT/$variant/native.stdout" > "$OUT/$variant/EXPECTED_BEFORE.verify"
 else
  test "$status" = 0
  grep -Fx RECEIVER_COMPONENT_OK "$OUT/$variant/native.stdout" > "$OUT/$variant/MARKER.verify"
  test "$(grep -Ec '^(CASE |MULTI |ORIGINAL_HEAP )' "$OUT/$variant/native.stdout")" = 30
 fi
 sha256sum -c "$OUT/$variant/SOURCE.sha256" > "$OUT/$variant/SOURCE.final.verify"
 sha256sum -c "$OUT/$variant/ELF.sha256" > "$OUT/$variant/ELF.final.verify"
done
(cd "$BASE";sha256sum -c "$RECEIVER_BASE_SOURCE_MANIFEST") > "$OUT/BASE_SOURCE.final.verify"
sha256sum -c "$OUT/COMPILERS.sha256" > "$OUT/COMPILERS.final.verify"
sha256sum -c "$OUT/PRODUCTION.sha256" > "$OUT/PRODUCTION.final.verify"
sha256sum -c "$OUT/GATE.sha256" > "$OUT/GATE.final.verify"
echo RECEIVER_PAIRED_COMPONENT_GATE_OK
