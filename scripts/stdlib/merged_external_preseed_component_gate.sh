#!/usr/bin/env bash
# Exact production helper, compiled by the pinned legacy seed. Type checking
# and source-built Madaros CLI consumer replay are separate qualification jobs.
set -euo pipefail
: "${SLURM_JOB_ID:?run this component gate under Slurm}"
: "${PRESEED_SEED_BIN:?pin the legacy positional compiler}"
: "${PRESEED_SEED_SHA256:?pin its SHA256}"
: "${PRESEED_AFTER_SHA256:?pin the production lower.sio}"
: "${PRESEED_IR_SHA256:?pin the production ir.sio}"
test "$#" = 4 || { echo "usage: $0 SOURCE_ROOT LOWER_AFTER IR_COMMON FRESH_ABSOLUTE_OUT" >&2;exit 64; }
SOURCE_ROOT=$(cd "$1" && pwd)
LOWER_AFTER=$(realpath "$2")
IR_COMMON=$(realpath "$3")
case "$4" in /*) ;;*) exit 64;;esac
OUT=$(realpath -m "$4")
case "$OUT" in "$SOURCE_ROOT"|"$SOURCE_ROOT"/*) echo "output must be outside the input snapshot" >&2;exit 64;;esac
test ! -e "$OUT";mkdir -p "$OUT"
trap 'r=$?;echo "$r" > "$OUT/STATUS";exit "$r"' EXIT
SELF=$(realpath "$0")
ROOT=$(cd "$(dirname "$SELF")/../.." && pwd)
FIXTURE="$ROOT/tests/stdlib/wave3-core/compiler_preseed_dependency.sio.in"
PROBE="$ROOT/tests/stdlib/wave3-core/compiler_preseed_probe.sio.in"
test -s "$SOURCE_ROOT/self-hosted/parser/ast.sio"
test -s "$FIXTURE";test -s "$PROBE"
ulimit -s524288;ulimit -c0
printf '%s  %s\n' "$PRESEED_SEED_SHA256" "$PRESEED_SEED_BIN" > "$OUT/COMPILER.sha256"
printf '%s  %s\n' "$PRESEED_AFTER_SHA256" "$LOWER_AFTER" "$PRESEED_IR_SHA256" "$IR_COMMON" > "$OUT/PRODUCTION_INPUTS.sha256"
sha256sum "$FIXTURE" "$PROBE" "$SELF" > "$OUT/GATE_INPUTS.sha256"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.before.verify"
sha256sum -c "$OUT/PRODUCTION_INPUTS.sha256" > "$OUT/PRODUCTION_INPUTS.before.verify"
mkdir "$OUT/capture"
cp "$LOWER_AFTER" "$OUT/capture/after-lower.sio"
cp "$IR_COMMON" "$OUT/capture/common-ir.sio"
test "$(grep -Fc 'if !LOWER_SKIP_USE_RECOLLECT_FOR_MERGED && head_owned.kind ==' "$OUT/capture/after-lower.sio")" = 2
sed -e 's/if !LOWER_SKIP_USE_RECOLLECT_FOR_MERGED && head_owned.kind == ItemKind::ItemFn {/if head_owned.kind == ItemKind::ItemFn {/' \
    -e 's/if !LOWER_SKIP_USE_RECOLLECT_FOR_MERGED && head_owned.kind == ItemKind::ItemImpl {/if head_owned.kind == ItemKind::ItemImpl {/' \
    "$OUT/capture/after-lower.sio" > "$OUT/capture/before-lower.sio"
diff -u "$OUT/capture/before-lower.sio" "$OUT/capture/after-lower.sio" > "$OUT/GUARDS_ONLY.patch" || test "$?" = 1
printf 'variant\tphase\texit_status\n' > "$OUT/RESULTS.tsv"
total=0
for variant in before after;do
 mkdir "$OUT/$variant"
 cp -a "$SOURCE_ROOT" "$OUT/$variant/source"
 cp "$OUT/capture/$variant-lower.sio" "$OUT/$variant/source/self-hosted/ir/lower.sio"
 cat "$PROBE" >> "$OUT/$variant/source/self-hosted/ir/lower.sio"
 cp "$OUT/capture/common-ir.sio" "$OUT/$variant/source/self-hosted/ir/ir.sio"
 cmp -n "$(wc -c < "$OUT/capture/$variant-lower.sio")" "$OUT/capture/$variant-lower.sio" "$OUT/$variant/source/self-hosted/ir/lower.sio" > "$OUT/$variant/PRODUCTION_PREFIX_IDENTICAL.verify"
 mkdir -p "$OUT/$variant/source/fixtures"
 cp "$FIXTURE" "$OUT/$variant/source/fixtures/dependency.sio"
 printf '%s\n' 'use ir::lower::{preseed_component_driver}' 'fn main()->i32 with IO,Alloc,Mut,Panic,Div {preseed_component_driver()}' > "$OUT/$variant/source/preseed_driver.sio"
 cd "$OUT/$variant/source"
 export SOUNIO_STDLIB_PATH="$PWD/self-hosted"
 find self-hosted stdlib fixtures -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/$variant/SOURCE.sha256"
 sha256sum preseed_driver.sio >> "$OUT/$variant/SOURCE.sha256"
 status=0
 timeout 900 "$PRESEED_SEED_BIN" preseed_driver.sio "$OUT/$variant/probe.elf" > "$OUT/$variant/compile.stdout" 2> "$OUT/$variant/compile.stderr" || status=$?
 printf '%s\tcompile\t%s\n' "$variant" "$status" >> "$OUT/RESULTS.tsv"
 if test "$status" = 0;then
  chmod +x "$OUT/$variant/probe.elf"
  sha256sum "$OUT/$variant/probe.elf" > "$OUT/$variant/ELF.sha256"
  timeout 180 "$OUT/$variant/probe.elf" > "$OUT/$variant/native.stdout" 2> "$OUT/$variant/native.stderr" || status=$?
  printf '%s\tnative\t%s\n' "$variant" "$status" >> "$OUT/RESULTS.tsv"
  for expected in 'MERGED_LAYOUTS 1' 'MERGED_VALUE_LAYOUTS 1' 'MERGED_FLOAT_FIELDS 1' 'MERGED_ENUM_VARIANTS 2' 'NONMERGED_FN_COUNT 2' 'NONMERGED_FLOAT_RETURNS 2' 'NONMERGED_FREE_FUNCTIONS 1' 'NONMERGED_METHODS 1' 'NONMERGED_REF_METHODS 1' 'NONMERGED_LAYOUTS 1' 'NONMERGED_FLOAT_FIELDS 1' 'NONMERGED_ENUM_VARIANTS 2';do
   grep -Fx "$expected" "$OUT/$variant/native.stdout" >> "$OUT/$variant/METADATA.verify"
  done
  if test "$variant" = before;then
   if test "$status" = 1;then grep -Fx 'MERGED_FN_COUNT 2' "$OUT/$variant/native.stdout" > "$OUT/$variant/EXPECTED_FAILURE.verify";else total=1;fi
  else
   if test "$status" = 0;then
    grep -Fx 'MERGED_FN_COUNT 0' "$OUT/$variant/native.stdout" > "$OUT/$variant/FN_COUNT.verify"
    grep -Fx PRESEED_COMPONENT_OK "$OUT/$variant/native.stdout" > "$OUT/$variant/MARKER.verify"
   else total=1
   fi
  fi
  sha256sum -c "$OUT/$variant/ELF.sha256" > "$OUT/$variant/ELF.final.verify"
 else total=1
 fi
 sha256sum -c "$OUT/$variant/SOURCE.sha256" > "$OUT/$variant/SOURCE.final.verify"
done
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.final.verify"
sha256sum -c "$OUT/PRODUCTION_INPUTS.sha256" > "$OUT/PRODUCTION_INPUTS.final.verify"
sha256sum -c "$OUT/GATE_INPUTS.sha256" > "$OUT/GATE_INPUTS.final.verify"
printf '%s\n' 'Legacy native component only. Full Madaros type/effect check and rebuilt CLI consumer replay are separate receipts.' > "$OUT/SCOPE.txt"
test "$total" = 0
echo PRESEED_COMPONENT_NATIVE_GATE_OK

