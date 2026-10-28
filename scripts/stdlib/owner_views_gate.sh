#!/usr/bin/env bash
# Real owner-bound spans: native bytes/NUL/boundaries and lifetime/privacy controls.
set -euo pipefail
: "${SLURM_JOB_ID:?run owner view qualification under Slurm}"
: "${MADAROS_RAW_BIN:?explicit raw compiler required}"
: "${OWNER_VIEWS_COMPILER_SHA256:?pin the raw compiler SHA256}"
: "${OWNER_VIEWS_PLATFORM_SO:?pin the platform runtime path}"
: "${OWNER_VIEWS_PLATFORM_SHA256:?pin the platform runtime SHA256}"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SELF=$(realpath "$0")
OUT=${1:?fresh absolute output directory required}
case "$OUT" in /*) ;;*) exit 64;;esac
test ! -e "$OUT";mkdir -p "$OUT"
trap 'r=$?;echo "$r" > "$OUT/STATUS";exit "$r"' EXIT
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib" SOUNIO_SOUC_ENGINE=madaros
export SOUNIO_REQUIRE_COMMITTED_MADAROS=0 SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
ulimit -s524288;ulimit -c0
# The wrapper's derived verification stamp is primed before source capture.
bin/souc --version > "$OUT/COMPILER_VERSION.txt"
printf '%s  %s\n' "$OWNER_VIEWS_COMPILER_SHA256" "$MADAROS_RAW_BIN" > "$OUT/COMPILER.sha256"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.before.verify"
printf '%s  %s\n' "$OWNER_VIEWS_PLATFORM_SHA256" "$OWNER_VIEWS_PLATFORM_SO" > "$OUT/PLATFORM.sha256"
sha256sum -c "$OUT/PLATFORM.sha256" > "$OUT/PLATFORM.before.verify"
find stdlib bin scripts/lib -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE.sha256"
sha256sum CHANGELOG.md CITATION.cff "$SELF" tests/stdlib/wave3-core/owner_views_facades.sio tests/stdlib/wave3-core/core_generic_lib_facade.sio >> "$OUT/SOURCE.sha256"
for name in owner_views owner_views_mutation owner_views_close owner_views_subview_mutation owner_views_escape owner_views_utf8_close owner_views_forge owner_views_utf8_forge;do
 sha256sum "tools/unicode/probes/$name.sio" >> "$OUT/SOURCE.sha256"
done
printf 'case\tphase\texit_status\n' > "$OUT/RESULTS.tsv"
run(){ local name=$1;shift;local r=0;"$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || r=$?;printf '%s\t%s\t%s\n' "${name%-*}" "${name##*-}" "$r" >> "$OUT/RESULTS.tsv";return "$r"; }
for file in stdlib/collections/byte_view.sio stdlib/text/utf8_view.sio stdlib/collections/mod.sio stdlib/collections/lib.sio stdlib/text/mod.sio stdlib/text/lib.sio;do
 name=$(printf '%s' "$file" | tr '/.' '__')
 run "$name-check" timeout 240 bin/souc check "$file"
done
for name in owner_views owner_views_facades core_generic_lib_facade;do
 if test "$name" = owner_views;then file=tools/unicode/probes/owner_views.sio;marker=OWNER_VIEWS_OK
 elif test "$name" = owner_views_facades;then file=tests/stdlib/wave3-core/owner_views_facades.sio;marker=OWNER_VIEWS_FACADES_OK
 else file=tests/stdlib/wave3-core/core_generic_lib_facade.sio;marker=CORE_GENERIC_LIB_FACADE_OK
 fi
 run "$name-check" timeout 240 bin/souc check "$file"
 run "$name-compile" timeout 300 bin/souc compile "$file" -o "$OUT/$name.elf"
 sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
 run "$name-native" timeout 120 "$OUT/$name.elf"
 grep -Fx "$marker" "$OUT/$name-native.stdout" > "$OUT/$name.marker.verify"
done
for name in owner_views_mutation owner_views_close owner_views_subview_mutation owner_views_escape owner_views_utf8_close owner_views_forge owner_views_utf8_forge;do
 expected=E038
 if test "$name" = owner_views_escape;then expected=E091;fi
 if test "$name" = owner_views_forge || test "$name" = owner_views_utf8_forge;then expected=E259;fi
 for phase in check compile;do
  rc=0
  if test "$phase" = check;then timeout 240 bin/souc check "tools/unicode/probes/$name.sio" > "$OUT/$name-$phase.stdout" 2> "$OUT/$name-$phase.stderr" || rc=$?
  else timeout 240 bin/souc compile "tools/unicode/probes/$name.sio" -o "$OUT/$name.elf" > "$OUT/$name-$phase.stdout" 2> "$OUT/$name-$phase.stderr" || rc=$?
  fi
  printf '%s\t%s\t%s\n' "$name" "$phase" "$rc" >> "$OUT/RESULTS.tsv"
  test "$rc" = 1;test ! -e "$OUT/$name.elf"
  grep -F "error[$expected]" "$OUT/$name-$phase.stdout" > "$OUT/$name-$phase.reason.verify"
  if grep -Fq 'error[E264]' "$OUT/$name-$phase.stdout";then exit 1;fi
 done
done
sha256sum -c "$OUT/SOURCE.sha256" > "$OUT/SOURCE.final.verify"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.final.verify"
sha256sum -c "$OUT/PLATFORM.sha256" > "$OUT/PLATFORM.final.verify"
sha256sum -c "$OUT/EXECUTABLES.sha256" > "$OUT/EXECUTABLES.final.verify"
printf '%s\n' 'Selected native view/API and lifetime/privacy qualification. Payload bytes are borrowed; no allocation-free runtime claim. HeapVec/HeapDeque and primitive C-string ABI remain separate.' > "$OUT/SCOPE.txt"
echo OWNER_VIEWS_API_GATE_OK
