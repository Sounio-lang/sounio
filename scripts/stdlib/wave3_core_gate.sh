#!/usr/bin/env bash
# Native core gates. A smoke receipt and a chunk-part receipt are separate claims.
set -euo pipefail
: "${SLURM_JOB_ID:?run under Slurm}"
: "${MADAROS_RAW_BIN:?qualified raw compiler ELF}"
: "${WAVE3_COMPILER_SHA256:?pin the qualified raw compiler digest}"
: "${WAVE3_PLATFORM_SO:?qualified libstdlib_platform.so path}"
: "${WAVE3_PLATFORM_SHA256:?pin the qualified platform runtime digest}"
test "$#" -ge 2
OUT=$1
PHASE=$2
case "$OUT" in /*) ;; *) echo "output must be absolute" >&2;exit 64;; esac
case "$PHASE" in smoke) test "$#" -eq 2;; chunks) test "$#" -eq 6;; *) echo "phase must be smoke or chunks" >&2;exit 64;; esac
test ! -e "$OUT"
mkdir -p "$OUT"
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_REQUIRE_COMMITTED_MADAROS=0 SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
test "$(basename "$WAVE3_PLATFORM_SO")" = libstdlib_platform.so
export LD_LIBRARY_PATH="$(dirname "$WAVE3_PLATFORM_SO")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s  %s\n' "$WAVE3_PLATFORM_SHA256" "$WAVE3_PLATFORM_SO" > "$OUT/PLATFORM.sha256"
sha256sum -c "$OUT/PLATFORM.sha256" > "$OUT/PLATFORM.before.verify"
ulimit -s524288
ulimit -c0
printf '%s  %s\n' "$WAVE3_COMPILER_SHA256" "$MADAROS_RAW_BIN" > "$OUT/COMPILER.sha256"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.before.verify"
find stdlib tests/stdlib/wave3-core tools/unicode -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_INPUTS.sha256"
sha256sum scripts/stdlib/wave3_core_gate.sh >> "$OUT/SOURCE_INPUTS.sha256"
printf '%s\n' "$PHASE" > "$OUT/PHASE"
run(){ local name=$1;shift;local status=0;"$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || status=$?;printf '%s\t%s\n' "$name" "$status" >> "$OUT/RESULTS.tsv";return "$status";}
printf 'phase\texit_status\n' > "$OUT/RESULTS.tsv"
if test "$PHASE" = smoke;then
 echo 'FOCUSED_NATIVE_SMOKE_ONLY: no full Unicode corpus acceptance implied' > "$OUT/SCOPE.txt"
 grep -F 'pub fn byte_buffer_budget' stdlib/collections/byte_buffer.sio > "$OUT/DEPENDENCY.txt"
 for name in utf8_owner utf8_exact_page utf8_source_literal utf8_facade text_format_edges text_interpolation unicode_properties unicode_conformance unicode_collation_root;do
  source=tests/stdlib/wave3-core/$name.sio
  run "$name-check" timeout 180 bin/souc check "$source"
  run "$name-compile" timeout 300 bin/souc compile "$source" -o "$OUT/$name.elf"
  sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
  run "$name-native" timeout 300 "$OUT/$name.elf"
  marker=$(sed -n 's|^//@ expect-stdout: ||p' "$source")
  test -n "$marker"
  grep -Fx "$marker" "$OUT/$name-native.stdout" > "$OUT/$name.marker.verify"
 done
 grep -Fx '91' "$OUT/unicode_conformance-native.stdout" > "$OUT/SMOKE_ROWS.verify"
 run original-format-compile timeout 180 bin/souc compile tests/stdlib/text/test_format_e2e.sio -o "$OUT/original-format.elf"
 run original-format-native timeout 120 "$OUT/original-format.elf"
 sha256sum "$OUT/original-format.elf" >> "$OUT/EXECUTABLES.sha256"
 sha256sum -c "$OUT/EXECUTABLES.sha256" > "$OUT/EXECUTABLES.final.verify"
 echo WAVE3_CORE_NATIVE_SMOKE_OK
else
 : "${WAVE3_CORE_ELF_SHA256:?pin the actual Sounio conformance ELF}"
 : "${WAVE3_FIXTURE_MANIFEST_SHA256:?pin the complete prepared manifest}"
 : "${WAVE3_FIXTURE_HASH_FILE:?pin the prepared file-hash list}"
 : "${WAVE3_FIXTURE_HASH_ROOT:?base directory for the prepared file-hash list}"
 : "${WAVE3_FIXTURE_HASH_SHA256:?pin identity of the file-hash list}"
 ELF=$3
 FIXTURES=$4
 printf '%s  %s\n' "$WAVE3_CORE_ELF_SHA256" "$ELF" > "$OUT/EXECUTABLE.sha256"
 printf '%s  %s\n' "$WAVE3_FIXTURE_MANIFEST_SHA256" "$FIXTURES/MANIFEST.tsv" > "$OUT/MANIFEST.sha256"
 printf '%s  %s\n' "$WAVE3_FIXTURE_HASH_SHA256" "$WAVE3_FIXTURE_HASH_FILE" > "$OUT/FIXTURE_HASH_LIST.sha256"
 sha256sum -c "$OUT/EXECUTABLE.sha256" > "$OUT/ELF.before.verify"
 sha256sum -c "$OUT/MANIFEST.sha256" > "$OUT/MANIFEST.before.verify"
 sha256sum -c "$OUT/FIXTURE_HASH_LIST.sha256" > "$OUT/FIXTURE_HASH_LIST.before.verify"
 (cd "$WAVE3_FIXTURE_HASH_ROOT" && sha256sum -c "$WAVE3_FIXTURE_HASH_FILE") > "$OUT/FIXTURE_BYTES.before.verify"
 echo 'ONE PART OF AN EXPLICIT PREPARED MANIFEST. Complete-manifest collection is a separate required gate.' > "$OUT/SCOPE.txt"
 g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation tools/unicode/run_chunks.cpp -o "$OUT/run_chunks"
 g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation tools/unicode/collect_chunks.cpp -o "$OUT/collect_chunks"
 run chunks "$OUT/run_chunks" "$ELF" "$FIXTURES" "$OUT/chunks" "$5" "$6"
 sha256sum -c "$OUT/EXECUTABLE.sha256" > "$OUT/ELF.final.verify"
 sha256sum -c "$OUT/MANIFEST.sha256" > "$OUT/MANIFEST.final.verify"
 sha256sum -c "$OUT/FIXTURE_HASH_LIST.sha256" > "$OUT/FIXTURE_HASH_LIST.final.verify"
 (cd "$WAVE3_FIXTURE_HASH_ROOT" && sha256sum -c "$WAVE3_FIXTURE_HASH_FILE") > "$OUT/FIXTURE_BYTES.final.verify"
 echo WAVE3_CORE_NATIVE_PART_OK
fi
sha256sum -c "$OUT/SOURCE_INPUTS.sha256" > "$OUT/SOURCE_INPUTS.final.verify"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.final.verify"
sha256sum -c "$OUT/PLATFORM.sha256" > "$OUT/PLATFORM.final.verify"
