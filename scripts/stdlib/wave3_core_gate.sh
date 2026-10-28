#!/usr/bin/env bash
# Focused smoke, one corpus part and complete-manifest collection are distinct.
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]];then
 AUTHORITY=slurm
elif [[ "${GITHUB_ACTIONS:-}" == true && "${CI:-}" == true &&
        "${GITHUB_RUN_ID:-}" =~ ^[0-9]+$ && "${GITHUB_SHA:-}" =~ ^[0-9a-fA-F]{40}$ &&
        -n "${GITHUB_WORKFLOW:-}" && -n "${GITHUB_REPOSITORY:-}" ]];then
 AUTHORITY=github-actions
else
 echo 'run under Slurm or a declared official GitHub Actions job' >&2;exit 78
fi
: "${WAVE3_COMPILER_SHA256:?pin the qualified compiler ELF digest}"
: "${WAVE3_PLATFORM_SO:?qualified libstdlib_platform.so path}"
: "${WAVE3_PLATFORM_SHA256:?pin the qualified platform runtime digest}"
test "$#" -ge 2
OUT=$1;PHASE=$2
case "$OUT" in /*) ;; *) echo 'output must be absolute' >&2;exit 64;; esac
case "$PHASE" in identity|smoke) test "$#" -eq 2;;chunks) test "$#" -eq 6;;collect) test "$#" -ge 5;;*) echo 'phase must be identity, smoke, chunks or collect' >&2;exit 64;;esac
COHERENT_KIND=${WAVE3_COHERENT_KIND:-}
FAMILY=${WAVE3_FIXTURE_FAMILY:-unicode}
CORPUS_RUN_ARGS=()
CORPUS_COLLECT_ARGS=()
if test -n "$COHERENT_KIND";then
 case "$PHASE" in chunks|collect) ;;*) echo "coherent kind requires corpus chunks or collect" >&2;exit 64;;esac
 case "$COHERENT_KIND:$FAMILY" in 1:unicode|2:locale|3:locale-metadata) ;;*) echo "coherent kind does not match fixture family" >&2;exit 64;;esac
 CORPUS_RUN_ARGS=(--coherent-kind "$COHERENT_KIND")
 CORPUS_COLLECT_ARGS=(--coherent-kind "$COHERENT_KIND")
fi
test ! -e "$OUT"
mkdir -p "$OUT"
printf "%s\n" "${COHERENT_KIND:-legacy-no-user-argv}" > "$OUT/ELF_ARGUMENT_KIND"
trap 'status=$?;printf "%s\n" "$status" > "$OUT/STATUS";exit "$status"' EXIT
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
if test "$AUTHORITY" = github-actions;then
 test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
 printf 'authority\tgithub-actions\nrepository\t%s\nworkflow\t%s\nrun_id\t%s\nsource_sha\t%s\n' "$GITHUB_REPOSITORY" "$GITHUB_WORKFLOW" "$GITHUB_RUN_ID" "$GITHUB_SHA" > "$OUT/AUTHORITY.tsv"
else
 printf 'authority\tslurm\njob_id\t%s\narray_task\t%s\nnode\t%s\n' "$SLURM_JOB_ID" "${SLURM_ARRAY_TASK_ID:-}" "${SLURMD_NODENAME:-}" > "$OUT/AUTHORITY.tsv"
fi
# Refuse engine/ELF bypasses instead of silently unsetting a caller's selection.
test -z "${SOUNIO_SOUC_BIN:-}"
test -z "${SOUNIO_MADAROS_BIN:-}"
test -z "${SOUNIO_SOUC_ENGINE:-}" || test "$SOUNIO_SOUC_ENGINE" = madaros
MODE=${WAVE3_COMPILER_MODE:-raw}
case "$MODE" in
 raw)
  : "${MADAROS_RAW_BIN:?qualified raw compiler ELF}"
  export SOUNIO_REQUIRE_COMMITTED_MADAROS=0
  ;;
 strict-wrapper)
  # Materialize only the repository's tracked gzip/digest pair. A foreign
  # raw override is a rejection, even if it could compile the smoke.
  test -z "${MADAROS_RAW_BIN:-}" || test "$MADAROS_RAW_BIN" = "$ROOT/bin/madaros-linux-x86_64"
  bash scripts/lib/materialize_madaros_prebuilt.sh > "$OUT/MATERIALIZE.stdout" 2> "$OUT/MATERIALIZE.stderr"
  export MADAROS_RAW_BIN="$ROOT/bin/madaros-linux-x86_64"
  export SOUNIO_REQUIRE_COMMITTED_MADAROS=1
  sha256sum bin/madaros-linux-x86_64.gz bin/madaros-linux-x86_64.sha256 > "$OUT/COMMITTED_PACKAGE.sha256"
  ;;
 *) echo 'compiler mode must be raw or strict-wrapper' >&2;exit 64;;
esac
printf '%s\n' "$MODE" > "$OUT/COMPILER_MODE"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=madaros SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
test "$(basename "$WAVE3_PLATFORM_SO")" = libstdlib_platform.so
export LD_LIBRARY_PATH="$(dirname "$WAVE3_PLATFORM_SO")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf '%s  %s\n' "$WAVE3_PLATFORM_SHA256" "$WAVE3_PLATFORM_SO" > "$OUT/PLATFORM.sha256"
printf '%s  %s\n' "$WAVE3_COMPILER_SHA256" "$MADAROS_RAW_BIN" > "$OUT/COMPILER.sha256"
sha256sum -c "$OUT/PLATFORM.sha256" > "$OUT/PLATFORM.before.verify"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.before.verify"
readelf -h "$MADAROS_RAW_BIN" > "$OUT/COMPILER_ELF_HEADER.txt"
ulimit -s524288
ulimit -c0
find stdlib tests/stdlib/wave3-core tools/unicode -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/SOURCE_INPUTS.sha256"
sha256sum CHANGELOG.md CITATION.cff scripts/stdlib/wave3_core_gate.sh bin/souc bin/madaros scripts/lib/{resolve_madaros,materialize_madaros_prebuilt}.sh >> "$OUT/SOURCE_INPUTS.sha256"
printf '%s\n' "$PHASE" > "$OUT/PHASE"
run(){ local name=$1;shift;local status=0;"$@" > "$OUT/$name.stdout" 2> "$OUT/$name.stderr" || status=$?;printf '%s\t%s\n' "$name" "$status" >> "$OUT/RESULTS.tsv";return "$status";}
printf 'phase\texit_status\n' > "$OUT/RESULTS.tsv"
run compiler-identity bin/souc --version
if test "$PHASE" = identity;then
 echo 'WRAPPER_AND_COMPILER_IDENTITY_ONLY: no library runtime acceptance' > "$OUT/SCOPE.txt"
 echo WAVE3_CORE_IDENTITY_OK
elif test "$PHASE" = smoke;then
 echo 'FOCUSED_NATIVE_SMOKE_ONLY: no full Unicode corpus acceptance implied' > "$OUT/SCOPE.txt"
 grep -F 'pub fn byte_buffer_budget' stdlib/collections/byte_buffer.sio > "$OUT/DEPENDENCY.txt"
 for name in utf8_owner utf8_exact_page utf8_source_literal utf8_facade text_format_edges text_interpolation unicode_properties unicode_conformance unicode_collation_root unicode_locale_canonical unicode_locale_contract unicode_text_facade unicode_text_lib_facade;do
  source=tests/stdlib/wave3-core/$name.sio
  run "$name-check" timeout 180 bin/souc check "$source"
  run "$name-compile" timeout 300 bin/souc compile "$source" -o "$OUT/$name.elf"
  sha256sum "$OUT/$name.elf" >> "$OUT/EXECUTABLES.sha256"
  run "$name-native" timeout 300 "$OUT/$name.elf"
  marker=$(sed -n 's|^//@ expect-stdout: ||p' "$source")
  test -n "$marker"
  grep -Fx "$marker" "$OUT/$name-native.stdout" > "$OUT/$name.marker.verify"
 done
 grep -Fx 91 "$OUT/unicode_conformance-native.stdout" > "$OUT/SMOKE_ROWS.verify"
 run original-format-compile timeout 180 bin/souc compile tests/stdlib/text/test_format_e2e.sio -o "$OUT/original-format.elf"
 run original-format-native timeout 120 "$OUT/original-format.elf"
 sha256sum "$OUT/original-format.elf" >> "$OUT/EXECUTABLES.sha256"
 sha256sum -c "$OUT/EXECUTABLES.sha256" > "$OUT/EXECUTABLES.final.verify"
 echo WAVE3_CORE_NATIVE_SMOKE_OK
else
 : "${WAVE3_CORE_ELF_SHA256:?pin the actual Sounio conformance ELF}"
 : "${WAVE3_FIXTURE_MANIFEST_SHA256:?pin the complete prepared manifest}"
 : "${WAVE3_FIXTURE_HASH_FILE:?prepared file-hash list}"
 : "${WAVE3_FIXTURE_HASH_ROOT:?base directory for the prepared file-hash list}"
 : "${WAVE3_FIXTURE_HASH_SHA256:?pin identity of the file-hash list}"
 : "${WAVE3_CORE_SOURCE_MANIFEST:?exact manifest used when compiling this ELF}"
 : "${WAVE3_CORE_SOURCE_ROOT:?base of the ELF source manifest}"
 : "${WAVE3_CORE_SOURCE_MANIFEST_SHA256:?pin source manifest identity}"
 ELF=$3;FIXTURES=$4
 printf '%s  %s\n' "$WAVE3_CORE_ELF_SHA256" "$ELF" > "$OUT/EXECUTABLE.sha256"
 printf '%s  %s\n' "$WAVE3_FIXTURE_MANIFEST_SHA256" "$FIXTURES/MANIFEST.tsv" > "$OUT/MANIFEST.sha256"
 printf '%s  %s\n' "$WAVE3_FIXTURE_HASH_SHA256" "$WAVE3_FIXTURE_HASH_FILE" > "$OUT/FIXTURE_HASH_LIST.sha256"
 printf '%s  %s\n' "$WAVE3_CORE_SOURCE_MANIFEST_SHA256" "$WAVE3_CORE_SOURCE_MANIFEST" > "$OUT/ELF_SOURCE_MANIFEST.sha256"
 for pin in EXECUTABLE MANIFEST FIXTURE_HASH_LIST ELF_SOURCE_MANIFEST;do sha256sum -c "$OUT/$pin.sha256" > "$OUT/$pin.before.verify";done
 (cd "$WAVE3_FIXTURE_HASH_ROOT" && sha256sum -c "$WAVE3_FIXTURE_HASH_FILE") > "$OUT/FIXTURE_BYTES.before.verify"
 (cd "$WAVE3_CORE_SOURCE_ROOT" && sha256sum -c "$WAVE3_CORE_SOURCE_MANIFEST") > "$OUT/ELF_SOURCE_BYTES.before.verify"
 FAMILY=${WAVE3_FIXTURE_FAMILY:-unicode}
 case "$FAMILY" in
  unicode) RUN_SOURCE=tools/unicode/run_chunks.cpp;COLLECT_SOURCE=tools/unicode/collect_chunks.cpp;;
  locale) RUN_SOURCE=tools/unicode/run_locale_chunks.cpp;COLLECT_SOURCE=tools/unicode/collect_locale_chunks.cpp;;
  locale-metadata) RUN_SOURCE=tools/unicode/run_locale_metadata_chunks.cpp;COLLECT_SOURCE=tools/unicode/collect_locale_metadata_chunks.cpp;;
  *) echo 'fixture family must be unicode, locale or locale-metadata' >&2;exit 64;;
 esac
 printf '%s\n' "$FAMILY" > "$OUT/FIXTURE_FAMILY"
 g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation "$RUN_SOURCE" -o "$OUT/run_chunks"
 g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation "$COLLECT_SOURCE" -o "$OUT/collect_chunks"
 sha256sum "$OUT/run_chunks" "$OUT/collect_chunks" > "$OUT/TOOLS.sha256"
 if test "$PHASE" = chunks;then
  echo 'ONE PART OF AN EXPLICIT PREPARED MANIFEST. Complete-manifest collection is a separate required gate.' > "$OUT/SCOPE.txt"
  run chunks "$OUT/run_chunks" "$ELF" "$FIXTURES" "$OUT/chunks" "$5" "$6" "${CORPUS_RUN_ARGS[@]}"
  echo WAVE3_CORE_NATIVE_PART_OK
 else
  echo 'COMPLETE PREPARED MANIFEST: every declared file and row must be collected exactly once.' > "$OUT/SCOPE.txt"
  RECEIPT_FORMAT=${WAVE3_PART_RECEIPT_FORMAT:-gate}
  case "$RECEIPT_FORMAT" in gate|legacy-worker) ;;*) echo 'part receipt format must be gate or legacy-worker' >&2;exit 64;;esac
  printf '%s\n' "$RECEIPT_FORMAT" > "$OUT/PART_RECEIPT_FORMAT"
  for part in "${@:5}";do
   parent=$(cd "$part/.." && pwd)
   if test "$RECEIPT_FORMAT" = gate;then
    test "$(cat "$parent/STATUS")" = 0
    sha256sum "$parent/STATUS" >> "$OUT/PART_RECEIPTS.sha256"
   else
    test "$(cat "$parent/WORKER_STATUS")" = 0
    test "$(cat "$parent/NATIVE_STATUS")" = 0
    sha256sum "$parent/WORKER_STATUS" "$parent/NATIVE_STATUS" >> "$OUT/PART_RECEIPTS.sha256"
   fi
   sha256sum "$parent/EXECUTABLE.sha256" "$part/STATUS" "$part/RESULTS.tsv" >> "$OUT/PART_RECEIPTS.sha256"
   if test -n "$COHERENT_KIND";then sha256sum "$part/ELF_INVOCATION.tsv" >> "$OUT/PART_RECEIPTS.sha256";fi
  done
  run collect "$OUT/collect_chunks" "${CORPUS_COLLECT_ARGS[@]}" "$FIXTURES" "$WAVE3_CORE_ELF_SHA256" "$OUT/summary" "${@:5}"
  sha256sum -c "$OUT/PART_RECEIPTS.sha256" > "$OUT/PART_RECEIPTS.final.verify"
  echo WAVE3_CORE_NATIVE_COMPLETE_OK
 fi
 sha256sum -c "$OUT/TOOLS.sha256" > "$OUT/TOOLS.final.verify"
 for pin in EXECUTABLE MANIFEST FIXTURE_HASH_LIST ELF_SOURCE_MANIFEST;do sha256sum -c "$OUT/$pin.sha256" > "$OUT/$pin.final.verify";done
 (cd "$WAVE3_FIXTURE_HASH_ROOT" && sha256sum -c "$WAVE3_FIXTURE_HASH_FILE") > "$OUT/FIXTURE_BYTES.final.verify"
 (cd "$WAVE3_CORE_SOURCE_ROOT" && sha256sum -c "$WAVE3_CORE_SOURCE_MANIFEST") > "$OUT/ELF_SOURCE_BYTES.final.verify"
fi
if test "$MODE" = strict-wrapper;then sha256sum -c "$OUT/COMMITTED_PACKAGE.sha256" > "$OUT/COMMITTED_PACKAGE.final.verify";fi
sha256sum -c "$OUT/SOURCE_INPUTS.sha256" > "$OUT/SOURCE_INPUTS.final.verify"
sha256sum -c "$OUT/COMPILER.sha256" > "$OUT/COMPILER.final.verify"
sha256sum -c "$OUT/PLATFORM.sha256" > "$OUT/PLATFORM.final.verify"

