#!/usr/bin/env bash
# Qualifies pinned data and the C++23 oracle, not Sounio APIs.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo 'Unicode table verification must run in a Slurm allocation.' >&2
  exit 2
fi
if [[ $# -ne 1 || -z "${ICU_PREFIX:-}" ]]; then
  echo 'usage: ICU_PREFIX=/isolated/icu79.1 tools/unicode/check_tables.sh NEW_OUTPUT_DIRECTORY' >&2
  exit 2
fi
SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$1"
mkdir "$OUT"
OUT="$(cd "$OUT" && pwd)"
trap 'r=$?;printf "%s\n" "$r" > "$OUT/EXIT_STATUS";date -u > "$OUT/ENDED_UTC";exit "$r"' EXIT
date -u > "$OUT/STARTED_UTC"
export LD_LIBRARY_PATH="$ICU_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
GEN="$SOURCE_ROOT/tools/unicode/unicode_generate.cpp"
REF="$SOURCE_ROOT/tests/reference/wave3_unicode_reference.cpp"
DATA="$SOURCE_ROOT/stdlib/text/data"
sha256sum "$GEN" "$REF" "$DATA/unicode18.bin" "$DATA/ucd18-source.tar.gz" "$DATA/UCD18-SOURCES.sha256" "$ICU_PREFIX/lib/libicuuc.so.79.1" "$ICU_PREFIX/lib/libicui18n.so.79.1" "$ICU_PREFIX/lib/libicudata.so.79.1" > "$OUT/INPUTS.sha256"
mkdir "$OUT/source"
tar -xzf "$DATA/ucd18-source.tar.gz" -C "$OUT/source"
(cd "$OUT/source" && sha256sum -c "$DATA/UCD18-SOURCES.sha256") > "$OUT/SOURCES.verify"
g++ -std=c++23 -O2 -Wall -Wextra -Werror "$GEN" -o "$OUT/generate" > "$OUT/generate-build.stdout" 2> "$OUT/generate-build.stderr"
g++ -std=c++23 -O2 -Wall -Wextra -Werror -Wno-misleading-indentation -I"$ICU_PREFIX/include" "$REF" -L"$ICU_PREFIX/lib" -licui18n -licuuc -licudata -o "$OUT/reference" > "$OUT/reference-build.stdout" 2> "$OUT/reference-build.stderr"
"$OUT/generate" "$OUT/source/ucd-18.0.0" "$OUT/unicode18.bin" "$OUT/REPORT.tsv" > "$OUT/generate.stdout" 2> "$OUT/generate.stderr"
"$OUT/generate" "$OUT/source/ucd-18.0.0" "$OUT/repeat.bin" "$OUT/REPORT.repeat.tsv" > "$OUT/repeat.stdout" 2> "$OUT/repeat.stderr"
cmp "$OUT/unicode18.bin" "$OUT/repeat.bin"
cmp "$OUT/REPORT.tsv" "$OUT/REPORT.repeat.tsv"
cmp "$DATA/unicode18.bin" "$OUT/unicode18.bin"
"$OUT/reference" "$OUT/unicode18.bin" "$OUT/source/ucd-18.0.0/NormalizationTest.txt" > "$OUT/reference.stdout" 2> "$OUT/reference.stderr"
"$OUT/reference" --mutate "$OUT/unicode18.bin" "$OUT/wrong-category.bin" > "$OUT/mutation.stdout" 2> "$OUT/mutation.stderr"
r=0
"$OUT/reference" "$OUT/wrong-category.bin" "$OUT/source/ucd-18.0.0/NormalizationTest.txt" > "$OUT/wrong-category.stdout" 2> "$OUT/wrong-category.stderr" || r=$?
printf '%s\n' "$r" > "$OUT/MUTATION_STATUS"
test "$r" -eq 1
grep -F 'property mismatch General_Category cp=65' "$OUT/wrong-category.stderr" > "$OUT/MUTATION_REJECTED"
mkdir "$OUT/wrong-version"
printf '%s\n' 'Unsupported Unicode version' > "$OUT/wrong-version/ReadMe.txt"
r=0
"$OUT/generate" "$OUT/wrong-version" "$OUT/wrong-version.bin" "$OUT/wrong-version.tsv" > "$OUT/wrong-version.stdout" 2> "$OUT/wrong-version.stderr" || r=$?
printf '%s\n' "$r" > "$OUT/VERSION_REJECTION_STATUS"
test "$r" -eq 1
grep -F 'expected final UCD18.0.0 ReadMe' "$OUT/wrong-version.stderr" > "$OUT/VERSION_REJECTED"
sha256sum -c "$OUT/INPUTS.sha256" > "$OUT/INPUTS.final.verify"
sha256sum "$OUT/generate" "$OUT/reference" "$OUT/unicode18.bin" "$OUT/REPORT.tsv" > "$OUT/OUTPUT.sha256"
printf '%s\n' 'UNICODE_TABLE_GATE_OK' > "$OUT/STATUS"
printf '%s\n' 'Qualified scope: generated Unicode18 tables and independent ICU79.1 oracle. Sounio algorithms, embedding, segmentation, collation and BIDI remain separate gates.' > "$OUT/SCOPE.txt"
cat "$OUT/STATUS"

