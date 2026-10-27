#!/usr/bin/env bash
# Full Unicode18 BIDI acceptance; no corpus subset/ICU whitelist mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
[[ $# == 1 && "$1" == /* && "$1" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "usage: $0 FRESH_ABSOLUTE_OUTPUT" >&2;exit 2; }
OUT="$(realpath -m -- "$1")"
[[ ! -e "$OUT" ]] || { echo "refuse existing output" >&2;exit 2; }
if [[ "${SLURM_JOB_ID:-}" =~ ^[1-9][0-9]*$ ]];then AUTHORITY="slurm:$SLURM_JOB_ID"
elif [[ "${GITHUB_ACTIONS:-}" == true && "${GITHUB_RUN_ID:-}" =~ ^[1-9][0-9]*$ && "${GITHUB_JOB:-}" =~ ^[A-Za-z0-9_.-]+$ ]];then AUTHORITY="github-actions:$GITHUB_RUN_ID:$GITHUB_JOB"
else echo "scheduled Slurm or identified official GitHub Actions context required" >&2;exit 2;fi
: "${MADAROS_RAW_BIN:?qualified raw compiler required}"
: "${STDLIB_PLATFORM_LIBRARY:?qualified platform library required}"
: "${ICU_PREFIX:?isolated ICU79.1 required}"
[[ "$MADAROS_RAW_BIN" == /* && "$STDLIB_PLATFORM_LIBRARY" == /* && "$ICU_PREFIX" == /* ]] || exit 2
RAW="$(realpath -e "$MADAROS_RAW_BIN")"
PLATFORM="$(realpath -e "$STDLIB_PLATFORM_LIBRARY")"
ICU="$(realpath -e "$ICU_PREFIX")"
[[ -x "$RAW" && -s "$PLATFORM" && "$(basename "$PLATFORM")" == libstdlib_platform.so ]] || exit 2
[[ "$(od -An -tx1 -N4 "$RAW"|tr -d ' \n')" == 7f454c46 && "$(od -An -tx1 -N4 "$PLATFORM"|tr -d ' \n')" == 7f454c46 ]] || exit 2
if [[ -f "$ICU/../SCOPE" ]] && grep -Fq "Diagnostic candidate only" "$ICU/../SCOPE";then echo "diagnostic ICU candidate cannot serve as the locked acceptance oracle" >&2;exit 2;fi
NODE="${NODE:-node}"
CXX="${CXX:-g++}"
WRAPPER="$ROOT/scripts/ci/souc-native-wrapper.sh"
BOUNDER="$ROOT/scripts/stdlib/bounded_command.mjs"
[[ -f "$WRAPPER" && -f "$BOUNDER" && -f "$ROOT/stdlib/text/data/ucd18-source.tar.gz" && -f "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256" ]] || { echo "missing mandatory source dependency" >&2;exit 2; }
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
mkdir "$OUT/upstream" "$OUT/runs"
printf '%s\n' "$AUTHORITY" > "$OUT/EXECUTION_AUTHORITY"
date -u +%FT%TZ > "$OUT/STARTED_UTC"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
export SOUNIO_SOUC_BIN="$RAW" SOUC_NATIVE_BIN="$RAW" SOUNIO_SOUC_RAW_MODE=modular
export LD_LIBRARY_PATH="$(dirname "$PLATFORM"):$ICU/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -s 524288
ulimit -c 0
finish(){
 local status=$? changed=0
 trap - EXIT
 for manifest in SOURCES TOOLS DERIVED EXECUTABLES ARTIFACTS;do
  if [[ -s "$OUT/$manifest.sha256" ]];then sha256sum -c "$OUT/$manifest.sha256" > "$OUT/$manifest.final.verify" 2>&1 || changed=1;fi
 done
 [[ "$changed" == 0 ]] || status=1
 date -u +%FT%TZ > "$OUT/ENDED_UTC"
 printf '%s\n' "$status" > "$OUT/STATUS"
 if [[ "$status" == 0 ]];then echo WAVE3_BIDI_GATE_OK;fi
 exit "$status"
}
trap finish EXIT
(
 cd "$ROOT"
 find stdlib -type f -print0
 find tests/stdlib/wave3-bidi -type f -print0
 printf '%s\0' tests/reference/wave3_bidi_cases.hpp tests/reference/wave3_bidi_generator.cpp tests/reference/wave3_bidi_reference.cpp tests/reference/wave3_bidi_diagnostic.cpp scripts/stdlib/wave3_bidi_gate.sh scripts/stdlib/bounded_command.mjs scripts/ci/souc-native-wrapper.sh
) | sort -zu | while IFS= read -r -d '' file;do sha256sum "$ROOT/$file";done > "$OUT/SOURCES.sha256"
sha256sum "$RAW" "$PLATFORM" "$ICU/lib/libicuuc.so.79.1" "$ICU/lib/libicui18n.so.79.1" "$ICU/lib/libicudata.so.79.1" "$(command -v "$NODE")" "$(command -v "$CXX")" > "$OUT/TOOLS.sha256"
"$CXX" --version > "$OUT/CXX_VERSION"
printf 'item\tphase\tstatus\n' > "$OUT/RESULTS.tsv"
: > "$OUT/EXECUTABLES.sha256"
phase(){
 local item="$1" name="$2" seconds="$3" bytes="$4" status=0
 shift 4
 "$NODE" "$BOUNDER" "$seconds" "$bytes" -- "$@" > "$OUT/$item.$name.log" 2>&1 || status=$?
 printf '%s\t%s\t%s\n' "$item" "$name" "$status" >> "$OUT/RESULTS.tsv"
 [[ "$status" == 0 ]]
}
phase corpus extract 60 1048576 tar -xzf "$ROOT/stdlib/text/data/ucd18-source.tar.gz" -C "$OUT/upstream"
(cd "$OUT/upstream";sha256sum -c "$ROOT/stdlib/text/data/UCD18-SOURCES.sha256") > "$OUT/UCD.final.verify"
UCD="$OUT/upstream/ucd-18.0.0"
phase generator compile 180 1048576 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$ROOT/tests/reference/wave3_bidi_generator.cpp" -o "$OUT/generator"
sha256sum "$OUT/generator" >> "$OUT/EXECUTABLES.sha256"
phase corpus generate 600 1048576 "$OUT/generator" "$UCD" "$OUT/corpus"
grep -Fx "BIDI18_CORPUS_GENERATED cases=861948 class_cases=770241 character_cases=91707 chunks=211" "$OUT/corpus.generate.log"
phase mutant generate 600 1048576 "$OUT/generator" --wrong-level-control "$UCD" "$OUT/mutant"
grep -Fx "BIDI_WRONG_LEVEL_CONTROL_GENERATED official_level=2 mutant_level=3 visual_order_unchanged=true" "$OUT/mutant.generate.log"
find "$OUT/corpus" "$OUT/mutant" -type f -print0 | sort -z | xargs -0 sha256sum > "$OUT/DERIVED.sha256"
sha256sum "$OUT/EXECUTION_AUTHORITY" "$OUT/CXX_VERSION" >> "$OUT/DERIVED.sha256"
failed=0
smoke_ready=0;conformance_ready=0;adversarial_ready=0;api_edges_ready=0
for fixture in smoke adversarial api_edges conformance;do
 source="$ROOT/tests/stdlib/wave3-bidi/$fixture.sio"
 if ! phase "$fixture" check 300 4194304 bash "$WRAPPER" check "$source";then failed=1;continue;fi
 if ! grep -Fxq "check: OK" "$OUT/$fixture.check.log";then failed=1;continue;fi
 if ! phase "$fixture" compile 300 4194304 bash "$WRAPPER" compile "$source" -o "$OUT/$fixture.elf";then failed=1;continue;fi
 if [[ ! -s "$OUT/$fixture.elf" || "$(od -An -tx1 -N4 "$OUT/$fixture.elf"|tr -d ' \n')" != 7f454c46 ]];then failed=1;continue;fi
 chmod +x "$OUT/$fixture.elf"
 sha256sum "$OUT/$fixture.elf" >> "$OUT/EXECUTABLES.sha256"
 printf -v "$fixture""_ready" %s 1
done
if [[ "$smoke_ready" == 1 ]];then
 if ! phase smoke native 120 1048576 "$OUT/smoke.elf";then failed=1
 elif [[ "$(grep -Fxc BIDI_SMOKE_OK "$OUT/smoke.native.log"||true)" != 1 ]];then failed=1;fi
fi
if [[ "$adversarial_ready" == 1 ]];then
 if ! phase adversarial native 120 1048576 "$OUT/adversarial.elf";then failed=1
 elif [[ "$(grep -Fxc BIDI_ADVERSARIAL_OK "$OUT/adversarial.native.log"||true)" != 1 ]];then failed=1;fi
fi
if [[ "$api_edges_ready" == 1 ]];then
 if ! phase api_edges native 120 1048576 "$OUT/api_edges.elf";then failed=1
 elif [[ "$(grep -Fxc BIDI_API_EDGES_OK "$OUT/api_edges.native.log"||true)" != 1 ]];then failed=1;fi
fi
completed=0
completed_cases=0
wrong_level_ok=0
if [[ "$conformance_ready" == 1 ]];then
 while IFS=$'\t' read -r chunk count bytes;do
  [[ "$chunk" =~ ^[0-9]{6}\.bin$ && "$count" =~ ^[1-9][0-9]*$ && "$bytes" =~ ^[1-9][0-9]*$ ]] || exit 1
  name="${chunk%.bin}"
  mkdir -p "$OUT/runs/$name/data"
  cp "$OUT/corpus/$chunk" "$OUT/runs/$name/data/current.bin"
  cmp "$OUT/corpus/$chunk" "$OUT/runs/$name/data/current.bin"
  if ! (cd "$OUT/runs/$name";phase "$name" native 180 1048576 "$OUT/conformance.elf");then failed=1;continue;fi
  if [[ "$(grep -Fxc BIDI_NATIVE_CONFORMANCE_OK "$OUT/$name.native.log"||true)" != 1 ]] || [[ "$(grep -Fxc "BIDI_CASES_OK $count" "$OUT/$name.native.log"||true)" != 1 ]];then failed=1;continue;fi
  completed=$((completed+1))
  completed_cases=$((completed_cases+count))
 done < "$OUT/corpus/CHUNKS.tsv"
 mkdir -p "$OUT/runs/wrong-level/data"
 cp "$OUT/mutant/wrong-level.bin" "$OUT/runs/wrong-level/data/current.bin"
 # Exact wrong-level rejection is mandatory even though visual order is unchanged.
 status=0
 (cd "$OUT/runs/wrong-level";"$NODE" "$BOUNDER" 120 1048576 -- "$OUT/conformance.elf") > "$OUT/wrong-level.native.log" 2>&1 || status=$?
 printf 'wrong-level\tnative-expected-1\t%s\n' "$status" >> "$OUT/RESULTS.tsv"
 if [[ "$status" != 1 ]] || ! grep -Fq BIDI_LEVEL_MISMATCH "$OUT/wrong-level.native.log" || grep -Fq BIDI_NATIVE_CONFORMANCE_OK "$OUT/wrong-level.native.log";then failed=1;else wrong_level_ok=1;fi
fi
expected="$(wc -l < "$OUT/corpus/CHUNKS.tsv")"
official_native=FAIL
if [[ "$completed" == "$expected" && "$expected" == 211 && "$completed_cases" == 861948 ]];then official_native=PASS;else failed=1;fi
printf '%s chunks=%s/211 cases=%s/861948\n' "$official_native" "$completed" "$completed_cases" > "$OUT/OFFICIAL_NATIVE_VERDICT"
printf '%s\n' "$wrong_level_ok" > "$OUT/WRONG_LEVEL_REJECTION_VERDICT"
echo "OFFICIAL_NATIVE_VERDICT $official_native chunks=$completed cases=$completed_cases"
icu_verdict=FAIL
# Independent ICU strict comparison remains mandatory; known disagreements fail.
if phase icu-reference compile 180 1048576 "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror -I"$ICU/include" "$ROOT/tests/reference/wave3_bidi_reference.cpp" -L"$ICU/lib" -licui18n -licuuc -licudata -o "$OUT/icu-reference";then
 sha256sum "$OUT/icu-reference" >> "$OUT/EXECUTABLES.sha256"
 ldd "$OUT/icu-reference" > "$OUT/ICU_BOUND_LIBRARIES"
 if ! phase icu-reference native 600 1048576 "$OUT/icu-reference" "$UCD" "$OUT/ICU_OPTIMIZED_LEVELS.tsv";then failed=1
 elif ! grep -Fq BIDI_ICU_OFFICIAL_OK "$OUT/icu-reference.native.log";then failed=1;else icu_verdict=PASS;fi
else failed=1;fi
printf '%s\n' "$icu_verdict" > "$OUT/ICU_CONFORMANCE_VERDICT"
echo "ICU_CONFORMANCE_VERDICT $icu_verdict"
find "$OUT/runs" -type f -print0 | sort -z | xargs -0 -r sha256sum > "$OUT/ARTIFACTS.sha256"
sha256sum "$OUT/RESULTS.tsv" "$OUT/OFFICIAL_NATIVE_VERDICT" "$OUT/WRONG_LEVEL_REJECTION_VERDICT" "$OUT/ICU_CONFORMANCE_VERDICT" "$OUT/"*.log >> "$OUT/ARTIFACTS.sha256"
if [[ -f "$OUT/ICU_OPTIMIZED_LEVELS.tsv" ]];then sha256sum "$OUT/ICU_OPTIMIZED_LEVELS.tsv" >> "$OUT/ARTIFACTS.sha256";fi
[[ "$failed" == 0 ]]
