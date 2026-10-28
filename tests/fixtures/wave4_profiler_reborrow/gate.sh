#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && -n "${SLURM_JOB_ID:-}" ]] || { echo "usage inside Slurm: gate.sh RAW_COMPILER SEALED_PLATFORM_DSO FRESH_RESULTS" >&2;exit 2; }
repo=$(cd -- "$(dirname -- "$0")/../../.." && pwd)
raw=$(realpath "$1");dso=$(realpath "$2");out=$(realpath -m "$3")
[[ -x "$raw" && -s "$dso" && ! -e "$out" ]]
mkdir -p "$out/before" "$out/after"
ulimit -s 524288;ulimit -c 0
export LC_ALL=C
export SOUNIO_LOOP_REGIONS="${SOUNIO_LOOP_REGIONS:-0}"
printf 'SOUNIO_LOOP_REGIONS=%s\n' "$SOUNIO_LOOP_REGIONS" > "$out/CONFIG"
export LD_LIBRARY_PATH="$(dirname "$dso")"
sha256sum "$raw" "$dso" > "$out/TOOLS.sha256"
cd "$repo"
cmp tests/stdlib/profile/test_profiler_e2e.sio tests/fixtures/wave4_profiler_reborrow/test_profiler.original.sio.in
cp -a stdlib "$out/before/"
cp -a stdlib "$out/after/"
cp tests/fixtures/wave4_profiler_reborrow/profiler.original.sio.in "$out/before/stdlib/profile/profiler.sio"
find stdlib tests/fixtures/wave4_profiler_reborrow tests/stdlib/profile -type f -print0|sort -z|xargs -0 sha256sum > "$out/SOURCES.sha256"
for state in before after;do
 export SOUNIO_STDLIB_PATH="$out/$state/stdlib"
 rc=0
 timeout 1200 "$raw" --check tests/stdlib/profile/test_profiler_e2e.sio > "$out/$state.check.stdout" 2>&1 ||rc=$?
 printf '%s\tcheck\t%s\n' "$state" "$rc" >> "$out/VERDICTS.tsv"
 if [[ "$state" == before ]];then
  [[ "$rc" == 1 ]];grep -Fq 'error[E039]' "$out/$state.check.stdout"
  rc=0;timeout 1200 "$raw" tests/stdlib/profile/test_profiler_e2e.sio -o "$out/$state.elf" > "$out/$state.compile.stdout" 2>&1 ||rc=$?
  printf '%s\tcompile\t%s\n' "$state" "$rc" >> "$out/VERDICTS.tsv"
  [[ "$rc" == 1 && ! -e "$out/$state.elf" ]]
  grep -Fq 'error[E039]' "$out/$state.compile.stdout"
 else
  [[ "$rc" == 0 ]];grep -Fq 'check: OK' "$out/$state.check.stdout"
  rc=0;timeout 1200 "$raw" tests/stdlib/profile/test_profiler_e2e.sio -o "$out/$state.elf" > "$out/$state.compile.stdout" 2>&1 ||rc=$?
  printf '%s\tcompile\t%s\n' "$state" "$rc" >> "$out/VERDICTS.tsv"
  [[ "$rc" == 0 && -s "$out/$state.elf" ]]
  grep -Fq 'Compilation successful!' "$out/$state.compile.stdout"
  chmod +x "$out/$state.elf"
  rc=0;timeout 30 /usr/bin/time -v -o "$out/$state.resource" "$out/$state.elf" > "$out/$state.native.stdout" 2>&1 ||rc=$?
  printf '%s\tnative\t%s\n' "$state" "$rc" >> "$out/VERDICTS.tsv"
  [[ "$rc" == 0 ]];grep -Fxq PROFILE_CONTRACT_OK "$out/$state.native.stdout"
 fi
done
for name in cast reborrow implicit held_shared_mutation;do
 source=tests/fixtures/wave4_profiler_reborrow/$name.sio
 rc=0;timeout 1200 "$raw" --check "$source" > "$out/$name.check.stdout" 2>&1 ||rc=$?
 printf '%s\tcheck\t%s\n' "$name" "$rc" >> "$out/VERDICTS.tsv"
 case "$name" in
 cast) [[ "$rc" == 1 ]];grep -Fq 'error[E039]' "$out/$name.check.stdout";;
 held_shared_mutation) [[ "$rc" == 1 ]];grep -Fq 'error[E038]' "$out/$name.check.stdout";;
 *) [[ "$rc" == 0 ]]
 timeout 1200 "$raw" "$source" -o "$out/$name.elf" > "$out/$name.compile.stdout" 2>&1
 [[ -s "$out/$name.elf" ]];chmod +x "$out/$name.elf"
 timeout 30 "$out/$name.elf" > "$out/$name.native.stdout" 2>&1
 grep -Fxq REBORROW_STEP_OK "$out/$name.native.stdout";;
 esac
done
sha256sum "$out/"*.elf > "$out/EXECUTABLES.sha256"
sha256sum -c "$out/TOOLS.sha256" > "$out/TOOLS.final.verify"
sha256sum -c "$out/SOURCES.sha256" > "$out/SOURCES.final.verify"
cmp tests/stdlib/profile/test_profiler_e2e.sio tests/fixtures/wave4_profiler_reborrow/test_profiler.original.sio.in
echo PROFILER_EXPLICIT_REBORROW_API_OK
