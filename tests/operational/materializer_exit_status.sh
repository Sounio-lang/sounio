#!/usr/bin/env bash
# Isolated launcher regression. Compile/execute C++ ELF controls only in Slurm.
# Usage: BEFORE_ROOT AFTER_ROOT FRESH_OUTPUT; neither input tree is modified.
set -euo pipefail
[[ $# == 3 && ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ && $(id -u) == 1000 ]] || exit 65
before=$(realpath "$1");after=$(realpath "$2");out=$3
[[ ! -e "$out" ]] || { echo 'fresh output required' >&2;exit 65; }
mkdir -p "$out";out=$(realpath "$out")
for input in "$before" "$after";do case "$out/" in "$input/"*) exit 65;;esac;done
for input in "$before" "$after";do sha256sum "$input/bin/souc" "$input/bin/madaros" "$input/scripts/lib/materialize_madaros_prebuilt.sh" "$input/scripts/lib/resolve_madaros.sh";done >"$out/INPUTS.sha256"
cat >"$out/override.cpp" <<'CPP'
#include <cstdio>
#ifndef EXPECTED_EXIT
#define EXPECTED_EXIT 17
#endif
int main(){std::puts("ACTUAL_CPP23_ELF_EXECUTED");return EXPECTED_EXIT;}
CPP
g++ -std=c++23 -O2 -DEXPECTED_EXIT=17 "$out/override.cpp" -o "$out/override17.elf" >"$out/compile17.stdout" 2>"$out/compile17.stderr"
g++ -std=c++23 -O2 -DEXPECTED_EXIT=23 "$out/override.cpp" -o "$out/override23.elf" >"$out/compile23.stdout" 2>"$out/compile23.stderr"
for executable in "$out/override17.elf" "$out/override23.elf";do
 magic=$(od -An -N4 -tx1 "$executable" | tr -d " \n")
 [[ $magic == 7f454c46 ]]
 printf "%s\tELF magic %s\n" "$executable" "$magic" >>"$out/ELF_TYPES.txt"
done
sha256sum "$out/override17.elf" "$out/override23.elf" >"$out/ELFS.sha256"
printf 'version\tlauncher\tprebuilt_state\toverride\tstrict\texpected\tactual\n' >"$out/RESULTS.tsv"
case_id=0
run_case(){
 local version=$1 launcher=$2 state=$3 override=$4 strict=$5 expected=$6 root fixture rc=0
 root=$before;[[ $version != after ]] || root=$after
 fixture="$out/case-$case_id";case_id=$((case_id+1))
 mkdir -p "$fixture"/{bin,scripts/lib,home,tmp,artifacts/self-hosted}
 cp "$root/bin/souc" "$root/bin/madaros" "$fixture/bin/"
 cp "$root/scripts/lib/materialize_madaros_prebuilt.sh" "$root/scripts/lib/resolve_madaros.sh" "$fixture/scripts/lib/"
 chmod +x "$fixture/bin/souc" "$fixture/bin/madaros"
 gzip -n -c "$out/override17.elf" >"$fixture/bin/madaros-linux-x86_64.gz"
 sha256sum "$out/override17.elf" >"$fixture/bin/madaros-linux-x86_64.sha256"
 case "$state" in
  empty-gzip) : >"$fixture/bin/madaros-linux-x86_64.gz";;
  empty-sha) : >"$fixture/bin/madaros-linux-x86_64.sha256";;
  corrupt-gzip) printf 'not gzip\n' >"$fixture/bin/madaros-linux-x86_64.gz";;
  missing-gzip) rm "$fixture/bin/madaros-linux-x86_64.gz";;
  missing-sha) rm "$fixture/bin/madaros-linux-x86_64.sha256";;
  invalid-sha) printf 'not-a-sha256\n' >"$fixture/bin/madaros-linux-x86_64.sha256";;
  wrong-sha) sha256sum "$out/override23.elf" >"$fixture/bin/madaros-linux-x86_64.sha256";;
  valid) :;;
  *)exit 65;;
 esac
 local -a envargs=()
 case "$override" in
  none) :;;
  raw) envargs+=(MADAROS_RAW_BIN="$out/override17.elf");;
  alias) envargs+=(SOUNIO_MADAROS_BIN="$out/override17.elf");;
  local) cp "$out/override17.elf" "$fixture/artifacts/self-hosted/madaros";;
  different) envargs+=(MADAROS_RAW_BIN="$out/override23.elf");;
  *)exit 65;;
 esac
 if [[ $strict == 1 ]];then envargs+=(SOUNIO_REQUIRE_COMMITTED_MADAROS=1);fi
 # HOME/TMP and every resolution root are fixture-local; no host artifacts,
 # inherited compiler overrides, shell compiler stubs or active checkout edits.
 : >"$fixture/fixture.sio"
 /usr/bin/time -v -o "$fixture/resource.txt" env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C HOME="$fixture/home" TMPDIR="$fixture/tmp" SOUNIO_SOUC_ENGINE=madaros "${envargs[@]}" bash "$fixture/bin/$launcher" check "$fixture/fixture.sio" >"$fixture/stdout" 2>"$fixture/stderr" || rc=$?
 printf '%s\n' "$rc" >"$fixture/status"
 printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$version" "$launcher" "$state" "$override" "$strict" "$expected" "$rc" >>"$out/RESULTS.tsv"
 [[ $rc == "$expected" ]]
 if [[ $expected == 17 || $expected == 23 ]];then grep -Fxq ACTUAL_CPP23_ELF_EXECUTED "$fixture/stdout";else [[ ! -s "$fixture/stdout" ]];fi
}
for launcher in souc madaros;do
 for state in empty-gzip empty-sha corrupt-gzip missing-gzip missing-sha invalid-sha wrong-sha;do
  # Actual failing-before behavior: loss of78 yields0 with no compiler executed.
  run_case before "$launcher" "$state" none 0 0
  run_case after "$launcher" "$state" none 0 78
  for override in raw alias local;do
   run_case before "$launcher" "$state" "$override" 0 17
   run_case after "$launcher" "$state" "$override" 0 17
   run_case before "$launcher" "$state" "$override" 1 78
   run_case after "$launcher" "$state" "$override" 1 78
  done
 done
 # Good committed payload still executes, with or without strict provenance.
 for version in before after;do
  run_case "$version" "$launcher" valid none 0 17
  run_case "$version" "$launcher" valid none 1 17
  run_case "$version" "$launcher" valid raw 1 17
  run_case "$version" "$launcher" valid different 0 23
  run_case "$version" "$launcher" valid different 1 78
 done
done
sha256sum -c "$out/INPUTS.sha256" >"$out/INPUTS.after.verify"
sha256sum -c "$out/ELFS.sha256" >"$out/ELFS.after.verify"
printf 'MATERIALIZER EXIT STATUS REGRESSION PASS; real ELF overrides and strict provenance preserved.\n' >"$out/READY"
(cd "$out" && find . -type f ! -name '*.elf' ! -name '*.gz' ! -name FILES.sha256 -print0|sort -z|xargs -0 sha256sum >FILES.sha256)
