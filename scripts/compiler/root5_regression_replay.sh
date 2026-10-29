#!/usr/bin/env bash
# Preparation never submits jobs. bind/case/collect require root's explicit
# authorization after RAW_READY and the seed smoke gate.
set -euo pipefail
action=${1:-}
[[ $# -gt 0 ]] && shift
base=/scratch/sounio-stdlib-wave3-20261027
expected_raw="$base/root-integrated05/results/madaros"
expected_ready="$base/root-integrated05/RAW_READY"
raw_root="$base/root-integrated05"
sdk_inventory(){
 (cd "$1" || exit 1
  LC_ALL=C find . -printf '%y\t%m\t%p\t%l\0' | LC_ALL=C sort -z) || return 1
 return 0
}
parse_raw_ready(){
 local file=$1 expected=$2 line extra
 IFS= read -r line < "$file" || return 1
 read -r raw_sha raw extra <<< "$line" || return 1
 [[ "$raw_sha" =~ ^[0-9a-f]{64}$ && "$raw" == "$expected" && -z "${extra:-}" ]] || return 1
 printf '%s  %s\n' "$raw_sha" "$raw" | cmp - "$file" || return 1
 return 0
}
expected_seed_sha=ccfaec48d1bae176091ef1f3401c2d863c1f05284390f3c50ca9d66e94c7114f
expected_lean_sha=16cab9c8700981608469f052a96733360e822db6598221432f438d404cd2913b
die(){ echo "root5-replay: $*" >&2; exit 2; }
load_config(){
 local dir=$1
 [[ -f "$dir/BIND.ready" && -f "$dir/binding/config.argv0" ]] || die "binding absent"
 mapfile -d '' -t cfg < "$dir/binding/config.argv0"
 [[ ${#cfg[@]} == 14 ]] || die "invalid binding shape"
 raw=${cfg[0]}; raw_sha=${cfg[1]}; ready=${cfg[2]}
 historical=${cfg[3]}; current=${cfg[4]}
 build_manifest=${cfg[5]}; build_cwd=${cfg[6]}
 seed=${cfg[7]}; tools_root=${cfg[8]}; root_head=${cfg[9]}
 time=${cfg[11]}; timeout_bin=${cfg[12]}; build_job=${cfg[13]}
 [[ ${cfg[10]} == "$dir" && "$raw" == "$expected_raw" && "$ready" == "$expected_ready" ]] || die "binding path mismatch"
}
verify_binding(){
 local dir=$1
 (cd "$dir" || exit 1
  sha256sum -c binding/IMMUTABLE.sha256 || exit 1) || return 1
 (cd "$build_cwd" || exit 1
  sha256sum -c "$build_manifest" || exit 1) || return 1
 sdk_inventory "$historical" | cmp - "$dir/binding/historical.inventory.argv0" || return 1
 sdk_inventory "$current" | cmp - "$dir/binding/current.inventory.argv0" || return 1
 return 0
}
case "$action" in
 bind)
  [[ $# == 10 ]] || die "bind D RAW_READY HISTORICAL_SDK CURRENT_SDK BUILD_MANIFEST BUILD_CWD SEED TOOLS_ROOT ACTUAL_ROOT_HEAD MAIN_BUILD_JOB_ID"
  d=$1; ready=$2; historical=$3; current=$4; build_manifest=$5; build_cwd=$6; seed=$7; tools_root=$8; root_head=$9; build_job=${10}
  [[ "$d" == /* && -d "$d/input/original" && ! -e "$d/BIND.ready" && ! -e "$d/binding" ]] || die "new prepared directory required"
  [[ "$ready" == "$expected_ready" && -f "$ready" && ! -L "$ready" ]] || die "real ROOT5 RAW_READY required"
  parse_raw_ready "$ready" "$expected_raw" || die "invalid canonical RAW_READY"
  [[ "$raw_sha" =~ ^[0-9a-f]{64}$ && "$raw" == "$expected_raw" && -z "${extra:-}" && -x "$raw" && ! -L "$raw" ]] || die "invalid RAW_READY"
  [[ ${SOUNIO_ROOT5_REPLAY_AUTHORIZATION:-} == "$raw_sha" ]] || die "root authorization for this exact raw is required"
  [[ "$root_head" =~ ^[0-9a-f]{40}$ && -f "$build_manifest" && -d "$build_cwd" ]] || die "sealed build provenance required"
  [[ -d "$historical/stdlib" && -d "$current/stdlib" && -f "$tools_root/tools-env-v3.sh" ]] || die "SDK/tools missing"
  [[ "$historical" == "$base/whole-integrated01/source" && "$current" == "$base/root-integrated05/source" ]] || die "SDK roots mismatch"
  [[ $(cat "$raw_root/results/BUILD_STATUS") == 0 ]] || die "BUILD_STATUS is not zero"
  [[ -s "$raw_root/SOURCE_HEAD" && -s "$raw_root/GIT_SOURCE_TREE" ]] || die "sealed source identity absent"
  [[ "$root_head" == "$(cat "$raw_root/SOURCE_HEAD")" ]] || die "SOURCE_HEAD mismatch"
  [[ "$build_job" =~ ^[1-9][0-9]*$ ]] || die "main build job id required"
  R="$tools_root"; source "$R/tools-env-v3.sh"
  time="$P/tools/time"; timeout_bin=$(command -v timeout)
  [[ -x "$time" && -x "$timeout_bin" && -x "$CXX" ]] || die "capture tools missing"
  "$timeout_bin" 30 sacct -X -n -P -j "$build_job" --format=JobIDRaw,State,ExitCode > "$d/BUILD_TERMINAL.initial"
  terminal=0
  while IFS='|' read -r jid state exit_code remainder; do
   if [[ "$jid" == "$build_job" && "$state" == COMPLETED && "$exit_code" == 0:0 ]]; then terminal=1; fi
  done < "$d/BUILD_TERMINAL.initial"
  [[ "$terminal" == 1 ]] || die "main build is not terminal COMPLETED Exit0"
  for name in SOURCE COMPILER SEED CAPACITY; do
   guard="$raw_root/results/$name.final.verify"
   [[ -s "$guard" && ! -L "$guard" ]] || die "final build guard missing"
  done
  for guard in "$raw_root"/results/*.final.verify; do
   [[ -f "$guard" && ! -L "$guard" ]] || die "invalid final guard"
   while IFS= read -r line; do [[ "$line" == *": OK" ]] || die "final build guard failed"; done < "$guard"
  done
  printf '%s  %s\n' "$expected_seed_sha" "$seed" | sha256sum -c -
  printf '%s  %s\n' "$expected_lean_sha" "$current/self-hosted/compiler/lean_single.sio" | sha256sum -c -
  (cd "$build_cwd"; sha256sum -c "$build_manifest") > "$d/BUILD.initial.verify"
  repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  mkdir "$d/binding" "$d/harness"
  cp "${BASH_SOURCE[0]}" "$d/harness/root5_regression_replay.sh"
  cp "$repo/tests/fixtures/wave4_root5_replay/census.cpp" "$d/harness/census.cpp"
  cp "$repo/tests/fixtures/wave4_root5_replay/CASES.tsv" "$d/harness/CASES.tsv"
  cp "$repo/tests/fixtures/wave4_root5_replay/launch.array.sbatch" "$d/harness/launch.array.sbatch"
  : > "$d/binding/IMMUTABLE.sha256"
  while IFS=$'\t' read -r id family old_rc rel sha bytes; do
   [[ "$id" =~ ^[0-9]{2}$ && "$rel" == tests/stdlib/*.sio && "$sha" =~ ^[0-9a-f]{64}$ ]] || die "invalid case row"
   original="$historical/$rel"; copy="$d/input/original/$rel"
   printf '%s  %s\n' "$sha" "$original" "$sha" "$copy" | sha256sum -c -
   [[ $(wc -c < "$copy") == "$bytes" ]] || die "test byte count mismatch"
   cmp "$original" "$copy"
   sha256sum "$original" "$copy" >> "$d/binding/IMMUTABLE.sha256"
  done < "$d/harness/CASES.tsv"
  printf '%s\0' "$raw" "$raw_sha" "$ready" "$historical" "$current" "$build_manifest" "$build_cwd" "$seed" "$tools_root" "$root_head" "$d" "$time" "$timeout_bin" "$build_job" > "$d/binding/config.argv0"
  cp "$ready" "$d/binding/RAW_READY.captured"
  cp "$d/BUILD_TERMINAL.initial" "$d/binding/BUILD_TERMINAL.captured"
  cp "$raw_root/results/BUILD_STATUS" "$raw_root/SOURCE_HEAD" "$raw_root/GIT_SOURCE_TREE" "$d/binding/"
  mkdir "$d/binding/final-guards"
  cp "$raw_root"/results/*.final.verify "$d/binding/final-guards/"
  sha256sum "$raw_root/results/BUILD_STATUS" "$raw_root/SOURCE_HEAD" "$raw_root/GIT_SOURCE_TREE" "$raw_root"/results/*.final.verify >> "$d/binding/IMMUTABLE.sha256"
  printf '%s  %s\n' "$raw_sha" "$raw" > "$d/binding/RAW.sha256"
  sha256sum "$raw" "$ready" "$seed" "$build_manifest" "$tools_root/tools-env-v3.sh" "$tools_root/private-tools-v3/g++" "$time" "$timeout_bin" >> "$d/binding/IMMUTABLE.sha256"
  sdk_inventory "$historical" > "$d/binding/historical.inventory.argv0"
  sdk_inventory "$current" > "$d/binding/current.inventory.argv0"
  for sdk in "$historical" "$current"; do
   find "$sdk" -type f -print0 | sort -z | xargs -0 sha256sum >> "$d/binding/IMMUTABLE.sha256"
  done
  find "$d/harness" "$d/binding" -type f ! -name IMMUTABLE.sha256 -print0 | sort -z | xargs -0 sha256sum >> "$d/binding/IMMUTABLE.sha256"
  # No build, replay or sbatch invocation occurs here.
  touch "$d/BIND.ready"
  ;;
 case)
  [[ $# == 2 && -n ${SLURM_JOB_ID:-} ]] || die "case D ARRAY_INDEX (Slurm only)"
  d=$1; index=$2
  [[ "$index" =~ ^([0-9]|10|11)$ ]] || die "invalid array index"
  load_config "$d"
  R="$tools_root"; source "$R/tools-env-v3.sh"
  [[ -x "$time" && -x "$timeout_bin" ]] || die "capture tools missing"
  row=$(sed -n "$((index+1))p" "$d/harness/CASES.tsv")
  IFS=$'\t' read -r id family old_rc rel sha bytes <<< "$row"
  [[ -n "$rel" && "$id" == "$(printf '%02d' "$index")" ]] || die "case index mismatch"
  out="$d/results/$id"
  [[ ! -e "$out" ]] || die "case output already exists"
  mkdir -p "$out"
  collector=0
  finish(){
   local rc=$?
   trap - EXIT
   if [[ "$rc" != 0 ]]; then collector=1; fi
   if [[ -f "$out/TOOLS.sha256" ]] && ! sha256sum -c "$out/TOOLS.sha256" >"$out/TOOLS.final.verify" 2>&1; then collector=1; fi
   if ! verify_binding "$d" >"$out/INPUTS.final.verify" 2>&1; then collector=1; fi
   printf '%s\n' "$collector" > "$out/COLLECTOR_STATUS"
   date -u +'%Y-%m-%dT%H:%M:%SZ' > "$out/ENDED_UTC"
   find "$out" -type f ! -name ARTIFACTS.sha256 -print0 | sort -z | xargs -0 sha256sum > "$out/ARTIFACTS.sha256"
   if [[ "$rc" != 0 ]]; then exit "$rc"; fi
   exit "$collector"
  }
  trap finish EXIT
  verify_binding "$d" > "$out/INPUTS.before.verify" 2>&1
  sha256sum "$time" "$timeout_bin" "$R/tools-env-v3.sh" "$CXX" > "$out/TOOLS.sha256"
  printf '%s\n' "$raw_sha" > "$out/RAW_SHA256"
  printf '%s\n' "$root_head" > "$out/ROOT_HEAD"
  cp "$d/harness/CASES.tsv" "$out/CASES.tsv"
  source_file="$d/input/original/$rel"
  ulimit -s 524288
  ulimit -c 0
  unset SOUNIO_NO_REGION_RECLAIM SOUNIO_LOOP_CHECKPOINT_ELISION SOUNIO_TARGET || true
  export SOUNIO_MADAROS_NOCACHE=1 SOUNIO_MADAROS_CACHE_READONLY=1
  for sdk_kind in historical current; do
   sdk="$historical"; [[ "$sdk_kind" == current ]] && sdk="$current"
   export SOUNIO_STDLIB_PATH="$sdk/stdlib"
   for mode in 0 1; do
    export SOUNIO_LOOP_REGIONS="$mode"
    dir="$out/$sdk_kind/loop$mode"; mkdir -p "$dir"
    printf 'SOUNIO_STDLIB_PATH=%s\nSOUNIO_LOOP_REGIONS=%s\nSOUNIO_LOOP_CHECKPOINT_ELISION=unset(default)\nSOUNIO_MADAROS_NOCACHE=1\nSOUNIO_MADAROS_CACHE_READONLY=1\n' "$SOUNIO_STDLIB_PATH" "$mode" > "$dir/ENVIRONMENT.txt"
    for phase in check compile; do
     prefix="$dir/$phase"
     args=("$phase" "$source_file")
     if [[ "$phase" == compile ]]; then args+=(-o "$dir/program.elf"); fi
     wrapped=("$time" -v -o "$prefix.resource" "$timeout_bin" --kill-after=30s 1200 "$raw" "${args[@]}")
     printf '%s\0' "${wrapped[@]}" > "$prefix.argv0"
     date -u +'%Y-%m-%dT%H:%M:%SZ' > "$prefix.started"
     set +e
     (cd "$sdk"; "${wrapped[@]}") > "$prefix.stdout" 2> "$prefix.stderr"
     rc=$?
     set -e
     printf '%s\n' "$rc" > "$prefix.STATUS"
     date -u +'%Y-%m-%dT%H:%M:%SZ' > "$prefix.ended"
     # A compiler failure is data. Always capture all eight phases.
     if [[ -f "$dir/program.elf" && "$phase" == compile ]]; then
      sha256sum "$dir/program.elf" > "$dir/program.elf.sha256"
     fi
    done
   done
  done
  ;;
 collect)
  [[ $# == 1 && -n ${SLURM_JOB_ID:-} ]] || die "collect D (Slurm only)"
  d=$1; load_config "$d"
  R="$tools_root"; source "$R/tools-env-v3.sh"
  verify_binding "$d" > "$d/COLLECT.before.verify" 2>&1
  [[ ! -e "$d/CENSUS" ]] || die "collection already exists"
  mkdir "$d/CENSUS"
  "$CXX" -std=c++23 -O2 -Wall -Wextra -Werror "$d/harness/census.cpp" -o "$d/CENSUS/census" >"$d/CENSUS/build.stdout" 2>"$d/CENSUS/build.stderr"
  set +e
  "$d/CENSUS/census" "$d" > "$d/CENSUS/census.stdout" 2>"$d/CENSUS/census.stderr"
  rc=$?
  set -e
  printf '%s\n' "$rc" > "$d/CENSUS/COLLECTOR_STATUS"
  for path in "$d"/results/*/ARTIFACTS.sha256; do
   sha256sum -c "$path" >> "$d/CENSUS/ARTIFACTS.final.verify" 2>&1 || rc=1
  done
  verify_binding "$d" > "$d/COLLECT.final.verify" 2>&1 || rc=1
  printf '%s\n' "$rc" > "$d/CENSUS/FINAL_STATUS"
  exit "$rc"
  ;;
 *) die "actions: bind, case, collect; preparation does not submit jobs" ;;
esac
