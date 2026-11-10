#!/usr/bin/env bash
# Current-run compiler transfer; hashes bind the tracked input bytes, not a
# complete-consumption proof. Never execute a downloaded compiler before verify.
set -euo pipefail
export LC_ALL=C
fail() { printf 'STDLIB_DEEP_RECEIPT_FAIL: %s\n' "$*" >&2; exit 65; }
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
work=$(mktemp -d "${TMPDIR:-/tmp}/stdlib-deep-receipt.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
check_revision() {
 local repo=$1 expected=$2 actual
 [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || fail 'expected Git SHA must be a full SHA1'
 actual=$(git -C "$repo" rev-parse HEAD) || fail 'cannot read checkout HEAD'
 [[ "$actual" == "$expected" ]] || fail 'checkout HEAD differs from expected Git SHA'
}
paths() {
 local repo=$1 output=$2 fixture
 git -C "$repo" ls-files -z -- bin scripts self-hosted stdlib > "$work/paths.nul" || fail 'cannot enumerate tracked inputs'
 : > "$output"
 while IFS= read -r -d '' fixture; do
  case "$fixture" in *$'\n'*|*$'\r'*|*\\*) fail 'unsupported manifest filename' ;; esac
  [[ -f "$repo/$fixture" ]] || fail "tracked input missing or not a file: $fixture"
  printf '%s\n' "$fixture" >> "$output"
 done < "$work/paths.nul"
 sort -o "$output" "$output"
 [[ -s "$output" ]] || fail 'empty tracked input list'
}
prepare() {
 local repo=$1 compiler=$2 output=$3 expected=$4 fixture
 check_revision "$repo" "$expected"
 [[ -f "$compiler" && -x "$compiler" ]] || fail 'fresh compiler is missing or not executable'
 [[ "$output" == /* ]] || fail 'artifact directory must be absolute'
 mkdir -p "$output"
 [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail 'artifact directory is not empty'
 printf '1\n' > "$output/RECEIPT_VERSION"
 printf '%s\n' "$expected" > "$output/SOURCE_GIT_SHA"
 paths "$repo" "$output/TRACKED_PATHS.txt"
 : > "$output/INPUTS.sha256"
 while IFS= read -r fixture; do
  (cd "$repo" && sha256sum -- "$fixture") >> "$output/INPUTS.sha256"
 done < "$output/TRACKED_PATHS.txt"
 cp -- "$compiler" "$output/madaros.elf"
 chmod 755 "$output/madaros.elf"
 (cd "$output" && sha256sum madaros.elf > COMPILER.sha256)
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_PREPARED %s\n' "$expected"
}
verify() {
 local repo=$1 artifact=$2 expected=$3 required line
 check_revision "$repo" "$expected"
 for required in RECEIPT_VERSION SOURCE_GIT_SHA TRACKED_PATHS.txt INPUTS.sha256 COMPILER.sha256 madaros.elf; do
  [[ -f "$artifact/$required" && ! -L "$artifact/$required" ]] || fail "missing or symlinked artifact file: $required"
 done
 printf '1\n' | cmp -s - "$artifact/RECEIPT_VERSION" || fail 'unsupported receipt version'
 printf '%s\n' "$expected" | cmp -s - "$artifact/SOURCE_GIT_SHA" || fail 'artifact source Git SHA mismatch'
 [[ "$(wc -l < "$artifact/COMPILER.sha256" | tr -d ' ')" == 1 ]] || fail 'compiler manifest must contain one entry'
 grep -Eq '^[0-9a-f]{64}  madaros\.elf$' "$artifact/COMPILER.sha256" || fail 'invalid compiler manifest'
 (cd "$artifact" && sha256sum -c COMPILER.sha256) || fail 'compiler hash mismatch'
 paths "$repo" "$work/current-paths.txt"
 cmp -s "$work/current-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'tracked input set differs'
 : > "$work/manifest-paths.txt"
 while IFS= read -r line; do
  [[ "$line" =~ ^[0-9a-f]{64}'  '(.+)$ ]] || fail 'invalid source manifest line'
  printf '%s\n' "${BASH_REMATCH[1]}" >> "$work/manifest-paths.txt"
 done < "$artifact/INPUTS.sha256"
 cmp -s "$work/manifest-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'source manifest set/order differs'
 (cd "$repo" && sha256sum -c "$artifact/INPUTS.sha256") || fail 'tracked source hash mismatch'
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_VERIFIED %s\n' "$expected"
}
selftest() {
 # An isolated disposable Git fixture; no project index/ref is changed.
 local repo="$work/fixture" artifact="$work/artifact" sha control rc
 mkdir -p "$repo"/{bin,scripts,self-hosted,stdlib}
 printf 'seed\n' > "$repo/bin/seed"
 printf 'script\n' > "$repo/scripts/build.sh"
 printf 'source\n' > "$repo/self-hosted/main.sio"
 printf 'stdlib\n' > "$repo/stdlib/mod.sio"
 git -c init.defaultBranch=main init -q "$repo"
 git -C "$repo" add bin scripts self-hosted stdlib
 git -C "$repo" -c commit.gpgsign=false -c core.hooksPath=/dev/null -c user.name=ReceiptFixture -c user.email=receipt-fixture@example.invalid commit -qm fixture
 sha=$(git -C "$repo" rev-parse HEAD)
 printf 'NONEXECUTED_TEST_COMPILER_BYTES\n' > "$work/compiler"
 chmod 755 "$work/compiler"
 "$script" prepare "$repo" "$work/compiler" "$artifact" "$sha" > "$work/prepare.log"
 "$script" verify "$repo" "$artifact" "$sha" > "$work/verify.log"
 # The compiler is version-bound, not attempt-bound: a failed-only CI rerun
 # may reuse this run's earlier successful build at the identical source SHA.
 GITHUB_RUN_ATTEMPT=2 "$script" verify "$repo" "$artifact" "$sha" > "$work/same-sha-rerun.log"
 printf 'RECEIPT_SAME_SHA_REUSE_ACCEPTED\n'
 rc=0
 GITHUB_RUN_ATTEMPT=2 "$script" verify "$repo" "$artifact" 0000000000000000000000000000000000000000 > "$work/different-sha-rerun.log" 2>&1 || rc=$?
 [[ "$rc" == 65 ]] || fail 'different-SHA reuse did not produce controlled exit65'
 grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/different-sha-rerun.log" || fail 'different-SHA reuse lacked refusal diagnostic'
 printf 'RECEIPT_DIFFERENT_SHA_REUSE_REJECTED\n'
 for control in source-git-sha compiler-bit source-file missing-manifest; do
  cp -a "$artifact" "$work/$control"
  case "$control" in
   source-git-sha) printf '%040d\n' 0 > "$work/$control/SOURCE_GIT_SHA" ;;
   compiler-bit) printf X >> "$work/$control/madaros.elf" ;;
   source-file) cp "$repo/stdlib/mod.sio" "$work/original-source"; printf X >> "$repo/stdlib/mod.sio" ;;
   missing-manifest) rm "$work/$control/INPUTS.sha256" ;;
  esac
  rc=0
  "$script" verify "$repo" "$work/$control" "$sha" > "$work/$control.log" 2>&1 || rc=$?
  [[ "$rc" == 65 ]] || fail "tamper did not produce controlled exit65: $control ($rc)"
  grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/$control.log" || fail "tamper lacked refusal diagnostic: $control"
  if [[ "$control" == source-file ]]; then cp "$work/original-source" "$repo/stdlib/mod.sio"; fi
  printf 'RECEIPT_TAMPER_REJECTED %s\n' "$control"
 done
 "$script" verify "$repo" "$artifact" "$sha" > "$work/final-verify.log"
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_SELFTEST_PASS\n'
}
case ${1:-} in
 prepare) [[ $# == 5 ]] || fail 'prepare REPO COMPILER ABSOLUTE_ARTIFACT_DIR EXPECTED_SHA'; prepare "$2" "$3" "$4" "$5" ;;
 verify) [[ $# == 4 ]] || fail 'verify REPO ARTIFACT_DIR EXPECTED_SHA'; verify "$2" "$3" "$4" ;;
 selftest) selftest ;;
 *) fail 'expected prepare, verify or selftest' ;;
esac
