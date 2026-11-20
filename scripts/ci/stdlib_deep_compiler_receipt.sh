#!/usr/bin/env bash
# Current-run compiler transfer. Never execute a downloaded compiler before verify.
set -euo pipefail
export LC_ALL=C
fail() { printf 'STDLIB_DEEP_RECEIPT_FAIL: %s\n' "$*" >&2; exit 65; }
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
work=$(mktemp -d "${TMPDIR:-/tmp}/stdlib-deep-receipt.XXXXXX")
trap 'rm -rf -- "$work"' EXIT

key() {
 local source=$1 engine=$2 target=$3 bootstrap=$4 toolchain=$5 inputs=$6 value
 [[ "$source" =~ ^[0-9a-f]{40}$ ]] || fail 'source key must be a full Git SHA1'
 for value in "$bootstrap" "$toolchain" "$inputs"; do
 [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail 'input key fields must be SHA256 digests'
 done
 printf 'verified-compiler-v2\nsource=%s\nengine=%s\ntarget=%s\nbootstrap=%s\ntoolchain=%s\ninputs=%s\n' \
 "$source" "$engine" "$target" "$bootstrap" "$toolchain" "$inputs" | sha256sum | awk '{print $1}'
}

check_revision() {
 local repo=$1 expected=$2 actual
 [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || fail 'expected Git SHA must be a full SHA1'
 actual=$(git -C "$repo" rev-parse HEAD) || fail 'cannot read checkout HEAD'
 [[ "$actual" == "$expected" ]] || fail 'checkout HEAD differs from expected Git SHA'
}

resolve_bootstrap() {
 local repo=$1 candidate
 [[ -z "${SOUC_BIN:-}${SOUNIO_SOUC_BIN:-}" ]] || fail 'custom bootstrap overrides are not allowed for this build'
 for candidate in "$repo/bin/souc-linux-x86_64" "$repo/bin/souc-lean-single-x86_64" "$repo/bin/souc"; do
 if [[ -x "$candidate" && "$(head -c2 "$candidate" 2>/dev/null)" != '#!' ]]; then
  printf '%s' "${candidate#"$repo"/}"
  return 0
 fi
 done
 fail 'could not resolve the build bootstrap ELF'
}

check_target() {
 local binary=$1 target=$2 magic elf_class data machine
 [[ "$target" == x86_64-linux ]] || fail "unsupported compiler target: $target"
 [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail 'compiler target does not match this Linux x86_64 host'
 magic=$(od -An -N4 -tx1 "$binary" | tr -d '[:space:]') || fail 'cannot read compiler ELF header'
 elf_class=$(od -An -j4 -N1 -tu1 "$binary" | tr -d '[:space:]') || fail 'cannot read compiler ELF class'
 data=$(od -An -j5 -N1 -tu1 "$binary" | tr -d '[:space:]') || fail 'cannot read compiler ELF byte order'
 machine=$(od -An -j18 -N2 -tx1 "$binary" | tr -d '[:space:]') || fail 'cannot read compiler ELF machine'
 [[ "$magic" == 7f454c46 && "$elf_class" == 2 && "$data" == 1 && "$machine" == 3e00 ]] \
 || fail 'compiler binary is not an x86_64 Linux ELF'
}

write_toolchain() {
 local repo=$1 output=$2
 {
 printf 'host_os=%s\n' "$(uname -s)"
 printf 'host_arch=%s\n' "$(uname -m)"
 printf 'host_kernel=%s\n' "$(uname -r)"
 printf 'glibc=%s\n' "$(getconf GNU_LIBC_VERSION 2>/dev/null || echo unavailable)"
 printf 'bash=%s\n' "$(bash --version | head -n1)"
 printf 'build_script_sha256=%s\n' "$(sha256sum "$repo/scripts/ci/build_modular_madaros.sh" | awk '{print $1}')"
 } > "$output/TOOLCHAIN.txt"
 (cd "$output" && sha256sum TOOLCHAIN.txt > TOOLCHAIN.sha256)
}

paths() {
 local repo=$1 output=$2 fixture
 git -C "$repo" ls-files -z -co --exclude-standard -- bin scripts self-hosted stdlib > "$work/paths.nul" || fail 'cannot enumerate source closure'
 : > "$output"
 while IFS= read -r -d '' fixture; do
 case "$fixture" in *$'\n'*|*$'\r'*|*\\*) fail 'unsupported manifest filename' ;; esac
 [[ -f "$repo/$fixture" && ! -L "$repo/$fixture" ]] || fail "source input missing, non-regular, or symlinked: $fixture"
 printf '%s\n' "$fixture" >> "$output"
 done < "$work/paths.nul"
 sort -o "$output" "$output"
 [[ -s "$output" ]] || fail 'empty source closure'
}
prepare() {
 local repo=$1 compiler=$2 output=$3 expected=$4 fixture bootstrap bootstrap_sha inputs_sha toolchain_sha build_key
 check_revision "$repo" "$expected"
 [[ -f "$compiler" && -x "$compiler" ]] || fail 'fresh compiler is missing or not executable'
 [[ "$output" == /* ]] || fail 'artifact directory must be absolute'
 mkdir -p "$output"
 [[ -z "$(find "$output" -mindepth 1 -maxdepth 1 -print -quit)" ]] || fail 'artifact directory is not empty'
 bootstrap=$(resolve_bootstrap "$repo")
 [[ -f "$repo/$bootstrap" && ! -L "$repo/$bootstrap" ]] || fail 'resolved bootstrap is not a regular file'
 check_target "$compiler" x86_64-linux
 printf '2\n' > "$output/RECEIPT_VERSION"
 printf '%s\n' "$expected" > "$output/SOURCE_GIT_SHA"
 printf 'madaros\n' > "$output/ENGINE"
 printf 'x86_64-linux\n' > "$output/TARGET"
 paths "$repo" "$output/TRACKED_PATHS.txt"
 : > "$output/INPUTS.sha256"
 while IFS= read -r fixture; do
  (cd "$repo" && sha256sum -- "$fixture") >> "$output/INPUTS.sha256"
 done < "$output/TRACKED_PATHS.txt"
 printf '%s\n' "$bootstrap" > "$output/BOOTSTRAP_PATH"
 (cd "$repo" && sha256sum -- "$bootstrap") > "$output/BOOTSTRAP.sha256"
 write_toolchain "$repo" "$output"
 cp -- "$compiler" "$output/madaros.elf"
 chmod 755 "$output/madaros.elf"
 (cd "$output" && sha256sum madaros.elf > COMPILER.sha256)
 inputs_sha=$(sha256sum "$output/INPUTS.sha256" | awk '{print $1}')
 toolchain_sha=$(sha256sum "$output/TOOLCHAIN.txt" | awk '{print $1}')
 bootstrap_sha=$(awk 'NR == 1 {print $1}' "$output/BOOTSTRAP.sha256")
 build_key=$(key "$expected" madaros x86_64-linux "$bootstrap_sha" "$toolchain_sha" "$inputs_sha")
 printf '%s\n' "$build_key" > "$output/BUILD_KEY"
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_PREPARED %s %s\n' "$expected" "$build_key"
}
verify() {
 local repo=$1 artifact=$2 expected=$3 required line bootstrap bootstrap_sha inputs_sha toolchain_sha actual_key
 check_revision "$repo" "$expected"
 for required in RECEIPT_VERSION SOURCE_GIT_SHA ENGINE TARGET BOOTSTRAP_PATH BOOTSTRAP.sha256 \
                TOOLCHAIN.txt TOOLCHAIN.sha256 BUILD_KEY TRACKED_PATHS.txt INPUTS.sha256 COMPILER.sha256 madaros.elf; do
 [[ -f "$artifact/$required" && ! -L "$artifact/$required" ]] || fail "missing or symlinked artifact file: $required"
 done
 printf '2\n' | cmp -s - "$artifact/RECEIPT_VERSION" || fail 'unsupported receipt version'
 printf '%s\n' "$expected" | cmp -s - "$artifact/SOURCE_GIT_SHA" || fail 'artifact source Git SHA mismatch'
 printf 'madaros\n' | cmp -s - "$artifact/ENGINE" || fail 'artifact engine mismatch'
 printf 'x86_64-linux\n' | cmp -s - "$artifact/TARGET" || fail 'artifact target mismatch'
 [[ "$(wc -l < "$artifact/COMPILER.sha256" | tr -d ' ')" == 1 ]] || fail 'compiler manifest must contain one entry'
 grep -Eq '^[0-9a-f]{64}  madaros\.elf$' "$artifact/COMPILER.sha256" || fail 'invalid compiler manifest'
 (cd "$artifact" && sha256sum -c COMPILER.sha256) || fail 'compiler hash mismatch'
 (cd "$artifact" && sha256sum -c TOOLCHAIN.sha256) || fail 'toolchain receipt hash mismatch'
 grep -Fxq "host_os=$(uname -s)" "$artifact/TOOLCHAIN.txt" || fail 'toolchain operating system differs from consumer'
 grep -Fxq "host_arch=$(uname -m)" "$artifact/TOOLCHAIN.txt" || fail 'toolchain architecture differs from consumer'
 [[ "$(wc -l < "$artifact/BOOTSTRAP.sha256" | tr -d ' ')" == 1 ]] || fail 'bootstrap manifest must contain one entry'
 bootstrap=$(cat "$artifact/BOOTSTRAP_PATH")
 [[ "$bootstrap" != /* && -f "$repo/$bootstrap" && ! -L "$repo/$bootstrap" ]] || fail 'bootstrap path is invalid'
 [[ "$(resolve_bootstrap "$repo")" == "$bootstrap" ]] || fail 'bootstrap path differs from the producer'
 (cd "$repo" && sha256sum -c "$artifact/BOOTSTRAP.sha256") || fail 'bootstrap hash mismatch'
 paths "$repo" "$work/current-paths.txt"
 cmp -s "$work/current-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'tracked input set differs'
 : > "$work/manifest-paths.txt"
 while IFS= read -r line; do
  [[ "$line" =~ ^[0-9a-f]{64}'  '(.+)$ ]] || fail 'invalid source manifest line'
  printf '%s\n' "${BASH_REMATCH[1]}" >> "$work/manifest-paths.txt"
 done < "$artifact/INPUTS.sha256"
 cmp -s "$work/manifest-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'source manifest set/order differs'
 (cd "$repo" && sha256sum -c "$artifact/INPUTS.sha256") || fail 'tracked source hash mismatch'
 check_target "$artifact/madaros.elf" x86_64-linux
 inputs_sha=$(sha256sum "$artifact/INPUTS.sha256" | awk '{print $1}')
 toolchain_sha=$(sha256sum "$artifact/TOOLCHAIN.txt" | awk '{print $1}')
 bootstrap_sha=$(awk 'NR == 1 {print $1}' "$artifact/BOOTSTRAP.sha256")
 actual_key=$(key "$expected" madaros x86_64-linux "$bootstrap_sha" "$toolchain_sha" "$inputs_sha")
 [[ "$(cat "$artifact/BUILD_KEY")" == "$actual_key" ]] || fail 'compiler build key mismatch'
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_VERIFIED %s %s\n' "$expected" "$actual_key"
}
selftest() {
 # An isolated disposable Git fixture; no project index/ref is changed.
 local repo="$work/fixture" artifact="$work/artifact" sha control rc compiler base_key changed_key value
 unset SOUC_BIN SOUNIO_SOUC_BIN
 mkdir -p "$repo"/{bin,scripts/ci,self-hosted,stdlib}
 python3 - "$repo/bin/souc-linux-x86_64" <<'PY'
import sys
header = bytearray(64)
header[:7] = b"\x7fELF\x02\x01\x01"
header[18:20] = b"\x3e\x00"
open(sys.argv[1], "wb").write(header)
PY
 chmod 755 "$repo/bin/souc-linux-x86_64"
 printf 'script\n' > "$repo/scripts/build.sh"
 printf 'build script\n' > "$repo/scripts/ci/build_modular_madaros.sh"
 printf 'source\n' > "$repo/self-hosted/main.sio"
 printf 'stdlib\n' > "$repo/stdlib/mod.sio"
 git -c init.defaultBranch=main init -q "$repo"
 git -C "$repo" add bin scripts self-hosted stdlib
 git -C "$repo" -c commit.gpgsign=false -c core.hooksPath=/dev/null -c user.name=ReceiptFixture -c user.email=receipt-fixture@example.invalid commit -qm fixture
 sha=$(git -C "$repo" rev-parse HEAD)
 compiler="$work/compiler"
 cp "$repo/bin/souc-linux-x86_64" "$compiler"
 chmod 755 "$compiler"
 printf 'untracked source\n' > "$repo/self-hosted/untracked.sio"
 "$script" prepare "$repo" "$compiler" "$artifact" "$sha" > "$work/prepare.log"
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
 base_key=$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})")
 [[ "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})")" == "$base_key" ]] \
  || fail 'identical build inputs produced different keys'
 for value in \
  "$("$script" key 0000000000000000000000000000000000000000 madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})")" \
  "$("$script" key "$sha" lean_single x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})")" \
  "$("$script" key "$sha" madaros aarch64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf e%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf e%.0s {1..64})" "$(printf d%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf e%.0s {1..64})")"; do
  [[ "$value" != "$base_key" ]] || fail 'build key ignored a source, engine, target, bootstrap, toolchain, or input dimension'
 done
 [[ "$(grep -c 'untracked.sio' "$artifact/TRACKED_PATHS.txt")" == 1 ]] || fail 'source closure omitted an untracked compiler input'
 printf 'RECEIPT_BUILD_KEY_DIMENSIONS_PASS\n'
 for control in source-git-sha compiler-bit source-file source-set target toolchain missing-manifest; do
  cp -a "$artifact" "$work/$control"
  case "$control" in
   source-git-sha) printf '%040d\n' 0 > "$work/$control/SOURCE_GIT_SHA" ;;
   compiler-bit) printf X >> "$work/$control/madaros.elf" ;;
   source-file) cp "$repo/stdlib/mod.sio" "$work/original-source"; printf X >> "$repo/stdlib/mod.sio" ;;
   source-set) printf 'new source\n' > "$repo/stdlib/new.sio" ;;
   target) printf 'aarch64-linux\n' > "$work/$control/TARGET" ;;
   toolchain) printf X >> "$work/$control/TOOLCHAIN.txt" ;;
   missing-manifest) rm "$work/$control/INPUTS.sha256" ;;
  esac
  rc=0
  "$script" verify "$repo" "$work/$control" "$sha" > "$work/$control.log" 2>&1 || rc=$?
  [[ "$rc" == 65 ]] || fail "tamper did not produce controlled exit65: $control ($rc)"
  grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/$control.log" || fail "tamper lacked refusal diagnostic: $control"
  if [[ "$control" == source-file ]]; then cp "$work/original-source" "$repo/stdlib/mod.sio"; fi
  if [[ "$control" == source-set ]]; then rm "$repo/stdlib/new.sio"; fi
  printf 'RECEIPT_TAMPER_REJECTED %s\n' "$control"
 done
 "$script" verify "$repo" "$artifact" "$sha" > "$work/final-verify.log"
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_SELFTEST_PASS\n'
}
case ${1:-} in
 prepare) [[ $# == 5 ]] || fail 'prepare REPO COMPILER ABSOLUTE_ARTIFACT_DIR EXPECTED_SHA'; prepare "$2" "$3" "$4" "$5" ;;
 verify) [[ $# == 4 ]] || fail 'verify REPO ARTIFACT_DIR EXPECTED_SHA'; verify "$2" "$3" "$4" ;;
 key) [[ $# == 7 ]] || fail 'key SOURCE_SHA ENGINE TARGET BOOTSTRAP_SHA TOOLCHAIN_SHA INPUTS_SHA'; key "$2" "$3" "$4" "$5" "$6" "$7" ;;
 selftest) selftest ;;
 *) fail 'expected prepare, verify or selftest' ;;
esac
