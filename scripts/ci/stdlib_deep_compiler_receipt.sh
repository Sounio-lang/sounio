#!/usr/bin/env bash
# Current-run compiler transfer. Never execute a downloaded compiler before verify.
set -euo pipefail
export LC_ALL=C
fail() { printf 'STDLIB_DEEP_RECEIPT_FAIL: %s\n' "$*" >&2; exit 65; }
script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")
work=$(mktemp -d "${TMPDIR:-/tmp}/stdlib-deep-receipt.XXXXXX")
trap 'rm -rf -- "$work"' EXIT

key() {
 local source=$1 engine=$2 target=$3 bootstrap=$4 toolchain=$5 inputs=$6 links=$7 value
 [[ "$source" =~ ^[0-9a-f]{40}$ ]] || fail 'source key must be a full Git SHA1'
 for value in "$bootstrap" "$toolchain" "$inputs" "$links"; do
 [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail 'input key fields must be SHA256 digests'
 done
 printf 'verified-compiler-v2\nsource=%s\nengine=%s\ntarget=%s\nbootstrap=%s\ntoolchain=%s\ninputs=%s\nlinks=%s\n' \
 "$source" "$engine" "$target" "$bootstrap" "$toolchain" "$inputs" "$links" | sha256sum | awk '{print $1}'
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

# TOOLCHAIN.txt describes the producer of the compiler bytes, not the runner
# that writes this receipt: on a build-cache hit those differ. The build writes
# COMPILER.toolchain beside its output (madaros_toolchain_manifest at build
# time, or the copy stored with the cache entry on a hit), and the receipt
# carries it. The fields that feed the output must match this checkout and
# environment; a compiler produced by another build script or with other
# SOUNIO_* settings is refused rather than relabelled.
current_build_facts() {
 local repo=$1 lib="$1/scripts/dev/madaros-cache.sh"
 [[ -f "$lib" ]] || fail 'missing scripts/dev/madaros-cache.sh for the build-environment fingerprint'
 # shellcheck source=/dev/null
 ( unset _SOUNIO_MADAROS_CACHE_LOADED; source "$lib" \
   && printf 'build_script_sha256=%s\nbuild_env_sha256=%s\n' "$(madaros_build_script_sha)" "$(madaros_env_fingerprint)" ) \
  || fail 'cannot compute the build-environment fingerprint'
}

write_toolchain() {
 local repo=$1 output=$2 compiler=$3 manifest="$3.toolchain" expected_keys
 [[ -f "$manifest" && ! -L "$manifest" ]] || fail 'compiler has no producer toolchain manifest (COMPILER.toolchain)'
 expected_keys=$'host_os\nhost_arch\nhost_kernel\nglibc\nbash\nbuild_script_sha256\nbuild_env_sha256'
 [[ "$(cut -d= -f1 "$manifest")" == "$expected_keys" ]] || fail 'producer toolchain manifest has unexpected fields'
 grep -Eq '^build_script_sha256=[0-9a-f]{64}$' "$manifest" || fail 'producer build-script digest is not a SHA256 digest'
 grep -Eq '^build_env_sha256=[0-9a-f]{64}$' "$manifest" || fail 'build-environment fingerprint is not a SHA256 digest'
 grep -Fxq "host_os=$(uname -s)" "$manifest" || fail 'compiler was produced on another operating system'
 grep -Fxq "host_arch=$(uname -m)" "$manifest" || fail 'compiler was produced on another architecture'
 current_build_facts "$repo" | cmp -s - <(tail -n 2 "$manifest") \
  || fail 'compiler was produced by another build script or build environment'
 cp -- "$manifest" "$output/TOOLCHAIN.txt"
 (cd "$output" && sha256sum TOOLCHAIN.txt > TOOLCHAIN.sha256)
}

# A symlink in the closure (main tracks scripts/dev/run_sio_test_suite.sh ->
# run_sio_test_suite_v2.sh) is an input like any other: sha256sum hashes its
# target's bytes, and SYMLINKS.txt binds the link text itself, so pointing it
# at a different file with identical bytes is still refused.
# Read the link text raw: command substitution strips trailing newlines, which
# would let a retarget to "name\n" hide behind an unchanged SYMLINKS.txt.
raw_link() {
 local target
 target=$(readlink -- "$1"; printf x) || return 1
 target=${target%x}
 target=${target%$'\n'}
 case "$target" in ''|*$'\n'*|*$'\r'*) return 1 ;; esac
 printf '%s' "$target"
}
link_ok() {
 local repo=$1 fixture=$2 target root resolved
 target=$(raw_link "$repo/$fixture") || return 1
 [[ "$target" != /* ]] || return 1
 root=$(realpath -e -- "$repo") || return 1
 resolved=$(realpath -e -- "$repo/$fixture") || return 1
 [[ "$resolved" == "$root"/* && -f "$resolved" && ! -L "$resolved" ]]
}
links() {
 local repo=$1 list=$2 output=$3 fixture
 : > "$output"
 while IFS= read -r fixture; do
  if [[ -L "$repo/$fixture" ]]; then
   printf '%s -> %s\n' "$fixture" "$(raw_link "$repo/$fixture")" >> "$output"
  fi
 done < "$list"
}

paths() {
 local repo=$1 output=$2 fixture root
 git -C "$repo" ls-files -z -co --exclude-standard -- bin scripts self-hosted stdlib > "$work/paths.nul" || fail 'cannot enumerate source closure'
 : > "$output"
 while IFS= read -r -d '' fixture; do
 case "$fixture" in *$'\n'*|*$'\r'*|*\\*) fail 'unsupported manifest filename' ;; esac
 if [[ -L "$repo/$fixture" ]]; then
  link_ok "$repo" "$fixture" || fail "source symlink must be relative and resolve to a regular file inside the checkout: $fixture"
 else
  [[ -f "$repo/$fixture" ]] || fail "source input missing or non-regular: $fixture"
 fi
 printf '%s\n' "$fixture" >> "$output"
 done < "$work/paths.nul"
 sort -o "$output" "$output"
 [[ -s "$output" ]] || fail 'empty source closure'
 # Every input, not only a final-component symlink, must resolve inside the
 # checkout: stdlib/pkg/mod.sio under a directory symlink to elsewhere escapes.
 root=$(realpath -e -- "$repo") || fail 'cannot resolve checkout root'
 (cd "$repo" && tr '\n' '\0' < "$output" | xargs -0 realpath -e -z --) > "$work/resolved.nul" \
  || fail 'cannot resolve source inputs'
 awk -v r="$root/" 'BEGIN { RS = "\0"; bad = 0 } index($0, r) != 1 { bad = 1 } END { exit bad }' "$work/resolved.nul" \
  || fail 'source input resolves outside the checkout'
}
prepare() {
 local repo=$1 compiler=$2 output=$3 expected=$4 fixture bootstrap bootstrap_sha inputs_sha toolchain_sha links_sha build_key
 check_revision "$repo" "$expected"
 [[ -f "$compiler" && -x "$compiler" ]] || fail 'fresh compiler is missing or not executable'
 [[ "$output" == /* ]] || fail 'artifact directory must be absolute'
 # An artifact directory inside a scanned root would enumerate its own receipt
 # files as inputs; resolve symlinks first so an alias cannot bypass this.
 local root resolved_output scanned
 root=$(realpath -e -- "$repo") || fail 'cannot resolve checkout root'
 resolved_output=$(realpath -m -- "$output") || fail 'cannot resolve artifact directory'
 for scanned in bin scripts self-hosted stdlib; do
  [[ "$resolved_output" != "$root/$scanned" && "$resolved_output" != "$root/$scanned"/* ]] \
   || fail "artifact directory must not be inside the scanned source root $scanned"
 done
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
 links "$repo" "$output/TRACKED_PATHS.txt" "$output/SYMLINKS.txt"
 : > "$output/INPUTS.sha256"
 while IFS= read -r fixture; do
  (cd "$repo" && sha256sum -- "$fixture") >> "$output/INPUTS.sha256"
 done < "$output/TRACKED_PATHS.txt"
 printf '%s\n' "$bootstrap" > "$output/BOOTSTRAP_PATH"
 (cd "$repo" && sha256sum -- "$bootstrap") > "$output/BOOTSTRAP.sha256"
 write_toolchain "$repo" "$output" "$compiler"
 cp -- "$compiler" "$output/madaros.elf"
 chmod 755 "$output/madaros.elf"
 (cd "$output" && sha256sum madaros.elf > COMPILER.sha256)
 inputs_sha=$(sha256sum "$output/INPUTS.sha256" | awk '{print $1}')
 toolchain_sha=$(sha256sum "$output/TOOLCHAIN.txt" | awk '{print $1}')
 bootstrap_sha=$(awk 'NR == 1 {print $1}' "$output/BOOTSTRAP.sha256")
 links_sha=$(sha256sum "$output/SYMLINKS.txt" | awk '{print $1}')
 build_key=$(key "$expected" madaros x86_64-linux "$bootstrap_sha" "$toolchain_sha" "$inputs_sha" "$links_sha")
 printf '%s\n' "$build_key" > "$output/BUILD_KEY"
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_PREPARED %s %s\n' "$expected" "$build_key"
}
verify() {
 local repo=$1 artifact=$2 expected=$3 required line bootstrap bootstrap_sha inputs_sha toolchain_sha links_sha actual_key
 check_revision "$repo" "$expected"
 for required in RECEIPT_VERSION SOURCE_GIT_SHA ENGINE TARGET BOOTSTRAP_PATH BOOTSTRAP.sha256 \
                TOOLCHAIN.txt TOOLCHAIN.sha256 BUILD_KEY TRACKED_PATHS.txt SYMLINKS.txt INPUTS.sha256 COMPILER.sha256 madaros.elf; do
 [[ -f "$artifact/$required" && ! -L "$artifact/$required" ]] || fail "missing or symlinked artifact file: $required"
 done
 printf '2\n' | cmp -s - "$artifact/RECEIPT_VERSION" || fail 'unsupported receipt version'
 printf '%s\n' "$expected" | cmp -s - "$artifact/SOURCE_GIT_SHA" || fail 'artifact source Git SHA mismatch'
 printf 'madaros\n' | cmp -s - "$artifact/ENGINE" || fail 'artifact engine mismatch'
 printf 'x86_64-linux\n' | cmp -s - "$artifact/TARGET" || fail 'artifact target mismatch'
 [[ "$(wc -l < "$artifact/COMPILER.sha256" | tr -d ' ')" == 1 ]] || fail 'compiler manifest must contain one entry'
 grep -Eq '^[0-9a-f]{64}  madaros\.elf$' "$artifact/COMPILER.sha256" || fail 'invalid compiler manifest'
 (cd "$artifact" && sha256sum -c COMPILER.sha256) || fail 'compiler hash mismatch'
 # Regenerate and compare bytes: a manifest checksumming any other unmodified
 # artifact file would pass sha256sum -c.
 (cd "$artifact" && sha256sum TOOLCHAIN.txt) | cmp -s - "$artifact/TOOLCHAIN.sha256" \
  || fail 'toolchain manifest does not checksum TOOLCHAIN.txt'
 grep -Fxq "host_os=$(uname -s)" "$artifact/TOOLCHAIN.txt" || fail 'toolchain operating system differs from consumer'
 grep -Fxq "host_arch=$(uname -m)" "$artifact/TOOLCHAIN.txt" || fail 'toolchain architecture differs from consumer'
 [[ "$(wc -l < "$artifact/BOOTSTRAP.sha256" | tr -d ' ')" == 1 ]] || fail 'bootstrap manifest must contain one entry'
 bootstrap=$(cat "$artifact/BOOTSTRAP_PATH")
 [[ "$bootstrap" != /* && -f "$repo/$bootstrap" && ! -L "$repo/$bootstrap" ]] || fail 'bootstrap path is invalid'
 [[ "$(resolve_bootstrap "$repo")" == "$bootstrap" ]] || fail 'bootstrap path differs from the producer'
 # Regenerate the manifest for the resolved bootstrap and compare bytes: a
 # manifest naming any other file, even a checked-out one, is refused.
 (cd "$repo" && sha256sum -- "$bootstrap") | cmp -s - "$artifact/BOOTSTRAP.sha256" \
  || fail 'bootstrap manifest does not match the resolved bootstrap path and bytes'
 paths "$repo" "$work/current-paths.txt"
 cmp -s "$work/current-paths.txt" "$artifact/TRACKED_PATHS.txt" || fail 'tracked input set differs'
 links "$repo" "$work/current-paths.txt" "$work/current-links.txt"
 cmp -s "$work/current-links.txt" "$artifact/SYMLINKS.txt" || fail 'source symlink set or target differs'
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
 links_sha=$(sha256sum "$artifact/SYMLINKS.txt" | awk '{print $1}')
 actual_key=$(key "$expected" madaros x86_64-linux "$bootstrap_sha" "$toolchain_sha" "$inputs_sha" "$links_sha")
 [[ "$(cat "$artifact/BUILD_KEY")" == "$actual_key" ]] || fail 'compiler build key mismatch'
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_VERIFIED %s %s\n' "$expected" "$actual_key"
}
selftest() {
 # An isolated disposable Git fixture; no project index/ref is changed.
 local repo="$work/fixture" artifact="$work/artifact" sha control rc compiler base_key changed_key value
 unset SOUC_BIN SOUNIO_SOUC_BIN
 mkdir -p "$repo"/{bin,scripts/ci,scripts/dev,self-hosted,stdlib}
 cp "$(dirname "$script")/../dev/madaros-cache.sh" "$repo/scripts/dev/madaros-cache.sh"
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
 # twin.sio has mod.sio's bytes: retargeting alias.sio to it leaves every
 # input hash unchanged, so only the SYMLINKS.txt binding can refuse it.
 printf 'stdlib\n' > "$repo/stdlib/twin.sio"
 ln -s mod.sio "$repo/stdlib/alias.sio"
 # top.sio links to a checkout file outside the scanned roots, so a twin whose
 # name differs only by a trailing newline is not itself enumerated.
 printf 'top\n' > "$repo/top.sio"
 ln -s ../top.sio "$repo/stdlib/top.sio"
 mkdir -p "$repo/stdlib/pkg" "$work/outside-pkg"
 printf 'pkg\n' > "$repo/stdlib/pkg/mod.sio"
 printf 'pkg\n' > "$work/outside-pkg/mod.sio"
 printf 'stdlib\n' > "$work/outside.sio"
 git -c init.defaultBranch=main init -q "$repo"
 git -C "$repo" add bin scripts self-hosted stdlib
 git -C "$repo" -c commit.gpgsign=false -c core.hooksPath=/dev/null -c user.name=ReceiptFixture -c user.email=receipt-fixture@example.invalid commit -qm fixture
 sha=$(git -C "$repo" rev-parse HEAD)
 compiler="$work/compiler"
 cp "$repo/bin/souc-linux-x86_64" "$compiler"
 chmod 755 "$compiler"
 # The build writes COMPILER.toolchain; the fixture compiler gets the manifest
 # the fixture's own cache library would write for it.
 producer_manifest() {
  ( unset _SOUNIO_MADAROS_CACHE_LOADED; cd "$repo" && source scripts/dev/madaros-cache.sh && madaros_toolchain_manifest )
 }
 producer_manifest > "$compiler.toolchain"
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
 base_key=$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")
 [[ "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")" == "$base_key" ]] \
  || fail 'identical build inputs produced different keys'
 for value in \
  "$("$script" key 0000000000000000000000000000000000000000 madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")" \
  "$("$script" key "$sha" lean_single x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")" \
  "$("$script" key "$sha" madaros aarch64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf e%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf e%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf f%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf e%.0s {1..64})" "$(printf f%.0s {1..64})")" \
  "$("$script" key "$sha" madaros x86_64-linux "$(printf b%.0s {1..64})" "$(printf c%.0s {1..64})" "$(printf d%.0s {1..64})" "$(printf e%.0s {1..64})")"; do
  [[ "$value" != "$base_key" ]] || fail 'build key ignored a source, engine, target, bootstrap, toolchain, input or symlink dimension'
 done
 [[ "$(grep -c 'untracked.sio' "$artifact/TRACKED_PATHS.txt")" == 1 ]] || fail 'source closure omitted an untracked compiler input'
 printf 'RECEIPT_BUILD_KEY_DIMENSIONS_PASS\n'
 # The key a real prepare writes moves with a build-environment flag and with an
 # equal-byte symlink retarget, not only with the hashed bytes.
 base_key=$(cat "$artifact/BUILD_KEY")
 cp "$compiler" "$work/compiler-env"
 SOUNIO_RECEIPT_SELFTEST_FLAG=1 producer_manifest > "$work/compiler-env.toolchain"
 SOUNIO_RECEIPT_SELFTEST_FLAG=1 "$script" prepare "$repo" "$work/compiler-env" "$work/artifact-env" "$sha" > "$work/prepare-env.log"
 [[ "$(cat "$work/artifact-env/BUILD_KEY")" != "$base_key" ]] || fail 'a build-environment flag left BUILD_KEY unchanged'
 grep -q '^build_env_sha256=' "$work/artifact-env/TOOLCHAIN.txt" || fail 'TOOLCHAIN.txt lacks the build-environment fingerprint'
 ln -sfn twin.sio "$repo/stdlib/alias.sio"
 "$script" prepare "$repo" "$compiler" "$work/artifact-link" "$sha" > "$work/prepare-link.log"
 ln -sfn mod.sio "$repo/stdlib/alias.sio"
 [[ "$(cat "$work/artifact-link/BUILD_KEY")" != "$base_key" ]] || fail 'an equal-byte symlink retarget left BUILD_KEY unchanged'
 printf 'RECEIPT_BUILD_KEY_ENV_AND_SYMLINK_PASS\n'
 # A cache hit hands over bytes built elsewhere: TOOLCHAIN.txt must name the
 # producer's runner, not this one. Before the fix the receipt re-described the
 # current runner, so this kernel line would have been replaced by uname -r.
 cp "$compiler" "$work/compiler-old"
 sed 's/^host_kernel=.*/host_kernel=producer-kernel-fixture/' "$compiler.toolchain" > "$work/compiler-old.toolchain"
 "$script" prepare "$repo" "$work/compiler-old" "$work/artifact-old" "$sha" > "$work/prepare-old.log"
 grep -Fxq 'host_kernel=producer-kernel-fixture' "$work/artifact-old/TOOLCHAIN.txt" \
  || fail 'receipt replaced the producer toolchain with the current runner'
 "$script" verify "$repo" "$work/artifact-old" "$sha" > "$work/verify-old.log"
 printf 'RECEIPT_PRODUCER_TOOLCHAIN_CARRIED\n'
 for control in missing stale-script other-env extra-field; do
  cp "$compiler" "$work/compiler-$control"
  case "$control" in
   missing) ;;
   stale-script) sed "s/^build_script_sha256=.*/build_script_sha256=$(printf 0%.0s {1..64})/" "$compiler.toolchain" > "$work/compiler-$control.toolchain" ;;
   other-env) cp "$work/compiler-env.toolchain" "$work/compiler-$control.toolchain" ;;
   extra-field) { cat "$compiler.toolchain"; printf 'note=x\n'; } > "$work/compiler-$control.toolchain" ;;
  esac
  rc=0
  "$script" prepare "$repo" "$work/compiler-$control" "$work/artifact-producer-$control" "$sha" > "$work/producer-$control.log" 2>&1 || rc=$?
  [[ "$rc" == 65 ]] || fail "producer toolchain control was accepted: $control ($rc)"
  grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/producer-$control.log" || fail "producer control lacked refusal diagnostic: $control"
  printf 'RECEIPT_PRODUCER_TOOLCHAIN_REJECTED %s\n' "$control"
 done
 ln -s "$repo/bin" "$work/bin-alias"
 for value in "$repo/bin/receipt" "$repo/stdlib/receipt" "$work/bin-alias/receipt"; do
  rc=0
  "$script" prepare "$repo" "$compiler" "$value" "$sha" > "$work/inside-root.log" 2>&1 || rc=$?
  [[ "$rc" == 65 && ! -e "$value" ]] || fail "artifact directory inside a scanned root was accepted: $value ($rc)"
 done
 printf 'RECEIPT_ARTIFACT_INSIDE_SOURCE_ROOT_REJECTED\n'
 for control in source-git-sha compiler-bit source-file source-set target toolchain missing-manifest bootstrap-manifest symlink-retarget symlink-escape toolchain-manifest dir-escape link-newline; do
  cp -a "$artifact" "$work/$control"
  case "$control" in
   source-git-sha) printf '%040d\n' 0 > "$work/$control/SOURCE_GIT_SHA" ;;
   compiler-bit) printf X >> "$work/$control/madaros.elf" ;;
   source-file) cp "$repo/stdlib/mod.sio" "$work/original-source"; printf X >> "$repo/stdlib/mod.sio" ;;
   source-set) printf 'new source\n' > "$repo/stdlib/new.sio" ;;
   target) printf 'aarch64-linux\n' > "$work/$control/TARGET" ;;
   toolchain) printf X >> "$work/$control/TOOLCHAIN.txt" ;;
   missing-manifest) rm "$work/$control/INPUTS.sha256" ;;
   # Name another checked-out file and rewrite BUILD_KEY to match it, so only
   # the bootstrap-path binding can refuse (a stale key would refuse anyway).
   bootstrap-manifest)
    (cd "$repo" && sha256sum -- scripts/build.sh) > "$work/$control/BOOTSTRAP.sha256"
    "$script" key "$sha" madaros x86_64-linux "$(awk '{print $1}' "$work/$control/BOOTSTRAP.sha256")" \
     "$(sha256sum "$work/$control/TOOLCHAIN.txt" | awk '{print $1}')" "$(sha256sum "$work/$control/INPUTS.sha256" | awk '{print $1}')" \
     "$(sha256sum "$work/$control/SYMLINKS.txt" | awk '{print $1}')" > "$work/$control/BUILD_KEY" ;;
   symlink-retarget) ln -sfn twin.sio "$repo/stdlib/alias.sio" ;;
   symlink-escape) ln -sfn ../../outside.sio "$repo/stdlib/alias.sio" ;;
   # A checksum of another unmodified file passes sha256sum -c; only the
   # regenerate-and-compare binding refuses it (BUILD_KEY is untouched).
   toolchain-manifest) (cd "$work/$control" && sha256sum TARGET) > "$work/$control/TOOLCHAIN.sha256" ;;
   # Equal bytes behind a locally excluded directory symlink to outside.
   dir-escape)
    mv "$repo/stdlib/pkg" "$work/pkg-saved"
    ln -s "$work/outside-pkg" "$repo/stdlib/pkg"
    printf 'stdlib/pkg\n' >> "$repo/.git/info/exclude" ;;
   link-newline)
    printf 'top\n' > "$repo/top.sio"$'\n'
    ln -sfn ../top.sio$'\n' "$repo/stdlib/top.sio" ;;
  esac
  rc=0
  "$script" verify "$repo" "$work/$control" "$sha" > "$work/$control.log" 2>&1 || rc=$?
  [[ "$rc" == 65 ]] || fail "tamper did not produce controlled exit65: $control ($rc)"
  grep -q '^STDLIB_DEEP_RECEIPT_FAIL:' "$work/$control.log" || fail "tamper lacked refusal diagnostic: $control"
  if [[ "$control" == source-file ]]; then cp "$work/original-source" "$repo/stdlib/mod.sio"; fi
  if [[ "$control" == source-set ]]; then rm "$repo/stdlib/new.sio"; fi
  if [[ "$control" == symlink-* ]]; then ln -sfn mod.sio "$repo/stdlib/alias.sio"; fi
  if [[ "$control" == dir-escape ]]; then
   rm "$repo/stdlib/pkg"; mv "$work/pkg-saved" "$repo/stdlib/pkg"
   sed -i '\|^stdlib/pkg$|d' "$repo/.git/info/exclude"
  fi
  if [[ "$control" == link-newline ]]; then rm "$repo/top.sio"$'\n'; ln -sfn ../top.sio "$repo/stdlib/top.sio"; fi
  printf 'RECEIPT_TAMPER_REJECTED %s\n' "$control"
 done
 "$script" verify "$repo" "$artifact" "$sha" > "$work/final-verify.log"
 printf 'STDLIB_DEEP_COMPILER_RECEIPT_SELFTEST_PASS\n'
}
case ${1:-} in
 prepare) [[ $# == 5 ]] || fail 'prepare REPO COMPILER ABSOLUTE_ARTIFACT_DIR EXPECTED_SHA'; prepare "$2" "$3" "$4" "$5" ;;
 verify) [[ $# == 4 ]] || fail 'verify REPO ARTIFACT_DIR EXPECTED_SHA'; verify "$2" "$3" "$4" ;;
 key) [[ $# == 8 ]] || fail 'key SOURCE_SHA ENGINE TARGET BOOTSTRAP_SHA TOOLCHAIN_SHA INPUTS_SHA LINKS_SHA'; key "$2" "$3" "$4" "$5" "$6" "$7" "$8" ;;
 selftest) selftest ;;
 *) fail 'expected prepare, verify or selftest' ;;
esac
