#!/usr/bin/env bash
# Simulated compilers only: no self-hosted builds or production cache writes.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_DIR="$(mktemp -d "${TMPDIR:-/tmp}/madaros-diagnostic-gate.XXXXXX")"
trap 'rm -rf "$TEST_DIR"' EXIT
mkdir -p "$TEST_DIR/scripts/dev" "$TEST_DIR/scripts/ci" "$TEST_DIR/self-hosted/compiler" "$TEST_DIR/stdlib"
cp "$ROOT/scripts/dev/madaros-cache.sh" "$TEST_DIR/scripts/dev/"
cp "$ROOT/scripts/dev/souc-build-lock.sh" "$TEST_DIR/scripts/dev/"
cp "$ROOT/scripts/ci/build_modular_madaros.sh" "$TEST_DIR/scripts/ci/"
printf 'fixture\n' > "$TEST_DIR/self-hosted/compiler/main.sio"
printf 'fixture\n' > "$TEST_DIR/self-hosted/compiler/lean_single.sio"
export SOUNIO_MADAROS_CACHE="$TEST_DIR/cache"
export SOUNIO_BUILD_LOCK="$TEST_DIR/test.lock"
export SOUNIO_BUILD_SLOTS=1
unset SOUNIO_STDLIB_PATH SOUNIO_MADAROS_NOCACHE SOUNIO_MADAROS_CACHE_READONLY
export MOCK_COUNT="$TEST_DIR/count"
# No shebang: the build wrapper rejects #! wrappers as seeds. Bash executes
# this ENOEXEC fixture as a shell script; it is never presented as a real ELF.
cat > "$TEST_DIR/compiler" <<'MOCK'
printf 'call\n' >> "$MOCK_COUNT"
case "${MOCK_MODE:-clean}" in
  stderr) printf 'error[E200]: undefined identifier `lost` at imported.sio:4\n' >&2 ;;
  stdout) printf 'E200 `lost` at line 4\n' ;;
  legacy) printf 'error: unknown variable lost\n' >&2 ;;
  failed) printf 'compiler failed\n' >&2; exit 23 ;;
  empty) exit 0 ;;
esac
cp "$0" "$2"
printf 'compile: fns=1\n'
MOCK
chmod +x "$TEST_DIR/compiler"
source "$TEST_DIR/scripts/dev/madaros-cache.sh"
fail() { echo "FAIL: $*" >&2; exit 1; }
call_count() { wc -l < "$MOCK_COUNT"; }
# Diagnostic scanning failures must fail closed rather than invert grep=2.
: > "$TEST_DIR/empty.log"
grep() { return 2; }
if madaros_build_log_clean "$TEST_DIR/empty.log"; then fail 'scanner failure certified clean'; fi
unset -f grep
if madaros_build_log_clean "$TEST_DIR/absent.log"; then fail 'missing log certified clean'; fi
for stage in seed madaros; do
  for mode in stderr stdout legacy; do
    export MOCK_MODE="$mode"
    key="$stage-$mode"
    out="$TEST_DIR/$key.elf"
    if madaros_cache_build_locked "$stage" "$key" "$out" "$TEST_DIR/compiler" input "$out" >"$TEST_DIR/$key.log" 2>&1; then
      fail "$key accepted rc=0 diagnostic"
    fi
    [[ ! -e "$out" && ! -e "$SOUNIO_MADAROS_CACHE/$stage/$key/artifact" ]] || fail "$key retained artifact/cache"
    grep -q 'refusing artifact' "$TEST_DIR/$key.log" || fail "$key missing refusal"
  done
  export MOCK_MODE=clean
  out="$TEST_DIR/$stage-clean.elf"
  madaros_cache_build_locked "$stage" "$stage-clean" "$out" "$TEST_DIR/compiler" input "$out" >"$TEST_DIR/clean.log" 2>&1
  [[ -s "$out" && -s "$SOUNIO_MADAROS_CACHE/$stage/$stage-clean/diagnostic.receipt" ]] || fail "$stage clean acceptance"
  before="$(call_count)"
  madaros_cache_build_locked "$stage" "$stage-clean" "$out" "$TEST_DIR/compiler" input "$out" >"$TEST_DIR/hit.log" 2>&1
  [[ "$(call_count)" == "$before" ]] || fail "$stage clean hit rebuilt"
  rm "$SOUNIO_MADAROS_CACHE/$stage/$stage-clean/diagnostic.receipt"
  madaros_cache_build_locked "$stage" "$stage-clean" "$out" "$TEST_DIR/compiler" input "$out" >"$TEST_DIR/legacy.log" 2>&1
  [[ "$(call_count)" != "$before" ]] || fail "$stage legacy cache accepted"
  grep -q 'unverified diagnostics' "$TEST_DIR/legacy.log" || fail "$stage legacy missing rebuild evidence"
  before="$(call_count)"
  printf 'e200-clean-v1 bogus\n' > "$SOUNIO_MADAROS_CACHE/$stage/$stage-clean/diagnostic.receipt"
  madaros_cache_build_locked "$stage" "$stage-clean" "$out" "$TEST_DIR/compiler" input "$out" >"$TEST_DIR/mismatch.log" 2>&1
  [[ "$(call_count)" != "$before" ]] || fail "$stage mismatched receipt accepted"
  for mode in failed empty; do
    export MOCK_MODE="$mode"
    out="$TEST_DIR/$stage-$mode.elf"
    rc=0
    madaros_cache_build_locked "$stage" "$stage-$mode" "$out" "$TEST_DIR/compiler" input "$out" >"$TEST_DIR/failure.log" 2>&1 || rc=$?
    [[ "$rc" -ne 0 && ! -e "$out" ]] || fail "$stage $mode accepted"
    if [[ "$mode" == failed ]]; then [[ "$rc" == 23 ]] || fail "$stage lost status 23"; fi
  done
done
# The real modular wrapper must propagate the helper refusal and must accept
# a clean simulated compile through the provided-seed route.
export SOUC_BIN="$TEST_DIR/compiler" MOCK_MODE=stderr
if bash "$TEST_DIR/scripts/ci/build_modular_madaros.sh" "$TEST_DIR/wrapper.elf" >"$TEST_DIR/wrapper-reject.log" 2>&1; then
  fail 'modular wrapper accepted E200'
fi
[[ ! -e "$TEST_DIR/wrapper.elf" ]] || fail 'wrapper retained rejected output'
export MOCK_MODE=clean
bash "$TEST_DIR/scripts/ci/build_modular_madaros.sh" "$TEST_DIR/wrapper.elf" >"$TEST_DIR/wrapper-clean.log" 2>&1
[[ -s "$TEST_DIR/wrapper.elf" ]] || fail 'wrapper rejected clean output'
# Exercise the wrapper's derived-seed path as well as a supplied seed.
unset SOUC_BIN SOUNIO_SOUC_BIN
mkdir -p "$TEST_DIR/bin"
cp "$TEST_DIR/compiler" "$TEST_DIR/bin/souc-linux-x86_64"
export MOCK_MODE=stderr SOUNIO_MADAROS_NOCACHE=1
if bash "$TEST_DIR/scripts/ci/build_modular_madaros.sh" "$TEST_DIR/derived.elf" >"$TEST_DIR/derived-reject.log" 2>&1; then
  fail 'seed derivation accepted E200'
fi
[[ ! -e "$TEST_DIR/derived.elf" ]] || fail 'derived wrapper retained rejected output'
export MOCK_MODE=clean
bash "$TEST_DIR/scripts/ci/build_modular_madaros.sh" "$TEST_DIR/derived.elf" >"$TEST_DIR/derived-clean.log" 2>&1
[[ -s "$TEST_DIR/derived.elf" ]] || fail 'derived wrapper rejected clean output'
echo 'PASS: seed and Madaros reject rc0 diagnostics; clean hits require receipts; legacy and mismatched receipts rebuild; scanner errors fail closed; failures preserve status; provided and derived seed wrappers reject E200 and accept clean simulations'
