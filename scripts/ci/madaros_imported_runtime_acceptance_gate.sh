#!/usr/bin/env bash
# #901: imported nominal layout identity must survive lowering and fail closed.
#
# This is a raw-ELF behavioral gate. It deliberately does not call bin/madaros:
# a wrapper can put a generated artifact outside the requested work directory,
# which makes an absent local a.out insufficient evidence that lowering failed.

set -euo pipefail
export LC_ALL=C

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
EXPECTED_RAW_SHA256="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_EXPECTED_SHA256:-}"
KEEP_WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_KEEP:-0}"
WORK="${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR:-}"
PASS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_nested_field_chain_main.sio"
MISS_SOURCE="$ROOT_DIR/tests/compiler/madaros_imported_runtime_acceptance/issue_901_known_layout_miss_main.sio"

fail() {
  echo "[madaros-imported-runtime-acceptance] FAIL: $*" >&2
  exit 1
}

portable_sha256() {
  sha256sum "$1" 2>/dev/null | awk '{print $1}' || shasum -a 256 "$1" | awk '{print $1}'
}

elf_magic() {
  od -An -tx1 -N4 "$1" 2>/dev/null | tr -d ' \n'
}

elf_u16() {
  od -An -tu2 -j "$2" -N2 "$1" 2>/dev/null | tr -d ' \n'
}

assert_executable_elf() {
  local path="$1"
  local label="$2"

  [[ -f "$path" && -s "$path" ]] || fail "$label is missing or empty: $path"
  [[ -x "$path" ]] || fail "$label is not executable: $path"
  [[ "$(elf_magic "$path")" == 7f454c46 ]] || fail "$label is not an ELF: $path"
  [[ "$(elf_u16 "$path" 16)" == 2 ]] || fail "$label is not ET_EXEC: $path"
  [[ "$(elf_u16 "$path" 18)" == 62 ]] || fail "$label is not x86-64: $path"
}

assert_no_fallback_marker() {
  local log="$1"

  if grep -Eiq \
    'native_prebundle:|source=fallback|fallback=|SELFHOST=fallback|compact modular IR table path|legacy compact IR differential enabled' \
    "$log"; then
    cat "$log" >&2
    fail "fallback or compact imported-IR marker observed in $log"
  fi
}

run_compile() {
  local label="$1"
  local source="$2"
  local output="$3"
  local log="$WORK/$label.compile.log"

  rm -f "$output"
  set +e
  (
    cd "$WORK/$label"
    exec env \
      -u MADAROS_RAW_BIN \
      -u SOUNIO_MADAROS_BIN \
      SOUNIO_STDLIB_PATH="$ROOT_DIR/stdlib" \
      "$RAW_MADAROS" --native-v2-compile "$source" "$output"
  ) >"$log" 2>&1
  CASE_RC=$?
  set -e

  # The raw native-v2 compiler emits a valid ELF without preserving +x; the
  # runtime witness must validate the binary, not the compiler's file mode.
  if [[ -e "$output" ]]; then
    chmod +x "$output"
  fi

  CASE_LOG="$log"
}

if [[ -n "${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_BIN:-}" ]]; then
  fail 'SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_BIN is no longer accepted; pass MADAROS_RAW_BIN as an executable ELF'
fi
[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit Madaros ELF'
assert_executable_elf "$RAW_MADAROS" 'Madaros input'
RAW_MADAROS="$(cd "$(dirname "$RAW_MADAROS")" && pwd)/$(basename "$RAW_MADAROS")"
[[ -f "$PASS_SOURCE" ]] || fail "positive witness is missing: $PASS_SOURCE"
[[ -f "$MISS_SOURCE" ]] || fail "negative witness is missing: $MISS_SOURCE"

RAW_SHA256="$(portable_sha256 "$RAW_MADAROS")"
if [[ -n "$EXPECTED_RAW_SHA256" && "$RAW_SHA256" != "$EXPECTED_RAW_SHA256" ]]; then
  fail "raw ELF SHA-256 mismatch: expected=$EXPECTED_RAW_SHA256 actual=$RAW_SHA256"
fi

if [[ -n "${SOUNIO_MADAROS_IMPORTED_RUNTIME_ACCEPTANCE_DIR:-}" ]]; then
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir -p "$WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-imported-runtime-acceptance.XXXXXX)"
fi
mkdir -p "$WORK/pass" "$WORK/miss"
if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

PASS_ELF="$WORK/pass/issue_901_nested_field_chain.elf"
run_compile pass "$PASS_SOURCE" "$PASS_ELF"
[[ "$CASE_RC" -eq 0 ]] || {
  cat "$CASE_LOG" >&2
  fail "typed nested-field witness did not compile (rc=$CASE_RC)"
}
assert_no_fallback_marker "$CASE_LOG"
assert_executable_elf "$PASS_ELF" 'positive witness ELF'

set +e
(cd "$WORK/pass" && "$PASS_ELF") >"$WORK/pass/runtime.log" 2>&1
pass_runtime_rc=$?
set -e
[[ "$pass_runtime_rc" -eq 0 ]] || {
  cat "$WORK/pass/runtime.log" >&2
  fail "typed nested-field witness ELF exited rc=$pass_runtime_rc"
}
grep -Fxq '520' "$WORK/pass/runtime.log" || {
  cat "$WORK/pass/runtime.log" >&2
  fail 'typed nested-field witness did not retain InnerState.family_id=520'
}
grep -Fxq 'ISSUE_901_NESTED_FIELD_CHAIN_OK' "$WORK/pass/runtime.log" || {
  cat "$WORK/pass/runtime.log" >&2
  fail 'typed nested-field witness lost its exact marker'
}

MISS_ELF="$WORK/miss/issue_901_known_layout_miss.elf"
run_compile miss "$MISS_SOURCE" "$MISS_ELF"
[[ "$CASE_RC" -ne 0 ]] || {
  cat "$CASE_LOG" >&2
  fail 'known-layout miss unexpectedly compiled'
}
assert_no_fallback_marker "$CASE_LOG"
[[ ! -e "$MISS_ELF" ]] || {
  cat "$CASE_LOG" >&2
  fail 'known-layout miss emitted a native artifact'
}
if grep -Fxq 'ISSUE_901_NESTED_FIELD_CHAIN_OK' "$CASE_LOG"; then
  cat "$CASE_LOG" >&2
  fail 'known-layout miss reached the positive runtime marker'
fi

echo "[madaros-imported-runtime-acceptance] PASS: direct raw ELF preserves nested nominal layout and rejects a non-materialized known-layout miss raw_sha256=$RAW_SHA256"
