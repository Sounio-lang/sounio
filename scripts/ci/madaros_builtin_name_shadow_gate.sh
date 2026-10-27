#!/usr/bin/env bash
# A bare read_i64/write_i64 call is not E175 when another module defines a
# private function of that name. Same-module calls keep the private signature.
# A non-builtin private function is still E175, and a wrong builtin arity is
# still E010. This pins the typecheck from issue #3238. Lowering still binds
# the bare call to the private function, so this gate does not run the fixture.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=../lib/classify_compile.sh
source "$ROOT_DIR/scripts/lib/classify_compile.sh"
FIX="$ROOT_DIR/tests/multimodule/builtin_name_shadow"
TAG="[madaros-builtin-name-shadow]"

fail() {
  echo "$TAG FAIL: $*" >&2
  exit 1
}

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "$TAG SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *) echo "$TAG SKIP: x86-64 Linux-only gate" >&2; exit 0 ;;
esac

if [[ -n "${SOUNIO_MADAROS_BUILTIN_NAME_SHADOW_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_BUILTIN_NAME_SHADOW_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-builtin-name-shadow.XXXXXX)"
fi
if [[ -z "${SOUNIO_MADAROS_BUILTIN_NAME_SHADOW_KEEP:-}" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

RAW="${SOUNIO_MADAROS_BUILTIN_NAME_SHADOW_BIN:-${MADAROS_RAW_BIN:-}}"
if [[ -z "$RAW" ]]; then
  RAW="$WORK/madaros-from-source.elf"
  echo "$TAG no MADAROS_RAW_BIN; building Madaros from source"
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$RAW" >"$WORK/build.log" 2>&1; then
    tail -n 40 "$WORK/build.log" >&2 || true
    fail "could not build Madaros from source"
  fi
fi
[[ -x "$RAW" ]] || fail "Madaros is missing or not executable: $RAW"
[[ "$(head -c 2 "$RAW")" != '#!' ]] || fail "not a raw ELF (a wrapper script?): $RAW"

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export MADAROS_RAW_BIN="$RAW"

expect_check_ok() {
  local label="$1" src="$2"
  local log="$WORK/$label.log"
  local rc=0
  [[ -f "$src" ]] || fail "$label: missing fixture $src"
  set +e
  MADAROS_RAW_BIN="$RAW" "$ROOT_DIR/bin/madaros" check "$src" >"$log" 2>&1
  rc=$?
  set -e
  [[ "$rc" -eq 0 ]] || {
    tail -n 40 "$log" >&2 || true
    fail "$label: check returned $rc"
  }
  if grep -q 'error\[E175\]' "$log"; then
    tail -n 40 "$log" >&2 || true
    fail "$label: check still reports E175"
  fi
  grep -q 'check: OK' "$log" || {
    tail -n 40 "$log" >&2 || true
    fail "$label: missing check: OK"
  }
}

expect_refusal() {
  local label="$1" src="$2" code="$3"
  local log="$WORK/$label.log"
  local rc=0
  # rc alone is not a refusal: a crash, timeout or OOM kill is not one (#2425).
  sounio_capture_rc rc "$log" -- "$ROOT_DIR/bin/madaros" check "$src"
  sounio_expect_refusal "$rc" "$log" "" "$code" || {
    cat "$log" >&2
    fail "$label: expected a clean $code refusal, got rc=$rc $(sounio_refusal_reason)"
  }
}

expect_check_ok basic "$FIX/main.sio"
echo "$TAG PASS(basic): bare read_i64/write_i64 are not private functions"

expect_refusal arity "$FIX/arity.sio" E010
if grep -q 'error\[E175\]' "$WORK/arity.log"; then
  cat "$WORK/arity.log" >&2
  fail "arity: wrong builtin arity was reported as a private function"
fi
echo "$TAG PASS(arity): wrong builtin arity is E010, not E175"

expect_refusal private "$ROOT_DIR/tests/multimodule/visibility_fn_private_main.sio" E175
echo "$TAG PASS(private): a non-builtin private function is still E175"
echo "$TAG PASS"
