#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[native-v2-capturing-closure-unsupported] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[native-v2-capturing-closure-unsupported] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

if [[ -n "${SOUC_BIN:-}" && "$SOUC_BIN" != "$ROOT_DIR"/* ]]; then
  echo "[native-v2-capturing-closure-unsupported] ignoring external SOUC_BIN outside this worktree: $SOUC_BIN"
  unset SOUC_BIN
fi

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT_DIR="${SOUNIO_NATIVE_V2_CAPTURING_CLOSURE_UNSUPPORTED_DIR:-$(mktemp -d /tmp/sounio-native-v2-capturing-closure-unsupported.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"
ZERO_CAPTURE="tests/selfhost/native_runtime/native_v2_closure_zero_capture_direct_42.sio"
DIRECT_CAPTURE="tests/selfhost/native_runtime/native_v2_closure_capture_direct_42.sio"
HOF_CAPTURE="tests/selfhost/native_runtime/native_v2_closure_capture_hof_apply_42.sio"
BOUND_HOF_CAPTURE="tests/selfhost/native_runtime/native_v2_closure_capture_hof_bound_42.sio"
NONTRANSPARENT_HOF_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_nontransparent_hof_fail.sio"
ALIAS_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_alias_fail.sio"
RETURN_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_return_fail.sio"
IGNORE_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_ignore_fail.sio"
INLINE_NONTRANSPARENT_HOF_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_inline_nontransparent_hof_fail.sio"
NONTRANSPARENT_HOF_GENERIC_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_nontransparent_hof_generic_fail.sio"
ALIAS_GENERIC_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_alias_generic_fail.sio"
RETURN_GENERIC_FAIL="tests/selfhost/native_runtime/native_v2_closure_capture_return_generic_fail.sio"
LOWER_SOURCE="self-hosted/ir/lower.sio"
DIAGNOSTIC="native-v2 capturing closure literals are not yet supported"
ZERO_ELF="$ARTIFACT_DIR/native_v2_closure_zero_capture_direct_42.native"
DIRECT_ELF="$ARTIFACT_DIR/native_v2_closure_capture_direct_42.native"
HOF_ELF="$ARTIFACT_DIR/native_v2_closure_capture_hof_apply_42.native"
BOUND_HOF_ELF="$ARTIFACT_DIR/native_v2_closure_capture_hof_bound_42.native"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

echo "[native-v2-capturing-closure-unsupported] souc=$SOUC_BIN"
echo "[native-v2-capturing-closure-unsupported] out=$OUT_DIR"

for path in \
  "$ZERO_CAPTURE" \
  "$DIRECT_CAPTURE" \
  "$HOF_CAPTURE" \
  "$BOUND_HOF_CAPTURE" \
  "$NONTRANSPARENT_HOF_FAIL" \
  "$ALIAS_FAIL" \
  "$RETURN_FAIL" \
  "$IGNORE_FAIL" \
  "$INLINE_NONTRANSPARENT_HOF_FAIL" \
  "$NONTRANSPARENT_HOF_GENERIC_FAIL" \
  "$ALIAS_GENERIC_FAIL" \
  "$RETURN_GENERIC_FAIL" \
  "$LOWER_SOURCE"; do
  if [[ ! -f "$path" ]]; then
    echo "[native-v2-capturing-closure-unsupported] FAIL: missing $path" >&2
    exit 1
  fi
done

if ! grep -q "$DIAGNOSTIC" "$LOWER_SOURCE"; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: missing native-v2 capture rejection diagnostic in $LOWER_SOURCE" >&2
  exit 1
fi

if ! grep -q 'if captures.count > 0' "$LOWER_SOURCE"; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: missing native-v2 capture count guard in $LOWER_SOURCE" >&2
  exit 1
fi

if ! grep -q 'allow_direct_capturing_closure_literal' "$LOWER_SOURCE"; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: missing direct captured-closure context guard in $LOWER_SOURCE" >&2
  exit 1
fi

run_log() {
  local name="$1"
  shift
  echo "[native-v2-capturing-closure-unsupported] running $name"
  "$@" >"$LOG_DIR/$name.log" 2>&1
}

expect_compile_fail() {
  local name="$1"
  local src="$2"
  local out="$ARTIFACT_DIR/$name.native"

  rm -f "$out"
  echo "[native-v2-capturing-closure-unsupported] running $name"
  set +e
  "$SOUC_BIN" compile "$src" -o "$out" >"$LOG_DIR/$name.log" 2>&1
  local rc=$?
  set -e

  if [[ "$rc" -eq 0 || -f "$out" ]]; then
    echo "[native-v2-capturing-closure-unsupported] FAIL: $src unexpectedly compiled" >&2
    cat "$LOG_DIR/$name.log" >&2 || true
    exit 1
  fi

  if ! grep -Eq 'capturing closure cannot be used as a value|capturing closure literals are not yet supported as values' "$LOG_DIR/$name.log"; then
    echo "[native-v2-capturing-closure-unsupported] FAIL: $src failed without captured-closure fail-closed diagnostic" >&2
    cat "$LOG_DIR/$name.log" >&2 || true
    exit 1
  fi
}

run_log zero_capture_compile "$SOUC_BIN" compile "$ZERO_CAPTURE" -o "$ZERO_ELF"
run_log direct_capture_compile "$SOUC_BIN" compile "$DIRECT_CAPTURE" -o "$DIRECT_ELF"
run_log hof_capture_compile "$SOUC_BIN" compile "$HOF_CAPTURE" -o "$HOF_ELF"
run_log bound_hof_capture_compile "$SOUC_BIN" compile "$BOUND_HOF_CAPTURE" -o "$BOUND_HOF_ELF"

if [[ ! -f "$ZERO_ELF" ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: zero-capture closure did not produce native ELF" >&2
  cat "$LOG_DIR/zero_capture_compile.log" >&2 || true
  exit 1
fi

if [[ ! -f "$DIRECT_ELF" ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: direct captured closure did not produce native ELF" >&2
  cat "$LOG_DIR/direct_capture_compile.log" >&2 || true
  exit 1
fi

if [[ ! -f "$HOF_ELF" ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: HOF captured closure did not produce native ELF" >&2
  cat "$LOG_DIR/hof_capture_compile.log" >&2 || true
  exit 1
fi

if [[ ! -f "$BOUND_HOF_ELF" ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: bound HOF captured closure did not produce native ELF" >&2
  cat "$LOG_DIR/bound_hof_capture_compile.log" >&2 || true
  exit 1
fi

chmod +x "$ZERO_ELF" "$DIRECT_ELF" "$HOF_ELF" "$BOUND_HOF_ELF" 2>/dev/null || true

set +e
"$ZERO_ELF" >"$LOG_DIR/zero_capture.stdout" 2>"$LOG_DIR/zero_capture.stderr"
runtime_rc=$?
set -e

if [[ "$runtime_rc" -ne 42 ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: zero-capture closure expected exit 42, got $runtime_rc" >&2
  cat "$LOG_DIR/zero_capture.stdout" >&2 || true
  cat "$LOG_DIR/zero_capture.stderr" >&2 || true
  exit 1
fi

set +e
"$DIRECT_ELF" >"$LOG_DIR/direct_capture.stdout" 2>"$LOG_DIR/direct_capture.stderr"
direct_runtime_rc=$?
set -e

if [[ "$direct_runtime_rc" -ne 42 ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: direct captured closure expected exit 42, got $direct_runtime_rc" >&2
  cat "$LOG_DIR/direct_capture.stdout" >&2 || true
  cat "$LOG_DIR/direct_capture.stderr" >&2 || true
  exit 1
fi

set +e
"$HOF_ELF" >"$LOG_DIR/hof_capture.stdout" 2>"$LOG_DIR/hof_capture.stderr"
hof_runtime_rc=$?
set -e

if [[ "$hof_runtime_rc" -ne 42 ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: HOF captured closure expected exit 42, got $hof_runtime_rc" >&2
  cat "$LOG_DIR/hof_capture.stdout" >&2 || true
  cat "$LOG_DIR/hof_capture.stderr" >&2 || true
  exit 1
fi

set +e
"$BOUND_HOF_ELF" >"$LOG_DIR/bound_hof_capture.stdout" 2>"$LOG_DIR/bound_hof_capture.stderr"
bound_hof_runtime_rc=$?
set -e

if [[ "$bound_hof_runtime_rc" -ne 42 ]]; then
  echo "[native-v2-capturing-closure-unsupported] FAIL: bound HOF captured closure expected exit 42, got $bound_hof_runtime_rc" >&2
  cat "$LOG_DIR/bound_hof_capture.stdout" >&2 || true
  cat "$LOG_DIR/bound_hof_capture.stderr" >&2 || true
  exit 1
fi

expect_compile_fail nontransparent_hof_fail "$NONTRANSPARENT_HOF_FAIL"
expect_compile_fail alias_fail "$ALIAS_FAIL"
expect_compile_fail return_fail "$RETURN_FAIL"
expect_compile_fail ignore_fail "$IGNORE_FAIL"
expect_compile_fail inline_nontransparent_hof_fail "$INLINE_NONTRANSPARENT_HOF_FAIL"
expect_compile_fail nontransparent_hof_generic_fail "$NONTRANSPARENT_HOF_GENERIC_FAIL"
expect_compile_fail alias_generic_fail "$ALIAS_GENERIC_FAIL"
expect_compile_fail return_generic_fail "$RETURN_GENERIC_FAIL"

echo "[native-v2-capturing-closure-unsupported] PASS: transparent direct/HOF captured closures stay native; escaping/nontransparent captured closure value use fails closed"
