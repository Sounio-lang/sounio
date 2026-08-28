#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"
sounio_require_souc

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

OUT_DIR="${SOUNIO_NATIVE_V2_OUT_PARAM_BOUNDARY_DIR:-$(mktemp -d /tmp/sounio-native-v2-out-param-boundary.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
ARTIFACT_DIR="$OUT_DIR/artifacts"

SAME_SRC="tests/selfhost/native_runtime/abi_out_param_boundary_same_file_42.sio"
IMPORTED_SRC="tests/selfhost/native_runtime/abi_out_param_boundary_imported_42.sio"
HELPER_SRC="tests/selfhost/native_runtime/abi_out_param_boundary_helper.sio"
BUNDLE="$ARTIFACT_DIR/abi_out_param_boundary_imported_42.bundle.sio"
SAME_BIN="$ARTIFACT_DIR/abi_out_param_boundary_same_file_42.native"
IMPORTED_BIN="$ARTIFACT_DIR/abi_out_param_boundary_imported_42.native"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR"

printf '[native-v2-out-param-boundary] souc=%s\n' "$SOUC_BIN"
printf '[native-v2-out-param-boundary] out=%s\n' "$OUT_DIR"

for src in "$SAME_SRC" "$IMPORTED_SRC" "$HELPER_SRC"; do
  if [[ ! -f "$src" ]]; then
    echo "[native-v2-out-param-boundary] FAIL: missing fixture $src" >&2
    exit 1
  fi
done

"$SOUC_BIN" check self-hosted/compiler/native_compile_driver.sio \
  >"$LOG_DIR/native_compile_driver.check.log" 2>&1
"$SOUC_BIN" check "$SAME_SRC" >"$LOG_DIR/same_file.check.log" 2>&1
"$SOUC_BIN" check "$IMPORTED_SRC" >"$LOG_DIR/imported.check.log" 2>&1

run_native_case() {
  local label="$1"
  local src="$2"
  local bin="$3"
  local compile_log="$LOG_DIR/${label}.compile.log"
  local stdout_log="$LOG_DIR/${label}.stdout"
  local stderr_log="$LOG_DIR/${label}.stderr"
  local runtime_rc=0

  "$SOUC_BIN" run self-hosted/compiler/native_compile_driver.sio -- \
    "$src" -o "$bin" >"$compile_log" 2>&1

  if [[ ! -x "$bin" ]]; then
    echo "[native-v2-out-param-boundary] FAIL: native binary not executable for $label: $bin" >&2
    tail -n 40 "$compile_log" >&2 || true
    exit 1
  fi

  set +e
  "$bin" >"$stdout_log" 2>"$stderr_log"
  runtime_rc=$?
  set -e

  if [[ "$runtime_rc" -ne 42 ]]; then
    echo "[native-v2-out-param-boundary] FAIL: expected exit 42 for $label, got $runtime_rc" >&2
    cat "$stdout_log" >&2 || true
    cat "$stderr_log" >&2 || true
    exit 1
  fi

  printf '[native-v2-out-param-boundary] ok: %s exit=%s\n' "$label" "$runtime_rc"
}

run_native_case same_file "$SAME_SRC" "$SAME_BIN"

"$SOUC_BIN" run self-hosted/compiler/native_prebundle.sio -- \
  "$IMPORTED_SRC" -o "$BUNDLE" --root tests/selfhost/native_runtime \
  >"$LOG_DIR/imported.prebundle.log" 2>&1

if [[ ! -s "$BUNDLE" ]]; then
  echo "[native-v2-out-param-boundary] FAIL: bundle not produced" >&2
  cat "$LOG_DIR/imported.prebundle.log" >&2 || true
  exit 1
fi

source_markers="$(awk 'BEGIN { n = 0 } /^\/\/ SOURCE:/ { n += 1 } END { print n }' "$BUNDLE")"
if [[ "$source_markers" -ne 2 ]]; then
  echo "[native-v2-out-param-boundary] FAIL: expected 2 SOURCE markers, got $source_markers" >&2
  grep '^// SOURCE:' "$BUNDLE" >&2 || true
  exit 1
fi

if grep -Eq '^(use |module )' "$BUNDLE"; then
  echo "[native-v2-out-param-boundary] FAIL: bundle still contains use/module lines" >&2
  grep -En '^(use |module )' "$BUNDLE" >&2 || true
  exit 1
fi

if ! grep -q '^// SOURCE: tests/selfhost/native_runtime/abi_out_param_boundary_helper.sio$' "$BUNDLE"; then
  echo "[native-v2-out-param-boundary] FAIL: bundle missing helper source marker" >&2
  grep '^// SOURCE:' "$BUNDLE" >&2 || true
  exit 1
fi

if ! grep -q '^// SOURCE: tests/selfhost/native_runtime/abi_out_param_boundary_imported_42.sio$' "$BUNDLE"; then
  echo "[native-v2-out-param-boundary] FAIL: bundle missing imported main source marker" >&2
  grep '^// SOURCE:' "$BUNDLE" >&2 || true
  exit 1
fi

run_native_case imported_helper "$BUNDLE" "$IMPORTED_BIN"

echo "[native-v2-out-param-boundary] PASS: same-file and prebundled imported &! out-param field stores exit 42"
