#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GENERATED_ADAPTERS="$ROOT_DIR/self-hosted/compiler/codegen/hardware/kaxi_adapters.sio"
PASS_MARKER="KAXI_ADAPTERS_SELF_CHECK_PASS"
DEFAULT_DEBUG_SOUC="$ROOT_DIR/target/debug/souc"
DEFAULT_RELEASE_SOUC="$ROOT_DIR/target/release/souc"

if [ ! -f "$GENERATED_ADAPTERS" ]; then
  echo "error: missing generated K-AXI adapters file: $GENERATED_ADAPTERS" >&2
  echo "hint: run the pure K-AXI emitter to generate this artifact first." >&2
  exit 2
fi

resolve_souc_bin() {
  if [ -n "${SOUC_BIN:-}" ]; then
    echo "$SOUC_BIN"
    return
  fi
  if [ -x "$DEFAULT_DEBUG_SOUC" ]; then
    echo "$DEFAULT_DEBUG_SOUC"
    return
  fi
  if [ -x "$DEFAULT_RELEASE_SOUC" ]; then
    echo "$DEFAULT_RELEASE_SOUC"
    return
  fi
  echo "souc"
}

SOUC_BIN="$(resolve_souc_bin)"

tmp_sio="$(mktemp "${TMPDIR:-/tmp}/omega_kaxi_adapter_self_check.XXXXXX.sio")"
cleanup() {
  rm -f "$tmp_sio"
}
trap cleanup EXIT

cat "$GENERATED_ADAPTERS" >"$tmp_sio"
cat >>"$tmp_sio" <<'SIO'

fn main() -> i32 with IO, Mut, Panic, Div, Alloc {
    if kaxi_adapters_self_check() {
        print("KAXI_ADAPTERS_SELF_CHECK_PASS\n")
        return 0 as i32
    } else {
        print("KAXI_ADAPTERS_SELF_CHECK_FAIL\n")
        return 1 as i32
    }
}
SIO

set +e
run_output="$(PATH="$ROOT_DIR:$PATH" "$SOUC_BIN" run "$tmp_sio" 2>&1)"
run_rc=$?
set -e

if [ "$run_rc" -ne 0 ]; then
  echo "error: souc run failed for K-AXI adapter self-check (exit=$run_rc bin=$SOUC_BIN)" >&2
  printf '%s\n' "$run_output" >&2
  exit "$run_rc"
fi

if ! printf '%s\n' "$run_output" | rg -q "$PASS_MARKER"; then
  echo "error: K-AXI adapter self-check did not emit required marker: $PASS_MARKER" >&2
  printf '%s\n' "$run_output" >&2
  exit 2
fi

printf '%s\n' "$run_output"
