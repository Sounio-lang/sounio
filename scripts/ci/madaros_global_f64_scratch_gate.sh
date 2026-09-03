#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_GLOBAL_F64_GATE_KEEP:-0}"
SOURCE="$ROOT_DIR/tests/run-pass/global_f64_scratch_add.sio"

fail() {
  echo "[madaros-global-f64] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_GLOBAL_F64_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_GLOBAL_F64_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-global-f64.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_GLOBAL_F64_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_GLOBAL_F64_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "rebuilt Madaros is missing or not executable: $MADAROS_ELF"

run_witness() {
  local source="$1"
  local marker="$2"
  local log="$3"
  local label="$4"

  set +e
  MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" run "$source" >"$log" 2>&1
  local run_rc=$?
  set -e

  if [[ "$run_rc" != "0" ]]; then
    cat "$log" >&2
    fail "$label exited rc=$run_rc"
  fi
  grep -Fxq "$marker" "$log" || {
    cat "$log" >&2
    fail "$label exact PASS marker missing"
  }
}

run_witness "$SOURCE" 'PASS global_f64_scratch_add' "$WORK/run.log" 'global f64 scratch witness'

SCALAR_SOURCE="$WORK/global_f64_scalar.sio"
cat >"$SCALAR_SOURCE" <<'SOUNIO'
//@ run-pass
//@ expect-stdout: PASS global_f64_scalar

var LEFT_SCALAR: f64 = 0.0
var RIGHT_SCALAR: f64 = 0.0

fn main() -> i32 with IO, Mut {
    LEFT_SCALAR = 0.0 - 1.0
    RIGHT_SCALAR = 1.0
    if LEFT_SCALAR + RIGHT_SCALAR != 0.0 { return 1 }
    println("PASS global_f64_scalar")
    0
}
SOUNIO

run_witness \
  "$SCALAR_SOURCE" \
  'PASS global_f64_scalar' \
  "$WORK/scalar.log" \
  'global f64 scalar witness'

BOUNDARY_SOURCE="$WORK/global_f64_after_256_globals.sio"
{
  printf '%s\n' '//@ run-pass'
  printf '%s\n' '//@ expect-stdout: PASS global_f64_after_256_globals'
  for ((i = 0; i < 257; i++)); do
    printf 'var PAD_%d: i64 = 0\n' "$i"
  done
  cat <<'SOUNIO'
var LEFT_BOUNDARY: [f64; 8] = [0.0; 8]
var RIGHT_BOUNDARY: [f64; 8] = [0.0; 8]

fn main() -> i32 with IO, Mut {
    LEFT_BOUNDARY[4] = 0.0 - 1.0
    RIGHT_BOUNDARY[4] = 1.0
    if LEFT_BOUNDARY[4] + RIGHT_BOUNDARY[4] != 0.0 { return 1 }
    println("PASS global_f64_after_256_globals")
    0
}
SOUNIO
} >"$BOUNDARY_SOURCE"

run_witness \
  "$BOUNDARY_SOURCE" \
  'PASS global_f64_after_256_globals' \
  "$WORK/boundary.log" \
  'global f64 after 256 globals witness'

echo "[madaros-global-f64] PASS: Madaros ELF preserves global f64 scalar and array-element typing past the previous 256-entry metadata boundary"
