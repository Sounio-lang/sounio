#!/usr/bin/env bash
# #1627 closeout gate: private extern "C" sqrt must not false-E175 complex builtins.
#
# Default Madaros is bin/madaros-linux-x86_64 (shipped). As of 2026-08-06 promote
# (sm_params Mut/Div/Panic + tip checker), the committed ELF is expected green.
# Override with MADAROS_BIN=... for a tip rebuild. REQUIRE_SHIPPED_GREEN=1 is
# redundant post-promote but kept for CI call sites.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MADAROS_BIN="${MADAROS_BIN:-}"
if [[ -z "$MADAROS_BIN" ]]; then
  if [[ -x "$ROOT/artifacts/self-hosted/madaros" ]]; then
    MADAROS_BIN="$ROOT/artifacts/self-hosted/madaros"
  else
    MADAROS_BIN="$ROOT/bin/madaros-linux-x86_64"
  fi
fi
if [[ ! -x "$MADAROS_BIN" ]]; then
  echo "missing Madaros ELF: $MADAROS_BIN" >&2
  exit 1
fi

FIX_DIR="$(mktemp -d /tmp/e175_1627_fix.XXXXXX)"
trap 'rm -rf "$FIX_DIR"' EXIT
mkdir -p "$FIX_DIR/stdlib/_e175_builtin_shadow_fixture"
cp -a "$ROOT/stdlib/." "$FIX_DIR/stdlib/"
cat >"$FIX_DIR/stdlib/_e175_builtin_shadow_fixture/shadow.sio" <<'EOF'
module _e175_builtin_shadow_fixture::shadow
extern "C" {
    fn sqrt(x: f64) -> f64;
}
pub fn shadow_id(x: f64) -> f64 { x }
EOF

MAIN="$(mktemp /tmp/e175_1627_main.XXXXXX.sio)"
cat >"$MAIN" <<'EOF'
use _e175_builtin_shadow_fixture::shadow::{shadow_id}
use complex::lib::{complex_new, complex_abs}
fn main() -> i32 with IO {
    print_f64(complex_abs(complex_new(3.0, 4.0)) + shadow_id(0.0))
    print("\n")
    0
}
EOF

export SOUNIO_STDLIB_PATH="$FIX_DIR/stdlib"
echo "== #1627 fixture check via $MADAROS_BIN =="
set +e
SOUNIO_SOUC_ENGINE=madaros MADAROS_RAW_BIN="$MADAROS_BIN" \
  "$ROOT/bin/souc" check "$MAIN" >"$FIX_DIR/out.txt" 2>"$FIX_DIR/err.txt"
rc=$?
set -e

if grep -q 'E175' "$FIX_DIR/out.txt" "$FIX_DIR/err.txt"; then
  echo "FAIL: E175 still present (#1627 not closed in $MADAROS_BIN)" >&2
  grep 'E175' "$FIX_DIR/out.txt" "$FIX_DIR/err.txt" | head -20 >&2
  exit 1
fi
if ! grep -qE 'verdict=0|check: OK' "$FIX_DIR/out.txt" "$FIX_DIR/err.txt"; then
  echo "FAIL: check did not pass rc=$rc" >&2
  tail -40 "$FIX_DIR/err.txt" >&2
  exit 1
fi

echo "PARTICLE_E175_1627_GATE_OK"
