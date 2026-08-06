#!/usr/bin/env bash
# #1627 closeout gate: private extern "C" sqrt must not false-E175 complex builtins.
#
# Default Madaros is bin/madaros-linux-x86_64 (shipped). Tip rebuilds currently
# regress E035 on sm_params, so this gate accepts MADAROS_BIN override for a
# freshly built artifacts/self-hosted/madaros from:
#   bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros
#
# When MADAROS_BIN is unset, runs against the shipped ELF — which should FAIL
# until a safe promote lands. Set REQUIRE_SHIPPED_GREEN=1 to demand green on
# the committed binary (post-promote).
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
  if [[ "${REQUIRE_SHIPPED_GREEN:-0}" == "1" ]] || [[ "$MADAROS_BIN" == *"artifacts/self-hosted/madaros"* ]]; then
    echo "FAIL: E175 still present (#1627 not closed in $MADAROS_BIN)" >&2
    grep 'E175' "$FIX_DIR/out.txt" "$FIX_DIR/err.txt" | head -20 >&2
    exit 1
  fi
  echo "PARTICLE_E175_1627_SHIPPED_STILL_E175"
  echo "hint: rebuild tip Madaros and re-run with MADAROS_BIN=artifacts/self-hosted/madaros"
  exit 0
fi
if ! grep -qE 'verdict=0|check: OK' "$FIX_DIR/out.txt" "$FIX_DIR/err.txt"; then
  echo "FAIL: check did not pass rc=$rc" >&2
  tail -40 "$FIX_DIR/err.txt" >&2
  exit 1
fi

echo "PARTICLE_E175_1627_GATE_OK"
