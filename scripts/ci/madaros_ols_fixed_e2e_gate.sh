#!/usr/bin/env bash
# Madaros-native OLS diagnostics E2E (fixed-array surface).
# Closes the D3 "OLS multi-mod E019" attention item for the cooks+OLS+shapiro
# path without claiming stats::validation slice methods are green.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
# Force default Madaros (do not inherit lean_single from caller)
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/stdlib/stats/test_ols_madaros_e2e.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/ols.elf"

echo "== madaros_ols_fixed_e2e_gate =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"
  tail -40 "$OUT/compile.log" || true
  exit 1
fi
chmod +x "$ELF"
LOG="$OUT/run.log"
if ! "$ELF" >"$LOG" 2>&1; then
  echo "FAIL: run"
  cat "$LOG" || true
  exit 1
fi
grep -q 'STATS_OLS_MADAROS_E2E_OK' "$LOG" || {
  echo "FAIL: missing sentinel"
  cat "$LOG" || true
  exit 1
}
grep -E 'cook_slope=2(\.0+)?(\s|$)|cook_slope=2\.000000' "$LOG" >/dev/null || {
  echo "FAIL: cook slope"
  cat "$LOG" || true
  exit 1
}
grep -E 'max_cooks=2\.25' "$LOG" >/dev/null || {
  echo "FAIL: max cooks"
  cat "$LOG" || true
  exit 1
}
grep -E 'ols_slope=0\.6' "$LOG" >/dev/null || {
  echo "FAIL: ols slope"
  cat "$LOG" || true
  exit 1
}

# Negative control: validation slice path must still be the E019 residual
NEG="$OUT/neg.sio"
cat >"$NEG" <<'EOF'
use stats::validation::{linear_regression}
fn main() -> i32 with IO, Mut, Div, Panic {
    var xt: [f64; 5] = [1.0, 2.0, 3.0, 4.0, 5.0]
    var yt: [f64; 5] = [2.0, 4.0, 5.0, 4.0, 5.0]
    let _fit = linear_regression(&xt, &yt)
    print("UNEXPECTED_VALIDATION_GREEN\n")
    return 0
}
EOF
if "$SOUC" compile "$NEG" -o "$OUT/neg.elf" >"$OUT/neg.log" 2>&1; then
  echo "FAIL: validation slice path unexpectedly compiled under Madaros"
  exit 1
fi
if ! grep -E 'E019|method calls are not supported' "$OUT/neg.log" >/dev/null; then
  echo "FAIL: expected E019 on validation import; got:"
  tail -30 "$OUT/neg.log" || true
  exit 1
fi

echo "MADAROS_OLS_FIXED_E2E_GATE_OK"
