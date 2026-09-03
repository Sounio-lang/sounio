#!/usr/bin/env bash
# Madaros-native OLS diagnostics E2E (fixed-array surface).
# cooks + ols_fixed + shapiro; validation fixed-buffer API is gated separately
# by scripts/ci/madaros_validation_import_gate.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
# Always pin this worktree's stdlib (never inherit a foreign SOUNIO_STDLIB_PATH).
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
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

# Positive control: stats::validation fixed-buffer API under Madaros
POS="$OUT/pos.sio"
cat >"$POS" <<'EOF'
use stats::validation::{linear_regression}
fn main() -> i32 with IO, Mut, Div, Panic {
    var xt: [f64; 256] = [0.0; 256]
    var yt: [f64; 256] = [0.0; 256]
    xt[0] = 1.0; xt[1] = 2.0; xt[2] = 3.0; xt[3] = 4.0; xt[4] = 5.0
    yt[0] = 2.0; yt[1] = 4.0; yt[2] = 5.0; yt[3] = 4.0; yt[4] = 5.0
    let fit = linear_regression(&xt, &yt, 5)
    if fit.slope > 0.59 && fit.slope < 0.61 {
        print("VALIDATION_POS_OK\n")
        return 0
    }
    1
}
EOF
if ! "$SOUC" compile "$POS" -o "$OUT/pos.elf" >"$OUT/pos.log" 2>&1; then
  echo "FAIL: validation fixed-buffer path should compile under Madaros"
  tail -40 "$OUT/pos.log" || true
  exit 1
fi
chmod +x "$OUT/pos.elf"
if ! "$OUT/pos.elf" >"$OUT/pos_run.log" 2>&1; then
  echo "FAIL: validation positive control run"
  cat "$OUT/pos_run.log" || true
  exit 1
fi
grep -q 'VALIDATION_POS_OK' "$OUT/pos_run.log" || {
  echo "FAIL: missing VALIDATION_POS_OK"
  cat "$OUT/pos_run.log" || true
  exit 1
}

echo "MADAROS_OLS_FIXED_E2E_GATE_OK"
