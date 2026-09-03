#!/usr/bin/env bash
# scripts/stats_madaros_singlefile_smoke_gate.sh
# Default Madaros single-file smoke (NO multi-module use chain).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
# Explicitly do NOT force lean_single — claim is default Madaros single-file.
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/stdlib/stats/test_madaros_singlefile_smoke.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/madaros_smoke.elf"
LOG="$OUT/run.log"
fail=0

echo "== stats_madaros_singlefile_smoke_gate: default engine (Madaros) =="
"$SOUC" --version 2>&1 | head -2 || true

if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"; tail -40 "$OUT/compile.log" || true; fail=1
else
  chmod +x "$ELF"
  if ! "$ELF" >"$LOG" 2>&1; then
    echo "FAIL: run"; cat "$LOG" || true; fail=1
  elif ! grep -q "STATS_MADAROS_SINGLEFILE_SMOKE_OK" "$LOG"; then
    echo "FAIL: missing sentinel"; cat "$LOG" || true; fail=1
  else
    grep '^MADAROS_SMOKE' "$LOG" || true
  fi
fi

if [[ $fail -eq 0 ]]; then
  echo "STATS_MADAROS_SINGLEFILE_SMOKE_GATE_OK"
  echo "claims_not_made: madaros_multimodule full_scipy_api"
  exit 0
fi
exit 1
