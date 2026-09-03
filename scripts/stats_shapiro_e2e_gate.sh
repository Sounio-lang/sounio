#!/usr/bin/env bash
# scripts/stats_shapiro_e2e_gate.sh — Shapiro–Wilk W/p E2E under lean_single
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
export SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-lean_single}"
SOUC="${SOUC:-$ROOT/bin/souc}"
SRC="tests/stdlib/stats/test_shapiro_e2e.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
ELF="$OUT/shapiro_e2e.elf"
LOG="$OUT/run.log"
fail=0

echo "== stats_shapiro_e2e_gate: engine=$SOUNIO_SOUC_ENGINE =="
if ! "$SOUC" compile "$SRC" -o "$ELF" >"$OUT/compile.log" 2>&1; then
  echo "FAIL: compile"; tail -30 "$OUT/compile.log" || true; fail=1
else
  chmod +x "$ELF"
  if ! "$ELF" >"$LOG" 2>&1; then
    echo "FAIL: run"; cat "$LOG" || true; fail=1
  elif ! grep -q "STATS_SHAPIRO_E2E_OK" "$LOG"; then
    echo "FAIL: missing sentinel"; cat "$LOG" || true; fail=1
  else
    grep '^SHAPIRO_E2E' "$LOG" || true
  fi
fi

# Optional pure-Python oracle on W (published definition)
if [[ $fail -eq 0 ]]; then
  set +e
  python3 - <<'PY' "$LOG"
import sys, math
log = open(sys.argv[1]).read()
# parse n10_w and skew_w
def grab(key):
    for tok in log.replace("\n"," ").split():
        if tok.startswith(key+"="):
            return float(tok.split("=",1)[1])
    return None
n10 = grab("n10_w"); skew = grab("skew_w"); n5 = grab("n5_w")
exp = {"n5_w": 0.9865881, "n10_w": 0.970158460121212, "skew_w": 0.738023544970166}
got = {"n5_w": n5, "n10_w": n10, "skew_w": skew}
ok = True
for k,e in exp.items():
    g = got[k]
    if g is None:
        print(f"oracle skip {k}: missing"); ok=False; continue
    err = abs(g-e)
    status = "OK" if err < 1e-6 else "FAIL"
    if err >= 1e-6: ok=False
    print(f"  {status} {k}: sounio={g} oracle={e} abs_err={err:.3e}")
sys.exit(0 if ok else 1)
PY
  orc=$?
  set -e
  if [[ $orc -ne 0 ]]; then echo "FAIL: oracle"; fail=1; fi
fi

if [[ $fail -eq 0 ]]; then
  echo "STATS_SHAPIRO_E2E_GATE_OK"
  exit 0
fi
exit 1
