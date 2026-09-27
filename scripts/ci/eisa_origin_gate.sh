#!/usr/bin/env bash
# R7 EReg.origin acceptance (founder 2026-08-20).
# Integer oracle + x+x=2u + independent negative + force-equal positive.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
# Workspace poison: a leftover SOUC_BIN can silently point at another tree.
unset SOUC_BIN || true
unset SOUNIO_SOUC_BIN || true

fail() { echo "EISA_ORIGIN_GATE_FAIL: $*" >&2; exit 1; }
pass() { echo "EISA_ORIGIN_GATE_OK: $*"; }

SOUC="${SOUC:-$ROOT/bin/souc}"
[[ -x "$SOUC" ]] || fail "no souc at $SOUC"

echo "souc=$SOUC"
"$SOUC" --version 2>&1 | head -2 || true

python3 docs/audit/repro/eisa_origin/oracle.py || fail "integer oracle"
pass "integer oracle (no floating point)"

run_ok() {
  local label="$1"
  local src="$2"
  local needle="$3"
  local out rc
  set +e
  out=$("$SOUC" run "$src" 2>&1)
  rc=$?
  set -e
  [[ $rc -eq 0 ]] || fail "$label rc=$rc: $out"
  echo "$out" | grep -Fq "$needle" || fail "$label missing '$needle': $out"
  pass "$label"
}

run_ok "correlated x+x" tests/run-pass/eisa_origin_correlated_add.sio "EISA_ORIGIN_CORR_ADD OK u=6 origin=1"
run_ok "independent negative" tests/run-pass/eisa_origin_independent_add.sio "EISA_ORIGIN_INDEP_ADD OK u=5 origin=0 no-fire"
run_ok "fusion mixed" tests/run-pass/eisa_origin_fusion_mixed.sio "EISA_ORIGIN_FUSION OK"
run_ok "esub envelope" tests/run-pass/eisa_origin_esub_same.sio "EISA_ORIGIN_ESUB OK u=6 origin=1"
run_ok "core W1-W5 nreg" tests/stdlib/eisa/test_eisa_core.sio "ALL PASS: eisa core W1 W2 W3 W4 W5"

# Positive control: force origin comparison always-equal. Independent 3-4-5
# must then fail (u becomes 3+4=7). Proves eisa_u_add_sub is reached.
CORE="$ROOT/stdlib/eisa/core.sio"
BAK="$(mktemp "${TMPDIR:-/tmp}/eisa-origin-core.XXXXXX")"
cp "$CORE" "$BAK"
restore_core() { cp "$BAK" "$CORE"; rm -f "$BAK"; }
trap restore_core EXIT

python3 - "$CORE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
s = p.read_text()
old = """pub fn eisa_origins_correlated(a: i64, b: i64) -> i64 {
    if a != 0 {
        if a == b {
            return 1
        }
    }
    0
}"""
new = """pub fn eisa_origins_correlated(a: i64, b: i64) -> i64 {
    let _keep_a = a
    let _keep_b = b
    1
}"""
if old not in s:
    raise SystemExit("hook text not found in core.sio")
p.write_text(s.replace(old, new, 1))
PY

set +e
force_out=$("$SOUC" run tests/run-pass/eisa_origin_independent_add.sio 2>&1)
force_rc=$?
set -e
restore_core
trap - EXIT

if [[ $force_rc -eq 0 ]]; then
  fail "FORCE-EQUAL independent still passed (comparison not reached): $force_out"
fi
echo "$force_out" | grep -Eq 'EISA_ORIGIN_INDEP_ADD FAIL|assert|FAIL' \
  || fail "FORCE-EQUAL failed without the independent FAIL marker: $force_out"
pass "FORCE-EQUAL positive control (independent now fails)"

# Non-regression: remaining stdlib/eisa tests that call egate / reconstruct EReg.
run_ok "isa nreg" tests/stdlib/eisa/test_eisa_isa.sio "ALL PASS: eisa isa P1 P2 P3 P4 P5"
run_ok "evm nreg" tests/stdlib/eisa/test_eisa_evm.sio "ALL PASS: eisa evm V1 V2 V3 V4 V5"
run_ok "backend nreg" tests/stdlib/eisa/test_eisa_backend.sio "ALL PASS: eisa backend B1 B2 B3 B4 B5 B6"

echo "EISA_ORIGIN_GATE_OK: all"
