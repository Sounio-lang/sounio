#!/usr/bin/env bash
# Token / source ceiling must REFUSE (E229), never silently drop input.
#
# Two witnesses (both generated, not committed — avoid multi‑MB blobs in git):
#
#   W1 source-byte wall: a complete valid `main` in the first bytes, then padding
#      to exactly 2097152 bytes, then MORE source (`fn should_not_be_dropped`).
#      Pre-fix: clips the buffer, parses main, exits 0 — trailing source GONE.
#      Post-fix: error[E229] source exceeds lexer byte buffer, nonzero rc.
#
#   W2 token wall: 2097152 comma tokens (one byte each) fill the token table with
#      no room for Eof. Post-fix: error[E229] token table full.
#
# Does NOT raise the 2097152 ceiling. Raising without refusal is the wrong fix.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

fail() { echo "TOKEN_TABLE_CEILING_GATE_FAIL: $*" >&2; exit 1; }
pass() { echo "TOKEN_TABLE_CEILING_GATE_OK: $*"; }

SOUC="${SOUC_BIN:-$ROOT_DIR/bin/souc}"
[[ -x "$SOUC" ]] || fail "souc missing: $SOUC"
ulimit -S -s 524288 2>/dev/null || true

WORK="$(mktemp -d "${TMPDIR:-/tmp}/sounio-token-ceiling.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

CAP=2097152
EXPECT="${TOKEN_CEILING_EXPECT:-refusal}"

# --- W1: source byte wall (the silent-clip honesty case) -------------------
W1="$WORK/w1_source_clip.sio"
python3 - <<PY
cap = $CAP
prefix = b"fn main() with IO {\n  println(\"ok\")\n  0\n}\n"
if len(prefix) >= cap:
    raise SystemExit("prefix too large")
# Fill to exactly CAP bytes (spaces are whitespace — still source that must not vanish).
pad = cap - len(prefix)
# Trailing REAL declarations past the wall — pre-fix must not see these.
extra = b"\nfn should_not_be_dropped() -> i64 { 42 }\n"
data = prefix + (b" " * pad) + extra
assert len(data) > cap
open(r"$W1", "wb").write(data)
print(f"W1 bytes={len(data)} cap={cap} prefix={len(prefix)} extra={len(extra)}")
PY

LOG1="$WORK/w1.log"
set +e
"$SOUC" check "$W1" >"$LOG1" 2>&1
rc1=$?
set -e
echo "  W1 source-clip: rc=$rc1"
tail -8 "$LOG1" | sed 's/^/  | /'

if [[ "$EXPECT" == "baseline_silent" ]]; then
  if grep -q 'error\[E229\]' "$LOG1"; then
    fail "W1 baseline_silent: unexpected E229 on pre-refusal toolchain"
  fi
  # The lie: trailing fn was dropped and check may still succeed.
  if [[ "$rc1" -eq 0 ]]; then
    echo "  W1 baseline CONFIRMED: rc=0 while file contains bytes past $CAP (silent clip)"
  else
    echo "  W1 baseline: rc=$rc1 without E229 (misparse/crash class — still not honest refusal)"
  fi
else
  [[ "$rc1" -ne 0 ]] || fail "W1: check exited 0 despite source past byte wall"
  grep -q 'error\[E229\]' "$LOG1" || fail "W1: missing error[E229]"
  grep -qiE 'byte buffer|2097152' "$LOG1" || fail "W1: E229 did not name the byte capacity"
  echo "  W1 OK: E229 on source past byte wall"
fi

# --- W2: token table wall (comma flood) ------------------------------------
W2="$WORK/w2_token_table.sio"
python3 - <<PY
open(r"$W2", "wb").write(b"," * $CAP)
print(f"W2 commas=$CAP")
PY

LOG2="$WORK/w2.log"
set +e
"$SOUC" check "$W2" >"$LOG2" 2>&1
rc2=$?
set -e
echo "  W2 token-table: rc=$rc2"
tail -6 "$LOG2" | sed 's/^/  | /'

if [[ "$EXPECT" == "baseline_silent" ]]; then
  if grep -q 'error\[E229\]' "$LOG2"; then
    fail "W2 baseline_silent: unexpected E229"
  fi
  echo "  W2 baseline OK: no E229 (rc=$rc2)"
  pass "baseline: silent clip / non-E229 failure still present (W1 rc=$rc1 W2 rc=$rc2)"
  exit 0
fi

[[ "$rc2" -ne 0 ]] || fail "W2: check exited 0 on full token table"
grep -q 'error\[E229\]' "$LOG2" || fail "W2: missing error[E229]"
grep -qiE 'token table full|2097152' "$LOG2" || fail "W2: E229 did not name the token capacity"
echo "  W2 OK: E229 on token table full"

pass "E229 refusal on source-byte and token-table ceilings"
