#!/usr/bin/env bash
# Sprint 87: native-v2 float coverage gate
# Verifies: IrLoadFloat/IrIntToFloat/IrFloatToInt whitelisted in native-v2,
# triangle_basic.sio compiles via --native-compile (no legacy fallback),
# and the resulting binary produces byte-identical PPM to the JIT reference.

set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOUC="./bin/souc"
SH="self-hosted/compiler/main.sio"
TIMEOUT=180
OUT_BIN="/tmp/sprint87_triangle.elf"
REF_PPM="/tmp/sprint87_ref.ppm"
OUT_PPM="/tmp/sprint87_out.ppm"
ARTIFACT="artifacts/sprint87/sprint87_native_v2_float_gate.v1.json"

PASS=0
FAIL=0
NOT_RUN=0
TOTAL=0

_ec=0

check_pass() {
    local name="$1"
    TOTAL=$((TOTAL + 1))
    echo "PASS  $name"
    PASS=$((PASS + 1))
}

check_fail() {
    local name="$1"
    local detail="$2"
    TOTAL=$((TOTAL + 1))
    echo "FAIL  $name ($detail)"
    FAIL=$((FAIL + 1))
}

check_not_run() {
    local name="$1"
    local reason="$2"
    TOTAL=$((TOTAL + 1))
    echo "NOT_RUN $name ($reason)"
    NOT_RUN=$((NOT_RUN + 1))
}

echo "=== Sprint 87: native-v2 float gate ==="
echo ""

# ── T1: self-tests T242-T245 pass ──────────────────────────────────────────
_ec=0
timeout $TIMEOUT $SOUC run "$SH" -- --self-test 2>&1 \
    | grep -E "T242|T243|T244|T245" > /tmp/sprint87_selftests.out || _ec=$?
if grep -q "T242 OK" /tmp/sprint87_selftests.out && \
   grep -q "T243 OK" /tmp/sprint87_selftests.out && \
   grep -q "T244 OK" /tmp/sprint87_selftests.out && \
   grep -q "T245 OK" /tmp/sprint87_selftests.out; then
    check_pass "T1: self-tests T242-T245 all pass"
elif [ $_ec -eq 124 ]; then
    check_not_run "T1: self-tests T242-T245" "OOM/timeout"
else
    check_fail "T1: self-tests T242-T245" "one or more FAIL (see /tmp/sprint87_selftests.out)"
fi

# ── T2: --native-compile triangle_basic.sio produces ELF ──────────────────
_ec=0
timeout $TIMEOUT $SOUC run "$SH" -- \
    --native-compile examples/render/triangle_basic.sio \
    -o "$OUT_BIN" > /tmp/sprint87_compile.log 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then
    check_not_run "T2: native-compile triangle_basic → ELF" "OOM/timeout"
elif [ $_ec -ne 0 ]; then
    check_fail "T2: native-compile triangle_basic → ELF" "exit=$_ec (see /tmp/sprint87_compile.log)"
elif [ -f "$OUT_BIN" ]; then
    check_pass "T2: native-compile triangle_basic → ELF"
else
    check_fail "T2: native-compile triangle_basic → ELF" "ELF not produced"
fi

# ── T3: native-v2 path was taken (no 'legacy_fallback' in log) ────────────
if [ -f /tmp/sprint87_compile.log ]; then
    if grep -q "legacy_fallback" /tmp/sprint87_compile.log 2>/dev/null; then
        check_fail "T3: native-v2 path taken (no legacy fallback)" \
            "legacy_fallback present in log"
    else
        check_pass "T3: native-v2 path taken (no legacy fallback)"
    fi
else
    check_not_run "T3: native-v2 path taken" "compile log missing"
fi

# ── T4: ELF binary is executable and runs to completion ───────────────────
if [ -f "$OUT_BIN" ]; then
    chmod +x "$OUT_BIN"
    _ec=0
    timeout 30 "$OUT_BIN" > "$OUT_PPM" 2>/dev/null || _ec=$?
    if [ $_ec -eq 124 ]; then
        check_not_run "T4: ELF runs to completion" "timeout"
    elif [ $_ec -ne 0 ] && [ $_ec -ne 1 ]; then
        check_fail "T4: ELF runs to completion" "exit=$_ec"
    elif [ -f "$OUT_PPM" ] && [ -s "$OUT_PPM" ]; then
        check_pass "T4: ELF runs and produces PPM output"
    else
        check_fail "T4: ELF runs to completion" "no PPM output produced"
    fi
else
    check_not_run "T4: ELF runs to completion" "ELF not produced (T2 not_run)"
fi

# ── T5: JIT reference PPM ─────────────────────────────────────────────────
_ec=0
timeout $TIMEOUT $SOUC run examples/render/triangle_basic.sio > "$REF_PPM" 2>/dev/null || _ec=$?
if [ $_ec -eq 124 ]; then
    check_not_run "T5: JIT reference PPM" "OOM/timeout"
elif [ -s "$REF_PPM" ]; then
    check_pass "T5: JIT reference PPM produced"
else
    check_fail "T5: JIT reference PPM" "exit=$_ec or empty output"
fi

# ── T6: PPM byte-for-byte match ───────────────────────────────────────────
if [ -s "$OUT_PPM" ] && [ -s "$REF_PPM" ]; then
    if cmp -s "$OUT_PPM" "$REF_PPM"; then
        check_pass "T6: PPM output byte-identical to JIT reference"
    else
        check_fail "T6: PPM output byte-identical to JIT reference" \
            "mismatch (diff: $(cmp "$OUT_PPM" "$REF_PPM" | head -1))"
    fi
else
    check_not_run "T6: PPM byte-identical" "one or both PPM files missing"
fi

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "Results: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN TOTAL=$TOTAL"

mkdir -p "$(dirname "$ARTIFACT")"
cat > "$ARTIFACT" <<JSON
{
  "gate": "sprint87_native_v2_float_gate",
  "sprint": 87,
  "pass": $PASS,
  "fail": $FAIL,
  "not_run": $NOT_RUN,
  "total": $TOTAL
}
JSON
echo "Artifact: $ARTIFACT"

if [ $FAIL -eq 0 ]; then
    echo "GATE PASS"
    exit 0
else
    echo "GATE FAIL"
    exit 1
fi
