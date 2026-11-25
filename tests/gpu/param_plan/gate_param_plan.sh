#!/usr/bin/env bash
# tests/gpu/param_plan/gate_param_plan.sh — kernel parameter plan gate
#
# Structural gate. Needs no GPU: it exercises the signature-derived parameter
# plan (self-hosted/gpu/gpu_param_plan.sio) through the public `--backend gpu`
# emission path.
#
# Three sections:
#   1. the plan selftest (unit level, builds kernel IR directly)
#   2. positive fixtures: must emit PTX with the expected `.param` count
#   3. negative fixtures: must fail closed with the expected refusal code
#
# Usage: bash tests/gpu/param_plan/gate_param_plan.sh
# Env:   SOUC            compiler entrypoint (default ./bin/souc)
#        MADAROS_RAW_BIN raw Madaros ELF, when testing a source build
# Exit:  0 = all pass, non-zero = failure count

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
WORK="$(mktemp -d /tmp/gpu-param-plan-gate.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS  $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL  $1: $2"; }

echo "=== GPU parameter plan gate ==="
echo "SOUC: $SOUC"
echo ""

if [ ! -x "$SOUC" ]; then
    echo "FATAL: compiler not executable: $SOUC"
    exit 1
fi

# ── Section 1: plan selftest ─────────────────────────────────────────────────

echo "--- Section 1: plan selftest ---"
selftest_log="$WORK/selftest.log"
if timeout 300 "$SOUC" run tests/gpu/param_plan/plan_selftest.sio >"$selftest_log" 2>&1; then
    if grep -q 'PASS: gpu_param_plan selftest' "$selftest_log"; then
        pass "selftest"
    else
        fail "selftest" "ran but did not print the PASS marker"
    fi
else
    fail "selftest" "exit $? — $(tail -3 "$selftest_log" | tr '\n' ' ')"
fi

# ── Section 2: positive fixtures ────────────────────────────────────────────

echo ""
echo "--- Section 2: positive fixtures ---"

# fixture:expected .param count
positives=(
    "pos_read_write_pair.sio:3"
    "pos_two_inputs.sio:4"
    "pos_scalar_middle.sio:4"
    "pos_in_place.sio:2"
    "pos_extent_later.sio:3"
)

for entry in "${positives[@]}"; do
    fixture="${entry%%:*}"
    want="${entry##*:}"
    src="tests/gpu/param_plan/$fixture"
    out="$WORK/${fixture%.sio}.ptx"
    log="$WORK/${fixture%.sio}.log"

    if ! timeout 300 "$SOUC" build "$src" -o "$out" --backend gpu >"$log" 2>&1; then
        fail "$fixture" "build failed — $(tail -2 "$log" | tr '\n' ' ')"
        continue
    fi
    if [ ! -s "$out" ]; then
        fail "$fixture" "no PTX written"
        continue
    fi

    got="$(grep -c '^    \.param \.' "$out")"
    if [ "$got" = "$want" ]; then
        pass "$fixture ($got params)"
    else
        fail "$fixture" "expected $want .param lines, got $got"
    fi
done

# ── Section 3: negative fixtures ────────────────────────────────────────────

echo ""
echo "--- Section 3: negative fixtures ---"

# fixture:expected refusal code
negatives=(
    "neg_no_extent.sio:no_extent"
    "neg_arity_over_cap.sio:arity_over_cap"
    "neg_no_output_slice.sio:no_output_slice"
    "neg_unsupported_type.sio:unsupported_param_type"
    "neg_multiple_extents.sio:multiple_extents"
)

for entry in "${negatives[@]}"; do
    fixture="${entry%%:*}"
    want="${entry##*:}"
    src="tests/gpu/param_plan/$fixture"
    out="$WORK/${fixture%.sio}.ptx"
    log="$WORK/${fixture%.sio}.log"

    timeout 300 "$SOUC" build "$src" -o "$out" --backend gpu >"$log" 2>&1
    rc=$?

    if [ "$rc" -eq 0 ]; then
        fail "$fixture" "build succeeded, expected a refusal"
        continue
    fi
    if grep -q "\[$want\]" "$log"; then
        pass "$fixture ($want)"
    else
        fail "$fixture" "expected refusal code [$want] — $(tail -2 "$log" | tr '\n' ' ')"
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "=== summary: pass=$PASS fail=$FAIL ==="
[ "$FAIL" -eq 0 ]
