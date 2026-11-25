#!/usr/bin/env bash
# tests/gpu/gate_launch_plan.sh — device-launch eligibility gate
#
# Structural gate, no GPU needed. The native compile path builds one parameter
# plan per `kernel fn` before codegen; `--gpu-launch-report` prints the verdict.
# This gate pins that verdict, so a shape cannot silently change side between
# "device" and "cpu" without a fixture moving with it.
#
# Usage: bash tests/gpu/gate_launch_plan.sh
# Env:   SOUC            compiler entrypoint (default ./bin/souc)
#        MADAROS_RAW_BIN raw Madaros ELF, when testing a source build
# Exit:  0 = all pass, non-zero = failure count

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC="${SOUC:-./bin/souc}"
WORK="$(mktemp -d /tmp/gpu-launch-plan-gate.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); echo "PASS  $1"; }
fail() { FAIL=$((FAIL + 1)); echo "FAIL  $1: $2"; }

echo "=== GPU launch eligibility gate ==="
echo "SOUC: $SOUC"
echo ""

if [ ! -x "$SOUC" ]; then
    echo "FATAL: compiler not executable: $SOUC"
    exit 1
fi

# ── Section 1: the whole supported launch corpus is device-eligible ─────────

echo "--- Section 1: supported corpus ---"
corpus_log="$WORK/corpus.log"
timeout 600 "$SOUC" build tests/run-pass/gpu_launch_vec_slices.sio \
    -o "$WORK/corpus.elf" --gpu-launch-report >"$corpus_log" 2>&1
corpus_rc=$?

if [ "$corpus_rc" -ne 0 ]; then
    fail "corpus" "build failed — $(tail -2 "$corpus_log" | tr '\n' ' ')"
else
    device="$(grep -c '^launch: .* device ' "$corpus_log")"
    cpu="$(grep -c '^launch: .* cpu:' "$corpus_log")"
    if [ "$device" = "6" ] && [ "$cpu" = "0" ]; then
        pass "corpus (6 device, 0 cpu)"
    else
        fail "corpus" "expected 6 device / 0 cpu, got $device / $cpu"
    fi
    # the report must name the roles it derived, not just the verdict
    if grep -q '^launch: vec_mul_launch device \[extent,slice_in,slice_in,slice_out\]$' "$corpus_log"; then
        pass "corpus roles (vec_mul_launch)"
    else
        fail "corpus roles" "vec_mul_launch did not report its derived roles"
    fi
fi

# ── Section 2: refused shapes report the CPU side and the code ──────────────

echo ""
echo "--- Section 2: refused shapes ---"

# fixture:expected refusal code
refused=(
    "neg_no_extent.sio:no_extent"
    "neg_no_output_slice.sio:no_output_slice"
    "neg_unsupported_type.sio:unsupported_param_type"
)

for entry in "${refused[@]}"; do
    fixture="${entry%%:*}"
    want="${entry##*:}"
    src="tests/gpu/param_plan/$fixture"
    log="$WORK/${fixture%.sio}.log"

    timeout 600 "$SOUC" build "$src" -o "$WORK/${fixture%.sio}.elf" --gpu-launch-report >"$log" 2>&1
    if grep -q "^launch: .* cpu:$want " "$log"; then
        pass "$fixture (cpu:$want)"
    else
        fail "$fixture" "expected a cpu:$want verdict — $(grep '^launch:' "$log" | head -1)"
    fi
done

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "=== summary: pass=$PASS fail=$FAIL ==="
[ "$FAIL" -eq 0 ]
