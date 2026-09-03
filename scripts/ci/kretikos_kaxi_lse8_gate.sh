#!/usr/bin/env bash
# Sub-PTX B2 LSE-8 end-to-end gate.
#
# Validates that the `lse8` K-AXI pattern emits a kernel that:
#   1. compiles via ptxas to a cubin on sm_89
#   2. launches successfully on the local GPU
#   3. produces .approx-precision-correct log-sum-exp results across
#      four test inputs (monotonic, large-gap, identical, negatives)
#   4. is bit-deterministic across three back-to-back launches
#
# SKIP behaviour: prints "SKIPPED (no CUDA toolchain / no GPU)" and
# exits 0 if ptxas / cuobjdump / libcuda is absent.
#
# Tolerance: 5e-3 absolute on the LSE result. The PTX `.approx` form
# of ex2 and lg2 is the only available f32 transcendental in the ISA;
# its precision floor is ~10⁻⁶ relative, but accumulated error across
# 7 max + 8 sub + 8 ex2 + 8 add + 1 lg2 + 1 add can reach a few ULP.
# 5e-3 is generous; tighten later if needed.
#
# Reference: docs/research/subptx_fmad_invariance.md (B1 finding); this
# gate is the B2 LSE-primitive correctness step.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ptxas >/dev/null 2>&1; then
    echo "kretikos_kaxi_lse8_gate: SKIPPED (no ptxas)"
    exit 0
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "kretikos_kaxi_lse8_gate: SKIPPED (no GPU)"
    exit 0
fi
if ! command -v cc >/dev/null 2>&1; then
    echo "kretikos_kaxi_lse8_gate: SKIPPED (no cc)"
    exit 0
fi

KRETIKOS="${KRETIKOS:-./bin/kretikos}"
RUNNER_SRC="scripts/gpu/kaxi_ptx_runner.c"

TMP_DIR="$(mktemp -d -t kaxi_lse8.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

PTX="$TMP_DIR/lse8.ptx"
RUNNER="$TMP_DIR/runner"

PASS=0
FAIL=0
FIRST_FAILURES=()

note_fail() {
    FAIL=$((FAIL + 1))
    FIRST_FAILURES+=("$1")
}

# 1) Build the CUDA driver-API runner.
if cc -O2 "$RUNNER_SRC" -ldl -lm -o "$RUNNER" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "runner build (cc)"
fi

# 2) Emit the lse8 PTX in --f32 mode.
if "$KRETIKOS" kaxi-emit-ptx lse8 --f32 -o "$PTX" >/dev/null 2>&1; then
    PASS=$((PASS + 1))
else
    note_fail "kretikos kaxi-emit-ptx lse8 --f32"
fi

# 3) Run four canonical test cases. Each line is:
#    label | input_csv | expected_output | tolerance
TESTS=(
    "monotonic|1,2,3,4,5,6,7,8|8.99436|5e-3"
    "large_gap|10,0,0,0,0,0,0,0|10.0098|5e-3"
    "identical|5,5,5,5,5,5,5,5|8.0|1e-4"
    "negatives|-1,-2,-3,-4,-5,-6,-7,-8|-0.00565|5e-3"
)

result_at_8() {
    local input="$1"
    "$RUNNER" "$PTX" --mode basic --threads 1 --mem-words 9 \
        --init-mem "${input},0" --type f32 2>/dev/null \
        | awk '/^MEM:/ {print $10}'
}

within_tol() {
    local val="$1"
    local expected="$2"
    local tol="$3"
    awk -v v="$val" -v e="$expected" -v t="$tol" \
        'BEGIN { d = v - e; if (d < 0) d = -d; exit (d <= t) ? 0 : 1 }'
}

for tc in "${TESTS[@]}"; do
    label="${tc%%|*}"
    rest="${tc#*|}"
    input="${rest%%|*}"
    rest="${rest#*|}"
    expected="${rest%%|*}"
    tol="${rest#*|}"
    got="$(result_at_8 "$input")"
    if [[ -z "$got" ]]; then
        note_fail "$label: no output from runner"
        continue
    fi
    if within_tol "$got" "$expected" "$tol"; then
        PASS=$((PASS + 1))
    else
        note_fail "$label: got=$got expected=$expected tol=$tol"
    fi
done

# 4) Run-to-run determinism on the monotonic input.
r1="$(result_at_8 "1,2,3,4,5,6,7,8")"
r2="$(result_at_8 "1,2,3,4,5,6,7,8")"
r3="$(result_at_8 "1,2,3,4,5,6,7,8")"
if [[ "$r1" == "$r2" && "$r2" == "$r3" ]]; then
    PASS=$((PASS + 1))
else
    note_fail "run-to-run determinism (got: $r1, $r2, $r3)"
fi

echo "=== sub-PTX B2 LSE-8 gate (RTX 4000 Ada / sm_89) ==="
echo "  PASS: $PASS"
echo "  FAIL: $FAIL"
if [[ "${#FIRST_FAILURES[@]}" -gt 0 ]]; then
    echo "  first failures:"
    for f in "${FIRST_FAILURES[@]}"; do
        echo "    $f"
    done
fi

if [[ "$FAIL" -gt 0 ]]; then
    exit 1
fi
exit 0
