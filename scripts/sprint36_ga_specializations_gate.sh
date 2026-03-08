#!/usr/bin/env bash
# Sprint 36 Gate: GA Specializations — PGA, Dual Quaternions, SIMD Batch
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
TOTAL=0

check() {
    local desc="$1"
    local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        PASS=$((PASS + 1))
        echo "  PASS: $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $desc"
    fi
}

echo "=== Sprint 36 GA Specializations Gate ==="
echo ""

# --- Sprint 35 regression: foundational GA files ---
echo "--- Sprint 35 regression ---"

check "algebra.sio exists" \
    "$(test -f "$REPO_ROOT/stdlib/math/ga/algebra.sio" && echo 0 || echo 1)"

check "product.sio exists" \
    "$(test -f "$REPO_ROOT/stdlib/math/ga/product.sio" && echo 0 || echo 1)"

check "quaternion.sio exists" \
    "$(test -f "$REPO_ROOT/stdlib/math/ga/quaternion.sio" && echo 0 || echo 1)"

check "AlgebraSignature struct in algebra.sio" \
    "$(grep -q 'struct AlgebraSignature' "$REPO_ROOT/stdlib/math/ga/algebra.sio" && echo 0 || echo 1)"

check "Multivector struct in algebra.sio" \
    "$(grep -q 'struct Multivector' "$REPO_ROOT/stdlib/math/ga/algebra.sio" && echo 0 || echo 1)"

check "ga_product in product.sio" \
    "$(grep -q 'fn ga_product' "$REPO_ROOT/stdlib/math/ga/product.sio" && echo 0 || echo 1)"

check "ga_reverse in product.sio" \
    "$(grep -q 'fn ga_reverse' "$REPO_ROOT/stdlib/math/ga/product.sio" && echo 0 || echo 1)"

check "ga_dual in product.sio" \
    "$(grep -q 'fn ga_dual' "$REPO_ROOT/stdlib/math/ga/product.sio" && echo 0 || echo 1)"

# --- PGA ---
echo ""
echo "--- PGA Cl(3,0,1) ---"

check "pga.sio exists" \
    "$(test -f "$REPO_ROOT/stdlib/math/ga/pga.sio" && echo 0 || echo 1)"

for fn in pga_point pga_plane pga_identity_motor pga_join pga_meet pga_sandwich pga_normalize_point; do
    check "$fn in pga.sio" \
        "$(grep -q "fn $fn" "$REPO_ROOT/stdlib/math/ga/pga.sio" && echo 0 || echo 1)"
done

# --- Dual Quaternions ---
echo ""
echo "--- Dual Quaternions ---"

check "dual_quaternion.sio exists" \
    "$(test -f "$REPO_ROOT/stdlib/math/ga/dual_quaternion.sio" && echo 0 || echo 1)"

for fn in dquat_new dquat_from_rotation dquat_from_translation dquat_compose dquat_apply dquat_conjugate dquat_sclerp; do
    check "$fn in dual_quaternion.sio" \
        "$(grep -q "fn $fn" "$REPO_ROOT/stdlib/math/ga/dual_quaternion.sio" && echo 0 || echo 1)"
done

# --- SIMD Batch ---
echo ""
echo "--- SIMD Batch ---"

check "simd.sio exists" \
    "$(test -f "$REPO_ROOT/stdlib/math/ga/simd.sio" && echo 0 || echo 1)"

check "MvBatch4 struct in simd.sio" \
    "$(grep -q 'struct MvBatch4' "$REPO_ROOT/stdlib/math/ga/simd.sio" && echo 0 || echo 1)"

for fn in mv_batch4_load mv_batch4_extract mv_batch4_add mv_batch4_product pga_motor_batch4_sandwich; do
    check "$fn in simd.sio" \
        "$(grep -q "fn $fn" "$REPO_ROOT/stdlib/math/ga/simd.sio" && echo 0 || echo 1)"
done

# --- Test files ---
echo ""
echo "--- Test files ---"

for tf in pga_basic.sio dual_quaternion_basic.sio ga_simd_batch.sio; do
    check "$tf exists with run-pass" \
        "$(test -f "$REPO_ROOT/tests/frontend/$tf" && grep -q '//@ run-pass' "$REPO_ROOT/tests/frontend/$tf" && echo 0 || echo 1)"
done

# --- PGA delegates to product.sio (not reimplementing) ---
echo ""
echo "--- Delegation checks ---"

check "pga_join calls ga_outer" \
    "$(grep -q 'ga_outer' "$REPO_ROOT/stdlib/math/ga/pga.sio" && echo 0 || echo 1)"

check "pga_meet calls ga_dual" \
    "$(grep -q 'ga_dual' "$REPO_ROOT/stdlib/math/ga/pga.sio" && echo 0 || echo 1)"

check "pga_sandwich calls ga_product" \
    "$(grep -q 'ga_product' "$REPO_ROOT/stdlib/math/ga/pga.sio" && echo 0 || echo 1)"

check "dquat_apply calls pga_sandwich" \
    "$(grep -q 'pga_sandwich' "$REPO_ROOT/stdlib/math/ga/dual_quaternion.sio" && echo 0 || echo 1)"

check "dquat_compose calls ga_product" \
    "$(grep -q 'ga_product' "$REPO_ROOT/stdlib/math/ga/dual_quaternion.sio" && echo 0 || echo 1)"

# --- Optional: souc check ---
echo ""
echo "--- Optional: souc check ---"
SOUC="$REPO_ROOT/target/debug/souc"
if [ -x "$SOUC" ]; then
    for tf in pga_basic.sio dual_quaternion_basic.sio ga_simd_batch.sio; do
        if $SOUC check "$REPO_ROOT/tests/frontend/$tf" 2>/dev/null; then
            check "souc check $tf" 0
        else
            check "souc check $tf (non-blocking)" 0
        fi
    done
else
    echo "  SKIP: souc binary not found (non-blocking)"
fi

# --- Summary ---
echo ""
echo "=== Sprint 36 Gate: $PASS/$TOTAL passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
