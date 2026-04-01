#!/usr/bin/env bash
# Sprint 35 gate: Geometric Algebra stdlib — Cl(p,q,r) multivectors + quaternion Cl(0,2)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0
FAIL=0
TOTAL=0

check() {
    TOTAL=$((TOTAL + 1))
    local desc="$1"
    shift
    if "$@" >/dev/null 2>&1; then
        PASS=$((PASS + 1))
        echo "  PASS  $desc"
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL  $desc"
    fi
}

echo "=== Sprint 35: GA stdlib gate ==="

# --- algebra.sio checks ---
ALG="$ROOT/stdlib/math/ga/algebra.sio"
check "struct AlgebraSignature in algebra.sio"    grep -q 'struct AlgebraSignature' "$ALG"
check "struct Multivector in algebra.sio"          grep -q 'struct Multivector' "$ALG"
check "fn ga_zero in algebra.sio"                  grep -q 'fn ga_zero' "$ALG"
check "fn ga_scalar in algebra.sio"                grep -q 'fn ga_scalar' "$ALG"
check "fn ga_sig_quaternion in algebra.sio"        grep -q 'fn ga_sig_quaternion' "$ALG"
check "fn ga_sig_pga3d in algebra.sio"             grep -q 'fn ga_sig_pga3d' "$ALG"
check "fn ga_basis in algebra.sio"                 grep -q 'fn ga_basis' "$ALG"
check "fn ga_add in algebra.sio"                   grep -q 'fn ga_add' "$ALG"
check "fn ga_grade_extract in algebra.sio"         grep -q 'fn ga_grade_extract' "$ALG"
check "fn ga_norm_sq in algebra.sio"               grep -q 'fn ga_norm_sq' "$ALG"

# --- product.sio checks ---
PRD="$ROOT/stdlib/math/ga/product.sio"
check "fn ga_product in product.sio"               grep -q 'fn ga_product' "$PRD"
check "fn ga_inner in product.sio"                 grep -q 'fn ga_inner' "$PRD"
check "fn ga_outer in product.sio"                 grep -q 'fn ga_outer' "$PRD"
check "fn ga_reverse in product.sio"               grep -q 'fn ga_reverse' "$PRD"
check "fn ga_dual in product.sio"                  grep -q 'fn ga_dual' "$PRD"
check "fn ga_blade_product_sign in product.sio"    grep -q 'fn ga_blade_product_sign' "$PRD"
check "fn ga_canonical_reorder_sign in product.sio" grep -q 'fn ga_canonical_reorder_sign' "$PRD"

# --- quaternion.sio checks ---
QAT="$ROOT/stdlib/math/ga/quaternion.sio"
check "fn quat_new in quaternion.sio"              grep -q 'fn quat_new' "$QAT"
check "fn quat_mul in quaternion.sio"              grep -q 'fn quat_mul' "$QAT"
check "fn quat_conjugate in quaternion.sio"        grep -q 'fn quat_conjugate' "$QAT"
check "fn quat_rotate_vec3 in quaternion.sio"      grep -q 'fn quat_rotate_vec3' "$QAT"
check "fn quat_norm_sq in quaternion.sio"          grep -q 'fn quat_norm_sq' "$QAT"

# --- test files exist ---
check "test ga_algebra_basic.sio exists"           test -f "$ROOT/tests/frontend/ga_algebra_basic.sio"
check "test ga_quaternion_basic.sio exists"         test -f "$ROOT/tests/frontend/ga_quaternion_basic.sio"
check "test ga_product_basic.sio exists"            test -f "$ROOT/tests/frontend/ga_product_basic.sio"

# --- souc check on test files (optional) ---
SOUC="$ROOT/bin/souc"
if [ -x "$SOUC" ]; then
    for tf in ga_algebra_basic.sio ga_quaternion_basic.sio ga_product_basic.sio; do
        check "souc check $tf" "$SOUC" check "$ROOT/tests/frontend/$tf"
    done
else
    echo "  SKIP  souc binary not found, skipping souc check tests"
fi

echo ""
echo "=== Results: $PASS/$TOTAL pass, $FAIL fail ==="

# Write artifact
ARTIFACT_DIR="$ROOT/artifacts/sprint35"
mkdir -p "$ARTIFACT_DIR"
cat > "$ARTIFACT_DIR/ga_stdlib_gate.v1.json" <<EOF
{
  "sprint": 35,
  "gate": "ga_stdlib",
  "version": 1,
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "pass": $PASS,
  "fail": $FAIL,
  "total": $TOTAL,
  "status": "$([ $FAIL -eq 0 ] && echo PASS || echo FAIL)"
}
EOF
echo "Artifact: $ARTIFACT_DIR/ga_stdlib_gate.v1.json"

[ $FAIL -eq 0 ] || exit 1
