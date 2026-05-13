#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_VERIFY_DIR:-$(mktemp -d /tmp/kretikos-ptx-kaxi-verify.XXXXXX)}"
mkdir -p "$OUT_DIR"

PATTERNS=(
  vec_add
  vec_sub
  vec_mul
  vec_div
  fma
)

PASS=0
FAIL=0

for pattern in "${PATTERNS[@]}"; do
  ptx_out="$OUT_DIR/${pattern}.ptx"
  kaxi_out="$OUT_DIR/${pattern}.kaxi"

  echo "verify: pattern=$pattern"

  # Emit PTX
  ./bin/kretikos emit-ptx "$pattern" -o "$ptx_out" >/dev/null 2>&1 || {
    echo "  FAIL: PTX emission failed"
    FAIL=$((FAIL + 1))
    continue
  }

  # Emit K-AXI
  ./bin/kretikos emit-kaxi "$pattern" -o "$kaxi_out" >/dev/null 2>&1 || {
    echo "  FAIL: K-AXI emission failed"
    FAIL=$((FAIL + 1))
    continue
  }

  # Structural checks
  ptx_has_ret=$(grep -c "ret;" "$ptx_out" || true)
  kaxi_has_ret=$(grep -c "ret seq=" "$kaxi_out" || true)

  if [[ "$ptx_has_ret" -lt 1 ]]; then
    echo "  FAIL: PTX missing ret"
    FAIL=$((FAIL + 1))
    continue
  fi

  if [[ "$kaxi_has_ret" -lt 1 ]]; then
    echo "  FAIL: K-AXI missing ret"
    FAIL=$((FAIL + 1))
    continue
  fi

  # Pattern-specific opcode checks
  case "$pattern" in
    vec_add)
      ptx_has=$(grep -c "add" "$ptx_out" || true)
      kaxi_has=$(grep -c "add r" "$kaxi_out" || true)
      ;;
    vec_sub)
      ptx_has=$(grep -c "sub" "$ptx_out" || true)
      kaxi_has=$(grep -c "sub r" "$kaxi_out" || true)
      ;;
    vec_mul)
      ptx_has=$(grep -c "mul" "$ptx_out" || true)
      kaxi_has=$(grep -c "mul r" "$kaxi_out" || true)
      ;;
    vec_div)
      ptx_has=$(grep -c "div" "$ptx_out" || true)
      kaxi_has=$(grep -c "div r" "$kaxi_out" || true)
      ;;
    fma)
      ptx_has=$(grep -c "fma" "$ptx_out" || true)
      kaxi_has=$(grep -c "fma r" "$kaxi_out" || true)
      ;;
    *)
      ptx_has=1
      kaxi_has=1
      ;;
  esac

  if [[ "$ptx_has" -lt 1 ]]; then
    echo "  FAIL: PTX missing $pattern opcode"
    FAIL=$((FAIL + 1))
    continue
  fi

  if [[ "$kaxi_has" -lt 1 ]]; then
    echo "  FAIL: K-AXI missing $pattern opcode"
    FAIL=$((FAIL + 1))
    continue
  fi

  echo "  PASS"
  PASS=$((PASS + 1))
done

echo ""
if [[ "$FAIL" -eq 0 ]]; then
  echo "kretikos_ptx_kaxi_verify: PASS ($PASS patterns)"
  exit 0
else
  echo "kretikos_ptx_kaxi_verify: FAIL ($PASS pass, $FAIL fail)"
  exit 1
fi
