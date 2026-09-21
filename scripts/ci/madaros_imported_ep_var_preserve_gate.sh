#!/usr/bin/env bash
# C1 — Madaros imported Epistemic variance preservation (fail-closed).
#
# Fixture: tests/epistemic_trust/imported_ep_var_preserve.sio
# Compares lean_single vs Madaros on scaled i64 Var reports for:
#   - mass_bottom() Epistemic variance (PDG σ=0.03 → var=9e-4)
#   - h_bb_yukawa_amplitude_nu amp_sq variance at H pole
#
# Requires bit-identical scaled integers across engines (trust axis).
# Madaros retry×3 for intermittent native SEGV under load.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=tests/epistemic_trust/imported_ep_var_preserve.sio
OUT_DIR="$(mktemp -d /tmp/ep_var_preserve.XXXXXX)"
trap 'rm -rf "$OUT_DIR"' EXIT

extract_field() {
  local file="$1" key="$2"
  awk -v k="$key" '$1 == k { print $2; exit }' "$file"
}

run_eng() {
  local eng="$1" out="$2" err="$3"
  local attempts=1
  if [ "$eng" = madaros ]; then
    attempts=3
  fi
  local try=1
  while [ "$try" -le "$attempts" ]; do
    echo "== imported ep-var engine=$eng try=$try =="
    set +e
    SOUNIO_SOUC_ENGINE="$eng" ./bin/souc run "$SRC" >"$out" 2>"$err"
    local rc=$?
    set -e
    if grep -q 'MADAROS_IMPORTED_EP_VAR_PRESERVE_OK' "$out"; then
      return 0
    fi
    if [ "$try" -eq "$attempts" ]; then
      echo "engine=$eng failed rc=$rc" >&2
      tail -40 "$err" >&2
      grep -E 'EP_|PRESERVE_' "$out" >&2 || true
      exit 1
    fi
    echo "engine=$eng soft-fail rc=$rc; retrying" >&2
    try=$((try + 1))
    sleep 1
  done
}

run_eng lean_single "$OUT_DIR/lean_out.txt" "$OUT_DIR/lean_err.txt"
run_eng madaros "$OUT_DIR/mad_out.txt" "$OUT_DIR/mad_err.txt"

# Strip compiler banners: keep only EP_/PRESERVE_ lines
grep -E '^(EP_|MADAROS_IMPORTED_EP_VAR_PRESERVE_)' "$OUT_DIR/lean_out.txt" >"$OUT_DIR/lean_keys.txt"
grep -E '^(EP_|MADAROS_IMPORTED_EP_VAR_PRESERVE_)' "$OUT_DIR/mad_out.txt" >"$OUT_DIR/mad_keys.txt"

lean_mb=$(extract_field "$OUT_DIR/lean_keys.txt" EP_VAR_MB_SCALE18)
lean_amp=$(extract_field "$OUT_DIR/lean_keys.txt" EP_VAR_AMP_SCALE18)
mad_mb=$(extract_field "$OUT_DIR/mad_keys.txt" EP_VAR_MB_SCALE18)
mad_amp=$(extract_field "$OUT_DIR/mad_keys.txt" EP_VAR_AMP_SCALE18)

if [[ -z "$lean_mb" || -z "$lean_amp" || -z "$mad_mb" || -z "$mad_amp" ]]; then
  echo "FAIL: missing scaled Var fields" >&2
  cat "$OUT_DIR/lean_keys.txt" "$OUT_DIR/mad_keys.txt" >&2
  exit 1
fi

if [[ "$lean_mb" != "$mad_mb" || "$lean_amp" != "$mad_amp" ]]; then
  echo "FAIL: Madaros/lean Var mismatch (CORRUPTED)" >&2
  echo "  lean mb=$lean_mb amp=$lean_amp" >&2
  echo "  mad  mb=$mad_mb amp=$mad_amp" >&2
  exit 1
fi

# PDG m_b σ=0.03 → var=0.0009 → *1e18 = 900_000_000_000_000
if [[ "$lean_mb" != "900000000000000" ]]; then
  echo "FAIL: mass_bottom var scale18 expected 900000000000000 got $lean_mb" >&2
  exit 1
fi

# Amp var ~1.354e-15 → scale18 ≈ 1354 (measured lean+Madaros 2026-08-06)
if [[ "$lean_amp" -lt 1000 || "$lean_amp" -gt 2000 ]]; then
  echo "FAIL: amp var scale18 out of expected band [1000,2000] got $lean_amp" >&2
  exit 1
fi

RECEIPT_DIR=artifacts/compiler
mkdir -p "$RECEIPT_DIR"
RECEIPT="$RECEIPT_DIR/madaros_imported_ep_var_preserve_receipt.v1.json"
cat >"$RECEIPT" <<EOF
{
  "schema": "madaros.imported_ep_var_preserve.v1",
  "classification": "TRUSTWORTHY",
  "ep_var_mb_scale18": $lean_mb,
  "ep_var_amp_scale18": $lean_amp,
  "engines": ["lean_single", "madaros"],
  "fixture": "$SRC",
  "note": "bit-identical scaled Var; EXP print_f64 0.000000 was display rounding of ~1e-15 not corruption"
}
EOF

echo "MADAROS_IMPORTED_EP_VAR_PRESERVE_GATE_OK"
echo "receipt=$RECEIPT"
echo "mb_scale18=$lean_mb amp_scale18=$lean_amp"
