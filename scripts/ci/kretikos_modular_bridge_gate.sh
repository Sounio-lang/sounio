#!/usr/bin/env bash
# Kretikos modular compiler bridge gate.
#
# Proves the narrow integration surface that should move first while the
# modular compiler lands:
#   checked Sounio source -> Kretikos profile classification
#   checked Sounio source -> K-AXI lowering witness
#   lowering witness -> certificate with explicit runtime/CUBIN status
#
# This gate intentionally does not claim arbitrary GPU lowering.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KRETIKOS="$ROOT_DIR/bin/kretikos"

if [[ ! -x "$KRETIKOS" ]]; then
  echo "error: missing Kretikos launcher: $KRETIKOS" >&2
  exit 1
fi

export SOUNIO_KRETIKOS_COMPILER="${SOUNIO_KRETIKOS_COMPILER:-$ROOT_DIR/bin/souc}"

OUT_DIR="${KRETIKOS_MODULAR_BRIDGE_OUT:-}"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kretikos-modular-bridge.XXXXXX")"
fi
mkdir -p "$OUT_DIR"

REAL_SOURCE="$ROOT_DIR/examples/kretikos/real_vec_add.sio"
LOWER_SOURCE="$ROOT_DIR/examples/kretikos/lower_vec_add_f32.sio"
PROFILE_LOG="$OUT_DIR/profile_source.log"
LOWERING_JSON="$OUT_DIR/kaxi_source_lowering.json"
CERT_JSON="$OUT_DIR/kaxi_lowering_certificate.json"

echo "=== Kretikos modular bridge gate ==="
echo "out_dir: $OUT_DIR"
"$KRETIKOS" compiler-info | tee "$OUT_DIR/compiler_info.log"

echo "[1/5] source check: runtime-backed profile source"
"$KRETIKOS" check "$REAL_SOURCE" >"$OUT_DIR/check_real_source.log" 2>&1

echo "[2/5] profile-source: runtime-backed classification"
"$KRETIKOS" profile-source "$REAL_SOURCE" | tee "$PROFILE_LOG"
grep -q 'profile=vec_add_f32' "$PROFILE_LOG"
grep -q 'runtime_backed=1' "$PROFILE_LOG"

echo "[3/5] source check: shape-lowering source"
"$KRETIKOS" check "$LOWER_SOURCE" >"$OUT_DIR/check_lower_source.log" 2>&1

echo "[4/5] K-AXI source lowering"
"$KRETIKOS" kaxi-lower-source "$LOWER_SOURCE" -o "$LOWERING_JSON" >"$OUT_DIR/kaxi_lower_source.log" 2>&1
"$KRETIKOS" kaxi-validate-evidence "$LOWERING_JSON" \
  --expect status=pass \
  --expect lowering.source_lowered_to_kaxi=true \
  --expect lowering.kaxi_pattern=source_vec_add_f32 \
  >"$OUT_DIR/kaxi_lower_source.validate.log" 2>&1

echo "[5/5] lowering certificate with honest runtime boundary"
"$KRETIKOS" kaxi-lowering-certificate "$LOWER_SOURCE" \
  -o "$CERT_JSON" \
  --force \
  --allow-runtime-blocked \
  >"$OUT_DIR/kaxi_lowering_certificate.log" 2>&1

mapfile -t CERT_PARTS < <("$KRETIKOS" kaxi-validate-evidence "$CERT_JSON" \
  --print status \
  --print lowering.source_lowered_to_kaxi \
  --print lowering.kaxi_pattern \
  --print runtime.blocked)

CERT_STATUS="${CERT_PARTS[0]}"
CERT_LOWERED="${CERT_PARTS[1]}"
CERT_PATTERN="${CERT_PARTS[2]}"
CERT_RUNTIME_BLOCKED="${CERT_PARTS[3]}"

case "$CERT_STATUS" in
  pass|partial_runtime_blocked) ;;
  *)
    echo "error: unexpected certificate status: $CERT_STATUS" >&2
    exit 1
    ;;
esac

if [[ "$CERT_LOWERED" != "true" || "$CERT_PATTERN" != "source_vec_add_f32" ]]; then
  echo "error: certificate did not preserve source->K-AXI lowering facts" >&2
  exit 1
fi

if [[ "$CERT_STATUS" == "partial_runtime_blocked" && "$CERT_RUNTIME_BLOCKED" != "true" ]]; then
  echo "error: partial runtime certificate did not mark runtime.blocked=true" >&2
  exit 1
fi

echo "kretikos_modular_bridge: PASS status=$CERT_STATUS runtime_blocked=$CERT_RUNTIME_BLOCKED"
echo "kretikos_modular_bridge: artifacts=$OUT_DIR"
