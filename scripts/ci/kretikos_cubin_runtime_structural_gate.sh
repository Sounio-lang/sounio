#!/usr/bin/env bash
# Kretikos CUBIN/runtime structural gate for the modular compiler bridge.
#
# This gate requires the runtime-backed vec_add_f32 profile to emit both PTX
# and CUBIN artifacts and to produce a full K-AXI lowering certificate. It does
# not require a local CUDA driver; driver execution remains a separate platform
# rung.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KRETIKOS="$ROOT_DIR/bin/kretikos"

if [[ ! -x "$KRETIKOS" ]]; then
  echo "error: missing Kretikos launcher: $KRETIKOS" >&2
  exit 1
fi

export SOUNIO_KRETIKOS_COMPILER="${SOUNIO_KRETIKOS_COMPILER:-$ROOT_DIR/bin/souc}"

OUT_DIR="${KRETIKOS_CUBIN_RUNTIME_OUT:-}"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kretikos-cubin-runtime.XXXXXX")"
fi
mkdir -p "$OUT_DIR"

PROFILE_SOURCE="$ROOT_DIR/examples/kretikos/real_vec_add.sio"
LOWER_SOURCE="$ROOT_DIR/examples/kretikos/lower_vec_add_f32.sio"
RUNTIME_DIR="$OUT_DIR/runtime"
CERT_JSON="$OUT_DIR/kaxi_lowering_certificate.json"

echo "=== Kretikos CUBIN/runtime structural gate ==="
echo "out_dir: $OUT_DIR"
"$KRETIKOS" compiler-info | tee "$OUT_DIR/compiler_info.log"

echo "[1/4] runtime-backed source bundle"
"$KRETIKOS" run-source "$PROFILE_SOURCE" -o "$RUNTIME_DIR" --force \
  >"$OUT_DIR/run_source.log" 2>&1

echo "[2/4] validate emitted source runtime profile"
"$KRETIKOS" kaxi-validate-evidence "$RUNTIME_DIR/kretikos_source_profile.v1.json" \
  --expect status=profiled \
  --expect profile=vec_add_f32 \
  --expect runtime_backed=true \
  --expect bundle_manifest=kretikos_bundle.v1.json \
  >"$OUT_DIR/source_profile.validate.log" 2>&1

echo "[3/4] validate emitted runtime bundle"
mapfile -t BUNDLE_PARTS < <("$KRETIKOS" kaxi-validate-evidence "$RUNTIME_DIR/kretikos_bundle.v1.json" \
  --print status \
  --print runtime_validation.status \
  --print runtime_validation.reason)

BUNDLE_STATUS="${BUNDLE_PARTS[0]}"
BUNDLE_RUNTIME_STATUS="${BUNDLE_PARTS[1]}"
BUNDLE_RUNTIME_REASON="${BUNDLE_PARTS[2]}"

if [[ "$BUNDLE_STATUS" != "emitted" ]]; then
  echo "error: runtime bundle was not emitted: $BUNDLE_STATUS" >&2
  exit 1
fi

case "$BUNDLE_RUNTIME_STATUS/$BUNDLE_RUNTIME_REASON" in
  pass/*|not_run/cuda_driver_missing) ;;
  *)
    echo "error: unexpected runtime classification: $BUNDLE_RUNTIME_STATUS/$BUNDLE_RUNTIME_REASON" >&2
    exit 1
    ;;
esac

echo "[4/4] strict source-to-K-AXI certificate with runtime bundle"
"$KRETIKOS" kaxi-lowering-certificate "$LOWER_SOURCE" \
  -o "$CERT_JSON" \
  --force \
  >"$OUT_DIR/kaxi_lowering_certificate.log" 2>&1

mapfile -t CERT_PARTS < <("$KRETIKOS" kaxi-validate-evidence "$CERT_JSON" \
  --print status \
  --print lowering.source_lowered_to_kaxi \
  --print lowering.kaxi_pattern \
  --print runtime.blocked \
  --print runtime.bundle_status \
  --print runtime.runtime_validation.status \
  --print runtime.runtime_validation.reason)

CERT_STATUS="${CERT_PARTS[0]}"
CERT_LOWERED="${CERT_PARTS[1]}"
CERT_PATTERN="${CERT_PARTS[2]}"
CERT_RUNTIME_BLOCKED="${CERT_PARTS[3]}"
CERT_BUNDLE_STATUS="${CERT_PARTS[4]}"
CERT_RUNTIME_STATUS="${CERT_PARTS[5]}"
CERT_RUNTIME_REASON="${CERT_PARTS[6]}"

if [[ "$CERT_STATUS" != "pass" ]]; then
  echo "error: certificate did not pass: $CERT_STATUS" >&2
  exit 1
fi

if [[ "$CERT_LOWERED" != "true" || "$CERT_PATTERN" != "source_vec_add_f32" ]]; then
  echo "error: certificate did not preserve source->K-AXI lowering facts" >&2
  exit 1
fi

if [[ "$CERT_RUNTIME_BLOCKED" != "false" || "$CERT_BUNDLE_STATUS" != "emitted" ]]; then
  echo "error: runtime bundle not present in certificate" >&2
  exit 1
fi

case "$CERT_RUNTIME_STATUS/$CERT_RUNTIME_REASON" in
  pass/*|not_run/cuda_driver_missing) ;;
  *)
  echo "error: unexpected local runtime classification: $CERT_RUNTIME_STATUS/$CERT_RUNTIME_REASON" >&2
  exit 1
  ;;
esac

echo "kretikos_cubin_runtime_structural: PASS status=$CERT_STATUS runtime=$CERT_RUNTIME_STATUS/$CERT_RUNTIME_REASON"
echo "kretikos_cubin_runtime_structural: artifacts=$OUT_DIR"
