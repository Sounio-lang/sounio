#!/usr/bin/env bash
# Kretikos semantic source-profile gate.
#
# Proves the narrow no-comment path:
#   checked Sounio source -> pure-Sounio source-shape recognizer
#   recognized source_vec_add_f32 -> runtime-backed Kretikos profile
#   profile -> lowering certificate with explicit runtime boundary

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KRETIKOS="$ROOT_DIR/bin/kretikos"

if [[ ! -x "$KRETIKOS" ]]; then
  echo "error: missing Kretikos launcher: $KRETIKOS" >&2
  exit 1
fi

export SOUNIO_KRETIKOS_COMPILER="${SOUNIO_KRETIKOS_COMPILER:-$ROOT_DIR/bin/souc}"

OUT_DIR="${KRETIKOS_SEMANTIC_PROFILE_OUT:-}"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kretikos-semantic-profile.XXXXXX")"
fi
mkdir -p "$OUT_DIR"

LOWER_SOURCE="$ROOT_DIR/examples/kretikos/lower_vec_add_f32.sio"
PROFILE_LOG="$OUT_DIR/profile_source.log"
RUN_DIR="$OUT_DIR/run_source"
LOWERING_JSON="$OUT_DIR/kaxi_source_lowering.json"
CERT_JSON="$OUT_DIR/kaxi_lowering_certificate.json"

echo "=== Kretikos semantic profile gate ==="
echo "out_dir: $OUT_DIR"
"$KRETIKOS" compiler-info | tee "$OUT_DIR/compiler_info.log"

if grep -Eq '^[[:space:]]*//[[:space:]]*kretikos:[[:space:]]*profile[[:space:]]*=' "$LOWER_SOURCE"; then
  echo "error: semantic profile gate source must not carry a kretikos profile directive" >&2
  exit 1
fi

echo "[1/5] source check: no-comment lowering source"
"$KRETIKOS" check "$LOWER_SOURCE" >"$OUT_DIR/check_lower_source.log" 2>&1

echo "[2/5] profile-source: pure-Sounio recognizer selects vec_add_f32"
"$KRETIKOS" profile-source "$LOWER_SOURCE" | tee "$PROFILE_LOG"
grep -q 'profile=vec_add_f32' "$PROFILE_LOG"
grep -q 'class=semantic_source_profile' "$PROFILE_LOG"
grep -q 'runtime_backed=1' "$PROFILE_LOG"

echo "[3/5] run-source: no-comment source emits runtime-backed bundle"
"$KRETIKOS" run-source "$LOWER_SOURCE" -o "$RUN_DIR" --force >"$OUT_DIR/run_source.log" 2>&1
"$KRETIKOS" kaxi-validate-evidence "$RUN_DIR/kretikos_source_profile.v1.json" \
  --expect status=profiled \
  --expect profile=vec_add_f32 \
  --expect runtime_backed=true \
  >"$OUT_DIR/run_source_profile.validate.log" 2>&1

echo "[4/5] K-AXI source lowering still recognizes source_vec_add_f32"
"$KRETIKOS" kaxi-lower-source "$LOWER_SOURCE" -o "$LOWERING_JSON" >"$OUT_DIR/kaxi_lower_source.log" 2>&1
"$KRETIKOS" kaxi-validate-evidence "$LOWERING_JSON" \
  --expect status=pass \
  --expect lowering.source_lowered_to_kaxi=true \
  --expect lowering.kaxi_pattern=source_vec_add_f32 \
  >"$OUT_DIR/kaxi_lower_source.validate.log" 2>&1

echo "[5/5] lowering certificate with semantic profile runtime bundle"
"$KRETIKOS" kaxi-lowering-certificate "$LOWER_SOURCE" \
  -o "$CERT_JSON" \
  --force \
  --allow-runtime-blocked \
  >"$OUT_DIR/kaxi_lowering_certificate.log" 2>&1

mapfile -t CERT_PARTS < <("$KRETIKOS" kaxi-validate-evidence "$CERT_JSON" \
  --print status \
  --print profile.name \
  --print lowering.kaxi_pattern \
  --print runtime.blocked)

CERT_STATUS="${CERT_PARTS[0]}"
CERT_PROFILE="${CERT_PARTS[1]}"
CERT_PATTERN="${CERT_PARTS[2]}"
CERT_RUNTIME_BLOCKED="${CERT_PARTS[3]}"

case "$CERT_STATUS" in
  pass|partial_runtime_blocked) ;;
  *)
    echo "error: unexpected certificate status: $CERT_STATUS" >&2
    exit 1
    ;;
esac

if [[ "$CERT_PROFILE" != "vec_add_f32" || "$CERT_PATTERN" != "source_vec_add_f32" ]]; then
  echo "error: certificate did not preserve semantic profile/lowering facts" >&2
  exit 1
fi

if [[ "$CERT_STATUS" == "partial_runtime_blocked" && "$CERT_RUNTIME_BLOCKED" != "true" ]]; then
  echo "error: partial runtime certificate did not mark runtime.blocked=true" >&2
  exit 1
fi

echo "kretikos_semantic_profile: PASS status=$CERT_STATUS profile=$CERT_PROFILE pattern=$CERT_PATTERN runtime_blocked=$CERT_RUNTIME_BLOCKED"
echo "kretikos_semantic_profile: artifacts=$OUT_DIR"
