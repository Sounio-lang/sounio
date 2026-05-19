#!/usr/bin/env bash
# Structural cross-backend semantic triangulation gate for Kretikos.
#
# This gate proves a narrow claim: the PTX, Metal, and SPIR-V emitters can all
# emit a vec_add-shaped kernel/artifact, and each artifact contains the backend
# marker for addition. It does not claim cross-backend runtime execution.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_CROSS_BACKEND_DIR:-$ROOT_DIR/artifacts/omega/kretikos_cross_backend_semantic}"
REPORT_JSON="${SOUNIO_KRETIKOS_CROSS_BACKEND_JSON:-$OUT_DIR/kretikos_cross_backend_semantic.v1.json}"
mkdir -p "$OUT_DIR"

PTX_OUT="$OUT_DIR/vec_add.ptx"
METAL_OUT="$OUT_DIR/vec_add.metal"
SPIRV_OUT="$OUT_DIR/vec_add.spv"

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

size_file() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

./bin/kretikos emit-ptx vec_add -o "$PTX_OUT" >/dev/null
./bin/kretikos emit-metal vec_add -o "$METAL_OUT" >/dev/null
./bin/kretikos emit-spirv vec_add -o "$SPIRV_OUT" >/dev/null

ptx_add_marker=false
metal_add_marker=false
spirv_magic_marker=false
spirv_fadd_marker=false

if grep -q 'add\.f32' "$PTX_OUT"; then ptx_add_marker=true; fi
if grep -q 'c\[tid\] = a\[tid\] + b_in\[tid\];' "$METAL_OUT"; then metal_add_marker=true; fi
if head -n 1 "$SPIRV_OUT" | grep -qx '119734787'; then spirv_magic_marker=true; fi
if grep -qx '327809' "$SPIRV_OUT"; then spirv_fadd_marker=true; fi

if [[ "$ptx_add_marker" != true || "$metal_add_marker" != true || "$spirv_magic_marker" != true || "$spirv_fadd_marker" != true ]]; then
  status="fail"
else
  status="pass"
fi

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ptx_sha="$(sha_file "$PTX_OUT")"
metal_sha="$(sha_file "$METAL_OUT")"
spirv_sha="$(sha_file "$SPIRV_OUT")"
ptx_bytes="$(size_file "$PTX_OUT")"
metal_bytes="$(size_file "$METAL_OUT")"
spirv_bytes="$(size_file "$SPIRV_OUT")"
artifacts_json="{\"metal\":{\"bytes\":$metal_bytes,\"path\":\"${METAL_OUT#$ROOT_DIR/}\",\"sha256\":\"$metal_sha\"},\"ptx\":{\"bytes\":$ptx_bytes,\"path\":\"${PTX_OUT#$ROOT_DIR/}\",\"sha256\":\"$ptx_sha\"},\"spirv\":{\"bytes\":$spirv_bytes,\"path\":\"${SPIRV_OUT#$ROOT_DIR/}\",\"sha256\":\"$spirv_sha\"}}"

./bin/kretikos json-emit \
  --raw-json "artifacts=$artifacts_json" \
  --array-strings "boundaries=structural_cross_backend_gate_only|does_not_claim_runtime_execution_for_metal_or_spirv|does_not_claim_general_backend_equivalence|does_not_claim_full_user_kernel_lowering|semantic_oracle_is_scalar_add_1_plus_2_equals_3" \
  --bool "checks.metal_add_marker=$metal_add_marker" \
  --bool "checks.ptx_add_marker=$ptx_add_marker" \
  --bool "checks.spirv_fadd_marker=$spirv_fadd_marker" \
  --bool "checks.spirv_magic_marker=$spirv_magic_marker" \
  --string "generated_at_utc=$generated_at_utc" \
  --int "oracle.a=1" \
  --int "oracle.b=2" \
  --int "oracle.expected=3" \
  --int "spirv.fadd_opcode_word=327809" \
  --string "schema=sounio.kretikos.cross-backend-semantic.v1" \
  --string "semantic=vec_add_f32" \
  --string "status=$status" \
  > "$REPORT_JSON"

./bin/kretikos kaxi-validate-evidence "$REPORT_JSON" \
  --expect "status=pass" \
  --expect "semantic=vec_add_f32" \
  --expect "checks.ptx_add_marker=true" \
  --expect "checks.metal_add_marker=true" \
  --expect "checks.spirv_magic_marker=true" \
  --expect "checks.spirv_fadd_marker=true" >/dev/null

echo "kretikos_cross_backend_semantic_gate: PASS report=${REPORT_JSON#$ROOT_DIR/}"
