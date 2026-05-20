#!/usr/bin/env bash
# Structural cross-backend semantic triangulation gate for Kretikos.
#
# This gate proves a narrow claim: the PTX, Metal, and SPIR-V emitters can all
# emit matching structural artifacts for a scalar add kernel and a small
# epistemic dual-output kernel. It does not claim cross-backend runtime
# execution.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_CROSS_BACKEND_DIR:-$ROOT_DIR/artifacts/omega/kretikos_cross_backend_semantic}"
REPORT_JSON="${SOUNIO_KRETIKOS_CROSS_BACKEND_JSON:-$OUT_DIR/kretikos_cross_backend_semantic.v1.json}"
PTXAS_ARCH="${SOUNIO_KRETIKOS_CROSS_BACKEND_PTXAS_ARCH:-sm_89}"
mkdir -p "$OUT_DIR"

VEC_PTX_OUT="$OUT_DIR/vec_add.ptx"
VEC_METAL_OUT="$OUT_DIR/vec_add.metal"
VEC_SPIRV_OUT="$OUT_DIR/vec_add.spv"
EPI_PTX_OUT="$OUT_DIR/epistemic_dual_output_f32.ptx"
EPI_METAL_OUT="$OUT_DIR/epistemic_dual_output_f32.metal"
EPI_SPIRV_OUT="$OUT_DIR/epistemic_dual_output_f32.spv"
VEC_CUBIN_OUT="$OUT_DIR/vec_add.cubin"
EPI_CUBIN_OUT="$OUT_DIR/epistemic_dual_output_f32.cubin"

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

size_file() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

./bin/kretikos emit-ptx vec_add -o "$VEC_PTX_OUT" >/dev/null
./bin/kretikos emit-metal vec_add -o "$VEC_METAL_OUT" >/dev/null
./bin/kretikos emit-spirv vec_add -o "$VEC_SPIRV_OUT" >/dev/null
./bin/kretikos emit-ptx epistemic_dual_output_f32 -o "$EPI_PTX_OUT" >/dev/null
./bin/kretikos emit-metal epistemic_dual_output_f32 -o "$EPI_METAL_OUT" >/dev/null
./bin/kretikos emit-spirv epistemic_dual_output_f32 -o "$EPI_SPIRV_OUT" >/dev/null

ptxas_bin="$(command -v ptxas || true)"
if [[ -z "$ptxas_bin" ]]; then
  echo "kretikos_cross_backend_semantic_gate: FAIL ptxas not found" >&2
  exit 1
fi
ptxas_version="$("$ptxas_bin" --version | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g')"
"$ptxas_bin" "$VEC_PTX_OUT" -arch="$PTXAS_ARCH" -o "$VEC_CUBIN_OUT"
"$ptxas_bin" "$EPI_PTX_OUT" -arch="$PTXAS_ARCH" -o "$EPI_CUBIN_OUT"

vec_ptx_add_marker=false
vec_ptx_global_id_marker=false
vec_metal_add_marker=false
vec_spirv_magic_marker=false
vec_spirv_fadd_marker=false
vec_ptxas_cubin_marker=false
epi_ptx_value_add_marker=false
epi_ptx_eps_mul_marker=false
epi_ptx_valid_and_marker=false
epi_ptx_provenance_xor_marker=false
epi_ptx_global_id_marker=false
epi_metal_value_add_marker=false
epi_metal_eps_square_sum_marker=false
epi_metal_valid_and_marker=false
epi_metal_provenance_xor_marker=false
epi_spirv_magic_marker=false
epi_spirv_fadd_marker=false
epi_spirv_fmul_marker=false
epi_spirv_valid_and_marker=false
epi_spirv_provenance_xor_marker=false
epi_ptxas_cubin_marker=false

if grep -q 'add\.f32' "$VEC_PTX_OUT"; then vec_ptx_add_marker=true; fi
if grep -q '%ctaid\.x' "$VEC_PTX_OUT" && grep -q '%ntid\.x' "$VEC_PTX_OUT" && grep -q 'mad\.lo\.u32' "$VEC_PTX_OUT"; then vec_ptx_global_id_marker=true; fi
if grep -q 'c\[tid\] = a\[tid\] + b_in\[tid\];' "$VEC_METAL_OUT"; then vec_metal_add_marker=true; fi
if head -n 1 "$VEC_SPIRV_OUT" | grep -qx '119734787'; then vec_spirv_magic_marker=true; fi
if grep -qx '327809' "$VEC_SPIRV_OUT"; then vec_spirv_fadd_marker=true; fi
if [[ -s "$VEC_CUBIN_OUT" ]]; then vec_ptxas_cubin_marker=true; fi

if grep -q 'add\.f32' "$EPI_PTX_OUT"; then epi_ptx_value_add_marker=true; fi
if grep -q 'mul\.rn\.f32' "$EPI_PTX_OUT"; then epi_ptx_eps_mul_marker=true; fi
if grep -q 'and\.b32' "$EPI_PTX_OUT"; then epi_ptx_valid_and_marker=true; fi
if grep -q 'xor\.b32' "$EPI_PTX_OUT"; then epi_ptx_provenance_xor_marker=true; fi
if grep -q '%ctaid\.x' "$EPI_PTX_OUT" && grep -q '%ntid\.x' "$EPI_PTX_OUT" && grep -q 'mad\.lo\.u32' "$EPI_PTX_OUT"; then epi_ptx_global_id_marker=true; fi
if grep -q 'out_value\[tid\] = a_value\[tid\] + b_value\[tid\];' "$EPI_METAL_OUT"; then epi_metal_value_add_marker=true; fi
if grep -q 'out_eps\[tid\] = (a_eps\[tid\] \* a_eps\[tid\]) + (b_eps\[tid\] \* b_eps\[tid\]);' "$EPI_METAL_OUT"; then epi_metal_eps_square_sum_marker=true; fi
if grep -q 'out_valid\[tid\] = a_valid\[tid\] & b_valid\[tid\];' "$EPI_METAL_OUT"; then epi_metal_valid_and_marker=true; fi
if grep -q 'out_prov\[tid\] = (tid \^ 0x12345678u) \^ out_valid\[tid\];' "$EPI_METAL_OUT"; then epi_metal_provenance_xor_marker=true; fi
if head -n 1 "$EPI_SPIRV_OUT" | grep -qx '119734787'; then epi_spirv_magic_marker=true; fi
if grep -qx '327809' "$EPI_SPIRV_OUT"; then epi_spirv_fadd_marker=true; fi
if grep -qx '327813' "$EPI_SPIRV_OUT"; then epi_spirv_fmul_marker=true; fi
if grep -qx '327877' "$EPI_SPIRV_OUT"; then epi_spirv_valid_and_marker=true; fi
if grep -qx '327879' "$EPI_SPIRV_OUT"; then epi_spirv_provenance_xor_marker=true; fi
if [[ -s "$EPI_CUBIN_OUT" ]]; then epi_ptxas_cubin_marker=true; fi

if [[ "$vec_ptx_add_marker" != true ||
      "$vec_ptx_global_id_marker" != true ||
      "$vec_metal_add_marker" != true ||
      "$vec_spirv_magic_marker" != true ||
      "$vec_spirv_fadd_marker" != true ||
      "$vec_ptxas_cubin_marker" != true ||
      "$epi_ptx_value_add_marker" != true ||
      "$epi_ptx_eps_mul_marker" != true ||
      "$epi_ptx_valid_and_marker" != true ||
      "$epi_ptx_provenance_xor_marker" != true ||
      "$epi_ptx_global_id_marker" != true ||
      "$epi_metal_value_add_marker" != true ||
      "$epi_metal_eps_square_sum_marker" != true ||
      "$epi_metal_valid_and_marker" != true ||
      "$epi_metal_provenance_xor_marker" != true ||
      "$epi_spirv_magic_marker" != true ||
      "$epi_spirv_fadd_marker" != true ||
      "$epi_spirv_fmul_marker" != true ||
      "$epi_spirv_valid_and_marker" != true ||
      "$epi_spirv_provenance_xor_marker" != true ||
      "$epi_ptxas_cubin_marker" != true ]]; then
  status="fail"
else
  status="pass"
fi

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
artifacts_json="{\"epistemic_dual_output_f32\":{\"cubin\":{\"bytes\":$(size_file "$EPI_CUBIN_OUT"),\"path\":\"${EPI_CUBIN_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$EPI_CUBIN_OUT")\"},\"metal\":{\"bytes\":$(size_file "$EPI_METAL_OUT"),\"path\":\"${EPI_METAL_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$EPI_METAL_OUT")\"},\"ptx\":{\"bytes\":$(size_file "$EPI_PTX_OUT"),\"path\":\"${EPI_PTX_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$EPI_PTX_OUT")\"},\"spirv\":{\"bytes\":$(size_file "$EPI_SPIRV_OUT"),\"path\":\"${EPI_SPIRV_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$EPI_SPIRV_OUT")\"}},\"vec_add_f32\":{\"cubin\":{\"bytes\":$(size_file "$VEC_CUBIN_OUT"),\"path\":\"${VEC_CUBIN_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$VEC_CUBIN_OUT")\"},\"metal\":{\"bytes\":$(size_file "$VEC_METAL_OUT"),\"path\":\"${VEC_METAL_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$VEC_METAL_OUT")\"},\"ptx\":{\"bytes\":$(size_file "$VEC_PTX_OUT"),\"path\":\"${VEC_PTX_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$VEC_PTX_OUT")\"},\"spirv\":{\"bytes\":$(size_file "$VEC_SPIRV_OUT"),\"path\":\"${VEC_SPIRV_OUT#$ROOT_DIR/}\",\"sha256\":\"$(sha_file "$VEC_SPIRV_OUT")\"}}}"

./bin/kretikos json-emit \
  --raw-json "artifacts=$artifacts_json" \
  --array-strings "boundaries=structural_cross_backend_gate_only|does_not_claim_runtime_execution_for_metal_or_spirv|does_not_claim_general_backend_equivalence|does_not_claim_full_user_kernel_lowering|vec_add_oracle_is_scalar_add_1_plus_2_equals_3|epistemic_dual_output_oracle_is_value_add_eps_square_sum_valid_and_provenance_xor" \
  --bool "checks.epistemic_dual_output_f32_metal_eps_square_sum_marker=$epi_metal_eps_square_sum_marker" \
  --bool "checks.epistemic_dual_output_f32_metal_provenance_xor_marker=$epi_metal_provenance_xor_marker" \
  --bool "checks.epistemic_dual_output_f32_metal_valid_and_marker=$epi_metal_valid_and_marker" \
  --bool "checks.epistemic_dual_output_f32_metal_value_add_marker=$epi_metal_value_add_marker" \
  --bool "checks.epistemic_dual_output_f32_ptx_eps_mul_marker=$epi_ptx_eps_mul_marker" \
  --bool "checks.epistemic_dual_output_f32_ptx_global_id_marker=$epi_ptx_global_id_marker" \
  --bool "checks.epistemic_dual_output_f32_ptx_provenance_xor_marker=$epi_ptx_provenance_xor_marker" \
  --bool "checks.epistemic_dual_output_f32_ptx_valid_and_marker=$epi_ptx_valid_and_marker" \
  --bool "checks.epistemic_dual_output_f32_ptx_value_add_marker=$epi_ptx_value_add_marker" \
  --bool "checks.epistemic_dual_output_f32_spirv_fadd_marker=$epi_spirv_fadd_marker" \
  --bool "checks.epistemic_dual_output_f32_spirv_fmul_marker=$epi_spirv_fmul_marker" \
  --bool "checks.epistemic_dual_output_f32_spirv_magic_marker=$epi_spirv_magic_marker" \
  --bool "checks.epistemic_dual_output_f32_spirv_provenance_xor_marker=$epi_spirv_provenance_xor_marker" \
  --bool "checks.epistemic_dual_output_f32_spirv_valid_and_marker=$epi_spirv_valid_and_marker" \
  --bool "checks.epistemic_dual_output_f32_ptxas_cubin_marker=$epi_ptxas_cubin_marker" \
  --bool "checks.vec_add_f32_metal_add_marker=$vec_metal_add_marker" \
  --bool "checks.vec_add_f32_ptx_add_marker=$vec_ptx_add_marker" \
  --bool "checks.vec_add_f32_ptx_global_id_marker=$vec_ptx_global_id_marker" \
  --bool "checks.vec_add_f32_ptxas_cubin_marker=$vec_ptxas_cubin_marker" \
  --bool "checks.vec_add_f32_spirv_fadd_marker=$vec_spirv_fadd_marker" \
  --bool "checks.vec_add_f32_spirv_magic_marker=$vec_spirv_magic_marker" \
  --string "generated_at_utc=$generated_at_utc" \
  --int "oracles.epistemic_dual_output_f32_a_eps_milli=10" \
  --int "oracles.epistemic_dual_output_f32_a_valid=1" \
  --int "oracles.epistemic_dual_output_f32_a_value=1" \
  --int "oracles.epistemic_dual_output_f32_b_eps_milli=20" \
  --int "oracles.epistemic_dual_output_f32_b_valid=1" \
  --int "oracles.epistemic_dual_output_f32_b_value=2" \
  --int "oracles.epistemic_dual_output_f32_expected_eps_square_sum_milli2=500" \
  --int "oracles.epistemic_dual_output_f32_expected_valid=1" \
  --int "oracles.epistemic_dual_output_f32_expected_value=3" \
  --int "oracles.epistemic_dual_output_f32_provenance_seed=305419896" \
  --int "oracles.vec_add_f32_a=1" \
  --int "oracles.vec_add_f32_b=2" \
  --int "oracles.vec_add_f32_expected=3" \
  --array-strings "semantics=vec_add_f32|epistemic_dual_output_f32" \
  --int "spirv.fadd_opcode_word=327809" \
  --int "spirv.fmul_opcode_word=327813" \
  --int "spirv.provenance_xor_opcode_word=327879" \
  --int "spirv.valid_and_opcode_word=327877" \
  --string "schema=sounio.kretikos.cross-backend-semantic.v1" \
  --string "status=$status" \
  --string "toolchain.ptxas_arch=$PTXAS_ARCH" \
  --string "toolchain.ptxas_path=$ptxas_bin" \
  --string "toolchain.ptxas_version=$ptxas_version" \
  > "$REPORT_JSON"

./bin/kretikos kaxi-validate-evidence "$REPORT_JSON" \
  --expect "status=pass" \
  --expect "checks.vec_add_f32_ptx_add_marker=true" \
  --expect "checks.vec_add_f32_ptx_global_id_marker=true" \
  --expect "checks.vec_add_f32_metal_add_marker=true" \
  --expect "checks.vec_add_f32_spirv_magic_marker=true" \
  --expect "checks.vec_add_f32_spirv_fadd_marker=true" \
  --expect "checks.epistemic_dual_output_f32_ptx_value_add_marker=true" \
  --expect "checks.epistemic_dual_output_f32_ptx_eps_mul_marker=true" \
  --expect "checks.epistemic_dual_output_f32_ptx_global_id_marker=true" \
  --expect "checks.epistemic_dual_output_f32_ptx_provenance_xor_marker=true" \
  --expect "checks.epistemic_dual_output_f32_ptx_valid_and_marker=true" \
  --expect "checks.epistemic_dual_output_f32_metal_value_add_marker=true" \
  --expect "checks.epistemic_dual_output_f32_metal_eps_square_sum_marker=true" \
  --expect "checks.epistemic_dual_output_f32_metal_provenance_xor_marker=true" \
  --expect "checks.epistemic_dual_output_f32_metal_valid_and_marker=true" \
  --expect "checks.epistemic_dual_output_f32_spirv_magic_marker=true" \
  --expect "checks.epistemic_dual_output_f32_spirv_fadd_marker=true" \
  --expect "checks.epistemic_dual_output_f32_spirv_fmul_marker=true" \
  --expect "checks.epistemic_dual_output_f32_spirv_provenance_xor_marker=true" \
  --expect "checks.epistemic_dual_output_f32_spirv_valid_and_marker=true" \
  --expect "checks.epistemic_dual_output_f32_ptxas_cubin_marker=true" \
  --expect "checks.vec_add_f32_ptxas_cubin_marker=true" >/dev/null

echo "kretikos_cross_backend_semantic_gate: PASS report=${REPORT_JSON#$ROOT_DIR/}"
