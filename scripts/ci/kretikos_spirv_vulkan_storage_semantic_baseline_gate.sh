#!/usr/bin/env bash
# Vulkan storage-buffer semantic baseline for Kretikos SPIR-V runtime work.
#
# This gate deliberately uses glslangValidator to build a reference SPIR-V
# storage-buffer vec-add shader. It proves the Vulkan descriptor/buffer/oracle
# route on this host before a Kretikos-emitted storage-buffer pattern is wired
# into the same runner.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_SPIRV_VULKAN_STORAGE_BASELINE_DIR:-$ROOT_DIR/artifacts/omega/kretikos_spirv_vulkan_storage_semantic_baseline}"
REPORT_JSON="${SOUNIO_KRETIKOS_SPIRV_VULKAN_STORAGE_BASELINE_JSON:-$OUT_DIR/kretikos_spirv_vulkan_storage_semantic_baseline.v1.json}"
GLSL_SRC="$OUT_DIR/storage_vec_add.comp"
SPV_BIN="$OUT_DIR/storage_vec_add.spv"
SPV_ASM="$OUT_DIR/storage_vec_add.spvasm.txt"
RUNNER_SRC="$ROOT_DIR/scripts/gpu/kretikos_spirv_vulkan_storage_vec_add.c"
RUNNER_BIN="$OUT_DIR/kretikos_spirv_vulkan_storage_vec_add"
STDOUT_TXT="$OUT_DIR/kretikos_spirv_vulkan_storage_vec_add.txt"

mkdir -p "$OUT_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "kretikos_spirv_vulkan_storage_semantic_baseline_gate: FAIL missing required command: $1" >&2
    exit 1
  fi
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

size_file() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

rel_path() {
  local p="$1"
  if [[ "$p" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${p#$ROOT_DIR/}"
  else
    printf '%s' "$p"
  fi
}

kv() {
  local key="$1"
  tr ' ' '\n' < "$STDOUT_TXT" | grep -m1 "^${key}=" | cut -d= -f2-
}

require_cmd cc
require_cmd glslangValidator
require_cmd jq
require_cmd spirv-dis
require_cmd spirv-val

cat > "$GLSL_SRC" <<'GLSL'
#version 450
layout(local_size_x = 1) in;
layout(set = 0, binding = 0) buffer A { float a[]; };
layout(set = 0, binding = 1) buffer B { float b[]; };
layout(set = 0, binding = 2) buffer C { float c[]; };
void main() {
  uint i = gl_GlobalInvocationID.x;
  c[i] = a[i] + b[i];
}
GLSL

glslangValidator -V "$GLSL_SRC" -o "$SPV_BIN" >/dev/null
spirv-val --target-env vulkan1.1 "$SPV_BIN"
spirv-dis "$SPV_BIN" > "$SPV_ASM"

grep -q 'OpEntryPoint GLCompute' "$SPV_ASM"
grep -q 'OpDecorate .* Binding 0' "$SPV_ASM"
grep -q 'OpDecorate .* Binding 1' "$SPV_ASM"
grep -q 'OpDecorate .* Binding 2' "$SPV_ASM"
grep -q 'OpFAdd' "$SPV_ASM"
grep -q 'OpStore' "$SPV_ASM"

cc -O2 "$RUNNER_SRC" -o "$RUNNER_BIN" -lvulkan -lm
"$RUNNER_BIN" "$SPV_BIN" | tee "$STDOUT_TXT"

grep -q 'status=pass' "$STDOUT_TXT"
grep -q 'max_abs_err=0' "$STDOUT_TXT"

generated_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
device_name="$(kv device_name)"
api_version="$(kv api_version)"
max_abs_err="$(kv max_abs_err)"
n="$(kv n)"

jq -n \
  --arg generated_at_utc "$generated_at_utc" \
  --arg device_name "$device_name" \
  --arg api_version "$api_version" \
  --arg max_abs_err "$max_abs_err" \
  --arg glsl_src "$(rel_path "$GLSL_SRC")" \
  --arg glsl_src_sha "$(sha_file "$GLSL_SRC")" \
  --arg spv_bin "$(rel_path "$SPV_BIN")" \
  --arg spv_bin_sha "$(sha_file "$SPV_BIN")" \
  --arg spv_asm "$(rel_path "$SPV_ASM")" \
  --arg spv_asm_sha "$(sha_file "$SPV_ASM")" \
  --arg runner_src "$(rel_path "$RUNNER_SRC")" \
  --arg runner_src_sha "$(sha_file "$RUNNER_SRC")" \
  --arg runner_bin "$(rel_path "$RUNNER_BIN")" \
  --arg runner_bin_sha "$(sha_file "$RUNNER_BIN")" \
  --arg stdout_path "$(rel_path "$STDOUT_TXT")" \
  --arg stdout_sha "$(sha_file "$STDOUT_TXT")" \
  --argjson n "$n" \
  --argjson glsl_src_bytes "$(size_file "$GLSL_SRC")" \
  --argjson spv_bin_bytes "$(size_file "$SPV_BIN")" \
  --argjson spv_asm_bytes "$(size_file "$SPV_ASM")" \
  --argjson runner_bin_bytes "$(size_file "$RUNNER_BIN")" \
  --argjson stdout_bytes "$(size_file "$STDOUT_TXT")" \
  '{
    schema: "sounio.kretikos.spirv-vulkan-storage-semantic-baseline.v1",
    generated_at_utc: $generated_at_utc,
    status: "pass",
    purpose: "prove Vulkan storage-buffer semantic oracle path before Kretikos emits storage-buffer SPIR-V",
    runtime: {
      device_name: $device_name,
      api_version: $api_version
    },
    artifacts: {
      reference_glsl: { path: $glsl_src, sha256: $glsl_src_sha, bytes: $glsl_src_bytes },
      reference_spirv_binary: { path: $spv_bin, sha256: $spv_bin_sha, bytes: $spv_bin_bytes },
      reference_spirv_disassembly: { path: $spv_asm, sha256: $spv_asm_sha, bytes: $spv_asm_bytes },
      runner_source: { path: $runner_src, sha256: $runner_src_sha },
      runner_binary: { path: $runner_bin, sha256: $runner_bin_sha, bytes: $runner_bin_bytes },
      stdout: { path: $stdout_path, sha256: $stdout_sha, bytes: $stdout_bytes }
    },
    checks: {
      glslang_reference_shader_compiles: true,
      spirv_val_vulkan11_pass: true,
      descriptor_bindings_0_1_2_present: true,
      vulkan_storage_buffer_dispatch_pass: true,
      cpu_oracle_max_abs_err_zero: true
    },
    oracle: {
      operation: "c[i] = a[i] + b[i]",
      n: $n,
      max_abs_err: $max_abs_err
    },
    boundaries: [
      "reference_glsl_baseline_not_kretikos_emitted_spirv",
      "proves_vulkan_storage_buffer_oracle_harness_only",
      "does_not_claim_kretikos_storage_buffer_lowering_yet",
      "does_not_claim_gpu_performance"
    ]
  }' > "$REPORT_JSON"

./bin/kretikos kaxi-validate-evidence "$REPORT_JSON" \
  --expect "status=pass" \
  --expect "checks.vulkan_storage_buffer_dispatch_pass=true" \
  --expect "checks.cpu_oracle_max_abs_err_zero=true" \
  --expect "oracle.max_abs_err=0" >/dev/null

echo "kretikos_spirv_vulkan_storage_semantic_baseline_gate: PASS report=$(rel_path "$REPORT_JSON")"
