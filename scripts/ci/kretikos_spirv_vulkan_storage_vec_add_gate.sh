#!/usr/bin/env bash
# Kretikos-emitted SPIR-V storage-buffer vec-add runtime gate.
#
# This gate reuses the Vulkan runner/oracle from the reference baseline, but
# replaces the reference GLSL module with the in-tree Kretikos SPIR-V emitter.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_SPIRV_VULKAN_STORAGE_DIR:-$ROOT_DIR/artifacts/omega/kretikos_spirv_vulkan_storage_vec_add}"
REPORT_JSON="${SOUNIO_KRETIKOS_SPIRV_VULKAN_STORAGE_JSON:-$OUT_DIR/kretikos_spirv_vulkan_storage_vec_add.v1.json}"
WORDS_TXT="$OUT_DIR/storage_vec_add.kretikos.words.txt"
SPV_BIN="$OUT_DIR/storage_vec_add.kretikos.spv"
SPV_ASM="$OUT_DIR/storage_vec_add.kretikos.spvasm.txt"
RUNNER_SRC="$ROOT_DIR/scripts/gpu/kretikos_spirv_vulkan_storage_vec_add.c"
RUNNER_BIN="$OUT_DIR/kretikos_spirv_vulkan_storage_vec_add"
STDOUT_TXT="$OUT_DIR/kretikos_spirv_vulkan_storage_vec_add.txt"

mkdir -p "$OUT_DIR"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "kretikos_spirv_vulkan_storage_vec_add_gate: FAIL missing required command: $1" >&2
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

require_cmd awk
require_cmd cc
require_cmd jq
require_cmd spirv-dis
require_cmd spirv-val

./bin/souc check self-hosted/gpu/kretikos_emit_spirv.sio
./bin/kretikos emit-spirv storage_vec_add -o "$WORDS_TXT" >/dev/null

awk '
  /^[[:space:]]*[0-9]+[[:space:]]*$/ {
    w = $1 + 0
    printf "%c%c%c%c", w % 256, int(w / 256) % 256, int(w / 65536) % 256, int(w / 16777216) % 256
  }
' "$WORDS_TXT" > "$SPV_BIN"

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
  --arg emitter_source "self-hosted/gpu/kretikos_emit_spirv.sio" \
  --arg emitter_source_sha "$(sha_file "$ROOT_DIR/self-hosted/gpu/kretikos_emit_spirv.sio")" \
  --arg words_path "$(rel_path "$WORDS_TXT")" \
  --arg words_sha "$(sha_file "$WORDS_TXT")" \
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
  --argjson words_bytes "$(size_file "$WORDS_TXT")" \
  --argjson spv_bin_bytes "$(size_file "$SPV_BIN")" \
  --argjson spv_asm_bytes "$(size_file "$SPV_ASM")" \
  --argjson runner_bin_bytes "$(size_file "$RUNNER_BIN")" \
  --argjson stdout_bytes "$(size_file "$STDOUT_TXT")" \
  '{
    schema: "sounio.kretikos.spirv-vulkan-storage-vec-add.v1",
    generated_at_utc: $generated_at_utc,
    status: "pass",
    purpose: "execute a Kretikos-emitted Vulkan storage-buffer vec-add SPIR-V module through the Vulkan oracle runner",
    runtime: {
      device_name: $device_name,
      api_version: $api_version
    },
    artifacts: {
      emitter_source: { path: $emitter_source, sha256: $emitter_source_sha },
      emitted_word_stream: { path: $words_path, sha256: $words_sha, bytes: $words_bytes },
      emitted_spirv_binary: { path: $spv_bin, sha256: $spv_bin_sha, bytes: $spv_bin_bytes },
      emitted_spirv_disassembly: { path: $spv_asm, sha256: $spv_asm_sha, bytes: $spv_asm_bytes },
      runner_source: { path: $runner_src, sha256: $runner_src_sha },
      runner_binary: { path: $runner_bin, sha256: $runner_bin_sha, bytes: $runner_bin_bytes },
      stdout: { path: $stdout_path, sha256: $stdout_sha, bytes: $stdout_bytes }
    },
    checks: {
      kretikos_emitter_source_checks: true,
      kretikos_word_stream_emits: true,
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
      "kretikos_emitted_narrow_storage_vec_add_pattern",
      "does_not_claim_general_storage_buffer_lowering",
      "does_not_claim_source_to_spirv_lowering",
      "does_not_claim_gpu_performance"
    ]
  }' > "$REPORT_JSON"

./bin/kretikos kaxi-validate-evidence "$REPORT_JSON" \
  --expect "status=pass" \
  --expect "checks.kretikos_word_stream_emits=true" \
  --expect "checks.vulkan_storage_buffer_dispatch_pass=true" \
  --expect "checks.cpu_oracle_max_abs_err_zero=true" \
  --expect "oracle.max_abs_err=0" >/dev/null

echo "kretikos_spirv_vulkan_storage_vec_add_gate: PASS report=$(rel_path "$REPORT_JSON")"
