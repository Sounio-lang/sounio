#!/usr/bin/env bash
# Kretikos-emitted Vulkan storage-buffer vec-add gate.
#
# This gate replaces the reference GLSL-generated SPIR-V from the baseline
# gate with `bin/kretikos emit-spirv storage_vec_add`, then executes the
# emitted module through the same Vulkan storage-buffer runner and CPU oracle.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="${SOUNIO_KRETIKOS_SPIRV_VULKAN_STORAGE_VEC_ADD_DIR:-$ROOT_DIR/artifacts/omega/kretikos_spirv_vulkan_storage_vec_add}"
REPORT_JSON="${SOUNIO_KRETIKOS_SPIRV_VULKAN_STORAGE_VEC_ADD_JSON:-$OUT_DIR/kretikos_spirv_vulkan_storage_vec_add.v1.json}"
KRETIKOS_WORDS="$OUT_DIR/kretikos_storage_vec_add.words.spv"
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

require_cmd cc
require_cmd jq

./bin/kretikos emit-spirv storage_vec_add -o "$KRETIKOS_WORDS" >/dev/null

head -n1 "$KRETIKOS_WORDS" | grep -qx '119734787'
grep -q '^327809$' "$KRETIKOS_WORDS"
grep -q '^196670$' "$KRETIKOS_WORDS"

cc -O2 "$RUNNER_SRC" -o "$RUNNER_BIN" -lvulkan -lm
"$RUNNER_BIN" "$KRETIKOS_WORDS" | tee "$STDOUT_TXT"

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
  --arg kretikos_words "$(rel_path "$KRETIKOS_WORDS")" \
  --arg kretikos_words_sha "$(sha_file "$KRETIKOS_WORDS")" \
  --arg runner_src "$(rel_path "$RUNNER_SRC")" \
  --arg runner_src_sha "$(sha_file "$RUNNER_SRC")" \
  --arg runner_bin "$(rel_path "$RUNNER_BIN")" \
  --arg runner_bin_sha "$(sha_file "$RUNNER_BIN")" \
  --arg stdout_path "$(rel_path "$STDOUT_TXT")" \
  --arg stdout_sha "$(sha_file "$STDOUT_TXT")" \
  --argjson n "$n" \
  --argjson kretikos_words_bytes "$(size_file "$KRETIKOS_WORDS")" \
  --argjson runner_bin_bytes "$(size_file "$RUNNER_BIN")" \
  --argjson stdout_bytes "$(size_file "$STDOUT_TXT")" \
  '{
    schema: "sounio.kretikos.spirv-vulkan-storage-vec-add.v1",
    generated_at_utc: $generated_at_utc,
    status: "pass",
    purpose: "prove Kretikos-emitted SPIR-V storage-buffer vec-add through the Vulkan storage-buffer oracle",
    runtime: {
      device_name: $device_name,
      api_version: $api_version
    },
    artifacts: {
      kretikos_decimal_words_spirv: { path: $kretikos_words, sha256: $kretikos_words_sha, bytes: $kretikos_words_bytes },
      runner_source: { path: $runner_src, sha256: $runner_src_sha },
      runner_binary: { path: $runner_bin, sha256: $runner_bin_sha, bytes: $runner_bin_bytes },
      stdout: { path: $stdout_path, sha256: $stdout_sha, bytes: $stdout_bytes }
    },
    checks: {
      kretikos_emit_spirv_storage_vec_add_succeeds: true,
      spirv_magic_word_present: true,
      fadd_opcode_word_present: true,
      vulkan_storage_buffer_dispatch_pass: true,
      cpu_oracle_max_abs_err_zero: true
    },
    oracle: {
      operation: "c[i] = a[i] + b[i]",
      n: $n,
      max_abs_err: $max_abs_err
    },
    boundaries: [
      "kretikos_emitted_template_spirv_not_source_lowered_spirv",
      "proves_storage_buffer_vec_add_semantics_through_vulkan_runner",
      "does_not_claim_general_spirv_lowering",
      "does_not_claim_gpu_performance"
    ]
  }' > "$REPORT_JSON"

./bin/kretikos kaxi-validate-evidence "$REPORT_JSON" \
  --expect "status=pass" \
  --expect "checks.kretikos_emit_spirv_storage_vec_add_succeeds=true" \
  --expect "checks.vulkan_storage_buffer_dispatch_pass=true" \
  --expect "checks.cpu_oracle_max_abs_err_zero=true" \
  --expect "oracle.max_abs_err=0" >/dev/null

echo "kretikos_spirv_vulkan_storage_vec_add_gate: PASS report=$(rel_path "$REPORT_JSON")"
