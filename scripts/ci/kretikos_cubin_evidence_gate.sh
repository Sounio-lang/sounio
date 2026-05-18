#!/usr/bin/env bash
# Evidence-carrying CUBIN gate for Kretikos.
#
# This is a hardware-backed gate: it requires CUDA Driver API execution to pass.
# The output JSON binds source, K-AXI, PTX, CUBIN, ptxas, runtime device, and a
# validation digest into one auditable artifact.

set -euo pipefail

ulimit -s unlimited 2>/dev/null || true

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

KRETIKOS="$ROOT_DIR/bin/kretikos"
SOURCE_PATH="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_SOURCE:-examples/kretikos/real_vec_add.sio}"
PTX_PATTERN="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_PTX_PATTERN:-vec_add}"
KAXI_PATTERN="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_KAXI_PATTERN:-$PTX_PATTERN}"
CUBIN_KIND="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_CUBIN_KIND:-vec_add_f32}"
RUNTIME_RUNG="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_RUNTIME_RUNG:-vec_add_f32}"
SM_TARGET="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_SM_TARGET:-sm80}"
OUT_DIR="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_DIR:-$ROOT_DIR/artifacts/omega/kretikos_cubin_evidence}"
OUT_JSON="${SOUNIO_KRETIKOS_CUBIN_EVIDENCE_JSON:-$OUT_DIR/kretikos_cubin_evidence_gate.v1.json}"

mkdir -p "$OUT_DIR"

rel_path() {
  local p="$1"
  if [[ "$p" == "$ROOT_DIR/"* ]]; then
    printf '%s' "${p#$ROOT_DIR/}"
  else
    printf '%s' "$p"
  fi
}

sha_file() {
  sha256sum "$1" | awk '{print $1}'
}

size_file() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

json_escape_arg() {
  printf '%s' "$1"
}

if [[ ! -x "$KRETIKOS" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL bin/kretikos missing or not executable" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_PATH" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL source not found: $SOURCE_PATH" >&2
  exit 1
fi
if ! command -v ptxas >/dev/null 2>&1; then
  echo "kretikos_cubin_evidence_gate: FAIL ptxas missing" >&2
  exit 1
fi

KAXI_ASM="$OUT_DIR/${KAXI_PATTERN}.kaxi"
KAXI_WITNESS="$OUT_DIR/${KAXI_PATTERN}.kaxi_witness.json"
BUNDLE_DIR="$OUT_DIR/bundle"
BUNDLE_JSON="$BUNDLE_DIR/kretikos_bundle.v1.json"
DIGEST_INPUT="$OUT_DIR/validation_digest_input.tsv"

"$KRETIKOS" kaxi-witness "$KAXI_PATTERN" \
  -o "$KAXI_WITNESS" \
  --asm-output "$KAXI_ASM" >/dev/null

"$KRETIKOS" bundle \
  -o "$BUNDLE_DIR" \
  --ptx-pattern "$PTX_PATTERN" \
  --cubin-kind "$CUBIN_KIND" \
  --require-toolchain \
  --require-runtime \
  --runtime-rung "$RUNTIME_RUNG" \
  -f >/dev/null

PTX_PATH="$BUNDLE_DIR/kretikos_${PTX_PATTERN}.ptx"
CUBIN_PATH="$BUNDLE_DIR/kretikos_${CUBIN_KIND}.cubin"
PTXAS_CUBIN_PATH="$BUNDLE_DIR/kretikos_${PTX_PATTERN}.ptxas.cubin"

for required in "$KAXI_ASM" "$KAXI_WITNESS" "$BUNDLE_JSON" "$PTX_PATH" "$CUBIN_PATH" "$PTXAS_CUBIN_PATH"; do
  if [[ ! -s "$required" ]]; then
    echo "kretikos_cubin_evidence_gate: FAIL required artifact missing or empty: $required" >&2
    exit 1
  fi
done

KAXI_STATUS="$("$KRETIKOS" kaxi-validate-evidence "$KAXI_WITNESS" --print-or-empty status)"
BUNDLE_STATUS="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty status)"
PTXAS_STATUS="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty toolchain_validation.ptxas.status)"
NVDISASM_STATUS="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty toolchain_validation.nvdisasm.status)"
RUNTIME_STATUS="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.status)"
RUNTIME_REASON="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.reason)"
RUNTIME_KERNEL="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.kernel)"
RUNTIME_DEVICE="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.device_name)"
RUNTIME_DRIVER="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.driver_version)"
RUNTIME_CC_MAJOR="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.cc_major)"
RUNTIME_CC_MINOR="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.cc_minor)"
RUNTIME_CUDA_RESULT="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty runtime_validation.cuda_result)"
PTXAS_PATH="$("$KRETIKOS" kaxi-validate-evidence "$BUNDLE_JSON" --print-or-empty toolchain_validation.ptxas.path)"

if [[ "$KAXI_STATUS" != "pass" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL K-AXI witness status=$KAXI_STATUS" >&2
  exit 1
fi
if [[ "$BUNDLE_STATUS" != "emitted" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL bundle status=$BUNDLE_STATUS" >&2
  exit 1
fi
if [[ "$PTXAS_STATUS" != "pass" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL ptxas status=$PTXAS_STATUS" >&2
  exit 1
fi
if [[ "$NVDISASM_STATUS" != "pass" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL nvdisasm status=$NVDISASM_STATUS" >&2
  exit 1
fi
if [[ "$RUNTIME_STATUS" != "pass" ]]; then
  echo "kretikos_cubin_evidence_gate: FAIL runtime status=$RUNTIME_STATUS reason=$RUNTIME_REASON" >&2
  exit 1
fi

SOURCE_SHA="$(sha_file "$SOURCE_PATH")"
SOURCE_BYTES="$(size_file "$SOURCE_PATH")"
KAXI_SHA="$(sha_file "$KAXI_ASM")"
KAXI_BYTES="$(size_file "$KAXI_ASM")"
KAXI_WITNESS_SHA="$(sha_file "$KAXI_WITNESS")"
PTX_SHA="$(sha_file "$PTX_PATH")"
PTX_BYTES="$(size_file "$PTX_PATH")"
CUBIN_SHA="$(sha_file "$CUBIN_PATH")"
CUBIN_BYTES="$(size_file "$CUBIN_PATH")"
PTXAS_CUBIN_SHA="$(sha_file "$PTXAS_CUBIN_PATH")"
PTXAS_CUBIN_BYTES="$(size_file "$PTXAS_CUBIN_PATH")"
BUNDLE_SHA="$(sha_file "$BUNDLE_JSON")"
PTX_TARGET="$(awk '/^\.target/ { print $2; exit }' "$PTX_PATH")"
PTX_VERSION="$(awk '/^\.version/ { print $2; exit }' "$PTX_PATH")"
PTXAS_VERSION="$(ptxas --version 2>&1 | tr '\n' ' ' | sed 's/[[:space:]][[:space:]]*/ /g; s/[[:space:]]$//')"
GENERATED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

{
  printf 'source_sha256\t%s\n' "$SOURCE_SHA"
  printf 'kaxi_pattern\t%s\n' "$KAXI_PATTERN"
  printf 'kaxi_sha256\t%s\n' "$KAXI_SHA"
  printf 'kaxi_witness_sha256\t%s\n' "$KAXI_WITNESS_SHA"
  printf 'ptx_sha256\t%s\n' "$PTX_SHA"
  printf 'ptx_pattern\t%s\n' "$PTX_PATTERN"
  printf 'cubin_sha256\t%s\n' "$CUBIN_SHA"
  printf 'ptxas_cubin_sha256\t%s\n' "$PTXAS_CUBIN_SHA"
  printf 'bundle_sha256\t%s\n' "$BUNDLE_SHA"
  printf 'sm_target\t%s\n' "$SM_TARGET"
  printf 'ptx_target\t%s\n' "$PTX_TARGET"
  printf 'ptxas_status\t%s\n' "$PTXAS_STATUS"
  printf 'runtime_status\t%s\n' "$RUNTIME_STATUS"
  printf 'runtime_reason\t%s\n' "$RUNTIME_REASON"
  printf 'runtime_rung\t%s\n' "$RUNTIME_RUNG"
  printf 'runtime_kernel\t%s\n' "$RUNTIME_KERNEL"
  printf 'runtime_device\t%s\n' "$RUNTIME_DEVICE"
  printf 'runtime_driver\t%s\n' "$RUNTIME_DRIVER"
  printf 'runtime_cc\t%s.%s\n' "$RUNTIME_CC_MAJOR" "$RUNTIME_CC_MINOR"
} > "$DIGEST_INPUT"
VALIDATION_DIGEST="$(sha_file "$DIGEST_INPUT")"

"$KRETIKOS" json-emit \
  --array-strings "boundaries=evidence_binds_source_kaxi_ptx_cubin_toolchain_runtime_and_digest|runtime_validation_is_cuda_driver_api_launch_and_copyback|ptxas_and_nvdisasm_must_pass|validation_digest_changes_if_any_bound_hash_or_runtime_identity_changes|does_not_claim_arbitrary_sounio_gpu_lowering|does_not_claim_general_cubin_correctness_beyond_selected_rung" \
  --string "bundle.path=$(rel_path "$BUNDLE_JSON")" \
  --string "bundle.sha256=$BUNDLE_SHA" \
  --int "cubin.bytes=$CUBIN_BYTES" \
  --string "cubin.kind=$CUBIN_KIND" \
  --string "cubin.path=$(rel_path "$CUBIN_PATH")" \
  --string "cubin.sha256=$CUBIN_SHA" \
  --string "cubin.sm_target=$SM_TARGET" \
  --string "feature=evidence-carrying-cubin" \
  --string "generated_at_utc=$GENERATED_AT_UTC" \
  --int "kaxi.bytes=$KAXI_BYTES" \
  --string "kaxi.path=$(rel_path "$KAXI_ASM")" \
  --string "kaxi.pattern=$KAXI_PATTERN" \
  --string "kaxi.sha256=$KAXI_SHA" \
  --string "kaxi.witness_json=$(rel_path "$KAXI_WITNESS")" \
  --string "kaxi.witness_sha256=$KAXI_WITNESS_SHA" \
  --int "ptx.bytes=$PTX_BYTES" \
  --string "ptx.path=$(rel_path "$PTX_PATH")" \
  --string "ptx.sha256=$PTX_SHA" \
  --string "ptx.target=$PTX_TARGET" \
  --string "ptx.version=$PTX_VERSION" \
  --string "ptxas.path=$PTXAS_PATH" \
  --string "ptxas.status=$PTXAS_STATUS" \
  --string "ptxas.version=$(json_escape_arg "$PTXAS_VERSION")" \
  --int "ptxas_output_cubin.bytes=$PTXAS_CUBIN_BYTES" \
  --string "ptxas_output_cubin.path=$(rel_path "$PTXAS_CUBIN_PATH")" \
  --string "ptxas_output_cubin.sha256=$PTXAS_CUBIN_SHA" \
  --string "runtime.cc_major=$RUNTIME_CC_MAJOR" \
  --string "runtime.cc_minor=$RUNTIME_CC_MINOR" \
  --string "runtime.cuda_result=$RUNTIME_CUDA_RESULT" \
  --string "runtime.device_name=$RUNTIME_DEVICE" \
  --string "runtime.driver_version=$RUNTIME_DRIVER" \
  --string "runtime.kernel=$RUNTIME_KERNEL" \
  --string "runtime.reason=$RUNTIME_REASON" \
  --string "runtime.rung=$RUNTIME_RUNG" \
  --string "runtime.status=$RUNTIME_STATUS" \
  --string "schema=sounio.kretikos.cubin-evidence-gate.v1" \
  --int "source.bytes=$SOURCE_BYTES" \
  --string "source.path=$SOURCE_PATH" \
  --string "source.sha256=$SOURCE_SHA" \
  --string "status=pass" \
  --string "validation_digest=$VALIDATION_DIGEST" \
  --string "validation_digest_input=$(rel_path "$DIGEST_INPUT")" \
  > "$OUT_JSON"

"$KRETIKOS" kaxi-validate-evidence "$OUT_JSON" \
  --expect "status=pass" \
  --expect "feature=evidence-carrying-cubin" \
  --expect "source.sha256=$SOURCE_SHA" \
  --expect "kaxi.sha256=$KAXI_SHA" \
  --expect "ptx.sha256=$PTX_SHA" \
  --expect "cubin.sha256=$CUBIN_SHA" \
  --expect "ptxas.status=pass" \
  --expect "runtime.status=pass" \
  --expect "runtime.rung=$RUNTIME_RUNG" \
  --expect "validation_digest=$VALIDATION_DIGEST" >/dev/null

echo "kretikos_cubin_evidence_gate: PASS report=$(rel_path "$OUT_JSON") digest=$VALIDATION_DIGEST"
