#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/sovereign_execution_kernel_product.runtime.v1"
EVIDENCE="$ROOT_DIR/tools/loom/evidence/loom-sovereign-execution-kernel-product-v1-20260831.txt"
PRODUCT_GATE="$ROOT_DIR/scripts/ci/sounio_loom_sovereign_execution_kernel_product_selftest.sh"

fail() {
  printf 'sounio-loom-sovereign-execution-kernel-product-freeze-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

value() {
  sed -n "s/^$1=//p" "$MANIFEST" | head -1
}

expect() {
  [[ "$(value "$1")" == "$2" ]] || fail "$1 diverged"
}

verify_file() {
  local path_key="$1" hash_key="$2" relative expected actual
  relative="$(value "$path_key")"
  [[ -n "$relative" && "$relative" != /* && "$relative" != *'..'* ]] || \
    fail "$path_key is unsafe"
  expected="$(value "$hash_key")"
  [[ "$expected" =~ ^[0-9a-f]{64}$ && -f "$ROOT_DIR/$relative" ]] || \
    fail "$hash_key is invalid"
  actual="$(sha256sum "$ROOT_DIR/$relative" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || fail "$relative drifted"
}

[[ -f "$MANIFEST" && -f "$EVIDENCE" && -x "$PRODUCT_GATE" ]] || \
  fail 'product freeze inputs are incomplete'
expect schema loom-sovereign-execution-kernel-product-runtime-v1
expect stage PRODUCT_EXECUTION_FROZEN
expect semantic_authority Sounio
expect semantic_action 9042
expect grant_residency Loom_kernel_memory
expect grant_is_bearer false
expect exported_token false
expect exported_handle false
expect interface_release_authority zero
expect same_uid_peer_isolation true
expect production_activation true
expect exec_attached true
expect write_attached false
expect commit_attached false
expect ci_attached false
expect claim_ready false
expect semantic_manifest_sha256 11a9099d8870646afc3b38302cbc98a257eb284784375cb4011be30214b42f71
expect material_manifest_sha256 2045439f1a07d737a0cb8370ad080a80cd0715db2966863539f3c0794d14d7e3
expect sounio_runtime_sha256 688ff9ce08e42e20f9a681f8388e4bde9d81867922f2084136f697c61d3db715

for pair in \
  contract_path:contract_sha256 \
  semantic_manifest_path:semantic_manifest_sha256 \
  material_manifest_path:material_manifest_sha256 \
  sounio_source_path:sounio_source_sha256 \
  sounio_entrypoint_path:sounio_entrypoint_sha256 \
  loom_source_path:loom_source_sha256 \
  exec_source_path:exec_source_sha256 \
  hook_source_path:hook_source_sha256 \
  sovereign_source_path:sovereign_source_sha256 \
  provider_fixture_path:provider_fixture_sha256 \
  c_stub_path:c_stub_sha256 \
  dune_path:dune_sha256 \
  loom_build_path:loom_build_sha256 \
  installer_path:installer_sha256 \
  coord_runtime_path:coord_runtime_sha256 \
  product_gate_path:product_gate_sha256 \
  freeze_gate_path:freeze_gate_sha256; do
  verify_file "${pair%%:*}" "${pair#*:}"
done

manifest_sha256="$(sha256sum "$MANIFEST" | awk '{print $1}')"
evidence_manifest_sha256="$(sed -n 's/^product_manifest_sha256=//p' "$EVIDENCE" | head -1)"
[[ "$evidence_manifest_sha256" == "$manifest_sha256" ]] || \
  fail 'evidence is not bound to this product manifest'

result="$($PRODUCT_GATE)"
result_sha256="$(printf '%s\n' "$result" | sha256sum | awk '{print $1}')"
expected_result_sha256="$(sed -n 's/^product_selftest_result_sha256=//p' "$EVIDENCE" | head -1)"
[[ "$result_sha256" == "$expected_result_sha256" ]] || \
  fail 'product selftest result diverged from evidence'

printf 'sounio-loom-sovereign-execution-kernel-product-freeze-selftest: PASS semantic_authority=Sounio action=9042 stage=PRODUCT_EXECUTION_FROZEN product_manifest_sha256=%s product_result_sha256=%s grant_residency=Loom_kernel_memory same_uid_peer_isolation=true production_activation=true exec_attached=true write_attached=false commit_attached=false ci_attached=false python_executed=false rust_executed=false claim_ready=false\n' \
  "$manifest_sha256" "$result_sha256"
