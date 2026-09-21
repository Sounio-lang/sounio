#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
MANIFEST="$ROOT_DIR/tools/loom/product_exec_ingress_dark.runtime.v1"

fail() {
  printf 'sounio-loom-product-exec-ingress-dark-freeze-selftest: FAIL: %s\n' \
    "$*" >&2
  exit 1
}

record_value() {
  local path="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate field: $key"
      found="$value"
    fi
  done <"$path"
  [[ -n "$found" ]] || fail "missing field: $key"
  printf '%s\n' "$found"
}

expect_field() {
  local key="$1" expected="$2" actual
  actual="$(record_value "$MANIFEST" "$key")"
  [[ "$actual" == "$expected" ]] ||
    fail "$key expected=$expected actual=$actual"
}

expect_binding() {
  local path_key="$1" hash_key="$2" relative expected actual
  relative="$(record_value "$MANIFEST" "$path_key")"
  [[ "$relative" =~ ^[A-Za-z0-9._/-]+$ && "$relative" != /* && \
     "/$relative/" != *'/../'* ]] || fail "unsafe binding: $path_key"
  [[ -f "$ROOT_DIR/$relative" && ! -L "$ROOT_DIR/$relative" ]] ||
    fail "missing binding: $relative"
  expected="$(record_value "$MANIFEST" "$hash_key")"
  actual="$(sha256sum "$ROOT_DIR/$relative" | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] ||
    fail "binding drift: $relative expected=$expected actual=$actual"
}

[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'runtime manifest is absent'
expect_field schema loom-product-exec-ingress-dark-runtime-v1
expect_field stage PRODUCT_DARK_ATTACHMENT_FROZEN
expect_field semantic_authority Sounio
expect_field semantic_action 9031
expect_field parent_exec_grant_action 9030
expect_field operational_language OCaml
expect_field operational_role OPERATIONAL_ATTACHMENT
expect_field runtime_version 2026.08.29.40
expect_field descriptor_transport unix-stream-inherited
expect_field descriptor_is_bearer false
expect_field descriptor_dark_attached true
expect_field distinct_uid_product_broker false
expect_field material_execution false
expect_field production_activation false
expect_field launch_open false
expect_field recycle_open false
expect_field exec_attached false
expect_field commit_attached false
expect_field ci_attached false
expect_field parity_open false
expect_field claim_ready false
expect_field python_executed false
expect_field rust_executed false

for spec in \
  garden_path:garden_sha256 \
  contract_path:contract_sha256 \
  evidence_path:evidence_sha256 \
  exec_ingress_source_path:exec_ingress_source_sha256 \
  hook_source_path:hook_source_sha256 \
  membrane_source_path:membrane_source_sha256 \
  cli_source_path:cli_source_sha256 \
  c_stub_path:c_stub_sha256 \
  dune_path:dune_sha256 \
  gate_path:gate_sha256 \
  counterexample_gate_path:counterexample_gate_sha256 \
  freeze_gate_path:freeze_gate_sha256 \
  action_9031_manifest_path:action_9031_manifest_sha256 \
  action_9031_runtime_path:action_9031_runtime_sha256 \
  resident_v5_manifest_path:resident_v5_manifest_sha256 \
  projection_path:projection_sha256 \
  parent_9030_manifest_path:parent_9030_manifest_sha256 \
  host_grant_manifest_path:host_grant_manifest_sha256 \
  process_witness_manifest_path:process_witness_manifest_sha256 \
  peer_material_judgment_path:peer_material_judgment_sha256; do
  expect_binding "${spec%%:*}" "${spec#*:}"
done

implementation_commit="$(record_value "$MANIFEST" implementation_commit)"
git -C "$ROOT_DIR" cat-file -e "$implementation_commit^{commit}" ||
  fail 'implementation commit is unavailable'
for spec in \
  exec_ingress_source_path:exec_ingress_source_sha256 \
  hook_source_path:hook_source_sha256 \
  membrane_source_path:membrane_source_sha256 \
  cli_source_path:cli_source_sha256 \
  c_stub_path:c_stub_sha256 \
  dune_path:dune_sha256 \
  gate_path:gate_sha256 \
  counterexample_gate_path:counterexample_gate_sha256; do
  path="$(record_value "$MANIFEST" "${spec%%:*}")"
  expected="$(record_value "$MANIFEST" "${spec#*:}")"
  actual="$(git -C "$ROOT_DIR" show "$implementation_commit:$path" | sha256sum | cut -d ' ' -f 1)"
  [[ "$actual" == "$expected" ]] ||
    fail "implementation commit binding drift: $path"
done

dark_output="$(bash "$(record_value "$MANIFEST" gate_path | \
  sed "s#^#$ROOT_DIR/#")")" || fail 'dark attachment gate failed'
[[ "$dark_output" == *'sounio-loom-product-exec-ingress-dark-selftest: PASS'* && \
   "$dark_output" == *'same_uid_self_broker=refused'* && \
   "$dark_output" == *'same_uid_fixture_escape=refused'* && \
   "$dark_output" == *'causal_sabotage=ALLOW0+HOOK_REFUSAL+NO_GRANT'* && \
   "$dark_output" == *'python_executed=false rust_executed=false'* && \
   "$dark_output" == *'exec_attached=false'* ]] ||
  fail "dark attachment receipt is incomplete: $dark_output"

counterexample_output="$(bash "$(record_value "$MANIFEST" \
  counterexample_gate_path | sed "s#^#$ROOT_DIR/#")")" ||
  fail 'historical counterexample gate failed'
[[ "$counterexample_output" == \
   *'sounio-loom-product-exec-ingress-counterexample-selftest: PASS'* && \
   "$counterexample_output" == *'frozen_counterexample=accepted'* && \
   "$counterexample_output" == *'counterexample_falsifies_product_attachment=true'* ]] ||
  fail "historical counterexample receipt is incomplete: $counterexample_output"

(
  flock -x 9
  dune build --root "$ROOT_DIR/tools/loom" src/loom.exe >/dev/null
) 9>"$ROOT_DIR/tools/loom/_build/.dune-build.lock"
runtime="$ROOT_DIR/tools/loom/_build/default/src/loom.exe"
runtime_sha256="$(sha256sum "$runtime" | cut -d ' ' -f 1)"
[[ "$runtime_sha256" == "$(record_value "$MANIFEST" runtime_sha256)" ]] ||
  fail 'native runtime hash drifted'
runtime_info="$($runtime runtime-version)"
[[ "$runtime_info" == *'runtime_version=2026.08.29.40'* && \
   "$runtime_info" == *'language=OCaml'* ]] ||
  fail "native runtime identity drifted: $runtime_info"

if rg -n 'DENY50[2-9]|DENY510|ALLOW code=0 reason=allow' \
  "$ROOT_DIR/tools/loom/src/loom_exec_ingress.ml" \
  "$ROOT_DIR/tools/loom/src/loom_hook.ml" \
  "$ROOT_DIR/tools/loom/src/loom_membrane.ml" >/dev/null; then
  fail 'OCaml copied a Sounio semantic expected-result string'
fi

printf '%s\n' \
  "sounio-loom-product-exec-ingress-dark-freeze-selftest: PASS semantic_authority=Sounio action=9031 parent_action=9030 operational_attachment=OCaml implementation_commit=$implementation_commit runtime_version=2026.08.29.40 runtime_sha256=$runtime_sha256 descriptor_dark_attached=true same_uid_self_broker=refused same_uid_fixture_escape=refused historical_counterexample=preserved causal_sabotage=PASS python_executed=false rust_executed=false distinct_uid_product_broker=false material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false"
