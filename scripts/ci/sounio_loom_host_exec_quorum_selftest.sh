#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-host-exec-quorum.XXXXXX")"
BROKER_ONE="$TEST_ROOT/broker-one"
BROKER_TWO="$TEST_ROOT/broker-two"
BARRIER_ONE="$TEST_ROOT/barrier-one"
BARRIER_TWO="$TEST_ROOT/barrier-two"
CONTROLLER_ONE="$TEST_ROOT/controller-one"
CONTROLLER_TWO="$TEST_ROOT/controller-two"
RESIDENT_ONE="$TEST_ROOT/resident-one"
RESIDENT_TWO="$TEST_ROOT/resident-two"
LANGUAGE_ONE="$TEST_ROOT/language-one"
LANGUAGE_TWO="$TEST_ROOT/language-two"
FIXTURE_ONE="$TEST_ROOT/fixture-one"
FIXTURE_TWO="$TEST_ROOT/fixture-two"
BUNDLE_ONE="$TEST_ROOT/fixtures-one.v1"
BUNDLE_TWO="$TEST_ROOT/fixtures-two.v1"
CONTROLLER_MANIFEST="$ROOT_DIR/tools/loom/exec_grant_controller.runtime.v1"
FIXTURE_MANIFEST="$ROOT_DIR/tools/loom/host_exec_quorum_fixture.freeze.v1"
RESIDENT_V4_MANIFEST="$ROOT_DIR/tools/loom/resident_membrane.runtime.v4"
LANGUAGE_MANIFEST="$ROOT_DIR/tools/loom/language_authority.freeze.v1"
FROZEN_CONTROLLER_ROOT="$TEST_ROOT/frozen-controller"
FROZEN_RESIDENT_V4_ROOT="$TEST_ROOT/frozen-resident-v4"
FROZEN_LANGUAGE_ROOT="$TEST_ROOT/frozen-language-authority"

cleanup() {
  rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

fail() {
  printf 'sounio-loom-host-exec-quorum-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

field() {
  local manifest="$1" key="$2" count line
  count="$(grep -c "^${key}=" "$manifest" || true)"
  [[ "$count" == 1 ]] || fail "manifest field $key occurs $count times: $manifest"
  line="$(grep -m1 "^${key}=" "$manifest")"
  printf '%s' "${line#*=}"
}

run_refusal() {
  local label="$1"
  shift
  local output status
  set +e
  output="$("$@" 2>&1)"
  status=$?
  set -e
  [[ $status -eq 70 ]] || fail "$label returned $status instead of 70: $output"
  printf '%s' "$output"
}

run_quorum() {
  "$BROKER_ONE" --selftest-exec-quorum \
    --controller-root "$ROOT_DIR" \
    --controller-manifest "$1" \
    --controller-runtime "$2" \
    --fixture-manifest "$3" \
    --fixture-bundle "$4" \
    --resident-runtime "$5" \
    --barrier-runtime "$6"
}

FROZEN_CONTROLLER_COMMIT="$(field "$CONTROLLER_MANIFEST" controller_commit)"
FROZEN_RESIDENT_V4_COMMIT="$(field "$RESIDENT_V4_MANIFEST" sounio_resident_v4_commit)"
FROZEN_LANGUAGE_COMMIT="$(field "$LANGUAGE_MANIFEST" sounio_executable_commit)"
mkdir -p "$FROZEN_CONTROLLER_ROOT" "$FROZEN_RESIDENT_V4_ROOT" \
  "$FROZEN_LANGUAGE_ROOT"
git -C "$ROOT_DIR" archive "$FROZEN_CONTROLLER_COMMIT" |
  tar -x -C "$FROZEN_CONTROLLER_ROOT"
git -C "$ROOT_DIR" archive "$FROZEN_RESIDENT_V4_COMMIT" |
  tar -x -C "$FROZEN_RESIDENT_V4_ROOT"
git -C "$ROOT_DIR" archive "$FROZEN_LANGUAGE_COMMIT" |
  tar -x -C "$FROZEN_LANGUAGE_ROOT"
[[ "$(sha256sum "$FROZEN_CONTROLLER_ROOT/tools/loom/src/loom_resident.ml" | cut -d ' ' -f 1)" == \
   "$(field "$CONTROLLER_MANIFEST" resident_source_sha256)" ]] ||
  fail 'frozen action-9030 resident source did not rehydrate'
bash "$ROOT_DIR/scripts/ci/sounio_loom_host_exec_quorum_fixture_freeze_selftest.sh" \
  >/dev/null
bash "$ROOT_DIR/scripts/ci/sounio_loom_principal_cell_barrier_selftest.sh" \
  >/dev/null

for output in "$BROKER_ONE" "$BROKER_TWO"; do
  SOUNIO_LOOM_KERNEL_PRINCIPAL_BROKER_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_kernel_principal_broker.sh" >/dev/null
done
for output in "$BARRIER_ONE" "$BARRIER_TWO"; do
  SOUNIO_LOOM_PRINCIPAL_CELL_BARRIER_INTEGRATED_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_loom_principal_cell_barrier_integrated.sh" \
      >/dev/null
done
for output in "$CONTROLLER_ONE" "$CONTROLLER_TWO"; do
  SOUNIO_LOOM_EXEC_GRANT_CONTROLLER_OUTPUT="$output" \
    bash "$FROZEN_CONTROLLER_ROOT/scripts/dev/build_loom_exec_grant_controller.sh" >/dev/null
done
for output in "$RESIDENT_ONE" "$RESIDENT_TWO"; do
  SOUNIO_LOOM_RESIDENT_MEMBRANE_V4_OUTPUT="$output" \
    bash "$FROZEN_RESIDENT_V4_ROOT/scripts/dev/build_sounio_loom_resident_membrane_v4.sh" \
      >/dev/null
done
for output in "$LANGUAGE_ONE" "$LANGUAGE_TWO"; do
  SOUNIO_LOOM_LANGUAGE_AUTHORITY_OUTPUT="$output" \
    bash "$FROZEN_LANGUAGE_ROOT/scripts/dev/build_sounio_loom_language_authority.sh" \
      >/dev/null
done
for output in "$FIXTURE_ONE" "$FIXTURE_TWO"; do
  SOUNIO_LOOM_HOST_EXEC_QUORUM_FIXTURE_OUTPUT="$output" \
    bash "$ROOT_DIR/scripts/dev/build_sounio_loom_host_exec_quorum_fixture.sh" \
      >/dev/null
done
"$FIXTURE_ONE" > "$BUNDLE_ONE"
"$FIXTURE_TWO" > "$BUNDLE_TWO"

cmp "$BROKER_ONE" "$BROKER_TWO" || fail 'two broker builds differ'
cmp "$BARRIER_ONE" "$BARRIER_TWO" || fail 'two integrated barrier builds differ'
cmp "$CONTROLLER_ONE" "$CONTROLLER_TWO" || fail 'two controller builds differ'
cmp "$RESIDENT_ONE" "$RESIDENT_TWO" || fail 'two resident builds differ'
cmp "$LANGUAGE_ONE" "$LANGUAGE_TWO" || fail 'two language-authority builds differ'
cmp "$FIXTURE_ONE" "$FIXTURE_TWO" || fail 'two Sounio fixture builds differ'
cmp "$BUNDLE_ONE" "$BUNDLE_TWO" || fail 'two Sounio fixture bundles differ'
[[ "$(sha256sum "$BARRIER_ONE" | cut -d ' ' -f 1)" == \
  7c67b2b022f9b6c5130573a6cac652a87db3b166a67a0c29673090bc1936d116 ]] ||
  fail 'integrated PrincipalCell runtime hash drifted'
[[ "$(sha256sum "$CONTROLLER_ONE" | cut -d ' ' -f 1)" == \
  a38d499bb01d29b470775fe1a2bd503c7a626b554e425634dbd14508a4747fa7 ]] ||
  fail 'frozen ExecGrantController runtime hash drifted'
[[ "$(sha256sum "$LANGUAGE_ONE" | cut -d ' ' -f 1)" == \
   "$(field "$LANGUAGE_MANIFEST" executable_sha256)" ]] ||
  fail 'frozen Sounio language-authority runtime hash drifted'
[[ "$(sha256sum "$BUNDLE_ONE" | cut -d ' ' -f 1)" == \
  523e132c4ab6a41ade56c2421472b092171627087fe4cf55ba4c74ac1f5d98fe ]] ||
  fail 'frozen Sounio fixture bundle hash drifted'

dependencies="$(
  ldd "$BROKER_ONE" 2>&1 || true
  ldd "$BARRIER_ONE" 2>&1 || true
  ldd "$CONTROLLER_ONE" 2>&1 || true
  ldd "$RESIDENT_ONE" 2>&1 || true
  ldd "$LANGUAGE_ONE" 2>&1 || true
)"
if printf '%s\n' "$dependencies" | grep -Eqi 'python|rust'; then
  fail 'integrated quorum has a prohibited Python or Rust runtime dependency'
fi
if rg -n 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow|code=491|code=499' \
  "$ROOT_DIR/tools/loom/src/loom_exec_quorum_lab.inc" >/dev/null; then
  fail 'material quorum module copied a Sounio expected result'
fi
if rg -n 'DENY49[1-9]|DENY500|DENY501|ALLOW code=0 reason=allow|code=491|code=499' \
  "$ROOT_DIR/tools/loom/src/loom_product_exec_ingress_host_canary.inc" >/dev/null; then
  fail 'product material canary copied a Sounio expected result'
fi
[[ "$(rg -c '/usr/bin/python3 --version' \
  "$ROOT_DIR/tools/loom/src/loom_product_exec_ingress_host_canary.inc")" == 1 && \
   "$(rg -c '/usr/bin/rustc --version' \
  "$ROOT_DIR/tools/loom/src/loom_product_exec_ingress_host_canary.inc")" == 1 ]] ||
  fail 'product prohibited-oracle controls drifted'
if rg -n 'SOUNIO_COORD_NATIVE_HOOK_SELFTEST' \
  "$ROOT_DIR/tools/loom/src/loom_product_exec_ingress_host_canary.inc" >/dev/null; then
  fail 'product host canary enabled the coordination-hook selftest bypass'
fi

result="$(run_quorum "$CONTROLLER_MANIFEST" "$CONTROLLER_ONE" \
  "$FIXTURE_MANIFEST" "$BUNDLE_ONE" "$RESIDENT_ONE" "$BARRIER_ONE")"
[[ "$result" == \
  'LOOM_HOST_EXEC_QUORUM_SELFTEST PASS semantic_authority=Sounio controller=OCaml role=EFFECT_PARITY material_layer=C++20+Linux material_role=MATERIAL_PARITY transitory=true single_resident_controller=true inherited_descriptor=true treatment=closed positive_semantics=ready positive_local=closed positive_local_reason=same-uid-principal exact_write_sabotage=open replay=closed controller_death=closed resident_death=closed wrong_generation=closed python=closed textual_receipt=closed principal_pidfd=bound principal_start_tick=bound principal_executable=bound principal_cgroup=bound principal_distinct_uid=false non_bearer_transport=measured same_uid_peer_isolation=false material_threshold_measured=true descriptor_barrier_causal=true material_grant=false material_execution=false barrier_release=false launch_open=false exec_attached=false parity_open=false claim_ready=false' ]] ||
  fail "integrated causal matrix diverged: $result"

protocol="$($BROKER_ONE --selftest-protocol)"
[[ "$protocol" == *'launch=closed recycle=closed unknown=denied'* ]] ||
  fail 'existing broker public protocol opened during integration'
public_barrier="$(run_refusal public-integrated-barrier "$BARRIER_ONE" \
  --internal-principal-cell)"
[[ "$public_barrier" == *'missing integrated environment: SOUNIO_LOOM_PRINCIPAL_CELL_INTERNAL'* ]] ||
  fail 'integrated PrincipalCell exposed an unmarked public entrypoint'

TAMPERED_CONTROLLER_MANIFEST="$TEST_ROOT/controller-manifest-tampered"
TAMPERED_CONTROLLER="$TEST_ROOT/controller-tampered"
TAMPERED_FIXTURE_MANIFEST="$TEST_ROOT/fixture-manifest-tampered"
TAMPERED_BUNDLE="$TEST_ROOT/bundle-tampered"
TAMPERED_BARRIER="$TEST_ROOT/barrier-tampered"
cp "$CONTROLLER_MANIFEST" "$TAMPERED_CONTROLLER_MANIFEST"
cp "$CONTROLLER_ONE" "$TAMPERED_CONTROLLER"
cp "$FIXTURE_MANIFEST" "$TAMPERED_FIXTURE_MANIFEST"
cp "$BUNDLE_ONE" "$TAMPERED_BUNDLE"
cp "$BARRIER_ONE" "$TAMPERED_BARRIER"
printf '\n' >> "$TAMPERED_CONTROLLER_MANIFEST"
printf 'X' >> "$TAMPERED_CONTROLLER"
printf '\n' >> "$TAMPERED_FIXTURE_MANIFEST"
printf 'X' >> "$TAMPERED_BUNDLE"
printf 'X' >> "$TAMPERED_BARRIER"

refusal="$(run_refusal controller-manifest-tamper run_quorum \
  "$TAMPERED_CONTROLLER_MANIFEST" "$CONTROLLER_ONE" "$FIXTURE_MANIFEST" \
  "$BUNDLE_ONE" "$RESIDENT_ONE" "$BARRIER_ONE")"
[[ "$refusal" == *'ExecGrantController manifest hash mismatch'* ]] ||
  fail 'controller manifest mutation reached the transaction'
refusal="$(run_refusal controller-runtime-tamper run_quorum \
  "$CONTROLLER_MANIFEST" "$TAMPERED_CONTROLLER" "$FIXTURE_MANIFEST" \
  "$BUNDLE_ONE" "$RESIDENT_ONE" "$BARRIER_ONE")"
[[ "$refusal" == *'controller runtime hash mismatch'* ]] ||
  fail 'controller runtime mutation reached the transaction'
refusal="$(run_refusal fixture-manifest-tamper run_quorum \
  "$CONTROLLER_MANIFEST" "$CONTROLLER_ONE" "$TAMPERED_FIXTURE_MANIFEST" \
  "$BUNDLE_ONE" "$RESIDENT_ONE" "$BARRIER_ONE")"
[[ "$refusal" == *'Sounio ExecQuorum fixtures manifest hash mismatch'* ]] ||
  fail 'fixture manifest mutation reached the transaction'
refusal="$(run_refusal fixture-bundle-tamper run_quorum \
  "$CONTROLLER_MANIFEST" "$CONTROLLER_ONE" "$FIXTURE_MANIFEST" \
  "$TAMPERED_BUNDLE" "$RESIDENT_ONE" "$BARRIER_ONE")"
[[ "$refusal" == *'Sounio fixture bundle hash mismatch'* ]] ||
  fail 'fixture bundle mutation reached the transaction'
refusal="$(run_refusal barrier-runtime-tamper run_quorum \
  "$CONTROLLER_MANIFEST" "$CONTROLLER_ONE" "$FIXTURE_MANIFEST" \
  "$BUNDLE_ONE" "$RESIDENT_ONE" "$TAMPERED_BARRIER")"
[[ "$refusal" == *'integrated PrincipalCell runtime hash mismatch'* ]] ||
  fail 'barrier runtime mutation reached the transaction'

printf 'sounio-loom-host-exec-quorum-selftest: PASS semantic_authority=Sounio controller=OCaml controller_role=EFFECT_PARITY material_layer=C++20+Linux material_role=MATERIAL_PARITY transitory=true single_resident_controller=true source_fixtures=Sounio frozen_commits_rehydrated=true controller_commit=%s resident_v4_commit=%s language_authority_commit=%s inherited_descriptor=true deterministic_rebuilds=broker+barrier+controller+resident+language-authority+fixtures treatment=closed positive_semantics=ready positive_local=closed positive_local_reason=same-uid-principal exact_write_sabotage=open causal_rule=three-object-quorum replay=closed controller_death=closed resident_death=closed wrong_generation=closed python=closed textual_receipt=closed controller_manifest_tamper=refused controller_runtime_tamper=refused fixture_manifest_tamper=refused fixture_bundle_tamper=refused barrier_runtime_tamper=refused principal_pidfd=bound principal_start_tick=bound principal_executable=bound principal_cgroup=bound principal_distinct_uid=false non_bearer_transport=measured same_uid_peer_isolation=false public_protocol=closed expected_results_encoded_in_material_layer=false prohibited_oracle_controls=python+rust runtime_dependencies=clean material_threshold_measured=true descriptor_barrier_causal=true material_grant=false material_execution=false barrier_release=false launch_open=false exec_attached=false parity_open=false claim_ready=false language_authority_binary_sha256=%s broker_source_sha256=%s quorum_module_sha256=%s product_canary_source_sha256=%s barrier_source_sha256=%s broker_binary_sha256=%s barrier_binary_sha256=%s\n' \
  "$FROZEN_CONTROLLER_COMMIT" "$FROZEN_RESIDENT_V4_COMMIT" \
  "$FROZEN_LANGUAGE_COMMIT" "$(sha256sum "$LANGUAGE_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_kernel_principal_broker.cpp" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_exec_quorum_lab.inc" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_product_exec_ingress_host_canary.inc" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_principal_cell_barrier_integrated.cpp" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BROKER_ONE" | cut -d ' ' -f 1)" \
  "$(sha256sum "$BARRIER_ONE" | cut -d ' ' -f 1)"
