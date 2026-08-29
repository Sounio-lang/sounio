#!/usr/bin/env bash

set -euo pipefail
umask 077

fail() {
  printf 'sounio-loom-host-exec-quorum-host-gate: FAIL reason=%s material_grant=false material_execution=false launch_open=false\n' "$*" >&2
  exit 1
}

unavailable() {
  printf 'sounio-loom-host-exec-quorum-host-gate: HOST_GATE_UNAVAILABLE reason=%s material_grant=false material_execution=false launch_open=false\n' "$*" >&2
  exit 77
}

usage() {
  printf 'usage: %s --release ABSOLUTE_PATH --expected-manifest-sha256 HEX\n' "$0" >&2
  exit 64
}

sha256_file() {
  sha256sum "$1" | cut -d ' ' -f 1
}

record_value() {
  local path="$1" key="$2" line name value found=''
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ "$line" == *=* ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    if [[ "$name" == "$key" ]]; then
      [[ -z "$found" ]] || fail "duplicate manifest field: $key"
      found="$value"
    fi
  done < "$path"
  [[ -n "$found" ]] || fail "manifest omitted field: $key"
  printf '%s\n' "$found"
}

RELEASE=''
EXPECTED_MANIFEST_SHA256=''
while [[ $# -gt 0 ]]; do
  case "$1" in
    --release)
      [[ $# -ge 2 ]] || usage
      RELEASE="$2"
      shift 2
      ;;
    --expected-manifest-sha256)
      [[ $# -ge 2 ]] || usage
      EXPECTED_MANIFEST_SHA256="$2"
      shift 2
      ;;
    *) usage ;;
  esac
done

[[ "$RELEASE" == /* && "$EXPECTED_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]] || usage
[[ "$(id -u)" == 0 && "$(id -g)" == 0 ]] || unavailable 'root identity is absent'
[[ "$(tr -d '\n' < /proc/1/comm 2>/dev/null)" == systemd ]] || unavailable 'PID 1 is not systemd'
[[ -d /run/systemd/system ]] || unavailable 'systemd runtime directory is absent'
for tool in systemctl systemd-run sha256sum stat hostname uname timeout readlink find sed cut; do
  command -v "$tool" >/dev/null 2>&1 || unavailable "required host tool is absent: $tool"
done
[[ -d "$RELEASE" && ! -L "$RELEASE" ]] || fail 'experimental release is absent or linked'
[[ "$(stat -c '%u:%g' "$RELEASE")" == 0:0 ]] || fail 'experimental release is not root-owned'
[[ -z "$(find "$RELEASE" -perm /022 -print -quit)" ]] || fail 'experimental release is writable by group or world'

MANIFEST="$RELEASE/release.manifest.v1"
[[ -f "$MANIFEST" && ! -L "$MANIFEST" ]] || fail 'experimental release manifest is absent or linked'
[[ "$(sha256_file "$MANIFEST")" == "$EXPECTED_MANIFEST_SHA256" ]] || fail 'experimental release manifest hash drifted'
[[ "$(record_value "$MANIFEST" schema)" == loom-host-exec-quorum-experiment-release-v1 ]] || fail 'release schema drifted'
[[ "$(record_value "$MANIFEST" stage)" == PARITY_OPEN_CANDIDATE ]] || fail 'release stage is not a parity candidate'
[[ "$(record_value "$MANIFEST" semantic_authority)" == Sounio ]] || fail 'release semantic authority drifted'
[[ "$(record_value "$MANIFEST" controller_language)" == OCaml ]] || fail 'release controller language drifted'
[[ "$(record_value "$MANIFEST" material_role)" == MATERIAL_PARITY ]] || fail 'release material role drifted'
for closed in material_grant material_execution launch_open recycle_open exec_attached commit_attached ci_attached parity_open claim_ready; do
  [[ "$(record_value "$MANIFEST" "$closed")" == false ]] || fail "release prematurely opened $closed"
done

verify_binding() {
  local path_key="$1" hash_key="$2" mode="$3" relative path
  relative="$(record_value "$MANIFEST" "$path_key")"
  [[ "$relative" =~ ^[A-Za-z0-9._/-]+$ && "$relative" != /* && "/$relative/" != *'/../'* ]] ||
    fail "release binding path is unsafe: $path_key"
  path="$RELEASE/$relative"
  [[ -f "$path" && ! -L "$path" ]] || fail "release binding is absent or linked: $relative"
  [[ "$(sha256_file "$path")" == "$(record_value "$MANIFEST" "$hash_key")" ]] || fail "release binding hash drifted: $relative"
  [[ "$(stat -c '%u:%g:%a' "$path")" == "0:0:$mode" ]] || fail "release binding metadata drifted: $relative"
  printf '%s\n' "$path"
}

BROKER="$(verify_binding broker_path broker_sha256 555)"
CONTROLLER_MANIFEST="$(verify_binding controller_manifest_path controller_manifest_sha256 444)"
CONTROLLER_RUNTIME="$(verify_binding controller_runtime_path controller_runtime_sha256 555)"
CONTROLLER_FROZEN_RESIDENT="$(verify_binding controller_frozen_resident_path controller_frozen_resident_sha256 444)"
CONTROLLER_FROZEN_CELL="$(verify_binding controller_frozen_cell_path controller_frozen_cell_sha256 444)"
CONTROLLER_FROZEN_SOURCE="$(verify_binding controller_frozen_source_path controller_frozen_source_sha256 444)"
FIXTURE_MANIFEST="$(verify_binding fixture_manifest_path fixture_manifest_sha256 444)"
FIXTURE_BUNDLE="$(verify_binding fixture_bundle_path fixture_bundle_sha256 444)"
RESIDENT_RUNTIME="$(verify_binding resident_runtime_path resident_runtime_sha256 555)"
LOCAL_BARRIER="$(verify_binding local_barrier_path local_barrier_sha256 555)"
HOST_BARRIER="$(verify_binding host_barrier_path host_barrier_sha256 555)"
PROCESS_WITNESS_CELL="$(verify_binding process_witness_cell_path process_witness_cell_sha256 555)"
PROCESS_WITNESS_PAYLOAD="$(verify_binding process_witness_payload_path process_witness_payload_sha256 555)"
PROCESS_WITNESS_MANIFEST="$(verify_binding process_witness_manifest_path process_witness_manifest_sha256 444)"
PROCESS_WITNESS_GARDEN="$(verify_binding process_witness_garden_path process_witness_garden_sha256 444)"
PRODUCT_RUNTIME="$(verify_binding product_exec_ingress_runtime_path product_exec_ingress_runtime_sha256 555)"
PRODUCT_LANGUAGE_RUNTIME="$(verify_binding product_language_runtime_path product_language_runtime_sha256 555)"
PRODUCT_RESIDENT_RUNTIME="$(verify_binding product_resident_runtime_path product_resident_runtime_sha256 555)"
PRODUCT_INGRESS_MANIFEST="$(verify_binding product_exec_ingress_manifest_path product_exec_ingress_manifest_sha256 444)"
PRODUCT_INGRESS_CONTRACT="$(verify_binding product_exec_ingress_contract_path product_exec_ingress_contract_sha256 444)"
PRODUCT_INGRESS_EVIDENCE="$(verify_binding product_exec_ingress_evidence_path product_exec_ingress_evidence_sha256 444)"
PRODUCT_LANE_CELL_CANARY_CONTRACT="$(verify_binding product_lane_cell_canary_contract_path product_lane_cell_canary_contract_sha256 444)"
[[ "$(record_value "$MANIFEST" controller_frozen_commit)" =~ ^[0-9a-f]{40}$ && \
   "$(record_value "$MANIFEST" resident_v4_frozen_commit)" =~ ^[0-9a-f]{40}$ && \
   "$(sha256_file "$CONTROLLER_FROZEN_RESIDENT")" == \
     "$(record_value "$CONTROLLER_MANIFEST" resident_source_sha256)" && \
   "$(sha256_file "$CONTROLLER_FROZEN_CELL")" == \
     "$(record_value "$CONTROLLER_MANIFEST" cell_source_sha256)" && \
   "$(sha256_file "$CONTROLLER_FROZEN_SOURCE")" == \
     "$(record_value "$CONTROLLER_MANIFEST" controller_source_sha256)" ]] ||
  fail 'frozen action-9030 controller provenance drifted'
[[ "$(record_value "$MANIFEST" process_witness_core)" == false && \
   "$(record_value "$MANIFEST" complete_effects)" == false ]] ||
  fail 'release preclaimed ProcessWitness completion'
[[ -s "$PROCESS_WITNESS_GARDEN" ]] || fail 'ProcessWitness Garden is empty'
AUTHORITY_ROOT="$RELEASE/$(record_value "$MANIFEST" authority_root_path)"
[[ -d "$AUTHORITY_ROOT" && ! -L "$AUTHORITY_ROOT" && -d "$AUTHORITY_ROOT/.git" ]] ||
  fail 'authority root topology is incomplete'
PRODUCT_ROOT="$RELEASE/$(record_value "$MANIFEST" product_authority_root_path)"
[[ "$PRODUCT_ROOT" == "$AUTHORITY_ROOT" && -d "$PRODUCT_ROOT/.git" && \
   "$(record_value "$MANIFEST" product_exec_ingress_action)" == 9031 && \
   "$(record_value "$MANIFEST" product_lane_cell_canary)" == false && \
   "$(record_value "$MANIFEST" distinct_uid_product_broker_canary)" == false ]] ||
  fail 'product ExecIngress release posture drifted'
[[ -s "$PRODUCT_INGRESS_MANIFEST" && -s "$PRODUCT_INGRESS_CONTRACT" && \
   -s "$PRODUCT_INGRESS_EVIDENCE" && -s "$PRODUCT_LANE_CELL_CANARY_CONTRACT" ]] ||
  fail 'product ExecIngress proof capsule is incomplete'

SYSTEMD_RUN="$(readlink -f "$(command -v systemd-run)")"
SYSTEMCTL="$(readlink -f "$(command -v systemctl)")"
[[ "$SYSTEMD_RUN" == /* && -x "$SYSTEMD_RUN" && "$SYSTEMCTL" == /* && -x "$SYSTEMCTL" ]] ||
  unavailable 'canonical systemd tools are unavailable'
read -r _ SYSTEMD_VERSION _ < <(systemctl --version | sed -n '1p')
[[ "$SYSTEMD_VERSION" =~ ^[0-9]+$ ]] || fail 'systemd version is not canonical'
(( SYSTEMD_VERSION >= 253 )) || unavailable 'systemd OpenFile transport requires version 253 or newer'

set +e
host_output="$(timeout --signal=TERM --kill-after=5s 150s \
  "$BROKER" --selftest-host-exec-quorum \
  --controller-manifest "$CONTROLLER_MANIFEST" \
  --controller-runtime "$CONTROLLER_RUNTIME" \
  --controller-root "$AUTHORITY_ROOT" \
  --fixture-manifest "$FIXTURE_MANIFEST" \
  --fixture-bundle "$FIXTURE_BUNDLE" \
  --resident-runtime "$RESIDENT_RUNTIME" \
  --barrier-runtime "$LOCAL_BARRIER" \
  --host-barrier-runtime "$HOST_BARRIER" \
  --systemd-run "$SYSTEMD_RUN" --systemctl "$SYSTEMCTL" 2>&1)"
host_status=$?
set -e
[[ $host_status -eq 0 ]] || fail "host broker matrix failed or timed out status=$host_status output=$host_output"
[[ "$host_output" == 'LOOM_HOST_EXEC_QUORUM_HOST_GATE PASS '* ]] || fail 'host broker receipt prefix diverged'
for expectation in \
  'semantic_authority=Sounio' 'controller=OCaml' \
  'controller_role=EFFECT_PARITY' 'material_role=MATERIAL_PARITY' \
  'systemd_transport_authority=false' 'dynamic_user=true' \
  'inherited_descriptor=true' 'arm_authority=false' \
  'single_resident_controller=true' 'treatment=closed' \
  'positive_host=open' 'positive_open_sentinels=1' \
  'exact_write_sabotage=open' 'sabotage_open_sentinels=1' \
  'total_open_sentinels=2' 'second_release=refused' \
  'replay=closed' 'controller_death=closed' 'resident_death=closed' \
  'wrong_generation=closed' 'python=closed' 'textual_receipt=closed' \
  'same_uid=closed' 'causal_rule=three-object-quorum' \
  'causal_sabotage=PASS' 'principal_pidfd=bound' \
  'principal_start_tick=bound' 'principal_executable=bound' \
  'principal_cgroup=bound' 'principal_distinct_uid=true' \
  'non_bearer_exec_quorum=true' 'descriptor_barrier_causal=true' \
  'linear_grant_consumption=true' 'material_grant=true' \
  'material_execution=false' 'launch_open=false' 'recycle_open=false' \
  'exec_attached=false' 'commit_attached=false' 'ci_attached=false' \
  'parity_open=false' 'claim_ready=false'; do
  [[ " $host_output " == *" $expectation "* ]] || fail "host broker receipt omitted $expectation"
done

set +e
product_output="$(timeout --signal=TERM --kill-after=5s 240s \
  "$BROKER" --selftest-product-exec-ingress-host \
  --product-root "$PRODUCT_ROOT" \
  --product-runtime "$PRODUCT_RUNTIME" \
  --product-language-runtime "$PRODUCT_LANGUAGE_RUNTIME" \
  --product-resident-runtime "$PRODUCT_RESIDENT_RUNTIME" \
  --systemd-run "$SYSTEMD_RUN" --systemctl "$SYSTEMCTL" 2>&1)"
product_status=$?
set -e
[[ $product_status -eq 0 ]] ||
  fail "product DynamicUser ExecIngress matrix failed or timed out status=$product_status output=$product_output"
[[ "$product_output" == 'LOOM_PRODUCT_EXEC_INGRESS_DYNAMIC_USER_HOST_GATE PASS '* ]] ||
  fail 'product DynamicUser ExecIngress receipt prefix diverged'
for expectation in \
  'semantic_authority=Sounio' 'action=9031' \
  'operational_attachment=OCaml' 'material_role=MATERIAL_PARITY' \
  'hostguardian=PID1-root' 'dynamic_user=true' \
  'lane_cell_canary_attached=true' 'lane_cell_pidfd=bound' \
  'lane_cell_start_tick=bound' 'lane_cell_executable=bound' \
  'lane_cell_cgroup=bound' 'inherited_descriptor=true' \
  'descriptor_open=systemd-OpenFile' 'descriptor_fd=3' \
  'descriptor_peer_pid=1' 'descriptor_peer_uid=0' \
  'descriptor_is_bearer=false' 'event_hash=bound' 'command_hash=bound' \
  'treatment=Sounio-DENY+hook-continues' \
  'sounio_allow_sabotage=hook-refused' 'binding_sabotage=refused' \
  'guardian_death=refused' 'same_uid=refused' \
  'missing_descriptor=refused' \
  'python_oracle=refused-before-execution' \
  'rust_oracle=refused-before-execution' 'causal_sabotage=PASS' \
  'command_executed=false' 'distinct_uid_product_broker_canary=true' \
  'fleet_lane_cell_attached=false' 'exec_cell_attached=false' \
  'material_execution=false' 'production_activation=false' \
  'launch_open=false' 'recycle_open=false' 'exec_attached=false' \
  'commit_attached=false' 'ci_attached=false' 'parity_open=false' \
  'claim_ready=false'; do
  [[ " $product_output " == *" $expectation "* ]] ||
    fail "product DynamicUser receipt omitted $expectation"
done

set +e
process_witness_output="$(timeout --signal=TERM --kill-after=5s 180s \
  "$BROKER" --selftest-host-process-witness \
  --controller-manifest "$CONTROLLER_MANIFEST" \
  --controller-runtime "$CONTROLLER_RUNTIME" \
  --controller-root "$AUTHORITY_ROOT" \
  --fixture-manifest "$FIXTURE_MANIFEST" \
  --fixture-bundle "$FIXTURE_BUNDLE" \
  --resident-runtime "$RESIDENT_RUNTIME" \
  --barrier-runtime "$LOCAL_BARRIER" \
  --process-witness-runtime "$PROCESS_WITNESS_CELL" \
  --process-witness-payload "$PROCESS_WITNESS_PAYLOAD" \
  --process-witness-manifest "$PROCESS_WITNESS_MANIFEST" \
  --systemd-run "$SYSTEMD_RUN" --systemctl "$SYSTEMCTL" 2>&1)"
process_witness_status=$?
set -e
[[ $process_witness_status -eq 0 ]] ||
  fail "host ProcessWitness matrix failed or timed out status=$process_witness_status output=$process_witness_output"
[[ "$process_witness_output" == 'LOOM_HOST_PROCESS_WITNESS_GATE PASS '* ]] ||
  fail 'host ProcessWitness receipt prefix diverged'
for expectation in \
  'semantic_authority=Sounio' 'controller=OCaml' \
  'dynamic_user=true' 'principal_distinct_uid=true' \
  'treatment=closed' 'positive=done' 'causal_bypass=done' \
  'causal_sabotage=PASS' 'wrong_generation=closed' \
  'payload_substitution=closed' 'forged_ready=closed' \
  'wrong_close=Sounio_refusal' 'controller_death=closed' \
  'broker_death_after_ready=closed' 'replay=closed' \
  'same_pid=true' 'start_tick=true' 'pidfd_at_ready=live' \
  'executable_transition=cell-to-Sounio' \
  'credential_unchanged=true' 'cgroup_unchanged=true' \
  'namespace_unchanged=true' 'environment_empty=true' \
  'descendants_empty=true' 'state_extinct=true' \
  'generation_extinct=true' 'authority_extinct=true' \
  'pidfd_extinct=true' 'cgroup_unpopulated=true' \
  'unit_inactive=true' 'extinction_omission=closed' \
  'complete_effects=false' 'process_witness_core=true' \
  'material_grant=true' 'material_execution=false' \
  'launch_open=false' 'recycle_open=false' 'exec_attached=false' \
  'commit_attached=false' 'ci_attached=false' 'parity_open=false' \
  'claim_ready=false'; do
  [[ " $process_witness_output " == *" $expectation "* ]] ||
    fail "host ProcessWitness receipt omitted $expectation"
done

printf 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS semantic_authority=Sounio controller_language=OCaml controller_role=EFFECT_PARITY material_language=C++20+Linux+systemd material_role=MATERIAL_PARITY transitory=true host=%s kernel=%s architecture=%s systemd_version=%s systemd_run_sha256=%s systemctl_sha256=%s release_manifest_sha256=%s broker_output_sha256=%s process_witness_output_sha256=%s product_exec_ingress_output_sha256=%s dynamic_user=true principal_distinct_uid=true non_bearer_exec_quorum=true descriptor_barrier_causal=true linear_grant_consumption=true positive_open_sentinels=1 sabotage_open_sentinels=1 total_open_sentinels=2 process_witness_core=true same_pid_execveat=true affirmative_extinction=true complete_effects=false product_lane_cell_canary=true distinct_uid_product_broker_canary=true fleet_lane_cell_attached=false exec_cell_attached=false material_grant=true material_execution=false production_activation=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n%s\n%s\n%s\n' \
  "$(hostname)" "$(uname -r)" "$(uname -m)" "$SYSTEMD_VERSION" \
  "$(sha256_file "$SYSTEMD_RUN")" "$(sha256_file "$SYSTEMCTL")" \
  "$EXPECTED_MANIFEST_SHA256" "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)" \
  "$(printf '%s\n' "$process_witness_output" | sha256sum | cut -d ' ' -f 1)" \
  "$(printf '%s\n' "$product_output" | sha256sum | cut -d ' ' -f 1)" \
  "$host_output" "$process_witness_output" "$product_output"
