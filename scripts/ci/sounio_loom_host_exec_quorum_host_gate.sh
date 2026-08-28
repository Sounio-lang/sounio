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
FIXTURE_MANIFEST="$(verify_binding fixture_manifest_path fixture_manifest_sha256 444)"
FIXTURE_BUNDLE="$(verify_binding fixture_bundle_path fixture_bundle_sha256 444)"
RESIDENT_RUNTIME="$(verify_binding resident_runtime_path resident_runtime_sha256 555)"
LOCAL_BARRIER="$(verify_binding local_barrier_path local_barrier_sha256 555)"
HOST_BARRIER="$(verify_binding host_barrier_path host_barrier_sha256 555)"
AUTHORITY_ROOT="$RELEASE/$(record_value "$MANIFEST" authority_root_path)"
[[ -d "$AUTHORITY_ROOT" && ! -L "$AUTHORITY_ROOT" && -d "$AUTHORITY_ROOT/.git" ]] ||
  fail 'authority root topology is incomplete'

SYSTEMD_RUN="$(readlink -f "$(command -v systemd-run)")"
SYSTEMCTL="$(readlink -f "$(command -v systemctl)")"
[[ "$SYSTEMD_RUN" == /* && -x "$SYSTEMD_RUN" && "$SYSTEMCTL" == /* && -x "$SYSTEMCTL" ]] ||
  unavailable 'canonical systemd tools are unavailable'

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

read -r _ SYSTEMD_VERSION _ < <(systemctl --version | sed -n '1p')
[[ "$SYSTEMD_VERSION" =~ ^[0-9]+$ ]] || fail 'systemd version is not canonical'
printf 'sounio-loom-host-exec-quorum-host-gate: HOST_MEASUREMENT_PASS semantic_authority=Sounio controller_language=OCaml controller_role=EFFECT_PARITY material_language=C++20+Linux+systemd material_role=MATERIAL_PARITY transitory=true host=%s kernel=%s architecture=%s systemd_version=%s systemd_run_sha256=%s systemctl_sha256=%s release_manifest_sha256=%s broker_output_sha256=%s dynamic_user=true principal_distinct_uid=true non_bearer_exec_quorum=true descriptor_barrier_causal=true linear_grant_consumption=true positive_open_sentinels=1 sabotage_open_sentinels=1 total_open_sentinels=2 material_grant=true material_execution=false launch_open=false recycle_open=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false\n%s\n' \
  "$(hostname)" "$(uname -r)" "$(uname -m)" "$SYSTEMD_VERSION" \
  "$(sha256_file "$SYSTEMD_RUN")" "$(sha256_file "$SYSTEMCTL")" \
  "$EXPECTED_MANIFEST_SHA256" "$(printf '%s\n' "$host_output" | sha256sum | cut -d ' ' -f 1)" \
  "$host_output"
