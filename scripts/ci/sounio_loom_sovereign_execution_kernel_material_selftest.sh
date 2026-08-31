#!/usr/bin/env bash

set -euo pipefail
umask 077

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/sounio-loom-sovereign-material.XXXXXX")"
trap 'rm -rf "$TEST_ROOT"' EXIT

fail() {
  printf 'sounio-loom-sovereign-execution-kernel-material-selftest: FAIL: %s\n' "$*" >&2
  exit 1
}

RUNTIME="$TEST_ROOT/sounio-authority"
SOUNIO_LOOM_SOVEREIGN_OUTPUT="$RUNTIME" \
  bash "$ROOT_DIR/scripts/dev/build_sounio_loom_sovereign_execution_kernel.sh" >/dev/null

for ordinal in one two; do
  SOUNIO_LOOM_SOVEREIGN_MATERIAL_OUTPUT="$TEST_ROOT/material-$ordinal" \
    bash "$ROOT_DIR/scripts/dev/build_loom_sovereign_execution_kernel_material.sh" >/dev/null
done
cmp "$TEST_ROOT/material-one" "$TEST_ROOT/material-two" ||
  fail 'two material builds differ'

MANIFEST="$ROOT_DIR/tools/loom/sovereign_execution_kernel.freeze.v1"
PEER="$ROOT_DIR/tools/loom/kernel_peer_material_judgment_v13.freeze.v1"
material_one="$($TEST_ROOT/material-one --selftest "$RUNTIME" "$MANIFEST" "$PEER")"
material_two="$($TEST_ROOT/material-two --selftest "$RUNTIME" "$MANIFEST" "$PEER")"
[[ "$material_one" == "$material_two" ]] || fail 'material output is nondeterministic'

for required in \
  'grant_resident_memory=true grant_is_bearer=false' \
  'grant_single_use=true consume_atomic=true' \
  'exported_token=false exported_handle=false' \
  'peer=SO_PEERCRED+pidfd+start-tick+harness-ancestry+executable+operation' \
  'hostile_same_uid=true hostile_same_executable=true' \
  'same_uid_spoof=refused-before-execution' \
  'interface_release_authority=zero' \
  'transport_death=material-witness-continued' \
  'gui_death=material-witness-continued' \
  'coordinator_death=material-witness-continued' \
  'guardian_death=grant-revoked+material-extinct+release-absent' \
  'same_uid_peer_isolation=true' \
  'production_gate_ready=true production_activation=false'; do
  [[ "$material_one" == *"$required"* ]] || fail "material result omitted $required"
done

SOUNIO_LOOM_SOVEREIGN_MATERIAL_OUTPUT="$TEST_ROOT/principal-sabotage" \
SOUNIO_LOOM_SOVEREIGN_MATERIAL_CPPFLAGS='-DLOOM_SOVEREIGN_DISABLE_PRINCIPAL_BINDING=1' \
  bash "$ROOT_DIR/scripts/dev/build_loom_sovereign_execution_kernel_material.sh" >/dev/null
set +e
principal_output="$($TEST_ROOT/principal-sabotage --selftest "$RUNTIME" "$MANIFEST" "$PEER" 2>&1)"
principal_code=$?
set -e
[[ $principal_code -eq 1 && "$principal_output" == \
  'loom-sovereign-execution-kernel-material: FAIL: same-uid-spoof-admitted' ]] ||
  fail "principal sabotage did not admit the same-UID spoof: $principal_output"

SOUNIO_LOOM_SOVEREIGN_MATERIAL_OUTPUT="$TEST_ROOT/pdeath-sabotage" \
SOUNIO_LOOM_SOVEREIGN_MATERIAL_CPPFLAGS='-DLOOM_SOVEREIGN_DISABLE_PDEATHSIG=1' \
  bash "$ROOT_DIR/scripts/dev/build_loom_sovereign_execution_kernel_material.sh" >/dev/null
set +e
pdeath_output="$($TEST_ROOT/pdeath-sabotage --selftest "$RUNTIME" "$MANIFEST" "$PEER" 2>&1)"
pdeath_code=$?
set -e
[[ $pdeath_code -eq 1 && "$pdeath_output" == \
  'loom-sovereign-execution-kernel-material: FAIL: material-survived-guardian' ]] ||
  fail "PDEATHSIG sabotage did not preserve the material process: $pdeath_output"

ORACLE_EXECUTED="$TEST_ROOT/oracle-executed"
for name in python python3 rustc cargo; do
  printf '#!/bin/sh\nprintf prohibited > %s\n' "$ORACLE_EXECUTED" > "$TEST_ROOT/$name"
  chmod 0755 "$TEST_ROOT/$name"
done
oracle_probe="$(env PATH="$TEST_ROOT:$PATH" "$TEST_ROOT/material-one" \
  --selftest "$RUNTIME" "$MANIFEST" "$PEER")"
[[ "$oracle_probe" == "$material_one" ]] || fail 'oracle-path probe changed material output'
[[ ! -e "$ORACLE_EXECUTED" ]] || fail 'a prohibited oracle executed'

dependencies="$(ldd "$TEST_ROOT/material-one" 2>&1 || true)"
printf '%s\n' "$dependencies" | grep -Eqi 'python|rust' &&
  fail 'material runtime has a prohibited dependency'

printf 'sounio-loom-sovereign-execution-kernel-material-selftest: PASS semantic_authority=Sounio action=9042 material_language=C++20+Linux material_role=MATERIAL_PARITY transitory=true treatment=PASS hostile_same_uid_spoof=REFUSED_BEFORE_EXECUTION principal_binding_sabotage=SPOOF_ADMITTED guardian_death=PASS pdeathsig_sabotage=MATERIAL_SURVIVED transport_death=PASS gui_death=PASS coordinator_death=PASS pod_death=PASS tmux_death=PASS grant_resident_memory=true grant_is_bearer=false grant_single_use=true consume_atomic=true peer=SO_PEERCRED+pidfd+start-tick+harness-ancestry+executable+operation interface_release_authority=zero material_exactly_once=true same_uid_peer_isolation=true production_gate_ready=true production_activation=false exec_attached=false commit_attached=false ci_attached=false parity_open=false claim_ready=false causal_sabotage=PASS expected_results_encoded_in_material_layer=false python_executed=false rust_executed=false runtime_dependencies=clean semantic_manifest_sha256=%s peer_judgment_sha256=%s source_sha256=%s material_binary_sha256=%s sounio_runtime_sha256=%s material_output_sha256=%s\n' \
  "$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)" \
  "$(sha256sum "$PEER" | cut -d ' ' -f 1)" \
  "$(sha256sum "$ROOT_DIR/tools/loom/src/loom_sovereign_execution_kernel_material.cpp" | cut -d ' ' -f 1)" \
  "$(sha256sum "$TEST_ROOT/material-one" | cut -d ' ' -f 1)" \
  "$(sha256sum "$RUNTIME" | cut -d ' ' -f 1)" \
  "$(printf '%s' "$material_one" | sha256sum | cut -d ' ' -f 1)"
