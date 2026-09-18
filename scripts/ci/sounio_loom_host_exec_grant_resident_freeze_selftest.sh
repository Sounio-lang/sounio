#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MANIFEST=tools/loom/host_exec_grant_resident_attachment.v1
EVIDENCE=tools/loom/evidence/loom-host-exec-grant-resident-attachment-v1-20260828.txt

fail() {
  printf 'sounio-loom-host-exec-grant-resident-freeze-selftest: FAIL reason=%s\n' "$*" >&2
  exit 1
}

expect_hash() {
  local path="$1" expected="$2"
  [[ -f "$path" ]] || fail "frozen path is absent: $path"
  [[ "$(sha256sum "$path" | cut -d ' ' -f 1)" == "$expected" ]] ||
    fail "frozen path hash drifted: $path"
}

require_line() {
  local path="$1" value="$2"
  grep -Fxq "$value" "$path" || fail "required frozen line is absent: $value"
}

expect_hash "$EVIDENCE" 5a41d4a74526687e4e3eadaf22f4cfef0c911fa4e78d9158bbb2434800752e02
expect_hash tools/loom/kernel_exec_grant_cell_authority.freeze.v1 8687d889e08f69190daaf3cdbee02741cde3ce62f136ba63df1fa9c2ccb0d051
expect_hash tools/loom/resident_membrane.runtime.v4 f61c93a3aefdbab792ed757faddf778017d34e0fa6bed97c565b56fe3147d473
expect_hash tools/loom/GARDEN_HOST_EXEC_GRANT_RESIDENT_ATTACHMENT_V1.md 29be7aa25ebcbede2f153ed5e42d9276e12e76716a735f6b6c81cce8d1b9dce7
expect_hash tools/loom/HOST_EXEC_GRANT_RESIDENT_ATTACHMENT_V1.md 3b32f4bc1f8c900afaa61ea378fdbfcb113863f61d1b2fcbdbc4f9d191da45a5
expect_hash tools/loom/HOST_PRINCIPAL_CELL_BARRIER_V1.md db0dfd903a45b9f3ebba3a95f5d56a2f53ab069c2a78dd0b49bb8e1d4569c608
expect_hash tools/loom/src/loom_kernel_principal_broker.cpp 7c304be6510ebef1b0ecc4c67c4cd57842472d3a7a4dd38359e393f10b33efd7

require_line "$MANIFEST" 'schema=loom-host-exec-grant-resident-attachment-freeze-v1'
require_line "$MANIFEST" 'stage=MATERIAL_MEASURED'
require_line "$MANIFEST" 'producing_language=Sounio'
require_line "$MANIFEST" 'language_role=SEMANTIC_AUTHORITY'
require_line "$MANIFEST" 'semantic_action=9030'
require_line "$MANIFEST" 'host_activation=PASS'
require_line "$MANIFEST" 'resident_action_9030_attached=true'
require_line "$MANIFEST" 'decision_transport_material=true'
require_line "$MANIFEST" 'semantic_decision_observed=DENY_ONLY'
require_line "$MANIFEST" 'material_grant=false'
require_line "$MANIFEST" 'material_execution=false'
require_line "$MANIFEST" 'barrier_release=false'
require_line "$MANIFEST" 'launch_open=false'
require_line "$MANIFEST" 'same_uid_peer_isolation=false'
require_line "$MANIFEST" 'exec_attached=false'
require_line "$MANIFEST" 'parity_open=false'
require_line "$MANIFEST" 'claim_ready=false'
require_line "$MANIFEST" 'evidence_sha256=5a41d4a74526687e4e3eadaf22f4cfef0c911fa4e78d9158bbb2434800752e02'

require_line "$EVIDENCE" 'source_commit=a899b5d6e5d641e357c5555e9b1419195b825e30'
require_line "$EVIDENCE" 'capsule_archive_sha256=00d4aa0580b096edc2cbe3caba58b010c84ebd546bf01a8098e676a9c509080c'
require_line "$EVIDENCE" 'result=HOST_ACTIVATION_PASS'
require_line "$EVIDENCE" 'resident_sequence_at_gate=0,1'
require_line "$EVIDENCE" 'resident_sequence_after_transport=3'
require_line "$EVIDENCE" 'resident_identity_unchanged=true'
require_line "$EVIDENCE" 'resident_sequence_monotonic=true'
require_line "$EVIDENCE" 'resident_poisoned_after_transport=false'
require_line "$EVIDENCE" 'local_current_material=DENY491'
require_line "$EVIDENCE" 'local_python_authority_laundering=DENY499'
require_line "$EVIDENCE" 'local_malformed_frame=DENY424'
require_line "$EVIDENCE" 'python_host_gate_oracle=refused-before-execution'
require_line "$EVIDENCE" 'python_oracle_executed=false'
require_line "$EVIDENCE" 'failed_promotion_rollback=PASS'
require_line "$EVIDENCE" 'host_activation=PASS'
require_line "$EVIDENCE" 'semantic_decision_observed=DENY_ONLY'
require_line "$EVIDENCE" 'material_grant=false'
require_line "$EVIDENCE" 'material_execution=false'
require_line "$EVIDENCE" 'barrier_release=false'
require_line "$EVIDENCE" 'launch_open=false'
require_line "$EVIDENCE" 'same_uid_peer_isolation=false'
require_line "$EVIDENCE" 'exec_attached=false'
require_line "$EVIDENCE" 'parity_open=false'
require_line "$EVIDENCE" 'claim_ready=false'

resident_gate="$(bash scripts/ci/sounio_loom_host_exec_grant_resident_selftest.sh)"
[[ "$resident_gate" == sounio-loom-host-exec-grant-resident-selftest:\ PASS* ]] ||
  fail 'source-fresh resident attachment gate failed'
[[ "$resident_gate" == *'current=DENY491 python=DENY499 malformed=DENY424'* ]] ||
  fail 'resident attachment decision controls drifted'
[[ "$resident_gate" == *'death=poisoned timeout=poisoned malformed_output=poisoned'* ]] ||
  fail 'resident attachment fault controls drifted'
[[ "$resident_gate" == *'launch_open=false material_grant=false material_execution=false exec_attached=false parity_open=false claim_ready=false' ]] ||
  fail 'resident attachment gate promoted beyond evidence'

capsule_gate="$(bash scripts/ci/sounio_loom_host_promotion_capsule_selftest.sh)"
[[ "$capsule_gate" == sounio-loom-host-promotion-capsule-selftest:\ PASS* ]] ||
  fail 'promotion capsule gate failed'
[[ "$capsule_gate" == *'python_oracle=refused_pre_execution python_oracle_executed=false'* ]] ||
  fail 'promotion capsule lost the Python-oracle negative control'
[[ "$capsule_gate" == *'launch=closed'* &&
   "$capsule_gate" == *'material_grant=false material_execution=false barrier_release=false'* ]] ||
  fail 'promotion capsule opened an unproven boundary'

printf 'sounio-loom-host-exec-grant-resident-freeze-selftest: PASS semantic_authority=Sounio action=9030 host=t560-proxmox host_activation=PASS resident_action_9030_attached=true resident_identity=persistent resident_sequence=0,1,3 current=DENY491 python=DENY499 malformed=DENY424 python_oracle=refused-before-execution decision_transport_material=true semantic_decision_observed=DENY_ONLY material_grant=false material_execution=false barrier_release=false launch_open=false same_uid_peer_isolation=false exec_attached=false parity_open=false claim_ready=false evidence_sha256=%s\n' \
  "$(sha256sum "$EVIDENCE" | cut -d ' ' -f 1)"
