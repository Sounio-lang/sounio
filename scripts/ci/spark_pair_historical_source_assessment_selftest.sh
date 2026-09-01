#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASSESSMENT="$ROOT_DIR/tools/cluster/spark_pair_historical_provenance.source-assessment.v1"
MATERIAL_OBSERVER="$ROOT_DIR/tools/cluster/spark_pair_historical_source_observation.material-parity.v1"
SEMANTIC_GATE="$ROOT_DIR/scripts/ci/spark_pair_historical_provenance_selftest.sh"

fail() {
  printf 'SPARK_PAIR_HISTORICAL_SOURCE_ASSESSMENT_SELFTEST_FAIL %s\n' "$*" >&2
  exit 1
}

assessment_value() {
  local key="$1" count value
  count="$(sed -n "s/^${key}=//p" "$ASSESSMENT" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "assessment key missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$ASSESSMENT")"
  [[ -n "$value" ]] || fail "assessment key empty: $key"
  printf '%s\n' "$value"
}

observer_value() {
  local key="$1" count value
  count="$(sed -n "s/^${key}=//p" "$MATERIAL_OBSERVER" | wc -l | tr -d ' ')"
  [[ "$count" == 1 ]] || fail "material observer key missing or duplicated: $key"
  value="$(sed -n "s/^${key}=//p" "$MATERIAL_OBSERVER")"
  [[ -n "$value" ]] || fail "material observer key empty: $key"
  printf '%s\n' "$value"
}

require_sha256() {
  local key="$1" value
  value="$(assessment_value "$key")"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "$key is not lowercase SHA-256"
  [[ "$value" != 0000000000000000000000000000000000000000000000000000000000000000 ]] || \
    fail "$key is zero"
}

require_observer_sha256() {
  local key="$1" value
  value="$(observer_value "$key")"
  [[ "$value" =~ ^[0-9a-f]{64}$ ]] || fail "$key is not lowercase SHA-256"
  [[ "$value" != 0000000000000000000000000000000000000000000000000000000000000000 ]] || \
    fail "$key is zero"
}

[[ -r "$ASSESSMENT" ]] || fail 'source assessment is missing'
[[ -r "$MATERIAL_OBSERVER" ]] || fail 'material observer receipt is missing'
semantic_result="$(bash "$SEMANTIC_GATE" --semantics-only)"
[[ "$semantic_result" == *'SPARK_PAIR_HISTORICAL_PROVENANCE_SELFTEST_PASS'* ]] || \
  fail 'frozen Sounio authority gate did not pass'

[[ "$(assessment_value status)" == NO_ADMISSIBLE_HISTORICAL_SOURCE ]] || \
  fail 'assessment claims an admissible historical source'
[[ "$(assessment_value semantic_authority)" == Sounio ]] || \
  fail 'assessment displaced Sounio semantic authority'
[[ "$(assessment_value semantic_frame)" == 9028 ]] || fail 'wrong semantic frame'
[[ "$(assessment_value source_classes_1_to_4_admitted)" == 0 ]] || \
  fail 'a class 1..4 source was promoted without evidence'
[[ "$(assessment_value frame_9028_state)" == HISTORICAL_SOURCE_EMPTY ]] || \
  fail 'frame 9028 left the empty state'
[[ "$(assessment_value offline_replay)" == CLOSED ]] || \
  fail 'offline replay opened without historical admission'
[[ "$(assessment_value source_digest_anchored_before_install)" == false ]] || \
  fail 'assessment promoted a complete source digest from current-layout evidence'
[[ "$(assessment_value partial_payload_current_layout_matches_capture_protocol_format)" == true ]] || \
  fail 'assessment lost the current etcd payload/trailer layout evidence'
[[ "$(assessment_value complete_source_bundle_digest_anchored_before_install)" == false ]] || \
  fail 'assessment promoted a complete bundle digest without evidence'
[[ "$(assessment_value no_retrospective_reconstruction)" == false ]] || \
  fail 'assessment overclaims non-retrospective construction'
[[ "$(assessment_value material_dispatch)" == false ]] || \
  fail 'assessment exposes a material dispatcher'
[[ "$(assessment_value effect)" == READ_ONLY_EVIDENCE_SYNTHESIS ]] || \
  fail 'assessment conflates synthesis with material observation'
[[ "$(assessment_value cluster_object_mutation)" == NONE ]] || \
  fail 'assessment mutated a Kubernetes object'
[[ "$(assessment_value assessment_host_configuration_mutation)" == NONE ]] || \
  fail 'assessment changed host configuration'

observer_path="$(assessment_value material_observer_receipt)"
[[ "$observer_path" == tools/cluster/spark_pair_historical_source_observation.material-parity.v1 ]] || \
  fail 'assessment points to the wrong material observer receipt'
read -r observer_sha _ < <(sha256sum "$MATERIAL_OBSERVER")
[[ "$observer_sha" == "$(assessment_value material_observer_receipt_sha256)" ]] || \
  fail 'material observer receipt digest changed'
[[ "$(observer_value status)" == PREINSTALL_FRAGMENTS_OBSERVED_NON_AUTHORITY ]] || \
  fail 'material observer receipt overclaims authority'
[[ "$(observer_value observer_role)" == READ_ONLY_NON_AUTHORITY ]] || \
  fail 'material observer displaced Sounio authority'
[[ "$(observer_value semantic_promotion)" == false ]] || \
  fail 'material observer promoted semantics'
[[ "$(observer_value rbd_map_mode)" == READ_ONLY ]] || \
  fail 'RBD map was not explicitly read-only'
[[ "$(observer_value rbd_map_flags)" == --read-only,--options,noudev ]] || \
  fail 'RBD read-only map flags changed'
[[ "$(observer_value ext4_mount_flags)" == ro,noload ]] || \
  fail 'ext4 observation could replay the journal'
[[ "$(observer_value rbd_device_before)" == UNMAPPED ]] || \
  fail 'RBD device was already mapped before observation'
[[ "$(observer_value rbd_device_after)" == UNMAPPED ]] || \
  fail 'RBD device remains mapped after observation'
[[ "$(observer_value kubernetes_volume_attachment_before)" == ABSENT ]] || \
  fail 'Kubernetes VolumeAttachment existed before observation'
[[ "$(observer_value kubernetes_volume_attachment_after)" == ABSENT ]] || \
  fail 'Kubernetes VolumeAttachment remains after observation'
[[ "$(observer_value rbd_source_bytes_mutated)" == false ]] || \
  fail 'material observer reports source-byte mutation'
[[ "$(observer_value target_host_configuration_mutation)" == NONE ]] || \
  fail 'material observer reports a target-host configuration mutation'
[[ "$(observer_value observer_local_known_hosts_mutation)" == \
  ONE_HOST_KEY_ADDED_DURING_FAILED_SSH_ATTEMPT ]] || \
  fail 'observer-local known_hosts side effect was hidden or changed'
[[ "$(observer_value etcd_checksum_observation_mount)" == NONE ]] || \
  fail 'embedded-checksum observation unexpectedly mounted a filesystem'
[[ "$(observer_value etcd_checksum_observation_rbd_map_mode)" == READ_ONLY ]] || \
  fail 'embedded-checksum observation did not use a read-only RBD map'
[[ "$(observer_value etcd_checksum_observation_rbd_device_before)" == UNMAPPED ]] || \
  fail 'checksum target was mapped before observation'
[[ "$(observer_value etcd_checksum_observation_rbd_device_after)" == UNMAPPED ]] || \
  fail 'checksum target remains mapped after observation'
[[ "$(observer_value etcd_checksum_observation_kubernetes_volume_attachment_before)" == ABSENT ]] || \
  fail 'checksum observation started with a Kubernetes VolumeAttachment'
[[ "$(observer_value etcd_checksum_observation_kubernetes_volume_attachment_after)" == ABSENT ]] || \
  fail 'checksum observation left a Kubernetes VolumeAttachment'
[[ "$(observer_value cleanup)" == COMPLETE ]] || \
  fail 'material observation cleanup is incomplete'
[[ "$(observer_value effect)" == "$(assessment_value material_observation_effect)" ]] || \
  fail 'assessment and material observer disagree on effect'
[[ "$(observer_value offline_replay)" == CLOSED ]] || \
  fail 'material observer opened offline replay'

etcd_sha_before="$(observer_value etcd_sha256_before)"
etcd_sha_after="$(observer_value etcd_sha256_after)"
etcd_checksum_observation_full_sha="$(observer_value etcd_checksum_observation_full_file_sha256)"
[[ "$etcd_sha_before" == "$etcd_sha_after" ]] || \
  fail 'etcd bytes changed across observation'
[[ "$etcd_sha_after" == "$(assessment_value candidate_2_source_sha256)" ]] || \
  fail 'assessment and material observer disagree on etcd digest'
[[ "$etcd_checksum_observation_full_sha" == "$etcd_sha_before" ]] || \
  fail 'checksum observation full-file digest is not the frozen etcd source bytes'
[[ "$etcd_checksum_observation_full_sha" == \
  "$(assessment_value candidate_2_checksum_observation_full_file_sha256)" ]] || \
  fail 'assessment and material receipt disagree on the checksum-observation full-file digest'
[[ "$(observer_value etcd_checksum_observation_full_file_matches_initial_before_after)" == true ]] || \
  fail 'material receipt lost the full-file digest glue across observations'
[[ "$(assessment_value candidate_2_checksum_observation_full_file_matches_source_sha256)" == true ]] || \
  fail 'assessment lost the checksum-observation/source digest glue'
require_observer_sha256 etcd_checksum_observation_full_file_sha256
etcd_payload_size="$(observer_value etcd_database_payload_size_bytes)"
etcd_checksum_size="$(observer_value etcd_embedded_checksum_size_bytes)"
etcd_file_size="$(observer_value etcd_file_size_bytes)"
[[ "$((etcd_payload_size + etcd_checksum_size))" == "$etcd_file_size" ]] || \
  fail 'etcd payload plus embedded checksum does not equal full file size'
etcd_embedded_checksum="$(observer_value etcd_embedded_checksum_hex)"
etcd_payload_sha256="$(observer_value etcd_database_payload_sha256)"
[[ "$etcd_embedded_checksum" =~ ^[0-9a-f]{64}$ ]] || \
  fail 'embedded etcd checksum is not lowercase SHA-256'
[[ "$etcd_embedded_checksum" == "$etcd_payload_sha256" ]] || \
  fail 'embedded etcd checksum does not match the measured database payload SHA-256'
[[ "$(observer_value etcd_embedded_checksum_matches_payload_sha256)" == true ]] || \
  fail 'material receipt does not attest checksum/payload equality'
[[ "$etcd_embedded_checksum" == "$(assessment_value candidate_2_embedded_checksum_sha256)" ]] || \
  fail 'assessment and material receipt disagree on the embedded checksum'
[[ "$etcd_payload_sha256" == "$(assessment_value candidate_2_database_payload_sha256)" ]] || \
  fail 'assessment and material receipt disagree on the database payload SHA-256'
[[ "$(assessment_value candidate_2_embedded_checksum_matches_payload_sha256)" == true ]] || \
  fail 'assessment lost checksum/payload equality'
require_observer_sha256 etcd_embedded_checksum_hex
require_observer_sha256 etcd_database_payload_sha256
[[ "$(observer_value velero_archive_sha256)" == \
  "$(assessment_value candidate_8_source_sha256)" ]] || \
  fail 'assessment and material observer disagree on Velero digest'

require_sha256 candidate_2_source_sha256
require_sha256 candidate_2_toolchain_sha256
require_sha256 candidate_2_embedded_checksum_sha256
require_sha256 candidate_2_database_payload_sha256
require_sha256 candidate_8_source_sha256
require_sha256 candidate_8_velero_backup_json_sha256
require_sha256 candidate_8_node0_sha256
require_sha256 candidate_8_node1_sha256
require_sha256 candidate_10_source_sha256
require_sha256 candidate_11_call_line_sha256
require_sha256 candidate_11_output_line_sha256

[[ "$(assessment_value candidate_2_class)" == PARTIAL_PREINSTALL_BACKUP:5 ]] || \
  fail 'etcd fragment was promoted beyond class 5'
[[ "$(assessment_value candidate_8_class)" == PARTIAL_PREINSTALL_BACKUP:5 ]] || \
  fail 'Velero fragment was promoted beyond class 5'

[[ "$(assessment_value candidate_2_source_sha256)" == \
  f7835e3ddb9e9b757405e6f78e6edf294ed1310d3412296c4d1c7f882583a7aa ]] || \
  fail 'etcd source digest changed'
[[ "$(assessment_value candidate_2_revision)" == 113864385 ]] || \
  fail 'etcd revision changed'
[[ "$(assessment_value candidate_2_key_count)" == 5739 ]] || \
  fail 'etcd key count changed'
[[ "$(assessment_value candidate_2_nodeset_name_and_uid)" == ABSENT ]] || \
  fail 'pre-NodeSet etcd snapshot unexpectedly contains the NodeSet'
[[ "$(assessment_value candidate_2_pireus_surface)" == ABSENT ]] || \
  fail 'pre-NodeSet etcd snapshot unexpectedly contains Pireus'
[[ "$(assessment_value candidate_2_node0_name_and_uid)" == PRESENT ]] || \
  fail 'etcd snapshot lost spark-3c59 identity'
[[ "$(assessment_value candidate_2_node1_name_and_uid)" == PRESENT ]] || \
  fail 'etcd snapshot lost spark-8e54 identity'
[[ "$(assessment_value candidate_2_device_plugin_0_name_and_uid)" == PRESENT ]] || \
  fail 'etcd snapshot lost device plugin 0 identity'
[[ "$(assessment_value candidate_2_device_plugin_1_name_and_uid)" == PRESENT ]] || \
  fail 'etcd snapshot lost device plugin 1 identity'

etcd_end="$(assessment_value candidate_2_capture_completed_utc)"
velero_end="$(assessment_value candidate_8_backup_completed_utc)"
nodeset_start="$(assessment_value nodeset_install_anchor_created_utc)"
[[ "$etcd_end" < "$nodeset_start" ]] || fail 'etcd snapshot is not before NodeSet creation'
[[ "$velero_end" < "$nodeset_start" ]] || fail 'Velero backup is not before NodeSet creation'
[[ "$(assessment_value candidate_2_payload_digest_evidence)" == \
  CURRENT_LAYOUT_TRAILING_SHA256_MATCHES_CURRENT_DATABASE_PAYLOAD ]] || \
  fail 'assessment lost the current etcd payload/trailer layout evidence'
[[ "$(assessment_value candidate_2_embedded_checksum_exact_value_externally_anchored_before_install)" == false ]] || \
  fail 'assessment overclaims an external temporal anchor for the embedded checksum value'
[[ "$(assessment_value candidate_2_source_file_digest_anchor)" == \
  FULL_FILE_SHA256_NOT_RECORDED_BEFORE_NODESET_CREATION ]] || \
  fail 'assessment overclaims the full etcd source-file digest anchor'
[[ "$(assessment_value candidate_2_complete_source_bundle_digest_anchor)" == NOT_PRESENT ]] || \
  fail 'assessment overclaims a complete source-bundle digest anchor'
[[ "$(assessment_value candidate_8_digest_anchor)" == \
  SHA256_NOT_RECORDED_BEFORE_NODESET_CREATION ]] || \
  fail 'assessment overclaims the Velero SHA-256 anchor'
[[ "$(assessment_value candidate_2_source_immutable_since_anchor)" == false ]] || \
  fail 'assessment overclaims etcd storage immutability'
[[ "$(assessment_value candidate_8_source_immutable_since_anchor)" == false ]] || \
  fail 'assessment overclaims Velero storage immutability'
[[ "$(observer_value velero_s3_versioning)" == DISABLED ]] || \
  fail 'Velero backing store unexpectedly claims versioning'
[[ "$(observer_value velero_s3_object_lock)" == NOT_PRESENT ]] || \
  fail 'Velero backing store unexpectedly claims object lock'
[[ "$(observer_value velero_backup_phase)" == \
  PARTIALLY_FAILED_ONLY_MISSING_INCLUDED_NAMESPACE_DARWIN_RESEARCH ]] || \
  fail 'Velero partial-failure classification changed'

[[ "$(observer_value time_machine_pre_nodeset_source)" == ABSENT ]] || \
  fail 'Time Machine unexpectedly claims a pre-NodeSet source'
[[ "$(observer_value time_machine_backupd_events_2026_08_29_through_nodeset_anchor)" == 0 ]] || \
  fail 'Time Machine backup event count changed for the cutoff window'
[[ "$(assessment_value candidate_9_pre_nodeset_source)" == ABSENT ]] || \
  fail 'assessment promoted a nonexistent Time Machine source'
[[ "$(assessment_value candidate_9_decision)" == NOT_INVOKED_NO_PRE_NODESET_SNAPSHOT_OR_CATALOG ]] || \
  fail 'assessment misclassified the Time Machine negative result'

[[ "$(observer_value beagle_git_blob_sha256)" == \
  "$(assessment_value candidate_10_source_sha256)" ]] || \
  fail 'assessment and material receipt disagree on the Beagle Git blob digest'
[[ "$(assessment_value candidate_10_class)" == MUTABLE_OR_CLOCK_ONLY_EXPORT:7 ]] || \
  fail 'Beagle planning chronology was promoted beyond class 7'
[[ "$(assessment_value candidate_10_custody)" == \
  UNSIGNED_SHALLOW_COMMIT_PARENT_ABSENT_NO_EXTERNAL_ANCHOR ]] || \
  fail 'assessment overclaims Beagle Git custody'
require_observer_sha256 beagle_git_blob_sha256
require_observer_sha256 rollout_kubectl_call_line_sha256
require_observer_sha256 rollout_kubectl_output_line_sha256
[[ "$(observer_value rollout_kubectl_call_line_sha256)" == \
  "$(assessment_value candidate_11_call_line_sha256)" ]] || \
  fail 'assessment and material receipt disagree on the rollout call-line digest'
[[ "$(observer_value rollout_kubectl_output_line_sha256)" == \
  "$(assessment_value candidate_11_output_line_sha256)" ]] || \
  fail 'assessment and material receipt disagree on the rollout output-line digest'
[[ "$(assessment_value candidate_11_class)" == MUTABLE_OR_CLOCK_ONLY_EXPORT:7 ]] || \
  fail 'mutable rollout chronology was promoted beyond class 7'

[[ "$(observer_value full_file_reverification_attempt_1_map)" == NOT_CREATED ]] || \
  fail 'failed full-file reverification attempt 1 created an RBD map'
[[ "$(observer_value full_file_reverification_attempt_2_map)" == NOT_CREATED ]] || \
  fail 'failed full-file reverification attempt 2 created an RBD map'
[[ "$(observer_value full_file_reverification_new_hashes)" == NONE ]] || \
  fail 'receipt claims an unobserved reverification digest'
[[ "$(observer_value full_file_reverification_secret_exposed_in_argv)" == false ]] || \
  fail 'reverification exposed a secret in argv'
[[ "$(observer_value final_clean_check_volumeattachments)" == ABSENT ]] || \
  fail 'final clean check found a VolumeAttachment'
[[ "$(observer_value final_clean_check_ephemeral_rbd_map)" == ABSENT ]] || \
  fail 'final clean check found an ephemeral RBD map'
[[ "$(assessment_value candidate_2_full_file_reverification)" == \
  NO_NEW_HASH_TWO_PREFLIGHT_REFUSALS_BEFORE_MAP_FINAL_STATE_CLEAN ]] || \
  fail 'assessment hides or overclaims the failed full-file reverification'

[[ "$(assessment_value content_addressed_preinstall_fragments)" == 2 ]] || \
  fail 'unexpected preinstall fragment count'
[[ "$(assessment_value current_layout_payload_checksum_matches)" == 1 ]] || \
  fail 'unexpected current-layout payload/checksum match count'
[[ "$(assessment_value complete_source_bundle_digest_anchors)" == 0 ]] || \
  fail 'a complete source-bundle digest anchor was promoted'
[[ "$(assessment_value negative_chronology_corroborations)" == 2 ]] || \
  fail 'unexpected negative chronology corroboration count'
[[ "$(assessment_value time_machine_pre_nodeset_sources)" == 0 ]] || \
  fail 'a nonexistent Time Machine source was counted'
[[ "$(assessment_value composite_closure)" == NOT_AVAILABLE ]] || \
  fail 'assessment claims composite closure'
[[ "$(assessment_value composite_leaf_profile)" == NOT_PRESENT ]] || \
  fail 'assessment claims a nonexistent class-4 leaf profile'
[[ "$(assessment_value class4_submission)" == FORBIDDEN_UNTIL_SOUNIO_LEAF_PROFILE_* ]] || \
  fail 'class-4 aggregate-mask submission is not forbidden'
[[ "$(assessment_value waiver)" == NOT_PRESENT ]] || fail 'unexpected founder waiver'
[[ "$(assessment_value external_llm)" == REVIEW_ONLY_* ]] || \
  fail 'external LLM is not explicitly review-only'

printf 'SPARK_PAIR_HISTORICAL_SOURCE_ASSESSMENT_SELFTEST_PASS fragments=2 etcd_file_sha256=f7835e3d etcd_payload_sha256=fe297ecf current_layout_payload_checksum=MATCH velero_sha256=8c406131 nodeset_anchor=2026-08-30T09:54:03Z class_1_to_4=0 complete_bundle_anchors=0 negative_chronology=2 time_machine_sources=0 composite_leaf_profile=NOT_PRESENT offline_replay=CLOSED rbd_cleanup=COMPLETE\n'
