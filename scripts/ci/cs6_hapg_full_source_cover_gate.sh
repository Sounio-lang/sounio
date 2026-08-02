#!/usr/bin/env bash
set -euo pipefail

if [[ $(python3 -B -c 'import sys; print(sys.flags.optimize)') != 0 ]]; then
  echo "H-APG cover gate error: Python optimization is forbidden" >&2
  exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

runner=scripts/research/cs6_hapg_full_source_cover_run.py
leaf_verifier=scripts/research/cs6_hapg_full_source_cover_verify.py
aggregator=scripts/research/cs6_hapg_full_source_cover_aggregate.py
kat_anchor=scripts/research/cs6_hapg_full_source_cover_kat_anchor.py
wrapper=scripts/research/cs6_hapg_full_source_cover_worker.cpp
slurm_job=scripts/research/cs6_hapg_full_source_cover_slurm_job.sh
contract=scripts/research/cs6_hapg_full_source_cover_contract_v6.txt
v5_contract=scripts/research/cs6_hapg_full_source_cover_contract_v5.txt
v4_contract=scripts/research/cs6_hapg_full_source_cover_contract_v4.txt
v3_contract=scripts/research/cs6_hapg_full_source_cover_contract_v3.txt
full53=scripts/research/receipts/cs6_affine_projective_cocycle_full53_retained_53_v1
v2_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v2_abort_8451_v1
v3_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v3_abort_8453_v1
v4_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1
v5_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v5_abort_8463_v1

for required in \
  "$runner" "$leaf_verifier" "$aggregator" "$kat_anchor" "$wrapper" "$slurm_job" "$contract" "$v5_contract" "$v4_contract" "$v3_contract" \
  "$v2_abort/manifest.txt" "$v2_abort/sacct.txt" "$v2_abort/config.txt" "$v2_abort/stderr.txt" \
  "$v3_abort/manifest.txt" "$v3_abort/sacct.txt" "$v3_abort/config.txt" \
  "$v3_abort/slurm-stderr.txt" "$v3_abort/repro-s0-stdout.txt" \
  "$v3_abort/repro-s0-stderr.txt" "$v3_abort/repro-s1-stdout.txt" \
  "$v3_abort/repro-s1-stderr.txt" "$v3_abort/hpg-full255-census.tsv" \
  "$v3_abort/hpg-full255-census-summary.txt" "$v3_abort/hpg-full255-stderr.jsonl" \
  "$v3_abort/challenge-spotcheck.json" \
  "$v4_abort/manifest.txt" "$v4_abort/files.sha256" "$v4_abort/sacct.txt" \
  "$v4_abort/config.txt" "$v4_abort/slurm-stdout.txt" \
  "$v4_abort/hpg-rc0-corpus.tar" "$v4_abort/corpus-files.sha256" \
  "$v4_abort/hpg-rc0-verifier-census.tsv" \
  "$v4_abort/hpg-rc0-verifier-census-summary.txt" \
  "$v4_abort/hpg-v5-kat-compat.tsv" "$v4_abort/hpg-v4-kat-corpus.tar" \
  "$v4_abort/hpg-v4-kat-corpus-files.sha256" \
  "$v4_abort/midpoint-discrete-negative-test.txt" \
  "$v4_abort/local-repro.tar" "$v4_abort/v4-hpg-verifier.py" \
  "$v5_abort/manifest.txt" "$v5_abort/files.sha256" "$v5_abort/jobs-8458-8463.sacct.psv" \
  "$v5_abort/logs/adaptive-8463.out" "$v5_abort/logs/adaptive-8463.err" \
  "$v5_abort/file-index-order-repro.txt"; do
  [[ -f $required ]] || {
    echo "H-APG cover gate error: missing $required" >&2
    exit 1
  }
done
bash -n "$slurm_job"

python3 -B - "$contract" "$v2_abort" "$v3_abort" "$v3_contract" "$v4_abort" "$v4_contract" "$v5_abort" "$v5_contract" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

path = Path(sys.argv[1])
abort_root = Path(sys.argv[2])
v3_abort_root = Path(sys.argv[3])
v3_contract_path = Path(sys.argv[4])
v4_abort_root = Path(sys.argv[5])
v4_contract_path = Path(sys.argv[6])
v5_abort_root = Path(sys.argv[7])
v5_contract_path = Path(sys.argv[8])
raw = path.read_bytes()
if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
    raise SystemExit("H-APG cover gate error: noncanonical frozen contract")
try:
    lines = raw.decode("ascii").splitlines()
except UnicodeError as error:
    raise SystemExit("H-APG cover gate error: frozen contract is not ASCII") from error
fields: dict[str, str] = {}
for line in lines:
    if line.count("=") != 1:
        raise SystemExit("H-APG cover gate error: malformed frozen contract")
    key, value = line.split("=", 1)
    if not key or not value or key in fields:
        raise SystemExit("H-APG cover gate error: duplicate or empty contract field")
    fields[key] = value

exact = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-contract.v6",
    "CONTRACT_STATE": "PRE_RESULT_FROZEN",
    "SUPERSEDES_V4_SHA256": "a308b4f0d32b4179ed17f1ffd7bbd4827fa81d9cb66162318ddecbc926a43293",
    "SUPERSEDES_V3_SHA256": "3e5f1c560356771e9d33582cab31b9776cf6f21d4eabcbc6e292523a2e9010e2",
    "RECOVERY_SCOPE": "V5_FILE_INDEX_ORDER_AND_STRONG_FRESH_KAT_PREREQUISITE",
    "SUPERSEDES_V5_SHA256": "5fe8436fe3384663d9046e754f1a27b4ac537e5a51ccc1d90469bd96f7ba3dab",
    "IMPLEMENTATION_PARENT_COMMIT": "7e09b89b94a773c6f5609dbc4b98f16dc22a9d5f",
    "V5_ABORT_RECEIPT_MANIFEST_SHA256": "17dcf5980db78f727e4428aad8763488809ef8c3bce2e301ad62384de97facd7",
    "V5_ABORT_FILES_INDEX_SHA256": "a6f9dd55cc2aff2d03adf3d73f52b67f2a673ed56e43221ccbd753c2a2aef4ba",
    "V5_ABORT_JOBS_SACCT_SHA256": "fdb6fd9b347ce5fb2c40965ca87391698aab2c940dd64549cd9ec97fbff622de",
    "V5_KAT_SLURM_JOB_ID": "8458",
    "V5_KAT_SLURM_STATE": "COMPLETED",
    "V5_KAT_SLURM_EXIT_CODE": "0:0",
    "V5_KAT_ARCHIVE_SHA256": "f288918070c73636458a61687ae1abf0199038b99801e1392ec4774b26ae7c7f",
    "V5_KAT_CONFIG_SHA256": "0cf799ebf131502cfd30347bc50060bc51a4c5853380ca58baf7dcbcdb1423f0",
    "V5_ADAPTIVE_SLURM_JOB_ID": "8463",
    "V5_ADAPTIVE_SLURM_STATE": "FAILED",
    "V5_ADAPTIVE_SLURM_EXIT_CODE": "1:0",
    "V5_ADAPTIVE_FAILURE_STAGE": "POST_RUN_INDEPENDENT_AGGREGATION_FILE_INDEX_GATE",
    "V5_ADAPTIVE_FAILURE_SIGNATURE": "aggregation_error_files_index_is_not_sorted",
    "V5_ADAPTIVE_RUNNER_REPORTED_EVALUATED_NODE_COUNT": "255",
    "V5_ADAPTIVE_RUNNER_REPORTED_WAVE_COUNT": "8",
    "V5_ADAPTIVE_ARCHIVE_PUBLISHED": "false",
    "V5_ADAPTIVE_RESULT_AVAILABLE": "false",
    "V5_ADAPTIVE_RUNNER_CANDIDATE_FIELD_IS_SCIENTIFIC_RESULT": "false",
    "V6_FILE_INDEX_ORDER": "ASCII_LEXICOGRAPHIC_POSIX_RELATIVE_TOKEN",
    "V6_KAT_PREREQUISITE_REQUIRED_FOR_ADAPTIVE": "true",
    "V6_KAT_ARCHIVE_SAFE_FULL_INDEX_REPLAY_REQUIRED": "true",
    "V6_KAT_LEAF_BY_LEAF_VERIFIER_REPLAY_REQUIRED": "true",
    "V6_KAT_SAME_GIT_HEAD_REQUIRED": "true",
    "V6_KAT_SAME_FROZEN_CONTRACT_REQUIRED": "true",
    "V6_KAT_SAME_BASE_DELTA_PREBUILT_AND_JOB_SCRIPT_REQUIRED": "true",
    "V6_KAT_SLURM_COMPLETED_ZERO_EXIT_REQUIRED": "true",
    "V6_KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT_REQUIRED": "true",
    "V6_KAT_CERTIFICATE_INCLUDED_IN_ADAPTIVE_BUNDLE": "true",
    "V6_KAT_CERTIFICATE_SHA256_BOUND_IN_ADAPTIVE_RUN_CONTRACT": "true",
    "V6_AGGREGATOR_REVALIDATES_KAT_ARCHIVE_AND_CERTIFICATE": "true",
    "V6_POST_RUN_FAILURE_DIAGNOSTIC_ARCHIVE_REQUIRED": "true",
    "V6_FRESH_SLURM_KAT_EXECUTED": "false",
    "V6_AUTHORITATIVE_ADAPTIVE_EXECUTED": "false",
    "V6_AUTHORITATIVE_RESULT_AVAILABLE": "false",
    "SLURM_CONFIG_SCHEMA": "sounio.cs6.hapg-full-source-cover-slurm-config.v3",
    "PREBUILT_MANIFEST_SCHEMA": "sounio.cs6.hapg-full-source-cover-prebuilt.v2",
    "RUN_CONTRACT_SCHEMA": "sounio.cs6.hapg-full-source-cover-run-contract.v2",
    "TRANSPORT_MANIFEST_SCHEMA": "sounio.cs6.hapg-full-source-cover-transport.v3",
    "AGGREGATION_SCHEMA": "sounio.cs6.hapg-full-source-cover-aggregation.v2",
    "KAT_PREREQUISITE_CERTIFICATE_SCHEMA": "sounio.cs6.hapg-full-source-cover-kat-prerequisite.v2",
    "KAT_PREREQUISITE_SACCT_ARTIFACT": "kat-prerequisite-sacct.txt",
    "KAT_PREREQUISITE_CERTIFICATE_ARTIFACT": "kat-prerequisite-certificate.txt",
    "KAT_PREREQUISITE_TRANSPORT_CONFIG_ARTIFACT": "transport-config.txt",
    "ADAPTIVE_RUN_CONTRACT_KAT_FIELDS": "KAT_PREREQUISITE_REQUIRED,KAT_PREREQUISITE_CERTIFICATE_SCHEMA,KAT_PREREQUISITE_CERTIFICATE_SHA256,KAT_PREREQUISITE_SACCT_SHA256,KAT_JOB_ID,KAT_ARCHIVE_SHA256,KAT_GIT_HEAD,KAT_FROZEN_CONTRACT_SHA256,KAT_BASE_REPO_BUNDLE_SHA256,KAT_BASE_GIT_HEAD,KAT_REPO_DELTA_BUNDLE_SHA256,KAT_PREBUILT_ARCHIVE_SHA256,KAT_PREBUILT_RUN_MANIFEST_SHA256,KAT_SLURM_JOB_SCRIPT_SHA256,KAT_END_UTC,KAT_COORDINATE_MANIFEST_SHA256,KAT_EXPECTED_RESULTS_SHA256,KAT_WAVE_CONTRACT_SHA256,KAT_WAVE_RESULT_SHA256,KAT_LEAF_EVIDENCE_VALID,KAT_HPG_VERIFIER_REPLAY_COUNT,KAT_HAPG_VERIFIER_REPLAY_COUNT,KAT_EVALUATED_NODE_COUNT,KAT_HPG_SIGNED_CHART_COUNT,KAT_HAPG_ATTEMPTED_COUNT,KAT_HAPG_CERTIFIED_COUNT,KAT_HAPG_UNCERTIFIED_COUNT,KAT_HAPG_RESCUE_COUNT,KAT_HPG_MUTATION_TESTS,KAT_HPG_MUTATIONS_REJECTED,KAT_HAPG_MUTATION_TESTS,KAT_HAPG_MUTATIONS_REJECTED,ADAPTIVE_SUBMIT_UTC,KAT_PREREQUISITE_VALID",
    "KAT_ANCHOR": "scripts/research/cs6_hapg_full_source_cover_kat_anchor.py",
    "ADAPTIVE_CONFIG_KAT_FIELDS": "KAT_ARCHIVE_PATH_KAT_ARCHIVE_SHA256_AND_KAT_JOB_ID",
    "POST_RUN_FAILURE_TRANSPORT_POLICY": "INDEPENDENT_AGGREGATOR_NONZERO_HASHED_NONPROMOTABLE_DIAGNOSTIC_ARCHIVE_REQUIRED",
    "V4_ADAPTIVE_ABORTED_SLURM_JOB_ID": "8455",
    "V4_ADAPTIVE_ABORT_STAGE": "H_PG_VERIFIER_WAVE_4",
    "V4_ADAPTIVE_ABORT_REPORTED_NODE_ID": "U02-0000000000_S02-0000000000",
    "V4_ADAPTIVE_ABORT_CONTROLLER_CLASS": "UNSTRUCTURED_EXPECTED_H_PG_VERIFIER_NEGATIVE",
    "V4_ADAPTIVE_ABORT_RUN_COMPLETE": "false",
    "V4_ADAPTIVE_ABORT_ARCHIVE_PUBLISHED": "false",
    "V4_ADAPTIVE_ABORT_EXACT_EVALUATION_COUNT": "UNKNOWN_SLURM_NODE_LOCAL_SCRATCH_NOT_RETAINED",
    "V4_ADAPTIVE_ABORT_EXECUTED_GIT_HEAD": "0c8e6ebd6d61f6eb891d958f9af57968512cba44",
    "V4_ADAPTIVE_REPO_DELTA_BUNDLE_SHA256": "2a874bda2485ed7e19ce0015a872da274b1d82cffe75607ddcb44379b0f9d304",
    "V4_ADAPTIVE_PREBUILT_ARCHIVE_SHA256": "94a8a0b0f76dc6aa6caede2cef675cd03f7f6934da5b59acaf93b099ee8f6ce4",
    "V4_ADAPTIVE_OUTPUT_DIRECTORY": "/orangefs/training/cs6-hapg-cover/0c8e6ebd6d61f6eb/results",
    "V4_LOCAL_REPRO_PUBLISHED_EVALUATED_NODE_COUNT": "15",
    "V4_LOCAL_REPRO_TOTAL_HPG_WORKER_ATTEMPTS": "31",
    "V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_ATTEMPTS": "16",
    "V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_RC0_COUNT": "14",
    "V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_CLASSIFIED_FAILURE_COUNT": "2",
    "V4_LOCAL_REPRO_HAPG_ATTEMPT_COUNT": "0",
    "V5_H_PG_EVENT_ORDER_OR_PLUS_SIDE_NEGATIVE_POLICY": "STRUCTURED_PROBE_FALSE_HAPG_INELIGIBLE_THEN_ADAPTIVE_SPLIT_IF_FROZEN_BUDGET_PERMITS",
    "V5_MIDPOINT_RECONSTRUCTION": "DISCRETE_BINARY64_OPERATION_BY_OPERATION_CANDIDATES_FOR_LO_PLUS_HALF_TIMES_HI_MINUS_LO",
    "V5_MIDPOINT_ARBITRARY_CONVEX_HULL_CONTAINMENT_ALLOWED": "false",
    "V5_H_PG_INVALID_NO_SIGNED_CHART_STATUS_SEMANTICS": "VERIFIER_COMPLETED_AND_PROBE_FALSE_OR_REQUIRED_PIVOT_UNSIGNED_HAPG_INELIGIBLE_NONE_ZERO_SENTINEL_NOT_CHART_NONEXISTENCE_PROOF",
    "V5_HPG_RC0_SELECTION_RULE": "EXACT_ROWS_WITH_RC_0",
    "V5_HPG_RC0_DIAGNOSTIC_NODE_COUNT": "54",
    "V5_HPG_RC0_DIAGNOSTIC_VERIFIER_RC0_COUNT": "54",
    "V5_HPG_RC0_DIAGNOSTIC_PROBE_FALSE_COUNT": "54",
    "V5_HPG_RC0_DIAGNOSTIC_CERTIFICATE_FALSE_COUNT": "54",
    "V5_HPG_RC0_DIAGNOSTIC_SUBDIVISION_TRUE_COUNT": "54",
    "V5_HPG_RC0_DIAGNOSTIC_MUTATION_TESTS": "4266",
    "V5_HPG_RC0_DIAGNOSTIC_MUTATIONS_REJECTED": "4266",
    "V5_KAT_HPG_MUTATION_TESTS": "4108",
    "V5_KAT_HPG_MUTATIONS_REJECTED": "4108",
    "V5_HAPG_ATTEMPT_EXPECTATION": "ZERO_DIAGNOSTIC_ONLY_NOT_EXECUTED_RESULT",
    "V5_HPG_DIAGNOSTIC_SCOPE": "LOCAL_REPLAY_OF_EXACT_V3_FULL255_RC0_SELECTION",
    "V5_HPG_DIAGNOSTIC_IS_SCIENTIFIC_RESULT": "false",
    "V5_KAT_COMPATIBILITY_SCOPE": "LOCAL_REPLAY_OF_V4_JOB_8454_HPG_RC0_RECEIPTS",
    "V5_FRESH_SLURM_KAT_EXECUTED": "true",
    "V5_AUTHORITATIVE_ADAPTIVE_EXECUTED": "false",
    "V5_AUTHORITATIVE_RESULT_AVAILABLE": "false",
    "V3_ADAPTIVE_ABORTED_SLURM_JOB_ID": "8453",
    "V3_ADAPTIVE_ABORT_STAGE": "H_PG_WAVE_1",
    "V3_ADAPTIVE_ABORT_REPORTED_NODE_ID": "U00-0000000000_S01-0000000000",
    "V3_ADAPTIVE_ABORT_CONTROLLER_CLASS": "UNRECOGNIZED_DECLARED_H_PG_CROSSING_SIGNATURE",
    "V3_ADAPTIVE_ABORT_RUN_COMPLETE": "false",
    "V3_ADAPTIVE_ABORT_ARCHIVE_PUBLISHED": "false",
    "V3_ADAPTIVE_ABORT_EXACT_EVALUATION_COUNT": "UNKNOWN_NODE_LOCAL_SCRATCH_NOT_RETAINED",
    "V3_ADAPTIVE_ABORT_EXECUTED_GIT_HEAD": "6ba7e469e3b42b47bdbcda3b0b63925f6c7d9d46",
    "V4_H_PG_NONTRANSVERSAL_CLASS": "H_PG_CROSSING",
    "V4_H_PG_NONTRANSVERSAL_SEMANTICS": "RIGOROUS_TRANSVERSALITY_NOT_CERTIFIED_NOT_PROOF_OF_ACTUAL_TANGENCY",
    "V4_H_PG_CROSSING_CERTIFIES_TRANSVERSALITY": "false",
    "V4_H_PG_CROSSING_OPERATIONAL_ACTION": "NO_SIGNED_CHART_THEN_SUBDIVIDE_OR_UNRESOLVED_BY_FROZEN_BUDGET",
    "V4_H_APG_UNDECLARED_GENERIC_FAILURE_CLASSES_ALLOWED": "false",
    "V4_H_APG_FAILURE_CENSUS_COMPLETE": "false",
    "V4_H_APG_UNKNOWN_FAILURE_POLICY": "RETURN_NONE_THEN_INFRASTRUCTURE_INVALID_AND_RUN_NOT_COMPLETE",
    "V3_HPG_FULL255_DIAGNOSTIC_NODE_COUNT": "255",
    "V3_HPG_FULL255_DIAGNOSTIC_PREFIX_UNKNOWN_COUNT": "2",
    "V3_HPG_FULL255_DIAGNOSTIC_PREFIX_UNKNOWN_COUNT_SEMANTICS": "SAME_TWO_NONTRANSVERSAL_ROWS_NOT_ADDITIONAL_PARTITION_CLASS",
    "V3_HPG_FULL255_DIAGNOSTIC_PRE_V4_CROSSING_COUNT": "16",
    "V4_HPG_FULL255_DIAGNOSTIC_POST_V4_CROSSING_COUNT": "18",
    "V4_HPG_FULL255_DIAGNOSTIC_POSTFIX_UNKNOWN_COUNT": "0",
    "V3_HPG_FULL255_DIAGNOSTIC_SIGNATURES_FIT_V3_CLASSIFIER": "false",
    "V4_HPG_FULL255_DIAGNOSTIC_SIGNATURES_FIT_V4_CLASSIFIER": "true",
    "V2_ABORTED_SLURM_JOB_ID": "8451",
    "V2_ABORT_STAGE": "PRE_SCIENCE_RUNTIME_PROVENANCE",
    "V2_ABORT_CLASS": "EMPTY_SYS_EXECUTABLE_UNDER_SLURM_EXPORT_NIL",
    "V2_ABORT_SCIENTIFIC_EVALUATIONS": "0",
    "PYTHON_RESOLUTION": "ABSOLUTE_REALPATH_FROM_COMMAND_V_PYTHON3",
    "GIT_TRANSPORT": "FROZEN_BASE_BUNDLE_PLUS_HASHED_INCREMENTAL_BUNDLE",
    "GIT_DELTA_REF_POLICY": "EXACT_SINGLE_HEAD_REF",
    "BASE_REPO_BUNDLE_GIT_HEAD": "6ca2515af28d58d025097f94c73025c0f5bc266d",
    "BASE_REPO_BUNDLE_SHA256": "cacd77ffa07966499f4614d3f84e03132bf01d765ca4fabc727c0701a9480389",
    "SOURCE": "N0",
    "TERMINAL_METHOD": "H_APG_ONLY",
    "TERMINAL_PREDICATE": "APG_COMPUTATION_VALID_AND_APG_CERTIFICATE_PASS",
    "WAVE_CONTRACT_CHAIN": "SHA256_PREVIOUS_WAVE_RESULT_AND_EXACT_NEXT_FRONTIER",
    "CHART_FREEZE_ORDER": "ALL_H_PG_PREPASSES_AND_VERIFICATIONS_THEN_ATOMIC_WAVE_CONTRACT_THEN_ANY_H_APG",
    "FRESH_REPLAY_SEMANTICS": "INDEPENDENT_RECERTIFICATION_SAME_CHARTS_DISTINCT_CHALLENGES_NOT_BITWISE_RECEIPT_REPRODUCTION",
    "WORKING_FILESYSTEM_POLICY": "NODE_LOCAL_TMP_THEN_HASHED_ARCHIVE_TRANSPORT",
    "BOUNDED_PILOT_EXECUTION_PATH": "SLURM_CPU_PREBUILT_NODE_LOCAL_TMP",
    "BOUNDED_PILOT_TRANSPORT_ROOT": "/orangefs/training",
    "BOUNDED_PILOT_SLURM_CLUSTER": "beagle-slurm-pilot",
    "BOUNDED_PILOT_SLURM_PARTITION": "gpu-orangefs",
    "BOUNDED_PILOT_SLURM_ACCOUNT": "lab",
    "BOUNDED_PILOT_SLURM_QOS": "normal",
    "BOUNDED_PILOT_SLURM_NODE": "gpuorangefs-r770-proxmox",
    "BOUNDED_PILOT_SLURM_NODES": "1",
    "BOUNDED_PILOT_SLURM_TASKS": "1",
    "BOUNDED_PILOT_SLURM_CPUS_PER_TASK": "32",
    "BOUNDED_PILOT_SLURM_ALLOCATED_CPUS": "120",
    "SLURM_ALLOCATION_SEMANTICS": "ONE_PINNED_NODE_EXCLUSIVE_PARTITION_ALLOCATES_ALL_120_EFFECTIVE_CPUS_WHILE_ONE_TASK_REQUESTS_32",
    "SLURM_CONTROL_PLANE_CHECK": "SCONTROL_RUNNING_UID_NODE_PARTITION_ACCOUNT_QOS_AND_RESOURCES",
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "OPTIMIZATION_LEVEL": "O0",
    "KAT_EXPECTED_ATTEMPTED": "53",
    "KAT_EXPECTED_H_PG_VALID": "52",
    "KAT_EXPECTED_H_APG_VALID": "52",
    "KAT_EXPECTED_H_APG_CERTIFIED": "48",
    "KAT_EXPECTED_H_APG_UNCERTIFIED": "4",
    "KAT_EXPECTED_H_APG_RESCUES": "20",
    "KAT_EXPECTED_HPG_MUTATION_TESTS": "4108",
    "KAT_EXPECTED_HPG_MUTATIONS_REJECTED": "4108",
    "KAT_EXPECTED_HAPG_MUTATION_TESTS": "5824",
    "KAT_EXPECTED_HAPG_MUTATIONS_REJECTED": "5824",
    "BOUNDED_PILOT_MAX_NODES": "255",
    "BOUNDED_PILOT_MAX_WAVES": "8",
    "BOUNDED_PILOT_MAX_U_DEPTH": "30",
    "BOUNDED_PILOT_MAX_S_DEPTH": "30",
    "BOUNDED_PILOT_JOBS": "32",
    "BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS": "300",
}
for key, value in exact.items():
    if fields.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: frozen contract mismatch for {key}")

sha_keys = (
    "SUPERSEDES_V5_SHA256",
    "SUPERSEDES_V4_SHA256",
    "SUPERSEDES_V3_SHA256",
    "PREPASS_WORKER_SHA256",
    "PREPASS_VERIFIER_SHA256",
    "H_APG_WRAPPER_SHA256",
    "H_APG_KERNEL_SHA256",
    "H_APG_ADAPTER_SHA256",
    "H_APG_NUMERIC_VERIFIER_SHA256",
    "RUNNER_SHA256",
    "AGGREGATOR_SHA256",
    "KAT_ANCHOR_SHA256",
    "EXACT_TREE_KERNEL_SHA256",
    "GATE_SHA256",
    "SLURM_JOB_SCRIPT_SHA256",
    "KAT_COORDINATE_MANIFEST_SHA256",
    "KAT_EXPECTED_RESULTS_SHA256",
    "KAT_ROOT_CHALLENGE",
    "BOUNDED_PILOT_ROOT_CHALLENGE",
    "BOUNDED_PILOT_REPLAY_ROOT_CHALLENGE",
    "PREBUILT_HPG_BINARY_SHA256",
    "PREBUILT_HAPG_BINARY_SHA256",
    "V2_ABORT_RECEIPT_MANIFEST_SHA256",
    "V2_ABORT_SACCT_SHA256",
    "V2_ABORT_CONFIG_SHA256",
    "V2_ABORT_STDERR_SHA256",
    "V3_ABORT_RECEIPT_MANIFEST_SHA256",
    "V3_ABORT_SACCT_SHA256",
    "V3_ABORT_CONFIG_SHA256",
    "V3_ABORT_SLURM_STDERR_SHA256",
    "V3_ABORT_REPRO_S0_STDOUT_SHA256",
    "V3_ABORT_REPRO_S0_STDERR_SHA256",
    "V3_ABORT_REPRO_S1_STDOUT_SHA256",
    "V3_ABORT_REPRO_S1_STDERR_SHA256",
    "V3_ABORT_HPG_FULL255_CENSUS_SHA256",
    "V3_ABORT_HPG_FULL255_CENSUS_SUMMARY_SHA256",
    "V3_ABORT_HPG_FULL255_STDERR_JSONL_SHA256",
    "V3_ABORT_HPG_CHALLENGE_SPOTCHECK_SHA256",
    "V4_ABORT_RECEIPT_MANIFEST_SHA256",
    "V4_ABORT_FILES_INDEX_SHA256",
    "V4_ABORT_SACCT_SHA256",
    "V4_ABORT_CONFIG_SHA256",
    "V4_ABORT_SLURM_STDOUT_SHA256",
    "V4_ABORT_HPG_RC0_CORPUS_SHA256",
    "V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256",
    "V4_ABORT_HPG_RC0_CENSUS_SHA256",
    "V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256",
    "V4_ABORT_HPG_V5_KAT_COMPAT_SHA256",
    "V4_ABORT_HPG_V4_KAT_CORPUS_SHA256",
    "V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256",
    "V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256",
    "V4_ABORT_LOCAL_REPRO_SHA256",
    "V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256",
    "V5_ABORT_RECEIPT_MANIFEST_SHA256",
    "V5_ABORT_FILES_INDEX_SHA256",
    "V5_ABORT_JOBS_SACCT_SHA256",
    "V5_KAT_ARCHIVE_SHA256",
    "V5_KAT_CONFIG_SHA256",
)
for key in sha_keys:
    if re.fullmatch(r"[0-9a-f]{64}", fields.get(key, "")) is None:
        raise SystemExit(f"H-APG cover gate error: invalid SHA-256 field {key}")
if re.fullmatch(r"[0-9a-f]{40}", fields.get("IMPLEMENTATION_COMMIT", "")) is None:
    raise SystemExit("H-APG cover gate error: invalid implementation commit")
if re.fullmatch(r"[0-9a-f]{40}", fields.get("IMPLEMENTATION_PARENT_COMMIT", "")) is None:
    raise SystemExit("H-APG cover gate error: invalid implementation parent")

repo_root = path.resolve().parents[2]
implementation = fields["IMPLEMENTATION_COMMIT"]
try:
    commit_type = subprocess.run(
        ["git", "-C", repo_root, "cat-file", "-t", implementation],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    parent = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", f"{implementation}^"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", repo_root, "merge-base", "--is-ancestor", implementation, "HEAD"],
        check=True,
        capture_output=True,
    )
except subprocess.CalledProcessError as error:
    raise SystemExit("H-APG cover gate error: implementation commit is unavailable or unrelated") from error
if commit_type != "commit" or parent != fields["IMPLEMENTATION_PARENT_COMMIT"]:
    raise SystemExit("H-APG cover gate error: implementation commit lineage mismatch")

implementation_bindings = {
    "PREPASS_WORKER_SHA256": "scripts/research/cs6_plucker_cocycle_probe.cpp",
    "PREPASS_VERIFIER_SHA256": "scripts/research/cs6_plucker_cocycle_verify.py",
    "H_APG_WRAPPER_SHA256": "scripts/research/cs6_hapg_full_source_cover_worker.cpp",
    "H_APG_KERNEL_SHA256": "scripts/research/cs6_affine_projective_cocycle_full53_probe.cpp",
    "H_APG_ADAPTER_SHA256": "scripts/research/cs6_hapg_full_source_cover_verify.py",
    "H_APG_NUMERIC_VERIFIER_SHA256": "scripts/research/cs6_affine_projective_cocycle_full53_verify.py",
    "RUNNER_SHA256": "scripts/research/cs6_hapg_full_source_cover_run.py",
    "AGGREGATOR_SHA256": "scripts/research/cs6_hapg_full_source_cover_aggregate.py",
    "KAT_ANCHOR_SHA256": "scripts/research/cs6_hapg_full_source_cover_kat_anchor.py",
    "EXACT_TREE_KERNEL_SHA256": "scripts/research/cs6_c1_full_source_cover_aggregate.py",
    "GATE_SHA256": "scripts/ci/cs6_hapg_full_source_cover_gate.sh",
    "SLURM_JOB_SCRIPT_SHA256": "scripts/research/cs6_hapg_full_source_cover_slurm_job.sh",
    "V4_ABORT_RECEIPT_MANIFEST_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/manifest.txt",
    "V4_ABORT_FILES_INDEX_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/files.sha256",
    "V4_ABORT_CONFIG_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/config.txt",
    "V4_ABORT_HPG_RC0_CORPUS_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/hpg-rc0-corpus.tar",
    "V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/corpus-files.sha256",
    "V4_ABORT_HPG_RC0_CENSUS_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/hpg-rc0-verifier-census.tsv",
    "V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/hpg-rc0-verifier-census-summary.txt",
    "V4_ABORT_HPG_V5_KAT_COMPAT_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/hpg-v5-kat-compat.tsv",
    "V4_ABORT_HPG_V4_KAT_CORPUS_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/hpg-v4-kat-corpus.tar",
    "V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/hpg-v4-kat-corpus-files.sha256",
    "V4_ABORT_LOCAL_REPRO_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/local-repro.tar",
    "V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/midpoint-discrete-negative-test.txt",
    "V4_ABORT_SACCT_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/sacct.txt",
    "V4_ABORT_SLURM_STDOUT_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/slurm-stdout.txt",
    "V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1/v4-hpg-verifier.py",
    "V5_ABORT_RECEIPT_MANIFEST_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v5_abort_8463_v1/manifest.txt",
    "V5_ABORT_FILES_INDEX_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v5_abort_8463_v1/files.sha256",
    "V5_ABORT_JOBS_SACCT_SHA256": "scripts/research/receipts/cs6_hapg_full_source_cover_v5_abort_8463_v1/jobs-8458-8463.sacct.psv",
}
for key, relative in implementation_bindings.items():
    try:
        blob = subprocess.run(
            ["git", "-C", repo_root, "show", f"{implementation}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"H-APG cover gate error: implementation blob missing: {relative}") from error
    if hashlib.sha256(blob).hexdigest() != fields[key]:
        raise SystemExit(f"H-APG cover gate error: implementation blob mismatch for {key}")


def parse_kv(receipt: Path) -> dict[str, str]:
    receipt_raw = receipt.read_bytes()
    if not receipt_raw.endswith(b"\n") or b"\r" in receipt_raw or b"\0" in receipt_raw:
        raise SystemExit(f"H-APG cover gate error: noncanonical receipt {receipt.name}")
    result: dict[str, str] = {}
    for row in receipt_raw.decode("ascii").splitlines():
        if row.count("=") != 1:
            raise SystemExit(f"H-APG cover gate error: malformed receipt {receipt.name}")
        key, value = row.split("=", 1)
        if not key or not value or key in result:
            raise SystemExit(f"H-APG cover gate error: duplicate receipt field {receipt.name}")
        result[key] = value
    return result


abort = parse_kv(abort_root / "manifest.txt")
abort_exact = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-v2-abort.v1",
    "EVIDENCE_CLASS": "SLURM_ACCOUNTING_AND_RAW_STAGE_LOG_NO_ATTESTATION",
    "SLURM_JOB_ID": fields["V2_ABORTED_SLURM_JOB_ID"],
    "SLURM_STATE": "FAILED",
    "SLURM_EXIT_CODE": "1:0",
    "SLURM_ALLOCATED_CPUS": fields["BOUNDED_PILOT_SLURM_ALLOCATED_CPUS"],
    "SLURM_REQUESTED_CPUS": fields["BOUNDED_PILOT_SLURM_CPUS_PER_TASK"],
    "SACCT_SHA256": fields["V2_ABORT_SACCT_SHA256"],
    "CONFIG_SHA256": fields["V2_ABORT_CONFIG_SHA256"],
    "STDERR_SHA256": fields["V2_ABORT_STDERR_SHA256"],
    "SCIENTIFIC_POPULATION_PARSED": "false",
    "SCIENTIFIC_WORKERS_LAUNCHED": "0",
    "SCIENTIFIC_EVALUATIONS": fields["V2_ABORT_SCIENTIFIC_EVALUATIONS"],
    "EXECUTION_PROVENANCE_ATTESTED": "false",
    "PROMOTION_ELIGIBLE": "false",
}
for key, value in abort_exact.items():
    if abort.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: v2 abort receipt mismatch for {key}")
sacct = (abort_root / "sacct.txt").read_text(encoding="ascii")
if not sacct.startswith(fields["V2_ABORTED_SLURM_JOB_ID"] + "|") or "|FAILED|1:0|" not in sacct:
    raise SystemExit("H-APG cover gate error: v2 abort sacct record mismatch")
stderr = (abort_root / "stderr.txt").read_text(encoding="ascii")
if "subprocess.run([python, \"--version\"]" not in stderr or not stderr.endswith(
    "PermissionError: [Errno 13] Permission denied: PosixPath('/tmp')\n"
):
    raise SystemExit("H-APG cover gate error: v2 abort call path mismatch")

v3_abort = parse_kv(v3_abort_root / "manifest.txt")
v3_contract_raw = v3_contract_path.read_bytes()
if hashlib.sha256(v3_contract_raw).hexdigest() != fields["SUPERSEDES_V3_SHA256"]:
    raise SystemExit("H-APG cover gate error: superseded v3 contract bytes mismatch")
v3_contract = parse_kv(v3_contract_path)
if v3_contract.get("SCHEMA") != "sounio.cs6.hapg-full-source-cover-contract.v3":
    raise SystemExit("H-APG cover gate error: superseded v3 contract schema mismatch")
v3_abort_exact = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-v3-adaptive-abort.v1",
    "EVIDENCE_CLASS": "SLURM_FAILURE_RAW_LOG_EXACT_REPRO_AND_FULL255_HPG_DIAGNOSTIC_NO_ATTESTATION",
    "SLURM_JOB_ID": fields["V3_ADAPTIVE_ABORTED_SLURM_JOB_ID"],
    "SLURM_STATE": "FAILED",
    "SLURM_EXIT_CODE": "1:0",
    "SACCT_SHA256": fields["V3_ABORT_SACCT_SHA256"],
    "CONFIG_SHA256": fields["V3_ABORT_CONFIG_SHA256"],
    "SLURM_STDERR_SHA256": fields["V3_ABORT_SLURM_STDERR_SHA256"],
    "EXECUTED_CONTRACT_SHA256": fields["SUPERSEDES_V3_SHA256"],
    "EXECUTED_GIT_HEAD": fields["V3_ADAPTIVE_ABORT_EXECUTED_GIT_HEAD"],
    "EXECUTED_RUNNER_SHA256": v3_contract["RUNNER_SHA256"],
    "EXECUTED_HPG_WORKER_BINARY_SHA256": v3_contract["PREBUILT_HPG_BINARY_SHA256"],
    "FAILURE_STAGE": fields["V3_ADAPTIVE_ABORT_STAGE"],
    "REPORTED_NODE_ID": fields["V3_ADAPTIVE_ABORT_REPORTED_NODE_ID"],
    "CONTROLLER_FAILURE_CLASS": fields["V3_ADAPTIVE_ABORT_CONTROLLER_CLASS"],
    "RUN_COMPLETE": fields["V3_ADAPTIVE_ABORT_RUN_COMPLETE"],
    "TRANSPORT_ARCHIVE_PUBLISHED": fields["V3_ADAPTIVE_ABORT_ARCHIVE_PUBLISHED"],
    "EXACT_EVALUATION_COUNT": fields["V3_ADAPTIVE_ABORT_EXACT_EVALUATION_COUNT"],
    "REPRO_S0_STDOUT_SHA256": fields["V3_ABORT_REPRO_S0_STDOUT_SHA256"],
    "REPRO_S0_STDERR_SHA256": fields["V3_ABORT_REPRO_S0_STDERR_SHA256"],
    "REPRO_S1_STDOUT_SHA256": fields["V3_ABORT_REPRO_S1_STDOUT_SHA256"],
    "REPRO_S1_STDERR_SHA256": fields["V3_ABORT_REPRO_S1_STDERR_SHA256"],
    "HPG_FULL255_CENSUS_SHA256": fields["V3_ABORT_HPG_FULL255_CENSUS_SHA256"],
    "HPG_FULL255_CENSUS_SUMMARY_SHA256": fields["V3_ABORT_HPG_FULL255_CENSUS_SUMMARY_SHA256"],
    "HPG_FULL255_STDERR_JSONL_SHA256": fields["V3_ABORT_HPG_FULL255_STDERR_JSONL_SHA256"],
    "HPG_CHALLENGE_SPOTCHECK_SHA256": fields["V3_ABORT_HPG_CHALLENGE_SPOTCHECK_SHA256"],
    "HPG_FULL255_NODE_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_NODE_COUNT"],
    "HPG_FULL255_PREEXISTING_CROSSING_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_PRE_V4_CROSSING_COUNT"],
    "HPG_FULL255_NONTRANSVERSAL_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_PREFIX_UNKNOWN_COUNT"],
    "HPG_FULL255_POST_V4_UNKNOWN_COUNT": fields["V4_HPG_FULL255_DIAGNOSTIC_POSTFIX_UNKNOWN_COUNT"],
    "HPG_FULL255_PREFIX_UNKNOWN_COUNT_SEMANTICS": fields["V3_HPG_FULL255_DIAGNOSTIC_PREFIX_UNKNOWN_COUNT_SEMANTICS"],
    "V4_CLASSIFICATION_TARGET": fields["V4_H_PG_NONTRANSVERSAL_CLASS"],
    "CLASSIFICATION_SEMANTICS": fields["V4_H_PG_NONTRANSVERSAL_SEMANTICS"],
    "EXECUTION_PROVENANCE_ATTESTED": "false",
    "PROMOTION_ELIGIBLE": "false",
}
for key, value in v3_abort_exact.items():
    if v3_abort.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: v3 abort receipt mismatch for {key}")
v3_sacct = (v3_abort_root / "sacct.txt").read_text(encoding="ascii")
if not v3_sacct.startswith(fields["V3_ADAPTIVE_ABORTED_SLURM_JOB_ID"] + "|") or "|FAILED|1:0|" not in v3_sacct:
    raise SystemExit("H-APG cover gate error: v3 abort sacct record mismatch")
v3_stderr = (v3_abort_root / "slurm-stderr.txt").read_text(encoding="ascii")
if fields["V3_ADAPTIVE_ABORT_REPORTED_NODE_ID"] not in v3_stderr or not v3_stderr.endswith(
    "RuntimeError: unexpected H-PG worker failure for U00-0000000000_S01-0000000000: rc=1\n"
):
    raise SystemExit("H-APG cover gate error: v3 abort controller call path mismatch")
for suffix in ("s0", "s1"):
    repro = (v3_abort_root / f"repro-{suffix}-stderr.txt").read_bytes().lower()
    if not repro.startswith(
        b"probe error: poincaremap error: possible nontransversal return to the section"
    ) or b"\ninner product of vector field and section gradient: [" not in repro:
        raise SystemExit(f"H-APG cover gate error: v3 {suffix} repro signature mismatch")
census = parse_kv(v3_abort_root / "hpg-full255-census-summary.txt")
census_expected = {
    "NODE_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_NODE_COUNT"],
    "H_PG_SUCCESS_COUNT": "54",
    "H_PG_INTERVAL_DOMAIN_COUNT": "1",
    "H_PG_CROSSING_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_PRE_V4_CROSSING_COUNT"],
    "H_PG_CAPD_SET_COUNT": "182",
    "H_PG_TIMEOUT_COUNT": "0",
    "H_PG_POINCARE_NONTRANSVERSAL_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_PREFIX_UNKNOWN_COUNT"],
    "UNKNOWN_RAW_FAILURE_COUNT": fields["V3_HPG_FULL255_DIAGNOSTIC_PREFIX_UNKNOWN_COUNT"],
    "SIGNATURES_FIT_CURRENT_FAILURE_CLASSES": fields["V3_HPG_FULL255_DIAGNOSTIC_SIGNATURES_FIT_V3_CLASSIFIER"],
}
for key, value in census_expected.items():
    if census.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: H-PG census mismatch for {key}")

v4_contract_raw = v4_contract_path.read_bytes()
if hashlib.sha256(v4_contract_raw).hexdigest() != fields["SUPERSEDES_V4_SHA256"]:
    raise SystemExit("H-APG cover gate error: superseded v4 contract bytes mismatch")
v4_contract = parse_kv(v4_contract_path)
if v4_contract.get("SCHEMA") != "sounio.cs6.hapg-full-source-cover-contract.v4":
    raise SystemExit("H-APG cover gate error: superseded v4 contract schema mismatch")
v5_contract_raw = v5_contract_path.read_bytes()
if hashlib.sha256(v5_contract_raw).hexdigest() != fields["SUPERSEDES_V5_SHA256"]:
    raise SystemExit("H-APG cover gate error: superseded v5 contract bytes mismatch")
v5_contract = parse_kv(v5_contract_path)
if v5_contract.get("SCHEMA") != "sounio.cs6.hapg-full-source-cover-contract.v5":
    raise SystemExit("H-APG cover gate error: superseded v5 contract schema mismatch")
v5_only_keys = {
    "IMPLEMENTATION_PARENT_COMMIT",
    "SUPERSEDES_V4_SHA256",
    "V4_ABORT_CONFIG_SHA256",
    "V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256",
    "V4_ABORT_FILES_INDEX_SHA256",
    "V4_ABORT_HPG_RC0_CENSUS_SHA256",
    "V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256",
    "V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256",
    "V4_ABORT_HPG_RC0_CORPUS_SHA256",
    "V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256",
    "V4_ABORT_HPG_V4_KAT_CORPUS_SHA256",
    "V4_ABORT_HPG_V5_KAT_COMPAT_SHA256",
    "V4_ABORT_LOCAL_REPRO_SHA256",
    "V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256",
    "V4_ABORT_RECEIPT_MANIFEST_SHA256",
    "V4_ABORT_SACCT_SHA256",
    "V4_ABORT_SLURM_STDOUT_SHA256",
    "V4_ADAPTIVE_ABORTED_SLURM_JOB_ID",
    "V4_ADAPTIVE_ABORT_ARCHIVE_PUBLISHED",
    "V4_ADAPTIVE_ABORT_CONTROLLER_CLASS",
    "V4_ADAPTIVE_ABORT_EXACT_EVALUATION_COUNT",
    "V4_ADAPTIVE_ABORT_EXECUTED_GIT_HEAD",
    "V4_ADAPTIVE_ABORT_REPORTED_NODE_ID",
    "V4_ADAPTIVE_ABORT_RUN_COMPLETE",
    "V4_ADAPTIVE_ABORT_STAGE",
    "V4_ADAPTIVE_OUTPUT_DIRECTORY",
    "V4_ADAPTIVE_PREBUILT_ARCHIVE_SHA256",
    "V4_ADAPTIVE_REPO_DELTA_BUNDLE_SHA256",
    "V4_LOCAL_REPRO_HAPG_ATTEMPT_COUNT",
    "V4_LOCAL_REPRO_PUBLISHED_EVALUATED_NODE_COUNT",
    "V4_LOCAL_REPRO_TOTAL_HPG_WORKER_ATTEMPTS",
    "V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_ATTEMPTS",
    "V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_CLASSIFIED_FAILURE_COUNT",
    "V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_RC0_COUNT",
    "V5_AUTHORITATIVE_ADAPTIVE_EXECUTED",
    "V5_AUTHORITATIVE_RESULT_AVAILABLE",
    "V5_FRESH_SLURM_KAT_EXECUTED",
    "V5_HAPG_ATTEMPT_EXPECTATION",
    "V5_HPG_DIAGNOSTIC_IS_SCIENTIFIC_RESULT",
    "V5_HPG_DIAGNOSTIC_SCOPE",
    "V5_HPG_RC0_DIAGNOSTIC_CERTIFICATE_FALSE_COUNT",
    "V5_HPG_RC0_DIAGNOSTIC_MUTATIONS_REJECTED",
    "V5_HPG_RC0_DIAGNOSTIC_MUTATION_TESTS",
    "V5_HPG_RC0_DIAGNOSTIC_NODE_COUNT",
    "V5_HPG_RC0_DIAGNOSTIC_PROBE_FALSE_COUNT",
    "V5_HPG_RC0_DIAGNOSTIC_SUBDIVISION_TRUE_COUNT",
    "V5_HPG_RC0_DIAGNOSTIC_VERIFIER_RC0_COUNT",
    "V5_HPG_RC0_SELECTION_RULE",
    "V5_H_PG_EVENT_ORDER_OR_PLUS_SIDE_NEGATIVE_POLICY",
    "V5_H_PG_INVALID_NO_SIGNED_CHART_STATUS_SEMANTICS",
    "V5_KAT_COMPATIBILITY_SCOPE",
    "V5_KAT_HPG_MUTATIONS_REJECTED",
    "V5_KAT_HPG_MUTATION_TESTS",
    "V5_MIDPOINT_ARBITRARY_CONVEX_HULL_CONTAINMENT_ALLOWED",
    "V5_MIDPOINT_RECONSTRUCTION",
}
changed_v4_keys = {
    "AGGREGATOR_SHA256",
    "GATE_SHA256",
    "H_APG_ADAPTER_SHA256",
    "IMPLEMENTATION_COMMIT",
    "PREPASS_VERIFIER_SHA256",
    "RECOVERY_SCOPE",
    "RUNNER_SHA256",
    "SCHEMA",
    "SLURM_JOB_SCRIPT_SHA256",
}
expected_v5_keys = set(v4_contract) | v5_only_keys
if set(v5_contract) != expected_v5_keys:
    missing = sorted(expected_v5_keys - set(v5_contract))
    extra = sorted(set(v5_contract) - expected_v5_keys)
    raise SystemExit(
        f"H-APG cover gate error: v5 contract key-set mismatch: missing={missing} extra={extra}"
    )
actual_changed_v4_keys = {
    key for key in v4_contract if v5_contract[key] != v4_contract[key]
}
if actual_changed_v4_keys != changed_v4_keys:
    raise SystemExit(
        "H-APG cover gate error: v5 changed-key set mismatch: "
        f"expected={sorted(changed_v4_keys)} actual={sorted(actual_changed_v4_keys)}"
    )
v6_only_keys = {
    "SUPERSEDES_V5_SHA256",
    "V5_ABORT_RECEIPT_MANIFEST_SHA256",
    "V5_ABORT_FILES_INDEX_SHA256",
    "V5_ABORT_JOBS_SACCT_SHA256",
    "V5_KAT_SLURM_JOB_ID",
    "V5_KAT_SLURM_STATE",
    "V5_KAT_SLURM_EXIT_CODE",
    "V5_KAT_ARCHIVE_SHA256",
    "V5_KAT_CONFIG_SHA256",
    "V5_ADAPTIVE_SLURM_JOB_ID",
    "V5_ADAPTIVE_SLURM_STATE",
    "V5_ADAPTIVE_SLURM_EXIT_CODE",
    "V5_ADAPTIVE_FAILURE_STAGE",
    "V5_ADAPTIVE_FAILURE_SIGNATURE",
    "V5_ADAPTIVE_RUNNER_REPORTED_EVALUATED_NODE_COUNT",
    "V5_ADAPTIVE_RUNNER_REPORTED_WAVE_COUNT",
    "V5_ADAPTIVE_ARCHIVE_PUBLISHED",
    "V5_ADAPTIVE_RESULT_AVAILABLE",
    "V5_ADAPTIVE_RUNNER_CANDIDATE_FIELD_IS_SCIENTIFIC_RESULT",
    "V6_FILE_INDEX_ORDER",
    "V6_KAT_PREREQUISITE_REQUIRED_FOR_ADAPTIVE",
    "V6_KAT_ARCHIVE_SAFE_FULL_INDEX_REPLAY_REQUIRED",
    "V6_KAT_LEAF_BY_LEAF_VERIFIER_REPLAY_REQUIRED",
    "V6_KAT_SAME_GIT_HEAD_REQUIRED",
    "V6_KAT_SAME_FROZEN_CONTRACT_REQUIRED",
    "V6_KAT_SAME_BASE_DELTA_PREBUILT_AND_JOB_SCRIPT_REQUIRED",
    "V6_KAT_SLURM_COMPLETED_ZERO_EXIT_REQUIRED",
    "V6_KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT_REQUIRED",
    "V6_KAT_CERTIFICATE_INCLUDED_IN_ADAPTIVE_BUNDLE",
    "V6_KAT_CERTIFICATE_SHA256_BOUND_IN_ADAPTIVE_RUN_CONTRACT",
    "V6_AGGREGATOR_REVALIDATES_KAT_ARCHIVE_AND_CERTIFICATE",
    "V6_POST_RUN_FAILURE_DIAGNOSTIC_ARCHIVE_REQUIRED",
    "V6_FRESH_SLURM_KAT_EXECUTED",
    "V6_AUTHORITATIVE_ADAPTIVE_EXECUTED",
    "V6_AUTHORITATIVE_RESULT_AVAILABLE",
    "KAT_ANCHOR",
    "KAT_ANCHOR_SHA256",
    "SLURM_CONFIG_SCHEMA",
    "PREBUILT_MANIFEST_SCHEMA",
    "RUN_CONTRACT_SCHEMA",
    "TRANSPORT_MANIFEST_SCHEMA",
    "AGGREGATION_SCHEMA",
    "KAT_PREREQUISITE_CERTIFICATE_SCHEMA",
    "KAT_PREREQUISITE_SACCT_ARTIFACT",
    "KAT_PREREQUISITE_CERTIFICATE_ARTIFACT",
    "KAT_PREREQUISITE_TRANSPORT_CONFIG_ARTIFACT",
    "ADAPTIVE_RUN_CONTRACT_KAT_FIELDS",
    "ADAPTIVE_CONFIG_KAT_FIELDS",
    "POST_RUN_FAILURE_TRANSPORT_POLICY",
    "BOUNDED_PILOT_SLURM_CLUSTER",
    "KAT_EXPECTED_HPG_MUTATION_TESTS",
    "KAT_EXPECTED_HPG_MUTATIONS_REJECTED",
    "KAT_EXPECTED_HAPG_MUTATION_TESTS",
    "KAT_EXPECTED_HAPG_MUTATIONS_REJECTED",
}
changed_v5_keys = {
    "AGGREGATOR_SHA256",
    "GATE_SHA256",
    "IMPLEMENTATION_COMMIT",
    "IMPLEMENTATION_PARENT_COMMIT",
    "RECOVERY_SCOPE",
    "RUNNER_SHA256",
    "SCHEMA",
    "SLURM_JOB_SCRIPT_SHA256",
    "V5_FRESH_SLURM_KAT_EXECUTED",
}
expected_v6_keys = set(v5_contract) | v6_only_keys
if set(fields) != expected_v6_keys:
    missing = sorted(expected_v6_keys - set(fields))
    extra = sorted(set(fields) - expected_v6_keys)
    raise SystemExit(
        f"H-APG cover gate error: v6 contract key-set mismatch: missing={missing} extra={extra}"
    )
actual_changed_v5_keys = {
    key for key in v5_contract if fields[key] != v5_contract[key]
}
if actual_changed_v5_keys != changed_v5_keys:
    raise SystemExit(
        "H-APG cover gate error: v6 changed-key set mismatch: "
        f"expected={sorted(changed_v5_keys)} actual={sorted(actual_changed_v5_keys)}"
    )
v4_abort = parse_kv(v4_abort_root / "manifest.txt")
v4_abort_exact = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-v4-adaptive-abort.v1",
    "EVIDENCE_CLASS": "SLURM_FAILURE_RAW_LOG_LOCAL_EXACT_REPRO_AND_FULL54_HPG_VERIFIER_DIAGNOSTIC_NO_ATTESTATION",
    "SLURM_JOB_ID": fields["V4_ADAPTIVE_ABORTED_SLURM_JOB_ID"],
    "SLURM_STATE": "FAILED",
    "SLURM_EXIT_CODE": "1:0",
    "SACCT_SHA256": fields["V4_ABORT_SACCT_SHA256"],
    "CONFIG_SHA256": fields["V4_ABORT_CONFIG_SHA256"],
    "SLURM_STDOUT_SHA256": fields["V4_ABORT_SLURM_STDOUT_SHA256"],
    "FILES_INDEX_SHA256": fields["V4_ABORT_FILES_INDEX_SHA256"],
    "FILE_COUNT": "13",
    "EXECUTED_GIT_HEAD": fields["V4_ADAPTIVE_ABORT_EXECUTED_GIT_HEAD"],
    "EXECUTED_CONTRACT_SHA256": fields["SUPERSEDES_V4_SHA256"],
    "EXECUTED_RUNNER_SHA256": v4_contract["RUNNER_SHA256"],
    "EXECUTED_HPG_WORKER_SOURCE_SHA256": fields["PREPASS_WORKER_SHA256"],
    "EXECUTED_HPG_WORKER_BINARY_SHA256": fields["PREBUILT_HPG_BINARY_SHA256"],
    "EXECUTED_HPG_VERIFIER_SHA256": fields["V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256"],
    "FAILURE_STAGE": fields["V4_ADAPTIVE_ABORT_STAGE"],
    "REPORTED_NODE_ID": fields["V4_ADAPTIVE_ABORT_REPORTED_NODE_ID"],
    "CONTROLLER_FAILURE_CLASS": fields["V4_ADAPTIVE_ABORT_CONTROLLER_CLASS"],
    "RUN_COMPLETE": fields["V4_ADAPTIVE_ABORT_RUN_COMPLETE"],
    "TRANSPORT_ARCHIVE_PUBLISHED": fields["V4_ADAPTIVE_ABORT_ARCHIVE_PUBLISHED"],
    "EXACT_EVALUATION_COUNT": fields["V4_ADAPTIVE_ABORT_EXACT_EVALUATION_COUNT"],
    "LOCAL_REPRO_SHA256": fields["V4_ABORT_LOCAL_REPRO_SHA256"],
    "LOCAL_REPRO_COMPLETE_THROUGH_ABORT": "true",
    "LOCAL_REPRO_PUBLISHED_EVALUATED_NODE_COUNT": fields["V4_LOCAL_REPRO_PUBLISHED_EVALUATED_NODE_COUNT"],
    "LOCAL_REPRO_TOTAL_HPG_WORKER_ATTEMPTS": fields["V4_LOCAL_REPRO_TOTAL_HPG_WORKER_ATTEMPTS"],
    "LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_ATTEMPTS": fields["V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_ATTEMPTS"],
    "LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_RC0_COUNT": fields["V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_RC0_COUNT"],
    "LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_CLASSIFIED_FAILURE_COUNT": fields["V4_LOCAL_REPRO_UNPUBLISHED_WAVE4_HPG_CLASSIFIED_FAILURE_COUNT"],
    "LOCAL_REPRO_HAPG_ATTEMPT_COUNT": fields["V4_LOCAL_REPRO_HAPG_ATTEMPT_COUNT"],
    "HPG_RC0_CORPUS_SHA256": fields["V4_ABORT_HPG_RC0_CORPUS_SHA256"],
    "HPG_RC0_CORPUS_FILES_SHA256": fields["V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256"],
    "HPG_RC0_CENSUS_SHA256": fields["V4_ABORT_HPG_RC0_CENSUS_SHA256"],
    "HPG_RC0_CENSUS_SUMMARY_SHA256": fields["V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256"],
    "HPG_V5_KAT_COMPAT_SHA256": fields["V4_ABORT_HPG_V5_KAT_COMPAT_SHA256"],
    "HPG_V4_KAT_CORPUS_SHA256": fields["V4_ABORT_HPG_V4_KAT_CORPUS_SHA256"],
    "HPG_V4_KAT_CORPUS_FILES_SHA256": fields["V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256"],
    "HPG_V4_KAT_SOURCE_SLURM_JOB_ID": "8454",
    "HPG_V4_KAT_SOURCE_ARCHIVE_SHA256": "9fd190ba0ac9508776bb6152bb1bc7a8c6844a0e9c9611f8f4ed7466887b9283",
    "HPG_V4_KAT_CORPUS_REGULAR_FILE_COUNT": "223",
    "HPG_V4_KAT_CORPUS_PORTABLE_REPLAY": "true",
    "MIDPOINT_DISCRETE_TEST_SHA256": fields["V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256"],
    "SELECTION_CENSUS_SHA256": fields["V3_ABORT_HPG_FULL255_CENSUS_SHA256"],
    "SELECTION_RULE": fields["V5_HPG_RC0_SELECTION_RULE"],
    "HPG_RC0_CORPUS_NODE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_NODE_COUNT"],
    "V5_VERIFIER_SHA256": fields["PREPASS_VERIFIER_SHA256"],
    "V5_VERIFIER_RC0_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_VERIFIER_RC0_COUNT"],
    "V5_PROBE_FALSE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_PROBE_FALSE_COUNT"],
    "V5_CERTIFICATE_FALSE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_CERTIFICATE_FALSE_COUNT"],
    "V5_SUBDIVISION_REQUIRED_TRUE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_SUBDIVISION_TRUE_COUNT"],
    "V5_MUTATION_TESTS": fields["V5_HPG_RC0_DIAGNOSTIC_MUTATION_TESTS"],
    "V5_MUTATIONS_REJECTED": fields["V5_HPG_RC0_DIAGNOSTIC_MUTATIONS_REJECTED"],
    "V5_KAT_MUTATION_TESTS": fields["V5_KAT_HPG_MUTATION_TESTS"],
    "V5_KAT_MUTATIONS_REJECTED": fields["V5_KAT_HPG_MUTATIONS_REJECTED"],
    "MIDPOINT_RECONSTRUCTION": fields["V5_MIDPOINT_RECONSTRUCTION"],
    "MIDPOINT_ARBITRARY_CONVEX_HULL_CONTAINMENT_ALLOWED": fields["V5_MIDPOINT_ARBITRARY_CONVEX_HULL_CONTAINMENT_ALLOWED"],
    "EVENT_ORDER_OR_PLUS_SIDE_NEGATIVE_POLICY": fields["V5_H_PG_EVENT_ORDER_OR_PLUS_SIDE_NEGATIVE_POLICY"],
    "H_PG_INVALID_NO_SIGNED_CHART_STATUS_SEMANTICS": fields["V5_H_PG_INVALID_NO_SIGNED_CHART_STATUS_SEMANTICS"],
    "V5_HAPG_ATTEMPT_EXPECTATION": fields["V5_HAPG_ATTEMPT_EXPECTATION"],
    "DIAGNOSTIC_EXECUTION_PROVENANCE_ATTESTED": "false",
    "PROMOTION_ELIGIBLE": "false",
}
for key, value in v4_abort_exact.items():
    if v4_abort.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: v4 abort receipt mismatch for {key}")

payload_names = {
    "config.txt",
    "corpus-files.sha256",
    "hpg-rc0-corpus.tar",
    "hpg-rc0-verifier-census-summary.txt",
    "hpg-rc0-verifier-census.tsv",
    "hpg-v5-kat-compat.tsv",
    "hpg-v4-kat-corpus.tar",
    "hpg-v4-kat-corpus-files.sha256",
    "local-repro.tar",
    "midpoint-discrete-negative-test.txt",
    "sacct.txt",
    "slurm-stdout.txt",
    "v4-hpg-verifier.py",
}
file_index_raw = (v4_abort_root / "files.sha256").read_bytes()
if hashlib.sha256(file_index_raw).hexdigest() != fields["V4_ABORT_FILES_INDEX_SHA256"]:
    raise SystemExit("H-APG cover gate error: v4 abort file-index digest mismatch")
indexed = {}
for line in file_index_raw.decode("ascii").splitlines():
    digest_token, name = line.split("  ", 1)
    if not re.fullmatch(r"[0-9a-f]{64}", digest_token) or name in indexed:
        raise SystemExit("H-APG cover gate error: malformed v4 abort file index")
    indexed[name] = digest_token
if set(indexed) != payload_names:
    raise SystemExit("H-APG cover gate error: v4 abort file-index set mismatch")
for name, expected_sha in indexed.items():
    if hashlib.sha256((v4_abort_root / name).read_bytes()).hexdigest() != expected_sha:
        raise SystemExit(f"H-APG cover gate error: v4 abort payload mismatch: {name}")

v4_sacct = (v4_abort_root / "sacct.txt").read_text(encoding="ascii")
if not v4_sacct.startswith("8455|") or "|FAILED|1:0|00:00:39|" not in v4_sacct:
    raise SystemExit("H-APG cover gate error: v4 abort sacct mismatch")
v4_stdout = (v4_abort_root / "slurm-stdout.txt").read_text(encoding="ascii")
if not v4_stdout.endswith(
    "RuntimeError: H-PG verification failed for U02-0000000000_S02-0000000000\n"
):
    raise SystemExit("H-APG cover gate error: v4 abort call path mismatch")
v4_config = parse_kv(v4_abort_root / "config.txt")
v4_config_expected = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-slurm-config.v2",
    "MODE": "adaptive",
    "BASE_REPO_BUNDLE_SHA256": fields["BASE_REPO_BUNDLE_SHA256"],
    "BASE_GIT_HEAD": fields["BASE_REPO_BUNDLE_GIT_HEAD"],
    "REPO_DELTA_BUNDLE_SHA256": fields["V4_ADAPTIVE_REPO_DELTA_BUNDLE_SHA256"],
    "PREBUILT_ARCHIVE_SHA256": fields["V4_ADAPTIVE_PREBUILT_ARCHIVE_SHA256"],
    "EXPECTED_GIT_HEAD": fields["V4_ADAPTIVE_ABORT_EXECUTED_GIT_HEAD"],
    "EXPECTED_CONTRACT_SHA256": fields["SUPERSEDES_V4_SHA256"],
    "OUTPUT_DIRECTORY": fields["V4_ADAPTIVE_OUTPUT_DIRECTORY"],
}
for key, value in v4_config_expected.items():
    if v4_config.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: v4 abort config mismatch for {key}")
audit_summary = parse_kv(v4_abort_root / "hpg-rc0-verifier-census-summary.txt")
summary_expected = {
    "SELECTION_CENSUS_SHA256": fields["V3_ABORT_HPG_FULL255_CENSUS_SHA256"],
    "SELECTION_RULE": fields["V5_HPG_RC0_SELECTION_RULE"],
    "SELECTION_ROW_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_NODE_COUNT"],
    "OLD_VERIFIER_SHA256": fields["V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256"],
    "V5_VERIFIER_SHA256": fields["PREPASS_VERIFIER_SHA256"],
    "V5_RC0_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_VERIFIER_RC0_COUNT"],
    "V5_PROBE_FALSE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_PROBE_FALSE_COUNT"],
    "V5_CERTIFICATE_FALSE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_CERTIFICATE_FALSE_COUNT"],
    "V5_SUBDIVISION_REQUIRED_TRUE_COUNT": fields["V5_HPG_RC0_DIAGNOSTIC_SUBDIVISION_TRUE_COUNT"],
    "V5_MUTATION_TESTS": fields["V5_HPG_RC0_DIAGNOSTIC_MUTATION_TESTS"],
    "V5_MUTATIONS_REJECTED": fields["V5_HPG_RC0_DIAGNOSTIC_MUTATIONS_REJECTED"],
    "KAT_V5_MUTATION_TESTS": fields["V5_KAT_HPG_MUTATION_TESTS"],
    "KAT_V5_MUTATIONS_REJECTED": fields["V5_KAT_HPG_MUTATIONS_REJECTED"],
}
for key, value in summary_expected.items():
    if audit_summary.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: v5 diagnostic summary mismatch for {key}")

v5_abort = parse_kv(v5_abort_root / "manifest.txt")
v5_abort_exact = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-v5-adaptive-abort.v1",
    "EVIDENCE_CLASS": "FRESH_KAT_PASS_AND_ADAPTIVE_POST_RUN_INDEX_GATE_FAILURE_NO_ADAPTIVE_ARCHIVE",
    "FROZEN_CONTRACT_SHA256": fields["SUPERSEDES_V5_SHA256"],
    "EXECUTED_GIT_HEAD": fields["IMPLEMENTATION_PARENT_COMMIT"],
    "BASE_REPO_BUNDLE_SHA256": fields["BASE_REPO_BUNDLE_SHA256"],
    "REPO_DELTA_BUNDLE_SHA256": "f18eca876a0dca56f3ba01ac32cac04f54e04dcc8df243cd49ef0f598f98de52",
    "PREBUILT_ARCHIVE_SHA256": "3f78b0d2e94534e7da4dd483c922aed54fbe2a3cb598a57b0ad862e7625c1688",
    "JOBS_SACCT_SHA256": fields["V5_ABORT_JOBS_SACCT_SHA256"],
    "KAT_JOB_ID": fields["V5_KAT_SLURM_JOB_ID"],
    "KAT_STATE": fields["V5_KAT_SLURM_STATE"],
    "KAT_EXIT_CODE": fields["V5_KAT_SLURM_EXIT_CODE"],
    "KAT_CONFIG_SHA256": fields["V5_KAT_CONFIG_SHA256"],
    "KAT_ARCHIVE_SHA256": fields["V5_KAT_ARCHIVE_SHA256"],
    "KAT_ARCHIVE_SIDECAR_SHA256": "15989fe6eece93c4da4054f23590a739521a1ed46ecd932a5e0ec770e1ba566c",
    "KAT_ARCHIVE_BYTES_RETAINED": "false",
    "KAT_ARCHIVE_SAFE_MEMBER_COUNT": "525",
    "KAT_RESULT_INDEXED_FILE_COUNT": "501",
    "KAT_TRANSPORT_MANIFEST_SHA256": "b71b75280451e2688792821a97d468aae4340464f344b77ebc6764fa49dabcb2",
    "KAT_RESULT_RUN_MANIFEST_SHA256": "0fbd95e37ea71bcd3fd8ff8b8eca29588ab1700c6f30aea3ca031cd16886a901",
    "KAT_RESULT_RUN_CONTRACT_SHA256": "13443e69e36104259e355f726e65365be6328b537a7d4f6db90806699a63f469",
    "KAT_RESULT_FILES_INDEX_SHA256": "6703897e2a23efb0633cf32c966a93680ba26bafe6b346b7992545ffa91a971e",
    "KAT_RESULT_SUMMARY_SHA256": "750d4400b8da0c755cd6627692ff561d8bea941fc7d205801704ebf178c87d55",
    "KAT_EVALUATED_NODE_COUNT": "53",
    "KAT_HPG_SIGNED_CHART_COUNT": "52",
    "KAT_HAPG_ATTEMPTED_COUNT": "52",
    "KAT_HAPG_CERTIFIED_COUNT": "48",
    "KAT_HAPG_RESCUE_COUNT": "20",
    "KAT_HPG_MUTATIONS": "4108/4108",
    "KAT_HAPG_MUTATIONS": "5824/5824",
    "KAT_AUTHORIZATION_SHA256": "7073311d87709fc0d583ca2b037280b374308371c8e2301c4fe0a1d4fe1fb61d",
    "FIRST_BARRIER_JOB_ID": "8460",
    "FIRST_BARRIER_STATE": "FAILED",
    "FIRST_BARRIER_FAILURE_CLASS": "SUBMISSION_MARKER_VISIBLE_BEFORE_BARRIER_GUARD",
    "FIRST_ADAPTIVE_JOB_ID": "8461",
    "FIRST_ADAPTIVE_STATE": "CANCELLED",
    "FIRST_ADAPTIVE_STARTED": "false",
    "SUCCESSFUL_BARRIER_JOB_ID": "8462",
    "SUCCESSFUL_BARRIER_STATE": "COMPLETED",
    "SUCCESSFUL_BARRIER_EXIT_CODE": "0:0",
    "SUCCESSFUL_BARRIER_ATTESTATION_SHA256": "d593a8153b00b6a9a26dbaa17ad4dfb086549d521b7829c59686b31cae24b1b0",
    "ADAPTIVE_JOB_ID": fields["V5_ADAPTIVE_SLURM_JOB_ID"],
    "ADAPTIVE_STATE": fields["V5_ADAPTIVE_SLURM_STATE"],
    "ADAPTIVE_EXIT_CODE": fields["V5_ADAPTIVE_SLURM_EXIT_CODE"],
    "ADAPTIVE_CONFIG_SHA256": "2154027e1fcea4e5e21375427355fce045a20da4975008f9f710291c310f94ce",
    "ADAPTIVE_SUBMISSION_SHA256": "a8c7ef8681ddecb1e073a3fc4f6ed921f1a6ff53b0fa82d026020e02b89c444c",
    "ADAPTIVE_SUBMIT_UTC": "2026-08-02T06:46:14",
    "ADAPTIVE_START_UTC": "2026-08-02T06:46:35",
    "ADAPTIVE_END_UTC": "2026-08-02T06:48:08",
    "ADAPTIVE_ALLOCATED_CPUS": "120",
    "ADAPTIVE_REQUESTED_CPUS": "32",
    "ADAPTIVE_RUNNER_REPORTED_EVALUATED_NODE_COUNT": fields[
        "V5_ADAPTIVE_RUNNER_REPORTED_EVALUATED_NODE_COUNT"
    ],
    "ADAPTIVE_RUNNER_REPORTED_WAVE_COUNT": fields[
        "V5_ADAPTIVE_RUNNER_REPORTED_WAVE_COUNT"
    ],
    "FAILURE_STAGE": fields["V5_ADAPTIVE_FAILURE_STAGE"],
    "FAILURE_SIGNATURE": fields["V5_ADAPTIVE_FAILURE_SIGNATURE"],
    "ROOT_CAUSE": "PATH_COMPONENT_ORDER_DIFFERS_FROM_POSIX_RELATIVE_TOKEN_ORDER_FOR_FRESH_REPLAY_SIBLINGS",
    "LOCALE_DEPENDENT": "false",
    "ADAPTIVE_RUNNER_RESULT_SCRATCH_RETAINED": "false",
    "ADAPTIVE_AGGREGATION_PUBLISHED": "false",
    "ADAPTIVE_TRANSPORT_ARCHIVE_PUBLISHED": fields["V5_ADAPTIVE_ARCHIVE_PUBLISHED"],
    "ADAPTIVE_RESULT_AVAILABLE": fields["V5_ADAPTIVE_RESULT_AVAILABLE"],
    "ADAPTIVE_RUNNER_CANDIDATE_FIELD_IS_SCIENTIFIC_RESULT": fields[
        "V5_ADAPTIVE_RUNNER_CANDIDATE_FIELD_IS_SCIENTIFIC_RESULT"
    ],
    "FILE_INDEX_ORDER_FIX": fields["V6_FILE_INDEX_ORDER"],
    "FILE_INDEX_REPRO_SHA256": "e22e20457c82088d9f49e5dd8a0019676299020589ec7e704ecb106db4af0165",
    "FILES_INDEX_SHA256": fields["V5_ABORT_FILES_INDEX_SHA256"],
    "FILE_COUNT": "33",
    "EXECUTION_PROVENANCE_ATTESTED": "false",
    "OPEN_PROBLEM_SOLVED": "false",
    "PROMOTION_ELIGIBLE": "false",
}
if set(v5_abort) != set(v5_abort_exact):
    missing = sorted(set(v5_abort_exact) - set(v5_abort))
    extra = sorted(set(v5_abort) - set(v5_abort_exact))
    raise SystemExit(
        f"H-APG cover gate error: v5 abort manifest key-set mismatch: missing={missing} extra={extra}"
    )
for key, value in v5_abort_exact.items():
    if v5_abort.get(key) != value:
        raise SystemExit(f"H-APG cover gate error: v5 abort receipt mismatch for {key}")

v5_payload_names = {
    "adaptive-config.txt",
    "adaptive-submission-attempt-8461-dependency-failed.txt",
    "adaptive-submission.txt",
    "adaptive-submit-guard.py",
    "file-index-order-repro.txt",
    "jobs-8458-8463.sacct.psv",
    "kat-archive.sha256",
    "kat-auth-barrier-job.sh",
    "kat-auth-barrier-job8462.txt",
    "kat-authorization.txt",
    "kat-config.txt",
    "kat-release-guard.py",
    "kat-result-files.sha256",
    "kat-run-contract.txt",
    "kat-run-manifest.txt",
    "kat-summary.txt",
    "kat-transport-manifest.txt",
    "kat-transport-slurm-job-record.txt",
    "logs/adaptive-8461.pre-cancel-scontrol.txt",
    "logs/adaptive-8463.err",
    "logs/adaptive-8463.final-scontrol.txt",
    "logs/adaptive-8463.out",
    "logs/kat-8458.err",
    "logs/kat-8458.out",
    "logs/kat-auth-barrier-8460.err",
    "logs/kat-auth-barrier-8460.final-scontrol.txt",
    "logs/kat-auth-barrier-8460.out",
    "logs/kat-auth-barrier-8462.err",
    "logs/kat-auth-barrier-8462.final-scontrol.txt",
    "logs/kat-auth-barrier-8462.guard.txt",
    "logs/kat-auth-barrier-8462.out",
    "logs/kat-auth-barrier-8462.scontrol.txt",
    "submit-adaptive-once.sh",
}
v5_index_raw = (v5_abort_root / "files.sha256").read_bytes()
if (
    not v5_index_raw.endswith(b"\n")
    or b"\r" in v5_index_raw
    or b"\0" in v5_index_raw
    or hashlib.sha256(v5_index_raw).hexdigest() != fields["V5_ABORT_FILES_INDEX_SHA256"]
):
    raise SystemExit("H-APG cover gate error: v5 abort file index envelope mismatch")
v5_indexed: dict[str, str] = {}
for line in v5_index_raw.decode("ascii").splitlines():
    if line.count("  ") != 1:
        raise SystemExit("H-APG cover gate error: malformed v5 abort file index")
    digest_token, name = line.split("  ", 1)
    pure = PurePosixPath(name)
    if (
        re.fullmatch(r"[0-9a-f]{64}", digest_token) is None
        or pure.is_absolute()
        or not pure.parts
        or ".." in pure.parts
        or name != pure.as_posix()
        or name in v5_indexed
    ):
        raise SystemExit("H-APG cover gate error: unsafe v5 abort file-index row")
    v5_indexed[name] = digest_token
if list(v5_indexed) != sorted(v5_indexed) or set(v5_indexed) != v5_payload_names:
    raise SystemExit("H-APG cover gate error: v5 abort file-index order or set mismatch")
v5_tree = list(v5_abort_root.rglob("*"))
if any(path.is_symlink() for path in v5_tree):
    raise SystemExit("H-APG cover gate error: symlink in v5 abort receipt")
actual_v5_payloads = {
    relative
    for file in v5_tree
    if file.is_file()
    for relative in (file.relative_to(v5_abort_root).as_posix(),)
    if relative not in {"manifest.txt", "files.sha256"}
}
if actual_v5_payloads != v5_payload_names:
    raise SystemExit("H-APG cover gate error: v5 abort unindexed payload set mismatch")
for name, expected_sha in v5_indexed.items():
    payload = v5_abort_root.joinpath(*PurePosixPath(name).parts)
    if payload.is_symlink() or hashlib.sha256(payload.read_bytes()).hexdigest() != expected_sha:
        raise SystemExit(f"H-APG cover gate error: v5 abort payload mismatch: {name}")

v5_jobs = {}
for row in (v5_abort_root / "jobs-8458-8463.sacct.psv").read_text(
    encoding="ascii"
).splitlines():
    columns = row.split("|")
    if len(columns) != 13 or columns[-1] != "" or columns[0] in v5_jobs:
        raise SystemExit("H-APG cover gate error: malformed v5 abort sacct row")
    v5_jobs[columns[0]] = columns
expected_job_states = {
    "8458": ("COMPLETED", "0:0"),
    "8460": ("FAILED", "1:0"),
    "8461": ("CANCELLED by 0", "0:0"),
    "8462": ("COMPLETED", "0:0"),
    "8463": ("FAILED", "1:0"),
}
expected_job_names = {
    "8458": "cs6-hapg-kat-v5",
    "8460": "cs6-hapg-kat-auth-v5",
    "8461": "cs6-hapg-adaptive-v5",
    "8462": "cs6-hapg-kat-auth-v5",
    "8463": "cs6-hapg-adaptive-v5",
}
if set(v5_jobs) != set(expected_job_states) or any(
    (v5_jobs[job_id][3], v5_jobs[job_id][4]) != state_exit
    for job_id, state_exit in expected_job_states.items()
) or any(
    v5_jobs[job_id][1] != expected_job_names[job_id]
    or v5_jobs[job_id][2] != "gpu-orangefs"
    for job_id in expected_job_names
):
    raise SystemExit("H-APG cover gate error: v5 abort Slurm state mismatch")
if (
    v5_jobs["8461"][6] != "None"
    or v5_jobs["8458"][5:8]
    != ["2026-08-02T06:28:07", "2026-08-02T06:28:07", "2026-08-02T06:29:38"]
    or v5_jobs["8463"][5:8]
    != [
        v5_abort["ADAPTIVE_SUBMIT_UTC"],
        v5_abort["ADAPTIVE_START_UTC"],
        v5_abort["ADAPTIVE_END_UTC"],
    ]
    or v5_jobs["8458"][9:12] != ["gpuorangefs-r770-proxmox", "120", "32"]
    or v5_jobs["8463"][9:12]
    != ["gpuorangefs-r770-proxmox", v5_abort["ADAPTIVE_ALLOCATED_CPUS"], v5_abort["ADAPTIVE_REQUESTED_CPUS"]]
    or any(
        not row[5] <= row[6] <= row[7]
        for job_id, row in v5_jobs.items()
        if job_id != "8461"
    )
    or not v5_jobs["8458"][7] <= v5_abort["ADAPTIVE_SUBMIT_UTC"]
    or not v5_jobs["8462"][7] <= v5_abort["ADAPTIVE_START_UTC"]
):
    raise SystemExit("H-APG cover gate error: v5 abort chronology mismatch")
v5_stdout = (v5_abort_root / "logs/adaptive-8463.out").read_text(encoding="ascii")
v5_stderr = (v5_abort_root / "logs/adaptive-8463.err").read_text(encoding="ascii")
if (
    "EVALUATED_NODE_COUNT=255\n" not in v5_stdout
    or "WAVE_COUNT=8\n" not in v5_stdout
    or "HAPG_FULL_SOURCE_COVER_CANDIDATE=false\n" not in v5_stdout
    or not v5_stderr.endswith("aggregation error: files index is not sorted\n")
):
    raise SystemExit("H-APG cover gate error: v5 adaptive failure signature mismatch")
repro = parse_kv(v5_abort_root / "file-index-order-repro.txt")
repro_exact = {
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-file-index-order-repro.v1",
    "FAILED_JOB_ID": fields["V5_ADAPTIVE_SLURM_JOB_ID"],
    "PRODUCER_ORDER_1": "fresh-replay/evaluations.tsv",
    "PRODUCER_ORDER_2": "fresh-replay-contract.txt",
    "PRODUCER_ORDER_3": "fresh-replay-terminals.tsv",
    "REQUIRED_ORDER_1": "fresh-replay-contract.txt",
    "REQUIRED_ORDER_2": "fresh-replay-terminals.tsv",
    "REQUIRED_ORDER_3": "fresh-replay/evaluations.tsv",
    "ROOT_CAUSE": "PATH_COMPONENT_ORDER_DIFFERS_FROM_POSIX_RELATIVE_TOKEN_ORDER",
    "LOCALE_DEPENDENT": "false",
    "MINIMAL_PRODUCER_CONSUMER_REPRO_PASS_AFTER_FIX": "true",
    "SCIENTIFIC_RESULT": "false",
    "PROMOTION_ELIGIBLE": "false",
}
if repro != repro_exact:
    raise SystemExit("H-APG cover gate error: v5 file-index repro mismatch")

false_fields = (
    "ABSTRACT_EXISTENCE_FALSIFIED_BY_BOUNDED_FAILURE",
    "GENERIC_CERTIFICATE_PASS_TERMINAL_ALLOWED",
    "AFFINE_TERMINAL_ALLOWED",
    "FIXED_PROJECTIVE_TERMINAL_ALLOWED",
    "BOXED_HOMOGENEOUS_TERMINAL_ALLOWED",
    "H_APG_RUNTIME_CHART_RESELECTION",
    "FPGA_EXECUTION",
    "U250_REQUIRED",
    "EXECUTION_PROVENANCE_ATTESTED",
    "FULL_SOURCE_CARRIER_PROVED",
    "INVARIANCE_PROVED",
    "CONE_FIELD_PROVED",
    "DOMINATED_SPLITTING_PROVED",
    "HYPERBOLICITY_PROVED",
    "CHAOTIC_ATTRACTOR_PROVED",
    "OPEN_PROBLEM_SOLVED",
    "NOVELTY_OR_PRIORITY_CLAIM",
    "PROMOTION_ELIGIBLE",
)
for key in false_fields:
    if fields.get(key) != "false":
        raise SystemExit(f"H-APG cover gate error: anti-promotion field is not false: {key}")
PY

expected=$(awk -F= '$1 == "GATE_SHA256" {print $2}' "$contract")
actual=$(sha256sum "$0" | awk '{print $1}')
[[ -n $expected && $actual == "$expected" ]] || {
  echo "H-APG cover gate error: frozen gate digest mismatch" >&2
  exit 1
}

check_binding() {
  local key=$1
  local path=$2
  local expected_sha
  local actual_sha
  expected_sha=$(awk -F= -v key="$key" '$1 == key {print $2}' "$contract")
  actual_sha=$(sha256sum "$path" | awk '{print $1}')
  [[ -n $expected_sha && $actual_sha == "$expected_sha" ]] || {
    echo "H-APG cover gate error: frozen source mismatch for $key" >&2
    exit 1
  }
}

check_binding PREPASS_WORKER_SHA256 scripts/research/cs6_plucker_cocycle_probe.cpp
check_binding PREPASS_VERIFIER_SHA256 scripts/research/cs6_plucker_cocycle_verify.py
check_binding H_APG_WRAPPER_SHA256 "$wrapper"
check_binding H_APG_KERNEL_SHA256 scripts/research/cs6_affine_projective_cocycle_full53_probe.cpp
check_binding H_APG_ADAPTER_SHA256 "$leaf_verifier"
check_binding H_APG_NUMERIC_VERIFIER_SHA256 scripts/research/cs6_affine_projective_cocycle_full53_verify.py
check_binding RUNNER_SHA256 "$runner"
check_binding AGGREGATOR_SHA256 "$aggregator"
check_binding KAT_ANCHOR_SHA256 "$kat_anchor"
check_binding EXACT_TREE_KERNEL_SHA256 scripts/research/cs6_c1_full_source_cover_aggregate.py
check_binding SLURM_JOB_SCRIPT_SHA256 "$slurm_job"
check_binding KAT_COORDINATE_MANIFEST_SHA256 scripts/research/cs6_affine_projective_cocycle_full53_coordinates_v1.tsv
check_binding KAT_EXPECTED_RESULTS_SHA256 "$full53/leaves.tsv"
check_binding V2_ABORT_RECEIPT_MANIFEST_SHA256 "$v2_abort/manifest.txt"
check_binding V2_ABORT_SACCT_SHA256 "$v2_abort/sacct.txt"
check_binding V2_ABORT_CONFIG_SHA256 "$v2_abort/config.txt"
check_binding V2_ABORT_STDERR_SHA256 "$v2_abort/stderr.txt"
check_binding V3_ABORT_RECEIPT_MANIFEST_SHA256 "$v3_abort/manifest.txt"
check_binding V3_ABORT_SACCT_SHA256 "$v3_abort/sacct.txt"
check_binding V3_ABORT_CONFIG_SHA256 "$v3_abort/config.txt"
check_binding V3_ABORT_SLURM_STDERR_SHA256 "$v3_abort/slurm-stderr.txt"
check_binding V3_ABORT_REPRO_S0_STDOUT_SHA256 "$v3_abort/repro-s0-stdout.txt"
check_binding V3_ABORT_REPRO_S0_STDERR_SHA256 "$v3_abort/repro-s0-stderr.txt"
check_binding V3_ABORT_REPRO_S1_STDOUT_SHA256 "$v3_abort/repro-s1-stdout.txt"
check_binding V3_ABORT_REPRO_S1_STDERR_SHA256 "$v3_abort/repro-s1-stderr.txt"
check_binding V3_ABORT_HPG_FULL255_CENSUS_SHA256 "$v3_abort/hpg-full255-census.tsv"
check_binding V3_ABORT_HPG_FULL255_CENSUS_SUMMARY_SHA256 "$v3_abort/hpg-full255-census-summary.txt"
check_binding V3_ABORT_HPG_FULL255_STDERR_JSONL_SHA256 "$v3_abort/hpg-full255-stderr.jsonl"
check_binding V3_ABORT_HPG_CHALLENGE_SPOTCHECK_SHA256 "$v3_abort/challenge-spotcheck.json"
check_binding V4_ABORT_RECEIPT_MANIFEST_SHA256 "$v4_abort/manifest.txt"
check_binding V4_ABORT_FILES_INDEX_SHA256 "$v4_abort/files.sha256"
check_binding V4_ABORT_SACCT_SHA256 "$v4_abort/sacct.txt"
check_binding V4_ABORT_CONFIG_SHA256 "$v4_abort/config.txt"
check_binding V4_ABORT_SLURM_STDOUT_SHA256 "$v4_abort/slurm-stdout.txt"
check_binding V4_ABORT_HPG_RC0_CORPUS_SHA256 "$v4_abort/hpg-rc0-corpus.tar"
check_binding V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256 "$v4_abort/corpus-files.sha256"
check_binding V4_ABORT_HPG_RC0_CENSUS_SHA256 "$v4_abort/hpg-rc0-verifier-census.tsv"
check_binding V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256 "$v4_abort/hpg-rc0-verifier-census-summary.txt"
check_binding V4_ABORT_HPG_V5_KAT_COMPAT_SHA256 "$v4_abort/hpg-v5-kat-compat.tsv"
check_binding V4_ABORT_HPG_V4_KAT_CORPUS_SHA256 "$v4_abort/hpg-v4-kat-corpus.tar"
check_binding V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256 "$v4_abort/hpg-v4-kat-corpus-files.sha256"
check_binding V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256 "$v4_abort/midpoint-discrete-negative-test.txt"
check_binding V4_ABORT_LOCAL_REPRO_SHA256 "$v4_abort/local-repro.tar"
check_binding V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256 "$v4_abort/v4-hpg-verifier.py"
check_binding V5_ABORT_RECEIPT_MANIFEST_SHA256 "$v5_abort/manifest.txt"
check_binding V5_ABORT_FILES_INDEX_SHA256 "$v5_abort/files.sha256"
check_binding V5_ABORT_JOBS_SACCT_SHA256 "$v5_abort/jobs-8458-8463.sacct.psv"
check_binding SUPERSEDES_V5_SHA256 "$v5_contract"
check_binding SUPERSEDES_V4_SHA256 "$v4_contract"
check_binding SUPERSEDES_V3_SHA256 "$v3_contract"

python3 -B - "$runner" "$leaf_verifier" "$aggregator" "$kat_anchor" <<'PY'
from pathlib import Path
import sys

for token in sys.argv[1:]:
    path = Path(token)
    compile(path.read_bytes(), str(path), "exec")
PY

python3 -B - "$v4_abort" "$v3_abort/hpg-full255-census.tsv" "$contract" \
  scripts/research/cs6_plucker_cocycle_verify.py "$runner" "$v4_contract" <<'PY'
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import csv
from fractions import Fraction
import hashlib
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import ModuleType

if sys.flags.optimize:
    raise SystemExit("H-APG cover gate error: Python optimization is forbidden")

abort_root = Path(sys.argv[1])
selection_census_path = Path(sys.argv[2])
contract_path = Path(sys.argv[3])
v5_verifier = Path(sys.argv[4])
runner_path = Path(sys.argv[5])
v4_contract_path = Path(sys.argv[6])
v4_verifier = abort_root / "v4-hpg-verifier.py"


def parse_kv_bytes(raw: bytes, label: str) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise AssertionError(f"noncanonical {label}")
    result: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        if line.count("=") != 1:
            raise AssertionError(f"malformed {label}")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise AssertionError(f"duplicate or empty {label} field")
        result[key] = value
    return result


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_unpack(archive_path: Path, destination: Path) -> set[str]:
    names: set[str] = set()
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
                or member.name in names
                or not (member.isdir() or member.isfile())
            ):
                raise AssertionError(f"unsafe tar member: {member.name}")
            names.add(member.name)
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                stream = archive.extractfile(member)
                if stream is None:
                    raise AssertionError(f"missing tar payload: {member.name}")
                target.write_bytes(stream.read())
    return names


def read_tsv(
    path: Path, expected_fields: list[str] | None = None
) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise AssertionError(f"noncanonical TSV: {path.name}")
    reader = csv.DictReader(raw.decode("ascii").splitlines(), delimiter="\t")
    if expected_fields is not None and reader.fieldnames != expected_fields:
        raise AssertionError(f"unexpected TSV header: {path.name}")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise AssertionError(f"malformed TSV row: {path.name}")
    return rows


contract = parse_kv_bytes(contract_path.read_bytes(), "v5 contract")
v4_frozen_contract_raw = v4_contract_path.read_bytes()
v4_frozen_contract = parse_kv_bytes(v4_frozen_contract_raw, "v4 contract")
assert sha(v4_frozen_contract_raw) == contract["SUPERSEDES_V4_SHA256"]
audit_rows = read_tsv(abort_root / "hpg-rc0-verifier-census.tsv")
assert len(audit_rows) == 54
assert [row["NODE_ID"] for row in audit_rows] == sorted(
    row["NODE_ID"] for row in audit_rows
)
selection_rows = read_tsv(selection_census_path)
selected_ids = {row["NODE_ID"] for row in selection_rows if row["RC"] == "0"}
assert len(selection_rows) == 255 and len(selected_ids) == 54
assert selected_ids == {row["NODE_ID"] for row in audit_rows}

with tempfile.TemporaryDirectory(prefix="cs6-hapg-v5-gate.") as temporary:
    temporary_path = Path(temporary)
    corpus_names = safe_unpack(
        abort_root / "hpg-rc0-corpus.tar", temporary_path / "corpus"
    )
    corpus_root = temporary_path / "corpus/hpg-rc0-corpus"
    expected_members = {"hpg-rc0-corpus"}
    for identity in selected_ids:
        expected_members.update(
            {
                f"hpg-rc0-corpus/{identity}",
                f"hpg-rc0-corpus/{identity}/input.txt",
                f"hpg-rc0-corpus/{identity}/receipt.txt",
            }
        )
    assert corpus_names == expected_members and len(corpus_names) == 163

    corpus_index: dict[str, str] = {}
    index_raw = (abort_root / "corpus-files.sha256").read_bytes()
    assert sha(index_raw) == contract["V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256"]
    for line in index_raw.decode("ascii").splitlines():
        digest_token, name = line.split("  ", 1)
        assert len(digest_token) == 64 and name not in corpus_index
        corpus_index[name] = digest_token
    expected_files = {
        f"{identity}/{filename}"
        for identity in selected_ids
        for filename in ("input.txt", "receipt.txt")
    }
    assert set(corpus_index) == expected_files and len(corpus_index) == 108
    for name, expected_sha in corpus_index.items():
        assert sha((corpus_root / name).read_bytes()) == expected_sha

    def run_verifier(verifier: Path, row: dict[str, str], mutations: bool):
        identity = row["NODE_ID"]
        command = [
            sys.executable,
            "-B",
            str(verifier),
            str(corpus_root / identity / "receipt.txt"),
            "--source-sha",
            contract["PREPASS_WORKER_SHA256"],
            "--input",
            str(corpus_root / identity / "input.txt"),
            "--challenge",
            row["RUN_CHALLENGE"],
        ]
        if mutations:
            command.append("--self-test-mutations")
        return subprocess.run(command, capture_output=True, timeout=180)

    def replay(row: dict[str, str]) -> tuple[str, int, int]:
        identity = row["NODE_ID"]
        input_raw = (corpus_root / identity / "input.txt").read_bytes()
        receipt_raw = (corpus_root / identity / "receipt.txt").read_bytes()
        assert sha(input_raw) == row["INPUT_SHA256"]
        assert sha(receipt_raw) == row["RECEIPT_SHA256"]
        assert row["HPG_WORKER_BINARY_SHA256"] == contract["PREBUILT_HPG_BINARY_SHA256"]
        assert row["SELECTION_CENSUS_SHA256"] == contract["V3_ABORT_HPG_FULL255_CENSUS_SHA256"]
        assert row["ORIGINAL_WORKER_SHA_MATCH"] == "true"

        old = run_verifier(v4_verifier, row, False)
        assert str(old.returncode) == row["OLD_RC"]
        assert (old.stderr == b"") == (row["OLD_STDERR_EMPTY"] == "true")
        assert sha(old.stderr) == row["OLD_STDERR_SHA256"]
        assert sha(old.stdout) == row["OLD_VERIFICATION_SHA256"]
        if old.returncode == 0:
            old_outcome = "PASS"
        elif old.stderr == b"verification error: local P2 event order or Plus-side witness failed\n":
            old_outcome = "EVENT_ORDER_OR_PLUS_SIDE"
        elif old.stderr.startswith(b"verification error: midpoint reconstruction mismatch:"):
            old_outcome = "MIDPOINT_RECONSTRUCTION"
        else:
            raise AssertionError(f"unexpected old verifier outcome: {identity}")
        assert old_outcome == row["OLD_OUTCOME"]

        plain = run_verifier(v5_verifier, row, False)
        mutated = run_verifier(v5_verifier, row, True)
        assert plain.returncode == mutated.returncode == 0
        assert plain.stderr == mutated.stderr == b""
        assert sha(plain.stdout) == row["V5_PLAIN_VERIFICATION_SHA256"]
        assert sha(mutated.stdout) == row["V5_MUTATION_VERIFICATION_SHA256"]
        plain_values = parse_kv_bytes(plain.stdout, "plain H-PG verification")
        mutation_values = parse_kv_bytes(mutated.stdout, "mutated H-PG verification")
        assert plain_values["RECEIPT_SHA256"] == row["RECEIPT_SHA256"]
        assert plain_values["PROBE_PASS"] == row["V5_PROBE_PASS"] == "false"
        assert plain_values["CERTIFICATE_PASS"] == row["V5_CERTIFICATE_PASS"] == "false"
        assert plain_values["SUBDIVISION_REQUIRED"] == row["V5_SUBDIVISION_REQUIRED"] == "true"
        assert row["V5_STRUCTURAL_PASS"] == "true"
        tests = int(mutation_values["MUTATION_TESTS"])
        rejected = int(mutation_values["MUTATIONS_REJECTED"])
        assert tests == int(row["V5_MUTATION_TESTS"]) == 79
        assert rejected == int(row["V5_MUTATIONS_REJECTED"]) == tests
        return old_outcome, tests, rejected

    with ThreadPoolExecutor(max_workers=16) as executor:
        replayed = list(executor.map(replay, audit_rows))
    outcomes = [outcome for outcome, _, _ in replayed]
    assert outcomes.count("PASS") == 25
    assert outcomes.count("EVENT_ORDER_OR_PLUS_SIDE") == 11
    assert outcomes.count("MIDPOINT_RECONSTRUCTION") == 18
    assert sum(tests for _, tests, _ in replayed) == 4266
    assert sum(rejected for _, _, rejected in replayed) == 4266

    local_names = safe_unpack(
        abort_root / "local-repro.tar", temporary_path / "local"
    )
    local_root = temporary_path / "local"
    assert len(local_names) == 121
    input_ids = {path.stem for path in (local_root / "inputs").glob("*.txt")}
    receipt_ids = {path.stem for path in (local_root / "hpg-receipts").glob("*.txt")}
    base_stderr_ids = {
        path.stem for path in (local_root / "hpg-stderr").glob("*.txt")
        if not path.name.endswith(".verifier.txt")
    }
    verifier_stderr_ids = {
        path.name.removesuffix(".verifier.txt")
        for path in (local_root / "hpg-stderr").glob("*.verifier.txt")
    }
    assert input_ids == receipt_ids == base_stderr_ids and len(input_ids) == 31
    expected_local_members = {
        "inputs",
        "hpg-receipts",
        "hpg-stderr",
        "wave-contracts",
        "wave-results",
        "run-contract.txt",
        *(f"inputs/{identity}.txt" for identity in input_ids),
        *(f"hpg-receipts/{identity}.txt" for identity in receipt_ids),
        *(f"hpg-stderr/{identity}.txt" for identity in base_stderr_ids),
        *(f"hpg-stderr/{identity}.verifier.txt" for identity in verifier_stderr_ids),
        *(f"wave-contracts/W{index:04d}.tsv" for index in range(4)),
        *(f"wave-results/W{index:04d}.tsv" for index in range(4)),
    }
    assert local_names == expected_local_members

    runner_raw = runner_path.read_bytes()
    runner = ModuleType("cs6_hapg_v5_gate_runner")
    runner.__file__ = str(runner_path.resolve())
    sys.modules[runner.__name__] = runner
    exec(compile(runner_raw, str(runner_path), "exec"), runner.__dict__)
    aggregate_path = runner_path.with_name("cs6_hapg_full_source_cover_aggregate.py")
    aggregate_raw = aggregate_path.read_bytes()
    aggregate = ModuleType("cs6_hapg_v5_gate_aggregate")
    aggregate.__file__ = str(aggregate_path.resolve())
    sys.modules[aggregate.__name__] = aggregate
    exec(compile(aggregate_raw, str(aggregate_path), "exec"), aggregate.__dict__)
    local_run_contract_raw = (local_root / "run-contract.txt").read_bytes()
    local_run_contract = parse_kv_bytes(local_run_contract_raw, "local run contract")
    local_run_contract_sha = sha(local_run_contract_raw)
    assert local_run_contract["FROZEN_CONTRACT_SHA256"] == contract["SUPERSEDES_V4_SHA256"]
    assert local_run_contract["MODE"] == "adaptive"
    assert local_run_contract["ROOT_CHALLENGE"] == contract["BOUNDED_PILOT_ROOT_CHALLENGE"]
    local_source_bindings = {
        "HPG_WORKER_SOURCE_SHA256": "PREPASS_WORKER_SHA256",
        "HPG_VERIFIER_SOURCE_SHA256": "PREPASS_VERIFIER_SHA256",
        "HAPG_WORKER_SOURCE_SHA256": "H_APG_WRAPPER_SHA256",
        "HAPG_KERNEL_SOURCE_SHA256": "H_APG_KERNEL_SHA256",
        "HAPG_VERIFIER_ADAPTER_SHA256": "H_APG_ADAPTER_SHA256",
        "HAPG_NUMERIC_VERIFIER_SHA256": "H_APG_NUMERIC_VERIFIER_SHA256",
        "SLURM_JOB_SCRIPT_SHA256": "SLURM_JOB_SCRIPT_SHA256",
    }
    assert all(
        local_run_contract[run_key] == v4_frozen_contract[contract_key]
        for run_key, contract_key in local_source_bindings.items()
    )

    previous_result_sha = "0" * 64
    published_ids: set[str] = set()
    frontier = [runner.Leaf(0, 0, 0, 0, "-", 0)]
    for wave_index, expected_count in enumerate((1, 2, 4, 8)):
        contract_path_local = local_root / f"wave-contracts/W{wave_index:04d}.tsv"
        result_path_local = local_root / f"wave-results/W{wave_index:04d}.tsv"
        parsed_wave = runner.VERIFY.parse_wave_contract(contract_path_local)
        parsed_result = aggregate.parse_wave_result(result_path_local)
        contract_raw = contract_path_local.read_bytes()
        expected_ids = {leaf.identity for leaf in frontier}
        assert len(frontier) == expected_count
        assert set(parsed_wave.rows) == set(parsed_result.rows) == expected_ids
        assert parsed_wave.headers["WAVE_INDEX"] == parsed_result.headers["WAVE_INDEX"] == str(wave_index)
        assert parsed_wave.headers["NODE_COUNT"] == parsed_result.headers["NODE_COUNT"] == str(expected_count)
        assert parsed_wave.headers["RUN_CONTRACT_SHA256"] == local_run_contract_sha
        assert parsed_wave.headers["ROOT_CHALLENGE"] == local_run_contract["ROOT_CHALLENGE"]
        assert all(
            parsed_wave.headers[key] == local_run_contract[key]
            for key in local_source_bindings
            if key != "SLURM_JOB_SCRIPT_SHA256"
        )
        assert parsed_wave.headers["PREVIOUS_WAVE_RESULT_SHA256"] == previous_result_sha
        assert parsed_result.headers["WAVE_CONTRACT_SHA256"] == sha(contract_raw)
        next_frontier = []
        for leaf in frontier:
            identity = leaf.identity
            wave_row = parsed_wave.rows[identity].values
            result_row = parsed_result.rows[identity]
            input_raw = (local_root / f"inputs/{identity}.txt").read_bytes()
            receipt_raw = (local_root / f"hpg-receipts/{identity}.txt").read_bytes()
            stderr_raw = (local_root / f"hpg-stderr/{identity}.txt").read_bytes()
            expected_decision, children = runner.split_leaf(leaf)
            assert input_raw == runner.leaf_input_bytes(leaf)
            assert wave_row["INPUT_SHA256"] == sha(input_raw)
            assert wave_row["HPG_RECEIPT_SHA256"] == sha(receipt_raw)
            assert wave_row["HPG_STDERR_SHA256"] == sha(stderr_raw)
            assert int(wave_row["HPG_RC"]) != 0
            assert runner.classify_worker_failure(stderr_raw, "H_PG") == wave_row["HPG_STATUS"]
            assert result_row["HPG_STATUS"] == wave_row["HPG_STATUS"]
            assert result_row["HAPG_ATTEMPTED"] == "false"
            assert result_row["HAPG_STATUS"] == "H_APG_NOT_ELIGIBLE"
            assert result_row["HAPG_RC"] == "0"
            assert result_row["HAPG_CHALLENGE"] == aggregate.ZERO_SHA256
            assert result_row["HAPG_RECEIPT_SHA256"] == aggregate.EMPTY_SHA256
            assert result_row["HAPG_STDERR_SHA256"] == aggregate.EMPTY_SHA256
            assert result_row["HAPG_VERIFICATION_SHA256"] == aggregate.ZERO_SHA256
            assert result_row["HAPG_PHYSICAL_SHA256"] == aggregate.ZERO_SHA256
            assert all(
                result_row[key] == "false" for key in aggregate.RESULT_COLUMNS[11:22]
            )
            assert result_row["DECISION"] == expected_decision
            assert result_row["TERMINAL_REASON"] == "-"
            next_frontier.extend(children)
        expected_next_sha = sha(runner.frontier_bytes(next_frontier))
        assert parsed_result.headers["NEXT_FRONTIER_SHA256"] == expected_next_sha
        published_ids.update(expected_ids)
        previous_result_sha = parsed_result.sha256
        frontier = next_frontier
        if wave_index < 3:
            next_wave = runner.VERIFY.parse_wave_contract(
                local_root / f"wave-contracts/W{wave_index + 1:04d}.tsv"
            )
            assert next_wave.headers["FRONTIER_SHA256"] == expected_next_sha
    assert len(published_ids) == 15
    wave4_ids = input_ids - published_ids
    assert wave4_ids == {leaf.identity for leaf in frontier} and len(wave4_ids) == 16
    wave4_frontier_sha = sha(runner.frontier_bytes(frontier))
    assert all(
        (local_root / f"inputs/{leaf.identity}.txt").read_bytes()
        == runner.leaf_input_bytes(leaf)
        for leaf in frontier
    )
    rc0_ids = {
        identity
        for identity in wave4_ids
        if (local_root / f"hpg-receipts/{identity}.txt").stat().st_size > 0
    }
    failure_ids = wave4_ids - rc0_ids
    assert len(rc0_ids) == len(verifier_stderr_ids) == 14
    assert rc0_ids == verifier_stderr_ids and len(failure_ids) == 2
    for identity in rc0_ids:
        receipt = local_root / f"hpg-receipts/{identity}.txt"
        input_path = local_root / f"inputs/{identity}.txt"
        assert (local_root / f"hpg-stderr/{identity}.txt").read_bytes() == b""
        challenge = next(
            line.split("=", 1)[1]
            for line in receipt.read_text(encoding="ascii").splitlines()
            if line.startswith("RUN_CHALLENGE=")
        )
        assert challenge == runner.VERIFY.hpg_leaf_challenge(
            local_run_contract["ROOT_CHALLENGE"],
            4,
            previous_result_sha,
            wave4_frontier_sha,
            identity,
            sha(input_path.read_bytes()),
        )
        replay_result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(v4_verifier),
                str(receipt),
                "--source-sha",
                contract["PREPASS_WORKER_SHA256"],
                "--input",
                str(input_path),
                "--challenge",
                challenge,
            ],
            capture_output=True,
            timeout=120,
        )
        assert replay_result.returncode == 1 and replay_result.stdout == b""
        assert replay_result.stderr == (
            local_root / f"hpg-stderr/{identity}.verifier.txt"
        ).read_bytes()
    for identity in failure_ids:
        stderr = (local_root / f"hpg-stderr/{identity}.txt").read_bytes()
        assert runner.classify_worker_failure(stderr, "H_PG") is not None
    identity = "U02-0000000000_S02-0000000000"
    input_path = local_root / f"inputs/{identity}.txt"
    receipt_path = local_root / f"hpg-receipts/{identity}.txt"
    receipt_challenge = next(
        line.split("=", 1)[1]
        for line in receipt_path.read_text(encoding="ascii").splitlines()
        if line.startswith("RUN_CHALLENGE=")
    )
    w3_result_raw = (local_root / "wave-results/W0003.tsv").read_bytes()
    w3_result_lines = w3_result_raw.decode("ascii").splitlines()
    w3_result_table = next(i for i, line in enumerate(w3_result_lines) if "\t" in line)
    w3_result_headers = parse_kv_bytes(
        ("\n".join(w3_result_lines[:w3_result_table]) + "\n").encode("ascii"),
        "local W3 result headers",
    )
    leaf = runner.Leaf(2, 0, 2, 0, "-", 4)
    assert runner.VERIFY.hpg_leaf_challenge(
        local_run_contract["ROOT_CHALLENGE"],
        4,
        sha(w3_result_raw),
        w3_result_headers["NEXT_FRONTIER_SHA256"],
        identity,
        sha(input_path.read_bytes()),
    ) == receipt_challenge
    plain = subprocess.run(
        [
            sys.executable,
            "-B",
            str(v5_verifier),
            str(receipt_path),
            "--source-sha",
            contract["PREPASS_WORKER_SHA256"],
            "--input",
            str(input_path),
            "--challenge",
            receipt_challenge,
        ],
        capture_output=True,
        timeout=120,
    )
    assert plain.returncode == 0 and plain.stderr == b""
    plain_values = parse_kv_bytes(plain.stdout, "aggregator matrix verification")
    hpg_result = runner.HpgResult(
        leaf,
        "H_PG_INVALID_NO_SIGNED_CHART",
        0,
        1,
        sha(input_path.read_bytes()),
        receipt_challenge,
        sha(receipt_path.read_bytes()),
        sha(b""),
        sha(plain.stdout),
        plain_values["PHYSICAL_SHA256"],
        False,
        False,
        (("NONE", 0),) * 4,
        False,
        0,
        0,
    )
    wave_raw = runner.hpg_contract_bytes(
        [hpg_result],
        "1" * 64,
        local_run_contract["ROOT_CHALLENGE"],
        sha(w3_result_raw),
        w3_result_headers["NEXT_FRONTIER_SHA256"],
        contract["PREPASS_WORKER_SHA256"],
        contract["PREPASS_VERIFIER_SHA256"],
        contract["H_APG_WRAPPER_SHA256"],
        contract["H_APG_KERNEL_SHA256"],
        contract["H_APG_ADAPTER_SHA256"],
        contract["H_APG_NUMERIC_VERIFIER_SHA256"],
    )
    aggregate_root = temporary_path / "aggregate-matrix"
    source_root = temporary_path / "aggregate-source"
    for directory in (
        "inputs",
        "hpg-receipts",
        "hpg-stderr",
        "hpg-verifications",
    ):
        (aggregate_root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, aggregate_root / f"inputs/{identity}.txt")
    shutil.copy2(receipt_path, aggregate_root / f"hpg-receipts/{identity}.txt")
    (aggregate_root / f"hpg-stderr/{identity}.txt").write_bytes(b"")
    (aggregate_root / f"hpg-verifications/{identity}.txt").write_bytes(plain.stdout)
    source_root.mkdir()
    shutil.copy2(v5_verifier, source_root / "cs6_plucker_cocycle_verify.py")
    wave_path = aggregate_root / "wave.tsv"
    wave_path.write_bytes(wave_raw)
    wave_lines = wave_raw.decode("ascii").splitlines()
    wave_table = wave_lines.index("\t".join(runner.VERIFY.WAVE_COLUMNS))
    wave_fields = wave_lines[wave_table + 1].split("\t")
    assert len(wave_fields) == len(runner.VERIFY.WAVE_COLUMNS)
    wave_values = dict(zip(runner.VERIFY.WAVE_COLUMNS, wave_fields, strict=True))
    assert wave_values["NODE_ID"] == identity
    assert wave_values["HPG_CHALLENGE"] == receipt_challenge
    # This is a direct verify_leaf_artifacts unit object, not a claim that the
    # one-row serialization is a complete copy of the real 16-row W4 frontier.
    wave = runner.VERIFY.WaveContract(
        {}, {identity: runner.VERIFY.WaveRow(wave_values)}, sha(wave_raw)
    )
    expected_leaf = aggregate.ExpectedLeaf(2, 0, 2, 0, "-", 4)
    result = {
        "WAVE_INDEX": "4",
        "NODE_ID": identity,
        "HPG_STATUS": "H_PG_INVALID_NO_SIGNED_CHART",
        "HAPG_ATTEMPTED": "false",
        "HAPG_STATUS": "H_APG_NOT_ELIGIBLE",
        "HAPG_RC": "0",
        "HAPG_CHALLENGE": "0" * 64,
        "HAPG_RECEIPT_SHA256": aggregate.EMPTY_SHA256,
        "HAPG_STDERR_SHA256": aggregate.EMPTY_SHA256,
        "HAPG_VERIFICATION_SHA256": "0" * 64,
        "HAPG_PHYSICAL_SHA256": "0" * 64,
        **{key: "false" for key in aggregate.RESULT_COLUMNS[11:22]},
        "DECISION": "SPLIT_S",
        "TERMINAL_REASON": "-",
    }

    def aggregate_call(candidate_wave, candidate_result):
        return aggregate.verify_leaf_artifacts(
            aggregate_root,
            source_root,
            wave_path,
            candidate_wave,
            expected_leaf,
            candidate_result,
            contract["PREPASS_WORKER_SHA256"],
            contract["H_APG_WRAPPER_SHA256"],
            local_run_contract["ROOT_CHALLENGE"],
            False,
            120,
        )

    valid_counts = aggregate_call(wave, result)
    assert valid_counts.hpg == 1 and valid_counts.hpg_mutations == 0
    verification_path = aggregate_root / f"hpg-verifications/{identity}.txt"
    stderr_path = aggregate_root / f"hpg-stderr/{identity}.txt"
    baseline_verification = verification_path.read_bytes()
    baseline_stderr = stderr_path.read_bytes()

    def expect_aggregate_reject(label: str, mutate) -> None:
        verification_path.write_bytes(baseline_verification)
        stderr_path.write_bytes(baseline_stderr)
        candidate_wave = copy.deepcopy(wave)
        candidate_result = dict(result)
        mutate(candidate_wave.rows[identity].values, candidate_result)
        try:
            aggregate_call(candidate_wave, candidate_result)
        except aggregate.AggregateError:
            return
        raise AssertionError(f"aggregator accepted forged H-PG outcome: {label}")

    expect_aggregate_reject(
        "status", lambda row, _: row.__setitem__("HPG_STATUS", "H_PG_CROSSING")
    )
    expect_aggregate_reject(
        "physical", lambda row, _: row.__setitem__("HPG_PHYSICAL_SHA256", "d" * 64)
    )
    expect_aggregate_reject(
        "probe", lambda row, _: row.__setitem__("HPG_PROBE_PASS", "true")
    )
    expect_aggregate_reject(
        "certificate", lambda row, _: row.__setitem__("HPG_CERTIFICATE_PASS", "true")
    )
    expect_aggregate_reject(
        "eligibility", lambda row, _: row.__setitem__("HAPG_ELIGIBLE", "true")
    )
    expect_aggregate_reject(
        "chart", lambda row, _: row.__setitem__("E1_R0_CHART", "X")
    )
    expect_aggregate_reject(
        "receipt", lambda row, _: row.__setitem__("HPG_RECEIPT_SHA256", "d" * 64)
    )
    expect_aggregate_reject(
        "verification digest",
        lambda row, _: row.__setitem__("HPG_VERIFICATION_SHA256", "d" * 64),
    )
    expect_aggregate_reject(
        "HAPG attempt", lambda _, candidate: candidate.__setitem__("HAPG_ATTEMPTED", "true")
    )

    def forge_stderr(row, _):
        stderr_path.write_bytes(b"forged\n")
        row["HPG_STDERR_SHA256"] = sha(stderr_path.read_bytes())

    expect_aggregate_reject("rc0 stderr", forge_stderr)

    def forge_stored_verification(row, _):
        verification_path.write_bytes(baseline_verification + b"FORGED=1\n")
        row["HPG_VERIFICATION_SHA256"] = sha(verification_path.read_bytes())

    expect_aggregate_reject("stored verification", forge_stored_verification)
    verification_path.write_bytes(baseline_verification)
    stderr_path.write_bytes(baseline_stderr)

kat_fields = [
    "NODE_ID",
    "INPUT_SHA256",
    "RECEIPT_SHA256",
    "RUN_CHALLENGE",
    "HPG_WORKER_BINARY_SHA256",
    "OLD_RC",
    "OLD_STDERR_EMPTY",
    "OLD_PLAIN_VERIFICATION_SHA256",
    "V5_RC",
    "V5_STDERR_EMPTY",
    "V5_PLAIN_VERIFICATION_SHA256",
    "PLAIN_BYTE_IDENTICAL",
    "V5_MUTATION_TESTS",
    "V5_MUTATIONS_REJECTED",
    "V5_MUTATION_VERIFICATION_SHA256",
]
kat_rows = read_tsv(abort_root / "hpg-v5-kat-compat.tsv", kat_fields)
assert len(kat_rows) == 52 and len({row["NODE_ID"] for row in kat_rows}) == 52
with tempfile.TemporaryDirectory(prefix="cs6-hapg-v5-kat-gate.") as temporary:
    kat_names = safe_unpack(
        abort_root / "hpg-v4-kat-corpus.tar", Path(temporary)
    )
    kat_root = Path(temporary)
    kat_index_raw = (abort_root / "hpg-v4-kat-corpus-files.sha256").read_bytes()
    kat_index: dict[str, str] = {}
    for line in kat_index_raw.decode("ascii").splitlines():
        digest_token, name = line.split("  ", 1)
        assert name not in kat_index
        kat_index[name] = digest_token
    actual_kat_files = {
        path.relative_to(kat_root).as_posix(): sha(path.read_bytes())
        for path in kat_root.rglob("*")
        if path.is_file()
    }
    assert kat_index == actual_kat_files and len(kat_index) == 223
    assert len(kat_names) == 229
    assert sha((kat_root / "files.sha256").read_bytes()) == (
        "18935b2452e0a594c062c68a6e977238eb51c37ebc9f3a35a96738636e98ccb2"
    )
    assert sha((kat_root / "run-manifest.txt").read_bytes()) == (
        "6d83688f0a624f0661f2abfa0ae7ac65a753cc5c2faff4ebeb8a955df8975c57"
    )
    inner_index: dict[str, str] = {}
    for line in (kat_root / "files.sha256").read_text(encoding="ascii").splitlines():
        digest_token, name = line.split("  ", 1)
        assert name not in inner_index
        inner_index[name] = digest_token
    for name, actual_sha in actual_kat_files.items():
        if name not in {"files.sha256", "run-manifest.txt"}:
            assert inner_index.get(name) == actual_sha
    kat_inputs = {path.stem for path in (kat_root / "inputs").glob("*.txt")}
    kat_receipts = {
        path.stem for path in (kat_root / "hpg-receipts").glob("*.txt")
    }
    kat_stderr = {path.stem for path in (kat_root / "hpg-stderr").glob("*.txt")}
    kat_verifications = {
        path.stem for path in (kat_root / "hpg-verifications").glob("*.txt")
    }
    assert kat_inputs == kat_receipts == kat_stderr and len(kat_inputs) == 53
    assert kat_verifications == {row["NODE_ID"] for row in kat_rows}
    wave_path = kat_root / "wave-contracts/W0000.tsv"
    kat_wave = runner.VERIFY.parse_wave_contract(wave_path)
    assert set(kat_wave.rows) == kat_inputs
    assert len(kat_wave.rows) == 53
    root_id = "U00-0000000000_S00-0000000000"
    assert kat_wave.rows[root_id].values["HPG_STATUS"] == "H_PG_INTERVAL_DOMAIN"
    assert kat_wave.rows[root_id].values["HPG_RC"] == "1"
    assert runner.classify_worker_failure(
        (kat_root / f"hpg-stderr/{root_id}.txt").read_bytes(), "H_PG"
    ) == "H_PG_INTERVAL_DOMAIN"

    def replay_kat(row: dict[str, str]) -> tuple[int, int]:
        identity = row["NODE_ID"]
        input_path = kat_root / f"inputs/{identity}.txt"
        receipt_path = kat_root / f"hpg-receipts/{identity}.txt"
        stderr_path = kat_root / f"hpg-stderr/{identity}.txt"
        stored_path = kat_root / f"hpg-verifications/{identity}.txt"
        assert sha(input_path.read_bytes()) == row["INPUT_SHA256"]
        assert sha(receipt_path.read_bytes()) == row["RECEIPT_SHA256"]
        assert stderr_path.read_bytes() == b""
        base_command = [
            str(receipt_path),
            "--source-sha",
            contract["PREPASS_WORKER_SHA256"],
            "--input",
            str(input_path),
            "--challenge",
            row["RUN_CHALLENGE"],
        ]

        def invoke(verifier: Path, mutations: bool):
            command = [sys.executable, "-B", str(verifier), *base_command]
            if mutations:
                command.append("--self-test-mutations")
            return subprocess.run(command, capture_output=True, timeout=180)

        old_plain = invoke(v4_verifier, False)
        old_mutated = invoke(v4_verifier, True)
        current_plain = invoke(v5_verifier, False)
        current_mutated = invoke(v5_verifier, True)
        assert row["HPG_WORKER_BINARY_SHA256"] == contract["PREBUILT_HPG_BINARY_SHA256"]
        assert row["OLD_RC"] == str(old_plain.returncode)
        assert row["OLD_STDERR_EMPTY"] == str(old_plain.stderr == b"").lower()
        assert row["V5_RC"] == str(current_plain.returncode)
        assert row["V5_STDERR_EMPTY"] == str(current_plain.stderr == b"").lower()
        assert row["PLAIN_BYTE_IDENTICAL"] == str(
            old_plain.stdout == current_plain.stdout
        ).lower()
        assert all(
            result.returncode == 0 and result.stderr == b""
            for result in (old_plain, old_mutated, current_plain, current_mutated)
        )
        assert old_plain.stdout == current_plain.stdout
        assert sha(old_plain.stdout) == row["OLD_PLAIN_VERIFICATION_SHA256"]
        assert sha(current_plain.stdout) == row["V5_PLAIN_VERIFICATION_SHA256"]
        assert old_mutated.stdout == stored_path.read_bytes()
        assert sha(stored_path.read_bytes()) == kat_wave.rows[identity].values[
            "HPG_VERIFICATION_SHA256"
        ]
        assert sha(current_mutated.stdout) == row["V5_MUTATION_VERIFICATION_SHA256"]
        old_values = parse_kv_bytes(old_mutated.stdout, "old KAT verification")
        current_values = parse_kv_bytes(
            current_mutated.stdout, "current KAT verification"
        )
        assert old_values["MUTATION_TESTS"] == old_values["MUTATIONS_REJECTED"] == "76"
        tests = int(current_values["MUTATION_TESTS"])
        rejected = int(current_values["MUTATIONS_REJECTED"])
        assert tests == int(row["V5_MUTATION_TESTS"]) == 79
        assert rejected == int(row["V5_MUTATIONS_REJECTED"]) == tests
        return tests, rejected

    with ThreadPoolExecutor(max_workers=16) as executor:
        kat_replayed = list(executor.map(replay_kat, kat_rows))
    assert sum(tests for tests, _ in kat_replayed) == 4108
    assert sum(rejected for _, rejected in kat_replayed) == 4108
    kat_summary = parse_kv_bytes(
        (kat_root / "summary.txt").read_bytes(), "v4 KAT summary"
    )
    assert {
        key: kat_summary[key]
        for key in (
            "MODE",
            "BOUNDED_RUN_COMPLETE",
            "INFRASTRUCTURE_VALID",
            "EVALUATED_NODE_COUNT",
            "HPG_SIGNED_CHART_COUNT",
            "HAPG_ATTEMPTED_COUNT",
            "HAPG_CERTIFIED_COUNT",
            "HAPG_RESCUE_COUNT",
            "HPG_MUTATION_TESTS",
            "HPG_MUTATIONS_REJECTED",
        )
    } == {
        "MODE": "kat",
        "BOUNDED_RUN_COMPLETE": "true",
        "INFRASTRUCTURE_VALID": "true",
        "EVALUATED_NODE_COUNT": "53",
        "HPG_SIGNED_CHART_COUNT": "52",
        "HAPG_ATTEMPTED_COUNT": "52",
        "HAPG_CERTIFIED_COUNT": "48",
        "HAPG_RESCUE_COUNT": "20",
        "HPG_MUTATION_TESTS": "3952",
        "HPG_MUTATIONS_REJECTED": "3952",
    }


def load(name: str, path: Path) -> ModuleType:
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path.resolve())
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


current = load("cs6_hapg_v5_midpoint_current", v5_verifier)
old = load("cs6_hapg_v5_midpoint_old", v4_verifier)
lower = Fraction.from_float(float.fromhex("0x1.0000000000000p+0"))
upper = Fraction.from_float(float.fromhex("0x1.0000000000001p+0"))
impossible = Fraction(9007199254740993, 9007199254740992)
current_source = current.Interval(lower, upper)
candidates = current.midpoint_candidates(current_source)
assert candidates == frozenset((lower, upper))
for candidate in candidates:
    current.require_midpoint_value(current.point(candidate), current_source, "gate")
for rejected in (
    current.point(impossible),
    current.Interval(lower, upper),
):
    try:
        current.require_midpoint_value(rejected, current_source, "gate")
    except current.VerificationError:
        pass
    else:
        raise AssertionError("discrete midpoint negative escaped")
old.require_midpoint_value(
    old.point(impossible), old.Interval(lower, upper), "old hull witness"
)
midpoint_receipt = parse_kv_bytes(
    (abort_root / "midpoint-discrete-negative-test.txt").read_bytes(),
    "midpoint negative receipt",
)
assert midpoint_receipt == {
    "SCHEMA": "sounio.cs6.hapg-v5-discrete-midpoint-negative-test.v1",
    "SOURCE_LOWER_BINARY64": "0x1.0000000000000p+0",
    "SOURCE_UPPER_BINARY64": "0x1.0000000000001p+0",
    "IMPOSSIBLE_INTERIOR": "9007199254740993/9007199254740992",
    "OLD_INTERVAL_ENVELOPE_CONTAINS_INTERIOR": "true",
    "LOWER_DISCRETE_CANDIDATE_ACCEPTED": "true",
    "UPPER_DISCRETE_CANDIDATE_ACCEPTED": "true",
    "IMPOSSIBLE_INTERIOR_REJECTED": "true",
    "NONPOINT_INTERVAL_REJECTED": "true",
}
print("HAPG_V5_ABORT_DIAGNOSTIC_GATE=54/54")
print("HAPG_V5_ABORT_MUTATIONS=4266/4266")
print("HAPG_V5_KAT_COMPAT_MUTATIONS=4108/4108")
print("HAPG_V5_LOCAL_REPRO_HPG_ATTEMPTS=31")
PY

python3 -B - "$runner" "$aggregator" "$v3_abort" "$contract" <<'PY'
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType

if sys.flags.optimize:
    raise SystemExit("H-APG cover gate error: Python optimization is forbidden")


def load(name: str, path: Path):
    path = path.resolve()
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__source_sha256__ = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


run = load("cs6_hapg_cover_gate_run", Path(sys.argv[1]))
aggregate = load("cs6_hapg_cover_gate_aggregate", Path(sys.argv[2]))
v3_abort = Path(sys.argv[3])
contract = {
    key: value
    for key, value in (
        line.split("=", 1)
        for line in Path(sys.argv[4]).read_text(encoding="ascii").splitlines()
    )
}
root = run.Leaf(0, 0, 0, 0, "-", 0)


def hpg(leaf: object, status: str = "HPG_VERIFIED_SIGNED_CHARTS"):
    return run.HpgResult(
        leaf,
        status,
        0,
        1,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        True,
        True,
        (("X", 1),) * 4,
        True,
        1,
        1,
    )


def hapg(leaf: object, **updates: object):
    values = {
        "leaf": leaf,
        "attempted": True,
        "status": "H_APG_UNCERTIFIED",
        "rc": 0,
        "elapsed_ms": 1,
        "challenge": "7" * 64,
        "receipt_sha": "8" * 64,
        "stderr_sha": "9" * 64,
        "verification_sha": "a" * 64,
        "physical_sha": "b" * 64,
        "probe_pass": True,
        "affine_pass": True,
        "projective_x_pass": False,
        "projective_y_pass": False,
        "projective_plus_pass": False,
        "projective_minus_pass": False,
        "homogeneous_pass": False,
        "apg_valid": True,
        "apg_pass": False,
        "apg_rescue": False,
        "generic_pass": True,
        "mutation_tests": 1,
        "mutations_rejected": 1,
    }
    values.update(updates)
    return run.HapgResult(**values)


checks = 0

assert run.KAT_ANCHOR.run_self_test() == 2
checks += 2
kat_anchor_tests, kat_anchor_rejected = run.KAT_ANCHOR.run_mutation_self_test()
assert kat_anchor_tests == kat_anchor_rejected == 22
checks += kat_anchor_tests

# Wave parsing rejects forged H-PG outcomes before the artifact aggregator runs.
negative_leaf = run.Leaf(2, 0, 2, 0, "-", 4)
run_contract_sha = "1" * 64
root_challenge = "2" * 64
previous_result_sha = "0" * 64
frontier_sha = run.digest_bytes(run.frontier_bytes([negative_leaf]))
input_sha = run.digest_bytes(run.leaf_input_bytes(negative_leaf))
challenge = run.VERIFY.hpg_leaf_challenge(
    root_challenge,
    negative_leaf.wave_index,
    previous_result_sha,
    frontier_sha,
    negative_leaf.identity,
    input_sha,
)


def negative_contract(certificate: bool = False) -> bytes:
    result = run.HpgResult(
        negative_leaf,
        "H_PG_INVALID_NO_SIGNED_CHART",
        0,
        1,
        input_sha,
        challenge,
        "3" * 64,
        run.EMPTY_SHA256,
        "5" * 64,
        "6" * 64,
        False,
        certificate,
        (("NONE", 0),) * 4,
        False,
        79,
        79,
    )
    return run.hpg_contract_bytes(
        [result],
        run_contract_sha,
        root_challenge,
        previous_result_sha,
        frontier_sha,
        "7" * 64,
        "8" * 64,
        "9" * 64,
        "a" * 64,
        "b" * 64,
        "c" * 64,
    )


def mutate_contract(raw: bytes, **updates: str) -> bytes:
    lines = raw.decode("ascii").splitlines()
    table = lines.index("\t".join(run.VERIFY.WAVE_COLUMNS))
    values = lines[table + 1].split("\t")
    indexes = {key: index for index, key in enumerate(run.VERIFY.WAVE_COLUMNS)}
    for key, value in updates.items():
        values[indexes[key]] = value
    lines[table + 1] = "\t".join(values)
    return ("\n".join(lines) + "\n").encode("ascii")


def parse_contract_bytes(raw: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="cs6-hapg-parser-gate.") as directory:
        path = Path(directory) / "wave.tsv"
        path.write_bytes(raw)
        run.VERIFY.parse_wave_contract(path)


valid_negative = negative_contract()
parse_contract_bytes(valid_negative)
parse_contract_bytes(negative_contract(certificate=True))
parse_contract_bytes(
    mutate_contract(
        valid_negative,
        HPG_STATUS="H_PG_CROSSING",
        HPG_RC="1",
        HPG_RECEIPT_SHA256=run.EMPTY_SHA256,
        HPG_STDERR_SHA256="d" * 64,
        HPG_VERIFICATION_SHA256="0" * 64,
        HPG_PHYSICAL_SHA256="0" * 64,
    )
)
parse_contract_bytes(
    mutate_contract(
        valid_negative,
        HPG_STATUS="H_PG_TIMEOUT",
        HPG_RC="124",
        HPG_RECEIPT_SHA256=run.EMPTY_SHA256,
        HPG_VERIFICATION_SHA256="0" * 64,
        HPG_PHYSICAL_SHA256="0" * 64,
    )
)
checks += 4

for updates in (
    {"HPG_PROBE_PASS": "true"},
    {"HPG_STDERR_SHA256": "d" * 64},
    {"HPG_RECEIPT_SHA256": run.EMPTY_SHA256},
    {"HPG_RECEIPT_SHA256": "0" * 64},
    {"HPG_STDERR_SHA256": "0" * 64},
    {"HPG_VERIFICATION_SHA256": run.EMPTY_SHA256},
    {"HPG_VERIFICATION_SHA256": "0" * 64},
    {"HAPG_ELIGIBLE": "true"},
    {
        "HPG_STATUS": "FORGED_FAILURE_CLASS",
        "HPG_RC": "1",
        "HPG_STDERR_SHA256": "d" * 64,
        "HPG_VERIFICATION_SHA256": "0" * 64,
        "HPG_PHYSICAL_SHA256": "0" * 64,
    },
    {
        "HPG_STATUS": "H_PG_CROSSING",
        "HPG_RC": "124",
        "HPG_STDERR_SHA256": "d" * 64,
        "HPG_VERIFICATION_SHA256": "0" * 64,
        "HPG_PHYSICAL_SHA256": "0" * 64,
    },
    {
        "HPG_STATUS": "H_PG_CAPD_SET",
        "HPG_RC": "1",
        "HPG_VERIFICATION_SHA256": "0" * 64,
        "HPG_PHYSICAL_SHA256": "0" * 64,
    },
):
    try:
        parse_contract_bytes(mutate_contract(valid_negative, **updates))
    except run.VERIFY.CoverVerificationError:
        checks += 1
    else:
        raise AssertionError(f"forged H-PG wave row accepted: {updates}")

nontransversal = (
    b"probe error: PoincareMap error: possible nontransversal return to the section \n"
    b"Inner product of vector field and section gradient: [-1, 1]\n"
)
assert run.classify_worker_failure(nontransversal, "H_PG") == "H_PG_CROSSING"
assert aggregate.classify_failure(nontransversal, "H_PG") == "H_PG_CROSSING"
checks += 2

# The same bytes never manufacture an undeclared H-APG class.
assert run.classify_worker_failure(nontransversal, "H_APG") is None
assert aggregate.classify_failure(nontransversal, "H_APG") is None
checks += 2

# Exact header without the CAPD normal-velocity diagnostic remains unknown.
header_only = nontransversal.splitlines()[0] + b"\n"
assert run.classify_worker_failure(header_only, "H_PG") is None
assert aggregate.classify_failure(header_only, "H_PG") is None
checks += 2

# Existing generic numeric failures are H-PG-only under the frozen class set.
interval_error = b"probe error: interval error: division by 0\n"
assert run.classify_worker_failure(interval_error, "H_PG") == "H_PG_INTERVAL_DOMAIN"
assert aggregate.classify_failure(interval_error, "H_PG") == "H_PG_INTERVAL_DOMAIN"
assert run.classify_worker_failure(interval_error, "H_APG") is None
assert aggregate.classify_failure(interval_error, "H_APG") is None
checks += 4

# This audits diagnostic classification only; it does not attest execution or prove cover.
# Reclassify every retained raw H-PG stderr with both independent implementations.
raw_stderr = {}
stderr_jsonl = (v3_abort / "hpg-full255-stderr.jsonl").read_bytes()
assert stderr_jsonl.endswith(b"\n") and b"\r" not in stderr_jsonl and b"\0" not in stderr_jsonl
for line in stderr_jsonl.decode("utf-8").splitlines():
    row = json.loads(line)
    raw = row["stderr_text"].encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == row["stderr_sha256"]
    assert row["node_id"] not in raw_stderr
    raw_stderr[row["node_id"]] = raw
counts = Counter()
census_raw = (v3_abort / "hpg-full255-census.tsv").read_bytes()
assert census_raw.endswith(b"\n") and b"\r" not in census_raw and b"\0" not in census_raw
census_reader = csv.DictReader(census_raw.decode("ascii").splitlines(), delimiter="\t")
assert census_reader.fieldnames == [
    "NODE_ID", "WAVE_INDEX", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX",
    "INPUT_SHA256", "RC", "STDERR_SHA256", "NORMALIZED_CLASS",
]
census_rows = list(census_reader)
census_by_id = {row["NODE_ID"]: row for row in census_rows}
assert len(census_by_id) == len(census_rows)

# Rebuild the exact full balanced waves 0..7 and bind all 255 input hashes.
expected_tree = {}
frontier = [root]
for wave_index in range(8):
    assert len(frontier) == 2**wave_index
    next_frontier = []
    for leaf in frontier:
        assert leaf.wave_index == wave_index and leaf.identity not in expected_tree
        expected_tree[leaf.identity] = leaf
        if wave_index < 7:
            _, children = run.split_leaf(leaf)
            next_frontier.extend(children)
    frontier = next_frontier
assert len(expected_tree) == 255
for row in census_rows:
    assert row["RC"] in {"0", "1"}
    leaf = expected_tree[row["NODE_ID"]]
    assert (
        row["WAVE_INDEX"], row["U_DEPTH"], row["U_INDEX"],
        row["S_DEPTH"], row["S_INDEX"],
    ) == tuple(
        str(value)
        for value in (
            leaf.wave_index, leaf.u_depth, leaf.u_index, leaf.s_depth, leaf.s_index,
        )
    )
    assert hashlib.sha256(run.leaf_input_bytes(leaf)).hexdigest() == row["INPUT_SHA256"]
    raw = raw_stderr[row["NODE_ID"]]
    assert hashlib.sha256(raw).hexdigest() == row["STDERR_SHA256"]
    runner_class = run.classify_worker_failure(raw, "H_PG")
    aggregate_class = aggregate.classify_failure(raw, "H_PG")
    assert runner_class == aggregate_class
    if row["RC"] == "0":
        assert runner_class is None and raw == b""
        counts["H_PG_SUCCESS"] += 1
    else:
        assert runner_class is not None
        counts[runner_class] += 1
    expected_class = row["NORMALIZED_CLASS"]
    if expected_class == "H_PG_POINCARE_NONTRANSVERSAL":
        expected_class = "H_PG_CROSSING"
    assert runner_class == expected_class or (
        row["RC"] == "0" and expected_class == "H_PG_SUCCESS"
    )
    # This checks classifier separation, not whether the bytes originated in H-APG.
    assert run.classify_worker_failure(raw, "H_APG") is None
    assert aggregate.classify_failure(raw, "H_APG") is None
assert set(census_by_id) == set(raw_stderr)
assert set(census_by_id) == set(expected_tree)
assert len(census_rows) == len(raw_stderr) == 255
assert counts == Counter(
    H_PG_SUCCESS=54,
    H_PG_INTERVAL_DOMAIN=1,
    H_PG_CROSSING=int(contract["V4_HPG_FULL255_DIAGNOSTIC_POST_V4_CROSSING_COUNT"]),
    H_PG_CAPD_SET=182,
)
checks += len(census_rows)

# Bind the challenge-invariance spot-check to exact census rows and both challenges.
spotcheck = json.loads((v3_abort / "challenge-spotcheck.json").read_text(encoding="ascii"))
assert spotcheck["schema"] == "sounio.cs6.hapg-full-source-cover-hpg-challenge-spotcheck.v1"
spotcheck_rows = spotcheck["rows"]
assert len(spotcheck_rows) == 8
spotcheck_groups = {}
for row in spotcheck_rows:
    coords = tuple(row["coords"])
    assert len(coords) == 4 and all(isinstance(value, int) and value >= 0 for value in coords)
    node_id = f"U{coords[0]:02d}-{coords[1]:010d}_S{coords[2]:02d}-{coords[3]:010d}"
    census_row = census_by_id[node_id]
    assert row["input_sha256"] == census_row["INPUT_SHA256"]
    assert str(row["rc"]) == census_row["RC"]
    assert row["stderr_sha256"] == census_row["STDERR_SHA256"]
    assert all(
        isinstance(row[key], str) and len(row[key]) == 64
        for key in ("input_sha256", "stderr_sha256", "stdout_physics_normalized_sha256")
    )
    spotcheck_groups.setdefault(coords, []).append(row)
assert len(spotcheck_groups) == 4
expected_challenges = {
    contract["BOUNDED_PILOT_ROOT_CHALLENGE"],
    contract["BOUNDED_PILOT_REPLAY_ROOT_CHALLENGE"],
}
for rows in spotcheck_groups.values():
    assert len(rows) == 2 and {row["challenge"] for row in rows} == expected_challenges
    invariant_keys = (
        "coords", "input_sha256", "rc", "stderr_sha256",
        "stdout_physics_normalized_sha256",
    )
    assert all(rows[0][key] == rows[1][key] for key in invariant_keys)
checks += len(spotcheck_rows)

# Generic certificates never terminate an H-APG-only leaf.
evaluations, frontier, _ = run.decide_adaptive_wave(
    [hpg(root)], [hapg(root)], "c" * 64, 1, 255, 8, 30, 30
)
assert evaluations[0].decision == "SPLIT_S" and len(frontier) == 2
checks += 1

# A valid APG pass is the only positive terminal predicate.
evaluations, frontier, _ = run.decide_adaptive_wave(
    [hpg(root)], [hapg(root, apg_pass=True)], "c" * 64, 1, 255, 8, 30, 30
)
assert evaluations[0].decision == "CERTIFIED" and not frontier
checks += 1

# TIMEOUT > AXIS_DEPTH > WAVE_LIMIT > NODE_BUDGET.
cases = (
    ("H_PG_TIMEOUT", 1, 1, 0, 0, "TIMEOUT"),
    ("HPG_VERIFIED_SIGNED_CHARTS", 1, 1, 0, 0, "AXIS_DEPTH"),
    ("HPG_VERIFIED_SIGNED_CHARTS", 1, 1, 30, 30, "WAVE_LIMIT"),
    ("HPG_VERIFIED_SIGNED_CHARTS", 2, 8, 30, 30, "NODE_BUDGET"),
)
for status, max_nodes, max_waves, max_u, max_s, reason in cases:
    evaluations, _, _ = run.decide_adaptive_wave(
        [hpg(root, status)],
        [hapg(root)],
        "c" * 64,
        1,
        max_nodes,
        max_waves,
        max_u,
        max_s,
    )
    assert evaluations[0].terminal_reason == reason
    checks += 1

# All-or-none admission: neither candidate may split when both do not fit.
_, first = run.split_leaf(root)
hpgs = [hpg(leaf) for leaf in first]
hapgs = [hapg(leaf) for leaf in first]
evaluations, next_frontier, _ = run.decide_adaptive_wave(
    hpgs, hapgs, "c" * 64, 3, 6, 8, 30, 30
)
assert not next_frontier
assert {item.terminal_reason for item in evaluations} == {"NODE_BUDGET"}
checks += 1

# A destination appearing during hard-link publication must survive untouched.
with tempfile.TemporaryDirectory(prefix="cs6-hapg-cover-race.") as directory:
    destination = Path(directory) / "frozen.tsv"
    original_link = run.os.link

    def racing_link(source, target, *args, **kwargs):
        Path(target).write_bytes(b"concurrent-owner\n")
        return original_link(source, target, *args, **kwargs)

    run.os.link = racing_link
    try:
        try:
            run.atomic_immutable_file(destination, b"ours\n")
        except FileExistsError:
            pass
        else:
            raise AssertionError("publication collision was accepted")
    finally:
        run.os.link = original_link
    assert destination.read_bytes() == b"concurrent-owner\n"
checks += 1

# Path-component ordering must match the canonical POSIX-token ordering.
with tempfile.TemporaryDirectory(prefix="cs6-hapg-cover-index-order.") as directory:
    index_root = Path(directory)
    (index_root / "fresh-replay").mkdir()
    (index_root / "fresh-replay/evaluations.tsv").write_bytes(b"nested\n")
    (index_root / "fresh-replay-contract.txt").write_bytes(b"contract\n")
    (index_root / "fresh-replay-terminals.tsv").write_bytes(b"terminals\n")
    raw_index = run.file_index(index_root)
    index_tokens = [
        line.split("  ", 1)[1] for line in raw_index.decode("ascii").splitlines()
    ]
    assert index_tokens == sorted(index_tokens) == [
        "fresh-replay-contract.txt",
        "fresh-replay-terminals.tsv",
        "fresh-replay/evaluations.tsv",
    ]
    (index_root / "files.sha256").write_bytes(raw_index)
    assert aggregate.verify_file_index(
        index_root,
        {
            "FILES_INDEX_SHA256": hashlib.sha256(raw_index).hexdigest(),
            "FILE_COUNT": "3",
        },
    ) == 3
checks += 1

# The exact historical path-component order remains rejected by the consumer.
with tempfile.TemporaryDirectory(prefix="cs6-hapg-cover-index-reject.") as directory:
    index_root = Path(directory)
    (index_root / "fresh-replay").mkdir()
    payloads = {
        "fresh-replay/evaluations.tsv": b"nested\n",
        "fresh-replay-contract.txt": b"contract\n",
        "fresh-replay-terminals.tsv": b"terminals\n",
    }
    for token, raw in payloads.items():
        (index_root / token).write_bytes(raw)
    old_order = (
        "fresh-replay/evaluations.tsv",
        "fresh-replay-contract.txt",
        "fresh-replay-terminals.tsv",
    )
    bad_index = "".join(
        f"{hashlib.sha256(payloads[token]).hexdigest()}  {token}\n"
        for token in old_order
    ).encode("ascii")
    (index_root / "files.sha256").write_bytes(bad_index)
    try:
        aggregate.verify_file_index(
            index_root,
            {
                "FILES_INDEX_SHA256": hashlib.sha256(bad_index).hexdigest(),
                "FILE_COUNT": "3",
            },
        )
    except aggregate.AggregateError as error:
        assert str(error) == "files index is not sorted"
    else:
        raise AssertionError("historical path-component file order was accepted")
checks += 1

mutation_total, mutation_rejected = aggregate.self_test_mutations()
assert mutation_total == mutation_rejected == 17
checks += mutation_total
assert checks == 339
print("HAPG_COVER_GATE_TESTS=339/339")
PY

[[ -f $full53/run-manifest.txt && -f $full53/files.sha256 ]] || {
  echo "H-APG cover gate error: retained full53 baseline is missing" >&2
  exit 1
}
[[ $(awk -F= '$1 == "H_APG_CS6_FULL53_SUPPORTED" {print $2}' \
  "$full53/run-manifest.txt") == true ]] || {
  echo "H-APG cover gate error: retained full53 baseline is unsupported" >&2
  exit 1
}
if [[ ${CS6_HAPG_DEEP_BASELINE_REPLAY:-0} == 1 ]]; then
  python3 -B scripts/research/cs6_affine_projective_cocycle_full53_retained_verify.py \
    "$full53"
fi

if [[ -n ${CS6_HAPG_COVER_BUNDLE:-} ]]; then
  retained_required=(
    CS6_HAPG_COVER_KAT_ARCHIVE
    CS6_HAPG_COVER_KAT_ARCHIVE_SHA256
    CS6_HAPG_COVER_KAT_JOB_ID
    CS6_HAPG_COVER_REPO_DELTA_SHA256
    CS6_HAPG_COVER_PREBUILT_ARCHIVE_SHA256
  )
  for key in "${retained_required[@]}"; do
    [[ -n ${!key:-} ]] || {
      echo "H-APG cover gate error: missing retained verification input: $key" >&2
      exit 1
    }
  done
  retained_aggregate=(
    python3 -B "$aggregator" "$CS6_HAPG_COVER_BUNDLE"
    --expected-contract-sha "$(sha256sum "$contract" | awk '{print $1}')"
    --expected-git-head "$(git rev-parse HEAD)"
    --kat-archive "$CS6_HAPG_COVER_KAT_ARCHIVE"
    --kat-archive-sha256 "$CS6_HAPG_COVER_KAT_ARCHIVE_SHA256"
    --kat-job-id "$CS6_HAPG_COVER_KAT_JOB_ID"
    --transport-repo-delta-sha256 "$CS6_HAPG_COVER_REPO_DELTA_SHA256"
    --transport-prebuilt-archive-sha256 "$CS6_HAPG_COVER_PREBUILT_ARCHIVE_SHA256"
    --self-test-mutations
  )
  if [[ -n ${CS6_HAPG_COVER_KAT_SACCT_FILE:-} ]]; then
    retained_aggregate+=(--kat-sacct-file "$CS6_HAPG_COVER_KAT_SACCT_FILE")
  fi
  "${retained_aggregate[@]}"
fi

echo "HAPG_FULL_SOURCE_COVER_GATE_PASS=true"
