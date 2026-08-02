#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

runner=scripts/research/cs6_hapg_full_source_cover_run.py
leaf_verifier=scripts/research/cs6_hapg_full_source_cover_verify.py
aggregator=scripts/research/cs6_hapg_full_source_cover_aggregate.py
wrapper=scripts/research/cs6_hapg_full_source_cover_worker.cpp
slurm_job=scripts/research/cs6_hapg_full_source_cover_slurm_job.sh
contract=scripts/research/cs6_hapg_full_source_cover_contract_v4.txt
v3_contract=scripts/research/cs6_hapg_full_source_cover_contract_v3.txt
full53=scripts/research/receipts/cs6_affine_projective_cocycle_full53_retained_53_v1
v2_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v2_abort_8451_v1
v3_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v3_abort_8453_v1

for required in \
  "$runner" "$leaf_verifier" "$aggregator" "$wrapper" "$slurm_job" "$contract" "$v3_contract" \
  "$v2_abort/manifest.txt" "$v2_abort/sacct.txt" "$v2_abort/config.txt" "$v2_abort/stderr.txt" \
  "$v3_abort/manifest.txt" "$v3_abort/sacct.txt" "$v3_abort/config.txt" \
  "$v3_abort/slurm-stderr.txt" "$v3_abort/repro-s0-stdout.txt" \
  "$v3_abort/repro-s0-stderr.txt" "$v3_abort/repro-s1-stdout.txt" \
  "$v3_abort/repro-s1-stderr.txt" "$v3_abort/hpg-full255-census.tsv" \
  "$v3_abort/hpg-full255-census-summary.txt" "$v3_abort/hpg-full255-stderr.jsonl" \
  "$v3_abort/challenge-spotcheck.json"; do
  [[ -f $required ]] || {
    echo "H-APG cover gate error: missing $required" >&2
    exit 1
  }
done
bash -n "$slurm_job"

python3 -B - "$contract" "$v2_abort" "$v3_abort" "$v3_contract" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
abort_root = Path(sys.argv[2])
v3_abort_root = Path(sys.argv[3])
v3_contract_path = Path(sys.argv[4])
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
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-contract.v4",
    "CONTRACT_STATE": "PRE_RESULT_FROZEN",
    "SUPERSEDES_V3_SHA256": "3e5f1c560356771e9d33582cab31b9776cf6f21d4eabcbc6e292523a2e9010e2",
    "RECOVERY_SCOPE": "DECLARED_H_PG_CROSSING_STDERR_CLASSIFICATION_ONLY",
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
    "SUPERSEDES_V3_SHA256",
    "PREPASS_WORKER_SHA256",
    "PREPASS_VERIFIER_SHA256",
    "H_APG_WRAPPER_SHA256",
    "H_APG_KERNEL_SHA256",
    "H_APG_ADAPTER_SHA256",
    "H_APG_NUMERIC_VERIFIER_SHA256",
    "RUNNER_SHA256",
    "AGGREGATOR_SHA256",
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
)
for key in sha_keys:
    if re.fullmatch(r"[0-9a-f]{64}", fields.get(key, "")) is None:
        raise SystemExit(f"H-APG cover gate error: invalid SHA-256 field {key}")
if re.fullmatch(r"[0-9a-f]{40}", fields.get("IMPLEMENTATION_COMMIT", "")) is None:
    raise SystemExit("H-APG cover gate error: invalid implementation commit")


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
check_binding SUPERSEDES_V3_SHA256 "$v3_contract"

python3 -B - "$runner" "$leaf_verifier" "$aggregator" <<'PY'
from pathlib import Path
import sys

for token in sys.argv[1:]:
    path = Path(token)
    compile(path.read_bytes(), str(path), "exec")
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

mutation_total, mutation_rejected = aggregate.self_test_mutations()
assert mutation_total == mutation_rejected == 17
checks += mutation_total
print(f"HAPG_COVER_GATE_TESTS={checks}/{checks}")
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
  python3 -B "$aggregator" "$CS6_HAPG_COVER_BUNDLE" \
    --expected-contract-sha "$(sha256sum "$contract" | awk '{print $1}')" \
    --expected-git-head "$(git rev-parse HEAD)" \
    --self-test-mutations
fi

echo "HAPG_FULL_SOURCE_COVER_GATE_PASS=true"
