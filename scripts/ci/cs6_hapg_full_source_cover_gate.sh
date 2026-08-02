#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

runner=scripts/research/cs6_hapg_full_source_cover_run.py
leaf_verifier=scripts/research/cs6_hapg_full_source_cover_verify.py
aggregator=scripts/research/cs6_hapg_full_source_cover_aggregate.py
wrapper=scripts/research/cs6_hapg_full_source_cover_worker.cpp
slurm_job=scripts/research/cs6_hapg_full_source_cover_slurm_job.sh
contract=scripts/research/cs6_hapg_full_source_cover_contract_v3.txt
full53=scripts/research/receipts/cs6_affine_projective_cocycle_full53_retained_53_v1
v2_abort=scripts/research/receipts/cs6_hapg_full_source_cover_v2_abort_8451_v1

for required in \
  "$runner" "$leaf_verifier" "$aggregator" "$wrapper" "$slurm_job" "$contract" \
  "$v2_abort/manifest.txt" "$v2_abort/sacct.txt" "$v2_abort/config.txt" "$v2_abort/stderr.txt"; do
  [[ -f $required ]] || {
    echo "H-APG cover gate error: missing $required" >&2
    exit 1
  }
done
bash -n "$slurm_job"

python3 -B - "$contract" "$v2_abort" <<'PY'
from __future__ import annotations

from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
abort_root = Path(sys.argv[2])
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
    "SCHEMA": "sounio.cs6.hapg-full-source-cover-contract.v3",
    "CONTRACT_STATE": "PRE_RESULT_FROZEN",
    "SUPERSEDES_V2_SHA256": "7d4bdcdf740fa67a4cd4cba171aaeb5e2ac56bcdaf034a94d4587c937a9056b5",
    "RECOVERY_SCOPE": "SLURM_LAUNCH_PLUMBING_ONLY",
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
    "SLURM_ALLOCATION_SEMANTICS": "EXCLUSIVE_PARTITION_ALLOCATES_FULL_EFFECTIVE_NODE_WHILE_CPUS_PER_TASK_REMAINS_REQUESTED_32",
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

python3 -B - "$runner" "$leaf_verifier" "$aggregator" <<'PY'
from pathlib import Path
import sys

for token in sys.argv[1:]:
    path = Path(token)
    compile(path.read_bytes(), str(path), "exec")
PY

python3 -B - "$runner" "$aggregator" <<'PY'
from __future__ import annotations

import hashlib
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
