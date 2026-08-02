#!/usr/bin/env bash
set -euo pipefail

if [[ $(python3 -B -c 'import sys; print(sys.flags.optimize)') != 0 ]]; then
  echo "V7-A carrier gate error: Python optimization is forbidden" >&2
  exit 1
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

contract=scripts/research/cs6_hapg_liouville_carrier_ablation_contract_v1.txt
coordinates=scripts/research/cs6_hapg_liouville_carrier_ablation_coordinates_v1.tsv
source=scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp
verifier=scripts/research/cs6_hapg_liouville_carrier_ablation_verify.py
runner=scripts/research/cs6_hapg_liouville_carrier_ablation_run.py
retained=scripts/research/cs6_hapg_liouville_carrier_ablation_retained_verify.py
slurm=scripts/research/cs6_hapg_liouville_carrier_ablation_slurm_job.sh
parent_source=scripts/research/cs6_plucker_cocycle_probe.cpp
parent_verifier=scripts/research/cs6_plucker_cocycle_verify.py
adaptive=scripts/research/receipts/cs6_hapg_full_source_cover_v6_jobs_8469_8470_v1/adaptive/evaluations.tsv
kat_coordinates=scripts/research/receipts/cs6_hapg_full_source_cover_v6_jobs_8469_8470_v1/kat/coordinates.tsv
kat_expected=scripts/research/receipts/cs6_hapg_full_source_cover_v6_jobs_8469_8470_v1/kat/expected-results.tsv
v3_census=scripts/research/receipts/cs6_hapg_full_source_cover_v3_abort_8453_v1/hpg-full255-census.tsv
v3_stderr=scripts/research/receipts/cs6_hapg_full_source_cover_v3_abort_8453_v1/hpg-full255-stderr.jsonl

for path in \
  "$contract" "$coordinates" "$source" "$verifier" "$runner" "$retained" "$slurm" \
  "$parent_source" "$parent_verifier" "$adaptive" "$kat_coordinates" \
  "$kat_expected" "$v3_census" "$v3_stderr"; do
  [[ -f $path && ! -L $path ]] || {
    echo "V7-A carrier gate error: missing regular file $path" >&2
    exit 1
  }
done

bash -n "$slurm"
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile "$verifier" "$runner" "$retained"

python3 -B - \
  "$contract" "$coordinates" "$source" "$verifier" "$runner" \
  "$parent_source" "$parent_verifier" "$adaptive" "$kat_coordinates" \
  "$kat_expected" "$v3_census" "$v3_stderr" <<'PY'
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


(
    contract_path,
    coordinate_path,
    source_path,
    verifier_path,
    runner_path,
    parent_source_path,
    parent_verifier_path,
    adaptive_path,
    kat_coordinate_path,
    kat_expected_path,
    census_path,
    stderr_path,
) = map(Path, sys.argv[1:])

checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        raise SystemExit(f"V7-A carrier gate error: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(path: Path) -> bytes:
    raw = path.read_bytes()
    check(raw.endswith(b"\n"), f"missing final LF: {path}")
    check(b"\r" not in raw and b"\0" not in raw, f"noncanonical bytes: {path}")
    check(raw.isascii(), f"non-ASCII bytes: {path}")
    return raw


def kv(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in canonical(path).decode("ascii").splitlines():
        check(line.count("=") == 1, f"malformed KV line: {path}")
        key, value = line.split("=", 1)
        check(bool(key and value and key not in fields), f"duplicate KV field: {path}")
        fields[key] = value
    return fields


contract = kv(contract_path)
check(
    digest(contract_path)
    == "decf9089e1dc9aae513f48c48a00e1c815a585b6ba7e9cd1c09b0b514fd58481",
    "frozen contract integral digest drift",
)
exact_contract = {
    "SCHEMA": "sounio.cs6.hapg-liouville-carrier-ablation-contract.v1",
    "CONTRACT_STATE": "PRE_RESULT_FROZEN",
    "BASE_COMMIT": "aced77f8207f71c254aa547ce5818ba2a1602ebe",
    "COORDINATE_MANIFEST_SHA256": "df665eceee8a45ea687a9f0bb643fe9fef28c800650482092be113f24fa41fdd",
    "ROOT_CHALLENGE": "47d47736b1c9c181f2041982c4ceb3cb3becf79a66b8740d990212d1a19eadc4",
    "CELL_COUNT": "40",
    "CARRIER_COUNT": "3",
    "CARRIERS": "C0HOTripletonSet,C0HORect2Set,C0Rect2Set",
    "MAXIMUM_EVALUATIONS": "120",
    "EXACT_ATTEMPT_MATRIX": "40_CELLS_TIMES_3_CARRIERS",
    "ATTEMPT_FREEZE_ORDER": "EXACT_120_ATTEMPTS_BEFORE_ANY_WORKER",
    "CARRIER_FALLBACK_ALLOWED": "false",
    "SAME_ACROSS_CARRIERS": "ODE,SECTION,CROSSING_DIRECTION,ORDER,RETURN_COUNT,TILE,INPUT_SHA256,CELL_CHALLENGE,COMPILER_FLAGS,CAPD_BUILD",
    "CAPD_VERSION": "5.3.0",
    "INTERVAL_BACKEND": "FILIB",
    "OPTIMIZATION_LEVEL": "O0",
    "GO_CARRIER": "VERIFIER_COMPLETE_40_OF_40_AND_REPAIRS_CAPD_TARGETS_24_OF_24_AND_PRESERVES_NO_CHART_8_OF_8_AND_PRESERVES_HPG_POSITIVE_8_OF_8_AND_ALL_FINITE_AND_DETERMINANT_COMPATIBLE_AND_REFERENCE_INVARIANT_AND_ALL_MUTATIONS_REJECTED",
    "PARTIAL_IMPROVEMENT_PROMOTION_ALLOWED": "false",
    "FPGA_EXECUTION": "false",
    "PROMOTION_ELIGIBLE": "false",
    "OPEN_PROBLEM_SOLVED": "false",
    "NOVELTY_OR_PRIORITY_CLAIMED": "false",
}
for key, value in exact_contract.items():
    check(contract.get(key) == value, f"frozen contract mismatch: {key}")
check(digest(coordinate_path) == contract["COORDINATE_MANIFEST_SHA256"], "coordinate digest drift")
check(digest(parent_source_path) == contract["PARENT_PROBE_SHA256"], "parent source digest drift")
check(digest(parent_verifier_path) == contract["PARENT_VERIFIER_SHA256"], "parent verifier digest drift")
check(digest(adaptive_path) == contract["V6_ADAPTIVE_EVALUATIONS_SHA256"], "adaptive evidence digest drift")
check(digest(kat_coordinate_path) == contract["V6_KAT_COORDINATES_SHA256"], "KAT coordinate digest drift")
check(digest(kat_expected_path) == contract["V6_KAT_EXPECTED_RESULTS_SHA256"], "KAT result digest drift")
check(digest(census_path) == contract["V3_FULL255_CENSUS_SHA256"], "v3 census digest drift")
check(digest(stderr_path) == contract["V3_FULL255_STDERR_JSONL_SHA256"], "v3 stderr digest drift")

coordinate_lines = canonical(coordinate_path).decode("ascii").splitlines()
header = "ORDINAL\tSAMPLE_CLASS\tSTRATUM\tNODE_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256"
check(header in coordinate_lines, "coordinate table header missing")
header_index = coordinate_lines.index(header)
actual_rows = list(csv.DictReader(coordinate_lines[header_index:], delimiter="\t"))
check(len(actual_rows) == 40, "coordinate row count mismatch")
check(len({row["NODE_ID"] for row in actual_rows}) == 40, "coordinate IDs are not unique")

adaptive_rows = list(csv.DictReader(canonical(adaptive_path).decode("ascii").splitlines(), delimiter="\t"))
census_rows = {
    row["NODE_ID"]: row
    for row in csv.DictReader(canonical(census_path).decode("ascii").splitlines(), delimiter="\t")
}
stderr_rows = {
    row["node_id"]: row["stderr_text"]
    for row in (json.loads(line) for line in canonical(stderr_path).decode("ascii").splitlines())
}


def medoids(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: row["NODE_ID"].encode("ascii"))
    size = len(ordered)
    ranks = [((2 * index + 1) * size) // (2 * count) for index in range(count)]
    check(len(set(ranks)) == count, "medoid rank collision")
    return [ordered[index] for index in ranks]


expected: list[tuple[str, str, dict[str, str], str]] = []
for u_depth, s_depth in (("2", "3"), ("3", "3"), ("3", "4")):
    eligible = [
        row for row in adaptive_rows
        if row["U_DEPTH"] == u_depth
        and row["S_DEPTH"] == s_depth
        and row["HPG_STATUS"] == "H_PG_CAPD_SET"
    ]
    for row in medoids(eligible, 8):
        expected.append(("CAPD_SET_TARGET", f"CAPD_U{u_depth}_S{s_depth}", row, census_rows[row["NODE_ID"]]["INPUT_SHA256"]))

for u_depth, s_depth in (("2", "2"), ("2", "3"), ("3", "3"), ("3", "4")):
    eligible = [
        row for row in adaptive_rows
        if row["U_DEPTH"] == u_depth
        and row["S_DEPTH"] == s_depth
        and row["HPG_STATUS"] == "H_PG_INVALID_NO_SIGNED_CHART"
    ]
    for row in medoids(eligible, 2):
        expected.append(("NO_SIGNED_CHART_CONTROL", f"NO_CHART_U{u_depth}_S{s_depth}", row, census_rows[row["NODE_ID"]]["INPUT_SHA256"]))

kat_lines = canonical(kat_expected_path).decode("ascii").splitlines()
kat_rows = list(csv.DictReader(kat_lines, delimiter="\t"))
kat_selected = sorted(
    [row for row in kat_rows if row["U_DEPTH"] == "8" and row["S_DEPTH"] == "8" and row["PROBE_PASS"] == "true"],
    key=lambda row: row["LEAF_ID"].encode("ascii"),
)
for u_depth, s_depth in (("12", "12"), ("13", "14"), ("14", "13"), ("16", "12")):
    eligible = sorted(
        [row for row in kat_rows if row["U_DEPTH"] == u_depth and row["S_DEPTH"] == s_depth and row["STATUS"] == "PROBE_VALID_CERTIFIED"],
        key=lambda row: row["LEAF_ID"].encode("ascii"),
    )
    check(bool(eligible), f"empty KAT regime U{u_depth}/S{s_depth}")
    kat_selected.append(eligible[0])
for row in kat_selected:
    adapter = {
        "NODE_ID": row["LEAF_ID"],
        "U_DEPTH": row["U_DEPTH"],
        "U_INDEX": row["U_INDEX"],
        "S_DEPTH": row["S_DEPTH"],
        "S_INDEX": row["S_INDEX"],
    }
    stratum = f"HPG_POSITIVE_U{row['U_DEPTH']}_S{row['S_DEPTH']}"
    expected.append(("HPG_POSITIVE_CONTROL", stratum, adapter, row["INPUT_SHA256"]))

check(len(expected) == 40, "reconstructed sample does not contain 40 cells")
for ordinal, (actual, reconstructed) in enumerate(zip(actual_rows, expected, strict=True), 1):
    sample_class, stratum, source_row, input_sha = reconstructed
    check(actual["ORDINAL"] == str(ordinal), f"ordinal drift at row {ordinal}")
    for key, value in (
        ("SAMPLE_CLASS", sample_class),
        ("STRATUM", stratum),
        ("NODE_ID", source_row["NODE_ID"]),
        ("U_DEPTH", source_row["U_DEPTH"]),
        ("U_INDEX", source_row["U_INDEX"]),
        ("S_DEPTH", source_row["S_DEPTH"]),
        ("S_INDEX", source_row["S_INDEX"]),
        ("PARENT_INPUT_SHA256", input_sha),
    ):
        check(actual[key] == value, f"selection reconstruction mismatch: row {ordinal} {key}")
    input_raw = (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={actual['U_DEPTH']}\n"
        f"U_INDEX={actual['U_INDEX']}\n"
        f"S_DEPTH={actual['S_DEPTH']}\n"
        f"S_INDEX={actual['S_INDEX']}\n"
    ).encode("ascii")
    check(hashlib.sha256(input_raw).hexdigest() == input_sha, f"input reconstruction mismatch: row {ordinal}")
    if sample_class == "CAPD_SET_TARGET":
        text = stderr_rows[actual["NODE_ID"]].lower()
        check("centeredtripletonset::evalaffinefunctional - empty intersection" in text, "CAPD target lacks empty-intersection signature")
        check("rq=[-nan, -nan]" in text, "CAPD target lacks NaN rQ signature")
        check(census_rows[actual["NODE_ID"]]["RC"] == "1", "CAPD target v3 RC mismatch")
    elif sample_class == "NO_SIGNED_CHART_CONTROL":
        check(census_rows[actual["NODE_ID"]]["RC"] == "0", "no-chart v3 RC mismatch")
        check(stderr_rows[actual["NODE_ID"]] == "", "no-chart v3 stderr is not empty")

source_text = canonical(source_path).decode("ascii")
for token in (
    "using capd::C0HOTripletonSet;",
    "using capd::C0HORect2Set;",
    "using capd::C0Rect2Set;",
    "return liouville_two_return_with<C0HOTripletonSet>(input);",
    "return liouville_two_return_with<C0HORect2Set>(input);",
    "return liouville_two_return_with<C0Rect2Set>(input);",
    "result.initial_hull = static_cast<IVector>(set);",
    '<< "V7_BINDING"',
    '<< "V7_FAILURE_BINDING"',
):
    check(source_text.count(token) == 1, f"carrier source token mismatch: {token}")
check(source_text.count("result.image = map(set, result.time, 2);") == 1, "carrier arms do not share one map call")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    check(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


verifier = load("cs6_v7_gate_verifier", verifier_path)
runner = load("cs6_v7_gate_runner", runner_path)
parent = verifier.load_parent_verifier()
dummy = "1" * 64
challenge = "2" * 64
carrier = "C0HORect2Set"
binding = verifier.expected_attempt_binding(challenge, carrier, dummy)
prefix = (
    "V7_BINDING"
    f" LIOUVILLE_CARRIER={carrier}"
    f" FROZEN_CONTRACT_SHA256={'3' * 64}"
    f" COORDINATE_MANIFEST_SHA256={'4' * 64}"
    f" RUN_CONTRACT_SHA256={dummy}"
    f" MANIFEST_ROW_SHA256={'5' * 64}"
    f" ATTEMPT_BINDING={binding}"
    " INITIAL0=[-0x1.0000000000001p+0,0x1.0000000000001p+0]"
    " INITIAL1=[-0x1.0000000000001p+0,0x1.0000000000001p+0]"
    " INITIAL2=[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]"
    " INITIAL3=[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]"
)
expected_prefix = {
    "LIOUVILLE_CARRIER": carrier,
    "FROZEN_CONTRACT_SHA256": "3" * 64,
    "COORDINATE_MANIFEST_SHA256": "4" * 64,
    "RUN_CONTRACT_SHA256": dummy,
    "MANIFEST_ROW_SHA256": "5" * 64,
    "ATTEMPT_BINDING": binding,
    "CELL_CHALLENGE": challenge,
}
fields, initial = verifier.parse_prefix(parent, prefix, expected_prefix)
check(fields["ATTEMPT_BINDING"] == binding and len(initial) == 4, "binding self-test failed")
total, rejected = verifier.wrapper_mutations(parent, prefix, expected_prefix)
check(total == rejected == 10, "binding mutation self-test failed")
geometry = parent.frozen_geometry()
source_record = {"U": parent.ZERO, "S": parent.ZERO}
expected_initial = [
    geometry["origin_x"], geometry["origin_y"], parent.ZERO, parent.ZERO
]
verifier.verify_initial_hull(parent, expected_initial, source_record)
initial_total, initial_rejected = verifier.initial_hull_mutations(
    parent, expected_initial, source_record
)
check(initial_total == initial_rejected == 4, "initial hull mutation self-test failed")
failure = (
    f"V7_FAILURE_BINDING LIOUVILLE_CARRIER={carrier} ATTEMPT_BINDING={binding}\n"
    "probe error: CenteredTripletonSet::evalAffineFunctional - empty intersection of rB and rQ. Report this error to CAPD developers!\n"
    "rB=[-1, 1]\n"
    "rQ=[-nan, -nan]\n\n"
).encode("ascii")
check(runner.classify_capd_set(failure, carrier, binding), "CAPD classifier rejected exact bound signature")
check(not runner.classify_capd_set(failure, "C0Rect2Set", binding), "CAPD classifier accepted wrong carrier")
check(not runner.classify_capd_set(b"probe error: empty intersection\nrQ=[-nan, -nan]\n", carrier, binding), "CAPD classifier accepted partial signature")
check(
    runner.FROZEN_CONTRACT_SHA256 == digest(contract_path),
    "runner frozen contract digest drift",
)
decision_cases = (
    ({"baseline_valid": False, "protocol_invalid": False, "control_ok": True, "repair_count": 24, "target_statuses": ["VERIFIED_COMPLETE"] * 24, "all_complete": True, "all_reference_invariant": True}, "RUN_INVALID"),
    ({"baseline_valid": True, "protocol_invalid": True, "control_ok": True, "repair_count": 24, "target_statuses": ["VERIFIED_COMPLETE"] * 24, "all_complete": True, "all_reference_invariant": True}, "RUN_INVALID"),
    ({"baseline_valid": True, "protocol_invalid": False, "control_ok": False, "repair_count": 24, "target_statuses": ["VERIFIED_COMPLETE"] * 24, "all_complete": True, "all_reference_invariant": True}, "NO_GO_CONTROL_REGRESSION"),
    ({"baseline_valid": True, "protocol_invalid": False, "control_ok": True, "repair_count": 24, "target_statuses": ["VERIFIED_COMPLETE"] * 24, "all_complete": True, "all_reference_invariant": True}, "GO"),
    ({"baseline_valid": True, "protocol_invalid": False, "control_ok": True, "repair_count": 24, "target_statuses": ["VERIFIED_COMPLETE"] * 24, "all_complete": True, "all_reference_invariant": False}, "RUN_INVALID"),
    ({"baseline_valid": True, "protocol_invalid": False, "control_ok": True, "repair_count": 0, "target_statuses": ["CAPD_SET_RQ_NAN"] * 24, "all_complete": False, "all_reference_invariant": False}, "NO_GO_ALL_FAILURES"),
    ({"baseline_valid": True, "protocol_invalid": False, "control_ok": True, "repair_count": 7, "target_statuses": ["VERIFIED_COMPLETE"] * 7 + ["CAPD_SET_RQ_NAN"] * 17, "all_complete": False, "all_reference_invariant": False}, "INCONCLUSIVE_PARTIAL"),
)
for arguments, expected_decision in decision_cases:
    check(
        runner.decide_carrier(**arguments) == expected_decision,
        f"decision reducer mismatch: {expected_decision}",
    )

parsed = runner.parse_coordinates(coordinate_path)
check(len(parsed) == 40, "runner coordinate parser count mismatch")
attempts = []
for coordinate in parsed:
    cell = runner.cell_challenge(contract["ROOT_CHALLENGE"], dummy, digest(coordinate_path), coordinate)
    group = []
    for carrier_name in runner.CARRIERS:
        group.append(runner.attempt_binding(cell, carrier_name, dummy))
        attempts.append((coordinate.node_id, carrier_name, cell, group[-1]))
    check(len(set(group)) == 3, f"attempt bindings collide: {coordinate.node_id}")
check(len(attempts) == 120 and len(set(attempts)) == 120, "attempt matrix is not unique 120")
for node in {row[0] for row in attempts}:
    check(len({row[2] for row in attempts if row[0] == node}) == 1, f"cell challenge differs across carriers: {node}")

print(f"V7_A_STATIC_ASSERTIONS={checks}")
print("V7_A_SELECTION_AUDIT=40/40")
print("V7_A_BINDING_MUTATIONS=10/10")
print("V7_A_INITIAL_HULL_MUTATIONS=4/4")
PY

if [[ -n ${CAPD_CONFIG:-} ]]; then
  [[ -x $CAPD_CONFIG ]] || {
    echo "V7-A carrier gate error: CAPD_CONFIG is not executable" >&2
    exit 1
  }
  source_sha=$(sha256sum "$source" | awk '{print $1}')
  binary=$(mktemp /tmp/cs6-v7-carrier-gate.XXXXXX)
  trap 'rm -f "$binary"' EXIT
  cxx=${CXX:-g++}
  # shellcheck disable=SC2046
  "$cxx" -std=c++17 $($CAPD_CONFIG --cflags) -O0 \
    -DCS6_WORKER_SOURCE_SHA256=\"$source_sha\" "$source" -o "$binary" \
    $($CAPD_CONFIG --libs)
  rm -f "$binary"
  trap - EXIT
  echo "V7_A_CAPD_COMPILE=PASS"
else
  echo "V7_A_CAPD_COMPILE=SKIPPED_NO_CAPD_CONFIG"
fi

echo "V7-A carrier gate: PASS"
