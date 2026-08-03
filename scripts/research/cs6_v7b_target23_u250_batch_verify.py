#!/usr/bin/env python3
"""Verify the retained VM100 build and DL380 U250 execution receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from fractions import Fraction
from pathlib import Path


EXPECTED_INPUT_SHA = "e1af962a1a7133ce092ceb60c3cf63551a87e884ec3f85acb5394b309c483935"
EXPECTED_OUTPUT_SHA = "d771d79a7df436f85b2d40b5f9a2bab53bfd42b4e934c9fb5794cb3a67a34ebf"
BASELINE_UUID = "13259b30-d0d2-d4db-deba-bfc0153a26d2"
FALSE_CLAIMS = (
    "RIGOROUS_INTERVAL_CERTIFICATE",
    "LEAF_WIDE_CERTIFICATE",
    "INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE",
    "GLOBAL_HPG_CERTIFICATE",
    "V7_B_ELIGIBILITY",
    "PROMOTION_ELIGIBLE",
    "OPEN_PROBLEM_SOLVED",
    "NOVELTY_OR_PRIORITY_CLAIMED",
)


def fail(message: str) -> None:
    raise ValueError(f"target-23 U250 verify error: {message}")


def key_values(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        fail(f"missing regular file: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            fail(f"malformed line in {path.name}: {line}")
        key, value = line.split("=", 1)
        if key in result:
            fail(f"duplicate key {key} in {path.name}")
        result[key] = value
    return result


def require(summary: dict[str, str], key: str, value: str) -> None:
    if summary.get(key) != value:
        fail(f"{key}: expected {value!r}, got {summary.get(key)!r}")


def recover_q(token: str) -> int:
    value = float(token)
    scaled = value * (1 << 40)
    recovered = round(scaled)
    if not math.isfinite(value) or scaled != recovered:
        fail(f"value is not an exact Q24.40 binary64: {token}")
    return recovered


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(receipt: Path) -> dict[str, str]:
    summary = key_values(receipt / "summary.txt")
    require(summary, "SCHEMA", "sounio.cs6.v7b-target23-u250-batch-summary.v1")
    for key, value in {
        "VM100_BUILD_PASS": "true",
        "HLS_CSIM_PASS": "true",
        "FPGA_EXECUTION": "true",
        "LEAVES": "331",
        "BIT_EXACT_WORDS": "2648",
        "BIT_MISMATCHES": "0",
        "EVENT_ORBITS_PASS": "331",
        "NEGATIVE_DETERMINANTS": "331",
        "INSIDE_BOTH_CAPD": "331",
        "INPUTS_SHA256": EXPECTED_INPUT_SHA,
        "EXPECTED_SHA256": EXPECTED_OUTPUT_SHA,
        "BASELINE_XCLBIN_UUID": BASELINE_UUID,
        "PCIE_LINK": "Gen3x16",
        "ARB_RECHECKS": "3",
        "ARB_CENTER_SIGN_CERTIFICATES": "3",
        "FPGA_INSIDE_ARB_RECHECKS": "0",
        "TARGET23_U250_BATCH_PASS": "true",
    }.items():
        require(summary, key, value)
    for key in FALSE_CLAIMS:
        require(summary, key, "false")
    if int(summary.get("REPEATS", "0")) < 3:
        fail("fewer than three timed repetitions")
    if int(summary.get("POWER_SAMPLES", "0")) < 3:
        fail("fewer than three power samples")
    for key in ("XCLBIN_SHA256", "KERNEL_SOURCE_SHA256", "HOST_BINARY_SHA256"):
        if not re.fullmatch(r"[0-9a-f]{64}", summary.get(key, "")):
            fail(f"invalid {key}")
    executed_uuid = summary.get("EXECUTED_XCLBIN_UUID", "").lower()
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", executed_uuid):
        fail("invalid executed xclbin UUID")
    if executed_uuid == BASELINE_UUID:
        fail("executed xclbin UUID equals baseline")
    for key in ("MEAN_KERNEL_SECONDS", "ORBITS_PER_SECOND", "BASELINE_POWER_WATTS", "MAX_RUN_POWER_WATTS"):
        if not math.isfinite(float(summary.get(key, "nan"))) or float(summary[key]) <= 0:
            fail(f"invalid positive metric {key}")

    reference_path = receipt / "reference.tsv"
    card_path = receipt / "card-results.tsv"
    if sha256(receipt / "card-results.tsv.bin") != EXPECTED_OUTPUT_SHA:
        fail("raw card output hash is not bit-exact")
    reference = list(csv.DictReader(reference_path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    card = list(csv.DictReader(card_path.read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(reference) != 331 or len(card) != 331:
        fail("reference/card cardinality mismatch")
    q_columns = {
        "EVENT1_TIME": "EVENT1_TIME_Q",
        "EVENT2_TIME": "EVENT2_TIME_Q",
        "X2": "X2_Q",
        "Y2": "Y2_Q",
        "ELL2": "ELL2_Q",
    }
    for index, (expected, actual) in enumerate(zip(reference, card, strict=True), 1):
        if actual["LEAF_INDEX"] != str(index) or expected["LEAF_INDEX"] != str(index):
            fail(f"leaf order drift at {index}")
        if actual["STEPS"] != expected["STEPS"]:
            fail(f"step mismatch at leaf {index}")
        for card_key, reference_key in q_columns.items():
            if recover_q(actual[card_key]) != int(expected[reference_key]):
                fail(f"{card_key} mismatch at leaf {index}")
        if actual["INSIDE_BOTH_CAPD"] != "true" or not float(actual["DETERMINANT"]) < 0:
            fail(f"scientific predicate failed at leaf {index}")

    power = list(csv.DictReader((receipt / "power-samples.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(power) != int(summary["POWER_SAMPLES"]):
        fail("power sample cardinality mismatch")
    if max(float(row["POWER_WATTS"]) for row in power) != float(summary["MAX_RUN_POWER_WATTS"]):
        fail("maximum power mismatch")
    if sum(row["PHASE"] == "kernel" for row in power) < 1:
        fail("no power sample taken during kernel execution")

    arb_rows = list(csv.DictReader((receipt / "arb-rechecks.tsv").read_text(encoding="ascii").splitlines(), delimiter="\t"))
    if len(arb_rows) != 3 or any(row["CERTIFICATE"] != "true" for row in arb_rows):
        fail("Arb recheck evidence mismatch")
    if any(row["FPGA_DETERMINANT_INSIDE_ARB"] != "false" for row in arb_rows):
        fail("receipt overstates FPGA containment in the narrow Arb intervals")
    expected_indices = ["319", "331", "329"]
    if [row["LEAF_INDEX"] for row in arb_rows] != expected_indices:
        fail("Arb recheck leaf selection drifted")
    for row in arb_rows:
        index = int(row["LEAF_INDEX"])
        worker = key_values(receipt / "arb" / f"leaf-{index}.txt")
        require(worker, "LEAF_ID", row["LEAF_ID"])
        require(worker, "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE", "true")
        require(worker, "DETERMINANT_STRICT_NEGATIVE", "true")
        lower = Fraction(worker["DETERMINANT_LOWER_Q"])
        upper = Fraction(worker["DETERMINANT_UPPER_Q"])
        if row["ARB_LOWER_Q"] != str(lower) or row["ARB_UPPER_Q"] != str(upper):
            fail(f"Arb endpoint transcript mismatch at leaf {index}")
        if not lower < upper < 0:
            fail(f"Arb interval does not certify a negative determinant at leaf {index}")
        card_det = Fraction.from_float(float(card[index - 1]["DETERMINANT"]))
        if lower < card_det < upper:
            fail(f"receipt says FPGA is outside Arb, but it is inside at leaf {index}")
        distance = lower - card_det if card_det < lower else card_det - upper
        if row["FPGA_TO_ARB_INTERVAL_DISTANCE_Q"] != str(distance):
            fail(f"FPGA/Arb separation mismatch at leaf {index}")
        if row["FPGA_DETERMINANT"] != card[index - 1]["DETERMINANT"]:
            fail(f"Arb/card determinant binding mismatch at leaf {index}")

    manifest = receipt / "artifact-files.sha256"
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        target = receipt / relative
        if target == manifest or not target.is_file() or sha256(target) != digest:
            fail(f"artifact hash mismatch: {relative}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = verify(args.receipt)
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-u250-batch-verification.v1")
    print(f"VERIFIED_LEAVES={summary['LEAVES']}")
    print(f"VERIFIED_BIT_EXACT_WORDS={summary['BIT_EXACT_WORDS']}")
    print(f"VERIFIED_ARB_RECHECKS={summary['ARB_RECHECKS']}")
    print("TARGET23_U250_BATCH_VERIFY_PASS=true")


if __name__ == "__main__":
    main()
