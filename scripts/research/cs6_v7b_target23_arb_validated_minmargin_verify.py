#!/usr/bin/env python3
"""Verify retained evidence from the Arb validated minimum-margin orbit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from fractions import Fraction
from pathlib import Path


CONTRACT_REL = Path("scripts/research/cs6_v7b_target23_arb_validated_minmargin_contract_v1.txt")
CAPD_RESULT_REL = Path("scripts/research/receipts/cs6_v7b_target23_decimal_center_replay_v1/results.tsv")
WORKER_REL = Path("scripts/research/cs6_v7b_target23_arb_validated_minmargin_worker.py")
LEAF_ID = "U08-0000000223_S09-0000000325"
CHALLENGE_DOMAIN = b"sounio.cs6.v7b-target23-arb-validated-minmargin-challenge.v1\0"
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-target23-arb-validated-minmargin-attempt.v1\0"
KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
FRACTION_RE = re.compile(r"-?[0-9]+(?:/[1-9][0-9]*)?")
HEX_RE = re.compile(r"[0-9a-f]+")
CARRIERS = ("C0HORECT2", "C0RECT2")
WORKER_KEYS = (
    "SCHEMA", "WORKER_SOURCE_SHA256", "PYTHON_VERSION", "PYTHON_FLINT_VERSION",
    "RUN_CHALLENGE", "ATTEMPT_BINDING", "LEAF_ID", "U_DEPTH", "U_INDEX",
    "S_DEPTH", "S_INDEX", "ARB_PRECISION_BITS", "ARB_THREADS", "TAYLOR_ORDER",
    "TIME_STEP_POWER", "STEPS_COMPLETED", "ADVANCE_CALLS", "PICARD_CALLS", "PICARD_CONTAINMENTS",
    "MAX_PICARD_ITERATIONS", "AMBIGUOUS_EVENT_STOPS", "EVENTS_VALIDATED",
    "INITIAL_STATE_CONTAINMENT",
    "EVENT1_TIME_LOWER_Q", "EVENT1_TIME_UPPER_Q", "EVENT2_TIME_LOWER_Q",
    "EVENT2_TIME_UPPER_Q", "INITIAL_NORMAL_LOWER_Q", "INITIAL_NORMAL_UPPER_Q",
    "FINAL_NORMAL_LOWER_Q", "FINAL_NORMAL_UPPER_Q", "EVENT2_ELL_LOWER_Q",
    "EVENT2_ELL_UPPER_Q", "Q0_AREA_LOWER_Q", "Q0_AREA_UPPER_Q",
    "DETERMINANT_LOWER_Q", "DETERMINANT_UPPER_Q", "MAX_GLOBAL_RADIUS_UPPER_Q",
    "MAX_LOCAL_REMAINDER_UPPER_Q", "ACCUMULATED_MU_H_UPPER_Q",
    "MAX_PICARD_CONTRACTION_UPPER_Q",
    "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE", "PICARD_CONTRACTION_OBLIGATION",
    "EVENT2_TRANSVERSALITY", "DETERMINANT_STRICT_NEGATIVE",
    "CAPD_USED_BY_WORKER", "LEAF_WIDE_CERTIFICATE",
    "INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE", "GLOBAL_HPG_CERTIFICATE",
    "V7_B_ELIGIBILITY", "PROMOTION_ELIGIBLE", "OPEN_PROBLEM_SOLVED",
    "NOVELTY_OR_PRIORITY_CLAIMED", "FPGA_EXECUTION",
)
SUMMARY_KEYS = (
    "SCHEMA", "PRE_EXECUTION_GIT_COMMIT", "WORKER_RC", "STDOUT_SHA256",
    "STDERR_SHA256", "RUN_COMPLETE", "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE",
    "LEAF_WIDE_CERTIFICATE", "INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE",
    "GLOBAL_HPG_CERTIFICATE", "V7_B_ELIGIBILITY", "PROMOTION_ELIGIBLE",
    "OPEN_PROBLEM_SOLVED", "NOVELTY_OR_PRIORITY_CLAIMED", "FPGA_EXECUTION",
)


def fail(message: str) -> None:
    raise SystemExit(f"Arb validated minimum-margin verify error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def kv_text(text: str, identity: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            fail(f"malformed key-value line: {identity}")
        key, value = line.split("=", 1)
        if not KEY_RE.fullmatch(key) or not value or key in fields:
            fail(f"invalid key-value field: {identity}:{key}")
        fields[key] = value
    return fields


def kv_file(path: Path) -> dict[str, str]:
    return kv_text(canonical(path), str(path))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def parse_fraction(token: str, identity: str) -> Fraction:
    if not FRACTION_RE.fullmatch(token):
        fail(f"invalid exact fraction: {identity}")
    value = Fraction(token)
    if str(value) != token:
        fail(f"noncanonical exact fraction: {identity}")
    return value


def binary64_fraction(token: str) -> Fraction:
    numerator, denominator = float.fromhex(token).as_integer_ratio()
    return Fraction(numerator, denominator)


def expected_bindings(root: Path, source_commit: str, wheel_sha256: str) -> tuple[str, str]:
    if len(source_commit) != 40 or not HEX_RE.fullmatch(source_commit):
        fail("invalid source commit")
    if len(wheel_sha256) != 64 or not HEX_RE.fullmatch(wheel_sha256):
        fail("invalid wheel digest")
    challenge = digest_bytes(
        CHALLENGE_DOMAIN
        + bytes.fromhex(digest(root / CONTRACT_REL))
        + bytes.fromhex(digest(root / CAPD_RESULT_REL))
        + bytes.fromhex(digest(root / WORKER_REL))
        + bytes.fromhex(wheel_sha256)
        + bytes.fromhex(source_commit)
    )
    binding = digest_bytes(
        ATTEMPT_DOMAIN + bytes.fromhex(challenge) + LEAF_ID.encode("ascii")
    )
    return challenge, binding


def capd_intervals(root: Path) -> dict[str, tuple[Fraction, Fraction]]:
    rows = list(csv.DictReader(canonical(root / CAPD_RESULT_REL).splitlines(), delimiter="\t"))
    selected = [row for row in rows if row.get("LEAF_ID") == LEAF_ID]
    if len(selected) != 1 or selected[0].get("POINTWISE_LEAF_PASS") != "true":
        fail("frozen CAPD comparison row missing or uncertified")
    row = selected[0]
    return {
        carrier: (
            binary64_fraction(row[f"{carrier}_LOWER"]),
            binary64_fraction(row[f"{carrier}_UPPER"]),
        )
        for carrier in CARRIERS
    }


def verify(receipt: Path, source_commit: str, wheel_sha256: str) -> dict[str, str]:
    root = Path.cwd()
    summary = kv_file(receipt / "execution-summary.txt")
    if tuple(summary) != SUMMARY_KEYS:
        fail("execution summary key population or order drifted")
    expected_summary = {
        "SCHEMA": "sounio.cs6.v7b-target23-arb-validated-minmargin-execution.v1",
        "PRE_EXECUTION_GIT_COMMIT": source_commit,
        "WORKER_RC": "0", "RUN_COMPLETE": "true",
        "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE": "false",
        "LEAF_WIDE_CERTIFICATE": "false", "INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false", "V7_B_ELIGIBILITY": "false",
        "PROMOTION_ELIGIBLE": "false", "OPEN_PROBLEM_SOLVED": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false", "FPGA_EXECUTION": "false",
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            fail(f"execution summary mismatch: {key}")

    stdout_path, stderr_path = receipt / "worker.stdout.txt", receipt / "worker.stderr.txt"
    stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
    if summary["STDOUT_SHA256"] != digest_bytes(stdout):
        fail("worker stdout hash mismatch")
    if summary["STDERR_SHA256"] != digest_bytes(stderr) or stderr:
        fail("worker stderr mismatch or nonempty")
    fields = kv_text(stdout.decode("ascii"), "worker stdout")
    if tuple(fields) != WORKER_KEYS:
        fail("worker field population or order drifted")

    challenge, binding = expected_bindings(root, source_commit, wheel_sha256)
    required = {
        "SCHEMA": "sounio.cs6.v7b-target23-arb-validated-minmargin-worker.v1",
        "WORKER_SOURCE_SHA256": digest(root / WORKER_REL),
        "PYTHON_FLINT_VERSION": "0.8.0", "RUN_CHALLENGE": challenge,
        "ATTEMPT_BINDING": binding, "LEAF_ID": LEAF_ID,
        "U_DEPTH": "8", "U_INDEX": "223", "S_DEPTH": "9", "S_INDEX": "325",
        "ARB_PRECISION_BITS": "256", "ARB_THREADS": "1", "TAYLOR_ORDER": "40",
        "TIME_STEP_POWER": "-8", "STEPS_COMPLETED": "1686", "ADVANCE_CALLS": "1791",
        "PICARD_CALLS": "1793", "PICARD_CONTAINMENTS": "1793",
        "AMBIGUOUS_EVENT_STOPS": "1", "EVENTS_VALIDATED": "2",
        "INITIAL_STATE_CONTAINMENT": "true",
        "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE": "true",
        "PICARD_CONTRACTION_OBLIGATION": "true", "EVENT2_TRANSVERSALITY": "true",
        "DETERMINANT_STRICT_NEGATIVE": "true", "CAPD_USED_BY_WORKER": "false",
        "LEAF_WIDE_CERTIFICATE": "false", "INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false", "V7_B_ELIGIBILITY": "false",
        "PROMOTION_ELIGIBLE": "false", "OPEN_PROBLEM_SOLVED": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false", "FPGA_EXECUTION": "false",
    }
    for key, value in required.items():
        if fields.get(key) != value:
            fail(f"worker field mismatch: {key}")
    for key in ("ADVANCE_CALLS", "PICARD_CALLS", "PICARD_CONTAINMENTS", "MAX_PICARD_ITERATIONS", "AMBIGUOUS_EVENT_STOPS"):
        if not fields[key].isdigit():
            fail(f"invalid counter: {key}")
    if (fields["PICARD_CALLS"] != fields["PICARD_CONTAINMENTS"]
            or int(fields["PICARD_CALLS"]) != int(fields["ADVANCE_CALLS"]) + 2
            or int(fields["ADVANCE_CALLS"]) < 1686):
        fail("Picard containment accounting mismatch")
    if not 1 <= int(fields["MAX_PICARD_ITERATIONS"]) <= 50:
        fail("Picard iteration bound violated")

    fractions = {
        key: parse_fraction(value, key)
        for key, value in fields.items()
        if key.endswith("_Q")
    }
    for prefix in ("EVENT1_TIME", "EVENT2_TIME", "INITIAL_NORMAL", "FINAL_NORMAL", "EVENT2_ELL", "Q0_AREA", "DETERMINANT"):
        if fractions[f"{prefix}_LOWER_Q"] >= fractions[f"{prefix}_UPPER_Q"]:
            fail(f"empty or reversed interval: {prefix}")
    if fractions["INITIAL_NORMAL_LOWER_Q"] <= 0 or fractions["FINAL_NORMAL_LOWER_Q"] <= 0:
        fail("normal velocity is not strictly positive")
    det_lower, det_upper = fractions["DETERMINANT_LOWER_Q"], fractions["DETERMINANT_UPPER_Q"]
    if det_upper >= 0:
        fail("validated determinant is not strictly negative")
    det_width = det_upper - det_lower
    if det_width > Fraction(1, 10**22):
        fail("validated determinant interval is wider than frozen threshold")
    if fractions["MAX_GLOBAL_RADIUS_UPPER_Q"] > Fraction(1, 10**12):
        fail("global state radius exceeds frozen threshold")
    if fractions["MAX_PICARD_CONTRACTION_UPPER_Q"] >= 1:
        fail("Picard contraction obligation violated")
    event2_width = fractions["EVENT2_TIME_UPPER_Q"] - fractions["EVENT2_TIME_LOWER_Q"]
    if event2_width > Fraction(1, 2**40):
        fail("second event bracket exceeds frozen threshold")
    if not (
        fractions["EVENT1_TIME_LOWER_Q"] < fractions["EVENT1_TIME_UPPER_Q"]
        < fractions["EVENT2_TIME_LOWER_Q"] < fractions["EVENT2_TIME_UPPER_Q"]
    ):
        fail("event ordering mismatch")

    intervals = capd_intervals(root)
    for carrier, (lower, upper) in intervals.items():
        if not lower < det_lower < det_upper < upper:
            fail(f"validated center enclosure escapes retained CAPD interval: {carrier}")

    command = canonical(receipt / "worker-command.txt").split()
    if (len(command) != 5 or command[1] != "-B" or Path(command[2]).name != WORKER_REL.name
            or command[3:] != [challenge, binding]):
        fail("worker command mismatch")

    provenance = receipt / "provenance"
    if canonical(provenance / "git-head.txt").strip() != source_commit:
        fail("Slurm payload Git identity mismatch")
    slurm = kv_file(provenance / "slurm-context.txt")
    if not slurm.get("SLURM_JOB_ID", "").isdigit() or slurm.get("SLURM_CPUS_PER_TASK") != "4":
        fail("Slurm execution attestation invalid")
    dependency = kv_file(provenance / "dependency-attestation.txt")
    if (dependency.get("PYTHON_FLINT_WHEEL_SHA256") != wheel_sha256
            or dependency.get("PYTHON_FLINT_VERSION") != "0.8.0"
            or len(dependency.get("FLINT_EXTENSION_SHA256", "")) != 64):
        fail("python-flint dependency attestation invalid")
    runtime = kv_file(provenance / "python-runtime.txt")
    if runtime.get("PYTHON_VERSION") != fields["PYTHON_VERSION"]:
        fail("Python runtime version mismatch")
    resolved_command = str(Path(command[0]).resolve())
    if resolved_command != str(Path(runtime.get("PYTHON_EXECUTABLE", "")).resolve()):
        fail("Python executable identity mismatch")

    return {
        "SLURM_JOB_ID": slurm["SLURM_JOB_ID"],
        "PICARD_CONTAINMENTS": fields["PICARD_CONTAINMENTS"],
        "DETERMINANT_LOWER_Q": str(det_lower),
        "DETERMINANT_UPPER_Q": str(det_upper),
        "DETERMINANT_WIDTH_Q": str(det_width),
        "MAX_GLOBAL_RADIUS_UPPER_Q": str(fractions["MAX_GLOBAL_RADIUS_UPPER_Q"]),
        "EVENT2_TIME_WIDTH_Q": str(event2_width),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    args = parser.parse_args()
    result = verify(args.receipt, args.source_commit, args.wheel_sha256)
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-arb-validated-minmargin-verification.v1")
    for key in ("SLURM_JOB_ID", "PICARD_CONTAINMENTS", "DETERMINANT_LOWER_Q",
                "DETERMINANT_UPPER_Q", "DETERMINANT_WIDTH_Q",
                "MAX_GLOBAL_RADIUS_UPPER_Q", "EVENT2_TIME_WIDTH_Q"):
        print(f"{key}={result[key]}")
    print("INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE=true")
    print("CAPD_COMPATIBLE_CENTER_ENCLOSURE=true")
    print("LEAF_WIDE_CERTIFICATE=false")
    print("INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("FPGA_EXECUTION=false")


if __name__ == "__main__":
    main()
