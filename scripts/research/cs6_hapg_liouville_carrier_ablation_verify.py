#!/usr/bin/env python3
"""Verify one bound V7-A carrier attempt using the frozen H-PG ledger."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence


SHA_ZERO = "0" * 64
CARRIERS = ("C0HOTripletonSet", "C0HORect2Set", "C0Rect2Set")
ATTEMPT_DOMAIN = b"sounio.cs6.hapg-liouville-carrier-ablation-attempt.v1\0"
PREFIX_MARKER = "V7_BINDING"
PREFIX_KEYS = (
    "LIOUVILLE_CARRIER",
    "FROZEN_CONTRACT_SHA256",
    "COORDINATE_MANIFEST_SHA256",
    "RUN_CONTRACT_SHA256",
    "MANIFEST_ROW_SHA256",
    "ATTEMPT_BINDING",
    "INITIAL0",
    "INITIAL1",
    "INITIAL2",
    "INITIAL3",
)
EXCLUDED_REFERENCE_RECORDS = {"LIOUVILLE", "LEAF_RESULT", "SUMMARY"}


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_parent_verifier() -> ModuleType:
    path = Path(__file__).with_name("cs6_plucker_cocycle_verify.py")
    spec = importlib.util.spec_from_file_location("cs6_v7_parent_verifier", path)
    if spec is None or spec.loader is None:
        fail("cannot load parent verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_attempt_binding(
    cell_challenge: str, carrier: str, run_contract_sha256: str
) -> str:
    return digest_bytes(
        ATTEMPT_DOMAIN
        + bytes.fromhex(cell_challenge)
        + b"\0"
        + carrier.encode("ascii")
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
    )


def split_receipt(raw: bytes) -> tuple[str, bytes]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail("receipt must be canonical LF-terminated bytes")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise VerificationError("receipt must be ASCII") from error
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0].startswith(PREFIX_MARKER + " "):
        fail("missing V7 binding prefix")
    core = ("\n".join(lines[1:]) + "\n").encode("ascii")
    return lines[0], core


def parse_prefix(
    parent: ModuleType,
    line: str,
    expected: Mapping[str, str],
) -> tuple[dict[str, str], list[object]]:
    tokens = line.split(" ")
    if len(tokens) != len(PREFIX_KEYS) + 1 or tokens[0] != PREFIX_MARKER:
        fail("V7 binding prefix grammar mismatch")
    fields: dict[str, str] = {}
    for token, key in zip(tokens[1:], PREFIX_KEYS, strict=True):
        if token.count("=") != 1:
            fail(f"malformed V7 binding token: {key}")
        actual_key, value = token.split("=", 1)
        if actual_key != key or not value or key in fields:
            fail(f"V7 binding key mismatch: {key}")
        fields[key] = value

    for key, value in expected.items():
        if key == "CELL_CHALLENGE":
            continue
        if fields.get(key) != value:
            fail(f"V7 binding value mismatch: {key}")
    if fields["LIOUVILLE_CARRIER"] not in CARRIERS:
        fail("unknown V7 carrier")
    for key in (
        "FROZEN_CONTRACT_SHA256",
        "COORDINATE_MANIFEST_SHA256",
        "RUN_CONTRACT_SHA256",
        "MANIFEST_ROW_SHA256",
        "ATTEMPT_BINDING",
    ):
        if parent.SHA_RE.fullmatch(fields[key]) is None:
            fail(f"noncanonical V7 digest: {key}")

    recomputed = expected_attempt_binding(
        expected["CELL_CHALLENGE"],
        fields["LIOUVILLE_CARRIER"],
        fields["RUN_CONTRACT_SHA256"],
    )
    if fields["ATTEMPT_BINDING"] != recomputed:
        fail("attempt binding mismatch")

    initial = []
    for index in range(4):
        try:
            initial.append(parent.parse_interval(fields[f"INITIAL{index}"]))
        except parent.VerificationError as error:
            raise VerificationError(f"invalid initial hull coordinate {index}") from error
    return fields, initial


def verify_initial_hull(
    parent: ModuleType,
    initial: Sequence[object],
    source: Mapping[str, object],
) -> None:
    if len(initial) != 4:
        fail("initial hull dimension mismatch")
    geometry = parent.frozen_geometry()
    source_u = parent.interval(source, "U")
    source_s = parent.interval(source, "S")
    expected = (
        geometry["origin_x"]
        + geometry["unstable_x"] * source_u
        + geometry["stable_x"] * source_s,
        geometry["origin_y"]
        + geometry["unstable_y"] * source_u
        + geometry["stable_y"] * source_s,
        parent.ZERO,
        parent.ZERO,
    )
    try:
        parent.require_tight_contains(initial[0], expected[0], "V7 INITIAL0")
        parent.require_tight_contains(initial[1], expected[1], "V7 INITIAL1")
    except parent.VerificationError as error:
        raise VerificationError("initial hull does not reconstruct frozen geometry") from error
    if initial[2] != parent.ZERO or initial[3] != parent.ZERO:
        fail("initial w or Liouville coordinate is not exact zero")


def initial_hull_mutations(
    parent: ModuleType,
    initial: Sequence[object],
    source: Mapping[str, object],
) -> tuple[int, int]:
    rejected = 0
    for index in range(4):
        candidate = list(initial)
        candidate[index] = parent.ONE
        try:
            verify_initial_hull(parent, candidate, source)
        except VerificationError:
            rejected += 1
        else:
            fail("initial-hull mutation escaped verifier")
    return 4, rejected


def record_tokens(core: bytes, marker: str) -> dict[str, str]:
    prefix = marker.encode("ascii") + b" "
    matches = [line for line in core.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        fail(f"core record cardinality mismatch: {marker}")
    fields: dict[str, str] = {}
    for token in matches[0].decode("ascii").split(" ")[1:]:
        if token.count("=") != 1:
            fail(f"malformed core record: {marker}")
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            fail(f"duplicate core record key: {marker}.{key}")
        fields[key] = value
    return fields


def reference_physical_sha256(core: bytes) -> str:
    retained: list[bytes] = []
    for line in core.splitlines():
        marker = line.split(b" ", 1)[0].decode("ascii")
        if marker not in EXCLUDED_REFERENCE_RECORDS:
            retained.append(line)
    return digest_bytes(b"\n".join(retained) + b"\n")


def pivot_sign(parent: ModuleType, record: Mapping[str, object]) -> str:
    chart = parent.string_value(record, "CHART")
    if chart == "NONE":
        return "NONE:0"
    pivot = parent.interval(record, "PIVOT")
    if pivot.upper < 0:
        return f"{chart}:-1"
    if pivot.lower > 0:
        return f"{chart}:+1"
    fail("eligible chart pivot is not sign definite")


def wrapper_mutations(
    parent: ModuleType,
    line: str,
    expected: Mapping[str, str],
) -> tuple[int, int]:
    tokens = line.split(" ")
    fields = {token.split("=", 1)[0]: token.split("=", 1)[1] for token in tokens[1:]}

    def replace(key: str, value: str) -> str:
        candidate = copy.deepcopy(fields)
        candidate[key] = value
        return PREFIX_MARKER + " " + " ".join(f"{name}={candidate[name]}" for name in PREFIX_KEYS)

    def changed_digest(key: str) -> str:
        return "1" * 64 if fields[key] == SHA_ZERO else SHA_ZERO

    alternate = next(carrier for carrier in CARRIERS if carrier != expected["LIOUVILLE_CARRIER"])
    mutations = [
        replace("LIOUVILLE_CARRIER", alternate),
        replace("FROZEN_CONTRACT_SHA256", changed_digest("FROZEN_CONTRACT_SHA256")),
        replace("COORDINATE_MANIFEST_SHA256", changed_digest("COORDINATE_MANIFEST_SHA256")),
        replace("RUN_CONTRACT_SHA256", changed_digest("RUN_CONTRACT_SHA256")),
        replace("MANIFEST_ROW_SHA256", changed_digest("MANIFEST_ROW_SHA256")),
        replace("ATTEMPT_BINDING", changed_digest("ATTEMPT_BINDING")),
        PREFIX_MARKER + " " + " ".join(tokens[2:]),
        line + " EXTRA=1",
        PREFIX_MARKER + " " + " ".join(reversed(tokens[1:])),
        line.replace(" INITIAL3=", " INITIAL2="),
    ]
    rejected = 0
    for candidate in mutations:
        try:
            parse_prefix(parent, candidate, expected)
        except VerificationError:
            rejected += 1
        else:
            fail("V7 binding mutation escaped verifier")
    return len(mutations), rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--challenge", required=True)
    parser.add_argument("--carrier", required=True, choices=CARRIERS)
    parser.add_argument("--frozen-contract-sha", required=True)
    parser.add_argument("--coordinate-manifest-sha", required=True)
    parser.add_argument("--run-contract-sha", required=True)
    parser.add_argument("--manifest-row-sha", required=True)
    parser.add_argument("--attempt-binding", required=True)
    parser.add_argument("--self-test-mutations", action="store_true")
    args = parser.parse_args(argv)

    parent = load_parent_verifier()
    expected = {
        "LIOUVILLE_CARRIER": args.carrier,
        "FROZEN_CONTRACT_SHA256": args.frozen_contract_sha,
        "COORDINATE_MANIFEST_SHA256": args.coordinate_manifest_sha,
        "RUN_CONTRACT_SHA256": args.run_contract_sha,
        "MANIFEST_ROW_SHA256": args.manifest_row_sha,
        "ATTEMPT_BINDING": args.attempt_binding,
        "CELL_CHALLENGE": args.challenge,
    }
    for label, value in (
        ("source", args.source_sha),
        ("challenge", args.challenge),
        ("frozen contract", args.frozen_contract_sha),
        ("coordinate manifest", args.coordinate_manifest_sha),
        ("run contract", args.run_contract_sha),
        ("manifest row", args.manifest_row_sha),
        ("attempt binding", args.attempt_binding),
    ):
        if parent.SHA_RE.fullmatch(value) is None:
            fail(f"{label} must be a lowercase SHA-256")

    raw = args.receipt.read_bytes()
    prefix, core = split_receipt(raw)
    _, initial = parse_prefix(parent, prefix, expected)
    leaf_input = parent.read_leaf_input(args.input)
    with tempfile.NamedTemporaryFile(prefix="cs6-v7-core-", suffix=".txt") as handle:
        handle.write(core)
        handle.flush()
        ledger = parent.parse_ledger(Path(handle.name))
    computed = parent.verify_ledger(ledger, args.source_sha, leaf_input, args.challenge)
    verify_initial_hull(parent, initial, ledger.records["SOURCE_TILE"])

    c1_det = parent.interval(ledger.records["C1_P2_CONTROL"], "DET")
    c2_det = parent.interval(ledger.records["C2_FULL_P2"], "HULL_DET")
    liouville_det = parent.interval(ledger.records["LIOUVILLE"], "DET")
    if not parent.joint_interval([c1_det, c2_det, liouville_det]):
        fail("C1, C2, and Liouville determinants lack a joint intersection")
    if liouville_det.upper >= 0:
        fail("Liouville determinant is not strictly negative")

    parent_total = parent_rejected = wrapper_total = wrapper_rejected = 0
    initial_total = initial_rejected = 0
    if args.self_test_mutations:
        parent_total, parent_rejected = parent.run_mutations(
            ledger, args.source_sha, leaf_input, args.challenge
        )
        wrapper_total, wrapper_rejected = wrapper_mutations(parent, prefix, expected)
        initial_total, initial_rejected = initial_hull_mutations(
            parent, initial, ledger.records["SOURCE_TILE"]
        )

    charts = ",".join(
        pivot_sign(parent, ledger.records[marker])
        for marker in (
            "HOMOGENEOUS_EVENT1_RAY0",
            "HOMOGENEOUS_EVENT1_RAY1",
            "HOMOGENEOUS_EVENT2_RAY0",
            "HOMOGENEOUS_EVENT2_RAY1",
        )
    )
    c1_token = record_tokens(core, "C1_P2_CONTROL")["DET"]
    c2_token = record_tokens(core, "C2_FULL_P2")["HULL_DET"]
    liouville_token = record_tokens(core, "LIOUVILLE")["DET"]
    initial_tokens = [prefix.split(f" INITIAL{index}=", 1)[1].split(" ", 1)[0] for index in range(4)]
    initial_sha = digest_bytes((" ".join(initial_tokens) + "\n").encode("ascii"))
    total = parent_total + wrapper_total + initial_total
    rejected = parent_rejected + wrapper_rejected + initial_rejected

    print("VERIFICATION_SCHEMA=sounio.cs6.hapg-liouville-carrier-ablation-verification.v1")
    print(f"LIOUVILLE_CARRIER={args.carrier}")
    print(f"ATTEMPT_BINDING={args.attempt_binding}")
    print(f"RECEIPT_SHA256={digest_bytes(raw)}")
    print(f"CORE_RECEIPT_SHA256={ledger.receipt_sha256}")
    print(f"PHYSICAL_SHA256={ledger.physical_sha256}")
    print(f"REFERENCE_PHYSICAL_SHA256={reference_physical_sha256(core)}")
    print(f"INITIAL_HULL_SHA256={initial_sha}")
    print(f"MUTATION_TESTS={total}")
    print(f"MUTATIONS_REJECTED={rejected}")
    print("ALL_FINITE=true")
    print("DETERMINANT_JOINT_COMPATIBLE=true")
    print(f"C1_DET={c1_token}")
    print(f"C2_HULL_DET={c2_token}")
    print(f"LIOUVILLE_DET={liouville_token}")
    print(f"CHART_TUPLE={charts}")
    print(f"PROBE_PASS={str(computed['PROBE_PASS']).lower()}")
    print(f"CERTIFICATE_PASS={str(computed['CERTIFICATE_PASS']).lower()}")
    print(f"SUBDIVISION_REQUIRED={str(computed['SUBDIVISION_REQUIRED']).lower()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, OSError) as error:
        print(f"verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
