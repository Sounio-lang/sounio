#!/usr/bin/env python3
"""Verify one bound V7-A.1 carrier-only Liouville checkpoint."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence


FROZEN_CONTRACT_SHA256 = "3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce"
COORDINATE_MANIFEST_SHA256 = "527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7"
ATTEMPT_DOMAIN = b"sounio.cs6.hapg-liouville-checkpoint-attempt.v1\0"
CARRIERS = ("C0HOTripletonSet", "C0HORect2Set", "C0Rect2Set")
EXPECTED_MUTATIONS_PER_CHECKPOINT = 46
ZERO_SHA256 = "0" * 64
ZERO_INTERVAL = "[-0x0.0000000000001p-1022,0x0.0000000000001p-1022]"
ONE_INTERVAL = "[0x1.fffffffffffffp-1,0x1.0000000000001p+0]"
NEGATIVE_ONE_INTERVAL = "[-0x1.0000000000001p+0,-0x1.fffffffffffffp-1]"

BINDING_KEYS = (
    "WORKER_SOURCE_SHA256",
    "INPUT_SHA256",
    "RUN_CHALLENGE",
    "LIOUVILLE_CARRIER",
    "FROZEN_CONTRACT_SHA256",
    "COORDINATE_MANIFEST_SHA256",
    "RUN_CONTRACT_SHA256",
    "MANIFEST_ROW_SHA256",
    "ATTEMPT_BINDING",
)
DECLARATION_KEYS = (
    "SCHEMA",
    "CAPD_SOURCE_TREE_DECLARED",
    "INTERVAL_BACKEND_DECLARED",
    "INTERVAL_SERIALIZATION",
    "SOURCE",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "U_TILES",
    "S_TILES",
    "ORDER",
    "RETURN_COUNT",
    "SECTION",
    "CROSSING_DIRECTION",
    "CHECKPOINT_SCOPE",
    "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED",
    "DOWNSTREAM_SECTION_RESIDENT_EXECUTED",
    "PROMOTION_ELIGIBLE",
)
SOURCE_KEYS = ("U", "S", "Q0_DET")
INITIAL_KEYS = ("X0", "X1", "X2", "X3")
LIOUVILLE_KEYS = (
    "TIME", "X0", "X1", "X2", "X3", "NU0", "NU2", "ELL", "EXP_ELL", "DET"
)
CHECKPOINT_KEYS = ("COMPLETE",)
RECORD_SPECS = (
    ("V7A1_BINDING", BINDING_KEYS),
    ("DECLARATIONS", DECLARATION_KEYS),
    ("SOURCE_TILE", SOURCE_KEYS),
    ("INITIAL_HULL", INITIAL_KEYS),
    ("LIOUVILLE", LIOUVILLE_KEYS),
    ("CHECKPOINT", CHECKPOINT_KEYS),
)


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_interval_verifier() -> ModuleType:
    path = Path(__file__).with_name("cs6_plucker_cocycle_verify.py")
    spec = importlib.util.spec_from_file_location("cs6_v7a1_interval_verifier", path)
    if spec is None or spec.loader is None:
        fail("cannot load exact interval verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_attempt_binding(
    challenge: str, carrier: str, run_contract_sha256: str
) -> str:
    return digest_bytes(
        ATTEMPT_DOMAIN
        + bytes.fromhex(challenge)
        + b"\0"
        + carrier.encode("ascii")
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
    )


def parse_record(line: str, marker: str, keys: Sequence[str]) -> dict[str, str]:
    tokens = line.split(" ")
    if len(tokens) != len(keys) + 1 or tokens[0] != marker:
        fail(f"record grammar mismatch: {marker}")
    fields: dict[str, str] = {}
    for token, expected_key in zip(tokens[1:], keys, strict=True):
        if token.count("=") != 1:
            fail(f"malformed record token: {marker}.{expected_key}")
        key, value = token.split("=", 1)
        if key != expected_key or not value or key in fields:
            fail(f"record key mismatch: {marker}.{expected_key}")
        fields[key] = value
    return fields


def parse_receipt(raw: bytes) -> dict[str, dict[str, str]]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail("receipt is not canonical LF-terminated text")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError("receipt is not ASCII") from error
    if len(lines) != len(RECORD_SPECS):
        fail("receipt record count mismatch")
    return {
        marker: parse_record(line, marker, keys)
        for line, (marker, keys) in zip(lines, RECORD_SPECS, strict=True)
    }


def serialize(records: Mapping[str, Mapping[str, str]]) -> bytes:
    lines = []
    for marker, keys in RECORD_SPECS:
        fields = records[marker]
        lines.append(marker + " " + " ".join(f"{key}={fields[key]}" for key in keys))
    return ("\n".join(lines) + "\n").encode("ascii")


def interval_fields(
    parent: ModuleType, fields: Mapping[str, str], keys: Sequence[str]
) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key in keys:
        try:
            parsed[key] = parent.parse_interval(fields[key])
        except parent.VerificationError as error:
            raise VerificationError(f"invalid interval: {key}") from error
    return parsed


def verify_records(
    parent: ModuleType,
    records: Mapping[str, Mapping[str, str]],
    expected: Mapping[str, str],
    leaf: object,
    expected_initial_sha: str | None,
    expected_det: str | None,
) -> dict[str, str]:
    binding = records["V7A1_BINDING"]
    for key in BINDING_KEYS:
        if binding[key] != expected[key]:
            fail(f"binding mismatch: {key}")
    for key in (
        "WORKER_SOURCE_SHA256",
        "INPUT_SHA256",
        "RUN_CHALLENGE",
        "FROZEN_CONTRACT_SHA256",
        "COORDINATE_MANIFEST_SHA256",
        "RUN_CONTRACT_SHA256",
        "MANIFEST_ROW_SHA256",
        "ATTEMPT_BINDING",
    ):
        if parent.SHA_RE.fullmatch(binding[key]) is None:
            fail(f"noncanonical digest: {key}")
    recomputed_binding = expected_attempt_binding(
        binding["RUN_CHALLENGE"],
        binding["LIOUVILLE_CARRIER"],
        binding["RUN_CONTRACT_SHA256"],
    )
    if binding["ATTEMPT_BINDING"] != recomputed_binding:
        fail("attempt binding does not recompute")

    declarations = records["DECLARATIONS"]
    exact_declarations = {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint.v1",
        "CAPD_SOURCE_TREE_DECLARED": "capd-5.3.0",
        "INTERVAL_BACKEND_DECLARED": "FILIB",
        "INTERVAL_SERIALIZATION": "ONE_ULP_OUTWARD_BINARY64_HEX",
        "SOURCE": "N0",
        "U_DEPTH": str(leaf.u_depth),
        "U_INDEX": str(leaf.u_index),
        "S_DEPTH": str(leaf.s_depth),
        "S_INDEX": str(leaf.s_index),
        "U_TILES": str(1 << leaf.u_depth),
        "S_TILES": str(1 << leaf.s_depth),
        "ORDER": "8",
        "RETURN_COUNT": "2",
        "SECTION": "COORDINATE_W_EQUALS_ZERO",
        "CROSSING_DIRECTION": "MINUS_PLUS",
        "CHECKPOINT_SCOPE": "LIOUVILLE_CARRIER_ONLY",
        "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED": "false",
        "DOWNSTREAM_SECTION_RESIDENT_EXECUTED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    if declarations != exact_declarations:
        fail("declaration record differs from frozen semantics")

    source = interval_fields(parent, records["SOURCE_TILE"], SOURCE_KEYS)
    initial = interval_fields(parent, records["INITIAL_HULL"], INITIAL_KEYS)
    liouville = interval_fields(parent, records["LIOUVILLE"], LIOUVILLE_KEYS)
    geometry = parent.frozen_geometry()
    logical_u, slack_u = parent.frozen_tile(
        geometry["radius_u"], leaf.u_index, 1 << leaf.u_depth
    )
    logical_s, slack_s = parent.frozen_tile(
        geometry["radius_s"], leaf.s_index, 1 << leaf.s_depth
    )
    try:
        parent.require_tile(source["U"], logical_u, slack_u, "source U")
        parent.require_tile(source["S"], logical_s, slack_s, "source S")
    except parent.VerificationError as error:
        raise VerificationError("source tile reconstruction failed") from error

    initial_x = (
        geometry["origin_x"]
        + geometry["unstable_x"] * source["U"]
        + geometry["stable_x"] * source["S"]
    )
    initial_y = (
        geometry["origin_y"]
        + geometry["unstable_y"] * source["U"]
        + geometry["stable_y"] * source["S"]
    )
    frame_det = (
        geometry["unstable_x"] * geometry["stable_y"]
        - geometry["stable_x"] * geometry["unstable_y"]
    )
    q0_det = frame_det * geometry["radius_u"] * geometry["radius_s"]
    try:
        parent.require_tight_contains(initial["X0"], initial_x, "initial X0")
        parent.require_tight_contains(initial["X1"], initial_y, "initial X1")
        parent.require_tight_contains(source["Q0_DET"], q0_det, "oriented Q0 determinant")
    except parent.VerificationError as error:
        raise VerificationError("initial geometry reconstruction failed") from error
    if initial["X2"] != parent.ZERO or initial["X3"] != parent.ZERO:
        fail("initial w or ell is not exact zero")

    if liouville["TIME"].lower <= 0:
        fail("return time is not strictly positive")
    if not liouville["X2"].contains_zero():
        fail("returned state does not contain the section")
    if liouville["ELL"] != liouville["X3"]:
        fail("ELL differs from returned auxiliary state")
    try:
        independent_exp = parent.exp_enclosure_negative(liouville["ELL"])
        parent.require_tight_contains(
            liouville["EXP_ELL"], independent_exp, "exp(ell)", 8192
        )
    except parent.VerificationError as error:
        raise VerificationError("independent exponential enclosure failed") from error

    calculated_nu0 = initial_x * initial_y - geometry["zs"]
    calculated_nu2 = (
        liouville["X0"] * liouville["X1"]
        - liouville["X2"]
        - geometry["zs"]
    )
    try:
        parent.require_tight_contains(liouville["NU0"], calculated_nu0, "nu0")
        parent.require_tight_contains(liouville["NU2"], calculated_nu2, "nu2")
    except parent.VerificationError as error:
        raise VerificationError("normal-velocity reconstruction failed") from error
    if liouville["NU0"].lower <= 0 or liouville["NU2"].lower <= 0:
        fail("normal velocity does not certify MinusPlus transversality")

    formula = (
        liouville["EXP_ELL"]
        * liouville["NU0"]
        / liouville["NU2"]
        * q0_det
    )
    try:
        parent.require_tight_contains(liouville["DET"], formula, "Liouville determinant")
    except parent.VerificationError as error:
        raise VerificationError("Liouville determinant identity failed") from error
    if liouville["DET"].upper >= 0:
        fail("Liouville determinant is not strictly negative")
    if records["CHECKPOINT"] != {"COMPLETE": "true"}:
        fail("checkpoint is not complete")

    initial_tokens = [records["INITIAL_HULL"][f"X{index}"] for index in range(4)]
    initial_sha = digest_bytes((" ".join(initial_tokens) + "\n").encode("ascii"))
    if expected_initial_sha is not None and initial_sha != expected_initial_sha:
        fail("positive-control initial hull differs from parent KAT")
    if expected_det is not None:
        try:
            parent_det = parent.parse_interval(expected_det)
        except parent.VerificationError as error:
            raise VerificationError("parent determinant KAT is malformed") from error
        if not liouville["DET"].overlaps(parent_det):
            fail("positive-control determinant does not overlap parent KAT")
    return {
        "INITIAL_HULL_SHA256": initial_sha,
        "LIOUVILLE_DET": records["LIOUVILLE"]["DET"],
        "PARENT_KAT_STATUS": "PASS" if expected_det is not None else "NOT_APPLICABLE",
    }


def replace_token(raw: bytes, marker: str, key: str, value: str) -> bytes:
    records = parse_receipt(raw)
    changed = copy.deepcopy(records)
    changed[marker][key] = value
    return serialize(changed)


def run_mutations(
    parent: ModuleType,
    raw: bytes,
    expected: Mapping[str, str],
    leaf: object,
    expected_initial_sha: str | None,
    expected_det: str | None,
) -> tuple[int, int]:
    def changed_digest(value: str) -> str:
        return "1" * 64 if value == ZERO_SHA256 else ZERO_SHA256

    records = parse_receipt(raw)
    alternate = next(
        carrier for carrier in CARRIERS if carrier != expected["LIOUVILLE_CARRIER"]
    )
    mutations: list[bytes] = []
    for key in (
        "WORKER_SOURCE_SHA256",
        "INPUT_SHA256",
        "RUN_CHALLENGE",
        "FROZEN_CONTRACT_SHA256",
        "COORDINATE_MANIFEST_SHA256",
        "RUN_CONTRACT_SHA256",
        "MANIFEST_ROW_SHA256",
        "ATTEMPT_BINDING",
    ):
        mutations.append(
            replace_token(raw, "V7A1_BINDING", key, changed_digest(records["V7A1_BINDING"][key]))
        )
    mutations.append(replace_token(raw, "V7A1_BINDING", "LIOUVILLE_CARRIER", alternate))
    for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "ORDER", "RETURN_COUNT"):
        mutations.append(replace_token(raw, "DECLARATIONS", key, "0"))
    mutations.extend(
        [
            replace_token(raw, "DECLARATIONS", "SECTION", "COORDINATE_X_EQUALS_ZERO"),
            replace_token(raw, "DECLARATIONS", "CROSSING_DIRECTION", "PLUS_MINUS"),
            replace_token(raw, "DECLARATIONS", "DOWNSTREAM_SECTION_RESIDENT_EXECUTED", "true"),
            replace_token(raw, "SOURCE_TILE", "U", ONE_INTERVAL),
            replace_token(raw, "SOURCE_TILE", "S", ONE_INTERVAL),
            replace_token(raw, "SOURCE_TILE", "Q0_DET", ONE_INTERVAL),
        ]
    )
    for key in INITIAL_KEYS:
        mutations.append(replace_token(raw, "INITIAL_HULL", key, ONE_INTERVAL))
    for key in ("TIME", "X0", "X1", "X2", "X3", "NU0", "NU2", "ELL", "EXP_ELL", "DET"):
        replacement = NEGATIVE_ONE_INTERVAL if key == "DET" else ONE_INTERVAL
        if key in {"TIME", "X3", "ELL"}:
            replacement = ZERO_INTERVAL
        mutations.append(replace_token(raw, "LIOUVILLE", key, replacement))
    both_erased = replace_token(raw, "LIOUVILLE", "X3", ZERO_INTERVAL)
    both_erased = replace_token(both_erased, "LIOUVILLE", "ELL", ZERO_INTERVAL)
    mutations.append(both_erased)
    mutations.append(replace_token(raw, "LIOUVILLE", "DET", ONE_INTERVAL))
    mutations.append(replace_token(raw, "CHECKPOINT", "COMPLETE", "false"))
    mutations.append(replace_token(raw, "LIOUVILLE", "TIME", "[0,1]"))
    mutations.append(replace_token(raw, "LIOUVILLE", "TIME", "[nan,nan]"))
    mutations.append(
        replace_token(raw, "LIOUVILLE", "TIME", "[0x1.0000000000001p+0,0x1.fffffffffffffp-1]")
    )
    lines = raw.splitlines(keepends=True)
    mutations.extend(
        [
            b"".join(lines[:-1]),
            raw + b"EXTRA VALUE=true\n",
            b"".join(reversed(lines)),
            b"".join(lines[:2] + [lines[3], lines[2]] + lines[4:]),
            raw.replace(b"V7A1_BINDING ", b"V7A1_BINDING EXTRA=1 ", 1),
        ]
    )

    rejected = 0
    for candidate in mutations:
        try:
            candidate_records = parse_receipt(candidate)
            verify_records(
                parent,
                candidate_records,
                expected,
                leaf,
                expected_initial_sha,
                expected_det,
            )
        except (VerificationError, parent.VerificationError, UnicodeError):
            rejected += 1
        else:
            fail("mutation escaped checkpoint verifier")
    if len(mutations) != EXPECTED_MUTATIONS_PER_CHECKPOINT:
        fail("mutation suite cardinality drift")
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
    parser.add_argument("--expected-initial-sha")
    parser.add_argument("--expected-det")
    parser.add_argument("--self-test-mutations", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test_mutations:
        fail("mutation self-tests are mandatory for V7-A.1")

    parent = load_interval_verifier()
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
    if args.frozen_contract_sha != FROZEN_CONTRACT_SHA256:
        fail("frozen contract digest mismatch")
    if args.coordinate_manifest_sha != COORDINATE_MANIFEST_SHA256:
        fail("coordinate manifest digest mismatch")
    if (args.expected_initial_sha is None) != (args.expected_det is None):
        fail("positive-control KAT arguments must be paired")
    if args.expected_initial_sha is not None and parent.SHA_RE.fullmatch(args.expected_initial_sha) is None:
        fail("positive-control initial hull SHA is malformed")

    try:
        leaf = parent.read_leaf_input(args.input)
    except parent.VerificationError as error:
        raise VerificationError("leaf input verification failed") from error
    if leaf.sha256 != digest_bytes(args.input.read_bytes()):
        fail("leaf input digest does not recompute")
    expected = {
        "WORKER_SOURCE_SHA256": args.source_sha,
        "INPUT_SHA256": leaf.sha256,
        "RUN_CHALLENGE": args.challenge,
        "LIOUVILLE_CARRIER": args.carrier,
        "FROZEN_CONTRACT_SHA256": args.frozen_contract_sha,
        "COORDINATE_MANIFEST_SHA256": args.coordinate_manifest_sha,
        "RUN_CONTRACT_SHA256": args.run_contract_sha,
        "MANIFEST_ROW_SHA256": args.manifest_row_sha,
        "ATTEMPT_BINDING": args.attempt_binding,
    }
    raw = args.receipt.read_bytes()
    records = parse_receipt(raw)
    computed = verify_records(
        parent,
        records,
        expected,
        leaf,
        args.expected_initial_sha,
        args.expected_det,
    )
    mutation_tests, mutations_rejected = run_mutations(
        parent,
        raw,
        expected,
        leaf,
        args.expected_initial_sha,
        args.expected_det,
    )
    if mutation_tests != mutations_rejected:
        fail("mutation rejection count mismatch")

    liouville_line = raw.splitlines(keepends=True)[4]
    print("VERIFICATION_SCHEMA=sounio.cs6.hapg-liouville-checkpoint-verification.v1")
    print(f"LIOUVILLE_CARRIER={args.carrier}")
    print(f"ATTEMPT_BINDING={args.attempt_binding}")
    print(f"RECEIPT_SHA256={digest_bytes(raw)}")
    print(f"INITIAL_HULL_SHA256={computed['INITIAL_HULL_SHA256']}")
    print(f"LIOUVILLE_RECORD_SHA256={digest_bytes(liouville_line)}")
    print(f"MUTATION_TESTS={mutation_tests}")
    print(f"MUTATIONS_REJECTED={mutations_rejected}")
    print("ALL_FINITE=true")
    print("SOURCE_TILE_RECONSTRUCTED=true")
    print("INITIAL_HULL_RECONSTRUCTED=true")
    print("EXP_ELL_RECOMPUTED=true")
    print("NORMAL_VELOCITIES_RECOMPUTED=true")
    print("LIOUVILLE_IDENTITY_VERIFIED=true")
    print("SECTION_CONTAINS_ZERO=true")
    print(f"LIOUVILLE_DET={computed['LIOUVILLE_DET']}")
    print(f"PARENT_KAT_STATUS={computed['PARENT_KAT_STATUS']}")
    print("CHECKPOINT_PASS=true")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, OSError) as error:
        print(f"verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
