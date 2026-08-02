#!/usr/bin/env python3
"""Verify one wave-frozen H-PG to fixed-chart H-APG leaf."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Mapping, Sequence


sys.dont_write_bytecode = True

class CoverVerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CoverVerificationError(message)


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
WAVE_SCHEMA = "sounio.cs6.hapg-full-source-cover-wave-contract.v1"
HAPG_SCHEMA = "sounio.cs6.hapg-full-source-cover-leaf.v1"
HAPG_EXECUTION_SCOPE = "ARBITRARY_MANIFEST_BOUND_DYADIC_LEAF_HAPG_CAPD_CPU"
HPG_CHALLENGE_DOMAIN = b"sounio.cs6.hapg-cover-hpg-leaf-challenge.v1\0"
SIGNED_CHART_STATUS = "HPG_VERIFIED_SIGNED_CHARTS"
CHART_MARKERS = (
    "HOMOGENEOUS_EVENT1_RAY0",
    "HOMOGENEOUS_EVENT1_RAY1",
    "HOMOGENEOUS_EVENT2_RAY0",
    "HOMOGENEOUS_EVENT2_RAY1",
)
WAVE_HEADERS: tuple[tuple[str, str | None], ...] = (
    ("SCHEMA", WAVE_SCHEMA),
    ("RUN_CONTRACT_SHA256", None),
    ("ROOT_CHALLENGE", None),
    ("WAVE_INDEX", None),
    ("PREVIOUS_WAVE_RESULT_SHA256", None),
    ("FRONTIER_SHA256", None),
    ("NODE_COUNT", None),
    ("HPG_WORKER_SOURCE_SHA256", None),
    ("HPG_VERIFIER_SOURCE_SHA256", None),
    ("HAPG_WORKER_SOURCE_SHA256", None),
    ("HAPG_KERNEL_SOURCE_SHA256", None),
    ("HAPG_VERIFIER_ADAPTER_SHA256", None),
    ("HAPG_NUMERIC_VERIFIER_SHA256", None),
    (
        "FREEZE_ORDER",
        "ALL_HPG_ATTEMPTS_VERIFIED_THEN_ATOMIC_WAVE_CONTRACT_THEN_ANY_HAPG",
    ),
)
WAVE_COLUMNS = (
    "WAVE_INDEX",
    "NODE_ID",
    "PARENT_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "INPUT_SHA256",
    "HPG_CHALLENGE",
    "HPG_STATUS",
    "HPG_RC",
    "HPG_RECEIPT_SHA256",
    "HPG_STDERR_SHA256",
    "HPG_VERIFICATION_SHA256",
    "HPG_PHYSICAL_SHA256",
    "HPG_PROBE_PASS",
    "HPG_CERTIFICATE_PASS",
    "E1_R0_CHART",
    "E1_R0_SIGN",
    "E1_R1_CHART",
    "E1_R1_SIGN",
    "E2_R0_CHART",
    "E2_R0_SIGN",
    "E2_R1_CHART",
    "E2_R1_SIGN",
    "HAPG_ELIGIBLE",
)
HPG_VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "LEAF_METHOD",
    "PROBE_PASS",
    "AFFINE_CERTIFICATE_PASS",
    "PROJECTIVE_X_CERTIFICATE_PASS",
    "PROJECTIVE_Y_CERTIFICATE_PASS",
    "PROJECTIVE_PLUS_CERTIFICATE_PASS",
    "PROJECTIVE_MINUS_CERTIFICATE_PASS",
    "HOMOGENEOUS_CERTIFICATE_PASS",
    "SUBDIVISION_REQUIRED",
    "CERTIFICATE_PASS",
)


def load_adjacent(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    if not path.is_file():
        fail(f"missing adjacent verifier kernel: {filename}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__source_sha256__ = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


HPG_CORE = load_adjacent(
    "cs6_hapg_cover_hpg_verify_core", "cs6_plucker_cocycle_verify.py"
)
HAPG_CORE = load_adjacent(
    "cs6_hapg_cover_hapg_verify_core",
    "cs6_affine_projective_cocycle_full53_verify.py",
)
HAPG_CORE.FIXED_HEADERS = tuple(
    (
        key,
        HAPG_SCHEMA
        if key == "SCHEMA"
        else HAPG_EXECUTION_SCOPE
        if key == "EXECUTION_SCOPE"
        else expected,
    )
    for key, expected in HAPG_CORE.FIXED_HEADERS
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        fail(f"{label} changed while being read")
    return raw


def parse_bool(token: str, label: str) -> bool:
    if token == "true":
        return True
    if token == "false":
        return False
    fail(f"noncanonical boolean: {label}")


def parse_int(token: str, label: str) -> int:
    if INT_RE.fullmatch(token) is None:
        fail(f"noncanonical integer: {label}")
    return int(token)


def canonical_leaf_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def hpg_leaf_challenge(
    root_challenge: str,
    wave_index: int,
    previous_wave_result_sha256: str,
    frontier_sha256: str,
    identity: str,
    input_sha256: str,
) -> str:
    for label, value in (
        ("root challenge", root_challenge),
        ("previous wave result", previous_wave_result_sha256),
        ("frontier", frontier_sha256),
        ("input", input_sha256),
    ):
        if SHA_RE.fullmatch(value) is None:
            fail(f"{label} must be lowercase SHA-256")
    return digest_bytes(
        HPG_CHALLENGE_DOMAIN
        + bytes.fromhex(root_challenge)
        + b"\0"
        + str(wave_index).encode("ascii")
        + b"\0"
        + bytes.fromhex(previous_wave_result_sha256)
        + b"\0"
        + bytes.fromhex(frontier_sha256)
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(input_sha256)
    )


@dataclass(frozen=True)
class WaveRow:
    values: Mapping[str, str]

    @property
    def chart_signs(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (self.values[f"E{event}_R{ray}_CHART"], int(self.values[f"E{event}_R{ray}_SIGN"]))
            for event, ray in ((1, 0), (1, 1), (2, 0), (2, 1))
        )


@dataclass(frozen=True)
class WaveContract:
    headers: Mapping[str, str]
    rows: Mapping[str, WaveRow]
    sha256: str


def frontier_bytes(rows: Mapping[str, WaveRow]) -> bytes:
    header = "NODE_ID\tPARENT_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256"
    body = [header]
    for identity in sorted(rows):
        values = rows[identity].values
        body.append(
            "\t".join(
                values[key]
                for key in (
                    "NODE_ID",
                    "PARENT_ID",
                    "U_DEPTH",
                    "U_INDEX",
                    "S_DEPTH",
                    "S_INDEX",
                    "INPUT_SHA256",
                )
            )
        )
    return ("\n".join(body) + "\n").encode("ascii")


def parse_wave_contract(path: Path) -> WaveContract:
    raw = stable_bytes(path, "wave contract")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise CoverVerificationError("wave contract must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail("wave contract must be canonical LF-terminated ASCII")
    lines = text.splitlines()
    header_count = len(WAVE_HEADERS)
    if len(lines) < header_count + 1:
        fail("wave contract is truncated")
    headers: dict[str, str] = {}
    for line, (expected_key, expected_value) in zip(
        lines[:header_count], WAVE_HEADERS, strict=True
    ):
        if line.count("=") != 1:
            fail(f"malformed wave header: {expected_key}")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            fail(f"noncanonical wave header: {expected_key}")
        if expected_value is not None and value != expected_value:
            fail(f"wave header contract mismatch: {expected_key}")
        headers[key] = value
    for key in (
        "RUN_CONTRACT_SHA256",
        "ROOT_CHALLENGE",
        "PREVIOUS_WAVE_RESULT_SHA256",
        "FRONTIER_SHA256",
        "HPG_WORKER_SOURCE_SHA256",
        "HPG_VERIFIER_SOURCE_SHA256",
        "HAPG_WORKER_SOURCE_SHA256",
        "HAPG_KERNEL_SOURCE_SHA256",
        "HAPG_VERIFIER_ADAPTER_SHA256",
        "HAPG_NUMERIC_VERIFIER_SHA256",
    ):
        if SHA_RE.fullmatch(headers[key]) is None:
            fail(f"wave header is not a SHA-256 digest: {key}")
    wave_index = parse_int(headers["WAVE_INDEX"], "WAVE_INDEX")
    node_count = parse_int(headers["NODE_COUNT"], "NODE_COUNT")
    if lines[header_count] != "\t".join(WAVE_COLUMNS):
        fail("wave contract column schema mismatch")
    data_lines = lines[header_count + 1 :]
    if len(data_lines) != node_count or node_count == 0:
        fail("wave contract row count mismatch")
    rows: dict[str, WaveRow] = {}
    for line in data_lines:
        fields = line.split("\t")
        if len(fields) != len(WAVE_COLUMNS) or any(not field for field in fields):
            fail("malformed wave contract row")
        values = dict(zip(WAVE_COLUMNS, fields, strict=True))
        if parse_int(values["WAVE_INDEX"], "row WAVE_INDEX") != wave_index:
            fail("row wave index mismatch")
        coordinates = tuple(
            parse_int(values[key], key)
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        )
        u_depth, u_index, s_depth, s_index = coordinates
        if u_depth > 30 or s_depth > 30:
            fail("wave row exceeds worker depth contract")
        if not (u_index < 1 << u_depth and s_index < 1 << s_depth):
            fail("wave row index is outside its dyadic depth")
        identity = canonical_leaf_id(*coordinates)
        if values["NODE_ID"] != identity or identity in rows:
            fail("wave row identity is noncanonical or duplicated")
        for key in (
            "INPUT_SHA256",
            "HPG_CHALLENGE",
            "HPG_RECEIPT_SHA256",
            "HPG_STDERR_SHA256",
            "HPG_VERIFICATION_SHA256",
            "HPG_PHYSICAL_SHA256",
        ):
            if SHA_RE.fullmatch(values[key]) is None:
                fail(f"wave row digest is malformed: {key}")
        if (
            values["HPG_RECEIPT_SHA256"] == ZERO_SHA256
            or values["HPG_STDERR_SHA256"] == ZERO_SHA256
        ):
            fail("materialized H-PG artifact uses an absence sentinel")
        hpg_rc = parse_int(values["HPG_RC"], "HPG_RC")
        probe = parse_bool(values["HPG_PROBE_PASS"], "HPG_PROBE_PASS")
        certificate = parse_bool(
            values["HPG_CERTIFICATE_PASS"], "HPG_CERTIFICATE_PASS"
        )
        eligible = parse_bool(values["HAPG_ELIGIBLE"], "HAPG_ELIGIBLE")
        chart_signs = tuple(
            (values[f"E{event}_R{ray}_CHART"], values[f"E{event}_R{ray}_SIGN"])
            for event, ray in ((1, 0), (1, 1), (2, 0), (2, 1))
        )
        if hpg_rc == 0 and (
            values["HPG_RECEIPT_SHA256"] == EMPTY_SHA256
            or values["HPG_STDERR_SHA256"] != EMPTY_SHA256
            or values["HPG_VERIFICATION_SHA256"] in {ZERO_SHA256, EMPTY_SHA256}
            or values["HPG_PHYSICAL_SHA256"] == ZERO_SHA256
        ):
            fail("successful H-PG row has inconsistent artifact sentinels")
        if eligible:
            if (
                values["HPG_STATUS"] != SIGNED_CHART_STATUS
                or values["HPG_RC"] != "0"
                or values["HPG_VERIFICATION_SHA256"] == ZERO_SHA256
                or values["HPG_PHYSICAL_SHA256"] == ZERO_SHA256
                or not probe
                or any(
                    chart not in HAPG_CORE.FULL53_CHARTS or sign not in {"-1", "1"}
                    for chart, sign in chart_signs
                )
            ):
                fail("H-APG eligible row lacks a verified signed-chart prepass")
        else:
            if any(chart != "NONE" or sign != "0" for chart, sign in chart_signs):
                fail("ineligible wave row carries a chart/sign tuple")
            if hpg_rc == 0:
                if (
                    values["HPG_STATUS"] != "H_PG_INVALID_NO_SIGNED_CHART"
                    or probe
                ):
                    fail("successful ineligible H-PG row has inconsistent sentinels")
            else:
                declared_failures = {
                    "H_PG_TIMEOUT",
                    "H_PG_INTERVAL_DOMAIN",
                    "H_PG_CROSSING",
                    "H_PG_CAPD_SET",
                }
                if (
                    values["HPG_STATUS"] not in declared_failures
                    or (hpg_rc == 124) != (values["HPG_STATUS"] == "H_PG_TIMEOUT")
                    or (
                        values["HPG_STATUS"] != "H_PG_TIMEOUT"
                        and values["HPG_STDERR_SHA256"] == EMPTY_SHA256
                    )
                    or values["HPG_VERIFICATION_SHA256"] != ZERO_SHA256
                    or values["HPG_PHYSICAL_SHA256"] != ZERO_SHA256
                    or probe
                    or certificate
                ):
                    fail("failed H-PG row has inconsistent failure fields")
        rows[identity] = WaveRow(values)
    if list(rows) != sorted(rows):
        fail("wave contract rows are not sorted by node ID")
    if digest_bytes(frontier_bytes(rows)) != headers["FRONTIER_SHA256"]:
        fail("wave contract frontier digest mismatch")
    for identity, row in rows.items():
        values = row.values
        expected_challenge = hpg_leaf_challenge(
            headers["ROOT_CHALLENGE"],
            wave_index,
            headers["PREVIOUS_WAVE_RESULT_SHA256"],
            headers["FRONTIER_SHA256"],
            identity,
            values["INPUT_SHA256"],
        )
        if values["HPG_CHALLENGE"] != expected_challenge:
            fail("wave row H-PG challenge derivation mismatch")
    return WaveContract(headers, rows, digest_bytes(raw))


def parse_kv_output(raw: bytes, keys: Sequence[str], label: str) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise CoverVerificationError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"{label} must be canonical LF-terminated ASCII")
    lines = text.splitlines()
    if len(lines) != len(keys):
        fail(f"{label} line count mismatch")
    result: dict[str, str] = {}
    for line, expected in zip(lines, keys, strict=True):
        if line.count("=") != 1:
            fail(f"malformed {label} line")
        key, value = line.split("=", 1)
        if key != expected or not value:
            fail(f"{label} key mismatch: {expected}")
        result[key] = value
    return result


def derive_chart_signs(ledger: object) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for marker in CHART_MARKERS:
        record = ledger.records[marker]
        chart = HPG_CORE.string_value(record, "CHART")
        pivot = HPG_CORE.interval(record, "PIVOT")
        sign = -1 if pivot.upper < 0 else 1 if pivot.lower > 0 else 0
        if chart not in HAPG_CORE.FULL53_CHARTS or sign == 0:
            fail(f"verified H-PG receipt lacks a signed selected chart: {marker}")
        result.append((chart, sign))
    return tuple(result)


def bool_text(value: bool) -> str:
    return str(value).lower()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--hapg-source-sha", required=True)
    parser.add_argument("--hpg-source-sha", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--wave-contract", required=True, type=Path)
    parser.add_argument("--hpg-receipt", required=True, type=Path)
    parser.add_argument("--hpg-verification", required=True, type=Path)
    parser.add_argument("--root-challenge", required=True)
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--require-hapg-terminal", action="store_true")
    parser.add_argument("--require-probe", action="store_true")
    args = parser.parse_args(argv)
    for label, value in (
        ("H-APG source", args.hapg_source_sha),
        ("H-PG source", args.hpg_source_sha),
        ("root challenge", args.root_challenge),
    ):
        if SHA_RE.fullmatch(value) is None:
            fail(f"{label} must be lowercase SHA-256")

    leaf_input = HAPG_CORE.read_leaf_input(args.input)
    identity = canonical_leaf_id(
        leaf_input.u_depth,
        leaf_input.u_index,
        leaf_input.s_depth,
        leaf_input.s_index,
    )
    wave = parse_wave_contract(args.wave_contract)
    if wave.headers["ROOT_CHALLENGE"] != args.root_challenge:
        fail("wave contract root challenge mismatch")
    row = wave.rows.get(identity)
    if row is None:
        fail("canonical input is absent from the wave contract")
    values = row.values
    if not parse_bool(values["HAPG_ELIGIBLE"], "HAPG_ELIGIBLE"):
        fail("H-APG verification is forbidden for an ineligible wave row")
    if values["INPUT_SHA256"] != leaf_input.sha256:
        fail("wave contract input digest mismatch")
    if wave.headers["HPG_WORKER_SOURCE_SHA256"] != args.hpg_source_sha:
        fail("wave contract H-PG source mismatch")
    if wave.headers["HAPG_WORKER_SOURCE_SHA256"] != args.hapg_source_sha:
        fail("wave contract H-APG source mismatch")
    if digest_bytes(Path(__file__).read_bytes()) != wave.headers["HAPG_VERIFIER_ADAPTER_SHA256"]:
        fail("executed H-APG adapter differs from the wave contract")
    hpg_core_path = Path(HPG_CORE.__file__).resolve()
    if (
        digest_bytes(hpg_core_path.read_bytes())
        != wave.headers["HPG_VERIFIER_SOURCE_SHA256"]
        or getattr(HPG_CORE, "__source_sha256__", None)
        != wave.headers["HPG_VERIFIER_SOURCE_SHA256"]
    ):
        fail("executed H-PG verifier differs from the wave contract")
    hapg_core_path = Path(HAPG_CORE.__file__).resolve()
    if (
        digest_bytes(hapg_core_path.read_bytes())
        != wave.headers["HAPG_NUMERIC_VERIFIER_SHA256"]
        or getattr(HAPG_CORE, "__source_sha256__", None)
        != wave.headers["HAPG_NUMERIC_VERIFIER_SHA256"]
    ):
        fail("executed H-APG numeric verifier differs from the wave contract")
    hapg_kernel_path = Path(__file__).resolve().with_name(
        "cs6_affine_projective_cocycle_full53_probe.cpp"
    )
    if (
        not hapg_kernel_path.is_file()
        or digest_bytes(hapg_kernel_path.read_bytes())
        != wave.headers["HAPG_KERNEL_SOURCE_SHA256"]
    ):
        fail("adjacent H-APG numeric kernel differs from the wave contract")

    hpg_receipt_raw = stable_bytes(args.hpg_receipt, "H-PG receipt")
    hpg_verification_raw = stable_bytes(args.hpg_verification, "H-PG verification")
    if digest_bytes(hpg_receipt_raw) != values["HPG_RECEIPT_SHA256"]:
        fail("H-PG receipt digest differs from the wave contract")
    if digest_bytes(hpg_verification_raw) != values["HPG_VERIFICATION_SHA256"]:
        fail("H-PG verification digest differs from the wave contract")
    hpg_input = HPG_CORE.read_leaf_input(args.input)
    hpg_ledger = HPG_CORE.parse_ledger(args.hpg_receipt)
    if (
        hpg_input.sha256 != leaf_input.sha256
        or hpg_ledger.receipt_sha256 != digest_bytes(hpg_receipt_raw)
    ):
        fail("H-PG recomputation reopened bytes different from the frozen snapshot")
    hpg_computed = HPG_CORE.verify_ledger(
        hpg_ledger,
        args.hpg_source_sha,
        hpg_input,
        values["HPG_CHALLENGE"],
    )
    hpg_verification = parse_kv_output(
        hpg_verification_raw, HPG_VERIFICATION_KEYS, "H-PG verification"
    )
    declared_probe = parse_bool(values["HPG_PROBE_PASS"], "wave HPG_PROBE_PASS")
    declared_certificate = parse_bool(
        values["HPG_CERTIFICATE_PASS"], "wave HPG_CERTIFICATE_PASS"
    )
    expected_method = "NONE"
    for candidate, key in (
        ("HOMOGENEOUS", "HOMOGENEOUS_CERTIFICATE_PASS"),
        ("AFFINE", "AFFINE_CERTIFICATE_PASS"),
        ("PROJECTIVE_X", "PROJECTIVE_X_CERTIFICATE_PASS"),
        ("PROJECTIVE_Y", "PROJECTIVE_Y_CERTIFICATE_PASS"),
        ("PROJECTIVE_PLUS", "PROJECTIVE_PLUS_CERTIFICATE_PASS"),
        ("PROJECTIVE_MINUS", "PROJECTIVE_MINUS_CERTIFICATE_PASS"),
    ):
        if hpg_computed[key]:
            expected_method = candidate
            break
    computed_fields = {
        "LEAF_METHOD": expected_method,
        "PROBE_PASS": bool_text(hpg_computed["PROBE_PASS"]),
        "AFFINE_CERTIFICATE_PASS": bool_text(hpg_computed["AFFINE_CERTIFICATE_PASS"]),
        "PROJECTIVE_X_CERTIFICATE_PASS": bool_text(hpg_computed["PROJECTIVE_X_CERTIFICATE_PASS"]),
        "PROJECTIVE_Y_CERTIFICATE_PASS": bool_text(hpg_computed["PROJECTIVE_Y_CERTIFICATE_PASS"]),
        "PROJECTIVE_PLUS_CERTIFICATE_PASS": bool_text(hpg_computed["PROJECTIVE_PLUS_CERTIFICATE_PASS"]),
        "PROJECTIVE_MINUS_CERTIFICATE_PASS": bool_text(hpg_computed["PROJECTIVE_MINUS_CERTIFICATE_PASS"]),
        "HOMOGENEOUS_CERTIFICATE_PASS": bool_text(hpg_computed["HOMOGENEOUS_CERTIFICATE_PASS"]),
        "SUBDIVISION_REQUIRED": bool_text(hpg_computed["SUBDIVISION_REQUIRED"]),
        "CERTIFICATE_PASS": bool_text(hpg_computed["CERTIFICATE_PASS"]),
    }
    if (
        hpg_verification["VERIFICATION_SCHEMA"]
        != "sounio.cs6.plucker-cocycle-leaf-verification.v1"
        or hpg_verification["RECEIPT_SHA256"] != hpg_ledger.receipt_sha256
        or hpg_verification["PHYSICAL_SHA256"] != hpg_ledger.physical_sha256
        or hpg_verification["PHYSICAL_SHA256"] != values["HPG_PHYSICAL_SHA256"]
        or parse_bool(hpg_verification["PROBE_PASS"], "verified PROBE_PASS")
        != hpg_computed["PROBE_PASS"]
        or parse_bool(
            hpg_verification["CERTIFICATE_PASS"], "verified CERTIFICATE_PASS"
        )
        != hpg_computed["CERTIFICATE_PASS"]
        or declared_probe != hpg_computed["PROBE_PASS"]
        or declared_certificate != hpg_computed["CERTIFICATE_PASS"]
        or any(hpg_verification[key] != value for key, value in computed_fields.items())
        or parse_int(hpg_verification["MUTATION_TESTS"], "stored H-PG mutations")
        != parse_int(
            hpg_verification["MUTATIONS_REJECTED"], "stored H-PG rejections"
        )
    ):
        fail("H-PG verification artifact differs from independent recomputation")
    chart_signs = derive_chart_signs(hpg_ledger)
    if chart_signs != row.chart_signs:
        fail("wave chart/sign tuple differs from the verified H-PG receipt")

    contract = HAPG_CORE.Full53LeafContract(
        leaf_id=identity,
        u_depth=leaf_input.u_depth,
        u_index=leaf_input.u_index,
        s_depth=leaf_input.s_depth,
        s_index=leaf_input.s_index,
        parent_input_sha256=leaf_input.sha256,
        parent_status=values["HPG_STATUS"],
        parent_receipt_sha256=values["HPG_RECEIPT_SHA256"],
        chart_signs=chart_signs,
        manifest_sha256=wave.sha256,
    )
    challenge = HAPG_CORE.full53_leaf_challenge(args.root_challenge, contract)
    hapg_ledger = HAPG_CORE.parse_ledger(args.receipt)
    computed = HAPG_CORE.verify_ledger(
        hapg_ledger,
        args.hapg_source_sha,
        leaf_input,
        contract,
        challenge,
    )
    total = rejected = 0
    if args.self_test_mutations:
        total, rejected = HAPG_CORE.run_mutations(
            hapg_ledger,
            args.hapg_source_sha,
            leaf_input,
            contract,
            challenge,
            args.root_challenge,
        )
    terminal = computed["APG_COMPUTATION_VALID"] and computed["APG_CERTIFICATE_PASS"]
    method = "H_APG_RESCUE" if computed["APG_RESCUE"] else "H_APG" if terminal else "NONE"
    output = (
        ("VERIFICATION_SCHEMA", "sounio.cs6.hapg-full-source-cover-leaf-verification.v1"),
        ("RECEIPT_SHA256", hapg_ledger.receipt_sha256),
        ("PHYSICAL_SHA256", hapg_ledger.physical_sha256),
        ("WAVE_CONTRACT_SHA256", wave.sha256),
        ("HPG_RECEIPT_SHA256", hpg_ledger.receipt_sha256),
        ("HPG_VERIFICATION_SHA256", digest_bytes(hpg_verification_raw)),
        ("LEAF_CHALLENGE", challenge),
        ("MUTATION_TESTS", str(total)),
        ("MUTATIONS_REJECTED", str(rejected)),
        ("LEAF_METHOD", method),
        ("PROBE_PASS", bool_text(computed["PROBE_PASS"])),
        ("AFFINE_CERTIFICATE_PASS", bool_text(computed["AFFINE_CERTIFICATE_PASS"])),
        ("PROJECTIVE_X_CERTIFICATE_PASS", bool_text(computed["PROJECTIVE_X_CERTIFICATE_PASS"])),
        ("PROJECTIVE_Y_CERTIFICATE_PASS", bool_text(computed["PROJECTIVE_Y_CERTIFICATE_PASS"])),
        ("PROJECTIVE_PLUS_CERTIFICATE_PASS", bool_text(computed["PROJECTIVE_PLUS_CERTIFICATE_PASS"])),
        ("PROJECTIVE_MINUS_CERTIFICATE_PASS", bool_text(computed["PROJECTIVE_MINUS_CERTIFICATE_PASS"])),
        ("HOMOGENEOUS_CERTIFICATE_PASS", bool_text(computed["HOMOGENEOUS_CERTIFICATE_PASS"])),
        ("APG_COMPUTATION_VALID", bool_text(computed["APG_COMPUTATION_VALID"])),
        ("APG_CERTIFICATE_PASS", bool_text(computed["APG_CERTIFICATE_PASS"])),
        ("APG_RESCUE", bool_text(computed["APG_RESCUE"])),
        ("GENERIC_CERTIFICATE_PASS", bool_text(computed["CERTIFICATE_PASS"])),
        ("HAPG_TERMINAL_CERTIFIED", bool_text(terminal)),
        ("HAPG_SUBDIVISION_REQUIRED", bool_text(not terminal)),
    )
    for key, value in output:
        print(f"{key}={value}")
    if args.require_probe and not computed["PROBE_PASS"]:
        return 3
    if args.require_hapg_terminal and not terminal:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (CoverVerificationError, HPG_CORE.VerificationError, HAPG_CORE.VerificationError) as error:
        print(f"verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
