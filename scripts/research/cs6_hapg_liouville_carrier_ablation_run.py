#!/usr/bin/env python3
"""Execute the frozen 40 x 3 V7-A Liouville carrier matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
FROZEN_CONTRACT_SHA256 = "decf9089e1dc9aae513f48c48a00e1c815a585b6ba7e9cd1c09b0b514fd58481"
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CELL_DOMAIN = b"sounio.cs6.hapg-liouville-carrier-ablation-cell.v1\0"
ATTEMPT_DOMAIN = b"sounio.cs6.hapg-liouville-carrier-ablation-attempt.v1\0"
CARRIERS = ("C0HOTripletonSet", "C0HORect2Set", "C0Rect2Set")
BASELINE = CARRIERS[0]
ALTERNATIVES = CARRIERS[1:]
CONTRACT_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_contract_v1.txt")
COORDINATES_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_coordinates_v1.tsv")
SOURCE_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
VERIFIER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_verify.py")
PARENT_VERIFIER_REL = Path("scripts/research/cs6_plucker_cocycle_verify.py")
KAT_COORDINATES_REL = Path(
    "scripts/research/receipts/cs6_hapg_full_source_cover_v6_jobs_8469_8470_v1/kat/coordinates.tsv"
)
VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "LIOUVILLE_CARRIER",
    "ATTEMPT_BINDING",
    "RECEIPT_SHA256",
    "CORE_RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "REFERENCE_PHYSICAL_SHA256",
    "INITIAL_HULL_SHA256",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "ALL_FINITE",
    "DETERMINANT_JOINT_COMPATIBLE",
    "C1_DET",
    "C2_HULL_DET",
    "LIOUVILLE_DET",
    "CHART_TUPLE",
    "PROBE_PASS",
    "CERTIFICATE_PASS",
    "SUBDIVISION_REQUIRED",
)


def die(message: str) -> None:
    raise SystemExit(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def bool_token(value: bool) -> str:
    return str(value).lower()


def parse_bool(value: str, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise RuntimeError(f"noncanonical boolean: {label}")


def canonical_kv(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    seen: set[str] = set()
    lines: list[str] = []
    for key, value in fields:
        if not key or not value or key in seen or "=" in key or "\n" in value:
            raise RuntimeError(f"noncanonical KV field: {key}")
        seen.add(key)
        lines.append(f"{key}={value}\n")
    path.write_bytes("".join(lines).encode("ascii"))


def parse_kv(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError(f"noncanonical KV file: {path}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RuntimeError(f"non-ASCII KV file: {path}") from error
    fields: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            raise RuntimeError(f"malformed KV line: {path}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise RuntimeError(f"duplicate or empty KV field: {path}")
        fields[key] = value
    return fields


def parse_verification(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RuntimeError("verifier output is not ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise RuntimeError("verifier output is not canonical")
    lines = text.splitlines()
    if len(lines) != len(VERIFICATION_KEYS):
        raise RuntimeError("verifier output line count mismatch")
    fields: dict[str, str] = {}
    for line, expected in zip(lines, VERIFICATION_KEYS, strict=True):
        if line.count("=") != 1:
            raise RuntimeError("malformed verifier output")
        key, value = line.split("=", 1)
        if key != expected or not value:
            raise RuntimeError(f"verifier output key mismatch: {expected}")
        fields[key] = value
    return fields


def leaf_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def leaf_input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        f"SCHEMA={INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


@dataclass(frozen=True)
class Coordinate:
    ordinal: int
    sample_class: str
    stratum: str
    node_id: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    input_sha256: str
    row_sha256: str


@dataclass(frozen=True)
class Attempt:
    index: int
    coordinate: Coordinate
    carrier: str
    cell_challenge: str
    binding: str

    @property
    def identity(self) -> str:
        return f"A{self.index:04d}"


@dataclass(frozen=True)
class Result:
    attempt: Attempt
    status: str
    worker_rc: int
    elapsed_ms: int
    stdout_sha256: str
    stderr_sha256: str
    verification_sha256: str = ZERO_SHA256
    core_receipt_sha256: str = ZERO_SHA256
    physical_sha256: str = ZERO_SHA256
    reference_sha256: str = ZERO_SHA256
    initial_sha256: str = ZERO_SHA256
    mutation_tests: int = 0
    mutations_rejected: int = 0
    all_finite: bool = False
    determinant_compatible: bool = False
    c1_det: str = "-"
    c2_det: str = "-"
    liouville_det: str = "-"
    chart_tuple: str = "-"
    probe_pass: bool = False
    certificate_pass: bool = False
    subdivision_required: bool = True


def parse_coordinates(path: Path) -> list[Coordinate]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError("coordinate manifest is noncanonical")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RuntimeError("coordinate manifest is not ASCII") from error
    header = (
        "ORDINAL\tSAMPLE_CLASS\tSTRATUM\tNODE_ID\tU_DEPTH\tU_INDEX\t"
        "S_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256"
    )
    try:
        header_index = lines.index(header)
    except ValueError as error:
        raise RuntimeError("coordinate table header missing") from error
    metadata = {}
    for line in lines[:header_index]:
        if line.count("=") != 1:
            raise RuntimeError("malformed coordinate metadata")
        key, value = line.split("=", 1)
        if not key or not value or key in metadata:
            raise RuntimeError("duplicate coordinate metadata")
        metadata[key] = value
    exact_metadata = {
        "SCHEMA": "sounio.cs6.hapg-liouville-carrier-ablation-coordinates.v1",
        "CONTRACT_STATE": "PRE_RESULT_FROZEN",
        "CELL_COUNT": "40",
        "CAPD_SET_TARGET_COUNT": "24",
        "NO_SIGNED_CHART_CONTROL_COUNT": "8",
        "HPG_POSITIVE_CONTROL_COUNT": "8",
        "CARRIER_COUNT": "3",
        "MAXIMUM_EVALUATIONS": "120",
    }
    for key, value in exact_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"coordinate metadata mismatch: {key}")

    coordinates: list[Coordinate] = []
    for raw_line in lines[header_index + 1 :]:
        fields = raw_line.split("\t")
        if len(fields) != 9:
            raise RuntimeError("coordinate row width mismatch")
        ordinal_text, sample_class, stratum, node, *numbers, input_sha = fields
        if any(not token.isdigit() for token in (ordinal_text, *numbers)):
            raise RuntimeError("coordinate integer is noncanonical")
        ordinal = int(ordinal_text)
        u_depth, u_index, s_depth, s_index = (int(token) for token in numbers)
        if ordinal != len(coordinates) + 1:
            raise RuntimeError("coordinate ordinal drift")
        if node != leaf_id(u_depth, u_index, s_depth, s_index):
            raise RuntimeError("coordinate node identity mismatch")
        if SHA_RE.fullmatch(input_sha) is None:
            raise RuntimeError("coordinate input digest is malformed")
        input_raw = leaf_input_bytes(u_depth, u_index, s_depth, s_index)
        if digest_bytes(input_raw) != input_sha:
            raise RuntimeError("coordinate input digest mismatch")
        coordinates.append(
            Coordinate(
                ordinal,
                sample_class,
                stratum,
                node,
                u_depth,
                u_index,
                s_depth,
                s_index,
                input_sha,
                digest_bytes((raw_line + "\n").encode("ascii")),
            )
        )
    if len(coordinates) != 40 or len({item.node_id for item in coordinates}) != 40:
        raise RuntimeError("coordinate manifest is not a unique 40-cell set")
    counts = {label: sum(item.sample_class == label for item in coordinates) for label in (
        "CAPD_SET_TARGET", "NO_SIGNED_CHART_CONTROL", "HPG_POSITIVE_CONTROL"
    )}
    if counts != {
        "CAPD_SET_TARGET": 24,
        "NO_SIGNED_CHART_CONTROL": 8,
        "HPG_POSITIVE_CONTROL": 8,
    }:
        raise RuntimeError("coordinate cohort population mismatch")
    return coordinates


def cell_challenge(
    root: str,
    run_contract_sha256: str,
    coordinate_manifest_sha256: str,
    coordinate: Coordinate,
) -> str:
    return digest_bytes(
        CELL_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
        + b"\0"
        + bytes.fromhex(coordinate_manifest_sha256)
        + b"\0"
        + bytes.fromhex(coordinate.row_sha256)
        + b"\0"
        + bytes.fromhex(coordinate.input_sha256)
    )


def attempt_binding(challenge: str, carrier: str, run_contract_sha256: str) -> str:
    return digest_bytes(
        ATTEMPT_DOMAIN
        + bytes.fromhex(challenge)
        + b"\0"
        + carrier.encode("ascii")
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
    )


def classify_capd_set(stderr: bytes, carrier: str, binding: str) -> bool:
    preamble = (
        f"V7_FAILURE_BINDING LIOUVILLE_CARRIER={carrier} "
        f"ATTEMPT_BINDING={binding}\n"
    ).encode("ascii")
    if not stderr.startswith(preamble):
        return False
    lowered = stderr[len(preamble):].lower()
    return (
        lowered.startswith(
            b"probe error: centeredtripletonset::evalaffinefunctional - empty intersection of rb and rq."
        )
        and b"\nrb=[" in lowered
        and lowered.endswith(b"\nrq=[-nan, -nan]\n\n")
    )


def historical_positive_charts(path: Path, selected: set[str]) -> dict[str, str]:
    lines = path.read_text(encoding="ascii").splitlines()
    header_prefix = "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\t"
    header_index = next((i for i, line in enumerate(lines) if line.startswith(header_prefix)), None)
    if header_index is None:
        raise RuntimeError("KAT coordinate header missing")
    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    result: dict[str, str] = {}
    for row in reader:
        node = row["LEAF_ID"]
        if node not in selected:
            continue
        parts = []
        for event in ("E1_R0", "E1_R1", "E2_R0", "E2_R1"):
            sign = row[f"{event}_SIGN"]
            sign = "+1" if sign == "1" else sign
            parts.append(f"{row[f'{event}_CHART']}:{sign}")
        result[node] = ",".join(parts)
    if set(result) != selected:
        raise RuntimeError("historical KAT chart coverage mismatch")
    return result


def write_attempt_contract(path: Path, attempts: Sequence[Attempt]) -> None:
    columns = (
        "ATTEMPT_INDEX", "ORDINAL", "SAMPLE_CLASS", "STRATUM", "NODE_ID",
        "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "INPUT_SHA256",
        "MANIFEST_ROW_SHA256", "CELL_CHALLENGE", "LIOUVILLE_CARRIER",
        "ATTEMPT_BINDING",
    )
    rows = ["\t".join(columns)]
    for attempt in attempts:
        coordinate = attempt.coordinate
        rows.append(
            "\t".join(
                (
                    str(attempt.index),
                    str(coordinate.ordinal),
                    coordinate.sample_class,
                    coordinate.stratum,
                    coordinate.node_id,
                    str(coordinate.u_depth),
                    str(coordinate.u_index),
                    str(coordinate.s_depth),
                    str(coordinate.s_index),
                    coordinate.input_sha256,
                    coordinate.row_sha256,
                    attempt.cell_challenge,
                    attempt.carrier,
                    attempt.binding,
                )
            )
        )
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


def negative_receipt_fields(result: Result) -> tuple[tuple[str, str], ...]:
    attempt = result.attempt
    coordinate = attempt.coordinate
    return (
        ("SCHEMA", "sounio.cs6.hapg-liouville-carrier-ablation-negative.v1"),
        ("ATTEMPT_INDEX", str(attempt.index)),
        ("NODE_ID", coordinate.node_id),
        ("SAMPLE_CLASS", coordinate.sample_class),
        ("LIOUVILLE_CARRIER", attempt.carrier),
        ("INPUT_SHA256", coordinate.input_sha256),
        ("CELL_CHALLENGE", attempt.cell_challenge),
        ("MANIFEST_ROW_SHA256", coordinate.row_sha256),
        ("ATTEMPT_BINDING", attempt.binding),
        ("WORKER_RC", str(result.worker_rc)),
        ("STDOUT_SHA256", result.stdout_sha256),
        ("STDERR_SHA256", result.stderr_sha256),
        ("CLASS", result.status),
        ("FAILURE_BINDING_AUTHENTICATED", bool_token(result.status == "CAPD_SET_RQ_NAN")),
        ("SCIENTIFIC_NEGATIVE", bool_token(result.status == "CAPD_SET_RQ_NAN")),
    )


def interval_fraction(token: str) -> tuple[Fraction, Fraction]:
    match = INTERVAL_RE.fullmatch(token)
    if match is None:
        raise RuntimeError("malformed interval token")
    lower = Fraction.from_float(float.fromhex(match.group(1)))
    upper = Fraction.from_float(float.fromhex(match.group(2)))
    if lower > upper:
        raise RuntimeError("reversed interval token")
    return lower, upper


def width_score(results: Sequence[Result]) -> tuple[Fraction, Fraction]:
    relative: list[Fraction] = []
    for result in results:
        lower, upper = interval_fraction(result.liouville_det)
        if upper >= 0:
            raise RuntimeError("winner score received nonnegative determinant")
        relative.append((upper - lower) / (-upper))
    return max(relative), sum(relative, Fraction(0))


def fraction_token(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def decide_carrier(
    *,
    baseline_valid: bool,
    protocol_invalid: bool,
    control_ok: bool,
    repair_count: int,
    target_statuses: Sequence[str],
    all_complete: bool,
    all_reference_invariant: bool,
) -> str:
    if not baseline_valid or protocol_invalid:
        return "RUN_INVALID"
    if not control_ok:
        return "NO_GO_CONTROL_REGRESSION"
    if repair_count == 24 and all_complete:
        return "GO" if all_reference_invariant else "RUN_INVALID"
    if repair_count == 0 and all(
        status == "CAPD_SET_RQ_NAN" for status in target_statuses
    ):
        return "NO_GO_ALL_FAILURES"
    if 1 <= repair_count <= 23 and all(
        status in {"VERIFIED_COMPLETE", "CAPD_SET_RQ_NAN"}
        for status in target_statuses
    ):
        return "INCONCLUSIVE_PARTIAL"
    return "RUN_INVALID"


def file_index(root: Path, excluded: set[str]) -> bytes:
    rows: list[str] = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise RuntimeError("symlink forbidden in result tree")
        rows.append(f"{digest(path)}  {relative}\n")
    return "".join(rows).encode("ascii")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root-challenge", required=True)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args(argv)
    if SHA_RE.fullmatch(args.root_challenge) is None:
        die("root challenge must be a lowercase SHA-256")
    if not (1 <= args.jobs <= 120) or not (1 <= args.timeout_seconds <= 3600):
        die("jobs or timeout is outside the frozen execution envelope")
    if not args.self_test_mutations:
        die("mutation self-tests are mandatory for the frozen V7-A protocol")

    repo = args.repo.resolve()
    worker = args.worker.resolve()
    provenance = args.provenance_dir.resolve()
    run_dir = args.run_dir.resolve()
    if not repo.is_dir() or not worker.is_file() or not os.access(worker, os.X_OK):
        die("repo or worker path is invalid")
    required_provenance = (
        "capd-cflags.txt", "capd-libs.txt", "capd-version.txt",
        "compile-command.txt", "compile-stderr.txt", "compile-stdout.txt",
        "compiler-version.txt", "dependencies.sha256",
        "runtime-libraries.sha256", "runtime-linkage.txt",
        "worker-source.cpp", "worker-source.sha256", "worker-binary.sha256",
        "slurm-context.txt", "node-uname.txt", "node-lscpu.txt",
        "node-runtime-linkage.txt", "node-runtime-libraries.sha256",
    )
    if any(not (provenance / name).is_file() for name in required_provenance):
        die("prebuilt provenance directory is incomplete")

    contract_path = repo / CONTRACT_REL
    coordinates_path = repo / COORDINATES_REL
    source_path = repo / SOURCE_REL
    verifier_path = repo / VERIFIER_REL
    parent_verifier_path = repo / PARENT_VERIFIER_REL
    runner_path = Path(__file__).resolve()
    for path in (contract_path, coordinates_path, source_path, verifier_path, parent_verifier_path, runner_path):
        if not path.is_file():
            die(f"required implementation file missing: {path}")
    contract = parse_kv(contract_path)
    coordinates_sha = digest(coordinates_path)
    frozen_contract_sha = digest(contract_path)
    if frozen_contract_sha != FROZEN_CONTRACT_SHA256:
        die("V7 frozen contract digest drift")
    if contract.get("CONTRACT_STATE") != "PRE_RESULT_FROZEN":
        die("V7 contract is not pre-result frozen")
    if contract.get("COORDINATE_MANIFEST_SHA256") != coordinates_sha:
        die("V7 coordinate manifest hash differs from frozen contract")
    if contract.get("ROOT_CHALLENGE") != args.root_challenge:
        die("root challenge differs from frozen contract")
    coordinates = parse_coordinates(coordinates_path)
    historical_kat_path = repo / KAT_COORDINATES_REL
    if digest(historical_kat_path) != contract.get("V6_KAT_COORDINATES_SHA256"):
        die("historical KAT coordinate digest differs from frozen contract")

    if (provenance / "capd-version.txt").read_text(encoding="ascii") != "5.3.0\n":
        die("prebuilt CAPD version is not frozen 5.3.0")
    cflags = (provenance / "capd-cflags.txt").read_text(encoding="ascii")
    if "-D__USE_FILIB__" not in cflags or "-frounding-math" not in cflags:
        die("prebuilt CAPD flags do not bind FILIB outward rounding")
    source_sha = digest(source_path)
    if (provenance / "worker-source.cpp").read_bytes() != source_path.read_bytes():
        die("prebuilt source snapshot differs from repository source")
    if (provenance / "worker-source.sha256").read_text(encoding="ascii") != source_sha + "\n":
        die("prebuilt source digest mismatch")
    worker_sha = digest(worker)
    if (provenance / "worker-binary.sha256").read_text(encoding="ascii") != worker_sha + "\n":
        die("prebuilt binary digest mismatch")

    if run_dir.exists():
        die("run directory already exists")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cs6-v7-carrier.", dir=run_dir.parent))
    complete = False
    try:
        for directory in (
            "inputs", "receipts", "negative-receipts", "stderr",
            "verifications", "verifier-stderr", "provenance",
        ):
            (work / directory).mkdir()
        snapshots = (
            (contract_path, work / "frozen-contract.txt"),
            (coordinates_path, work / "coordinates.tsv"),
            (source_path, work / "worker-source.cpp"),
            (verifier_path, work / "attempt-verifier.py"),
            (parent_verifier_path, work / "cs6_plucker_cocycle_verify.py"),
            (runner_path, work / "runner.py"),
            (repo / KAT_COORDINATES_REL, work / "historical-kat-coordinates.tsv"),
        )
        for source, target in snapshots:
            shutil.copy2(source, target)
        provenance_files = sorted(provenance.iterdir(), key=lambda path: path.name)
        if any(path.is_symlink() or not path.is_file() for path in provenance_files):
            raise RuntimeError("prebuilt provenance must contain regular files only")
        for path in provenance_files:
            if path.name == "worker-binary":
                continue
            shutil.copy2(path, work / "provenance" / path.name)

        git_head = subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "-C", repo, "status", "--short", "--untracked-files=all"],
            check=True, capture_output=True,
        ).stdout
        (work / "git-head.txt").write_text(git_head + "\n", encoding="ascii")
        (work / "git-status.txt").write_bytes(git_status)
        (work / "python-version.txt").write_bytes(
            subprocess.run([sys.executable, "--version"], check=True, capture_output=True).stdout
        )

        provenance_index = file_index(work / "provenance", set())
        (work / "provenance.sha256").write_bytes(provenance_index)
        run_contract_fields = (
            ("SCHEMA", "sounio.cs6.hapg-liouville-carrier-ablation-run-contract.v1"),
            ("FROZEN_CONTRACT_SHA256", frozen_contract_sha),
            ("COORDINATE_MANIFEST_SHA256", coordinates_sha),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("GIT_HEAD", git_head),
            ("WORKER_SOURCE_SHA256", source_sha),
            ("WORKER_BINARY_SHA256", worker_sha),
            ("ATTEMPT_VERIFIER_SHA256", digest(verifier_path)),
            ("PARENT_VERIFIER_SHA256", digest(parent_verifier_path)),
            ("RUNNER_SHA256", digest(runner_path)),
            ("PROVENANCE_INDEX_SHA256", digest_bytes(provenance_index)),
            ("HISTORICAL_KAT_COORDINATES_SHA256", digest(historical_kat_path)),
            ("CELL_COUNT", "40"),
            ("CARRIER_COUNT", "3"),
            ("ATTEMPT_COUNT", "120"),
            ("ATTEMPT_FREEZE_ORDER", "EXACT_120_ATTEMPTS_BEFORE_ANY_WORKER"),
            ("JOBS", str(args.jobs)),
            ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
            ("SELF_TEST_MUTATIONS", bool_token(args.self_test_mutations)),
            ("CAPD_VERSION", "5.3.0"),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("FPGA_EXECUTION", "false"),
            ("EXECUTION_PROVENANCE_ATTESTED", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        )
        canonical_kv(work / "run-contract.txt", run_contract_fields)
        run_contract_sha = digest(work / "run-contract.txt")

        attempts: list[Attempt] = []
        for coordinate in coordinates:
            challenge = cell_challenge(
                args.root_challenge, run_contract_sha, coordinates_sha, coordinate
            )
            for carrier in CARRIERS:
                attempts.append(
                    Attempt(
                        len(attempts) + 1,
                        coordinate,
                        carrier,
                        challenge,
                        attempt_binding(challenge, carrier, run_contract_sha),
                    )
                )
        if len(attempts) != 120:
            raise RuntimeError("attempt matrix cardinality mismatch")
        write_attempt_contract(work / "attempt-contract.tsv", attempts)

        for coordinate in coordinates:
            raw = leaf_input_bytes(
                coordinate.u_depth, coordinate.u_index,
                coordinate.s_depth, coordinate.s_index,
            )
            if digest_bytes(raw) != coordinate.input_sha256:
                raise RuntimeError("input reconstruction drift after attempt freeze")
            (work / "inputs" / f"{coordinate.node_id}.txt").write_bytes(raw)

        python = Path(sys.executable).resolve()
        verifier_snapshot = work / "attempt-verifier.py"

        def run_attempt(attempt: Attempt) -> Result:
            coordinate = attempt.coordinate
            identity = attempt.identity
            receipt_path = work / "receipts" / f"{identity}.txt"
            stderr_path = work / "stderr" / f"{identity}.txt"
            verification_path = work / "verifications" / f"{identity}.txt"
            verifier_stderr_path = work / "verifier-stderr" / f"{identity}.txt"
            command = [
                str(worker),
                str(coordinate.u_depth), str(coordinate.u_index),
                str(coordinate.s_depth), str(coordinate.s_index),
                coordinate.input_sha256, attempt.cell_challenge, attempt.carrier,
                frozen_contract_sha, coordinates_sha, run_contract_sha,
                coordinate.row_sha256, attempt.binding,
            ]
            started = time.monotonic_ns()
            try:
                completed_process = subprocess.run(
                    command, capture_output=True, timeout=args.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                elapsed = (time.monotonic_ns() - started) // 1_000_000
                stdout = error.stdout or b""
                stderr = error.stderr or b""
                receipt_path.write_bytes(stdout)
                stderr_path.write_bytes(stderr)
                return Result(
                    attempt, "TIMEOUT", 124, elapsed,
                    digest_bytes(stdout), digest_bytes(stderr),
                )
            elapsed = (time.monotonic_ns() - started) // 1_000_000
            stdout = completed_process.stdout
            stderr = completed_process.stderr
            receipt_path.write_bytes(stdout)
            stderr_path.write_bytes(stderr)
            base = Result(
                attempt,
                "UNKNOWN_FAILURE",
                completed_process.returncode,
                elapsed,
                digest_bytes(stdout),
                digest_bytes(stderr),
            )
            if completed_process.returncode != 0:
                status = "CAPD_SET_RQ_NAN" if (
                    completed_process.returncode == 1
                    and not stdout
                    and classify_capd_set(stderr, attempt.carrier, attempt.binding)
                ) else "UNKNOWN_FAILURE"
                result = replace(base, status=status)
                canonical_kv(
                    work / "negative-receipts" / f"{identity}.txt",
                    negative_receipt_fields(result),
                )
                return result
            if stderr or not stdout:
                return base

            verification_command = [
                python, "-B", verifier_snapshot, receipt_path,
                "--source-sha", source_sha,
                "--input", work / "inputs" / f"{coordinate.node_id}.txt",
                "--challenge", attempt.cell_challenge,
                "--carrier", attempt.carrier,
                "--frozen-contract-sha", frozen_contract_sha,
                "--coordinate-manifest-sha", coordinates_sha,
                "--run-contract-sha", run_contract_sha,
                "--manifest-row-sha", coordinate.row_sha256,
                "--attempt-binding", attempt.binding,
            ]
            if args.self_test_mutations:
                verification_command.append("--self-test-mutations")
            verification = subprocess.run(verification_command, capture_output=True)
            verification_path.write_bytes(verification.stdout)
            verifier_stderr_path.write_bytes(verification.stderr)
            if verification.returncode != 0 or verification.stderr:
                return replace(
                    base,
                    status="VERIFIER_FAILURE",
                    verification_sha256=digest_bytes(verification.stdout),
                )
            fields = parse_verification(verification.stdout)
            mutation_tests = int(fields["MUTATION_TESTS"])
            mutations_rejected = int(fields["MUTATIONS_REJECTED"])
            if fields["LIOUVILLE_CARRIER"] != attempt.carrier:
                raise RuntimeError("verified carrier mismatch")
            if fields["ATTEMPT_BINDING"] != attempt.binding:
                raise RuntimeError("verified attempt binding mismatch")
            if fields["RECEIPT_SHA256"] != digest_bytes(stdout):
                raise RuntimeError("verified receipt hash mismatch")
            return Result(
                attempt=attempt,
                status="VERIFIED_COMPLETE",
                worker_rc=0,
                elapsed_ms=elapsed,
                stdout_sha256=digest_bytes(stdout),
                stderr_sha256=EMPTY_SHA256,
                verification_sha256=digest_bytes(verification.stdout),
                core_receipt_sha256=fields["CORE_RECEIPT_SHA256"],
                physical_sha256=fields["PHYSICAL_SHA256"],
                reference_sha256=fields["REFERENCE_PHYSICAL_SHA256"],
                initial_sha256=fields["INITIAL_HULL_SHA256"],
                mutation_tests=mutation_tests,
                mutations_rejected=mutations_rejected,
                all_finite=parse_bool(fields["ALL_FINITE"], "ALL_FINITE"),
                determinant_compatible=parse_bool(
                    fields["DETERMINANT_JOINT_COMPATIBLE"],
                    "DETERMINANT_JOINT_COMPATIBLE",
                ),
                c1_det=fields["C1_DET"],
                c2_det=fields["C2_HULL_DET"],
                liouville_det=fields["LIOUVILLE_DET"],
                chart_tuple=fields["CHART_TUPLE"],
                probe_pass=parse_bool(fields["PROBE_PASS"], "PROBE_PASS"),
                certificate_pass=parse_bool(
                    fields["CERTIFICATE_PASS"], "CERTIFICATE_PASS"
                ),
                subdivision_required=parse_bool(
                    fields["SUBDIVISION_REQUIRED"], "SUBDIVISION_REQUIRED"
                ),
            )

        results: list[Result] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_attempt, attempt): attempt for attempt in attempts}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: result.attempt.index)
        if len(results) != 120:
            raise RuntimeError("result matrix cardinality mismatch")

        matrix = {(result.attempt.coordinate.node_id, result.attempt.carrier): result for result in results}
        if len(matrix) != 120:
            raise RuntimeError("result matrix identity collision")
        unknown = [result for result in results if result.status in {"UNKNOWN_FAILURE", "TIMEOUT", "VERIFIER_FAILURE"}]
        target_coordinates = [item for item in coordinates if item.sample_class == "CAPD_SET_TARGET"]
        no_chart_coordinates = [item for item in coordinates if item.sample_class == "NO_SIGNED_CHART_CONTROL"]
        positive_coordinates = [item for item in coordinates if item.sample_class == "HPG_POSITIVE_CONTROL"]
        historical = historical_positive_charts(
            work / "historical-kat-coordinates.tsv",
            {item.node_id for item in positive_coordinates},
        )

        baseline_target = [matrix[(item.node_id, BASELINE)] for item in target_coordinates]
        baseline_no_chart = [matrix[(item.node_id, BASELINE)] for item in no_chart_coordinates]
        baseline_positive = [matrix[(item.node_id, BASELINE)] for item in positive_coordinates]
        baseline_valid = (
            all(result.status == "CAPD_SET_RQ_NAN" for result in baseline_target)
            and all(
                result.status == "VERIFIED_COMPLETE" and not result.probe_pass
                for result in baseline_no_chart
            )
            and all(
                result.status == "VERIFIED_COMPLETE"
                and result.probe_pass
                and result.chart_tuple == historical[result.attempt.coordinate.node_id]
                for result in baseline_positive
            )
        )

        group_invariance: dict[str, bool] = {}
        group_mismatch: dict[str, bool] = {}
        for coordinate in coordinates:
            successes = [
                matrix[(coordinate.node_id, carrier)]
                for carrier in CARRIERS
                if matrix[(coordinate.node_id, carrier)].status == "VERIFIED_COMPLETE"
            ]
            group_invariance[coordinate.node_id] = (
                len(successes) >= 2
                and len({result.reference_sha256 for result in successes}) == 1
                and len({result.initial_sha256 for result in successes}) == 1
            )
            group_mismatch[coordinate.node_id] = (
                len(successes) >= 2 and not group_invariance[coordinate.node_id]
            )

        completed_results = [
            result for result in results if result.status == "VERIFIED_COMPLETE"
        ]
        mutation_ok = all(
            result.mutation_tests > 0
            and result.mutation_tests == result.mutations_rejected
            for result in completed_results
        )
        physical_ok = all(
            result.all_finite and result.determinant_compatible
            for result in completed_results
        )
        protocol_invalid = (
            bool(unknown)
            or not mutation_ok
            or not physical_ok
            or any(group_mismatch.values())
        )

        decisions: dict[str, str] = {BASELINE: "BASELINE_VALID" if baseline_valid else "RUN_INVALID"}
        repairs: dict[str, int] = {BASELINE: 0}
        scores: dict[str, tuple[Fraction, Fraction]] = {}
        for carrier in ALTERNATIVES:
            carrier_results = [matrix[(item.node_id, carrier)] for item in coordinates]
            targets = [matrix[(item.node_id, carrier)] for item in target_coordinates]
            no_chart = [matrix[(item.node_id, carrier)] for item in no_chart_coordinates]
            positive = [matrix[(item.node_id, carrier)] for item in positive_coordinates]
            repair_count = sum(result.status == "VERIFIED_COMPLETE" for result in targets)
            repairs[carrier] = repair_count
            control_ok = (
                all(
                    result.status == "VERIFIED_COMPLETE"
                    and not result.probe_pass
                    and result.chart_tuple
                    == matrix[(result.attempt.coordinate.node_id, BASELINE)].chart_tuple
                    for result in no_chart
                )
                and all(
                    result.status == "VERIFIED_COMPLETE"
                    and result.probe_pass
                    and result.chart_tuple
                    == historical[result.attempt.coordinate.node_id]
                    for result in positive
                )
            )
            decision = decide_carrier(
                baseline_valid=baseline_valid,
                protocol_invalid=protocol_invalid,
                control_ok=control_ok,
                repair_count=repair_count,
                target_statuses=[result.status for result in targets],
                all_complete=all(
                    result.status == "VERIFIED_COMPLETE" for result in carrier_results
                ),
                all_reference_invariant=all(
                    group_invariance[item.node_id] for item in coordinates
                ),
            )
            if decision == "GO":
                scores[carrier] = width_score(carrier_results)
            decisions[carrier] = decision

        go_carriers = [carrier for carrier in ALTERNATIVES if decisions[carrier] == "GO"]
        tie_rank = {"C0HORect2Set": 0, "C0Rect2Set": 1}
        winner = "NONE"
        if go_carriers:
            winner = min(
                go_carriers,
                key=lambda carrier: (*scores[carrier], tie_rank[carrier]),
            )
        run_valid = baseline_valid and not protocol_invalid and all(
            decisions[carrier] != "RUN_INVALID" for carrier in ALTERNATIVES
        )

        result_columns = (
            "ATTEMPT_INDEX", "ORDINAL", "SAMPLE_CLASS", "STRATUM", "NODE_ID",
            "LIOUVILLE_CARRIER", "STATUS", "WORKER_RC", "ELAPSED_MS",
            "INPUT_SHA256", "MANIFEST_ROW_SHA256", "CELL_CHALLENGE",
            "ATTEMPT_BINDING", "STDOUT_SHA256", "STDERR_SHA256",
            "VERIFICATION_SHA256", "CORE_RECEIPT_SHA256", "PHYSICAL_SHA256",
            "REFERENCE_PHYSICAL_SHA256", "INITIAL_HULL_SHA256",
            "MUTATION_TESTS", "MUTATIONS_REJECTED", "ALL_FINITE",
            "DETERMINANT_JOINT_COMPATIBLE", "C1_DET", "C2_HULL_DET",
            "LIOUVILLE_DET", "CHART_TUPLE", "PROBE_PASS",
            "CERTIFICATE_PASS", "SUBDIVISION_REQUIRED",
        )
        result_rows = ["\t".join(result_columns)]
        for result in results:
            attempt = result.attempt
            coordinate = attempt.coordinate
            result_rows.append(
                "\t".join(
                    (
                        str(attempt.index), str(coordinate.ordinal),
                        coordinate.sample_class, coordinate.stratum,
                        coordinate.node_id, attempt.carrier, result.status,
                        str(result.worker_rc), str(result.elapsed_ms),
                        coordinate.input_sha256, coordinate.row_sha256,
                        attempt.cell_challenge, attempt.binding,
                        result.stdout_sha256, result.stderr_sha256,
                        result.verification_sha256, result.core_receipt_sha256,
                        result.physical_sha256, result.reference_sha256,
                        result.initial_sha256, str(result.mutation_tests),
                        str(result.mutations_rejected), bool_token(result.all_finite),
                        bool_token(result.determinant_compatible), result.c1_det,
                        result.c2_det, result.liouville_det, result.chart_tuple,
                        bool_token(result.probe_pass), bool_token(result.certificate_pass),
                        bool_token(result.subdivision_required),
                    )
                )
            )
        (work / "results.tsv").write_text("\n".join(result_rows) + "\n", encoding="ascii")

        decision_rows = [
            "LIOUVILLE_CARRIER\tROLE\tDECISION\tCAPD_REPAIRS\tMAX_RELATIVE_WIDTH\tSUM_RELATIVE_WIDTH"
        ]
        roles = {
            "C0HOTripletonSet": "BASELINE_HO_TRIPLETON",
            "C0HORect2Set": "CAUSAL_HO_DOUBLETON_ABLATION",
            "C0Rect2Set": "TAYLOR_DOUBLETON_ROBUSTNESS_CONTROL",
        }
        for carrier in CARRIERS:
            score = scores.get(carrier)
            decision_rows.append(
                "\t".join(
                    (
                        carrier, roles[carrier], decisions[carrier],
                        str(repairs[carrier]),
                        "-" if score is None else fraction_token(score[0]),
                        "-" if score is None else fraction_token(score[1]),
                    )
                )
            )
        (work / "decisions.tsv").write_text("\n".join(decision_rows) + "\n", encoding="ascii")

        summary_fields = (
            ("SCHEMA", "sounio.cs6.hapg-liouville-carrier-ablation-summary.v1"),
            ("RUN_COMPLETE", "true"),
            ("RUN_VALID", bool_token(run_valid)),
            ("FROZEN_CONTRACT_SHA256", frozen_contract_sha),
            ("COORDINATE_MANIFEST_SHA256", coordinates_sha),
            ("RUN_CONTRACT_SHA256", run_contract_sha),
            ("ATTEMPT_CONTRACT_SHA256", digest(work / "attempt-contract.tsv")),
            ("CELL_COUNT", "40"),
            ("ATTEMPT_COUNT", "120"),
            ("VERIFIED_COMPLETE_COUNT", str(sum(result.status == "VERIFIED_COMPLETE" for result in results))),
            ("CAPD_SET_RQ_NAN_COUNT", str(sum(result.status == "CAPD_SET_RQ_NAN" for result in results))),
            ("UNKNOWN_FAILURE_COUNT", str(len(unknown))),
            ("BASELINE_PREREQUISITE_VALID", bool_token(baseline_valid)),
            ("PROTOCOL_INVALID", bool_token(protocol_invalid)),
            ("C0HOTripletonSet_DECISION", decisions["C0HOTripletonSet"]),
            ("C0HORect2Set_DECISION", decisions["C0HORect2Set"]),
            ("C0Rect2Set_DECISION", decisions["C0Rect2Set"]),
            ("C0HORect2Set_CAPD_REPAIRS", str(repairs["C0HORect2Set"])),
            ("C0Rect2Set_CAPD_REPAIRS", str(repairs["C0Rect2Set"])),
            ("GO_CARRIER_COUNT", str(len(go_carriers))),
            ("V7_B_WINNER", winner),
            ("MUTATION_TESTS", str(sum(result.mutation_tests for result in results))),
            ("MUTATIONS_REJECTED", str(sum(result.mutations_rejected for result in results))),
            ("REFERENCE_INVARIANCE_CELL_COUNT", str(sum(group_invariance.values()))),
            ("REFERENCE_INVARIANCE_UNWITNESSED_COUNT", str(sum(not value for value in group_invariance.values()))),
            ("TOTAL_WORKER_ELAPSED_MS", str(sum(result.elapsed_ms for result in results))),
            ("FPGA_EXECUTION", "false"),
            ("EXECUTION_PROVENANCE_ATTESTED", "false"),
            ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ("HYPERBOLICITY_PROVED", "false"),
            ("CHAOTIC_ATTRACTOR_PROVED", "false"),
            ("OPEN_PROBLEM_SOLVED", "false"),
            ("NOVELTY_OR_PRIORITY_CLAIMED", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        )
        canonical_kv(work / "summary.txt", summary_fields)

        result_index = file_index(work, {"result-files.sha256", "run-manifest.txt", "files.sha256"})
        (work / "result-files.sha256").write_bytes(result_index)
        manifest_fields = (
            ("SCHEMA", "sounio.cs6.hapg-liouville-carrier-ablation-run-manifest.v1"),
            ("RUN_COMPLETE", "true"),
            ("RUN_VALID", bool_token(run_valid)),
            ("GIT_HEAD", git_head),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("WORKER_SOURCE_SHA256", source_sha),
            ("WORKER_BINARY_SHA256", worker_sha),
            ("FROZEN_CONTRACT_SHA256", frozen_contract_sha),
            ("COORDINATE_MANIFEST_SHA256", coordinates_sha),
            ("RUN_CONTRACT_SHA256", run_contract_sha),
            ("ATTEMPT_CONTRACT_SHA256", digest(work / "attempt-contract.tsv")),
            ("RESULTS_SHA256", digest(work / "results.tsv")),
            ("DECISIONS_SHA256", digest(work / "decisions.tsv")),
            ("SUMMARY_SHA256", digest(work / "summary.txt")),
            ("RESULT_FILES_INDEX_SHA256", digest_bytes(result_index)),
            ("ATTEMPT_COUNT", "120"),
            ("GO_CARRIER_COUNT", str(len(go_carriers))),
            ("V7_B_WINNER", winner),
            ("PROMOTION_ELIGIBLE", "false"),
        )
        canonical_kv(work / "run-manifest.txt", manifest_fields)
        final_index = file_index(work, {"files.sha256"})
        (work / "files.sha256").write_bytes(final_index)
        os.replace(work, run_dir)
        complete = True
    finally:
        if not complete:
            if args.keep_failed:
                print(f"FAILED_WORK_DIR={work}", file=sys.stderr)
            else:
                shutil.rmtree(work, ignore_errors=True)

    summary = parse_kv(run_dir / "summary.txt")
    print(f"RUN_DIR={run_dir}")
    for key in (
        "RUN_VALID", "ATTEMPT_COUNT", "VERIFIED_COMPLETE_COUNT",
        "CAPD_SET_RQ_NAN_COUNT", "BASELINE_PREREQUISITE_VALID",
        "C0HORect2Set_DECISION", "C0Rect2Set_DECISION", "V7_B_WINNER",
        "MUTATION_TESTS", "MUTATIONS_REJECTED",
    ):
        print(f"{key}={summary[key]}")
    return 0 if summary["RUN_VALID"] == "true" else 2


if __name__ == "__main__":
    raise SystemExit(main())
