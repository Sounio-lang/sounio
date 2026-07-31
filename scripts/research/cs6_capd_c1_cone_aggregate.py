#!/usr/bin/env python3
"""Validate a full-source CS6 C1 ledger and aggregate global cone hulls."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


EDGES = ("N0->N0", "N0->N1", "N1->N0")
SOURCE_GEOMETRY = {
    "N0": (Fraction("0"), Fraction("0.004"), Fraction("0.3")),
    "N1": (
        Fraction("0.019771776972779206"),
        Fraction("0.0015"),
        Fraction("0.3"),
    ),
}
Q = {
    "N0": (Fraction("1"), Fraction("2.3023784599059653")),
    "N1": (
        Fraction("0.06526711140171336"),
        Fraction("2.3023784599059653"),
    ),
}
C0_EXPECTED_CONTRACT = {
    "CERTIFICATE_KIND": "CAPD_RIGOROUS_COVERING_AGGREGATE_V1",
    "CAPD_SOURCE_TREE_DECLARED": "capd-5.3.0",
    "INTERVAL_BACKEND_DECLARED": "FILIB",
    "MAP": "P^6",
    "SECTION_ORIENTATION": "MinusPlus",
    "ORDER": "8",
    "ZSEC": "[22.327463739099997, 22.327463739100004]",
    "ORIGIN": (
        "{[15.186446520640784, 15.186446520640787],"
        "[10.908543194765464, 10.908543194765468]}"
    ),
    "UNSTABLE": (
        "{[-0.6743031621419977, -0.67430316214199748],"
        "[-0.73845463335624284, -0.73845463335624262]}"
    ),
    "STABLE": (
        "{[-0.94170446778164529, -0.94170446778164507],"
        "[0.33644122125579118, 0.33644122125579129]}"
    ),
    "FRAME_DETERMINANT": "[-0.92226940685332637, -0.9222694068533257]",
    "N0_LOCAL": (
        "{[0, 0],[0, 0],[0.0039999999999999992, 0.004000000000000001],"
        "[0.29999999999999993, 0.30000000000000004]}"
    ),
    "N1_LOCAL": (
        "{[0.019771776972779202, 0.019771776972779209],[0, 0],"
        "[0.0014999999999999998, 0.0015000000000000002],"
        "[0.29999999999999993, 0.30000000000000004]}"
    ),
    "HSETS_DISJOINT": "true",
    "FRAME_RIGOROUSLY_INVERTIBLE": "true",
    "GRID": "N0_U:200,N1_U:75,SUPPORT_S:75,EXIT_S:1200",
    "EXECUTION_TRUST_MODEL": "AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION",
    "REMOTE_ATTESTATION_PRESENT": "false",
    "INDEPENDENT_REPLAY_REQUIRED": "true",
    "LEDGER_RECORDS": "42825",
    "ADJACENCY_MATRIX": "[[1,1],[1,0]]",
    "FIBONACCI_COVERINGS_PROVED": "true",
    "POSITIVE_ENTROPY_PROVED": "true",
    "POINCARE_ENTROPY_LOWER_BOUND_FORMULA": "log(phi)/6",
    "UNIFORM_HYPERBOLICITY_PROVED": "false",
    "CHAOTIC_ATTRACTOR_PROVED": "false",
    "FLOW_ENTROPY_BOUND_PROVED": "false",
}
C0_COVERING_RELATIONS = (
    "COVERING_RELATION=N0->N0 DEGREE=-1 PROVED=true",
    "COVERING_RELATION=N0->N1 DEGREE=-1 PROVED=true",
    "COVERING_RELATION=N1->N0 DEGREE=1 PROVED=true",
)
C1_VECTOR_FIELD = (
    "par:zs;var:x,y,w;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs;"
)
C1_LIOUVILLE_FIELD = (
    "par:zs;var:x,y,w,ell;fun:2*y*y-x*y,x*y-y*(w+zs)/2,x*y-w-zs,"
    "x-y-(w+zs)/2-1;"
)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
GRID_RE = re.compile(r"^N0_U:(\d+),N1_U:(\d+),S:(\d+)$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^]]+)\]$")
HEX_BOUND_RE = re.compile(
    r"^([+-]?)0x([01])(?:\.([0-9a-f]{1,13}))?p([+-](?:0|[1-9][0-9]{0,3}))$"
)
LEDGER_INTERVAL_FIELDS = (
    "SOURCE_U", "SOURCE_S", "A00", "A01", "A10", "A11",
    "TILE_M00", "TILE_DET_M_NAIVE", "TILE_DET_M_EXPANDED",
    "C1_RETURN_TIME", "INTEGRAL_DIVERGENCE", "EXP_INTEGRAL_DIVERGENCE",
    "DET_LIOUVILLE",
    "NU0", "NU1", "NU2", "NU3", "NU4", "NU5", "NU6",
    "T1", "T2", "T3", "T4", "T5", "T6",
)
LEDGER_REQUIRED_FIELDS = frozenset(LEDGER_INTERVAL_FIELDS) | {
    "SOURCE", "TARGET", "EDGE", "U_INDEX", "S_INDEX",
    "TILE_CONE_DIAGNOSTIC", "LIOUVILLE_INVERTIBLE",
}


@dataclass(frozen=True)
class RationalInterval:
    lower: Fraction
    upper: Fraction

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("reversed rational interval")


@dataclass
class Hull:
    lower: Fraction | None = None
    upper: Fraction | None = None
    lower_key: tuple[str, int, int] | None = None
    upper_key: tuple[str, int, int] | None = None

    def include(self, value: RationalInterval, key: tuple[str, int, int]) -> None:
        if self.lower is None or value.lower < self.lower:
            self.lower = value.lower
            self.lower_key = key
        if self.upper is None or value.upper > self.upper:
            self.upper = value.upper
            self.upper_key = key

    def interval(self) -> RationalInterval:
        if self.lower is None or self.upper is None:
            raise SystemExit("empty derivative hull")
        return RationalInterval(self.lower, self.upper)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256_manifest(path: Path, label: str) -> set[Path]:
    seen: set[Path] = set()
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise SystemExit(f"empty {label} manifest")
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise SystemExit(f"malformed {label} manifest line")
        artifact = Path(match.group(2))
        canonical = artifact.resolve()
        if canonical in seen or not artifact.is_file():
            raise SystemExit(f"invalid {label} manifest artifact: {artifact}")
        seen.add(canonical)
        if sha256_file(artifact) != match.group(1):
            raise SystemExit(f"{label} artifact hash mismatch: {artifact}")
    return seen


def linkage_paths(text: str) -> set[Path]:
    paths: set[Path] = set()
    for line in text.splitlines():
        match = re.search(r"=> (/[^ ]+)", line)
        if match is None:
            match = re.match(r"\s*(/[^ ]+)", line)
        if match is not None:
            paths.add(Path(match.group(1)).resolve())
    return paths


def selector_expected(total: int, index: int, count: int) -> int:
    if index >= total:
        return 0
    return (total - 1 - index) // count + 1


def parse_nonnegative_int(text: str | None, field: str) -> int:
    if text is None or re.fullmatch(r"0|[1-9][0-9]*", text) is None:
        raise SystemExit(f"invalid {field}: {text}")
    return int(text)


def parse_unique_values(text: str, path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith(("EDGE=", "COVERING_RELATION=")):
            continue
        key, value = line.split("=", 1)
        if key in result:
            raise SystemExit(f"{path}: duplicate key {key}")
        result[key] = value
    return result


def parse_interval(
    text: str, field: str, *, require_hex: bool = False
) -> RationalInterval:
    match = INTERVAL_RE.fullmatch(text)
    if match is None:
        raise SystemExit(f"invalid {field} interval: {text}")
    try:
        def parse_bound(value: str) -> Fraction:
            hex_match = HEX_BOUND_RE.fullmatch(value)
            if require_hex and hex_match is None:
                raise ValueError("ledger endpoint is not canonical hexadecimal binary64")
            if hex_match is not None:
                decoded = float.fromhex(value)
                if not decoded == decoded or decoded in (float("inf"), float("-inf")):
                    raise ValueError("non-finite hexadecimal endpoint")
                fractional_digits = hex_match.group(3) or ""
                exact_mantissa = Fraction(
                    int(hex_match.group(2) + fractional_digits, 16),
                    16 ** len(fractional_digits),
                )
                exponent = int(hex_match.group(4))
                if exact_mantissa == 0:
                    if fractional_digits or exponent != 0:
                        raise ValueError("non-canonical hexadecimal zero")
                elif exponent < -1022 or exponent > 1023:
                    raise ValueError("hexadecimal exponent outside binary64 format")
                exact_literal = (
                    exact_mantissa * (2 ** exponent)
                    if exponent >= 0
                    else exact_mantissa / (2 ** (-exponent))
                )
                if hex_match.group(1) == "-":
                    exact_literal = -exact_literal
                exact_binary64 = Fraction.from_float(decoded)
                if exact_literal != exact_binary64:
                    raise ValueError("hexadecimal endpoint is not exactly binary64")
                return exact_binary64
            return Fraction(value)

        return RationalInterval(parse_bound(match.group(1)), parse_bound(match.group(2)))
    except (OverflowError, ValueError, ZeroDivisionError) as error:
        raise SystemExit(f"invalid {field} interval: {text}") from error


def interval_add(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    return RationalInterval(left.lower + right.lower, left.upper + right.upper)


def interval_sub(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    return RationalInterval(left.lower - right.upper, left.upper - right.lower)


def interval_mul(left: RationalInterval, right: RationalInterval) -> RationalInterval:
    products = (
        left.lower * right.lower,
        left.lower * right.upper,
        left.upper * right.lower,
        left.upper * right.upper,
    )
    return RationalInterval(min(products), max(products))


def interval_square(value: RationalInterval) -> RationalInterval:
    if value.lower <= 0 <= value.upper:
        return RationalInterval(Fraction(0), max(value.lower**2, value.upper**2))
    return RationalInterval(min(value.lower**2, value.upper**2), max(value.lower**2, value.upper**2))


def interval_scale(value: RationalInterval, scale: Fraction) -> RationalInterval:
    if scale < 0:
        raise ValueError("negative scale")
    return RationalInterval(scale * value.lower, scale * value.upper)


def interval_div_positive(
    numerator: RationalInterval, denominator: RationalInterval
) -> RationalInterval:
    if numerator.lower <= 0 or denominator.lower <= 0:
        raise ValueError("positive interval division requires positive operands")
    return interval_mul(
        numerator,
        RationalInterval(Fraction(1, denominator.upper), Fraction(1, denominator.lower)),
    )


def overlaps(left: RationalInterval, right: RationalInterval) -> bool:
    return left.lower <= right.upper and right.lower <= left.upper


def contains(outer: RationalInterval, inner: RationalInterval) -> bool:
    return outer.lower <= inner.lower and inner.upper <= outer.upper


def reject_symlink_tree(root: Path, label: str) -> None:
    if root.is_symlink():
        raise SystemExit(f"{label} contains symlink: {root}")
    for directory, names, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in (*names, *files):
            path = base / name
            if path.is_symlink():
                raise SystemExit(f"{label} contains symlink: {path}")


def cone_intervals(
    edge: str, entries: tuple[RationalInterval, RationalInterval, RationalInterval, RationalInterval]
) -> tuple[RationalInterval, RationalInterval, RationalInterval, RationalInterval, RationalInterval]:
    source, target = edge.split("->")
    p, q, r, s = entries
    source_positive, source_negative = Q[source]
    target_positive, target_negative = Q[target]
    m00 = interval_sub(
        interval_sub(
            interval_scale(interval_square(p), target_positive),
            interval_scale(interval_square(r), target_negative),
        ),
        RationalInterval(source_positive, source_positive),
    )
    m01 = interval_sub(
        interval_scale(interval_mul(p, q), target_positive),
        interval_scale(interval_mul(r, s), target_negative),
    )
    m11 = interval_add(
        interval_sub(
            interval_scale(interval_square(q), target_positive),
            interval_scale(interval_square(s), target_negative),
        ),
        RationalInterval(source_negative, source_negative),
    )
    determinant_naive = interval_sub(interval_mul(m00, m11), interval_square(m01))
    map_determinant = interval_sub(interval_mul(p, s), interval_mul(q, r))
    determinant_expanded = interval_sub(
        interval_add(
            interval_sub(
                interval_add(
                    interval_sub(
                        interval_scale(interval_square(p), target_positive * source_negative),
                        interval_scale(interval_square(r), target_negative * source_negative),
                    ),
                    interval_scale(interval_square(s), source_positive * target_negative),
                ),
                interval_scale(interval_square(q), source_positive * target_positive),
            ),
            RationalInterval(-source_positive * source_negative, -source_positive * source_negative),
        ),
        interval_scale(interval_square(map_determinant), target_positive * target_negative),
    )
    return m00, m01, m11, determinant_naive, determinant_expanded


def cone_diagnostic(
    edge: str, entries: tuple[RationalInterval, RationalInterval, RationalInterval, RationalInterval]
) -> tuple[RationalInterval, RationalInterval, RationalInterval, RationalInterval, RationalInterval, Fraction]:
    m00, m01, m11, determinant_naive, determinant_expanded = cone_intervals(edge, entries)
    trace = interval_add(m00, m11)
    if m00.lower <= 0 or determinant_expanded.lower <= 0 or trace.upper <= 0:
        raise SystemExit(
            f"global full-source Sylvester predicate failed for {edge}: "
            f"m00.lower={m00.lower} det.expanded.lower={determinant_expanded.lower}"
        )
    return (
        m00, m01, m11, determinant_naive, determinant_expanded,
        determinant_expanded.lower / trace.upper,
    )


def algebra_selftest() -> None:
    singleton_cases = (
        ("N0->N0", (Fraction(2), Fraction(0), Fraction(0), Fraction(1, 2))),
        ("N0->N1", (Fraction(-6), Fraction(1, 5), Fraction(1, 100), Fraction(1, 1000))),
        ("N1->N0", (Fraction(3, 2), Fraction(-1, 7), Fraction(1, 90), Fraction(-1, 500))),
    )
    for edge, values in singleton_cases:
        entries = tuple(RationalInterval(value, value) for value in values)
        _, _, _, naive, expanded = cone_intervals(edge, entries)
        if naive != expanded:
            raise SystemExit(f"expanded determinant identity selftest failed: {edge}")

    cancellation_witness = (
        RationalInterval(Fraction("-8.7688880834563889"), Fraction("-1.8270792510353773")),
        RationalInterval(Fraction("-0.37613695338121922"), Fraction("0.37608677886874053")),
        RationalInterval(Fraction("-0.058396689609620689"), Fraction("0.058396316025623007")),
        RationalInterval(Fraction("-0.0063380542042827939"), Fraction("0.0063380134508601981")),
    )
    m00, _, _, naive, expanded = cone_intervals("N0->N0", cancellation_witness)
    if not (m00.lower > 0 and naive.lower < 0 and expanded.lower > 0):
        raise SystemExit("expanded determinant cancellation witness failed")


def parse_ledger_line(raw_line: bytes) -> dict[str, str]:
    try:
        line = raw_line.decode("ascii").rstrip("\n")
    except UnicodeDecodeError as error:
        raise SystemExit("non-ASCII ledger record") from error
    tokens = line.split()
    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            raise SystemExit("malformed ledger token")
        key, value = token.split("=", 1)
        if key in fields:
            raise SystemExit(f"duplicate ledger field: {key}")
        fields[key] = value
    return fields


def ideal_tile(source: str, u_index: int, s_index: int, u_tiles: int, s_tiles: int) -> tuple[RationalInterval, RationalInterval]:
    center_u, radius_u, radius_s = SOURCE_GEOMETRY[source]
    u_step = 2 * radius_u / u_tiles
    s_step = 2 * radius_s / s_tiles
    u_lower = center_u - radius_u + u_index * u_step
    s_lower = -radius_s + s_index * s_step
    return (
        RationalInterval(u_lower, u_lower + u_step),
        RationalInterval(s_lower, s_lower + s_step),
    )


def check_tight_enclosure(actual: RationalInterval, ideal: RationalInterval, field: str) -> None:
    tolerance = Fraction("2e-14")
    if actual.lower > ideal.lower or actual.upper < ideal.upper:
        raise SystemExit(f"{field} does not enclose its canonical tile")
    if ideal.lower - actual.lower > tolerance or actual.upper - ideal.upper > tolerance:
        raise SystemExit(f"{field} is wider than canonical tolerance")


def validate_c0_certificate(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"missing retained C0 certificate: {path}")
    text = path.read_text(encoding="ascii")
    values = parse_unique_values(text, path)
    for key, expected in C0_EXPECTED_CONTRACT.items():
        if values.get(key) != expected:
            raise SystemExit(f"C0 certificate {key} mismatch")
    hash_keys = (
        "SOURCE_SHA256",
        "EXECUTABLE_SHA256",
        "CAPD_CONFIG_SHA256",
        "CAPD_CFLAGS_SHA256",
        "CAPD_LIBS_SHA256",
        "CXX_DRIVER_SHA256",
        "CXX_VERSION_SHA256",
        "SHARD_BUNDLE_SHA256",
        "LEDGER_SHA256",
    )
    for key in hash_keys:
        if not HASH_RE.fullmatch(values.get(key, "")):
            raise SystemExit(f"C0 certificate {key} is not SHA-256")
    c0_source = Path(__file__).with_name("cs6_capd_fibonacci_covering.cpp")
    if not c0_source.is_file() or values["SOURCE_SHA256"] != sha256_file(c0_source):
        raise SystemExit("C0 certificate source does not match the frozen local C0 machine")
    if parse_nonnegative_int(values.get("SHARDS"), "C0 SHARDS") < 1:
        raise SystemExit("C0 certificate has no shards")
    if not values.get("SLURM_JOB_ID") or any(
        character.isspace() for character in values["SLURM_JOB_ID"]
    ):
        raise SystemExit("C0 certificate has invalid SLURM_JOB_ID")
    relations = tuple(
        line for line in text.splitlines() if line.startswith("COVERING_RELATION=")
    )
    if relations != C0_COVERING_RELATIONS:
        raise SystemExit("C0 certificate covering-relation contract mismatch")
    return values


def revalidate_c0_evidence(run_dir: Path, certificate: Path) -> dict[str, str]:
    values = validate_c0_certificate(certificate)
    c0_run_dir = run_dir / "c0-run"
    retained_c0_aggregator = run_dir / "c0-aggregator.py"
    canonical_c0_aggregator = Path(__file__).with_name(
        "cs6_capd_fibonacci_covering_aggregate.py"
    )
    c0_source = Path(__file__).with_name("cs6_capd_fibonacci_covering.cpp")
    if (
        not c0_run_dir.is_dir()
        or not retained_c0_aggregator.is_file()
        or not canonical_c0_aggregator.is_file()
    ):
        raise SystemExit("retained C0 raw evidence bundle is missing")
    reject_symlink_tree(c0_run_dir, "retained C0 raw evidence bundle")
    if retained_c0_aggregator.read_bytes() != canonical_c0_aggregator.read_bytes():
        raise SystemExit("retained C0 aggregator does not match canonical local code")
    shards = parse_nonnegative_int(values.get("SHARDS"), "C0 SHARDS")
    if shards < 1:
        raise SystemExit("C0 certificate has no shards")
    with tempfile.TemporaryDirectory(prefix="cs6-c0-reaggregate-") as temporary:
        temporary_path = Path(temporary)
        fresh_ledger = temporary_path / "ledger.txt"
        fresh_certificate = temporary_path / "certificate.txt"
        completed = subprocess.run(
            (
                sys.executable,
                str(canonical_c0_aggregator),
                "--run-dir",
                str(c0_run_dir),
                "--shards",
                str(shards),
                "--source",
                str(c0_source),
                "--ledger-output",
                str(fresh_ledger),
                "--certificate-output",
                str(fresh_certificate),
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise SystemExit(f"retained C0 raw evidence failed reaggregation{suffix}")
        if fresh_certificate.read_bytes() != certificate.read_bytes():
            raise SystemExit("retained C0 certificate does not match raw evidence")
        if sha256_file(fresh_ledger) != values["LEDGER_SHA256"]:
            raise SystemExit("retained C0 canonical ledger hash mismatch")
    return values


def validate_manifest(run_dir: Path, source: Path, shards: int) -> tuple[dict[str, str], tuple[int, int, int]]:
    manifest_path = run_dir / "run-manifest.txt"
    snapshot = run_dir / "proof-source.cpp"
    binary = run_dir / "proof-binary"
    c0_certificate = run_dir / "c0-certificate.txt"
    for path in (manifest_path, snapshot, binary, source, c0_certificate):
        if not path.is_file():
            raise SystemExit(f"missing provenance artifact: {path}")
    manifest = parse_unique_values(manifest_path.read_text(encoding="ascii"), manifest_path)
    required = {
        "MANIFEST_KIND": "CS6_CAPD_C1_CONE_RUN_V1",
        "RUN_COMPLETE": "true",
        "EXECUTION_TRUST_MODEL": (
            "SAME_UID_ACTIVE_SLURM_ALLOCATION_INCLUDES_EXECUTION_NODE_"
            "NO_REMOTE_ATTESTATION"
        ),
        "REMOTE_ATTESTATION_PRESENT": "false",
        "INDEPENDENT_REPLAY_REQUIRED": "true",
        "C1_SET": "C1Rect2Set",
        "C1_INITIAL_DERIVATIVE": "B*R_SOURCE_TANGENT_ZERO_NORMAL",
        "SHARDS": str(shards),
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise SystemExit(f"manifest {key} mismatch")
    if not manifest.get("SLURM_JOB_ID") or any(
        character.isspace() for character in manifest["SLURM_JOB_ID"]
    ):
        raise SystemExit("manifest SLURM_JOB_ID mismatch")
    for key in (
        "SOURCE_SHA256", "EXECUTABLE_SHA256", "C0_CERTIFICATE_SHA256",
        "CAPD_CONFIG_SHA256", "CAPD_CFLAGS_SHA256", "CAPD_LIBS_SHA256",
        "CXX_DRIVER_SHA256", "CXX_VERSION_SHA256", "CAPD_VERSION_SHA256",
        "CAPD_PC_SHA256", "CAPD_LIBRARY_MANIFEST_SHA256",
        "CAPD_HEADER_MANIFEST_SHA256", "RUNTIME_LINKAGE_SHA256",
        "RUNTIME_LIBRARY_MANIFEST_SHA256", "SLURM_JOB_RECORD_SHA256",
        "SLURM_VERSION_SHA256", "SLURM_HOSTNAMES_SHA256",
        "C0_AGGREGATOR_SHA256",
    ):
        if not HASH_RE.fullmatch(manifest.get(key, "")):
            raise SystemExit(f"manifest {key} is not SHA-256")
    if not (manifest["SOURCE_SHA256"] == sha256_file(source) == sha256_file(snapshot)):
        raise SystemExit("proof source snapshot/hash mismatch")
    if manifest["EXECUTABLE_SHA256"] != sha256_file(binary):
        raise SystemExit("proof binary hash mismatch")
    if manifest["C0_CERTIFICATE_SHA256"] != sha256_file(c0_certificate):
        raise SystemExit("retained C0 certificate hash mismatch")
    validate_c0_certificate(c0_certificate)
    retained = {
        "CAPD_CONFIG_SHA256": run_dir / "capd-config-retained",
        "CAPD_CFLAGS_SHA256": run_dir / "capd-cflags.txt",
        "CAPD_LIBS_SHA256": run_dir / "capd-libs.txt",
        "CXX_DRIVER_SHA256": run_dir / "compiler-driver-retained",
        "CXX_VERSION_SHA256": run_dir / "compiler-version.txt",
        "CAPD_VERSION_SHA256": run_dir / "capd-version.txt",
        "CAPD_PC_SHA256": run_dir / "capd.pc",
        "CAPD_LIBRARY_MANIFEST_SHA256": run_dir / "capd-libraries.sha256",
        "CAPD_HEADER_MANIFEST_SHA256": run_dir / "capd-headers.sha256",
        "RUNTIME_LINKAGE_SHA256": run_dir / "runtime-linkage.txt",
        "RUNTIME_LIBRARY_MANIFEST_SHA256": run_dir / "runtime-libraries.sha256",
        "SLURM_JOB_RECORD_SHA256": run_dir / "slurm-job.txt",
        "SLURM_VERSION_SHA256": run_dir / "slurm-version.txt",
        "SLURM_HOSTNAMES_SHA256": run_dir / "slurm-hostnames.txt",
        "C0_AGGREGATOR_SHA256": run_dir / "c0-aggregator.py",
    }
    for key, path in retained.items():
        if not path.is_file() or manifest[key] != sha256_file(path):
            raise SystemExit(f"retained provenance mismatch: {key}")
    if (run_dir / "capd-version.txt").read_text(encoding="ascii") != "5.3.0\n":
        raise SystemExit("retained CAPD version mismatch")
    cflags = (run_dir / "capd-cflags.txt").read_text(encoding="ascii").split()
    if "-D__USE_FILIB__" not in cflags or "-frounding-math" not in cflags:
        raise SystemExit("retained CAPD interval backend flags mismatch")
    if "Version: 5.3.0" not in (run_dir / "capd.pc").read_text(encoding="utf-8"):
        raise SystemExit("retained CAPD pkg-config version mismatch")
    library_tokens = (run_dir / "capd-libs.txt").read_text(encoding="utf-8").split()
    expected_libraries = {Path(token).resolve() for token in library_tokens}
    if not expected_libraries or any(not path.is_file() for path in expected_libraries):
        raise SystemExit("retained CAPD library argument set is not path-bound")
    actual_libraries = validate_sha256_manifest(
        run_dir / "capd-libraries.sha256", "CAPD library"
    )
    if actual_libraries != expected_libraries:
        raise SystemExit("CAPD library manifest does not match linker arguments")

    include_roots = {
        Path(token[2:]).resolve() for token in cflags if token.startswith("-I")
    }
    if not include_roots or any(not root.is_dir() for root in include_roots):
        raise SystemExit("retained CAPD include roots are invalid")
    expected_headers = {
        path.resolve()
        for root in include_roots
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    actual_headers = validate_sha256_manifest(
        run_dir / "capd-headers.sha256", "CAPD header"
    )
    if actual_headers != expected_headers:
        raise SystemExit("CAPD header manifest does not match include trees")

    runtime_text = (run_dir / "runtime-linkage.txt").read_text(encoding="utf-8")
    expected_runtime = linkage_paths(runtime_text)
    actual_runtime = validate_sha256_manifest(
        run_dir / "runtime-libraries.sha256", "runtime library"
    )
    if not expected_runtime or actual_runtime != expected_runtime:
        raise SystemExit("runtime library manifest does not match linkage record")
    slurm_record = (run_dir / "slurm-job.txt").read_text(encoding="utf-8")
    slurm_tokens = set(slurm_record.split())
    execution_uid = manifest.get("EXECUTION_UID", "")
    execution_node = manifest.get("EXECUTION_NODE", "")
    slurm_nodelist = manifest.get("SLURM_NODELIST", "")
    if (
        re.fullmatch(r"[0-9]+", execution_uid) is None
        or not execution_node
        or any(character.isspace() for character in execution_node)
        or not slurm_nodelist
        or any(character.isspace() for character in slurm_nodelist)
        or f"JobId={manifest.get('SLURM_JOB_ID', '')}" not in slurm_tokens
        or not ({"JobState=RUNNING", "JobState=COMPLETING"} & slurm_tokens)
        or f"NodeList={slurm_nodelist}" not in slurm_tokens
        or not any(
            token.startswith("UserId=") and token.endswith(f"({execution_uid})")
            for token in slurm_tokens
        )
    ):
        raise SystemExit("retained Slurm control-plane record mismatch")
    allocated_nodes = set(
        (run_dir / "slurm-hostnames.txt").read_text(encoding="utf-8").splitlines()
    )
    if not allocated_nodes or execution_node not in allocated_nodes:
        raise SystemExit("execution node is not in retained Slurm allocation")
    match = GRID_RE.fullmatch(manifest.get("GRID", ""))
    if match is None:
        raise SystemExit("invalid manifest grid")
    grid = tuple(int(match.group(index)) for index in range(1, 4))
    if min(grid) < 1:
        raise SystemExit("non-positive manifest grid")
    n0_u, n1_u, s_tiles = grid
    if manifest.get("RAW_TILES") != str((n0_u + n1_u) * s_tiles):
        raise SystemExit("manifest raw tile count mismatch")
    if manifest.get("EDGE_RECORDS") != str((2 * n0_u + n1_u) * s_tiles):
        raise SystemExit("manifest edge record count mismatch")
    if int(manifest.get("ORDER", "0")) < 1:
        raise SystemExit("invalid manifest order")
    return manifest, grid


def validate_output_paths(ledger: Path, certificate: Path, run_dir: Path, source: Path) -> None:
    inputs = {source.resolve()}
    inputs.update(path.resolve() for path in run_dir.iterdir())
    for output in (ledger, certificate):
        if not output.parent.is_dir():
            raise SystemExit(f"output parent is not a directory: {output.parent}")
        if output.exists() or output.is_symlink():
            raise SystemExit(f"refusing existing output: {output}")
        if output.resolve() in inputs:
            raise SystemExit(f"output aliases proof input: {output}")


def write_fresh_atomic(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link refuses a destination created by a concurrent/stale run.
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def rational_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--certificate-output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path(__file__).with_name("cs6_capd_c1_cone.cpp"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    algebra_selftest()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    manifest, (n0_u, n1_u, s_tiles) = validate_manifest(args.run_dir, args.source, args.shards)
    validate_output_paths(args.ledger_output, args.certificate_output, args.run_dir, args.source)
    c0_path = args.run_dir / "c0-certificate.txt"
    c0_values = revalidate_c0_evidence(args.run_dir, c0_path)

    seen: set[tuple[str, int, int]] = set()
    per_edge_indices = {edge: set() for edge in EDGES}
    source_fingerprints: dict[tuple[str, int, int], tuple[str, ...]] = {}
    hulls = {edge: [Hull(), Hull(), Hull(), Hull()] for edge in EDGES}
    min_liouville: Fraction | None = None
    canonical_records: dict[tuple[str, int, int], bytes] = {}
    bundle_digest = hashlib.sha256()
    bundle_digest.update((args.run_dir / "run-manifest.txt").read_bytes())
    bundle_digest.update(c0_path.read_bytes())

    for ordinal in range(1, args.shards + 1):
        output_path = args.run_dir / f"shard-{ordinal}.txt"
        ledger_path = args.run_dir / f"ledger-{ordinal}.txt"
        if not output_path.is_file() or not ledger_path.is_file():
            raise SystemExit(f"missing shard artifact {ordinal}")
        output_bytes = output_path.read_bytes()
        output_values = parse_unique_values(output_bytes.decode("ascii"), output_path)
        required_output = {
            "SCHEMA": "sounio.cs6.capd-c1-cone.v1",
            "CAPD_SOURCE_TREE_DECLARED": "capd-5.3.0",
            "INTERVAL_BACKEND_DECLARED": "FILIB",
            "C1_SET": "C1Rect2Set",
            "C1_INITIAL_DERIVATIVE": "B*R_SOURCE_TANGENT_ZERO_NORMAL",
            "MAP": "P^6",
            "RETURNS_PER_MAP": "6",
            "SECTION_ORIENTATION": "MinusPlus",
            "Q_DECIMAL_INTERPRETATION": "exact-decimal-input-outward-interval",
            "LEDGER_ENDPOINT_ENCODING": "outward-one-ulp-exact-hexadecimal-binary64",
            "Q_N0": "1,-2.3023784599059653",
            "Q_N1": "0.06526711140171336,-2.3023784599059653",
            "CONE_DETERMINANT_FORM": "expanded-exact-cancellation-before-interval-evaluation",
            "C1_LIOUVILLE_FINAL_RETURN_OVERLAP_REQUIRED": "true",
            "LIOUVILLE_EXPONENTIAL_OPERAND_EMITTED": "true",
            "C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_REQUIRED": "true",
            "VECTOR_FIELD_CAPD": C1_VECTOR_FIELD,
            "LIOUVILLE_FIELD_CAPD": C1_LIOUVILLE_FIELD,
            "ZSEC": C0_EXPECTED_CONTRACT["ZSEC"],
            "ORIGIN": C0_EXPECTED_CONTRACT["ORIGIN"],
            "UNSTABLE": C0_EXPECTED_CONTRACT["UNSTABLE"],
            "STABLE": C0_EXPECTED_CONTRACT["STABLE"],
            "FRAME_DETERMINANT": C0_EXPECTED_CONTRACT["FRAME_DETERMINANT"],
            "N0_LOCAL": C0_EXPECTED_CONTRACT["N0_LOCAL"],
            "N1_LOCAL": C0_EXPECTED_CONTRACT["N1_LOCAL"],
            "HSETS_DISJOINT": "true",
            "GRID": manifest["GRID"],
            "SHARD": f"{ordinal}/{args.shards}",
            "LEDGER_ENABLED": "true",
            "SHARD_PASS": "true",
            "FULL_SOURCE_GLOBAL_HULL_TESTED": "false",
            "PAIRWISE_CHORD_CONE_CONDITION_PROVED": "false",
            "TANGENT_CONE_CONDITION_PROVED": "false",
            "LIOUVILLE_INVERTIBILITY_PROVED": "false",
            "UNIFORM_HYPERBOLICITY_PROVED": "false",
            "CHAOTIC_ATTRACTOR_PROVED": "false",
        }
        for key, expected in required_output.items():
            if output_values.get(key) != expected:
                raise SystemExit(f"shard {ordinal}: {key} mismatch")
        if output_values.get("ORDER") != manifest["ORDER"]:
            raise SystemExit(f"shard {ordinal}: order mismatch")
        selector_index = ordinal - 1
        expected_n0 = selector_expected(n0_u * s_tiles, selector_index, args.shards)
        expected_n1 = selector_expected(n1_u * s_tiles, selector_index, args.shards)
        expected_raw_local = expected_n0 + expected_n1
        expected_records_local = 2 * expected_n0 + expected_n1
        expected_counts_local = {
            "RAW_TILES_EXPECTED": expected_raw_local,
            "RAW_TILES_PROCESSED": expected_raw_local,
            "RAW_TILES_VALID": expected_raw_local,
            "EDGE_RECORDS_EXPECTED": expected_records_local,
            "EDGE_RECORDS_WRITTEN": expected_records_local,
        }
        for field, expected in expected_counts_local.items():
            actual = parse_nonnegative_int(output_values.get(field), field)
            if actual != expected:
                raise SystemExit(
                    f"shard {ordinal}: {field} mismatch: {actual}/{expected}"
                )
        reported_tile_cone_passes = parse_nonnegative_int(
            output_values.get("TILE_CONE_DIAGNOSTIC_PASSES"),
            "TILE_CONE_DIAGNOSTIC_PASSES",
        )
        if reported_tile_cone_passes > expected_records_local:
            raise SystemExit(f"shard {ordinal}: impossible tile cone pass count")
        frame_det = parse_interval(output_values.get("FRAME_DETERMINANT", ""), "FRAME_DETERMINANT")
        if frame_det.lower <= 0 <= frame_det.upper or output_values.get("FRAME_RIGOROUSLY_INVERTIBLE") != "true":
            raise SystemExit(f"shard {ordinal}: singular frame")
        raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
        if len(raw_lines) != expected_records_local:
            raise SystemExit(f"shard {ordinal}: ledger count mismatch")

        observed_tile_cone_passes = 0
        for raw_line in raw_lines:
            fields = parse_ledger_line(raw_line)
            if set(fields) != LEDGER_REQUIRED_FIELDS:
                raise SystemExit(
                    "ledger field set mismatch: "
                    f"missing={LEDGER_REQUIRED_FIELDS - set(fields)} "
                    f"extra={set(fields) - LEDGER_REQUIRED_FIELDS}"
                )
            source, target, edge = fields["SOURCE"], fields["TARGET"], fields["EDGE"]
            if edge not in EDGES or edge != f"{source}->{target}":
                raise SystemExit("invalid ledger edge")
            u_index, s_index = int(fields["U_INDEX"]), int(fields["S_INDEX"])
            u_limit = n0_u if source == "N0" else n1_u
            if not (0 <= u_index < u_limit and 0 <= s_index < s_tiles):
                raise SystemExit(f"ledger index out of range: {(edge, u_index, s_index)}")
            linear_index = u_index * s_tiles + s_index
            if linear_index % args.shards != selector_index:
                raise SystemExit(
                    f"ledger record assigned to wrong shard: {(edge, u_index, s_index)}"
                )
            key = (edge, u_index, s_index)
            if key in seen:
                raise SystemExit(f"duplicate ledger key: {key}")
            seen.add(key)
            per_edge_indices[edge].add((u_index, s_index))
            values = {
                name: parse_interval(fields[name], name, require_hex=True)
                for name in LEDGER_INTERVAL_FIELDS
            }
            if fields["TILE_CONE_DIAGNOSTIC"] not in {"true", "false"}:
                raise SystemExit(f"invalid tile cone diagnostic: {key}")
            observed_tile_cone_passes += fields["TILE_CONE_DIAGNOSTIC"] == "true"
            ideal_u, ideal_s = ideal_tile(source, u_index, s_index, u_limit, s_tiles)
            check_tight_enclosure(values["SOURCE_U"], ideal_u, f"{key} SOURCE_U")
            check_tight_enclosure(values["SOURCE_S"], ideal_s, f"{key} SOURCE_S")
            if values["C1_RETURN_TIME"].lower <= 0:
                raise SystemExit(f"non-positive C1 return time: {key}")
            if fields["LIOUVILLE_INVERTIBLE"] != "true" or values["DET_LIOUVILLE"].lower <= 0:
                raise SystemExit(f"Liouville invertibility failed: {key}")
            for crossing in range(7):
                if values[f"NU{crossing}"].lower <= 0:
                    raise SystemExit(f"non-positive normal velocity: {key} crossing={crossing}")
            previous = RationalInterval(Fraction(0), Fraction(0))
            for crossing in range(1, 7):
                current = values[f"T{crossing}"]
                if current.lower <= previous.upper:
                    raise SystemExit(f"non-increasing return times: {key}")
                previous = current
            if not overlaps(values["C1_RETURN_TIME"], values["T6"]):
                raise SystemExit(f"C1/Liouville sixth-return time mismatch: {key}")
            liouville_formula = interval_div_positive(
                interval_mul(
                    values["EXP_INTEGRAL_DIVERGENCE"], values["NU0"]
                ),
                values["NU6"],
            )
            if not contains(liouville_formula, values["DET_LIOUVILLE"]):
                raise SystemExit(f"Liouville determinant formula mismatch: {key}")
            c1_determinant = interval_sub(
                interval_mul(values["A00"], values["A11"]),
                interval_mul(values["A01"], values["A10"]),
            )
            source_radius_u, source_radius_s = SOURCE_GEOMETRY[source][1:]
            target_radius_u, target_radius_s = SOURCE_GEOMETRY[target][1:]
            normalized_liouville = interval_scale(
                values["DET_LIOUVILLE"],
                (source_radius_u * source_radius_s)
                / (target_radius_u * target_radius_s),
            )
            if not overlaps(c1_determinant, normalized_liouville):
                raise SystemExit(f"C1/Liouville normalized determinant mismatch: {key}")
            fingerprint = tuple(fields[name] for name in (
                "SOURCE_U", "SOURCE_S", "C1_RETURN_TIME", "INTEGRAL_DIVERGENCE",
                "EXP_INTEGRAL_DIVERGENCE",
                "DET_LIOUVILLE", "NU0", "NU1", "NU2", "NU3", "NU4", "NU5", "NU6",
                "T1", "T2", "T3", "T4", "T5", "T6",
            ))
            source_key = (source, u_index, s_index)
            if source_key in source_fingerprints and source_fingerprints[source_key] != fingerprint:
                raise SystemExit(f"cross-edge source evidence drift: {source_key}")
            source_fingerprints[source_key] = fingerprint
            min_liouville = (
                liouville_formula.lower
                if min_liouville is None
                else min(min_liouville, liouville_formula.lower)
            )
            for index, name in enumerate(("A00", "A01", "A10", "A11")):
                hulls[edge][index].include(values[name], key)
            canonical_records[key] = raw_line

        if observed_tile_cone_passes != reported_tile_cone_passes:
            raise SystemExit(f"shard {ordinal}: tile cone diagnostic count mismatch")

        bundle_digest.update(output_path.name.encode("ascii") + b"\0" + hashlib.sha256(output_bytes).digest())
        ledger_bytes = b"".join(raw_lines)
        bundle_digest.update(ledger_path.name.encode("ascii") + b"\0" + hashlib.sha256(ledger_bytes).digest())

    expected_counts = {"N0->N0": n0_u * s_tiles, "N0->N1": n0_u * s_tiles, "N1->N0": n1_u * s_tiles}
    for edge, expected in expected_counts.items():
        if len(per_edge_indices[edge]) != expected:
            raise SystemExit(f"incomplete edge partition {edge}: {len(per_edge_indices[edge])}/{expected}")
    expected_records = sum(expected_counts.values())
    if len(seen) != expected_records or len(source_fingerprints) != (n0_u + n1_u) * s_tiles:
        raise SystemExit("full-source partition count mismatch")
    if min_liouville is None or min_liouville <= 0:
        raise SystemExit("global Liouville determinant lower bound is non-positive")

    diagnostics = {}
    for edge in EDGES:
        entries = tuple(hull.interval() for hull in hulls[edge])
        diagnostics[edge] = (entries, cone_diagnostic(edge, entries))

    edge_rank = {edge: index for index, edge in enumerate(EDGES)}
    ordered_keys = sorted(canonical_records, key=lambda item: (edge_rank[item[0]], item[1], item[2]))
    canonical_ledger = b"".join(canonical_records[key] for key in ordered_keys)
    ledger_sha = hashlib.sha256(canonical_ledger).hexdigest()
    lines = [
        "CERTIFICATE_KIND=CAPD_RIGOROUS_C1_CONE_AGGREGATE_V1",
        f"GRID={manifest['GRID']}",
        f"ORDER={manifest['ORDER']}",
        "C1_SET=C1Rect2Set",
        "MAP=P^6",
        "SECTION_ORIENTATION=MinusPlus",
        "Q_DECIMAL_INTERPRETATION=exact-decimal-rational",
        "GLOBAL_SYLVESTER_ARITHMETIC=exact-rational-over-outward-serialized-CAPD-endpoints",
        "CONE_DETERMINANT_FORM=expanded-exact-cancellation-before-interval-evaluation",
        f"SHARDS={args.shards}",
        f"SOURCE_SHA256={manifest['SOURCE_SHA256']}",
        f"EXECUTABLE_SHA256={manifest['EXECUTABLE_SHA256']}",
        f"C0_CERTIFICATE_SHA256={manifest['C0_CERTIFICATE_SHA256']}",
        f"C0_AGGREGATOR_SHA256={manifest['C0_AGGREGATOR_SHA256']}",
        f"C0_SOURCE_SHA256={c0_values['SOURCE_SHA256']}",
        f"C0_EXECUTABLE_SHA256={c0_values['EXECUTABLE_SHA256']}",
        f"C0_LEDGER_SHA256={c0_values['LEDGER_SHA256']}",
        f"C0_SHARD_BUNDLE_SHA256={c0_values['SHARD_BUNDLE_SHA256']}",
        f"C0_EXECUTION_TRUST_MODEL_DECLARED={c0_values['EXECUTION_TRUST_MODEL']}",
        "C0_EXECUTION_PROVENANCE_VERIFIED=false",
        f"C1_EXECUTION_TRUST_MODEL={manifest['EXECUTION_TRUST_MODEL']}",
        f"C1_CAPD_CONFIG_SHA256={manifest['CAPD_CONFIG_SHA256']}",
        f"C1_CAPD_CFLAGS_SHA256={manifest['CAPD_CFLAGS_SHA256']}",
        f"C1_CAPD_LIBS_SHA256={manifest['CAPD_LIBS_SHA256']}",
        f"C1_CAPD_LIBRARY_MANIFEST_SHA256={manifest['CAPD_LIBRARY_MANIFEST_SHA256']}",
        f"C1_CAPD_HEADER_MANIFEST_SHA256={manifest['CAPD_HEADER_MANIFEST_SHA256']}",
        f"C1_RUNTIME_LIBRARY_MANIFEST_SHA256={manifest['RUNTIME_LIBRARY_MANIFEST_SHA256']}",
        f"C1_CXX_DRIVER_SHA256={manifest['CXX_DRIVER_SHA256']}",
        f"C1_SLURM_JOB_ID={manifest['SLURM_JOB_ID']}",
        f"C1_SLURM_JOB_RECORD_SHA256={manifest['SLURM_JOB_RECORD_SHA256']}",
        f"C1_EXECUTION_NODE={manifest['EXECUTION_NODE']}",
        f"C1_EXECUTION_UID={manifest['EXECUTION_UID']}",
        f"C1_SHARD_BUNDLE_SHA256={bundle_digest.hexdigest()}",
        f"C1_LEDGER_RECORDS={expected_records}",
        f"C1_LEDGER_SHA256={ledger_sha}",
        f"MIN_DET_LIOUVILLE_LOWER_RATIONAL={rational_text(min_liouville)}",
    ]
    uniform_mu: Fraction | None = None
    for edge in EDGES:
        entries, (m00, m01, m11, determinant_naive, determinant_expanded, mu) = diagnostics[edge]
        uniform_mu = mu if uniform_mu is None else min(uniform_mu, mu)
        contributor_fields = []
        for name, hull in zip(("A00", "A01", "A10", "A11"), hulls[edge]):
            assert hull.lower_key is not None and hull.upper_key is not None
            contributor_fields.extend((
                f"{name}_LOWER_CONTRIBUTOR={hull.lower_key[1]}:{hull.lower_key[2]}",
                f"{name}_UPPER_CONTRIBUTOR={hull.upper_key[1]}:{hull.upper_key[2]}",
            ))
        lines.extend((
            f"EDGE={edge} GLOBAL_A00=[{rational_text(entries[0].lower)},{rational_text(entries[0].upper)}]",
            f"EDGE={edge} GLOBAL_A01=[{rational_text(entries[1].lower)},{rational_text(entries[1].upper)}]",
            f"EDGE={edge} GLOBAL_A10=[{rational_text(entries[2].lower)},{rational_text(entries[2].upper)}]",
            f"EDGE={edge} GLOBAL_A11=[{rational_text(entries[3].lower)},{rational_text(entries[3].upper)}]",
            f"EDGE={edge} GLOBAL_HULL_CONTRIBUTORS={' '.join(contributor_fields)}",
            f"EDGE={edge} M00_LOWER_RATIONAL={rational_text(m00.lower)} DET_M_NAIVE_LOWER_RATIONAL={rational_text(determinant_naive.lower)} DET_M_EXPANDED_LOWER_RATIONAL={rational_text(determinant_expanded.lower)} MU_LOWER_RATIONAL={rational_text(mu)} GLOBAL_SYLVESTER_PASS=true",
        ))
    assert uniform_mu is not None and uniform_mu > 0
    lines.extend((
        f"UNIFORM_CONE_GAP_LOWER_RATIONAL={rational_text(uniform_mu)}",
        "FULL_SOURCE_C1_DERIVATIVE_ENCLOSURE_PROVED=true",
        "C0_RAW_EVIDENCE_REAGGREGATED=true",
        "C0_C1_SEMANTIC_CONTRACT_BOUND=true",
        "C1_DYNAMICAL_SYSTEM_PREAMBLE_BOUND=true",
        "C1_EXPLICIT_CAPD_ARTIFACT_SETS_BOUND=true",
        "GLOBAL_FULL_SOURCE_HULL_TESTED=true",
        "PAIRWISE_CHORD_CONE_CONDITION_PROVED=true",
        "TANGENT_CONE_CONDITION_PROVED=true",
        "LIOUVILLE_INVERTIBILITY_PROVED=true",
        "C1_LIOUVILLE_FINAL_RETURN_BOUND=true",
        "LIOUVILLE_DETERMINANT_OPERAND_CONSISTENCY_BOUND=true",
        "LIOUVILLE_EXPONENTIATION_RECOMPUTED_BY_AGGREGATOR=false",
        "C1_LIOUVILLE_NORMALIZED_DETERMINANT_OVERLAP_BOUND=true",
        "FIBONACCI_COVERINGS_PROVED=true",
        "POSITIVE_ENTROPY_PROVED=true",
        "COMBINED_C0_C1_MATHEMATICAL_EVIDENCE_COMPLETE=true",
        "COMBINED_C0_C1_EXECUTION_PROVENANCE_ATTESTED=false",
        "UNIFORM_HYPERBOLICITY_PROVED=false",
        "UNIFORM_HYPERBOLICITY_BLOCKER=EXPLICIT_COMPACT_INVARIANT_CONE_THEOREM_NOT_BOUND",
        "CHAOTIC_ATTRACTOR_PROVED=false",
        "FLOW_ENTROPY_BOUND_PROVED=false",
        "REMOTE_ATTESTATION_PRESENT=false",
        "INDEPENDENT_REPLAY_REQUIRED=true",
    ))
    certificate = ("\n".join(lines) + "\n").encode("ascii")
    write_fresh_atomic(args.ledger_output, canonical_ledger)
    write_fresh_atomic(args.certificate_output, certificate)
    print(f"CS6_CAPD_C1_CONE_AGGREGATE PASS records={expected_records}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
