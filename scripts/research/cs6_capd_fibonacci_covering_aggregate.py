#!/usr/bin/env python3
"""Validate and aggregate a complete sharded CS6 CAPD covering proof."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path


EDGES = ("N0->N0", "N0->N1", "N1->N0")
ROLES = ("support", "left_exit", "right_exit")
DEGREES = {"N0->N0": -1, "N0->N1": -1, "N1->N0": 1}
SOURCE_GEOMETRY = {
    "N0": (Decimal("0"), Decimal("0.004"), Decimal("0.3")),
    "N1": (
        Decimal("0.019771776972779206"),
        Decimal("0.0015"),
        Decimal("0.3"),
    ),
}
EXPECTED_GRID = (200, 75, 75, 1200)
EXPECTED_PREAMBLE = {
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
}
PREAMBLE_KEYS = tuple(EXPECTED_PREAMBLE)
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+), ([^]]+)\]$")
STAT_RE = re.compile(
    r"^EDGE=(N0->N0|N0->N1|N1->N0) DEGREE=(-?1) "
    r"ROLE=(support|left_exit|right_exit) EXPECTED=(\d+) "
    r"PROCESSED=(\d+) PASS=(\d+) MIN_MARGIN=([^ ]+) "
    r"RETURN_TIME=\[([^,]+),([^]]+)\] "
    r"MIN_INITIAL_NORMAL_VELOCITY=([^ ]+) MIN_NORMAL_VELOCITY=([^ ]+) "
    r"MAX_PHYSICAL_DIAMETER=([^ ]+)$"
)
LEDGER_RE = re.compile(
    r"^EDGE=(N0->N0|N0->N1|N1->N0) "
    r"ROLE=(support|left_exit|right_exit) U_INDEX=(\d+) S_INDEX=(\d+) "
    r"SOURCE_U=(\[[^]]+\]) SOURCE_S=(\[[^]]+\]) "
    r"IMAGE_U=(\[[^]]+\]) IMAGE_S=(\[[^]]+\]) "
    r"INITIAL_NORMAL_VELOCITY=(\[[^]]+\]) "
    r"NORMAL_VELOCITY=(\[[^]]+\]) RETURN_TIME=(\[[^]]+\]) "
    r"PHYSICAL_DIAMETER=([^ ]+) MARGIN=([^ ]+) PASS=(true|false)$"
)


@dataclass(frozen=True)
class LedgerRecord:
    edge: str
    role: str
    u_index: int
    s_index: int
    source_u: tuple[Decimal, Decimal]
    source_s: tuple[Decimal, Decimal]
    image_u: tuple[Decimal, Decimal]
    image_s: tuple[Decimal, Decimal]
    initial_normal: tuple[Decimal, Decimal]
    final_normal: tuple[Decimal, Decimal]
    return_time: tuple[Decimal, Decimal]
    physical_diameter: Decimal
    margin: Decimal
    raw_line: bytes

    @property
    def key(self) -> tuple[str, str, int, int]:
        return self.edge, self.role, self.u_index, self.s_index


@dataclass
class AggregateStat:
    expected: int = 0
    processed: int = 0
    passed: int = 0
    min_margin: Decimal | None = None
    min_return_time: Decimal | None = None
    max_return_time: Decimal | None = None
    min_initial_normal: Decimal | None = None
    min_final_normal: Decimal | None = None
    max_physical_diameter: Decimal | None = None

    def add_record(self, record: LedgerRecord) -> None:
        self.expected += 1
        self.processed += 1
        self.passed += 1
        self.min_margin = minimum(self.min_margin, record.margin)
        self.min_return_time = minimum(self.min_return_time, record.return_time[0])
        self.max_return_time = maximum(self.max_return_time, record.return_time[1])
        self.min_initial_normal = minimum(
            self.min_initial_normal, record.initial_normal[0]
        )
        self.min_final_normal = minimum(self.min_final_normal, record.final_normal[0])
        self.max_physical_diameter = maximum(
            self.max_physical_diameter, record.physical_diameter
        )


@dataclass(frozen=True)
class ReportedStat:
    expected: int
    processed: int
    passed: int
    min_margin: Decimal
    min_return_time: Decimal
    max_return_time: Decimal
    min_initial_normal: Decimal
    min_final_normal: Decimal
    max_physical_diameter: Decimal


def minimum(current: Decimal | None, value: Decimal) -> Decimal:
    return value if current is None or value < current else current


def maximum(current: Decimal | None, value: Decimal) -> Decimal:
    return value if current is None or value > current else current


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_unique_key_values(text: str, path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.startswith("EDGE="):
            continue
        key, value = line.split("=", 1)
        if key in result and not key.startswith("PARTITION_RELATION"):
            raise SystemExit(f"{path}: duplicate key {key}")
        result[key] = value
    return result


def parse_interval(text: str, field: str) -> tuple[Decimal, Decimal]:
    match = INTERVAL_RE.match(text)
    if match is None:
        raise SystemExit(f"invalid {field} interval: {text}")
    lower, upper = Decimal(match.group(1)), Decimal(match.group(2))
    if not lower.is_finite() or not upper.is_finite() or lower > upper:
        raise SystemExit(f"non-finite or reversed {field} interval: {text}")
    return lower, upper


def ideal_source_box(
    edge: str, role: str, u_index: int, s_index: int
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    source = edge.split("->", 1)[0]
    center_u, radius_u, radius_s = SOURCE_GEOMETRY[source]
    n0_u, n1_u, support_s, exit_s = EXPECTED_GRID
    u_tiles = n0_u if source == "N0" else n1_u
    if role == "support":
        u_step = 2 * radius_u / u_tiles
        u_lower = center_u - radius_u + u_index * u_step
        u_upper = u_lower + u_step
        s_tiles = support_s
    else:
        u_lower = u_upper = center_u + (-radius_u if role == "left_exit" else radius_u)
        s_tiles = exit_s
    s_step = 2 * radius_s / s_tiles
    s_lower = -radius_s + s_index * s_step
    return u_lower, u_upper, s_lower, s_lower + s_step


def encloses_tightly(
    actual: tuple[Decimal, Decimal], ideal: tuple[Decimal, Decimal], field: str
) -> None:
    tolerance = Decimal("2e-14")
    if actual[0] > ideal[0] or actual[1] < ideal[1]:
        raise SystemExit(f"{field} does not enclose its canonical tile")
    if ideal[0] - actual[0] > tolerance or actual[1] - ideal[1] > tolerance:
        raise SystemExit(f"{field} is wider than the canonical tile tolerance")


def recompute_margin(record: LedgerRecord) -> Decimal:
    u_lower, u_upper = record.image_u
    s_lower, s_upper = record.image_s
    negative_infinity = Decimal("-Infinity")
    outside_u = negative_infinity
    if u_upper < -1:
        outside_u = -1 - u_upper
    elif u_lower > 1:
        outside_u = u_lower - 1

    if record.role == "support":
        inside_s = min(s_lower + 1, 1 - s_upper)
        return max(outside_u, inside_s)

    degree = DEGREES[record.edge]
    if degree == 1:
        return -1 - u_upper if record.role == "left_exit" else u_lower - 1
    return u_lower - 1 if record.role == "left_exit" else -1 - u_upper


def validate_record(record: LedgerRecord) -> None:
    n0_u, n1_u, support_s, exit_s = EXPECTED_GRID
    source = record.edge.split("->", 1)[0]
    u_limit = n0_u if source == "N0" else n1_u
    if record.role == "support":
        if not (0 <= record.u_index < u_limit and 0 <= record.s_index < support_s):
            raise SystemExit(f"ledger index out of range: {record.key}")
    elif record.u_index != 0 or not (0 <= record.s_index < exit_s):
        raise SystemExit(f"ledger face index out of range: {record.key}")

    ideal = ideal_source_box(*record.key)
    encloses_tightly(record.source_u, ideal[:2], f"{record.key} SOURCE_U")
    encloses_tightly(record.source_s, ideal[2:], f"{record.key} SOURCE_S")

    if record.return_time[0] <= 0:
        raise SystemExit(f"non-positive return time: {record.key}")
    if record.initial_normal[0] <= 0 or record.final_normal[0] <= 0:
        raise SystemExit(f"non-positive section normal velocity: {record.key}")
    if not record.physical_diameter.is_finite() or record.physical_diameter < 0:
        raise SystemExit(f"invalid physical diameter: {record.key}")
    if not record.margin.is_finite() or record.margin <= 0:
        raise SystemExit(f"non-positive reported margin: {record.key}")

    recomputed = recompute_margin(record)
    if not recomputed.is_finite() or recomputed <= 0:
        raise SystemExit(f"covering predicate fails on ledger image: {record.key}")
    tolerance = Decimal("1e-12") * max(Decimal(1), abs(recomputed), abs(record.margin))
    if abs(recomputed - record.margin) > tolerance:
        raise SystemExit(f"reported margin disagrees with ledger image: {record.key}")


def parse_ledger_record(raw_line: bytes) -> LedgerRecord:
    line = raw_line.decode("ascii").rstrip("\n")
    match = LEDGER_RE.match(line)
    if match is None or match.group(14) != "true":
        raise SystemExit("malformed or failing ledger record")
    record = LedgerRecord(
        edge=match.group(1),
        role=match.group(2),
        u_index=int(match.group(3)),
        s_index=int(match.group(4)),
        source_u=parse_interval(match.group(5), "SOURCE_U"),
        source_s=parse_interval(match.group(6), "SOURCE_S"),
        image_u=parse_interval(match.group(7), "IMAGE_U"),
        image_s=parse_interval(match.group(8), "IMAGE_S"),
        initial_normal=parse_interval(match.group(9), "INITIAL_NORMAL_VELOCITY"),
        final_normal=parse_interval(match.group(10), "NORMAL_VELOCITY"),
        return_time=parse_interval(match.group(11), "RETURN_TIME"),
        physical_diameter=Decimal(match.group(12)),
        margin=Decimal(match.group(13)),
        raw_line=raw_line,
    )
    validate_record(record)
    return record


def parse_reported_stats(output: str, shard: int) -> dict[tuple[str, str], ReportedStat]:
    result: dict[tuple[str, str], ReportedStat] = {}
    for line in output.splitlines():
        match = STAT_RE.match(line)
        if match is None:
            continue
        edge, degree, role = match.group(1), int(match.group(2)), match.group(3)
        key = edge, role
        if degree != DEGREES[edge] or key in result:
            raise SystemExit(f"shard {shard}: malformed or duplicate stat {key}")
        result[key] = ReportedStat(
            expected=int(match.group(4)),
            processed=int(match.group(5)),
            passed=int(match.group(6)),
            min_margin=Decimal(match.group(7)),
            min_return_time=Decimal(match.group(8)),
            max_return_time=Decimal(match.group(9)),
            min_initial_normal=Decimal(match.group(10)),
            min_final_normal=Decimal(match.group(11)),
            max_physical_diameter=Decimal(match.group(12)),
        )
    expected_keys = {(edge, role) for edge in EDGES for role in ROLES}
    if set(result) != expected_keys:
        raise SystemExit(f"shard {shard}: incomplete stat set")
    return result


def compare_reported_to_ledger(
    reported: dict[tuple[str, str], ReportedStat],
    local: dict[tuple[str, str], AggregateStat],
    shard: int,
) -> None:
    for key in reported:
        left, right = reported[key], local[key]
        count = right.processed
        if not (left.expected == left.processed == left.passed == count):
            raise SystemExit(f"shard {shard}: count mismatch for {key}")
        if count == 0:
            continue
        ledger_values = (
            right.min_margin,
            right.min_return_time,
            right.max_return_time,
            right.min_initial_normal,
            right.min_final_normal,
            right.max_physical_diameter,
        )
        reported_values = (
            left.min_margin,
            left.min_return_time,
            left.max_return_time,
            left.min_initial_normal,
            left.min_final_normal,
            left.max_physical_diameter,
        )
        if ledger_values != reported_values:
            raise SystemExit(f"shard {shard}: numeric summary mismatch for {key}")


def expected_keys() -> set[tuple[str, str, int, int]]:
    n0_u, n1_u, support_s, exit_s = EXPECTED_GRID
    keys: set[tuple[str, str, int, int]] = set()
    for edge in ("N0->N0", "N0->N1"):
        keys.update(
            (edge, "support", u, s)
            for u in range(n0_u)
            for s in range(support_s)
        )
    keys.update(
        ("N1->N0", "support", u, s)
        for u in range(n1_u)
        for s in range(support_s)
    )
    for edge in EDGES:
        for role in ("left_exit", "right_exit"):
            keys.update((edge, role, 0, s) for s in range(exit_s))
    return keys


def validate_manifest(run_dir: Path, source: Path, shards: int) -> dict[str, str]:
    manifest_path = run_dir / "run-manifest.txt"
    snapshot = run_dir / "proof-source.cpp"
    binary = run_dir / "proof-binary"
    for path in (manifest_path, snapshot, binary, source):
        if not path.is_file():
            raise SystemExit(f"missing provenance artifact: {path}")
    manifest = parse_unique_key_values(
        manifest_path.read_text(encoding="ascii"), manifest_path
    )
    required = {
        "MANIFEST_KIND": "CS6_CAPD_FIBONACCI_RUN_V1",
        "RUN_COMPLETE": "true",
        "EXECUTION_TRUST_MODEL": (
            "AUTHORIZED_FOUNDRY_SLURM_CPU_TCB_NO_ATTESTATION"
        ),
        "REMOTE_ATTESTATION_PRESENT": "false",
        "INDEPENDENT_REPLAY_REQUIRED": "true",
        "GRID": EXPECTED_PREAMBLE["GRID"],
        "ORDER": "8",
        "SHARDS": str(shards),
    }
    for key, value in required.items():
        if manifest.get(key) != value:
            raise SystemExit(f"manifest {key} mismatch")
    for key in (
        "SOURCE_SHA256",
        "EXECUTABLE_SHA256",
        "CAPD_CONFIG_SHA256",
        "CAPD_CFLAGS_SHA256",
        "CAPD_LIBS_SHA256",
        "CXX_DRIVER_SHA256",
        "CXX_VERSION_SHA256",
    ):
        if not HASH_RE.match(manifest.get(key, "")):
            raise SystemExit(f"manifest {key} is not SHA-256")
    source_sha = sha256_file(source)
    if not (
        manifest["SOURCE_SHA256"] == source_sha == sha256_file(snapshot)
    ):
        raise SystemExit("proof source snapshot/hash mismatch")
    if manifest["EXECUTABLE_SHA256"] != sha256_file(binary):
        raise SystemExit("proof executable hash mismatch")
    retained = {
        "CAPD_CONFIG_SHA256": run_dir / "capd-config-retained",
        "CAPD_CFLAGS_SHA256": run_dir / "capd-cflags.txt",
        "CAPD_LIBS_SHA256": run_dir / "capd-libs.txt",
        "CXX_DRIVER_SHA256": run_dir / "compiler-driver-retained",
        "CXX_VERSION_SHA256": run_dir / "compiler-version.txt",
    }
    for key, path in retained.items():
        if not path.is_file() or manifest[key] != sha256_file(path):
            raise SystemExit(f"retained provenance mismatch: {key}")
    for key in ("CAPD_CONFIG_PATH", "CXX_PATH", "CXX_VERSION", "SLURM_JOB_ID"):
        if not manifest.get(key):
            raise SystemExit(f"manifest missing {key}")
    if (run_dir / "compiler-version.txt").read_text(encoding="ascii") != manifest[
        "CXX_VERSION"
    ]:
        raise SystemExit("compiler version text mismatch")
    return manifest


def validate_output_paths(
    ledger_output: Path, certificate_output: Path, run_dir: Path, source: Path, shards: int
) -> None:
    outputs = (ledger_output, certificate_output)
    resolved = [path.resolve(strict=False) for path in outputs]
    if resolved[0] == resolved[1]:
        raise SystemExit("ledger and certificate outputs must be distinct")

    inputs = {
        source.resolve(),
        (run_dir / "run-manifest.txt").resolve(),
        (run_dir / "proof-source.cpp").resolve(),
        (run_dir / "proof-binary").resolve(),
    }
    for ordinal in range(1, shards + 1):
        inputs.add((run_dir / f"shard-{ordinal}.txt").resolve())
        inputs.add((run_dir / f"ledger-{ordinal}.txt").resolve())

    for path, canonical in zip(outputs, resolved):
        if not path.parent.is_dir():
            raise SystemExit(f"output parent is not a directory: {path.parent}")
        if path.exists() or path.is_symlink():
            raise SystemExit(f"refusing existing output: {path}")
        if canonical in inputs:
            raise SystemExit(f"output aliases a proof input: {path}")


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
        # Linking, unlike replace/rename, refuses a destination created by a
        # concurrent or stale run. The certificate is linked last by main().
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--certificate-output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).with_name("cs6_capd_fibonacci_covering.cpp"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    manifest = validate_manifest(args.run_dir, args.source, args.shards)
    validate_output_paths(
        args.ledger_output,
        args.certificate_output,
        args.run_dir,
        args.source,
        args.shards,
    )
    all_records: dict[tuple[str, str, int, int], LedgerRecord] = {}
    global_stats = {
        (edge, role): AggregateStat() for edge in EDGES for role in ROLES
    }
    preamble: dict[str, str] | None = None
    bundle_digest = hashlib.sha256()
    bundle_digest.update((args.run_dir / "run-manifest.txt").read_bytes())
    bundle_digest.update(bytes.fromhex(manifest["SOURCE_SHA256"]))
    bundle_digest.update(bytes.fromhex(manifest["EXECUTABLE_SHA256"]))

    for ordinal in range(1, args.shards + 1):
        output_path = args.run_dir / f"shard-{ordinal}.txt"
        ledger_path = args.run_dir / f"ledger-{ordinal}.txt"
        if not output_path.is_file() or not ledger_path.is_file():
            raise SystemExit(f"missing shard artifact for ordinal {ordinal}")
        output_bytes = output_path.read_bytes()
        output = output_bytes.decode("ascii")
        values = parse_unique_key_values(output, output_path)
        required = {
            "SHARD": f"{ordinal}/{args.shards}",
            "LEDGER_ENABLED": "true",
            "SHARD_PASS": "true",
            "FIBONACCI_COVERINGS_PROVED": "false",
            "POSITIVE_ENTROPY_PROVED": "false",
            "UNIFORM_HYPERBOLICITY_PROVED": "false",
            "CHAOTIC_ATTRACTOR_PROVED": "false",
            "FLOW_ENTROPY_BOUND_PROVED": "false",
        }
        for key, expected in required.items():
            if values.get(key) != expected:
                raise SystemExit(f"shard {ordinal}: {key} mismatch")
        current_preamble = {key: values.get(key, "") for key in PREAMBLE_KEYS}
        if current_preamble != EXPECTED_PREAMBLE:
            raise SystemExit(f"shard {ordinal}: frozen proof preamble mismatch")
        if preamble is None:
            preamble = current_preamble
        elif current_preamble != preamble:
            raise SystemExit(f"shard {ordinal}: proof preamble drift")
        if "FIRST_FAILURE" in output or "FIRST_EXCEPTION" in output:
            raise SystemExit(f"shard {ordinal}: failure marker present")

        reported = parse_reported_stats(output, ordinal)
        local_stats = {
            (edge, role): AggregateStat() for edge in EDGES for role in ROLES
        }
        raw_lines = ledger_path.read_bytes().splitlines(keepends=True)
        if len(raw_lines) != int(values.get("LEDGER_RECORDS", "-1")):
            raise SystemExit(f"shard {ordinal}: ledger record count mismatch")
        for raw_line in raw_lines:
            record = parse_ledger_record(raw_line)
            if record.key in all_records:
                raise SystemExit(f"duplicate ledger key: {record.key}")
            all_records[record.key] = record
            local_stats[(record.edge, record.role)].add_record(record)
            global_stats[(record.edge, record.role)].add_record(record)
        compare_reported_to_ledger(reported, local_stats, ordinal)

        for path, content in ((output_path, output_bytes), (ledger_path, b"".join(raw_lines))):
            bundle_digest.update(path.name.encode("ascii") + b"\0")
            bundle_digest.update(hashlib.sha256(content).digest())

    wanted = expected_keys()
    actual = set(all_records)
    if actual != wanted:
        raise SystemExit(
            f"ledger partition mismatch: missing={len(wanted - actual)} "
            f"extra={len(actual - wanted)}"
        )

    role_rank = {role: rank for rank, role in enumerate(ROLES)}
    ordered_keys = sorted(
        all_records,
        key=lambda item: (EDGES.index(item[0]), role_rank[item[1]], item[2], item[3]),
    )
    canonical_ledger = b"".join(all_records[key].raw_line for key in ordered_keys)
    ledger_digest = hashlib.sha256(canonical_ledger)

    assert preamble is not None
    lines = ["CERTIFICATE_KIND=CAPD_RIGOROUS_COVERING_AGGREGATE_V1"]
    lines.extend(f"{key}={preamble[key]}" for key in PREAMBLE_KEYS)
    lines.extend(
        (
            f"SHARDS={args.shards}",
            f"SOURCE_SHA256={manifest['SOURCE_SHA256']}",
            f"EXECUTABLE_SHA256={manifest['EXECUTABLE_SHA256']}",
            f"CAPD_CONFIG_SHA256={manifest['CAPD_CONFIG_SHA256']}",
            f"CAPD_CFLAGS_SHA256={manifest['CAPD_CFLAGS_SHA256']}",
            f"CAPD_LIBS_SHA256={manifest['CAPD_LIBS_SHA256']}",
            f"CXX_DRIVER_SHA256={manifest['CXX_DRIVER_SHA256']}",
            f"CXX_VERSION_SHA256={manifest['CXX_VERSION_SHA256']}",
            f"SLURM_JOB_ID={manifest['SLURM_JOB_ID']}",
            f"EXECUTION_TRUST_MODEL={manifest['EXECUTION_TRUST_MODEL']}",
            f"REMOTE_ATTESTATION_PRESENT={manifest['REMOTE_ATTESTATION_PRESENT']}",
            f"INDEPENDENT_REPLAY_REQUIRED={manifest['INDEPENDENT_REPLAY_REQUIRED']}",
            f"SHARD_BUNDLE_SHA256={bundle_digest.hexdigest()}",
            f"LEDGER_RECORDS={len(ordered_keys)}",
            f"LEDGER_SHA256={ledger_digest.hexdigest()}",
        )
    )
    for edge in EDGES:
        for role in ROLES:
            stat = global_stats[(edge, role)]
            lines.append(
                f"EDGE={edge} DEGREE={DEGREES[edge]} ROLE={role} "
                f"EXPECTED={stat.expected} PROCESSED={stat.processed} PASS={stat.passed} "
                f"MIN_MARGIN={stat.min_margin} RETURN_TIME=[{stat.min_return_time},"
                f"{stat.max_return_time}] "
                f"MIN_INITIAL_NORMAL_VELOCITY={stat.min_initial_normal} "
                f"MIN_NORMAL_VELOCITY={stat.min_final_normal} "
                f"MAX_PHYSICAL_DIAMETER={stat.max_physical_diameter}"
            )
        lines.append(f"COVERING_RELATION={edge} DEGREE={DEGREES[edge]} PROVED=true")
    lines.extend(
        (
            "ADJACENCY_MATRIX=[[1,1],[1,0]]",
            "FIBONACCI_COVERINGS_PROVED=true",
            "POSITIVE_ENTROPY_PROVED=true",
            "POINCARE_ENTROPY_LOWER_BOUND_FORMULA=log(phi)/6",
            "UNIFORM_HYPERBOLICITY_PROVED=false",
            "CHAOTIC_ATTRACTOR_PROVED=false",
            "FLOW_ENTROPY_BOUND_PROVED=false",
        )
    )
    certificate = ("\n".join(lines) + "\n").encode("ascii")
    write_fresh_atomic(args.ledger_output, canonical_ledger)
    write_fresh_atomic(args.certificate_output, certificate)
    print(f"CS6_CAPD_FIBONACCI_AGGREGATE PASS records={len(ordered_keys)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
