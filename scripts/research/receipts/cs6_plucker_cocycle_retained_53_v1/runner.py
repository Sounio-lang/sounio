#!/usr/bin/env python3
"""Run the event-normalized CS6 P1 cocycle on the retained 53 coordinates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CHALLENGE_DOMAIN = b"sounio.cs6.plucker-cocycle-leaf-challenge.v1\0"
ZERO_SHA256 = "0" * 64
PARENT_RUNS = (
    "cs6_c1_full_source_cover_core_scout_v1",
    "cs6_c1_full_source_cover_boundary_scout_v1",
    "cs6_c1_full_source_cover_dense_boundary_scout_v1",
)
VERIFICATION_KEYS = (
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


def die(message: str) -> None:
    raise SystemExit(message)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical_kv(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    path.write_bytes(
        "".join(f"{key}={value}\n" for key, value in fields).encode("ascii")
    )


def parse_kv_bytes(raw: bytes, expected_keys: Sequence[str]) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RuntimeError("non-ASCII verifier output") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise RuntimeError("noncanonical verifier output")
    lines = text.splitlines()
    if len(lines) != len(expected_keys):
        raise RuntimeError("verifier output line count mismatch")
    result: dict[str, str] = {}
    for line, expected in zip(lines, expected_keys, strict=True):
        if line.count("=") != 1:
            raise RuntimeError("malformed verifier output")
        key, value = line.split("=", 1)
        if key != expected or not value:
            raise RuntimeError(f"verifier output key mismatch: {expected}")
        result[key] = value
    return result


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


def leaf_challenge(root: str, identity: str, input_sha: str) -> str:
    return digest_bytes(
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(input_sha)
    )


@dataclass(frozen=True, order=True)
class Leaf:
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    input_sha: str

    @property
    def identity(self) -> str:
        return leaf_id(self.u_depth, self.u_index, self.s_depth, self.s_index)


@dataclass(frozen=True)
class Metrics:
    c1_width: float
    affine_width: float
    fixed_best_width: float
    homogeneous_width: float
    minimum_pivot_score: float
    chart_switches: int
    receipt_bytes: int


@dataclass(frozen=True)
class LeafResult:
    leaf: Leaf
    status: str
    method: str
    probe_pass: bool
    affine: bool
    projective_x: bool
    projective_y: bool
    projective_plus: bool
    projective_minus: bool
    homogeneous: bool
    certificate: bool
    subdivision: bool
    challenge: str
    receipt_sha: str
    verification_sha: str
    physical_sha: str
    worker_rc: int
    elapsed_ms: int
    metrics: Metrics | None


def parse_coordinate_manifest(path: Path, repo: Path) -> list[Leaf]:
    raw = path.read_bytes()
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RuntimeError("coordinate manifest must be ASCII") from error
    if not raw.endswith(b"\n") or len(lines) != 57:
        raise RuntimeError("coordinate manifest line count or terminator mismatch")
    if lines[:4] != [
        "SCHEMA=sounio.cs6.plucker-cocycle-coordinates.v1",
        "PARENT_COORDINATE_SET=CS6_C1_FULL_SOURCE_COVER_RETAINED_53",
        "LEAF_COUNT=53",
        "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256",
    ]:
        raise RuntimeError("coordinate manifest header mismatch")
    leaves: list[Leaf] = []
    for line in lines[4:]:
        fields = line.split("\t")
        if len(fields) != 6:
            raise RuntimeError("coordinate manifest row width mismatch")
        identity, *numbers, input_sha = fields
        if any(not token.isdigit() for token in numbers) or SHA_RE.fullmatch(input_sha) is None:
            raise RuntimeError("coordinate manifest row grammar mismatch")
        leaf = Leaf(*(int(token) for token in numbers), input_sha)
        if identity != leaf.identity:
            raise RuntimeError("coordinate manifest identity mismatch")
        canonical = leaf_input_bytes(
            leaf.u_depth, leaf.u_index, leaf.s_depth, leaf.s_index
        )
        if digest_bytes(canonical) != leaf.input_sha:
            raise RuntimeError("coordinate manifest input hash mismatch")
        leaves.append(leaf)
    if len(leaves) != 53 or len({leaf.identity for leaf in leaves}) != 53:
        raise RuntimeError("coordinate manifest is not a unique 53-leaf set")
    if leaves != sorted(leaves, key=lambda leaf: leaf.identity):
        raise RuntimeError("coordinate manifest is not canonically ordered")

    discovered: dict[str, str] = {}
    receipts = repo / "scripts/research/receipts"
    for parent in PARENT_RUNS:
        for input_path in sorted((receipts / parent / "inputs").glob("*.txt")):
            input_sha = digest(input_path)
            previous = discovered.setdefault(input_path.stem, input_sha)
            if previous != input_sha:
                raise RuntimeError("parent retained inputs conflict")
    expected = {leaf.identity: leaf.input_sha for leaf in leaves}
    if discovered != expected:
        raise RuntimeError("coordinate manifest differs from retained parent inputs")
    return leaves


def record_values(raw: bytes, marker: str) -> dict[str, str]:
    prefix = marker.encode("ascii") + b" "
    matches = [line for line in raw.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"receipt record cardinality mismatch: {marker}")
    values: dict[str, str] = {}
    for token in matches[0].decode("ascii").split(" ")[1:]:
        if token.count("=") != 1:
            raise RuntimeError(f"malformed receipt token: {marker}")
        key, value = token.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate receipt token: {marker} {key}")
        values[key] = value
    return values


def interval_bounds(token: str) -> tuple[float, float]:
    match = INTERVAL_RE.fullmatch(token)
    if match is None:
        raise RuntimeError("malformed retained interval")
    lower, upper = (float.fromhex(value) for value in match.groups())
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise RuntimeError("nonfinite or inverted retained interval")
    return lower, upper


def interval_width(record: dict[str, str], key: str) -> float:
    lower, upper = interval_bounds(record[key])
    width = upper - lower
    if not math.isfinite(width) or width <= 0.0:
        raise RuntimeError("retained diagnostic interval has nonpositive width")
    return width


def extract_metrics(receipt: Path) -> Metrics:
    raw = receipt.read_bytes()
    c1_width = interval_width(record_values(raw, "C1_P2_CONTROL"), "DET")
    affine_width = interval_width(record_values(raw, "AFFINE_CARRIER"), "DET")
    fixed_records = [
        record_values(raw, marker)
        for marker in ("PROJECTIVE_X", "PROJECTIVE_Y", "PROJECTIVE_PLUS", "PROJECTIVE_MINUS")
    ]
    fixed_widths = [interval_width(record, "DET") for record in fixed_records if record["ELIGIBLE"] == "true"]
    if not fixed_widths:
        raise RuntimeError("no eligible fixed final projective chart")
    homogeneous_width = interval_width(record_values(raw, "PLUCKER_COCYCLE"), "DET")
    ray_markers = (
        "HOMOGENEOUS_EVENT1_RAY0",
        "HOMOGENEOUS_EVENT1_RAY1",
        "HOMOGENEOUS_EVENT2_RAY0",
        "HOMOGENEOUS_EVENT2_RAY1",
    )
    rays = [record_values(raw, marker) for marker in ray_markers]
    scores: list[float] = []
    for ray in rays:
        if ray["ELIGIBLE"] != "true":
            raise RuntimeError("computed leaf has an ineligible dynamic chart")
        lower, upper = interval_bounds(ray["PIVOT"])
        margin = min(abs(lower), abs(upper))
        norm_squared = 2.0 if ray["CHART"] in {"PLUS", "MINUS"} else 1.0
        score = margin * margin / norm_squared
        if not math.isfinite(score) or score <= 0.0:
            raise RuntimeError("dynamic chart score is nonfinite or nonpositive")
        scores.append(score)
    chart_switches = sum(rays[index]["CHART"] != rays[index + 2]["CHART"] for index in range(2))
    return Metrics(
        c1_width,
        affine_width,
        min(fixed_widths),
        homogeneous_width,
        min(scores),
        chart_switches,
        len(raw),
    )


def dependency_paths(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        raise RuntimeError("compiler dependency file is malformed")
    return sorted({Path(item) for item in shlex.split(text.split(":", 1)[1])})


def dependency_manifest(paths: Sequence[Path], source: Path) -> bytes:
    rows: list[str] = []
    for path in paths:
        if path == source:
            rows.append(f"{digest(path)}  BUNDLE/worker-source.cpp")
        elif path.is_file():
            rows.append(f"{digest(path)}  {path}")
    if not rows:
        raise RuntimeError("compiler emitted no hashable dependencies")
    return ("\n".join(sorted(set(rows))) + "\n").encode("ascii")


def known_interval_domain_failure(stderr: bytes) -> bool:
    lowered = stderr.lower()
    return b"interval error:" in lowered and (
        b"division by 0" in lowered or b"division by zero" in lowered
    )


def bool_value(values: dict[str, str], key: str) -> bool:
    if values[key] not in {"true", "false"}:
        raise RuntimeError(f"noncanonical verifier boolean: {key}")
    return values[key] == "true"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def decimal(value: float) -> str:
    return format(value, ".17g")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capd-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root-challenge", required=True)
    parser.add_argument("--coordinate-manifest", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args(argv)
    if SHA_RE.fullmatch(args.root_challenge) is None:
        die("root challenge must be lowercase SHA-256")
    if not 1 <= args.jobs <= 32:
        die("jobs must be in [1,32]")
    if not 1 <= args.timeout_seconds <= 3600:
        die("timeout must be in [1,3600]")

    repo = Path(__file__).resolve().parents[2]
    source = repo / "scripts/research/cs6_plucker_cocycle_probe.cpp"
    verifier = repo / "scripts/research/cs6_plucker_cocycle_verify.py"
    runner = Path(__file__).resolve()
    coordinates = args.coordinate_manifest or (
        repo / "scripts/research/cs6_plucker_cocycle_coordinates_v1.tsv"
    )
    for required in (source, verifier, runner, coordinates):
        if not required.is_file():
            die(f"missing runner input: {required}")
    leaves = parse_coordinate_manifest(coordinates, repo)

    capd_config = args.capd_config.resolve()
    if not capd_config.is_file() or not os.access(capd_config, os.X_OK):
        die("capd-config is not executable")
    cxx_found = shutil.which(args.cxx)
    if cxx_found is None:
        die(f"C++ compiler not found: {args.cxx}")
    cxx = Path(cxx_found).resolve()
    python = Path(sys.executable).resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        die("run directory already exists")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cs6-plucker-cocycle.", dir=run_dir.parent))
    complete = False
    try:
        for directory in ("inputs", "receipts", "verifications", "stderr"):
            (work / directory).mkdir()
        source_snapshot = work / "worker-source.cpp"
        verifier_snapshot = work / "leaf-verifier.py"
        runner_snapshot = work / "runner.py"
        coordinate_snapshot = work / "coordinates.tsv"
        for source_path, target in (
            (source, source_snapshot),
            (verifier, verifier_snapshot),
            (runner, runner_snapshot),
            (coordinates, coordinate_snapshot),
        ):
            shutil.copy2(source_path, target)

        def capd(option: str) -> str:
            result = subprocess.run(
                [capd_config, option], check=True, capture_output=True, text=True
            )
            return result.stdout.strip()

        capd_version = capd("--modversion")
        capd_cflags = capd("--cflags")
        capd_libs = capd("--libs")
        if capd_version != "5.3.0":
            die(f"unsupported CAPD version: {capd_version}")
        flags = shlex.split(capd_cflags)
        if "-D__USE_FILIB__" not in flags or "-frounding-math" not in flags:
            die("CAPD config lacks FILIB outward-rounding flags")
        canonical_kv(
            work / "run-contract.txt",
            (
                ("SCHEMA", "sounio.cs6.plucker-cocycle-run-contract.v1"),
                ("SOURCE", "N0"),
                ("ROOT_CHALLENGE", args.root_challenge),
                ("COORDINATE_MANIFEST_SHA256", digest(coordinate_snapshot)),
                ("EXPECTED_LEAF_COUNT", "53"),
                ("JOBS", str(args.jobs)),
                ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
                ("DISCRETE_POINCARE_COCYCLE", "true"),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ),
        )
        (work / "capd-cflags.txt").write_text(capd_cflags + "\n", encoding="ascii")
        (work / "capd-libs.txt").write_text(capd_libs + "\n", encoding="ascii")
        (work / "capd-version.txt").write_text(capd_version + "\n", encoding="ascii")
        compiler_version = subprocess.run(
            [cxx, "--version"], check=True, capture_output=True
        ).stdout.rstrip(b"\n")
        (work / "compiler-version.txt").write_bytes(compiler_version + b"\n")
        (work / "python-version.txt").write_bytes(
            subprocess.run([python, "--version"], check=True, capture_output=True).stdout
        )
        (work / "git-head.txt").write_bytes(
            subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True).stdout
        )
        (work / "git-status.txt").write_bytes(
            subprocess.run(
                ["git", "-C", repo, "status", "--short", "--untracked-files=all"],
                check=True,
                capture_output=True,
            ).stdout
        )

        source_sha = digest(source_snapshot)
        dependency_file = work / "dependencies.d"
        binary = work / "worker-binary"
        compile_command = [
            str(cxx),
            "-std=c++17",
            *flags,
            "-O0",
            f'-DCS6_WORKER_SOURCE_SHA256="{source_sha}"',
            str(source_snapshot),
            "-MD",
            "-MF",
            str(dependency_file),
            "-o",
            str(binary),
            *shlex.split(capd_libs),
        ]
        (work / "compile-command.txt").write_text(
            shlex.join(compile_command) + "\n", encoding="ascii"
        )
        compiled = subprocess.run(compile_command, capture_output=True)
        (work / "compile-stdout.txt").write_bytes(compiled.stdout)
        (work / "compile-stderr.txt").write_bytes(compiled.stderr)
        if compiled.returncode != 0:
            die(f"worker compilation failed: {compiled.returncode}")
        dependencies_before = dependency_manifest(
            dependency_paths(dependency_file), source_snapshot
        )
        (work / "dependencies.sha256").write_bytes(dependencies_before)
        link_rows = sorted(
            f"{digest(Path(item))}  {Path(item)}"
            for item in shlex.split(capd_libs)
            if Path(item).is_file()
        )
        if not link_rows:
            die("CAPD link flags contain no hashable files")
        (work / "link-inputs.sha256").write_text(
            "\n".join(link_rows) + "\n", encoding="ascii"
        )
        linkage = subprocess.run(["ldd", binary], check=True, capture_output=True, text=True)
        (work / "runtime-linkage.txt").write_text(linkage.stdout, encoding="ascii")
        runtime_paths: set[Path] = set()
        for line in linkage.stdout.splitlines():
            fields = line.split()
            candidate = None
            if "=>" in fields and fields.index("=>") + 1 < len(fields):
                candidate = fields[fields.index("=>") + 1]
            elif fields and fields[0].startswith("/"):
                candidate = fields[0]
            if candidate and candidate.startswith("/") and Path(candidate).is_file():
                runtime_paths.add(Path(candidate))
        (work / "runtime-libraries.sha256").write_text(
            "".join(f"{digest(path)}  {path}\n" for path in sorted(runtime_paths)),
            encoding="ascii",
        )

        def unresolved(leaf: Leaf, status: str, challenge: str, receipt_sha: str, rc: int, elapsed: int) -> LeafResult:
            return LeafResult(
                leaf, status, "NONE", False, False, False, False, False, False,
                False, False, True, challenge, receipt_sha, ZERO_SHA256,
                ZERO_SHA256, rc, elapsed, None,
            )

        def run_leaf(leaf: Leaf) -> LeafResult:
            identity = leaf.identity
            input_path = work / "inputs" / f"{identity}.txt"
            receipt_path = work / "receipts" / f"{identity}.txt"
            verification_path = work / "verifications" / f"{identity}.txt"
            stderr_path = work / "stderr" / f"{identity}.txt"
            input_raw = leaf_input_bytes(
                leaf.u_depth, leaf.u_index, leaf.s_depth, leaf.s_index
            )
            input_path.write_bytes(input_raw)
            if digest_bytes(input_raw) != leaf.input_sha:
                raise RuntimeError(f"canonical input drift: {identity}")
            challenge = leaf_challenge(args.root_challenge, identity, leaf.input_sha)
            command = [
                str(binary), str(leaf.u_depth), str(leaf.u_index),
                str(leaf.s_depth), str(leaf.s_index), leaf.input_sha, challenge,
            ]
            started = time.monotonic_ns()
            try:
                worker = subprocess.run(command, capture_output=True, timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                elapsed = (time.monotonic_ns() - started) // 1_000_000
                receipt_path.write_bytes(error.stdout or b"")
                stderr_path.write_bytes(error.stderr or b"")
                return unresolved(leaf, "COMPUTATION_UNRESOLVED_TIMEOUT", challenge, digest(receipt_path), 124, elapsed)
            elapsed = (time.monotonic_ns() - started) // 1_000_000
            receipt_path.write_bytes(worker.stdout)
            stderr_path.write_bytes(worker.stderr)
            if worker.returncode != 0:
                if known_interval_domain_failure(worker.stderr):
                    return unresolved(
                        leaf, "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN", challenge,
                        digest(receipt_path), worker.returncode, elapsed,
                    )
                raise RuntimeError(f"unexpected worker failure for {identity}: rc={worker.returncode}")
            if worker.stderr:
                raise RuntimeError(f"worker emitted stderr for {identity}")
            verification = subprocess.run(
                [
                    python, verifier_snapshot, receipt_path, "--source-sha", source_sha,
                    "--input", input_path, "--challenge", challenge, "--require-probe",
                ],
                capture_output=True,
            )
            verification_path.write_bytes(verification.stdout)
            if verification.returncode != 0 or verification.stderr:
                (work / "stderr" / f"{identity}.verifier.txt").write_bytes(verification.stderr)
                raise RuntimeError(f"leaf verification failed for {identity}")
            values = parse_kv_bytes(verification.stdout, VERIFICATION_KEYS)
            probe_pass = bool_value(values, "PROBE_PASS")
            certificate = bool_value(values, "CERTIFICATE_PASS")
            subdivision = bool_value(values, "SUBDIVISION_REQUIRED")
            if not probe_pass or certificate == subdivision:
                raise RuntimeError(f"inconsistent verified status for {identity}")
            metrics = extract_metrics(receipt_path)
            return LeafResult(
                leaf,
                "PROBE_VALID_CERTIFIED" if certificate else "PROBE_VALID_UNCERTIFIED",
                values["LEAF_METHOD"],
                probe_pass,
                bool_value(values, "AFFINE_CERTIFICATE_PASS"),
                bool_value(values, "PROJECTIVE_X_CERTIFICATE_PASS"),
                bool_value(values, "PROJECTIVE_Y_CERTIFICATE_PASS"),
                bool_value(values, "PROJECTIVE_PLUS_CERTIFICATE_PASS"),
                bool_value(values, "PROJECTIVE_MINUS_CERTIFICATE_PASS"),
                bool_value(values, "HOMOGENEOUS_CERTIFICATE_PASS"),
                certificate,
                subdivision,
                challenge,
                values["RECEIPT_SHA256"],
                digest(verification_path),
                values["PHYSICAL_SHA256"],
                worker.returncode,
                elapsed,
                metrics,
            )

        results: list[LeafResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_leaf, leaf): leaf for leaf in leaves}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: result.leaf.identity)

        valid = [result for result in results if result.probe_pass]
        if not valid:
            raise RuntimeError("no leaf completed the homogeneous experiment")
        audit = valid[0]
        audit_id = audit.leaf.identity
        mutation = subprocess.run(
            [
                python, verifier_snapshot, work / "receipts" / f"{audit_id}.txt",
                "--source-sha", source_sha,
                "--input", work / "inputs" / f"{audit_id}.txt",
                "--challenge", audit.challenge,
                "--self-test-mutations", "--require-probe",
            ],
            capture_output=True,
        )
        (work / "mutation-audit.txt").write_bytes(mutation.stdout)
        (work / "mutation-audit-stderr.txt").write_bytes(mutation.stderr)
        if mutation.returncode != 0 or mutation.stderr:
            raise RuntimeError("mutation audit failed")
        mutation_values = parse_kv_bytes(mutation.stdout, VERIFICATION_KEYS)
        mutation_tests = int(mutation_values["MUTATION_TESTS"])
        mutation_rejected = int(mutation_values["MUTATIONS_REJECTED"])
        if mutation_tests == 0 or mutation_tests != mutation_rejected:
            raise RuntimeError("mutation audit did not reject every mutation")

        columns = (
            "LEAF_ID", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "STATUS",
            "METHOD", "PROBE_PASS", "AFFINE_PASS", "PROJECTIVE_X_PASS",
            "PROJECTIVE_Y_PASS", "PROJECTIVE_PLUS_PASS", "PROJECTIVE_MINUS_PASS",
            "HOMOGENEOUS_PASS", "CERTIFICATE_PASS", "SUBDIVISION_REQUIRED",
            "INPUT_SHA256", "LEAF_CHALLENGE", "RECEIPT_SHA256", "STDERR_SHA256",
            "VERIFICATION_SHA256", "PHYSICAL_SHA256", "WORKER_RC", "ELAPSED_MS",
            "C1_DET_WIDTH", "AFFINE_DET_WIDTH", "FIXED_BEST_DET_WIDTH",
            "HOMOGENEOUS_DET_WIDTH", "MINIMUM_PIVOT_SCORE", "CHART_SWITCHES",
            "RECEIPT_BYTES",
        )
        rows = ["\t".join(columns)]
        for result in results:
            metric_values = ("-",) * 7 if result.metrics is None else (
                decimal(result.metrics.c1_width),
                decimal(result.metrics.affine_width),
                decimal(result.metrics.fixed_best_width),
                decimal(result.metrics.homogeneous_width),
                decimal(result.metrics.minimum_pivot_score),
                str(result.metrics.chart_switches),
                str(result.metrics.receipt_bytes),
            )
            leaf = result.leaf
            stderr_sha = digest(work / "stderr" / f"{leaf.identity}.txt")
            rows.append("\t".join((
                leaf.identity, str(leaf.u_depth), str(leaf.u_index), str(leaf.s_depth),
                str(leaf.s_index), result.status, result.method,
                str(result.probe_pass).lower(), str(result.affine).lower(),
                str(result.projective_x).lower(), str(result.projective_y).lower(),
                str(result.projective_plus).lower(), str(result.projective_minus).lower(),
                str(result.homogeneous).lower(), str(result.certificate).lower(),
                str(result.subdivision).lower(), leaf.input_sha, result.challenge,
                result.receipt_sha, stderr_sha, result.verification_sha, result.physical_sha,
                str(result.worker_rc), str(result.elapsed_ms), *metric_values,
            )))
        (work / "leaves.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")

        metrics = [result.metrics for result in valid if result.metrics is not None]
        affine_count = sum(result.affine for result in valid)
        homogeneous_count = sum(result.homogeneous for result in valid)
        rescue_count = sum(result.homogeneous and not result.affine for result in valid)
        loss_count = sum(result.affine and not result.homogeneous for result in valid)
        c1_ratios = [item.homogeneous_width / item.c1_width for item in metrics]
        homogeneous_ratios = [item.homogeneous_width / item.affine_width for item in metrics]
        fixed_ratios = [item.homogeneous_width / item.fixed_best_width for item in metrics]
        summary_fields = (
            ("SCHEMA", "sounio.cs6.plucker-cocycle-summary.v1"),
            ("COORDINATE_COUNT", str(len(results))),
            ("PROBE_VALID_COUNT", str(len(valid))),
            ("COMPUTATION_UNRESOLVED_COUNT", str(len(results) - len(valid))),
            ("AFFINE_CERTIFIED_COUNT", str(affine_count)),
            ("HOMOGENEOUS_CERTIFIED_COUNT", str(homogeneous_count)),
            ("HOMOGENEOUS_RESCUE_COUNT", str(rescue_count)),
            ("AFFINE_LOSS_COUNT", str(loss_count)),
            ("HOMOGENEOUS_STRICT_SUPERSET_OF_AFFINE", str(loss_count == 0 and rescue_count > 0).lower()),
            ("PROJECTIVE_FIXED_ANY_CERTIFIED_COUNT", str(sum(
                result.projective_x or result.projective_y or result.projective_plus or result.projective_minus
                for result in valid
            ))),
            ("HOMOGENEOUS_NARROWER_THAN_AFFINE_COUNT", str(sum(item.homogeneous_width < item.affine_width for item in metrics))),
            ("HOMOGENEOUS_NARROWER_THAN_FIXED_BEST_COUNT", str(sum(item.homogeneous_width < item.fixed_best_width for item in metrics))),
            ("HOMOGENEOUS_NARROWER_THAN_C1_COUNT", str(sum(item.homogeneous_width < item.c1_width for item in metrics))),
            ("MIN_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(min(c1_ratios))),
            ("MEDIAN_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(statistics.median(c1_ratios))),
            ("MEAN_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(mean(c1_ratios))),
            ("MAX_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(max(c1_ratios))),
            ("MIN_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO", decimal(min(homogeneous_ratios))),
            ("MEDIAN_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO", decimal(statistics.median(homogeneous_ratios))),
            ("MEAN_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO", decimal(mean(homogeneous_ratios))),
            ("MAX_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO", decimal(max(homogeneous_ratios))),
            ("MIN_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO", decimal(min(fixed_ratios))),
            ("MEDIAN_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO", decimal(statistics.median(fixed_ratios))),
            ("MEAN_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO", decimal(mean(fixed_ratios))),
            ("MAX_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO", decimal(max(fixed_ratios))),
            ("MINIMUM_CERTIFIED_PIVOT_SCORE", decimal(min(item.minimum_pivot_score for item in metrics))),
            ("CHART_SWITCH_COUNT", str(sum(item.chart_switches for item in metrics))),
            ("ONE_RAY_CHART_SWITCH_LEAF_COUNT", str(sum(item.chart_switches == 1 for item in metrics))),
            ("TWO_RAY_CHART_SWITCH_LEAF_COUNT", str(sum(item.chart_switches == 2 for item in metrics))),
            ("TOTAL_WORKER_ELAPSED_MS", str(sum(result.elapsed_ms for result in results))),
            ("MEAN_VALID_WORKER_ELAPSED_MS", decimal(mean([float(result.elapsed_ms) for result in valid]))),
            ("MEAN_RECEIPT_BYTES", decimal(mean([float(item.receipt_bytes) for item in metrics]))),
            ("MUTATION_TESTS", str(mutation_tests)),
            ("MUTATIONS_REJECTED", str(mutation_rejected)),
            (
                "H_PG_CS6_SUPPORTED_ON_RETAINED_COORDINATES",
                str(loss_count == 0 and rescue_count > 0).lower(),
            ),
            ("DISCRETE_POINCARE_COCYCLE", "true"),
            ("CONTINUOUS_RICCATI_INTEGRATED", "false"),
            ("GENERAL_GRASSMANN_PLUCKER_INTEGRATOR", "false"),
            ("EXECUTION_TRUST_MODEL", "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION"),
            ("EXECUTION_PROVENANCE_ATTESTED", "false"),
            ("U250_USED", "false"),
            ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ("HYPERBOLICITY_PROVED", "false"),
            ("CHAOTIC_ATTRACTOR_PROVED", "false"),
            ("NOVELTY_OR_PRIORITY_CLAIM", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        )
        canonical_kv(work / "summary.txt", summary_fields)

        dependencies_after = dependency_manifest(dependency_paths(dependency_file), source_snapshot)
        if dependencies_before != dependencies_after:
            raise RuntimeError("compile dependency changed during execution")
        binary.unlink()
        dependency_file.unlink()
        artifacts = (
            "capd-cflags.txt", "capd-libs.txt", "capd-version.txt",
            "compile-command.txt", "compile-stderr.txt", "compile-stdout.txt",
            "compiler-version.txt", "coordinates.tsv", "dependencies.sha256",
            "git-head.txt", "git-status.txt", "leaf-verifier.py", "leaves.tsv",
            "link-inputs.sha256", "mutation-audit-stderr.txt",
            "mutation-audit.txt", "python-version.txt", "run-contract.txt",
            "runner.py", "runtime-libraries.sha256", "runtime-linkage.txt",
            "summary.txt", "worker-source.cpp",
        )
        manifest_fields: list[tuple[str, str]] = [
            ("SCHEMA", "sounio.cs6.plucker-cocycle-run-manifest.v1"),
            ("RUN_COMPLETE", "true"),
            ("CAPD_VERSION", capd_version),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("SOURCE_SHA256", source_sha),
            ("VERIFIER_SHA256", digest(verifier_snapshot)),
            ("RUNNER_SHA256", digest(runner_snapshot)),
            ("COORDINATE_MANIFEST_SHA256", digest(coordinate_snapshot)),
            ("LEAF_COUNT", str(len(results))),
            ("PROBE_VALID_COUNT", str(len(valid))),
            ("MUTATION_TESTS", str(mutation_tests)),
            ("PROMOTION_ELIGIBLE", "false"),
        ]
        for name in artifacts:
            manifest_fields.append((
                name.upper().replace("-", "_").replace(".", "_") + "_SHA256",
                digest(work / name),
            ))
        canonical_kv(work / "run-manifest.txt", manifest_fields)
        os.replace(work, run_dir)
        complete = True
    finally:
        if not complete:
            if args.keep_failed:
                print(f"FAILED_WORK_DIR={work}", file=sys.stderr)
            else:
                shutil.rmtree(work, ignore_errors=True)

    print(f"RUN_DIR={run_dir}")
    print("COORDINATE_COUNT=53")
    print(f"PROBE_VALID_COUNT={len(valid)}")
    print(f"AFFINE_CERTIFIED_COUNT={affine_count}")
    print(f"HOMOGENEOUS_CERTIFIED_COUNT={homogeneous_count}")
    print(f"HOMOGENEOUS_RESCUE_COUNT={rescue_count}")
    print("FULL_SOURCE_CARRIER_PROVED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
