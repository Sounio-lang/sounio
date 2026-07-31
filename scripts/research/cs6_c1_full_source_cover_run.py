#!/usr/bin/env python3
"""Compile once and execute a deterministic CS6 C1 scout or adaptive cover."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_RE = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CHALLENGE_DOMAIN = b"sounio.cs6.c1-cover-leaf-challenge.v1\0"
ZERO_SHA256 = "0" * 64
NODE_COLUMNS = (
    "NODE_ID",
    "PARENT_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "ACTION",
    "METHOD",
    "INPUT_PATH",
    "INPUT_SHA256",
    "LEAF_CHALLENGE",
    "RECEIPT_PATH",
    "RECEIPT_SHA256",
    "VERIFICATION_PATH",
    "VERIFICATION_SHA256",
    "PHYSICAL_SHA256",
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
    result: dict[str, str] = {}
    lines = text.splitlines()
    if len(lines) != len(expected_keys):
        raise RuntimeError("verifier output line count mismatch")
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
    material = (
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(input_sha)
    )
    return digest_bytes(material)


def stratum_index(depth: int, position: int, grid: int) -> int:
    return ((2 * position + 1) * (1 << depth)) // (2 * grid)


def parse_depth_pairs(token: str) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for item in token.split(","):
        match = PAIR_RE.fullmatch(item)
        if match is None:
            die(f"invalid depth pair: {item}")
        pair = (int(match.group(1)), int(match.group(2)))
        if pair[0] > 30 or pair[1] > 30:
            die("depth pair exceeds worker contract")
        if pair in pairs:
            die(f"duplicate depth pair: {item}")
        pairs.append(pair)
    if not pairs:
        die("at least one depth pair is required")
    return tuple(pairs)


@dataclass(frozen=True, order=True)
class Leaf:
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int

    @property
    def identity(self) -> str:
        return leaf_id(self.u_depth, self.u_index, self.s_depth, self.s_index)


@dataclass(frozen=True)
class LeafResult:
    leaf: Leaf
    status: str
    method: str
    certificate: bool
    subdivision: bool
    input_sha: str
    challenge: str
    receipt_sha: str
    verification_sha: str
    physical_sha: str
    worker_rc: int
    elapsed_ms: int


@dataclass(frozen=True)
class CoverNode:
    leaf: Leaf
    parent: str
    action: str
    result: LeafResult


def scout_leaves(
    pairs: Sequence[tuple[int, int]], grid: int, include_root: bool
) -> list[Leaf]:
    for u_depth, s_depth in pairs:
        if grid > 1 << u_depth or grid > 1 << s_depth:
            raise RuntimeError(
                f"grid {grid} exceeds dyadic tiles at depth pair {u_depth}:{s_depth}"
            )
    samples = [
        Leaf(
            u_depth,
            stratum_index(u_depth, u_position, grid),
            s_depth,
            stratum_index(s_depth, s_position, grid),
        )
        for u_depth, s_depth in pairs
        for u_position in range(grid)
        for s_position in range(grid)
    ]
    if len(set(samples)) != len(samples):
        raise RuntimeError("scout strata do not map to unique dyadic leaves")
    root = Leaf(0, 0, 0, 0)
    if include_root:
        if root in samples:
            raise RuntimeError("--include-root duplicates a configured scout leaf")
        samples.append(root)
    expected = len(pairs) * grid * grid + int(include_root)
    if len(samples) != expected or len(set(samples)) != expected:
        raise RuntimeError("scout leaf cardinality differs from its contract")
    return sorted(samples, key=lambda leaf: leaf.identity)


def split_leaf(leaf: Leaf, max_axis_depth: int) -> tuple[str, tuple[Leaf, Leaf]] | None:
    if leaf.u_depth <= leaf.s_depth:
        if leaf.u_depth >= max_axis_depth:
            return None
        return (
            "SPLIT_U",
            (
                Leaf(leaf.u_depth + 1, 2 * leaf.u_index, leaf.s_depth, leaf.s_index),
                Leaf(
                    leaf.u_depth + 1,
                    2 * leaf.u_index + 1,
                    leaf.s_depth,
                    leaf.s_index,
                ),
            ),
        )
    if leaf.s_depth >= max_axis_depth:
        return None
    return (
        "SPLIT_S",
        (
            Leaf(leaf.u_depth, leaf.u_index, leaf.s_depth + 1, 2 * leaf.s_index),
            Leaf(
                leaf.u_depth,
                leaf.u_index,
                leaf.s_depth + 1,
                2 * leaf.s_index + 1,
            ),
        ),
    )


def build_adaptive_tree(
    evaluate_wave: Callable[[Sequence[Leaf]], Sequence[LeafResult]],
    max_nodes: int,
    max_axis_depth: int,
) -> tuple[list[LeafResult], dict[str, CoverNode], list[list[str]]]:
    if max_nodes < 1 or not 0 <= max_axis_depth <= 30:
        raise RuntimeError("invalid adaptive tree bounds")
    root = Leaf(0, 0, 0, 0)
    frontier = [root]
    parents = {root.identity: "-"}
    allocated = 1
    results: list[LeafResult] = []
    nodes: dict[str, CoverNode] = {}
    waves: list[list[str]] = []

    while frontier:
        frontier.sort(key=lambda leaf: leaf.identity)
        waves.append([leaf.identity for leaf in frontier])
        wave_results = sorted(evaluate_wave(frontier), key=lambda item: item.leaf.identity)
        if [item.leaf for item in wave_results] != frontier:
            raise RuntimeError("adaptive evaluator returned a noncanonical frontier")
        results.extend(wave_results)
        next_frontier: list[Leaf] = []
        for result in wave_results:
            identity = result.leaf.identity
            if identity in nodes:
                raise RuntimeError(f"adaptive evaluator repeated node: {identity}")
            if result.certificate == result.subdivision:
                raise RuntimeError(f"inconsistent adaptive result: {identity}")
            if result.certificate:
                action = "CERTIFIED"
            else:
                split = split_leaf(result.leaf, max_axis_depth)
                if split is None or allocated + 2 > max_nodes:
                    action = "UNRESOLVED"
                else:
                    action, children = split
                    allocated += 2
                    for child in children:
                        if child.identity in parents:
                            raise RuntimeError(
                                f"adaptive split repeated child: {child.identity}"
                            )
                        parents[child.identity] = identity
                        next_frontier.append(child)
            nodes[identity] = CoverNode(
                result.leaf, parents[identity], action, result
            )
        frontier = next_frontier

    if len(nodes) != allocated:
        raise RuntimeError("adaptive tree node accounting mismatch")
    return results, nodes, waves


def cover_node_fields(node: CoverNode) -> tuple[str, ...]:
    identity = node.leaf.identity
    result = node.result
    if node.action != "CERTIFIED":
        method = "NONE"
        evidence = ("-",) * 8
    else:
        method = result.method
        verification_present = result.verification_sha != ZERO_SHA256
        physical_present = result.physical_sha != ZERO_SHA256
        evidence = (
            f"inputs/{identity}.txt",
            result.input_sha,
            result.challenge,
            f"receipts/{identity}.txt",
            result.receipt_sha,
            f"verifications/{identity}.txt" if verification_present else "-",
            result.verification_sha if verification_present else "-",
            result.physical_sha if physical_present else "-",
        )
    fields = (
        identity,
        node.parent,
        str(node.leaf.u_depth),
        str(node.leaf.u_index),
        str(node.leaf.s_depth),
        str(node.leaf.s_index),
        node.action,
        method,
        *evidence,
    )
    if len(fields) != len(NODE_COLUMNS):
        raise RuntimeError("adaptive node row differs from aggregate schema")
    return fields


def nodes_tsv_bytes(nodes: dict[str, CoverNode]) -> bytes:
    rows = ["\t".join(NODE_COLUMNS)]
    rows.extend("\t".join(cover_node_fields(nodes[identity])) for identity in sorted(nodes))
    return ("\n".join(rows) + "\n").encode("ascii")


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capd-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root-challenge", required=True)
    parser.add_argument("--mode", choices=("scout", "adaptive"), default="scout")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--depth-pairs",
        default="8:8,12:12,14:14,15:15,12:16,16:12",
    )
    parser.add_argument("--include-root", action="store_true")
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-axis-depth", type=int)
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args(argv)

    if SHA_RE.fullmatch(args.root_challenge) is None:
        die("root challenge must be lowercase SHA-256")
    if not 1 <= args.jobs <= 32:
        die("jobs must be in [1,32]")
    if not 1 <= args.timeout_seconds <= 3600:
        die("timeout must be in [1,3600]")
    pairs: tuple[tuple[int, int], ...] = ()
    ordered_leaves: list[Leaf] = []
    if args.mode == "scout":
        if args.max_nodes is not None or args.max_axis_depth is not None:
            die("adaptive bounds cannot be supplied in scout mode")
        if not 1 <= args.grid <= 8:
            die("grid must be in [1,8]")
        pairs = parse_depth_pairs(args.depth_pairs)
        try:
            ordered_leaves = scout_leaves(pairs, args.grid, args.include_root)
        except RuntimeError as error:
            die(str(error))
    else:
        if args.max_nodes is None or args.max_axis_depth is None:
            die("adaptive mode requires --max-nodes and --max-axis-depth")
        if args.max_nodes < 1:
            die("max-nodes must be positive")
        if not 0 <= args.max_axis_depth <= 30:
            die("max-axis-depth must be in [0,30]")

    repo = Path(__file__).resolve().parents[2]
    source = repo / "scripts/research/cs6_c1_full_source_cover_probe.cpp"
    verifier = repo / "scripts/research/cs6_c1_full_source_cover_leaf_verify.py"
    runner = Path(__file__).resolve()
    for required in (source, verifier, runner):
        if not required.is_file():
            die(f"missing runner input: {required}")

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

    work = Path(tempfile.mkdtemp(prefix=".cs6-c1-full-source-cover.", dir=run_dir.parent))
    complete = False
    try:
        for directory in ("inputs", "receipts", "verifications", "stderr"):
            (work / directory).mkdir()
        source_snapshot = work / "worker-source.cpp"
        verifier_snapshot = work / "leaf-verifier.py"
        runner_snapshot = work / "runner.py"
        shutil.copy2(source, source_snapshot)
        shutil.copy2(verifier, verifier_snapshot)
        shutil.copy2(runner, runner_snapshot)

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
        if "-D__USE_FILIB__" not in shlex.split(capd_cflags):
            die("CAPD config does not select FILIB")
        if "-frounding-math" not in shlex.split(capd_cflags):
            die("CAPD config omits -frounding-math")
        if args.mode == "scout":
            contract_fields = (
                ("SCHEMA", "sounio.cs6.c1-full-source-cover-scout-contract.v1"),
                ("SOURCE", "N0"),
                ("MODE", "scout"),
                ("ROOT_CHALLENGE", args.root_challenge),
                ("DEPTH_PAIRS", args.depth_pairs),
                ("GRID", str(args.grid)),
                ("INCLUDE_ROOT", str(args.include_root).lower()),
                ("EXPECTED_LEAF_COUNT", str(len(ordered_leaves))),
                ("JOBS", str(args.jobs)),
                ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
                ("SCOUT_ONLY", "true"),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
            )
        else:
            assert args.max_nodes is not None
            assert args.max_axis_depth is not None
            contract_fields = (
                ("SCHEMA", "sounio.cs6.c1-full-source-cover-adaptive-contract.v1"),
                ("SOURCE", "N0"),
                ("MODE", "adaptive"),
                ("ROOT_CHALLENGE", args.root_challenge),
                ("ROOT_NODE", leaf_id(0, 0, 0, 0)),
                ("TRAVERSAL", "DETERMINISTIC_BREADTH_FIRST"),
                ("SPLIT_RULE", "SHALLOWER_AXIS_TIE_U"),
                ("MAX_NODES", str(args.max_nodes)),
                ("MAX_AXIS_DEPTH", str(args.max_axis_depth)),
                ("JOBS", str(args.jobs)),
                ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
                ("SCOUT_ONLY", "false"),
                ("AGGREGATION_REQUIRED", "true"),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
            )
        canonical_kv(work / "run-contract.txt", contract_fields)
        (work / "capd-cflags.txt").write_text(capd_cflags + "\n", encoding="ascii")
        (work / "capd-libs.txt").write_text(capd_libs + "\n", encoding="ascii")
        (work / "capd-version.txt").write_text(capd_version + "\n", encoding="ascii")
        (work / "compiler-version.txt").write_bytes(
            subprocess.run([cxx, "--version"], check=True, capture_output=True).stdout
        )
        (work / "python-version.txt").write_bytes(
            subprocess.run([python, "--version"], check=True, capture_output=True).stdout
        )
        (work / "git-head.txt").write_bytes(
            subprocess.run(
                ["git", "-C", repo, "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
            ).stdout
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
            *shlex.split(capd_cflags),
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
        compile_result = subprocess.run(compile_command, capture_output=True)
        (work / "compile-stdout.txt").write_bytes(compile_result.stdout)
        (work / "compile-stderr.txt").write_bytes(compile_result.stderr)
        if compile_result.returncode != 0:
            die(f"worker compilation failed: {compile_result.returncode}")

        dep_paths = dependency_paths(dependency_file)
        dependencies_before = dependency_manifest(dep_paths, source_snapshot)
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

        verification_keys = (
            "VERIFICATION_SCHEMA",
            "RECEIPT_SHA256",
            "PHYSICAL_SHA256",
            "MUTATION_TESTS",
            "MUTATIONS_REJECTED",
            "LEAF_METHOD",
            "SUBDIVISION_REQUIRED",
            "CERTIFICATE_PASS",
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
            input_sha = digest_bytes(input_raw)
            challenge = leaf_challenge(args.root_challenge, identity, input_sha)
            command = [
                str(binary),
                str(leaf.u_depth),
                str(leaf.u_index),
                str(leaf.s_depth),
                str(leaf.s_index),
                input_sha,
                challenge,
            ]
            started = time.monotonic_ns()
            try:
                worker = subprocess.run(
                    command, capture_output=True, timeout=args.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                elapsed = (time.monotonic_ns() - started) // 1_000_000
                receipt_path.write_bytes(error.stdout or b"")
                stderr_path.write_bytes(error.stderr or b"")
                return LeafResult(
                    leaf, "COMPUTATION_UNRESOLVED_TIMEOUT", "NONE", False, True,
                    input_sha, challenge, digest(receipt_path), "0" * 64, "0" * 64,
                    124, elapsed,
                )
            elapsed = (time.monotonic_ns() - started) // 1_000_000
            receipt_path.write_bytes(worker.stdout)
            stderr_path.write_bytes(worker.stderr)
            if worker.returncode != 0:
                if known_interval_domain_failure(worker.stderr):
                    return LeafResult(
                        leaf, "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN", "NONE",
                        False, True, input_sha, challenge, digest(receipt_path),
                        "0" * 64, "0" * 64, worker.returncode, elapsed,
                    )
                raise RuntimeError(
                    f"unexpected worker failure for {identity}: rc={worker.returncode}"
                )
            if worker.stderr:
                raise RuntimeError(f"worker emitted stderr for {identity}")
            verification = subprocess.run(
                [
                    python,
                    verifier_snapshot,
                    receipt_path,
                    "--source-sha",
                    source_sha,
                    "--input",
                    input_path,
                    "--challenge",
                    challenge,
                ],
                capture_output=True,
            )
            verification_path.write_bytes(verification.stdout)
            if verification.returncode != 0 or verification.stderr:
                (work / "stderr" / f"{identity}.verifier.txt").write_bytes(
                    verification.stderr
                )
                raise RuntimeError(f"leaf verification failed for {identity}")
            values = parse_kv_bytes(verification.stdout, verification_keys)
            certificate = values["CERTIFICATE_PASS"] == "true"
            subdivision = values["SUBDIVISION_REQUIRED"] == "true"
            if certificate == subdivision:
                raise RuntimeError(f"inconsistent terminal status for {identity}")
            return LeafResult(
                leaf,
                "CERTIFIED" if certificate else "SUBDIVISION_REQUIRED",
                values["LEAF_METHOD"],
                certificate,
                subdivision,
                input_sha,
                challenge,
                values["RECEIPT_SHA256"],
                digest(verification_path),
                values["PHYSICAL_SHA256"],
                worker.returncode,
                elapsed,
            )

        def evaluate_wave(leaves: Sequence[Leaf]) -> list[LeafResult]:
            wave_results: list[LeafResult] = []
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.jobs
            ) as executor:
                futures = {executor.submit(run_leaf, leaf): leaf for leaf in leaves}
                for future in concurrent.futures.as_completed(futures):
                    wave_results.append(future.result())
            wave_results.sort(key=lambda result: result.leaf.identity)
            return wave_results

        cover_nodes: dict[str, CoverNode] = {}
        waves: list[list[str]] = []
        if args.mode == "scout":
            results = evaluate_wave(ordered_leaves)
        else:
            assert args.max_nodes is not None
            assert args.max_axis_depth is not None
            results, cover_nodes, waves = build_adaptive_tree(
                evaluate_wave, args.max_nodes, args.max_axis_depth
            )
        results.sort(key=lambda result: result.leaf.identity)

        certified = [result for result in results if result.certificate]
        mutation_tests = mutation_rejected = 0
        if certified:
            audit = certified[0]
            identity = audit.leaf.identity
            mutation = subprocess.run(
                [
                    python,
                    verifier_snapshot,
                    work / "receipts" / f"{identity}.txt",
                    "--source-sha",
                    source_sha,
                    "--input",
                    work / "inputs" / f"{identity}.txt",
                    "--challenge",
                    audit.challenge,
                    "--self-test-mutations",
                    "--require-terminal",
                ],
                capture_output=True,
            )
            (work / "mutation-audit.txt").write_bytes(mutation.stdout)
            (work / "mutation-audit-stderr.txt").write_bytes(mutation.stderr)
            if mutation.returncode != 0 or mutation.stderr:
                raise RuntimeError("retained mutation audit failed")
            mutation_values = parse_kv_bytes(mutation.stdout, verification_keys)
            mutation_tests = int(mutation_values["MUTATION_TESTS"])
            mutation_rejected = int(mutation_values["MUTATIONS_REJECTED"])
            if mutation_tests == 0 or mutation_tests != mutation_rejected:
                raise RuntimeError("mutation audit did not reject every mutation")
        else:
            (work / "mutation-audit.txt").write_text(
                "MUTATION_AUDIT_SKIPPED=NO_CERTIFIED_LEAF\n", encoding="ascii"
            )
            (work / "mutation-audit-stderr.txt").write_bytes(b"")

        header = (
            "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tSTATUS\tMETHOD\t"
            "CERTIFICATE_PASS\tSUBDIVISION_REQUIRED\tINPUT_SHA256\tLEAF_CHALLENGE\t"
            "RECEIPT_SHA256\tVERIFICATION_SHA256\tPHYSICAL_SHA256\tWORKER_RC\tELAPSED_MS\n"
        )
        rows = [header]
        for result in results:
            leaf = result.leaf
            rows.append(
                "\t".join(
                    (
                        leaf.identity,
                        str(leaf.u_depth),
                        str(leaf.u_index),
                        str(leaf.s_depth),
                        str(leaf.s_index),
                        result.status,
                        result.method,
                        str(result.certificate).lower(),
                        str(result.subdivision).lower(),
                        result.input_sha,
                        result.challenge,
                        result.receipt_sha,
                        result.verification_sha,
                        result.physical_sha,
                        str(result.worker_rc),
                        str(result.elapsed_ms),
                    )
                )
                + "\n"
            )
        evaluation_name = "scout.tsv" if args.mode == "scout" else "evaluations.tsv"
        (work / evaluation_name).write_text("".join(rows), encoding="ascii")

        if args.mode == "adaptive":
            (work / "nodes.tsv").write_bytes(nodes_tsv_bytes(cover_nodes))

        subdivision_count = sum(
            result.status == "SUBDIVISION_REQUIRED" for result in results
        )
        computation_unresolved_count = sum(
            result.status.startswith("COMPUTATION_UNRESOLVED")
            for result in results
        )
        if args.mode == "scout":
            summary_rows = [
                ("SCHEMA", "sounio.cs6.c1-full-source-cover-scout-summary.v1"),
                ("MODE", "scout"),
                ("SCOUT_ONLY", "true"),
                ("LEAF_COUNT", str(len(results))),
                ("CERTIFIED_COUNT", str(len(certified))),
                ("SUBDIVISION_REQUIRED_COUNT", str(subdivision_count)),
                ("COMPUTATION_UNRESOLVED_COUNT", str(computation_unresolved_count)),
                ("MUTATION_TESTS", str(mutation_tests)),
                ("MUTATIONS_REJECTED", str(mutation_rejected)),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
                ("PROJECTIVE_RICCATI_INTEGRATED", "false"),
                ("HYPERBOLICITY_PROVED", "false"),
                ("CHAOTIC_ATTRACTOR_PROVED", "false"),
                ("U250_USED", "false"),
            ]
        else:
            assert args.max_nodes is not None
            assert args.max_axis_depth is not None
            terminal_nodes = [
                node
                for node in cover_nodes.values()
                if node.action in {"CERTIFIED", "UNRESOLVED"}
            ]
            unresolved_nodes = [
                node for node in terminal_nodes if node.action == "UNRESOLVED"
            ]
            depth_limited = sum(
                split_leaf(node.leaf, args.max_axis_depth) is None
                for node in unresolved_nodes
            )
            budget_limited = len(unresolved_nodes) - depth_limited
            summary_rows = [
                ("SCHEMA", "sounio.cs6.c1-full-source-cover-adaptive-summary.v1"),
                ("MODE", "adaptive"),
                ("SCOUT_ONLY", "false"),
                ("BOUNDED_RUN_COMPLETE", "true"),
                ("EVALUATED_NODE_COUNT", str(len(results))),
                ("TREE_NODE_COUNT", str(len(cover_nodes))),
                ("WAVE_COUNT", str(len(waves))),
                (
                    "SPLIT_NODE_COUNT",
                    str(
                        sum(
                            node.action in {"SPLIT_U", "SPLIT_S"}
                            for node in cover_nodes.values()
                        )
                    ),
                ),
                (
                    "CERTIFIED_TERMINAL_COUNT",
                    str(sum(node.action == "CERTIFIED" for node in terminal_nodes)),
                ),
                ("UNRESOLVED_TERMINAL_COUNT", str(len(unresolved_nodes))),
                ("DEPTH_LIMITED_UNRESOLVED_COUNT", str(depth_limited)),
                ("NODE_BUDGET_LIMITED_UNRESOLVED_COUNT", str(budget_limited)),
                ("SUBDIVISION_REQUIRED_EVALUATION_COUNT", str(subdivision_count)),
                (
                    "COMPUTATION_UNRESOLVED_EVALUATION_COUNT",
                    str(computation_unresolved_count),
                ),
                ("MAX_NODES", str(args.max_nodes)),
                ("MAX_AXIS_DEPTH", str(args.max_axis_depth)),
                (
                    "MAX_U_DEPTH_REACHED",
                    str(max(node.leaf.u_depth for node in cover_nodes.values())),
                ),
                (
                    "MAX_S_DEPTH_REACHED",
                    str(max(node.leaf.s_depth for node in cover_nodes.values())),
                ),
                ("MUTATION_TESTS", str(mutation_tests)),
                ("MUTATIONS_REJECTED", str(mutation_rejected)),
                ("AGGREGATION_REQUIRED", "true"),
                ("PROMOTION_ELIGIBLE", "false"),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
                ("PROJECTIVE_RICCATI_INTEGRATED", "false"),
                ("HYPERBOLICITY_PROVED", "false"),
                ("CHAOTIC_ATTRACTOR_PROVED", "false"),
                ("U250_USED", "false"),
            ]
        canonical_kv(work / "summary.txt", summary_rows)

        dependencies_after = dependency_manifest(dep_paths, source_snapshot)
        if dependencies_before != dependencies_after:
            raise RuntimeError("compile dependency changed during execution")

        artifact_names = [
            "run-contract.txt",
            "worker-source.cpp",
            "leaf-verifier.py",
            "runner.py",
            "worker-binary",
            "compile-command.txt",
            "dependencies.sha256",
            "link-inputs.sha256",
            "runtime-libraries.sha256",
            evaluation_name,
            "summary.txt",
            "mutation-audit.txt",
        ]
        if args.mode == "adaptive":
            artifact_names.append("nodes.tsv")
        if args.mode == "scout":
            manifest_schema = "sounio.cs6.c1-full-source-cover-scout-manifest.v1"
            count_fields = (
                ("LEAF_COUNT", str(len(results))),
                ("CERTIFIED_COUNT", str(len(certified))),
                ("SCOUT_ONLY", "true"),
            )
        else:
            manifest_schema = "sounio.cs6.c1-full-source-cover-adaptive-manifest.v1"
            count_fields = (
                ("EVALUATED_NODE_COUNT", str(len(results))),
                ("TREE_NODE_COUNT", str(len(cover_nodes))),
                (
                    "CERTIFIED_TERMINAL_COUNT",
                    str(
                        sum(
                            node.action == "CERTIFIED"
                            for node in cover_nodes.values()
                        )
                    ),
                ),
                (
                    "UNRESOLVED_TERMINAL_COUNT",
                    str(
                        sum(
                            node.action == "UNRESOLVED"
                            for node in cover_nodes.values()
                        )
                    ),
                ),
                ("SCOUT_ONLY", "false"),
                ("AGGREGATION_REQUIRED", "true"),
            )
        manifest_fields: list[tuple[str, str]] = [
            ("SCHEMA", manifest_schema),
            ("MODE", args.mode),
            ("RUN_COMPLETE", "true"),
            ("CAPD_VERSION", capd_version),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("SOURCE_SHA256", source_sha),
            *count_fields,
            ("MUTATION_TESTS", str(mutation_tests)),
            ("EXECUTION_TRUST_MODEL", "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION"),
            ("REMOTE_ATTESTATION_PRESENT", "false"),
            ("INDEPENDENT_REPLAY_REQUIRED", "true"),
            ("PROMOTION_ELIGIBLE", "false"),
            ("FULL_SOURCE_CARRIER_PROVED", "false"),
        ]
        for name in artifact_names:
            manifest_fields.append(
                (name.upper().replace("-", "_").replace(".", "_") + "_SHA256", digest(work / name))
            )
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
    print(f"MODE={args.mode}")
    if args.mode == "scout":
        print(f"LEAF_COUNT={len(results)}")
        print(f"CERTIFIED_COUNT={len(certified)}")
        print("SCOUT_ONLY=true")
    else:
        print(f"EVALUATED_NODE_COUNT={len(results)}")
        print(f"TREE_NODE_COUNT={len(cover_nodes)}")
        print(
            "UNRESOLVED_TERMINAL_COUNT="
            + str(
                sum(
                    node.action == "UNRESOLVED"
                    for node in cover_nodes.values()
                )
            )
        )
        print("SCOUT_ONLY=false")
        print("AGGREGATION_REQUIRED=true")
    print("FULL_SOURCE_CARRIER_PROVED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
