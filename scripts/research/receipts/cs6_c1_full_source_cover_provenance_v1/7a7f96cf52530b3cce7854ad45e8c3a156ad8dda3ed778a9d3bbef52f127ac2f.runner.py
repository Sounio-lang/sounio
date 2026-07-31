#!/usr/bin/env python3
"""Compile once and execute a deterministic CS6 C1 full-source scout."""

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
from typing import Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
PAIR_RE = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CHALLENGE_DOMAIN = b"sounio.cs6.c1-cover-leaf-challenge.v1\0"


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
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--depth-pairs",
        default="8:8,12:12,14:14,15:15,12:16,16:12",
    )
    parser.add_argument("--include-root", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args(argv)

    if SHA_RE.fullmatch(args.root_challenge) is None:
        die("root challenge must be lowercase SHA-256")
    if not 1 <= args.jobs <= 32:
        die("jobs must be in [1,32]")
    if not 1 <= args.grid <= 8:
        die("grid must be in [1,8]")
    if not 1 <= args.timeout_seconds <= 3600:
        die("timeout must be in [1,3600]")
    pairs = parse_depth_pairs(args.depth_pairs)

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
        canonical_kv(
            work / "run-contract.txt",
            (
                ("SCHEMA", "sounio.cs6.c1-full-source-cover-scout-contract.v1"),
                ("SOURCE", "N0"),
                ("ROOT_CHALLENGE", args.root_challenge),
                ("DEPTH_PAIRS", args.depth_pairs),
                ("GRID", str(args.grid)),
                ("INCLUDE_ROOT", str(args.include_root).lower()),
                ("JOBS", str(args.jobs)),
                ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
                ("SCOUT_ONLY", "true"),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ),
        )
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

        leaves = {
            Leaf(du, stratum_index(du, u, args.grid), ds, stratum_index(ds, s, args.grid))
            for du, ds in pairs
            for u in range(args.grid)
            for s in range(args.grid)
        }
        if args.include_root:
            leaves.add(Leaf(0, 0, 0, 0))
        ordered_leaves = sorted(leaves, key=lambda leaf: leaf.identity)

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

        results: list[LeafResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_leaf, leaf): leaf for leaf in ordered_leaves}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
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
        (work / "scout.tsv").write_text("".join(rows), encoding="ascii")

        summary_rows = [
            ("SCHEMA", "sounio.cs6.c1-full-source-cover-scout-summary.v1"),
            ("SCOUT_ONLY", "true"),
            ("LEAF_COUNT", str(len(results))),
            ("CERTIFIED_COUNT", str(len(certified))),
            (
                "SUBDIVISION_REQUIRED_COUNT",
                str(sum(result.status == "SUBDIVISION_REQUIRED" for result in results)),
            ),
            (
                "COMPUTATION_UNRESOLVED_COUNT",
                str(sum(result.status.startswith("COMPUTATION_UNRESOLVED") for result in results)),
            ),
            ("MUTATION_TESTS", str(mutation_tests)),
            ("MUTATIONS_REJECTED", str(mutation_rejected)),
            ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ("PROJECTIVE_RICCATI_INTEGRATED", "false"),
            ("HYPERBOLICITY_PROVED", "false"),
            ("CHAOTIC_ATTRACTOR_PROVED", "false"),
            ("U250_USED", "false"),
        ]
        canonical_kv(work / "summary.txt", summary_rows)

        dependencies_after = dependency_manifest(dep_paths, source_snapshot)
        if dependencies_before != dependencies_after:
            raise RuntimeError("compile dependency changed during scout")

        artifact_names = (
            "run-contract.txt",
            "worker-source.cpp",
            "leaf-verifier.py",
            "runner.py",
            "worker-binary",
            "compile-command.txt",
            "dependencies.sha256",
            "link-inputs.sha256",
            "runtime-libraries.sha256",
            "scout.tsv",
            "summary.txt",
            "mutation-audit.txt",
        )
        manifest_fields: list[tuple[str, str]] = [
            ("SCHEMA", "sounio.cs6.c1-full-source-cover-scout-manifest.v1"),
            ("RUN_COMPLETE", "true"),
            ("CAPD_VERSION", capd_version),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("SOURCE_SHA256", source_sha),
            ("LEAF_COUNT", str(len(results))),
            ("CERTIFIED_COUNT", str(len(certified))),
            ("MUTATION_TESTS", str(mutation_tests)),
            ("SCOUT_ONLY", "true"),
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
    print(f"LEAF_COUNT={len(results)}")
    print(f"CERTIFIED_COUNT={len(certified)}")
    print("SCOUT_ONLY=true")
    print("FULL_SOURCE_CARRIER_PROVED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
