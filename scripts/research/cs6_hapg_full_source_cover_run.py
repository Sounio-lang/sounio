#!/usr/bin/env python3
"""Run a two-stage H-PG to fixed-chart H-APG KAT or adaptive cover."""

from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import errno
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
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Callable, Mapping, Sequence


sys.dont_write_bytecode = True

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
KAT_CERTIFICATE_SCHEMA = "sounio.cs6.hapg-full-source-cover-kat-prerequisite.v2"
KAT_RUN_CONTRACT_EVIDENCE_KEYS = (
    "KAT_COORDINATE_MANIFEST_SHA256",
    "KAT_EXPECTED_RESULTS_SHA256",
    "KAT_WAVE_CONTRACT_SHA256",
    "KAT_WAVE_RESULT_SHA256",
    "KAT_LEAF_EVIDENCE_VALID",
    "KAT_HPG_VERIFIER_REPLAY_COUNT",
    "KAT_HAPG_VERIFIER_REPLAY_COUNT",
    "KAT_EVALUATED_NODE_COUNT",
    "KAT_HPG_SIGNED_CHART_COUNT",
    "KAT_HAPG_ATTEMPTED_COUNT",
    "KAT_HAPG_CERTIFIED_COUNT",
    "KAT_HAPG_UNCERTIFIED_COUNT",
    "KAT_HAPG_RESCUE_COUNT",
    "KAT_HPG_MUTATION_TESTS",
    "KAT_HPG_MUTATIONS_REJECTED",
    "KAT_HAPG_MUTATION_TESTS",
    "KAT_HAPG_MUTATIONS_REJECTED",
)
NODE_COLUMNS = (
    "NODE_ID",
    "PARENT_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "WAVE_INDEX",
    "ACTION",
    "TERMINAL_REASON",
    "WAVE_CONTRACT_SHA256",
)
RESULT_HEADERS: tuple[tuple[str, str | None], ...] = (
    ("SCHEMA", "sounio.cs6.hapg-full-source-cover-wave-result.v1"),
    ("WAVE_INDEX", None),
    ("WAVE_CONTRACT_SHA256", None),
    ("NODE_COUNT", None),
    ("NEXT_FRONTIER_SHA256", None),
    ("DECISION_POLICY", "H_APG_ONLY_S_BIASED_BALANCED_ALL_OR_NONE_WAVE_ADMISSION"),
    ("CAP_PRECEDENCE", "TIMEOUT_THEN_AXIS_DEPTH_THEN_WAVE_LIMIT_THEN_NODE_BUDGET"),
)
RESULT_COLUMNS = (
    "WAVE_INDEX",
    "NODE_ID",
    "HPG_STATUS",
    "HAPG_ATTEMPTED",
    "HAPG_STATUS",
    "HAPG_RC",
    "HAPG_CHALLENGE",
    "HAPG_RECEIPT_SHA256",
    "HAPG_STDERR_SHA256",
    "HAPG_VERIFICATION_SHA256",
    "HAPG_PHYSICAL_SHA256",
    "HAPG_PROBE_PASS",
    "AFFINE_PASS",
    "PROJECTIVE_X_PASS",
    "PROJECTIVE_Y_PASS",
    "PROJECTIVE_PLUS_PASS",
    "PROJECTIVE_MINUS_PASS",
    "HOMOGENEOUS_PASS",
    "APG_VALID",
    "APG_PASS",
    "APG_RESCUE",
    "GENERIC_CERTIFICATE_PASS",
    "DECISION",
    "TERMINAL_REASON",
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
HAPG_VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "WAVE_CONTRACT_SHA256",
    "HPG_RECEIPT_SHA256",
    "HPG_VERIFICATION_SHA256",
    "LEAF_CHALLENGE",
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
    "APG_COMPUTATION_VALID",
    "APG_CERTIFICATE_PASS",
    "APG_RESCUE",
    "GENERIC_CERTIFICATE_PASS",
    "HAPG_TERMINAL_CERTIFIED",
    "HAPG_SUBDIVISION_REQUIRED",
)
CHART_MARKERS = (
    "HOMOGENEOUS_EVENT1_RAY0",
    "HOMOGENEOUS_EVENT1_RAY1",
    "HOMOGENEOUS_EVENT2_RAY0",
    "HOMOGENEOUS_EVENT2_RAY1",
)
EXECUTION_DIRECTORIES = (
    "inputs",
    "hpg-receipts",
    "hpg-stderr",
    "hpg-verifications",
    "hapg-receipts",
    "hapg-stderr",
    "hapg-verifications",
    "wave-contracts",
    "wave-results",
)


def die(message: str) -> None:
    raise SystemExit(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def bool_text(value: bool) -> str:
    return str(value).lower()


def canonical_kv(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    path.write_bytes("".join(f"{key}={value}\n" for key, value in fields).encode("ascii"))


def create_execution_directories(root: Path) -> None:
    for directory in EXECUTION_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=False)


def parse_kv_output(raw: bytes, keys: Sequence[str], label: str) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RuntimeError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError(f"{label} is not canonical LF-terminated ASCII")
    lines = text.splitlines()
    if len(lines) != len(keys):
        raise RuntimeError(f"{label} line count mismatch")
    result: dict[str, str] = {}
    for line, expected in zip(lines, keys, strict=True):
        if line.count("=") != 1:
            raise RuntimeError(f"malformed {label} line")
        key, value = line.split("=", 1)
        if key != expected or not value:
            raise RuntimeError(f"{label} key mismatch: {expected}")
        result[key] = value
    return result


def parse_bool(token: str, label: str) -> bool:
    if token == "true":
        return True
    if token == "false":
        return False
    raise RuntimeError(f"noncanonical boolean: {label}")


def load_module(name: str, path: Path) -> ModuleType:
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__source_sha256__ = digest_bytes(raw)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


VERIFY = load_module(
    "cs6_hapg_cover_verify_adapter",
    Path(__file__).resolve().with_name("cs6_hapg_full_source_cover_verify.py"),
)
KAT_ANCHOR = load_module(
    "cs6_hapg_cover_kat_anchor",
    Path(__file__).resolve().with_name("cs6_hapg_full_source_cover_kat_anchor.py"),
)


@dataclass(frozen=True)
class Leaf:
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    parent_id: str
    wave_index: int

    @property
    def identity(self) -> str:
        return VERIFY.canonical_leaf_id(
            self.u_depth, self.u_index, self.s_depth, self.s_index
        )

    @property
    def area(self) -> Fraction:
        return Fraction(1, 1 << (self.u_depth + self.s_depth))


@dataclass(frozen=True)
class HpgResult:
    leaf: Leaf
    status: str
    rc: int
    elapsed_ms: int
    input_sha: str
    challenge: str
    receipt_sha: str
    stderr_sha: str
    verification_sha: str
    physical_sha: str
    probe_pass: bool
    certificate_pass: bool
    chart_signs: tuple[tuple[str, int], ...]
    eligible: bool
    mutation_tests: int
    mutations_rejected: int


@dataclass(frozen=True)
class HapgResult:
    leaf: Leaf
    attempted: bool
    status: str
    rc: int
    elapsed_ms: int
    challenge: str
    receipt_sha: str
    stderr_sha: str
    verification_sha: str
    physical_sha: str
    probe_pass: bool
    affine_pass: bool
    projective_x_pass: bool
    projective_y_pass: bool
    projective_plus_pass: bool
    projective_minus_pass: bool
    homogeneous_pass: bool
    apg_valid: bool
    apg_pass: bool
    apg_rescue: bool
    generic_pass: bool
    mutation_tests: int
    mutations_rejected: int


@dataclass(frozen=True)
class Evaluation:
    leaf: Leaf
    hpg: HpgResult
    hapg: HapgResult
    decision: str
    terminal_reason: str
    wave_contract_sha: str


@dataclass(frozen=True)
class TreeNode:
    leaf: Leaf
    action: str
    terminal_reason: str
    wave_contract_sha: str


@dataclass(frozen=True)
class KatExpectation:
    chart_signs: tuple[tuple[str, int], ...]
    apg_pass: bool
    apg_rescue: bool


def leaf_input_bytes(leaf: Leaf) -> bytes:
    return (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={leaf.u_depth}\n"
        f"U_INDEX={leaf.u_index}\n"
        f"S_DEPTH={leaf.s_depth}\n"
        f"S_INDEX={leaf.s_index}\n"
    ).encode("ascii")


def frontier_bytes(leaves: Sequence[Leaf]) -> bytes:
    rows = ["NODE_ID\tPARENT_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256"]
    for leaf in sorted(leaves, key=lambda item: item.identity):
        rows.append(
            "\t".join(
                (
                    leaf.identity,
                    leaf.parent_id,
                    str(leaf.u_depth),
                    str(leaf.u_index),
                    str(leaf.s_depth),
                    str(leaf.s_index),
                    digest_bytes(leaf_input_bytes(leaf)),
                )
            )
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def split_leaf(leaf: Leaf) -> tuple[str, tuple[Leaf, Leaf]]:
    next_wave = leaf.wave_index + 1
    if leaf.s_depth <= leaf.u_depth:
        return (
            "SPLIT_S",
            (
                Leaf(
                    leaf.u_depth,
                    leaf.u_index,
                    leaf.s_depth + 1,
                    2 * leaf.s_index,
                    leaf.identity,
                    next_wave,
                ),
                Leaf(
                    leaf.u_depth,
                    leaf.u_index,
                    leaf.s_depth + 1,
                    2 * leaf.s_index + 1,
                    leaf.identity,
                    next_wave,
                ),
            ),
        )
    return (
        "SPLIT_U",
        (
            Leaf(
                leaf.u_depth + 1,
                2 * leaf.u_index,
                leaf.s_depth,
                leaf.s_index,
                leaf.identity,
                next_wave,
            ),
            Leaf(
                leaf.u_depth + 1,
                2 * leaf.u_index + 1,
                leaf.s_depth,
                leaf.s_index,
                leaf.identity,
                next_wave,
            ),
        ),
    )


def classify_worker_failure(stderr: bytes, prefix: str) -> str | None:
    lowered = stderr.lower()
    if prefix == "H_PG" and b"interval error:" in lowered and (
        b"division by 0" in lowered or b"division by zero" in lowered
    ):
        return "H_PG_INTERVAL_DOMAIN"
    if prefix == "H_PG" and (
        b"one-step newton crossing was not available" in lowered
        or (
            lowered.startswith(
                b"probe error: poincaremap error: possible nontransversal return to the section"
            )
            and b"\ninner product of vector field and section gradient: [" in lowered
        )
    ):
        return "H_PG_CROSSING"
    if prefix == "H_PG" and (
        b"centeredtripletonset::evalaffinefunctional - empty intersection" in lowered
        and b"rq=[-nan, -nan]" in lowered
    ):
        return "H_PG_CAPD_SET"
    if prefix == "H_APG" and any(
        marker in lowered
        for marker in (
            b"frozen tm2 pivot sign was not certified",
            b"tm2 reciprocal pivot contains zero",
            b"tm2 reciprocal center contains zero",
            b"tm2 reciprocal is noncontractive",
        )
    ):
        return "H_APG_FROZEN_CHART"
    return None


def atomic_immutable_file(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    linked = False
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RuntimeError("short write while freezing artifact")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        os.unlink(temporary)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if path.read_bytes() != raw:
            raise RuntimeError("frozen artifact differs after publication")
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def dependency_paths(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        raise RuntimeError("compiler dependency file is malformed")
    return sorted({Path(item) for item in shlex.split(text.split(":", 1)[1])})


def dependency_manifest(paths: Sequence[Path], snapshots: Mapping[Path, str]) -> bytes:
    rows: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        label = snapshots.get(path.resolve(), str(path))
        rows.append(f"{digest(path)}  {label}")
    if not rows:
        raise RuntimeError("compiler emitted no hashable dependencies")
    return ("\n".join(sorted(set(rows))) + "\n").encode("ascii")


def compile_worker(
    cxx: Path,
    flags: Sequence[str],
    libraries: Sequence[str],
    source: Path,
    binary: Path,
    dependency_file: Path,
    source_sha: str,
    command_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> list[Path]:
    command = [
        str(cxx),
        "-std=c++17",
        *flags,
        "-O0",
        f'-DCS6_WORKER_SOURCE_SHA256="{source_sha}"',
        str(source),
        "-MD",
        "-MF",
        str(dependency_file),
        "-Wl,--trace",
        "-o",
        str(binary),
        *libraries,
    ]
    canonical_command = [
        f"BUNDLE/{Path(token).name}"
        if token in {str(source), str(binary), str(dependency_file)}
        else token
        for token in command
    ]
    command_path.write_text(shlex.join(canonical_command) + "\n", encoding="ascii")
    result = subprocess.run(command, capture_output=True)
    stdout_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"worker compilation failed: {source.name} rc={result.returncode}")
    traced: set[Path] = set()
    for line in (result.stdout + b"\n" + result.stderr).splitlines():
        candidate = Path(line.decode("utf-8", errors="surrogateescape").strip())
        if candidate.is_file():
            traced.add(candidate.resolve())
    if not traced:
        raise RuntimeError("linker emitted no hashable --trace inputs")
    return sorted(traced)


def canonicalize_dependency_file(path: Path, work: Path) -> None:
    temporary_prefix = str(work.resolve()) + "/"
    text = path.read_text(encoding="utf-8")
    canonical = text.replace(temporary_prefix, "BUNDLE/")
    if str(work.resolve()) in canonical:
        raise RuntimeError("compiler dependency record retained a temporary path")
    path.write_text(canonical, encoding="utf-8")


def hpg_contract_bytes(
    results: Sequence[HpgResult],
    run_contract_sha: str,
    root_challenge: str,
    previous_result_sha: str,
    frontier_sha: str,
    hpg_source_sha: str,
    hpg_verifier_sha: str,
    hapg_source_sha: str,
    hapg_kernel_sha: str,
    adapter_sha: str,
    hapg_verifier_sha: str,
) -> bytes:
    if not results:
        raise RuntimeError("cannot freeze an empty H-PG wave")
    wave_index = results[0].leaf.wave_index
    rows = ["\t".join(VERIFY.WAVE_COLUMNS)]
    for result in sorted(results, key=lambda item: item.leaf.identity):
        if result.leaf.wave_index != wave_index:
            raise RuntimeError("mixed wave indices in H-PG freeze")
        charts: list[str] = []
        for chart, sign in result.chart_signs:
            charts.extend((chart, str(sign)))
        fields = (
            str(wave_index),
            result.leaf.identity,
            result.leaf.parent_id,
            str(result.leaf.u_depth),
            str(result.leaf.u_index),
            str(result.leaf.s_depth),
            str(result.leaf.s_index),
            result.input_sha,
            result.challenge,
            result.status,
            str(result.rc),
            result.receipt_sha,
            result.stderr_sha,
            result.verification_sha,
            result.physical_sha,
            bool_text(result.probe_pass),
            bool_text(result.certificate_pass),
            *charts,
            bool_text(result.eligible),
        )
        if len(fields) != len(VERIFY.WAVE_COLUMNS):
            raise RuntimeError("H-PG wave row width mismatch")
        rows.append("\t".join(fields))
    headers = (
        ("SCHEMA", VERIFY.WAVE_SCHEMA),
        ("RUN_CONTRACT_SHA256", run_contract_sha),
        ("ROOT_CHALLENGE", root_challenge),
        ("WAVE_INDEX", str(wave_index)),
        ("PREVIOUS_WAVE_RESULT_SHA256", previous_result_sha),
        ("FRONTIER_SHA256", frontier_sha),
        ("NODE_COUNT", str(len(results))),
        ("HPG_WORKER_SOURCE_SHA256", hpg_source_sha),
        ("HPG_VERIFIER_SOURCE_SHA256", hpg_verifier_sha),
        ("HAPG_WORKER_SOURCE_SHA256", hapg_source_sha),
        ("HAPG_KERNEL_SOURCE_SHA256", hapg_kernel_sha),
        ("HAPG_VERIFIER_ADAPTER_SHA256", adapter_sha),
        ("HAPG_NUMERIC_VERIFIER_SHA256", hapg_verifier_sha),
        (
            "FREEZE_ORDER",
            "ALL_HPG_ATTEMPTS_VERIFIED_THEN_ATOMIC_WAVE_CONTRACT_THEN_ANY_HAPG",
        ),
    )
    return (
        "".join(f"{key}={value}\n" for key, value in headers)
        + "\n".join(rows)
        + "\n"
    ).encode("ascii")


def absent_hapg(leaf: Leaf, status: str) -> HapgResult:
    return HapgResult(
        leaf,
        False,
        status,
        0,
        0,
        ZERO_SHA256,
        EMPTY_SHA256,
        EMPTY_SHA256,
        ZERO_SHA256,
        ZERO_SHA256,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        0,
        0,
    )


def wave_result_bytes(
    evaluations: Sequence[Evaluation], next_frontier_sha: str
) -> bytes:
    if not evaluations:
        raise RuntimeError("cannot publish an empty wave result")
    wave_index = evaluations[0].leaf.wave_index
    contract_sha = evaluations[0].wave_contract_sha
    rows = ["\t".join(RESULT_COLUMNS)]
    for evaluation in sorted(evaluations, key=lambda item: item.leaf.identity):
        hapg = evaluation.hapg
        fields = (
            str(wave_index),
            evaluation.leaf.identity,
            evaluation.hpg.status,
            bool_text(hapg.attempted),
            hapg.status,
            str(hapg.rc),
            hapg.challenge,
            hapg.receipt_sha,
            hapg.stderr_sha,
            hapg.verification_sha,
            hapg.physical_sha,
            bool_text(hapg.probe_pass),
            bool_text(hapg.affine_pass),
            bool_text(hapg.projective_x_pass),
            bool_text(hapg.projective_y_pass),
            bool_text(hapg.projective_plus_pass),
            bool_text(hapg.projective_minus_pass),
            bool_text(hapg.homogeneous_pass),
            bool_text(hapg.apg_valid),
            bool_text(hapg.apg_pass),
            bool_text(hapg.apg_rescue),
            bool_text(hapg.generic_pass),
            evaluation.decision,
            evaluation.terminal_reason,
        )
        if len(fields) != len(RESULT_COLUMNS):
            raise RuntimeError("wave result row width mismatch")
        rows.append("\t".join(fields))
    headers = (
        ("SCHEMA", "sounio.cs6.hapg-full-source-cover-wave-result.v1"),
        ("WAVE_INDEX", str(wave_index)),
        ("WAVE_CONTRACT_SHA256", contract_sha),
        ("NODE_COUNT", str(len(evaluations))),
        ("NEXT_FRONTIER_SHA256", next_frontier_sha),
        (
            "DECISION_POLICY",
            "H_APG_ONLY_S_BIASED_BALANCED_ALL_OR_NONE_WAVE_ADMISSION",
        ),
        (
            "CAP_PRECEDENCE",
            "TIMEOUT_THEN_AXIS_DEPTH_THEN_WAVE_LIMIT_THEN_NODE_BUDGET",
        ),
    )
    return (
        "".join(f"{key}={value}\n" for key, value in headers)
        + "\n".join(rows)
        + "\n"
    ).encode("ascii")


def run_hpg_leaf(
    leaf: Leaf,
    worker: Path,
    verifier: Path,
    python: Path,
    source_sha: str,
    root_challenge: str,
    previous_result_sha: str,
    frontier_sha: str,
    work: Path,
    timeout: int,
    mutations: bool,
) -> HpgResult:
    identity = leaf.identity
    input_path = work / "inputs" / f"{identity}.txt"
    receipt_path = work / "hpg-receipts" / f"{identity}.txt"
    stderr_path = work / "hpg-stderr" / f"{identity}.txt"
    verification_path = work / "hpg-verifications" / f"{identity}.txt"
    input_raw = leaf_input_bytes(leaf)
    input_path.write_bytes(input_raw)
    input_sha = digest_bytes(input_raw)
    challenge = VERIFY.hpg_leaf_challenge(
        root_challenge,
        leaf.wave_index,
        previous_result_sha,
        frontier_sha,
        identity,
        input_sha,
    )
    command = [
        str(worker),
        str(leaf.u_depth),
        str(leaf.u_index),
        str(leaf.s_depth),
        str(leaf.s_index),
        input_sha,
        challenge,
    ]
    started = time.monotonic_ns()
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        elapsed = (time.monotonic_ns() - started) // 1_000_000
        receipt_path.write_bytes(error.stdout or b"")
        stderr_path.write_bytes(error.stderr or b"")
        return HpgResult(
            leaf,
            "H_PG_TIMEOUT",
            124,
            elapsed,
            input_sha,
            challenge,
            digest(receipt_path),
            digest(stderr_path),
            ZERO_SHA256,
            ZERO_SHA256,
            False,
            False,
            (("NONE", 0),) * 4,
            False,
            0,
            0,
        )
    elapsed = (time.monotonic_ns() - started) // 1_000_000
    receipt_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    if result.returncode != 0:
        status = classify_worker_failure(result.stderr, "H_PG")
        if status is None:
            raise RuntimeError(
                f"unexpected H-PG worker failure for {identity}: rc={result.returncode}"
            )
        return HpgResult(
            leaf,
            status,
            result.returncode,
            elapsed,
            input_sha,
            challenge,
            digest(receipt_path),
            digest(stderr_path),
            ZERO_SHA256,
            ZERO_SHA256,
            False,
            False,
            (("NONE", 0),) * 4,
            False,
            0,
            0,
        )
    if result.stderr:
        raise RuntimeError(f"H-PG worker emitted stderr for {identity}")
    verify_command = [
        str(python),
        "-B",
        str(verifier),
        str(receipt_path),
        "--source-sha",
        source_sha,
        "--input",
        str(input_path),
        "--challenge",
        challenge,
    ]
    if mutations:
        verify_command.append("--self-test-mutations")
    verification = subprocess.run(verify_command, capture_output=True, timeout=timeout)
    verification_path.write_bytes(verification.stdout)
    if verification.returncode != 0 or verification.stderr:
        (work / "hpg-stderr" / f"{identity}.verifier.txt").write_bytes(
            verification.stderr
        )
        raise RuntimeError(f"H-PG verification failed for {identity}")
    values = parse_kv_output(
        verification.stdout, HPG_VERIFICATION_KEYS, "H-PG verification"
    )
    if (
        values["VERIFICATION_SCHEMA"]
        != "sounio.cs6.plucker-cocycle-leaf-verification.v1"
        or values["RECEIPT_SHA256"] != digest(receipt_path)
    ):
        raise RuntimeError(f"H-PG verification binding mismatch for {identity}")
    probe = parse_bool(values["PROBE_PASS"], "HPG PROBE_PASS")
    certificate = parse_bool(values["CERTIFICATE_PASS"], "HPG CERTIFICATE_PASS")
    mutation_tests = int(values["MUTATION_TESTS"])
    mutations_rejected = int(values["MUTATIONS_REJECTED"])
    if mutations and (mutation_tests == 0 or mutation_tests != mutations_rejected):
        raise RuntimeError(f"H-PG mutation audit failed for {identity}")
    ledger = VERIFY.HPG_CORE.parse_ledger(receipt_path)
    chart_signs: list[tuple[str, int]] = []
    signed = True
    for marker in CHART_MARKERS:
        record = ledger.records[marker]
        chart = VERIFY.HPG_CORE.string_value(record, "CHART")
        pivot = VERIFY.HPG_CORE.interval(record, "PIVOT")
        sign = -1 if pivot.upper < 0 else 1 if pivot.lower > 0 else 0
        if chart not in VERIFY.HAPG_CORE.FULL53_CHARTS or sign == 0:
            signed = False
        chart_signs.append((chart, sign))
    eligible = probe and signed
    if not eligible:
        chart_signs = [("NONE", 0)] * 4
    return HpgResult(
        leaf,
        VERIFY.SIGNED_CHART_STATUS if eligible else "H_PG_INVALID_NO_SIGNED_CHART",
        0,
        elapsed,
        input_sha,
        challenge,
        digest(receipt_path),
        digest(stderr_path),
        digest(verification_path),
        values["PHYSICAL_SHA256"],
        probe,
        certificate,
        tuple(chart_signs),
        eligible,
        mutation_tests,
        mutations_rejected,
    )


def run_hapg_leaf(
    hpg: HpgResult,
    worker: Path,
    adapter: Path,
    python: Path,
    hpg_source_sha: str,
    hapg_source_sha: str,
    root_challenge: str,
    wave_contract: Path,
    work: Path,
    timeout: int,
    mutations: bool,
) -> HapgResult:
    leaf = hpg.leaf
    if not hpg.eligible or leaf.identity == "U00-0000000000_S00-0000000000":
        return absent_hapg(leaf, "H_APG_NOT_ELIGIBLE")
    wave = VERIFY.parse_wave_contract(wave_contract)
    contract = VERIFY.HAPG_CORE.Full53LeafContract(
        leaf_id=leaf.identity,
        u_depth=leaf.u_depth,
        u_index=leaf.u_index,
        s_depth=leaf.s_depth,
        s_index=leaf.s_index,
        parent_input_sha256=hpg.input_sha,
        parent_status=hpg.status,
        parent_receipt_sha256=hpg.receipt_sha,
        chart_signs=hpg.chart_signs,
        manifest_sha256=wave.sha256,
    )
    challenge = VERIFY.HAPG_CORE.full53_leaf_challenge(root_challenge, contract)
    identity = leaf.identity
    receipt_path = work / "hapg-receipts" / f"{identity}.txt"
    stderr_path = work / "hapg-stderr" / f"{identity}.txt"
    verification_path = work / "hapg-verifications" / f"{identity}.txt"
    chart_args: list[str] = []
    for chart, sign in hpg.chart_signs:
        chart_args.extend((chart, str(sign)))
    command = [
        str(worker),
        str(leaf.u_depth),
        str(leaf.u_index),
        str(leaf.s_depth),
        str(leaf.s_index),
        hpg.input_sha,
        hpg.receipt_sha,
        *chart_args,
        wave.sha256,
        challenge,
    ]
    started = time.monotonic_ns()
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        elapsed = (time.monotonic_ns() - started) // 1_000_000
        receipt_path.write_bytes(error.stdout or b"")
        stderr_path.write_bytes(error.stderr or b"")
        return HapgResult(
            leaf,
            True,
            "H_APG_TIMEOUT",
            124,
            elapsed,
            challenge,
            digest(receipt_path),
            digest(stderr_path),
            ZERO_SHA256,
            ZERO_SHA256,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            0,
            0,
        )
    elapsed = (time.monotonic_ns() - started) // 1_000_000
    receipt_path.write_bytes(result.stdout)
    stderr_path.write_bytes(result.stderr)
    if result.returncode != 0:
        status = classify_worker_failure(result.stderr, "H_APG")
        if status is None:
            raise RuntimeError(
                f"unexpected H-APG worker failure for {identity}: rc={result.returncode}"
            )
        missing = absent_hapg(leaf, status)
        return HapgResult(
            leaf,
            True,
            status,
            result.returncode,
            elapsed,
            challenge,
            digest(receipt_path),
            digest(stderr_path),
            missing.verification_sha,
            missing.physical_sha,
            missing.probe_pass,
            missing.affine_pass,
            missing.projective_x_pass,
            missing.projective_y_pass,
            missing.projective_plus_pass,
            missing.projective_minus_pass,
            missing.homogeneous_pass,
            missing.apg_valid,
            missing.apg_pass,
            missing.apg_rescue,
            missing.generic_pass,
            0,
            0,
        )
    if result.stderr:
        raise RuntimeError(f"H-APG worker emitted stderr for {identity}")
    input_path = work / "inputs" / f"{identity}.txt"
    hpg_receipt = work / "hpg-receipts" / f"{identity}.txt"
    hpg_verification = work / "hpg-verifications" / f"{identity}.txt"
    command = [
        str(python),
        "-B",
        str(adapter),
        str(receipt_path),
        "--hapg-source-sha",
        hapg_source_sha,
        "--hpg-source-sha",
        hpg_source_sha,
        "--input",
        str(input_path),
        "--wave-contract",
        str(wave_contract),
        "--hpg-receipt",
        str(hpg_receipt),
        "--hpg-verification",
        str(hpg_verification),
        "--root-challenge",
        root_challenge,
    ]
    if mutations:
        command.append("--self-test-mutations")
    verification = subprocess.run(command, capture_output=True, timeout=timeout)
    verification_path.write_bytes(verification.stdout)
    if verification.returncode != 0 or verification.stderr:
        (work / "hapg-stderr" / f"{identity}.verifier.txt").write_bytes(
            verification.stderr
        )
        raise RuntimeError(f"H-APG verification failed for {identity}")
    values = parse_kv_output(
        verification.stdout, HAPG_VERIFICATION_KEYS, "H-APG verification"
    )
    if (
        values["VERIFICATION_SCHEMA"]
        != "sounio.cs6.hapg-full-source-cover-leaf-verification.v1"
        or values["RECEIPT_SHA256"] != digest(receipt_path)
        or values["WAVE_CONTRACT_SHA256"] != wave.sha256
        or values["HPG_RECEIPT_SHA256"] != hpg.receipt_sha
    ):
        raise RuntimeError(f"H-APG verification binding mismatch for {identity}")
    mutation_tests = int(values["MUTATION_TESTS"])
    mutations_rejected = int(values["MUTATIONS_REJECTED"])
    if mutations and (mutation_tests == 0 or mutation_tests != mutations_rejected):
        raise RuntimeError(f"H-APG mutation audit failed for {identity}")
    apg_valid = parse_bool(values["APG_COMPUTATION_VALID"], "APG valid")
    apg_pass = parse_bool(values["APG_CERTIFICATE_PASS"], "APG pass")
    terminal = parse_bool(values["HAPG_TERMINAL_CERTIFIED"], "HAPG terminal")
    if terminal != (apg_valid and apg_pass):
        raise RuntimeError(f"H-APG terminal predicate mismatch for {identity}")
    return HapgResult(
        leaf,
        True,
        "H_APG_CERTIFIED"
        if terminal
        else "H_APG_UNCERTIFIED"
        if apg_valid
        else "H_APG_INVALID",
        0,
        elapsed,
        challenge,
        digest(receipt_path),
        digest(stderr_path),
        digest(verification_path),
        values["PHYSICAL_SHA256"],
        parse_bool(values["PROBE_PASS"], "HAPG PROBE_PASS"),
        parse_bool(values["AFFINE_CERTIFICATE_PASS"], "affine pass"),
        parse_bool(values["PROJECTIVE_X_CERTIFICATE_PASS"], "projective X"),
        parse_bool(values["PROJECTIVE_Y_CERTIFICATE_PASS"], "projective Y"),
        parse_bool(values["PROJECTIVE_PLUS_CERTIFICATE_PASS"], "projective plus"),
        parse_bool(values["PROJECTIVE_MINUS_CERTIFICATE_PASS"], "projective minus"),
        parse_bool(values["HOMOGENEOUS_CERTIFICATE_PASS"], "homogeneous pass"),
        apg_valid,
        apg_pass,
        parse_bool(values["APG_RESCUE"], "APG rescue"),
        parse_bool(values["GENERIC_CERTIFICATE_PASS"], "generic certificate"),
        mutation_tests,
        mutations_rejected,
    )


def parse_contract(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RuntimeError("frozen contract must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError("frozen contract must be canonical LF-terminated ASCII")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            raise RuntimeError("malformed frozen contract")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise RuntimeError("duplicate or empty frozen contract field")
        result[key] = value
    return result


def validate_kat_prerequisite_certificate(
    fields: Mapping[str, str], contract: Mapping[str, str]
) -> None:
    expected = {
        "SCHEMA": KAT_CERTIFICATE_SCHEMA,
        "CERTIFICATE_SCOPE": "AUTHORITATIVE_V6_ADAPTIVE_PREREQUISITE",
        "KAT_SCHEMA_PROFILE": "v6",
        "KAT_PREREQUISITE_VALID": "true",
        "KAT_ROOT_CHALLENGE": contract.get("KAT_ROOT_CHALLENGE"),
        "KAT_COORDINATE_MANIFEST_SHA256": contract.get(
            "KAT_COORDINATE_MANIFEST_SHA256"
        ),
        "KAT_EXPECTED_RESULTS_SHA256": contract.get("KAT_EXPECTED_RESULTS_SHA256"),
        "KAT_LEAF_EVIDENCE_VALID": "true",
        "KAT_HPG_VERIFIER_REPLAY_COUNT": "52",
        "KAT_HAPG_VERIFIER_REPLAY_COUNT": "52",
        "KAT_EVALUATED_NODE_COUNT": contract.get("KAT_EXPECTED_ATTEMPTED"),
        "KAT_HPG_SIGNED_CHART_COUNT": contract.get("KAT_EXPECTED_H_PG_VALID"),
        "KAT_HAPG_ATTEMPTED_COUNT": contract.get("KAT_EXPECTED_H_APG_VALID"),
        "KAT_HAPG_CERTIFIED_COUNT": contract.get("KAT_EXPECTED_H_APG_CERTIFIED"),
        "KAT_HAPG_UNCERTIFIED_COUNT": contract.get(
            "KAT_EXPECTED_H_APG_UNCERTIFIED"
        ),
        "KAT_HAPG_RESCUE_COUNT": contract.get("KAT_EXPECTED_H_APG_RESCUES"),
        "KAT_HPG_MUTATION_TESTS": contract.get("KAT_EXPECTED_HPG_MUTATION_TESTS"),
        "KAT_HPG_MUTATIONS_REJECTED": contract.get(
            "KAT_EXPECTED_HPG_MUTATIONS_REJECTED"
        ),
        "KAT_HAPG_MUTATION_TESTS": contract.get(
            "KAT_EXPECTED_HAPG_MUTATION_TESTS"
        ),
        "KAT_HAPG_MUTATIONS_REJECTED": contract.get(
            "KAT_EXPECTED_HAPG_MUTATIONS_REJECTED"
        ),
        "KAT_ANCHOR_SOURCE_SHA256": contract.get("KAT_ANCHOR_SHA256"),
        "KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT": "true",
    }
    if (
        contract.get("KAT_PREREQUISITE_CERTIFICATE_SCHEMA")
        != KAT_CERTIFICATE_SCHEMA
        or any(fields.get(key) != value for key, value in expected.items())
        or any(
            SHA_RE.fullmatch(fields.get(key, "")) is None
            or fields.get(key) == ZERO_SHA256
            for key in ("KAT_WAVE_CONTRACT_SHA256", "KAT_WAVE_RESULT_SHA256")
        )
    ):
        raise RuntimeError(
            "KAT prerequisite lacks the frozen v6 leaf-evidence bindings"
        )


def verify_slurm_allocation(contract: Mapping[str, str]) -> tuple[str, bytes]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    node = os.environ.get("SLURMD_NODENAME", "")
    expected_environment = {
        "SLURM_JOB_NUM_NODES": contract["BOUNDED_PILOT_SLURM_NODES"],
        "SLURM_NTASKS": contract["BOUNDED_PILOT_SLURM_TASKS"],
        "SLURM_CPUS_PER_TASK": contract["BOUNDED_PILOT_SLURM_CPUS_PER_TASK"],
    }
    if (
        not job_id.isdigit()
        or not node
        or any(os.environ.get(key) != value for key, value in expected_environment.items())
    ):
        raise RuntimeError("Slurm allocation environment differs from the frozen contract")
    scontrol = shutil.which("scontrol")
    if scontrol is None:
        raise RuntimeError("scontrol is required for authoritative scientific execution")
    result = subprocess.run(
        [scontrol, "-o", "show", "job", job_id], capture_output=True, check=True
    )
    raw = result.stdout
    try:
        text = raw.decode("ascii").strip()
    except UnicodeError as error:
        raise RuntimeError("Slurm control-plane record must be ASCII") from error
    fields = {
        key: value
        for token in shlex.split(text)
        if "=" in token
        for key, value in (token.split("=", 1),)
    }
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", fields.get("UserId", ""))
    expected_fields = {
        "JobId": job_id,
        "JobState": "RUNNING",
        "Partition": contract["BOUNDED_PILOT_SLURM_PARTITION"],
        "Account": contract["BOUNDED_PILOT_SLURM_ACCOUNT"],
        "QOS": contract["BOUNDED_PILOT_SLURM_QOS"],
        "NodeList": contract["BOUNDED_PILOT_SLURM_NODE"],
        "NumNodes": contract["BOUNDED_PILOT_SLURM_NODES"],
        "NumTasks": contract["BOUNDED_PILOT_SLURM_TASKS"],
        "NumCPUs": contract["BOUNDED_PILOT_SLURM_ALLOCATED_CPUS"],
        "CPUs/Task": contract["BOUNDED_PILOT_SLURM_CPUS_PER_TASK"],
    }
    if (
        any(fields.get(key) != value for key, value in expected_fields.items())
        or node != expected_fields["NodeList"]
        or user is None
        or int(user.group(1)) != os.getuid()
    ):
        raise RuntimeError("Slurm control-plane record differs from the frozen allocation")
    canonical = (text + "\n").encode("ascii")
    return node, canonical


def parse_kat_population(
    coordinate_manifest: Path, expected_results: Path
) -> tuple[list[Leaf], dict[str, KatExpectation]]:
    coordinate_lines = coordinate_manifest.read_text(encoding="ascii").splitlines()
    header = (
        "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256\t"
        "PARENT_STATUS\tPARENT_RECEIPT_SHA256\tE1_R0_CHART\tE1_R0_SIGN\t"
        "E1_R1_CHART\tE1_R1_SIGN\tE2_R0_CHART\tE2_R0_SIGN\tE2_R1_CHART\t"
        "E2_R1_SIGN"
    )
    try:
        header_index = coordinate_lines.index(header)
    except ValueError as error:
        raise RuntimeError("KAT coordinate manifest header mismatch") from error
    rows = coordinate_lines[header_index + 1 :]
    if len(rows) != 53:
        raise RuntimeError("KAT coordinate population is not 53 leaves")
    result_lines = expected_results.read_text(encoding="ascii").splitlines()
    result_header = result_lines[0].split("\t")
    expected_by_id = {
        values["LEAF_ID"]: values
        for values in (
            dict(zip(result_header, line.split("\t"), strict=True))
            for line in result_lines[1:]
        )
    }
    leaves: list[Leaf] = []
    expectations: dict[str, KatExpectation] = {}
    for line in rows:
        fields = line.split("\t")
        if len(fields) != 16:
            raise RuntimeError("KAT coordinate row width mismatch")
        identity = fields[0]
        coordinates = tuple(int(token) for token in fields[1:5])
        leaf = Leaf(*coordinates, "-", 0)
        if identity != leaf.identity or identity in expectations:
            raise RuntimeError("KAT coordinate identity mismatch")
        if digest_bytes(leaf_input_bytes(leaf)) != fields[5]:
            raise RuntimeError("KAT coordinate input digest mismatch")
        charts = tuple(
            (fields[index], int(fields[index + 1]))
            for index in (8, 10, 12, 14)
        )
        expected = expected_by_id.get(identity)
        if expected is None:
            raise RuntimeError("KAT leaf is absent from retained full53 results")
        expectations[identity] = KatExpectation(
            charts,
            expected["APG_PASS"] == "true",
            expected["APG_RESCUE"] == "true",
        )
        leaves.append(leaf)
    leaves.sort(key=lambda item: item.identity)
    if [leaf.identity for leaf in leaves] != sorted(expectations):
        raise RuntimeError("KAT population order or result set mismatch")
    return leaves, expectations


def evaluate_parallel(
    leaves: Sequence[Leaf], jobs: int, function: Callable[[Leaf], object]
) -> list[object]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        results = list(executor.map(function, leaves))
    return sorted(results, key=lambda item: item.leaf.identity)


def freeze_wave_contract(
    work: Path,
    results: Sequence[HpgResult],
    run_contract_sha: str,
    root_challenge: str,
    previous_result_sha: str,
    frontier_sha: str,
    hpg_source_sha: str,
    hpg_verifier_sha: str,
    hapg_source_sha: str,
    hapg_kernel_sha: str,
    adapter_sha: str,
    hapg_verifier_sha: str,
) -> tuple[Path, str]:
    raw = hpg_contract_bytes(
        results,
        run_contract_sha,
        root_challenge,
        previous_result_sha,
        frontier_sha,
        hpg_source_sha,
        hpg_verifier_sha,
        hapg_source_sha,
        hapg_kernel_sha,
        adapter_sha,
        hapg_verifier_sha,
    )
    wave_index = results[0].leaf.wave_index
    path = work / "wave-contracts" / f"W{wave_index:04d}.tsv"
    atomic_immutable_file(path, raw)
    parsed = VERIFY.parse_wave_contract(path)
    if parsed.sha256 != digest_bytes(raw):
        raise RuntimeError("wave contract changed during freeze verification")
    return path, parsed.sha256


def decide_adaptive_wave(
    hpg_results: Sequence[HpgResult],
    hapg_results: Sequence[HapgResult],
    wave_contract_sha: str,
    allocated_nodes: int,
    max_nodes: int,
    max_waves: int,
    max_u_depth: int,
    max_s_depth: int,
) -> tuple[list[Evaluation], list[Leaf], list[TreeNode]]:
    if [item.leaf for item in hpg_results] != [item.leaf for item in hapg_results]:
        raise RuntimeError("H-PG and H-APG wave populations differ")
    fixed: dict[str, tuple[str, str]] = {}
    candidates: list[tuple[HpgResult, HapgResult, str, tuple[Leaf, Leaf]]] = []
    for hpg, hapg in zip(hpg_results, hapg_results, strict=True):
        leaf = hpg.leaf
        if hapg.apg_valid and hapg.apg_pass:
            fixed[leaf.identity] = ("CERTIFIED", "H_APG")
            continue
        if hpg.status == "H_PG_TIMEOUT" or hapg.status == "H_APG_TIMEOUT":
            fixed[leaf.identity] = ("UNRESOLVED", "TIMEOUT")
            continue
        action, children = split_leaf(leaf)
        axis_limited = (
            action == "SPLIT_S" and leaf.s_depth >= max_s_depth
        ) or (action == "SPLIT_U" and leaf.u_depth >= max_u_depth)
        if axis_limited:
            fixed[leaf.identity] = ("UNRESOLVED", "AXIS_DEPTH")
            continue
        candidates.append((hpg, hapg, action, children))

    if candidates:
        wave_index = candidates[0][0].leaf.wave_index
        if wave_index + 1 >= max_waves:
            for hpg, _, _, _ in candidates:
                fixed[hpg.leaf.identity] = ("UNRESOLVED", "WAVE_LIMIT")
            candidates = []
        elif allocated_nodes + 2 * len(candidates) > max_nodes:
            for hpg, _, _, _ in candidates:
                fixed[hpg.leaf.identity] = ("UNRESOLVED", "NODE_BUDGET")
            candidates = []

    split_by_id = {
        hpg.leaf.identity: (action, children)
        for hpg, _, action, children in candidates
    }
    evaluations: list[Evaluation] = []
    nodes: list[TreeNode] = []
    next_frontier: list[Leaf] = []
    hapg_by_id = {item.leaf.identity: item for item in hapg_results}
    for hpg in hpg_results:
        identity = hpg.leaf.identity
        if identity in split_by_id:
            action, children = split_by_id[identity]
            reason = "-"
            next_frontier.extend(children)
        else:
            action, reason = fixed[identity]
        evaluation = Evaluation(
            hpg.leaf,
            hpg,
            hapg_by_id[identity],
            action,
            reason,
            wave_contract_sha,
        )
        evaluations.append(evaluation)
        nodes.append(TreeNode(hpg.leaf, action, reason, wave_contract_sha))
    next_frontier.sort(key=lambda item: item.identity)
    if len({leaf.identity for leaf in next_frontier}) != len(next_frontier):
        raise RuntimeError("adaptive wave generated duplicate children")
    return evaluations, next_frontier, nodes


def run_one_wave(
    leaves: Sequence[Leaf],
    jobs: int,
    work: Path,
    hpg_worker: Path,
    hpg_verifier: Path,
    hapg_worker: Path,
    adapter: Path,
    python: Path,
    hpg_source_sha: str,
    hpg_verifier_sha: str,
    hapg_source_sha: str,
    hapg_kernel_sha: str,
    adapter_sha: str,
    hapg_verifier_sha: str,
    run_contract_sha: str,
    root_challenge: str,
    previous_result_sha: str,
    timeout: int,
    mutations: bool,
) -> tuple[list[HpgResult], list[HapgResult], Path, str, str]:
    ordered = sorted(leaves, key=lambda item: item.identity)
    frontier_raw = frontier_bytes(ordered)
    frontier_sha = digest_bytes(frontier_raw)
    hpg_results = evaluate_parallel(
        ordered,
        jobs,
        lambda leaf: run_hpg_leaf(
            leaf,
            hpg_worker,
            hpg_verifier,
            python,
            hpg_source_sha,
            root_challenge,
            previous_result_sha,
            frontier_sha,
            work,
            timeout,
            mutations,
        ),
    )
    hpg_results = [item for item in hpg_results if isinstance(item, HpgResult)]
    if len(hpg_results) != len(ordered):
        raise RuntimeError("H-PG evaluator lost a frontier node")
    contract_path, contract_sha = freeze_wave_contract(
        work,
        hpg_results,
        run_contract_sha,
        root_challenge,
        previous_result_sha,
        frontier_sha,
        hpg_source_sha,
        hpg_verifier_sha,
        hapg_source_sha,
        hapg_kernel_sha,
        adapter_sha,
        hapg_verifier_sha,
    )
    # Rehash the immutable freeze immediately before any H-APG subprocess starts.
    if digest(contract_path) != contract_sha:
        raise RuntimeError("wave contract drifted before H-APG execution")
    hapg_results = evaluate_parallel(
        ordered,
        jobs,
        lambda leaf: run_hapg_leaf(
            next(item for item in hpg_results if item.leaf.identity == leaf.identity),
            hapg_worker,
            adapter,
            python,
            hpg_source_sha,
            hapg_source_sha,
            root_challenge,
            contract_path,
            work,
            timeout,
            mutations,
        ),
    )
    hapg_results = [item for item in hapg_results if isinstance(item, HapgResult)]
    if digest(contract_path) != contract_sha:
        raise RuntimeError("wave contract drifted during H-APG execution")
    return hpg_results, hapg_results, contract_path, contract_sha, frontier_sha


def publish_wave_result(
    work: Path, evaluations: Sequence[Evaluation], next_frontier: Sequence[Leaf]
) -> tuple[Path, str, str]:
    next_sha = digest_bytes(frontier_bytes(next_frontier))
    raw = wave_result_bytes(evaluations, next_sha)
    wave_index = evaluations[0].leaf.wave_index
    path = work / "wave-results" / f"W{wave_index:04d}.tsv"
    atomic_immutable_file(path, raw)
    return path, digest_bytes(raw), next_sha


def run_kat(
    leaves: Sequence[Leaf],
    expectations: Mapping[str, KatExpectation],
    **wave_args: object,
) -> tuple[list[Evaluation], list[tuple[int, str, str, str, str]], bool]:
    hpg_results, hapg_results, _, contract_sha, frontier_sha = run_one_wave(
        leaves, **wave_args
    )
    hapg_by_id = {item.leaf.identity: item for item in hapg_results}
    evaluations = [
        Evaluation(
            hpg.leaf,
            hpg,
            hapg_by_id[hpg.leaf.identity],
            "KAT_ONLY",
            "-",
            contract_sha,
        )
        for hpg in hpg_results
    ]
    _, result_sha, next_sha = publish_wave_result(
        wave_args["work"], evaluations, []  # type: ignore[arg-type]
    )
    root = "U00-0000000000_S00-0000000000"
    if len(hpg_results) != 53 or len(hapg_results) != 53:
        raise RuntimeError("KAT attempt count mismatch")
    if next(item for item in hpg_results if item.leaf.identity == root).status != "H_PG_INTERVAL_DOMAIN":
        raise RuntimeError("KAT root failure class mismatch")
    signed = [item for item in hpg_results if item.eligible]
    if len(signed) != 52:
        raise RuntimeError("KAT signed H-PG count mismatch")
    for hpg in hpg_results:
        expected = expectations[hpg.leaf.identity]
        if hpg.leaf.identity == root:
            if hpg.chart_signs != (("NONE", 0),) * 4:
                raise RuntimeError("KAT root chart sentinel mismatch")
        elif hpg.chart_signs != expected.chart_signs:
            raise RuntimeError(f"KAT chart tuple mismatch: {hpg.leaf.identity}")
        hapg = hapg_by_id[hpg.leaf.identity]
        if hpg.leaf.identity != root and (
            hapg.apg_pass != expected.apg_pass
            or hapg.apg_rescue != expected.apg_rescue
            or not hapg.apg_valid
        ):
            raise RuntimeError(f"KAT H-APG outcome mismatch: {hpg.leaf.identity}")
    paired = [item for item in hapg_results if item.attempted]
    if (
        len(paired) != 52
        or sum(item.apg_pass for item in paired) != 48
        or sum(not item.apg_pass for item in paired) != 4
        or sum(item.apg_rescue for item in paired) != 20
    ):
        raise RuntimeError("KAT aggregate H-APG counts mismatch")
    if any(
        item.mutation_tests == 0 or item.mutation_tests != item.mutations_rejected
        for item in hpg_results
        if item.verification_sha != ZERO_SHA256
    ) or any(
        item.mutation_tests == 0 or item.mutation_tests != item.mutations_rejected
        for item in paired
    ):
        raise RuntimeError("KAT mutation audit mismatch")
    wave_rows = [(0, frontier_sha, contract_sha, result_sha, next_sha)]
    return evaluations, wave_rows, True


def run_adaptive(
    max_nodes: int,
    max_waves: int,
    max_u_depth: int,
    max_s_depth: int,
    **wave_args: object,
) -> tuple[list[Evaluation], dict[str, TreeNode], list[tuple[int, str, str, str, str]], bool]:
    frontier = [Leaf(0, 0, 0, 0, "-", 0)]
    previous_result_sha = ZERO_SHA256
    allocated = 1
    evaluations: list[Evaluation] = []
    nodes: dict[str, TreeNode] = {}
    wave_rows: list[tuple[int, str, str, str, str]] = []
    for wave_index in range(max_waves):
        if not frontier:
            break
        if any(
            leaf.wave_index != wave_index
            or leaf.u_depth + leaf.s_depth != wave_index
            for leaf in frontier
        ):
            raise RuntimeError("adaptive frontier violates BFS depth invariant")
        local_args = dict(wave_args)
        local_args["previous_result_sha"] = previous_result_sha
        hpg_results, hapg_results, _, contract_sha, frontier_sha = run_one_wave(
            frontier, **local_args
        )
        wave_evaluations, next_frontier, wave_nodes = decide_adaptive_wave(
            hpg_results,
            hapg_results,
            contract_sha,
            allocated,
            max_nodes,
            max_waves,
            max_u_depth,
            max_s_depth,
        )
        for node in wave_nodes:
            if node.leaf.identity in nodes:
                raise RuntimeError("adaptive tree repeated a node")
            nodes[node.leaf.identity] = node
        evaluations.extend(wave_evaluations)
        result_path, result_sha, next_sha = publish_wave_result(
            wave_args["work"], wave_evaluations, next_frontier  # type: ignore[arg-type]
        )
        if digest(result_path) != result_sha:
            raise RuntimeError("wave result changed after publication")
        wave_rows.append((wave_index, frontier_sha, contract_sha, result_sha, next_sha))
        allocated += len(next_frontier)
        if allocated > max_nodes:
            raise RuntimeError("adaptive tree exceeded the node budget")
        previous_result_sha = result_sha
        frontier = next_frontier
    if frontier:
        raise RuntimeError("adaptive run ended with an unpublished frontier")
    if len(nodes) != allocated or len(evaluations) != len(nodes):
        raise RuntimeError("adaptive tree accounting mismatch")
    full_candidate = all(node.action == "CERTIFIED" for node in nodes.values() if node.action in {"CERTIFIED", "UNRESOLVED"})
    return evaluations, nodes, wave_rows, full_candidate


def run_fresh_replays(
    original_evaluations: Sequence[Evaluation],
    original_run_contract_sha: str,
    original_root_challenge: str,
    replay_root_challenge: str,
    **wave_args: object,
) -> tuple[list[Evaluation], list[tuple[int, str, str, str, str]], bool]:
    if replay_root_challenge == original_root_challenge:
        raise RuntimeError("fresh replay root challenge must be distinct")
    certified = sorted(
        (item for item in original_evaluations if item.decision == "CERTIFIED"),
        key=lambda item: (item.leaf.wave_index, item.leaf.identity),
    )
    work = wave_args["work"]
    if not isinstance(work, Path):
        raise RuntimeError("fresh replay work root is not a path")
    replay_work = work / "fresh-replay"
    replay_work.mkdir()
    create_execution_directories(replay_work)
    terminal_raw = (
        "NODE_ID\tWAVE_INDEX\tORIGINAL_HPG_RECEIPT_SHA256\t"
        "ORIGINAL_HAPG_RECEIPT_SHA256\n"
        + "".join(
            f"{item.leaf.identity}\t{item.leaf.wave_index}\t"
            f"{item.hpg.receipt_sha}\t{item.hapg.receipt_sha}\n"
            for item in certified
        )
    ).encode("ascii")
    (work / "fresh-replay-terminals.tsv").write_bytes(terminal_raw)
    replay_contract_fields = (
        ("SCHEMA", "sounio.cs6.hapg-full-source-cover-fresh-replay-contract.v1"),
        ("PARENT_RUN_CONTRACT_SHA256", original_run_contract_sha),
        ("ORIGINAL_ROOT_CHALLENGE", original_root_challenge),
        ("FRESH_REPLAY_ROOT_CHALLENGE", replay_root_challenge),
        ("CERTIFIED_TERMINAL_COUNT", str(len(certified))),
        ("CERTIFIED_TERMINALS_SHA256", digest_bytes(terminal_raw)),
        ("POLICY", "EVERY_CERTIFIED_TERMINAL_FRESH_HPG_FREEZE_HAPG"),
        ("PROMOTION_ELIGIBLE", "false"),
    )
    replay_contract_path = work / "fresh-replay-contract.txt"
    canonical_kv(replay_contract_path, replay_contract_fields)
    replay_contract_sha = digest(replay_contract_path)

    groups: list[list[Evaluation]] = []
    for item in certified:
        if not groups or groups[-1][0].leaf.wave_index != item.leaf.wave_index:
            groups.append([])
        groups[-1].append(item)
    previous_result_sha = ZERO_SHA256
    replay_evaluations: list[Evaluation] = []
    wave_rows: list[tuple[int, str, str, str, str]] = []
    for group_index, group in enumerate(groups):
        leaves = [item.leaf for item in group]
        local_args = dict(wave_args)
        local_args.update(
            {
                "work": replay_work,
                "root_challenge": replay_root_challenge,
                "previous_result_sha": previous_result_sha,
                "run_contract_sha": replay_contract_sha,
                "mutations": False,
            }
        )
        hpg_results, hapg_results, _, contract_sha, frontier_sha = run_one_wave(
            leaves, **local_args
        )
        original_by_id = {item.leaf.identity: item for item in group}
        hapg_by_id = {item.leaf.identity: item for item in hapg_results}
        group_evaluations: list[Evaluation] = []
        for hpg in hpg_results:
            original = original_by_id[hpg.leaf.identity]
            hapg = hapg_by_id[hpg.leaf.identity]
            if (
                not hpg.eligible
                or hpg.chart_signs != original.hpg.chart_signs
                or not hapg.attempted
                or not hapg.apg_valid
                or not hapg.apg_pass
            ):
                raise RuntimeError(
                    f"fresh H-APG replay failed for {hpg.leaf.identity}"
                )
            group_evaluations.append(
                Evaluation(
                    hpg.leaf,
                    hpg,
                    hapg,
                    "REPLAY_CERTIFIED",
                    "-",
                    contract_sha,
                )
            )
        next_group = (
            [item.leaf for item in groups[group_index + 1]]
            if group_index + 1 < len(groups)
            else []
        )
        _, result_sha, next_sha = publish_wave_result(
            replay_work, group_evaluations, next_group
        )
        wave_rows.append(
            (group[0].leaf.wave_index, frontier_sha, contract_sha, result_sha, next_sha)
        )
        replay_evaluations.extend(group_evaluations)
        previous_result_sha = result_sha
    write_global_ledgers(replay_work, replay_evaluations, wave_rows)
    complete = len(replay_evaluations) == len(certified)
    return replay_evaluations, wave_rows, complete


def nodes_bytes(nodes: Mapping[str, TreeNode]) -> bytes:
    rows = ["\t".join(NODE_COLUMNS)]
    for identity in sorted(nodes):
        node = nodes[identity]
        leaf = node.leaf
        rows.append(
            "\t".join(
                (
                    identity,
                    leaf.parent_id,
                    str(leaf.u_depth),
                    str(leaf.u_index),
                    str(leaf.s_depth),
                    str(leaf.s_index),
                    str(leaf.wave_index),
                    node.action,
                    node.terminal_reason,
                    node.wave_contract_sha,
                )
            )
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def write_global_ledgers(
    work: Path,
    evaluations: Sequence[Evaluation],
    wave_rows: Sequence[tuple[int, str, str, str, str]],
) -> None:
    evaluation_columns = (
        "WAVE_INDEX",
        "NODE_ID",
        "PARENT_ID",
        "U_DEPTH",
        "U_INDEX",
        "S_DEPTH",
        "S_INDEX",
        "WAVE_CONTRACT_SHA256",
        "HPG_STATUS",
        "HPG_RECEIPT_SHA256",
        "HPG_VERIFICATION_SHA256",
        "HAPG_STATUS",
        "HAPG_RECEIPT_SHA256",
        "HAPG_VERIFICATION_SHA256",
        "APG_VALID",
        "APG_PASS",
        "APG_RESCUE",
        "GENERIC_CERTIFICATE_PASS",
        "DECISION",
        "TERMINAL_REASON",
    )
    rows = ["\t".join(evaluation_columns)]
    timing_rows = ["WAVE_INDEX\tNODE_ID\tHPG_ELAPSED_MS\tHAPG_ELAPSED_MS"]
    negative_rows = ["WAVE_INDEX\tNODE_ID\tHPG_STATUS\tHAPG_STATUS\tDECISION\tTERMINAL_REASON"]
    for evaluation in sorted(
        evaluations, key=lambda item: (item.leaf.wave_index, item.leaf.identity)
    ):
        leaf = evaluation.leaf
        rows.append(
            "\t".join(
                (
                    str(leaf.wave_index),
                    leaf.identity,
                    leaf.parent_id,
                    str(leaf.u_depth),
                    str(leaf.u_index),
                    str(leaf.s_depth),
                    str(leaf.s_index),
                    evaluation.wave_contract_sha,
                    evaluation.hpg.status,
                    evaluation.hpg.receipt_sha,
                    evaluation.hpg.verification_sha,
                    evaluation.hapg.status,
                    evaluation.hapg.receipt_sha,
                    evaluation.hapg.verification_sha,
                    bool_text(evaluation.hapg.apg_valid),
                    bool_text(evaluation.hapg.apg_pass),
                    bool_text(evaluation.hapg.apg_rescue),
                    bool_text(evaluation.hapg.generic_pass),
                    evaluation.decision,
                    evaluation.terminal_reason,
                )
            )
        )
        timing_rows.append(
            f"{leaf.wave_index}\t{leaf.identity}\t{evaluation.hpg.elapsed_ms}\t{evaluation.hapg.elapsed_ms}"
        )
        if not evaluation.hapg.apg_pass:
            negative_rows.append(
                "\t".join(
                    (
                        str(leaf.wave_index),
                        leaf.identity,
                        evaluation.hpg.status,
                        evaluation.hapg.status,
                        evaluation.decision,
                        evaluation.terminal_reason,
                    )
                )
            )
    (work / "evaluations.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")
    (work / "timings.tsv").write_text("\n".join(timing_rows) + "\n", encoding="ascii")
    (work / "negative-outcomes.tsv").write_text(
        "\n".join(negative_rows) + "\n", encoding="ascii"
    )
    wave_lines = [
        "WAVE_INDEX\tFRONTIER_SHA256\tWAVE_CONTRACT_SHA256\tWAVE_RESULT_SHA256\tNEXT_FRONTIER_SHA256"
    ]
    wave_lines.extend("\t".join(map(str, row)) for row in wave_rows)
    (work / "waves.tsv").write_text("\n".join(wave_lines) + "\n", encoding="ascii")


def file_index(root: Path) -> bytes:
    rows: list[str] = []
    for path in sorted(
        root.rglob("*"),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if path.is_symlink():
            raise RuntimeError(f"symlink is forbidden in run bundle: {path}")
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            raise RuntimeError(f"Python bytecode is forbidden in run bundle: {path}")
        if not path.is_file() or path in {
            root / "files.sha256",
            root / "run-manifest.txt",
        }:
            continue
        rows.append(f"{digest(path)}  {path.relative_to(root).as_posix()}")
    return ("\n".join(rows) + "\n").encode("ascii")


def verify_prebuilt_bundle(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("prebuilt bundle must be a regular directory")
    index_path = root / "files.sha256"
    manifest_path = root / "run-manifest.txt"
    if any(path.is_symlink() or not path.is_file() for path in (index_path, manifest_path)):
        raise RuntimeError("prebuilt bundle envelope is missing or unsafe")
    raw = index_path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError("prebuilt file index is noncanonical")
    indexed: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        if line.count("  ") != 1:
            raise RuntimeError("malformed prebuilt file index")
        sha256, token = line.split("  ", 1)
        pure = PurePosixPath(token)
        if (
            SHA_RE.fullmatch(sha256) is None
            or pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or token != pure.as_posix()
            or token in indexed
        ):
            raise RuntimeError("unsafe or duplicate prebuilt file-index row")
        indexed[token] = sha256
    if list(indexed) != sorted(indexed):
        raise RuntimeError("prebuilt file index is not sorted")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("symlink is forbidden in prebuilt bundle")
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            raise RuntimeError("Python bytecode is forbidden in prebuilt bundle")
        if not path.is_file() or path in {index_path, manifest_path}:
            continue
        token = path.relative_to(root).as_posix()
        actual[token] = digest(path)
    if indexed != actual:
        raise RuntimeError("prebuilt file index differs from the exact file set")
    manifest = parse_contract(manifest_path)
    expected_keys = {
        "SCHEMA",
        "RUN_COMPLETE",
        "MODE",
        "CAPD_VERSION",
        "INTERVAL_BACKEND",
        "OPTIMIZATION_LEVEL",
        "FROZEN_CONTRACT_SHA256",
        "HPG_WORKER_SOURCE_SHA256",
        "HPG_VERIFIER_SOURCE_SHA256",
        "HAPG_WORKER_SOURCE_SHA256",
        "HAPG_KERNEL_SOURCE_SHA256",
        "HAPG_VERIFIER_ADAPTER_SHA256",
        "HAPG_NUMERIC_VERIFIER_SHA256",
        "RUNNER_SHA256",
        "AGGREGATOR_SHA256",
        "KAT_ANCHOR_SHA256",
        "EXACT_TREE_KERNEL_SHA256",
        "GATE_SHA256",
        "SLURM_JOB_SCRIPT_SHA256",
        "HPG_WORKER_BINARY_SHA256",
        "HAPG_WORKER_BINARY_SHA256",
        "FILES_INDEX_SHA256",
        "FILE_COUNT",
        "PROMOTION_ELIGIBLE",
    }
    if set(manifest) != expected_keys:
        raise RuntimeError("prebuilt manifest field set mismatch")
    declared_files = {
        "FROZEN_CONTRACT_SHA256": "cs6_hapg_full_source_cover_contract_v6.txt",
        "HPG_WORKER_SOURCE_SHA256": "cs6_plucker_cocycle_probe.cpp",
        "HPG_VERIFIER_SOURCE_SHA256": "cs6_plucker_cocycle_verify.py",
        "HAPG_WORKER_SOURCE_SHA256": "cs6_hapg_full_source_cover_worker.cpp",
        "HAPG_KERNEL_SOURCE_SHA256": "cs6_affine_projective_cocycle_full53_probe.cpp",
        "HAPG_VERIFIER_ADAPTER_SHA256": "cs6_hapg_full_source_cover_verify.py",
        "HAPG_NUMERIC_VERIFIER_SHA256": "cs6_affine_projective_cocycle_full53_verify.py",
        "RUNNER_SHA256": "cs6_hapg_full_source_cover_run.py",
        "AGGREGATOR_SHA256": "cs6_hapg_full_source_cover_aggregate.py",
        "KAT_ANCHOR_SHA256": "cs6_hapg_full_source_cover_kat_anchor.py",
        "EXACT_TREE_KERNEL_SHA256": "cs6_c1_full_source_cover_aggregate.py",
        "GATE_SHA256": "cs6_hapg_full_source_cover_gate.sh",
        "SLURM_JOB_SCRIPT_SHA256": "cs6_hapg_full_source_cover_slurm_job.sh",
        "HPG_WORKER_BINARY_SHA256": "hpg-worker-binary",
        "HAPG_WORKER_BINARY_SHA256": "hapg-worker-binary",
    }
    if any(manifest[key] != indexed.get(filename) for key, filename in declared_files.items()):
        raise RuntimeError("prebuilt manifest declaration differs from indexed bytes")
    if (
        manifest["SCHEMA"] != "sounio.cs6.hapg-full-source-cover-prebuilt.v2"
        or manifest["RUN_COMPLETE"] != "true"
        or manifest["MODE"] != "prepare"
        or manifest["CAPD_VERSION"] != "5.3.0"
        or manifest["INTERVAL_BACKEND"] != "FILIB"
        or manifest["OPTIMIZATION_LEVEL"] != "O0"
        or manifest["FILES_INDEX_SHA256"] != digest_bytes(raw)
        or manifest["FILE_COUNT"] != str(len(indexed))
        or manifest["PROMOTION_ELIGIBLE"] != "false"
        or (root / "git-status.txt").read_bytes() != b""
        or manifest["HPG_WORKER_BINARY_SHA256"]
        != indexed.get("hpg-worker-binary")
        or manifest["HAPG_WORKER_BINARY_SHA256"]
        != indexed.get("hapg-worker-binary")
    ):
        raise RuntimeError("prebuilt manifest binding mismatch")
    return manifest


def publish_directory_noreplace(work: Path, destination: Path) -> None:
    parent = destination.parent.resolve(strict=True)
    if work.parent.resolve() != parent or not destination.name:
        raise RuntimeError("run publication requires a sibling temporary directory")
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise RuntimeError("exclusive run publication requires renameat2")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        os.fsync(parent_fd)
        result = renameat2(
            parent_fd,
            os.fsencode(work.name),
            parent_fd,
            os.fsencode(destination.name),
            1,
        )
        if result != 0:
            number = ctypes.get_errno()
            if number == errno.EEXIST:
                raise RuntimeError("run destination already exists or is a symlink")
            raise RuntimeError(f"exclusive run publication failed: errno={number}")
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prepare", "kat", "adaptive"), required=True)
    parser.add_argument("--capd-config", type=Path)
    parser.add_argument("--prebuilt-dir", type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root-challenge")
    parser.add_argument("--replay-root-challenge")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-waves", type=int)
    parser.add_argument("--max-u-depth", type=int)
    parser.add_argument("--max-s-depth", type=int)
    parser.add_argument("--coordinate-manifest", type=Path)
    parser.add_argument("--expected-results", type=Path)
    parser.add_argument("--kat-archive", type=Path)
    parser.add_argument("--kat-archive-sha256")
    parser.add_argument("--kat-job-id")
    parser.add_argument("--transport-repo-delta-sha256")
    parser.add_argument("--transport-prebuilt-archive-sha256")
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--enforce-frozen-contract", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "adaptive" and not args.enforce_frozen_contract:
        die("v6 adaptive mode requires the frozen Slurm contract")
    if args.mode == "prepare":
        if args.root_challenge is not None:
            die("root challenge is forbidden in prepare mode")
        if args.replay_root_challenge is not None:
            die("replay root challenge is forbidden in prepare mode")
        if args.capd_config is None or args.prebuilt_dir is not None:
            die("prepare mode requires capd-config and forbids prebuilt-dir")
        if args.self_test_mutations:
            die("receipt mutation tests are forbidden in prepare mode")
        if any(
            value is not None
            for value in (
                args.kat_archive,
                args.kat_archive_sha256,
                args.kat_job_id,
                args.transport_repo_delta_sha256,
                args.transport_prebuilt_archive_sha256,
            )
        ):
            die("KAT prerequisite arguments are forbidden in prepare mode")
    else:
        if args.root_challenge is None or SHA_RE.fullmatch(args.root_challenge) is None:
            die("root challenge must be lowercase SHA-256")
        if (args.capd_config is None) == (args.prebuilt_dir is None):
            die("scientific modes require exactly one of capd-config or prebuilt-dir")
        if args.mode == "adaptive":
            if (
                args.replay_root_challenge is None
                or SHA_RE.fullmatch(args.replay_root_challenge) is None
                or args.replay_root_challenge == args.root_challenge
            ):
                die("adaptive mode requires a distinct replay root challenge")
            if (
                args.kat_archive is None
                or args.kat_archive_sha256 is None
                or SHA_RE.fullmatch(args.kat_archive_sha256) is None
                or args.kat_job_id is None
                or not args.kat_job_id.isdigit()
                or args.transport_repo_delta_sha256 is None
                or SHA_RE.fullmatch(args.transport_repo_delta_sha256) is None
                or args.transport_prebuilt_archive_sha256 is None
                or SHA_RE.fullmatch(args.transport_prebuilt_archive_sha256) is None
            ):
                die("adaptive mode requires a transport-bound completed KAT archive")
        elif args.replay_root_challenge is not None:
            die("replay root challenge is forbidden in KAT mode")
        elif any(
            value is not None
            for value in (
                args.kat_archive,
                args.kat_archive_sha256,
                args.kat_job_id,
                args.transport_repo_delta_sha256,
                args.transport_prebuilt_archive_sha256,
            )
        ):
            die("KAT prerequisite arguments are forbidden outside adaptive mode")
    if not 1 <= args.jobs <= 64:
        die("jobs must be in [1,64]")
    if not 1 <= args.timeout_seconds <= 3600:
        die("timeout must be in [1,3600]")
    if args.mode == "adaptive":
        if None in (
            args.max_nodes,
            args.max_waves,
            args.max_u_depth,
            args.max_s_depth,
        ):
            die("adaptive mode requires all node, wave, and depth limits")
        assert args.max_nodes is not None
        assert args.max_waves is not None
        assert args.max_u_depth is not None
        assert args.max_s_depth is not None
        if (
            args.max_nodes < 1
            or args.max_waves < 1
            or not 0 <= args.max_u_depth <= 30
            or not 0 <= args.max_s_depth <= 30
        ):
            die("invalid adaptive limits")
    elif args.mode == "kat" and any(
        value is not None
        for value in (
            args.max_nodes,
            args.max_waves,
            args.max_u_depth,
            args.max_s_depth,
        )
    ):
        die("adaptive limits are forbidden in KAT mode")
    elif args.mode == "prepare" and any(
        value is not None
        for value in (
            args.max_nodes,
            args.max_waves,
            args.max_u_depth,
            args.max_s_depth,
            args.coordinate_manifest,
            args.expected_results,
        )
    ):
        die("scientific population arguments are forbidden in prepare mode")

    repo = Path(__file__).resolve().parents[2]
    research = repo / "scripts/research"
    frozen_contract = research / "cs6_hapg_full_source_cover_contract_v6.txt"
    frozen_v5_contract = research / "cs6_hapg_full_source_cover_contract_v5.txt"
    frozen_v4_contract = research / "cs6_hapg_full_source_cover_contract_v4.txt"
    frozen_v3_contract = research / "cs6_hapg_full_source_cover_contract_v3.txt"
    hpg_source = research / "cs6_plucker_cocycle_probe.cpp"
    hpg_verifier = research / "cs6_plucker_cocycle_verify.py"
    hapg_wrapper = research / "cs6_hapg_full_source_cover_worker.cpp"
    hapg_kernel = research / "cs6_affine_projective_cocycle_full53_probe.cpp"
    hapg_core = research / "cs6_affine_projective_cocycle_full53_verify.py"
    adapter = research / "cs6_hapg_full_source_cover_verify.py"
    runner = Path(__file__).resolve()
    aggregator = research / "cs6_hapg_full_source_cover_aggregate.py"
    kat_anchor = research / "cs6_hapg_full_source_cover_kat_anchor.py"
    exact_tree_kernel = research / "cs6_c1_full_source_cover_aggregate.py"
    gate = repo / "scripts/ci/cs6_hapg_full_source_cover_gate.sh"
    slurm_job_script = research / "cs6_hapg_full_source_cover_slurm_job.sh"
    v2_abort = research / "receipts/cs6_hapg_full_source_cover_v2_abort_8451_v1"
    v2_abort_manifest = v2_abort / "manifest.txt"
    v2_abort_sacct = v2_abort / "sacct.txt"
    v2_abort_config = v2_abort / "config.txt"
    v2_abort_stderr = v2_abort / "stderr.txt"
    v3_abort = research / "receipts/cs6_hapg_full_source_cover_v3_abort_8453_v1"
    v3_abort_manifest = v3_abort / "manifest.txt"
    v3_abort_sacct = v3_abort / "sacct.txt"
    v3_abort_config = v3_abort / "config.txt"
    v3_abort_slurm_stderr = v3_abort / "slurm-stderr.txt"
    v3_abort_repro_s0_stdout = v3_abort / "repro-s0-stdout.txt"
    v3_abort_repro_s0_stderr = v3_abort / "repro-s0-stderr.txt"
    v3_abort_repro_s1_stdout = v3_abort / "repro-s1-stdout.txt"
    v3_abort_repro_s1_stderr = v3_abort / "repro-s1-stderr.txt"
    v3_abort_census = v3_abort / "hpg-full255-census.tsv"
    v3_abort_census_summary = v3_abort / "hpg-full255-census-summary.txt"
    v3_abort_stderr_jsonl = v3_abort / "hpg-full255-stderr.jsonl"
    v3_abort_challenge_spotcheck = v3_abort / "challenge-spotcheck.json"
    v4_abort = research / "receipts/cs6_hapg_full_source_cover_v4_abort_8455_v1"
    v4_abort_manifest = v4_abort / "manifest.txt"
    v4_abort_files = v4_abort / "files.sha256"
    v4_abort_sacct = v4_abort / "sacct.txt"
    v4_abort_config = v4_abort / "config.txt"
    v4_abort_slurm_stdout = v4_abort / "slurm-stdout.txt"
    v4_abort_corpus = v4_abort / "hpg-rc0-corpus.tar"
    v4_abort_corpus_files = v4_abort / "corpus-files.sha256"
    v4_abort_census = v4_abort / "hpg-rc0-verifier-census.tsv"
    v4_abort_census_summary = v4_abort / "hpg-rc0-verifier-census-summary.txt"
    v4_abort_kat_compat = v4_abort / "hpg-v5-kat-compat.tsv"
    v4_abort_kat_corpus = v4_abort / "hpg-v4-kat-corpus.tar"
    v4_abort_kat_corpus_files = v4_abort / "hpg-v4-kat-corpus-files.sha256"
    v4_abort_midpoint_test = v4_abort / "midpoint-discrete-negative-test.txt"
    v4_abort_local_repro = v4_abort / "local-repro.tar"
    v4_abort_v4_verifier = v4_abort / "v4-hpg-verifier.py"
    v5_abort = research / "receipts/cs6_hapg_full_source_cover_v5_abort_8463_v1"
    v5_abort_manifest = v5_abort / "manifest.txt"
    v5_abort_files = v5_abort / "files.sha256"
    v5_abort_sacct = v5_abort / "jobs-8458-8463.sacct.psv"
    required = [
        frozen_contract,
        frozen_v5_contract,
        frozen_v4_contract,
        frozen_v3_contract,
        hpg_source,
        hpg_verifier,
        hapg_wrapper,
        hapg_kernel,
        hapg_core,
        adapter,
        runner,
        aggregator,
        kat_anchor,
        exact_tree_kernel,
        gate,
        slurm_job_script,
        v2_abort_manifest,
        v2_abort_sacct,
        v2_abort_config,
        v2_abort_stderr,
        v3_abort_manifest,
        v3_abort_sacct,
        v3_abort_config,
        v3_abort_slurm_stderr,
        v3_abort_repro_s0_stdout,
        v3_abort_repro_s0_stderr,
        v3_abort_repro_s1_stdout,
        v3_abort_repro_s1_stderr,
        v3_abort_census,
        v3_abort_census_summary,
        v3_abort_stderr_jsonl,
        v3_abort_challenge_spotcheck,
        v4_abort_manifest,
        v4_abort_files,
        v4_abort_sacct,
        v4_abort_config,
        v4_abort_slurm_stdout,
        v4_abort_corpus,
        v4_abort_corpus_files,
        v4_abort_census,
        v4_abort_census_summary,
        v4_abort_kat_compat,
        v4_abort_kat_corpus,
        v4_abort_kat_corpus_files,
        v4_abort_midpoint_test,
        v4_abort_local_repro,
        v4_abort_v4_verifier,
        v5_abort_manifest,
        v5_abort_files,
        v5_abort_sacct,
    ]
    for path in required:
        if not path.is_file():
            die(f"missing runner input: {path}")
    contract = parse_contract(frozen_contract)
    if (
        args.enforce_frozen_contract
        and args.mode != "prepare"
        and contract.get("KAT_PREREQUISITE_CERTIFICATE_SCHEMA")
        != KAT_CERTIFICATE_SCHEMA
    ):
        die("frozen v6 contract does not require the KAT certificate v2 schema")
    if args.enforce_frozen_contract and args.mode != "prepare":
        if args.mode == "kat":
            expected = {
                "root": contract["KAT_ROOT_CHALLENGE"],
                "jobs": contract["BOUNDED_PILOT_JOBS"],
                "timeout": contract["BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS"],
            }
        else:
            expected = {
                "root": contract["BOUNDED_PILOT_ROOT_CHALLENGE"],
                "jobs": contract["BOUNDED_PILOT_JOBS"],
                "timeout": contract["BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS"],
                "max_nodes": contract["BOUNDED_PILOT_MAX_NODES"],
                "max_waves": contract["BOUNDED_PILOT_MAX_WAVES"],
                "max_u": contract["BOUNDED_PILOT_MAX_U_DEPTH"],
                "max_s": contract["BOUNDED_PILOT_MAX_S_DEPTH"],
                "replay_root": contract["BOUNDED_PILOT_REPLAY_ROOT_CHALLENGE"],
            }
        actual = {
            "root": args.root_challenge,
            "jobs": str(args.jobs),
            "timeout": str(args.timeout_seconds),
        }
        if args.mode == "adaptive":
            actual.update(
                {
                    "max_nodes": str(args.max_nodes),
                    "max_waves": str(args.max_waves),
                    "max_u": str(args.max_u_depth),
                    "max_s": str(args.max_s_depth),
                    "replay_root": str(args.replay_root_challenge),
                }
            )
        if actual != expected or not args.self_test_mutations:
            die("arguments differ from the frozen authoritative contract")

    coordinates = args.coordinate_manifest or (
        research / "cs6_affine_projective_cocycle_full53_coordinates_v1.tsv"
    )
    expected_results = args.expected_results or (
        research
        / "receipts/cs6_affine_projective_cocycle_full53_retained_53_v1/leaves.tsv"
    )
    if args.mode == "kat":
        for path in (coordinates, expected_results):
            if not path.is_file():
                die(f"missing KAT input: {path}")

    capd_config = args.capd_config.resolve() if args.capd_config is not None else None
    if capd_config is not None and (
        not capd_config.is_file() or not os.access(capd_config, os.X_OK)
    ):
        die("capd-config is not executable")
    prebuilt_dir = args.prebuilt_dir.resolve() if args.prebuilt_dir is not None else None
    cxx_found = shutil.which(args.cxx) if capd_config is not None else None
    if capd_config is not None and cxx_found is None:
        die(f"C++ compiler not found: {args.cxx}")
    cxx = Path(cxx_found).resolve() if cxx_found is not None else None
    if not sys.executable or not os.path.isabs(sys.executable):
        die("Python executable identity is missing or not absolute")
    python = Path(sys.executable).resolve(strict=True)
    if not python.is_file() or not os.access(python, os.X_OK):
        die("Python executable identity is not an executable regular file")
    run_dir = args.run_dir.resolve()
    if re.fullmatch(r"/[A-Za-z0-9._/-]+", str(run_dir)) is None:
        die("run directory contains a character unsafe for canonical build records")
    if run_dir.exists() or run_dir.is_symlink():
        die("run directory already exists")
    if (
        args.enforce_frozen_contract
        and args.mode in {"kat", "adaptive"}
        and not run_dir.is_relative_to(Path("/tmp").resolve())
    ):
        die("authoritative scientific work must publish first on node-local /tmp")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.", dir=run_dir.parent.resolve())
    )
    complete = False
    try:
        for directory in (
            "inputs",
            "hpg-receipts",
            "hpg-stderr",
            "hpg-verifications",
            "hapg-receipts",
            "hapg-stderr",
            "hapg-verifications",
            "wave-contracts",
            "wave-results",
        ):
            (work / directory).mkdir()
        snapshots = {
            hpg_source: work / hpg_source.name,
            hpg_verifier: work / hpg_verifier.name,
            hapg_wrapper: work / hapg_wrapper.name,
            hapg_kernel: work / hapg_kernel.name,
            hapg_core: work / hapg_core.name,
            adapter: work / adapter.name,
            runner: work / runner.name,
            aggregator: work / aggregator.name,
            kat_anchor: work / kat_anchor.name,
            exact_tree_kernel: work / exact_tree_kernel.name,
            gate: work / gate.name,
            slurm_job_script: work / slurm_job_script.name,
            frozen_contract: work / frozen_contract.name,
            frozen_v5_contract: work / "v5-executed-contract.txt",
            frozen_v4_contract: work / "v4-executed-contract.txt",
            frozen_v3_contract: work / "v3-executed-contract.txt",
            v2_abort_manifest: work / "v2-abort-manifest.txt",
            v2_abort_sacct: work / "v2-abort-sacct.txt",
            v2_abort_config: work / "v2-abort-config.txt",
            v2_abort_stderr: work / "v2-abort-stderr.txt",
            v3_abort_manifest: work / "v3-abort-manifest.txt",
            v3_abort_sacct: work / "v3-abort-sacct.txt",
            v3_abort_config: work / "v3-abort-config.txt",
            v3_abort_slurm_stderr: work / "v3-abort-slurm-stderr.txt",
            v3_abort_repro_s0_stdout: work / "v3-abort-repro-s0-stdout.txt",
            v3_abort_repro_s0_stderr: work / "v3-abort-repro-s0-stderr.txt",
            v3_abort_repro_s1_stdout: work / "v3-abort-repro-s1-stdout.txt",
            v3_abort_repro_s1_stderr: work / "v3-abort-repro-s1-stderr.txt",
            v3_abort_census: work / "v3-abort-hpg-full255-census.tsv",
            v3_abort_census_summary: work / "v3-abort-hpg-full255-census-summary.txt",
            v3_abort_stderr_jsonl: work / "v3-abort-hpg-full255-stderr.jsonl",
            v3_abort_challenge_spotcheck: work / "v3-abort-challenge-spotcheck.json",
            v4_abort_manifest: work / "v4-abort-manifest.txt",
            v4_abort_files: work / "v4-abort-files.sha256",
            v4_abort_sacct: work / "v4-abort-sacct.txt",
            v4_abort_config: work / "v4-abort-config.txt",
            v4_abort_slurm_stdout: work / "v4-abort-slurm-stdout.txt",
            v4_abort_corpus: work / "v4-abort-hpg-rc0-corpus.tar",
            v4_abort_corpus_files: work / "v4-abort-corpus-files.sha256",
            v4_abort_census: work / "v4-abort-hpg-rc0-verifier-census.tsv",
            v4_abort_census_summary: work / "v4-abort-hpg-rc0-verifier-census-summary.txt",
            v4_abort_kat_compat: work / "v4-abort-hpg-v5-kat-compat.tsv",
            v4_abort_kat_corpus: work / "v4-abort-hpg-v4-kat-corpus.tar",
            v4_abort_kat_corpus_files: work / "v4-abort-hpg-v4-kat-corpus-files.sha256",
            v4_abort_midpoint_test: work / "v4-abort-midpoint-discrete-negative-test.txt",
            v4_abort_local_repro: work / "v4-abort-local-repro.tar",
            v4_abort_v4_verifier: work / "v4-abort-v4-hpg-verifier.py",
            v5_abort_manifest: work / "v5-abort-manifest.txt",
            v5_abort_files: work / "v5-abort-files.sha256",
            v5_abort_sacct: work / "v5-abort-sacct.psv",
        }
        if args.mode == "kat":
            snapshots[coordinates.resolve()] = work / "kat-coordinates.tsv"
            snapshots[expected_results.resolve()] = work / "kat-expected-results.tsv"
        for source, target in snapshots.items():
            shutil.copy2(source, target)
            target.chmod(0o444)
        snapshot_digests = {target: digest(target) for target in snapshots.values()}

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
        git_status = subprocess.run(
            ["git", "-C", repo, "status", "--short", "--untracked-files=all"],
            check=True,
            capture_output=True,
        ).stdout
        (work / "git-status.txt").write_bytes(git_status)
        if args.enforce_frozen_contract and git_status:
            die("authoritative execution requires a clean source checkout")

        hpg_source_snapshot = snapshots[hpg_source]
        hpg_verifier_snapshot = snapshots[hpg_verifier]
        hapg_wrapper_snapshot = snapshots[hapg_wrapper]
        adapter_snapshot = snapshots[adapter]
        hpg_source_sha = digest(hpg_source_snapshot)
        hpg_verifier_sha = digest(hpg_verifier_snapshot)
        hapg_source_sha = digest(hapg_wrapper_snapshot)
        adapter_sha = digest(adapter_snapshot)
        if (
            getattr(VERIFY, "__source_sha256__", None) != adapter_sha
            or getattr(VERIFY.HPG_CORE, "__source_sha256__", None)
            != hpg_verifier_sha
            or getattr(VERIFY.HAPG_CORE, "__source_sha256__", None)
            != digest(snapshots[hapg_core])
            or getattr(KAT_ANCHOR, "__source_sha256__", None)
            != digest(snapshots[kat_anchor])
        ):
            die("in-process verifier bytes differ from frozen source snapshots")
        if args.enforce_frozen_contract:
            source_bindings = {
                "SUPERSEDES_V5_SHA256": digest(snapshots[frozen_v5_contract]),
                "SUPERSEDES_V4_SHA256": digest(snapshots[frozen_v4_contract]),
                "SUPERSEDES_V3_SHA256": digest(snapshots[frozen_v3_contract]),
                "PREPASS_WORKER_SHA256": hpg_source_sha,
                "PREPASS_VERIFIER_SHA256": hpg_verifier_sha,
                "H_APG_WRAPPER_SHA256": hapg_source_sha,
                "H_APG_KERNEL_SHA256": digest(snapshots[hapg_kernel]),
                "H_APG_ADAPTER_SHA256": adapter_sha,
                "H_APG_NUMERIC_VERIFIER_SHA256": digest(snapshots[hapg_core]),
                "RUNNER_SHA256": digest(snapshots[runner]),
                "AGGREGATOR_SHA256": digest(snapshots[aggregator]),
                "KAT_ANCHOR_SHA256": digest(snapshots[kat_anchor]),
                "EXACT_TREE_KERNEL_SHA256": digest(snapshots[exact_tree_kernel]),
                "GATE_SHA256": digest(snapshots[gate]),
                "SLURM_JOB_SCRIPT_SHA256": digest(snapshots[slurm_job_script]),
                "V2_ABORT_RECEIPT_MANIFEST_SHA256": digest(snapshots[v2_abort_manifest]),
                "V2_ABORT_SACCT_SHA256": digest(snapshots[v2_abort_sacct]),
                "V2_ABORT_CONFIG_SHA256": digest(snapshots[v2_abort_config]),
                "V2_ABORT_STDERR_SHA256": digest(snapshots[v2_abort_stderr]),
                "V3_ABORT_RECEIPT_MANIFEST_SHA256": digest(snapshots[v3_abort_manifest]),
                "V3_ABORT_SACCT_SHA256": digest(snapshots[v3_abort_sacct]),
                "V3_ABORT_CONFIG_SHA256": digest(snapshots[v3_abort_config]),
                "V3_ABORT_SLURM_STDERR_SHA256": digest(snapshots[v3_abort_slurm_stderr]),
                "V3_ABORT_REPRO_S0_STDOUT_SHA256": digest(snapshots[v3_abort_repro_s0_stdout]),
                "V3_ABORT_REPRO_S0_STDERR_SHA256": digest(snapshots[v3_abort_repro_s0_stderr]),
                "V3_ABORT_REPRO_S1_STDOUT_SHA256": digest(snapshots[v3_abort_repro_s1_stdout]),
                "V3_ABORT_REPRO_S1_STDERR_SHA256": digest(snapshots[v3_abort_repro_s1_stderr]),
                "V3_ABORT_HPG_FULL255_CENSUS_SHA256": digest(snapshots[v3_abort_census]),
                "V3_ABORT_HPG_FULL255_CENSUS_SUMMARY_SHA256": digest(snapshots[v3_abort_census_summary]),
                "V3_ABORT_HPG_FULL255_STDERR_JSONL_SHA256": digest(snapshots[v3_abort_stderr_jsonl]),
                "V3_ABORT_HPG_CHALLENGE_SPOTCHECK_SHA256": digest(snapshots[v3_abort_challenge_spotcheck]),
                "V4_ABORT_RECEIPT_MANIFEST_SHA256": digest(snapshots[v4_abort_manifest]),
                "V4_ABORT_FILES_INDEX_SHA256": digest(snapshots[v4_abort_files]),
                "V4_ABORT_SACCT_SHA256": digest(snapshots[v4_abort_sacct]),
                "V4_ABORT_CONFIG_SHA256": digest(snapshots[v4_abort_config]),
                "V4_ABORT_SLURM_STDOUT_SHA256": digest(snapshots[v4_abort_slurm_stdout]),
                "V4_ABORT_HPG_RC0_CORPUS_SHA256": digest(snapshots[v4_abort_corpus]),
                "V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256": digest(snapshots[v4_abort_corpus_files]),
                "V4_ABORT_HPG_RC0_CENSUS_SHA256": digest(snapshots[v4_abort_census]),
                "V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256": digest(snapshots[v4_abort_census_summary]),
                "V4_ABORT_HPG_V5_KAT_COMPAT_SHA256": digest(snapshots[v4_abort_kat_compat]),
                "V4_ABORT_HPG_V4_KAT_CORPUS_SHA256": digest(snapshots[v4_abort_kat_corpus]),
                "V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256": digest(snapshots[v4_abort_kat_corpus_files]),
                "V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256": digest(snapshots[v4_abort_midpoint_test]),
                "V4_ABORT_LOCAL_REPRO_SHA256": digest(snapshots[v4_abort_local_repro]),
                "V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256": digest(snapshots[v4_abort_v4_verifier]),
                "V5_ABORT_RECEIPT_MANIFEST_SHA256": digest(snapshots[v5_abort_manifest]),
                "V5_ABORT_FILES_INDEX_SHA256": digest(snapshots[v5_abort_files]),
                "V5_ABORT_JOBS_SACCT_SHA256": digest(snapshots[v5_abort_sacct]),
            }
            if (
                contract.get("SCHEMA")
                != "sounio.cs6.hapg-full-source-cover-contract.v6"
                or contract.get("CONTRACT_STATE") != "PRE_RESULT_FROZEN"
                or contract.get("FRESH_REPLAY_SEMANTICS")
                != "INDEPENDENT_RECERTIFICATION_SAME_CHARTS_DISTINCT_CHALLENGES_NOT_BITWISE_RECEIPT_REPRODUCTION"
                or any(contract.get(key) != value for key, value in source_bindings.items())
            ):
                die("scientific sources differ from the frozen v6 contract")
            if args.mode == "kat" and (
                digest(snapshots[coordinates.resolve()])
                != contract["KAT_COORDINATE_MANIFEST_SHA256"]
                or digest(snapshots[expected_results.resolve()])
                != contract["KAT_EXPECTED_RESULTS_SHA256"]
            ):
                die("KAT population or expected result bytes differ from the frozen v6 contract")
            if args.mode in {"kat", "adaptive"} and (
                prebuilt_dir is None or not os.environ.get("SLURM_JOB_ID", "").isdigit()
            ):
                die("authoritative scientific execution requires a Slurm prebuilt job")
        hpg_binary = work / "hpg-worker-binary"
        hapg_binary = work / "hapg-worker-binary"
        snapshot_labels = {
            target.resolve(): f"BUNDLE/{target.name}" for target in snapshots.values()
        }
        hpg_dep: Path | None = None
        hapg_dep: Path | None = None
        hpg_dependencies: bytes | None = None
        hapg_dependencies: bytes | None = None
        prebuilt_manifest_sha = ZERO_SHA256
        if capd_config is not None:
            assert cxx is not None

            def capd(option: str) -> str:
                result = subprocess.run(
                    [capd_config, option], check=True, capture_output=True, text=True
                )
                return result.stdout.strip()

            capd_version = capd("--modversion")
            capd_cflags = capd("--cflags")
            capd_libs = capd("--libs")
            flags = shlex.split(capd_cflags)
            libraries = shlex.split(capd_libs)
            if (
                capd_version != "5.3.0"
                or "-D__USE_FILIB__" not in flags
                or "-frounding-math" not in flags
            ):
                die("CAPD configuration differs from the frozen FILIB contract")
            (work / "capd-version.txt").write_text(
                capd_version + "\n", encoding="ascii"
            )
            (work / "capd-cflags.txt").write_text(
                capd_cflags + "\n", encoding="ascii"
            )
            (work / "capd-libs.txt").write_text(capd_libs + "\n", encoding="ascii")
            (work / "compiler-version.txt").write_bytes(
                subprocess.run([cxx, "--version"], check=True, capture_output=True).stdout
            )
            hpg_dep = work / "hpg-dependencies.d"
            hapg_dep = work / "hapg-dependencies.d"
            hpg_link_inputs = compile_worker(
                cxx,
                flags,
                libraries,
                hpg_source_snapshot,
                hpg_binary,
                hpg_dep,
                hpg_source_sha,
                work / "hpg-compile-command.txt",
                work / "hpg-compile-stdout.txt",
                work / "hpg-compile-stderr.txt",
            )
            hapg_link_inputs = compile_worker(
                cxx,
                flags,
                libraries,
                hapg_wrapper_snapshot,
                hapg_binary,
                hapg_dep,
                hapg_source_sha,
                work / "hapg-compile-command.txt",
                work / "hapg-compile-stdout.txt",
                work / "hapg-compile-stderr.txt",
            )
            hpg_dependencies = dependency_manifest(
                dependency_paths(hpg_dep), snapshot_labels
            )
            hapg_dependencies = dependency_manifest(
                dependency_paths(hapg_dep), snapshot_labels
            )
            (work / "hpg-dependencies.sha256").write_bytes(hpg_dependencies)
            (work / "hapg-dependencies.sha256").write_bytes(hapg_dependencies)
            link_rows = [
                f"{digest(path)}  {path}"
                for path in sorted(set(hpg_link_inputs + hapg_link_inputs))
            ]
            if not link_rows:
                raise RuntimeError("CAPD link flags contain no hashable libraries")
            (work / "link-inputs.sha256").write_text(
                "\n".join(link_rows) + "\n", encoding="ascii"
            )
            build_mode = "SOURCE_COMPILED_IN_PROCESS"
        else:
            assert prebuilt_dir is not None
            prebuilt_manifest = verify_prebuilt_bundle(prebuilt_dir)
            expected_prebuilt = {
                "FROZEN_CONTRACT_SHA256": digest(snapshots[frozen_contract]),
                "HPG_WORKER_SOURCE_SHA256": hpg_source_sha,
                "HPG_VERIFIER_SOURCE_SHA256": hpg_verifier_sha,
                "HAPG_WORKER_SOURCE_SHA256": hapg_source_sha,
                "HAPG_KERNEL_SOURCE_SHA256": digest(snapshots[hapg_kernel]),
                "HAPG_VERIFIER_ADAPTER_SHA256": adapter_sha,
                "HAPG_NUMERIC_VERIFIER_SHA256": digest(snapshots[hapg_core]),
                "RUNNER_SHA256": digest(snapshots[runner]),
                "AGGREGATOR_SHA256": digest(snapshots[aggregator]),
                "KAT_ANCHOR_SHA256": digest(snapshots[kat_anchor]),
                "EXACT_TREE_KERNEL_SHA256": digest(snapshots[exact_tree_kernel]),
                "GATE_SHA256": digest(snapshots[gate]),
                "SLURM_JOB_SCRIPT_SHA256": digest(snapshots[slurm_job_script]),
            }
            if any(
                prebuilt_manifest[key] != value
                for key, value in expected_prebuilt.items()
            ):
                raise RuntimeError("prebuilt sources differ from the current frozen run")
            origin = work / "prebuilt-origin"
            shutil.copytree(prebuilt_dir, origin, symlinks=False)
            staged_manifest = verify_prebuilt_bundle(origin)
            if staged_manifest != prebuilt_manifest:
                raise RuntimeError("prebuilt manifest changed during staging")
            prebuilt_manifest = staged_manifest
            shutil.copy2(origin / "hpg-worker-binary", hpg_binary)
            shutil.copy2(origin / "hapg-worker-binary", hapg_binary)
            hpg_binary.chmod(0o555)
            hapg_binary.chmod(0o555)
            if (
                digest(hpg_binary) != prebuilt_manifest["HPG_WORKER_BINARY_SHA256"]
                or digest(hapg_binary)
                != prebuilt_manifest["HAPG_WORKER_BINARY_SHA256"]
            ):
                raise RuntimeError("prebuilt worker binary changed during staging")
            if args.enforce_frozen_contract and (
                prebuilt_manifest["HPG_WORKER_BINARY_SHA256"]
                != contract["PREBUILT_HPG_BINARY_SHA256"]
                or prebuilt_manifest["HAPG_WORKER_BINARY_SHA256"]
                != contract["PREBUILT_HAPG_BINARY_SHA256"]
            ):
                die("prebuilt binary digest differs from the frozen v6 contract")
            capd_version = (origin / "capd-version.txt").read_text(
                encoding="ascii"
            ).strip()
            capd_cflags = (origin / "capd-cflags.txt").read_text(
                encoding="ascii"
            ).strip()
            capd_libs = (origin / "capd-libs.txt").read_text(
                encoding="ascii"
            ).strip()
            if (
                capd_version != "5.3.0"
                or "-D__USE_FILIB__" not in shlex.split(capd_cflags)
                or "-frounding-math" not in shlex.split(capd_cflags)
            ):
                raise RuntimeError("prebuilt CAPD contract mismatch")
            prebuilt_manifest_sha = digest(origin / "run-manifest.txt")
            build_mode = "VERIFIED_PREBUILT_BUNDLE"

        linkage_rows: list[str] = []
        runtime_paths: set[Path] = set()
        for label, binary in (("HPG", hpg_binary), ("HAPG", hapg_binary)):
            linkage = subprocess.run(
                ["ldd", binary], check=True, capture_output=True, text=True
            ).stdout
            if "=> not found" in linkage:
                raise RuntimeError(f"{label} worker has an unresolved runtime library")
            linkage_rows.append(f"[{label}]\n{linkage}")
            for line in linkage.splitlines():
                fields = line.split()
                candidate = None
                if "=>" in fields and fields.index("=>") + 1 < len(fields):
                    candidate = fields[fields.index("=>") + 1]
                elif fields and fields[0].startswith("/"):
                    candidate = fields[0]
                if candidate and candidate.startswith("/") and Path(candidate).is_file():
                    runtime_paths.add(Path(candidate))
        (work / "runtime-linkage.txt").write_text(
            "".join(linkage_rows), encoding="ascii"
        )
        if not runtime_paths:
            raise RuntimeError("workers expose no hashable runtime libraries")
        runtime_libraries = "".join(
            f"{digest(path)}  {path}\n" for path in sorted(runtime_paths)
        ).encode("ascii")
        (work / "runtime-libraries.sha256").write_bytes(runtime_libraries)
        if prebuilt_dir is not None and runtime_libraries != (
            origin / "runtime-libraries.sha256"
        ).read_bytes():
            raise RuntimeError("compute runtime libraries differ from prepared runtime")
        (work / "build-mode.txt").write_text(build_mode + "\n", encoding="ascii")

        if args.mode == "prepare":
            assert hpg_dep is not None and hapg_dep is not None
            assert hpg_dependencies is not None and hapg_dependencies is not None
            if dependency_manifest(dependency_paths(hpg_dep), snapshot_labels) != hpg_dependencies:
                raise RuntimeError("H-PG compile dependencies changed during preparation")
            if dependency_manifest(dependency_paths(hapg_dep), snapshot_labels) != hapg_dependencies:
                raise RuntimeError("H-APG compile dependencies changed during preparation")
            for snapshot, expected_sha in snapshot_digests.items():
                if digest(snapshot) != expected_sha:
                    raise RuntimeError(
                        f"frozen input changed during preparation: {snapshot.name}"
                    )
            if args.enforce_frozen_contract and (
                digest(hpg_binary) != contract["PREBUILT_HPG_BINARY_SHA256"]
                or digest(hapg_binary) != contract["PREBUILT_HAPG_BINARY_SHA256"]
            ):
                die("prepared binary digest differs from the frozen v6 contract")
            canonicalize_dependency_file(hpg_dep, work)
            canonicalize_dependency_file(hapg_dep, work)
            index = file_index(work)
            (work / "files.sha256").write_bytes(index)
            manifest_fields = (
                ("SCHEMA", "sounio.cs6.hapg-full-source-cover-prebuilt.v2"),
                ("RUN_COMPLETE", "true"),
                ("MODE", "prepare"),
                ("CAPD_VERSION", capd_version),
                ("INTERVAL_BACKEND", "FILIB"),
                ("OPTIMIZATION_LEVEL", "O0"),
                ("FROZEN_CONTRACT_SHA256", digest(snapshots[frozen_contract])),
                ("HPG_WORKER_SOURCE_SHA256", hpg_source_sha),
                ("HPG_VERIFIER_SOURCE_SHA256", hpg_verifier_sha),
                ("HAPG_WORKER_SOURCE_SHA256", hapg_source_sha),
                ("HAPG_KERNEL_SOURCE_SHA256", digest(snapshots[hapg_kernel])),
                ("HAPG_VERIFIER_ADAPTER_SHA256", adapter_sha),
                ("HAPG_NUMERIC_VERIFIER_SHA256", digest(snapshots[hapg_core])),
                ("RUNNER_SHA256", digest(snapshots[runner])),
                ("AGGREGATOR_SHA256", digest(snapshots[aggregator])),
                ("KAT_ANCHOR_SHA256", digest(snapshots[kat_anchor])),
                ("EXACT_TREE_KERNEL_SHA256", digest(snapshots[exact_tree_kernel])),
                ("GATE_SHA256", digest(snapshots[gate])),
                ("SLURM_JOB_SCRIPT_SHA256", digest(snapshots[slurm_job_script])),
                ("HPG_WORKER_BINARY_SHA256", digest(hpg_binary)),
                ("HAPG_WORKER_BINARY_SHA256", digest(hapg_binary)),
                ("FILES_INDEX_SHA256", digest_bytes(index)),
                ("FILE_COUNT", str(len(index.splitlines()))),
                ("PROMOTION_ELIGIBLE", "false"),
            )
            canonical_kv(work / "run-manifest.txt", manifest_fields)
            publish_directory_noreplace(work, run_dir)
            complete = True
            print(f"RUN_DIR={run_dir}")
            print("MODE=prepare")
            print(f"HPG_WORKER_BINARY_SHA256={digest(run_dir / 'hpg-worker-binary')}")
            print(f"HAPG_WORKER_BINARY_SHA256={digest(run_dir / 'hapg-worker-binary')}")
            print("PROMOTION_ELIGIBLE=false")
            return 0

        slurm_job_verified = False
        slurm_job_record_sha = ZERO_SHA256
        execution_node = os.uname().nodename
        kat_certificate = None
        kat_certificate_fields: dict[str, str] = {}
        adaptive_submit_utc = "NONE"
        if args.enforce_frozen_contract:
            execution_node, slurm_job_record = verify_slurm_allocation(contract)
            (work / "slurm-job-record.txt").write_bytes(slurm_job_record)
            slurm_job_record_sha = digest(work / "slurm-job-record.txt")
            slurm_job_verified = True
            if args.mode == "adaptive":
                assert args.kat_archive is not None
                assert args.kat_archive_sha256 is not None
                assert args.kat_job_id is not None
                assert args.transport_repo_delta_sha256 is not None
                assert args.transport_prebuilt_archive_sha256 is not None
                try:
                    control_text = slurm_job_record.decode("ascii").strip()
                except UnicodeError as error:
                    raise RuntimeError("Slurm job record must be ASCII") from error
                control_fields = {
                    key: value
                    for token in shlex.split(control_text)
                    if "=" in token
                    for key, value in (token.split("=", 1),)
                }
                adaptive_job_id = os.environ["SLURM_JOB_ID"]
                adaptive_submit_utc = control_fields.get("SubmitTime", "")
                if (
                    control_fields.get("JobId") != adaptive_job_id
                    or not adaptive_submit_utc
                    or adaptive_submit_utc == "Unknown"
                ):
                    raise RuntimeError("adaptive Slurm submission identity is unavailable")
                kat_sacct = KAT_ANCHOR.query_live_sacct(args.kat_job_id)
                kat_expectations = KAT_ANCHOR.KatAnchorExpectations(
                    kat_job_id=args.kat_job_id,
                    kat_archive_sha256=args.kat_archive_sha256,
                    expected_git_head=(work / "git-head.txt").read_text(
                        encoding="ascii"
                    ).strip(),
                    expected_contract_sha256=digest(snapshots[frozen_contract]),
                    expected_base_repo_bundle_sha256=contract[
                        "BASE_REPO_BUNDLE_SHA256"
                    ],
                    expected_base_git_head=contract["BASE_REPO_BUNDLE_GIT_HEAD"],
                    expected_repo_delta_bundle_sha256=args.transport_repo_delta_sha256,
                    expected_prebuilt_archive_sha256=args.transport_prebuilt_archive_sha256,
                    expected_prebuilt_run_manifest_sha256=prebuilt_manifest_sha,
                    expected_slurm_job_script_sha256=digest(
                        snapshots[slurm_job_script]
                    ),
                )
                kat_certificate = KAT_ANCHOR.certify_kat_anchor(
                    archive_path=args.kat_archive,
                    sidecar_path=Path(f"{args.kat_archive}.sha256"),
                    sacct_bytes=kat_sacct,
                    adaptive_job_id=adaptive_job_id,
                    adaptive_submit_utc=adaptive_submit_utc,
                    expectations=kat_expectations,
                )
                kat_certificate_fields = kat_certificate.as_dict()
                validate_kat_prerequisite_certificate(
                    kat_certificate_fields, contract
                )
                if (
                    kat_certificate_fields.get("KAT_PREREQUISITE_VALID") != "true"
                    or kat_certificate_fields.get("KAT_JOB_ID") != args.kat_job_id
                    or kat_certificate_fields.get("KAT_ARCHIVE_SHA256")
                    != args.kat_archive_sha256
                    or kat_certificate_fields.get("ADAPTIVE_JOB_ID")
                    != adaptive_job_id
                    or kat_certificate_fields.get("ADAPTIVE_SUBMIT_UTC")
                    != adaptive_submit_utc
                ):
                    raise RuntimeError("KAT prerequisite certificate is not authoritative")
                (work / "kat-prerequisite-sacct.txt").write_bytes(kat_sacct)
                (work / "kat-prerequisite-certificate.txt").write_bytes(
                    kat_certificate.as_bytes()
                )
        working_filesystem_policy = (
            "NODE_LOCAL_TMP_THEN_HASHED_ARCHIVE_TRANSPORT"
            if args.enforce_frozen_contract
            else "CALLER_SELECTED_NO_ARCHIVE_GUARANTEE"
        )
        run_contract_fields = [
            ("SCHEMA", "sounio.cs6.hapg-full-source-cover-run-contract.v2"),
            ("FROZEN_CONTRACT_SHA256", digest(snapshots[frozen_contract])),
            ("MODE", args.mode),
            ("SOURCE", "N0"),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("TRAVERSAL", "DETERMINISTIC_BREADTH_FIRST"),
            ("SPLIT_RULE", "S_IF_S_DEPTH_LE_U_DEPTH_ELSE_U"),
            ("TERMINAL_PREDICATE", "APG_COMPUTATION_VALID_AND_APG_CERTIFICATE_PASS"),
            ("HPG_WORKER_SOURCE_SHA256", hpg_source_sha),
            ("HPG_VERIFIER_SOURCE_SHA256", hpg_verifier_sha),
            ("HAPG_WORKER_SOURCE_SHA256", hapg_source_sha),
            ("HAPG_KERNEL_SOURCE_SHA256", digest(snapshots[hapg_kernel])),
            ("HAPG_VERIFIER_ADAPTER_SHA256", adapter_sha),
            ("HAPG_NUMERIC_VERIFIER_SHA256", digest(snapshots[hapg_core])),
            ("KAT_ANCHOR_SHA256", digest(snapshots[kat_anchor])),
            ("SLURM_JOB_SCRIPT_SHA256", digest(snapshots[slurm_job_script])),
            ("BUILD_MODE", build_mode),
            ("PREBUILT_RUN_MANIFEST_SHA256", prebuilt_manifest_sha),
            ("SLURM_JOB_ID", os.environ.get("SLURM_JOB_ID", "NONE")),
            ("EXECUTION_NODE", execution_node),
            ("SLURM_JOB_VERIFIED", bool_text(slurm_job_verified)),
            ("SLURM_JOB_RECORD_SHA256", slurm_job_record_sha),
            (
                "WORKING_FILESYSTEM_POLICY",
                working_filesystem_policy,
            ),
            ("JOBS", str(args.jobs)),
            ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
            ("MUTATION_AUDIT", bool_text(args.self_test_mutations)),
            ("LOCAL_PROCESS_ORDERED_HASH_CHAIN", "true"),
            ("EXECUTION_PROVENANCE_ATTESTED", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        ]
        if args.mode == "adaptive":
            assert kat_certificate is not None
            run_contract_fields.extend(
                (
                    ("MAX_NODES", str(args.max_nodes)),
                    ("MAX_WAVES", str(args.max_waves)),
                    ("MAX_U_DEPTH", str(args.max_u_depth)),
                    ("MAX_S_DEPTH", str(args.max_s_depth)),
                    ("ALL_OR_NONE_WAVE_ADMISSION", "true"),
                    ("FRESH_REPLAY_ROOT_CHALLENGE", str(args.replay_root_challenge)),
                    ("KAT_PREREQUISITE_REQUIRED", "true"),
                    (
                        "KAT_PREREQUISITE_CERTIFICATE_SCHEMA",
                        contract["KAT_PREREQUISITE_CERTIFICATE_SCHEMA"],
                    ),
                    (
                        "KAT_PREREQUISITE_CERTIFICATE_SHA256",
                        kat_certificate.sha256,
                    ),
                    (
                        "KAT_PREREQUISITE_SACCT_SHA256",
                        kat_certificate_fields["KAT_SACCT_SHA256"],
                    ),
                    ("KAT_JOB_ID", kat_certificate_fields["KAT_JOB_ID"]),
                    (
                        "KAT_ARCHIVE_SHA256",
                        kat_certificate_fields["KAT_ARCHIVE_SHA256"],
                    ),
                    (
                        "KAT_GIT_HEAD",
                        kat_certificate_fields["KAT_EXPECTED_GIT_HEAD"],
                    ),
                    (
                        "KAT_FROZEN_CONTRACT_SHA256",
                        kat_certificate_fields["KAT_FROZEN_CONTRACT_SHA256"],
                    ),
                    (
                        "KAT_BASE_REPO_BUNDLE_SHA256",
                        kat_certificate_fields["KAT_BASE_REPO_BUNDLE_SHA256"],
                    ),
                    (
                        "KAT_BASE_GIT_HEAD",
                        kat_certificate_fields["KAT_BASE_GIT_HEAD"],
                    ),
                    (
                        "KAT_REPO_DELTA_BUNDLE_SHA256",
                        kat_certificate_fields["KAT_REPO_DELTA_BUNDLE_SHA256"],
                    ),
                    (
                        "KAT_PREBUILT_ARCHIVE_SHA256",
                        kat_certificate_fields["KAT_PREBUILT_ARCHIVE_SHA256"],
                    ),
                    (
                        "KAT_PREBUILT_RUN_MANIFEST_SHA256",
                        kat_certificate_fields["KAT_PREBUILT_RUN_MANIFEST_SHA256"],
                    ),
                    (
                        "KAT_SLURM_JOB_SCRIPT_SHA256",
                        kat_certificate_fields["KAT_SLURM_JOB_SCRIPT_SHA256"],
                    ),
                    ("KAT_END_UTC", kat_certificate_fields["KAT_END_UTC"]),
                    *tuple(
                        (key, kat_certificate_fields[key])
                        for key in KAT_RUN_CONTRACT_EVIDENCE_KEYS
                    ),
                    ("ADAPTIVE_SUBMIT_UTC", adaptive_submit_utc),
                    ("KAT_PREREQUISITE_VALID", "true"),
                )
            )
        else:
            run_contract_fields.extend(
                (
                    ("KAT_COORDINATE_MANIFEST_SHA256", digest(snapshots[coordinates.resolve()])),
                    ("KAT_EXPECTED_RESULTS_SHA256", digest(snapshots[expected_results.resolve()])),
                )
            )
        canonical_kv(work / "run-contract.txt", run_contract_fields)
        run_contract_sha = digest(work / "run-contract.txt")

        common_wave_args = {
            "jobs": args.jobs,
            "work": work,
            "hpg_worker": hpg_binary,
            "hpg_verifier": hpg_verifier_snapshot,
            "hapg_worker": hapg_binary,
            "adapter": adapter_snapshot,
            "python": python,
            "hpg_source_sha": hpg_source_sha,
            "hpg_verifier_sha": hpg_verifier_sha,
            "hapg_source_sha": hapg_source_sha,
            "hapg_kernel_sha": digest(snapshots[hapg_kernel]),
            "adapter_sha": adapter_sha,
            "hapg_verifier_sha": digest(snapshots[hapg_core]),
            "run_contract_sha": run_contract_sha,
            "root_challenge": args.root_challenge,
            "previous_result_sha": ZERO_SHA256,
            "timeout": args.timeout_seconds,
            "mutations": args.self_test_mutations,
        }
        nodes: dict[str, TreeNode] = {}
        fresh_replay_evaluations: list[Evaluation] = []
        fresh_replay_wave_rows: list[tuple[int, str, str, str, str]] = []
        fresh_replay_complete = False
        if args.mode == "kat":
            leaves, expectations = parse_kat_population(
                snapshots[coordinates.resolve()], snapshots[expected_results.resolve()]
            )
            evaluations, wave_rows, supported = run_kat(
                leaves, expectations, **common_wave_args
            )
        else:
            evaluations, nodes, wave_rows, supported = run_adaptive(
                args.max_nodes,
                args.max_waves,
                args.max_u_depth,
                args.max_s_depth,
                **common_wave_args,
            )
            (work / "nodes.tsv").write_bytes(nodes_bytes(nodes))
            assert args.root_challenge is not None
            assert args.replay_root_challenge is not None
            (
                fresh_replay_evaluations,
                fresh_replay_wave_rows,
                fresh_replay_complete,
            ) = run_fresh_replays(
                evaluations,
                run_contract_sha,
                args.root_challenge,
                args.replay_root_challenge,
                **common_wave_args,
            )
            supported = supported and fresh_replay_complete
        write_global_ledgers(work, evaluations, wave_rows)
        hpg_mutations = sum(item.hpg.mutation_tests for item in evaluations)
        hpg_rejected = sum(item.hpg.mutations_rejected for item in evaluations)
        hapg_mutations = sum(item.hapg.mutation_tests for item in evaluations)
        hapg_rejected = sum(item.hapg.mutations_rejected for item in evaluations)
        terminal_nodes = [
            node for node in nodes.values() if node.action in {"CERTIFIED", "UNRESOLVED"}
        ]
        unresolved_nodes = [node for node in terminal_nodes if node.action == "UNRESOLVED"]
        unresolved_area = sum((node.leaf.area for node in unresolved_nodes), Fraction(0))
        summary_fields = [
            ("SCHEMA", "sounio.cs6.hapg-full-source-cover-summary.v1"),
            ("MODE", args.mode),
            ("BOUNDED_RUN_COMPLETE", "true"),
            ("INFRASTRUCTURE_VALID", "true"),
            ("EVALUATED_NODE_COUNT", str(len(evaluations))),
            ("WAVE_COUNT", str(len(wave_rows))),
            ("HPG_SIGNED_CHART_COUNT", str(sum(item.hpg.eligible for item in evaluations))),
            ("HAPG_ATTEMPTED_COUNT", str(sum(item.hapg.attempted for item in evaluations))),
            ("HAPG_CERTIFIED_COUNT", str(sum(item.hapg.apg_pass for item in evaluations))),
            ("HAPG_RESCUE_COUNT", str(sum(item.hapg.apg_rescue for item in evaluations))),
            ("HPG_MUTATION_TESTS", str(hpg_mutations)),
            ("HPG_MUTATIONS_REJECTED", str(hpg_rejected)),
            ("HAPG_MUTATION_TESTS", str(hapg_mutations)),
            ("HAPG_MUTATIONS_REJECTED", str(hapg_rejected)),
            ("FRESH_REPLAY_TERMINAL_COUNT", str(len(fresh_replay_evaluations))),
            ("FRESH_REPLAY_WAVE_COUNT", str(len(fresh_replay_wave_rows))),
            ("FRESH_REPLAY_COMPLETE", bool_text(fresh_replay_complete)),
            ("TREE_NODE_COUNT", str(len(nodes))),
            ("CERTIFIED_TERMINAL_COUNT", str(sum(node.action == "CERTIFIED" for node in terminal_nodes))),
            ("UNRESOLVED_TERMINAL_COUNT", str(len(unresolved_nodes))),
            ("UNRESOLVED_AREA_NUMERATOR", str(unresolved_area.numerator)),
            ("UNRESOLVED_AREA_DENOMINATOR", str(unresolved_area.denominator)),
            ("HAPG_FULL_SOURCE_COVER_CANDIDATE", bool_text(args.mode == "adaptive" and supported)),
            ("AGGREGATION_REQUIRED", bool_text(args.mode == "adaptive")),
            ("EXECUTION_PROVENANCE_ATTESTED", "false"),
            ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ("HYPERBOLICITY_PROVED", "false"),
            ("CHAOTIC_ATTRACTOR_PROVED", "false"),
            ("OPEN_PROBLEM_SOLVED", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        ]
        canonical_kv(work / "summary.txt", summary_fields)
        if hpg_dep is not None and hpg_dependencies is not None:
            if dependency_manifest(dependency_paths(hpg_dep), snapshot_labels) != hpg_dependencies:
                raise RuntimeError("H-PG compile dependencies changed during execution")
        if hapg_dep is not None and hapg_dependencies is not None:
            if dependency_manifest(dependency_paths(hapg_dep), snapshot_labels) != hapg_dependencies:
                raise RuntimeError("H-APG compile dependencies changed during execution")
        for snapshot, expected_sha in snapshot_digests.items():
            if digest(snapshot) != expected_sha:
                raise RuntimeError(f"frozen input changed during execution: {snapshot.name}")
        if hpg_dep is not None:
            canonicalize_dependency_file(hpg_dep, work)
        if hapg_dep is not None:
            canonicalize_dependency_file(hapg_dep, work)
        index = file_index(work)
        (work / "files.sha256").write_bytes(index)
        manifest_fields = (
            ("SCHEMA", "sounio.cs6.hapg-full-source-cover-run-manifest.v1"),
            ("RUN_COMPLETE", "true"),
            ("MODE", args.mode),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("CAPD_VERSION", capd_version),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("RUN_CONTRACT_SHA256", run_contract_sha),
            ("FILES_INDEX_SHA256", digest_bytes(index)),
            ("FILE_COUNT", str(len(index.splitlines()))),
            ("EVALUATED_NODE_COUNT", str(len(evaluations))),
            ("WAVE_COUNT", str(len(wave_rows))),
            ("LOCAL_PROCESS_ORDERED_HASH_CHAIN", "true"),
            ("EXECUTION_PROVENANCE_ATTESTED", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        )
        canonical_kv(work / "run-manifest.txt", manifest_fields)
        publish_directory_noreplace(work, run_dir)
        complete = True
    finally:
        if not complete and work.exists():
            if args.keep_failed:
                print(f"FAILED_WORK_DIR={work}", file=sys.stderr)
            else:
                shutil.rmtree(work, ignore_errors=True)

    print(f"RUN_DIR={run_dir}")
    print(f"MODE={args.mode}")
    print(f"EVALUATED_NODE_COUNT={len(evaluations)}")
    print(f"WAVE_COUNT={len(wave_rows)}")
    print(f"HAPG_FULL_SOURCE_COVER_CANDIDATE={bool_text(args.mode == 'adaptive' and supported)}")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
