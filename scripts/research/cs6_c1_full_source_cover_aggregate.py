#!/usr/bin/env python3
"""Verify an exact dyadic CS6 C1 source-cover tree and its leaf receipts."""

from __future__ import annotations

import argparse
import copy
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
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
NODE_RE = re.compile(r"^U(0|[1-9][0-9]*)-([0-9]{10})_S(0|[1-9][0-9]*)-([0-9]{10})$")
CHALLENGE_DOMAIN = b"sounio.cs6.c1-cover-leaf-challenge.v1\0"
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
ACTIONS = {"SPLIT_U", "SPLIT_S", "CERTIFIED", "UNRESOLVED"}
METHODS = {"NONE", "AFFINE", "PROJECTIVE_X", "PROJECTIVE_Y", "PROJECTIVE_PLUS", "PROJECTIVE_MINUS"}
COLUMNS = (
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
VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "LEAF_METHOD",
    "SUBDIVISION_REQUIRED",
    "CERTIFICATE_PASS",
)


class CoverError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise CoverError(message)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != before.st_size:
        fail(f"{label} changed while being snapshotted")
    return raw


def snapshot_file(source: Path, destination: Path, label: str, executable: bool = False) -> bytes:
    raw = stable_bytes(source, label)
    destination.write_bytes(raw)
    destination.chmod(0o500 if executable else 0o400)
    return raw


def node_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def challenge(root: str, identity: str, input_sha: str) -> str:
    return digest_bytes(
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(input_sha)
    )


def leaf_input_bytes(node: "Node") -> bytes:
    return (
        f"SCHEMA={INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={node.u_depth}\n"
        f"U_INDEX={node.u_index}\n"
        f"S_DEPTH={node.s_depth}\n"
        f"S_INDEX={node.s_index}\n"
    ).encode("ascii")


@dataclass(frozen=True)
class Node:
    identity: str
    parent: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    action: str
    method: str = "NONE"
    input_path: str = "-"
    input_sha: str = "-"
    leaf_challenge: str = "-"
    receipt_path: str = "-"
    receipt_sha: str = "-"
    verification_path: str = "-"
    verification_sha: str = "-"
    physical_sha: str = "-"

    @property
    def logical_u(self) -> tuple[Fraction, Fraction]:
        denominator = 1 << self.u_depth
        return Fraction(self.u_index, denominator), Fraction(self.u_index + 1, denominator)

    @property
    def logical_s(self) -> tuple[Fraction, Fraction]:
        denominator = 1 << self.s_depth
        return Fraction(self.s_index, denominator), Fraction(self.s_index + 1, denominator)

    @property
    def area(self) -> Fraction:
        return Fraction(1, 1 << (self.u_depth + self.s_depth))


def make_node(
    u_depth: int,
    u_index: int,
    s_depth: int,
    s_index: int,
    action: str,
    parent: str = "-",
) -> Node:
    return Node(
        node_id(u_depth, u_index, s_depth, s_index),
        parent,
        u_depth,
        u_index,
        s_depth,
        s_index,
        action,
    )


def parse_int(token: str, label: str) -> int:
    if INT_RE.fullmatch(token) is None:
        fail(f"noncanonical integer: {label}")
    return int(token)


def parse_tree(path: Path) -> dict[str, Node]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise CoverError("tree must be ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail("tree must be canonical LF-terminated ASCII")
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != COLUMNS:
        fail("tree header mismatch")
    nodes: dict[str, Node] = {}
    for line_number, line in enumerate(lines[1:], 2):
        fields = line.split("\t")
        if len(fields) != len(COLUMNS):
            fail(f"tree column count mismatch on line {line_number}")
        values = dict(zip(COLUMNS, fields, strict=True))
        u_depth = parse_int(values["U_DEPTH"], "U_DEPTH")
        u_index = parse_int(values["U_INDEX"], "U_INDEX")
        s_depth = parse_int(values["S_DEPTH"], "S_DEPTH")
        s_index = parse_int(values["S_INDEX"], "S_INDEX")
        if u_depth > 30 or s_depth > 30:
            fail("tree depth exceeds worker contract")
        if not (u_index < 1 << u_depth and s_index < 1 << s_depth):
            fail("tree index out of range")
        identity = values["NODE_ID"]
        if identity != node_id(u_depth, u_index, s_depth, s_index):
            fail("node identity differs from coordinates")
        if identity in nodes:
            fail(f"duplicate node: {identity}")
        if values["ACTION"] not in ACTIONS or values["METHOD"] not in METHODS:
            fail("unknown tree action or method")
        node = Node(
            identity,
            values["PARENT_ID"],
            u_depth,
            u_index,
            s_depth,
            s_index,
            values["ACTION"],
            values["METHOD"],
            values["INPUT_PATH"],
            values["INPUT_SHA256"],
            values["LEAF_CHALLENGE"],
            values["RECEIPT_PATH"],
            values["RECEIPT_SHA256"],
            values["VERIFICATION_PATH"],
            values["VERIFICATION_SHA256"],
            values["PHYSICAL_SHA256"],
        )
        nodes[identity] = node
    if not nodes:
        fail("tree contains no nodes")
    return nodes


def dependency_paths(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        fail("compiler dependency file is malformed")
    return sorted({Path(item) for item in shlex.split(text.split(":", 1)[1])})


def dependency_manifest(paths: Sequence[Path], source: Path) -> bytes:
    rows: list[str] = []
    for path in paths:
        if path == source:
            rows.append(f"{digest(path)}  BUNDLE/worker-source.cpp")
        elif path.is_file():
            rows.append(f"{digest(path)}  {path}")
    if not rows:
        fail("compiler emitted no hashable dependencies")
    return ("\n".join(sorted(set(rows))) + "\n").encode("ascii")


def file_manifest(paths: Sequence[Path]) -> bytes:
    rows = [f"{digest(path)}  {path}" for path in sorted(set(paths)) if path.is_file()]
    return (("\n".join(rows) + "\n") if rows else "").encode("ascii")


def verify_file_manifest(raw: bytes, bundled_source: Path | None = None) -> None:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise CoverError("file manifest must be ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail("file manifest is noncanonical")
    for line in text.splitlines():
        if line.count("  ") != 1:
            fail("file manifest row is malformed")
        expected, token = line.split("  ", 1)
        if SHA_RE.fullmatch(expected) is None:
            fail("file manifest digest is malformed")
        if token == "BUNDLE/worker-source.cpp":
            if bundled_source is None:
                fail("file manifest needs the bundled worker source")
            path = bundled_source
        else:
            path = Path(token)
        if not path.is_file() or digest(path) != expected:
            fail(f"file manifest input changed or disappeared: {token}")


def publish_audit_bundle(
    build_root: Path, destination: Path, replay_ledger: bytes
) -> str:
    if not destination.name or destination.name in {".", ".."}:
        fail("audit output has no directory name")
    try:
        parent = destination.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise CoverError("audit output parent does not exist") from error
    source_names = (
        "worker-source.cpp",
        "compile-command.txt",
        "dependencies.sha256",
        "link-inputs.sha256",
        "runtime-linkage.txt",
        "runtime-libraries.sha256",
    )
    payloads = {
        name: stable_bytes(build_root / name, name)
        for name in source_names
    }
    payloads["replay-ledger.tsv"] = replay_ledger
    rows = [f"{digest_bytes(payloads[name])}  {name}" for name in payloads]
    index = ("\n".join(sorted(rows)) + "\n").encode("ascii")
    manifest = (
        "SCHEMA=sounio.cs6.c1-full-source-cover-replay-audit.v1\n"
        f"FILES_INDEX_SHA256={digest_bytes(index)}\n"
        f"FILE_COUNT={len(rows)}\n"
        "REPLAY_LEDGER_RETAINED=true\n"
        "REPLAY_RECEIPTS_RETAINED=false\n"
    ).encode("ascii")
    payloads["files.sha256"] = index
    payloads["audit-manifest.txt"] = manifest

    parent_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    temporary_name = temporary.name
    directory_descriptor: int | None = None
    published = False
    renamed = False
    created_identity: tuple[int, int] | None = None

    def stable_payload(name: str) -> bytes:
        assert directory_descriptor is not None
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        try:
            before = os.fstat(descriptor)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            identity_before = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            identity_after = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            raw = b"".join(chunks)
            if identity_before != identity_after or len(raw) != before.st_size:
                fail(f"audit payload changed while being verified: {name}")
            return raw
        finally:
            os.close(descriptor)

    def rename_noreplace(source_name: str, final_name: str) -> None:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            fail("exclusive audit publication requires renameat2")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            parent_descriptor,
            os.fsencode(source_name),
            parent_descriptor,
            os.fsencode(final_name),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise CoverError("audit output already exists or is a symlink")
        raise CoverError(
            f"exclusive audit publication failed: errno={error_number}"
        )

    try:
        directory_descriptor = os.open(
            temporary_name,
            os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(directory_descriptor)
        created_identity = (opened.st_dev, opened.st_ino)
        for name, raw in payloads.items():
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_descriptor,
            )
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        fail(f"short write while publishing audit file: {name}")
                    view = view[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(directory_descriptor)
        for name, raw in payloads.items():
            if stable_payload(name) != raw:
                fail(f"audit payload changed before publication: {name}")
        os.fsync(parent_descriptor)
        rename_noreplace(temporary_name, destination.name)
        renamed = True
        os.fsync(parent_descriptor)
        visible = os.stat(
            destination.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        if (visible.st_dev, visible.st_ino) != created_identity:
            fail("audit output directory changed during publication")
        for name, raw in payloads.items():
            if stable_payload(name) != raw:
                fail(f"published audit payload differs from its commitment: {name}")
        os.fsync(parent_descriptor)
        published = True
        return digest_bytes(manifest)
    finally:
        if not published:
            if directory_descriptor is not None:
                for name in reversed(tuple(payloads)):
                    try:
                        os.unlink(name, dir_fd=directory_descriptor)
                    except FileNotFoundError:
                        pass
            try:
                visible_name = destination.name if renamed else temporary_name
                visible = os.stat(
                    visible_name, dir_fd=parent_descriptor, follow_symlinks=False
                )
                if (visible.st_dev, visible.st_ino) == created_identity:
                    os.rmdir(visible_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            os.fsync(parent_descriptor)
        if directory_descriptor is not None:
            os.close(directory_descriptor)
        os.close(parent_descriptor)


def expected_children(node: Node) -> tuple[str, str]:
    if node.action == "SPLIT_U":
        return (
            node_id(node.u_depth + 1, 2 * node.u_index, node.s_depth, node.s_index),
            node_id(node.u_depth + 1, 2 * node.u_index + 1, node.s_depth, node.s_index),
        )
    if node.action == "SPLIT_S":
        return (
            node_id(node.u_depth, node.u_index, node.s_depth + 1, 2 * node.s_index),
            node_id(node.u_depth, node.u_index, node.s_depth + 1, 2 * node.s_index + 1),
        )
    return ()


def verify_sweep(terminals: Sequence[Node]) -> None:
    boundaries = sorted({point for node in terminals for point in node.logical_u})
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != 1:
        fail("terminal U projection does not span the root")
    for left, right in zip(boundaries, boundaries[1:], strict=False):
        if left == right:
            continue
        midpoint = (left + right) / 2
        active = sorted(
            (node.logical_s for node in terminals if node.logical_u[0] <= midpoint < node.logical_u[1]),
            key=lambda interval: (interval[0], interval[1]),
        )
        cursor = Fraction(0)
        for lower, upper in active:
            if lower != cursor:
                fail("logical sweep detected a gap or overlap")
            if upper <= lower:
                fail("logical sweep detected an empty terminal")
            cursor = upper
        if cursor != 1:
            fail("logical sweep does not span the S axis")


def verify_structure(nodes: Mapping[str, Node]) -> tuple[list[Node], Fraction, Fraction]:
    root_identity = node_id(0, 0, 0, 0)
    if root_identity not in nodes or nodes[root_identity].parent != "-":
        fail("tree has no canonical root")
    if sum(node.parent == "-" for node in nodes.values()) != 1:
        fail("tree has multiple roots")

    children_by_parent: dict[str, set[str]] = {}
    for node in nodes.values():
        if node.identity == root_identity:
            continue
        if node.parent not in nodes:
            fail(f"missing parent for {node.identity}")
        children_by_parent.setdefault(node.parent, set()).add(node.identity)
    for node in nodes.values():
        actual = children_by_parent.get(node.identity, set())
        expected = set(expected_children(node))
        if actual != expected:
            fail(f"tree closure mismatch at {node.identity}")
        if node.action in {"SPLIT_U", "SPLIT_S"}:
            if node.method != "NONE" or any(
                value != "-"
                for value in (
                    node.input_path,
                    node.input_sha,
                    node.leaf_challenge,
                    node.receipt_path,
                    node.receipt_sha,
                    node.verification_path,
                    node.verification_sha,
                    node.physical_sha,
                )
            ):
                fail("internal node carries terminal evidence")

    terminals = sorted(
        (node for node in nodes.values() if node.action in {"CERTIFIED", "UNRESOLVED"}),
        key=lambda node: node.identity,
    )
    if not terminals:
        fail("tree has no terminals")
    total_area = sum((node.area for node in terminals), Fraction(0))
    if total_area != 1:
        fail("terminal areas do not sum exactly to one")
    verify_sweep(terminals)
    accepted_area = sum(
        (node.area for node in terminals if node.action == "CERTIFIED"), Fraction(0)
    )
    unresolved_area = total_area - accepted_area
    return terminals, accepted_area, unresolved_area


def safe_artifact(root: Path, token: str) -> Path:
    pure = PurePosixPath(token)
    if token == "-" or pure.is_absolute() or ".." in pure.parts or not pure.parts:
        fail(f"unsafe artifact path: {token}")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            fail(f"symlink artifact path is forbidden: {token}")
    resolved_root = root.resolve()
    resolved = current.resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        fail(f"missing or escaping artifact: {token}")
    return resolved


def parse_verification(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise CoverError("verification output must be ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail("verification output is noncanonical")
    lines = text.splitlines()
    if len(lines) != len(VERIFICATION_KEYS):
        fail("verification output line count mismatch")
    result: dict[str, str] = {}
    for line, expected in zip(lines, VERIFICATION_KEYS, strict=True):
        if line.count("=") != 1:
            fail("malformed verification output")
        key, value = line.split("=", 1)
        if key != expected or not value:
            fail(f"verification key mismatch: {expected}")
        result[key] = value
    return result


def verify_terminal_artifacts(
    node: Node,
    bundle_root: Path,
    verifier: Path,
    source_sha: str,
    root_challenge: str,
    worker: Path | None,
    replay_root: str | None,
    audit_mutations: bool,
    timeout_seconds: int,
) -> tuple[bytes | None, int, int]:
    if node.action != "CERTIFIED":
        if node.method != "NONE":
            fail("unresolved terminal declares a certificate method")
        if any(
            value != "-"
            for value in (
                node.input_path,
                node.input_sha,
                node.leaf_challenge,
                node.receipt_path,
                node.receipt_sha,
                node.verification_path,
                node.verification_sha,
                node.physical_sha,
            )
        ):
            fail("unresolved terminal carries certificate evidence")
        return None, 0, 0
    if node.method == "NONE":
        fail("certified terminal omits its method")
    for value in (
        node.input_sha,
        node.leaf_challenge,
        node.receipt_sha,
        node.verification_sha,
        node.physical_sha,
    ):
        if SHA_RE.fullmatch(value) is None:
            fail("certified terminal contains a malformed digest")
    original_input = safe_artifact(bundle_root, node.input_path)
    original_receipt = safe_artifact(bundle_root, node.receipt_path)
    original_verification = safe_artifact(bundle_root, node.verification_path)
    input_raw = stable_bytes(original_input, "terminal input")
    receipt_raw = stable_bytes(original_receipt, "terminal receipt")
    verification_raw = stable_bytes(original_verification, "terminal verification")
    if input_raw != leaf_input_bytes(node):
        fail("terminal input coordinates differ from tree node")
    if digest_bytes(input_raw) != node.input_sha or digest_bytes(receipt_raw) != node.receipt_sha:
        fail("terminal input or receipt hash mismatch")
    if digest_bytes(verification_raw) != node.verification_sha:
        fail("terminal verification hash mismatch")
    expected_challenge = challenge(root_challenge, node.identity, node.input_sha)
    if node.leaf_challenge != expected_challenge:
        fail("terminal challenge derivation mismatch")

    with tempfile.TemporaryDirectory(prefix="cs6-c1-cover-leaf-snapshot-") as directory:
        snapshot_root = Path(directory)
        input_path = snapshot_root / "input.txt"
        receipt_path = snapshot_root / "receipt.txt"
        input_path.write_bytes(input_raw)
        receipt_path.write_bytes(receipt_raw)
        verified = subprocess.run(
            [
                sys.executable,
                verifier,
                receipt_path,
                "--source-sha",
                source_sha,
                "--input",
                input_path,
                "--challenge",
                node.leaf_challenge,
                "--require-terminal",
            ],
            capture_output=True,
            timeout=timeout_seconds,
        )
    if verified.returncode != 0 or verified.stderr:
        fail(f"stored terminal verification failed: {node.identity}")
    values = parse_verification(verified.stdout)
    stored_values = parse_verification(verification_raw)
    if values != stored_values:
        fail("fresh verifier output differs from retained verification")
    if (
        values["CERTIFICATE_PASS"] != "true"
        or values["SUBDIVISION_REQUIRED"] != "false"
        or values["LEAF_METHOD"] != node.method
        or values["RECEIPT_SHA256"] != node.receipt_sha
        or values["PHYSICAL_SHA256"] != node.physical_sha
    ):
        fail("terminal verification claims mismatch")

    mutation_tests = mutation_rejected = 0
    if audit_mutations:
        with tempfile.TemporaryDirectory(prefix="cs6-c1-cover-mutation-") as directory:
            mutation_root = Path(directory)
            mutation_input = mutation_root / "input.txt"
            mutation_receipt = mutation_root / "receipt.txt"
            mutation_input.write_bytes(input_raw)
            mutation_receipt.write_bytes(receipt_raw)
            mutation = subprocess.run(
                [
                    sys.executable,
                    verifier,
                    mutation_receipt,
                    "--source-sha",
                    source_sha,
                    "--input",
                    mutation_input,
                    "--challenge",
                    node.leaf_challenge,
                    "--self-test-mutations",
                    "--require-terminal",
                ],
                capture_output=True,
                timeout=timeout_seconds,
            )
        if mutation.returncode != 0 or mutation.stderr:
            fail("leaf mutation audit failed")
        mutation_values = parse_verification(mutation.stdout)
        mutation_tests = int(mutation_values["MUTATION_TESTS"])
        mutation_rejected = int(mutation_values["MUTATIONS_REJECTED"])
        if mutation_tests == 0 or mutation_tests != mutation_rejected:
            fail("leaf mutation audit did not reject every mutation")

    if worker is None or replay_root is None:
        return None, mutation_tests, mutation_rejected
    replay_challenge = challenge(replay_root, node.identity, node.input_sha)
    with tempfile.TemporaryDirectory(prefix="cs6-c1-cover-replay-") as directory:
        replay_root_path = Path(directory)
        replay_input = replay_root_path / "input.txt"
        replay_receipt = replay_root_path / "receipt.txt"
        replay_input.write_bytes(input_raw)
        result = subprocess.run(
            [
                worker,
                str(node.u_depth),
                str(node.u_index),
                str(node.s_depth),
                str(node.s_index),
                node.input_sha,
                replay_challenge,
            ],
            capture_output=True,
            timeout=timeout_seconds,
        )
        replay_receipt.write_bytes(result.stdout)
        if result.returncode != 0 or result.stderr:
            fail(f"fresh worker replay failed: {node.identity}")
        replay = subprocess.run(
            [
                sys.executable,
                verifier,
                replay_receipt,
                "--source-sha",
                source_sha,
                "--input",
                replay_input,
                "--challenge",
                replay_challenge,
                "--require-terminal",
            ],
            capture_output=True,
            timeout=timeout_seconds,
        )
        if replay.returncode != 0 or replay.stderr:
            fail(f"fresh receipt replay verification failed: {node.identity}")
        replay_values = parse_verification(replay.stdout)
        if (
            replay_values["CERTIFICATE_PASS"] != "true"
            or replay_values["SUBDIVISION_REQUIRED"] != "false"
            or replay_values["LEAF_METHOD"] != node.method
            or replay_values["RECEIPT_SHA256"] != digest_bytes(result.stdout)
            or replay_values["PHYSICAL_SHA256"] != node.physical_sha
        ):
            fail("fresh replay certificate claims mismatch")
        replay_row = (
            "\t".join(
                (
                    node.identity,
                    replay_challenge,
                    digest_bytes(result.stdout),
                    digest_bytes(replay.stdout),
                    replay_values["PHYSICAL_SHA256"],
                    replay_values["LEAF_METHOD"],
                )
            )
            + "\n"
        ).encode("ascii")
    return replay_row, mutation_tests, mutation_rejected


def compile_canonical_worker(
    source: Path,
    source_sha: str,
    capd_config: Path,
    cxx_name: str,
    directory: Path,
    timeout_seconds: int,
) -> tuple[Path, dict[str, str]]:
    if not capd_config.is_file() or not os.access(capd_config, os.X_OK):
        fail("capd-config is not executable")
    cxx_found = shutil.which(cxx_name)
    if cxx_found is None:
        fail("C++ compiler is unavailable")
    cxx = Path(cxx_found).resolve()
    capd_config_sha = digest(capd_config)
    compiler_sha = digest(cxx)

    def capd(option: str) -> str:
        result = subprocess.run(
            [capd_config, option],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return result.stdout.strip()

    try:
        version = capd("--modversion")
        cflags = capd("--cflags")
        libraries = capd("--libs")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CoverError("capd-config failed during canonical replay build") from error
    if version != "5.3.0":
        fail("canonical replay requires CAPD 5.3.0")
    flag_tokens = shlex.split(cflags)
    if "-D__USE_FILIB__" not in flag_tokens or "-frounding-math" not in flag_tokens:
        fail("canonical replay requires FILIB and rounding-math")
    binary = directory / "canonical-worker"
    dependency_file = directory / "dependencies.d"
    library_tokens = shlex.split(libraries)
    command = [
        str(cxx),
        "-std=c++17",
        *flag_tokens,
        "-O0",
        f'-DCS6_WORKER_SOURCE_SHA256="{source_sha}"',
        str(source),
        "-MD",
        "-MF",
        str(dependency_file),
        "-o",
        str(binary),
        *library_tokens,
    ]
    normalized_command = [
        "BUNDLE/dependencies.d" if token == str(dependency_file)
        else "BUNDLE/worker-source.cpp" if token == str(source)
        else "BUNDLE/worker-binary" if token == str(binary)
        else token
        for token in command
    ]
    command_bytes = (shlex.join(normalized_command) + "\n").encode("ascii")
    try:
        compiled = subprocess.run(
            command, capture_output=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        raise CoverError("canonical worker compilation timed out") from error
    if compiled.returncode != 0:
        fail("canonical worker compilation failed")
    dependency_set = dependency_paths(dependency_file)
    dependencies = dependency_manifest(dependency_set, source)
    link_paths = [Path(token) for token in library_tokens if token.startswith("/")]
    link_inputs = file_manifest(link_paths)
    if not link_inputs:
        fail("CAPD link flags contain no hashable file inputs")
    try:
        stable_compile = subprocess.run(
            command, capture_output=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as error:
        raise CoverError("canonical stability compilation timed out") from error
    if stable_compile.returncode != 0:
        fail("canonical stability compilation failed")
    if dependency_manifest(dependency_paths(dependency_file), source) != dependencies:
        fail("compiler dependency closure changed during canonical build")
    if file_manifest(link_paths) != link_inputs:
        fail("link inputs changed during canonical build")
    if digest(capd_config) != capd_config_sha or digest(cxx) != compiler_sha:
        fail("compiler or capd-config changed during canonical build")
    try:
        linkage = subprocess.run(
            ["ldd", binary], check=True, capture_output=True, text=True,
            timeout=timeout_seconds,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CoverError("runtime linkage capture failed") from error
    runtime_paths: set[Path] = set()
    for line in linkage.stdout.splitlines():
        fields = line.split()
        candidate: str | None = None
        if "=>" in fields and fields.index("=>") + 1 < len(fields):
            candidate = fields[fields.index("=>") + 1]
        elif fields and fields[0].startswith("/"):
            candidate = fields[0]
        if candidate and candidate.startswith("/") and Path(candidate).is_file():
            runtime_paths.add(Path(candidate))
    runtime_libraries = file_manifest(sorted(runtime_paths))
    source_raw = stable_bytes(source, "canonical worker source snapshot")
    if digest_bytes(source_raw) != source_sha:
        fail("canonical worker source snapshot changed during compilation")
    build_artifacts = {
        "worker-source.cpp": source_raw,
        "compile-command.txt": command_bytes,
        "dependencies.sha256": dependencies,
        "link-inputs.sha256": link_inputs,
        "runtime-linkage.txt": linkage.stdout.encode("utf-8"),
        "runtime-libraries.sha256": runtime_libraries,
    }
    for name, raw in build_artifacts.items():
        path = directory / name
        path.write_bytes(raw)
        path.chmod(0o400)
    binary.chmod(0o500)
    return binary, {
        "WORKER_BINARY_SHA256": digest(binary),
        "COMPILE_COMMAND_SHA256": digest_bytes(command_bytes),
        "CAPD_CONFIG_SHA256": capd_config_sha,
        "COMPILER_SHA256": compiler_sha,
        "DEPENDENCIES_SHA256": digest_bytes(dependencies),
        "LINK_INPUTS_SHA256": digest_bytes(link_inputs),
        "RUNTIME_LINKAGE_SHA256": digest_bytes(linkage.stdout.encode("utf-8")),
        "RUNTIME_LIBRARIES_SHA256": digest_bytes(runtime_libraries),
        "CAPD_VERSION": version,
        "INTERVAL_BACKEND": "FILIB",
        "OPTIMIZATION_LEVEL": "O0",
    }


def self_test_mutations() -> tuple[int, int]:
    root = make_node(0, 0, 0, 0, "SPLIT_U")
    left = make_node(1, 0, 0, 0, "CERTIFIED", root.identity)
    right = make_node(1, 1, 0, 0, "SPLIT_S", root.identity)
    low = make_node(1, 1, 1, 0, "CERTIFIED", right.identity)
    high = make_node(1, 1, 1, 1, "CERTIFIED", right.identity)
    valid = {node.identity: node for node in (root, left, right, low, high)}
    verify_structure(valid)

    mutations: list[dict[str, Node]] = []
    candidate = copy.deepcopy(valid)
    candidate.pop(high.identity)
    mutations.append(candidate)
    candidate = copy.deepcopy(valid)
    candidate[low.identity] = replace(candidate[low.identity], parent=root.identity)
    mutations.append(candidate)
    candidate = copy.deepcopy(valid)
    candidate[left.identity] = replace(candidate[left.identity], action="SPLIT_S")
    mutations.append(candidate)
    candidate = copy.deepcopy(valid)
    candidate[root.identity] = replace(candidate[root.identity], parent=left.identity)
    mutations.append(candidate)
    candidate = copy.deepcopy(valid)
    extra = make_node(2, 0, 2, 0, "CERTIFIED")
    candidate[extra.identity] = extra
    mutations.append(candidate)
    candidate = copy.deepcopy(valid)
    candidate[right.identity] = replace(candidate[right.identity], action="CERTIFIED")
    mutations.append(candidate)
    candidate = copy.deepcopy(valid)
    candidate[root.identity] = replace(candidate[root.identity], action="SPLIT_S")
    mutations.append(candidate)

    rejected = 0
    for candidate in mutations:
        try:
            verify_structure(candidate)
        except CoverError:
            rejected += 1
        else:
            fail("structural mutation escaped exact verifier")

    gap_overlap = (
        make_node(2, 0, 0, 0, "CERTIFIED"),
        make_node(2, 1, 0, 0, "CERTIFIED"),
        make_node(2, 1, 0, 0, "CERTIFIED"),
        make_node(2, 3, 0, 0, "CERTIFIED"),
    )
    if sum((node.area for node in gap_overlap), Fraction(0)) != 1:
        fail("self-test construction lost equal area")
    try:
        verify_sweep(gap_overlap)
    except CoverError:
        rejected += 1
    else:
        fail("equal-area gap/overlap mutation escaped sweep")
    return len(mutations) + 1, rejected


def write_certificate(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    if not path.name or path.name in {".", ".."}:
        fail("certificate output has no filename")
    try:
        parent = path.parent.resolve(strict=True)
    except FileNotFoundError as error:
        raise CoverError("certificate output parent does not exist") from error
    if not parent.is_dir():
        fail("certificate output parent is not a directory")
    destination = parent / path.name
    if os.path.lexists(destination):
        fail("certificate output already exists or is a symlink")
    raw = "".join(f"{key}={value}\n" for key, value in fields).encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o600)
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            raise CoverError("certificate output appeared during publication") from error
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tree", nargs="?", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--root-challenge")
    parser.add_argument("--replay-root-challenge")
    parser.add_argument("--capd-config", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--require-local-full-source", action="store_true")
    parser.add_argument("--require-full-source", action="store_true")
    args = parser.parse_args(argv)

    mutation_tests, mutation_rejected = self_test_mutations()
    if args.tree is None:
        if not args.self_test_mutations:
            fail("tree is required unless only self-testing mutations")
        print(f"MUTATION_TESTS={mutation_tests}")
        print(f"MUTATIONS_REJECTED={mutation_rejected}")
        return 0
    required = {
        "tree": args.tree,
        "bundle root": args.bundle_root,
        "root challenge": args.root_challenge,
        "output": args.output,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        fail(f"missing aggregate arguments: {','.join(missing)}")
    assert args.tree is not None
    assert args.bundle_root is not None
    assert args.root_challenge is not None
    assert args.output is not None
    if SHA_RE.fullmatch(args.root_challenge) is None:
        fail("root challenge must be lowercase SHA-256")
    if not 1 <= args.timeout_seconds <= 3600:
        fail("timeout must be in [1,3600]")
    if args.replay_root_challenge is not None and SHA_RE.fullmatch(
        args.replay_root_challenge
    ) is None:
        fail("replay root challenge must be lowercase SHA-256")
    if args.replay_root_challenge == args.root_challenge:
        fail("replay root challenge must differ from the retained challenge")
    replay_arguments = (
        args.capd_config,
        args.replay_root_challenge,
        args.audit_dir,
    )
    if any(value is not None for value in replay_arguments) and not all(
        value is not None for value in replay_arguments
    ):
        fail("capd-config, replay root challenge, and audit-dir are one contract")
    if args.audit_dir is not None:
        output_identity = args.output.parent.resolve(strict=True) / args.output.name
        audit_identity = args.audit_dir.parent.resolve(strict=True) / args.audit_dir.name
        if output_identity == audit_identity:
            fail("certificate output and audit directory must differ")

    aggregate_path = Path(__file__).resolve()
    canonical_dir = aggregate_path.parent
    original_verifier = canonical_dir / "cs6_c1_full_source_cover_leaf_verify.py"
    original_source = canonical_dir / "cs6_c1_full_source_cover_probe.cpp"
    if not original_verifier.is_file() or not original_source.is_file():
        fail("canonical verifier or worker source is missing")
    snapshot_directory = tempfile.TemporaryDirectory(prefix="cs6-c1-cover-inputs-")
    snapshot_root = Path(snapshot_directory.name)
    source = snapshot_root / "worker-source.cpp"
    verifier = snapshot_root / "leaf-verifier.py"
    tree_snapshot = snapshot_root / "nodes.tsv"
    source_raw = snapshot_file(original_source, source, "canonical worker source")
    verifier_raw = snapshot_file(original_verifier, verifier, "canonical leaf verifier")
    tree_raw = snapshot_file(args.tree, tree_snapshot, "cover tree")
    source_sha = digest_bytes(source_raw)
    verifier_sha = digest_bytes(verifier_raw)
    aggregate_sha = digest_bytes(stable_bytes(aggregate_path, "canonical aggregator"))
    tree_sha = digest_bytes(tree_raw)

    nodes = parse_tree(tree_snapshot)
    terminals, accepted_area, unresolved_area = verify_structure(nodes)
    ledger_bytes = "".join(
        "\t".join(
            (
                node.identity,
                node.parent,
                node.action,
                node.method,
                node.input_sha,
                node.leaf_challenge,
                node.receipt_sha,
                node.verification_sha,
                node.physical_sha,
            )
        )
        + "\n"
        for node in sorted(nodes.values(), key=lambda item: item.identity)
    ).encode("ascii")
    bundle_ledger_sha = digest_bytes(ledger_bytes)

    compile_metadata = {
        "WORKER_BINARY_SHA256": "0" * 64,
        "COMPILE_COMMAND_SHA256": "0" * 64,
        "CAPD_CONFIG_SHA256": "0" * 64,
        "COMPILER_SHA256": "0" * 64,
        "DEPENDENCIES_SHA256": "0" * 64,
        "LINK_INPUTS_SHA256": "0" * 64,
        "RUNTIME_LINKAGE_SHA256": "0" * 64,
        "RUNTIME_LIBRARIES_SHA256": "0" * 64,
        "CAPD_VERSION": "NOT_RUN",
        "INTERVAL_BACKEND": "NOT_RUN",
        "OPTIMIZATION_LEVEL": "NOT_RUN",
    }
    replay_directory: tempfile.TemporaryDirectory[str] | None = None
    worker: Path | None = None
    if args.replay_root_challenge is not None:
        assert args.capd_config is not None
        replay_directory = tempfile.TemporaryDirectory(
            prefix="cs6-c1-cover-canonical-build-"
        )
        worker, compile_metadata = compile_canonical_worker(
            source,
            source_sha,
            args.capd_config.resolve(),
            args.cxx,
            Path(replay_directory.name),
            args.timeout_seconds,
        )

    replay_rows: list[bytes] = []
    audit_bundle_sha = "0" * 64
    audit_bundle_published = False
    verified_count = 0
    leaf_mutation_tests = leaf_mutations_rejected = 0
    audited_methods: set[str] = set()
    observed_methods = {
        node.method for node in terminals if node.action == "CERTIFIED"
    }
    try:
        for node in terminals:
            audit_leaf = node.action == "CERTIFIED" and node.method not in audited_methods
            replay_row, audited, rejected = verify_terminal_artifacts(
                node,
                args.bundle_root.resolve(),
                verifier,
                source_sha,
                args.root_challenge,
                worker,
                args.replay_root_challenge,
                audit_leaf,
                args.timeout_seconds,
            )
            if node.action == "CERTIFIED":
                verified_count += 1
                if replay_row is not None:
                    replay_rows.append(replay_row)
            if audited:
                audited_methods.add(node.method)
                leaf_mutation_tests += audited
                leaf_mutations_rejected += rejected
        if stable_bytes(original_source, "canonical worker source") != source_raw:
            fail("canonical worker source changed during aggregation")
        if stable_bytes(original_verifier, "canonical leaf verifier") != verifier_raw:
            fail("canonical leaf verifier changed during aggregation")
        if stable_bytes(args.tree, "cover tree") != tree_raw:
            fail("cover tree changed during aggregation")
        if stable_bytes(source, "worker source snapshot") != source_raw:
            fail("worker source snapshot changed during aggregation")
        if stable_bytes(verifier, "leaf verifier snapshot") != verifier_raw:
            fail("leaf verifier snapshot changed during aggregation")
        if stable_bytes(tree_snapshot, "cover tree snapshot") != tree_raw:
            fail("cover tree snapshot changed during aggregation")
        if digest(aggregate_path) != aggregate_sha:
            fail("canonical aggregator changed during aggregation")
        if replay_directory is not None:
            assert args.audit_dir is not None
            assert worker is not None
            build_root = Path(replay_directory.name)
            if digest(worker) != compile_metadata["WORKER_BINARY_SHA256"]:
                fail("canonical worker binary changed during replay")
            capd_config = args.capd_config.resolve()
            if digest(capd_config) != compile_metadata["CAPD_CONFIG_SHA256"]:
                fail("capd-config changed during replay")
            current_cxx = shutil.which(args.cxx)
            if current_cxx is None or digest(Path(current_cxx).resolve()) != compile_metadata[
                "COMPILER_SHA256"
            ]:
                fail("C++ compiler changed during replay")
            artifact_keys = {
                "compile-command.txt": "COMPILE_COMMAND_SHA256",
                "dependencies.sha256": "DEPENDENCIES_SHA256",
                "link-inputs.sha256": "LINK_INPUTS_SHA256",
                "runtime-linkage.txt": "RUNTIME_LINKAGE_SHA256",
                "runtime-libraries.sha256": "RUNTIME_LIBRARIES_SHA256",
            }
            artifact_bytes: dict[str, bytes] = {}
            for name, metadata_key in artifact_keys.items():
                raw = stable_bytes(build_root / name, name)
                if digest_bytes(raw) != compile_metadata[metadata_key]:
                    fail(f"canonical build artifact changed during replay: {name}")
                artifact_bytes[name] = raw
            if stable_bytes(
                build_root / "worker-source.cpp", "bundled worker source"
            ) != source_raw:
                fail("bundled worker source differs from compiled source")
            verify_file_manifest(artifact_bytes["dependencies.sha256"], source)
            verify_file_manifest(artifact_bytes["link-inputs.sha256"])
            verify_file_manifest(artifact_bytes["runtime-libraries.sha256"])
            audit_bundle_sha = publish_audit_bundle(
                build_root, args.audit_dir, b"".join(sorted(replay_rows))
            )
            audit_bundle_published = True
    except subprocess.TimeoutExpired as error:
        raise CoverError("terminal verification or replay timed out") from error
    finally:
        if replay_directory is not None:
            replay_directory.cleanup()
        snapshot_directory.cleanup()
    certified_count = sum(node.action == "CERTIFIED" for node in terminals)
    unresolved_count = len(terminals) - certified_count
    all_verified = (
        certified_count > 0
        and unresolved_count == 0
        and verified_count == certified_count
    )
    fresh_replay_all = (
        args.replay_root_challenge is not None
        and len(replay_rows) == certified_count
        and certified_count > 0
    )
    methods_mutation_audited = audited_methods == observed_methods and bool(observed_methods)
    replay_ledger = b"".join(sorted(replay_rows))
    replay_ledger_sha = digest_bytes(replay_ledger) if replay_rows else "0" * 64
    local_full_source = (
        accepted_area == 1
        and unresolved_count == 0
        and all_verified
        and fresh_replay_all
        and mutation_tests > 0
        and mutation_tests == mutation_rejected
        and leaf_mutation_tests > 0
        and leaf_mutation_tests == leaf_mutations_rejected
        and methods_mutation_audited
        and audit_bundle_published
    )
    full_source = False
    fields = (
        ("SCHEMA", "sounio.cs6.c1-full-source-cover-certificate.v1"),
        ("SOURCE", "N0"),
        ("TREE_SHA256", tree_sha),
        ("SOURCE_SHA256", source_sha),
        ("LEAF_VERIFIER_SHA256", verifier_sha),
        ("AGGREGATOR_SHA256", aggregate_sha),
        ("BUNDLE_LEDGER_SHA256", bundle_ledger_sha),
        ("WORKER_BINARY_SHA256", compile_metadata["WORKER_BINARY_SHA256"]),
        ("COMPILE_COMMAND_SHA256", compile_metadata["COMPILE_COMMAND_SHA256"]),
        ("CAPD_CONFIG_SHA256", compile_metadata["CAPD_CONFIG_SHA256"]),
        ("COMPILER_SHA256", compile_metadata["COMPILER_SHA256"]),
        ("DEPENDENCIES_SHA256", compile_metadata["DEPENDENCIES_SHA256"]),
        ("LINK_INPUTS_SHA256", compile_metadata["LINK_INPUTS_SHA256"]),
        ("RUNTIME_LINKAGE_SHA256", compile_metadata["RUNTIME_LINKAGE_SHA256"]),
        ("RUNTIME_LIBRARIES_SHA256", compile_metadata["RUNTIME_LIBRARIES_SHA256"]),
        ("REPLAY_LEDGER_SHA256", replay_ledger_sha),
        ("REPLAY_AUDIT_MANIFEST_SHA256", audit_bundle_sha),
        ("ROOT_CHALLENGE", args.root_challenge),
        (
            "REPLAY_ROOT_CHALLENGE",
            args.replay_root_challenge or "0" * 64,
        ),
        ("CAPD_VERSION", compile_metadata["CAPD_VERSION"]),
        ("INTERVAL_BACKEND", compile_metadata["INTERVAL_BACKEND"]),
        ("OPTIMIZATION_LEVEL", compile_metadata["OPTIMIZATION_LEVEL"]),
        ("NODE_COUNT", str(len(nodes))),
        ("TERMINAL_COUNT", str(len(terminals))),
        ("CERTIFIED_TERMINALS", str(certified_count)),
        ("UNRESOLVED_TERMINALS", str(unresolved_count)),
        ("ROOT_LOGICAL_COVER_EXACT", "true"),
        ("LOGICAL_INTERIORS_DISJOINT", "true"),
        ("ACCEPTED_AREA_NUMERATOR", str(accepted_area.numerator)),
        ("ACCEPTED_AREA_DENOMINATOR", str(accepted_area.denominator)),
        ("UNRESOLVED_AREA_NUMERATOR", str(unresolved_area.numerator)),
        ("UNRESOLVED_AREA_DENOMINATOR", str(unresolved_area.denominator)),
        ("ALL_TERMINALS_VERIFIED", str(all_verified).lower()),
        ("FRESH_REPLAY_ALL", str(fresh_replay_all).lower()),
        ("REPLAY_LEDGER_RETAINED", str(audit_bundle_published).lower()),
        ("REPLAY_RECEIPTS_RETAINED", "false"),
        ("MUTATION_TESTS", str(mutation_tests)),
        ("MUTATIONS_REJECTED", str(mutation_rejected)),
        ("LEAF_MUTATION_TESTS", str(leaf_mutation_tests)),
        ("LEAF_MUTATIONS_REJECTED", str(leaf_mutations_rejected)),
        ("OBSERVED_CERTIFICATE_METHODS", ",".join(sorted(observed_methods)) or "NONE"),
        ("MUTATION_AUDITED_METHODS", ",".join(sorted(audited_methods)) or "NONE"),
        ("ALL_OBSERVED_METHODS_MUTATION_AUDITED", str(methods_mutation_audited).lower()),
        (
            "POSITIVE_PROJECTIVE_METHOD_EXERCISED",
            str(any(method.startswith("PROJECTIVE_") for method in observed_methods)).lower(),
        ),
        (
            "LOCAL_FULL_SOURCE_CERTIFICATE_COMPLETE",
            str(local_full_source).lower(),
        ),
        ("EXECUTION_PROVENANCE_ATTESTED", "false"),
        ("REMOTE_ATTESTATION_PRESENT", "false"),
        ("PROMOTION_ELIGIBLE", "false"),
        ("FULL_SOURCE_CARRIER_PROVED", str(full_source).lower()),
        ("PROJECTIVE_RICCATI_INTEGRATED", "false"),
        ("HYPERBOLICITY_PROVED", "false"),
        ("CHAOTIC_ATTRACTOR_PROVED", "false"),
    )
    write_certificate(args.output, fields)
    for key, value in fields:
        print(f"{key}={value}")
    if args.require_full_source and not full_source:
        return 3
    if args.require_local_full_source and not local_full_source:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CoverError as error:
        print(f"cover error: {error}", file=sys.stderr)
        raise SystemExit(1)
