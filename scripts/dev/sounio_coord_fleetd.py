#!/usr/bin/env python3
"""Durable desired-state reconciliation for Sounio agent fleets."""

from __future__ import annotations

import argparse
import base64
import contextlib
import dataclasses
import fcntl
import functools
import hashlib
import hmac
import json
import os
import secrets
import shutil
import sqlite3
import subprocess
import sys
import time
import tempfile
import tomllib
import uuid
from pathlib import Path
from typing import Any, Iterator


PROTOCOL_VERSION = 1
RUNTIME_VERSION = "2026.08.23.3"
SCHEMA_VERSION = "1"
ZERO_HASH = "0" * 64
SAFE_TOKEN = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"


class FleetdError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class LaneSpec:
    slot: str
    enabled: bool
    restart: str
    cwd: Path
    kind: str | None
    home: Path | None
    agent: str | None
    lane: str | None
    session_id: str | None
    identity: str | None
    command: tuple[str, ...]

    def canonical(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "command": list(self.command),
            "cwd": str(self.cwd),
            "enabled": self.enabled,
            "home": str(self.home) if self.home else None,
            "identity": self.identity,
            "kind": self.kind,
            "lane": self.lane,
            "restart": self.restart,
            "session_id": self.session_id,
            "slot": self.slot,
        }

    @property
    def desired_hash(self) -> str:
        return digest_json(self.canonical())


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_write_secret_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FleetdError(f"refusing to replace capability file: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(canonical_json(value))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def failpoint(name: str) -> None:
    if os.environ.get("SOUNIO_FLEET_FAILPOINT") != name:
        return
    print(f"FLEET_FAILPOINT name={name} exit=197", file=sys.stderr, flush=True)
    os._exit(197)


def slug(value: str, limit: int = 96) -> str:
    cleaned = "".join(character if character in SAFE_TOKEN else "-" for character in value)
    cleaned = cleaned.strip("-")[:limit]
    if not cleaned:
        raise FleetdError("slot becomes empty after normalization")
    return cleaned


def resolve_path(value: str, base: Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def git_common_dir(cwd: Path) -> Path:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise FleetdError(f"not an attached Git worktree: {cwd}")
    return Path(result.stdout.strip()).resolve()


def default_db(cwd: Path) -> Path:
    override = os.environ.get("SOUNIO_FLEET_DB")
    if override:
        return Path(override).expanduser().resolve()
    return git_common_dir(cwd) / "sounio-fleet" / "fleet.db"


def agentd_state_root(cwd: Path) -> Path:
    override = os.environ.get("SOUNIO_AGENTD_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return git_common_dir(cwd) / "sounio-agentd"


def fleet_agent_command() -> Path:
    sibling = Path(__file__).resolve().with_name("sounio-fleet-agent-runtime")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling
    override = os.environ.get("SOUNIO_FLEET_AGENT_COMMAND")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FleetdError("sounio-fleet-agent-runtime is not installed beside fleetd")


def load_config(path: Path) -> list[LaneSpec]:
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise FleetdError(f"cannot read fleet config {path}: {exc}") from exc
    if document.get("version") != 1:
        raise FleetdError("fleet config requires version = 1")
    raw_lanes = document.get("lane")
    if not isinstance(raw_lanes, list) or not raw_lanes:
        raise FleetdError("fleet config requires at least one [[lane]]")
    base = path.parent.resolve()
    lanes: list[LaneSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_lanes, start=1):
        if not isinstance(raw, dict):
            raise FleetdError(f"lane {index} must be a table")
        slot = raw.get("slot")
        if not isinstance(slot, str) or not slot.strip():
            raise FleetdError(f"lane {index} requires a non-empty slot")
        if slot in seen:
            raise FleetdError(f"duplicate fleet slot: {slot}")
        seen.add(slot)
        cwd_value = raw.get("cwd")
        if not isinstance(cwd_value, str) or not cwd_value:
            raise FleetdError(f"lane {slot} requires cwd")
        cwd = resolve_path(cwd_value, base)
        kind = raw.get("kind")
        command_value = raw.get("command", [])
        if kind is not None and not isinstance(kind, str):
            raise FleetdError(f"lane {slot} kind must be a string")
        if not isinstance(command_value, list) or any(
            not isinstance(item, str) or not item for item in command_value
        ):
            raise FleetdError(f"lane {slot} command must be an array of strings")
        command = tuple(command_value)
        if bool(kind) == bool(command):
            raise FleetdError(f"lane {slot} requires exactly one of kind or command")
        restart = raw.get("restart", "never")
        if restart not in {"never", "on-failure", "always"}:
            raise FleetdError(
                f"lane {slot} restart must be never, on-failure, or always"
            )
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise FleetdError(f"lane {slot} enabled must be boolean")
        home_value = raw.get("home")
        home = resolve_path(home_value, base) if isinstance(home_value, str) else None
        if kind and home is None:
            raise FleetdError(f"lane {slot} with kind requires home")
        agent = raw.get("agent")
        lane = raw.get("lane")
        session_id = raw.get("session_id")
        identity = raw.get("identity")
        for key, value in (
            ("agent", agent),
            ("lane", lane),
            ("session_id", session_id),
            ("identity", identity),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise FleetdError(f"lane {slot} {key} must be a non-empty string")
        if command and (not agent or not session_id):
            raise FleetdError(f"lane {slot} command requires agent and session_id")
        if identity is not None and identity not in {"exact", "bootstrap", "standalone"}:
            raise FleetdError(f"lane {slot} has unsupported identity: {identity}")
        lanes.append(
            LaneSpec(
                slot=slot,
                enabled=enabled,
                restart=restart,
                cwd=cwd,
                kind=kind,
                home=home,
                agent=agent,
                lane=lane,
                session_id=session_id,
                identity=identity,
                command=command,
            )
        )
    return lanes


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if path.is_symlink():
        raise FleetdError(f"refusing symlink fleet database: {path}")
    connection = sqlite3.connect(path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY,
            event_id TEXT NOT NULL UNIQUE,
            occurred_utc TEXT NOT NULL,
            event_type TEXT NOT NULL,
            slot TEXT NOT NULL,
            causal_key TEXT NOT NULL UNIQUE,
            payload TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS slot_view (
            slot TEXT PRIMARY KEY,
            desired_json TEXT,
            desired_hash TEXT,
            observed_json TEXT,
            observed_fingerprint TEXT,
            decision TEXT,
            reason TEXT,
            last_event_seq INTEGER NOT NULL,
            updated_utc TEXT NOT NULL
        );
        """
    )
    existing = connection.execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()
    if existing and existing["value"] != SCHEMA_VERSION:
        raise FleetdError(
            f"unsupported fleet database schema: {existing['value']}"
        )
    connection.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (SCHEMA_VERSION,),
    )
    connection.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('database_id', ?)",
        (str(uuid.uuid4()),),
    )
    connection.commit()
    os.chmod(path, 0o600)
    return connection


@contextlib.contextmanager
def writer_lock(db_path: Path) -> Iterator[None]:
    lock_path = db_path.with_suffix(db_path.suffix + ".lock")
    lock_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        os.chmod(lock_path, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def event_hash(
    seq: int,
    occurred_utc: str,
    event_type: str,
    slot: str,
    causal_key: str,
    payload_text: str,
    prev_hash: str,
) -> str:
    material = {
        "causal_key": causal_key,
        "event_type": event_type,
        "occurred_utc": occurred_utc,
        "payload": payload_text,
        "prev_hash": prev_hash,
        "seq": seq,
        "slot": slot,
    }
    return digest_json(material)


def apply_event_to_view(connection: sqlite3.Connection, event: sqlite3.Row) -> None:
    row = connection.execute(
        "SELECT * FROM slot_view WHERE slot = ?", (event["slot"],)
    ).fetchone()
    current = dict(row) if row else {
        "slot": event["slot"],
        "desired_json": None,
        "desired_hash": None,
        "observed_json": None,
        "observed_fingerprint": None,
        "decision": None,
        "reason": None,
    }
    payload = json.loads(event["payload"])
    if event["event_type"] == "DESIRED_DECLARED":
        current["desired_json"] = canonical_json(payload["desired"])
        current["desired_hash"] = payload["desired_hash"]
    elif event["event_type"] == "OBSERVATION":
        current["observed_json"] = canonical_json(payload)
        current["observed_fingerprint"] = payload["fingerprint"]
    elif event["event_type"] == "RECONCILE_PLANNED":
        current["decision"] = payload["decision"]
        current["reason"] = payload["reason"]
    connection.execute(
        """
        INSERT INTO slot_view(
            slot, desired_json, desired_hash, observed_json,
            observed_fingerprint, decision, reason, last_event_seq, updated_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(slot) DO UPDATE SET
            desired_json=excluded.desired_json,
            desired_hash=excluded.desired_hash,
            observed_json=excluded.observed_json,
            observed_fingerprint=excluded.observed_fingerprint,
            decision=excluded.decision,
            reason=excluded.reason,
            last_event_seq=excluded.last_event_seq,
            updated_utc=excluded.updated_utc
        """,
        (
            current["slot"],
            current["desired_json"],
            current["desired_hash"],
            current["observed_json"],
            current["observed_fingerprint"],
            current["decision"],
            current["reason"],
            event["seq"],
            event["occurred_utc"],
        ),
    )


def append_event(
    connection: sqlite3.Connection,
    event_type: str,
    slot: str,
    causal_key: str,
    payload: dict[str, Any],
) -> tuple[sqlite3.Row, bool]:
    payload_text = canonical_json(payload)
    connection.execute("BEGIN IMMEDIATE")
    try:
        existing = connection.execute(
            "SELECT * FROM events WHERE causal_key = ?", (causal_key,)
        ).fetchone()
        if existing:
            if (
                existing["event_type"] != event_type
                or existing["slot"] != slot
                or existing["payload"] != payload_text
            ):
                raise FleetdError(f"causal key collision: {causal_key}")
            connection.commit()
            return existing, False
        tail = connection.execute(
            "SELECT seq, event_hash FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = int(tail["seq"]) + 1 if tail else 1
        previous = tail["event_hash"] if tail else ZERO_HASH
        occurred = utc_now()
        hashed = event_hash(
            seq, occurred, event_type, slot, causal_key, payload_text, previous
        )
        event_id = f"evt-{hashed[:24]}"
        connection.execute(
            """
            INSERT INTO events(
                seq, event_id, occurred_utc, event_type, slot,
                causal_key, payload, prev_hash, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                seq,
                event_id,
                occurred,
                event_type,
                slot,
                causal_key,
                payload_text,
                previous,
                hashed,
            ),
        )
        event = connection.execute(
            "SELECT * FROM events WHERE seq = ?", (seq,)
        ).fetchone()
        if event is None:
            raise FleetdError(f"inserted event {seq} disappeared")
        apply_event_to_view(connection, event)
        connection.commit()
        return event, True
    except Exception:
        connection.rollback()
        raise


def transition_key(
    connection: sqlite3.Connection, prefix: str, slot: str, payload: dict[str, Any]
) -> str:
    tail = connection.execute("SELECT max(seq) AS seq FROM events").fetchone()
    anchor = int(tail["seq"] or 0)
    return f"{prefix}:{slot}:after-{anchor}:{digest_json(payload)}"


def unchanged_event(
    connection: sqlite3.Connection,
    event_type: str,
    slot: str,
    payload: dict[str, Any],
) -> sqlite3.Row | None:
    row = connection.execute(
        """
        SELECT * FROM events
        WHERE event_type = ? AND slot = ?
        ORDER BY seq DESC LIMIT 1
        """,
        (event_type, slot),
    ).fetchone()
    if row and row["payload"] == canonical_json(payload):
        return row
    return None


def verify_chain(connection: sqlite3.Connection) -> tuple[int, str]:
    previous = ZERO_HASH
    expected_seq = 1
    rows = connection.execute("SELECT * FROM events ORDER BY seq").fetchall()
    for row in rows:
        if row["seq"] != expected_seq:
            raise FleetdError(
                f"event sequence gap: expected {expected_seq}, found {row['seq']}"
            )
        if row["prev_hash"] != previous:
            raise FleetdError(f"event {row['seq']} has an invalid previous hash")
        try:
            parsed_payload = json.loads(row["payload"])
        except json.JSONDecodeError as exc:
            raise FleetdError(f"event {row['seq']} payload is not JSON") from exc
        if row["payload"] != canonical_json(parsed_payload):
            raise FleetdError(f"event {row['seq']} payload is not canonical JSON")
        calculated = event_hash(
            row["seq"],
            row["occurred_utc"],
            row["event_type"],
            row["slot"],
            row["causal_key"],
            row["payload"],
            row["prev_hash"],
        )
        if calculated != row["event_hash"]:
            raise FleetdError(f"event {row['seq']} hash mismatch")
        if row["event_id"] != f"evt-{calculated[:24]}":
            raise FleetdError(f"event {row['seq']} id mismatch")
        previous = calculated
        expected_seq += 1
    return len(rows), previous


def verify_views(connection: sqlite3.Connection) -> int:
    expected: dict[str, dict[str, Any]] = {}
    for event in connection.execute("SELECT * FROM events ORDER BY seq"):
        current = expected.setdefault(
            event["slot"],
            {
                "slot": event["slot"],
                "desired_json": None,
                "desired_hash": None,
                "observed_json": None,
                "observed_fingerprint": None,
                "decision": None,
                "reason": None,
            },
        )
        try:
            payload = json.loads(event["payload"])
            if event["event_type"] == "DESIRED_DECLARED":
                current["desired_json"] = canonical_json(payload["desired"])
                current["desired_hash"] = payload["desired_hash"]
            elif event["event_type"] == "OBSERVATION":
                current["observed_json"] = canonical_json(payload)
                current["observed_fingerprint"] = payload["fingerprint"]
            elif event["event_type"] == "RECONCILE_PLANNED":
                current["decision"] = payload["decision"]
                current["reason"] = payload["reason"]
        except (KeyError, TypeError, ValueError) as exc:
            raise FleetdError(
                f"event {event['seq']} has an invalid {event['event_type']} payload"
            ) from exc
        current["last_event_seq"] = event["seq"]
        current["updated_utc"] = event["occurred_utc"]
    actual_rows = {
        row["slot"]: dict(row)
        for row in connection.execute("SELECT * FROM slot_view ORDER BY slot")
    }
    if actual_rows != expected:
        raise FleetdError(
            "materialized fleet view mismatch; verify the log, then run rebuild-views"
        )
    return len(actual_rows)


def verify_state(connection: sqlite3.Connection) -> tuple[int, str, int]:
    count, tail = verify_chain(connection)
    views = verify_views(connection)
    return count, tail, views


def verify_config_coverage(
    connection: sqlite3.Connection, specs: list[LaneSpec]
) -> None:
    configured = {spec.slot for spec in specs}
    tracked = {
        row["slot"]
        for row in connection.execute(
            "SELECT slot FROM slot_view WHERE desired_json IS NOT NULL"
        )
    }
    omitted = sorted(tracked - configured)
    if omitted:
        raise FleetdError(
            "fleet config silently omitted tracked slots; retain them with "
            f"enabled = false: {','.join(omitted)}"
        )


def rebuild_views(connection: sqlite3.Connection) -> None:
    verify_chain(connection)
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute("DELETE FROM slot_view")
        for event in connection.execute("SELECT * FROM events ORDER BY seq"):
            apply_event_to_view(connection, event)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


@functools.lru_cache(maxsize=1)
def openssl_command() -> str:
    configured = os.environ.get("SOUNIO_FLEET_OPENSSL")
    if configured and not Path(configured).is_absolute():
        raise FleetdError("SOUNIO_FLEET_OPENSSL must be an absolute path")
    command = configured or shutil.which("openssl")
    if not command:
        raise FleetdError("OpenSSL is required for Ed25519 fleet anchors")
    executable = Path(command).resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise FleetdError(f"OpenSSL executable is not usable: {executable}")
    try:
        result = subprocess.run(
            [str(executable), "version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise FleetdError("OpenSSL version probe timed out") from exc
    words = result.stdout.split()
    try:
        major = int(words[1].split(".", maxsplit=1)[0])
    except (IndexError, ValueError) as exc:
        raise FleetdError("OpenSSL returned an unparseable version") from exc
    if result.returncode != 0 or words[0] != "OpenSSL" or major < 3:
        raise FleetdError("OpenSSL 3 or newer is required for Ed25519 anchors")
    return str(executable)


def run_openssl(arguments: list[str], *, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            [openssl_command(), *arguments],
            input=input_bytes,
            check=False,
            capture_output=True,
            timeout=30.0,
        )
    except subprocess.TimeoutExpired as exc:
        raise FleetdError("OpenSSL timed out during Ed25519 operation") from exc
    if result.returncode != 0:
        reason = result.stderr.decode("utf-8", errors="replace").strip()
        raise FleetdError(f"OpenSSL refused Ed25519 operation: {reason}")
    return result.stdout


def generate_anchor_keypair(private_key: Path, public_key: Path) -> None:
    for path in (private_key, public_key):
        if path.exists() or path.is_symlink():
            raise FleetdError(f"refusing to replace anchor key: {path}")
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sounio-fleet-keygen-") as raw_temp:
        temporary = Path(raw_temp)
        staged_private = temporary / "private.pem"
        staged_public = temporary / "public.pem"
        run_openssl(
            ["genpkey", "-algorithm", "ED25519", "-out", str(staged_private)]
        )
        run_openssl(
            [
                "pkey",
                "-in",
                str(staged_private),
                "-pubout",
                "-out",
                str(staged_public),
            ]
        )
        os.replace(staged_private, private_key)
        os.replace(staged_public, public_key)
    os.chmod(private_key, 0o600)
    os.chmod(public_key, 0o644)


def validate_private_key(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FleetdError(f"private anchor key is missing or a symlink: {path}")
    if path.stat().st_mode & 0o077:
        raise FleetdError(f"private anchor key permissions are not 600: {path}")


def public_key_fingerprint(public_key: Path) -> str:
    if public_key.is_symlink() or not public_key.is_file():
        raise FleetdError(f"public anchor key is missing or a symlink: {public_key}")
    der = run_openssl(
        ["pkey", "-pubin", "-in", str(public_key), "-outform", "DER"]
    )
    return hashlib.sha256(der).hexdigest()


def sign_ed25519(private_key: Path, material: bytes) -> bytes:
    validate_private_key(private_key)
    with tempfile.TemporaryDirectory(prefix="sounio-fleet-sign-") as raw_temp:
        temporary = Path(raw_temp)
        message = temporary / "message"
        signature = temporary / "signature"
        message.write_bytes(material)
        run_openssl(
            [
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_key),
                "-in",
                str(message),
                "-out",
                str(signature),
            ]
        )
        return signature.read_bytes()


def verify_ed25519(public_key: Path, material: bytes, signature: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="sounio-fleet-verify-") as raw_temp:
        temporary = Path(raw_temp)
        message = temporary / "message"
        signature_path = temporary / "signature"
        message.write_bytes(material)
        signature_path.write_bytes(signature)
        run_openssl(
            [
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey",
                str(public_key),
                "-in",
                str(message),
                "-sigfile",
                str(signature_path),
            ]
        )


def read_anchor(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FleetdError(f"anchor is missing or a symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetdError(f"cannot read fleet anchor {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise FleetdError(f"unsupported fleet anchor: {path}")
    return value


def anchor_document_digest(document: dict[str, Any]) -> str:
    return digest_json(document)


def verify_anchor_document(
    connection: sqlite3.Connection,
    path: Path,
    public_key: Path,
    expected_previous: str | None = None,
) -> dict[str, Any]:
    document = read_anchor(path)
    payload = document.get("payload")
    encoded_signature = document.get("signature_base64")
    if not isinstance(payload, dict) or not isinstance(encoded_signature, str):
        raise FleetdError(f"anchor omits payload or signature: {path}")
    if payload.get("algorithm") != "Ed25519":
        raise FleetdError(f"anchor uses an unsupported algorithm: {path}")
    key_fingerprint = public_key_fingerprint(public_key)
    if payload.get("public_key_sha256") != key_fingerprint:
        raise FleetdError(f"anchor public-key identity mismatch: {path}")
    database = connection.execute(
        "SELECT value FROM meta WHERE key = 'database_id'"
    ).fetchone()
    if not database or payload.get("database_id") != database["value"]:
        raise FleetdError(f"anchor belongs to a different fleet database: {path}")
    count = payload.get("event_count")
    if not isinstance(count, int) or count < 1:
        raise FleetdError(f"anchor has an invalid event count: {path}")
    event = connection.execute(
        "SELECT event_hash FROM events WHERE seq = ?", (count,)
    ).fetchone()
    if not event or payload.get("tail_hash") != event["event_hash"]:
        raise FleetdError(f"anchor does not match its event-log prefix: {path}")
    if (
        expected_previous is not None
        and payload.get("previous_anchor_sha256") != expected_previous
    ):
        raise FleetdError(f"anchor chain predecessor mismatch: {path}")
    try:
        signature = base64.b64decode(encoded_signature, validate=True)
    except ValueError as exc:
        raise FleetdError(f"anchor signature is not canonical base64: {path}") from exc
    verify_ed25519(public_key, canonical_json(payload).encode("utf-8"), signature)
    return document


def anchor_paths(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob("anchor-*.json"))


def verify_anchor_directory(
    connection: sqlite3.Connection, directory: Path, public_key: Path
) -> tuple[int, dict[str, Any] | None]:
    verify_state(connection)
    previous = ZERO_HASH
    latest: dict[str, Any] | None = None
    paths = anchor_paths(directory)
    if not paths:
        raise FleetdError(f"no signed fleet anchors found in {directory}")
    previous_count = 0
    for path in paths:
        document = verify_anchor_document(
            connection, path, public_key, expected_previous=previous
        )
        payload = document["payload"]
        if payload["event_count"] <= previous_count:
            raise FleetdError(f"fleet anchors are not strictly increasing: {path}")
        previous_count = payload["event_count"]
        previous = anchor_document_digest(document)
        latest = document
    return len(paths), latest


def create_anchor(
    connection: sqlite3.Connection,
    directory: Path,
    private_key: Path,
    public_key: Path,
) -> Path:
    count, tail, _ = verify_state(connection)
    if count < 1:
        raise FleetdError("cannot anchor an empty fleet event log")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    previous = ZERO_HASH
    paths = anchor_paths(directory)
    if paths:
        _, latest = verify_anchor_directory(connection, directory, public_key)
        if latest is None:
            raise FleetdError("signed fleet anchor directory has no latest anchor")
        if latest["payload"]["event_count"] == count:
            return paths[-1]
        previous = anchor_document_digest(latest)
    database = connection.execute(
        "SELECT value FROM meta WHERE key = 'database_id'"
    ).fetchone()
    if not database:
        raise FleetdError("fleet database identity is missing")
    payload = {
        "algorithm": "Ed25519",
        "anchored_utc": utc_now(),
        "database_id": database["value"],
        "event_count": count,
        "previous_anchor_sha256": previous,
        "public_key_sha256": public_key_fingerprint(public_key),
        "tail_hash": tail,
    }
    signature = sign_ed25519(
        private_key, canonical_json(payload).encode("utf-8")
    )
    document = {
        "payload": payload,
        "signature_base64": base64.b64encode(signature).decode("ascii"),
        "version": 1,
    }
    verify_ed25519(
        public_key,
        canonical_json(payload).encode("utf-8"),
        signature,
    )
    output = directory / f"anchor-{count:020d}-{tail[:16]}.json"
    atomic_write_secret_json(output, document)
    return output


def mapping_path(spec: LaneSpec) -> Path:
    return agentd_state_root(spec.cwd) / "fleet-slots" / f"{slug(spec.slot)}.json"


def read_mapping(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"mapping-unreadable:{exc.__class__.__name__}"
    if not isinstance(value, dict):
        return None, "mapping-invalid-root"
    return value, None


def parse_fleet_status(output: str, slot: str) -> dict[str, str]:
    for line in output.splitlines():
        if not line.startswith("FLEET_SLOT_STATUS "):
            continue
        fields: dict[str, str] = {}
        for token in line.split()[1:]:
            key, separator, value = token.partition("=")
            if separator:
                fields[key] = value
        if fields.get("slot") == slot:
            return fields
    return {}


def desired_mapping_mismatch(spec: LaneSpec, mapping: dict[str, Any]) -> str | None:
    expected: dict[str, str] = {"worktree": str(spec.cwd)}
    if spec.kind:
        expected["agent"] = spec.kind
    if spec.agent:
        expected["agent"] = spec.agent
    if spec.lane:
        expected["lane"] = spec.lane
    if spec.session_id:
        expected["session_id"] = spec.session_id
    if spec.identity:
        expected["identity"] = spec.identity
    if spec.command:
        original_digest = hashlib.sha256(
            canonical_json(list(spec.command)).encode("utf-8")
        ).hexdigest()
        start_capability_id = mapping.get("start_capability_id")
        if isinstance(start_capability_id, str) and start_capability_id:
            env_command = shutil.which("env")
            if env_command is None:
                return "capability-env-unavailable"
            wrapped = [
                str(Path(env_command).resolve()),
                f"SOUNIO_FLEET_START_CAPABILITY_ID={start_capability_id}",
                *spec.command,
            ]
            expected["command"] = Path(env_command).name
            expected["argv_digest"] = hashlib.sha256(
                canonical_json(wrapped).encode("utf-8")
            ).hexdigest()
            expected["original_argv_digest"] = original_digest
        else:
            expected["command"] = Path(spec.command[0]).name
            expected["argv_digest"] = original_digest
    for key, value in expected.items():
        observed = mapping.get(key)
        if key == "worktree" and isinstance(observed, str):
            observed = str(Path(observed).resolve())
        if observed != value:
            return f"desired-{key}-mismatch"
    return None


def observe_spec(spec: LaneSpec) -> dict[str, Any]:
    try:
        path = mapping_path(spec)
    except FleetdError:
        base: dict[str, Any] = {
            "state": "drifted",
            "reason": "worktree-unavailable",
            "mapping_path": "-",
            "mapping_hash": None,
            "generation": None,
        }
        base["fingerprint"] = digest_json(base)
        return base
    mapping, mapping_error = read_mapping(path)
    if mapping_error:
        base = {
            "state": "drifted",
            "reason": mapping_error,
            "mapping_path": str(path),
            "mapping_hash": None,
            "generation": None,
        }
    elif mapping is None:
        base = {
            "state": "absent",
            "reason": "slot-mapping-absent",
            "mapping_path": str(path),
            "mapping_hash": None,
            "generation": None,
        }
    else:
        mismatch = desired_mapping_mismatch(spec, mapping)
        try:
            result = subprocess.run(
                [
                    str(fleet_agent_command()),
                    "status",
                    "--cwd",
                    str(spec.cwd),
                    "--slot",
                    spec.slot,
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            fields = parse_fleet_status(result.stdout, spec.slot)
        except subprocess.TimeoutExpired:
            fields = {"state": "unreachable"}
        state = fields.get("state", "drifted")
        reason = "fleet-runtime-observation"
        if mismatch:
            state = "drifted"
            reason = mismatch
        elif not fields:
            state = "drifted"
            reason = "fleet-runtime-omitted-slot"
        elif state == "active":
            reason = "supervisor-generation-verified"
        elif state == "unreachable":
            reason = "supervisor-unreachable-or-probe-timeout"
        elif state == "drifted":
            reason = "supervisor-generation-drift"
        base = {
            "state": state,
            "reason": reason,
            "mapping_path": str(path),
            "mapping_hash": digest_json(mapping),
            "generation": mapping.get("instance_id"),
            "agent": mapping.get("agent"),
            "lane": mapping.get("lane"),
            "session_id": mapping.get("session_id"),
            "argv_digest": mapping.get("argv_digest"),
            "start_capability_id": mapping.get("start_capability_id"),
            "harness_pid": fields.get("harness_pid"),
            "attached_clients": fields.get("attached_clients"),
        }
    fingerprint_source = dict(base)
    base["fingerprint"] = digest_json(fingerprint_source)
    return base


def previous_observation_state(
    connection: sqlite3.Connection, slot: str, current_seq: int
) -> str | None:
    rows = connection.execute(
        """
        SELECT payload FROM events
        WHERE slot = ? AND event_type = 'OBSERVATION' AND seq < ?
        ORDER BY seq DESC LIMIT 1
        """,
        (slot, current_seq),
    ).fetchall()
    if not rows:
        return None
    return json.loads(rows[0]["payload"]).get("state")


def decide(
    connection: sqlite3.Connection,
    spec: LaneSpec,
    observation: dict[str, Any],
    observation_seq: int,
) -> tuple[str, str]:
    state = observation["state"]
    if state == "drifted":
        return "blocked", observation["reason"]
    if state == "unreachable":
        return "blocked", "probe-unreachable-start-not-authorized"
    if not spec.enabled:
        if state == "active":
            return "hold", "disabled-active-stop-policy-not-authorized"
        return "noop", "disabled"
    if state == "active":
        return "noop", "desired-state-satisfied"
    if spec.restart == "never":
        return "hold", "restart-policy-never"
    if spec.restart == "on-failure":
        previous = previous_observation_state(connection, spec.slot, observation_seq)
        if previous is not None and previous != "active":
            return "hold", "on-failure-requires-prior-active-observation"
    if state == "absent":
        return "start", f"restart-policy-{spec.restart}"
    return "blocked", f"unsupported-observed-state:{state}"


def declare_desired(
    connection: sqlite3.Connection, spec: LaneSpec
) -> tuple[sqlite3.Row, bool]:
    payload = {"desired": spec.canonical(), "desired_hash": spec.desired_hash}
    unchanged = unchanged_event(connection, "DESIRED_DECLARED", spec.slot, payload)
    if unchanged:
        return unchanged, False
    return append_event(
        connection,
        "DESIRED_DECLARED",
        spec.slot,
        transition_key(connection, "desired", spec.slot, payload),
        payload,
    )


def record_observation(
    connection: sqlite3.Connection, spec: LaneSpec, observation: dict[str, Any]
) -> tuple[sqlite3.Row, bool]:
    unchanged = unchanged_event(connection, "OBSERVATION", spec.slot, observation)
    if unchanged:
        return unchanged, False
    return append_event(
        connection,
        "OBSERVATION",
        spec.slot,
        transition_key(connection, "observe", spec.slot, observation),
        observation,
    )


def record_plan(
    connection: sqlite3.Connection,
    spec: LaneSpec,
    observation: dict[str, Any],
    observation_seq: int,
    decision: str,
    reason: str,
) -> tuple[sqlite3.Row, bool]:
    payload = {
        "decision": decision,
        "desired_hash": spec.desired_hash,
        "observation_fingerprint": observation["fingerprint"],
        "observation_seq": observation_seq,
        "reason": reason,
    }
    unchanged = unchanged_event(
        connection, "RECONCILE_PLANNED", spec.slot, payload
    )
    if unchanged:
        return unchanged, False
    return append_event(
        connection,
        "RECONCILE_PLANNED",
        spec.slot,
        transition_key(connection, "plan", spec.slot, payload),
        payload,
    )


def capability_events(
    connection: sqlite3.Connection, capability_id: str
) -> list[tuple[str, dict[str, Any]]]:
    matched: list[tuple[str, dict[str, Any]]] = []
    for row in connection.execute(
        """
        SELECT event_type, payload FROM events
        WHERE event_type IN (
            'CAPABILITY_ISSUED', 'CAPABILITY_PUBLISHED',
            'CAPABILITY_CONSUMED', 'CAPABILITY_REVOKED'
        ) ORDER BY seq
        """
    ):
        payload = json.loads(row["payload"])
        if payload.get("capability_id") == capability_id:
            matched.append((row["event_type"], payload))
    return matched


def capability_document_digest(document: dict[str, Any]) -> str:
    public = {key: value for key, value in document.items() if key != "_path"}
    return hashlib.sha256(
        (canonical_json(public) + "\n").encode("utf-8")
    ).hexdigest()


def validate_capability_document(
    issued_event: sqlite3.Row,
    authority: dict[str, Any],
    document: dict[str, Any],
) -> None:
    capability_id = authority.get("capability_id")
    token = document.get("token")
    if not isinstance(capability_id, str) or not isinstance(token, str):
        raise FleetdError("capability document omits its identity or secret")
    required = {
        key: value
        for key, value in authority.items()
        if key not in {"issued_unix", "observation_seq", "token_hash"}
    }
    required["issued_event_hash"] = issued_event["event_hash"]
    required["version"] = 1
    if any(document.get(key) != value for key, value in required.items()):
        raise FleetdError(f"capability file {capability_id} binding was altered")
    observed_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(observed_hash, str(authority.get("token_hash", ""))):
        raise FleetdError(
            f"capability {capability_id} secret does not match its issuance"
        )


def publish_capability(
    connection: sqlite3.Connection,
    issued_event: sqlite3.Row,
    authority: dict[str, Any],
    document: dict[str, Any],
    output_path: Path,
) -> sqlite3.Row:
    validate_capability_document(issued_event, authority, document)
    payload = {
        "action": authority["action"],
        "capability_id": authority["capability_id"],
        "document_path": str(output_path),
        "document_sha256": capability_document_digest(document),
        "issued_event_hash": issued_event["event_hash"],
        "issued_event_seq": issued_event["seq"],
        "slot": authority["slot"],
    }
    event, _ = append_event(
        connection,
        "CAPABILITY_PUBLISHED",
        authority["slot"],
        f"capability-published:{authority['capability_id']}",
        payload,
    )
    return event


def recover_unpublished_capabilities(connection: sqlite3.Connection) -> int:
    recovered = 0
    issued_rows = connection.execute(
        "SELECT * FROM events WHERE event_type = 'CAPABILITY_ISSUED' ORDER BY seq"
    ).fetchall()
    for issued_event in issued_rows:
        authority = json.loads(issued_event["payload"])
        capability_id = authority.get("capability_id")
        raw_path = authority.get("capability_path")
        if not isinstance(capability_id, str) or not isinstance(raw_path, str):
            continue
        events = capability_events(connection, capability_id)
        kinds = [kind for kind, _ in events]
        if "CAPABILITY_PUBLISHED" in kinds or any(
            kind in {"CAPABILITY_CONSUMED", "CAPABILITY_REVOKED"}
            for kind in kinds
        ):
            continue
        path = absolute_without_symlink_resolution(raw_path)
        if not path.is_file() or path.is_symlink():
            append_event(
                connection,
                "CAPABILITY_REVOKED",
                authority["slot"],
                f"capability-revoked:{capability_id}:crash-before-publish",
                {
                    "capability_id": capability_id,
                    "reason": "crash-before-capability-publish",
                },
            )
            recovered += 1
            continue
        document = read_capability_file(path)
        validate_capability_document(issued_event, authority, document)
        publish_capability(connection, issued_event, authority, document, path)
        recovered += 1
    return recovered


def outstanding_capability_exists(
    connection: sqlite3.Connection,
    slot: str,
    desired_hash: str,
    observation_fingerprint: str,
) -> bool:
    issued: dict[str, dict[str, Any]] = {}
    terminal: set[str] = set()
    for row in connection.execute(
        """
        SELECT event_type, payload FROM events
        WHERE event_type IN (
            'CAPABILITY_ISSUED', 'CAPABILITY_CONSUMED', 'CAPABILITY_REVOKED'
        ) ORDER BY seq
        """
    ):
        payload = json.loads(row["payload"])
        capability_id = payload.get("capability_id")
        if not isinstance(capability_id, str):
            continue
        if row["event_type"] == "CAPABILITY_ISSUED":
            issued[capability_id] = payload
        else:
            terminal.add(capability_id)
    now = int(time.time())
    return any(
        capability_id not in terminal
        and payload.get("slot") == slot
        and payload.get("desired_hash") == desired_hash
        and payload.get("observation_fingerprint") == observation_fingerprint
        and int(payload.get("expires_unix", 0)) >= now
        for capability_id, payload in issued.items()
    )


def issue_start_capability(
    connection: sqlite3.Connection,
    spec: LaneSpec,
    observation: dict[str, Any],
    observation_seq: int,
    output_path: Path,
    ttl_seconds: int,
) -> str:
    if ttl_seconds < 1 or ttl_seconds > 86400:
        raise FleetdError("capability TTL must be between 1 and 86400 seconds")
    if outstanding_capability_exists(
        connection, spec.slot, spec.desired_hash, observation["fingerprint"]
    ):
        raise FleetdError(
            f"an unconsumed start capability already exists for slot {spec.slot}"
        )
    capability_id = f"cap-{uuid.uuid4()}"
    token = secrets.token_urlsafe(48)
    issued_unix = int(time.time())
    payload = {
        "action": "start",
        "capability_id": capability_id,
        "capability_path": str(output_path),
        "desired_hash": spec.desired_hash,
        "expires_unix": issued_unix + ttl_seconds,
        "issued_unix": issued_unix,
        "observation_fingerprint": observation["fingerprint"],
        "observation_seq": observation_seq,
        "slot": spec.slot,
        "token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    event, _ = append_event(
        connection,
        "CAPABILITY_ISSUED",
        spec.slot,
        f"capability-issued:{capability_id}",
        payload,
    )
    failpoint("start-capability:issued")
    document = {
        "action": "start",
        "capability_id": capability_id,
        "capability_path": str(output_path),
        "desired_hash": spec.desired_hash,
        "expires_unix": payload["expires_unix"],
        "issued_event_hash": event["event_hash"],
        "observation_fingerprint": observation["fingerprint"],
        "slot": spec.slot,
        "token": token,
        "version": 1,
    }
    try:
        atomic_write_secret_json(output_path, document)
        failpoint("start-capability:file-written")
        publish_capability(connection, event, payload, document, output_path)
    except Exception:
        append_event(
            connection,
            "CAPABILITY_REVOKED",
            spec.slot,
            f"capability-revoked:{capability_id}:write-failed",
            {
                "capability_id": capability_id,
                "reason": "capability-file-write-failed",
            },
        )
        raise
    return capability_id


def read_capability_file(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FleetdError(f"capability file is missing or a symlink: {path}")
    if path.stat().st_mode & 0o077:
        raise FleetdError(f"capability file permissions are not private: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetdError(f"cannot read capability file {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise FleetdError(f"unsupported capability document: {path}")
    return value


def load_capability_documents(paths: list[str]) -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.absolute()
        document = read_capability_file(path)
        slot = document.get("slot")
        if not isinstance(slot, str) or not slot:
            raise FleetdError(f"capability file has no slot: {path}")
        if slot in documents:
            raise FleetdError(f"multiple capability files target slot {slot}")
        document["_path"] = str(path)
        documents[slot] = document
    return documents


def consume_start_capability(
    connection: sqlite3.Connection,
    spec: LaneSpec,
    observation: dict[str, Any],
    document: dict[str, Any],
) -> str:
    capability_id = document.get("capability_id")
    token = document.get("token")
    if not isinstance(capability_id, str) or not isinstance(token, str):
        raise FleetdError("capability document omits its identity or secret")
    events = capability_events(connection, capability_id)
    issued = [payload for kind, payload in events if kind == "CAPABILITY_ISSUED"]
    if len(issued) != 1:
        raise FleetdError(f"capability {capability_id} has no unique issuance event")
    if any(kind == "CAPABILITY_CONSUMED" for kind, _ in events):
        raise FleetdError(f"capability {capability_id} was already consumed")
    if any(kind == "CAPABILITY_REVOKED" for kind, _ in events):
        raise FleetdError(f"capability {capability_id} was revoked")
    published = [
        payload for kind, payload in events if kind == "CAPABILITY_PUBLISHED"
    ]
    authority = issued[0]
    required = {
        "action": "start",
        "capability_id": capability_id,
        "desired_hash": spec.desired_hash,
        "observation_fingerprint": observation["fingerprint"],
        "slot": spec.slot,
    }
    if any(authority.get(key) != value for key, value in required.items()):
        raise FleetdError(f"capability {capability_id} does not authorize this transition")
    if any(document.get(key) != value for key, value in required.items()):
        raise FleetdError(f"capability file {capability_id} binding was altered")
    if authority.get("capability_path") is not None:
        if len(published) != 1:
            raise FleetdError(f"capability {capability_id} was not uniquely published")
    if int(authority.get("expires_unix", 0)) < int(time.time()):
        raise FleetdError(f"capability {capability_id} expired")
    observed_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(observed_hash, str(authority.get("token_hash", ""))):
        raise FleetdError(f"capability {capability_id} secret does not match its issuance")
    if authority.get("capability_path") is not None and (
        published[0].get("document_sha256") != capability_document_digest(document)
    ):
        raise FleetdError(f"capability {capability_id} publication digest drifted")
    append_event(
        connection,
        "CAPABILITY_CONSUMED",
        spec.slot,
        f"capability-consumed:{capability_id}",
        {
            "action": "start",
            "capability_id": capability_id,
            "desired_hash": spec.desired_hash,
            "observation_fingerprint": observation["fingerprint"],
            "slot": spec.slot,
        },
    )
    failpoint("start-action:authority-consumed")
    return capability_id


def file_receipt(raw_path: str) -> dict[str, Any]:
    path = absolute_without_symlink_resolution(raw_path)
    if path.is_symlink() or not path.is_file():
        raise FleetdError(f"evidence file is missing or a symlink: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size": size}


def events_with_identity(
    connection: sqlite3.Connection,
    event_types: tuple[str, ...],
    identity_key: str,
    identity: str,
) -> list[tuple[sqlite3.Row, dict[str, Any]]]:
    placeholders = ",".join("?" for _ in event_types)
    rows = connection.execute(
        f"SELECT * FROM events WHERE event_type IN ({placeholders}) ORDER BY seq",
        event_types,
    ).fetchall()
    matched: list[tuple[sqlite3.Row, dict[str, Any]]] = []
    for row in rows:
        payload = json.loads(row["payload"])
        if payload.get(identity_key) == identity:
            matched.append((row, payload))
    return matched


def active_start_authority(
    connection: sqlite3.Connection,
    slot: str,
    observation: dict[str, Any],
) -> tuple[sqlite3.Row, dict[str, Any]]:
    matches: list[tuple[sqlite3.Row, dict[str, Any]]] = []
    for row in connection.execute(
        """
        SELECT * FROM events
        WHERE event_type = 'ACTION_COMMITTED' AND slot = ?
        ORDER BY seq
        """,
        (slot,),
    ):
        payload = json.loads(row["payload"])
        if (
            payload.get("action") == "start"
            and payload.get("status") == "committed"
            and payload.get("generation") == observation.get("generation")
            and payload.get("argv_digest") == observation.get("argv_digest")
            and payload.get("capability_id")
            == observation.get("start_capability_id")
        ):
            matches.append((row, payload))
    if len(matches) != 1:
        raise FleetdError(
            f"checkpoint requires one capability-bound active generation: {slot}"
        )
    capability_id = matches[0][1].get("capability_id")
    if not isinstance(capability_id, str):
        raise FleetdError("active start receipt omits its capability identity")
    consumed = [
        payload
        for kind, payload in capability_events(connection, capability_id)
        if kind == "CAPABILITY_CONSUMED" and payload.get("action") == "start"
    ]
    if len(consumed) != 1:
        raise FleetdError(
            f"active generation has no unique consumed authority: {capability_id}"
        )
    return matches[0]


def create_checkpoint(
    connection: sqlite3.Connection,
    specs: list[LaneSpec],
    slot: str,
    checkpoint_kind: str,
    summary_file: str,
    evidence_files: list[str],
) -> str:
    verify_state(connection)
    verify_config_coverage(connection, specs)
    matches = [spec for spec in specs if spec.slot == slot]
    if len(matches) != 1:
        raise FleetdError(f"fleet config has no unique slot: {slot}")
    spec = matches[0]
    declare_desired(connection, spec)
    observation = observe_spec(spec)
    observation_event, _ = record_observation(connection, spec, observation)
    if observation["state"] != "active" or not observation.get("argv_digest"):
        raise FleetdError(
            f"checkpoint requires an active argv-attested slot: {slot}"
        )
    start_event, start_receipt = active_start_authority(
        connection, slot, observation
    )
    summary = file_receipt(summary_file)
    evidence = [file_receipt(path) for path in evidence_files]
    if not evidence:
        raise FleetdError("checkpoint requires at least one evidence file")
    checkpoint_id = f"chk-{uuid.uuid4()}"
    payload = {
        "argv_digest": observation["argv_digest"],
        "checkpoint_id": checkpoint_id,
        "checkpoint_kind": checkpoint_kind,
        "evidence": evidence,
        "generation": observation["generation"],
        "observation_seq": int(observation_event["seq"]),
        "slot": slot,
        "start_action_event_hash": start_event["event_hash"],
        "start_action_event_seq": start_event["seq"],
        "start_action_id": start_receipt["action_id"],
        "start_capability_id": start_receipt["capability_id"],
        "state": "draft",
        "summary": summary,
    }
    event, _ = append_event(
        connection,
        "CHECKPOINT_DRAFTED",
        slot,
        f"checkpoint-drafted:{checkpoint_id}",
        payload,
    )
    print(
        "FLEET_CHECKPOINT_DRAFTED "
        f"checkpoint_id={checkpoint_id} kind={checkpoint_kind} slot={slot} "
        f"event_seq={event['seq']} evidence={len(evidence)}"
    )
    return checkpoint_id


def checkpoint_evidence_status(
    draft: dict[str, Any],
) -> tuple[str, list[str]]:
    expected_receipts = [draft["summary"], *draft["evidence"]]
    mismatches: list[str] = []
    for expected in expected_receipts:
        try:
            actual = file_receipt(expected["path"])
        except FleetdError:
            mismatches.append(expected["path"])
            continue
        if actual != expected:
            mismatches.append(expected["path"])
    return digest_json(expected_receipts), mismatches


def refuse_checkpoint_evidence(
    connection: sqlite3.Connection,
    draft: dict[str, Any],
    checkpoint_id: str,
    mismatches: list[str],
    reason: str,
) -> None:
    append_event(
        connection,
        "CHECKPOINT_REFUSED",
        draft["slot"],
        transition_key(
            connection,
            "checkpoint-refused",
            draft["slot"],
            {"checkpoint_id": checkpoint_id, "reason": reason},
        ),
        {
            "checkpoint_id": checkpoint_id,
            "mismatch_count": len(mismatches),
            "reason": reason,
        },
    )


def verify_checkpoint(connection: sqlite3.Connection, checkpoint_id: str) -> int:
    verify_state(connection)
    events = events_with_identity(
        connection,
        ("CHECKPOINT_DRAFTED", "CHECKPOINT_VERIFIED", "CHECKPOINT_REFUSED"),
        "checkpoint_id",
        checkpoint_id,
    )
    drafts = [
        (row, payload)
        for row, payload in events
        if row["event_type"] == "CHECKPOINT_DRAFTED"
    ]
    if len(drafts) != 1:
        raise FleetdError(f"checkpoint has no unique draft: {checkpoint_id}")
    if any(row["event_type"] == "CHECKPOINT_VERIFIED" for row, _ in events):
        raise FleetdError(f"checkpoint was already verified: {checkpoint_id}")
    draft_event, draft = drafts[0]
    evidence_set_digest, mismatches = checkpoint_evidence_status(draft)
    if mismatches:
        refuse_checkpoint_evidence(
            connection, draft, checkpoint_id, mismatches, "evidence-drift"
        )
        raise FleetdError(
            f"checkpoint evidence drifted: {checkpoint_id} mismatches={len(mismatches)}"
        )
    event, _ = append_event(
        connection,
        "CHECKPOINT_VERIFIED",
        draft["slot"],
        f"checkpoint-verified:{checkpoint_id}",
        {
            "checkpoint_id": checkpoint_id,
            "draft_event_hash": draft_event["event_hash"],
            "draft_event_seq": draft_event["seq"],
            "evidence_set_digest": evidence_set_digest,
            "state": "verified",
        },
    )
    print(
        "FLEET_CHECKPOINT_VERIFIED "
        f"checkpoint_id={checkpoint_id} event_seq={event['seq']} "
        f"evidence_set_digest={evidence_set_digest}"
    )
    return 0


def verified_checkpoint(
    connection: sqlite3.Connection, checkpoint_id: str
) -> tuple[sqlite3.Row, dict[str, Any], sqlite3.Row, dict[str, Any]]:
    events = events_with_identity(
        connection,
        ("CHECKPOINT_DRAFTED", "CHECKPOINT_VERIFIED"),
        "checkpoint_id",
        checkpoint_id,
    )
    drafts = [
        (row, payload)
        for row, payload in events
        if row["event_type"] == "CHECKPOINT_DRAFTED"
    ]
    verified = [
        (row, payload)
        for row, payload in events
        if row["event_type"] == "CHECKPOINT_VERIFIED"
    ]
    if len(drafts) != 1 or len(verified) != 1:
        raise FleetdError(
            f"handoff requires a uniquely verified checkpoint: {checkpoint_id}"
        )
    return drafts[0][0], drafts[0][1], verified[0][0], verified[0][1]


def ensure_handoff_capability(
    connection: sqlite3.Connection,
    prepared: sqlite3.Row,
    prepared_payload: dict[str, Any],
    output_path: Path,
    ttl_seconds: int,
) -> str:
    recover_unpublished_capabilities(connection)
    handoff_id = str(prepared_payload["handoff_id"])
    for issued_event in connection.execute(
        "SELECT * FROM events WHERE event_type = 'CAPABILITY_ISSUED' ORDER BY seq"
    ):
        authority = json.loads(issued_event["payload"])
        if (
            authority.get("action") != "accept-handoff"
            or authority.get("handoff_id") != handoff_id
        ):
            continue
        capability_id = str(authority.get("capability_id", ""))
        events = capability_events(connection, capability_id)
        kinds = [kind for kind, _ in events]
        if "CAPABILITY_REVOKED" in kinds:
            continue
        if "CAPABILITY_PUBLISHED" in kinds or "CAPABILITY_CONSUMED" in kinds:
            if authority.get("capability_path") != str(output_path):
                raise FleetdError(
                    "prepared handoff capability path changed during recovery"
                )
            return capability_id
    capability_id = f"cap-{uuid.uuid4()}"
    token = secrets.token_urlsafe(48)
    issued_unix = int(time.time())
    issued_payload = {
        "action": "accept-handoff",
        "capability_id": capability_id,
        "capability_path": str(output_path),
        "checkpoint_id": prepared_payload["checkpoint_id"],
        "evidence_set_digest": prepared_payload["evidence_set_digest"],
        "expires_unix": issued_unix + ttl_seconds,
        "handoff_id": handoff_id,
        "issued_unix": issued_unix,
        "prepared_event_hash": prepared["event_hash"],
        "prepared_event_seq": prepared["seq"],
        "slot": prepared["slot"],
        "to_agent": prepared_payload["to_agent"],
        "to_lane": prepared_payload["to_lane"],
        "token_hash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
    }
    issued_event, _ = append_event(
        connection,
        "CAPABILITY_ISSUED",
        prepared["slot"],
        f"capability-issued:{capability_id}",
        issued_payload,
    )
    failpoint("handoff-prepare:authority-issued")
    document = {
        key: value
        for key, value in issued_payload.items()
        if key not in {"issued_unix", "token_hash"}
    }
    document.update(
        {
            "issued_event_hash": issued_event["event_hash"],
            "token": token,
            "version": 1,
        }
    )
    try:
        atomic_write_secret_json(output_path, document)
        failpoint("handoff-prepare:file-written")
        publish_capability(
            connection, issued_event, issued_payload, document, output_path
        )
    except Exception:
        append_event(
            connection,
            "CAPABILITY_REVOKED",
            prepared["slot"],
            f"capability-revoked:{capability_id}:write-failed",
            {
                "capability_id": capability_id,
                "reason": "capability-file-write-failed",
            },
        )
        raise
    return capability_id


def prepare_handoff(
    connection: sqlite3.Connection,
    checkpoint_id: str,
    to_agent: str,
    to_lane: str,
    output_path: Path,
    ttl_seconds: int,
) -> str:
    verify_state(connection)
    if ttl_seconds < 1 or ttl_seconds > 86400:
        raise FleetdError("handoff capability TTL must be between 1 and 86400 seconds")
    _, draft, verified_event, verified = verified_checkpoint(
        connection, checkpoint_id
    )
    evidence_set_digest, mismatches = checkpoint_evidence_status(draft)
    if mismatches or verified.get("evidence_set_digest") != evidence_set_digest:
        refuse_checkpoint_evidence(
            connection,
            draft,
            checkpoint_id,
            mismatches,
            "evidence-drift-before-handoff",
        )
        raise FleetdError(
            f"checkpoint evidence drifted before handoff: {checkpoint_id} "
            f"mismatches={len(mismatches)}"
        )
    existing = events_with_identity(
        connection,
        ("HANDOFF_PREPARED", "HANDOFF_ACCEPTED"),
        "checkpoint_id",
        checkpoint_id,
    )
    if any(row["event_type"] == "HANDOFF_ACCEPTED" for row, _ in existing):
        raise FleetdError(f"checkpoint handoff was already accepted: {checkpoint_id}")
    prepared_existing = [
        (row, payload)
        for row, payload in existing
        if row["event_type"] == "HANDOFF_PREPARED"
    ]
    if prepared_existing:
        if len(prepared_existing) != 1:
            raise FleetdError(f"checkpoint has non-unique handoff: {checkpoint_id}")
        prepared, prepared_payload = prepared_existing[0]
        required = {
            "capability_path": str(output_path),
            "checkpoint_id": checkpoint_id,
            "evidence_set_digest": evidence_set_digest,
            "to_agent": to_agent,
            "to_lane": to_lane,
        }
        if any(prepared_payload.get(key) != value for key, value in required.items()):
            raise FleetdError("prepared handoff recovery binding changed")
        capability_id = ensure_handoff_capability(
            connection, prepared, prepared_payload, output_path, ttl_seconds
        )
        print(
            "FLEET_HANDOFF_RECOVERED "
            f"handoff_id={prepared_payload['handoff_id']} "
            f"checkpoint_id={checkpoint_id} capability_id={capability_id}"
        )
        return str(prepared_payload["handoff_id"])
    handoff_id = f"handoff-{uuid.uuid4()}"
    prepared, _ = append_event(
        connection,
        "HANDOFF_PREPARED",
        draft["slot"],
        f"handoff-prepared:{handoff_id}",
        {
            "capability_path": str(output_path),
            "checkpoint_id": checkpoint_id,
            "evidence_set_digest": evidence_set_digest,
            "handoff_id": handoff_id,
            "state": "prepared",
            "to_agent": to_agent,
            "to_lane": to_lane,
            "verified_checkpoint_event_hash": verified_event["event_hash"],
            "verified_checkpoint_event_seq": verified_event["seq"],
        },
    )
    failpoint("handoff-prepare:prepared")
    prepared_payload = json.loads(prepared["payload"])
    capability_id = ensure_handoff_capability(
        connection, prepared, prepared_payload, output_path, ttl_seconds
    )
    print(
        "FLEET_HANDOFF_PREPARED "
        f"handoff_id={handoff_id} checkpoint_id={checkpoint_id} "
        f"capability_id={capability_id} event_seq={prepared['seq']}"
    )
    return handoff_id


def validate_handoff_capability(
    connection: sqlite3.Connection,
    handoff_id: str,
    agent: str,
    lane: str,
    evidence_set_digest: str,
    document: dict[str, Any],
) -> tuple[str, dict[str, Any], bool]:
    capability_id = document.get("capability_id")
    token = document.get("token")
    if not isinstance(capability_id, str) or not isinstance(token, str):
        raise FleetdError("handoff capability omits its identity or secret")
    events = capability_events(connection, capability_id)
    issued = [payload for kind, payload in events if kind == "CAPABILITY_ISSUED"]
    if len(issued) != 1:
        raise FleetdError(f"handoff capability has no unique issuance: {capability_id}")
    consumed = [
        payload for kind, payload in events if kind == "CAPABILITY_CONSUMED"
    ]
    if len(consumed) > 1:
        raise FleetdError(f"handoff capability was consumed more than once: {capability_id}")
    if any(kind == "CAPABILITY_REVOKED" for kind, _ in events):
        raise FleetdError(f"handoff capability was revoked: {capability_id}")
    authority = issued[0]
    published = [
        payload for kind, payload in events if kind == "CAPABILITY_PUBLISHED"
    ]
    required = {
        "action": "accept-handoff",
        "capability_id": capability_id,
        "evidence_set_digest": evidence_set_digest,
        "handoff_id": handoff_id,
        "to_agent": agent,
        "to_lane": lane,
    }
    if any(authority.get(key) != value for key, value in required.items()):
        raise FleetdError("handoff capability does not authorize this recipient")
    if any(document.get(key) != value for key, value in required.items()):
        raise FleetdError("handoff capability binding was altered")
    if len(published) != 1:
        raise FleetdError(f"handoff capability was not uniquely published: {capability_id}")
    if published[0].get("document_sha256") != capability_document_digest(document):
        raise FleetdError("handoff capability publication digest drifted")
    if not consumed and int(authority.get("expires_unix", 0)) < int(time.time()):
        raise FleetdError(f"handoff capability expired: {capability_id}")
    observed_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(observed_hash, str(authority.get("token_hash", ""))):
        raise FleetdError("handoff capability secret does not match its issuance")
    if consumed:
        if any(consumed[0].get(key) != value for key, value in required.items()):
            raise FleetdError("consumed handoff capability binding drifted")
    return capability_id, authority, bool(consumed)


def consume_handoff_capability(
    connection: sqlite3.Connection,
    capability_id: str,
    authority: dict[str, Any],
    handoff_id: str,
    agent: str,
    lane: str,
    evidence_set_digest: str,
) -> None:
    append_event(
        connection,
        "CAPABILITY_CONSUMED",
        authority["slot"],
        f"capability-consumed:{capability_id}",
        {
            "action": "accept-handoff",
            "capability_id": capability_id,
            "evidence_set_digest": evidence_set_digest,
            "handoff_id": handoff_id,
            "slot": authority["slot"],
            "to_agent": agent,
            "to_lane": lane,
        },
    )
    failpoint("handoff-accept:authority-consumed")


def accept_handoff(
    connection: sqlite3.Connection,
    handoff_id: str,
    agent: str,
    lane: str,
    capability_path: Path,
    public_key: Path,
    anchor_dir: Path,
) -> int:
    verify_state(connection)
    prepared_events = events_with_identity(
        connection,
        ("HANDOFF_PREPARED", "HANDOFF_ACCEPTED"),
        "handoff_id",
        handoff_id,
    )
    prepared = [
        (row, payload)
        for row, payload in prepared_events
        if row["event_type"] == "HANDOFF_PREPARED"
    ]
    if len(prepared) != 1:
        raise FleetdError(f"handoff has no unique prepared state: {handoff_id}")
    if any(row["event_type"] == "HANDOFF_ACCEPTED" for row, _ in prepared_events):
        raise FleetdError(f"handoff was already accepted: {handoff_id}")
    prepared_event, prepared_payload = prepared[0]
    if prepared_payload.get("to_agent") != agent or prepared_payload.get("to_lane") != lane:
        raise FleetdError("handoff recipient does not match the prepared state")
    checkpoint_id = prepared_payload.get("checkpoint_id")
    if not isinstance(checkpoint_id, str):
        raise FleetdError("prepared handoff omits its checkpoint identity")
    _, draft, verified_event, verified_payload = verified_checkpoint(
        connection, checkpoint_id
    )
    evidence_set_digest, mismatches = checkpoint_evidence_status(draft)
    evidence_bound = (
        verified_payload.get("evidence_set_digest") == evidence_set_digest
        and prepared_payload.get("evidence_set_digest") == evidence_set_digest
        and prepared_payload.get("verified_checkpoint_event_hash")
        == verified_event["event_hash"]
    )
    if mismatches or not evidence_bound:
        refuse_checkpoint_evidence(
            connection,
            draft,
            checkpoint_id,
            mismatches,
            "evidence-drift-before-accept",
        )
        raise FleetdError(
            f"checkpoint evidence drifted before handoff acceptance: {checkpoint_id} "
            f"mismatches={len(mismatches)}"
        )
    _, latest_anchor = verify_anchor_directory(connection, anchor_dir, public_key)
    covered_event = max(int(prepared_event["seq"]), int(verified_event["seq"]))
    if (
        latest_anchor is None
        or latest_anchor["payload"]["event_count"] < covered_event
    ):
        raise FleetdError(
            f"handoff preparation is not covered by a signed anchor: {handoff_id}"
        )
    document = read_capability_file(capability_path)
    capability_id, authority, already_consumed = validate_handoff_capability(
        connection,
        handoff_id,
        agent,
        lane,
        evidence_set_digest,
        document,
    )
    request_payload = {
        "anchor_event_count": latest_anchor["payload"]["event_count"],
        "anchor_tail_hash": latest_anchor["payload"]["tail_hash"],
        "capability_id": capability_id,
        "checkpoint_id": checkpoint_id,
        "evidence_set_digest": evidence_set_digest,
        "handoff_id": handoff_id,
        "state": "requested",
        "to_agent": agent,
        "to_lane": lane,
    }
    request_events = events_with_identity(
        connection,
        ("HANDOFF_ACCEPTANCE_REQUESTED",),
        "handoff_id",
        handoff_id,
    )
    if request_events:
        if len(request_events) != 1:
            raise FleetdError("handoff has non-unique acceptance request")
        _, prior_request = request_events[0]
        stable = {
            key: request_payload[key]
            for key in (
                "capability_id",
                "checkpoint_id",
                "evidence_set_digest",
                "handoff_id",
                "state",
                "to_agent",
                "to_lane",
            )
        }
        if any(prior_request.get(key) != value for key, value in stable.items()):
            raise FleetdError("handoff acceptance request binding drifted")
        covered_count = int(prior_request.get("anchor_event_count", 0))
        covered_row = connection.execute(
            "SELECT event_hash FROM events WHERE seq = ?", (covered_count,)
        ).fetchone()
        if (
            covered_row is None
            or covered_row["event_hash"] != prior_request.get("anchor_tail_hash")
            or covered_count > int(latest_anchor["payload"]["event_count"])
        ):
            raise FleetdError("recorded handoff anchor prefix no longer verifies")
        request_payload = prior_request
    else:
        append_event(
            connection,
            "HANDOFF_ACCEPTANCE_REQUESTED",
            authority["slot"],
            f"handoff-acceptance-requested:{handoff_id}",
            request_payload,
        )
    failpoint("handoff-accept:requested")
    if not already_consumed:
        consume_handoff_capability(
            connection,
            capability_id,
            authority,
            handoff_id,
            agent,
            lane,
            evidence_set_digest,
        )
    event, _ = append_event(
        connection,
        "HANDOFF_ACCEPTED",
        authority["slot"],
        f"handoff-accepted:{handoff_id}",
        {
            "anchor_event_count": request_payload["anchor_event_count"],
            "anchor_tail_hash": request_payload["anchor_tail_hash"],
            "capability_id": capability_id,
            "checkpoint_id": checkpoint_id,
            "evidence_set_digest": evidence_set_digest,
            "handoff_id": handoff_id,
            "state": "accepted",
            "to_agent": agent,
            "to_lane": lane,
        },
    )
    print(
        "FLEET_HANDOFF_ACCEPTED "
        f"handoff_id={handoff_id} checkpoint_id={authority['checkpoint_id']} "
        f"event_seq={event['seq']} anchor_events={request_payload['anchor_event_count']}"
    )
    return 0


def launch_arguments(spec: LaneSpec, capability_id: str) -> list[str]:
    base = [str(fleet_agent_command())]
    if spec.kind:
        if spec.home is None:
            raise FleetdError(f"lane {spec.slot} kind has no home")
        return [
            *base,
            "launch-kind",
            "--slot",
            spec.slot,
            "--kind",
            spec.kind,
            "--home",
            str(spec.home),
            "--cwd",
            str(spec.cwd),
            "--start-capability-id",
            capability_id,
            "--no-attach",
        ]
    if spec.agent is None or spec.session_id is None:
        raise FleetdError(f"lane {spec.slot} command has no stable identity")
    arguments = [
        *base,
        "launch",
        "--slot",
        spec.slot,
        "--agent",
        spec.agent,
        "--session-id",
        spec.session_id,
        "--identity",
        spec.identity or "exact",
        "--cwd",
        str(spec.cwd),
        "--start-capability-id",
        capability_id,
        "--no-attach",
    ]
    if spec.lane:
        arguments.extend(["--lane", spec.lane])
    return [*arguments, "--", *spec.command]


def start_action_id(
    slot: str,
    desired_hash: str,
    observation_fingerprint: str,
    capability_id: str,
) -> str:
    return digest_json(
        {
            "action": "start",
            "capability_id": capability_id,
            "desired_hash": desired_hash,
            "observation": observation_fingerprint,
            "slot": slot,
        }
    )


def apply_start(
    connection: sqlite3.Connection,
    spec: LaneSpec,
    observation: dict[str, Any],
    observation_seq: int,
    capability_id: str,
) -> bool:
    action_id = start_action_id(
        spec.slot,
        spec.desired_hash,
        observation["fingerprint"],
        capability_id,
    )
    requested = {
        "action": "start",
        "action_id": action_id,
        "capability_id": capability_id,
        "desired_hash": spec.desired_hash,
        "observation_seq": observation_seq,
    }
    append_event(
        connection,
        "ACTION_REQUESTED",
        spec.slot,
        f"action-request:{action_id}",
        requested,
    )
    failpoint("start-action:requested")
    arguments = launch_arguments(spec, capability_id)
    try:
        result = subprocess.run(
            arguments,
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
        return_code = result.returncode
        output = (result.stdout or result.stderr).strip()
    except subprocess.TimeoutExpired as exc:
        return_code = 124
        timeout_output = exc.stdout or exc.stderr or "fleet launch timed out"
        output = (
            timeout_output.decode("utf-8", errors="replace")
            if isinstance(timeout_output, bytes)
            else timeout_output
        ).strip()
    failpoint("start-action:launched")
    committed_observation = observe_spec(spec) if return_code == 0 else None
    if committed_observation is not None and (
        committed_observation.get("state") != "active"
        or committed_observation.get("start_capability_id") != capability_id
    ):
        return_code = 125
        output = f"{output}\nstart capability did not bind the active generation".strip()
        committed_observation = None
    outcome = {
        "action": "start",
        "action_id": action_id,
        "capability_id": capability_id,
        "exit_code": return_code,
        "output_digest": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "status": "committed" if return_code == 0 else "failed",
    }
    if committed_observation is not None:
        outcome.update(
            {
                "argv_digest": committed_observation.get("argv_digest"),
                "generation": committed_observation.get("generation"),
                "observed_state": committed_observation.get("state"),
            }
        )
    append_event(
        connection,
        "ACTION_COMMITTED" if return_code == 0 else "ACTION_FAILED",
        spec.slot,
        f"action-result:{action_id}:{return_code}:{outcome['output_digest']}",
        outcome,
    )
    print(
        "FLEET_ACTION "
        f"slot={spec.slot} action=start status={outcome['status']} "
        f"action_id={action_id[:16]} exit_code={return_code}"
    )
    return return_code == 0


def recover_start_actions(
    connection: sqlite3.Connection, specs: list[LaneSpec]
) -> int:
    specs_by_slot = {spec.slot: spec for spec in specs}
    issued: dict[str, tuple[sqlite3.Row, dict[str, Any]]] = {}
    consumed: dict[str, tuple[sqlite3.Row, dict[str, Any]]] = {}
    for row in connection.execute(
        """
        SELECT * FROM events
        WHERE event_type IN ('CAPABILITY_ISSUED', 'CAPABILITY_CONSUMED')
        ORDER BY seq
        """
    ):
        payload = json.loads(row["payload"])
        capability_id = payload.get("capability_id")
        if not isinstance(capability_id, str) or payload.get("action") != "start":
            continue
        if row["event_type"] == "CAPABILITY_ISSUED":
            issued[capability_id] = (row, payload)
        else:
            consumed[capability_id] = (row, payload)
    failed = 0
    for capability_id, (_, consumption) in consumed.items():
        issuance = issued.get(capability_id)
        if issuance is None:
            raise FleetdError(
                f"consumed start capability has no issuance: {capability_id}"
            )
        _, authority = issuance
        slot = consumption.get("slot")
        spec = specs_by_slot.get(str(slot))
        if spec is None:
            raise FleetdError(
                f"pending start action has no configured slot: {capability_id}"
            )
        fingerprint = str(consumption.get("observation_fingerprint", ""))
        authority_desired_hash = str(authority.get("desired_hash", ""))
        action_id = start_action_id(
            spec.slot, authority_desired_hash, fingerprint, capability_id
        )
        action_rows = events_with_identity(
            connection,
            ("ACTION_REQUESTED", "ACTION_COMMITTED", "ACTION_FAILED"),
            "action_id",
            action_id,
        )
        if any(
            row["event_type"] in {"ACTION_COMMITTED", "ACTION_FAILED"}
            for row, _ in action_rows
        ):
            continue
        if authority_desired_hash != spec.desired_hash:
            raise FleetdError(
                f"pending start action desired state drifted: {capability_id}"
            )
        observation = observe_spec(spec)
        if any(row["event_type"] == "ACTION_REQUESTED" for row, _ in action_rows) and (
            observation.get("state") == "active"
            and observation.get("argv_digest")
            and observation.get("generation")
            and observation.get("start_capability_id") == capability_id
        ):
            outcome = {
                "action": "start",
                "action_id": action_id,
                "argv_digest": observation["argv_digest"],
                "capability_id": capability_id,
                "exit_code": 0,
                "generation": observation["generation"],
                "observed_state": "active",
                "output_digest": digest_json(observation),
                "recovered_after_crash": True,
                "status": "committed",
            }
            append_event(
                connection,
                "ACTION_COMMITTED",
                spec.slot,
                f"action-result:{action_id}:recovered:{outcome['output_digest']}",
                outcome,
            )
            print(
                "FLEET_ACTION_RECOVERED "
                f"slot={spec.slot} action=start action_id={action_id[:16]}"
            )
            continue
        observation_seq = int(authority.get("observation_seq", 0))
        recovery_observation = {
            "fingerprint": fingerprint,
        }
        if not apply_start(
            connection,
            spec,
            recovery_observation,
            observation_seq,
            capability_id,
        ):
            failed += 1
    return failed


def cycle(
    connection: sqlite3.Connection,
    specs: list[LaneSpec],
    *,
    apply: bool,
    capabilities: dict[str, dict[str, Any]] | None = None,
    emit: bool = True,
) -> int:
    verify_state(connection)
    recover_unpublished_capabilities(connection)
    verify_config_coverage(connection, specs)
    blocked = 0
    failed = recover_start_actions(connection, specs) if apply else 0
    capabilities = capabilities or {}
    used_capabilities: set[str] = set()
    for spec in specs:
        declare_desired(connection, spec)
        observation = observe_spec(spec)
        observation_event, inserted = record_observation(connection, spec, observation)
        decision, reason = decide(
            connection, spec, observation, int(observation_event["seq"])
        )
        record_plan(
            connection,
            spec,
            observation,
            int(observation_event["seq"]),
            decision,
            reason,
        )
        if emit:
            print(
                "FLEET_RECONCILE "
                f"slot={spec.slot} observed={observation['state']} "
                f"decision={decision} reason={reason} "
                f"observation={'new' if inserted else 'deduplicated'}"
            )
        if decision == "blocked":
            blocked += 1
        elif decision == "start" and apply:
            document = capabilities.get(spec.slot)
            if document is None:
                failed += 1
                append_event(
                    connection,
                    "ACTION_REFUSED",
                    spec.slot,
                    transition_key(
                        connection,
                        "action-refused",
                        spec.slot,
                        {
                            "action": "start",
                            "observation_fingerprint": observation["fingerprint"],
                            "reason": "linear-capability-required",
                        },
                    ),
                    {
                        "action": "start",
                        "observation_fingerprint": observation["fingerprint"],
                        "reason": "linear-capability-required",
                    },
                )
                print(
                    "FLEET_ACTION "
                    f"slot={spec.slot} action=start status=refused "
                    "reason=linear-capability-required"
                )
                continue
            used_capabilities.add(spec.slot)
            try:
                capability_id = consume_start_capability(
                    connection, spec, observation, document
                )
            except FleetdError as exc:
                failed += 1
                refusal_reason = slug(str(exc), limit=160)
                refusal_payload = {
                    "action": "start",
                    "capability_id": str(document.get("capability_id", "unknown")),
                    "observation_fingerprint": observation["fingerprint"],
                    "reason": refusal_reason,
                }
                append_event(
                    connection,
                    "CAPABILITY_REFUSED",
                    spec.slot,
                    transition_key(
                        connection,
                        "capability-refused",
                        spec.slot,
                        refusal_payload,
                    ),
                    refusal_payload,
                )
                print(
                    "FLEET_ACTION "
                    f"slot={spec.slot} action=start status=refused "
                    f"reason={refusal_reason}"
                )
                continue
            if apply_start(
                connection,
                spec,
                observation,
                int(observation_event["seq"]),
                capability_id,
            ):
                after = observe_spec(spec)
                after_event, _ = record_observation(connection, spec, after)
                after_decision, after_reason = decide(
                    connection, spec, after, int(after_event["seq"])
                )
                record_plan(
                    connection,
                    spec,
                    after,
                    int(after_event["seq"]),
                    after_decision,
                    after_reason,
                )
                if after["state"] != "active":
                    failed += 1
            else:
                failed += 1
    unused = sorted(set(capabilities) - used_capabilities)
    for slot in unused:
        failed += 1
        print(
            "FLEET_ACTION "
            f"slot={slot} action=start status=refused reason=capability-not-applicable"
        )
    return 2 if failed else (1 if blocked else 0)


def observe_cycle(
    connection: sqlite3.Connection, specs: list[LaneSpec]
) -> int:
    verify_state(connection)
    verify_config_coverage(connection, specs)
    drifted = 0
    for spec in specs:
        declare_desired(connection, spec)
        observation = observe_spec(spec)
        _, inserted = record_observation(connection, spec, observation)
        if observation["state"] == "drifted":
            drifted += 1
        print(
            "FLEET_OBSERVE "
            f"slot={spec.slot} state={observation['state']} "
            f"reason={observation['reason']} "
            f"observation={'new' if inserted else 'deduplicated'}"
        )
    return 1 if drifted else 0


def authorize_start(
    connection: sqlite3.Connection,
    specs: list[LaneSpec],
    slot: str,
    output_path: Path,
    ttl_seconds: int,
) -> int:
    verify_state(connection)
    recover_unpublished_capabilities(connection)
    verify_config_coverage(connection, specs)
    matches = [spec for spec in specs if spec.slot == slot]
    if len(matches) != 1:
        raise FleetdError(f"fleet config has no unique slot: {slot}")
    spec = matches[0]
    declare_desired(connection, spec)
    observation = observe_spec(spec)
    observation_event, _ = record_observation(connection, spec, observation)
    decision, reason = decide(
        connection, spec, observation, int(observation_event["seq"])
    )
    record_plan(
        connection,
        spec,
        observation,
        int(observation_event["seq"]),
        decision,
        reason,
    )
    if decision != "start":
        raise FleetdError(
            f"slot {slot} is not authorizable: decision={decision} reason={reason}"
        )
    capability_id = issue_start_capability(
        connection,
        spec,
        observation,
        int(observation_event["seq"]),
        output_path,
        ttl_seconds,
    )
    print(
        "FLEET_CAPABILITY_ISSUED "
        f"capability_id={capability_id} slot={slot} action=start "
        f"expires_unix={int(time.time()) + ttl_seconds} path={output_path}"
    )
    return 0


def print_events(connection: sqlite3.Connection, slot: str | None, limit: int) -> None:
    if slot:
        rows = connection.execute(
            "SELECT * FROM events WHERE slot = ? ORDER BY seq DESC LIMIT ?",
            (slot, limit),
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM events ORDER BY seq DESC LIMIT ?", (limit,)
        ).fetchall()
    for row in reversed(rows):
        print(
            "FLEET_EVENT "
            f"seq={row['seq']} id={row['event_id']} type={row['event_type']} "
            f"slot={row['slot']} utc={row['occurred_utc']} hash={row['event_hash']}"
        )


def print_status(connection: sqlite3.Connection, slot: str | None) -> int:
    if slot:
        rows = connection.execute(
            "SELECT * FROM slot_view WHERE slot = ?", (slot,)
        ).fetchall()
    else:
        rows = connection.execute("SELECT * FROM slot_view ORDER BY slot").fetchall()
    blocked = 0
    for row in rows:
        observed = json.loads(row["observed_json"]) if row["observed_json"] else {}
        decision = row["decision"] or "unknown"
        if decision == "blocked":
            blocked += 1
        print(
            "FLEET_STATUS "
            f"slot={row['slot']} observed={observed.get('state', 'unknown')} "
            f"generation={observed.get('generation') or '-'} "
            f"decision={decision} reason={row['reason'] or '-'} "
            f"last_event_seq={row['last_event_seq']}"
        )
    print(f"fleet_slots={len(rows)} blocked={blocked}")
    return 1 if blocked else 0


def explain(connection: sqlite3.Connection, slot: str, limit: int) -> int:
    row = connection.execute(
        "SELECT * FROM slot_view WHERE slot = ?", (slot,)
    ).fetchone()
    if not row:
        raise FleetdError(f"unknown fleet slot: {slot}")
    desired = json.loads(row["desired_json"]) if row["desired_json"] else {}
    observed = json.loads(row["observed_json"]) if row["observed_json"] else {}
    print(f"slot={slot}")
    print(f"desired_hash={row['desired_hash'] or '-'}")
    print(f"desired_enabled={str(desired.get('enabled', False)).lower()}")
    print(f"restart_policy={desired.get('restart', '-')}")
    print(f"observed_state={observed.get('state', 'unknown')}")
    print(f"observed_generation={observed.get('generation') or '-'}")
    print(f"decision={row['decision'] or 'unknown'}")
    print(f"reason={row['reason'] or '-'}")
    print_events(connection, slot, limit)
    return 1 if row["decision"] == "blocked" else 0


def common_db_path(args: argparse.Namespace) -> Path:
    if args.db:
        return Path(args.db).expanduser().resolve()
    return default_db(Path(args.cwd).resolve())


def absolute_without_symlink_resolution(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.absolute()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sounio-fleet")
    root.add_argument("--db")
    root.add_argument("--cwd", default=os.getcwd())
    subparsers = root.add_subparsers(dest="command_name", required=True)

    subparsers.add_parser("runtime-version")
    keygen_parser = subparsers.add_parser("keygen")
    keygen_parser.add_argument("--private-key", required=True)
    keygen_parser.add_argument("--public-key", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--config", default="fleet.toml")
    for name in ("observe", "plan", "reconcile"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="fleet.toml")
        if name == "reconcile":
            command.add_argument("--apply", action="store_true")
            command.add_argument("--capability", action="append", default=[])
    authorize_parser = subparsers.add_parser("authorize")
    authorize_parser.add_argument("--config", default="fleet.toml")
    authorize_parser.add_argument("--slot", required=True)
    authorize_parser.add_argument("--action", choices=("start",), default="start")
    authorize_parser.add_argument("--out", required=True)
    authorize_parser.add_argument("--ttl", type=int, default=600)
    watch_parser = subparsers.add_parser("watch")
    watch_parser.add_argument("--config", default="fleet.toml")
    watch_parser.add_argument("--interval", type=float, default=2.0)
    watch_parser.add_argument("--cycles", type=int, default=0)
    watch_parser.add_argument("--apply", action="store_true")
    events_parser = subparsers.add_parser("events")
    events_parser.add_argument("--slot")
    events_parser.add_argument("--limit", type=int, default=50)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--slot")
    explain_parser = subparsers.add_parser("explain")
    explain_parser.add_argument("--slot", required=True)
    explain_parser.add_argument("--limit", type=int, default=20)
    subparsers.add_parser("verify-log")
    subparsers.add_parser("rebuild-views")
    anchor_parser = subparsers.add_parser("anchor-log")
    anchor_parser.add_argument("--private-key", required=True)
    anchor_parser.add_argument("--public-key", required=True)
    anchor_parser.add_argument("--anchor-dir", required=True)
    verify_anchor_parser = subparsers.add_parser("verify-anchors")
    verify_anchor_parser.add_argument("--public-key", required=True)
    verify_anchor_parser.add_argument("--anchor-dir", required=True)
    checkpoint_create = subparsers.add_parser("checkpoint-create")
    checkpoint_create.add_argument("--config", default="fleet.toml")
    checkpoint_create.add_argument("--slot", required=True)
    checkpoint_create.add_argument(
        "--kind", choices=("cognitive", "scientific"), required=True
    )
    checkpoint_create.add_argument("--summary-file", required=True)
    checkpoint_create.add_argument("--evidence", action="append", required=True)
    checkpoint_verify = subparsers.add_parser("checkpoint-verify")
    checkpoint_verify.add_argument("--checkpoint-id", required=True)
    handoff_prepare = subparsers.add_parser("handoff-prepare")
    handoff_prepare.add_argument("--checkpoint-id", required=True)
    handoff_prepare.add_argument("--to-agent", required=True)
    handoff_prepare.add_argument("--to-lane", required=True)
    handoff_prepare.add_argument("--capability-out", required=True)
    handoff_prepare.add_argument("--ttl", type=int, default=600)
    handoff_accept = subparsers.add_parser("handoff-accept")
    handoff_accept.add_argument("--handoff-id", required=True)
    handoff_accept.add_argument("--agent", required=True)
    handoff_accept.add_argument("--lane", required=True)
    handoff_accept.add_argument("--capability", required=True)
    handoff_accept.add_argument("--public-key", required=True)
    handoff_accept.add_argument("--anchor-dir", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command_name == "runtime-version":
        print(f"protocol_version={PROTOCOL_VERSION}")
        print(f"runtime_version={RUNTIME_VERSION}")
        print(f"schema_version={SCHEMA_VERSION}")
        return 0
    if args.command_name == "keygen":
        private_key = absolute_without_symlink_resolution(args.private_key)
        public_key = absolute_without_symlink_resolution(args.public_key)
        generate_anchor_keypair(private_key, public_key)
        print(
            "FLEET_ANCHOR_KEY_GENERATED "
            f"private_key={private_key} public_key={public_key} "
            f"public_key_sha256={public_key_fingerprint(public_key)}"
        )
        return 0
    db_path = common_db_path(args)
    with connect_db(db_path) as connection:
        if args.command_name == "verify-log":
            count, tail, views = verify_state(connection)
            print(
                f"FLEET_LOG_VERIFIED events={count} views={views} tail_hash={tail}"
            )
            return 0
        if args.command_name == "anchor-log":
            private_key = absolute_without_symlink_resolution(args.private_key)
            public_key = absolute_without_symlink_resolution(args.public_key)
            directory = absolute_without_symlink_resolution(args.anchor_dir)
            with writer_lock(db_path):
                output = create_anchor(
                    connection, directory, private_key, public_key
                )
            document = read_anchor(output)
            print(
                "FLEET_LOG_ANCHORED "
                f"events={document['payload']['event_count']} "
                f"tail_hash={document['payload']['tail_hash']} path={output}"
            )
            return 0
        if args.command_name == "verify-anchors":
            public_key = absolute_without_symlink_resolution(args.public_key)
            directory = absolute_without_symlink_resolution(args.anchor_dir)
            count, latest = verify_anchor_directory(
                connection, directory, public_key
            )
            if latest is None:
                raise FleetdError("signed fleet anchor directory has no latest anchor")
            print(
                "FLEET_ANCHORS_VERIFIED "
                f"anchors={count} latest_events={latest['payload']['event_count']} "
                f"latest_tail_hash={latest['payload']['tail_hash']}"
            )
            return 0
        if args.command_name == "rebuild-views":
            with writer_lock(db_path):
                rebuild_views(connection)
            print("FLEET_VIEWS_REBUILT")
            return 0
        if args.command_name == "events":
            verify_chain(connection)
            print_events(connection, args.slot, args.limit)
            return 0
        if args.command_name == "status":
            verify_state(connection)
            return print_status(connection, args.slot)
        if args.command_name == "explain":
            verify_state(connection)
            return explain(connection, args.slot, args.limit)
        if args.command_name == "checkpoint-verify":
            with writer_lock(db_path):
                return verify_checkpoint(connection, args.checkpoint_id)
        if args.command_name == "handoff-prepare":
            output_path = absolute_without_symlink_resolution(args.capability_out)
            with writer_lock(db_path):
                prepare_handoff(
                    connection,
                    args.checkpoint_id,
                    args.to_agent,
                    args.to_lane,
                    output_path,
                    args.ttl,
                )
            return 0
        if args.command_name == "handoff-accept":
            capability_path = absolute_without_symlink_resolution(args.capability)
            public_key = absolute_without_symlink_resolution(args.public_key)
            anchor_dir = absolute_without_symlink_resolution(args.anchor_dir)
            with writer_lock(db_path):
                return accept_handoff(
                    connection,
                    args.handoff_id,
                    args.agent,
                    args.lane,
                    capability_path,
                    public_key,
                    anchor_dir,
                )
        config_path = Path(args.config).expanduser().resolve()
        specs = load_config(config_path)
        if args.command_name == "init":
            with writer_lock(db_path):
                verify_state(connection)
                verify_config_coverage(connection, specs)
                for spec in specs:
                    declare_desired(connection, spec)
            print(
                f"FLEET_INITIALIZED db={db_path} config={config_path} lanes={len(specs)}"
            )
            return 0
        if args.command_name == "authorize":
            output_path = Path(args.out).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            with writer_lock(db_path):
                return authorize_start(
                    connection,
                    specs,
                    args.slot,
                    output_path.absolute(),
                    args.ttl,
                )
        if args.command_name == "checkpoint-create":
            with writer_lock(db_path):
                create_checkpoint(
                    connection,
                    specs,
                    args.slot,
                    args.kind,
                    args.summary_file,
                    args.evidence,
                )
            return 0
        if args.command_name == "observe":
            with writer_lock(db_path):
                return observe_cycle(connection, specs)
        if args.command_name in {"plan", "reconcile"}:
            apply = args.command_name == "reconcile" and args.apply
            capabilities = (
                load_capability_documents(args.capability)
                if args.command_name == "reconcile"
                else {}
            )
            if capabilities and not apply:
                raise FleetdError("capability files require reconcile --apply")
            with writer_lock(db_path):
                return cycle(
                    connection,
                    specs,
                    apply=apply,
                    capabilities=capabilities,
                )
        if args.command_name == "watch":
            if args.interval <= 0:
                raise FleetdError("watch interval must be positive")
            if args.apply:
                raise FleetdError(
                    "watch cannot hold reusable mutation authority; use one-shot "
                    "reconcile --apply --capability"
                )
            cycles = 0
            while True:
                specs = load_config(config_path)
                with writer_lock(db_path):
                    result = cycle(connection, specs, apply=False)
                cycles += 1
                if args.cycles and cycles >= args.cycles:
                    return result
                time.sleep(args.interval)
    raise FleetdError(f"unknown command: {args.command_name}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FleetdError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
