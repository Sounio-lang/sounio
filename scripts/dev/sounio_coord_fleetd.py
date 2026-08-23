#!/usr/bin/env python3
"""Durable desired-state reconciliation for Sounio agent fleets."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Iterator


PROTOCOL_VERSION = 1
RUNTIME_VERSION = "2026.08.23.1"
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


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


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
        return existing, False
    connection.execute("BEGIN IMMEDIATE")
    try:
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
        expected["command"] = Path(spec.command[0]).name
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
        if previous != "active":
            return "hold", "on-failure-requires-prior-active-observation"
    if state in {"absent", "unreachable"}:
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


def launch_arguments(spec: LaneSpec) -> list[str]:
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
        "--no-attach",
    ]
    if spec.lane:
        arguments.extend(["--lane", spec.lane])
    return [*arguments, "--", *spec.command]


def apply_start(
    connection: sqlite3.Connection,
    spec: LaneSpec,
    observation: dict[str, Any],
    observation_seq: int,
) -> bool:
    action_id = digest_json(
        {
            "action": "start",
            "desired_hash": spec.desired_hash,
            "observation": observation["fingerprint"],
            "slot": spec.slot,
        }
    )
    requested = {
        "action": "start",
        "action_id": action_id,
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
    arguments = launch_arguments(spec)
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
    outcome = {
        "action": "start",
        "action_id": action_id,
        "exit_code": return_code,
        "output_digest": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "status": "committed" if return_code == 0 else "failed",
    }
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


def cycle(
    connection: sqlite3.Connection,
    specs: list[LaneSpec],
    *,
    apply: bool,
    emit: bool = True,
) -> int:
    verify_state(connection)
    verify_config_coverage(connection, specs)
    blocked = 0
    failed = 0
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
            if apply_start(
                connection, spec, observation, int(observation_event["seq"])
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sounio-fleet")
    root.add_argument("--db")
    root.add_argument("--cwd", default=os.getcwd())
    subparsers = root.add_subparsers(dest="command_name", required=True)

    subparsers.add_parser("runtime-version")
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--config", default="fleet.toml")
    for name in ("observe", "plan", "reconcile"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", default="fleet.toml")
        if name == "reconcile":
            command.add_argument("--apply", action="store_true")
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
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command_name == "runtime-version":
        print(f"protocol_version={PROTOCOL_VERSION}")
        print(f"runtime_version={RUNTIME_VERSION}")
        print(f"schema_version={SCHEMA_VERSION}")
        return 0
    db_path = common_db_path(args)
    with connect_db(db_path) as connection:
        if args.command_name == "verify-log":
            count, tail, views = verify_state(connection)
            print(
                f"FLEET_LOG_VERIFIED events={count} views={views} tail_hash={tail}"
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
        if args.command_name == "observe":
            with writer_lock(db_path):
                return observe_cycle(connection, specs)
        if args.command_name in {"plan", "reconcile"}:
            apply = args.command_name == "reconcile" and args.apply
            with writer_lock(db_path):
                return cycle(connection, specs, apply=apply)
        if args.command_name == "watch":
            if args.interval <= 0:
                raise FleetdError("watch interval must be positive")
            cycles = 0
            while True:
                specs = load_config(config_path)
                with writer_lock(db_path):
                    result = cycle(connection, specs, apply=args.apply)
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
