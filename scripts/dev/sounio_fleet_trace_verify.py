#!/usr/bin/env python3
"""Independent refinement checker for a concrete Sounio fleet Event Log."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ZERO_HASH = "0" * 64
CERTIFICATE_VERSION = 1


class TraceError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def event_digest(row: sqlite3.Row) -> str:
    return digest_json(
        {
            "causal_key": row["causal_key"],
            "event_type": row["event_type"],
            "occurred_utc": row["occurred_utc"],
            "payload": row["payload"],
            "prev_hash": row["prev_hash"],
            "seq": row["seq"],
            "slot": row["slot"],
        }
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceError(message)


def string_field(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    require(isinstance(value, str) and bool(value), f"{context} omits {key}")
    return value


def open_read_only(path: Path) -> sqlite3.Connection:
    require(path.is_file() and not path.is_symlink(), f"fleet database is invalid: {path}")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    return connection


def openssl() -> str:
    command = shutil.which("openssl")
    require(command is not None, "OpenSSL is required to verify fleet anchors")
    executable = str(Path(command).resolve())
    result = subprocess.run(
        [executable, "version"], check=False, capture_output=True, text=True, timeout=5.0
    )
    words = result.stdout.split()
    require(result.returncode == 0 and len(words) >= 2, "OpenSSL version probe failed")
    try:
        major = int(words[1].split(".", maxsplit=1)[0])
    except ValueError as exc:
        raise TraceError("OpenSSL version is not parseable") from exc
    require(major >= 3, "OpenSSL 3 or newer is required for Ed25519")
    return executable


def run_openssl(arguments: list[str], input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        [openssl(), *arguments],
        input=input_bytes,
        check=False,
        capture_output=True,
        timeout=10.0,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise TraceError(f"OpenSSL verification failed: {detail or result.returncode}")
    return result.stdout


def public_key_fingerprint(public_key: Path) -> str:
    require(public_key.is_file() and not public_key.is_symlink(), "public key is invalid")
    der = run_openssl(["pkey", "-pubin", "-in", str(public_key), "-outform", "DER"])
    return hashlib.sha256(der).hexdigest()


def verify_signature(public_key: Path, material: bytes, signature: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="sounio-trace-anchor-") as raw:
        directory = Path(raw)
        message = directory / "message"
        signature_file = directory / "signature"
        message.write_bytes(material)
        signature_file.write_bytes(signature)
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
                str(signature_file),
            ]
        )


def verify_anchors(
    rows_by_seq: dict[int, sqlite3.Row],
    database_id: str,
    anchor_dir: Path,
    public_key: Path,
) -> dict[tuple[int, str], dict[str, Any]]:
    paths = sorted(anchor_dir.glob("anchor-*.json")) if anchor_dir.is_dir() else []
    require(bool(paths), f"no signed fleet anchors found in {anchor_dir}")
    fingerprint = public_key_fingerprint(public_key)
    previous_digest = ZERO_HASH
    previous_count = 0
    verified: dict[tuple[int, str], dict[str, Any]] = {}
    for path in paths:
        require(path.is_file() and not path.is_symlink(), f"anchor is invalid: {path}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TraceError(f"cannot read anchor {path}: {exc}") from exc
        require(isinstance(document, dict) and document.get("version") == 1, f"unsupported anchor: {path}")
        payload = document.get("payload")
        encoded = document.get("signature_base64")
        require(isinstance(payload, dict) and isinstance(encoded, str), f"anchor omits payload or signature: {path}")
        count = payload.get("event_count")
        tail = payload.get("tail_hash")
        require(isinstance(count, int) and count > previous_count, f"anchor sequence is not increasing: {path}")
        require(isinstance(tail, str) and count in rows_by_seq, f"anchor prefix is absent: {path}")
        require(rows_by_seq[count]["event_hash"] == tail, f"anchor tail mismatch: {path}")
        require(payload.get("algorithm") == "Ed25519", f"anchor algorithm mismatch: {path}")
        require(payload.get("database_id") == database_id, f"anchor database mismatch: {path}")
        require(payload.get("public_key_sha256") == fingerprint, f"anchor key mismatch: {path}")
        require(payload.get("previous_anchor_sha256") == previous_digest, f"anchor predecessor mismatch: {path}")
        try:
            signature = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise TraceError(f"anchor signature is not canonical base64: {path}") from exc
        verify_signature(public_key, canonical_json(payload).encode("utf-8"), signature)
        verified[(count, tail)] = payload
        previous_count = count
        previous_digest = digest_json(document)
    return verified


def receipt_digest(draft: dict[str, Any]) -> str:
    summary = draft.get("summary")
    evidence = draft.get("evidence")
    require(isinstance(summary, dict), "checkpoint draft omits summary receipt")
    require(isinstance(evidence, list) and bool(evidence), "checkpoint draft omits evidence receipts")
    for receipt in [summary, *evidence]:
        require(isinstance(receipt, dict), "checkpoint receipt is not an object")
        require(isinstance(receipt.get("path"), str), "checkpoint receipt omits path")
        digest = receipt.get("sha256")
        require(isinstance(digest, str) and len(digest) == 64, "checkpoint receipt omits SHA-256")
        require(isinstance(receipt.get("size"), int) and receipt["size"] >= 0, "checkpoint receipt size is invalid")
    return digest_json([summary, *evidence])


def verify_trace(
    connection: sqlite3.Connection,
    *,
    anchor_dir: Path | None,
    public_key: Path | None,
) -> dict[str, Any]:
    meta = {
        row["key"]: row["value"]
        for row in connection.execute("SELECT key, value FROM meta ORDER BY key")
    }
    require(meta.get("schema_version") == "1", "fleet schema version is not 1")
    database_id = meta.get("database_id")
    require(isinstance(database_id, str) and bool(database_id), "fleet database identity is missing")
    rows = connection.execute("SELECT * FROM events ORDER BY seq").fetchall()
    previous = ZERO_HASH
    rows_by_seq: dict[int, sqlite3.Row] = {}
    parsed: dict[int, dict[str, Any]] = {}
    for expected, row in enumerate(rows, start=1):
        require(row["seq"] == expected, f"event sequence gap at {expected}")
        require(row["prev_hash"] == previous, f"event {expected} previous hash mismatch")
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError as exc:
            raise TraceError(f"event {expected} payload is not JSON") from exc
        require(isinstance(payload, dict), f"event {expected} payload is not an object")
        require(row["payload"] == canonical_json(payload), f"event {expected} payload is not canonical")
        calculated = event_digest(row)
        require(calculated == row["event_hash"], f"event {expected} hash mismatch")
        require(row["event_id"] == f"evt-{calculated[:24]}", f"event {expected} id mismatch")
        rows_by_seq[expected] = row
        parsed[expected] = payload
        previous = calculated

    verified_anchors: dict[tuple[int, str], dict[str, Any]] = {}
    if (anchor_dir is None) != (public_key is None):
        raise TraceError("--anchor-dir and --public-key must be provided together")
    if anchor_dir is not None and public_key is not None:
        verified_anchors = verify_anchors(rows_by_seq, database_id, anchor_dir, public_key)

    capabilities: dict[str, dict[str, Any]] = {}
    recovery_budgets: dict[str, dict[str, Any]] = {}
    actions: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    handoffs: dict[str, dict[str, Any]] = {}
    observations: dict[int, tuple[sqlite3.Row, dict[str, Any]]] = {}
    abstract_trace: list[dict[str, Any]] = []
    accepted_claims: list[dict[str, Any]] = []
    stopped_claims: list[dict[str, Any]] = []

    for row in rows:
        seq = int(row["seq"])
        event_type = str(row["event_type"])
        payload = parsed[seq]
        label = "Stutter"
        if event_type == "OBSERVATION":
            observations[seq] = (row, payload)
        elif event_type == "RECOVERY_BUDGET_ISSUED":
            budget_id = string_field(payload, "budget_id", f"event {seq}")
            require(budget_id not in recovery_budgets, f"recovery budget issued twice: {budget_id}")
            require(payload.get("action") == "recover-start", f"recovery budget action is invalid: {budget_id}")
            require(isinstance(payload.get("max_starts"), int) and 0 < payload["max_starts"] <= 64, f"recovery budget bound is invalid: {budget_id}")
            require(isinstance(payload.get("backoff_seconds"), int) and payload["backoff_seconds"] >= 0, f"recovery budget backoff is invalid: {budget_id}")
            require(isinstance(payload.get("issued_unix"), int), f"recovery budget issue time is invalid: {budget_id}")
            require(
                isinstance(payload.get("expires_unix"), int)
                and payload["expires_unix"] >= payload["issued_unix"],
                f"recovery budget validity interval is invalid: {budget_id}",
            )
            recovery_budgets[budget_id] = {
                "issued": (row, payload),
                "published": None,
                "revoked": None,
                "spent": [],
            }
        elif event_type == "RECOVERY_BUDGET_PUBLISHED":
            budget_id = string_field(payload, "budget_id", f"event {seq}")
            state = recovery_budgets.get(budget_id)
            require(state is not None and state["published"] is None, f"recovery budget publication is invalid: {budget_id}")
            issued_row, issued = state["issued"]
            require(row["slot"] == issued_row["slot"] == issued.get("slot"), f"recovery budget publication slot mismatch: {budget_id}")
            require(payload.get("issued_event_seq") == issued_row["seq"], f"recovery budget publication sequence mismatch: {budget_id}")
            require(payload.get("issued_event_hash") == issued_row["event_hash"], f"recovery budget publication hash mismatch: {budget_id}")
            state["published"] = (row, payload)
        elif event_type == "RECOVERY_BUDGET_REVOKED":
            budget_id = string_field(payload, "budget_id", f"event {seq}")
            state = recovery_budgets.get(budget_id)
            require(state is not None and state["revoked"] is None, f"recovery budget revocation is invalid: {budget_id}")
            require(not state["spent"], f"spent recovery budget was revoked: {budget_id}")
            state["revoked"] = (row, payload)
        elif event_type == "RECOVERY_BUDGET_SPENT":
            budget_id = string_field(payload, "budget_id", f"event {seq}")
            state = recovery_budgets.get(budget_id)
            require(state is not None and state["published"] is not None and state["revoked"] is None, f"unpublished or revoked recovery budget was spent: {budget_id}")
            issued_row, issued = state["issued"]
            ordinal = payload.get("ordinal")
            require(ordinal == len(state["spent"]) + 1, f"recovery budget ordinal is not contiguous: {budget_id}")
            require(ordinal <= issued.get("max_starts"), f"recovery budget exceeded its bound: {budget_id}")
            spent_unix = payload.get("spent_unix")
            require(isinstance(spent_unix, int), f"recovery budget spend time is invalid: {budget_id}")
            require(
                issued.get("issued_unix") <= spent_unix <= issued.get("expires_unix"),
                f"recovery budget was spent outside its validity interval: {budget_id}",
            )
            if state["spent"]:
                previous_spent_unix = state["spent"][-1][1].get("spent_unix")
                require(
                    spent_unix >= previous_spent_unix + issued.get("backoff_seconds"),
                    f"recovery budget backoff was violated: {budget_id}",
                )
            for key in ("slot", "desired_hash", "max_starts", "backoff_seconds"):
                require(payload.get(key) == issued.get(key), f"recovery budget spend {key} mismatch: {budget_id}")
            capability_id = string_field(payload, "capability_id", f"recovery budget {budget_id}")
            require(all(entry[1].get("capability_id") != capability_id for entry in state["spent"]), f"recovery child capability is duplicated: {capability_id}")
            state["spent"].append((row, payload))
        elif event_type == "CAPABILITY_ISSUED":
            capability_id = string_field(payload, "capability_id", f"event {seq}")
            require(capability_id not in capabilities, f"capability {capability_id} has duplicate issuance")
            action = string_field(payload, "action", f"capability {capability_id}")
            require(action in {"start", "stop", "accept-handoff"}, f"capability {capability_id} action is invalid")
            capabilities[capability_id] = {"issued": (row, payload), "published": None, "terminal": None}
            parent_budget = payload.get("parent_recovery_budget_id")
            if parent_budget is not None:
                state = recovery_budgets.get(str(parent_budget))
                require(state is not None, f"recovery child has no budget: {capability_id}")
                ordinal = payload.get("recovery_ordinal")
                matching = [
                    spending
                    for _, spending in state["spent"]
                    if spending.get("ordinal") == ordinal
                    and spending.get("capability_id") == capability_id
                ]
                require(len(matching) == 1, f"recovery child is not bound to one spend: {capability_id}")
                require(payload.get("slot") == matching[0].get("slot"), f"recovery child slot mismatch: {capability_id}")
                require(payload.get("desired_hash") == matching[0].get("desired_hash"), f"recovery child desired hash mismatch: {capability_id}")
                label = "IssueStartCapability"
            elif action == "start":
                label = "IssueStartCapability"
            elif action == "stop":
                label = "IssueStopCapability"
            else:
                label = "PrepareHandoff"
        elif event_type == "CAPABILITY_PUBLISHED":
            capability_id = string_field(payload, "capability_id", f"event {seq}")
            state = capabilities.get(capability_id)
            require(state is not None, f"published capability has no issuance: {capability_id}")
            require(state["published"] is None, f"capability was published twice: {capability_id}")
            issued_row, issued = state["issued"]
            require(row["slot"] == issued_row["slot"], f"capability publication slot mismatch: {capability_id}")
            require(payload.get("action") == issued.get("action"), f"capability publication action mismatch: {capability_id}")
            require(payload.get("issued_event_seq") == issued_row["seq"], f"capability publication sequence mismatch: {capability_id}")
            require(payload.get("issued_event_hash") == issued_row["event_hash"], f"capability publication hash mismatch: {capability_id}")
            state["published"] = (row, payload)
            label = "Stutter"
        elif event_type in {"CAPABILITY_CONSUMED", "CAPABILITY_REVOKED"}:
            capability_id = string_field(payload, "capability_id", f"event {seq}")
            state = capabilities.get(capability_id)
            require(state is not None, f"terminal capability has no issuance: {capability_id}")
            require(state["terminal"] is None, f"capability has multiple terminal events: {capability_id}")
            issued_row, issued = state["issued"]
            if event_type == "CAPABILITY_CONSUMED" and issued.get("capability_path") is not None:
                require(state["published"] is not None, f"unpublished capability was consumed or revoked: {capability_id}")
            require(row["slot"] == issued_row["slot"], f"capability terminal slot mismatch: {capability_id}")
            if event_type == "CAPABILITY_CONSUMED":
                require(payload.get("action") == issued.get("action"), f"capability consumption action mismatch: {capability_id}")
                if issued.get("action") == "start":
                    keys = ("slot", "desired_hash", "observation_fingerprint")
                elif issued.get("action") == "stop":
                    keys = (
                        "slot",
                        "desired_hash",
                        "observation_fingerprint",
                        "generation",
                        "argv_digest",
                        "start_capability_id",
                    )
                else:
                    keys = ("slot", "handoff_id", "evidence_set_digest", "to_agent", "to_lane")
                for key in keys:
                    require(payload.get(key) == issued.get(key), f"capability consumption {key} mismatch: {capability_id}")
                label = "Stutter"
            else:
                label = "Stutter"
            state["terminal"] = (row, payload)
        elif event_type == "ACTION_REQUESTED":
            action_id = string_field(payload, "action_id", f"event {seq}")
            capability_id = string_field(payload, "capability_id", f"action {action_id}")
            require(action_id not in actions, f"action request is duplicated: {action_id}")
            capability = capabilities.get(capability_id)
            require(capability is not None, f"action has no capability: {action_id}")
            issued = capability["issued"][1]
            terminal = capability["terminal"]
            require(terminal is not None and terminal[0]["event_type"] == "CAPABILITY_CONSUMED", f"action uses unconsumed capability: {action_id}")
            action = string_field(payload, "action", f"action {action_id}")
            require(action == issued.get("action") and action in {"start", "stop"}, f"action kind does not match capability: {action_id}")
            identity = {
                    "action": "start",
                    "capability_id": capability_id,
                    "desired_hash": payload.get("desired_hash"),
                    "observation": issued.get("observation_fingerprint"),
                    "slot": row["slot"],
            }
            if action == "stop":
                identity["action"] = "stop"
                identity["generation"] = payload.get("generation")
            expected = digest_json(identity)
            require(action_id == expected, f"action identity is not capability-bound: {action_id}")
            require(payload.get("desired_hash") == issued.get("desired_hash"), f"action desired hash mismatch: {action_id}")
            for key in ("generation", "argv_digest", "start_capability_id") if action == "stop" else ():
                require(payload.get(key) == issued.get(key), f"stop action {key} mismatch: {action_id}")
            actions[action_id] = {"action": action, "requested": (row, payload), "terminal": None}
            label = "Stutter"
        elif event_type in {"ACTION_COMMITTED", "ACTION_FAILED"}:
            action_id = string_field(payload, "action_id", f"event {seq}")
            state = actions.get(action_id)
            require(state is not None, f"action result has no request: {action_id}")
            require(state["terminal"] is None, f"action has multiple results: {action_id}")
            requested = state["requested"][1]
            require(payload.get("capability_id") == requested.get("capability_id"), f"action result capability mismatch: {action_id}")
            action = state["action"]
            require(payload.get("action") == action, f"action result kind mismatch: {action_id}")
            if event_type == "ACTION_COMMITTED":
                require(payload.get("status") == "committed", f"committed action status mismatch: {action_id}")
                require(isinstance(payload.get("generation"), str) and bool(payload["generation"]), f"committed action omits generation: {action_id}")
                require(isinstance(payload.get("argv_digest"), str) and len(payload["argv_digest"]) == 64, f"committed action omits full argv digest: {action_id}")
                if action == "start":
                    label = "StartWithLinearCapability"
                else:
                    for key in ("generation", "argv_digest", "start_capability_id"):
                        require(payload.get(key) == requested.get(key), f"committed stop {key} mismatch: {action_id}")
                    require(payload.get("observed_state") == "absent", f"committed stop did not reach absent: {action_id}")
                    stopped_claims.append(
                        {
                            "argv_digest": payload["argv_digest"],
                            "generation": payload["generation"],
                            "start_capability_id": payload.get("start_capability_id"),
                            "stop_capability_id": payload["capability_id"],
                        }
                    )
                    label = "StopWithLinearCapability"
            else:
                require(payload.get("status") == "failed", f"failed action status mismatch: {action_id}")
                label = "Stutter"
            state["terminal"] = (row, payload)
        elif event_type == "CHECKPOINT_DRAFTED":
            checkpoint_id = string_field(payload, "checkpoint_id", f"event {seq}")
            require(checkpoint_id not in checkpoints, f"checkpoint draft is duplicated: {checkpoint_id}")
            observation_seq = payload.get("observation_seq")
            require(isinstance(observation_seq, int) and observation_seq in observations, f"checkpoint observation is absent: {checkpoint_id}")
            observation_row, observation = observations[observation_seq]
            require(observation_row["slot"] == row["slot"] and observation.get("state") == "active", f"checkpoint observation is not active: {checkpoint_id}")
            require(observation.get("argv_digest") == payload.get("argv_digest"), f"checkpoint argv mismatch: {checkpoint_id}")
            action_id = string_field(payload, "start_action_id", f"checkpoint {checkpoint_id}")
            capability_id = string_field(payload, "start_capability_id", f"checkpoint {checkpoint_id}")
            require(observation.get("start_capability_id") == capability_id, f"checkpoint observation capability mismatch: {checkpoint_id}")
            action = actions.get(action_id)
            require(action is not None and action["terminal"] is not None, f"checkpoint start action is absent: {checkpoint_id}")
            action_row, action_payload = action["terminal"]
            require(action_row["event_type"] == "ACTION_COMMITTED", f"checkpoint start action did not commit: {checkpoint_id}")
            require(action_payload.get("capability_id") == capability_id, f"checkpoint start capability mismatch: {checkpoint_id}")
            require(action_payload.get("generation") == payload.get("generation"), f"checkpoint generation mismatch: {checkpoint_id}")
            require(action_payload.get("argv_digest") == payload.get("argv_digest"), f"checkpoint action argv mismatch: {checkpoint_id}")
            require(payload.get("start_action_event_seq") == action_row["seq"], f"checkpoint action sequence mismatch: {checkpoint_id}")
            require(payload.get("start_action_event_hash") == action_row["event_hash"], f"checkpoint action hash mismatch: {checkpoint_id}")
            checkpoints[checkpoint_id] = {"draft": (row, payload), "verified": None, "evidence_digest": receipt_digest(payload)}
            label = "CreateCheckpoint"
        elif event_type == "CHECKPOINT_VERIFIED":
            checkpoint_id = string_field(payload, "checkpoint_id", f"event {seq}")
            state = checkpoints.get(checkpoint_id)
            require(state is not None, f"verified checkpoint has no draft: {checkpoint_id}")
            require(state["verified"] is None, f"checkpoint verified twice: {checkpoint_id}")
            draft_row = state["draft"][0]
            require(payload.get("draft_event_seq") == draft_row["seq"], f"checkpoint draft sequence mismatch: {checkpoint_id}")
            require(payload.get("draft_event_hash") == draft_row["event_hash"], f"checkpoint draft hash mismatch: {checkpoint_id}")
            require(payload.get("evidence_set_digest") == state["evidence_digest"], f"checkpoint evidence digest mismatch: {checkpoint_id}")
            state["verified"] = (row, payload)
            label = "VerifyCheckpoint"
        elif event_type == "HANDOFF_PREPARED":
            handoff_id = string_field(payload, "handoff_id", f"event {seq}")
            checkpoint_id = string_field(payload, "checkpoint_id", f"handoff {handoff_id}")
            require(handoff_id not in handoffs, f"handoff preparation is duplicated: {handoff_id}")
            checkpoint = checkpoints.get(checkpoint_id)
            require(checkpoint is not None and checkpoint["verified"] is not None, f"handoff checkpoint is not verified: {handoff_id}")
            verified_row, verified = checkpoint["verified"]
            require(payload.get("verified_checkpoint_event_seq") == verified_row["seq"], f"handoff checkpoint sequence mismatch: {handoff_id}")
            require(payload.get("verified_checkpoint_event_hash") == verified_row["event_hash"], f"handoff checkpoint hash mismatch: {handoff_id}")
            require(payload.get("evidence_set_digest") == verified.get("evidence_set_digest"), f"handoff evidence mismatch: {handoff_id}")
            handoffs[handoff_id] = {"prepared": (row, payload), "requested": None, "accepted": None}
            label = "Stutter"
        elif event_type == "HANDOFF_ACCEPTANCE_REQUESTED":
            handoff_id = string_field(payload, "handoff_id", f"event {seq}")
            state = handoffs.get(handoff_id)
            require(state is not None and state["requested"] is None, f"handoff acceptance request is invalid: {handoff_id}")
            prepared_row, prepared = state["prepared"]
            capability_id = string_field(payload, "capability_id", f"handoff {handoff_id}")
            capability = capabilities.get(capability_id)
            require(capability is not None and capability["published"] is not None, f"handoff acceptance authority is not published: {handoff_id}")
            issued = capability["issued"][1]
            for key in ("checkpoint_id", "evidence_set_digest", "handoff_id", "to_agent", "to_lane"):
                require(payload.get(key) == prepared.get(key) == issued.get(key), f"handoff request {key} mismatch: {handoff_id}")
            anchor_count = payload.get("anchor_event_count")
            anchor_tail = payload.get("anchor_tail_hash")
            require(isinstance(anchor_count, int) and anchor_count >= prepared_row["seq"], f"handoff anchor does not cover preparation: {handoff_id}")
            require(anchor_count in rows_by_seq and rows_by_seq[anchor_count]["event_hash"] == anchor_tail, f"handoff anchor is not a log prefix: {handoff_id}")
            if verified_anchors:
                require((anchor_count, anchor_tail) in verified_anchors, f"handoff anchor signature is unverified: {handoff_id}")
            state["requested"] = (row, payload)
            label = "AnchorVerifiedPrefix"
        elif event_type == "HANDOFF_ACCEPTED":
            handoff_id = string_field(payload, "handoff_id", f"event {seq}")
            state = handoffs.get(handoff_id)
            require(state is not None and state["accepted"] is None, f"accepted handoff is not unique: {handoff_id}")
            require(state["requested"] is not None, f"accepted handoff has no anchored request: {handoff_id}")
            _, prepared = state["prepared"]
            _, requested = state["requested"]
            capability_id = string_field(payload, "capability_id", f"handoff {handoff_id}")
            capability = capabilities.get(capability_id)
            require(capability is not None and capability["terminal"] is not None, f"accepted handoff has no consumed authority: {handoff_id}")
            terminal_row, terminal = capability["terminal"]
            require(terminal_row["event_type"] == "CAPABILITY_CONSUMED", f"accepted handoff authority was not consumed: {handoff_id}")
            for key in ("checkpoint_id", "evidence_set_digest", "handoff_id", "to_agent", "to_lane", "anchor_event_count", "anchor_tail_hash"):
                require(payload.get(key) == requested.get(key), f"accepted handoff {key} mismatch: {handoff_id}")
            require(terminal.get("handoff_id") == handoff_id and terminal.get("capability_id") == capability_id, f"accepted handoff capability mismatch: {handoff_id}")
            checkpoint = checkpoints[str(prepared["checkpoint_id"])]
            draft = checkpoint["draft"][1]
            accepted_claims.append(
                {
                    "anchor_event_count": payload["anchor_event_count"],
                    "argv_digest": draft["argv_digest"],
                    "checkpoint_id": prepared["checkpoint_id"],
                    "evidence_set_digest": prepared["evidence_set_digest"],
                    "handoff_capability_id": capability_id,
                    "handoff_id": handoff_id,
                    "recipient": f"{prepared['to_agent']}/{prepared['to_lane']}",
                    "start_capability_id": draft["start_capability_id"],
                }
            )
            state["accepted"] = (row, payload)
            label = "AcceptAnchoredHandoff"
        abstract_trace.append({"event_seq": seq, "event_type": event_type, "formal_action": label})

    if accepted_claims:
        require(bool(verified_anchors), "accepted handoffs require independently verified anchors")
    for budget_id, state in recovery_budgets.items():
        issued = state["issued"][1]
        require(len(state["spent"]) <= int(issued["max_starts"]), f"recovery budget exceeded its bound: {budget_id}")
        for _, spending in state["spent"]:
            capability_id = str(spending["capability_id"])
            capability = capabilities.get(capability_id)
            require(capability is not None, f"recovery spend has no child capability: {capability_id}")
            child = capability["issued"][1]
            require(child.get("parent_recovery_budget_id") == budget_id, f"recovery child parent mismatch: {capability_id}")
            require(child.get("recovery_ordinal") == spending.get("ordinal"), f"recovery child ordinal mismatch: {capability_id}")
    invariants = {
        "accepted_handoff_has_unique_causal_trace": True,
        "accepted_handoff_preserves_argv": True,
        "accepted_handoff_preserves_evidence": True,
        "accepted_handoff_recipient_matches": True,
        "accepted_handoff_uses_one_start_capability": True,
        "capability_consumption_is_linear": True,
        "event_log_hash_chain_valid": True,
        "recovery_budget_spending_is_bounded": True,
        "signed_anchor_is_log_prefix": True,
        "stop_targets_one_exact_generation": True,
    }
    certificate = {
        "abstract_trace": abstract_trace,
        "accepted_handoffs": accepted_claims,
        "certificate_version": CERTIFICATE_VERSION,
        "database_id": database_id,
        "event_count": len(rows),
        "invariants": invariants,
        "stopped_generations": stopped_claims,
        "tail_hash": previous,
    }
    certificate["certificate_sha256"] = digest_json(certificate)
    return certificate


def write_certificate(path: Path, certificate: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(canonical_json(certificate) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-fleet-trace-verify")
    result.add_argument("--db", required=True)
    result.add_argument("--anchor-dir")
    result.add_argument("--public-key")
    result.add_argument("--certificate")
    return result


def main() -> int:
    args = parser().parse_args()
    db = Path(args.db).expanduser().absolute()
    anchor_dir = Path(args.anchor_dir).expanduser().absolute() if args.anchor_dir else None
    public_key = Path(args.public_key).expanduser().absolute() if args.public_key else None
    with open_read_only(db) as connection:
        certificate = verify_trace(
            connection, anchor_dir=anchor_dir, public_key=public_key
        )
    if args.certificate:
        write_certificate(Path(args.certificate).expanduser().absolute(), certificate)
    print(
        "FLEET_TRACE_CONFORMS "
        f"events={certificate['event_count']} "
        f"accepted={len(certificate['accepted_handoffs'])} "
        f"invariants={len(certificate['invariants'])} "
        f"tail_hash={certificate['tail_hash']} "
        f"certificate_sha256={certificate['certificate_sha256']}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, sqlite3.Error, TraceError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
