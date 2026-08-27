#!/usr/bin/env python3
"""Versioned causal experiment receipts for Sounio coordination."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PREREG_SCHEMA = "sounio.causal_experiment.prereg.v1"
OUTCOME_SCHEMA = "sounio.causal_experiment.outcome.v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
RESULT = re.compile(r"^[A-Za-z0-9._:/+-]+=(PASS|FAIL)$")
GIT_OBJECT = re.compile(r"^[0-9a-f]{40,64}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PREREG_FIELDS = {
    "base_commit",
    "base_tree",
    "control_predicate",
    "created_utc",
    "experiment_id",
    "falsifier",
    "intervention",
    "owner",
    "resources",
    "schema",
    "stage",
    "statement",
    "treatment_predicate",
}
OUTCOME_FIELDS = {
    "closed_utc",
    "control",
    "evidence",
    "experiment_id",
    "owner",
    "prereg_path",
    "prereg_sha256",
    "schema",
    "stage",
    "subject_commit",
    "subject_tree",
    "treatment",
    "verdict",
}
OWNER_FIELDS = {"agent", "branch", "lane", "worktree"}


class Refusal(RuntimeError):
    pass


def git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        input=input_bytes,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise Refusal(detail or f"git {' '.join(args)} failed")
    return result.stdout


def git_text(root: Path, *args: str) -> str:
    return git(root, *args).decode("utf-8").strip()


def worktree() -> Path:
    configured = os.environ.get("SOUNIO_COORD_WORKTREE")
    root = configured or git_text(Path.cwd(), "rev-parse", "--show-toplevel")
    return Path(root).resolve()


def state_dir() -> Path:
    configured = os.environ.get("SOUNIO_COORD_STATE_DIR")
    if not configured:
        raise Refusal("coordination state directory was not provided")
    return Path(configured).resolve()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_text(label: str, value: str) -> str:
    if not value or any(mark in value for mark in ("\n", "\r", "\t")):
        raise Refusal(f"{label} must be non-empty and single-line")
    if len(value) > 4096:
        raise Refusal(f"{label} exceeds 4096 characters")
    return value


def utc_text(label: str, value: str) -> str:
    safe_text(label, value)
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise Refusal(f"{label} is not a canonical UTC timestamp") from error
    return value


def slug(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:80]
    return token or "unnamed"


def repo_path(root: Path, value: str, *, must_exist: bool = False) -> tuple[str, Path]:
    raw = Path(value)
    if raw.is_absolute():
        raise Refusal(f"receipt and evidence paths must be repository-relative: {value}")
    normalized = Path(os.path.normpath(value))
    if not normalized.parts or normalized.parts[0] == ".." or normalized == Path("."):
        raise Refusal(f"path escapes or names no repository file: {value}")
    relative = normalized.as_posix()
    candidate = root / normalized
    if candidate.is_symlink():
        raise Refusal(f"symlink evidence is not accepted: {relative}")
    absolute = candidate.resolve(strict=False)
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise Refusal(f"path escapes repository through a symlink: {value}") from error
    if must_exist and not absolute.is_file():
        raise Refusal(f"file does not exist: {relative}")
    return relative, absolute


def load_json(path: Path, expected_schema: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise Refusal(f"cannot read canonical receipt {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema") != expected_schema:
        raise Refusal(f"receipt schema is not {expected_schema}: {path}")
    if raw != canonical_bytes(value):
        raise Refusal(f"receipt is not canonical JSON: {path}")
    return value, raw


def atomic_write(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise Refusal(f"refusing to replace existing receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".causal-write.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def parse_claim(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Refusal(f"active coordination claim not found: {path.stem}")
    claim: dict[str, Any] = {"files": [], "resources": []}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator:
            continue
        if key == "file":
            claim["files"].append(value)
        elif key == "resource":
            claim["resources"].append(value)
        else:
            claim[key] = value
    try:
        expired = int(time.time()) > int(claim["last_seen_epoch"]) + int(
            claim["ttl_seconds"]
        )
    except (KeyError, ValueError) as error:
        raise Refusal("coordination claim has invalid lease metadata") from error
    if expired:
        raise Refusal(f"coordination claim expired: {claim.get('claim_id', path.stem)}")
    return claim


def path_scope(value: str) -> str:
    return value[:-3].rstrip("/") if value.endswith("/**") else value.rstrip("/")


def path_covers(claimed: str, requested: str) -> bool:
    claimed_scope = path_scope(claimed)
    return bool(claimed_scope) and (
        requested == claimed_scope or requested.startswith(f"{claimed_scope}/")
    )


def resource_parts(value: str) -> tuple[str, str, bool]:
    kind, separator, name = value.partition(":")
    if not separator or kind not in {"concept", "diagnostic", "gate", "api"}:
        raise Refusal(f"invalid typed resource: {value}")
    wildcard = name.endswith("/**")
    normalized = name[:-3].rstrip("/") if wildcard else name
    if not normalized or not re.fullmatch(r"[A-Za-z0-9._:/@+-]+", normalized):
        raise Refusal(f"invalid typed resource: {value}")
    return kind, normalized, wildcard


def resource_covers(claimed: str, requested: str) -> bool:
    claimed_kind, claimed_name, wildcard = resource_parts(claimed)
    requested_kind, requested_name, _ = resource_parts(requested)
    if claimed_kind != requested_kind:
        return False
    return claimed_name == requested_name or (
        wildcard and requested_name.startswith(f"{claimed_name}/")
    )


def claim_for(root: Path, agent: str, lane: str) -> dict[str, Any]:
    claim_id = f"{slug(agent)}--{slug(lane)}"
    claim = parse_claim(state_dir() / "claims" / f"{claim_id}.claim")
    if claim.get("agent") != agent or claim.get("lane") != lane:
        raise Refusal("coordination claim owner mismatch")
    if Path(str(claim.get("worktree", ""))).resolve() != root:
        raise Refusal(f"coordination claim belongs to {claim.get('worktree', 'unknown')}")
    branch = git_text(root, "branch", "--show-current")
    if claim.get("branch") != branch:
        raise Refusal(f"coordination claim branch changed from {claim.get('branch')}")
    return claim


def authorize_claim(
    root: Path,
    claim: dict[str, Any],
    paths: list[str],
    resources: list[str],
) -> None:
    for requested in paths:
        if not any(path_covers(claimed, requested) for claimed in claim["files"]):
            raise Refusal(f"claim does not cover causal receipt path: {requested}")
    for requested in resources:
        resource_parts(requested)
        if not any(
            resource_covers(claimed, requested) for claimed in claim["resources"]
        ):
            raise Refusal(f"claim does not cover causal resource: {requested}")


def claimed_files_clean(root: Path, claim: dict[str, Any]) -> None:
    for value in claim["files"]:
        if Path(value).is_absolute():
            continue
        scope = path_scope(value)
        if not scope:
            raise Refusal("claim contains an empty file scope")
        status = git_text(
            root, "status", "--porcelain=v1", "--untracked-files=all", "--", scope
        )
        if status:
            raise Refusal(f"claimed path is dirty before causal receipt operation: {scope}")


def object_at(root: Path, commit: str, path: str) -> bytes:
    try:
        return git(root, "show", f"{commit}:{path}")
    except Refusal as error:
        raise Refusal(f"{path} is not committed at {commit}") from error


def require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Refusal(f"commit {ancestor} is not an ancestor of {descendant}")


def evidence_entry(root: Path, subject: str, role: str, value: str) -> dict[str, str]:
    relative, absolute = repo_path(root, value, must_exist=True)
    current = absolute.read_bytes()
    committed = object_at(root, subject, relative)
    if current != committed:
        raise Refusal(f"evidence differs from subject commit {subject}: {relative}")
    return {"path": relative, "role": role, "sha256": sha256(committed)}


def validate_prereg_owner(
    root: Path, prereg: dict[str, Any], agent: str | None, lane: str | None
) -> None:
    owner = prereg.get("owner")
    if not isinstance(owner, dict):
        raise Refusal("preregistration owner is missing")
    if set(owner) != OWNER_FIELDS:
        raise Refusal("preregistration owner fields are invalid")
    for key in ("agent", "branch", "lane", "worktree"):
        safe_text(f"owner {key}", str(owner.get(key, "")))
    if not Path(str(owner["worktree"])).is_absolute():
        raise Refusal("preregistration owner worktree is not absolute")
    if agent is not None or lane is not None:
        if Path(str(owner["worktree"])).resolve() != root:
            raise Refusal("preregistration belongs to another worktree")
        if owner.get("branch") != git_text(root, "branch", "--show-current"):
            raise Refusal("preregistration branch differs from the active branch")
    if agent is not None and owner.get("agent") != agent:
        raise Refusal("preregistration agent mismatch")
    if lane is not None and owner.get("lane") != lane:
        raise Refusal("preregistration lane mismatch")


def validate_preregistration(root: Path, prereg: dict[str, Any]) -> None:
    if set(prereg) != PREREG_FIELDS:
        raise Refusal("preregistration fields are invalid")
    if prereg.get("stage") != "preregistered":
        raise Refusal("preregistration stage is invalid")
    if not SAFE_ID.fullmatch(str(prereg.get("experiment_id", ""))):
        raise Refusal("preregistration experiment id is invalid")
    for key in (
        "statement",
        "falsifier",
        "intervention",
        "treatment_predicate",
        "control_predicate",
    ):
        safe_text(key.replace("_", " "), str(prereg.get(key, "")))
    utc_text("created UTC", str(prereg.get("created_utc", "")))
    base_commit = str(prereg.get("base_commit", ""))
    base_tree = str(prereg.get("base_tree", ""))
    if not GIT_OBJECT.fullmatch(base_commit) or not GIT_OBJECT.fullmatch(base_tree):
        raise Refusal("preregistration base commit or tree is invalid")
    if git_text(root, "rev-parse", f"{base_commit}^{{tree}}") != base_tree:
        raise Refusal("preregistration base tree does not match its commit")
    resources = prereg.get("resources")
    if not isinstance(resources, list) or not resources:
        raise Refusal("preregistration resources are missing")
    for resource in resources:
        if not isinstance(resource, str):
            raise Refusal("preregistration resource is invalid")
        resource_parts(resource)
    if resources != sorted(set(resources)):
        raise Refusal("preregistration resources are not canonical")


def validate_outcome(outcome: dict[str, Any]) -> None:
    if set(outcome) != OUTCOME_FIELDS:
        raise Refusal("outcome fields are invalid")
    if outcome.get("stage") != "outcome":
        raise Refusal("outcome stage is invalid")
    if not SAFE_ID.fullmatch(str(outcome.get("experiment_id", ""))):
        raise Refusal("outcome experiment id is invalid")
    utc_text("closed UTC", str(outcome.get("closed_utc", "")))
    treatment = safe_text("treatment result", str(outcome.get("treatment", "")))
    control = safe_text("control result", str(outcome.get("control", "")))
    if not RESULT.fullmatch(treatment) or not RESULT.fullmatch(control):
        raise Refusal("outcome treatment and control are invalid")
    verdict = outcome.get("verdict")
    if verdict not in {"supported", "falsified", "inconclusive"}:
        raise Refusal("outcome verdict is invalid")
    if verdict == "supported" and not (
        treatment.endswith("=PASS") and control.endswith("=PASS")
    ):
        raise Refusal("supported outcome lacks passing treatment and control")
    if verdict == "falsified" and not treatment.endswith("=FAIL"):
        raise Refusal("falsified outcome requires a failed treatment predicate")
    if not GIT_OBJECT.fullmatch(str(outcome.get("subject_commit", ""))):
        raise Refusal("outcome subject commit is invalid")
    if not GIT_OBJECT.fullmatch(str(outcome.get("subject_tree", ""))):
        raise Refusal("outcome subject tree is invalid")
    if not SHA256.fullmatch(str(outcome.get("prereg_sha256", ""))):
        raise Refusal("outcome preregistration digest is invalid")
    safe_text("outcome preregistration path", str(outcome.get("prereg_path", "")))


def open_experiment(args: argparse.Namespace) -> None:
    root = worktree()
    agent = safe_text("agent", args.agent)
    lane = safe_text("lane", args.lane)
    resources = sorted(set(args.resource))
    if not resources:
        raise Refusal("experiment-open requires at least one --resource")
    relative, receipt = repo_path(root, args.receipt)
    claim = claim_for(root, agent, lane)
    authorize_claim(root, claim, [relative], resources)
    claimed_files_clean(root, claim)
    experiment_id = args.id or (
        f"exp-{time.time_ns()}-{os.getpid()}-{secrets.token_hex(2)}"
    )
    if not SAFE_ID.fullmatch(experiment_id):
        raise Refusal("experiment id supports only letters, numbers, dot, underscore, and dash")
    head = git_text(root, "rev-parse", "HEAD")
    value: dict[str, Any] = {
        "base_commit": head,
        "base_tree": git_text(root, "rev-parse", "HEAD^{tree}"),
        "control_predicate": safe_text("control predicate", args.control_predicate),
        "created_utc": now_utc(),
        "experiment_id": experiment_id,
        "falsifier": safe_text("falsifier", args.falsifier),
        "intervention": safe_text("intervention", args.intervention),
        "owner": {
            "agent": agent,
            "branch": git_text(root, "branch", "--show-current"),
            "lane": lane,
            "worktree": str(root),
        },
        "resources": resources,
        "schema": PREREG_SCHEMA,
        "stage": "preregistered",
        "statement": safe_text("statement", args.statement),
        "treatment_predicate": safe_text(
            "treatment predicate", args.treatment_predicate
        ),
    }
    data = canonical_bytes(value)
    atomic_write(receipt, data)
    print(
        f"EXPERIMENT_OPEN id={experiment_id} receipt={relative} "
        f"prereg_sha256={sha256(data)} base_commit={head}"
    )


def close_experiment(args: argparse.Namespace) -> None:
    root = worktree()
    agent = safe_text("agent", args.agent)
    lane = safe_text("lane", args.lane)
    prereg_relative, prereg_path = repo_path(root, args.prereg, must_exist=True)
    outcome_relative, outcome_path = repo_path(root, args.outcome)
    if prereg_relative == outcome_relative:
        raise Refusal("outcome receipt must be distinct from preregistration")
    prereg, prereg_raw = load_json(prereg_path, PREREG_SCHEMA)
    validate_preregistration(root, prereg)
    validate_prereg_owner(root, prereg, agent, lane)
    resources = prereg.get("resources")
    if not isinstance(resources, list) or not all(isinstance(v, str) for v in resources):
        raise Refusal("preregistration resources are invalid")
    treatment_evidence_paths = [
        repo_path(root, value, must_exist=True)[0]
        for value in args.treatment_evidence
    ]
    control_evidence_paths = [
        repo_path(root, value, must_exist=True)[0] for value in args.control_evidence
    ]
    if set(treatment_evidence_paths) & set(control_evidence_paths):
        raise Refusal("treatment and control evidence paths must be distinct")
    claim = claim_for(root, agent, lane)
    authorize_claim(
        root,
        claim,
        [prereg_relative, outcome_relative, *treatment_evidence_paths, *control_evidence_paths],
        resources,
    )
    claimed_files_clean(root, claim)

    subject = git_text(root, "rev-parse", "HEAD")
    if object_at(root, subject, prereg_relative) != prereg_raw:
        raise Refusal("preregistration differs from the subject commit")
    parent = git_text(root, "rev-parse", "HEAD^")
    if object_at(root, parent, prereg_relative) != prereg_raw:
        raise Refusal("preregistration was not committed before the subject commit")
    base_commit = str(prereg.get("base_commit", ""))
    require_ancestor(root, base_commit, subject)
    if base_commit == subject:
        raise Refusal("experiment has no commit after preregistration")

    treatment = safe_text("treatment result", args.treatment)
    control = safe_text("control result", args.control)
    if not RESULT.fullmatch(treatment) or not RESULT.fullmatch(control):
        raise Refusal("treatment and control must use NAME=PASS or NAME=FAIL")
    verdict = args.verdict
    if verdict == "supported" and not (
        treatment.endswith("=PASS") and control.endswith("=PASS")
    ):
        raise Refusal("supported verdict requires passing treatment and sabotage control")
    if verdict == "falsified" and not treatment.endswith("=FAIL"):
        raise Refusal("falsified verdict requires a failed treatment predicate")
    if not args.treatment_evidence or not args.control_evidence:
        raise Refusal("close requires separate treatment and control evidence")
    evidence = [
        evidence_entry(root, subject, "treatment", value)
        for value in args.treatment_evidence
    ] + [
        evidence_entry(root, subject, "control", value)
        for value in args.control_evidence
    ]
    evidence.sort(key=lambda item: (item["role"], item["path"]))
    value: dict[str, Any] = {
        "closed_utc": now_utc(),
        "control": control,
        "evidence": evidence,
        "experiment_id": prereg.get("experiment_id"),
        "owner": prereg.get("owner"),
        "prereg_path": prereg_relative,
        "prereg_sha256": sha256(prereg_raw),
        "schema": OUTCOME_SCHEMA,
        "stage": "outcome",
        "subject_commit": subject,
        "subject_tree": git_text(root, "rev-parse", "HEAD^{tree}"),
        "treatment": treatment,
        "verdict": verdict,
    }
    data = canonical_bytes(value)
    atomic_write(outcome_path, data)
    print(
        f"EXPERIMENT_CLOSED id={value['experiment_id']} verdict={verdict} "
        f"outcome={outcome_relative} outcome_sha256={sha256(data)} "
        f"subject_commit={subject}"
    )


def verify_chain(
    root: Path,
    prereg_value: str,
    outcome_value: str,
    *,
    agent: str | None,
    lane: str | None,
    head: str | None,
    require_supported: bool,
) -> dict[str, str]:
    prereg_relative, prereg_path = repo_path(root, prereg_value, must_exist=True)
    outcome_relative, outcome_path = repo_path(root, outcome_value, must_exist=True)
    prereg, prereg_raw = load_json(prereg_path, PREREG_SCHEMA)
    outcome, outcome_raw = load_json(outcome_path, OUTCOME_SCHEMA)
    validate_preregistration(root, prereg)
    validate_prereg_owner(root, prereg, agent, lane)
    validate_outcome(outcome)
    if outcome.get("experiment_id") != prereg.get("experiment_id"):
        raise Refusal("outcome belongs to another experiment")
    if outcome.get("owner") != prereg.get("owner"):
        raise Refusal("outcome owner differs from preregistration")
    if outcome.get("prereg_path") != prereg_relative:
        raise Refusal("outcome preregistration path mismatch")
    if outcome.get("prereg_sha256") != sha256(prereg_raw):
        raise Refusal("outcome preregistration digest mismatch")
    if require_supported and outcome.get("verdict") != "supported":
        raise Refusal("handoff requires a supported causal experiment")
    checked_head = head or git_text(root, "rev-parse", "HEAD")
    if object_at(root, checked_head, prereg_relative) != prereg_raw:
        raise Refusal("preregistration is not committed unchanged at handoff HEAD")
    if object_at(root, checked_head, outcome_relative) != outcome_raw:
        raise Refusal("outcome is not committed unchanged at handoff HEAD")
    subject = str(outcome.get("subject_commit", ""))
    require_ancestor(root, subject, checked_head)
    if outcome.get("subject_tree") != git_text(root, "rev-parse", f"{subject}^{{tree}}"):
        raise Refusal("outcome subject tree mismatch")
    parent = git_text(root, "rev-parse", f"{subject}^")
    if object_at(root, parent, prereg_relative) != prereg_raw:
        raise Refusal("preregistration was not committed before the subject commit")
    require_ancestor(root, str(prereg.get("base_commit", "")), subject)

    evidence = outcome.get("evidence")
    if not isinstance(evidence, list):
        raise Refusal("outcome evidence is invalid")
    if not all(isinstance(item, dict) for item in evidence):
        raise Refusal("outcome evidence entry is invalid")
    if evidence != sorted(evidence, key=lambda item: (item.get("role"), item.get("path"))):
        raise Refusal("outcome evidence order is not canonical")
    roles: set[str] = set()
    paths_by_role: dict[str, set[str]] = {"treatment": set(), "control": set()}
    for item in evidence:
        role = str(item.get("role", ""))
        path = str(item.get("path", ""))
        digest = str(item.get("sha256", ""))
        if role not in {"treatment", "control"}:
            raise Refusal(f"unknown causal evidence role: {role}")
        if set(item) != {"path", "role", "sha256"}:
            raise Refusal("causal evidence entry has unexpected fields")
        if path in paths_by_role[role]:
            raise Refusal(f"duplicate causal evidence path: {path}")
        if sha256(object_at(root, subject, path)) != digest:
            raise Refusal(f"causal evidence digest mismatch: {path}")
        paths_by_role[role].add(path)
        roles.add(role)
    if roles != {"treatment", "control"}:
        raise Refusal("outcome must retain both treatment and control evidence")
    if paths_by_role["treatment"] & paths_by_role["control"]:
        raise Refusal("treatment and control evidence paths must remain distinct")
    if agent is not None and lane is not None:
        claim = claim_for(root, agent, lane)
        resources = [str(value) for value in prereg.get("resources", [])]
        all_paths = [prereg_relative, outcome_relative]
        all_paths.extend(path for values in paths_by_role.values() for path in values)
        authorize_claim(root, claim, all_paths, resources)
        claimed_files_clean(root, claim)
        changed = git_text(root, "diff", "--name-only", subject, checked_head).splitlines()
        for changed_path in changed:
            if changed_path == outcome_relative:
                continue
            if any(path_covers(scope, changed_path) for scope in claim["files"]):
                raise Refusal(
                    "claimed implementation changed after causal observation: "
                    f"{changed_path}"
                )
    return {
        "experiment_id": str(prereg.get("experiment_id", "")),
        "outcome_sha256": sha256(outcome_raw),
        "prereg_sha256": sha256(prereg_raw),
        "subject_commit": subject,
        "verdict": str(outcome.get("verdict", "")),
    }


def verify_experiment(args: argparse.Namespace) -> None:
    result = verify_chain(
        worktree(),
        args.prereg,
        args.outcome,
        agent=args.agent,
        lane=args.lane,
        head=args.head,
        require_supported=args.require_supported,
    )
    print(
        "CAUSAL_VERIFIED "
        + " ".join(f"{key}={value}" for key, value in result.items())
    )


def status_experiment(args: argparse.Namespace) -> None:
    root = worktree()
    prereg_relative, prereg_path = repo_path(root, args.prereg, must_exist=True)
    prereg, prereg_raw = load_json(prereg_path, PREREG_SCHEMA)
    validate_preregistration(root, prereg)
    committed = False
    try:
        committed = object_at(root, "HEAD", prereg_relative) == prereg_raw
    except Refusal:
        committed = False
    if not args.outcome:
        print(
            f"EXPERIMENT_STATUS id={prereg.get('experiment_id')} state=open "
            f"prereg_committed={'yes' if committed else 'no'} "
            f"prereg_sha256={sha256(prereg_raw)}"
        )
        return
    result = verify_chain(
        root,
        args.prereg,
        args.outcome,
        agent=None,
        lane=None,
        head=None,
        require_supported=False,
    )
    print(
        f"EXPERIMENT_STATUS id={result['experiment_id']} state={result['verdict']} "
        f"prereg_committed=yes prereg_sha256={result['prereg_sha256']} "
        f"outcome_sha256={result['outcome_sha256']} "
        f"subject_commit={result['subject_commit']}"
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)

    open_parser = commands.add_parser("open")
    open_parser.add_argument("--agent", required=True)
    open_parser.add_argument("--lane", required=True)
    open_parser.add_argument("--id")
    open_parser.add_argument("--receipt", required=True)
    open_parser.add_argument("--statement", required=True)
    open_parser.add_argument("--falsifier", required=True)
    open_parser.add_argument("--intervention", required=True)
    open_parser.add_argument("--treatment-predicate", required=True)
    open_parser.add_argument("--control-predicate", required=True)
    open_parser.add_argument("--resource", action="append", default=[])
    open_parser.set_defaults(function=open_experiment)

    close_parser = commands.add_parser("close")
    close_parser.add_argument("--agent", required=True)
    close_parser.add_argument("--lane", required=True)
    close_parser.add_argument("--prereg", required=True)
    close_parser.add_argument("--outcome", required=True)
    close_parser.add_argument(
        "--verdict", choices=("supported", "falsified", "inconclusive"), required=True
    )
    close_parser.add_argument("--treatment", required=True)
    close_parser.add_argument("--control", required=True)
    close_parser.add_argument("--treatment-evidence", action="append", default=[])
    close_parser.add_argument("--control-evidence", action="append", default=[])
    close_parser.set_defaults(function=close_experiment)

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--prereg", required=True)
    verify_parser.add_argument("--outcome", required=True)
    verify_parser.add_argument("--agent")
    verify_parser.add_argument("--lane")
    verify_parser.add_argument("--head")
    verify_parser.add_argument("--require-supported", action="store_true")
    verify_parser.set_defaults(function=verify_experiment)

    status_parser = commands.add_parser("status")
    status_parser.add_argument("--prereg", required=True)
    status_parser.add_argument("--outcome")
    status_parser.set_defaults(function=status_experiment)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except Refusal as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
