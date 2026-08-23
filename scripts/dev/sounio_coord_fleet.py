#!/usr/bin/env python3
"""Persistent fleet slots backed by the detached Sounio agent supervisor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
RUNTIME_VERSION = "2026.08.23.3"
UUID_RE = re.compile(
    r"(?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)
SAFE_TOKEN = re.compile(r"[^A-Za-z0-9._-]+")


class FleetError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchPlan:
    agent: str
    lane: str
    session_id: str
    identity: str
    command: list[str]


def slug(value: str, limit: int = 96) -> str:
    cleaned = SAFE_TOKEN.sub("-", value).strip("-")
    if not cleaned:
        raise FleetError("identity becomes empty after normalization")
    return cleaned[:limit]


def session_lane(session_id: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", session_id)[:24] or "unknown"
    return f"session-{token}"


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
        raise FleetError(f"not an attached Git worktree: {cwd}")
    return Path(result.stdout.strip()).resolve()


def state_root(cwd: Path) -> Path:
    override = os.environ.get("SOUNIO_AGENTD_DIR")
    root = Path(override).expanduser() if override else git_common_dir(cwd) / "sounio-agentd"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(root, 0o700)
    return root.resolve()


def slot_paths(root: Path, slot: str) -> dict[str, Path]:
    directory = root / "fleet-slots"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    name = slug(slot)
    return {
        "mapping": directory / f"{name}.json",
        "lock": directory / f"{name}.lock",
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(4)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def command_argv_digest(command: list[str]) -> str:
    encoded = json.dumps(
        command, ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FleetError(f"cannot read fleet slot mapping {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FleetError(f"invalid fleet slot mapping: {path}")
    return value


def agentd_command() -> Path:
    sibling = Path(__file__).resolve().with_name("sounio-agentd-runtime")
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling
    override = os.environ.get("SOUNIO_AGENTD_COMMAND")
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FleetError("sounio-agentd-runtime is not installed beside the fleet launcher")


def parse_status(output: str) -> dict[str, str]:
    status: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            status[key] = value
    return status


def run_agentd(
    arguments: list[str], *, capture: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(agentd_command()), *arguments],
        check=False,
        capture_output=capture,
        text=True,
    )


def probe_mapping(mapping: dict[str, Any], root: Path) -> tuple[str, dict[str, str]]:
    required = (
        "agent",
        "lane",
        "session_id",
        "worktree",
        "instance_id",
        "command",
        "argv_digest",
    )
    if any(not isinstance(mapping.get(key), str) or not mapping[key] for key in required):
        return "drifted", {}
    result = run_agentd(
        [
            "status",
            "--agent",
            mapping["agent"],
            "--lane",
            mapping["lane"],
            "--cwd",
            mapping["worktree"],
            "--state-dir",
            str(root),
        ]
    )
    if result.returncode != 0:
        return "unreachable", {}
    status = parse_status(result.stdout)
    immutable = {
        "agent": mapping["agent"],
        "lane": mapping["lane"],
        "session_id": mapping["session_id"],
        "worktree": str(Path(mapping["worktree"]).resolve()),
        "instance_id": mapping["instance_id"],
        "command": mapping["command"],
        "argv_digest": mapping["argv_digest"],
    }
    observed = dict(status)
    if observed.get("worktree"):
        observed["worktree"] = str(Path(observed["worktree"]).resolve())
    if any(observed.get(key) != value for key, value in immutable.items()):
        return "drifted", status
    if status.get("state") != "active":
        return "unreachable", status
    return "active", status


def latest_uuid(paths: list[Path]) -> str | None:
    candidates: list[tuple[int, str]] = []
    for path in paths:
        try:
            match = UUID_RE.search(path.name)
            if match and path.stat().st_size > 0:
                candidates.append((path.stat().st_mtime_ns, match.group("uuid").lower()))
        except OSError:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def claude_project_dir(home: Path, cwd: Path) -> Path:
    encoded = re.sub(r"[^A-Za-z0-9_-]", "-", str(cwd.resolve()))
    return home / ".claude" / "projects" / encoded


def latest_codex_uuid(paths: list[Path], cwd: Path) -> str | None:
    candidates: list[tuple[int, str]] = []
    expected_cwd = cwd.resolve()
    for path in paths:
        try:
            if path.stat().st_size == 0:
                continue
            with path.open(encoding="utf-8") as handle:
                metadata = json.loads(handle.readline())
            payload = metadata.get("payload", {})
            if metadata.get("type") != "session_meta" or not isinstance(payload, dict):
                continue
            if payload.get("source") == "exec" or "exec" in str(payload.get("originator", "")):
                continue
            recorded_cwd = payload.get("cwd")
            if not isinstance(recorded_cwd, str) or Path(recorded_cwd).resolve() != expected_cwd:
                continue
            session_id = payload.get("session_id") or payload.get("id")
            if not isinstance(session_id, str) or UUID_RE.fullmatch(session_id) is None:
                continue
            candidates.append((path.stat().st_mtime_ns, session_id.lower()))
        except (json.JSONDecodeError, OSError):
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def require_program(names: list[str]) -> str:
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return resolved
    raise FleetError(f"required agent command is not installed: {' or '.join(names)}")


def resolve_kind(kind: str, home: Path, slot: str, cwd: Path) -> LaunchPlan:
    if kind == "claude":
        executable = require_program(["claude"])
        existing = latest_uuid(list(claude_project_dir(home, cwd).glob("*.jsonl")))
        if existing:
            command = [executable, "--resume", existing, "--setting-sources", "user,local"]
            session_id = existing
            identity = "exact"
        else:
            session_id = str(uuid.uuid4())
            command = [
                executable,
                "--session-id",
                session_id,
                "--setting-sources",
                "user,local",
            ]
            identity = "exact"
        return LaunchPlan("claude", session_lane(session_id), session_id, identity, command)

    if kind == "codex":
        executable = require_program(["codex"])
        existing = latest_codex_uuid(
            list((home / ".codex" / "sessions").glob("**/*.jsonl")), cwd
        )
        if existing:
            session_id = existing
            command = [executable, "resume", existing]
            identity = "exact"
        else:
            # Codex currently has no caller-supplied UUID for a fresh TUI session.
            # The slot mapping still prevents duplicate launches and survives tmux;
            # the next generation resumes the real persisted UUID exactly.
            session_id = str(uuid.uuid4())
            command = [executable]
            identity = "bootstrap"
        return LaunchPlan("codex", session_lane(session_id), session_id, identity, command)

    if kind == "kimi":
        executable = require_program(["kimi", "kimi-cli"])
        session_id = str(uuid.uuid4())
        return LaunchPlan(
            "kimi",
            f"fleet-{slug(slot)}",
            session_id,
            "standalone",
            [executable, "--continue"],
        )

    if kind == "grok":
        executable = require_program(["grok"])
        session_id = str(uuid.uuid4())
        return LaunchPlan(
            "grok", f"fleet-{slug(slot)}", session_id, "standalone", [executable]
        )

    if kind == "cursor":
        executable = require_program(["cursor-agent"])
        session_id = str(uuid.uuid4())
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_continue-fallback",
            executable,
        ]
        return LaunchPlan(
            "cursor", f"fleet-{slug(slot)}", session_id, "standalone", command
        )

    if kind == "empryo":
        executable = require_program(["em", "empryo"])
        session_id = str(uuid.uuid4())
        return LaunchPlan(
            "empryo", f"fleet-{slug(slot)}", session_id, "standalone", [executable]
        )

    raise FleetError(f"unsupported fleet agent kind: {kind}")


def mapping_for(
    slot: str,
    plan: LaunchPlan,
    worktree: Path,
    instance_id: str,
    command_name: str,
    argv_digest: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_VERSION,
        "runtime_version": RUNTIME_VERSION,
        "slot": slot,
        "agent": plan.agent,
        "lane": plan.lane,
        "session_id": plan.session_id,
        "identity": plan.identity,
        "worktree": str(worktree),
        "instance_id": instance_id,
        "command": command_name,
        "argv_digest": argv_digest,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def start_or_reuse(
    slot: str, plan: LaunchPlan, worktree: Path
) -> tuple[dict[str, Any], str]:
    root = state_root(worktree)
    paths = slot_paths(root, slot)
    with paths["lock"].open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if paths["mapping"].is_file():
            existing = read_json(paths["mapping"])
            state, _ = probe_mapping(existing, root)
            if state == "active":
                return existing, "reattached"
            if state == "drifted":
                raise FleetError(
                    f"fleet slot {slot} identity drifted; refusing to attach or replace it"
                )
            paths["mapping"].unlink(missing_ok=True)

        result = run_agentd(
            [
                "start",
                "--agent",
                plan.agent,
                "--lane",
                plan.lane,
                "--session-id",
                plan.session_id,
                "--cwd",
                str(worktree),
                "--state-dir",
                str(root),
                "--",
                *plan.command,
            ]
        )
        if result.returncode != 0:
            raise FleetError((result.stderr or result.stdout).strip() or "agentd start failed")
        start_status = result.stdout.strip()
        status_result = run_agentd(
            [
                "status",
                "--agent",
                plan.agent,
                "--lane",
                plan.lane,
                "--cwd",
                str(worktree),
                "--state-dir",
                str(root),
            ]
        )
        if status_result.returncode != 0:
            raise FleetError("new supervisor did not return a verified status")
        status = parse_status(status_result.stdout)
        mapping = mapping_for(
            slot,
            plan,
            worktree,
            status.get("instance_id", ""),
            status.get("command", ""),
            status.get("argv_digest", ""),
        )
        if (
            not mapping["instance_id"]
            or not mapping["command"]
            or not mapping["argv_digest"]
        ):
            raise FleetError("new supervisor omitted immutable identity fields")
        atomic_write_json(paths["mapping"], mapping)
        if start_status.startswith("AGENTD_ALREADY_RUNNING"):
            return mapping, "recovered"
        return mapping, "started"


def attach(mapping: dict[str, Any], root: Path) -> int:
    arguments = [
        "attach",
        "--agent",
        mapping["agent"],
        "--lane",
        mapping["lane"],
        "--cwd",
        mapping["worktree"],
        "--state-dir",
        str(root),
    ]
    last_result: subprocess.CompletedProcess[str] | None = None
    for attempt in range(20):
        last_result = run_agentd(arguments, capture=False)
        if last_result.returncode == 0:
            return 0
        state, _ = probe_mapping(mapping, root)
        if state != "active" or attempt == 19:
            break
        time.sleep(0.1)
    return last_result.returncode if last_result is not None else 1


def launch(args: argparse.Namespace, plan: LaunchPlan) -> int:
    worktree = Path(args.cwd).resolve()
    if not worktree.is_dir():
        raise FleetError(f"worktree does not exist: {worktree}")
    mapping, action = start_or_reuse(args.slot, plan, worktree)
    print(
        "FLEET_SLOT "
        f"action={action} slot={args.slot} agent={mapping['agent']} lane={mapping['lane']} "
        f"session_id={mapping['session_id']} identity={mapping['identity']} "
        f"instance_id={mapping['instance_id']}",
        flush=True,
    )
    if args.no_attach:
        return 0
    return attach(mapping, state_root(Path(mapping["worktree"])))


def explicit_plan(args: argparse.Namespace) -> LaunchPlan:
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise FleetError("launch requires a command after --")
    lane = args.lane or (
        session_lane(args.session_id)
        if args.identity in {"exact", "bootstrap"}
        else f"fleet-{slug(args.slot)}"
    )
    return LaunchPlan(args.agent, lane, args.session_id, args.identity, command)


def status_command(args: argparse.Namespace) -> int:
    worktree = Path(args.cwd).resolve()
    root = state_root(worktree)
    directory = root / "fleet-slots"
    mappings = [] if not directory.is_dir() else sorted(directory.glob("*.json"))
    count = 0
    unhealthy = 0
    for path in mappings:
        mapping = read_json(path)
        if args.slot and mapping.get("slot") != args.slot:
            continue
        state, status = probe_mapping(mapping, root)
        if state != "active":
            unhealthy += 1
        print(
            "FLEET_SLOT_STATUS "
            f"state={state} slot={mapping.get('slot', '-')} "
            f"agent={mapping.get('agent', '-')} lane={mapping.get('lane', '-')} "
            f"session_id={mapping.get('session_id', '-')} "
            f"identity={mapping.get('identity', '-')} "
            f"instance_id={mapping.get('instance_id', '-')} "
            f"argv_digest={mapping.get('argv_digest', '-')} "
            f"harness_pid={status.get('harness_pid', '-')} "
            f"attached_clients={status.get('attached_clients', '0')} "
            f"worktree={mapping.get('worktree', '-')}",
        )
        count += 1
    print(f"fleet_slots={count} unhealthy={unhealthy}")
    return 1 if unhealthy else 0


def stop_command(args: argparse.Namespace) -> int:
    worktree = Path(args.cwd).resolve()
    root = state_root(worktree)
    paths = slot_paths(root, args.slot)
    with paths["lock"].open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not paths["mapping"].is_file():
            print(f"FLEET_SLOT_STOPPED slot={args.slot} state=absent")
            return 0
        mapping = read_json(paths["mapping"])
        state, _ = probe_mapping(mapping, root)
        if state == "drifted":
            raise FleetError(f"fleet slot {args.slot} identity drifted; refusing to stop it")
        if state == "active":
            result = run_agentd(
                [
                    "stop",
                    "--agent",
                    mapping["agent"],
                    "--lane",
                    mapping["lane"],
                    "--cwd",
                    mapping["worktree"],
                    "--state-dir",
                    str(root),
                ]
            )
            if result.returncode != 0:
                raise FleetError((result.stderr or result.stdout).strip())
        paths["mapping"].unlink(missing_ok=True)
        print(f"FLEET_SLOT_STOPPED slot={args.slot} state={state}")
    return 0


def plan_kind_command(args: argparse.Namespace) -> int:
    plan = resolve_kind(
        args.kind,
        Path(args.home).expanduser().resolve(),
        args.slot,
        Path(args.cwd).resolve(),
    )
    print(f"agent={plan.agent}")
    print(f"lane={plan.lane}")
    print(f"session_id={plan.session_id}")
    print(f"identity={plan.identity}")
    print(f"command={shlex.join(plan.command)}")
    return 0


def continue_fallback(arguments: list[str]) -> int:
    if len(arguments) != 1:
        raise FleetError("_continue-fallback requires one executable")
    executable = arguments[0]
    first = subprocess.run([executable, "--continue"], check=False)
    if first.returncode == 0:
        return 0
    os.execv(executable, [executable])
    return 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sounio-fleet-agent")
    subparsers = root.add_subparsers(dest="command_name", required=True)

    subparsers.add_parser("runtime-version")

    launch_parser = subparsers.add_parser("launch")
    launch_parser.add_argument("--slot", required=True)
    launch_parser.add_argument("--agent", required=True)
    launch_parser.add_argument("--lane")
    launch_parser.add_argument("--session-id", required=True)
    launch_parser.add_argument(
        "--identity", choices=("exact", "bootstrap", "standalone"), default="exact"
    )
    launch_parser.add_argument("--cwd", required=True)
    launch_parser.add_argument("--no-attach", action="store_true")
    launch_parser.add_argument("command", nargs=argparse.REMAINDER)

    kind_parser = subparsers.add_parser("launch-kind")
    kind_parser.add_argument("--slot", required=True)
    kind_parser.add_argument(
        "--kind",
        required=True,
        choices=("claude", "codex", "kimi", "grok", "cursor", "empryo"),
    )
    kind_parser.add_argument("--home", required=True)
    kind_parser.add_argument("--cwd", required=True)
    kind_parser.add_argument("--no-attach", action="store_true")

    plan_parser = subparsers.add_parser("plan-kind")
    plan_parser.add_argument("--slot", required=True)
    plan_parser.add_argument(
        "--kind",
        required=True,
        choices=("claude", "codex", "kimi", "grok", "cursor", "empryo"),
    )
    plan_parser.add_argument("--home", required=True)
    plan_parser.add_argument("--cwd", default=os.getcwd())

    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--cwd", default=os.getcwd())
    status_parser.add_argument("--slot")

    stop_parser = subparsers.add_parser("stop")
    stop_parser.add_argument("--cwd", default=os.getcwd())
    stop_parser.add_argument("--slot", required=True)

    fallback_parser = subparsers.add_parser("_continue-fallback")
    fallback_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command_name == "runtime-version":
        print(f"protocol_version={PROTOCOL_VERSION}")
        print(f"runtime_version={RUNTIME_VERSION}")
        return 0
    if args.command_name == "launch":
        return launch(args, explicit_plan(args))
    if args.command_name == "launch-kind":
        plan = resolve_kind(
            args.kind,
            Path(args.home).expanduser().resolve(),
            args.slot,
            Path(args.cwd).resolve(),
        )
        return launch(args, plan)
    if args.command_name == "plan-kind":
        return plan_kind_command(args)
    if args.command_name == "status":
        return status_command(args)
    if args.command_name == "stop":
        return stop_command(args)
    if args.command_name == "_continue-fallback":
        return continue_fallback(args.arguments)
    raise FleetError(f"unknown command: {args.command_name}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FleetError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
