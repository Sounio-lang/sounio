#!/usr/bin/env python3
"""Bridge Claude Code and Codex lifecycle hooks to bin/sounio-coord."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
CONFLICT_OWNER = re.compile(r"existing_claim=\S+ agent=(\S+) lane=(\S+)")
MESSAGE_ID = re.compile(r"^MESSAGE id=(\S+) ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    return parser.parse_args()


def read_event() -> dict[str, Any]:
    try:
        value = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def repo_root(cwd: str) -> Path | None:
    result = subprocess.run(
        ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def target_path(cwd: str, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cwd) / path
    return Path(os.path.abspath(path))


def target_repo_root(path: Path) -> Path | None:
    probe = path if path.is_dir() else path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return repo_root(str(probe))


def git_common_dir(root: Path) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = root / common
    return common.resolve()


def paths_for_target(
    cwd: str, session_root: Path, paths: list[str]
) -> tuple[Path, list[str]] | None:
    roots: dict[Path, list[str]] = {}
    session_common = git_common_dir(session_root)
    for value in paths:
        absolute = target_path(cwd, value)
        root = target_repo_root(absolute)
        if root is None or git_common_dir(root) != session_common:
            return None
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            return None
        roots.setdefault(root, []).append(str(relative))
    if len(roots) != 1:
        return None
    root, relative_paths = next(iter(roots.items()))
    return root, list(dict.fromkeys(relative_paths))


def safe_token(value: str, limit: int = 24) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:limit]
    return token or "unknown"


def run_coord(
    root: Path, *args: str, worktree: Path | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SOUNIO_COORD_TTL_SECONDS"] = env.get(
        "SOUNIO_COORD_HOOK_TTL_SECONDS", "1800"
    )
    return subprocess.run(
        [str(root / "bin" / "sounio-coord"), *args],
        cwd=worktree or root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def tmux_endpoint(root: Path) -> tuple[str, str] | None:
    tmux_value = os.environ.get("TMUX", "")
    pane = os.environ.get("TMUX_PANE", "")
    socket = tmux_value.partition(",")[0]
    if not socket or not pane:
        return None
    try:
        result = subprocess.run(
            [
                "tmux",
                "-S",
                socket,
                "display-message",
                "-p",
                "-t",
                pane,
                "#{pane_id}|#{pane_current_path}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    pane_id, separator, pane_cwd = result.stdout.strip().partition("|")
    if not separator or not pane_id or not pane_cwd:
        return None
    pane_root = repo_root(pane_cwd)
    if pane_root is None or pane_root.resolve() != root.resolve():
        return None
    return socket, pane_id


def agentd_endpoint() -> tuple[str, str] | None:
    socket_path = os.environ.get("SOUNIO_AGENTD_SOCKET", "")
    token_file = os.environ.get("SOUNIO_AGENTD_TOKEN_FILE", "")
    if not socket_path or not token_file:
        return None
    socket = Path(socket_path)
    token = Path(token_file)
    if not socket.exists() or not token.is_file():
        return None
    return str(socket.resolve()), str(token.resolve())


def process_worktree(context_root: Path) -> Path:
    supervised = os.environ.get("SOUNIO_AGENTD_WORKTREE", "")
    if not supervised:
        return context_root
    physical_root = repo_root(supervised)
    if physical_root is None:
        return context_root
    if git_common_dir(physical_root) != git_common_dir(context_root):
        return context_root
    return physical_root


def refresh_delivery_endpoint(
    tool_root: Path, root: Path, agent: str, lane: str
) -> None:
    if agent.startswith("claude"):
        harness = "claude"
    elif agent.startswith("codex"):
        harness = "codex"
    else:
        return
    ttl = os.environ.get("SOUNIO_COORD_HOOK_TTL_SECONDS", "1800")
    supervised = agentd_endpoint()
    if supervised is not None:
        socket, token_file = supervised
        result = run_coord(
            tool_root,
            "endpoint-register",
            "--agent",
            agent,
            "--lane",
            lane,
            "--harness",
            harness,
            "--transport",
            "agentd",
            "--address",
            socket,
            "--socket",
            socket,
            "--token-file",
            token_file,
            "--ttl-seconds",
            ttl,
            worktree=root,
        )
        if result.returncode != 0:
            sys.stderr.write(
                "sounio coordination agentd endpoint warning: "
                f"{result.stderr or result.stdout}"
            )
        return

    endpoint = tmux_endpoint(root)
    if endpoint is None:
        return
    socket, pane = endpoint
    result = run_coord(
        tool_root,
        "endpoint-register",
        "--agent",
        agent,
        "--lane",
        lane,
        "--harness",
        harness,
        "--transport",
        "tmux",
        "--address",
        pane,
        "--socket",
        socket,
        "--ttl-seconds",
        ttl,
        worktree=root,
    )
    if result.returncode != 0:
        sys.stderr.write(
            "sounio coordination delivery endpoint warning: "
            f"{result.stderr or result.stdout}"
        )


def process_identity() -> tuple[int, str, str, str, str] | None:
    pid = os.getppid()
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        _, separator, tail = stat.rpartition(")")
        fields = tail.split()
        pid_start = fields[19]
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        pid_namespace = os.readlink("/proc/self/ns/pid")
        host = os.uname().nodename
    except (IndexError, OSError):
        return None
    if not separator or not pid_start or not boot_id or not pid_namespace or not host:
        return None
    return pid, pid_start, boot_id, pid_namespace, host


def refresh_process_presence(
    tool_root: Path,
    process_root: Path,
    agent: str,
    lane: str,
    session_id: str,
    claim_root: Path | None = None,
) -> bool:
    identity = process_identity()
    if identity is None:
        sys.stderr.write(
            "sounio coordination process-presence warning: "
            "could not identify the harness process\n"
        )
        return False
    if agent.startswith("claude"):
        harness = "claude"
    elif agent.startswith("codex"):
        harness = "codex"
    else:
        return False
    pid, pid_start, boot_id, pid_namespace, host = identity
    ttl = os.environ.get("SOUNIO_COORD_HOOK_TTL_SECONDS", "1800")
    result = run_coord(
        tool_root,
        "presence-register",
        "--agent",
        agent,
        "--lane",
        lane,
        "--harness",
        harness,
        "--session-id",
        session_id,
        "--pid",
        str(pid),
        "--pid-start",
        pid_start,
        "--boot-id",
        boot_id,
        "--pid-namespace",
        pid_namespace,
        "--host",
        host,
        "--ttl-seconds",
        ttl,
        worktree=process_root,
    )
    if result.returncode != 0 and "claim not found:" in result.stderr:
        scope = run_coord(
            tool_root,
            "scope",
            *scope_args(agent, lane, f"active {agent} session"),
            worktree=claim_root or process_root,
        )
        if scope.returncode == 0:
            result = run_coord(
                tool_root,
                "presence-register",
                "--agent",
                agent,
                "--lane",
                lane,
                "--harness",
                harness,
                "--session-id",
                session_id,
                "--pid",
                str(pid),
                "--pid-start",
                pid_start,
                "--boot-id",
                boot_id,
                "--pid-namespace",
                pid_namespace,
                "--host",
                host,
                "--ttl-seconds",
                ttl,
                worktree=process_root,
            )
    if result.returncode != 0:
        sys.stderr.write(
            "sounio coordination process-presence warning: "
            f"{result.stderr or result.stdout}"
        )
        return False
    return True


def scope_args(agent: str, lane: str, intent: str) -> list[str]:
    return ["--agent", agent, "--lane", lane, "--intent", intent]


def extract_paths(event: dict[str, Any]) -> list[str]:
    tool_name = str(event.get("tool_name", ""))
    tool_input = event.get("tool_input", {})
    paths: list[str] = []

    if isinstance(tool_input, dict):
        for key in ("file_path", "notebook_path"):
            value = tool_input.get(key)
            if isinstance(value, str) and value:
                paths.append(value)

    if tool_name in {"apply_patch", "Edit", "Write"}:
        patch = ""
        if isinstance(tool_input, str):
            patch = tool_input
        elif isinstance(tool_input, dict):
            for key in ("patch", "input"):
                value = tool_input.get(key)
                if isinstance(value, str):
                    patch = value
                    break
        paths.extend(PATCH_PATH.findall(patch))

    return list(dict.fromkeys(paths))


def notify_conflict(
    tool_root: Path,
    worktree: Path,
    agent: str,
    lane: str,
    paths: list[str],
    stderr: str,
) -> None:
    owner = CONFLICT_OWNER.search(stderr)
    if not owner:
        return
    to_agent, to_lane = owner.groups()
    message = f"Write conflict requested by {agent}/{lane}: {', '.join(paths)}"
    run_coord(
        tool_root,
        "send",
        "--agent",
        agent,
        "--lane",
        lane,
        "--to-agent",
        to_agent,
        "--to-lane",
        to_lane,
        "--kind",
        "request",
        "--message",
        message,
        worktree=worktree,
    )


def main() -> int:
    args = parse_args()
    event = read_event()
    cwd = str(event.get("cwd") or os.getcwd())
    root = repo_root(cwd)
    if root is None:
        return 0
    tool_root = repo_root(str(Path(__file__).resolve().parent))
    if (
        tool_root is None
        or git_common_dir(tool_root) != git_common_dir(root)
        or not (tool_root / "bin" / "sounio-coord").is_file()
    ):
        tool_root = root
    if not (tool_root / "bin" / "sounio-coord").is_file():
        return 0

    event_name = str(event.get("hook_event_name", ""))
    raw_session_id = str(event.get("session_id", "unknown"))
    session_id = safe_token(raw_session_id)
    agent = safe_token(args.agent)
    lane = f"session-{session_id}"
    intent = f"active {agent} session"
    common = scope_args(agent, lane, intent)
    presence_root = process_worktree(root)

    if event_name == "SessionEnd":
        if not refresh_process_presence(
            tool_root, presence_root, agent, lane, raw_session_id, claim_root=root
        ):
            sys.stderr.write(
                "coordination refused: this process cannot end another live "
                "lane generation\n"
            )
            return 2
        run_coord(
            tool_root,
            "endpoint-unregister",
            "--agent",
            agent,
            "--lane",
            lane,
            worktree=presence_root,
        )
        run_coord(
            tool_root,
            "presence-unregister",
            "--agent",
            agent,
            "--lane",
            lane,
            worktree=presence_root,
        )
        run_coord(
            tool_root,
            "release",
            "--agent",
            agent,
            "--lane",
            lane,
            "--reason",
            "agent session ended",
            worktree=root,
        )
        return 0

    if event_name == "PreToolUse":
        paths = extract_paths(event)
        if not paths:
            return 0
        target = paths_for_target(cwd, root, paths)
        if target is None:
            sys.stderr.write(
                "coordination refused: write paths must resolve to one worktree "
                "attached to the current Sounio repository\n"
            )
            return 2
        target_root, target_paths = target
        if not refresh_process_presence(
            tool_root, presence_root, agent, lane, raw_session_id, claim_root=root
        ):
            sys.stderr.write(
                "coordination refused: this process does not own the live "
                "lane generation\n"
            )
            return 2

        result = run_coord(
            tool_root,
            "authorize",
            "--agent",
            agent,
            "--files",
            *target_paths,
            worktree=target_root,
        )
        if result.returncode == 0:
            refresh_delivery_endpoint(tool_root, presence_root, agent, lane)
            return 0

        if target_root == root:
            result = run_coord(
                tool_root,
                "scope",
                *common,
                "--files",
                *target_paths,
                worktree=root,
            )
        if result.returncode != 0:
            notify_conflict(
                tool_root, root, agent, lane, target_paths, result.stderr
            )
            sys.stderr.write(result.stderr or "coordination scope update failed\n")
            return 2
        refresh_delivery_endpoint(tool_root, presence_root, agent, lane)
        return 0

    if event_name == "SessionStart":
        result = run_coord(tool_root, "scope", *common, worktree=root)
    else:
        result = run_coord(
            tool_root,
            "heartbeat",
            "--agent",
            agent,
            "--lane",
            lane,
            worktree=root,
        )
        if result.returncode != 0:
            result = run_coord(tool_root, "scope", *common, worktree=root)
    claim_is_in_attached_worktree = (
        result.returncode != 0 and "claim belongs to worktree " in result.stderr
    )
    if result.returncode != 0 and not claim_is_in_attached_worktree:
        sys.stderr.write(f"sounio coordination warning: {result.stderr}")
        return 0

    if not refresh_process_presence(
        tool_root, presence_root, agent, lane, raw_session_id, claim_root=root
    ):
        return 2
    refresh_delivery_endpoint(tool_root, presence_root, agent, lane)

    if event_name == "SessionStart":
        print(
            f"Sounio coordination joined: agent={agent} lane={lane}. "
            "Use this same agent/lane with `bin/sounio-coord scope` before "
            "write-bearing Bash commands."
        )

    if event_name in {"UserPromptSubmit", "PostToolUse"}:
        inbox = run_coord(
            tool_root,
            "inbox",
            "--agent",
            agent,
            "--lane",
            lane,
            "--directed-only",
            "--newest-first",
            "--limit",
            "12",
            worktree=root,
        )
        lines = [line for line in inbox.stdout.splitlines() if line.startswith("MESSAGE ")]
        omitted = 0
        for line in inbox.stdout.splitlines():
            if line.startswith("inbox_omitted="):
                try:
                    omitted = int(line.partition("=")[2])
                except ValueError:
                    omitted = 0
        if lines:
            print("Recent directed Sounio lane messages waiting for this agent:")
            print("\n".join(lines))
            message_ids = [
                match.group(1)
                for line in lines
                if (match := MESSAGE_ID.match(line)) is not None
            ]
            if message_ids:
                injection = run_coord(
                    tool_root,
                    "injected",
                    "--agent",
                    agent,
                    "--lane",
                    lane,
                    "--messages",
                    *message_ids,
                    worktree=root,
                )
                if injection.returncode != 0:
                    sys.stderr.write(
                        "sounio coordination injection receipt warning: "
                        f"{injection.stderr}"
                    )
            if omitted:
                print(
                    f"{omitted} older directed message(s) omitted. Inspect them with "
                    f"`bin/sounio-coord inbox --agent {agent} --lane {lane} "
                    "--directed-only --newest-first`."
                )
            request_ids = [
                match.group(1)
                for line in lines
                if " kind=request " in line
                and (match := MESSAGE_ID.match(line)) is not None
            ]
            if request_ids:
                print(
                    "Reply on-thread with "
                    f"`bin/sounio-coord send --agent {agent} --lane {lane} "
                    "--kind reply --reply-to <request-id> --message \"<text>\"`. "
                    f"Pending request ids: {', '.join(request_ids)}."
                )
            print(
                "After handling one, acknowledge it with "
                f"bin/sounio-coord ack --agent {agent} --lane {lane} --message <id>."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
