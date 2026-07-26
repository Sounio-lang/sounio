#!/usr/bin/env python3
"""Bridge Claude Code and Codex lifecycle hooks to bin/sounio-coord."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
CONFLICT_OWNER = re.compile(r"existing_claim=\S+ agent=(\S+) lane=(\S+)")


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


def safe_token(value: str, limit: int = 24) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]", "_", value)[:limit]
    return token or "unknown"


def worktree_token(root: Path) -> str:
    """A short, stable token identifying THIS worktree.

    Two things depend on this, both learned from issue #1477:

    1. `session_id` is sometimes absent from the hook event. Falling back to the
       literal string "unknown" put every agent in that state on the same lane
       `session-unknown`, where they collided with each other and blocked
       Edit/Write for entire sessions.
    2. A lane that does not name the worktree collides with ITSELF when one
       session works in more than one worktree: the claim is bound to the first
       worktree and every later tool call is refused with
       "claim belongs to worktree ...". Agents legitimately run in
       .claude/worktrees/<name> while a claim was registered against the repo
       root, so the paths never actually conflicted.

    Including the worktree in the lane makes both impossible.
    """
    return safe_token(hashlib.sha1(str(root.resolve()).encode()).hexdigest()[:10])


def run_coord(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SOUNIO_COORD_TTL_SECONDS"] = env.get(
        "SOUNIO_COORD_HOOK_TTL_SECONDS", "1800"
    )
    return subprocess.run(
        [str(root / "bin" / "sounio-coord"), *args],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


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
    root: Path,
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
        root,
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
    )


def main() -> int:
    args = parse_args()
    event = read_event()
    cwd = str(event.get("cwd") or os.getcwd())
    root = repo_root(cwd)
    if root is None or not (root / "bin" / "sounio-coord").is_file():
        return 0

    event_name = str(event.get("hook_event_name", ""))
    raw_session = str(event.get("session_id") or "").strip()
    agent = safe_token(args.agent)
    wt = worktree_token(root)
    # Lane is scoped to (session, worktree). See worktree_token for why both are
    # required. When session_id is absent the worktree token alone still keeps
    # concurrent agents apart, instead of funnelling them into "session-unknown".
    if raw_session:
        lane = f"session-{safe_token(raw_session)}-{wt}"
    else:
        lane = f"session-wt-{wt}"
    intent = f"active {agent} session"
    common = scope_args(agent, lane, intent)

    if event_name == "SessionEnd":
        run_coord(
            root,
            "release",
            "--agent",
            agent,
            "--lane",
            lane,
            "--reason",
            "agent session ended",
        )
        return 0

    if event_name == "PreToolUse":
        paths = extract_paths(event)
        if not paths:
            return 0
        result = run_coord(root, "scope", *common, "--files", *paths)
        if result.returncode != 0:
            notify_conflict(root, agent, lane, paths, result.stderr)
            sys.stderr.write(result.stderr or "coordination scope update failed\n")
            return 2
        return 0

    if event_name == "SessionStart":
        result = run_coord(root, "scope", *common)
    else:
        result = run_coord(
            root, "heartbeat", "--agent", agent, "--lane", lane
        )
        if result.returncode != 0:
            result = run_coord(root, "scope", *common)
    if result.returncode != 0:
        sys.stderr.write(f"sounio coordination warning: {result.stderr}")
        return 0

    if event_name == "SessionStart":
        print(
            f"Sounio coordination joined: agent={agent} lane={lane}. "
            "Use this same agent/lane with `bin/sounio-coord scope` before "
            "write-bearing Bash commands."
        )

    if event_name in {"UserPromptSubmit", "PostToolUse"}:
        inbox = run_coord(
            root, "inbox", "--agent", agent, "--lane", lane
        )
        lines = [line for line in inbox.stdout.splitlines() if line.startswith("MESSAGE ")]
        if lines:
            print("Sounio lane messages waiting for this agent:")
            print("\n".join(lines))
            print(
                "After handling one, acknowledge it with "
                f"bin/sounio-coord ack --agent {agent} --lane {lane} --message <id>."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
