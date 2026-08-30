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
import tempfile
import time
from pathlib import Path
from typing import Any


PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$", re.MULTILINE)
CONFLICT_OWNER = re.compile(r"existing_claim=\S+ agent=(\S+) lane=(\S+)")

# The harness kills a hook that overruns its configured timeout (10s in
# .claude/settings.json), and a killed PreToolUse hook stalls the tool call for
# the whole budget first. So the hook keeps its own, smaller deadline: every
# subprocess is bounded, and once the budget is gone the remaining coordination
# work is skipped rather than started. Coordination is advisory — arriving late
# is fine, blocking an agent for ten seconds is not.
BUDGET_SECONDS = float(os.getenv("SOUNIO_COORD_HOOK_BUDGET_SECONDS", "8"))

# `inbox` re-reads every message file on each call, so its cost grows with the
# store (~0.8s at 742 messages). PostToolUse fires on every single tool call,
# which made that the hook's dominant cost. Messages are still checked on every
# user turn, and at most this often during a long run of tool calls.
INBOX_INTERVAL_SECONDS = float(
    os.getenv("SOUNIO_COORD_INBOX_INTERVAL_SECONDS", "60")
)

# Everything the hook prints is injected into the agent's context. A lane that
# has never acked a broadcast currently sees 64 messages (~49KB, measured
# 2026-08-25) on its first turn, so only the newest are shown inline and the
# rest are pointed at.
INBOX_DISPLAY_LIMIT = int(os.getenv("SOUNIO_COORD_INBOX_DISPLAY_LIMIT", "20"))

TIMED_OUT = 124

# Identity-compared marker so a synthesised result can never be confused with a
# real exit code from sounio-coord.
SKIPPED_ARGS = ("<skipped>",)  # a tuple; subprocess.run always sets args to a list

_DEADLINE = time.monotonic() + BUDGET_SECONDS


def remaining_budget() -> float:
    return _DEADLINE - time.monotonic()


def warn(message: str) -> None:
    sys.stderr.write(f"sounio coordination warning: {message}\n")


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


def skipped(reason: str) -> subprocess.CompletedProcess[str]:
    """A result standing in for a call that was never made, or was cut short."""
    return subprocess.CompletedProcess(SKIPPED_ARGS, TIMED_OUT, "", reason)


def was_skipped(result: subprocess.CompletedProcess[str]) -> bool:
    """True only for results this module synthesised, never for a real exit code."""
    return result.args is SKIPPED_ARGS


def repo_root(cwd: str) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=max(0.5, min(3.0, remaining_budget())),
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
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


def run_coord(
    root: Path, *args: str, timeout: float = 4.0
) -> subprocess.CompletedProcess[str]:
    budget = min(timeout, remaining_budget())
    if budget <= 0.2:
        return skipped(f"skipped `{args[0] if args else '?'}`: hook budget exhausted")

    env = os.environ.copy()
    env["SOUNIO_COORD_TTL_SECONDS"] = env.get(
        "SOUNIO_COORD_HOOK_TTL_SECONDS", "1800"
    )
    try:
        return subprocess.run(
            [str(root / "bin" / "sounio-coord"), *args],
            cwd=root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=budget,
        )
    except subprocess.TimeoutExpired:
        return skipped(
            f"`sounio-coord {args[0] if args else '?'}` exceeded {budget:.1f}s"
        )
    except OSError as e:
        return skipped(f"could not run sounio-coord: {e}")


def scope_args(agent: str, lane: str, intent: str) -> list[str]:
    return ["--agent", agent, "--lane", lane, "--intent", intent]


def inbox_stamp(root: Path, agent: str, lane: str) -> Path:
    key = hashlib.md5(f"{root}\0{agent}\0{lane}".encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / f"sounio-coord-inbox-{key}.stamp"


def inbox_due(stamp: Path, interval: float) -> bool:
    if interval <= 0:
        return True
    try:
        return (time.time() - stamp.stat().st_mtime) >= interval
    except OSError:
        return True


def mark_inbox_checked(stamp: Path) -> None:
    try:
        stamp.touch()
    except OSError:
        pass


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
        timeout=2.0,
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
            timeout=4.0,
        )
        return 0

    if event_name == "PreToolUse":
        paths = extract_paths(event)
        if not paths:
            return 0
        result = run_coord(root, "scope", *common, "--files", *paths, timeout=4.0)
        if was_skipped(result):
            # Never block a write because coordination was slow or unavailable —
            # exit 2 here would deny the tool call outright.
            warn(f"{result.stderr.strip()}; proceeding without a lease")
            return 0
        if result.returncode != 0:
            notify_conflict(root, agent, lane, paths, result.stderr)
            sys.stderr.write(result.stderr or "coordination scope update failed\n")
            return 2
        return 0

    if event_name == "SessionStart":
        result = run_coord(root, "scope", *common, timeout=4.0)
    else:
        result = run_coord(
            root, "heartbeat", "--agent", agent, "--lane", lane, timeout=3.0
        )
        if result.returncode != 0 and not was_skipped(result):
            result = run_coord(root, "scope", *common, timeout=4.0)
    if result.returncode != 0:
        warn(result.stderr.strip() or "coordination update failed")
        return 0

    if event_name == "SessionStart":
        print(
            f"Sounio coordination joined: agent={agent} lane={lane}. "
            "Use this same agent/lane with `bin/sounio-coord scope` before "
            "write-bearing Bash commands."
        )

    if event_name in {"UserPromptSubmit", "PostToolUse"}:
        stamp = inbox_stamp(root, agent, lane)
        # A user turn is the point where waiting messages matter most, so it
        # always checks; the per-tool-call firehose is throttled.
        if event_name == "UserPromptSubmit" or inbox_due(stamp, INBOX_INTERVAL_SECONDS):
            inbox = run_coord(
                root, "inbox", "--agent", agent, "--lane", lane, timeout=4.0
            )
            if was_skipped(inbox):
                warn(inbox.stderr.strip())
                return 0
            mark_inbox_checked(stamp)
            lines = [
                line for line in inbox.stdout.splitlines() if line.startswith("MESSAGE ")
            ]
            if lines:
                withheld = 0
                if 0 < INBOX_DISPLAY_LIMIT < len(lines):
                    withheld = len(lines) - INBOX_DISPLAY_LIMIT
                    lines = lines[-INBOX_DISPLAY_LIMIT:]
                print("Sounio lane messages waiting for this agent:")
                print("\n".join(lines))
                if withheld:
                    print(
                        f"({withheld} older message(s) not shown — read them with "
                        f"bin/sounio-coord inbox --agent {agent} --lane {lane})"
                    )
                print(
                    "After handling one, acknowledge it with "
                    f"bin/sounio-coord ack --agent {agent} --lane {lane} --message <id>."
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
