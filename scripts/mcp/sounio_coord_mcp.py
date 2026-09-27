#!/usr/bin/env python3
"""MCP server: agent talk + attention bus over bin/sounio-coord.

Exposes the live Sounio coordination bus (claims, send/inbox/ack) and the
Attention Charter P0 snapshot so Cursor / Claude agents can coordinate without
guessing who owns which files.

Prerequisites:
  pip install "mcp[cli]"
  Run from a Sounio worktree (or set SOUNIO_REPO_ROOT).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "sounio-coord",
    instructions=(
        "Sounio multi-agent coordination bus. Use coord_brief / attention_p0 "
        "before writing. Use coord_claim before edits. Use coord_send / "
        "coord_inbox / coord_ack to talk to other agents. Equation: 5=1+2 — "
        "only P0 work closing compiler sovereignty (1) or epistemic honesty (2) "
        "may hold write attention. See .claude/ATTENTION_CHARTER.md."
    ),
)


def _repo_root() -> Path:
    env = os.environ.get("SOUNIO_REPO_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    # scripts/mcp/sounio_coord_mcp.py → repo root
    candidate = here.parents[2]
    if (candidate / "bin" / "sounio-coord").is_file():
        return candidate
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    raise RuntimeError("cannot locate Sounio repo root; set SOUNIO_REPO_ROOT")


def _coord(*args: str, timeout: float = 30.0) -> dict[str, Any]:
    root = _repo_root()
    binary = root / "bin" / "sounio-coord"
    if not binary.is_file():
        return {"ok": False, "error": f"missing {binary}"}
    proc = subprocess.run(
        [str(binary), *args],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "args": list(args),
        "repo_root": str(root),
    }


@mcp.tool()
async def attention_p0() -> dict[str, Any]:
    """Return the Attention Charter equation and current P0 queue.

    Read this before starting write-bearing work. Only slots that close
    horizon 1 (compiler) or 2 (epistemic honesty) may become active_p0.
    """

    root = _repo_root()
    path = root / ".claude" / "attention_p0.v1.json"
    charter = root / ".claude" / "ATTENTION_CHARTER.md"
    if not path.is_file():
        return {"ok": False, "error": f"missing {path}"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "ok": True,
        "equation": data.get("equation", "5 = 1 + 2"),
        "active_p0": data.get("active_p0"),
        "charter_path": str(charter),
        "p0_path": str(path),
        "policy": data.get("policy", {}),
        "slots": data.get("slots", []),
        "notes": data.get("notes", []),
    }


@mcp.tool()
async def coord_brief() -> dict[str, Any]:
    """Startup-sized coordination summary (claims, conflicts, worktrees)."""

    return _coord("brief")


@mcp.tool()
async def coord_status(all_worktrees: bool = False) -> dict[str, Any]:
    """Full coordination status. Set all_worktrees=true for a slower full scan."""

    if all_worktrees:
        return _coord("status", "--all-worktrees", timeout=120.0)
    return _coord("status")


@mcp.tool()
async def coord_check() -> dict[str, Any]:
    """Fail when active file-claim conflicts exist."""

    return _coord("check")


@mcp.tool()
async def coord_claim(
    agent: str,
    lane: str,
    intent: str,
    files: list[str],
) -> dict[str, Any]:
    """Reserve an exact write set for one lane. Put files last; quote globs.

    Example files: ["self-hosted/ir/lower.sio", "scripts/ci/foo_gate.sh"]
    """

    if not files:
        return {"ok": False, "error": "files must be a non-empty list"}
    return _coord(
        "claim",
        "--agent",
        agent,
        "--lane",
        lane,
        "--intent",
        intent,
        "--files",
        *files,
    )


@mcp.tool()
async def coord_heartbeat(agent: str, lane: str) -> dict[str, Any]:
    """Refresh an existing claim lease."""

    return _coord("heartbeat", "--agent", agent, "--lane", lane)


@mcp.tool()
async def coord_release(agent: str, lane: str, reason: str) -> dict[str, Any]:
    """Release a claim and record the handoff reason."""

    return _coord(
        "release",
        "--agent",
        agent,
        "--lane",
        lane,
        "--reason",
        reason,
    )


@mcp.tool()
async def coord_send(
    agent: str,
    lane: str,
    message: str,
    kind: str = "info",
    to_agent: str = "",
    to_lane: str = "",
) -> dict[str, Any]:
    """Send a directed or broadcast message on the agent bus.

    kind: info | request | reply | blocker | handoff
    Omit to_agent/to_lane to broadcast (visible to all inboxes).
    """

    args = [
        "send",
        "--agent",
        agent,
        "--lane",
        lane,
        "--kind",
        kind,
        "--message",
        message,
    ]
    if to_agent:
        args.extend(["--to-agent", to_agent])
    if to_lane:
        args.extend(["--to-lane", to_lane])
    return _coord(*args)


@mcp.tool()
async def coord_inbox(agent: str, lane: str, show_all: bool = False) -> dict[str, Any]:
    """Show unread (or all) messages for one agent/lane."""

    args = ["inbox", "--agent", agent, "--lane", lane]
    if show_all:
        args.append("--all")
    return _coord(*args)


@mcp.tool()
async def coord_ack(agent: str, lane: str, message_id: str) -> dict[str, Any]:
    """Acknowledge a message after acting on it."""

    return _coord(
        "ack",
        "--agent",
        agent,
        "--lane",
        lane,
        "--message",
        message_id,
    )


@mcp.tool()
async def coord_prune() -> dict[str, Any]:
    """Remove expired claims and messages (shepherd hygiene)."""

    return _coord("prune")


if __name__ == "__main__":
    mcp.run()
