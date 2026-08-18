#!/usr/bin/env python3
"""agent-bus MCP server — real-time push between the agents on this pod.

WHY A SERVER AND NOT JUST THE CLI. scripts/dev/agent-bus.sh gives every agent a
shared log and expiring leases, but it is pull: an agent hears you when it next
runs `brief`. For BeagleCockpit that is not enough — the whole point there is to
know things as they happen. MCP's 2025-11-25 spec has the primitive for it:
a client calls `resources/subscribe` on a URI, and the server sends
`notifications/resources/updated` when that resource changes. The client re-reads
on its own. That is push, and it is what this server exists to provide.

    https://modelcontextprotocol.io/specification/2025-11-25/server/resources

DEPENDENCIES: none. Stdlib only, stdio transport, JSON-RPC framed line-by-line.
The MCP Python SDK is not installed on this pod and adding a network install to
the critical path of agent coordination would be its own hazard.

STORAGE is the same /workspace/.agents/bus the shell CLI uses, so an agent with
no MCP client still participates through the CLI and still shows up here. The
two are one channel with two doors.
"""
import json
import os
import subprocess
import sys
import threading
import time

PROTOCOL_VERSION = "2025-11-25"
SERVER_NAME = "agent-bus"
SERVER_VERSION = "1.0.0"

BUS = os.environ.get("AGENT_BUS_DIR", "/workspace/.agents/bus")
EVENTS = os.path.join(BUS, "events.jsonl")
LEASES = os.path.join(BUS, "leases")
HAZARDS = os.path.join(BUS, "hazards")
CLI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev", "agent-bus.sh")

ME = os.environ.get("AGENT_ID") or os.path.basename(os.environ.get("HOME", "unknown"))

RESOURCES = [
    ("bus://events",   "Bus events",   "Every agent's status/finding/blocker/done posts, newest last."),
    ("bus://hazards",  "Live hazards", "Environment facts that make measurements LIE rather than fail."),
    ("bus://leases",   "Held leases",  "Exclusive claims on shared resources, with time remaining."),
    ("bus://agents",   "Active agents","Agents that posted in the last two hours."),
]

_out_lock = threading.Lock()
_subscriptions = set()
_sub_lock = threading.Lock()


def send(obj):
    """One JSON object per line. Guarded because the watcher thread also writes."""
    with _out_lock:
        sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def notify(method, params=None):
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    send(msg)


def ok(req_id, result):
    send({"jsonrpc": "2.0", "id": req_id, "result": result})


def err(req_id, code, message):
    send({"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}})


def cli(*args):
    """Delegate mutations to the shell CLI so the two doors cannot drift apart."""
    try:
        p = subprocess.run(["bash", CLI, *args], capture_output=True, text=True, timeout=30)
        return (p.stdout + p.stderr).strip(), p.returncode
    except Exception as e:                                    # noqa: BLE001
        return f"agent-bus CLI failed: {e}", 1


def read_resource(uri):
    if uri == "bus://events":
        try:
            with open(EVENTS, encoding="utf-8", errors="replace") as fh:
                return "".join(fh.readlines()[-200:])
        except OSError:
            return ""
    if uri == "bus://hazards":
        out, _ = cli("hazard", "list")
        return out
    if uri == "bus://leases":
        out, _ = cli("leases")
        return out
    if uri == "bus://agents":
        out, _ = cli("who")
        return out
    return None


def bus_fingerprint():
    """Cheap change detector: sizes and mtimes of everything the bus stores."""
    parts = []
    try:
        st = os.stat(EVENTS)
        parts.append(("e", st.st_size, int(st.st_mtime)))
    except OSError:
        parts.append(("e", -1, 0))
    for d in (LEASES, HAZARDS):
        try:
            parts.append((d, tuple(sorted(os.listdir(d)))))
        except OSError:
            parts.append((d, ()))
    return repr(parts)


def watcher():
    """Poll the bus and push resources/updated to whoever subscribed.

    Polling rather than inotify on purpose: the bus lives on a shared volume
    that may be a network mount, where inotify is unreliable, and a 400 ms poll
    of two small directories costs nothing next to what these agents do.
    """
    last = bus_fingerprint()
    while True:
        time.sleep(0.4)
        cur = bus_fingerprint()
        if cur == last:
            continue
        last = cur
        with _sub_lock:
            targets = list(_subscriptions)
        for uri in targets:
            notify("notifications/resources/updated", {"uri": uri})


TOOLS = [
    {
        "name": "bus_post",
        "description": "Post to the shared agent bus. Every other agent sees it, and any "
                       "subscribed client is pushed a resources/updated for bus://events.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["status", "finding", "blocker", "done"]},
                "text": {"type": "string", "description": "What changed. Be specific."},
            },
            "required": ["kind", "text"],
        },
    },
    {
        "name": "bus_hazard",
        "description": "Record an environment fact that will make other agents' measurements "
                       "LIE rather than fail: a poisoned env var, a stale artifact, a checkout "
                       "parked on another branch. Those are the expensive ones.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
                "text": {"type": "string"},
                "clear": {"type": "boolean", "description": "Clear this hazard instead of adding it."},
            },
            "required": ["slug"],
        },
    },
    {
        "name": "bus_claim",
        "description": "Take an exclusive, expiring lease on a shared resource (a build lock, a "
                       "file, a lane). Fails if another agent holds it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "resource": {"type": "string"},
                "ttl_minutes": {"type": "integer", "default": 90},
            },
            "required": ["resource"],
        },
    },
    {
        "name": "bus_release",
        "description": "Release a lease you hold.",
        "inputSchema": {
            "type": "object",
            "properties": {"resource": {"type": "string"}},
            "required": ["resource"],
        },
    },
    {
        "name": "bus_brief",
        "description": "Hazards, leases and recent events in one read. Run this before starting "
                       "anything that touches shared state.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def call_tool(name, args):
    if name == "bus_post":
        out, _ = cli("post", args.get("kind", "status"), args.get("text", ""))
        return out
    if name == "bus_hazard":
        if args.get("clear"):
            out, _ = cli("hazard", "clear", args["slug"])
        else:
            out, _ = cli("hazard", "add", args["slug"], args.get("text", ""))
        return out
    if name == "bus_claim":
        out, rc = cli("claim", args["resource"], str(args.get("ttl_minutes", 90)))
        return out if rc == 0 else f"REFUSED: {out}"
    if name == "bus_release":
        out, _ = cli("release", args["resource"])
        return out
    if name == "bus_brief":
        out, _ = cli("brief")
        return out
    raise KeyError(name)


def handle(msg):
    method = msg.get("method")
    req_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        ok(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                # subscribe is the one that matters: it is what turns this from a
                # pull channel into a push one.
                "resources": {"subscribe": True, "listChanged": False},
                "tools": {"listChanged": False},
            },
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "Shared channel between the agents on this pod. Subscribe to bus://events "
                "and bus://hazards to be told the moment another agent posts. Post when your "
                "state changes; claim before touching a shared resource."
            ),
        })
        return

    if method in ("notifications/initialized", "notifications/cancelled"):
        return

    if method == "ping":
        ok(req_id, {})
        return

    if method == "resources/list":
        ok(req_id, {"resources": [
            {"uri": u, "name": n, "description": d, "mimeType": "text/plain"}
            for u, n, d in RESOURCES
        ]})
        return

    if method == "resources/read":
        uri = params.get("uri", "")
        body = read_resource(uri)
        if body is None:
            err(req_id, -32602, f"unknown resource: {uri}")
            return
        ok(req_id, {"contents": [{"uri": uri, "mimeType": "text/plain", "text": body}]})
        return

    if method == "resources/subscribe":
        uri = params.get("uri", "")
        if read_resource(uri) is None:
            err(req_id, -32602, f"unknown resource: {uri}")
            return
        with _sub_lock:
            _subscriptions.add(uri)
        ok(req_id, {})
        return

    if method == "resources/unsubscribe":
        with _sub_lock:
            _subscriptions.discard(params.get("uri", ""))
        ok(req_id, {})
        return

    if method == "tools/list":
        ok(req_id, {"tools": TOOLS})
        return

    if method == "tools/call":
        name = params.get("name", "")
        try:
            text = call_tool(name, params.get("arguments") or {})
        except KeyError:
            err(req_id, -32602, f"unknown tool: {name}")
            return
        except Exception as e:                                # noqa: BLE001
            ok(req_id, {"content": [{"type": "text", "text": f"error: {e}"}], "isError": True})
            return
        ok(req_id, {"content": [{"type": "text", "text": text}]})
        return

    if req_id is not None:
        err(req_id, -32601, f"method not found: {method}")


def main():
    os.makedirs(BUS, exist_ok=True)
    threading.Thread(target=watcher, daemon=True).start()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            handle(msg)
        except Exception as e:                                # noqa: BLE001
            if msg.get("id") is not None:
                err(msg["id"], -32603, f"internal error: {e}")


if __name__ == "__main__":
    main()
