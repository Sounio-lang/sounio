import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPException
from pathlib import Path
import traceback
from contextlib import closing
from typing import Optional

WEBSERVER_HOST = "localhost"
WEBSERVER_ENDPOINT = "/api/provenance/call"
PORT_FILE_SUFFIX = "-provenance-port.txt"

# This hook is best-effort provenance telemetry. The provenance webserver is
# usually not running, so "no port file" and "connection refused" are the normal
# case, not errors: report nothing and exit 0. Only genuinely unexpected faults
# get a one-line note on stderr (visible under `claude --debug`), and even those
# never fail the hook — a PostToolUse hook that exits non-zero prints a full
# traceback into the session on every file edit.
DEBUG = os.getenv("PROVENANCE_HOOK_DEBUG") == "1"


class ProvenanceHookError(RuntimeError):
    pass


def note(message):
    """Emit a diagnostic without ever failing the hook."""
    print(f"provenance hook: {message}", file=sys.stderr)


def http_request(method, host, port, location, *, body: Optional[bytes] = None, headers={}, timeout=None) -> bytes:
    with closing(HTTPConnection(host, port, timeout=timeout)) as connection:
        connection.request(method, location, body=body, headers=headers)
        # The response must be read before the connection closes, otherwise the
        # server sees an aborted request.
        return connection.getresponse().read()


def get_server_port() -> Optional[int]:
    """Port the provenance webserver announced, or None if it is not running."""
    claude_root = os.getenv("CLAUDE_PROJECT_DIR")
    if not claude_root:
        if DEBUG:
            note("CLAUDE_PROJECT_DIR is unset")
        return None

    path_hash = hashlib.md5(claude_root.encode('utf-8')).hexdigest()
    port_file = Path(tempfile.gettempdir()) / (path_hash + PORT_FILE_SUFFIX)

    try:
        return int(port_file.read_text("utf-8").strip())
    except (FileNotFoundError, NotADirectoryError):
        # Server not running for this project — the expected state.
        if DEBUG:
            note(f"no port file at {port_file}")
        return None
    except (ValueError, OSError) as e:
        # Present but unreadable or malformed: worth a word, still not fatal.
        note(f"ignoring unusable port file {port_file}: {e}")
        return None


def send_diff_to_webserver(file_path, timestamp_ms):
    port = get_server_port()
    if port is None:
        return None

    url = f"http://{WEBSERVER_HOST}:{port}{WEBSERVER_ENDPOINT}"

    try:
        payload = {"file_path": file_path, "timestamp": timestamp_ms}
        return http_request(
            "POST",
            WEBSERVER_HOST,
            port=port,
            location=WEBSERVER_ENDPOINT,
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={'Content-Type': 'application/json'},
            timeout=0.5
        )

    except (HTTPException, OSError, ConnectionError) as e:
        # Stale port file, server shut down mid-session, request timed out.
        if DEBUG:
            note(f"network error while sending diff to {url}: {e}")
        return None


def extract_file_path(tool_name, tool_input):
    if tool_name in ["Write", "Edit", "MultiEdit"]:
        return tool_input.get('file_path', 'unknown')
    if tool_name == "NotebookEdit":
        return tool_input.get('notebook_path', 'unknown')
    return 'unknown'


def main():
    data = json.load(sys.stdin)
    tool_name = data.get('tool_name', 'unknown')

    modification_tools = [
        "Write", "Edit", "MultiEdit", "NotebookEdit"
    ]

    if tool_name in modification_tools:
        tool_input = data.get('tool_input', {})
        file_path = extract_file_path(tool_name, tool_input)
        if file_path:
            timestamp_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            send_diff_to_webserver(file_path, timestamp_ms)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Never fail the tool call over telemetry.
        if DEBUG:
            traceback.print_exc(file=sys.stderr)
        else:
            note(f"unexpected error: {sys.exc_info()[1]}")
    sys.exit(0)
