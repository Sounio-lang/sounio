#!/usr/bin/env bash
# Smoke checks for tools/lsp/sounio-lsp.sh and parse_diagnostics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
LSP_SCRIPT="$SCRIPT_DIR/sounio-lsp.sh"
DIAG_PARSER="$SCRIPT_DIR/parse_diagnostics.sh"

log() {
    printf '[lsp-smoke] %s\n' "$*" >&2
}

fail() {
    printf '[lsp-smoke][FAIL] %s\n' "$*" >&2
    exit 1
}

require_bin() {
    local bin="$1"
    if ! command -v "$bin" >/dev/null 2>&1; then
        fail "missing required command: $bin"
    fi
}

require_bin jq
require_bin python3
require_bin bash

resolve_souc_for_smoke() {
    if [[ -n "${SOUNIO_LSP_SOUC_BIN:-}" ]] && [[ -x "${SOUNIO_LSP_SOUC_BIN:-}" ]]; then
        printf '%s\n' "$SOUNIO_LSP_SOUC_BIN"
        return 0
    fi
    OMEGA_SOUC_REQUIRE_PINNED="${OMEGA_SOUC_REQUIRE_PINNED:-1}" \
    OMEGA_SOUC_ALLOW_LOCAL_FALLBACK="${OMEGA_SOUC_ALLOW_LOCAL_FALLBACK:-0}" \
        bash "$ROOT_DIR/scripts/omega/omega_resolve_souc_bin.sh" --print-path
}

SOUNIO_LSP_SOUC_BIN="$(resolve_souc_for_smoke)"
export SOUNIO_LSP_SOUC_BIN
export SOUNIO_REPO_HARD_NO_RUST="${SOUNIO_REPO_HARD_NO_RUST:-1}"
export SOUNIO_LSP_STRICT_NO_RUST="${SOUNIO_LSP_STRICT_NO_RUST:-1}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

test_diag_parser() {
    log "diagnostic parser conversion"
    local input_file out_file
    input_file="$TMP_DIR/diag-input.txt"
    out_file="$TMP_DIR/diag-output.json"
    cat >"$input_file" <<'EOF'
error[E017]: type mismatch
  --> /tmp/main.sio:42:10
  |
42|     let x: i64 = "hello"
  |                  ^^^^^^^ expected i64, found string
EOF
    "$DIAG_PARSER" <"$input_file" >"$out_file"
    jq -e '
        length == 1
        and .[0].code == "E017"
        and .[0].severity == 1
        and .[0].range.start.line == 41
        and .[0].range.start.character == 9
    ' "$out_file" >/dev/null || fail "parse_diagnostics output does not match expected shape"
}

test_lifecycle_framed() {
    log "initialize/shutdown/exit framed exchange"
    local out_file err_file
    out_file="$TMP_DIR/lifecycle.out"
    err_file="$TMP_DIR/lifecycle.err"
    python3 - "$LSP_SCRIPT" "$out_file" "$err_file" <<'PY'
import json
import re
import subprocess
import sys

lsp_script, out_path, err_path = sys.argv[1:4]

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
    {"jsonrpc": "2.0", "method": "initialized", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}},
]

wire = bytearray()
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    wire.extend(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    wire.extend(body)

p = subprocess.Popen(
    ["bash", lsp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
out, err = p.communicate(bytes(wire), timeout=60)
open(out_path, "wb").write(out)
open(err_path, "wb").write(err)
if p.returncode != 0:
    raise SystemExit(f"lsp lifecycle run failed with code {p.returncode}")

payloads = []
idx = 0
while idx < len(out):
    sep = out.find(b"\r\n\r\n", idx)
    if sep < 0:
        break
    headers = out[idx:sep].decode("utf-8", "replace")
    match = re.search(r"Content-Length:\s*([0-9]+)", headers, flags=re.IGNORECASE)
    if not match:
        raise SystemExit("missing Content-Length in response headers")
    length = int(match.group(1))
    body_start = sep + 4
    body_end = body_start + length
    payload = json.loads(out[body_start:body_end].decode("utf-8", "replace"))
    payloads.append(payload)
    idx = body_end

init = next((p for p in payloads if p.get("id") == 1), None)
shutdown = next((p for p in payloads if p.get("id") == 2), None)
if init is None:
    raise SystemExit("initialize response missing")
if shutdown is None:
    raise SystemExit("shutdown response missing")
caps = init.get("result", {}).get("capabilities", {})
if not caps.get("hoverProvider") or not caps.get("definitionProvider"):
    raise SystemExit("capabilities missing hover/definition provider")
PY
}

test_didopen_publish_diagnostics() {
    log "didOpen publishes diagnostics"
    local bad_file out_file err_file
    bad_file="$TMP_DIR/lsp_bad_input.sio"
    out_file="$TMP_DIR/didopen.out"
    err_file="$TMP_DIR/didopen.err"
    cat >"$bad_file" <<'EOF'
fn main() -> i32 {
    let x: i64 = "hello"
    return 0
}
EOF
    python3 - "$LSP_SCRIPT" "$bad_file" "$out_file" "$err_file" <<'PY'
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import quote

lsp_script, bad_file, out_path, err_path = sys.argv[1:5]
bad_file = pathlib.Path(bad_file).resolve()
uri = "file://" + quote(str(bad_file))

text = bad_file.read_text(encoding="utf-8")
messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {"textDocument": {"uri": uri, "languageId": "sounio", "version": 1, "text": text}},
    },
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}},
]

wire = bytearray()
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    wire.extend(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    wire.extend(body)

p = subprocess.Popen(
    ["bash", lsp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
out, err = p.communicate(bytes(wire), timeout=120)
open(out_path, "wb").write(out)
open(err_path, "wb").write(err)
if p.returncode != 0:
    raise SystemExit(f"lsp didOpen run failed with code {p.returncode}")

payloads = []
idx = 0
while idx < len(out):
    sep = out.find(b"\r\n\r\n", idx)
    if sep < 0:
        break
    headers = out[idx:sep].decode("utf-8", "replace")
    match = re.search(r"Content-Length:\s*([0-9]+)", headers, flags=re.IGNORECASE)
    if not match:
        raise SystemExit("missing Content-Length in didOpen response headers")
    length = int(match.group(1))
    body_start = sep + 4
    body_end = body_start + length
    payload = json.loads(out[body_start:body_end].decode("utf-8", "replace"))
    payloads.append(payload)
    idx = body_end

publish = [
    p for p in payloads
    if p.get("method") == "textDocument/publishDiagnostics"
]
if not publish:
    raise SystemExit("missing publishDiagnostics notification")

diagnostics = publish[-1].get("params", {}).get("diagnostics", [])
if not diagnostics:
    raise SystemExit("publishDiagnostics has empty diagnostics array for invalid source")
if diagnostics[0].get("severity") != 1:
    raise SystemExit("first diagnostic is not severity error")
PY
}

test_didchange_unsaved_snapshot_on_save() {
    log "didChange unsaved text is used by didSave diagnostics"
    local source_file out_file err_file
    source_file="$TMP_DIR/lsp_changed_buffer.sio"
    out_file="$TMP_DIR/didchange_save.out"
    err_file="$TMP_DIR/didchange_save.err"
    cat >"$source_file" <<'EOF'
fn main() -> i32 {
    let x: i64 = 7
    return 0
}
EOF
    python3 - "$LSP_SCRIPT" "$source_file" "$out_file" "$err_file" <<'PY'
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import quote

lsp_script, source_file, out_path, err_path = sys.argv[1:5]
source_file = pathlib.Path(source_file).resolve()
uri = "file://" + quote(str(source_file))
original_text = source_file.read_text(encoding="utf-8")
changed_text = """fn main() -> i32 {
    let x: i64 = "not-int"
    return 0
}
"""

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {"textDocument": {"uri": uri, "languageId": "sounio", "version": 1, "text": original_text}},
    },
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": uri, "version": 2},
            "contentChanges": [{"text": changed_text}],
        },
    },
    {"jsonrpc": "2.0", "method": "textDocument/didSave", "params": {"textDocument": {"uri": uri}}},
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}},
]

wire = bytearray()
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    wire.extend(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    wire.extend(body)

p = subprocess.Popen(
    ["bash", lsp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
out, err = p.communicate(bytes(wire), timeout=120)
open(out_path, "wb").write(out)
open(err_path, "wb").write(err)
if p.returncode != 0:
    raise SystemExit(f"lsp didChange/didSave run failed with code {p.returncode}")

payloads = []
idx = 0
while idx < len(out):
    sep = out.find(b"\r\n\r\n", idx)
    if sep < 0:
        break
    headers = out[idx:sep].decode("utf-8", "replace")
    match = re.search(r"Content-Length:\s*([0-9]+)", headers, flags=re.IGNORECASE)
    if not match:
        raise SystemExit("missing Content-Length in didChange/didSave response headers")
    length = int(match.group(1))
    body_start = sep + 4
    body_end = body_start + length
    payload = json.loads(out[body_start:body_end].decode("utf-8", "replace"))
    payloads.append(payload)
    idx = body_end

publish = [
    p for p in payloads
    if p.get("method") == "textDocument/publishDiagnostics"
]
if len(publish) < 2:
    raise SystemExit("expected at least two publishDiagnostics notifications (didOpen and didSave)")

last_diagnostics = publish[-1].get("params", {}).get("diagnostics", [])
if not last_diagnostics:
    raise SystemExit("didSave diagnostics are empty; expected in-memory changed text to produce errors")
if last_diagnostics[0].get("severity") != 1:
    raise SystemExit("didSave first diagnostic is not severity error")
PY
}

test_hover_definition_roundtrip() {
    log "hover/definition roundtrip (no crash)"
    local good_file out_file err_file
    good_file="$TMP_DIR/lsp_good_input.sio"
    out_file="$TMP_DIR/hover_def.out"
    err_file="$TMP_DIR/hover_def.err"
    cat >"$good_file" <<'EOF'
fn id(x: i64) -> i64 {
    return x
}

fn main() -> i32 {
    let v: i64 = id(2)
    return 0
}
EOF
    python3 - "$LSP_SCRIPT" "$good_file" "$out_file" "$err_file" <<'PY'
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import quote

lsp_script, good_file, out_path, err_path = sys.argv[1:5]
good_file = pathlib.Path(good_file).resolve()
uri = "file://" + quote(str(good_file))

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
    {"jsonrpc": "2.0", "id": 3, "method": "textDocument/hover", "params": {"textDocument": {"uri": uri}, "position": {"line": 0, "character": 4}}},
    {"jsonrpc": "2.0", "id": 4, "method": "textDocument/definition", "params": {"textDocument": {"uri": uri}, "position": {"line": 5, "character": 16}}},
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}},
]

wire = bytearray()
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    wire.extend(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    wire.extend(body)

p = subprocess.Popen(
    ["bash", lsp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
out, err = p.communicate(bytes(wire), timeout=120)
open(out_path, "wb").write(out)
open(err_path, "wb").write(err)
if p.returncode != 0:
    raise SystemExit(f"lsp hover/definition run failed with code {p.returncode}")

payloads = []
idx = 0
while idx < len(out):
    sep = out.find(b"\r\n\r\n", idx)
    if sep < 0:
        break
    headers = out[idx:sep].decode("utf-8", "replace")
    match = re.search(r"Content-Length:\s*([0-9]+)", headers, flags=re.IGNORECASE)
    if not match:
        raise SystemExit("missing Content-Length in hover/definition response headers")
    length = int(match.group(1))
    body_start = sep + 4
    body_end = body_start + length
    payload = json.loads(out[body_start:body_end].decode("utf-8", "replace"))
    payloads.append(payload)
    idx = body_end

hover = next((p for p in payloads if p.get("id") == 3), None)
definition = next((p for p in payloads if p.get("id") == 4), None)
if hover is None:
    raise SystemExit("hover response missing")
if definition is None:
    raise SystemExit("definition response missing")
if "error" in hover:
    raise SystemExit(f"hover returned JSON-RPC error: {hover['error']}")
if "error" in definition:
    raise SystemExit(f"definition returned JSON-RPC error: {definition['error']}")
PY
}

test_multi_uri_diagnostics_isolation() {
    log "multi-document didSave sequence keeps diagnostics isolated per URI"
    local file_a file_b out_file err_file
    file_a="$TMP_DIR/lsp_multi_a.sio"
    file_b="$TMP_DIR/lsp_multi_b.sio"
    out_file="$TMP_DIR/multi_uri.out"
    err_file="$TMP_DIR/multi_uri.err"

    cat >"$file_a" <<'EOF'
fn main() -> i32 {
    let a: i64 = "bad-a"
    return 0
}
EOF
    cat >"$file_b" <<'EOF'
fn main() -> i32 {
    let b: i64 = 1
    return 0
}
EOF

    python3 - "$LSP_SCRIPT" "$file_a" "$file_b" "$out_file" "$err_file" <<'PY'
import json
import pathlib
import re
import subprocess
import sys
from urllib.parse import quote

lsp_script, file_a, file_b, out_path, err_path = sys.argv[1:6]
file_a = pathlib.Path(file_a).resolve()
file_b = pathlib.Path(file_b).resolve()
uri_a = "file://" + quote(str(file_a))
uri_b = "file://" + quote(str(file_b))
text_a = file_a.read_text(encoding="utf-8")
text_b = file_b.read_text(encoding="utf-8")

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {"textDocument": {"uri": uri_a, "languageId": "sounio", "version": 1, "text": text_a}},
    },
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {"textDocument": {"uri": uri_b, "languageId": "sounio", "version": 1, "text": text_b}},
    },
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": uri_b, "version": 2},
            "contentChanges": [{"text": 'fn main() -> i32 {\n    let b: i64 = "bad-b"\n    return 0\n}\n'}],
        },
    },
    {"jsonrpc": "2.0", "method": "textDocument/didSave", "params": {"textDocument": {"uri": uri_b}}},
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didChange",
        "params": {
            "textDocument": {"uri": uri_b, "version": 3},
            "contentChanges": [{"text": "fn main() -> i32 {\n    let b: i64 = 2\n    return 0\n}\n"}],
        },
    },
    {"jsonrpc": "2.0", "method": "textDocument/didSave", "params": {"textDocument": {"uri": uri_b}}},
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}},
]

wire = bytearray()
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    wire.extend(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    wire.extend(body)

p = subprocess.Popen(
    ["bash", lsp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
out, err = p.communicate(bytes(wire), timeout=180)
open(out_path, "wb").write(out)
open(err_path, "wb").write(err)
if p.returncode != 0:
    raise SystemExit(f"lsp multi-uri run failed with code {p.returncode}")

payloads = []
idx = 0
while idx < len(out):
    sep = out.find(b"\r\n\r\n", idx)
    if sep < 0:
        break
    headers = out[idx:sep].decode("utf-8", "replace")
    match = re.search(r"Content-Length:\s*([0-9]+)", headers, flags=re.IGNORECASE)
    if not match:
        raise SystemExit("missing Content-Length in multi-uri response headers")
    length = int(match.group(1))
    body_start = sep + 4
    body_end = body_start + length
    payload = json.loads(out[body_start:body_end].decode("utf-8", "replace"))
    payloads.append(payload)
    idx = body_end

publishes = [
    p for p in payloads
    if p.get("method") == "textDocument/publishDiagnostics"
]
if not publishes:
    raise SystemExit("missing publishDiagnostics notifications in multi-uri scenario")

by_uri = {}
for p in publishes:
    uri = p.get("params", {}).get("uri")
    if uri:
        by_uri.setdefault(uri, []).append(p.get("params", {}).get("diagnostics", []))

if uri_a not in by_uri:
    raise SystemExit("missing diagnostics stream for uri_a")
if uri_b not in by_uri:
    raise SystemExit("missing diagnostics stream for uri_b")

if not any(d for d in by_uri[uri_a] if len(d) > 0):
    raise SystemExit("uri_a should keep non-empty diagnostics")

# For uri_b we expect one failing save and then a clean save.
if len(by_uri[uri_b]) < 2:
    raise SystemExit("uri_b should have multiple diagnostic publications")
if len(by_uri[uri_b][-2]) == 0:
    raise SystemExit("uri_b penultimate diagnostics should report the introduced error")
if len(by_uri[uri_b][-1]) != 0:
    raise SystemExit("uri_b final diagnostics should be clean after fix")
PY
}

test_strict_no_rust_failfast() {
    log "strict no-rust rejects unpinned SOUC_BIN override"
    local out_file err_file
    out_file="$TMP_DIR/strict.out"
    err_file="$TMP_DIR/strict.err"
    if SOUNIO_LSP_STRICT_NO_RUST=1 SOUNIO_LSP_SOUC_BIN=/bin/echo \
        bash "$LSP_SCRIPT" </dev/null >"$out_file" 2>"$err_file"; then
        fail "strict no-rust should have rejected unpinned SOUC_BIN override"
    fi
    if ! grep -q "strict no-rust mode requires pinned souc" "$err_file"; then
        fail "strict no-rust failure message not found"
    fi
}

test_invalid_timeout_config_failfast() {
    log "invalid timeout config fails fast"
    local out_file err_file
    out_file="$TMP_DIR/invalid-timeout.out"
    err_file="$TMP_DIR/invalid-timeout.err"
    if SOUNIO_LSP_CHECK_TIMEOUT_SEC=not_a_number \
        bash "$LSP_SCRIPT" </dev/null >"$out_file" 2>"$err_file"; then
        fail "invalid SOUNIO_LSP_CHECK_TIMEOUT_SEC should fail fast"
    fi
    if ! grep -q "SOUNIO_LSP_CHECK_TIMEOUT_SEC must be a positive integer" "$err_file"; then
        fail "invalid timeout failure message not found"
    fi
}

test_timeout_emits_diagnostic() {
    if ! command -v timeout >/dev/null 2>&1; then
        log "timeout command missing; skipping timeout diagnostic test"
        return 0
    fi

    log "timeout emits explicit diagnostic"
    local fake_souc source_file out_file err_file
    fake_souc="$TMP_DIR/fake_souc_timeout.sh"
    source_file="$TMP_DIR/lsp_timeout_input.sio"
    out_file="$TMP_DIR/timeout_diag.out"
    err_file="$TMP_DIR/timeout_diag.err"

    cat >"$fake_souc" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "check" ]]; then
    sleep 2
    exit 0
fi
exit 0
EOF
    chmod +x "$fake_souc"

    cat >"$source_file" <<'EOF'
fn main() -> i32 {
    return 0
}
EOF

    python3 - "$LSP_SCRIPT" "$source_file" "$fake_souc" "$out_file" "$err_file" <<'PY'
import json
import os
import pathlib
import re
import subprocess
import sys
from urllib.parse import quote

lsp_script, source_file, fake_souc, out_path, err_path = sys.argv[1:6]
source_file = pathlib.Path(source_file).resolve()
uri = "file://" + quote(str(source_file))
text = source_file.read_text(encoding="utf-8")

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"capabilities": {}}},
    {
        "jsonrpc": "2.0",
        "method": "textDocument/didOpen",
        "params": {"textDocument": {"uri": uri, "languageId": "sounio", "version": 1, "text": text}},
    },
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit", "params": {}},
]

wire = bytearray()
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
    wire.extend(f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8"))
    wire.extend(body)

env = os.environ.copy()
env["SOUNIO_LSP_STRICT_NO_RUST"] = "0"
env["SOUNIO_LSP_CHECK_TIMEOUT_SEC"] = "1"
env["SOUNIO_LSP_SOUC_BIN"] = fake_souc

p = subprocess.Popen(
    ["bash", lsp_script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    env=env,
)
out, err = p.communicate(bytes(wire), timeout=60)
open(out_path, "wb").write(out)
open(err_path, "wb").write(err)
if p.returncode != 0:
    raise SystemExit(f"lsp timeout diagnostic run failed with code {p.returncode}")

payloads = []
idx = 0
while idx < len(out):
    sep = out.find(b"\r\n\r\n", idx)
    if sep < 0:
        break
    headers = out[idx:sep].decode("utf-8", "replace")
    match = re.search(r"Content-Length:\s*([0-9]+)", headers, flags=re.IGNORECASE)
    if not match:
        raise SystemExit("missing Content-Length in timeout diagnostic response headers")
    length = int(match.group(1))
    body_start = sep + 4
    body_end = body_start + length
    payload = json.loads(out[body_start:body_end].decode("utf-8", "replace"))
    payloads.append(payload)
    idx = body_end

publish = [p for p in payloads if p.get("method") == "textDocument/publishDiagnostics"]
if not publish:
    raise SystemExit("missing publishDiagnostics for timeout scenario")

diagnostics = publish[-1].get("params", {}).get("diagnostics", [])
if not diagnostics:
    raise SystemExit("expected synthetic timeout diagnostic, got empty diagnostics")
first = diagnostics[0]
if first.get("code") != "SOUNIO_LSP_CHECK_TIMEOUT":
    raise SystemExit(f"unexpected timeout diagnostic code: {first.get('code')}")
if "timed out" not in first.get("message", ""):
    raise SystemExit(f"timeout diagnostic message missing timeout text: {first.get('message')}")
PY
}

main() {
    test_diag_parser
    test_lifecycle_framed
    test_didopen_publish_diagnostics
    test_didchange_unsaved_snapshot_on_save
    test_hover_definition_roundtrip
    test_multi_uri_diagnostics_isolation
    test_strict_no_rust_failfast
    test_invalid_timeout_config_failfast
    test_timeout_emits_diagnostic
    log "PASS"
}

main "$@"
