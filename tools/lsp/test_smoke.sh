#!/usr/bin/env bash
# Smoke checks for tools/lsp/sounio-lsp.sh and parse_diagnostics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
out, err = p.communicate(bytes(wire), timeout=40)
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
out, err = p.communicate(bytes(wire), timeout=80)
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
out, err = p.communicate(bytes(wire), timeout=80)
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

main() {
    test_diag_parser
    test_lifecycle_framed
    test_didopen_publish_diagnostics
    test_hover_definition_roundtrip
    test_strict_no_rust_failfast
    log "PASS"
}

main "$@"
