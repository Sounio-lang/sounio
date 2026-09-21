#!/usr/bin/env bash
# scripts/ci/sounio_editor_tooling_support_gate.sh
#
# Bounded support gate for the Sounio editor-tooling preview. This proves the
# public CLI hooks for formatter, REPL, and LSP preview, static editor
# package wiring, and a green pure-Sounio LSP rebuild under Madaros. It
# intentionally does not claim a mature IDE.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

ARTIFACT_DIR="${SOUNIO_EDITOR_TOOLING_ARTIFACT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/sounio-editor-tooling-support.XXXXXX")}"
mkdir -p "$ARTIFACT_DIR"

LOG_PATH="$ARTIFACT_DIR/editor_tooling_support.log"
SUMMARY_JSON="$ARTIFACT_DIR/editor_tooling_support_status.v1.json"
: > "$LOG_PATH"

pass=0
fail=0
warn=0
STEP_ROWS=()
WARN_ROWS=()

record_step() {
  local name="$1"
  local status="$2"
  local detail="$3"
  STEP_ROWS+=("$name"$'\t'"$status"$'\t'"$detail")
  case "$status" in
    pass) pass=$((pass + 1)) ;;
    fail) fail=$((fail + 1)) ;;
    warn) warn=$((warn + 1)) ;;
  esac
}

run_step() {
  local name="$1"
  shift
  echo "[editor-tooling] $name" | tee -a "$LOG_PATH"
  if "$@" >>"$LOG_PATH" 2>&1; then
    record_step "$name" pass "$*"
  else
    record_step "$name" fail "$*"
  fi
}

warn_step() {
  local name="$1"
  local detail="$2"
  WARN_ROWS+=("$name"$'\t'"$detail")
  record_step "$name" warn "$detail"
  echo "[editor-tooling] warning: $name: $detail" | tee -a "$LOG_PATH"
}

run_step "cli-help-editor-verbs" bash -c '
  set -euo pipefail
  help="$(./bin/souc --help)"
  grep -q "souc format <file.sio>" <<<"$help"
  grep -q "souc fmt <file.sio>" <<<"$help"
  grep -q "souc repl" <<<"$help"
  grep -q "souc lsp --stdio" <<<"$help"
'

run_step "formatter-idempotent" env -u SOUC_BIN -u SOUNIO_SOUC_ENGINE \
  bash scripts/gates/g5a_formatter_idempotent.sh

run_step "repl-eval" env -u SOUC_BIN -u SOUNIO_SOUC_ENGINE \
  bash scripts/gates/g5b_repl_eval.sh

run_step "lsp-preview-smoke" env \
  SOUNIO_LSP_STRICT_NO_RUST=0 \
  SOUC_BIN="$ROOT_DIR/bin/souc" \
  SOUNIO_LSP_SOUC_BIN="$ROOT_DIR/bin/souc" \
  bash tools/lsp/test_smoke.sh

run_step "lsp-cli-initialize" python3 - <<'PY'
import json
import subprocess

messages = [
    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}},
    {"jsonrpc": "2.0", "method": "exit"},
]
stream = b""
for msg in messages:
    body = json.dumps(msg, separators=(",", ":")).encode()
    stream += b"Content-Length: %d\r\n\r\n" % len(body) + body

proc = subprocess.run(
    ["./bin/souc", "lsp", "--stdio"],
    input=stream,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=15,
    check=False,
)
out = proc.stdout.decode("utf-8", "replace")
assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
for needle in (
    '"hoverProvider":true',
    '"completionProvider"',
    '"documentFormattingProvider":true',
    '"codeActionProvider"',
    '"documentSymbolProvider":true',
    '"signatureHelpProvider"',
):
    assert needle in out, needle
PY

run_step "editor-assets-static-contract" python3 - <<'PY'
import json
from pathlib import Path

pkg = json.loads(Path("tools/editors/vscode/package.json").read_text())
commands = {entry["command"] for entry in pkg["contributes"]["commands"]}
for required in (
    "sounio.restartServer",
    "sounio.runFile",
    "sounio.checkFile",
    "sounio.startRepl",
    "sounio.formatDocument",
    "sounio.showOutline",
    "sounio.renameSymbol",
    "sounio.showSignature",
):
    assert required in commands, required

languages = pkg["contributes"]["languages"]
assert any(lang["id"] == "sounio" and ".sio" in lang["extensions"] for lang in languages)
assert Path("tools/editors/vscode/src/extension.ts").read_text().count("souc lsp") >= 1
assert "souc lsp --stdio" in Path("tools/editors/README.md").read_text()
assert "command = \"souc\"" in Path("tools/editors/helix/languages.toml").read_text()
assert "cmd = { 'souc', 'lsp', '--stdio' }" in Path("tools/editors/neovim/lspconfig.lua").read_text()
PY

run_step "pure-sounio-lsp-rebuild" ./bin/souc compile \
  self-hosted/lsp/server.sio -o "$ARTIFACT_DIR/pure-sounio-lsp.bin"

python3 - "$SUMMARY_JSON" "$pass" "$fail" "$warn" "${STEP_ROWS[@]}" -- "${WARN_ROWS[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone

summary_path = sys.argv[1]
pass_count = int(sys.argv[2])
fail_count = int(sys.argv[3])
warn_count = int(sys.argv[4])
rest = sys.argv[5:]
split = rest.index("--")
step_raw = rest[:split]
warn_raw = rest[split + 1:]

def parse_row(row):
    name, status, detail = row.split("\t", 2)
    return {"name": name, "status": status, "detail": detail}

def parse_warn(row):
    name, detail = row.split("\t", 1)
    return {"name": name, "detail": detail}

summary = {
    "schema": "sounio.editor_tooling_support.v1",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "status": "pass" if fail_count == 0 else "fail",
    "pass": pass_count,
    "fail": fail_count,
    "warn": warn_count,
    "steps": [parse_row(row) for row in step_raw],
    "warnings": [parse_warn(row) for row in warn_raw],
    "claim_boundary": {
        "claim": "SOTA-preview editor tooling, not mature IDE support",
        "proved": [
            "public bin/souc format/fmt CLI",
            "public bin/souc repl CLI",
            "public bin/souc lsp --stdio preview server",
            "formatter idempotency gate",
            "file-backed REPL eval gate",
            "bash LSP smoke and initialize capability smoke",
            "VS Code/Helix/Neovim static editor asset contract",
            "pure-Sounio LSP rebuild under Madaros",
            "semanticTokens/full/delta (probe-tested in tools/lsp/test_protocol.sh)",
            "incremental text synchronization (probe-tested in tools/lsp/test_protocol.sh)",
            "unopened-file workspace indexing (probe-tested in tools/lsp/test_protocol.sh)",
            "type hierarchy prepare/supertypes/subtypes (probe-tested in tools/lsp/test_protocol.sh)",
        ],
        "not_proved": [
            "marketplace-quality VS Code release",
            "notebook or AI assistant integration",
        ],
    },
}

Path = __import__("pathlib").Path
Path(summary_path).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, indent=2, sort_keys=True))
PY

if [[ "$fail" -gt 0 ]]; then
  echo "[editor-tooling] FAIL: $fail failing steps; see $LOG_PATH" >&2
  exit 1
fi

echo "[editor-tooling] PASS: $pass pass, $warn warnings; summary=$SUMMARY_JSON"
