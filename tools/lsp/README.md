# Sounio LSP Server

`tools/lsp/sounio-lsp.sh` is a minimal LSP (Language Server Protocol) server for `.sio` files.
It wraps `souc` and speaks JSON-RPC over stdio with proper `Content-Length` framing.

## Files

- `tools/lsp/sounio-lsp.sh`: LSP server (bash + jq)
- `tools/lsp/parse_diagnostics.sh`: converts compiler stderr to LSP `Diagnostic[]`

## Supported LSP Methods

- Lifecycle: `initialize`, `initialized`, `shutdown`, `exit`
- Sync: `textDocument/didOpen`, `textDocument/didChange`, `textDocument/didSave`, `textDocument/didClose`
- Features: `textDocument/hover`, `textDocument/definition`
- Notifications: `textDocument/publishDiagnostics`

## VSCode Setup

Use this in `settings.json`:

```json
{
  "sounio.lsp.path": "${workspaceFolder}/tools/lsp/sounio-lsp.sh"
}
```

## Neovim Setup (`lspconfig`)

```lua
require('lspconfig.configs').sounio = {
  default_config = {
    cmd = { 'tools/lsp/sounio-lsp.sh' },
    filetypes = { 'sounio' },
    root_dir = function(fname)
      return vim.fs.dirname(vim.fs.find({'d.toml', '.git'}, { upward = true, path = fname })[1])
    end,
  },
}
```

## Manual Test

Quick smoke test (raw JSON fallback):

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | \
  tools/lsp/sounio-lsp.sh 2>/dev/null
```

Protocol-correct test with `Content-Length`:

```bash
python3 - <<'PY'
import json, subprocess
msg = {"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}
body = json.dumps(msg, separators=(",", ":")).encode()
wire = b"Content-Length: " + str(len(body)).encode() + b"\\r\\n\\r\\n" + body
p = subprocess.Popen(
    ["bash", "tools/lsp/sounio-lsp.sh"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)
out, err = p.communicate(wire, timeout=10)
print(out.decode("utf-8", "replace"))
print(err.decode("utf-8", "replace"))
PY
```

## Notes

- `jq` is required.
- LSP line/character are 0-based; compiler diagnostics are 1-based.
- The server kills stale check processes before a new diagnostic run.
- `SOUNIO_LSP_CHECK_TIMEOUT_SEC` controls `souc check` timeout (default: `60`).
- when a `souc check` times out or fails without parseable diagnostics, the server emits a synthetic diagnostic (`source: "sounio-lsp"`) instead of returning an empty list.

## No-Rust Strict Mode

`tools/lsp/sounio-lsp.sh` supports strict no-rust resolution and verification:

- `SOUNIO_LSP_STRICT_NO_RUST`:
  - defaults to `SOUNIO_REPO_HARD_NO_RUST` (or `1` when unset)
  - accepts: `1/true/yes/on` and `0/false/no/off`
- `SOUNIO_LSP_SOUC_BIN`:
  - optional explicit override for `souc` binary path
  - when strict mode is enabled, override must point inside:
    - `.pinned-souc/`
    - `artifacts/omega/souc-bin/`
  - strict mode requires `<binary>.sha256` and `<binary>.sig`; sha256 is validated at startup

## Smoke Test

Run the local smoke suite:

```bash
bash tools/lsp/test_smoke.sh
```

This validates:

- parser conversion (`parse_diagnostics.sh`)
- framed lifecycle (`initialize`/`shutdown`/`exit`)
- `didOpen -> publishDiagnostics` flow
- `didClose -> publishDiagnostics([])` cleanup flow
- `didChange` unsaved buffer used by `didSave` diagnostics
- `hover/definition` request roundtrip
- multi-document `didSave` sequencing keeps diagnostics isolated per URI
- strict no-rust fail-fast behavior
- explicit synthetic diagnostics for timeout and non-timeout check failures

CI/automation entrypoint:

```bash
bash scripts/lsp_smoke_gate.sh
```

Gate marker emitted on success: `LSP_SMOKE_PASS`.

When a document is open in the LSP session, diagnostics/hover/definition use the
in-memory buffer snapshot (not only on-disk file contents). This keeps editor
feedback aligned with unsaved changes.
