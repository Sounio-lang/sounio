# Sounio LSP Server - BIG EDITION 🚀

`tools/lsp/sounio-lsp.sh` is a full-featured LSP (Language Server Protocol) server for `.sio` files.
It wraps `souc` and speaks JSON-RPC over stdio with proper `Content-Length` framing.

## What's New in BIG LSP v1.0.0

The BIG LSP brings a comprehensive IDE experience to Sounio with **15+ LSP features**:

| Feature | Method | Description |
|---------|--------|-------------|
| ✅ Diagnostics | `textDocument/publishDiagnostics` | Real-time error reporting |
| ✅ Hover | `textDocument/hover` | Type information on hover |
| ✅ Go to Definition | `textDocument/definition` | Jump to symbol definition |
| ✅ **Autocompletion** | `textDocument/completion` | Context-aware suggestions |
| ✅ **Formatting** | `textDocument/formatting` | Document formatting via `souc fmt` |
| ✅ **Code Actions** | `textDocument/codeAction` | Quick fixes and refactorings |
| ✅ **Document Symbols** | `textDocument/documentSymbol` | Outline view |
| ✅ **Workspace Symbols** | `workspace/symbol` | Global symbol search |
| ✅ **Find References** | `textDocument/references` | Find all usages |
| ✅ **Rename** | `textDocument/rename` | Safe symbol renaming |
| ✅ **Signature Help** | `textDocument/signatureHelp` | Function parameter hints |
| ✅ **Document Highlights** | `textDocument/documentHighlight` | Highlight symbol occurrences |
| ✅ **Prepare Rename** | `textDocument/prepareRename` | Rename preview |

## Files

- `tools/lsp/sounio-lsp.sh`: Full-featured LSP server (bash + jq + python3)
- `tools/lsp/parse_diagnostics.sh`: Converts compiler stderr to LSP `Diagnostic[]`
- `editors/vscode/`: VSCode extension with full LSP client support

## Supported LSP Methods

### Lifecycle
- `initialize` - Server capabilities announcement
- `initialized` - Client initialized
- `shutdown` - Graceful shutdown
- `exit` - Process exit

### Text Synchronization
- `textDocument/didOpen` - Document opened
- `textDocument/didChange` - Document changed
- `textDocument/didSave` - Document saved
- `textDocument/didClose` - Document closed

### Language Features
- `textDocument/hover` - Type information
- `textDocument/definition` - Go to definition
- `textDocument/completion` - Autocomplete
- `textDocument/formatting` - Format document
- `textDocument/rangeFormatting` - Format selection
- `textDocument/codeAction` - Quick fixes
- `textDocument/documentSymbol` - Document outline
- `workspace/symbol` - Workspace symbol search
- `textDocument/references` - Find references
- `textDocument/rename` - Rename symbol
- `textDocument/prepareRename` - Rename preparation
- `textDocument/signatureHelp` - Signature help
- `textDocument/documentHighlight` - Document highlights

### Notifications
- `textDocument/publishDiagnostics` - Error/warning publishing

## VSCode Setup

### Settings

Add to `settings.json`:

```json
{
  "sounio.serverPath": "${workspaceFolder}/tools/lsp/sounio-lsp.sh",
  "sounio.trace.server": "messages",
  "sounio.inlayHints.enabled": true,
  "sounio.inlayHints.typeHints": true,
  "sounio.inlayHints.effectHints": true,
  "sounio.inlayHints.confidenceHints": true,
  "sounio.epistemic.enabled": true,
  "sounio.epistemic.showBadges": true
}
```

### Available Commands

| Command | Keybinding | Description |
|---------|------------|-------------|
| `Sounio: Run Current File` | `F5` | Run the current file |
| `Sounio: Run Current File (JIT)` | `Shift+F5` | Run with JIT compilation |
| `Sounio: Check Current File` | `Ctrl+Shift+B` | Type-check the file |
| `Sounio: Format Document` | - | Format the current document |
| `Sounio: Organize Imports` | - | Sort and clean imports |
| `Sounio: Add Import...` | - | Add a new import |
| `Sounio: Show Document Outline` | - | Open symbol outline |
| `Sounio: Find All References` | - | Find symbol usages |
| `Sounio: Rename Symbol` | - | Rename symbol safely |
| `Sounio: Show Signature Help` | - | Show function signature |
| `Sounio: Show AST` | - | Display AST in terminal |
| `Sounio: Show HIR` | - | Display HIR in terminal |
| `Sounio: Show HLIR` | - | Display HLIR in terminal |
| `Sounio: Toggle Epistemic Mode` | `Ctrl+Shift+E` | Toggle epistemic visualization |
| `Sounio: Show Confidence Info` | `Ctrl+Shift+C` | Show confidence panel |
| `Sounio: Show Provenance Chain` | `Ctrl+Shift+P` | Show provenance panel |
| `Sounio: Show Uncertainty Info` | - | Show uncertainty analysis |
| `Sounio: Start REPL` | - | Start interactive REPL |
| `Sounio: Restart Language Server` | - | Restart the LSP |

### Code Lens

The extension provides code lens actions above function definitions:
- ▶️ Run - Run the file
- ✓ Check - Type-check the file

### Inlay Hints

The LSP supports inlay hints for:
- Type annotations on `let` bindings
- Confidence levels on epistemic values
- Effect annotations
- Unit type hints

## Neovim Setup (`lspconfig`)

```lua
require('lspconfig.configs').sounio = {
  default_config = {
    cmd = { 'tools/lsp/sounio-lsp.sh' },
    filetypes = { 'sounio' },
    root_dir = function(fname)
      return vim.fs.dirname(vim.fs.find({'d.toml', '.git'}, { upward = true, path = fname })[1])
    end,
    settings = {
      sounio = {
        inlayHints = { enabled = true },
        epistemic = { enabled = true }
      }
    }
  },
}

-- Keymaps
vim.keymap.set('n', 'gd', vim.lsp.buf.definition, { buffer = bufnr })
vim.keymap.set('n', 'K', vim.lsp.buf.hover, { buffer = bufnr })
vim.keymap.set('n', '<leader>rn', vim.lsp.buf.rename, { buffer = bufnr })
vim.keymap.set('n', 'gr', vim.lsp.buf.references, { buffer = bufnr })
vim.keymap.set('n', '<leader>ca', vim.lsp.buf.code_action, { buffer = bufnr })
vim.keymap.set('n', '<leader>f', vim.lsp.buf.format, { buffer = bufnr })
vim.keymap.set('n', '<leader>ds', vim.lsp.buf.document_symbol, { buffer = bufnr })
vim.keymap.set('n', '<leader>ws', vim.lsp.buf.workspace_symbol, { buffer = bufnr })
```

## Emacs Setup (`lsp-mode`)

```elisp
(use-package lsp-mode
  :hook (sounio-mode . lsp)
  :commands lsp)

;; Register Sounio LSP
(with-eval-after-load 'lsp-mode
  (add-to-list 'lsp-language-id-configuration '(sounio-mode . "sounio"))
  
  (lsp-register-client
   (make-lsp-client :new-connection (lsp-stdio-connection "tools/lsp/sounio-lsp.sh")
                    :major-modes '(sounio-mode)
                    :server-id 'sounio-lsp)))
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
wire = b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
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

## Smoke Test Suite

Run the local smoke suite:

```bash
bash tools/lsp/test_smoke.sh
```

This validates:

- Parser conversion (`parse_diagnostics.sh`)
- Framed lifecycle (`initialize`/`shutdown`/`exit`)
- `didOpen -> publishDiagnostics` flow
- `didClose -> publishDiagnostics([])` cleanup flow
- `didChange` unsaved buffer used by `didSave` diagnostics
- `hover/definition` request roundtrip
- `completion` request roundtrip
- `formatting` request roundtrip
- `documentSymbol` request roundtrip
- `codeAction` request roundtrip
- `references` request roundtrip
- `rename` request roundtrip
- `signatureHelp` request roundtrip
- `documentHighlight` request roundtrip
- Multi-document `didSave` sequencing keeps diagnostics isolated per URI
- Multi-document `didOpen -> didChange -> didSave` roundtrip on both URIs
- Strict no-rust fail-fast behavior
- Explicit synthetic diagnostics for timeout and non-timeout check failures

CI/automation entrypoint:

```bash
bash scripts/lsp_smoke_gate.sh
```

Gate marker emitted on success: `LSP_SMOKE_PASS`.

## Notes

- `jq` and `python3` are required.
- LSP line/character are 0-based; compiler diagnostics are 1-based.
- The server kills stale check processes before a new diagnostic run.
- `SOUNIO_LSP_CHECK_TIMEOUT_SEC` controls `souc check` timeout (default: `60`).
- When a `souc check` times out or fails without parseable diagnostics, the server emits a synthetic diagnostic (`source: "sounio-lsp"`) instead of returning an empty list.
- When a document is open in the LSP session, diagnostics/hover/definition use the in-memory buffer snapshot (not only on-disk file contents). This keeps editor feedback aligned with unsaved changes.
- The server keeps a per-URI check token and suppresses stale diagnostic publishes, so rapid saves/changes across multiple open documents do not cross-contaminate results.

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

## Architecture

The BIG LSP is built with:

1. **Bash core** - Message parsing and dispatch
2. **jq** - JSON manipulation
3. **Python3** - Complex text processing (completions, symbols, references)
4. **souc compiler** - Source of truth for types, diagnostics, and formatting

This hybrid approach provides:
- Fast startup (no compilation needed)
- Easy modification (bash is readable)
- Powerful text processing (Python for AST analysis)
- Full compiler integration (always accurate)
