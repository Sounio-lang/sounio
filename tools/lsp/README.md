# Sounio LSP Server

A minimal Language Server Protocol (LSP) implementation for the Sounio programming language. This server wraps the `souc` compiler to provide IDE features.

## Architecture

Since Sounio's Poseidon VM has no async runtime and limited I/O capabilities, the LSP server is implemented as a bash script that:

1. Reads JSON-RPC messages from stdin
2. Dispatches to the Sounio compiler (`souc check`) for type checking
3. Parses compiler diagnostics and converts to LSP format
4. Returns JSON-RPC responses on stdout

## Requirements

- `bash` (4.0+)
- `jq` (JSON processor)
- `souc` Sounio compiler binary

## Installation

The LSP server is included in the Sounio repository at `tools/lsp/sounio-lsp.sh`.

## VSCode Configuration

Add to your VSCode `settings.json`:

```json
{
    "sounio.serverPath": "${workspaceFolder}/tools/lsp/sounio-lsp.sh"
}
```

Or configure through the UI:
1. Open Settings (Ctrl+,)
2. Search for "Sounio"
3. Set "Server Path" to the full path to `sounio-lsp.sh`

## Neovim Configuration

Using `lspconfig`:

```lua
-- Add custom LSP config for Sounio
local lspconfig = require('lspconfig')
local configs = require('lspconfig.configs')

-- Define the Sounio LSP if not already defined
if not configs.sounio then
    configs.sounio = {
        default_config = {
            cmd = { 'tools/lsp/sounio-lsp.sh' },
            filetypes = { 'sounio' },
            root_dir = function(fname)
                return vim.fs.dirname(vim.fs.find({'d.toml', '.git'}, { upward = true, path = fname })[1])
            end,
            settings = {},
        },
    }
end

-- Setup the LSP
lspconfig.sounio.setup{}
```

Using built-in LSP (without lspconfig):

```lua
vim.api.nvim_create_autocmd('FileType', {
    pattern = 'sounio',
    callback = function(args)
        vim.lsp.start({
            name = 'sounio-lsp',
            cmd = { '/path/to/sounio/tools/lsp/sounio-lsp.sh' },
            root_dir = vim.fs.root(args.buf, { 'd.toml', '.git' }),
        })
    end,
})
```

## Emacs Configuration

Using `lsp-mode`:

```elisp
(require 'lsp-mode)

(add-to-list 'lsp-language-id-configuration '(sounio-mode . "sounio"))

(lsp-register-client
 (make-lsp-client :new-connection (lsp-stdio-connection '("/path/to/sounio/tools/lsp/sounio-lsp.sh"))
                  :activation-fn (lsp-activate-on "sounio")
                  :server-id 'sounio-lsp))
```

## Supported Features

### Phase 1 (Current)

- **Diagnostics**: Type errors and warnings from `souc check`
- **Hover**: Type information at cursor position
- **Go to Definition**: Basic definition navigation (AST-based)
- **Document Sync**: Full document synchronization

### Lifecycle

- `initialize` - Server capabilities negotiation
- `initialized` - Client initialized notification
- `shutdown` - Graceful shutdown
- `exit` - Process termination

### Document Synchronization

- `textDocument/didOpen` - Document opened, initial check
- `textDocument/didSave` - Document saved, re-check
- `textDocument/didClose` - Document closed, clear diagnostics

## Compiler Integration

The LSP server calls the following `souc` commands:

```bash
# Type checking
souc check <file.sio>

# Type checking with AST output
souc check <file.sio> --show-ast

# Type checking with type information
souc check <file.sio> --show-types
```

## Diagnostic Format

Compiler diagnostics are parsed from stderr and converted to LSP format:

```
error[E017]: type mismatch
  --> path/to/file.sio:42:10
  |
42|     let x: i64 = "hello"
  |                   ^^^^^^^ expected i64, found string
```

Becomes:

```json
{
    "range": {
        "start": {"line": 41, "character": 9},
        "end": {"line": 41, "character": 16}
    },
    "severity": 1,
    "code": "E017",
    "source": "sounio",
    "message": "type mismatch: expected i64, found string"
}
```

Note: LSP uses 0-based line/character indices, while the Sounio compiler uses 1-based.

## Error Codes

Common Sounio error codes:

- `E001-E009`: Parse errors
- `E010`: Undeclared variable
- `E017`: Type mismatch
- `E035`: Missing effect
- `E037/E038`: Borrow conflict
- `E040`: Linear type violation

## Troubleshooting

### Server not starting

Check that `jq` is installed:
```bash
jq --version
```

Check that `souc` binary exists:
```bash
ls -la artifacts/omega/souc-bin/souc-linux-x86_64
# or
which souc
```

### No diagnostics appearing

1. Check VSCode Output panel → "Sounio LSP Trace"
2. Verify file has `.sio` extension
3. Run `souc check <file>` manually to see compiler output

### Process hanging

The LSP server kills stale check processes before starting new ones. If a process hangs:
```bash
pkill -f "souc check"
```

## Development

To test the LSP server manually:

```bash
# Basic lifecycle test
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}' | \
    tools/lsp/sounio-lsp.sh 2>/dev/null

# Should output Content-Length header + JSON response
```

Test diagnostic parsing:

```bash
# Create a file with an error
cat > /tmp/test.sio << 'EOF'
fn main() {
    let x: i64 = "hello"
}
EOF

# Run check and parse diagnostics
./artifacts/omega/souc-bin/souc-linux-x86_64 check /tmp/test.sio 2>&1 | \
    tools/lsp/parse_diagnostics.sh
```

## Limitations

- Full document sync only (no incremental updates)
- Limited hover/definition support (compiler doesn't expose detailed location info yet)
- No completion support (requires symbol index)
- No workspace symbols
- Formatting not implemented (use `souc fmt` separately)

## Future Enhancements

- Incremental document sync
- Completion provider
- Symbol search
- Rename refactoring
- Inlay hints for types
- Code actions (quick fixes)
