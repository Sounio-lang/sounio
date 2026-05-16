# Sounio LSP

`./bin/souc lsp` runs the Sounio Language Server — a pure-Sounio JSON-RPC
implementation that speaks LSP over stdio.

## What you get

Eight methods, all in pure Sounio (no Python, no jq, no bash hybrid):

| Feature | LSP method |
|---|---|
| Lifecycle | `initialize`, `initialized`, `shutdown`, `exit` |
| Diagnostics (push) | `textDocument/publishDiagnostics` on `didOpen` / `didChange` |
| Hover | `textDocument/hover` — keyword / type / effect / stdlib table |
| Completions | `textDocument/completion` — static table + document scan |
| Go-to-definition | `textDocument/definition` |
| References | `textDocument/references` (honors `includeDeclaration`) |
| Rename | `textDocument/rename` (emits `WorkspaceEdit`) |

Diagnostics are sourced from a real `souc check` subprocess via
fork+execve, so every error you see in the IDE is one the compiler
actually reports.

For non-IDE consumers, the same diagnostics are available via the CLI:

```bash
./bin/souc check --json path/to/file.sio
# {"schema":"sounio.diagnostic.v1","uri":"file://...","diagnostics":[...]}
```

The JSON conforms to `tools/shared/diagnostic_schema.json` — the
canonical wire format also consumed by the MCP server.

Type-checker-inferred function info at a position:

```bash
./bin/souc inspect --pos LINE:COL path/to/file.sio
# {"schema":"sounio.inspect.v1","name":"helper",
#  "signature":"fn helper : arity 1 -> i64 with Mut,Panic", ...}
```

`signature` is the real type-checker output (effects included). The
LSP hover handler currently uses a static lookup table for keywords /
stdlib names; wiring it to call `souc inspect` for user-defined
functions is straightforward future work.

## Install

The server is a single Sounio source file. Build it once:

```bash
./bin/souc compile self-hosted/lsp/server.sio -o bin/sounio-lsp
```

Or just invoke `./bin/souc lsp` — the wrapper auto-builds
`bin/sounio-lsp` when the source is newer than the binary.

## Editor setup

### VS Code

Install the `vscode-sounio` extension from `tools/editors/vscode/`.
It defaults to `serverPath: "souc"` and spawns `souc lsp --stdio`.

For an in-tree dev session where `souc` is not on PATH, set:

```jsonc
// .vscode/settings.json
{ "sounio.serverPath": "/absolute/path/to/sounio/bin/souc" }
```

### Helix

See `tools/editors/helix/languages.toml` (forthcoming).

### Neovim

See `tools/editors/neovim/lspconfig.lua` (forthcoming).

## Architecture

- `self-hosted/lsp/server.sio` — single-file Sounio program, ~2000 lines.
  Reads framed JSON-RPC from stdin byte-at-a-time, dispatches by method,
  emits framed responses on stdout.
- `tools/shared/diagnostic_schema.json` — canonical wire format for
  diagnostics shared with the MCP server.

## Limits (v0)

- Single-document state: `didOpen` / `didChange` overwrite one in-memory
  buffer. Cross-file go-to-def and references will land when the
  document store grows beyond one buffer.
- Hover content is sourced from a static table (50 names) of keywords,
  primitive types, effects, and stdlib functions. Inferred-type display
  awaits a `souc inspect --pos` subcommand.
- No formatting, no code-actions, no semantic tokens. Capabilities
  advertised match what the server actually answers.

## Deprecated

The bash + jq + python3 LSP that used to live here is preserved as
`sounio-lsp.sh.deprecated` for one release, then removed.
