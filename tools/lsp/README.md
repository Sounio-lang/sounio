# Sounio LSP Preview

`./bin/souc lsp --stdio` starts the checked Sounio language-server preview over
JSON-RPC stdio.

The current preview route is `tools/lsp/sounio-lsp.sh`, a bash + jq server that
delegates diagnostics, formatting, and compiler-backed checks to the active
`bin/souc` wrapper. The pure-Sounio server source at `self-hosted/lsp/server.sio`
is still present, but rebuilding it under the active Madaros path is a separate
blocker tracked by `scripts/ci/sounio_editor_tooling_support_gate.sh`.

No pure-Sounio LSP rebuild under current Madaros is claimed or demonstrated by
this preview gate.

## Checked Surface

The bounded editor-tooling gate is:

```bash
bash scripts/ci/sounio_editor_tooling_support_gate.sh
```

It proves:

- public `bin/souc format` / `bin/souc fmt`
- public `bin/souc repl`
- public `bin/souc lsp --stdio`
- formatter idempotency through `scripts/gates/g5a_formatter_idempotent.sh`
- file-backed REPL evaluation through `scripts/gates/g5b_repl_eval.sh`
- preview LSP smoke through `tools/lsp/test_smoke.sh`
- initialize/capability response from `souc lsp --stdio`
- static VS Code, Helix, and Neovim editor wiring

The preview LSP advertises the core editing features expected by modern LSP
clients: diagnostics, hover, definition, completion, formatting/range
formatting/on-type formatting, code actions, document symbols, workspace
symbols for the preview surface, rename, references, document highlights,
signature help, selection/folding ranges, and selected commands.

## Editor Setup

### VS Code

Install the `vscode-sounio` extension from `tools/editors/vscode/`.
It defaults to `serverPath: "souc"` and spawns `souc lsp --stdio`.

For an in-tree dev session where `souc` is not on `PATH`, set:

```jsonc
// .vscode/settings.json
{ "sounio.serverPath": "/absolute/path/to/sounio/bin/souc" }
```

### Helix

See `tools/editors/helix/languages.toml`.

### Neovim

See `tools/editors/neovim/lspconfig.lua`.

## Boundaries

This is a SOTA-preview support contract, not mature IDE support. Do not claim:

- pure-Sounio LSP rebuild under current Madaros
- semantic token delta support
- incremental text synchronization
- unopened-file workspace indexing
- marketplace-quality VS Code release
- notebook integration
- AI assistant integration

`tools/lsp/test_protocol.sh` remains the intended richer protocol path for the
pure-Sounio server. Treat it as a future closure gate until it passes against
the active compiler.
