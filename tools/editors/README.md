# Editor integrations

The Sounio preview LSP speaks JSON-RPC over stdio and is invoked as
`souc lsp --stdio`. In the checked preview contract this routes through
`tools/lsp/sounio-lsp.sh`; rebuilding the pure-Sounio server from
`self-hosted/lsp/server.sio` is tracked separately.

| Editor | Path | Status |
|---|---|---|
| VS Code | `editors/vscode/` | Preview extension with syntax highlighting, snippets, command wiring, and LSP client |
| Helix | `editors/helix/languages.toml` | LSP-only (no tree-sitter grammar yet) |
| Neovim | `editors/neovim/lspconfig.lua` | LSP via `nvim-lspconfig` |

See `tools/lsp/README.md` for the LSP feature surface.

## Quick install

### VS Code

```bash
cd tools/editors/vscode
npm install
npm run compile
# Then F5 inside VS Code to launch an extension-host window, or
# `vsce package` and install the resulting .vsix.
```

If `souc` isn't on PATH, the extension falls back to
`<workspace>/bin/souc`, so dragging a Sounio source repo into VS Code
uses the in-tree preview tooling.

### Helix

```bash
cat tools/editors/helix/languages.toml >> ~/.config/helix/languages.toml
```

Open a `.sio` file in Helix; `hx --health sounio` should now list
`sounio-lsp` as configured.

### Neovim

```bash
mkdir -p ~/.config/nvim/lua
cp tools/editors/neovim/lspconfig.lua ~/.config/nvim/lua/sounio.lua
# Then add `require('sounio')` to ~/.config/nvim/init.lua
```

Requires `neovim/nvim-lspconfig`.

## Sharing the TextMate grammar

`editors/vscode/syntaxes/sounio.tmLanguage.json` is the syntax-
highlighting source of truth. VS Code uses it directly. For Helix /
Neovim a tree-sitter grammar is the right long-term path; until that
exists, syntax highlighting in those editors will be plain or relies on
external plugins.
