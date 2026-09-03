# Changelog

## 1.1.1

- Published under the `sounio-lang` namespace on Open VSX. 1.1.0
  briefly landed under `sounio` but was held inactive due to
  unverified-namespace policy. No code changes from 1.1.0 — same
  pure-Sounio LSP wiring, same eight LSP methods.

## 1.1.0

- Switched the language client to the pure-Sounio LSP (`souc lsp`).
  Eight LSP methods now wired end-to-end: initialize, shutdown,
  textDocument/publishDiagnostics, hover, completion, definition,
  references, rename.
- Extension auto-resolves the `souc` launcher to
  `<workspace>/bin/souc` when one exists in the workspace, removing
  the need to set `sounio.serverPath` for in-tree dev sessions.

## 1.0.0

- Initial release with TextMate grammar, snippets, language
  configuration, and a bash + python3 + jq LSP hybrid (now removed).
