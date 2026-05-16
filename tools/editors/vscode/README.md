# Sounio Language Support for VS Code

VS Code extension for the [Sounio](https://github.com/sounio-lang/sounio)
programming language — epistemic computing at the horizon of certainty.

## Features (v1.1.0)

Powered by the pure-Sounio Language Server (`souc lsp`):

- **Diagnostics** — type errors highlighted as you save.
- **Hover** — info on Sounio keywords, primitive types, effects, and
  stdlib functions.
- **Completion** — context-aware completions from the stdlib plus
  identifiers in the current file.
- **Go to Definition** (F12) — jump to where a name is declared.
- **Find References** (Shift+F12) — list every use of an identifier.
- **Rename Symbol** (F2) — rename across the current file with a
  workspace-edit preview.
- **Syntax Highlighting** — TextMate grammar covering effects, units of
  measure, refinement types, and Sounio's epistemic keywords.

## Setup

The extension spawns the Sounio compiler as a language server:

1. Install Sounio (the `souc` CLI) — see the
   [main repo](https://github.com/sounio-lang/sounio).
2. Make sure `souc` is on `PATH`, *or* set `sounio.serverPath` in your
   VS Code settings to an absolute path (e.g.
   `"/path/to/sounio/bin/souc"`). When opening a Sounio repo, the
   extension also falls back to `<workspace>/bin/souc` automatically.

The LSP binary builds itself the first time the extension launches it
(`souc lsp` auto-compiles `bin/sounio-lsp` from
`self-hosted/lsp/server.sio` when missing).

## Commands

| Keybinding | Command |
|---|---|
| `F5` | Sounio: Run Current File |
| `Shift+F5` | Sounio: Run Current File (JIT) |
| `Ctrl+Shift+B` | Sounio: Check Current File |
| `Ctrl+Shift+C` | Sounio: Show Confidence Info |
| `Ctrl+Shift+P` | Sounio: Show Provenance Chain |
| `Ctrl+Shift+E` | Sounio: Toggle Epistemic Mode |

## Configuration

| Setting | Default | Description |
|---|---|---|
| `sounio.serverPath` | `souc` | Path to the Sounio compiler. |
| `sounio.trace.server` | `off` | LSP wire-protocol tracing. |
| `sounio.epistemic.enabled` | `true` | Confidence badges and provenance UI. |
| `sounio.epistemic.confidenceThreshold` | `0.8` | Lower confidence is flagged. |

## License

Dual-licensed under MIT OR Apache-2.0. See `LICENSE`.
