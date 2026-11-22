# Sounio Language Support for VS Code

## Canonical source and transition

The editor client and canonical grammar are maintained in [Sounio-lang/sounio-grammar](https://github.com/Sounio-lang/sounio-grammar). Submit client, grammar, configuration and snippet changes there. The two former in-tree clients have been reconciled in that repository; this directory remains a transitional copy. The historical instructions and feature lists below do not define the current release contract.

Install the VSIX from the [editor consolidation prerelease](https://github.com/Sounio-lang/sounio-grammar/releases/tag/editor-consolidation-20260920), verifying its published checksum, and configure the installed compiler through the external client's README. This is a GitHub prerelease, not a Marketplace publication. The LSP server remains with the compiler, and its supported behavior depends on the selected compiler distribution.

Core consumer migration is tracked in [PR #2554](https://github.com/Sounio-lang/sounio/pull/2554). Keep this copy until the remaining consumers have transitioned and passed their tests.

---

VS Code extension for the [Sounio](https://github.com/sounio-lang/sounio)
programming language — epistemic computing at the horizon of certainty.

## Features (preview)

Powered by the checked preview Language Server route (`souc lsp --stdio`):

- **Diagnostics** — type errors highlighted as you save.
- **Hover** — info on Sounio keywords, primitive types, effects, and
  stdlib functions.
- **Completion** — context-aware completions from the stdlib plus
  identifiers in the current file.
- **Go to Definition** (F12) — jump to where a name is declared.
- **Find References** (Shift+F12) — list every use of an identifier.
- **Rename Symbol** (F2) — rename across the current file with a
  workspace-edit preview.
- **Formatting** — routes through the checked `souc format` surface.
- **REPL terminal** — opens the file-backed `souc repl` preview.
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

The checked preview route uses the in-tree `bin/souc` wrapper. Rebuilding the
pure-Sounio server from `self-hosted/lsp/server.sio` is a separate compiler
blocker and should not be presented as green until `tools/lsp/test_protocol.sh`
or an equivalent gate passes.

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
