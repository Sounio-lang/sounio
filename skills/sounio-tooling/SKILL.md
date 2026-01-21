---
name: sounio-tooling
description: "Work on Sounio tooling: LSP, VSCode extension, playground (WASM), and Jupyter kernel; use when editing `compiler/src/lsp/`, `editors/vscode/`, `playground/`, or `jupyter/`."
---

# Sounio Tooling

## Overview

Make changes to developer tooling without breaking the compiler’s APIs or the editing experience.

## Workflow

### 1) Pick the target

- LSP server: `compiler/src/lsp/`
- VSCode extension: `editors/vscode/`
- Playground: `playground/`
- Jupyter kernel: `jupyter/`

### 2) Keep interfaces stable

- When changing AST/types/diagnostics, update LSP features that depend on them (hover, completion, semantic tokens).
- Prefer adding new fields over breaking existing protocol payload shapes.

## References

- Tooling map: `references/tooling-navigation.md`
