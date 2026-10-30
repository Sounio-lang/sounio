---
name: sounio-tooling
description: "Work on Sounio tooling: LSP, VS Code extension, and Jupyter kernel; use when editing `tools/lsp/`, `self-hosted/lsp/`, `tools/editors/vscode/`, or `ecosystem/sounio-jupyter/`."
---

# Sounio Tooling

## Overview

Make changes to developer tooling without breaking the compiler’s APIs or the editing experience.

## Workflow

### 1) Pick the target

- LSP server: `tools/lsp/` and `self-hosted/lsp/`
- VS Code extension: `tools/editors/vscode/`
- Jupyter kernel: `ecosystem/sounio-jupyter/`

### 2) Keep interfaces stable

- When changing AST/types/diagnostics, update LSP features that depend on them (hover, completion, semantic tokens).
- Prefer adding new fields over breaking existing protocol payload shapes.

## References

- Tooling map: `references/tooling-navigation.md`
