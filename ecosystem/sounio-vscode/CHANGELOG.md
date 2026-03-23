# Changelog

## 0.2.0

- Merged full-featured extension with LSP support and graceful fallback
- Added **Rustism Detector** — real-time detection of Rust patterns in Sounio code with quick-fix code actions (semicolons, `&mut`, `let mut`, macros, closures, attributes)
- Added epistemic mode: confidence badges, provenance chain viewer, uncertainty analysis panels
- Added 13 commands (Run, Check, Show AST/HIR/HLIR, REPL, epistemic tools, Add Import)
- Added code lens (Run/Check above functions)
- Added inlay hints (type hints, confidence hints for epistemic values)
- Enhanced TextMate grammar: added doc comments, block comments
- Fixed snippets: corrected match expression (removed Rust-style commas), added `mainmut`, `implmut`, `fnref`, `fnapply`, `use`, `println`, `assert`, `arr`
- Added semantic token types: effect, unit, refinement, confidence, provenance
- Added confidence-level theme colors (green/yellow/red)
- Extension icon from official Sounio brand pack

## 0.1.0

- Initial release: syntax highlighting, snippets, check-on-save diagnostics
