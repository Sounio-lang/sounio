# Feature flags (compiler crate)

Common backends/features from `CLAUDE.md`:

- `--features jit` (Cranelift JIT)
- `--features llvm` (LLVM backend)
- `--features gpu` (GPU codegen)
- `--features lsp` (language server)
- `--features smt` (Z3 refinements)
- `--features ontology` (ontology integration)

When adding new backend code, keep non-default features isolated behind `#[cfg(feature = \"...\")]`.

