# Codegen navigation (repo-local)

## Key directories

- Backends: `compiler/src/codegen/`
- Native binary emission/linking: `compiler/src/backend/native/`
- IRs: `compiler/src/hir/`, `compiler/src/sir/`, `compiler/src/hlir/`, `compiler/src/mir/`

## Quick searches

- Cranelift pipeline: `rg -n \"cranelift\" compiler/src/codegen -S`
- LLVM pipeline: `rg -n \"inkwell|llvm\" compiler/src/codegen/llvm -S`
- GPU pipeline: `rg -n \"PTX|SPIR\" compiler/src/codegen/gpu -S`

