<!-- docs:meta
topic_id: repo.docs.implementation.self-hosted-compiler
authority: repo_only
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.implementation.self-hosted-compiler
-->

# Self-Hosted Compiler

The self-hosted compiler is the primary implementation surface for Sounio today. It should not be described as a small six-file experiment anymore. The active tree is a decomposed compiler and tooling environment with dedicated directories for syntax, checking, IR, code generation, tooling, and bootstrap work.

## 1. Operating model

Treat these statements as the current contract:

- contributors should learn the compiler from `self-hosted/`
- users should validate behavior through the checked artifact under `artifacts/omega/`
- Rust-side crates still participate in artifact production, but they are not the main map of language implementation anymore

## 2. Top-level subsystem map

Current self-hosted directories with meaningful compiler roles:

- `self-hosted/lexer/`: tokenization, cursor, numeric parsing, and token tables
- `self-hosted/parser/`: AST-building syntax modules, expressions, items, statements, patterns, and recovery helpers
- `self-hosted/resolve/`: name resolution, imports, module lookup, and path handling
- `self-hosted/check/`: semantic core for types, inference, effects, epistemics, units, ownership, traits, refinements, exhaustiveness, monomorphization, and related checks
- `self-hosted/effects/`: dedicated effect representations, handler support, and checker-adjacent effect logic
- `self-hosted/hlir/`: higher-level lowering and strategy-oriented transforms
- `self-hosted/ir/`: lower IR, serialization, normalization, optimization, SSA, DCE, inlining, and debug helpers
- `self-hosted/native/`: native lowering, frames, relocations, ELF/object handling, and backend tests
- `self-hosted/wasm/`: lowering, module assembly, encoding, and WASM-oriented support
- `self-hosted/gpu/`: PTX, SPIR-V, Metal, tensor-oriented, and GPU lowering work
- `self-hosted/llvm/`: LLVM-oriented backend work kept as a separate subtree
- `self-hosted/lsp/`: protocol, diagnostics, hover, goto-definition, completions, rename, and code actions
- `self-hosted/bootstrap/`: bootstrap driver and multi-stage verification helpers
- `self-hosted/tools/`: support tooling around the self-hosted environment

## 3. Driver-facing entry points

The fastest way to orient yourself is:

- `self-hosted/compiler/main.sio`
- `self-hosted/compiler/module_loader.sio`
- `self-hosted/compiler/lexer.sio`
- `self-hosted/compiler/parser.sio`
- `self-hosted/compiler/typecheck.sio`
- `self-hosted/compiler/gen.sio`

Those files are useful entry points, but they are not the full implementation. In most cases they fan out into the subsystem directories listed above.

## 4. What "self-hosted-first" means in practice

When you are tracing compiler behavior:

1. start from the checked fixture or failing program
2. confirm the behavior with the checked artifact when possible
3. move into the relevant `self-hosted/` subsystem
4. only then drop into Rust-side packaging or bridge code if the issue is clearly outside the self-hosted implementation

Examples:

- syntax bug: `self-hosted/lexer/` and `self-hosted/parser/`
- type or effect bug: `self-hosted/check/` and `self-hosted/effects/`
- confidence, provenance, or unit issue: `self-hosted/check/epistemic.sio` and `self-hosted/check/units.sio`
- native or object emission issue: `self-hosted/native/`
- GPU or backend capability question: confirm with `souc info`, then inspect `self-hosted/gpu/` or `self-hosted/llvm/`

## 5. Artifact reality versus source-tree reality

The source tree contains more capability than the checked public artifact exposes by default.

For the current checked JIT artifact:

- version: `Madaros v0.80.0`
- enabled backend: Cranelift JIT
- disabled in the checked artifact: LLVM, GPU codegen, LSP, SMT, ontology, distributed, package-manager features

For the separate checked GPU artifact:

- enabled backend path: GPU codegen
- JIT disabled in that artifact
- public GPU emission path: `build --backend gpu`

That means the right phrasing is:

- "the repo contains GPU and LLVM backend work" when discussing source layout
- "the checked JIT artifact exposes Cranelift JIT by default" when discussing the default path
- "the checked GPU artifact exposes GPU codegen and PTX emission" when discussing the GPU path

## 6. Validation surfaces

Use these to keep self-hosted documentation honest:

- `tests/run-pass/` for supported behavior
- `tests/compile-fail/` for expected refusals
- `tests/stdlib/` plus committed status JSON for stdlib and science lanes
- `self-hosted/test_ir.sio` and `self-hosted/ir/` for IR-facing work
- `scripts/sounio-verify` and `Makefile.verify` for deeper bootstrap and IR inspection work

## 7. What to avoid documenting

- do not describe the compiler as primarily implemented under `crates/souc/src/`
- do not claim LLVM, GPU, or LSP support from directory presence alone
- do not describe all stdlib modules as equally active; the repository contains active, stub, and disabled surfaces side by side
- do not imply that a passing `check` means every execution path is equally mature

This file is the contributor-facing map for the self-hosted compiler. Pair it with `docs/compiler/COMPILER_ARCHITECTURE_OVERVIEW.md` for the broader pipeline view and `docs/implementation/TOOLING_SUMMARY.md` for the current toolchain.
