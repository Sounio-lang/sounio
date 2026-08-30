<!-- docs:meta
topic_id: website.docs.compiler
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler
-->

# Sounio Compiler Architecture Overview

This is the current contributor-facing map of the Sounio compiler. The important correction versus older documentation is that Sounio is now best understood as a self-hosted-first compiler with an artifact-backed public workflow, not as a Rust-crate-first compiler that happens to have some self-hosted experiments.

## 1. Current operating model

There are two views you need to hold at the same time:

- implementation view: learn the compiler from `self-hosted/`
- public behavior view: validate claims against the checked compiler artifact under `artifacts/omega/`

The checked artifacts used by the docs are:

```bash
bin/souc
artifacts/omega/souc-bin/souc-linux-x86_64-gpu
```

On the current snapshot, `souc info` for the default JIT profile reports:

- version `1.0.0-beta.4`
- **Cranelift JIT NOT compiled** — `souc info` prints `[-] Cranelift JIT - rebuild
  with --features jit`. Measured 2026-08-27: no artifact enables it, no build path
  passes the feature, and the binary exports no Cranelift symbol. This bullet
  previously read "Cranelift JIT enabled" — not compiled
- LLVM disabled in the checked artifact
- GPU codegen disabled in the checked JIT artifact
- LSP, SMT, distributed, and package-manager features disabled in the checked artifact
- ontology CLI on the checked artifact via `souc ontology <init|info|search|is-subclass|list|lock|update|diff|deprecations|verify>`

> **Two binaries, two different `souc ontology` CLIs — measured 2026-08-26.** The
> command name is shared and the subcommand sets barely overlap, so which one you
> get depends on which binary you are holding:
>
> | | checked artifact (`artifacts/omega/souc-bin/souc-linux-x86_64-gpu`) | default `bin/souc` (Madaros v0.80.0) |
> |---|---|---|
> | subcommands | `init info search is-subclass list lock update diff deprecations verify` | `resolve search ancestors is-subclass map` |
> | `ontology resolve GO:0008150` | not a subcommand — prints usage | `label=biological_process`, `iri=…GO_0008150`, `provenance=local:go` |
>
> Both are real; `souc ontology bogussub` gives `unknown ontology subcommand` on
> the default binary, so its dispatch is genuine and not a fallback. A CURIE
> outside the bundled slice (`CHEBI:15377`, `LOINC:2345-7`) answers `unresolved`
> — the slice is partial, which is not the same as the command being absent, and
> reading it as absence is an easy mistake to make.
>
> `souc --help` does not list `ontology` at all on the default binary. That
> omission is the whole trap: a survey of `--help` concludes the command does not
> exist, and it does.

For the separate checked GPU profile, `souc info` reports:

- version `1.0.0-beta.4`
- GPU codegen enabled
- Cranelift JIT disabled
- public PTX emission through `build --backend gpu`
- the same disabled feature families unless you rebuild a different artifact

## 2. Pipeline map

```text
source (.sio)
  -> lexer
  -> parser
  -> resolve
  -> semantic checking
  -> HLIR
  -> lower IR
  -> backend lowering
  -> executable artifact / runtime path
```

Mapped to the current tree:

- `self-hosted/lexer/`
- `self-hosted/parser/`
- `self-hosted/resolve/`
- `self-hosted/check/` and `self-hosted/effects/`
- `self-hosted/hlir/`
- `self-hosted/ir/`
- `self-hosted/native/`, `self-hosted/wasm/`, `self-hosted/gpu/`, `self-hosted/llvm/`

## 3. Frontend

The syntax stack is decomposed into focused directories instead of a single monolithic frontend:

- `self-hosted/lexer/` covers cursor movement, token definitions, tables, and number parsing
- `self-hosted/parser/` splits expressions, statements, items, types, patterns, AST structures, and recovery helpers
- `self-hosted/compiler/lexer.sio` and `self-hosted/compiler/parser.sio` are useful driver-facing entry points, but the real implementation detail is underneath them

Resolution work then moves into `self-hosted/resolve/`, which is the right place to inspect import, path, and module lookup behavior.

## 4. Semantic core

The center of the compiler is `self-hosted/check/`. It is large because many language features are enforced there:

- ordinary typing and inference: `types.sio`, `infer.sio`, `check.sio`, `env.sio`, `defs.sio`
- effects: `effects.sio`, `effects_row.sio`, plus the dedicated `self-hosted/effects/` subtree
- epistemics: `epistemic.sio`
- units: `units.sio`
- ownership and borrowing: `ownership.sio`, `borrow.sio`, `borrows.sio`, `lifetimes.sio`
- pattern reasoning: `patterns.sio`, `pat_decision.sio`, `exhaustiveness.sio`
- traits, specialization, and refinements: `traits.sio`, `specialization.sio`, `refinement.sio`

If a docs claim involves types, effects, provenance, confidence, units, or resource rules, this is usually the first place to verify it.

## 5. IR and optimization layers

After semantic checking, lowering continues through dedicated IR layers:

- `self-hosted/hlir/` for higher-level lowering and strategy-oriented transforms
- `self-hosted/ir/` for lower IR, serialization, normalization, SSA, constant propagation, DCE, inlining, and backend-facing transforms
- `self-hosted/analysis/` for escape, lifetime, and abstract-interpretation work

Practical files for orientation:

- `self-hosted/hlir/lower.sio`
- `self-hosted/ir/lower.sio`
- `self-hosted/ir/optimize.sio`
- `self-hosted/ir/ssa.sio`
- `self-hosted/ir/serialize.sio`

## 6. Backends and tooling surfaces

Backend-oriented work is distributed by target:

- `self-hosted/native/` for native lowering, frame layout, relocation, ELF support, and backend tests
- `self-hosted/wasm/` for WASM lowering and encoding
- `self-hosted/gpu/` for PTX, SPIR-V, Metal, tensor-oriented, and GPU lowering work
- `self-hosted/llvm/` for LLVM-specific paths

Tooling-adjacent subsystems live nearby:

- `self-hosted/lsp/` for language-server functionality
- `self-hosted/bootstrap/` for multi-stage verification work
- `self-hosted/tools/` for support tooling

The repo therefore contains more backend and tooling work than any one checked
artifact exposes. Always separate "implemented in source" from "enabled in the
artifact you are discussing."

## 7. Evidence-backed status

The most useful committed status signals today are:

- `artifacts/stdlib/stdlib_reliability_status.v1.json`
- `artifacts/stdlib/stdlib_science_pipeline_status.v1.json`
- `artifacts/stdlib/stdlib_hyper_execution_status.v1.json`

Current status:

- stdlib reliability totals: `251 pass / 0 fail / 0 skip / 251 total`
- stdlib inventory: `927` `.sio` files, `0` disabled files, `0` stub module files, `119` active module entrypoints
- science pipeline: `2/2` required lanes passing
- hyper execution: `7/7` required lanes passing (`nn`, `onn`, `qnn`, `snn`, `spnn`, `quantnn`, `math`)
- science runtime regressions remain tracked separately and currently show `0` failures under soft local enforcement

These artifacts matter more than aspirational architecture diagrams when you are documenting what is reliable right now.

## 8. Recommended navigation order

If you are new to the implementation:

1. run `bin/souc info`
2. run `artifacts/omega/souc-bin/souc-linux-x86_64-gpu info` if the claim touches GPU codegen
3. read `self-hosted/compiler/main.sio`
4. inspect the relevant subsystem directory under `self-hosted/`
5. confirm the claim with a run-pass, compile-fail, or `build --backend gpu` fixture
6. read the committed status JSON before documenting stdlib or science support

## 9. Common documentation mistakes to avoid

- describing the active compiler as primarily `crates/souc/src/*`
- claiming backend support from source-tree presence alone
- describing GPU as globally disabled because the default JIT profile reports it off
- assuming all stdlib modules are equally callable
- treating `check` success as proof that every runtime and codegen path is equally mature

Use this file as the current architecture map. Historical deep reports can still be useful, but contributor-facing docs should anchor to the self-hosted tree and the checked artifact first.
