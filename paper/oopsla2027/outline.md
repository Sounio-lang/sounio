# Sounio: A Self-Hosted Systems Language for Verifiable Scientific Computing

**Target:** OOPSLA 2027 (two review rounds: ~April and ~October)
**Authors:** Demetrios Chiuratto Agourakis, Marli Gerenutti
**Format:** PACMPL, ACM small format

---

## The Pitch

This is the SYSTEMS paper. Not type theory — engineering.

"We built a 410,000-line self-hosted systems language for scientific
computing, with a three-stage verified bootstrap, native ELF/Mach-O code
generation, and a Scientific IR that treats epistemic operations as
first-class instructions. Here's how it works and how it performs."

OOPSLA reviewers want: ambitious system, real implementation, honest evaluation.

---

## Section Plan

### 1. Introduction
- The gap: systems languages lack scientific semantics, scientific languages
  lack systems performance
- Sounio bridges this: 138K self-hosted compiler + 272K stdlib
- Three-stage bootstrap with SHA-256 parity verification
- Contributions:
  1. Scientific IR (SIR) with epistemic/ODE/tensor instructions
  2. Self-hosted compiler with verified bootstrap
  3. Multi-backend codegen (native ELF/Mach-O, LLVM, Cranelift, GPU)
  4. Evaluation on pharmacokinetic modeling and uncertainty benchmarks

### 2. Language Design (from user's perspective)
- Epistemic types: Knowledge<T> with automatic GUM propagation
- Linear types: resource safety for GPU buffers, file handles
- Algebraic effects: IO, Mut, Div, Async, GPU, Epistemic
- Units of measure: compile-time dimensional analysis
- Code examples in Sounio syntax (NOT Rust)

### 3. Compiler Architecture
- Pipeline: Source → Lexer → Parser → AST → Check → HIR → SIR → HLIR → Backend
- Bidirectional type inference with epistemic extensions
- The type checker: 138K lines, handles generics, turbofish, bounds

### 4. Scientific IR (SIR) — The Novel Contribution
- Why a domain-specific IR for science (vs lowering to LLVM directly)
- SIR instruction set: epistemic ops, ODE integration, tensor ops, autodiff
- Optimization passes:
  - Variance fusion (consecutive epistemic additions → single RSS)
  - Exact elision (ε=0 skips variance computation)
  - SIMD vectorization (value + variance in parallel)
- Example: SIR dump for a PBPK function

### 5. Self-Hosted Bootstrap
- Stage 1: Rust-hosted compiler compiles Sounio compiler to SOBC
- Stage 2: Stage-1 compiler compiles itself to SOBC
- Stage 3: SHA-256 parity check (Stage 1 output = Stage 2 output)
- The journey: from 0 to 138K lines self-hosting
- SOIR v1 IR format specification

### 6. Code Generation
- Native backend: ELF64 (Linux), Mach-O (macOS)
- LLVM backend (partial)
- GPU: PTX (NVIDIA), Metal (Apple)
- The ODE bridge: Sounio RHS functions called from Rust RK4 solver

### 7. Evaluation
- 7.1 Micro-benchmarks: uncertainty propagation vs Python/Julia (Table)
- 7.2 ODE validation: caffeine PK, exponential decay (analytical comparison)
- 7.3 Compile-time overhead: epistemic checking adds 12%
- 7.4 Bootstrap verification: Stage 1 = Stage 2 (deterministic)
- 7.5 Standard library coverage: 272K lines across 20+ domains

### 8. Related Work
- Scientific languages: Julia, Fortran, Chapel, X10
- Self-hosted compilers: Go, Rust, Zig
- Domain-specific IRs: Halide, TVM, MLIR
- Uncertainty libraries: Python uncertainties, Julia Measurements.jl

### 9. Conclusion

---

## Source Material

Most content can be extracted from:
- POPL draft sections 5-7 (Implementation, Evaluation)
- TechRxiv preprint (sounio_arxiv_draft.md)
- docs/compiler/ARCHITECTURE.md
- docs/RUSTLESS_CUTOVER.md, RUSTLESS_COMPLETE.md
- benchmark data in benchmarks/ and artifacts/omega/

## What's New vs. POPL

| Aspect | ICFP (Theory) | OOPSLA (Systems) |
|--------|---------------|------------------|
| Core contribution | Type theory + Lean proofs | Compiler engineering + SIR |
| Formal content | Typing rules, metatheory | Architecture, pipeline |
| Lean proofs | Central | Supporting evidence |
| Benchmarks | None | Central |
| LOC counts | None | Central |
| Bootstrap | Not mentioned | Full section |
| GPU codegen | Not mentioned | Section |
| Related work | PL theory papers | Systems papers |
