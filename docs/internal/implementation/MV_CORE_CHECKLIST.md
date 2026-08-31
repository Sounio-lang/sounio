<!-- docs:meta
topic_id: repo.docs.internal.implementation.mv-core-checklist
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.implementation.mv-core-checklist
-->

# Minimum Viable Sounio Core (MVSC) — Checklist

This checklist defines the smallest coherent subset of Sounio that feels “real” end-to-end: parse → typecheck → diagnose → (optionally) run, while preserving Sounio’s epistemic invariants.

## Non‑Negotiables (Language Identity)

1. **Epistemic integrity**: uncertainty (metrology) and confidence (trust) are distinct channels and never silently discarded.
2. **Effects are explicit**: side effects are visible in types (`with IO`, `with Async`, …) and enforced by the compiler.
3. **Sounio syntax is Sounio**: no Rust macros (`println!`), no `&mut` (use `&!`), and examples must match what the compiler accepts.

## Definition of Done (Core)

### 1) Syntax + Parsing
- `let`/`var`, `fn`, `struct`, basic expressions, control flow (`if`, `for`, `while`).
- References: `&T` and `&!T` parse and roundtrip in AST.
- Effects syntax parses (`fn f() -> T with IO, Async { ... }`).
- `async`/`.await` parses (even if runtime is partial).
- Error recovery produces actionable diagnostics (not cascades).

### 2) Type System + Effects
- Type checking for primitives, structs, references, arrays/slices.
- Effect checking: missing effects are detected (e.g., calling `print` requires `IO`).
- Clear rule: what is *pure* vs *effectful* (especially for epistemic ops).

### 3) Epistemic Core (Knowledge)
Backed by `stdlib/epistemic/SEMANTICS.md` invariants:
- `Knowledge<T>` exists as a first-class type and cannot implicitly coerce to `T`.
- “No silent unwrap”: extracting `T` requires an explicit operation.
- Confidence monotonicity under pure transforms is enforced.
- Uncertainty propagation never contracts unless using an explicit evidence fusion operator.
- Provenance is append-only under transforms.

### 4) Units + Refinements (Minimal)
- Units: parsing and basic dimensional checking for a small built-in set (enough to reject obvious mismatches).
- Refinements: parse `{ x: T | predicate }`, typecheck predicate variables, and produce meaningful errors.
- Integration target: `Knowledge[{x:T|P}]` is representable, even if advanced proving is gated behind a feature.

### 5) MIR + One Backend Path
- SSA validity checks for MIR (dominance/phi sanity) run in debug/test builds.
- At least one reliable execution path:
  - Either interpreter for core constructs, or
  - Cranelift path for a small subset (enough to run `tests/run-pass` style programs).
- Optimization passes are effect-aware and conservative around memory and calls.

### 6) Tests That Prove “Realness”
- A small set of run-pass tests that cover:
  - effects (`IO`) and a pure function
  - references (`&`/`&!`)
  - basic struct usage
  - minimal epistemic ops (construct + explicit unwrap)
- compile-fail tests that cover:
  - missing effect annotation
  - illegal implicit Knowledge→T coercion
  - simple unit mismatch
  - simple refinement violation (when enabled)

## Repo Hygiene (Prevents Drift)
- Every example in `examples/` is either:
  - Guaranteed to compile in CI, or
  - Marked clearly as “aspirational” and excluded from quick-start commands.
- `docs/guide/LLM_PROGRAMMING_GUIDE.md` stays aligned with the compiler; “known limitations” are kept current.

## Fast Validation Loop

From `compiler/`:
- `cargo test` (compiler unit/integration tests)
- `cargo run -- check examples/hello.sio` (or a canonical minimal example)

From repo root:
- Keep `tests/run-pass` and `tests/compile-fail` runnable via the harness the project uses.

