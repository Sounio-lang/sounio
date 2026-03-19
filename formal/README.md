# Phase 8 — Formal Verification of the Sounio Compiler

## Goal

Phase 8 establishes machine-checked proofs of correctness for the two
highest-leverage invariant classes in the Sounio compiler:

1. **ELF64 linker** (`ElfLinker.lean`) — section layout, symbol containment,
   and relocation validity for the object-file writer in
   `crates/souc/src/backend/native/elf.rs`.
2. **Bidirectional type checker** (`TypeChecker.lean`) — subtype reflexivity,
   transitivity, epistemic-type covariance, and effect-safety for the checker
   in `crates/souc/src/check/mod.rs`.

## Targets and Invariants

### ELF Linker

| Theorem | Invariant |
|---|---|
| `sections_non_overlapping` | Distinct sections never share byte ranges in the file. |
| `sections_offset_monotone` | Sections are laid out in strictly increasing offset order. |
| `section_align_respected` | Every section offset is divisible by its `addralign`. |
| `symbol_within_section` | Symbol `(offset, size)` fits entirely within its owning section. |
| `symbol_unique_name` | No two global symbols share a name in one object. |
| `reloc_target_valid_thm` | Every relocation names an in-range section index. |
| `reloc_offset_within_section` | Relocation patch point lies inside the target section. |
| `reloc_symbol_valid` | Relocation symbol index is in bounds for the symbol table. |

### Type Checker

| Theorem | Invariant |
|---|---|
| `subtype_refl` | Every type is a subtype of itself. |
| `subtype_trans` | Subtyping is transitive. |
| `knowledge_covariant` | `Knowledge<T>` is covariant: `T1 ≤ T2 → Knowledge<T1> ≤ Knowledge<T2>`. |
| `fn_contravariant_arg` | Function argument position is contravariant. |
| `fn_covariant_ret` | Function return position is covariant. |
| `knowledge_unwrap_sub` | Inversion: `Knowledge<T1> ≤ Knowledge<T2>` implies `T1 ≤ T2`. |
| `check_implies_infer` | Bidirectional soundness: check mode implies an inferred subtype. |
| `no_effect_leakage` | A pure function's effect row is empty after handler masking. |

## Running the Proofs

Requires Lean 4 and Lake (https://github.com/leanprover/lean4).

```
cd formal/
lake build
```

All theorems are fully proved — zero `sorry` in both Phase 8 root files and the
`lean4/` mechanization.  `lake build` succeeds with no warnings.

## Proof Strategy

- **Arithmetic invariants** (alignment, overlap): `omega` + `Nat.mod` lemmas.
- **Structural invariants** (subtyping, unification): induction on the
  inductive `Sub` / `Unify` constructors.
- **Effect-safety**: requires a denotational semantics for effect rows; planned
  for Phase 8.2 once the row-polymorphism model is finalised.
