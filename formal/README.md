# Phase 8 — Formal Verification of the Sounio Compiler

## Goal

Phase 8 establishes machine-checked proofs of correctness for the two
highest-leverage invariant classes in the Sounio compiler:

1. **ELF64 linker** (`ElfLinker.lean`) — section layout, symbol containment,
   and relocation validity, modelling the object-file writer in
   `self-hosted/native/elf.sio`.
2. **Bidirectional type checker** (`TypeChecker.lean`) — subtype reflexivity,
   transitivity, epistemic-type covariance, and effect-safety, modelling the
   checker in `self-hosted/check/` (`check.sio`, `infer.sio`, `effects.sio`).

Both are models written by hand in Lean. They are not extracted from, or
mechanically linked to, the `.sio` sources they describe. (The Rust paths this
file used to cite no longer exist; the compiler is the self-hosted Sounio tree.)

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

### What "proved" means here

Measured 2026-10-05 over the 308 tracked `.lean` files under `formal/` (excluding
`formal/lake/packages/`), with comments and string literals stripped before
matching:

| Count | Value |
|---|---:|
| `sorry` in code | 0 |
| `axiom` declarations | 61, in 10 files |
| `native_decide` uses | 741, in 110 files |

The 61 axioms are in `Epistemic.lean` (20), `OctonionAlgebra.lean` (15),
`HessianAD.lean` (6), `lean4/SounioIEEE754Spec.lean` (6),
`SecondOrderGUM.lean` (4), `lean4/SounioFloatInstance.lean` (4),
`TypeCheckerSoundness.lean` (3), `NonAssocHessian.lean` (1),
`lean4/SounioErdos90UnitSpectrum.lean` (1) and
`lean4/SounioImpossibilityChain.lean` (1). A theorem that depends on one of
them is proved relative to that axiom; `#print axioms <name>` shows which.
`native_decide` trusts the compiled evaluator (it adds `Lean.ofReduceBool` to
the trusted base) rather than the kernel alone.

To re-count: `scripts/dev/gen_axiom_inventory.sh` regenerates
[`AXIOM_INVENTORY.md`](AXIOM_INVENTORY.md) (every axiom by file, name and line;
`native_decide` per file). It walks `git ls-files`, strips `--` and `/- -/`
comments and string literals, then counts `\bsorry\b`, declarations matching
`^\s*axiom\s`, and `\bnative_decide\b` (`grep` over the raw files over-counts,
because many files state in their doc comments that they use no `sorry` or no
axioms). CI checks the inventory is current. The two `AxiomReport.lean` files
added since (one per Lake package) declare no axioms; CI runs them to print
`#print axioms` for the headline theorems (`scripts/ci/lean_axiom_report.sh`).

**EL+ closure.** `OntologyELPlusClosureComplete.lean` proves (`subBPlusC_iff`)
that, for concepts in the TBox's finite universe (`conceptUniv t`), the
boolean saturation oracle answers *true* if and only if the subsumption is
derivable in the derivation calculus `Der` defined in Lean
(`subBPlusC t C D = true ↔ Der t C D`). Completeness is with respect to `Der`;
the canonical model is a device inside the proof, not a separately stated
EL+ semantics that the theorem is about. `stdlib/ontology/elplus.sio` is a hand-written
mirror of the saturation, not extracted code: the link between the proof and
the executable is by construction and by test, not by proof (TOUR.md section 2).

## Proof Strategy

- **Arithmetic invariants** (alignment, overlap): `omega` + `Nat.mod` lemmas.
- **Structural invariants** (subtyping, unification): induction on the
  inductive `Sub` / `Unify` constructors.
- **Effect-safety**: requires a denotational semantics for effect rows; planned
  for Phase 8.2 once the row-polymorphism model is finalised.
