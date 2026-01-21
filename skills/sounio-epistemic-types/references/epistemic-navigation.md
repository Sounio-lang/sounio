# Epistemic navigation

## Semantics + contracts

- `stdlib/epistemic/SEMANTICS.md` (non-negotiable invariants)
- `docs/api/EPISTEMIC_API.md` (API notes for promotion lattice, KEC, etc.)

## Compiler implementation (Rust)

- `compiler/src/epistemic/knowledge.rs` (core `Knowledge` structure + ontology binding types)
- `compiler/src/epistemic/promotion.rs` (uncertainty model promotion lattice)
- `compiler/src/epistemic/kec.rs` (backend selection logic)
- `compiler/src/epistemic/provenance.rs` (provenance representation)
- `compiler/src/dependent/types.rs` (dependent epistemic types)
- `compiler/src/dependent/subtyping.rs` (subtyping rules + proofs)

## Frontend surface

- `compiler/src/ast/mod.rs` (`TypeExpr::Knowledge`, `TypeExpr::Quantity`, `TypeExpr::Refinement`, …)
- `compiler/src/parser/tests/epistemic.rs` (parser ground truth)

## Useful searches

- `rg -n \"TypeExpr::Knowledge|Knowledge \\{\" compiler/src`
- `rg -n \"Confidence|Uncertainty|Provenance\" compiler/src/epistemic stdlib/epistemic`
