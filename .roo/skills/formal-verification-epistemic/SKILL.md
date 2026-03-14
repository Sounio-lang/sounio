---
name: formal-verification-epistemic
description: Integrate formal verification tools (SMT solvers, Lean) to prove properties of epistemic programs, such as uncertainty bounds, confidence guarantees, and provenance integrity, establishing Sounio as a language with mathematically verified scientific claims.
---

# Formal Verification of Epistemic Programs

## When to use this skill

Use this skill when you need to:

- Prove that a Sounio program’s uncertainty propagation obeys specified bounds
- Verify that confidence updates follow Bayesian coherence
- Ensure provenance chains are tamper‑proof and complete
- Integrate SMT solvers (Z3, cvc5) or proof assistants (Lean, Coq) with the Sounio compiler
- Add refinement‑type constraints that involve epistemic quantities

## When NOT to use this skill

- For ordinary type checking (use the existing checker)
- For runtime validation of epistemic values (use the standard library’s runtime checks)
- For performance optimizations unrelated to verification

## Inputs required

- Specification of the property to be verified (e.g., “output variance ≤ 0.1”)
- References to formal definitions (probability theory, measure theory, GUM)
- Choice of verification backend (SMT, Lean, custom)
- Expected interaction model (compile‑time proof, runtime proof certificate)

## Workflow

1. **Understand the existing verification infrastructure**
   - Read `docs/compiler/REFINEMENT_TYPES.md` and `self‑hosted/check/refinement.sio`
   - Examine the SMT‑solver integration (if any) in `self‑hosted/smt/`
   - Review the epistemic semantics (`docs/epistemic/SEMANTICS.md`)

2. **Design the verification extension**
   - Decide which properties are verifiable (variance bounds, confidence thresholds, provenance acyclicity)
   - Choose a verification backend and define the interface
   - Determine whether proofs are constructed at compile time or verified at runtime

3. **Extend the refinement‑type system**
   - Add new refinement predicates that refer to epistemic attributes (`variance`, `confidence`, `provenance`)
   - Modify the refinement checker to interact with the chosen verification backend

4. **Implement the verification backend integration**
   - Write a bridge that translates Sounio epistemic constraints into SMT‑LIB or Lean terms
   - Handle proof reconstruction and error reporting

5. **Write tests**
   - Add compile‑pass tests where verification succeeds
   - Add compile‑fail tests where verification fails (with appropriate error messages)
   - Include cross‑validation with numerical simulations

6. **Update documentation**
   - Extend the refinement‑types guide with epistemic examples
   - Document the verification backend’s capabilities and limitations
   - Provide a tutorial on proving a simple epistemic property

## Examples

### Variance‑bound refinement

```sio
// Function that guarantees output variance ≤ 0.01
fn stable_measurement(x: Knowledge<f64>) -> { y: Knowledge<f64> | y.var() ≤ 0.01 } {
    // Implementation that preserves the variance bound
    x.scale(1.0)  // placeholder
}
```

### Confidence monotonicity proof

```sio
// Lemma: combining independent evidence never reduces confidence
lemma confidence_monotone(
    a: Knowledge<f64>,
    b: Knowledge<f64>
) -> { combined: Knowledge<f64> |
    combined.confidence.mean() ≥ a.confidence.mean() &&
    combined.confidence.mean() ≥ b.confidence.mean()
} {
    // Proof constructed via SMT solver
    a.combine(b)
}
```

### Provenance acyclicity

```sio
// Refinement that ensures a provenance chain contains no cycles
type AcyclicKnowledge<T> = {
    k: Knowledge<T> |
    is_acyclic(k.provenance)
}

fn derive_safely(source: AcyclicKnowledge<f64>) -> AcyclicKnowledge<f64> {
    // Derivation that preserves acyclicity
    source.transform("square", |x| x * x)
}
```

## Troubleshooting / edge cases

- **Undecidability**: many epistemic properties are not decidable; fall back to runtime checks or approximate proofs.
- **Solver performance**: complex constraints may time out; consider using approximate solvers or heuristics.
- **Proof erosion**: when the compiler changes, existing proofs may break; design a proof‑versioning scheme.

## Related files

- `self‑hosted/check/refinement.sio` – refinement‑type checking
- `self‑hosted/smt/` – SMT‑solver integration (if present)
- `docs/compiler/REFINEMENT_TYPES.md` – refinement‑type documentation
- `docs/epistemic/SEMANTICS.md` – formal semantics of epistemic types

## Next steps

After implementing formal verification for epistemic programs, consider:

- Extending verification to other scientific domains (units of measure, physical constraints)
- Generating proof certificates that can be independently checked
- Integrating with theorem‑proving communities (e.g., Mathlib for Lean)