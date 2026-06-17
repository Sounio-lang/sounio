---
name: deep-algebra
description: Work on deep algebraic formalization, proofs, and structures in Lean 4 and Sounio
user-invocable: true
allowed-tools: Bash, Read, Edit, Write, Glob, Grep
---

# Deep Algebra

Use this skill for formal algebra work in the Sounio repository:
- Algebraic structures (groups, rings, fields, octonions, sedenions)
- Representation theory and automorphism lifts
- Polynomial, Galois, and number-theoretic proofs
- Lean 4 metatheory (weakening, substitution, progress, preservation)

## Entry points

```bash
cd /workspace/sounio
ls formal/lean4/Sounio*.lean
ls formal/lean4/*.lean
```

## Build

```bash
cd formal/lean4
lake build <ModuleName>
# or
lake build
```

## Check axioms / sorry

```bash
lake env lean --run scripts/check_axioms.lean MyTheorem 2>/dev/null || true
```

For any new theorem, run `#print axioms MyTheorem` to ensure the axiom footprint is expected.

## Pattern: adding a new algebraic witness

1. Define the structure and its operations in a Lean file under `formal/lean4/`.
2. Prove closure, associativity / alternative laws, and any automorphism / representation claims.
3. Add a Sounio example or stdlib module that mirrors the construction.
4. Run `lake build <Module>` and the matching gate.
5. Request math-review offload before commit.

## Offload requirements

Every new theorem statement, axiom, or numeric / algebraic claim requires `bin/llm-offload -t math-review -p xai` and a log entry in `.claude/llm_offload_log.md`.

## Common files

| Topic | File pattern |
|---|---|
| Octonion / sedenion automorphisms | `formal/lean4/SounioG2Derivations.lean` |
| Epistemic effect metatheory | `formal/lean4/EpistemicEffectsV2.lean` |
| SAT / colouring certificates | `formal/lean4/SounioSatColouring*.lean` |
| Algebraic witnesses | `formal/lean4/Sounio*Witness.lean` |
| Core utilities | `formal/lean4/Sounio*.lean` |

## Caution

- Do not use `native_decide` for infinite or analytic claims; it is only for finite combinatorial witnesses.
- Keep Mathlib-free unless the task explicitly allows the dependency.
- Separate `sorry` placeholders from discharged theorems in reports.
