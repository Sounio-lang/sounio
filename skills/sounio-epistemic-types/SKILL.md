---
name: sounio-epistemic-types
description: "Work on Sounio epistemic computing: `Knowledge`/uncertainty/confidence/provenance, ontology bindings, and dependent epistemic subtyping/proofs; use when editing `stdlib/epistemic/`."
---

# Sounio Epistemic Types

## Overview

Make changes to Sounio’s epistemic system without breaking its invariants or collapsing uncertainty into confidence (or vice versa).

## Workflow

### 1) Start from the invariants

- Read `stdlib/epistemic/SEMANTICS.md` and treat it as the spec for “what must never be violated”.
- When changing semantics, update tests (or add new ones) that protect those invariants.

### 2) Locate the implementation layer you’re editing

- Stdlib semantics + user-facing APIs: `stdlib/epistemic/`

### 3) Define purity/effects explicitly

- If an epistemic operation changes confidence or provenance, decide whether it must be effectful (and enforce via effect checking).
- Keep “confidence monotone under pure transforms” enforceable by construction.

## References

- Epistemic map + search patterns: `references/epistemic-navigation.md`
