<!-- docs:meta
topic_id: repo.docs.internal.concepts.nonassociative-order
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.nonassociative-order
-->

# Nonassociative Order


Status: **executable**

Concept-ID: `SOUNIO-NONASSOCIATIVE-ORDER`

## Founder Intent

When grouping or interaction history changes a result, the language must retain
that order instead of normalizing it under associative assumptions.

## Mathematical Core

```text
[a,b,c] = (a*b)*c - a*(b*c)
```

The associator and its norm are executable mathematical objects.

## Current Surfaces

- `stdlib/algebra/associator_field.sio`
- `stdlib/epistemic/uncertain_octonion.sio`
- `stdlib/epistemic/perturbation_graph.sio`
- `self-hosted/ir/ir.sio` (`IrAssociator`)
- `self-hosted/native/lower_ir.sio`
- K-AXI/GPU associator kernels

## Required Invariants

- `NonAssoc` is an effect obligation, not evidence of a physical ontology.
- Parenthesization remains explicit where the algebra is nonassociative.
- Correlation and order-induced variance are not double counted.
- Mathematical identities, structural interpretations, and physical claims
  remain separately labeled.

## Claims Forbidden

- A system is physically octonionic merely because the associator models it.
- `kappa * norm_sq(associator)` is a physical variance term without a binding,
  units, and discriminating experiment.
