<!-- docs:meta
topic_id: repo.docs.internal.concepts.relational-associator
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.relational-associator
-->

# Relational Associator


Status: **executable**

Concept-ID: `SOUNIO-RELATIONAL-ASSOCIATOR`

## Founder Intent

When an explicitly declared mediation rule closes intermediate context, the
language must preserve grouping and expose any exact difference between
`(a odot b) odot c` and `a odot (b odot c)`.

## Mathematical Core

For the frozen D1 rule,

```text
x odot y = (2*x + y) / 3
[a,b,c]_odot = (a odot b) odot c - a odot (b odot c)
```

At `a=3/10`, `b=3/5`, and `c=9/10`, the two groupings are `17/30`
and `13/30`; the exact associator is `2/15`.

## Ontology Binding

`stdlib/ontology/relational_dynamics.sio` distinguishes participant state,
dyadic relational state, mediated history, grouping structure, predictive
state expansion, bounded witness, causal receipt, and clinical authority.

This first binding is a parallel nominal boundary: the ontology module and its
negative witnesses re-express the kernel's distinctions, but a runtime D1
receipt is not yet transported as an ontology-typed result. Direct result
identity remains a later interface rather than an implied bridge.

## Required Invariants

- Ordered leaves, rule, bounds, and observation schema are common across the
  two groupings.
- Intermediate closure is explicit; ordinary function composition is not
  claimed to be nonassociative.
- An associative addition control is evaluated on the same leaves.
- A state-expansion rival promotes grouping structure and replays a total,
  exact transition table for this frozen two-state family.
- Exact rational numerators and denominators are retained before reduction.

## Claims Forbidden

- The operation is a discovered law of psychotherapy, psychiatry, chemistry,
  or physiology.
- The witness establishes a real causal mechanism or a treatment rule.
- Algebraic adjacency to quantum or octonionic systems is ontological identity.
