<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-05-10-epistemic-fermentation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-05-10-epistemic-fermentation
-->

# Epistemic Fermentation

> **Status**: Garden seed with executable path bridge | **Last validated**: 2026-05-10 | **Source**: live session capture

## Butterfly

> like a "bread" i.e., the path of the fermentation and type of fermento
> defines the bread type

Same ingredients do not make the same bread when the fermentation path differs.
Likewise, the same numeric value and the same confidence should not always be
the same knowledge.

This seed names that pressure: **epistemic fermentation**. The route by which a
value became admissible is part of the value's meaning.

## Core Idea

Two evidence values can share the same surface:

- value: `42000`
- confidence: `980` permille

and still be different knowledge because one was measured through a GUM-style
path while the other was imputed through a model path.

The Garden claim is not "these numbers are clinically meaningful." They are
deliberately inert test numbers. The claim is that Sounio should be able to make
the path difference executable: a consumer that requires measured evidence
should reject an imputed value even when the point estimate and confidence match.

## Connections

- [`2026-05-09-novelty-weather-map.md`](2026-05-09-novelty-weather-map.md)
  identifies Metrological Compilation as the first candidate for promotion.
- [`stdlib/epistemic/path.sio`](../../../../stdlib/epistemic/path.sio)
  provides path tags, typed wrappers, surface equivalence, fermentation
  equivalence, and path-family gates.
- [`tests/run-pass/epistemic_fermentation.sio`](../../../../tests/run-pass/epistemic_fermentation.sio)
  demonstrates same surface value, different path-bearing types, and accepted
  path-specific consumers.
- [`tests/compile-fail/epistemic_fermentation_wrong_path.sio`](../../../../tests/compile-fail/epistemic_fermentation_wrong_path.sio)
  demonstrates that an imputed value cannot be passed to a measured-only
  consumer.
- [`tests/stdlib/epistemic/test_path_fermentation.sio`](../../../../tests/stdlib/epistemic/test_path_fermentation.sio)
  tests the reusable path surface: same surface, different fermentation, and
  measured-family rejection for imputed and simulation paths. It also checks
  that path erasure preserves the source family and a discharge reason tag.
- [`tests/compile-fail/epistemic_path_wrong_wrapper.sio`](../../../../tests/compile-fail/epistemic_path_wrong_wrapper.sio)
  tests the reusable wrapper refusal at the call boundary.
- [`tests/compile-fail/epistemic_path_erasure_requires_discharge.sio`](../../../../tests/compile-fail/epistemic_path_erasure_requires_discharge.sio)
  tests that a path-bearing payload is not accepted where an explicitly
  discharged, path-erased value is required.
- [`tests/compile-fail/epistemic_path_direct_payload_construction.sio`](../../../../tests/compile-fail/epistemic_path_direct_payload_construction.sio)
  tests that external code cannot fabricate a `PathKnowledgeI64` literal.
- [`tests/compile-fail/epistemic_path_direct_wrapper_construction.sio`](../../../../tests/compile-fail/epistemic_path_direct_wrapper_construction.sio)
  tests that external code cannot wrap an imputed payload as a measured witness.
- [`tests/compile-fail/epistemic_path_private_field_access.sio`](../../../../tests/compile-fail/epistemic_path_private_field_access.sio)
  tests that external code must use public accessors instead of reading private
  witness fields directly.
- [`stdlib/epistemic/SEMANTICS.md`](../../../../stdlib/epistemic/SEMANTICS.md)
  already states that computational history is part of epistemic state.

## Evidence State

| Layer | Status |
| --- | --- |
| `Garden` | The bread/ferment image is captured as a repo-local research seed. |
| `Hypothesis` | Knowledge equivalence should be path-sensitive, not only value/confidence-sensitive. |
| `Executable` | Current Sounio type checking distinguishes measured and imputed wrappers, `epistemic::path` separately tests surface equivalence and fermentation equivalence, and module-private structs block external witness fabrication. |
| `Claim-ready` | No. This is a local witness, not a novelty proof, public research claim, or clinical artifact. |

## What This Is Not

- Not a clinical dosing test.
- Not proof that Sounio has fully general first-class path-dependent knowledge types.
- Not a priority claim.
- Not a complete metrological compiler design.
- Not a reason to delete legacy epistemic storage or query paths.
- Not a cryptographic anti-forgery proof. The current bridge proves
  module-private construction and field opacity for this compiler path, plus
  typed call-boundary refusal and explicit discharge.

## Next Executable Bridge

Lift the path bridge from integer-tagged wrappers into a generic epistemic path
surface:

- a generic `PathKnowledge<T, P>` shape when the compiler surface can support it
- explicit promotion rules from payload tags into type-level path witnesses
- a discharge audit that records why path information was intentionally erased
- formal contracts for constructor invariants once the local specification
  surface is ready
- generic opacity tests once `PathKnowledge<T, P>` exists

The next proof should show that "same value, same confidence" is only a surface
equivalence, while admissibility remains path-sensitive.
