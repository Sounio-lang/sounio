<!-- docs:meta
topic_id: repo.docs.internal.concepts.policy-observation-associator
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.policy-observation-associator
-->

# Policy-Observation Associator

Concept-ID: `SOUNIO-POLICY-OBSERVATION-ASSOCIATOR`

Status: executable finite synthetic specification

Canonical surface:
`stdlib/epistemic/proof_carrying_policy_observation_associator.sio`

## Meaning

This concept names a typed counterexample to associativity for one declared
partial policy-observation composition operator. Its carrier contains adjacent,
disjoint fragments of a frozen ordered observation program. The executable
witness uses the same typed operands, the same order, and the same operator in
both parenthesizations.

The three operands are:

1. a D5 policy context containing committed anchor evidence;
2. an evidence-commit boundary;
3. a pending synthetic coverage probe.

`((a * b) * c)` places the policy inside the scope that resolves the pending
probe and therefore withholds it. `(a * (b * c))` commits the probe before the
outer policy is composed. The fixture declares committed evidence monotone, so
the outer policy cannot erase that commitment retroactively.

## Dependencies

- `SOUNIO-POLICY-STATE-FEEDBACK` supplies the policy, coverage, burden, and
  provenance categories consumed by the D6 atoms.
- `SOUNIO-NONASSOCIATIVE-ORDER` supplies the generic associativity vocabulary.
- `SOUNIO-RELATIONAL-ASSOCIATOR` is an earlier exact counterexample on a
  different weighted rational carrier.
- `SOUNIO-REBRACKETING-AUTHORITY` is a separate compiler-level concept. A D6
  scientific receipt neither authorizes nor certifies an IR rewrite.
- Native `Contest<T, ...>`, `TyContest`, and `IrContest` remain compiler-owned
  surfaces. D6 uses ordinary library receipt structs and does not construct or
  bind the compiler's contest index.

## Ontology Binding

`stdlib/ontology/policy_observation_associator.sio` distinguishes composition
carriers, observation resolutions, composition-law artifacts, state
representations, evidence boundaries, and interpretation or authority claims.

This is currently a parallel nominal boundary. The ontology module and its
negative witnesses independently re-express the kernel's distinctions, but a
runtime D6 receipt is not yet transported as an ontology-typed result. A
result-identity bridge requires separate source-to-IR evidence.

## Semantic Boundary

This concept does not replace or broaden those dependencies. In particular:

- it is a counterexample for a partial operation on one composable triple, not
  a theorem about a total magma or every observation algebra;
- it is not ordinary function composition or monadic bind;
- it does not claim a causal policy-state mechanism;
- it does not establish an empirical psychiatric order effect;
- it does not establish statistical positivity, policy value, consent,
  suffering, diagnosis, treatment, or clinical authority;
- its non-retroactive commitment rule is declared inside the frozen fixture,
  not inferred as a universal law of evidence.

Because this carrier has no declared additive group, D6 witnesses inequality
directly. It does not manufacture a subtraction-valued algebraic associator.

## Rival Representation

The flattened ordered payload is identical under both parenthesizations and its
base-101 concatenation is associative on the witness; the two receipt
identities remain distinct. Retaining the grouping tree expands the payload:
an explicit frozen-tree replay recovers status `WITHHELD` or `COMMITTED`. D6
therefore does not claim irreducible memory after state expansion.

The base-101 flat checksum, base-31 evidence fingerprint, and grouping-tree
fingerprint are bounded audit conveniences. Only the six frozen flat checksums
are shown collision-free; no recurrence is a general identity or
authentication scheme.

## Execution Evidence

The reusable imported kernel is compiler-checked conformance evidence on the
current Madaros multimodule path. Runtime evidence comes from the separately
executed scalar native witness and the independent Python enumerator. The
imported kernel is not claimed as runtime-executed until multimodule thin-link
execution succeeds.

## Acceptance Surface

The D6 gate must:

- resolve canonical `bin/souc` to Madaros without engine fallback;
- typecheck the kernel, ontology, and imported API witness;
- execute the scalar native witness and independent exhaustive oracle;
- observe all category boundaries as compiler rejections;
- verify this Concept-ID and its declared dependencies;
- recursively execute the D5 gate.


## Claims Forbidden

- Do not claim clinical validity, ClinicalAuthority, or ClinicalRelease from this contract alone.
- Do not claim compiler-wide integration or production cutover from fixture evidence alone.
- Do not claim scientific truth or independent replay beyond the scoped witnesses bound in `bindings.tsv`.
