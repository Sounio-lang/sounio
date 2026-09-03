<!-- docs:meta
topic_id: repo.docs.research.erdos-sat-smt-unsat-scope-2026-06-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erdos-sat-smt-unsat-scope-2026-06-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Erdős, SAT/SMT, and UNSAT Scope for Sounio

This note pins the current claim boundary between the two Erdős-shaped lanes
that keep getting mentally conflated:

- **Erdős #90** is the planar unit-distance count problem `u(n)`: how many pairs
  at distance 1 can occur among `n` planar points.
- **Erdős #508** is Hadwiger-Nelson: the chromatic number of the plane.

These lanes both use exact geometry, graph construction, and certificate
checking, but they are not the same theorem.

## Current external facts checked 2026-06-23

- Erdős Problems lists **#90** as the unit-distance problem: whether every
  `n`-point planar set has at most `n^{1+O(1/log log n)}` unit-distance pairs.
  Source: https://www.erdosproblems.com/90.
- Erdős Problems lists **#508** as the chromatic number of the plane: the
  smallest number of colours needed to colour `R^2` so unit-distance pairs have
  different colours. Source: https://www.erdosproblems.com/508.
- The known Hadwiger-Nelson lower-bound breakthrough remains de Grey 2018:
  finite unit-distance graphs that are not 4-colourable, with a 1581-vertex
  graph in the original paper. Source: https://arxiv.org/abs/1804.02385.
- The current public status of Hadwiger-Nelson is still the open interval
  `5 <= chi(R^2) <= 7`; a `chi >= 6` claim would require a concrete finite
  unit-distance graph plus an independently checked 5-colouring UNSAT
  certificate. Erdős #508 is not solved by the Erdős #90 advance.
- In May 2026, the Erdős #90 conjectural `n^{1+o(1)}` bound was displaced by
  an OpenAI-generated construction and human-verified companion exposition.
  Sources: https://openai.com/index/model-disproves-discrete-geometry-conjecture/
  and https://arxiv.org/abs/2605.20695.
- Sawin's explicit follow-up gives a concrete exponent: infinitely many
  `n` have more than `n^1.014` unit-distance pairs, improving the recent
  inexplicit exponent and still belonging to #90, not #508. Source:
  https://arxiv.org/abs/2605.20579.
- SAT proof infrastructure is moving toward independent replay and verified
  checkers. SAT Competition 2025 requires UNSAT proofs and documents verified
  checker pipelines. Source: https://satcompetition.github.io/2025/output.html.
- VeriPB/CakePB remains the pseudo-Boolean proof-checking line for SAT/PB
  certificates, with 2026 material noting the latest VeriPB/CakePB usage and a
  new proof-format generation. Sources:
  https://satcompetition.github.io/2026/downloads/checkers/veripb.pdf and
  https://www.cril.univ-artois.fr/PB26/.
- PBLean shows the theorem-prover-facing direction: importing VeriPB PB kernel
  certificates into Lean 4 through a proved reflective checker. Source:
  https://arxiv.org/html/2602.08692v2.

## What Sounio currently has

Sounio has three relevant proof styles in the repo:

1. **SAT/UNSAT graph-colouring certificates.** The Erdős/Hadwiger lane already
   has exact graph-colouring encodings, LRAT/DRAT-style proof replay, and a
   Lean-facing SAT bridge for known examples. This supports `chi >= 5` style
   claims when geometry plus non-4-colourability are both independently checked.
2. **SMT/QF_LRA UNSAT certificates.** `stdlib/theorem/qflra_exact.sio` and
   `stdlib/theorem/smt.sio` establish the small integrated SMT path: exact
   arithmetic rows, Farkas-style UNSAT witnesses, assumption-core checks, and
   bounded DPLL(T) contracts.
3. **Lorenz i256 proof-carrying numeric receipts.** The Lorenz lane is not a
   SAT solver but follows the same discipline: producer artifacts are not
   accepted as theorems until a smaller checker verifies exact division,
   enclosure, containment, local-flowpipe, readiness, and nonclaim masks.

The new executable scope gate is:

- `stdlib/theorem/erdos_scope.sio`
- `tests/run-pass/erdos_sat_smt_unsat_scope_tiny.sio`

It makes the dangerous distinctions executable:

- #90 is not #508.
- A `chi(R^2) >= 6` preflight requires problem #508, lower bound 6, a concrete
  unit-distance graph, a 5-colour UNSAT certificate, and independent replay mask
  `31`.
- The 2026 #90 disproof can be recorded as external/human-verified while still
  not being a Sounio formalization.
- Solver-only UNSAT is not accepted without independent checker and kernel
  replay gates.

## Operational interpretation

For Sounio, the right architecture is:

```text
producer/search/solver
  -> explicit instance manifest
  -> proof object or numeric receipt
  -> tiny checker
  -> portfolio entry
  -> external or Lean replay when the claim is public-facing
```

That gives a clean split:

- SAT can search aggressively with CDCL, symmetry breaking, domain encodings,
  and FRAT/LRAT-style proof hints.
- SMT can search or eliminate aggressively, but UNSAT must collapse to Farkas,
  Alethe/LFSC-style reconstruction, or another small replayable kernel.
- PB/Cardinality should move toward VeriPB/CakePB/PBLean shape: rich producer,
  elaborator, tiny checked kernel.
- Lorenz/i256 dynamics should keep following the same receipt discipline:
  i128/i256 arithmetic is an enabling substrate, but the claim is the checked
  enclosure/flowpipe/invariant object, not the raw high-precision orbit.

## Claim boundaries

- **We have:** a Sounio architecture that can express SAT, SMT, PB, and Lorenz
  numeric proof obligations as replayable certificates.
- **We have:** known `chi >= 5`-class graph-colouring infrastructure and
  Hadwiger-Nelson scope gates.
- **We do not have:** a `chi(R^2) >= 6` theorem, a concrete non-5-colourable
  finite unit-distance graph, or a Sounio-formalized proof of the 2026 #90
  disproof.
- **Next real upgrade:** wire these scope gates into solver portfolio metadata
  only after the portfolio parser tail is stable again; before that, keep them
  as focused theorem-library checks.
