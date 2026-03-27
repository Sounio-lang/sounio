<!-- docs:meta
topic_id: repo.docs.research.sedenion-zero-divisor-geometry-report
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-zero-divisor-geometry-report
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Sedenion Zero-Divisor Geometry: Primitive Two-Support Census

Reproduction command:

```bash
python3 scripts/research/generate_sedenion_zero_divisor_geometry.py
```

Artifact:

- `artifacts/research/sedenion_zero_divisor_geometry.v1.json`

## What This Checks

The generator mirrors `self-hosted/ir/algebra.sio::ir_cd_sigma` exactly and
enumerates primitive sign-quotiented two-support imaginary vectors
`e_i +/- e_j` for `1 <= i < j <= 15`. It then computes all pairs `(a, b)` with
`a * b = 0` using exact integer arithmetic.

This keeps the research artifact aligned with the compiler's own
Cayley-Dickson sign law rather than a separate textbook implementation.

## Stable Counts

- `336` ordered primitive zero-divisor pairs
- `168` unordered projective classes
- `84` primitive projective vectors that actually participate
- `42` distinct `2 + 2` support quartets
- `7` disconnected xor-labeled graph components

These are deterministic outputs of the checked-in script and match the usual
`336 = 2 x 168` headline.

## Structural Findings

1. Every primitive participating vertex is mixed-half.
   The support is always one lower index in `{1, ..., 7}` and one upper index
   in `{8, ..., 15}`.

2. The participating set is not all mixed-half vectors.
   Exactly `28` mixed candidates are excluded. Empirically, the excluded cases
   are precisely the ones touching `e8` or having xor-label `8`
   (the diagonal family `e_i +/- e_{i+8}`).

3. The annihilation graph splits into `7` identical xor fibers.
   Each component has:
   - `12` vertices
   - `24` unordered annihilation pairs
   - degree `4` at every vertex
   - bipartition sizes `6 + 6`
   - a single constant xor label in `{9, ..., 15}`

4. The `42` support quartets are perfectly uniform.
   Each unordered projective pair uses a support union of the form
   `{i, j, k, l}` with `2` lower and `2` upper indices, and each such quartet
   supports exactly `4` unordered projective pairs.

5. The annihilator signature is uniform.
   For every participating primitive vertex, left and right multiplication have
   rank `12`, hence nullity `4`.

## Interpretation

The primitive zero-divisor geometry does not look like a single Fano-plane-like
incidence object. What the census suggests instead is a **Fano-indexed xor
bundle**:

- `7` xor fibers
- each fiber carrying `12` primitive vertices and `24` projective pairs
- `42` assessor-style support quartets spread uniformly across the bundle

That is a much more rigid picture than "168 unrelated classes", but it still
does **not** expose a natural `11`-fold structure. In the primitive census, the
stable factors are `4`, `6`, `7`, `12`, `24`, `42`, `84`, `168`, and `336`.

## Relation To `1848 = 11 x 168`

The script explicitly probes three natural fibrations:

- support quartets: `42 x 4 = 168`
- xor components: `7 x 24 = 168`
- annihilator signatures: completely uniform, so no nontrivial factorization

None of them produces a stable factor `11`. The current evidence therefore
pushes in one direction:

- the `11` in `1848 = 11 x 168` appears to belong to the associator side of the
  tower geometry
- the primitive zero-divisor side seems governed instead by `7` xor fibers and
  `42` support quartets

That is not a proof, but it is a clean computational constraint on any deeper
conjecture.

## Next Lean-Friendly Targets

- Prove the `336 / 168` counts by `native_decide`.
- Prove the `7`-component xor decomposition.
- Prove that every primitive participating vertex has nullity `4`.
- Formalize the `42` support quartets and compare them to de Marrais'
  assessor language without over-identifying the two structures.
