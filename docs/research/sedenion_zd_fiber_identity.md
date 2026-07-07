<!-- docs:meta
topic_id: repo.docs.research.sedenion-zd-fiber-identity
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-zd-fiber-identity
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The isomorphism type of the sedenion ZD fibers (executed exactly)

**One line.** Each of the 7 sedenion zero-divisor fibers is isomorphic to **`K_{6,6}` minus a 2-factor of
three disjoint 4-cycles** (`K_{6,6} − 3·K_{2,2}`), the bipartite 3-block color-mismatch graph. Certified
by its common-neighbor profile `(4:6, 2:24, 0:36)`, executed in Sounio (decidable ℤ-equality) and verified
on **three independent legs** — souc, a Python oracle, and Lean `native_decide`.

## From "how many" to "what shape"

Brick 1 (`sedenion_e8_boundary.md`) fixed **which** 84 of 112 primitives participate. Brick 2
(`sedenion_zd_fibers.md`) fixed **how** they annihilate: 7 fibers by `L = lo^hi ∈ {9..15}`, each 12
vertices / 24 edges / degree 4 / bipartite (6,6) / connected. This brick fixes **what each fiber IS**.

A 4-regular bipartite graph on 6+6 vertices whose complement (in `K_{6,6}`) is a 2-factor of three
4-cycles is `K_{6,6} − 3·K_{2,2}`: partition each side into 3 pairs; a vertex is adjacent to everything
on the other side except its own pair-block (adjacency = "different block"). The two signs `±` of each
support always fall in the same block, so the sign is inert to the block structure.

## The BFS-free certificate: common-neighbor profile

Over the 66 vertex-pairs of a fiber, the number of common annihilators (common neighbors) is exactly:

| common neighbors | pairs | meaning |
|---|---|---|
| 4 | 6  | same side, same `K_{2,2}` block (co-blocked) |
| 2 | 24 | same side, different block |
| 0 | 36 | opposite sides (bipartite → no common neighbor) |

Given brick 2's structure (4-regular bipartite 6+6), the profile `(6, 24, 36)` is the signature of
`K_{6,6} − 3·K_{2,2}`. It needs no graph traversal, so it is exactly what the souc test executes; the
**rigorous** "complement = three 4-cycles" isomorphism (which needs a bipartition + cycle walk) is
discharged by the Python oracle (`COMPLEMENT_C4 = 7`).

## Honest scope

The Python geometry report (`SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md`) already noted the fibers'
uniform assessor-like structure. This brick's delta: it **executes** the isomorphism-type certificate in
Sounio (the common-neighbor profile), gives the **explicit identity** `K_{6,6} − 3·K_{2,2}`, and verifies
it element-wise on three legs. All three legs transcribe the same Cayley–Dickson sign law, so the cross-
check certifies implementation-agreement; Lean `native_decide` is the independent-checker leg.

## Certification

- **Executed in Sounio:** `tests/run-pass/sedenion_zd_fiber_identity.sio` (self-contained, no #637).
  Verdict `FIBER_ID OK`.
- **Cross-toolchain:** `scripts/ci/sedenion_zd_fiber_identity_gate.sh` diffs the 7 specific fiber
  profile records souc-vs-oracle (identical; both `VERTICES 84 / FIBER_ID OK`, oracle `COMPLEMENT_C4 7`).
  Registered in CI (Contracts). Confirmed under `bin/souc` and stage2.
- **Lean:** `formal/lean4/SounioSedenionFiberIdentity.lean` `native_decide`-proves `fiber_profile`
  (every fiber's profile is `(6,24,36)`). `lake build` green; verified by the Lean Proofs CI job.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_zd_fiber_identity.sio
python3 scripts/research/sedenion_zd_fiber_identity_oracle.py
bash scripts/ci/sedenion_zd_fiber_identity_gate.sh
(cd formal/lean4 && lake build SounioSedenionFiberIdentity)
```
