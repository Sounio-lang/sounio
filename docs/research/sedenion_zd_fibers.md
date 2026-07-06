<!-- docs:meta
topic_id: repo.docs.research.sedenion-zd-fibers
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.sedenion-zd-fibers
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# The 7-fiber structure of the sedenion zero-divisor graph (executed exactly)

**One line.** The 84 participating mixed-half sedenion primitives split into **exactly 7 fibers indexed by
the algebraic label `L = lo XOR hi ∈ {9..15}`**, each a **12-vertex, 24-edge, degree-4, bipartite (6,6),
connected** graph, and **annihilation never crosses fibers** (`a · b = 0 ⟹ L(a) = L(b)`). Executed by the
running language from the exact Cayley–Dickson product (decidable ℤ-equality, no float) and cross-verified
element-wise against an independent Python implementation.

## Prior work and contribution (honest scope)

The counts — 7 xor-fibers, 12 vertices / 24 pairs each, degree 4, bipartite 6+6 — were already found
**empirically, in Python**, in `SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md` (findings #3, #5). This document
does **not** claim that discovery. Its contribution:

1. **Executes** the fiber decomposition in the running Sounio language as decidable ℤ-equality — the
   fiber sizes, the uniform degree 4, the intra-fiber closure, and the 24 edges/fiber.
2. **Pins the fiber label to a closed form**: the report described "a single constant xor label in
   `{9..15}`"; here the index is exactly `L(v) = lo XOR hi`, and annihilation is proven to preserve it.
3. **Cross-verifies** the 7 specific fiber records element-wise across two souc builds and an independent
   Python oracle (guarding against souc's silent-miscompile mode).
4. **Connects it to the e8-boundary** (`sedenion_e8_boundary.md`): participation requires `lo^hi ≠ 8`, and
   among participants `lo^hi` *is* the fiber index — the same xor-grade, forbidden at 8 and partitioning
   at `{9..15}`.

As with the e8-boundary, all three legs transcribe the *same* Cayley–Dickson sign law, so the cross-check
certifies implementation-agreement, not spec-independence; the independent-spec leg is Lean `native_decide`
(see below).

## Result

| Fiber label `L = lo^hi` | vertices | edges | degree | bipartition | connected |
|---|---|---|---|---|---|
| 9, 10, 11, 12, 13, 14, 15 (each) | 12 | 24 | 4 | (6, 6) | yes |
| **total** | **84** | **168** | — | — | 7 components |

- **Intra-fiber closure** (executed): every annihilation edge satisfies `L(a) = L(b)` — `INTRA_BAD = 0`.
- **Uniform degree** (executed): every participating vertex annihilates exactly 4 partners — `DEGREE_BAD = 0`.
- **Connected + bipartite (6,6)** (oracle-verified, BFS): the two facts the souc test does not execute
  (they need graph traversal); the Python oracle confirms all 7 fibers.

## Certification

- **Executed exactly in Sounio:** `tests/run-pass/sedenion_zd_fibers.sio` (self-contained Lean-bridge
  `prim_prod`, no `[i64;2048]` import → no #637). Verdict line `FIBERS OK`, gated by the run-pass output gate.
- **Cross-toolchain verified:** `scripts/ci/sedenion_zd_fibers_gate.sh` diffs the 7 **specific** fiber
  records (`L, size, edges`) souc-vs-oracle — element-wise identical, both `PARTICIPATE 84 / DEGREE_BAD 0 /
  INTRA_BAD 0 / FIBERS OK`, oracle also `BIPARTITE_OK 7 / CONNECTED_OK 7`. Registered in CI (Contracts).
  Confirmed identical under committed `bin/souc` and a fresh stage2.

## Reproduce

```bash
SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/sedenion_zd_fibers.sio
python3 scripts/research/sedenion_zd_fibers_oracle.py
bash scripts/ci/sedenion_zd_fibers_gate.sh          # CROSS-VERIFIED 7/7
```

## Lean-friendly next target

Prove by `native_decide`: the annihilation relation on the 84 participating primitives has each connected
component equal to a level set of `L = lo ⊕ hi`, with `|component| = 12` and every vertex of degree 4 —
turning the executed fiber decomposition into a formal theorem alongside `SounioZeroDivisorBridge.lean`.
The `7 × 24 = 168` edge count is the same `168 = |PSL(2,7)|`, now exhibited as a disjoint union of 7
identical 12-vertex fibers.
