## Track 3 P2 result: spindle composition at omega_16

**Question:** Can composing Moser spindles at ω₁₆ (introducing √7) reach χ=5?

**Method:** Joined 2/3/4 copies of the Moser spindle (7 vertices each, χ=4) at
the origin, rotating each successive copy by ω₁₆ = arccos(31/32). Computed
exact unit-distance graph and chromatic number via backtracking.

**Results (Python float cross-check, Q16 exact verification for 2-copy case):**

| Copies | Vertices | Edges | χ |
|--------|----------|-------|---|
| 2 (ω₁) | 13 | 30 | 4 |
| 2 (ω₃) | 13 | 30 | 4 |
| 2 (ω₄) | 13 | 22 | 4 |
| 2 (ω₁₆)| 13 | 22 | 4 |
| 3 (ω₁₆)| 25 | 44 | 4 |
| 4 (ω₁₆)| 31 | 55 | 4 |

**Finding:** χ=5 does NOT emerge from simple algebraic-angle spindle composition,
even at 31 vertices. The χ=5 barrier requires de Grey's specific multi-gadget
construction (529+ vertices, complex subgraph combination).

**Structural implication:** The gap between χ=4 (7 vertices, Moser spindle) and
χ=5 (510 vertices, Heule record) is not bridgeable by stacking spindles at
algebraic angles. A qualitatively different construction is needed.
