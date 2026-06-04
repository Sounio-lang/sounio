# ORC definition — LOCKED decision (2026-06-04)

**Decision.** The manuscript/headline Ollivier–Ricci curvature is the **WEIGHTED GraphRicciCurvature**
(α = 0.5) on the committed `*_edges_FINAL.csv`:

| | EN | ES | ZH | NL |
|---|---|---|---|---|
| **weighted GRC (headline)** | −0.258 | −0.155 | −0.214 | −0.270 |
| native exact unweighted-uniform (confirmation) | −0.137 | −0.068 | −0.144 | −0.196 |

The **native Sounio exact machinery computes the UNWEIGHTED-uniform-hop** ORC and is reported as an
**independent, exact-over-ℚ, Farkas-certified, Slurm-reproducible confirmation of the hyperbolic SIGN**
(κ̄ < 0 for all four) — *not* as the headline magnitude. Both definitions are negative for all four
languages, so the scientific conclusion is **robust to the ORC definition**.

**Rules.**
- Headline numbers → weighted GRC on the committed FINAL files. Retire the stale `statistical_tests_v6.4.json`
  (−0.197 family).
- Native exact numbers (−0.137/−0.068/−0.144/−0.196) → cite only as the *unweighted-uniform* independent
  certification of the sign. Never equate them with the weighted headline (mixing flag #1 in
  `orc_definition_matrix.json`).

**Future work (option A, not required).** Extend the native exact machinery to a *weighted* ORC
(weighted shortest-path ground distance + weight-proportional lazy-walk measure; exact min-cost-flow over ℚ
with rational R1.Strength weights). That would make the *certified* number match the weighted headline.
Substantial engineering; deferred because the **sign**, which both definitions agree on, is what the
scientific claim rests on.
