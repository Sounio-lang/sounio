# HeuleGraph510 — exact unit-distance certification (Phase B, in progress)

**Status:** exact certifier built and run on real GraphData coordinates; field
audited from the data. Awaiting 4 missing vertices to close the converse audit at
the published 510/2508.

## Method
`scripts/research/heule510_exact_certifier.py` — a self-contained exact evaluator
over the number field **K = ℚ(√3,√5,√11)** (8-dim, basis `{1,√3,√5,√11,√15,√33,√55,
√165}`, XOR-graded multiplication, exact `Fraction` coefficients, field inversion via
the conjugate-norm). It parses the Mathematica `.vtx` coordinates with a recursive-
descent parser over the field (no `sympy` — sympy auto-refolds √3·√11→√33 and
corrupts coefficient extraction), then computes squared distances exactly and tests
`‖u−v‖² == 1` as an exact field identity. No floating point anywhere.

## Findings
- **Field assertion from the data:** the realised radicands are exactly
  `{3,5,11,15,33,55,165}` — **√7 is absent**. de Grey's *full* G needs √7 (the
  L-assembly rotation 2·arcsin(1/8)), but the trimmed 510-record drops every
  √7-bearing vertex, so HeuleGraph510 lives in the **degree-8** field ℚ(√3,√5,√11),
  not the degree-16 ℚ(√3,√5,√7,√11). (The data adjudicated the field; it was not
  assumed — exactly the audit discipline that guards against silent mis-measurement.)
- **Partial certification:** of the 506 vertices supplied (the GraphData export
  arrived 4 short of 510), **2449 pairs are at exact unit distance**; the published
  graph has 510 vertices / 2508 edges. Two vertices are isolated in the 506-subgraph,
  consistent with their neighbors lying among the 4 missing vertices.

## To close
Supply the 4 missing vertices → recompute; expected exact result **510 vertices,
2508 unit-distance edges**, every edge certified `‖·‖²=1` in ℚ(√3,√5,√11), plus the
converse audit (no undeclared pair at exact distance 1). χ≥5 remains Phase C
(UNSAT certificate), unfaked.
