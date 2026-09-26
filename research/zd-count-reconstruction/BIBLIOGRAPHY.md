# Prior-art comparison and novelty decision

Inspected during 2026-09-09 local session (UTC 2026-09-10).
This is a targeted theorem comparison, not a complete citation census.

| Source and inspected location | Established result relevant here | Consequence for our claim |
|---|---|---|
| Zhilina, Part II, Definitions 2.2–2.4, Notations 3.11/3.13, Corollary 3.21 | Same doubling convention; normalized basis-pair lines; component C(x) indexed by the product of the two coordinates. | G_{n,W}=C(e_W) under the explicit vertex identification in NOTE.md. XOR restriction is not an object-level novelty. |
| Zhilina, Part II, Theorem 4.6, Lemmas 5.1/5.9/5.11, Remark 5.10 | Recursive component construction and recovery using recognizable vertices, neighbourhoods and column structure. | Construction and reconstruction from full adjacency are prior art; compare the smaller input (E,τ), not merely “recursive recognition.” |
| Zhilina, Part II, Theorem 5.15, Corollary 5.16 and Theorem 5.17 | Whole-graph reconstruction of algebra parameters and equivalence of graph/algebra isomorphism. | Does not itself state the fixed-component two-count or admissible-weight theorem. The component-level lemmas above are the closer baseline. |
| de Marrais, Placeholder Substructures I, Theorem 7; II, §2 and Theorem 10 | Dyad indices related by XOR with G+S; signed Emanation Tables; the power-of-two number-hub case. | With ambient generator H=2^n and S=W, the full dyad is e_a±e_{H+(a⊕W)}. The reset/hub family is known. ET assessors, signed dyad lines, and our paired signed matrix have different vertex counts and must not be conflated. |
| de Marrais, Placeholder Substructures III, §§4–5, Theorems 15–16 | Changes of dyad zero-products under insertion of high bits; two successive changes restore specified entries. | Bit-driven construction and sign-sensitive recurrences cannot be advertised as new. These statements do not supply an inverse from two global counts. |

Primary sources:

- Svetlana Zhilina, *Orthogonality graphs of real Cayley–Dickson algebras.
  Part II: The subgraph on pairs of basis elements*, International Journal
  of Algebra and Computation **31**(4) (2021), 691–725.
  [Publisher, DOI 10.1142/S0218196721500338](https://doi.org/10.1142/S0218196721500338);
  [full author text](https://arxiv.org/html/2608.28163v1).
  The publisher records publication on 12 May 2021; the 2026 arXiv deposit
  must not be mistaken for the date of first publication.
- Robert P. C. de Marrais, *Placeholder Substructures I: The Road from NKS
  to Scale-Free Networks is Paved with Zero Divisors* (2007).
  [Author text](https://arxiv.org/html/math/0703745v1).
- Robert P. C. de Marrais, *Placeholder Substructures II: Meta-Fractals,
  Made of Box-Kites, Fill Infinite-Dimensional Skies* (2007).
  [Author text](https://arxiv.org/html/0704.0026v2).
- Robert P. C. de Marrais, *Placeholder Substructures III: A Bit-String-Driven
  “Recipe Theory” for Infinite-Dimensional Zero-Divisor Spaces* (2007).
  [Author text](https://arxiv.org/html/0704.0112v3).

Additional primary sources screened: Zhilina Part I
[author text](https://arxiv.org/html/2608.28176v1) for its overlap with Part II;
de Marrais *Flying Higher Than a Box-Kite* (2002)
[abstract](https://arxiv.org/abs/math/0207003). The latter HTML full-text
request failed; only its abstract and its placement in Zhilina's references
were examined. It is not recorded as a full-text negative match.

## Searches and limits

Searches included the combinations:
- “Cayley-Dickson” + “edges” + “triangles”;
- “Cayley-Dickson” + “complete invariant”;
- “Cayley-Dickson” + “isomorphism” + “components” + Zhilina;
- “Placeholder Substructures” + “isomorphism”;
- “Cayley-Dickson” + “one bit” + graph;
- “Cayley-Dickson” + “scalar” + “triangles”;
- the exact Part II title with “cited”.

The exact-title search and publisher references were inspected. This did
not establish exhaustive forward-citation coverage. Searches returning no
matching theorem do not prove absence of one. Non-primary mirrors and
unrelated results were not used as mathematical evidence.

## Decision

**Rejected as novelty:** the underlying component family, XOR grouping,
recursive zero-product construction, the reset/hub case, and reconstruction
from an entire unlabelled component.

**Derived refinement:** the inverse from (E,τ) is a short arithmetic induction
on the signed recurrence. It is not sufficiently separated from known
recursive structure to justify advertising a new classification theory.
The exact count-completeness statement was not located in the inspected
primary statements.

**Retained candidate:** uniform injectivity within each origin class for
every admissible (a,b), hence a sharp upper bound of two on scalar fibres.
Its essential obligations are preservation of the admissible weights and
reset-versus-T separation within the core origin, not extending a finite
scan. NOTE.md supplies the arithmetic, the existing Lean files supply the
unbounded formal result, and the direct product controls verify the examples.

**Publication assessment:** potentially a short reconstruction/compression
note, with explicit credit for the known component structure. Priority is
not established. This assessment is an inference from the stated comparison;
neither search coverage nor LLM review settles originality or acceptance.
