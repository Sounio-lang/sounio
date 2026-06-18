<!-- docs:meta
topic_id: repo.docs.research.erdos-168-chromatic-separation
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.erdos-168-chromatic-separation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# A machine-checked study of zero-divisor / associator chromatic obstructions for unit-distance graphs (Erdős #508/#704)

**Status:** internal research note, 2026-05-25. Consolidates a six-step
investigation, fully reproduced in Sounio and machine-checked in Lean 4
(16 theorems, no `sorry`). Branch `proof/sedenion-unordered-injectivity-168`.

> **Honest scope up front.** This is **not** a resolution of Erdős #508
> (chromatic number of the plane) or #704 (growth of χ(ℝ^d)), and it does **not**
> improve any known bound. It is a rigorous mapping of *what the verified
> 168-theorem / sedenion zero-divisor / associator structure can and cannot do
> to the chromatic number* of unit-distance-type graphs, together with a precise
> structural reason the natural "algebraic → geometric" bridge fails. The single
> positive result is a Level-2 *algebraic separation* (non-associativity is
> necessary and sufficient to exceed χ = 2 in this family).

---

## 1. Background

The companion formal development establishes (all by `native_decide`):

- `SounioCayleyDickson.lean` — the Cayley–Dickson sign function `cdSigma a b bits`
  with `e_a · e_b = cdSigma(a,b,bits) · e_{a⊕b}`; the 168 non-Fano triples of 𝕆.
- `SounioZeroDivisorBridge.lean` — exactly **84** valid primitive imaginary
  sedenions `v = e_lo ± e_hi` (`lo∈1..7, hi∈9..15, lo⊕hi≠8`, `validPrims`), and
  exactly **168** unordered projective zero-divisor classes (`unorderedZDPairs`,
  pairs `(u,v)` with `u·v=0`), in 7 xor-fibers of 24.

All arithmetic below reuses `cdSigma`/`cd_sigma_ct` verbatim, so the graphs are
computed from the *same* multiplication table the bridge theorems are proved
against. Every chromatic claim is over **integer** coordinates, hence exact.

## 2. Setup

Points are integer vectors in ℝ¹⁶ (sedenion coordinates). For a fixed primitive
`v` (or ZD pair `(u,v)`), we study conflict graphs on a point set, edge defined by
one of:

- **Linear ZD surgery** (right/left mult. by a primitive): edge ⟺ `‖(p_i−p_j)·v‖² = ‖v‖² = 2`.
- **Associator-distance surgery**: edge ⟺ `‖((p_i−p_j)·u)·v‖² = 4` (since `u·v=0`,
  `assoc(p,u,v)=(p·u)·v`).
- **Associator conflict** (non-distance): edge ⟺ `‖assoc(p_i,p_j,c)‖² = T`. This is
  *bilinear in (p_i,p_j)* — not a function of `p_i−p_j` — so it is **not**
  translation-invariant.
- **Euclidean unit-distance** (realizable): edge ⟺ `‖p_i−p_j‖² = T`; scaling by
  `1/√T` makes this an induced unit-distance graph in ℝ¹⁶.

Chromatic number is computed exactly by brute force on small probes; on larger
probes we use BFS for χ≥3 (exact) and an explicit coloring for χ≤2 (exact);
exact χ≥4 on ~100+ vertices is infeasible (non-k-colorability is exponential).

Reproduce: `souc compile examples/erdos/moser_zd_probe.sio -o /tmp/m && /tmp/m`;
`cd formal/lean4 && lake build SounioErdosUnitDistance`.

## 3. Results (six steps)

| # | Construction | Point set | Outcome | Lean witness |
|---|---|---|---|---|
| 1 | linear right-mult, edge=‖·v‖²==2 | 7-vertex probe | χ=2; 4/84 surgeries change edges, 0 raise χ | `classical_chromatic_eq_two`, `zd_surgeries_changing_edges_eq_four`, `no_zd_surgery_raises_chromatic` |
| 2 | associator-distance `(p·u)·v`, ==4 | 7-vertex probe | χ=2; **168/168** classes change edges, 0 raise χ | `associator_class_count_168`, `all_associator_surgeries_change_edges`, `no_associator_surgery_raises_chromatic` |
| 3 | scale linear+assoc-dist | 137 binary + 273 signed (504 graphs, BFS) | 0 non-bipartite; linear χ=2 is a **theorem** | `bigProbe_card`, `linear_surgery_total_parity_2colors` |
| 4a | left-mult | 137 + 273 | identical to right; invariant is **two-sided** | `left_surgery_total_parity_2colors` |
| 4b | **associator conflict** ‖assoc‖²==4 | 137 binary, all 84 c | **non-bipartite ∀c (χ≥3)**; total-parity broken 0/84 | `associator_conflict_triangle`, `cWit_is_first` |
| 5 | conflict, sparse regime | T=8 conflict graph | clique-driven χ∈[8,11] (ω≥8); **rigorous sparse χ>ω** | `associator_conflict_induced_C5` |
| 6 | **realizable** Euclidean ‖diff‖²==T | 84 primitives; 137 binary | **NULL** — bipartite (T=6) / clique-driven (T=2,4); no sparse χ>ω | (none; null result) |

## 4. The two structural facts

**(A) Distance-based ZD surgery cannot exceed χ = 2 — and this is forced by a
two-sided parity invariant.** For every primitive `v` and either multiplication
side, total Hamming-weight parity is a proper 2-coloring of the twisted graph on
the complete weight-≤2 binary probe (137 points). Proved:
`linear_surgery_total_parity_2colors`, `left_surgery_total_parity_2colors`.
Equivalently: every linear twisted unit edge joins points of opposite parity.
(Intuition: `‖M_v(d)‖²=2` forces `M_v(d)` to have two ±1 entries; parity is a
homomorphism preserved by the linear map.) Over 504 graphs (two complete
families) BFS finds **zero** non-bipartite cases.

**(B) The associator conflict relation breaks the parity ceiling — χ ≥ 3, and
the gap reaches the sparse (clique-poor) regime.** Because `assoc(x,y,c)` is
bilinear and not a function of `x−y`, the conflict graph is not
translation-invariant and the parity coloring fails (0/84). For every primitive
`c` the conflict graph is non-bipartite. The high χ is partly *clique-driven*
(an explicit K₈, so χ ≥ 8 but ω ≈ χ — the trivial near-simplex regime), but it
is **not only** clique-driven: extracting (exactly) a triangle-free induced
subgraph that is non-bipartite gives χ > ω with the smallest possible clique.
Concretely, the T=8 conflict graph contains the **induced 5-cycle**
`{e₃, e₂+e₆, e₂, e₄, e₈}` — ω=2, χ=3. Proved: `associator_conflict_triangle`
(K₃ ⇒ χ≥3) and `associator_conflict_induced_C5` (induced C₅ ⇒ χ>ω).

## 5. The precise obstruction to the geometric bridge

A new Erdős bound would require a graph that is simultaneously (i) high χ in the
sparse regime (χ ≫ ω) and (ii) realizable as a Euclidean unit-distance graph.
Step 6 shows these are, for this structure, in tension:

- The sparse χ>ω gap (Step 5) lives in the **associator conflict** relation,
  which is *bilinear, not a distance* — hence **not realizable** as a
  unit-distance graph.
- The **realizable** Euclidean distance graphs on the same (or generic) integer
  point sets are bipartite (T=6) or clique-driven (T=2,4): their triangle-free
  regions are bipartite, so they exhibit **no** sparse χ>ω.

So the chromatic obstruction the 168/ZD/associator structure provides is exactly
the part that is **not** geometrically realizable, and the realizable part
carries no sparse obstruction. This is the concrete reason the bridge fails —
not a failure of search but a structural incompatibility on these point sets.

## 6. What is and isn't claimed

- **Claimed (machine-checked):** a Level-2 *algebraic separation* — distance-based
  ZD surgery is χ≤2 (two-sided parity theorem), while the non-associative
  associator conflict relation reaches χ≥3 and a rigorous sparse χ>ω. Non-
  associativity is necessary and sufficient to exceed χ=2 in this family.
- **Not claimed:** any bound on χ(ℝ²) (#508) or χ(ℝ^d) (#704). The induced C₅ is a
  trivial unit pentagon; the realizable graphs give at most the trivial simplex
  bound; nothing beats known results.

## 7. Reproducibility

- Sounio: `stdlib/hypercomplex_graph/erdos_unit_distance.sio` (self-contained;
  84 primitives, 168 ZD pairs, all four edge relations, exact chromatic number /
  BFS / triangle-free extraction / shortest-odd-cycle). Driver:
  `examples/erdos/moser_zd_probe.sio` (7 stages + summary).
- Lean: `formal/lean4/SounioErdosUnitDistance.lean` — **16 theorems**, all
  `native_decide`, **no `sorry`**, axiom profile `propext`/`Quot.sound`/native
  reduction only. Builds against the proof branch's `SounioZeroDivisorBridge`.
- External review: each Lean iteration passed an `xai`/Grok 4.1 math-review
  (see `.claude/llm_offload_log.md`, 2026-05-25): all theorems OK, no overclaim
  beyond the checked finite instances.

## 8. Open frontier

The genuine bridge is open: a *realizable* edge relation that inherits the
associator's clique-poor odd-cycle structure (sparse χ>ω that is also a Euclidean
unit-distance graph). It is not obvious such a relation exists on sedenion point
sets — plausibly as hard as the underlying Erdős problems. Other untouched
directions: larger χ−ω gaps (e.g. a triangle-free χ=4 / Grötzsch-type induced
subgraph), and pathion (32D) constructions for #704.

---

*Commits (proof branch): `8d9302c5` (1) · `de59a1d5` (2) · `deeb4389` (3) ·
`8b48c5e2` (4) · `f3a1407d` (5) · `d4bad078` (6).*
