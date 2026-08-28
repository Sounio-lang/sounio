# Hadwiger–Nelson / χ(ℝ²) ≥ 5 — deep literature review (for the Sounio attack)

*Compiled 2026-05-29 to direct the `souc_sat` / de Grey-core line. EN-UK orthography
(repo doc convention). Every claim is sourced; arXiv / Geombinatorics references at
the end. This is research notes, not a submission artefact.*

---

## 0. TL;DR — the strategic reframe (read this first)

The literature makes one thing unambiguous and it **inverts the naive plan**:

> **Refuting the 4-colouring of a *given* 5-chromatic unit-distance graph is the
> *easy* part.** Modern CDCL solvers dispatch the 1581-vertex de Grey CNF in
> seconds, and the 509-vertex graphs in well under a second. The *hard* parts —
> the parts that earned papers and burned ~100 000 CPU-hours — were (1) **finding**
> a 5-chromatic unit-distance graph at all (de Grey 2018) and (2) **minimising** it
> (Heule 2018–19 via proof-trimming; Parts 2019–20 via vertex selection).

Therefore Sounio should **not** try to re-discover or re-minimise a core (that is a
multi-year research programme already done by others). The **novel, achievable
flagship** is:

> **The first end-to-end *machine-checked, exact-arithmetic, self-hosted* certificate
> that χ(ℝ²) ≥ 5** — geometry certified over an exact number field, graph → CNF with a
> proven soundness lemma, refutation by the Sounio self-hosted solver, and the DRAT/LRAT
> proof closed by a *formally verified* checker, the whole chain composed in Lean.

Nobody has published that full chain: de Grey/Heule used floating-point embeddings and
*unverified* solvers (the DRAT refutation was checked by `drat-trim`, but the
geometry→graph→CNF half was not machine-checked); Parts produced a *human-verifiable*
proof but not a proof-assistant one. **That gap is the edge of novelty** (CLAUDE.md §6.10).

---

## 1. The problem and the standing bounds

The **chromatic number of the plane** CNP = χ(ℝ²) is the least number of colours to
colour ℝ² so that no two points at Euclidean distance exactly 1 share a colour.
Classical bounds 4 ≤ CNP ≤ 7: lower bound 4 from the **Moser spindle** (7 vertices) and
the **Golomb graph** (10 vertices); upper bound 7 from a hexagonal tiling
(Isbell/Hadwiger). The value may even depend on the set-theoretic axioms
(Shelah–Soifer).

**2018 breakthrough (de Grey):** CNP ≥ 5, via an explicit **1581-vertex** non-4-colourable
unit-distance graph — computer-assisted, independently SAT-verified (Aaronson, others).
Independently, **Exoo–Ismailescu** gave another χ ≥ 5 construction.

Still open / out of scope for us:
- **CNP ≥ 6 in the plane is OPEN.** 6-chromatic constructions exist only for *two-distance*
  graphs (Parts 2020) and *odd-distance* graphs — not the unit-distance plane. Do **not**
  chase χ ≥ 6 in ℝ²; it is an unsolved problem.
- Fractional chromatic number record ≈ 383/102; "measurable" CNP ≥ 6 partial results.

---

## 2. Timeline of small 5-chromatic unit-distance graphs (the minimisation race)

| vertices | edges | who / how | date | ref |
|---:|---:|---|---|---|
| 1581 | — | **de Grey** — first χ≥5 construction | Apr 2018 | [deG18] |
| ~826 | 4273 | Polymath16 vertex-removal | 2018 | [PM16] |
| 826→633→553 | — | **Heule** — SAT + clausal-proof trimming | 2018 | [Heu18] |
| 553 | 2720/2840 | Heule / Parts | Jul 2019 | [Heu18][Par20a] |
| **529** | 2670 (Heule) / 2630 (Parts) | **Heule** "Trimming Graphs via Clausal Proof Optimisation" — more *symmetric*; ~100k CPU-h | 2019 | [Heu19] |
| 525 | 2605 | Parts | Jul 2019 | [Par20a] |
| 510 | 2508 → 2502 | Parts | Aug 2019 – Mar 2020 | [Par20a] |
| **509** | **2442** | **Parts — current world record** (built from a 374-vtx + 136-vtx subgraph) | 2020 | [Par20a] |

Parts also produced a **human-verifiable** proof for χ≥5 [Par20b] (Geombinatorics 30/2),
where the case enumeration is small enough to check by hand — relevant because it implies
the 509 graph's non-4-colourability is *structurally* simple, not just SAT-opaque.

Public data: Heule's 553/529 edge lists are on his CMU page and the Polymath16 wiki;
the Parts graphs are in **House of Graphs** ("Parts Graph …") and Wolfram
`GraphData["PartsGraph509"]`. **The repo does not yet carry any of these edge lists**
(`examples/erdos/degrey_*.sio` reconstruct the *spindle / small fragments*, not a full
core).

---

## 3. The SAT encoding of k-colouring (what everyone uses)

Given graph G=(V,E) and k colours, variables `x[v,c]` = "vertex v has colour c"
(**direct / one-hot** encoding, 1-based literal `v*k + c + 1` — exactly what
`souc_sat.sio::build_spindle_3col` already emits):

- **At-least-one** per vertex: `(x[v,0] ∨ … ∨ x[v,k-1])`.
- **Edge / proper-colouring**: for each `(u,v)∈E` and colour c: `(¬x[u,c] ∨ ¬x[v,c])`.
- **At-most-one** per vertex is *optional* for UNSAT (a multi-coloured vertex only makes
  satisfaction *easier*, so omitting AMO is sound for refutation — and cheaper). Heule
  omits it for the refutation direction. We already omit it; that is correct.

`G` is k-colourable ⟺ the CNF is SAT. To certify χ(G) ≥ 5 we prove the **4-colouring CNF
is UNSAT**. The encoding-soundness lemma (CNF UNSAT ⟹ G not 4-colourable ⟹, with the
exact embedding, χ(ℝ²) ≥ 5) is the half that must be done **in Lean** for the flagship.

---

## 4. Symmetry breaking — the one encoder change that actually matters here

Colours are fully interchangeable, so the colour-permutation group **S_k** (k! = 24 for
4 colours) bloats the search. Two families of predicates:

- **Lex-leader** (Crawford–Ginsberg–Luks–Roy 1996): for each symmetry σ post
  `⟨x₁,…,xₙ⟩ ≤_lex σ⟨x₁,…,xₙ⟩`, keeping only the lexicographically least member of each
  orbit. Computing a *complete* lex-leader set is **NP-hard** (CP 2024 shows the barrier
  is graph-non-isomorphism certification); incomplete sets are used in practice.
- **Value precedence** (Law–Lee 2004; Walsh 2006–07 proved `PRECEDENCE ≡ VALSYM`): for
  *interchangeable values* — exactly the colour case — enforce *"the first time colour j
  is used precedes the first time colour k is used, for all j<k."* This breaks the **entire**
  S_k symmetry, is polynomial-size, and propagates well. **This is the right tool for graph
  colouring.**

### Decisive, evidence-backed insight for de Grey graphs (corrected 2026-05-29)

> **Every unit-distance graph in the plane is K₄-free, so its clique number ω = 3**
> (four mutually unit-distant points cannot exist in ℝ²; three give an equilateral
> triangle).
>
> Precolouring a clique of size m fixes m colours; the **residual colour symmetry is
> exactly S_{k−m}**. For the de Grey **4-colouring** (k=4) precolouring **one triangle**
> (m=ω=3) leaves **S_{4−3}=S₁ = trivial** — i.e. **clique-precolour of a single triangle
> already breaks the whole colour-permutation group S₄ completely.** A single leftover
> colour has nothing to permute with.

**Correction of an earlier draft claim:** an earlier version of this note asserted that
clique-precolour "fixes only 3 of 4 colours, leaving residual symmetry, so value precedence
is required." That is **wrong** — the residual is S₁, not a non-trivial group.
**Empirically confirmed**: on the spindle (k=3, ω=3) and K₈/₇, clique-precolour and value
precedence give *identical* conflict counts (13→2 and 46 165→1), both `drat-trim s VERIFIED`,
because both are complete when k−ω ≤ 1.

**Where each tool actually helps:**
- **Clique-precolour** is complete and cheapest **when k − ω ≤ 1** — which *is* the de Grey
  4-colouring case. Use it there.
- **Value precedence** (now implemented, `add_value_precedence`) is the general tool that
  bites when **k − ω ≥ 2**: triangle-free graphs (ω=2, k≥4), 5-colourings, or the
  solver-benchmark family. It breaks the same S_k without needing to *find* a clique.
- The genuinely harder, higher-value symmetry for de Grey is the **graph-automorphism
  group** (the "large rotation/reflection group" the Polymath threads mention), which needs
  `Aut(G)` from nauty/bliss/saucy plus lex-leader (NP-hard to make complete). **But** since
  refuting a *given* core is already trivial for any CDCL solver, this only matters for a
  *kissat-beating benchmark story*, **not for the χ≥5 critical path.** Keep it staged.

**Recommended encoder for the core (`souc_sat` F2):** for the 4-colouring, **triangle
precolour is sufficient for colour symmetry** (already landed). Value precedence is
available (`SB=2`) and composes (`SB=3`) but is redundant here; reserve it for k−ω≥2 work.

These are CNF clauses added to the formula; the DRAT refutation is *of the augmented
formula*, and satisfiability-preservation is the meta-lemma (the **no-SB control run**
empirically backs it up: spindle is UNSAT with *and* without any SB).

---

## 5. Proof systems, verified checkers, and parallelism (the V-track + cluster)

### Proof formats
- **DRUP/DRAT** (Wetzler–Heule–Hunt, FMCAD 2013 / SAT 2014): deletion + reverse-unit-prop,
  with RAT for non-RUP lemmas. The de-facto SAT-competition format. `souc_sat` emits this
  (streamed). `drat-trim` is the reference checker/trimmer (it also **extracts trimmed
  subgraphs** — the exact mechanism Heule used to minimise!).
- **LRAT / LPR** (Cruz-Filipe et al.; Tan–Heule–Myreen): DRAT **plus hints**, so a checker
  needs no propagation search ⇒ amenable to *formal verification*. `drat-trim cnf drat -L
  out.lrat` produces LRAT.

### Formally verified checkers (target for the flagship)
- **Lean 4 core `Std.Tactic.BVDecide`** (ex-`leanprover/leansat`, merged nightly
  2024-08-29): ships a **verified LRAT checker** with a Lean soundness proof; `bv_decide`
  runs CaDiCaL→LRAT→verified-check, `bv_check file.lrat` checks a stored LRAT. **This is the
  natural target for `formal/lean4/`** — feed our LRAT to the *already-verified* Lean checker
  and compose with the geometry+encoding lemmas.
- **cake_lpr** (Tan–Heule–Myreen, CAV 2021): verified **machine-code** LRAT/LPR checker via
  CakeML — the compiler is out of the trusted base. Strongest TCB story; good as the
  independent second checker.
- **Coq** checker (used to validate the 200 TB Pythagorean proof).

### Parallel solving for scale (the 384-core cluster)
- **Cube-and-Conquer** (Heule–Kullmann–Wieringa–Biere 2011/12): a look-ahead solver splits
  the formula into millions of **cubes**; CDCL conquers each; near-linear speed-up on
  thousands of cores; clausal proofs **compose**. Used for the **Boolean Pythagorean
  Triples** (200 TB DRAT, Coq-verified), **Schur Number Five**, and the **Empty Hexagon**.
- **Proofix** (SAT 2025, "Problem Partitioning via Proof Prefixes"): *automatic* partitioning
  that often beats hand-picked cubes — relevant if/when we scale beyond a single core.
- **Clause-sharing portfolios** (e.g. Mallob): our `portfolio.sh` / SLURM array is the
  lightweight version (no clause sharing yet — P3 in the roadmap).

For a *given* 509-vertex 4-colouring, **none of this is needed** — it is a small CNF
(~2036 vars, ~10k clauses). C&C only earns its keep if we attempt something genuinely
larger (higher-dimensional analogues, or a 6-chromatic *search*, which is open and out of
scope). Keep C&C staged; do not over-engineer for an instance that solves in milliseconds.

---

## 6. Exact geometry / the number field (where Sounio is actually differentiated)

The published constructions embed vertices with **floating-point** coordinates and check
unit distances numerically (de Grey, Heule) or by hand-tracked algebra (Parts). The de
Grey / Moser-spindle family lives in **ℚ(√3, √11)** (the spindle has dist² values that
close over this field); Parts' coordinates use minimal polynomials over a comparable real
field. The repo's **degree-16 kernel ℚ(√3, √5, √7, √11)** (`degrey_fieldtower.sio`) is a
*superfield* of what these graphs need, so it can represent every vertex **exactly** and
verify each edge `dist² = 1` with **no floating point** — `degrey_q3q11_spindle.sio` and
`degrey_fragment_q3q11.sio` already do this for the spindle and a 2-spindle fragment
(`dist² = 576` exact at scale ×12).

> **This is the Sounio strengthening of the result:** an exact-arithmetic certification of
> the embedding closes the one gap the SAT half never addressed — that the abstract graph
> *is actually realisable at unit distance in ℝ²*. Combined with a verified LRAT checker,
> the entire geometry→graph→CNF→UNSAT→χ≥5 chain becomes machine-checked.

---

## 7. Gap analysis → concrete roadmap mapped to repo files

| Step | What | Repo target | Status |
|---|---|---|---|
| G | Acquire a core's **edge list** (Heule **529** / Parts **510**) — vendored from `marijnheule/CNP-SAT`. | `examples/erdos/data/degrey_529.edge`, `parts_510.edge`, `degrey_529.vtx` | **DONE** |
| E | Encode 4-colouring (DIMACS edge-file reader) + triangle precolour / value precedence. | `souc_sat.sio` graph-file mode | **DONE** |
| S | Refute G₅₂₉ with `souc_sat` (streamed DRAT) → **`drat-trim s VERIFIED`** ⟹ **χ(G₅₂₉)≥5**. | `souc_sat.sio` | **DONE (Part A)** — 33 s, 327 k conflicts, 72 MB proof |
| B | Reconstruct `degrey_529.vtx` exactly over ℚ(√3,√5,√11); certify every edge `dist²=1` (no floats). | `degrey_geometry.sio` (denominator-extended `Q16` kernel + Mathematica parser) | **DONE (Part B)** — 2670/2670 edges `dist²=1` exact, no `SQRT_ERR`/`DIV_ERR` |
| V | DRAT → **LRAT** (`drat-trim -L`, 36 MB) → **`cake_lpr` (formally-verified CakeML checker) `s VERIFIED UNSAT`**. | `examples/erdos/verify_lrat_cake.sh`, `CAKE_LPR_RESULT.md` | **DONE** — SAT leg machine-checked by a verified checker |
| T₁ | **Geometry leg in Lean 4**: `g529_all_edges_unit_distance` — all 2670 edges `dist²=1` exact over ℚ(√3,√5,√7,√11), `native_decide`, no `sorry`. | `formal/lean4/SounioDeGreyUnitDistance.lean` (gen by `gen_lean_geometry.sh`) | **DONE** |
| T₂ | Compose: CNF UNSAT ⟹ not 4-colourable; ∧ unit-distance ⟹ χ(ℝ²)≥5. | `formal/lean4/` | **TODO** (depends on verified LRAT check) |

**Done (2026-05-29):** value precedence `add_value_precedence` landed (`SB=2`/`SB=3`),
spindle + K₈/₇ `drat-trim s VERIFIED`. Finding: it is **redundant with triangle precolour
for the de Grey k=4 case** (k−ω≤1 ⇒ both complete), so it is *not* the lever for χ≥5; it is
kept as the general tool for k−ω≥2.

**Revised highest-leverage next step:** since symmetry breaking is *not* on the χ≥5 critical
path (refutation of a given core is trivial), the real work is **data + exact geometry**:
1. Acquire an edge list — start with **Heule's 529** (most symmetric ⇒ simplest exact
   coordinates) or **Parts' 509** (fewest vertices). Sources: Heule CMU page / Polymath16
   wiki; House of Graphs "Parts Graph"; Wolfram `GraphData`.
2. Reconstruct its vertices **exactly** in the ℚ(√3,√5,√7,√11) kernel and certify every
   edge `dist²=1` with no floating point (extend `degrey_fieldtower.sio`).
3. Build the 4-colouring CNF (+ triangle precolour), refute with `souc_sat` (streamed DRAT),
   `drat-trim s VERIFIED` — **the first exact + self-hosted χ≥5 refutation.**
4. Then the Lean V-track: DRAT→LRAT→`Std.Tactic.BVDecide` verified checker + geometry/
   encoding lemmas ⟹ end-to-end machine-checked χ(ℝ²)≥5.

---

## 8. References

- **[deG18]** A. D. N. J. de Grey. *The chromatic number of the plane is at least 5.*
  Geombinatorics 28(1):18–31, 2018. arXiv:1804.02385.
- **[Heu18]** M. J. H. Heule. *Computing small unit-distance graphs with chromatic number 5.*
  Geombinatorics XXVIII:32–50, 2018. arXiv:1805.12181. (1581 → 553 via SAT proof-trimming.)
- **[Heu19]** M. J. H. Heule. *Trimming Graphs Using Clausal Proof Optimization.* CP 2019,
  LNCS 11802:251–267. arXiv:1907.00929. (553 → **529**, more symmetric, ~100k CPU-h.)
- **[Par20a]** J. Parts. *Graph minimization, focusing on the example of 5-chromatic
  unit-distance graphs in the plane.* Geombinatorics 29(4):137–166, 2020. (→ **509**, record.)
- **[Par20b]** J. Parts. *The chromatic number of the plane is at least 5 — a human-verifiable
  proof.* Geombinatorics 30(2):77–102, 2020.
- **[EI18]** G. Exoo, D. Ismailescu. *The chromatic number of the plane is at least 5: a new
  proof.* (Independent χ≥5.)
- **[CGLR96]** Crawford, Ginsberg, Luks, Roy. *Symmetry-breaking predicates for search
  problems.* KR 1996. (Lex-leader.)
- **[LL04]** Y. Law, J. Lee. *Global constraints for integer and set value precedence.* CP 2004.
- **[Wal07]** T. Walsh. *Symmetry breaking using value precedence.* (PRECEDENCE ≡ VALSYM.)
- **[CP24]** *The Complexity of Symmetry Breaking Beyond Lex-Leader.* CP 2024. (NP-hardness /
  graph-non-iso barrier for complete SBPs.)
- **[WHH13]** Wetzler, Heule, Hunt. *Trimming while checking clausal proofs / DRAT-trim.*
  FMCAD 2013; SAT 2014.
- **[HKM16]** Heule, Kullmann, Marek. *Solving and verifying the Boolean Pythagorean Triples
  problem via Cube-and-Conquer.* SAT 2016, LNCS 9710:228–245. (200 TB DRAT, Coq-verified.)
- **[HKWB12]** Heule, Kullmann, Wieringa, Biere. *Cube and Conquer: guiding CDCL SAT solvers
  by lookaheads.* HVC 2011, LNCS 7261:50–65.
- **[THM21]** Y. K. Tan, M. Heule, M. Myreen. *cake_lpr: Verified propagation redundancy
  checking in CakeML.* CAV 2021; extended in IJSTTT 2022.
- **[LeanSAT]** leanprover/leansat → merged into Lean 4 core as `Std.Tactic.BVDecide`
  (verified LRAT checker), nightly 2024-08-29.
- **[Proofix25]** *Problem Partitioning via Proof Prefixes.* SAT 2025, LIPIcs vol. 341.
- **[Schur5]** M. Heule. *Schur Number Five.* AAAI 2018. **[EmptyHex]** Heule, Scheucher.
  *The Empty Hexagon Number.* 2024.
