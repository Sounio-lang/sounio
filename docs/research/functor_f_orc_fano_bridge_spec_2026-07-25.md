<!-- docs:meta
topic_id: repo.docs.research.functor-f-orc-fano-bridge-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-orc-fano-bridge-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — the algebra ↔ ord-M bridge: Ollivier-Ricci curvature on the Fano graph is symmetry-forced

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `M_CHARACTERISED` (characterised negative of *symmetry-coincidence* type, not an identity)
**Parents:** `functor_f_g2_tower_closure_spec_2026-07-25.md` (`Q_GREEN`), `functor_f_petitot_bridge_spec_2026-07-25.md` (`B_OBSTRUCTED`, the sibling ord-P edge)
**Harness:** `scripts/research/functor_f_orc_fano_bridge_contract.py`

---

## 0. What this is, and the verdict type fixed in advance

The second cross-column edge: does the algebraic column (associator / G₂ 3-form φ / 4-form
ψ) reach the *metric-morphodynamic* column, taken here as **Ollivier-Ricci curvature (ORC)**
on the canonical graph the algebra builds? The programme forbids the identity reading
(D3: "ORC pattern ≡ associator/ψ locus"). So the verdict type was **named before computing**:
an *operational* probe whose bar is a **canonical algebra→ORC map** — an algebraic invariant
that *determines a varying ORC*, a match of control **type**, never a coincidence of counts.

The honest outcome is a **characterised obstruction of symmetry-coincidence type**. On the
canonical Fano graph the ORC is a single scalar **forced by graph edge-transitivity** — a
symmetry strictly larger than, and blind to, the algebra's oriented data. Any match between
"uniform ORC" and "uniform associator magnitude 2" is therefore a symmetry/dimension
coincidence, exactly the failure mode the task warned against.

---

## 1. Method discipline (why this is falsifiable)

Per the ord-P lesson: **derive the graph from the algebra, count what the algebra canonically
supplies BEFORE computing curvature, and demand a control-type match, not a dimension count.**
Two guards were built in before any curvature was read:

1. **Solver audit (`M0B`).** The whole result rests on a hand-rolled optimal-transport
   (Wasserstein-1) solver. It is validated *in-contract* against closed forms — `LLY(Kₙ) =
   n/(n−1)` for `n = 4..8`, interior path edge `= 0`, and `d`-regular tree central edge
   `= −2(d−2)/d` for `d = 3,4,5` — before it is trusted, in parallel to the octonion axiom
   audit (`M0`).
2. **Control graphs (`M3`).** Graphs with **zero** Fano/octonion content are run through the
   same solver, so "the value is symmetry-forced, not algebra-forced" is *measured*, not argued.

---

## 2. Construction (canonical, not invented)

- **Nodes** = the 7 imaginary units `e₁..e₇`. **Lines** = the 7 Fano lines
  `FANO = [(i,j,i⊕j) for i in 1..7 for j>i if (i⊕j)>j]` (each a quaternion triple).
- **Graph A — edge rule:** two units are adjacent **iff they lie on a common Fano line**.
  Because the Fano plane is a `2-(7,3,1)` design, **every** pair lies on **exactly one** line,
  so the 7 line-triangles' 1-skeleton is the **complete graph `K₇`** (`21 = C(7,2)`). Stating
  the rule honestly *is* the first finding: the natural 7-node incidence graph collapses.
- **Graph B — Heawood** (the point–line incidence / Levi graph): 14 nodes (7 points + 7 lines),
  point ~ line iff incident; 3-regular, triangle-free, girth 6. This *retains* the incidence
  structure (no pair-covering collapse) and is the canonical "Fano graph" of graph theory.

**Curvature.** Standard Ollivier-Ricci at idleness `p`,
`κ_p(x,y) = 1 − W₁(m_x^p, m_y^p)/d(x,y)`, with `m_x^p` = mass `p` at `x` and `(1−p)/deg(x)`
on each neighbour; and **Lin–Lu–Yau** `κ_LLY = lim_{p→1} κ_p/(1−p)`. `W₁` is the exact
graph-metric Wasserstein-1, from the in-file integer min-cost-flow solver. Reported at `p=0`
(vanilla OR) and as `κ_LLY`; `κ_LLY` is exact (identical rational at idleness resolution
`M=200` and `M=400`, i.e. on the top linear piece).

---

## 3. Findings

| Clause | Result | Reading |
|---|---|---|
| `M0_CORE_AUDIT` | inherited octonion core passes its axioms | foundation verified before use |
| `M0B_OT_SOLVER_AUDIT` | OT solver reproduces `n/(n−1)`, path`=0`, `−2(d−2)/d` | the tool is audited before it is trusted |
| `M1_INCIDENCE_COLLAPSE` | edge rule → `K₇` (`21=C(7,2)`, 6-reg); Heawood built (14 nodes, 3-reg, triangle-free, girth 6). **Count before curvature:** of `C(7,3)=35` unit-triples, **7 on-line** (associator `0`), **28 off-line** (associator magnitude **exactly 2**); φ support (unordered) `=` the 7 lines; canonical algebra-derived edge weight on `A` (`|φ|` on each pair's unique line) `=` the **singleton `{1}`** | the algebra supplies *uniform* data across the design |
| `M2_ORC_UNIFORM` | `A(K₇)`: `LLY = 7/6`, `OR(p=0) = 5/6`. `B(Heawood)`: `LLY = OR(p=0) = −2/3`. **Single-valued across all 21 edges** each (edge-transitive); `LLY` exact (`M=200 ≡ M=400`) | ORC is one constant — the symmetry *is* the test |
| `M3_SYMMETRY_NOT_ALGEBRA` | Möbius–Kantor `GP(8,3)` (3-reg, girth 6, **no octonions**) → `−2/3` `=` Heawood; `K₈` → `8/7`. Closed forms `n/(n−1)` `[Kₙ]` and `−2(d−2)/d` `[d`-reg girth ≥6]`; the `2` is the **universal tree numerator** (`d=4→−1`, `d=5→−6/5`) | the value is symmetry-forced, reproduced with zero algebra |
| `M4_ORIENTATION_BLIND` | **16 of 128** line-orientation sign-vectors give genuine composition (octonion) algebras, carrying **16 distinct signed G₂ 3-forms φ** (distinct associators), yet **all** induce the identical graphs `A`, `B` → identical ORC | ORC `= f(`unoriented incidence design`)`; the signed φ/associator/ψ is invisible to it |

Verdict: `FUNCTOR_F_ORC_VERDICT M_CHARACTERISED`.

---

## 4. The obstruction, precisely

Two independent, converging arguments locate it.

**(a) 0-DOF / edge-transitivity.** A canonical algebra→ORC correspondence would require ORC
to be a *non-trivial function of a varying algebraic invariant*. But both canonical graphs are
edge-transitive, so ORC is provably a **single constant** — it carries zero degrees of freedom
able to resolve any line-level or unit-level algebraic difference. There is no variation to
correlate; the observed "uniform ORC ↔ uniform associator magnitude 2" is a **match of two
constants**, i.e. a coincidence, not a map. The controls (`M3`) show the constants come out
of the pure incidence counts: `7/6` from `|V|=7` (not the seven imaginary units — `K₈` gives
`8/7`), and `−2/3` from `d=3` via the universal tree numerator `2` (not the associator
magnitude 2 — Möbius–Kantor reproduces `−2/3` with no octonions anywhere).

**(b) Orientation-blindness / control-TYPE mismatch.** The genuinely algebraic datum is the
**oriented** structure constant `φ_{ijk} = ⟨e_i e_j, e_k⟩` — the seven line *orientations* /
associator signs are what distinguish the octonions from a bare `2-(7,3,1)` design. `M4`
measures that **16** sign-vectors yield real octonion algebras with **16 distinct signed φ**
(hence distinct associators and ψ), and **every one of them induces the byte-identical graph**.
The graph is a function of `|φ|`'s *support* alone — the unoriented incidence design — which
is invariant under the whole orientation group that *moves* the algebraic invariants. So ORC
is blind, by construction, to precisely the data the algebraic invariants encode. This is a
control-**type** mismatch, the bar the task set, and it is stronger than (a) alone.

Together: the algebra→ORC edge does not carry a canonical correspondence; the symmetric graph's
curvature is fixed by a symmetry that has already quotiented away the algebra's oriented content.

---

## 5. What this is NOT

- **Not** "ORC is uninteresting" — the numbers are exact and correct; the point is *what
  determines them*.
- **Not** an identity claim in either direction (D3 respected) — the verdict is operational.
- **Not** a proof that *no* construction can see the algebra. The obstruction is specific to
  the **unweighted** incidence graph and to any **canonical** weighting (which `M1` shows is
  the singleton `{1}` — uniform, so weighted ORC equals unweighted up to a global scale).
  Hand-picked, non-canonical edge weights would be **fabrication** and are out of scope. A
  genuinely different, *orientation-sensitive* directed/signed construction that could resolve
  the line orientations is not excluded — it is a **next edge**, §6, not this probe's scope.
- **Not** a `14/28` result. No `14/28` split of the objects under test was found; the measured
  split is **7 on-line / 28 off-line** of `C(7,3)=35`. It would not matter: ORC is a single
  constant regardless of any split, which is the whole point.

---

## 6. Place in the ladder

```
G_GREEN         uniformity across 7 lines
H_CHARACTERISED argmax-b obstruction, b_cov fix
E_GREEN         continuous-orbit equivariance
K_CHARACTERISED field functoriality up to ord-1 correction
P_GREEN         correction is the coherent G2 3-form
Q_GREEN         G2 form-tower closes; algebraic invariants terminate at ord-2
B_OBSTRUCTED    algebra -> Petitot (ord-P): cusp canonical, butterfly obstructed
M_CHARACTERISED algebra <-> ORC (ord-M): curvature symmetry-forced, algebra-blind
```

The algebraic column is saturated (`Q`); **both** cross-column probes now return clean,
located obstructions (`B` at ord-P, `M` at ord-M) rather than forced matches — the programme's
preferred kind of result. The two negatives share a mechanism: the algebra's *oriented /
signed* content (butterfly `x⁴` factor at ord-P; line orientations at ord-M) is exactly what
the target column cannot see.

## 7. Suggested next edges

1. **Orientation-sensitive ord-M:** build a *directed* or *signed* graph whose edge signs are
   the line orientations `φ_{ijk}`, and a curvature that respects them (signed/magnetic OR);
   test whether *that* varies with an algebraic invariant. This is the construction `M4`
   predicts is necessary — and the honest way to reopen the edge.
2. **Prove edge-transitivity forces the constant** from `Aut(K₇)=S₇ ⊋ PSL(2,7)` and
   `Aut(Heawood)=PGL(2,7) ⊋ PSL(2,7)` — upgrades `M2`/`M4` from measured to structural.
3. **External write-up:** `G→Q` (closed algebraic story) + `B` + `M` (two honest cross-column
   obstructions sharing one mechanism) is a complete, self-contained arc.

---

## 8. Reproduce

```bash
python3 scripts/research/functor_f_orc_fano_bridge_contract.py
# expect: M0,M0B,M1,M2,M3,M4 PASS, FUNCTOR_F_ORC_VERDICT M_CHARACTERISED
```

Pure Python (numpy); CD sign law self-contained; embeds the `M0` core axiom-audit and the
`M0B` optimal-transport solver audit before either is used.

---

## 9. AI disclosure

Probe, contract, and spec produced under human direction (2026-07-25). The verdict type
(operational, never identity) and the count-before-curvature discipline were fixed before
computing; an advisor review predicted the dimension-coincidence failure mode, required the
`M0B` solver self-audit and the zero-content control graphs (`M3`), corrected an initial weak
alternativity test to the composition-algebra discriminator (yielding the measured `16/128`
in `M4`), and flagged the `14/28` mismatch. All measured numbers are exact rationals from the
in-file solver. Claims bounded by the six named clauses. The §10 math-review offload raised
no flags (all clauses `[OK]`), so no wording changes were required. No clinical content.
GAIDeT-ICMJE 2025.

## 10. Math-review offload

`bin/llm-offload -t math-review -p xai -i docs/research/functor_f_orc_fano_bridge_spec_2026-07-25.md`
— Grok 4.3 reviewed every clause and returned **all `[OK]`, "No errors found; all derivations
tight."** It independently confirmed: the `S(2,3,7)` incidence ⇒ `K₇` collapse; the Heawood
construction; `LLY(K₇)=7/6`, `OR=5/6`, Heawood `=−2/3`, matching `n/(n−1)` and `−2(d−2)/d`;
the Möbius–Kantor `−2/3` and `K₈` `8/7` controls as "symmetry forcing"; the `35=7+28`
associator split; the `16/128` composition-algebra count; and "edge-transitivity ⇒ ORC
constant … no degrees of freedom remain for algebraic invariants". No flags to address, so no
edits to the six clauses; this note added. Raw output dir: `/tmp/llm-offload-v09t35/`.
