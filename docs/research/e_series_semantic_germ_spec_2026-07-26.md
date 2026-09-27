<!-- docs:meta
topic_id: repo.docs.research.e-series-semantic-germ-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.e-series-semantic-germ-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# E-series semantic germ — a Petitot potential with E₆/E₇/E₈ bifurcation structure, and the weak form of ADE–Wildgen made testable

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN`
**Parents:** `ade_wildgen_mckay_spec_2026-07-26.md` (verdict `STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE`; the promotion path this rung executes), `functor_f_petitot_bridge_spec_2026-07-25.md` (the cross-column discipline: operational bridge, never identity), `functor_f_e6_albert_shadow_spec_2026-07-25.md` (`PHI_IS_THE_E6_CUBIC_CROSSTERM`, re-audited as O1/O2), `petitot-semantic-potential.md` (the morphodynamic frame)
**Harness:** `scripts/research/e_series_semantic_germ_contract.py`
**Gate:** `scripts/ci/e_series_semantic_germ_gate.sh`

---

## 0. What this is, and the verdict type fixed in advance

The ADE–Wildgen rung separated two forms of the conjecture and found the weak form
**undecidable within the programme**, for one stated reason (its clause `C3`):

> the programme's operative Petitot germs are A-series (cusp `x⁴ = A₃`, butterfly
> `x⁶ = A₅`). **No `E`-series germ has ever been constructed on the semantic side.**

The promotion path it named: *construct an `E`-series germ or `F₄`-natural semantic
object with the same verdict-before-computing discipline.* This rung constructs the
`E`-series germ. The verdict type was fixed before computing:

- **not** a proof of the weak form (construction ≠ evidence that semantic morphology
  *requires* the `E`-series);
- **not** a refutation either;
- an **operational test surface**: the semantic `E`-series object now exists, is
  gated, and carries a stated falsifiable signature. The weak form is thereby
  **testable**; its truth value stays **open**.

Verdict: `E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN`.

---

## 1. The construction

A Petitot-style morphodynamic potential on **two state variables** `(x, y)` (the
one-state cusp/butterfly cannot reach the exceptional germs — every `Aₖ` is
one-variable, every `Eₖ` has corank 2), with controls = the miniversal unfolding
monomials:

| potential | germ at the organising centre | μ | unfolding monomials (controls) |
|---|---|---|---|
| `V_E6 = x³ + y⁴ + Σ cᵢmᵢ` | `E₆` | 6 | `1, x, y, y², xy, xy²` |
| `V_E7 = x³ + xy³ + Σ cᵢmᵢ` | `E₇` | 7 | `1, y, y², x, xy, x², x²y` |
| `V_E8 = x³ + y⁵ + Σ cᵢmᵢ` | `E₈` | 8 | `1, x, y, y², y³, xy, xy², xy³` |

Semantic reading (Petitot's dictionary, stated as reading not theorem): the
organising centre is the degenerate semantic morphology; a generic small control
value splits it into `μ` elementary positions (the Morsification census, `G3`);
the bifurcation set in control space is the opposition/catastrophe locus. Two
honesty notes, both enforced by the contract's output rather than smoothed over:

- The `E`-germs are odd-degree (corank-2) degenerate critical points, **not**
  minima — the "positions" are the critical points of the unfolding, counted over
  ℂ. The real sub-census (real critical points, Morse indices, minima = wells) is
  real-form-dependent and is reported as data, not as a law: for the reference
  deformation `V + t(xy + x + y)`, `t = 1/10`, the contract measures 6/7/8
  nondegenerate complex critical points (all `A₁`) and real counts `0/1/0`.
- This is exactly the richness Wildgen's four-actant semantics was argued to need:
  `μ = 6, 7, 8` elementary positions against `3` (cusp) and `5` (butterfly) for
  the programme's operative A-series germs.

---

## 2. What is computed

**G-side — the germs and their bifurcation structure (exact rational arithmetic).**

| Clause | Result | Reading |
|---|---|---|
| `G1_GERMS_MILNOR` | `μ(E₆)=6, μ(E₇)=7, μ(E₈)=8` by exact rational Buchberger, with standard-monomial bases equal to the unfolding monomial sets above; controls: cusp `μ=3`, butterfly `μ=5` | the germs are the claimed Arnold singularities; control counts match |
| `G2_E_TYPE_JET` | the cubic 3-jet of each `E`-germ is a **perfect cube** (binary Hessian of the cubic ≡ 0, exact); `D₄`/`D₅` controls split into 3 / 2 distinct linear factors (binary-cubic discriminant, exact) | the organising centres are `E`-type, not `D`-type, at the jet level |
| `G3_MORSIFICATION_CENSUS` | explicit small real deformation `V + t(xy+x+y)`, `t=1/10`, splits each centre into exactly `μ` distinct nondegenerate (`A₁`) critical points over ℂ (max residual `≤ 5e-14`, all Hessian dets bounded away from 0, pairwise separated); real sub-census reported as data | the semantic content: 6/7/8 elementary positions |
| `G4_E6_ADJACENCY_FULL` | **full** Arnold sub-singularity list of `E₆` witnessed: `A₁` (G3), `A₂, A₃, A₄, A₅, D₄, D₅` | the `E₆` bifurcation set contains its entire expected stratification |
| `G5_E7_ADJACENCY_SPINE` | `A₁..A₅, D₄, D₅, D₆, E₆` witnessed — full list **except `A₆`** | one short of full (scoped, §4) |
| `G6_E8_ADJACENCY_SPINE` | `A₁..A₅, D₄, D₅, D₆, D₇, E₆, E₇` witnessed — full list **except `A₆, A₇`** | two short of full (scoped, §4) |

Each adjacency witness is an explicit 1-parameter deformation curve
`T + W(t)`, `W(t) → 0`, whose origin-germ type is the target type — verified
**exactly** (gradient vanishes at the origin; Hessian corank; **local** Milnor
number by exact linear algebra on the truncated local ring `Q[x,y]/𝔪^(N+1)` —
unit inversion = truncated geometric series — not the global quotient, which
would count critical points the deformation moves elsewhere; cubic-jet factor
structure). Type constancy on the punctured parameter line is verified exactly
at `t = 1, 1/8, 1/64`; for single-monomial witnesses it is also a theorem
(quasi-homogeneous scaling `(x,y) ↦ (s^{wₓ}x, s^{w_y}y)` maps `T + tm` onto
`s·(T + t′m)` for any `t′ > 0`). A type-`S` fibre accumulating at the central
`E`-fibre is precisely the adjacency `S < T` in Arnold's sense. Sample
witnesses: `E₆ + t·y³ → D₄` for all `t ≠ 0`; `E₈ + t·xy³ → E₇`;
`E₈ + t·y⁴ → E₆`; `E₈ + (t²x²y + 2t·xy³) → D₇`; `E₆ + (t²x² + 2t·xy²) → A₅`.

**O-side — verification against the octonion/associator structure.**

| Clause | Result | Reading |
|---|---|---|
| `O1_PHI_IS_CUBIC_CROSSTERM` | `Re(x·y·z)` bracketing-independent on 𝕆 (dev `9e-15`, 200 imaginary triples); `= −φ(x,y,z)` on imaginary triples (dev `8e-15`) | inherited `E1+E3` re-audited on the self-contained CD core: `φ` **is** the imaginary restriction of the `E₆`/Albert cubic cross-term |
| `O2_ASSOCIATOR_SEPARATE` | `‖[e₁,e₂,e₄]‖ = 2` while `φ(e₁,e₂,e₄) = 0` | the vector associator (the `ψ`/non-associative side) and the scalar cubic cross-term are separate objects (inherited `E4` re-audited) |
| `O3_MAGIC_SQUARE_CHAIN` | Tits `T(A, J₃(𝕆))`: `0+1·26+52 = 78` (`E₆`), `3+3·26+52 = 133` (`E₇`), `14+7·26+52 = 248` (`E₈`) | the `E`-labels of the constructed germs are exactly the octonionic magic-square tower (inherited `C4` re-audited) |
| `O4_NO_FORM_IDENTITY` | `φ(x,x,x) = 0` and `Re(x³) = 0` for random imaginary `x` (max `9e-16`) — no nonzero **symmetric** cubic form on `Im 𝕆` arises from the Albert cross-term — while the semantic germ's cubic `u³` is a commutative polynomial cube, nonzero generically | **divergence, honestly measured:** the semantic `E`-germ is **not** octonion-derived. The bridge is the `E`-label (magic square on the algebra side; Arnold label on the singularity side), not a form identity. `D3` quarantine kept. |

The O-side verdict mirrors the parent rung's discipline: the genuine continuous
bridge octonions → `E`-series is **Freudenthal–Tits** (O3), and `φ` sits inside
the `E₆` cubic (O1) — but the morphodynamic germ constructed here is a
catastrophe-theoretic object sharing the **label**, not a pushforward of an
octonionic form (O4). The boundaries `C1` (`G₂` not simply-laced; no
`SU(2)`-McKay partner) and `C2` (no order-5/icosahedral content in the Fano
group) of the parent rung are untouched and not re-opened.

---

## 3. What the weak form's test now is

With the semantic `E`-series object gated, the weak form ("exceptional geometry,
somewhere in the `G₂…E₈` complex, governs rich semantic morphology") has an
operational test surface:

1. **Necessity direction (falsifier).** If every semantic morphology the
   programme instruments closes with A-series germs (cusp/butterfly sufficing, as
   for the semiotic square so far), the `E`-germs are unnecessary — the weak
   form is then unsupported by the programme's evidence (though not logically
   refuted).
2. **Sufficiency direction (support).** A semantic phenomenon whose morphology
   provably requires `μ ≥ 6` positions with an `E`-type (cubic-cube, corank-2)
   organising centre — e.g. a six-actant structure whose bifurcation set
   exhibits the `G4` stratification — would be positive evidence for the weak
   form, and the `O3` chain would then tie it to the octonionic tower.

Neither is decided here. The verdict stays `WEAK_FORM_OPEN`; what changed is
that the programme now **has the object the test runs on**.

---

## 4. What this is NOT (scope, and what was scoped out)

- **Not a proof or refutation of the weak form** — construction makes it
  testable, not true.
- **Not a form-level octonion derivation of the germs** (O4 is a measured
  divergence). **Not** an identity between the bifurcation set and the
  ZD/associator loci (`D3` kept).
- **Not the full E₇/E₈ adjacency closure.** Missing: `A₆ < E₇`; `A₆, A₇ < E₈`.
  These need critical points **away from the origin** (within the natural
  one-cubic-parameter families the residual-coefficient conditions force
  irrational parameter relations — e.g. `b⁵ = −8` — so no rational at-origin
  witness exists there). Off-origin witnesses need only the translation of the
  same exact machinery to a rational critical point; scoped out as the natural
  follow-up, not a gap in what is claimed.
- **Not the monodromy/vanishing-cycle verification** (the Dynkin diagram of the
  intersection form, which would tie the germ's geometry to the Weyl group of
  the `E`-algebra directly rather than via the label). Heavier machinery;
  scoped out.
- **Not an answer to the routon nullity-histogram multiplicities** (the open
  combinatorial question from the Level-7 rung). Separate problem, untouched.
- **Not clinical, not a semantic-identity (`D3`-class) claim.**

---

## 5. Reproduce

```bash
.venv/bin/python3 scripts/research/e_series_semantic_germ_contract.py
# expect: G1..G6, O1..O4 PASS (~4 s),
#   E_SERIES_GERM_VERDICT E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN
#   E_SERIES_SEMANTIC_GERM_OK

bash scripts/ci/e_series_semantic_germ_gate.sh
# expect: E_SERIES_SEMANTIC_GERM_GATE_OK
```

Pure Python + NumPy, self-contained, deterministic. Exact parts: rational
Buchberger (Milnor numbers, unfolding bases), exact linear algebra on the
truncated local ring (local Milnor numbers at the organising centre),
binary-cubic Hessian/discriminant (jet factor structure), exact evaluation of
every adjacency witness at three rational parameter values. Numeric parts
(`G3` census, O-side audits): fixed seeds. A `--probe` mode prints the
candidate-witness battery used to fix the witness tables.

---

## 6. AI disclosure

Probe, contract, gate and this spec produced under human direction (2026-07-26),
in the lane `e-series-semantic-germ-20260726`. The verdict type (operational
test surface, never identity) and the scope boundaries were fixed before
computing. All Milnor numbers, coranks, and cubic factor structures are computed exactly
from first principles (rational Buchberger, exact local-ring linear algebra,
binary-cubic Hessian/discriminant tests); the mapping from these computed
invariants to Arnold's type names (`A_μ`, `D_μ`, `E_μ`) applies the standard
classification theorem for simple singularities, which is cited theory, not
re-proved here. The adjacency witnesses were found by a scripted probe battery
(kept in the contract under `--probe`) and are asserted only where the exact
classifier confirms the type at three parameter values (`t = 1, 1/8, 1/64`);
for single-monomial witnesses type-constancy in `t` is additionally a theorem
(quasi-homogeneous scaling), and for the multi-term resonant curves the
coefficient relations are polynomial identities in `t` (verified as printed). The two closures left partial
(`A₆ < E₇`; `A₆, A₇ < E₈`) are named as scope, not smoothed over. Math-facing;
gated on the math-review offload under `.claude/AGENT_OFFLOAD_POLICY.md`
(logged in `.claude/llm_offload_log.md`). No clinical content.
GAIDeT-ICMJE 2025.
