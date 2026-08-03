<!-- docs:meta
topic_id: repo.docs.research.ade-wildgen-mckay-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.ade-wildgen-mckay-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# ADE–Wildgen conjecture — the McKay correspondence, computed, and a verdict on decidability

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE`
**Parents:** `petitot-semantic-potential.md` (§4, "the deep conjecture", flagged interpretive frontier),
`functor_f_exceptional_frontier_note_2026-07-25.md` (the boundary discipline this rung keeps),
`functor_f_e6_albert_shadow_spec_2026-07-25.md`, `functor_f_e8_capstone_spec_2026-07-26.md` (the continuous bridge)
**Harness:** `scripts/research/ade_wildgen_mckay_contract.py`
**Gate:** `scripts/ci/ade_wildgen_mckay_gate.sh`

---

## 0. The conjecture, stated precisely

From `petitot-semantic-potential.md §4`: Wildgen's four-actant semantics requires the
exceptional singularities `E₆, E₇, E₈` (Arnold); the octonions generate the exceptional Lie
algebras via the Freudenthal–Tits magic square (`G₂ → F₄ → E₆ → E₇ → E₈`); Arnold's
exceptional *singularities* and the exceptional *Lie algebras* share their ADE/Dynkin labels
via the **McKay correspondence**; therefore "the organizing centre of rich semantic
morphology is the octonionic exceptional structure". The parent doc flags this as the
interpretive frontier with a live landmine (analogy mistaken for theorem).

Two forms must be separated:

- **Strong form** — semantic morphology is governed by the *same* exceptional structure as
  the octonions, with McKay as the bridge: the finite/Dynkin content on the singularity side
  coincides with the finite/Dynkin content on the octonion side.
- **Weak form** — exceptional geometry (somewhere in the `G₂…E₈` complex) governs rich
  (four-actant) semantic morphology.

This rung computes the McKay correspondence for `E₆/E₇/E₈` from first principles, compares it
with the `G₂`/octonion structure the rupture programme actually instruments, and gives a
verdict on each form.

---

## 1. What is computed (task b)

**M-side — the label-match is real mathematics.** The binary polyhedral groups are built as
explicit unit quaternions (`2T ⊂ 2O`, `2T ⊂ 2I ⊂ SU(2)`), closure verified; conjugacy classes
enumerated; character tables computed by the Burnside class-algebra algorithm (structure
constants → common eigenvectors → central characters → degrees), self-checked by recovering
the natural `SU(2)` 2-dimensional character as a computed row. The McKay fusion matrix
`N(ρ,σ) = ⟨V₂⊗ρ, σ⟩` is then computed exactly:

| Group | \|Γ\| | classes | irrep dims = affine marks | fusion graph | h = Σd |
|---|---|---|---|---|---|
| `2T` | 24 | 7 | `1,1,1,2,2,2,3` | affine `E₆`, arms `(2,2,2)`, spectral radius 2 | 12 |
| `2O` | 48 | 8 | `1,1,2,2,2,3,3,4` | affine `E₇`, arms `(1,3,3)`, spectral radius 2 | 18 |
| `2I` | 120 | 9 | `1,2,2,3,3,4,4,5,6` | affine `E₈`, arms `(1,2,5)`, spectral radius 2 | 30 |

In each case the trivial representation sits at the extending node (a leaf adjacent to the
natural-representation node; its deletion leaves the finite `E₆/E₇/E₈` diagram with arms
`(1,2,2)/(1,2,3)/(1,2,4)` and spectral radius `< 2`). The McKay correspondence is therefore
**verified computation inside this repo**, not citation: `ADE`-singularities ↔ finite
`SU(2)` subgroups ↔ exceptional Lie labels.

**C-side — comparison with the `G₂`/octonion structure.**

| Clause | Result | Reading |
|---|---|---|
| `C1_G2_EXCLUDED_FROM_SU2_MCKAY` | every McKay fusion matrix is symmetric (the natural rep is self-dual) ⇒ simply-laced only; the `G₂` Cartan matrix `[[2,−1],[−3,2]]` is not symmetric; folding `D₄` under triality reproduces it exactly | **`G₂` has no `SU(2)`-McKay partner.** The octonion automorphism group enters the ADE world only via the `D₄` triality fold, not via McKay. |
| `C2_FANO_FINITE_CONTENT` | `PSL(2,7) = GL(3,2)`, order 168, element orders `{1,2,3,4,7}` (profile `1,21,56,42,48`); point-stabilizer has the `S₄` signature; its derived subgroup has the `A₄` signature; no element of order 5, `5 ∤ 168`; `7` divides none of `\|2T\|,\|2O\|,\|2I\|` | at the Fano/finite level the octonion symmetry **contains** the `E₆`/`E₇` polyhedral groups (tetrahedral `A₄`, octahedral `S₄`) but **not** the `E₈` icosahedral group `A₅`. Finite level only: the continuous `G₂ ⊃ SU(2) ⊃ 2I`; the exclusion is of the *combinatorial* symmetry the programme instruments. |
| `C3_GERMS_MILNOR` | cusp germ `x⁴ = A₃` (`μ=3`), butterfly germ `x⁶ = A₅` (`μ=5`); `E₆ = x³+y⁴`, `E₇ = x³+xy³`, `E₈ = x³+y⁵` have `μ = 6,7,8` (exact rational Buchberger) | the programme's **operative Petitot germs are A-series** (`petitot_potential.py`, R3/R4 `Φ_fp`). No `E`-series germ has ever been constructed on the semantic side. |
| `C4_MAGIC_SQUARE_LINK` | Tits construction `T(A, J₃(𝕆)) = der(A) ⊕ (Im A ⊗ J₃(𝕆)₀) ⊕ der(J₃(𝕆))`: `0+1·26+52 = 78` (`E₆`), `3+3·26+52 = 133` (`E₇`, `der ℍ = su(2)`), `14+7·26+52 = 248` (`E₈`) | the genuine continuous bridge octonions → `E`-series is **Freudenthal–Tits**, already gated (`functor_f` `e6`/`e7`/`e8` contracts: `φ` = `E₆` cubic cross-term; `E₈` has no small-rep invariant) — **not** McKay. |

---

## 2. Evidence for and against (task a)

**For the conjecture.**

1. The ADE label-match is real and now computed in-repo (M1–M4): Arnold's `E₆/E₇/E₈`
   singularities and the exceptional Lie algebras genuinely share finite structure via McKay.
2. The octonionic tower genuinely reaches `E₆/E₇/E₈` (magic square; `C4`; gated by the
   `functor_f` exceptional rungs), and the programme's central form `φ` **is** the imaginary
   restriction of the `E₆` cubic cross-term (`PHI_IS_THE_E6_CUBIC_CROSSTERM`).
3. The `E₆`/`E₇` *finite* polyhedral content (`A₄`, `S₄`) does live inside the octonion's
   Fano symmetry `PSL(2,7)` (`C2`) — a genuine, computed subgroup coincidence.

**Against.**

1. **`G₂` is not in the McKay series at all** (`C1`): it is not simply-laced, so no
   `SU(2)` subgroup corresponds to it. The programme's symmetry and the singularity-side
   finite groups meet only through the `D₄` fold — a different theorem from McKay.
2. **The `E₈` finite content is absent from the Fano group** (`C2`): `5 ∤ 168`. The one
   exceptional singularity whose polyhedral group is icosahedral — the one Wildgen's richest
   morphology would most plausibly need — is exactly the one with no Fano-level presence.
3. **The semantic side has never used an `E`-series germ** (`C3`): the operative
   morphodynamic potentials are the cusp (`A₃`) and the butterfly (`A₅`). Whatever Wildgen's
   four-actant semantics requires, *this* programme has only ever instrumented A-series
   catastrophes on the Petitot side — so even a perfect structural bridge on the algebra side
   would not yet touch the semantic side.
4. The continuous bridge that does exist (`C4`, magic square) explicitly tops out: `E₈` has
   no small-representation invariant for `φ` to live in (`E8_NO_SMALL_REP`).

---

## 3. Verdict (task c)

> **`ADE_WILDGEN_VERDICT STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE`**

- **Strong form: refuted as stated** (obstructed at every computable checkpoint). The
  octonion/`G₂` side and the `E₆/E₇/E₈` singularity side do **not** share the same finite
  structure: `G₂` is excluded from the `SU(2)` McKay series (`C1`); the `E₈` icosahedral
  content is missing from the Fano group (`C2`); and the semantic germs in use are A-series,
  not E-series (`C3`). Any claim of the form "McKay identifies the octonionic exceptional
  structure with Wildgen's singularities" is the named landmine — an analogy mistaken for a
  theorem.
- **Weak form: currently undecidable within the programme.** The continuous bridge exists
  (magic square; `φ` inside the `E₆` cubic — gated), so the exceptional geometry genuinely
  reaches from the octonions to the `E`-series *as Lie algebras*. But the conjecture's
  semantic half — an `E₆/E₇/E₈` catastrophe germ realized in a morphodynamic potential, or
  (per the frontier note) an `F₄`-natural-not-`G₂`-natural object on the semantic side — has
  never been constructed. Until it is, the weak form is neither provable nor refutable here:
  the programme has no semantic `E`-series object to test it on.
- **Not claimed:** that the weak form is false; that McKay itself is anything but the
  verified mathematics of §1; any clinical or semantic-identity (`D3`-class) content.

The one honest path to promotion (unchanged from the frontier note, now with the finite
obstructions quantified): construct an `E`-series germ or `F₄`-natural semantic object with
the same verdict-before-computing discipline. `C2` says the `E₈` version of that object
cannot come from Fano combinatorics; `C1` says its symmetry argument cannot run through
`SU(2)`-McKay on `G₂`; `C4` says the continuous route runs through the magic square and
stops below `E₈` for explicit small-rep invariants.

---

## 4. Reproduce

```bash
bash scripts/ci/ade_wildgen_mckay_gate.sh
# or directly:
.venv/bin/python3 scripts/research/ade_wildgen_mckay_contract.py
# expect: M1..M4, C1..C4 PASS, ADE_WILDGEN_MCKAY_OK
```

Pure Python (numpy), deterministic (fixed seeds), self-contained: quaternion construction of
`2T/2O/2I`, Burnside character tables, McKay fusion, `GL(3,2)` enumeration, exact rational
Buchberger for the Milnor numbers.

---

## 5. AI disclosure

Probe, contract, gate and this spec produced under human direction (2026-07-26), in the lane
`ade-wildgen-mckay-20260726`. The McKay computation is verified numerically from first
principles (no cited character tables); the `G₂`-exclusion, the `PSL(2,7)` subgroup content
and the Milnor numbers are computed exactly. The verdict separates a refuted strong form
from an undecidable weak form and asserts **no** semantic exceptional claim (`D3`-class
quarantine kept). Math-facing; gated on the math-review offload under
`.claude/AGENT_OFFLOAD_POLICY.md` (logged in `.claude/llm_offload_log.md`). No clinical
content. GAIDeT-ICMJE 2025.
