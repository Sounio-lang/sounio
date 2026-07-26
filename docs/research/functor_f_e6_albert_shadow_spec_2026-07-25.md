<!-- docs:meta

topic_id: repo.docs.research.functor-f-e6-albert-shadow-2026-07-25
authority: repo_only
audience: researchers
last_validated: 2026-07-26
validated_by: claude
source_of_truth: repo.docs.research.functor-f-e6-albert-shadow-2026-07-25
-->

# Functor F — the associator is the G₂ shadow of the E₆/Albert cubic form

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PHI_IS_G2_SHADOW_OF_E6_CUBIC` (a concrete exceptional bridge)
**Parents:** `functor_f_g2_coherence_spec_2026-07-25.md` (the `φ` 3-form), `functor_f_exceptional_frontier_note_2026-07-25.md` (the boundary; this rung is the concrete algebra that note licensed)
**Harness:** `scripts/research/functor_f_e6_albert_shadow_contract.py`

---

## 0. The result

The whole Functor-F programme studies the octonion associator `[x,y,z]`. This rung places
it inside the exceptional tower — **concretely**, not via the Petitot semantic conjecture:

> The octonion triple product `(x,y,z) ↦ x·y·z` splits `G₂`-equivariantly into two pieces.
> Its **real part `Re(x·y·z)`** is bracketing-independent and is exactly the octonion
> cross-term of the **cubic form (determinant) `N` of the Albert algebra `J₃(𝕆)`** — the
> `F₄`-invariant that `E₆` preserves projectively. Its complementary **imaginary part's
> bracketing-ambiguity is the associator `[x,y,z]` = the `G₂` 3-form `φ`** — the functor-F
> object. So **Functor F lives in the `G₂` (non-associative) complement of the `E₆` cubic
> invariant.**

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `E1_RE_WELL_DEFINED` | `Re[x,y,z] = 0` (associator purely imaginary); `Re((xy)z) = Re(x(yz))` | `Re(x·y·z)` is a **well-defined, bracketing-independent** trilinear form on 𝕆. |
| `E2_G2_PRESERVES_ALBERT_CUBIC` | the Albert cubic form `N(J₃(𝕆)) = abc − a·n(x) − b·n(y) − c·n(z) + 2Re(xyz)` is invariant under `G₂ = Aut(𝕆)` acting on the off-diagonal octonions (rel dev `3e-15`) | `G₂ ⊂ F₄ = Aut(J₃(𝕆))`; `Re(xyz)` is the octonion cross-term of the `E₆`/`F₄` invariant. |
| `E3_ASSOCIATOR_IS_G2_PIECE` | the associator is imaginary (`Re=0`, `‖·‖≠0`), `φ₁₂₃ = +1` | the complementary piece is the `G₂` 3-form `φ`, the functor-F object. |
| `E4_SPLIT` | `x·y·z = Re-part` (the `E₆`/Albert cubic term) `+ Im-part` whose bracketing-ambiguity `= [x,y,z] = φ` | the two pieces are the whole content of the octonion trilinear form. |

Verdict: `FUNCTOR_F_E6_VERDICT PHI_IS_G2_SHADOW_OF_E6_CUBIC`.

---

## 2. Why this is the exceptional bridge — and why it is honest

The exceptional-frontier note (`functor_f_exceptional_frontier_note`) fixed the boundary:
no *semantic* `E₆/E₇/E₈` claim, and it named the **one honest way** the frontier becomes
more than an analogy — *exhibit a functor-F object that is exceptional-natural, concretely*.
This rung is exactly that step, done concretely:

- The associator (functor-F's object) is not merely "adjacent" to the exceptional tower;
  it is **one of the two `G₂`-irreducible pieces** of the single octonion trilinear form,
  the other being an `F₄`/`E₆` invariant (the Albert cubic form's octonion cross-term).
- The bracketing-independence of `Re(xyz)` (its associativity in the real part) is *what
  makes the `E₆` cubic form well-defined over the non-associative octonions* — and the
  functor-F associator is precisely the non-associative remainder that the cubic form
  discards. Functor F is the `E₆`-cubic form's blind spot.

This is concrete linear algebra on 𝕆 and `J₃(𝕆)`, verified. It makes **no** claim that
semantics is exceptional (that stays quarantined, `D3`-class), and it does **not** construct
`E₆` — it exhibits the `F₄`/`E₆` invariant and the functor-F object as complementary pieces
of one trilinear form, with `G₂ ⊂ F₄` verified.

---

## 3. What this is NOT

- **Not** a construction of `E₆`/`E₇`/`E₈`, nor a claim they are realised by the programme.
- **Not** the Petitot semantic conjecture — that four-actant morphology *is* the exceptional
  structure. That remains the quarantined landmine (`petitot §4`, `D3`-class).
- **Not** `∀n` — it is a statement about the octonions 𝕆 and `J₃(𝕆)`.
- **Not** an identity, not clinical.

---

## 4. Place in the ladder — the exceptional bridge, made concrete

```
G_GREEN..Q_GREEN        the G2 φ/ψ tower (closed algebraic column)
PSL27_THREADS_THE_TOWER the Fano/PSL(2,7) unification of φ/ψ/ZD
PHI_IS_G2_SHADOW_OF_E6_CUBIC   the associator φ is the G2 non-associative complement of
                        the E6/Albert cubic invariant -> functor F sits, concretely, at the
                        base of the exceptional tower (G2 ⊂ F4 = Aut(J3(O)))
```

---

## 5. Suggested next edges

1. **`ψ` and the `F₄`/`E₆` structure** — does the co-associator 4-form `ψ` correspond to a
   piece of a higher `J₃(𝕆)` invariant (the quartic / the adjoint)?
2. **Up the square** — the same real/imaginary split for `E₇` (the Freudenthal triple system
   over `J₃(𝕆)`) and `E₈` (magic-square `(𝕆,𝕆)`); is there a functor-F object at each rung?
3. **The `PSL(2,7)`/Fano structure inside `F₄`** — does the Fano threading of the ZD fibres
   correspond to a Weyl/subgroup structure of `F₄`? (Concrete; not the semantic claim.)

---

## 6. Reproduce

```bash
python3 scripts/research/functor_f_e6_albert_shadow_contract.py
# expect: E1..E4 PASS, FUNCTOR_F_E6_VERDICT PHI_IS_G2_SHADOW_OF_E6_CUBIC
```

Pure Python (numpy); octonions + the `J₃(𝕆)` cubic form; embeds a `G₂`-invariance check.

---

## 7. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26), when the operator
pushed to stop refusing the exceptional frontier and attack it. The result is concrete
algebra (the octonion trilinear split, the Albert cubic form, `G₂`-invariance), verified
numerically; it makes no semantic exceptional claim and does not construct `E₆`. Claims
bounded by the four named clauses. Commit gated on the §10 math-review offload. No clinical
content. GAIDeT-ICMJE 2025.
