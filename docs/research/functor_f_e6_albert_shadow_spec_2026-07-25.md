<!-- docs:meta
topic_id: repo.docs.research.functor-f-e6-albert-shadow-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-e6-albert-shadow-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — the G₂ 3-form φ *is* the E₆/Albert cubic cross-term

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PHI_IS_THE_E6_CUBIC_CROSSTERM` (a concrete exceptional bridge; framing corrected)
**Parents:** `functor_f_g2_coherence_spec_2026-07-25.md` (the `φ` 3-form), `functor_f_exceptional_frontier_note_2026-07-25.md` (the boundary this rung is the concrete instance of)
**Harness:** `scripts/research/functor_f_e6_albert_shadow_contract.py`

---

## 0. The result (and a correction)

The Functor-F programme is built on the octonion `G₂` 3-form `φ` (and its dual `ψ`). This
rung places `φ` inside the exceptional tower — **concretely**, not via the Petitot semantic
conjecture:

> For imaginary octonions, `Re(x·y·z) = −φ(x,y,z)`. The real part `Re(x·y·z)` is exactly the
> octonion cross-term of the **cubic form (determinant) `N` of the Albert algebra `J₃(𝕆)`**
> — the `F₄`-invariant that `E₆` preserves projectively. Therefore **the `G₂` 3-form `φ`
> *is* the imaginary restriction of the `E₆`/Albert cubic invariant's octonion cross-term.**
> Functor-F's central form sits *inside* the `E₆` cubic.

> **Correction (2026-07-26).** An earlier draft of this rung said "the *associator* is the
> `G₂` shadow / complement, and the associator `= φ`". That conflated two different objects:
> the **scalar** 3-form `φ(x,y,z) = ⟨xy,z⟩` and the **vector-valued** associator
> `[x,y,z] = (xy)z − x(yz)` (which is the `ψ`/4-form side, `[e_a,e_b,e_c] = −2ψ_{abcd}e_d`).
> The corrected statement is cleaner and stronger: `φ` **is** the cubic cross-term (`E3`),
> and the associator is a *separate* object (`E4`).

---

## 1. Results

| Clause | Result | Reading |
|---|---|---|
| `E1_RE_WELL_DEFINED` | `Re[x,y,z] = 0` (associator purely imaginary); `Re((xy)z) = Re(x(yz))` | `Re(x·y·z)` is a **well-defined, bracketing-independent** trilinear form on 𝕆. |
| `E2_G2_PRESERVES_ALBERT_CUBIC` | `N(J₃(𝕆)) = abc − a·n(x) − b·n(y) − c·n(z) + 2Re(xyz)` is `G₂`-invariant (rel dev `3e-15`) | `G₂ ⊂ F₄ = Aut(J₃(𝕆))`; `Re(xyz)` is the octonion cross-term of the `E₆`/`F₄` invariant. |
| `E3_PHI_IS_CUBIC_CROSSTERM` | for imaginary `x,y,z`, `Re(x·y·z) = −φ(x,y,z)` (dev `4e-15`) | **the `G₂` 3-form `φ` IS the imaginary restriction of the `E₆` cubic cross-term.** |
| `E4_ASSOCIATOR_IS_SEPARATE` | the associator `[x,y,z]` is **vector**-valued (`‖·‖≈14`, the `ψ`/4-form side); `φ` is **scalar** | `φ` (the cubic cross-term) and the associator (the non-associative / `ψ` part) are **different objects** — the earlier conflation is corrected. |
| `E5_CUBIC_SPECIFIC` | `Re(word)` bracketing-independent at length `≤3` (`5e-15`), **not** at `≥4` (`spread ~70`); `Re[x,y,z]=0` vs `Re([x,y,z]·w)≠0` | the bridge is **cubic-specific** — `φ` is 3-linear = the cubic degree, so it matches `E₆` and does **not** lift to the `E₇` quartic by this mechanism. |

Verdict: `FUNCTOR_F_E6_VERDICT PHI_IS_THE_E6_CUBIC_CROSSTERM`.

---

## 2. Why this is the exceptional bridge — and why it is honest

The exceptional-frontier note named the one honest way the frontier becomes more than an
analogy: *exhibit a functor-F object that is exceptional-natural, concretely.* This rung is
that step, in its corrected and stronger form:

- `φ` — the form the whole coherence/tower column (`P_GREEN`, `Q_GREEN`) is built on — is
  **not merely adjacent** to the exceptional tower: it is (the imaginary restriction of) the
  octonion cross-term of the `F₄`/`E₆` cubic invariant `N(J₃(𝕆))`. The bracketing-independence
  of `Re(xyz)` (`E1`) is exactly what makes that cubic invariant well-defined over the
  non-associative octonions, and `φ` is that well-defined trilinear form on the imaginary
  part.
- The **vector** associator `[x,y,z]` (the `ψ`/4-form side) is the *separate*, genuinely
  non-associative object; it is what does **not** reduce to a scalar cubic invariant.

Concrete linear algebra on 𝕆 and `J₃(𝕆)`, verified. **No** semantic exceptional claim
(quarantined, `D3`-class); **no** construction of `E₆`.

---

## 3. What this is NOT

- **Not** a construction of `E₆`/`E₇`/`E₈`, nor a claim they are realised by the programme.
- **Not** the Petitot semantic conjecture (`petitot §4`, quarantined landmine).
- **Not** the earlier (retracted) "associator = φ = G₂ complement" framing — see the §0
  correction.
- **Not** `∀n`; **not** an identity; **not** clinical.

---

## 4. Place in the ladder

```
G_GREEN..Q_GREEN        the G2 φ/ψ tower (closed algebraic column)
PSL27_THREADS_THE_TOWER the Fano/PSL(2,7) unification of φ/ψ/ZD
PHI_IS_THE_E6_CUBIC_CROSSTERM   φ IS the imaginary restriction of the E6/Albert cubic
                        invariant's octonion cross-term (G2 ⊂ F4 = Aut(J3(O))); functor-F's
                        central form sits inside the E6 cubic (cubic-specific; no E7 lift)
```

---

## 5. Suggested next edges

1. **`ψ` (4-form) and a higher `J₃(𝕆)` invariant** — the associator (the `ψ` side) is the
   degree-4 object; does it appear in the `E₇` quartic (Freudenthal triple system), which is
   the honest degree-4 exceptional invariant (built via trace/adjoint, *not* `Re`-of-a-word)?
2. **[RESOLVED as a boundary, `E5`]** the `Re`-of-word split does not lift to `E₇`.
3. **The `PSL(2,7)`/Fano structure inside `F₄`** — concrete subgroup/Weyl question.

---

## 6. Reproduce

```bash
python3 scripts/research/functor_f_e6_albert_shadow_contract.py
# expect: E1..E5 PASS, FUNCTOR_F_E6_VERDICT PHI_IS_THE_E6_CUBIC_CROSSTERM
```

Pure Python (numpy); octonions + the `J₃(𝕆)` cubic form; embeds a `G₂`-invariance check.

---

## 7. AI disclosure

Probe, contract, and note produced under human direction (2026-07-26). The first version
claimed "the associator is the `G₂` shadow/complement of the `E₆` cubic, `= φ`"; a self-check
found this conflated the scalar 3-form `φ` with the vector associator, and the **corrected**
result (`φ = Re(xyz)|imaginary = the cubic cross-term`; associator separate) is recorded here
along with the retraction (§0). Concrete algebra, verified numerically; no semantic claim, no
`E₆` construction. Commit gated on the §10 math-review offload. No clinical content.
GAIDeT-ICMJE 2025.
