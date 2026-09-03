<!-- docs:meta
topic_id: repo.docs.research.functor-f-phi-jets-vanish-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-phi-jets-vanish-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — the coupling-jet G₂ 3-form φ(a₁,a₂,a₃) vanishes: a structural theorem

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `PHI_JETS_VANISH_PROVEN` (6/6)
**Parent:** `functor_f_petitot_bridge_spec_2026-07-25.md` (`B_OBSTRUCTED`, clause B4)
**Harness:** `scripts/research/functor_f_phi_jets_vanish_contract.py`

---

## 0. What is proved, and its type

The Petitot bridge rung (`B_OBSTRUCTED`) rested, at clause **B4**, on a *measurement*: the
canonical antisymmetric-cubic G₂ invariant `φ(a₁,a₂,a₃)` evaluated on the three coupling
associator-jets was `0` over all 840 sampled 3-line configurations, with the note
"structural proof open". This document closes that gap. It turns the measurement into a
**theorem** over the octonion core.

> **Verdict type, named before computing.** `PHI_JETS_VANISH_PROVEN` is a **positive
> structural theorem** (a "green" result). Its *content*, however, is the **negative** for
> the Petitot bridge: it is *because* φ vanishes structurally that the algebra supplies no
> canonical `x⁴` "butterfly factor". The green must not be misread as "the butterfly
> stratum closes"; it is the proof that it cannot close by this route.

> **Scope, stated up front.** The theorem is about **this coupling pattern** — one
> off-line substitution per associator slot, in the specific jet shape below — over the
> **octonion core** (`bits=3`, Fano plane PG(2,2)). It is **not** a claim that associator
> triples never satisfy `φ≠0` in general (false), and it is **not** the completeness claim
> "no `x⁴` factor of any kind exists" (that remains the parent's disclaimer: the bridge
> argument needs only the domain-bounded rank-3 and this φ(jets)=0, not a completeness
> theorem for invariants). D3 is respected: what follows is a combinatorial fact about the
> Fano plane, **not** an "X ≡ ZD/associator/Petitot" identity.

---

## 1. Setup

Base **Fano line** `(i,j,k)` — three imaginary basis indices with `i^j = k`, hence

```
i ^ j ^ k = 0.                                                        (line identity)
```

The four **off-line units** are the complement `{1..7} \ {i,j,k}`. The coupling uses three
distinct off-line units `u₁,u₂,u₃` and forms the three **coupling jets**

```
a₁ = [e_{u₁}, e_j,   e_k ]      a₂ = [e_i,   e_{u₂}, e_k ]      a₃ = [e_i,   e_j,   e_{u₃}]
```

where `[x,y,z] = (xy)z − x(yz)` is the associator, and `d₁,d₂,d₃ ∈ ℝ` are jet scalars
(the pure jet is `d=1`).

---

## 2. Each jet is a single basis axis of magnitude 2

For octonion basis units, `omul(e_a,e_b) = cds(a,b)·e_{a^b}` (single axis `a^b`, sign
`cds(a,b) = ±1`). Expanding the associator on basis units:

```
[e_a,e_b,e_c] = ( cds(a,b)·cds(a^b,c) − cds(b,c)·cds(a,b^c) ) · e_{a^b^c}.        (†)
```

So an associator of three basis units is **always supported on the single axis `a^b^c`**,
with a coefficient that is either `0` (associative / Fano-line triple) or `±2` (the two
sign products differ, `+1` and `−1`, giving `±2`).

Applying (†) to the three jets and using the line identity `i^j^k=0`:

```
axis(a₁) = u₁^j^k = u₁ ^ (j^k) = u₁ ^ i         (since j^k = i)
axis(a₂) = i^u₂^k = u₂ ^ (i^k) = u₂ ^ j         (since i^k = j)
axis(a₃) = i^j^u₃ = u₃ ^ (i^j) = u₃ ^ k         (since i^j = k)
```

Each substituted triple is **not** a Fano line — e.g. `{u₁,j,k}` has
`u₁^j^k = u₁^i ≠ 0` because `u₁` is off-line (`u₁ ∉ {i,j,k}`, so `u₁ ≠ i`) — hence each
coefficient in (†) is `±2`, not `0`. Therefore

```
a_m = d_m · s_m · 2 · e_{axis(a_m)},     s_m ∈ {+1,−1},     axis(a_m) ≠ 0.
```

The sign `s_m` is recorded but **not load-bearing** (see §4). This is clause **C1** of the
contract (verified as an exact single-axis identity over all 168 configurations).

---

## 3. The XOR obstruction — the heart of the proof

The G₂ 3-form on basis units is `φ₃(e_p,e_q,e_r) = ⟨e_p e_q, e_r⟩ = cds(p,q)·[p^q = r]`,
which is **nonzero only when `p^q = r`, i.e. `p^q^r = 0`** (the triple is a Fano line). By
trilinearity,

```
φ(a₁,a₂,a₃) = 8 · d₁d₂d₃ · s₁s₂s₃ · φ₃( e_{axis(a₁)}, e_{axis(a₂)}, e_{axis(a₃)} ).
```

So the entire question is whether the **axis-triple** forms a Fano line. Compute its XOR:

```
axis(a₁) ^ axis(a₂) ^ axis(a₃)
  = (u₁^i) ^ (u₂^j) ^ (u₃^k)
  = (u₁^u₂^u₃) ^ (i^j^k)
  = u₁ ^ u₂ ^ u₃.                                                     (line identity)
```

The line indices cancel completely; the obstruction is carried entirely by the off-line
units. It remains to show `u₁^u₂^u₃ ≠ 0`.

### 3.1 Lemma (no line lies in the off-line complement)

> **Lemma.** In the Fano plane PG(2,2), for any line `L`, no three points of the
> 4-point complement `{1..7}\L` form a line. Equivalently, for any three distinct
> off-line units, `u₁^u₂^u₃ ≠ 0`.

**Proof (incidence counting, no computation).** In a projective plane any two distinct
lines meet in **exactly one** point. Let `M ≠ L` be any line. Then `|M ∩ L| = 1`, so `M`
has exactly `3 − 1 = 2` points off `L`. A line contained in the complement of `L` would
have `0` points on `L` — contradicting `|M ∩ L| = 1` for every `M ≠ L` (and `L` itself is
not in its own complement). Hence no line lies in the 4-point complement. Since three
distinct nonzero indices XOR to `0` **iff** they form a Fano line, `u₁^u₂^u₃ ≠ 0`. ∎

The contract verifies this exhaustively as clause **C2**: across all `7` lines, all
`C(4,3)=4` off-line triples each (28 checks) — **zero** are lines. This is the *structural
reason* the theorem holds for **every** line, not a coincidence for `(1,2,3)`: the base
line and its off-line complement are forced into general position by the incidence axioms.

### 3.2 Conclusion of the vanishing

By §3 and the Lemma, `axis(a₁)^axis(a₂)^axis(a₃) = u₁^u₂^u₃ ≠ 0`, so the axis-triple is
**never** a Fano line, so `φ₃(e_{axis₁},e_{axis₂},e_{axis₃}) = 0`, so

```
φ(a₁,a₂,a₃) = 0     for all base lines, all off-line assignments, all real d₁,d₂,d₃.   ∎
```

Contract clauses **C3** (XOR ≠ 0 over all configs), **C4** (axis-triple never in `FANO`),
**C5** (φ = 0 over `168 × 6 = 1008` config × scalings, `d=1` and random `d`).

---

## 4. What is and is not load-bearing (honesty ledger)

- **The XOR obstruction (§3) is the whole proof.** It kills φ on its own.
- **Sign prefactor `s₁s₂s₃`** — *not* load-bearing. φ₃ is trilinear and the axis-triple
  factor is identically `0`; the signs never enter. (This is why the given decomposition
  `φ = 8·(signs)·φ₃(axes)` is quoted with signs, yet the signs are not computed.)
- **Pairwise-distinctness of the three axes** — *not* load-bearing, and in fact **fails**
  in `112/168` configurations (measured, clause `C3_NOTE`). "Never a Fano line" is
  guaranteed by the XOR obstruction whether or not the axes are distinct. When they are
  distinct, φ still vanishes by XOR ≠ 0; when they coincide, φ vanishes a fortiori.
- **The jet scalars `d`** — the result is *proved* for all real `d` by trilinearity, not
  merely sampled. The random-`d` rows of `C5` are corroboration, not the argument.
- **Octonion scope.** The proof uses the Fano plane PG(2,2) and the fact that a basis
  associator is a single axis — both specific to `bits=3`. Past the octonions
  (sedenions+) an associator of units need not be a single axis, so the argument does not
  extend and **no `∀n` claim is made**.

---

## 5. Consequence for the Petitot bridge

This strengthens the parent `B_OBSTRUCTED` result at exactly one point: clause **B4** was
"φ(jets)=0 measured over 840 configs (structural proof open)"; it is now a theorem. The
downstream reading is unchanged and **must stay** unchanged: there is no canonical
algebra-forced `x⁴` butterfly factor, so the 3↔3 dimension match is a coincidence, not a
canonical map. This document proves the φ-vanishing *premise*; it does **not** upgrade the
bridge conclusion beyond what the parent's domain-bounded rank-3 already licenses. The
algebra/Petitot divergence remains located precisely at the mediating (butterfly) stratum,
now on a proved rather than measured footing.

---

## 6. Reproduce

```
python3 scripts/research/functor_f_phi_jets_vanish_contract.py
# => FUNCTOR_F_PHI_JETS_VERDICT PHI_JETS_VANISH_PROVEN   (exit 0)
```

Clauses: `C0` core axiom audit · `C1` axis exhibited (±2·e_axis, axis=u_m^line_m) ·
`C2` no line in the off-line complement (28 checks) · `C3` XOR obstruction ·
`C4` axis-triple never a Fano line · `C5` φ = 0 over 1008 config × scalings.
