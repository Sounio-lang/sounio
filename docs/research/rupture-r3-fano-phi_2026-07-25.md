<!-- docs:meta
topic_id: repo.docs.research.rupture-r3-fano-phi-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-r3-fano-phi-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# R3 — first-principles Φ from the Fano-neighbourhood associator jet

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** B-contract operational close (`R3_GREEN`); D3 still forbidden; field-level follow-on is R4  
**Harness:** `scripts/research/rupture_r3_fano_restriction_probe.py`  
**Parents:** `rupture-abcd-claims_2026-07-24.md` §B, `petitot-semantic-potential.md`,
`rupture-as-singularity.md`

---

## 1. What was wrong with the previous candidate Φ

The 2026-07-24 probe used

\[
a = -1 + c\,\|\alpha\|^2,\qquad b=\tau
\]

with a fitted scale \(c=0.85\) and **no** odd jet. That showed only that an
ε-driven fold crossing is *arrangeable* (R3_HINT). It could not address
clause (iii): direction of the associator in \(\mathrm{Im}\,𝕆\).

---

## 2. Exact jet lemma (executed)

Fix a Fano line \(L_\star=\{e_i,e_j,e_k\}\) with \(e_i e_j=e_k\) (associative triple).
For any off-line unit \(e_u\) (\(u\notin\{i,j,k\}\)) and \(\varepsilon\in\mathbb{R}\),

\[
\bigl[e_i+\varepsilon e_u,\; e_j,\; e_k\bigr]
\;=\;
\varepsilon\,\bigl[e_u,\; e_j,\; e_k\bigr].
\]

**Proof sketch:** the associator is trilinear (over \(\mathbb{R}\)) and
\([e_i,e_j,e_k]=0\) on a Fano line, so only the \(\varepsilon\)-term survives.

**Structure of the pure triple associator** (CD sign law, bits=3), worked line
\((1,2,3)\):

| \(u\) | \([e_u,e_2,e_3]\) | support |
|---|---|---|
| 4 | \(+2\,e_5\) | single axis |
| 5 | \(-2\,e_4\) | single axis |
| 6 | \(-2\,e_7\) | single axis |
| 7 | \(+2\,e_6\) | single axis |

So ambient coupling to one off-line direction produces an associator that is
**pure, directed, and linear in \(\varepsilon\)**:

\[
\alpha(\varepsilon)
= \varepsilon\cdot(\pm 2)\,e_m
\qquad\Rightarrow\qquad
\|\alpha\|=2|\varepsilon|,
\quad
\alpha_m=\pm 2\varepsilon.
\]

The **direction** is the choice of \(u\) (which axis \(e_m\)) and the **sign** of
\(\varepsilon\). Norm alone erases the sign.

---

## 3. First-principles Φ_fp

### Split of responsibilities

| Piece | Source | Role |
|---|---|---|
| Internal opposition depth \(A_0\) | **semantic** (Greimas / Petitot input) | double-well scale on the isolated square |
| Semantic tilt \(\tau\) | **semantic** | which contrary is biased when \(\alpha=0\) |
| Even jet \(\|\alpha\|^2/4\) | **algebra (𝕆)** | ambient non-associativity strength |
| Odd jet \(\alpha_m/2\) | **algebra (𝕆)** | ambient direction |

The isolated Fano square is Booleanisable (associator 0). It does **not** by
itself generate Petitot’s non-Boolean topology. Ambient jet terms are the only
𝕆-derived deformation of the control plane.

### Definition

Cusp potential \(V=x^4/4 + a\,x^2/2 + b\,x\), fold \(\Delta=4a^3+27b^2\).

\[
\boxed{
\begin{aligned}
a &= A_0 + \frac{\|\alpha\|^2}{4},\\
b &= \tau + \frac{\alpha_m}{2},
\end{aligned}
}
\qquad A_0=-1
\quad\text{(unit choice for opposition depth).}
\]

**Why these normalisations (not free fits):**

- On the pure table, \(\|\alpha\|^2/4=\varepsilon^2\) and \(\alpha_m/2=\pm\varepsilon\), so both
  jets are measured in the **same** dimensionless \(\varepsilon\)-units as the
  neighbourhood coordinate.
- \(A_0=-1\) places the pure square (\(\varepsilon=0,\tau=0\)) at the classical
  symmetric bistable cusp point used in every textbook cusp; it is a choice of
  **unit**, not a regression against data.
- No coefficient \(c\) is fitted to force a crossing.

### What Φ_fp is *not*

- Not an isomorphism \(B(V)\cong ZD(𝕊)\) (D3 still false).
- Not a derivation of \(A_0\) from 𝕆 (semantic input stays semantic).
- Not yet a separation of **contrariety** (2→1 merge) vs **contradiction**
  (antipodal) as full Petitot strata — that needs at least a second control
  path / butterfly; see §5.

---

## 4. B-contract status under Φ_fp

| Clause | Statement | Status under harness |
|---|---|---|
| **JET** | linearity + single-axis pure associators | **PASS** (all 4 off-line units on worked line) |
| **(i)** | \(\varepsilon=0\Rightarrow\alpha=0\Rightarrow(a,b)=(A_0,\tau)\) | **PASS** |
| **(ii)** | increasing \(|\varepsilon|\) yields fold crossing (2→1 wells) for some \(\tau\) | **PASS** |
| **(iii)** | \(\mathrm{sign}(\varepsilon)\) flips \(b\) (even \(a\)) and flips which well is deeper | **PASS** |
| **(iii+)** | contrariety vs contradiction as **two path classes** from the same jet | **PASS** (operational) |
| **D3** | semantic square = algebraic locus | **FORBIDDEN** |

**Verdict:** `R3_GREEN` — operational B-contract under Φ_fp.  
**Still not D3** (no identity of varieties / potentials with ZD).

Norm-only control: \(\Phi(\alpha)=(A_0+\|\alpha\|^2/4,\;\tau)\) is identical for
\(\pm\varepsilon\) and **cannot** flip wells — so direction is load-bearing, not cosmetic.

---

## 5. Clause (iii+) — two path classes from one jet

The even/odd split of \(\alpha\) under Φ_fp yields two control paths from the
same Fano-neighbourhood coupling:

### Path C — contrariety (even jet only)

Set \(\tau=-\alpha_m/2\) so \(b\equiv 0\). Sweep \(\varepsilon\):

- start bistable, symmetric poles at \(\pm 1\);
- both wells approach the origin;
- end monostable **neutral** (\(x=0\)).

Operational reading: both contraries can dissolve into a neutral term
(“both false” / neutralisation is possible).

### Path D — contradiction (odd jet)

Set \(\tau=0\) so \(b=\alpha_m/2\). Sweep \(\varepsilon\):

- start bistable;
- for \(|\varepsilon|\gtrsim 0.5\), monostable **polar** (\(|x|\gtrsim 0.5\));
- \(\mathrm{sign}(\varepsilon)\) selects which pole;
- no neutral monostable on the path.

Operational reading: one pole deepens as the other vanishes; “neither” does
not occur.

### Why this is non-Booleanisability (operational)

A Boolean lattice \(2^2\) has a **single** complement type. Hosting both a
*neutralisation* path class and a *polar-selection* path class as distinct
moves is exactly Petitot’s obstruction, realised here as two path classes in
the cusp plane **from the same algebraic jet**. This is an **operational
witness**, not a topos-theoretic theorem and not D3.

---

## 6. What remains after R3_GREEN

1. Multi-line field: two Fano lines meeting in one unit (system-level obstruction).
2. Optional: lift \(A_0\) from a declared semantic potential on the quaternion
   subalgebra (still not from 𝕆 alone).
3. Keep D3 closed until a natural \(\Phi\) without unit-choice narrative exists
   *and* a variety-level correspondence is proved (not expected soon).

---

## 7. Reproduce

```bash
python3 scripts/research/rupture_r3_fano_restriction_probe.py
# expect: JET_LEMMA PASS, CLAUSE_I/II/III/III_PLUS PASS, R3_VERDICT R3_GREEN
# expect marker: D3_identity_still_forbidden

bash scripts/ci/rupture_abcd_contracts_gate.sh
# expect: RUPTURE_ABCD_CONTRACTS_OK
```

---

## 8. AI disclosure

Derivation and harness under human direction (2026-07-25). No clinical claims.
GAIDeT-ICMJE 2025.
