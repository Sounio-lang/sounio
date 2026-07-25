<!-- docs:meta
topic_id: repo.docs.research.rupture-abcd-claims-2026-07-24
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-abcd-claims-2026-07-24
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Rupture A+B+C+D — claim ladder, singularity orders, and executable contracts

**Date:** 2026-07-24  
**Orthography:** EN-UK  
**Status:** executable contract ladder green on `main` (PRs #1432, #1446); not a clinical or ML result  
**Discursive map:** `rupture-programme-synthesis_2026-07-25.md`  
**Parents:** `nonassociativity-as-rupture.md` (PR #1237), `rupture-as-singularity.md`,
`petitot-semantic-potential.md`, `sedenion_zd_fibers.md`, Frente A measure tests,
`probe-corrected-protocol.md`  
**Harnesses:** `scripts/research/rupture_r2_{fiber_measure_contract,full_tubular_probe}.py`,
`scripts/research/rupture_r3_fano_restriction_probe.py`,
`scripts/research/rupture_r4_fano_field_contract.py`,
`scripts/ci/rupture_abcd_contracts_gate.sh`

---

## 0. Purpose

This note freezes the **claim ladder** for the rupture programme after the A+B+C+D
synthesis. It does four jobs:

| Letter | Job |
|---|---|
| **D** | Split “same event on both sides” into referee-safe levels D0–D3 |
| **C** | Classify singularity **orders** so instruments are not smuggled into each other |
| **A / R2** | Define the first honest **measure-theoretic** object on sedenion annihilation |
| **B / R3** | State the Fano-restriction → Petitot-strata hypothesis as a **contract**, not a match |

**Hard rule:** clinical, ECSS, depression, and risk readings stay **outside** R2 and R3
until those contracts are green on pure algebra / morphodynamics.

---

## 1. D — What “same event” may mean

The load-bearing sentence in `rupture-as-singularity.md`:

> Rupture-as-singularity is the same event on both sides — `det L_x = 0`.

is **ambiguous**. Use only the graded reading below.

| Level | Statement | Status |
|---|---|---|
| **D0** | \(x\) is a zero-divisor \(\iff L_x\colon y\mapsto x\cdot y\) is singular \(\iff \det L_x = 0\) (Dugger–Isaksen) | **definition / theorem** |
| **D1** | The family \(\{L_x\}\) degenerating on \(\{\det=0\}\) is the **same kind of object** as a Thom family of operators (Hessian / left-multiplication) whose bifurcation set is a singular locus | **structural parallel** — language of families, **not** variety isomorphism |
| **D2** | In the Cayley–Dickson tower, the catastrophe set of two-unit sums is empty on ℍ,𝕆 and **born** at 𝕆→𝕊 (84/210), growing under further doubling | **executed** (`catastrophe_cd` lineage) |
| **D3** | A Petitot semantic potential’s bifurcation set **is** (or is dual to) the ZD locus or the octonion associator | **not established** — and the repo documents an **honest divergence** |

### Referee-safe reformulation (replace unqualified D3)

**Keep (D0+D1+D2):**

> In the CD tower, \(\{\det L_x=0\}\) is the singular locus of left-multiplication. It is empty
> on the division algebras and appears at the 𝕆→𝕊 doubling. As a family of operators in \(x\),
> that locus is a bifurcation set in Thom’s **operational** sense. \(|\det L_x|\) (scale-normalised)
> is a **positive, graded** distance-to-algebraic-rupture.

**Do not assert without B-green:**

> Semantic rupture (Petitot) and epistemic rupture (annihilation) are the same singularity structure.

**Replace with H-bridge:**

> Both faces admit family-degeneracy as the form of obstruction. The load-bearing geometric
> unification of the **algebraic** faces is G₂ (associator ≅ G₂ 3-form; ZD ≅ G₂ / V₂(ℝ⁷)).
> Unification with morphodynamic semantics is a **programme** requiring the Fano-restriction
> hypothesis (R3), not identity D3.

---

## 2. C — Orders of singularity (instrumentation table)

| Order | Object | Min. algebra | Positive invariant | Repo instrument | Must not claim |
|---|---|---|---|---|---|
| **0** | commutator \([a,b]\) | ℍ | \(\|[a,b]\|\) | secondary | associator / ZD |
| **1a** | associator \([a,b,c]\) | 𝕆 | \(\|[a,b,c]\|\); non-Fano \(\|\cdot\|^2=4\) | `oct_associator`, `associator_field` | semantic jump |
| **1b** | G₂ 3-form | 𝕆 | alternating form = associator | #1230 hinge | clinical state |
| **2** | \(\det L_x=0\) | 𝕊 | \(|\det L_x|\), \(d_{\mathrm{sing}}\) | catastrophe scan | Petitot square |
| **2′** | annihilating pair \(x\cdot y=0\) | 𝕊 | fiber label \(L=\mathrm{lo}\oplus\mathrm{hi}\) | `sedenion_zd_fibers` | “invisibility” slogan |
| **2″** | **composed** annihilation | matrices / 𝕊 stacks | principal-angle **alignment** | probe-corrected-protocol | gap alone |
| **3** | Massey / Borromean | topology / A∞ | ternary class | #1225 | ZD pairs |
| **M** | Ollivier–Ricci \(\kappa\) | metric graph | \(\kappa\), law of \(\kappa\) | `orc.sio`, epistemic curvature | associator / ZD |
| **P** | bifurcation of \(V(\cdot;c)\) | control space | stratum codimension | `petitot_potential.py` | 𝕆/𝕊 identity |

### Non-collapse rules

1. **Ord 1 ⇏ ord 2** — 𝕆 is non-associative and still a division algebra.  
2. **Ord 2 ⇏ ord M** — algebra ≠ transport geometry.  
3. **Ord P ⇏ ord 1/2** without a functor (R3 / D3).  
4. **Ord 3 ⇏ ord 2′** — sister metaphors; separate proofs.  
5. **Ord 2″ ≠ ord 2′** — locus without alignment is not compositional annihilation (rotating control).

**G₂** binds ord 1 and ord 2 in the algebraic column. **ORC (M)** and **Petitot (P)** are
parallel faces until a green functor says otherwise.

---

## 3. A / R2 — First honest statistical object on 𝕊

### Already executed (do not re-litigate)

| Brick | Fact | Evidence |
|---|---|---|
| Locus census | 84 projective primitives, 168 pairs, 42 quartets | `generate_sedenion_zero_divisor_geometry.py` |
| Fibers | 7 fibers \(L\in\{9..15\}\), 12 verts, deg 4, bipartite 6+6, **intra-fiber only** | `sedenion_zd_fibers.sio` + oracle + CI gate |
| Measure slice (Frente A) | on-locus \(\mathbb{E}[r]=\mathrm{Var}[r]=0\) exact over ℚ; off-locus \(\mathbb{E}=0\), \(\mathrm{Var}>0\) exact | `sedenion_measure_annihilation_{exact,general,bigint}.sio` |
| Alignment probe | gap alone is false positive; principal angles separate aligned vs rotating | `probe-corrected-protocol.md` |

### Definitions

- **Scale-free singularity distance** (continuous relaxation; not required for discrete R2-partial):
  \[
  d_{\mathrm{sing}}(x)=\frac{|\det L_x|^{1/16}}{\|x\|}.
  \]
- **Fiber label** on primitives: \(L(v)=\mathrm{lo}\oplus\mathrm{hi}\).  
- **Reference laws:** \(\mu_{\mathrm{loc}}\) (on locus / per fiber), \(\mu_{\varepsilon}\) (tubular),
  \(\mu_{\mathrm{G}}\) (sphere / Gaussian control), \(\mu_{\mathrm{rot}}\) (rotating dead subspaces — ord 2″).

### Discriminants (ordered)

1. \(p_{\mathrm{ann}}(\varepsilon)=\mathbb{P}(\|X\cdot Y\|<\varepsilon\mid \|X\|,\|Y\|\ge\delta)\)  
2. \(p_{\mathrm{fiber}}=\mathbb{P}(L(X)=L(Y)\mid\text{annihilation})\) — must be 1 on structural support  
3. \(\mathrm{Var}[r_k]\) of product coordinates — Frente A contract  
4. Subspace alignment for depth-\(T\) composition (ord 2″)  
5. **Never** `gap_dominance` alone  

### Contracts

**R2-partial (green today if harness passes):**

> (i) All annihilating primitive edges are intra-fiber (`INTRA_BAD=0`).  
> (ii) On the canonical rational slice, on-locus Var\(=0\) and off-locus Var\(=1/150\) (Frente A numbers).  
> (iii) Uniform random mixed-half pairs annihilate at rate \(\ll 1\) vs structured fiber partners.

**R2-full (measured, not proved):**

> Exact anchors on the discriminant hypersurface + continuous \(d_{\mathrm{sing}}\) tubular
> measurements that separate \(\mu_{\mathrm{loc}}\) from \(\mu_{\mathrm{G}}\) across all seven fibers.

| Layer | Content | Status |
|---|---|---|
| **Exact** | \(\det L_x=0\), rank 12 on all 84 primitives; transversal vanishing order 4 on all 168 edges; exact poly per fiber | PASS |
| **Measured** | uniform MC tube upper bound; local \(d_{\mathrm{sing}}\sim t^{1/4}\) slopes uniform across 7 fibers; model tube mass estimator (declared approx.) | `R2_FULL_MEASURED` |
| **Not claimed** | continuous law as theorem; D3; clinical content | — |

Executables:
- partial: `scripts/research/rupture_r2_fiber_measure_contract.py`
- full (measured): `scripts/research/rupture_r2_full_tubular_probe.py`

Note: `docs/research/rupture-r2-full-tubular_2026-07-25.md`

### Explicit non-claims (R2)

- No clinical mapping of \(\det L_x\).  
- No “statistical invisibility” slogan without \(\mu\) and discriminants (1)–(4).  
- No D3 / Petitot identification.

---

## 4. B / R3 — Fano restriction → Petitot strata

### Documented divergence (must remain)

| | isolated square | field of squares |
|---|---|---|
| **Petitot** | already not Booleanisable (strata topology) | — |
| **𝕆** | one Fano line ≅ ℍ → associative, Booleanisable | cross-line associator \(\neq 0\) |

### Hypothesis (reconciliation)

> No real semantic square is isolated. The extra topology Petitot needs on a lone square is the
> **shadow** of ambient non-associativity — the restriction of the associator to a neighbourhood
> of one Fano line.

### B-contract (R3-worked)

Exhibit one **worked opposition** and a candidate map
\(\Phi:\{\text{configs near }L_\star\}\to(a,b)\) into the cusp control plane
\(V=x^4/4+a x^2/2+b x\), \(\Delta=4a^3+27b^2\), such that:

1. **(i)** Pure \(L_\star\) configurations: associator \(=0\); \(\Phi\) does not require non-Fano data.  
2. **(ii)** Minimal non-Fano perturbation drives \(\Phi\) across the fold \(\Delta=0\) (well count changes).  
3. **(iii)** Crossing **type** (2→1 merge vs antipodal appearance) correlates with associator
   **direction** in \(\mathrm{Im}\,𝕆\), not only \(\|\mathrm{assoc}\|\).

**Fail condition:** every natural \(\Phi\) either never needs non-Fano to match Petitot-like strata,
or non-Fano only rescales \(V\) without changing stratum type → parallel formalisms; keep D1 only.

Executable probe: `scripts/research/rupture_r3_fano_restriction_probe.py`.  
Derivation (2026-07-25): `docs/research/rupture-r3-fano-phi_2026-07-25.md`.

### Φ_fp (first principles — R3_GREEN operational)

Exact jet: \([e_i+\varepsilon e_u,e_j,e_k]=\varepsilon[e_u,e_j,e_k]\), single-axis support.

\[
a = A_0 + \frac{\|\alpha\|^2}{4},\qquad
b = \tau + \frac{\alpha_m}{2},\qquad A_0=-1\text{ (unit choice)}.
\]

| Verdict | Meaning |
|---|---|
| `R3_PARTIAL` | JET + (i)+(ii)+(iii) under Φ_fp |
| `R3_GREEN` | + (iii+) even/odd path classes = contrariety vs contradiction — **current** |

**Path C (contrariety):** \(\tau=-\alpha_m/2\) (\(b\equiv 0\)) → neutral monostable.  
**Path D (contradiction):** \(\tau=0\) (\(b=\alpha_m/2\)) → polar monostable, sign selects pole.

Detail: `docs/research/rupture-r3-fano-phi_2026-07-25.md`.

### Explicit non-claims (R3)

- No isomorphism \(B(V_{\mathrm{cusp}})\cong ZD(𝕊)\) (**D3 still forbidden**).  
- No Wildgen E₆–E₈ = magic square **theorem** (label conjecture only).  
- No ECSS / affect / depression smuggle.  
- `R3_GREEN` = operational path-class non-Booleanisability, **not** a topos theorem and not D3.

---

## 4b. R4 — multi-line Fano field (system level)

R3 is the neighbourhood of **one** line. R4 is the **field of seven squares**
plus Φ_fp path classes sourced from the **cross-line** jet.

| ID | Fact | Status |
|---|---|---|
| F1–F2 | 7 lines associative; pairwise meet in 1 unit | PASS |
| F3–F4 | cross-line non-Fano \(\|\mathrm{assoc}\|=2\); census 7+28 | PASS |
| F5 | two-line mixing jet linear; L1-internal \(\|\alpha\|=0\), cross \(\|\alpha\|=2\) | PASS |
| F6 | worked pair shares one term | PASS |
| F7 | multi-line Φ_fp Path C/D from cross jet (not single-line off-unit) | PASS |

| Verdict | Meaning |
|---|---|
| `R4_PARTIAL` | F1–F6 field census |
| `R4_GREEN` | F1–F7 multi-line path classes — **current** |

Harness: `scripts/research/rupture_r4_fano_field_contract.py`  
Note: `docs/research/rupture-r4-fano-field_2026-07-25.md`

---

## 5. Claim ladder diagram

```text
[D0] det L_x = 0  ⇔  ZD                         (def/thm)
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
      [D2]           [C ord2]       [A / R2]
   tower birth     singularity    measure+fiber
   𝕆→𝕊             table          contracts
         │              │
         │              ├ ord1 associator / G₂
         │              ├ ord2″ alignment
         │              ├ ord3 Massey
         │              └ ord M ORC (parallel)
         ▼
      [D1] Thom-as-family-degeneracy language
         │
         ✗ not automatic
         ▼
      [D3] Petitot square = ZD/associator     FORBIDDEN without R3
         │
         ▼
      [B / R3] Φ: Fano neighbourhood → (a,b)
               non-Fano ⇔ fold crossing + type
```

---

## 6. What may be said in public text *now*

### May say

1. ZD = singular locus of \(L_x\); empty on 𝕆; born at 𝕊; 7-fiber structure executed and gated.  
2. Exact rational measures on an algebraic slice give \(\mathrm{Var}(\text{product coords})=0\) on-locus
   and exact positive variance off-locus (Frente A).  
3. Compositional annihilation is diagnosed by subspace alignment, not spectral gap alone.  
4. Octonion Booleanises the *isolated* Fano square and obstructs at the *field* of squares
   (divergence from Petitot).  
5. Thom parallel is **family degeneracy of operators** (D1), not identity of semantic potentials (D3).

### Must not say

1. Unqualified “same event on both sides” (D3).  
2. \(\det L_x\) measures mood, depression, or suicide risk.  
3. Non-associativity improves generic ML tasks as evidence for the thesis.  
4. ORC = associator = ZD under one scalar.  
5. ADE/Wildgen as theorem.

---

## 7. Gate

```bash
bash scripts/ci/rupture_abcd_contracts_gate.sh
# expects: RUPTURE_ABCD_CONTRACTS_OK
# R2: PARTIAL PASS
# R3: GREEN (single-line path classes) + D3_identity_still_forbidden
# R4: GREEN (field + multi-line path classes) + D3_identity_still_forbidden
```

---

## 8. Relation to existing docs

| Doc | Role after this note |
|---|---|
| `nonassociativity-as-rupture.md` | programme map; unclaimed jewels |
| `rupture-as-singularity.md` | D0–D2 evidence; **read its D3 sentence through §1 of this note** |
| `petitot-semantic-potential.md` | divergence + R3 setup |
| `relational-annihilation-geometry.md` | clinical **hypothesis** only — quarantine |
| ECSS formal model | uses ord 1 emissions; not ord 2 measure |

---

## 9. AI disclosure

Synthesis and contract harnesses drafted under human direction (2026-07-24). Substantive geometry
reuses prior math-reviewed / CI-gated census and measure work. No clinical claims. GAIDeT-ICMJE 2025.
