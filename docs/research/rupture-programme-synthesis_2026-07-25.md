<!-- docs:meta
topic_id: repo.docs.research.rupture-programme-synthesis-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.rupture-programme-synthesis-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Non-associativity as the algebra of rupture — programme synthesis (executed state)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Audience:** researchers, referees, and future agents  
**Status:** synthesis of an **executable** claim ladder on `main` after PRs
[#1432](https://github.com/Sounio-lang/sounio/pull/1432) and
[#1446](https://github.com/Sounio-lang/sounio/pull/1446)  
**One-command proof surface:**

```bash
bash scripts/ci/rupture_abcd_contracts_gate.sh
# → RUPTURE_ABCD_CONTRACTS_OK
```

This document is the **discursive** map. The **contract** map remains
`rupture-abcd-claims_2026-07-24.md`. Neither is a clinical paper, an ML benchmark
claim, or a proof of D3 (semantic potential ≡ algebraic locus).

---

## 0. Thesis in one paragraph

Composition that is not context-free is the shared name of two research faces:
**semantic** rupture (how meanings fail to reassociate) and **epistemic** rupture
(how knowledge fails to invert / annihilates). Classical formalisms mark rupture
with a *negative* object (missing map, unglued section, diverging metric, vanished
attractor). The Cayley–Dickson tower supplies **positive, graded, computable**
invariants: the octonion **associator** (G₂ 3-form) and the sedenion
**zero-divisor / \(\det L_x\)** singularity. The programme instruments both,
separates singularity *orders*, forbids claim smuggling, and ships a gated
executable stack from combinatorial annihilation through Fano morphodynamics to
the multi-line field.

---

## 1. What is *not* the contribution

| Non-claim | Why |
|---|---|
| “Octonion nets beat Mamba on Dyck” | Task without rupture; non-assoc need not help (#1232-class) |
| “\(\det L_x\) measures depression / suicide risk” | Clinical mapping is hypothesis only; quarantined |
| “Petitot potential = ZD variety” (**D3**) | Documented divergence; operational bridge ≠ identity |
| “Continuous tubular law is a theorem” | `R2_FULL_MEASURED` — exact anchors + MC, not a proof |
| “R3/R4_GREEN = topos non-Booleanisability” | Path-class witness in the cusp plane only |
| Fixed-point Madaros / self-hosting | Orthogonal compiler track |

The contribution is **instrumentation + claim discipline** for rupture as algebra.

---

## 2. Claim ladder D0–D3 (what “same event” may mean)

From `rupture-as-singularity.md` and the referee split in the claims doc:

| Level | Content | Status on `main` |
|---|---|---|
| **D0** | \(x\) ZD \(\iff L_x\) singular \(\iff \det L_x=0\) | definition / theorem |
| **D1** | \(\{L_x\}\) is a *Thom-style family* degenerating on a singular locus | structural parallel (language of families) |
| **D2** | Catastrophe set empty on ℍ,𝕆; **born** at 𝕆→𝕊 (84/210 two-unit sums) | executed (CD tower scan lineage) |
| **D3** | Petitot bifurcation set **is** the ZD locus / associator | **FORBIDDEN** — honest divergence stands |

**Keep (D0–D2):** graded distance-to-algebraic-rupture via \(|\det L_x|\) (normalised).  
**Replace unqualified D3 with H-bridge:** both faces admit *family degeneracy*;
algebraic faces unify under G₂; morphodynamic semantics is a *programme*
(R3/R4), not an identity theorem.

---

## 3. Orders of singularity (instrumentation table)

Do not collapse sensors.

| Order | Object | Min. algebra | Positive invariant | Repo surface |
|---|---|---|---|---|
| **1a/1b** | associator / G₂ 3-form | 𝕆 | \(\|[a,b,c]\|\); non-Fano \(\|\cdot\|^2=4\) | `oct_associator`, `associator_field` |
| **2** | \(\det L_x=0\) | 𝕊 | \(d_{\mathrm{sing}}\), tube mass | R2 full probe |
| **2′** | annihilating pair | 𝕊 | fiber \(L=\mathrm{lo}\oplus\mathrm{hi}\) | `sedenion_zd_fibers`, R2 partial |
| **2″** | composed annihilation | stacks | subspace **alignment** | probe-corrected-protocol (separate) |
| **3** | Massey / Borromean | topology | ternary class | #1225 lineage |
| **M** | Ollivier–Ricci \(\kappa\) | graphs | curvature law | ORC / ECSS (parallel) |
| **P** | bifurcation of \(V(\cdot;c)\) | control space | stratum type | Petitot / Φ_fp |

**Non-collapse:** ord 1 ⇏ ord 2 (𝕆 is still a division algebra); ord 2 ⇏ ord M;
ord P ⇏ ord 1/2 without a functor; G₂ binds 1 and 2 in the *algebraic* column only.

---

## 4. Executable contracts (the green ladder)

### 4.1 R2 — annihilation as statistical / continuous object (A)

| Contract | Meaning | Verdict |
|---|---|---|
| **R2-partial** | 84 verts / 168 edges / 7 fibers; `INTRA_BAD=0`; Frente A \(\mathrm{Var}=0\) on-locus, \(1/150\) off; random ann. ≪ structured | **PASS** |
| **R2_FULL_MEASURED** | Exact: det=0, rank 12 on all primitives; vanishing order 4 on all edges; exact poly leading 256. Measured: MC tube ≪ \(\mu_G\); \(d_{\mathrm{sing}}\sim t^{1/4}\) slopes fiber-uniform | **PASS** (not a theorem) |

Harnesses:
- `scripts/research/rupture_r2_fiber_measure_contract.py`
- `scripts/research/rupture_r2_full_tubular_probe.py`
- note: `rupture-r2-full-tubular_2026-07-25.md`

### 4.2 R3 — Fano neighbourhood → Petitot path classes (B)

**Exact jet:** \([e_i+\varepsilon e_u,e_j,e_k]=\varepsilon[e_u,e_j,e_k]\), single-axis support.

**Φ_fp** (unit \(A_0=-1\) for opposition depth — semantic unit choice, not a fit):

\[
a=A_0+\frac{\|\alpha\|^2}{4},\qquad
b=\tau+\frac{\alpha_m}{2}.
\]

| Path | Dial | Outcome |
|---|---|---|
| **C — contrariety** | \(\tau=-\alpha_m/2\) \(\Rightarrow b\equiv 0\) | monostable **neutral** \(x=0\) |
| **D — contradiction** | \(\tau=0\) \(\Rightarrow b=\alpha_m/2\) | monostable **polar**; \(\mathrm{sign}(\varepsilon)\) selects pole |

Isolated Fano square: associator 0 (Booleanisable). Ambient jet supplies the
extra control that Boolean \(2^2\) cannot host as *two* opposition types.
Verdict: **`R3_GREEN`** (operational).  
Harness: `rupture_r3_fano_restriction_probe.py` · note: `rupture-r3-fano-phi_2026-07-25.md`

### 4.3 R4 — field of seven squares (system level)

| Fact | Result |
|---|---|
| 7 lines associative; 21 pairs meet in exactly 1 unit | PASS |
| Census \(\binom{7}{3}\): 7 Fano + 28 non-Fano (\(\|\mathrm{assoc}\|=2\)) | PASS |
| L1-internal residual \(\|\alpha\|=0\); L1⊕L2 cross \(\|\alpha\|=2\) | PASS |
| Φ_fp Path C/D **sourced from the cross-line jet** (not single-line off-unit) | **`R4_GREEN`** |

The system residual is the load-bearing contrast: coupling a second square through
the shared term produces an obstruction invisible to pure L1 perturbation.

Harness: `rupture_r4_fano_field_contract.py` · note: `rupture-r4-fano-field_2026-07-25.md`

---

## 5. How the pieces fit (architecture)

```text
                    G₂ (algebraic spine)
                   /                    \
          associator (ord 1)        ZD / det L_x (ord 2)
               |                          |
          R3 single-line jet          R2 fibers + d_sing tube
               |                          |
          Φ_fp Path C / D              R2_FULL_MEASURED
               \                          /
                \                        /
                 R4 multi-line field
                 (cross jet → same Φ_fp paths)
                          |
                    D1 Thom-language
                          |
                    D3 ── ✗ still closed
                          |
              ORC / ECSS / clinical ── outside gate
```

**Compiler role** (from the original rupture synthesis): infrastructure to compute
in the algebra of rupture — associator, VJP, ZD locus, discrete curvature — not
an ML contest.

---

## 6. Relation to prior literature map

The literature synthesis (`nonassociativity-as-rupture.md`, PR #1237) stated four
unclaimed jewels. Mapping to executed state:

| Jewel (programme) | Executed status |
|---|---|
| Associator as positive graded rupture + functor F | Ord 1 instrumented; F as homology functor still open |
| Sedenion ZD = statistical invisibility | R2-partial + R2_FULL_MEASURED (measure layer); “invisibility” slogan not claimed |
| G₂ unifies semantic + epistemic faces | Algebraic faces only; morphodynamic face via R3/R4 operational paths |
| Positivization vs Sneed/Abramsky/Amari/Thom negatives | \(\det L_x\), associator, \(d_{\mathrm{sing}}\) are the positive objects |

**Divergence with Petitot** (`petitot-semantic-potential.md`) remains a feature:
the octonion Booleanises an *isolated* square and obstructs at the *field* —
exactly why R4 exists.

---

## 7. What to cite when making claims

| If you want to say… | Cite / run |
|---|---|
| ZD locus + 7 fibers, exact | `sedenion_zd_fibers` gates + R2 partial |
| Exact measure on annihilating slice | Frente A tests / R2 partial |
| Continuous tube / order-4 contact | R2 full probe (`R2_FULL_MEASURED`) |
| Associator jet near Fano + path classes | R3 probe (`R3_GREEN`) |
| Field obstruction + multi-line paths | R4 probe (`R4_GREEN`) |
| “Same singularity as Thom” | **D0–D2 only** — not D3 |
| Anything clinical / ECSS | separate declaration; not this gate |

---

## 8. Open edges (honest queue)

1. **Functor F** — homology of meaning as a formal functor, not only path classes.  
2. **Ord 2″** — subspace-alignment probe on **non-sedenion** models (LSTM/S4), with rotating control.  
3. **R2 continuous law as theorem** — lift measured \(t^{1/4}\) contact to a proof.  
4. **#1237 merge / ADE–Wildgen** — literature map vs catastrophe-controversy landmine.  
5. **External paper** — this synthesis is the skeleton; prose for arXiv still needs human voice and offload review under repo policy.

---

## 9. Reproduce everything

```bash
# full ladder
bash scripts/ci/rupture_abcd_contracts_gate.sh

# individuals
python3 scripts/research/rupture_r2_fiber_measure_contract.py
python3 scripts/research/rupture_r2_full_tubular_probe.py
python3 scripts/research/rupture_r3_fano_restriction_probe.py
python3 scripts/research/rupture_r4_fano_field_contract.py
```

Merged landings: **#1432** (R3_GREEN + R4_GREEN), **#1446** (R2_FULL_MEASURED).

---

## 10. AI disclosure

Programme instrumentation and this synthesis drafted under human direction
(2026-07-24/25). Math-facing claims remain bounded by the named gates and the
D0–D3 discipline. No clinical or patient-level claim. GAIDeT-ICMJE 2025.
