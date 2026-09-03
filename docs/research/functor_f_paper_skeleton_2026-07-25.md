<!-- docs:meta
topic_id: repo.docs.research.functor-f-paper-skeleton-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-paper-skeleton-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — paper skeleton (submission-ready outline, not prose)

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** SKELETON — section map + bounded claims + reproducibility, no narrative prose
**Consolidates:** the Functor-F programme, `docs/research/functor_f_*_spec_2026-07-25.md`
(twelve rungs) and `docs/research/rupture-programme-synthesis_2026-07-25.md`
**Does not supersede** `docs/papers/rupture_functor_f_synthesis_2026-07-25.md` (a broader
draft covering Mercyful Learning and the Falsification Ledger, which are out of scope
here); this skeleton is the narrower, Functor-F-only submission plan.

---

## 0. How to read this document

This is an **outline**, not a paper. Every claim below is a one-line gloss of a
theorem or measurement whose full statement lives in the cited spec, and whose
truth value is whatever its named executable contract returns — not what this
skeleton says about it. Where a verdict is a characterised **obstruction**
(`H_CHARACTERISED`, `K_CHARACTERISED`, `B_OBSTRUCTED`, `M_CHARACTERISED`), the
obstruction is the result, not a defect to explain away.

---

## 1. Title options

1. *Functor F: a G₂-Equivariant Bridge from the Octonion Associator to Morphodynamic
   Stratification — Closed Algebra, Located Obstructions, and One PSL(2,7) Threading the
   Order Tower*
2. *The Associator as a G₂-Equivariant Functor: an Executable Algebraic Column and Two
   Honest Cross-Column Failures*
3. *From Associator Jets to Cusp Strata: G₂-Covariance, Coherence, and the Limits of the
   Bridge to Petitot and Ollivier–Ricci Curvature*
4. *Instrumentation, Not Identity: Functor F and the Discipline of Bounded Claims in an
   Executable Rupture Programme*

Preferred for submission: **#1** (leads with the closed result, names the cross-column
findings as "located obstructions" rather than negatives, and closes on the positive
PSL(2,7) unification; avoids the D3-adjacent phrase "same singularity").

---

## 2. Abstract (target 150 words)

> We construct **Functor F**, a jet-functorial map from the octonion associator to the
> stratification homology of a cusp potential, and characterise it exactly. On a single
> Fano line the assignment is uniform, but its polar coordinate — read off by a
> coordinate `argmax` — is **not** G₂-covariant; we exhibit the obstruction under generic
> and continuous automorphisms and give a constructive fix, a natural pairing, restoring
> equivariance everywhere. Lifted to cross-line couplings, F is strictly additive exactly
> on associative sub-fields and picks up one covariant ord-1 correction — the associator
> itself — elsewhere; that correction is proved to be the G₂ 3-form under its defining
> contraction identity, and the resulting tower of invariants `{δ,φ,ψ}` is proved to close
> at ord-2. Three cross-column probes follow: the bridges to Petitot's morphodynamics and
> to Ollivier–Ricci curvature are **canonically obstructed**, each for a located,
> mechanistic reason; the bridge to ord-3 (Massey/Borromean) is instead **located** — the
> secondary ternary operation is empty in the octonions (a division algebra) and, in the
> sedenions, coincides exactly with the ord-2 zero-divisor fibre, though its quotient
> admits no canonical fill without imposed A∞ structure. Finally, looking at the
> **symmetry** rather than the bare products, a **positive** cross-order result: one Fano
> plane and its automorphism group `PSL(2,7)` (the full 168, transitive) index all three
> layers — the `φ` 3-form's lines (ord-1) label the sedenion zero-divisor fibres, the `ψ`
> 4-form's coassociative planes (ord-2) are their supports, and the ord-3 operation lives
> there — so the algebraic tower and the ZD-orbit geometry are one Fano/`PSL(2,7)`
> structure. Twelve executable contracts bound every claim; no identity between algebra
> and semantics is asserted.

(≈185 words.)

---

## 3. Contributions — framed as INSTRUMENTATION + CLAIM DISCIPLINE

Per `rupture-programme-synthesis §1` ("what is NOT the contribution") and the **D3
FORBIDDEN** rule (Petitot's bifurcation set is never claimed to *be* the associator/ZD
locus), this paper's contribution is explicitly **not** a benchmark result and **not** an
identity theorem across columns. It is:

1. **An instrumented, G₂-equivariant functor** from associator jets to stratification
   data, with every non-obvious step (uniformity, covariance, coherence, closure)
   discharged by a named executable contract rather than asserted.
2. **A discipline for reporting obstructions as first-class results.** Four of the twelve
   rungs are *characterisations of failure* (`H_CHARACTERISED`, `K_CHARACTERISED`,
   `B_OBSTRUCTED`, `M_CHARACTERISED`), each naming its verdict type **before** computing,
   each locating the mechanism (a non-invariant coordinate; a non-associative coupling; a
   vanishing invariant; a symmetry that quotients away the algebra's oriented content) —
   not merely reporting "no bridge found".
3. **A closed algebraic column.** The chain `G_GREEN → Q_GREEN` proves the associator
   3-form `φ` and the co-associator 4-form `ψ` are the **only** algebraic rupture
   invariants the octonion core generates under contraction — the tower terminates at
   ord-2. This is a boundary result: it tells the programme where the algebraic sensor
   saturates and where it must hand off to a **functor**, not further contraction, to
   reach ord-M or ord-P (`rupture-programme-synthesis §3`, "non-collapse" rule).
4. **Three cross-column probes — two located obstructions and one located bridge.** The
   algebra reaches Petitot's cusp canonically but not the butterfly (`B_OBSTRUCTED`,
   sharpened to a proved vanishing by `PHI_JETS_VANISH_PROVEN`); the algebra's oriented
   content is invisible to Ollivier–Ricci curvature on the canonical Fano graph for a
   symmetry-forced reason (`M_CHARACTERISED`); and the ord-3 (Massey/Borromean) secondary
   ternary structure is **located** — empty in 𝕆, coinciding in 𝕊 with the ord-2
   zero-divisor fibre (`SECONDARY_TERNARY_LOCATED`), with a 2-dim quotient that the bare
   algebra cannot canonically fill (`NO_CANONICAL_FILL`, a recorded near-miss refuted by a
   bracketing-consistency test). The cross-column frontier is therefore **not uniformly
   negative**: ord-P and ord-M are coincidences, ord-3 genuinely attaches to ord-2.
5. **A positive cross-order unification via symmetry.** Looking at the symmetry rather
   than the bare products, one Fano plane and the full `PSL(2,7)` (168 collineations,
   transitive) index all three layers at once: the `φ` 3-form's 7 lines (ord-1) are the
   sedenion ZD fibre *labels*, the `ψ` 4-form's 7 coassociative planes (ord-2) are their
   *supports* (the two `φ`/`ψ` Hodge-dual), and the ord-3 secondary operation lives on
   those fibres — so the algebraic tower and the prior 168-orbit ZD geometry are one
   Fano/`PSL(2,7)` structure (`PSL27_THREADS_THE_TOWER`). The 7-fibre/168 datum is prior
   work; the `φ`/`ψ`-tower thread is the contribution. Shared indexing and symmetry, not
   an identity (D3 respected).

**What this is explicitly not**, restated from the synthesis: not a claim that
non-associative structure helps any learned model; not a clinical mapping; not a proof
that the continuous R2 tube law is a theorem; not a topos-non-Booleanisability result
beyond the cusp-plane path classes; not a self-hosting/compiler claim.

---

## 4. Section map (claim → executable gate)

| § | Working title | Claim (bounded) | Contract (harness) | Verdict token |
|---|---|---|---|---|
| 1 | Introduction | Positions Functor F as the answer to synthesis open-edge #1 ("Functor F — homology of meaning as a formal functor, not only path classes") | — (framing only) | — |
| 2 | Background: octonions, G₂, the Fano plane | States `Aut(𝕆)=G₂`, the CD sign law, PG(2,2) incidence, and the cusp potential `V(x;a,b)` inherited from R3/R4 | `rupture_r3_fano_restriction_probe.py`, `rupture_r4_fano_field_contract.py` | `R3_GREEN`, `R4_GREEN` |
| 3 | Functor F on a single line: uniformity | For every Fano line and off-line unit, the jet is single-axis, `‖α‖=2`, and Φ_fp/path-class/Betti behaviour is uniform | `functor_f_g2_covariance_contract.py` | `FUNCTOR_F_G2_VERDICT G_GREEN` (6/6) |
| 4 | The covariance obstruction and its fix | Coordinate `argmax`-`b` is **not** G₂-covariant under generic automorphisms (100% of 200 sampled `g` move it); the pairing `b_cov=⟨α,e_m⟩` is | `functor_f_g2_equivariance_contract.py` | `FUNCTOR_F_G2_EQUIV_VERDICT H_CHARACTERISED` (4/4) |
| 5 | Equivariance over a continuous orbit | Under an explicit one-parameter `exp(t·𝔤₂)` subgroup, `b_cov` and the induced path class are invariant; the naive `argmax` path can **flip poles** mid-orbit | `functor_f_phi_fp_equivariant_contract.py` | `FUNCTOR_F_PHI_EQUIV_VERDICT E_GREEN` (4/4) |
| 6 | Functoriality on morphisms: the field of couplings | F is strictly additive on cross-line couplings **iff** the coupling stays inside an associative (Fano) sub-algebra (14/42 configs); the defect on the remaining 28 is exactly the cross associator, ord-1, G₂-covariant | `functor_f_field_functoriality_contract.py` | `FUNCTOR_F_FIELD_VERDICT K_CHARACTERISED` (4/4) |
| 7 | Coherence of the ord-1 correction | The correction of §6 **is** the G₂ 3-form `φ`, and `φ` obeys the exact contraction identity `φ·φ = δδ−δδ−ψ` (all `7⁴` tuples, worst deviation `0.0`) | `functor_f_g2_coherence_contract.py` | `FUNCTOR_F_COHERENCE_VERDICT P_GREEN` (5/5) |
| 8 | Closure of the algebraic tower | Every pairwise contraction of `{φ,ψ}` returns a member of `{δ,φ,ψ}` (four exact identities, integer coefficients `24,4,−2,−4`); no ord-3 invariant is generated | `functor_f_g2_tower_closure_contract.py` | `FUNCTOR_F_TOWER_VERDICT Q_GREEN` (5/5) |
| 9 | Cross-column I: the algebra → Petitot bridge | The cusp (2 canonical G₂-invariants: depth+tilt) closes canonically; the butterfly's third control does not — the antisymmetric cubic invariant `φ(α₁,α₂,α₃)` vanishes on the coupling jets (measured over 840 configs) | `functor_f_petitot_bridge_contract.py` | `FUNCTOR_F_PETITOT_VERDICT B_OBSTRUCTED` (5/5) |
| 10 | The vanishing, proved | `φ(a₁,a₂,a₃)=0` is upgraded from measurement to a structural theorem: single-axis associator + PG(2,2) incidence lemma (no Fano line lies in a 4-point complement) + trilinearity | `functor_f_phi_jets_vanish_contract.py` | `FUNCTOR_F_PHI_JETS_VERDICT PHI_JETS_VANISH_PROVEN` (6/6) |
| 11 | Cross-column II: the algebra ↔ ord-M (Ollivier–Ricci) bridge | On the canonical Fano-incidence graphs (K₇; Heawood), ORC is a single edge-transitivity-forced scalar with **zero degrees of freedom**; 16 distinct signed octonion tables (16 distinct `φ`) all induce the identical graph and identical curvature — the bridge is a symmetry/dimension coincidence, not a map | `functor_f_orc_fano_bridge_contract.py` | `FUNCTOR_F_ORC_VERDICT M_CHARACTERISED` (6/6) |
| 12 | Cross-column III: the algebra ↔ ord-3 (Massey/Borromean) | The secondary ternary operation (defined where the primary associator vanishes) is **empty in 𝕆** (division algebra — the algebraic reason the octonion associator is not a Massey object) and, in 𝕊, **is** the ord-2 zero-divisor fibre: 42 ZD with `ker L_b=ker R_b=`4-dim (the merged `seam_coincidence` `lo⊕hi` fibre); indeterminacy `14` vs generic `16` gives a distinguishable 2-dim quotient; no Borromean triple exists | `functor_f_ord3_secondary_ternary_contract.py` | `FUNCTOR_F_ORD3_VERDICT SECONDARY_TERNARY_LOCATED` (6/6) |
| 13 | Can the ord-3 quotient be filled? | The 2-dim quotient is **reachable** by intrinsic ternary composites but **not canonically fillable**: the four bracketings/orderings span the full quotient (rank 2, all 42 ZD), so the value is bracketing-selected not forced; 𝕊 has no differential, so classical Massey cannot pick a representative. An initial `(a·c)·b` "canonical invariant" is refuted by a bracketing-consistency test (recorded near-miss) | `functor_f_ord3_quotient_fill_contract.py` | `FUNCTOR_F_ORD3FILL_VERDICT NO_CANONICAL_FILL` (6/6) |
| 14 | The Fano/PSL(2,7) thread (positive) | One Fano plane + the full `PSL(2,7)` (168, transitive) index all three layers: `φ` lines (ord-1) = ZD fibre labels; `ψ` coassociative 4-planes (ord-2) = ZD fibre supports (`φ`/`ψ` dual); ord-3 op lives there. The algebraic tower and the prior 168-orbit ZD geometry are one Fano/`PSL(2,7)` structure (7-fibre/168 is prior; the `φ`/`ψ` thread is new) | `functor_f_fano_psl27_thread_contract.py` | `FUNCTOR_F_FANO_VERDICT PSL27_THREADS_THE_TOWER` (7/7) |
| 15 | Discussion | The algebraic column is closed (§3–8); the three cross-column probes (§9–13) return **located, mechanistic** results — two coincidence-obstructions (ord-P, ord-M) and one attachment of ord-3 to ord-2 — and §14 turns the symmetry into a **positive** unification of the whole tower; the frontier is not uniformly negative | synthesis §5 architecture diagram | — |
| 16 | What is not claimed | See §5 below | — | — |
| 17 | Related work | See §6 below | — | — |
| App. A | Reproducibility | See §7 below | all twelve contracts | all PASS |

---

## 5. What is not claimed (explicit)

- **Not D3.** No identity "Petitot's bifurcation set ≡ the associator/ZD locus" is
  claimed anywhere in this arc (§9, §11 are the two places this could be smuggled in,
  and both are explicitly typed *operational*, named before computing).
- **Not a construction of G₂ or Aut(𝕆).** The programme samples verified generic and
  continuous automorphisms (§4, §5); it never enumerates or constructs the group itself.
- **Not the Mac Lane pentagon.** §7's identity is the G₂ contraction identity on the
  3-form/4-form pair, not a monoidal coherence pentagon — the octonions carry no
  4-fold monoidal structure for which a literal pentagon is well-posed.
- **Not a completeness theorem for algebraic invariants.** §9–10's obstruction needs
  only the domain-bounded rank (`≤3` for a 3-coupling field, since rank is bounded by
  the number of continuous DOF) and the measured/proved vanishing of `φ(jets)`; it does
  not claim `{δ,φ,ψ}` (or any finite set) exhausts every conceivable invariant of the
  coupling jets.
- **Not a claim that no algebra→Petitot or algebra→ORC bridge can ever exist.** §9 and
  §11 rule out the *canonical* construction each probe tested (the natural cubic
  invariant; the unweighted incidence graph and any canonically-uniform weighting of
  it) — not every conceivable construction. Orientation-sensitive / signed variants are
  named as open next edges, not attempted here.
- **Not a claim of a canonical ord-3 invariant.** §12–13 locate the secondary ternary
  operation on the ord-2 ZD fibre and show its 2-dim quotient is *reachable* but
  *bracketing-selected*, not canonically filled; a genuine ord-3 invariant would require
  *imposing* A∞/differential structure (the open positive follow-up), which is not done
  here. The `(a·c)·b` near-miss is reported and refuted, not claimed.
- **Not the (topological) Massey product.** §12–13's object is a secondary ternary
  operation on a plain (differential-free) algebra; only Massey's *definedness pattern*
  (secondary lives where primary vanishes) transfers. The empirical path-Massey negative
  in `BORROMEAN_AINFINITY.md` is a *different* argument, cited but not leaned on.
- **Not a claim about non-algebraic orders in general.** §8's closure result is scoped
  to the **algebraic column** only (ord-1 `φ`, ord-2 `ψ`); it says nothing about ord-M
  (curvature) or ord-P (bifurcation) as sensors in their own right — only that the
  algebra cannot supply a third *algebraic* invariant to hand them.
- **Not a claim that F "fails" at the field level (§6).** F is additive exactly on
  associative couplings and picks up one controlled, covariant, ord-1 defect elsewhere
  — this is a characterisation, not a refutation of functoriality.
- **Not a clinical, benchmark, or trained-model claim of any kind.** Consistent with
  `rupture-programme-synthesis §1`.
- **Not a claim that this skeleton's base rung (jet-functoriality, provisionally
  `F_GREEN`) has been re-verified in this checkout.** See §7.0 below — it is cited as
  the documented parent of §3's contract but its own contract file is not present in
  this branch's working tree at the time of writing.

---

## 6. Related work (pointers only)

- **Bryant, R.** *Some remarks on G₂-structures* — the 3-form/4-form (`φ`/`ψ`) pair and
  their contraction identities used verbatim in §7–8 (standard G₂ geometry, not
  rediscovered here; this paper's contribution is verifying them exactly on the repo's
  octonion core and drawing the rupture-programme consequence).
- **Karigiannis, S.** *Flows of G2 structures* and related survey material — same
  contraction-identity normalisation family as §7–8; cite for the standard derivation
  this paper's `P3`/`Q1`–`Q4` clauses reproduce.
- **Petitot, J.** *Morphogenèse du Sens* — the cusp/butterfly catastrophe-theoretic
  semantic model that Φ_fp (R3/R4) targets and that §9–10 test the algebra against;
  the divergence recorded at D3 (`rupture-programme-synthesis §1`, `petitot-semantic-
  potential.md §3`) is a load-bearing prior result, not a new finding of this paper —
  §9 sharpens *where* the divergence is located (butterfly, not cusp).
- **Reggiani, S.** (2024) — continuous G₂ acting transitively on a sedenion
  zero-divisor manifold; **published, distinct** from this repo's own separate finding
  (frozen-168 permutation action on discrete ZD fibers, see repo memory
  `cd-tower-168-acts-on-zd-fibers`). Cited here only to bound the scope of "G₂ acting
  on CD-tower structures" as a research area with a prior, independent published
  result — this paper's G₂-equivariance rungs (§3–5) act on **associator jets**, a
  different object, and make no claim of overlap or priority dispute with Reggiani.
- **Koebisu, S.** (arXiv:2512.13002) — sedenion `det L_x` factorisation underlying the
  R2 tube-law measurement that motivates the ord-2 sensor in `rupture-programme-
  synthesis §3`; not itself a Functor-F result, cited for the ord-2 column this paper's
  §8 closure result is contrasted against.
- **Ollivier, Y.; Lin–Lu–Yau** — Ollivier–Ricci and Lin–Lu–Yau curvature, the ord-M
  sensor targeted (and found symmetry-obstructed) in §11.

---

## 7. Reproducibility appendix

### 7.0 Honest gap: the base rung

`functor_f_g2_covariance_spec_2026-07-25.md` (§3's contract) names
`functor_f_jet_functoriality_spec_2026-07-25.md` (verdict `F_GREEN`, 7/7 clauses) as its
parent — the base jet-functorial witness the whole ladder builds on. That spec and its
contract (`functor_f_jet_functoriality_contract.py`, gate
`functor_f_jet_functoriality_gate.sh`) are **not present in this branch's working tree**
at the time of writing this skeleton; they exist on a sibling research branch
(`research/rupture-ord2-alignment-20260725`, commit `6693894dc`, "Functor F
jet-functoriality witness (F_GREEN 7/7)") that has not been merged here. This skeleton
therefore cites `F_GREEN` as the documented antecedent of `G_GREEN` but does **not**
report a freshly re-run verdict for it — only the twelve contracts in §7.1, all physically
present and re-executed in this session, are load-bearing for this document's claims.
Before submission, either merge that branch's rung or re-derive it on this branch and
re-verify.

### 7.1 The twelve executable contracts (all re-run live, this session)

| Contract (`scripts/research/`) | Verdict token | Clauses | Exit | CI gate |
|---|---|---|---|---|
| `functor_f_g2_covariance_contract.py` | `FUNCTOR_F_G2_VERDICT G_GREEN` | 6/6 PASS | 0 | `scripts/ci/functor_f_g2_covariance_gate.sh` |
| `functor_f_g2_equivariance_contract.py` | `FUNCTOR_F_G2_EQUIV_VERDICT H_CHARACTERISED` | 4/4 PASS | 0 | — (direct invocation) |
| `functor_f_phi_fp_equivariant_contract.py` | `FUNCTOR_F_PHI_EQUIV_VERDICT E_GREEN` | 4/4 PASS | 0 | — |
| `functor_f_field_functoriality_contract.py` | `FUNCTOR_F_FIELD_VERDICT K_CHARACTERISED` | 4/4 PASS | 0 | — |
| `functor_f_g2_coherence_contract.py` | `FUNCTOR_F_COHERENCE_VERDICT P_GREEN` | 5/5 PASS | 0 | — |
| `functor_f_g2_tower_closure_contract.py` | `FUNCTOR_F_TOWER_VERDICT Q_GREEN` | 5/5 PASS | 0 | — |
| `functor_f_petitot_bridge_contract.py` | `FUNCTOR_F_PETITOT_VERDICT B_OBSTRUCTED` | 5/5 PASS | 0 | — |
| `functor_f_phi_jets_vanish_contract.py` | `FUNCTOR_F_PHI_JETS_VERDICT PHI_JETS_VANISH_PROVEN` | 6/6 PASS | 0 | — |
| `functor_f_orc_fano_bridge_contract.py` | `FUNCTOR_F_ORC_VERDICT M_CHARACTERISED` | 6/6 PASS | 0 | — |
| `functor_f_ord3_secondary_ternary_contract.py` | `FUNCTOR_F_ORD3_VERDICT SECONDARY_TERNARY_LOCATED` | 6/6 PASS | 0 | — |
| `functor_f_ord3_quotient_fill_contract.py` | `FUNCTOR_F_ORD3FILL_VERDICT NO_CANONICAL_FILL` | 6/6 PASS | 0 | — |
| `functor_f_fano_psl27_thread_contract.py` | `FUNCTOR_F_FANO_VERDICT PSL27_THREADS_THE_TOWER` | 7/7 PASS | 0 | — |

All twelve are pure-Python (numpy), self-contained, and independently re-audit the
inherited octonion core (a `Q0`/`M0`/`B0`/`C0`/`P0`-style axiom check) before using it —
each contract distrusts the shared core rather than assuming it.

```bash
for f in functor_f_g2_covariance functor_f_g2_equivariance functor_f_phi_fp_equivariant \
         functor_f_field_functoriality functor_f_g2_coherence functor_f_g2_tower_closure \
         functor_f_petitot_bridge functor_f_phi_jets_vanish functor_f_orc_fano_bridge \
         functor_f_ord3_secondary_ternary functor_f_ord3_quotient_fill; do
  python3 scripts/research/${f}_contract.py || echo "FAIL: $f"
done
```

### 7.2 Independent (non-authorial) verification summaries — folded in honestly

Two rungs received an independent adversarial-verification pass (separate
reimplementation, different RNG/algebra convention, attempted refutation). **Neither
was refuted.** Their own recorded caveats are kept, not smoothed over:

**`PHI_JETS_VANISH_PROVEN` (φ(jets)=0 on the octonion core) — two independent passes,
both `CONFIRMED`, high confidence.**
- Independent recursive Cayley–Dickson doubling (conjugate-convention) reproduced every
  decisive number: single-axis jets of magnitude exactly 2, 168 Fano-plane configs, 0/168
  XOR collapses, 0/168 axis-triples on a Fano line.
- A second, fully independent reimplementation used an unrelated (Baez-convention)
  explicit sign table in **exact rational arithmetic** and reproduced `φ(jets)=0`
  *exactly* (not a floating-point artifact) over 840 configurations.
- Both passes confirmed a deliberate D3-probe inside the contract: generic (non-jet)
  associator triples **do** give `φ≠0` within 20k random samples — i.e. the vanishing is
  a genuine, non-vacuous fact about this specific jet shape, not a trivial identity, and
  no "associator ≡ X" identity is smuggled in.
- Scope caveat (already stated in the spec, independently confirmed accurate): the proof
  is octonion-core-specific (`bits=3`); no `∀n` claim; does not prove `x⁴`
  non-existence; the honest-negative framing for the Petitot bridge is accurate.

**`M_CHARACTERISED` (algebra ↔ ord-M / Ollivier–Ricci obstruction) — two independent
passes, both `CONFIRMED`, high confidence.**
- Independent `scipy.linprog` optimal-transport solver (vs. the contract's in-file
  min-cost-flow) and an independently built Fano-line/graph construction reproduced
  every exact rational: `K₇` `LLY=7/6`, `OR=5/6`; Heawood `LLY=OR=−2/3`; `K₈=8/7`;
  Möbius–Kantor `GP(8,3)=−2/3` (zero-octonion control); tree edge `=−2(d−2)/d`; the
  7-on-line/28-off-line associator split with magnitudes exactly `{0,2}`; the
  16-of-128 composition-algebra count with 16 distinct signed `φ` collapsing to one
  graph.
- One reviewer's own naive LP at `p→1−1e-7` initially disagreed (Heawood `LLY=4/3`
  instead of `−2/3`) — traced to float amplification in their method, not a defect in
  the contract, which sidesteps it with exact-rational min-cost-flow and an
  `M=200 == M=400` exactness guard.
- Caveat both reviewers flagged and neither treated as fatal: the obstruction is *partly
  by construction* (the unweighted incidence graph discards orientation before curvature
  is computed), and edge-transitivity ⇒ constant-ORC is *measured and asserted*, not
  proved from `Aut(K₇)=S₇`. Both are already named as open next edges in the spec's §6/§7,
  not hidden.
- No overclaim found in either pass: "coincidence" framing is treated as demonstrated
  (not merely asserted) via the octonion-free control graph, and D3 is respected
  throughout (no identity claimed).

**Net effect on this skeleton's claims:** none of the load-bearing numbers in §9–11
change; the independent passes raise confidence in `PHI_JETS_VANISH_PROVEN` and
`M_CHARACTERISED` specifically, and both caveats they raised (orientation-blindness
partly built into the construction; edge-transitivity-forces-constant not yet a proved
theorem) are already carried as named open edges rather than resolved claims. No
`REFUTED` or `OVERCLAIM` verdict was returned by either pass; had one been, it would be
listed here rather than in this document's prose.

### 7.3 One-command full-ladder check (proposed, not yet wired)

No single gate currently runs all twelve Functor-F contracts together (only
`functor_f_g2_covariance_gate.sh` exists as a dedicated CI gate; the rest are verified by
direct `python3` invocation grepping their verdict token, as in §7.1). A follow-up before
submission should add a `functor_f_full_ladder_gate.sh` analogous to
`rupture_abcd_contracts_gate.sh`, so the whole `G→Q, B, PHI_JETS, M, ord-3, PSL27` arc is one command.
