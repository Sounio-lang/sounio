<!-- docs:meta
topic_id: repo.docs.research.paper-a-readme
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-readme
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — assembly README

**Working title:** *Manufacturing Precision Is a Type Error: Compile-Time Anti-Garbling
for Uncertainty-Typed Languages*
**Target:** PLDI / OOPSLA (bug-class + type-discipline shape); ECOOP fallback. Artifact-evaluable.
**Status (2026-08-25):** full-prose draft of every section except §1 (abstract done, intro
is skeleton); ready for a prose intro pass and a single-file merge.

**One-line thesis.** Uncertainty libraries assume operand independence and silently
understate variance when it fails; reading this through the Blackwell/QIF order makes it an
*anti-garbling* (manufacturing information), and carrying each value's noise-symbol
source-set in the type turns the independence assumption into a checked precondition (E230 +
proved-disjoint certificate). Kernel-checked core; running prototypes; **wired + source-verified compiler** (Madaros v0.80.0, integration commit `4ac63da51f`). Closed draft: `paper_A_MERGED_2026-08-25.md`.

---

## Section order → source file → status

| § | Title | File | Status |
|---|---|---|---|
| — | Abstract | `paper_A_antigarbling_skeleton_2026-08-25.md` | ✅ done (~200w) |
| 1 | Introduction | `paper_A_section1_draft_2026-08-25.md` | ✅ full prose (skeleton file retains the original bullets/outline) |
| 2 | The defect class, by example | `paper_A_section2_draft_2026-08-25.md` | ✅ full prose |
| 3 | Preliminaries | `paper_A_sections3_10_11_draft_2026-08-25.md` | ✅ full prose |
| 4 | Anti-garbling as the soundness criterion | `paper_A_sections4_5_draft_2026-08-25.md` | ✅ full prose |
| 5 | The type system | `paper_A_sections4_5_draft_2026-08-25.md` | ✅ full prose |
| 6 | Metatheory | `paper_A_section6_draft_2026-08-25.md` | ✅ full prose |
| 7 | Implementation | `paper_A_sections7_9_draft_2026-08-25.md` | ✅ full prose |
| 8 | Evaluation | `paper_A_section8_draft_2026-08-25.md` | ✅ full prose |
| 9 | Related work | `paper_A_sections7_9_draft_2026-08-25.md` | ✅ full prose |
| 10 | Limitations | `paper_A_sections3_10_11_draft_2026-08-25.md` | ✅ full prose |
| 11 | Conclusion | `paper_A_sections3_10_11_draft_2026-08-25.md` | ✅ full prose |

**Context (not part of the paper):** `publishable_novelty_assessment_2026-08-25.md` — the
prior-art-ranked assessment that scoped this paper (and papers B, C); records the two
adversarial gates.

---

## Grounding index (what backs each claim; all reproduced 2026-08-25)

| Claim | Artifact | Status |
|---|---|---|
| Defect is real (`ep_mul(x,x)=2m²v` vs `ep_square=4m²v`) | `stdlib/epistemic/knowledge.sio:112,154` | source |
| add/sub asymmetry (understate vs conservative) | `knowledge.sio:96,105` | source |
| tests use independent operands only | `knowledge.sio:290–310` | source |
| naive add sound ⟺ zero-cov; gap = 2·cov (Lemma 1) | `docs/research/lean/SounioAntiGarblingModel.lean` | kernel-checked, axiom-free |
| base calculus progress + preservation; `gAddMeta`/`gMulMeta` = the §2 ops; validity preserved | `formal/lean4/EpistemicEffectsV2.lean` (`:92,94,223,324,559,626`) | mechanized (Lean 4.33.1) |
| **NS-extended calculus**: Lemma 1 general form, Lemma 2 (`Covers`), NS progress + preservation, `exact_preservation`, Theorem 6.4 (`typed_agfree`, `soundness_star`), x+x sabotage witness | `formal/lean4/EpistemicEffectsNS.lean` (gate `scripts/ci/ns_metatheory_lean_gate.sh`) | mechanized 2026-08-30 (Lean 4.33.1, Mathlib-free, no sorry) |
| NS carrier + sound add runs (`x+x`→4, naive→2) | `docs/research/sounio/noise_symbols.sio` | souc-green |
| NS dataflow flags shared-source add | `docs/research/sounio/ns_dataflow.sio` | souc-green |
| five acceptance controls incl. sabotage causality | `docs/research/sounio/ns_contract.sio` | souc-green (5/5 PASS) |
| correlated operator (escape valve) | `stdlib/epistemic/gum_supplement1.sio` (`gum_s1_add_correlated`) | in-tree, orphaned |
| clinical WARN (AUC=450, CI[362,538], boundary 400) | `examples/vancomycin_auc_epistemic.sio` (lean_single; Madaros fails closed on builtin Knowledge arithmetic, #1706) | run-pass |
| same receipt, engine-portable (Madaros + lean_single): CrCl 52.08±5.26, CL 2.22±0.22, AUC 450.07±44.44, CI [361.2, 539.0], WARN | `examples/vancomycin_auc_affine.sio` (`stdlib/epistemic/affine`) | run-pass on both engines, 2026-08-31 |
| **RQ4 two-compartment flip rate**: interval sum ρ=1 silences 311/909 true WARNs (34.2 %), Var ratio 0.500; phase decomposition Cov<0 silences 0, chain over-states 300× (1,894 spurious) | `docs/research/sounio/rq4_vanco_two_compartment_flip.sio` (5,000-patient deterministic cohort, Madaros v0.80.0) | **measured 2026-08-31**, deterministic, identity check error 0 |
| **RQ4 Monte Carlo adequacy**: Var_MC/Var_T = 0.999 (0.857–1.158); WARN agreement 99.4 % (±2σ rule), 94.8 % vs true quantile; Var_N/Var_MC = 300.9 | `docs/research/sounio/rq4_vanco_mc_adequacy.sio` (5 000 × 1 000 draws; Madaros + lean_single byte-identical) | **measured 2026-08-31** |
| **Partition lemma**: invariance of a sum to its shared sources ⟹ `Cov ≤ 0` ⟹ naive add conservative; full partition ⟹ `Cov = −Var a` | `formal/lean4/EpistemicEffectsNS.lean` (`inner_nonpos_of_partition` …; gate 16 theorems) | mechanized 2026-08-31 |
| 2-compartment (shared-source sum) is future | `stdlib/clinical/vancomycin_pbpk.sio:49,52` | stubbed |
| compiler wire (E230, `noise_sets.sio`, N1–N4) | integration commit `4ac63da51f` (base `06e85a6ada`) | **landed + source-verified** (Madaros v0.80.0); four controls + both gates green; xai+zai reviewed |

Now **measured on the wired compiler**: the corpus false-positive rate (RQ3, 6/95, all
characterized) and the compiler-level sabotage witness (RQ2, `ns_antigarbling_gate.sh`).
Still genuinely `[pending]`: interprocedural parameter projection (§10). RQ4's two-compartment
flip rate was measured 2026-08-31 (`paper_A_rq4_two_compartment_flip_2026-08-31.md`).

---

## Prior-art citations (cite in §1 and §9)

- Comba & Stolfi 1993 — affine arithmetic (noise symbols).
- Goubault & Putot — *Static Analysis of Finite Precision Computations*, VMCAI 2011;
  *Perturbed affine arithmetic for invariant computation*, 2008 (arXiv:0807.2961); zonotope
  intersection, 2010 (arXiv:1002.2236). Fluctuat.
- Blackwell 1953 — comparison of experiments (informativeness order).
- McIver, Morgan, Smith, Espinoza & Meinicke — *Abstract channels and their robust
  information-leakage ordering*, POST 2014.
- Alvim, Chatzikokolakis, McIver, Morgan, Palamidessi & Smith — *The Science of Quantitative
  Information Flow*, Springer 2020.
- Bornholt, Mytkowicz & McKinley — *Uncertain⟨T⟩*, ASPLOS 2014.
- Giordano — *Measurements.jl*. ISO/IEC 98-3 (GUM). Ferson — p-boxes.
- NumFuzz (2024, arXiv:2405.04612); Bean (2025, arXiv:2501.14550); type-based rounding-error
  analysis (2025, arXiv:2501.14598).

---

## Remaining before submission

1. ~~**§1 intro prose**~~ — ✅ done (`paper_A_section1_draft_2026-08-25.md`). Content-complete.
2. ~~**Merge**~~ — ✅ done: `paper_A_MERGED_2026-08-25.md` (single file, 11 sections + abstract
   in order, 1114 lines; notation unified `m, v, Cov, ⟨·,·⟩, Knowledge⟨T,N⟩`).
3. ~~**Land the wire (N1–N4)**~~ — ✅ **done + source-verified.** N1–N3 wired into the checker
   (E230 gate, `noise_sets.sio`, dataflow, sabotage knob), built from source (Madaros v0.80.0),
   xai+zai math-reviewed (round 1 caught 3, round 2 clean). Integration commit `4ac63da51f` on
   base `06e85a6ada` (branch `fable/ns-antigarbling-integration-20260825`), awaiting codex's
   line-by-line merge review. §8's RQ2/RQ3 now carry the measured wired-compiler numbers.
   Excluded as separate slices: interprocedural arg→param projection, the completeness proofs.
4. ~~**Figures**~~ — ✅ done, `figures/paper_A/`: `fig1_two_formula_defect.svg` (§2.1),
   `fig2_noiseset_dataflow.svg` (§5.3/§8.2), `fig3_vancomycin_warn.svg` (§8.4).
5. ~~**Prior-art gate sign-off**~~ — ✅ done: `paper_A_priorart_gate_signoff_2026-08-25.md`.
   Three gates (affine arithmetic; QIF/Blackwell; a deep adversarial pass) — narrow claim
   **survives in the intersection form**, assertable with attribution; residual (patents not
   exhaustively searched) stated.
