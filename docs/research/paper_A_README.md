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
proved-disjoint certificate). Kernel-checked core; running prototypes; specified compiler wire.

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
| NS carrier + sound add runs (`x+x`→4, naive→2) | `docs/research/sounio/noise_symbols.sio` | souc-green |
| NS dataflow flags shared-source add | `docs/research/sounio/ns_dataflow.sio` | souc-green |
| five acceptance controls incl. sabotage causality | `docs/research/sounio/ns_contract.sio` | souc-green (5/5 PASS) |
| correlated operator (escape valve) | `stdlib/epistemic/gum_supplement1.sio` (`gum_s1_add_correlated`) | in-tree, orphaned |
| clinical WARN (AUC=450, CI[362,538], boundary 400) | `examples/vancomycin_auc_epistemic.sio` | run-pass |
| 2-compartment (shared-source sum) is future | `stdlib/clinical/vancomycin_pbpk.sio:49,52` | stubbed |
| compiler wire (E230, `noise_sets.sio`, N1–N4) | synthesis §26 | authorized, **pending** |

`[pending wire]` in §8/§7: corpus false-positive rate (RQ3), compiler-level sabotage witness
(RQ2), full two-compartment flip rate (RQ4). Do not report these as measured until N3–N4 land.

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

1. ~~**§1 intro prose**~~ — ✅ done (`paper_A_section1_draft_2026-08-25.md`). Every section is
   now full prose; the draft is content-complete.
2. **Merge** the section files into one document in the order above; unify notation
   (`m, v, Cov, ⟨·,·⟩`, `Knowledge⟨T, N⟩`) — already consistent across drafts.
3. **Land the wire (N1–N4)** to convert the `[pending wire]` evaluation items into measured
   numbers — the single biggest strengthener; coordinate per synthesis §26.
4. **Figures:** (a) the two-formula defect (§2.1); (b) the noise-set dataflow on `x+x` vs
   `x+y` (§5.3/§8.2); (c) the vancomycin CI crossing the 400 boundary and its contraction
   under understatement (§8.4).
5. **Prior-art gate sign-off** — the narrow novelty claim is CANDIDATE pending an explicit
   adversarial prior-art pass beyond the two gates run here (per synthesis §24/§25).
