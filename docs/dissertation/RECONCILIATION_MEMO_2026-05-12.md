<!-- docs:meta
topic_id: repo.docs.dissertation.reconciliation-memo-2026-05-12
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.reconciliation-memo-2026-05-12
-->

> **⚠️ SUPERSEDED — 2026-05-21.** Operator directive: the PBPK28 biomaterials track and the
> broader Sounio epistemic-computing thesis are **one dissertation**, not the two-lane split
> recorded below. The "binding authority" finding in this memo (that the September–October
> defense is the biomaterials master's *and* that the Sounio epistemic thesis is a separate,
> longer-horizon project) no longer governs. The audit (`audit/gap_report.json`,
> `audit/discovery_log.md`) and `pbpk_claim_truth_table.md` have been re-framed to one unified
> evidence surface. This memo is retained verbatim **for history only** — do not cite the
> two-track split as current policy.

# Dissertation Architecture Reconciliation

**Date:** 2026-05-12  
**Branch:** claude/refine-local-plan-KAgIS  
**Commit:** 8dff1ed988bcdaf61bc83e8fc4dda44bb9db7f0a  
**Auditor:** Claude Code  
**Human answer (2026-05-12):** PUC-SP — the September–October defense is the biomaterials master's.  
**Resolved action:** §§4.1–4.9 are final. §4.10 (Sobol/Cut-HDMR) is the next deliverable.
The Sounio epistemic-computing thesis is a separate, longer-horizon project.

---

## 1. Verdict

**Verdict: (C) — Two distinct dissertations sharing infrastructure.**

The repository contains two architecturally separate PBPK kernels (`PBPKState14` / `PBPKState28`) that are *not* terminology drift from each other — they differ in state dimensionality, thermodynamic model class, and drug coverage. Semaglutide exists exclusively in the PBPK28 path; the claim truth table (Artefact B's binding authority) has no knowledge of PBPK28 or semaglutide and was dated before the PBPK28 code landed. The drug-numbering convention in the commit messages (#3=tirzepatide, #4=vancomycin) is incompatible with the rapamycin+semaglutide two-drug spine of Artefact A. The two systems share stdlib infrastructure (Tsit5 solver patterns, GUM propagation, Lean 4 obligations) but target different theses.

---

## 2. Evidence Inventory

### 2.1 PBPK Kernel State Count

**Two kernels coexist in the repo. They are NOT the same model.**

**PBPK14 — well-stirred, 14 scalar states:**

```
stdlib/darwin_pbpk/tsit5_pbpk14.sio:119
pub struct PBPKState14 {
    blood: f64, liver: f64, kidney: f64, brain: f64,
    heart: f64, lung: f64, muscle: f64, adipose: f64,
    gut: f64, skin: f64, bone: f64, spleen: f64,
    pancreas: f64, other: f64
}
```

One concentration per organ (well-stirred model). Used by: rapamycin epistemic path,
tacrolimus, haloperidol, vancomycin, olanzapine.

**PBPK28 — permeability-limited, 28 states (14 × {C_v, C_t}):**

```
stdlib/darwin_pbpk/core/pbpk28_params.sio:26
struct PBPKState28 {
    cv: [f64; 14],   // vascular concentration per organ
    ct: [f64; 14],   // tissue/interstitial concentration per organ
}
```

Two concentrations per organ coupled by a PS (permeability-surface-area) product.  
Used by: rapamycin (Cypher DES chapter §4), semaglutide (SUSTAIN).  
Documentation in the same file: *"14 organs × {vascular C_v, interstitial C_t} with a
permeability-surface-area product PS coupling."*

These are different architectural choices: PBPK14 assumes instantaneous vascular-tissue
equilibrium; PBPK28 adds a transport rate-limiting layer. The claim truth table in
`docs/dissertation/pbpk_claim_truth_table.md` uses "PBPK14" throughout and makes no
reference to PBPK28 — it cannot be read as covering both.

---

### 2.2 Drug-Arm Implementation Status

| Drug arm | Status | Key files | CI gate at HEAD |
|---|---|---|---|
| Rapamycin — PBPK14 epistemic | IMPLEMENTED + VALIDATED | `stdlib/darwin_pbpk/epistemic_pbpk14.sio`, `stdlib/darwin_pbpk/drugs/rapamycin.sio` | `scripts/ci/dissertation_pbpk_suite_gate.sh` ✓ |
| Rapamycin — PBPK28 (§4 chapter) | IMPLEMENTED + PARITY GATE | `stdlib/darwin_pbpk/tsit5_pbpk28.sio`, `stdlib/darwin_pbpk/core/pbpk28_params.sio`, `stdlib/darwin_pbpk/tmdd/fkbp12_mtorc1.sio`, `stdlib/darwin_pbpk/pd/coronary_smc_prolif.sio`, `stdlib/darwin_pbpk/compartments/coronary_smc.sio` | `scripts/ci/dissertation_pbpk28_parity_gate.sh` (9/9 per `chapter_04.md`; not re-run in this audit) |
| Semaglutide (SUSTAIN / PBPK28) | IMPLEMENTED + VALIDATED | `stdlib/darwin_pbpk/tmdd/glp1r.sio`, `stdlib/darwin_pbpk/pd/bergman_glucose_insulin.sio`, `stdlib/darwin_pbpk/scenarios/semaglutide_sc_depot.sio`, `tests/run-pass/dissertation_pbpk28_parity_ref_semaglutide.sio` | `darwin_tmdd_glp1r_smoke`, `darwin_pd_bergman_smoke` run-pass tests present |
| Tirzepatide (dual GIP/GLP-1) | IMPLEMENTED + VALIDATED | `stdlib/darwin_pbpk/drugs/tirzepatide.sio`, `stdlib/darwin_pbpk/validation/tirzepatide_sc_pbpk.sio`, `stdlib/darwin_pbpk/pd/glp1_gipr_gum.sio` | `dissertation_pbpk_suite_gate.sh` (tirzepatide entry present) |
| Vancomycin (ICU TDM) | IMPLEMENTED + VALIDATED | `stdlib/darwin_pbpk/drugs/vancomycin.sio`, `stdlib/darwin_pbpk/validation/vancomycin_icu_pbpk.sio`, `stdlib/clinical/vancomycin_pbpk.sio`, `stdlib/darwin_pbpk/pd/vancomycin_auc_gum.sio` | `dissertation_pbpk_suite_gate.sh` ✓ |
| Tacrolimus + sirolimus DDI | IMPLEMENTED + VALIDATED | `stdlib/darwin_pbpk/drugs/tacrolimus.sio`, `stdlib/darwin_pbpk/validation/tacrolimus_oral_pbpk.sio`, `stdlib/darwin_pbpk/ddi/tacrolimus_sirolimus_ddi.sio`, `stdlib/darwin_pbpk/validation/tacrolimus_sirolimus_ddi_clinical.sio`, `stdlib/darwin_pbpk/pd/tacrolimus_trough_gum.sio` | `dissertation_pbpk_suite_gate.sh` ✓ + Lean 4 obligations |
| Haloperidol + olanzapine (psychiatric) | IMPLEMENTED + VALIDATED | `stdlib/darwin_pbpk/drugs/haloperidol.sio`, `stdlib/darwin_pbpk/drugs/olanzapine.sio`, `stdlib/darwin_pbpk/pgx/cyp2d6_haloperidol.sio`, `stdlib/darwin_pbpk/pd/mtor_collision.sio`, `stdlib/darwin_pbpk/ddi/polypharmacy_fano.sio`, `stdlib/darwin_pbpk/population/pop_pbpk_pd.sio` | `dissertation_pbpk_suite_gate.sh` (8 new entries) ✓ |

**Observation:** Semaglutide exists in **zero** files on the PBPK14 / epistemic-computing
path. It exists solely within the PBPK28 scenario suite. The drug-arm numbering found in
commit messages (`e4123ab [dissertation] Add tirzepatide (drug #3)`,
`31b4f16 [dissertation] vancomycin TDM — drug #4`) establishes tirzepatide as drug #3
and vancomycin as #4 in the epistemic thesis — positions inconsistent with semaglutide
being that thesis's second drug.

---

### 2.3 Lean 4 Status

**Status: PARTIALLY IMPLEMENTED — statement-level, algebraic proofs are future work.**

Files present in `formal/lean4/`:

- `formal/lean4/SounioTacrolimusDosingSafety.lean` — dosing safety theorem statement;
  monotonicity of C24h w.r.t. F_oral, Vc, CL stated. Proof obligation declared; discharge
  is future work ("algebraic proofs reduce to the abstract Fréchet `pb_apply2_monotone`
  lemma").
- `formal/lean4/SounioTacrolimusDDI.lean` — two monotonicity properties for the
  tacrolimus+sirolimus DDI Fréchet enclosure; statement-level only.

Files present in `formal/` (top-level):

`SecondOrderGUM.lean`, `Epistemic.lean`, `GUM.lean`, `NonAssocHessian.lean`,
`FanoLabellingOrbits.lean`, `OctonionAlgebra.lean`, `KnowledgeArithmeticSoundness.lean`,
`GradientTopology.lean`, `EpistemicGemm.lean`, and others.

**Critical finding:** `formal/FanoLabellingOrbits.lean` and `formal/OctonionAlgebra.lean`
exist — these directly back the 168-theorem claim. However, the tacrolimus Lean files are
obligation stubs, not discharged proofs. The Fano/octonion Lean files were not audited for
discharge completeness in this memo.

**Safe dissertation language:** "The Lean 4 proof obligations for tacrolimus dosing safety
and DDI are formalised as statement-level imports; the algebraic discharges are scheduled
as future work." Do not say "The Lean 4 proofs are complete."

---

### 2.4 Canonical Dissertation Outline

**Finding: No single canonical dissertation outline file (`outline.md`, `toc.md`, spine
declaration) exists in `docs/dissertation/`. The spine must be reconstructed from secondary
evidence.**

Files examined:

| File | Spine it implies |
|---|---|
| `docs/dissertation/pbpk_claim_truth_table.md` (last validated 2026-03-07) | PBPK14 rapamycin + haloperidol; no PBPK28, no semaglutide, no tirzepatide |
| `docs/dissertation/chapter_clinical_verified_outline.md` | Vancomycin TDM chapter with tacrolimus cross-synthesis; 10-section structure |
| `docs/dissertation/handoff/chapter_04.md` | §4 of rapamycin/semaglutide PBPK28 chapter; explicit: "NOT the vancomycin chapter" |
| `docs/dissertation/handoff/psychiatric_pgx_mtor_168_pop_package.md` (today) | Four drug-class chapters: rapa → vanco → tac → halo+olz; psychiatric capstone |
| `docs/dissertation/ADVISOR_HANDOFF.md` | 3D anatomical viewer website; rapamycin + semaglutide tour; traffic-light compile gate per drug |

The `chapter_04.md` handoff declares its scope: **"§4 of the rapamycin/semaglutide PBPK28
chapter (NOT the vancomycin chapter)"** — this is the clearest evidence that the PBPK28
work and the vancomycin/tacrolimus/epistemic work are chapters of two different documents.

The psychiatric package (today) lists its position as "Chapter 4+ of: rapamycin → vanco →
tac → halo" — a 4-drug-class structure with no PBPK28, no semaglutide.

---

### 2.5 Truth Table Contents

`docs/dissertation/pbpk_claim_truth_table.md` was last validated **2026-03-07**. It
predates the PBPK28 merge (PR #127, `dadae02`, which landed after the Stage G-ζ-1 commits
visible in the log). Its claims table:

**Drug arms it covers (repo-backed):**
- Rapamycin — PBPK14 Tsit5/GUM (CPU only)
- Haloperidol — via PBPK14 + BBB sub-model
- Vancomycin — AUC-guided Knightian gate
- Tacrolimus — PBPK14 validation, GUM budget, DDI module

**Drug arms NOT in the table:**
- Semaglutide — absent
- PBPK28 — absent (the word "PBPK28" does not appear in the table)
- Tirzepatide — absent (post-dates the table)
- Olanzapine / psychiatric suite — absent (added today)

**Conclusion:** the truth table is the binding authority for the Sounio epistemic-computing
dissertation. It has no jurisdiction over the PBPK28/semaglutide chapter (Artefact A).
Artefact A's binding authority is the `chapter_04.md` handoff + the PBPK28 parity gate.

---

### 2.6 Git History Evidence

```
dadae02  Merge pull request #127 from Sounio-lang/dissertation/3d-frontend-stage-f
         (Stage G-ε/ζ: PBPK28 + semaglutide stdlib + 3D viewer)
0f691a2  [docs] dissertation tacrolimus thrust — CI gate, chapter outline, claim table
c57a95e  [formal] Lean 4 obligations for tacrolimus dosing safety + DDI
31b4f16  [dissertation] vancomycin TDM — drug #4, AUC-guided dosing + Knightian gate
e4123ab  [dissertation] Add tirzepatide (drug #3) — dual GIP/GLP-1 PBPK + ISO budget
```

**Two parallel trajectories are visible:**

1. **Stage G-δ/ε/ζ track** (branch `dissertation/3d-frontend-stage-f`, merged as PR #127):
   PBPK28 kernel, semaglutide, 3D anatomical viewer, Cypher DES stent scenario, Higuchi
   release, FKBP12/mTORC1 TMDD, Bergman PD, ADVISOR_HANDOFF walkthrough for advisor/
   committee — all characterised by "Stage G-ε-N" commit messages. This is the
   **biomaterials thesis** track.

2. **Multi-drug epistemic track** (main + current branch): drug #3=tirzepatide,
   #4=vancomycin, #5=tacrolimus+sirolimus, #6(implicit)=haloperidol+olanzapine; compile-
   time confidence gates; Knightian uncertainty; Lean 4 obligations; 168-theorem; cross-
   drug ISO budget; population PBPK/PD. This is the **Sounio epistemic-computing thesis**
   track.

The two tracks were merged into `main` side by side (PR #127 for Stage G; the tacrolimus
and psychiatric work on the main line). They share `stdlib/darwin_pbpk/` as a common
library but target different manuscripts.

**No commit message references PUC-SP, Moema, or the September-October defense timeline**
— those are editorial-manuscript-level facts not yet in version control.

---

## 3. Reconciliation Actions

### Verdict = (C) — Two dissertations

**For the biomaterials thesis (Artefact A — §§4.1–4.9):**

- §§4.1–4.9 are FINAL for this thesis. Do not rewrite, renumber, or reframe them.
- The kernel is PBPK28. "PBPK14" language must not appear in this manuscript.
- Semaglutide (SUSTAIN) is the second case study. It is IMPLEMENTED + VALIDATED.
- Pending sections (§4.10 Sobol/Cut-HDMR) continue in this document without reconciliation
  dependency.
- The truth table (`pbpk_claim_truth_table.md`) does NOT govern claims in this manuscript.
  The governing artefact is `docs/dissertation/handoff/chapter_04.md` + the PBPK28 parity
  gate output.
- §§4.7–4.9 (Hessian-corrected GUM, cross-drug ISO budget, transversal epistemic
  infrastructure) belong in this thesis as contributions of the PBPK28 dissertation. They
  are NOT shared with the Sounio thesis unless explicitly cross-cited.

**For the Sounio epistemic-computing thesis (Artefact B):**

- Spine: rapamycin (PBPK14 epistemic) → tirzepatide → vancomycin → tacrolimus+sirolimus →
  haloperidol+olanzapine (psychiatric capstone). Semaglutide does NOT appear in this
  thesis unless a PBPK14 semaglutide path is separately committed.
- The psychiatric package (`docs/dissertation/handoff/psychiatric_pgx_mtor_168_pop_package.md`)
  is correctly positioned as the capstone chapter.
- `docs/dissertation/pbpk_claim_truth_table.md` should be updated to add tirzepatide and
  the psychiatric suite as `repo-backed` entries.
- Lean 4 obligations are correctly framed as future work; do not claim discharge.

**Action items for the writing thread:**

1. **Immediately:** Confirm which defense (Sept–Oct) targets which manuscript (answer to
   §4 below).
2. **Biomaterials thesis:** Continue §4.10 (Sobol/Cut-HDMR) with no rewrite needed.
   Reference the PBPK28 parity gate, not `dissertation_pbpk_suite_gate.sh`.
3. **Sounio thesis:** The psychiatric capstone package is ready. The next writing deliverable
   is the chapter 1/2 introduction that frames the four drug-class arc and positions
   PBPK14 correctly (not PBPK28).
4. **Do NOT mix kernels:** The Sounio thesis must not reference PBPK28 as its rapamycin
   model. The rapamycin PBPK14 epistemic path (`epistemic_pbpk14.sio`) is a different model
   from the PBPK28 path and they have different claims.
5. **Truth table maintenance:** Add tirzepatide and the psychiatric suite to
   `pbpk_claim_truth_table.md`; mark semaglutide as `repo-backed` only in the PBPK28
   context.

---

## 4. The Single Question for the Human

> **Is the September–October defense the PUC-SP biomaterials master's (PBPK28,
> rapamycin + semaglutide, advisor Dr. Moema Haussen), OR the Sounio epistemic-computing
> thesis (four drug-class chapters, compile-time gates, Lean 4)?**

This one answer determines everything:

- If **biomaterials (PUC-SP)**: §4.10 is the next section; §§4.1–4.9 are final; the Sounio
  thesis is a separate, longer-horizon project.
- If **Sounio epistemic**: the psychiatric capstone package is the last chapter; §§4.1–4.9
  are a different, parallel manuscript not on the Sept–Oct critical path; the next writing
  deliverable is the Sounio thesis introduction.
- If **both on the same timeline**: escalate immediately — two 60–80 page dissertations
  cannot share a single Sept–Oct window without a committee agreement on scope reduction.

---

## 5. Files to Be Touched

This audit produces **one new file only** (this memo). No chapter prose was written or
modified.

| Action | File | Reason |
|---|---|---|
| ADD (this memo) | `docs/dissertation/RECONCILIATION_MEMO_2026-05-12.md` | Audit deliverable |
| PENDING (human decision) | `docs/dissertation/pbpk_claim_truth_table.md` | Add tirzepatide + psychiatric entries once thesis spine confirmed |
| NO TOUCH | `§§4.1–4.9` files in `/mnt/project/` | Out of scope; biomaterials thesis working tree |
| NO TOUCH | Any stdlib `.sio` files | Audit only, no fixes |
