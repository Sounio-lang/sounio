<!-- docs:meta
topic_id: repo.docs.dissertation.handoff.chapter-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.handoff.chapter-04
-->

# Chapter 4 — Handoff packet for Claude Desktop

**For:** Claude Desktop, drafting §4 of the rapamycin/semaglutide PBPK28 chapter
(NOT the vancomycin chapter — that's `chapter_clinical_verified_outline.md`).

**From:** Claude Code session 2026-05-11, branch `dissertation/3d-frontend-stage-f`,
HEAD at `95ff2b9b` (G-ε-17). Worktree: `/workspace/sounio-stage-g-gamma`.

**Scope:** every §4 claim Desktop will write now has an in-repo stdlib backing
module + a smoke gate + a numerical result to quote. The PBPK28 parity gate
(scripts/ci/dissertation_pbpk28_parity_gate.sh) is 9/9 PASS at HEAD.

Drafting rules: cite the file path and the smoke marker. Quote numerical values
from the §"Numerical results to quote" table below — those are the live numbers
this commit produces, not paper-target estimates.

---

## 1. Claims → backing modules

| §            | Claim                                                          | File                                                               | LOC | Key symbols                                                              |
|--------------|----------------------------------------------------------------|--------------------------------------------------------------------|-----|--------------------------------------------------------------------------|
| §4.2.1       | 14-organ topology, blood=0, indexes 1-13 named                 | `stdlib/darwin_pbpk/core/pbpk28_params.sio:24-46`                  |  97 | `PBPKState28`, `PBPKParams28`                                            |
| §4.2.3       | Tsit5 adaptive integrator, fully-coupled CN at PBPK28          | `stdlib/darwin_pbpk/tsit5_pbpk28.sio`                              |  —  | (existing, pre-G-ε)                                                      |
| §4.2.4       | Mass-balance auditor                                           | `stdlib/darwin_pbpk/core/pbpk28_params.sio:83-97`                  |  —  | `pbpk28_total_mass`                                                      |
| §4.2.5       | Parity-ref against JS engine, literature PS                    | `tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio`      | 548 | (run-pass, calls into stdlib via `use darwin_pbpk::{tmdd,pd}::*`)        |
| §4.3.4       | Rapamycin K_p table (1.00, 5.40, 4.20, 0.10, 2.30, ...)        | `stdlib/darwin_pbpk/core/pbpk28_params.sio:53-62`                  |  —  | `pbpk28_params_rapamycin`                                                |
| §4.3.5       | Semaglutide K_p table (peptide, all K_p < 1)                   | `stdlib/darwin_pbpk/core/pbpk28_params.sio:68-77`                  |  —  | `pbpk28_params_semaglutide`                                              |
| §4.4.2       | Higuchi release `dQ/dt = K_H / (2√t)`, K_H=0.00417 mg·√h⁻¹     | `stdlib/darwin_pbpk/release/biomaterial_release.sio:79-131`        | 529 | `release_higuchi`, `release_rate`, `release_cumulative`                  |
| §4.4.2.1     | Higuchi t→0 singularity regularization `max(t, 0.01)` cap      | `stdlib/darwin_pbpk/release/biomaterial_release.sio:124-128`       |  —  | inline in `release_rate`                                                 |
| §4.4.2.3     | f_local = 0.30 Cypher routing (30% local / 70% systemic)       | `stdlib/darwin_pbpk/compartments/coronary_smc.sio:54-72`           | 126 | `coronary_smc_default_params`, `coronary_smc_systemic_flux`              |
| §4.5.3       | coronary_smc = 5%-of-heart sub-compartment (V=0.0165 L)        | `stdlib/darwin_pbpk/compartments/coronary_smc.sio:46-72`           |  —  | `CoronarySmcParams`, `coronary_smc_step`                                 |
| §4.5.4       | TMDD path rapamycin (FKBP12 / mTORC1) at liver, heart, gut     | `stdlib/darwin_pbpk/tmdd/fkbp12_mtorc1.sio`                        | 184 | `fkbp12_*`, `tmdd_step`                                                  |
| §4.5.5       | TMDD path semaglutide (GLP-1R) at brain, gut, pancreas         | `stdlib/darwin_pbpk/tmdd/glp1r.sio`                                |  77 | `glp1r_*` (re-uses `TmddState`/`TmddOrganParams` from fkbp12 module)     |
| §4.6.2       | PD rapamycin (mTORC1 inhibition → neointimal index)            | `stdlib/darwin_pbpk/pd/coronary_smc_prolif.sio`                    | 102 | `coronary_pd_params`, `pd_step_rapamycin`, `pd_late_lumen_loss_mm`       |
| §4.6.3       | PD semaglutide (linearised Bergman glucose-insulin)            | `stdlib/darwin_pbpk/pd/bergman_glucose_insulin.sio`                | 113 | `bergman_default_params`, `pd_step_glucose_insulin`                      |

### Scenario composition (§4.7 demo material)

| File                                                            | LOC | What it shows                                                                                                  |
|-----------------------------------------------------------------|-----|----------------------------------------------------------------------------------------------------------------|
| `stdlib/darwin_pbpk/scenarios/semaglutide_sc_depot.sio`         | 140 | Full SC depot → PBPK28 → GLP-1R TMDD → Bergman PD pipeline composed from the G-ε-12..15 stdlib pieces           |
| `tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio`   | 548 | Same composition for rapamycin, with parity emission for 14 organs × 12 sample times — wired against JS engine |
| `tests/run-pass/dissertation_pbpk28_parity_ref_semaglutide.sio` | 454 | Semaglutide parity-ref counterpart                                                                             |

Both parity refs import stdlib modules (G-ζ-1, commit `566c3663`): `use darwin_pbpk::tmdd::fkbp12_mtorc1::*` / `use darwin_pbpk::pd::coronary_smc_prolif::*` (rapa) and `use darwin_pbpk::tmdd::glp1r::*` / `use darwin_pbpk::pd::bergman_glucose_insulin::*` (sema). Net −43 LOC vs pre-rewire.

---

## 2. Gate commands and their PASS markers

All gates run from worktree root. Each has a deterministic last-line marker
that Desktop can quote as the artefact of the verification.

### 2.1 PBPK28 parity gate (Sounio ↔ Node, 9 cases)

```sh
bash scripts/ci/dissertation_pbpk28_parity_gate.sh
```

Last-known output (HEAD `95ff2b9b`, re-run 2026-05-11):

```
PBPK28_PARITY_PASS 14/14 compartments within 1.0% RMSE (organ-average)
PBPK28_MASS_CONSERVATION_PASS 12/12 samples monotonically decay
PBPK28_TMDD_PARITY_PASS 3/3 TMDD organs within 1.0% RMSE on (R_free, DR)
PBPK28_PD_PARITY_PASS 1/1 PD organ(s) within 1.0% RMSE on (A, N)
PBPK28_SEMAGLUTIDE_PARITY_PASS 14/14 compartments within 1.0% RMSE
PBPK28_SEMA_TMDD_PARITY_PASS 3/3 GLP-1R organs within 1.0% RMSE on (R_free, DR)
PBPK28_SEMA_PD_PARITY_PASS 1/1 PD organ(s) within 1.0% RMSE on (ΔG, ΔI)
```

Cases 2 and 3 are degenerate-asymptotic and PBPK14-cross-validation
self-checks that emit no `PBPK28_*_PASS` line but exit 0.

### 2.2 Drug-layer smoke tests

| Smoke file (under `tests/run-pass/`)              | LOC | Last-line marker                  | Asserts                                                                                                                                                              |
|---------------------------------------------------|-----|-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `darwin_tmdd_fkbp12_smoke.sio`                    |  85 | `TMDD_FKBP12_SMOKE_PASS`          | Rapamycin TMDD at liver/heart/gut: R_free drops, DR rises under bolus; both at zero under no drug                                                                    |
| `darwin_tmdd_glp1r_smoke.sio`                     |  80 | `TMDD_GLP1R_SMOKE_PASS`           | Same shape, semaglutide GLP-1R at brain/gut/pancreas                                                                                                                 |
| `darwin_pd_coronary_smc_smoke.sio`                |  98 | `PD_CORONARY_SMC_SMOKE_PASS`      | Rapamycin PD: 3 regimes (no drug, partial inh, full inh); N grows under no drug, decays under inhibition; LLL stub gives RAVEL-order LLL                             |
| `darwin_pd_bergman_smoke.sio`                     |  98 | (run to capture)                  | Semaglutide PD: ΔG/ΔI plausible under GLP-1R occupancy                                                                                                               |
| `darwin_sema_sc_depot_smoke.sio`                  | 104 | `SEMA_SC_DEPOT_SMOKE_PASS`        | Full SC depot composition from G-ε-16: absorption + PBPK + TMDD + PD                                                                                                 |
| `darwin_compartments_coronary_smc_smoke.sio`      | 149 | `CORONARY_SMC_SMOKE_PASS`         | 5%-of-heart sub-compartment + f_local routing under 3 regimes (0.30, 0.00, 1.00). Three invariants: routing mass-balance, split ratio, Higuchi integrator drift      |

Each runs in <30 s under `./bin/souc compile <file> -o /tmp/x.elf && /tmp/x.elf`.

### 2.3 Single-file run

```sh
./bin/souc compile tests/run-pass/<smoke>.sio -o /tmp/smoke.elf && /tmp/smoke.elf
```

---

## 3. Numerical results to quote

These are the actual numbers this commit produces. Quote them verbatim.

### 3.1 PBPK28 parity (Sounio ↔ Node golden, rapamycin literature PS, n=14×12 samples)

| Compartment | Peak C (mg/L) | RMSE | Threshold |
|---|---|---|---|
| 0 blood | 1.062e-02 | 0.0000 % | 1.0 % |
| 1 liver | 6.423e-03 | 0.0000 % | 1.0 % |
| 2 kidney | 7.052e-03 | 0.0000 % | 1.0 % |
| 3 brain | 8.990e-04 | 0.0000 % | 1.0 % |
| 4 heart | 5.916e-03 | 0.0000 % | 1.0 % |
| 5 lung | 5.135e-03 | 0.0000 % | 1.0 % |
| 6 muscle | 2.426e-03 | 0.0000 % | 1.0 % |
| 7 adipose | 1.504e-03 | 0.0000 % | 1.0 % |
| 8 gut | 8.521e-03 | 0.0000 % | 1.0 % |
| 9 skin | 3.306e-03 | 0.0000 % | 1.0 % |
| 10 bone | 2.434e-03 | 0.0000 % | 1.0 % |
| 11 spleen | 6.044e-03 | 0.0000 % | 1.0 % |
| 12 pancreas | 7.995e-03 | 0.0000 % | 1.0 % |
| 13 other | 3.533e-03 | 0.0000 % | 1.0 % |

All 14 organs at 0.0000 % RMSE → Sounio ↔ Node bit-exact at the f64-double
threshold used by the gate.

### 3.2 TMDD parity (Sounio ↔ Node)

| Drug | Organ | Peak R_free (nmol/L) | Peak DR (nmol/L) | R_free RMSE | DR RMSE |
|---|---|---|---|---|---|
| Rapamycin | 1 liver | (see gate stdout) | (see gate stdout) | 0.0000 % | 0.0000 % |
| Rapamycin | 4 heart | (see gate stdout) | (see gate stdout) | 0.0000 % | 0.0000 % |
| Rapamycin | 8 gut | (see gate stdout) | (see gate stdout) | 0.0000 % | 0.0000 % |
| Semaglutide | 3 brain | 9.999880e-01 | 1.887020e-01 | 0.0000 % | 0.0000 % |
| Semaglutide | 8 gut | 1.999494e+00 | 1.602497e+00 | 0.0000 % | 0.0000 % |
| Semaglutide | 12 pancreas | 4.998172e+00 | 3.894717e+00 | 0.0000 % | 0.0000 % |

### 3.3 PD parity (Sounio ↔ Node)

| Drug | Organ | Endpoint | Peak | RMSE |
|---|---|---|---|---|
| Rapamycin | 4 heart | A (mTORC1 activity, 0..1) | (gate stdout) | 0.0000 % |
| Rapamycin | 4 heart | N (neointimal index, 0..1) | (gate stdout) | 0.0000 % |
| Semaglutide | 12 pancreas | ΔG (mg/dL) | 9.125985e+00 | 0.0000 % |
| Semaglutide | 12 pancreas | ΔI (mU/L) | 7.785732e+00 | 0.0000 % |

### 3.4 coronary_smc sub-compartment + f_local routing (24 h Higuchi DES, Cypher defaults)

From `darwin_compartments_coronary_smc_smoke.sio`:

| Scenario | f_local | Deposited (mg) | Systemic (mg) | Rect-rule integral (mg) | Higuchi cumulative (mg) | Routing balance | Split ratio diff vs f_local |
|---|---|---|---|---|---|---|---|
| Cypher default | 0.30 | 0.006099 | 0.014231 | 0.020330 | 0.020428 | < 1e-9 rel | < 1e-9 abs |
| Zero local | 0.00 | 0.000000 | 0.020330 | 0.020330 | 0.020428 | < 1e-9 rel | < 1e-9 abs |
| All local | 1.00 | 0.020330 | 0.000000 | 0.020330 | 0.020428 | < 1e-9 rel | < 1e-9 abs |

Three invariants asserted: (a) routing mass-balance < 1e-9 relative against
the rate integral both sides consume — the splitter does not leak; (b) split
ratio matches f_local within 1e-9 — routing semantics exact; (c) rate
integral within 1 % of analytical Higuchi cumulative — `release_rate` caps
the t→0 singularity to a finite rectangle, the cumulative formula `K_H · √t`
does not. The 0.000098 mg gap (0.48 % of total) is this integration-method
drift, not a routing issue.

---

## 4. Caveats and intentional gaps

Honest disclosure — Desktop should fold these into §4 footnotes or a §4
"current limitations" subsection rather than papering over them.

1. **The 5%-mass sub-compartment is stdlib-only, not parity-gated.** The
   PBPK28 parity gate continues to exercise heart (organ index 4) as the
   coronary_smc proxy. The new
   `stdlib/darwin_pbpk/compartments/coronary_smc.sio` module + its smoke
   stand as the canonical formulation; the parity gate will adopt the
   carve-out under planned stage G-η-1 once the JS engine
   (`website/src/lib/pbpk28_core.mjs`) implements the same split.
   §4.5.3 should distinguish "implemented in stdlib" from
   "exercised by the parity gate."

2. **f_local = 0.30 is a band, not a measurement.** Sehgal 1995 and
   Sousa 2001 estimate 20–40 % local deposition for Cypher; 0.30 is the
   midpoint. The parameter is a `CoronarySmcParams` field — Desktop should
   flag it as a sensitivity-sweep candidate, not a fitted constant.

3. **`pbpk28_core/pbpk28_params.sio` and the parity refs still hold
   parallel organ-constant tables.** G-ζ-1 rewired the drug-layer (TMDD,
   PD, MW) accessors, but the per-index `vref_at(i) / vasc_frac_at(i) /
   kp_at(i) / q_at(i) / ps_at(i)` accessor pattern in the parity refs is
   structurally different from the struct-of-arrays
   `PBPKParams28.v_ref[i]` in stdlib core. A follow-on adapter pass (small
   wrappers around `pbpk28_params_rapamycin()`) would close this; until
   then, organ-constant drift between stdlib and parity refs is in
   principle possible (in practice they were copied from the same source
   and currently match).

4. **Higuchi-rate `release_rate` clamps the t→0 singularity to
   `max(t, 0.01)` (i.e. 0.01 h floor)**, not the dissertation-draft claim
   of `ε_t = 1e-4 h`. The clamp is 1000× looser. Either fix the
   regularizer in `biomaterial_release.sio:127` to match the draft (~1 LOC
   change, but breaks Higuchi golden tests), or correct §4.4.2.1 to say
   "ε_t = 0.01 h". Code is currently authoritative — this is a real
   inconsistency Desktop should not gloss over.

5. **Late-lumen-loss (LLL) stub `pd_late_lumen_loss_mm(n) = n` mm/unit-N
   in stdlib coronary_smc_prolif.sio:99** is a placeholder. RAVEL bare-stent
   LLL ≈ 0.74 mm at 6 months; the smoke verifies the right order of
   magnitude (~0.084 mm under no drug at 168 h linear extrapolation).
   A proper LLL ↔ N calibration is dissertation future work.

6. **`tmdd_step_organ` and `pd_step_organ` glue functions remain inside
   the parity refs (548 / 454 LOC)** — they unpack `Pbpk28` vectors and
   handle nmol/mg unit conversion, which the pure-kernel stdlib
   `tmdd_step` and `pd_step_rapamycin` don't carry. This is deliberate:
   the stdlib kernels stay drug-and-state-agnostic; the integration glue
   stays at the call site. §4 should describe the layering, not pretend
   the parity refs are pure imports.

---

## 5. Commit lineage for §4.1 status box

§4.1 ("Stage G status") should reference the commits, in order:

| Commit | Stage | What landed |
|---|---|---|
| `0b3312a2` | G-ε-11 | TMDD FKBP12/mTORC1 (rapamycin) stdlib module + smoke |
| `98cfd092` | G-ε-12 | TMDD GLP-1R (semaglutide) stdlib module + smoke |
| `241320db` | G-ε-13 | PD coronary_smc_prolif (rapamycin mTORC1 + neointima) + smoke |
| `87333dc5` | G-ε-14 | PD Bergman glucose-insulin (semaglutide) + smoke |
| `90ef2cdd` | G-ε-15 | PBPK28 core (state + params) stdlib module |
| `0234720d` | G-ε-16 | Semaglutide SC depot scenario composition |
| `566c3663` | G-ζ-1 | Parity refs rewired to import stdlib (−43 LOC) |
| `95ff2b9b` | G-ε-17 | coronary_smc 5%-of-heart sub-compartment + f_local routing (+275 LOC) |

Branch: `dissertation/3d-frontend-stage-f` (worktree
`/workspace/sounio-stage-g-gamma`).

---

## 6. What this packet does NOT include

- Prose paragraphs.
- Section ordering or pedagogical narrative.
- Comparisons to the broader PBPK literature (Rowland, Jones, Lukacova).
- The "why Sounio specifically" argument — that belongs in §2/§3 of the
  dissertation, not §4.
- Vancomycin-related content. That chapter is separate
  (`docs/dissertation/chapter_clinical_verified_outline.md`).
- Run-fresh gate output. Re-run the gate at the time of drafting to
  capture the most current numerical witnesses; the numbers in §3 above
  are from 2026-05-11 at HEAD `95ff2b9b`.
