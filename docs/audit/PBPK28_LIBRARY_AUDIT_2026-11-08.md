<!-- docs:meta
topic_id: repo.docs.audit.pbpk28-library-audit-2026-11-08
authority: repo_only
audience: users
last_validated: 2026-11-08
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk28-library-audit-2026-11-08
-->

# PBPK28 library audit — 2026-11-08

Scope: every drug the library carries on the 28-compartment model (14 organs x
{vascular, interstitial}), measured on `main` after PR #3481 (fentanyl). Every
number below is produced by a command in the last section; nothing is quoted
from memory. Literature values are either marked *as cited by the repository*
(not independently verified here) or *verified* (PubMed, with DOI).

## 1. Inventory

15 drugs, 18 parameter sets (risperidone has parent + 9-OH, venlafaxine parent +
ODV, semaglutide has a core and a drug-module copy):

clozapine, olanzapine, quetiapine, risperidone (+9-OH), aripiprazole, lithium,
tacrolimus, everolimus, vancomycin, tirzepatide, semaglutide, rapamycin,
venlafaxine (+ODV), morphine, fentanyl.

Not on PBPK28, despite files under `drugs/` and tests named
`dissertation_pbpk28_parity_ref_*`: **haloperidol** (PBPK14 well-stirred
parameters, `haloperidol_pbpk_params`) and **midazolam** (closed-form oral
first-pass + CYP3A inhibition, no 28-compartment transport). Counting them as
PBPK28 drugs would overstate the library by two; the test names are misleading
but are hard-coded in `scripts/ci/dissertation_pbpk28_parity_gate.sh`, so they
are left as they are here.

## 2. Structural invariants (gated)

`tests/run-pass/darwin_pbpk28_library_invariants.sio` gates, for all 18 sets:
v_ref > 0, q > 0 and ps > 0 for every exchanging organ, kp > 0, vasc_frac in
[0, 1], V_ss above V_b, and **finiteness of every checked value and of V_ss**
(`math::pure::is_finite`, which rejects NaN and +-Inf; the sign and range tests
alone do not, because NaN compares false against every bound and +Inf passes any
`> 0`). **All 18 pass on Madaros and lean_single.** The class of defect found in
the sedation-weaning morphine template (lung PS = 0, an unreachable tissue layer)
does not occur anywhere in the library.

The gate carries its own negative controls, so discrimination does not depend on
a manual mutant: lung PS = 0, NaN in kp / ps / q / vasc_frac / v_ref / cl_central,
and +Inf in kp and v_ref, each injected into the finite morphine set, must be
rejected, and the finite baseline must still pass (9 rejections, both engines).
The first version of the gate lacked the finiteness checks and printed
`PBPK28_LIBRARY_INVARIANTS_PASS` for ps[lung] = NaN, kp[adipose] = +Inf
(V_ss = inf) and q[liver] = NaN. That is what the review of this PR found, and the
controls above are its regression test. The lung PS = 0 control moves V_ss from
157.515 to 151.833 L, the figure the morphine header documents for that defect.

## 3. Distribution volume against the target each module declares

V_ss = V_b + sum V_i (f_v + (1 - f_v) Kp) over reachable layers. "Declared
target" is the number in the module's own header, **as cited by the repository,
not verified here**.

| # | Set | V_ss (L) | Declared target (L) | Deviation | Status |
|---|---|---|---|---|---|
| 1 | clozapine | 408.89 | ~400 | +2.2 % | within 10 % |
| 2 | olanzapine | 665.63 | ~700 | -4.9 % | within 10 % |
| 3 | quetiapine | 326.31 | ~700 | -53.4 % | declared: adipose capped, full-Vd variant is follow-up |
| 4 | aripiprazole | 386.90 | 404 (label) | -4.2 % | within 10 % |
| 5 | lithium | 44.02 | 45 | -2.2 % | within 10 % |
| 6 | tacrolimus | 253.79 | ~1300 (Staatz 2004, as cited) | -80.5 % | declared: `tacrolimus_fullvd_params` exists |
| 7 | everolimus | 48.67 | 110 (label) | -55.8 % | declared: Kp copied from rapamycin, pending SIO-550 |
| 8 | vancomycin | 46.13 | 49 (Vc 0.40 + Vp 0.30 L/kg) | -5.9 % | within 10 % |
| 9 | tirzepatide | 21.57 | none declared | n/a | no V_ss target; Kp copied from semaglutide |
| 10 | risperidone parent | 112.65 | ~115 | -2.0 % | within 10 % |
| 11 | risperidone 9-OH | 452.10 | 487 (paliperidone label) | -7.2 % | within 10 % |
| 12 | rapamycin | 48.67 | 840 (12 L/kg, Kahan 2001, as cited) | -94.2 % | declared: full-Vd parameters exist |
| 13 | venlafaxine parent | 107.81 | none declared | n/a | no V_ss target in the repository |
| 14 | venlafaxine ODV | 86.43 | none declared | n/a | no V_ss target in the repository |
| 15 | semaglutide (core) | 21.57 | none declared | n/a | no V_ss target |
| 16 | semaglutide (drug module) | 21.57 | none declared | n/a | identical to row 15 (`semaglutide_drugs_consistency`) |
| 17 | morphine | 157.52 | 231 (Mazoit 2007, as cited by the module, unverified); 136 (Bouwmeester 2004, verified) | -31.8 % / +15.8 % | declared tension (morphine header, limitation 1) |
| 18 | fentanyl | 510.18 | none declared for V_ss | n/a | declared tension is the terminal t1/2: 21.0 h simulated against 219 min (McClain & Hug 1980, verified) |

Totals reconcile with the 18 sets the gate enumerates: **7 within 10 %** (rows
1, 2, 4, 5, 8, 10, 11), **5 far from a declared target, with the tension declared**
(rows 3, 6, 7, 12, 17), **6 without a V_ss target** (rows 9, 13, 14, 15, 16, 18);
7 + 5 + 6 = 18.

Observations, all measured:

- Seven sets sit within 10 % of the target their own header declares; five are
  far from it and say so in their headers; six have no V_ss target at all.
- Rapamycin and everolimus have the same V_ss to six digits (48.671031 L) and
  semaglutide and tirzepatide the same (21.569860 L): the Kp vectors are copies, and
  the copies are declared in the module headers ("Kp/PS mirror the rapamycin
  vector ... assumed, pending SIO-550"; "Kp/PS mirror the semaglutide peptide
  vector"). They are assumptions, not independent parameterisations.
- Venlafaxine's 1.54 L/kg has no declared target. Two broad PubMed searches
  (venlafaxine pharmacokinetics volume of distribution; venlafaxine
  O-desmethylvenlafaxine clinical pharmacokinetics review) returned no abstract
  stating a steady-state volume, so no value is asserted here; it needs a
  verified source before it is compared.
- Aripiprazole carries elimination as organ sinks (`cl_central = 0`, CYP3A4 0.52 and
  CYP2D6 0.49 L/h on the liver), so no MRT = V_ss/CL is defined from the
  parameter set alone.

## 4. What each test tier actually establishes

| Tier | Establishes | Drugs today |
|---|---|---|
| P, parity | the Sounio scenario equals the website engine (`pbpk28_core.mjs`), RMSE < 1 % per compartment on cavg, `dissertation_pbpk28_parity_gate.sh` | the 13 pre-existing drugs (one case each) |
| A, analytic | a value fixed by hand before the run: runtime clearance identity, MRT = V_ss/CL and AUC = Dose/CL on a converged grid, mass identity | 14 of 15: morphine and fentanyl (own tests); clozapine, olanzapine, quetiapine, lithium, tacrolimus, everolimus, vancomycin, tirzepatide, risperidone (parent and 9-OH), rapamycin, venlafaxine (parent and ODV), semaglutide (`darwin_pbpk28_library_moments.sio`, added 2026-11-09); also the clozapine x fluvoxamine dynamic layer |
| L, literature | a PubMed-verified published value, deviation reported | morphine (V_ss vs Bouwmeester 2004), fentanyl (terminal t1/2 vs McClain & Hug 1980, [10.1038/clpt.1980.138](https://doi.org/10.1038/clpt.1980.138)) |

Parity proves two implementations agree; it does not show that either is
pharmacologically right. Under the criterion that a result counts as done only
when an explicit model predicted it, 2 of 15 drugs were at tier A when this
audit was written.

**Update 2026-11-09.** `tests/run-pass/darwin_pbpk28_library_moments.sio` applies
the tier-A gates to the 14 remaining sets that carry elimination in `cl_central`,
against the V_ss/CL column of section 3, fixed before the run. All 14 pass on
Madaros (19 s) and lean_single (74 s), with identical numbers: runtime clearance
equal to `cl_central` to six digits, MRT within 0.24 % (largest: quetiapine,
3.8754 against 3.8846 h), AUC within 0.24 %, quadrature converged (< 0.2 %), tail
< 1e-13 and mass residual ~1e-13 (lean_single prints them unrounded). The AUC
excess is trapezoid error, not model error: it falls with the step at second
order. The step grows geometrically late (0.5 % of t) and the horizon is
80 x MRT, so the slow peptide sets (tirzepatide 354 h, semaglutide 280 h) cost
no more than the fast ones. **14 of 15 drugs are now at tier A.** Aripiprazole is
the exception: its elimination is two CYP sinks on the liver with
`cl_central = 0`, so V_ss/CL has no closed form from the parameter set alone.

Tier A here means internal consistency of each parameter set with the PBPK28
transport. It is not tier L: no V_ss or CL of the 13 older drugs is checked
against a verified published value yet.

## 5. Venlafaxine ODV

- **Mass-balance defect: fixed.** Issue #2718 ("parity reference and website JS
  engine still fabricate ODV mass") was closed by PR #2914 on 2026-10-13.
  Verified on the current tree, Madaros: `darwin_venlafaxine_formation_sink_implicit`
  passes (formed ODV against f_m x absorbed, with a bound, across phenotypes and
  dt) and the venlafaxine parity reference runs with the implicit liver sink.
- **Open scientific gap, honestly tagged.** `darwin_venlafaxine_xr_odv_ratio_literature`
  is `known-failure` and prints `LITERATURE_FAIL_HONEST outside_mask=9`: the
  steady-state ODV/parent ratio is outside the published band for PM (0.3465
  against < 0.3) and UM (2.502 against > 5.2) and inside for IM (1.127) and NM
  (1.912). The tag attributes it to hepatic flow capping formation. It is the
  same kind of gap as the clozapine x fluvoxamine 5-10x: the model is
  mechanically consistent and the in-vitro-derived fractions do not reach the
  clinic.

## 6. Reproduce

```bash
./bin/souc run tests/run-pass/darwin_pbpk28_library_invariants.sio   # sections 2, 3
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/darwin_pbpk28_library_invariants.sio
./bin/souc run tests/run-pass/darwin_venlafaxine_formation_sink_implicit.sio       # section 5
./bin/souc run tests/run-pass/darwin_venlafaxine_xr_odv_ratio_literature.sio       # section 5 (rc 1 is the declared known-failure)
gh api repos/Sounio-lang/sounio/issues/2718 --jq '{state,closed_at}'
```
