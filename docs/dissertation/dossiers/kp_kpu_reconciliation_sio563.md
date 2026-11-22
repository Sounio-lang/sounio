<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.kp-kpu-reconciliation-sio563
authority: repo_only
audience: users
last_validated: 2026-11-09
validated_by: SIO-563
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.kp-kpu-reconciliation-sio563
-->

# Kp, Kpu and Kp,uu reconciliation (SIO-563)

This report binds the Rodgers-Rowland unit repair to implementation base
`3765b7d024a46a67c7a13675ba1e92857cffb8bd`. It does not reclassify fixed Kp
tables as mechanistic predictions, and it does not treat computational parity
as clinical validation.

## Declared concentration convention

- `fu_p = C_plasma,unbound / C_plasma,total`.
- `fu_t = C_tissue,unbound / C_tissue,total`.
- `Kpu = C_tissue,total / C_plasma,unbound`.
- `Kp = C_tissue,total / C_plasma,total = fu_p * Kpu`.
- `Kp,uu = C_tissue,unbound / C_plasma,unbound = (fu_t / fu_p) * Kp`.
- Therefore `Kp = (fu_p / fu_t) * Kp,uu`.

The neutral implementation under review does not use blood:plasma ratio, Ka,
or pKa. The unit and sensitivity gates therefore hold tissue composition and
the non-target physicochemical input fixed, and record B:P and Ka as not used
or recomputed. PBPK28 tissue:vascular Kp arrays are not automatically
convertible to the tissue:plasma convention without a separate denominator
reconciliation.

## Implemented repair and historical PR disposition

Before this change, `rr_neutral` divided Kpu by `fu_plasma`, divided the logP
sensitivity by `fu_plasma`, and assigned a negative fu sensitivity. The repair
uses `Kp = fu_plasma * Kpu`,
`dKp/dlogP = fu_plasma * P * ln(10) * lipid_composite`, and
`dKp/dfu_p = Kpu > 0`. The diagonal GUM variance in
`rr_scaled_uncertainty` is numerically unchanged by the sign correction because
that coefficient is squared; its comment is corrected.

The local 12-step Newton square root also remained inaccurate at the actual
rapamycin raw-variance magnitudes. It is replaced by `math::pure::sqrt`. The
module already delegates its exponential to `math::pure::exp` through #846;
absolute `10^-3` and `10^7` anchors guard that current path.

Historical status checked live on 2026-11-09:

- PR #2703 is open at head
  `7f6d0234fca281fd809ba0e2c9bed13b2a342925`, with merge status conflicting.
  Its unit and square-root findings remained applicable; its broader helper and
  composition-table patch is not imported wholesale because current main has
  since changed the exponential path and the composition audit is separate.
- PR #2729 is closed without merge at head
  `01ea92de4d1b0ac1407e320a29ad24c956ac607d`. Its unit correction was not
  present on the implementation base.

## Sensitivity contract and discriminating controls

The executable Sounio controls use `logP = 0`, `fu_p = 0.2`, and central steps
`h_fu = h_logP = 1e-6`. Tissue composition is fixed. T5 checks the closed-form
conversion in all 14 compartments; T6 checks the positive fu derivative; T7
checks the logP derivative; T8 squares the reported standard uncertainty back
to its analytical variance at the rapamycin point (`logP = 7`, `fu_p = 0.08`,
`u_logP = 1.05`, `u_fu = 0.02`); T9 anchors the shared power path.

The C++23 oracle is an independent component control for the concentration
definition, finite-difference derivatives, uncertainty propagation and square
root. It does not read values emitted by the Sounio process and therefore is
not, by itself, a cross-language comparator. The discriminating Sounio T5-T9
controls and the deliberately failing historical mutants establish the Sounio
implementation boundary.

Measured negative controls:

- With the historical division retained, T5, T6 and T7 matched `0/14`; the
  independent C++23 oracle passed.
- After the unit repair but before replacing the local square root, T8 recovered
  `0/14` analytical variances.
- With both repairs, T5-T8 matched `14/14`, T9 passed, both changed Sounio files
  checked, and the module ran to `PASS`.

## Fifteen-drug result trace

The 15 named drug families below include parallel PBPK28/PBPK14 models and
parent/metabolite variants where the repository defines them. No listed
constructor calls `rr_neutral`; “R-R conceptual” means a fixed table is
attributed to an R-R pattern, not that the repaired function generated it.

| Drug family | Current Kp source at the implementation base | Dependency and impact |
| --- | --- | --- |
| Rapamycin | `core/pbpk28_params.sio::pbpk28_params_rapamycin`; `epistemic_pbpk14.sio::ep14_rapamycin_params` | Fixed PBPK28 and PBPK14 tables; no direct change. `drugs/rapamycin.sio::rapamycin_priors` carries a copied `rr_scale = 2.4306`, so its conceptual provenance remains separate from call-graph impact. |
| Venlafaxine | `core/pbpk28_params.sio::pbpk28_params_venlafaxine_parent` and `pbpk28_params_venlafaxine_odv` | Fixed, R-R-conceptual tables; no executable dependency on `rr_neutral`. ODV ratios remain provenance-pending. |
| Semaglutide | `core/pbpk28_params.sio::pbpk28_params_semaglutide`, wrapped by `drugs/semaglutide.sio::semaglutide_pbpk28_params` | Fixed peptide table; R-R explicitly declared inapplicable. No direct change. |
| Tirzepatide | `scenarios/tirzepatide_sc_depot.sio::tirz_pbpk28_params` | Fixed analogue of the semaglutide table. No direct change; drug-specific refinement remains unverified. |
| Everolimus | `scenarios/everolimus_des.sio::evr_pbpk28_params` | Fixed analogue of the historical rapamycin table. No direct change; inherits its provenance gap. |
| Tacrolimus | `scenarios/tacrolimus_oral.sio::tac_pbpk28_params`; parallel table in `drugs/tacrolimus.sio::tacrolimus_mean_params` | Fixed, literature-anchored table. No direct change. |
| Olanzapine | `scenarios/olanzapine_oral.sio::olz_pbpk28_params`; PBPK14 twin in `drugs/olanzapine.sio::olanzapine_pbpk_params` | Fixed R-R-conceptual pattern calibrated to Vd; not an independent R-R prediction. No direct change. |
| Clozapine | `scenarios/clozapine_oral.sio::clz_pbpk28_params` | Fixed R-R-conceptual estimate. No direct change. |
| Quetiapine | `scenarios/quetiapine_oral.sio::qtp_pbpk28_params` | Fixed R-R-conceptual estimate. No direct change. |
| Lithium | `scenarios/lithium_oral.sio::li_pbpk28_params` | Fixed ECF/measured-anchor rubric; the neutral R-R function is not applicable. No direct change. |
| Vancomycin | `scenarios/vancomycin_iv.sio::vnc_pbpk28_params` | Fixed ECF/measured-anchor table. No direct change. |
| Risperidone | `scenarios/risperidone_oral.sio::rsp_parent_params` and `rsp_oh9_params` | Fixed assumed parent and 9-OH tables. No direct change. |
| Aripiprazole | `scenarios/aripiprazole_oral.sio::arp_pbpk28_params` | Fixed assumed table. The source exists on this base; the older dossier note that its PR was queued is stale. No direct change. |
| Haloperidol | `drugs/haloperidol.sio::haloperidol_pbpk_params` | Fixed PBPK14 table calibrated to label Vd. The prior R-R attempt is explicitly superseded. No direct change. |
| Midazolam | `drugs/midazolam.sio::midazolam_params_with_clh`, wrapped by `midazolam_pbpk_params` | Fixed nominal PBPK14 table. No direct change; the neutral R-R function is not applicable to this weak base. |

The rapamycin and semaglutide parity fixtures also contain private copies of
their Kp arrays; the venlafaxine parity fixture calls the core constructors.
None calls `rr_neutral`. This repair therefore corrects the reference module
and its raw uncertainties; it does not retroactively change or validate all
scenario results.

## Affected, unaffected and not verified

Affected:

- `core/rodgers_rowland.sio::rr_neutral` raw Kp, analytical coefficients and
  `kp_unc_raw`.
- `core/rodgers_rowland.sio::rr_sqrt` and therefore the numerical square-root
  step in `rr_scaled_uncertainty`; this is independently discriminated by T8
  from the units and derivative-sign repair.
- The sign description in `rr_scaled_uncertainty` and the inverted Kp/Kp,uu
  identity comment in `epistemic_pbpk28.sio`.
- Any future caller that consumes `RRResult.kp` or `kp_unc_raw`.

Unaffected by the measured call graph:

- All 15 drug-family Kp constructors above, which use fixed arrays.
- The BBB Kp,uu observer, whose source separately computes unbound-tissue over
  unbound-plasma concentration.

Not verified by this card:

- The full Rodgers-Rowland neutral equation and tissue-composition table against
  the paper body; SIO-564 owns the composition/source audit.
- Scientific provenance of fixed arrays already marked pending in the SIO-550
  dossier.
- A publication figure manifest tying every historical plot to one of the
  constructors above. This report traces executable Kp sources, not unversioned
  graphics.
- Clinical predictive validity. The broad gate keeps two clinical items pending
  because `obs_n() == 0`; it does not promote them to validation.

## Reproducible receipts

All numerical execution ran under Slurm on x86_64. C++ used `-std=c++23`.
The compiler was rebuilt from the implementation-base source in the allocation:

- source-tracking seed SHA-256:
  `2c4642b04d5c6d4cb11ee6097abcd3a96f6a38c56243325ea6d94fbf31e4bbd9`
- Madaros SHA-256:
  `94e1c90e3a47cbcbab2cb52a6a815d4e8f5f4e045b94ae1fc92e61f84442a5c0`
- final C++23 oracle executable SHA-256:
  `a315b9660abbf4cd356eb05b947bc389aef153fca15730b033b108fa25940580`
- final C++23 oracle source SHA-256:
  `c315d0f872f77d7f8d3cef01762d1a7c3eb909d63228b4c0c569650996815d2b`
- final `rodgers_rowland.sio` SHA-256:
  `a373c63a81823cf6e37d62a34deb94c2047e0f4cc8cbba8ec02a77fe1b283608`
- final `epistemic_pbpk28.sio` SHA-256:
  `97f9e7af675861b47d33055aa46b4099d0329f1f054c24040a7a73de3cd5236f`
- The targeted allocation executed the same code at pre-review source hashes
  `fd5bed62832aa4bdf1201db45e24e13409bcb460697997becfad694858119fea`
  and
  `67472993228375669c8f71f8f7550a1cbb00ef48e1310f1480e74ab7661cba21`;
  the final hashes above differ only by reviewer-requested comments.
- final targeted receipt: Slurm allocation 22623, SHA-256
  `c5a5b6ad0e2609abec2693db1dfa3b055bdbcbb22734b2ee0b0d48c770b9de3c`
- final dissertation-suite receipt: Slurm allocation 22623, SHA-256
  `98854dc402fa620267e6c0593ffea61f2d2a78fbe6dbbd37f6dd6361ae5e5bf9`
- conversion negative-control receipt: allocation 22606, SHA-256
  `e0e5c451f396f2542dfc84661b0fcaffcd01002c73ea4e673809504957226575`
- square-root negative-control receipt: allocation 22616, SHA-256
  `ad4a46f718aabc24e21123c37447e607c7fdf5cde67acea2f6908c6120f71de5`

The final dissertation suite result is 50/53 passing, two explicitly pending
clinical cases, and one documented `FAIL_HONEST`. Those three non-passing
classifications are preserved and are not treated as regressions or successes.
