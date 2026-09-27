<!-- docs:meta
topic_id: repo.docs.audit.darwin-pbpk-blood-as-plasma-bbb-driver-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.darwin-pbpk-blood-as-plasma-bbb-driver-2026-09-26
-->

# Dispatch: darwin_pbpk drives the BBB and reports "plasma" with whole-blood concentration (2026-09-26)

**Status:** OPEN. Investigation only; no code changed. The fix changes
absolute concentrations, brain exposures and PD endpoints, so it needs an
operator decision and a math review (`CLAUDE.md` §10) first. Raised in a
math review and a code review of PR #2698. The sign correction for
`rb_ratio > 1` comes from lane `claude(gracious-bardeen)`.

## Claim

`PBPKState14.blood` is a **whole-blood** concentration: `pbpk_ode` converts
it with `c_plasma = c_blood / prm.rb_ratio` (`tsit5_pbpk14.sio:381`).
Every PBPK-to-BBB coupling in `stdlib/darwin_pbpk` nevertheless passes
`sys_st.blood` where plasma is expected:

- **BBB driver.** It is the argument `c_plasma` of `bbb_rk4_step` /
  `bbb_ode`, which forms `fu_plasma * c_plasma` (`bbb/bbb_core.sio:87–92`).
- **Reported "plasma" series.** It is stored as `trace.plasma[]`, and the
  "unbound plasma" AUC is `fu_plasma * AUC(blood)`.

The unbound driving concentration and the reported unbound-plasma exposure
are therefore off by the factor `rb_ratio`.

## Evidence

**No caller converts.** All six couplings take `sys_st.blood` unchanged:

| Module | Driver variable | Trace |
|---|---|---|
| `bbb/bbb_coupled.sio:112,136` | `c_plasma_prev`, `c_plasma_now` | `trace.plasma` |
| `scenarios/oral_rapamycin_bbb.sio:103,135` (`oral_bbb_run`) | `c_prev`, `c_now` | `trace.plasma` |
| `scenarios/steady_state_runner.sio:160,201` | `c_prev`, `c_now` | `trace.plasma`; also the solver-step AUC accumulator `auc_p` (`:178`) behind `auc_plasma_u` (`:226`) |
| `scenarios/des_sirolimus_bbb.sio:127,156` | `c_prev`, `c_now` | `trace.plasma` |
| `scenarios/oral_haloperidol_bbb.sio:132,178,260` | `c_prev`, `c_now` | `trace.c_plasma` |

**The intent is plasma.**
- `bbb_ode` names its argument `c_plasma`.
- `bbb_coupled.sio`'s header describes "a plasma trajectory", and its driver
  variables are named `c_plasma_*`.
- `kpuu_auc` is documented as AUC(C_isf_u) / AUC(C_plasma_u) (Fridén
  framework).
- `bbb_rapamycin_params().fu_plasma = 0.08` is an in-vitro **plasma**
  dialysis value (Schreiber 1991, `bbb/bbb_rapamycin.sio:83–89`).

The one reference implementation that converts explicitly,
`tests/run-pass/dissertation_pbpk28_parity_ref_haloperidol.sio:241`
(`plasma = c[0] / rb_ratio()`), does so with `rb_ratio = 1`.

**Chronology.** `rapamycin_mean_params().rb_ratio = 0.58` dates from
`8ee3df3d6` (2026-03-05). The BBB module came later, in `31a266a89`
(2026-04-22), so for the mean model the shortcut was wrong from the start.
`rapamycin_fullvd_params().rb_ratio = 36` (Yatscoff 1995) was set the next
day (`d37bb9c2d`).

**Part of it is deliberate.** The systemic calibration commits
(`b76ebb67f`, `d37bb9c2d`) compare the blood compartment with clinical
sirolimus values: "C_max ~2 ng/mL vs clinical 5 ng/mL". Sirolimus
monitoring is done in whole blood, so reporting blood is right for that
comparison. What is wrong is:
- calling it plasma;
- multiplying it by a plasma unbound fraction;
- driving the BBB with it.

## Impact by drug

The size and sign follow from plasma = blood / rb.

| Parameter set | `rb_ratio` | Unbound plasma AUC, BBB driver and (linearly) ISF/ICF exposure |
|---|---:|---|
| haloperidol, midazolam | 1.0 | unaffected |
| olanzapine | 0.96 | ~4% low |
| `rapamycin_mean_params`, `ep14_rapamycin_params` and derivatives (pop_sim, biomaterial_release, des_sirolimus, gum_vs_mc, rapamycin_clinical) | 0.58 | **~42% low** |
| `rapamycin_fullvd_params` | 36.0 | **~36× high** |
| tacrolimus | 15.0 | 15× high wherever a BBB or "unbound plasma" readout uses it |

The factor applies to quantities linear in the driver: unbound plasma
concentration and AUC, and the BBB driver. Because the BBB chain is linear,
it also applies to ISF and ICF exposure. It does **not** carry over to PD:
the Hill response saturates, so PD endpoints change by less than the
driver. On the mean model, inhibition rises 68–69% while the driver rises
×1.724 (measurement 1). PD must be re-simulated per model, not rescaled.

Scale-free quantities survive: Kp,uu from the AUC ratio, cell-to-ISF ratios,
t_max lag and accumulation ratios. This holds **only because** the
driver-to-ISF/ICF chain is linear and time-invariant (passive, linear BBB
kinetics) and `rb_ratio` is a positive constant. Saturable transport,
Michaelis–Menten efflux or nonlinear binding would break it. The exact
common ×1.724 scaling in measurement 1 confirms linearity for this
parameter regime, not in general.

The steady-state runner's per-step AUC accumulator was added by PR #2698.
It integrates the same whole-blood value, so the conversion must cover it
too. Its code and its quadrature test now say so explicitly. That test
compares two whole-blood integrals, and a constant `rb_ratio` rescaling
leaves its relative error unchanged, so it stays valid after the fix.

## Measurements

All on Madaros md5 `5764851f`, in a scratch copy of `stdlib/`. No repository
file was changed.

**1. Dissertation PD demo.** `examples/dissertation_oral_pd_demo.sio`
(rapamycin mean, `rb_ratio = 0.58`), with `oral_bbb_run`'s driver and trace
converted to `blood / rb_ratio` (three lines):

| Output | Blood driver (current) | Plasma driver | Change |
|---|---:|---:|---:|
| C_max_plasma (mg/L) | 0.001755 | 0.003027 | ×1.7248 displayed; exact scale 1/0.58 = 1.7241 |
| C_plasma @ 24 h | 0.001387 | 0.002391 | ×1.72 |
| C_icf @ 24 h | 0.000633 | 0.001091 | ×1.72 |
| max_inhibition | 0.036075 | 0.060614 | **+68%** |
| mean_inhibition | 0.020975 | 0.035485 | **+69%** |
| inhibition_AUC | 1.006824 | 1.703285 | **+69%** |

**2. The current validation cannot see it.**
`tests/stdlib/darwin_pbpk/bbb/test_bbb_validation_rapamycin.sio`
(`ep14_rapamycin_params`, rb 0.58) was run with `bbb_coupled_run`'s driver
and trace converted the same way:
- original and converted both compile cleanly, run with rc=0 and print
  `BBB_VALIDATION_RAPAMYCIN_OK`;
- its four benchmarks (Kp,uu from the AUC ratio, cell/ISF ratio, ISF t_max
  lag, non-negativity) are all scale-free.

A first attempt at this measurement referenced a variable that does not
exist in `bbb_coupled_run` (`sys_prm` instead of `systemic_prm`). It failed
to compile, and its "failure" was discarded.

## Proposed fix (not applied)

1. At each of the six couplings, drive the BBB with
   `c_plasma = sys_st.blood / sys_prm.rb_ratio`.
2. Keep a whole-blood series for clinical comparisons, and name it as such
   (for example `trace.blood`). Store plasma separately.
3. Compute the unbound plasma AUC as `fu_plasma * AUC(blood) / rb_ratio`.
4. Add a guard test with `rb_ratio != 1`. At steady state under constant
   infusion, `fu_isf * C_isf` must approach
   `kpuu_brain * fu_plasma * C_blood / rb_ratio`. That is an absolute
   check, which the current benchmarks lack. The identity holds exactly
   only:
   - at true steady state (the test must allow for the approach);
   - with `C_isf` the **total** ISF concentration;
   - if `kpuu_brain` is the model's own implied unbound steady-state ratio,
     that is, with no separate active-transport asymmetry making the
     realised ratio differ from the parameter.

   Stated this way, the guard tests the wiring, not a parameter
   definition.
5. Re-derive every dissertation-facing BBB and PD number (before/after
   table) and run `bin/llm-offload -t math-review`.

## Decision needed

- **Q1.** Were the BBB parameters (`ps_bbb`, `kpuu_brain`, `ps_mem`,
  `kpuu_cell`) or the PD Hill parameters fitted against a blood-driven
  model? If so, converting the driver means refitting them, not just
  correcting the scale.
- **Q2.** Which quantity should the dissertation report as "plasma"
  concentration for sirolimus: whole blood (clinical convention) or plasma?

## Math review (2026-09-27)

Default fan-out `bin/llm-offload -t math-review`. Grok 4.7 (xAI direct,
after the gateway leg timed out) and Kimi K3 (gateway) both returned
verdicts. zai was rate-limited (1313) and local was down; both are errors,
not passes.

Both confirm the core claim: the driver and the reported unbound exposure
are off by exactly `rb_ratio`, and the sign follows `rb − 1`. Accepted and
applied:
- Grok, **overreach**: the impact table applied the factor to PD, but the
  Hill response is saturating (+68–69% measured versus ×1.724). The rows now
  cover linear quantities only, with a PD note.
- Grok, **wrong**: "×1.725 (= 1/0.58)". In fact 1/0.58 = 1.7241, and 1.7248
  is the ratio of the rounded displayed values.
- Both, **tightenable**: the scale-free claim needs linear, time-invariant
  BBB kinetics.
- Kimi, **tightenable**: the guard identity's hypotheses (steady state, total
  C_isf, `kpuu_brain` equal to the implied ratio). Now stated.

