<!-- docs:meta
topic_id: repo.docs.audit.pbpk28-cn-rannacher-mass-balance-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk28-cn-rannacher-mass-balance-2026-09-26
-->

# PBPK28 CN floor-clamp mass injection: caller inventory, Rannacher fix, per-consumer gates

**Date:** 2026-09-26
**Branch:** `claude/pbpk28-cn-rannacher` (base `origin/main` @ `98315edcdb`)
**Engines:** committed `bin/souc-lean-single-x86_64` (CI engine), and Madaros built from `98315edcdb` with `make build-madaros` (md5 `5764851f3d229372e26aac1e951c95e1`, built by the sibling session that wrote the dispatch below)
**Companion dispatch:** the root-cause isolation, literature, and TR-BDF2 measurements are in the sibling dispatch `docs/audit/PBPK28_CN_FLOOR_CLAMP_MASS_INJECTION_DISPATCH_2026-09-26.md` (branch `audit/pbpk28-cn-floor-clamp-dispatch`). This document does not repeat that isolation. It covers the fix, the caller inventory, and the before/after numbers.

## Defect, in one paragraph

`pbpk28_full_cn_step` (`stdlib/darwin_pbpk/tsit5_pbpk28.sio`) is Crank–Nicolson. CN is A-stable but not L-stable. Its per-organ Schur solve conserves mass. However, after a non-smooth event, the stiff vascular↔interstitial modes ring with alternating sign, and the kernel's `if c < 0 { 0 }` floors turn that ringing into mass. Minimal repro (venlafaxine parent, `cl_central = 0`, 50 mg into blood, dt = 0.5 h): the body holds 85.065244 mg after 1 step and 89.100032 mg after 8. Under this branch's clamp-free Rannacher step the same case holds 50.000000 mg at every step, with `|ΔM|/M₀ ≤ 8e-14` and no negative concentration.

## Fix

The published kernel is **not modified**. `pbpk28_full_cn_step` stays bit-identical, so every parity reference, smoke test and golden output built on it is unchanged. The fix is a new module, `stdlib/darwin_pbpk/theta_pbpk28.sio`, and the bolus-IV consumers move onto it:

- `pbpk28_theta_step_sink_mut` — the same Schur elimination with implicit weight θ·dt and explicit weight (1−θ)·dt, an optional organ sink, and **no floor**.
- `pbpk28_rannacher_step_sink_mut` — two backward-Euler half steps on a start-up step, and clamp-free CN otherwise. `pbpk28_rannacher_startup_steps() = 2` (Giles & Carter: two CN steps replaced by four half steps). With one start-up step, the rapamycin bolus at dt = 0.5 h books 2.7e-5 mg of negative mass; with two it books 3.1e-10 mg.
- `pbpk28_theta_step_routed_sink_mut` / `pbpk28_trbdf2_step_routed_sink_mut` — the same steps with an optional input route (`input_organ` ∈ 1..13 moves dt·rel from the blood row to that organ's vascular row, e.g. portal first-pass into the liver). Added at the request of the venlafaxine lanes. The unrouted entry points are wrappers and print byte-identical output.
- `pbpk28_trbdf2_step_sink_mut` — L-stable TR-BDF2, built from two θ-steps. It is for stiff modes re-excited away from a dosing event. The venlafaxine scenario's owner uses it for the implicit CYP2D6 liver sink. It is not used by the consumers switched here.
- `PBPK28Ledger` — administered, eliminated and sink-removed mass, plus per-organ AUC, each booked with the method's own quadrature weights. So `M(T) + eliminated + sink − administered = M(0)` holds to rounding. Negative concentrations are **booked** (`min_conc`, `neg_mass`), not floored. `pbpk28_bolus_run_mut` fails the run (returns false) if booked negative mass exceeds 5e-7 mg.

Why Rannacher and not a smaller dt or bookkeeping: every affected consumer has a single dosing event (a bolus at t = 0). Rannacher start-up removes the injection completely at the consumers' own dt, so no consumer had to fall back to option 2 (a smaller dt) or option 3 (book the clamped mass and fail). No tolerance was loosened anywhere.

## Inventory: every caller, at its production dt

Residual = M(T) + eliminated − administered − M(0), with elimination booked in the method's own quadrature. For the shipped kernel, that quadrature is CL·trapezoid(C_b), which is exact for CN, so the residual is exactly the floor-injected mass. Source: `docs/audit/repro/pbpk28_rannacher_consumers_probe.sio` and `…_scenarios_probe.sio`, lean_single; Madaros agrees to rounding.

| caller | input | dt (h) | residual, shipped (mg) | rel. | status / residual after |
|---|---|---:|---:|---:|---|
| `epistemic_pbpk28.sio` `ep28_simulate_dt` (first-order GUM) | 5 mg bolus | 0.05 | **1.408197** | 0.28 | **switched** → 1.6e-12 mg (3.2e-13) |
| `epistemic_pbpk28_hessian.sio` `h28_simulate_auc` | 5 mg bolus | 0.1 | **2.585006** | 0.52 | **switched** → 1.5e-12 mg |
| `cumulants.sio` `m5_simulate_auc` | 5 mg bolus | 0.1 | **2.585006** | 0.52 | **switched** → 1.5e-12 mg |
| `validation/pbpk28_mc_cross_validation.sio` `mc28_auc_fast` | 5 mg bolus | 0.5 | **6.985257** | 1.40 | **switched** → 8.6e-13 mg |
| `validation/pbpk28_mc_prior_family_sweep.sio` `ms28_auc_fast` | 5 mg bolus | 0.5 | **6.985257** | 1.40 | **switched** → 8.6e-13 mg |
| `validation/pbpk28_sobol_pce.sio` `sp28_auc_fast`, rapamycin | 5 mg bolus | 0.5 | **6.985257** | 1.40 | **switched** → 8.6e-13 mg |
| same, semaglutide | 1 mg bolus | 0.5 | **0.282869** | 0.28 | **switched** → 1.3e-13 mg |
| `validation/pbpk28_rapamycin_clinical.sio` (oral depot, F·ka·A) | 2 mg oral | 0.05 | 3.2e-14 | 1e-13 | not affected — unchanged |
| `validation/pbpk28_semaglutide_clinical.sio` (SC depot) | 1 mg SC | 0.05 | 2.7e-13 | 4e-13 | not affected — unchanged |
| `scenarios/semaglutide_sc_depot.sio` (+ TMDD + PD) | 1 mg SC | 0.5 (smoke test) | 6.1e-14 | 8e-14 | not affected — unchanged |
| `scenarios/venlafaxine_xr.sio` parent + ODV | 75 mg XR | 0.5 | CN floors 3.6e-4 (parent), 2.7e-6 … 1.1e-4 (ODV) | 5e-6 | not edited here — see below |
| `tsit5_pbpk28.sio` `pbpk28_dt_convergence_auc` | bolus | caller | (bolus-biased like ep28) | — | **no callers**; kernel file left untouched |
| `tests/run-pass/darwin_pbpk28_smoke.sio`, parity refs (rapamycin, semaglutide, degenerate) | bolus | 0.001 | floors never fire at this dt (dispatch) | — | unchanged (existing gates) |
| `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio` | XR | 0.5 | own copy of the floored kernel; not measured here | — | unchanged (existing gate) |
| `docs/audit/repro/smoke_pbpk28_cn_imported.sio` | 1 step | 0.05 | — | — | Madaros SIGSEGV repro, not a consumer |

PBPK14 (`tsit5_pbpk14.sio`) is an explicit Tsit5 kernel, not CN, and has no negativity clamp. Out of scope.

**Venlafaxine XR.** At dt = 0.5 h, the scenario's mass balance is dominated by a separate defect: the explicit CYP2D6 formation step, where CL_form·dt/V_liver ≫ 1, followed by a liver floor. That creates 12.8 mg (PM), 66.3 mg (IM), 200.8 mg (NM) and 603.3 mg (UM) out of 75 mg released over 120 h. The CN floors add only the numbers in the table. That scenario is owned by other sessions: the formation sink by branch `claude/determined-kilby-3fc1ca`, and the TR-BDF2 wiring plus gut fix by `claude/competent-mcclintock-b137de`, built on this branch's `theta_pbpk28.sio` (commit `99eca282d`). This branch does not edit `venlafaxine_xr.sio`.

## Before → after, printed outputs of the switched modules

ep28 and the Hessian: lean_single, with Madaros matching every substantive field; only its fixed-point print of values below 1e-6 differs. MC, sweep and Sobol: lean_single only. Madaros exits 182 (`handles full`) on the MC and sweep modules, and the Sobol module does not type-check under Madaros (E259). All three failures happen on `main` as well as on this branch.

Exact reference: AUC_blood(0–∞) = Dose/CL = 5/12.4 = **0.403226** mg·h/L for rapamycin.

### `epistemic_pbpk28.sio` (first-order GUM, dt = 0.05)

| output | before | after |
|---|---:|---:|
| AUC_blood mean | 0.516790 | **0.403226** |
| AUC_blood SD | 0.228175 | 0.183456 (= Dose/CL²·√31.827344, analytic) |
| CV(AUC_blood) | 0.441523 | 0.454971 |
| AUC_liver | 2.284939 | 1.804839 |
| C_brain(24 h) | 5.10e-5 | 3.95e-5 |
| sens[0] cl_hepatic / sens[2] fu | 0.697376 / 0.301837 | 0.697609 / 0.301938 (analytic 0.697606 / 0.301942) |
| sens[4] kp_liver / sens[5] kp_kidney | 2.23e-4 / 1.11e-4 | 5e-24 / 3e-24 |
| AUC confidence | 0.671038 | 0.671068 |
| AUC at dt = 0.05 / 0.025 / 0.0125 | 0.516790 / 0.417365 / 0.403333 | 0.403226 at all three |
| tests | 9/9 | 9/9 (TEST 5, 7, 9 re-derived; see commit) |

The non-zero Kp share of the AUC_blood variance was entirely a solver artefact. AUC_blood over 0–168 h is (Dose − M(168 h))/CL, and M(168 h)/Dose ≈ 4e-18. TEST 5 now computes its dominant pins analytically from the priors, keeps the 1e-4 band, and requires the Kp tail to stay below 1e-12. It detects a lost Kp Jacobian column (#1497) through each Kp's share of its own organ endpoint.

### `epistemic_pbpk28_hessian.sio` (dt = 0.1)

| output | before | after |
|---|---:|---:|
| AUC_ref | 0.611694 | **0.403226** |
| var first / second order | 0.067863 / 0.087119 | 0.033656 / 0.043184 |
| ∂AUC/∂CL | −0.046164 | −0.032519 (analytic −Dose/CL² = −0.032518) |
| H[0][0] ∂²AUC/∂CL² | 0.007454 | 0.005245 (analytic 2·Dose/CL³ = 0.005245) |
| H[3][3] (kp_brain) | −0.261626 | −4.4e-9 (rounding noise) |
| ρ_literal(CL) / ρ_normalized(CL) | 0.380433 / 0.072365 | 0.380000 / 0.072200 (analytic σ/CL, ½(σ/CL)²) |
| ρ_literal / ρ_normalized, kp_brain | 0.349828 / 0.061190 | 0 / 0 (guarded, see below) |
| \|mean shift\| / AUC_ref | 0.192536 | 0.206996 |

Once the Kp columns are rounding noise, the nonlinearity ratios would print noise divided by noise: ρ_normalized(kp_adipose) came out as 1,075,556 before the guard was added. `h28_first_order_resolvable` now reports ρ = 0 when |cᵢ|σᵢ ≤ 1e-9·|AUC_ref|. That floor is about 25–50× above the finite-difference rounding floor and about 10⁷× below the smallest real effect (cl_renal). The Hessian self-test gained TEST 7 (mass identity); 7/7 and 3/3 pass.

### `validation/pbpk28_mc_cross_validation.sio` (LogNormal, N = 2000, seed 1729, dt = 0.5)

| output | before | after |
|---|---:|---:|
| MC mean AUC | 1.086296 | 0.476478 |
| u_GUM / u_Hessian / u_MC | 0.228175 / 0.295160 / 0.357945 | 0.183456 / 0.207808 / **0.211790** |
| rel_GUM | 0.362543 | 0.133783 |
| rel_Hess | 0.175405 | **0.018801** |
| verdict | `…_LOGNORMAL_OUTPUT` (neither criterion met) | `…_LOGNORMAL_HESSIAN_PASS` (Hessian ≤ 0.10 met) |

The comparison used to set an MC at dt = 0.5 (+140% bias) against a GUM at dt = 0.05 (+28%) and a Hessian at dt = 0.1 (+52%). Its "strongly nonlinear regime" reading was largely solver artefact. With one unbiased kernel for all three, the second-order GUM agrees with MC to 1.9%.

### `validation/pbpk28_mc_prior_family_sweep.sio` (dt = 0.5)

| prior | u_MC before → after | rel_GUM before → after | rel_Hess before → after | Hessian ≤ 0.10 |
|---|---|---|---|---|
| Gaussian | 0.935626 → 0.564984 | 0.756126 → 0.675290 | 0.684532 → 0.632187 | NO → NO |
| LogNormal | 0.357945 → 0.211790 | 0.362542 → 0.133783 | 0.175404 → **0.018800** | NO → **YES** |
| TruncNormal | 0.483584 → 0.288417 | 0.528159 → 0.363920 | 0.389641 → 0.279486 | NO → NO |
| verdict | `MC_PRIOR_FAMILY_SWEEP_OUTPUT` | | | → `MC_PRIOR_FAMILY_SWEEP_PASS` |

The module's printed scientific conclusion, that no prior family resolves the GUM/MC discord, **no longer holds**: the LogNormal prior meets the Hessian criterion.

### `cumulants.sio` (M5 fourth-order budget, dt = 0.1; scratch probe, because `tests/run-pass/pbpk28_m5_gum_4th_order.sio` does not compile on main: E035, and E259 under Madaros)

| output | before (u_MC pin 0.357945) | after (u_MC pin 0.211790) |
|---|---:|---:|
| u_1st | 0.260506 | 0.183456 |
| u_2nd_hessian | 0.295160 | 0.207808 |
| u_total (4th order) | 0.378674 | 0.266505 |
| rel_hess_residual | 0.175404 | 0.018801 |
| rel_fourth_residual | 0.057910 | 0.258345 |
| "fourth-order improves on the Hessian" | YES | **NO** |

The canonical u_MC pin was re-derived from the rerun above and now lives in `m5_pbpk28_u_mc_canonical()`. The M5 claim that the fourth-order cumulant budget improves agreement with MC does **not survive** the corrected kernel. The assertion in the m5 test was left as it is: it will report `…_OUTPUT` honestly once that test compiles again.

### `validation/pbpk28_sobol_pce.sio` (dt = 0.5)

| output (Saltelli N = 512) | before | after |
|---|---:|---:|
| rapamycin S_i CL_renal / fu | 0.029295 / 0.049450 | 0.018363 / 0.094698 |
| rapamycin S_i kp_liver / kp_kidney | 3e-6 / 5.2e-5 | ~0 / 1.1e-9 |
| rapamycin S_Ti CL_hepatic / kp_liver | 1.000000 / 1.2e-9 | 1.000000 / 1.3e-15 |
| semaglutide S_i fu / kp_brain / kp_kidney | 0.986151 / 0.009152 / 0.018073 | 0.950518 / 0.006599 / 0.009373 |
| semaglutide S_Ti CL_proteolytic / fu / kp_liver | 0.689513 / 0.583407 / 0.002126 | 0.657361 / 0.555677 / 0.004522 |
| semaglutide ρ_add = Σ S_i | 1.013376 | 0.966490 |
| tests | 5/5 + 5/5 | 6/6 + 6/6 |

The header's old claim, a "~0.5% bias that cancels in variance ratios", did not describe the kernel. The injection depends on the sampled parameters, which is visible in the Kp indices. Pre-existing and unchanged: the rapamycin first-order S_i[CL_hepatic] prints 0.000000 while S_Ti[CL_hepatic] = 1.000000. That looks like an estimator issue in `epistemic::sobol` and was not examined here. Runtime: the ledger bookkeeping makes each model evaluation about 1.6–2× slower under lean_single (MC 361 s → 586 s; Sobol 1483 s → 2331 s).

## Gates added

| gate | where | what it pins |
|---|---|---|
| `tests/run-pass/pbpk28_theta_closed_system_mass.sio` | CI (run-pass), both engines | CL = 0 closed systems, 3 profiles × dt ∈ {0.5, 0.1, 0.05} × {CN, Rannacher, TR-BDF2}: \|ΔM\|/M₀ at every step ≤ `pbpk28_mass_tol_rel`; no booked negative mass for Rannacher/TR-BDF2; CL + liver sink + input closes the ledger, both with blood input and with the input routed into the liver. |
| `tests/run-pass/pbpk28_consumer_mass_balance.sio` | CI (run-pass) | each switched consumer's own simulation path at its production dt closes M(T) + CL·AUC − Dose to rounding, with no visible negative mass. lean_single only: under Madaros the Sobol module does not type-check on main (pre-existing E259), so this import cannot build there. |
| ep28 TEST 7 / 9, Hessian TEST 7, MC TEST 6, sweep TEST 7, Sobol TEST 6 (both drugs) | module self-tests (`dissertation_pbpk_suite_gate.sh`) | the same identity inside each module's own run; MC and sweep return non-zero on failure. |

Bounds (`theta_pbpk28.sio`): `|residual|/scale ≤ max(1e-12, sub_steps·1e-15)`, a heuristic rounding budget for an identity that is exact in real arithmetic. The math review flagged it as heuristic, not a proven FP bound, and the wording says so. Measured residuals sit 3–30× below it, and the defect is 10⁹× above it. Booked negative mass must stay ≤ 5e-7 mg, half the last digit of a 6-decimal mg print. That is a resolution bound: no second-order linear method is positivity-preserving at these dt (Bolley & Crouzeix 1978).

## Not done here (requires its own dispatch)

- `docs/dissertation/results/*` still record the floor-biased numbers, and they are **stale**: `pbpk28_epistemic_runs_v1.txt` (0.516790, 2.284939, 0.611694), `pbpk28_epistemic_v1.md`, `m5_gum_4th_order_v1.md` (u_MC 0.357945), `m6_prior_update_v1.md` (u_MC 0.549197 → 0.357945; both values predate this fix), `runs/m6_full_stack_v1.txt`, `mc_cross_validation_lognormal_v2.md` (MC mean 1.204004 and its "Jensen" attribution), and anything cited from them. Regenerating them, and the dissertation prose, is external-facing work: it needs the offload triad (`bin/llm-offload --raw <draft> deepseek xai gemini`) and was not authorised here.
- Repeated dosing: every switched consumer has one bolus at t = 0. The sibling dispatch measured that Rannacher must be re-triggered after each dose, and that for depot forcing at dt = 0.5 TR-BDF2 behaves better. Any future multi-dose consumer should take that into account.
- `tsit5_pbpk28.sio::pbpk28_dt_convergence_auc` has no callers and still runs the floored kernel. The kernel file is owned by another session and was left untouched by design.
- The venlafaxine XR scenario and its parity reference: see Inventory.

## Madaros hazard observed by a sibling session (not triggered here)

Madaros built from `98315edcd` aliases arrays on whole-struct reassignment (`z = x`, `*dst = src`). Only `let y = x` and element-wise copies are safe. `theta_pbpk28.sio` never reassigns a struct. Stage saves are element-wise (`p28t_save`, `p28t_bdf2_combine`), and ledgers and states are fresh `var` initialisations. Every probe and test in this branch prints identical substantive output on both engines.

## Reproduce

On the workspace, from a worktree of this branch:

```bash
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN
export SOUNIO_STDLIB_PATH=$PWD/stdlib; ulimit -s 524288
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile docs/audit/repro/pbpk28_rannacher_consumers_probe.sio -o /tmp/c.elf && /tmp/c.elf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile docs/audit/repro/pbpk28_rannacher_scenarios_probe.sio -o /tmp/s.elf && /tmp/s.elf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/pbpk28_theta_closed_system_mass.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/pbpk28_consumer_mass_balance.sio
```

The consumers probe runs in about 50 s; the scenario probe in about 7 s. The MC module takes about 6 min under lean_single, and the Madaros build of it hits the known `handles full` wall (exit 182) on both base and branch.

## Literature

Cited from the author's knowledge; the scite connector was unavailable in this session. Verify before quoting externally.

- Rannacher R. Finite element solution of diffusion problems with irregular data. *Numer. Math.* 43 (1984) 309–327.
- Giles MB, Carter R. Convergence analysis of Crank–Nicolson and Rannacher time-marching. *J. Comput. Finance* 9(4) (2006) 89–112.
- Bank RE et al. Transient simulation of silicon devices and circuits. *IEEE Trans. CAD* 4 (1985) 436–451.
- Hosea ME, Shampine LF. Analysis and implementation of TR-BDF2. *Appl. Numer. Math.* 20 (1996) 21–37.
- Bolley C, Crouzeix M. Conservation de la positivité lors de la discrétisation des problèmes d'évolution paraboliques. *RAIRO Anal. Numér.* 12 (1978) 237–245.
- Shampine LF. Conservation laws and the numerical solution of ODEs. *Comput. Math. Appl.* 12B (1986) 1287–1296.
