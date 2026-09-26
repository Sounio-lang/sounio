<!-- docs:meta
topic_id: repo.docs.audit.pbpk28-cn-floor-clamp-mass-injection-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk28-cn-floor-clamp-mass-injection-dispatch-2026-09-26
-->

# PBPK28 CN transport step injects mass through its floor clamps — dispatch

**Date:** 2026-09-26
**Base:** `origin/main` @ `98315edcdb` (remote worktree `/workspace/worktrees/claude-pbpk28-cn-mass`, detached)
**Engines:** Madaros built from that commit with `make build-madaros` (`artifacts/self-hosted/madaros`, md5 `5764851f3d229372e26aac1e951c95e1`); cross-checked against committed `bin/souc-lean-single-x86_64` (md5 `0cb08380bfa77fe2905271ea35c341cf`)
**Owner:** unassigned (`stdlib/darwin_pbpk/tsit5_pbpk28.sio`, shared by every PBPK28 consumer)
**Status:** evidence recorded; root cause **confirmed** by isolation; fix **proposed, not applied**. No file under `stdlib/` or `self-hosted/` is changed by this dispatch.
**Found while:** fixing an unrelated venlafaxine ODV formation-step bug (reported as commit `5e57fad6` on branch `claude/venlafaxine-odv-mass-balance`; at the time of writing that commit is in neither this checkout, the workspace checkout nor `origin`). That bug and this one are separate; do not conflate them.

## Summary

`pbpk28_full_cn_step` does not conserve mass at the step sizes most consumers use.
The per-organ Schur elimination is **correct**: with the clamps removed, the step
conserves total mass to about 5 × 10⁻¹⁴ relative, as a linear one-step method must.
The defect is the three `if x < 0.0 { 0.0 } else { x }` floors in
`pbpk28_cn_apply_schur` (`tsit5_pbpk28.sio:52,57,58`).

Crank–Nicolson is A-stable but not L-stable. With the kernel's half-step h = dt/2
(`tsit5_pbpk28.sio:77`), its amplification factor is R(z) = (1 − z)/(1 + z) with
z = hλ = dt·λ/2, where λ > 0 is a decay rate such as (Q+PS)/V_v. R tends to −1 as
z grows. The PBPK28 vascular blocks are extremely stiff (rapamycin lung:
(Q+PS)/V_v ≈ 3.8 × 10⁴ h⁻¹), so every practical dt leaves a weakly damped,
sign-alternating mode (R ≈ −0.998 at dt = 0.05 h). That mode drives
concentrations negative. The floor then zeros them, which adds mass that was
never dosed.

In the dissertation's own rapamycin configuration this inflates blood AUC(0–168 h) by
**+28% at dt = 0.05 h**, **+52% at dt = 0.1 h** and **+140% at dt = 0.5 h**.
Those are the step sizes of the first-order GUM, the Hessian budget, and the
MC / Sobol-PCE / prior-sweep modules. The first-order GUM mean recorded in
`docs/dissertation/results/pbpk28_epistemic_runs_v1.txt` (`AUC_blood … 0.516790`)
and the Hessian reference (`AUC_ref … 0.611694`) match, to every printed digit, the
clamp-biased values measured here. The mass-balance value is Dose/CL = 5/12.4 = **0.403226**.

No existing gate tests a closed system, or tests the exact discrete mass identity
at a production step size. Every PBPK28 mass check this audit found passes under
the defect.
Section [Why no gate caught it](#why-no-gate-caught-it) explains each one.

## Repro (minimal)

```sounio
use darwin_pbpk::core::pbpk28_params::*
use darwin_pbpk::tsit5_pbpk28::*

fn main() -> i32 with IO, Mut, Div, Panic {
    var p = pbpk28_params_venlafaxine_parent()
    p.cl_central = 0.0                       // closed system: no sink
    var st = pbpk28_state_zero()
    st.cv[0] = 10.0                          // 50 mg, all in blood
    var i: i32 = 0
    while i < 6 {
        st = pbpk28_full_cn_step(st, p, 0.0, 0.5)   // no source
        println(pbpk28_total_mass(st, p))
        i = i + 1
    }
    0
}
```

The output is identical on both engines:
`85.065244, 85.561513, 88.694805, 88.695442, 89.100032, 89.100032`.
A closed system must print `50.000000` six times.

Full probes, committed alongside this dispatch:

| Probe | What it measures |
|---|---|
| `docs/audit/repro/pbpk28_cn_clamp_mass_probe.sio` | Repro plus isolation. Runs the stdlib kernel, a bit-exact in-probe copy with clamps ON (also recording the pre-clamp state), and the same copy with clamps OFF. |
| `docs/audit/repro/pbpk28_cn_clamp_blast_probe.sio` | 4 drug profiles × 5 dt values, 5 mg IV bolus, nominal CL, 168 h. Reports AUC with clamps on and off, clamp-injected mass, and the exact discrete mass identity. |
| `docs/audit/repro/pbpk28_cn_fix_candidates_probe.sio` | The shipped kernel against four clamp-free linear schemes on all four `ep28` endpoints, rapamycin and semaglutide. |

## Isolation: the floors account for 100% of the drift

Output of `pbpk28_cn_clamp_mass_probe.sio`, case 1 (venlafaxine parent, CL = 0, dt = 0.5 h):

| step | M, shipped kernel (mg) | mass injected by floors (mg) | most negative pre-clamp entry | M, clamps OFF (mg) | Schur-step relative drift |
|---:|---:|---:|---:|---:|---:|
| 1 | 85.065244 | 35.065244 | −7.013049 | 50.000000 | −1.7e-14 |
| 2 | 85.561513 | 0.496269 | −0.504540 | 50.000000 | −4.0e-15 |
| 3 | 88.694805 | 3.133292 | −0.626658 | 50.000000 | −1.1e-14 |
| 4 | 88.695442 | 0.000637 | −0.002211 | 50.000000 | −9.3e-15 |
| 5 | 89.100032 | 0.404590 | −0.080918 | 50.000000 | −9.0e-15 |
| 6 | 89.100032 | 0.000000 | +0.007239 | 50.000000 | +4.8e-16 |
| **Σ** | **growth 39.100032** | **39.100032** | | final drift −3.1e-14 | |

- The in-probe copy with clamps ON equals the stdlib kernel **bit for bit**
  (`std_vs_copy_maxdiff = 0` at every step). The isolation therefore tests the
  shipped arithmetic, not a paraphrase of it.
- The "Schur-step drift" column is mass before the floors, minus mass after the
  previous step. It stays at roundoff: the Schur algebra is conservative.
- Total growth equals total floor injection to every printed digit.
- With clamps OFF, mass stays at 50.000000, but the state rings down to −8.9 mg/L.
  The floor was hiding that oscillation, not preventing it.
- Case 2 (PS = 0 everywhere, pure Q-convection): growth 1.093577 mg equals injection
  1.093577 mg. The "flat, then growing" alternation in the original report is the
  sign-alternating mode crossing zero on every other step.

## Mechanism

The system is linear with constant coefficients: ẋ = A x + b. Let W be the
diagonal matrix of compartment volumes. Then **1**ᵀW A = −CL·e_bloodᵀ: the
Q-coupling and PS-coupling columns telescope, leaving only the blood-row
elimination. Any linear one-step method preserves this linear invariant exactly
(Shampine 1986). For CN it gives the **exact discrete identity**

  M(t_{n+1}) = M(t_n) − CL · dt · (C_b,n + C_b,n+1)/2 + dt · rel_mid

so the trapezoidal blood AUC satisfies M(T) + CL·AUC_trap = Dose (+ released mass)
to roundoff, at any dt. With clamps off, the measured residual is ≤ 5 × 10⁻¹³ in
every cell of up to 3360 steps, and at most 4.0 × 10⁻¹² over all 20 (drug, dt)
cells. The largest value is at 168 000 steps (semaglutide, dt = 0.001 h).

The floor is a nonlinear projection that is not mass-conserving, so it breaks the
identity. The floor fires only when the scheme produces negatives, and CN produces
them here because of stiffness:

| profile | stiffest vascular block | (Q+PS)/V_v (h⁻¹) | R(z) at dt = 0.5 / 0.1 / 0.05 / 0.001 h | CN positivity bound 2/λ (h) |
|---|---|---:|---|---:|
| rapamycin | lung (5) | 37 647 | −0.9998 / −0.9989 / −0.9979 / −0.899 | 5.3e-5 |
| venlafaxine parent | lung (5) | 28 973 | −0.9997 / −0.9986 / −0.9972 / −0.871 | 6.9e-5 |
| semaglutide | gut (8) | 2 951 | −0.9973 / −0.9865 / −0.9733 / −0.192 | 6.8e-4 |
| blood row, all profiles | ΣQ_i / V_b | 150 | −0.948 / −0.765 / −0.579 / — | — |

Here z = (dt/2)·λ, following the kernel's h = dt/2. The last column is the largest
dt at which the diagonal of CN's explicit factor I + (dt/2)A stays non-negative,
the usual sufficient condition for CN to preserve positivity. It is below one
second for rapamycin. By the Bolley–Crouzeix theorem, no linear method above first
order preserves positivity unconditionally (Bolley & Crouzeix 1978). Two separate facts
follow. The floor breaks conservation because it is a non-conservative map. And no
second-order linear replacement can *guarantee* positivity at these dt without it.
The fix therefore has to remove the floor and handle any residual negativity
explicitly, rather than look for a scheme that makes the floor safe.
The kernel header's claim of "no operator-splitting bias" is true of the algebra and
false of the shipped step.

## Blast radius

### Consumers and the dt each one runs

| consumer | dt (h) | dissertation role |
|---|---:|---|
| `epistemic_pbpk28.sio` (`ep28_simulate`) | 0.05 | first-order GUM budget, all four endpoints |
| `epistemic_pbpk28_hessian.sio` | 0.1 | second-order (Hessian) budget |
| `cumulants.sio` (`m5_simulate_auc`) | 0.1 | cumulant / M5 layer |
| `validation/pbpk28_mc_cross_validation.sio` | 0.5 | MC ↔ GUM cross-validation (also "M6 canonical MC truth", `cumulants.sio:447`) |
| `validation/pbpk28_mc_prior_family_sweep.sio` | 0.5 | prior-family sweep |
| `validation/pbpk28_sobol_pce.sio` | 0.5 | Sobol / PCE indices (§4.10) |
| `validation/pbpk28_rapamycin_clinical.sio`, `…_semaglutide_clinical.sio` | 0.05 | clinical-profile validation (oral / SC depot) |
| `scenarios/venlafaxine_xr.sio` | 0.5 | venlafaxine XR parent + ODV |
| `scenarios/semaglutide_sc_depot.sio` | caller-supplied | SC depot + TMDD + PD |
| `tests/run-pass/darwin_pbpk28_smoke.sio`, parity refs (rapamycin, semaglutide) | 0.001 | gates; the floor never fires at this dt |
| `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio` | 0.5 | a separate copy of the kernel with the same floors (`:305–311`) |

### Measured bias (5 mg IV bolus into blood, nominal CL, 0–168 h, `pbpk28_cn_clamp_blast_probe.sio`)

Reference: clamps OFF, which satisfies the exact identity and reproduces
AUC_blood = (Dose − M(168 h))/CL at every dt.

| profile | dt = 0.5 | dt = 0.1 | dt = 0.05 | dt = 0.01 | dt = 0.001 |
|---|---:|---:|---:|---:|---:|
| rapamycin, AUC_blood bias | **+139.7%** | **+51.7%** | **+28.2%** | +0.022% | 0 |
| — mass injected (mg, of 5 mg dosed) | 4.394 | 2.316 | 1.330 | 0.0011 | 0 |
| semaglutide, AUC_blood bias | +28.3% | +6.6% | +3.2% | +0.020% | 0 |
| venlafaxine parent, AUC_blood bias | +349.0% | +93.8% | +43.9% | +0.12% | 0 |
| venlafaxine ODV, AUC_blood bias | +212.1% | +65.4% | +31.7% | +0.10% | 0 |

All four `ep28` endpoints for rapamycin, shipped kernel against the dt = 0.001 h
reference (`pbpk28_cn_fix_candidates_probe.sio`, method 0):

| endpoint | reference | dt = 0.05 (GUM) | dt = 0.1 (Hessian) | dt = 0.5 (MC / Sobol) |
|---|---:|---:|---:|---:|
| AUC_blood (mg·h/L) | 0.403226 | 0.516790 (+28.2%) | 0.611694 (+51.7%) | 0.966553 (+139.7%) |
| AUC_liver (mg·h/L) | 1.804839 | 2.284939 (+26.6%) | 2.641239 (+46.3%) | 3.391329 (+87.9%) |
| AUC_kidney (mg·h/L) | 1.487097 | 1.882824 (+26.6%) | 2.176358 (+46.3%) | 2.794287 (+87.9%) |
| C_brain(24 h) (mg/L) | 3.9493e-5 | 5.1023e-5 (+29.2%) | 5.6927e-5 (+44.1%) | **0** (floored) |

### Recorded dissertation numbers that carry the bias

- `docs/dissertation/results/pbpk28_epistemic_runs_v1.txt:14` — `AUC_blood(0-168h) mean 0.516790`,
  and `:43` `AUC_liver 2.284939`. These equal the clamp-biased dt = 0.05 values above,
  to every printed digit.
- `…/pbpk28_epistemic_runs_v1.txt:70` and `docs/dissertation/results/pbpk28_epistemic_v1.md:96` —
  Hessian `AUC_ref = 0.611694`. This equals the clamp-biased dt = 0.1 value.
- `docs/dissertation/results/mc_cross_validation_lognormal_v2.md:38` — MC mean
  AUC 1.204004, "Jensen upward bias from convexity of 1/CL". The MC runs at
  dt = 0.5 h, where the nominal-parameter AUC is already 0.966553 against a
  mass-balance value of 0.403226. The MC-vs-GUM comparison sets a dt = 0.5 MC
  (+140% bias) against a dt = 0.05 GUM (+28%) and a dt = 0.1 Hessian (+52%). Its
  discrepancy therefore contains solver artefact of unknown share, and the Jensen
  attribution cannot stand until it is rerun. *Not re-derived here: this dispatch
  does not rerun the MC.*
- Sensitivity coefficients, variances, CV(AUC), the confidence score and the Sobol /
  PCE indices all come from finite differences or samples of the biased map. The
  bias depends on the parameters. The stiff rates are set by Q, PS and V_v, while Kp
  and CL change the modal residues and how much mass is left to be clipped. So the
  derivatives are contaminated too, not merely offset. Their magnitude is
  **not measured** in this dispatch.

### Not affected

- The PBPK28 parity gate's case 1 (REF dt = 0.001 ↔ ALT dt = 0.0005) and the smoke
  test. They run where the floor never fires: 0 clamp events measured at
  dt = 0.001 h for all four profiles.
- PBPK14 (`tsit5_pbpk14.sio`), a different kernel. Not examined here.

## Why no gate caught it

| check | what it tests | why it passes under the defect |
|---|---|---|
| `darwin_pbpk28_smoke.sio` invariant 2 | M_{n+1} ≤ 1.0001·M_n at each step | dt = 0.001 h, where the floor never fires. CL > 0 masks small injections. |
| `dissertation_pbpk28_parity_gate.sh` case 4 | M monotone non-increasing at 12 samples | Runs the parity-ref copy at dt = 0.001, with CL > 0. The header (`:30–32`, `:346`) also promises "decay matches ∫cl_hep·C_b dt within 5%". **The awk body (`:361–405`) never computes that integral**; it checks monotonicity only. |
| `epistemic_pbpk28.sio` TEST 7 | M_final ≥ 0 and M_final < M_init | With CL > 0, everything is eliminated by 168 h (M_final = 3.3e-17), so injected mass is invisible. The recorded run passes with 1.33 mg injected. |
| `epistemic_pbpk28.sio` TEST 9 | \|AUC(dt = 0.0125) − Dose/CL\| < 0.01 | It gates only the **finest** dt, never the production dt = 0.05, which is off by 0.114 (28%). The O(dt²) ratio and the relative difference are printed but not gated. |
| venlafaxine case 13 / mass conservation | parent + ODV ≤ released | A one-sided upper bound over dosed mass; not a closed-system identity. |

No check uses a closed system (CL = 0, rel = 0). None uses the exact discrete
identity M(T) + CL·AUC = Dose. None runs at the step size its consumer uses.

## A second, independent error in coarse-dt AUCs (found while testing fixes)

Blood exchanges with the organs on a timescale of V_b/ΣQ = 1/150 h ≈ 24 s. A trapezoid over
a 0.5 h interval starting at the bolus therefore charges roughly
dt·C_b(0)/2 ≈ 0.25 mg·h/L that the true curve does not contain. That is a mismatch
between method and quadrature, not a solver error. With backward Euler,
rectangle-rule AUC is exact, while the trapezoid gives 0.653226 = 0.403226 + 0.25.
CN-plus-trapezoid is exact for a different reason: the volume-weighted sum of the CN
equations is itself the trapezoidal rule on the scalar mass ODE, so the identity holds
for any CN trajectory. That is why the clamps-off CN AUC is right even though its state
trajectory rings badly (next section). **Any fix must accumulate
AUCs with the integrator's own quadrature weights.** Swapping the integrator under
an existing trapezoid would replace one bias with another.

## Proposed fix (measured, not applied)

`pbpk28_cn_fix_candidates_probe.sio`, clamp-free, each method with its own quadrature.
The reference is CN at dt = 0.001 h.

Rapamycin C_b(24 h) and C_brain(24 h), relative error / most negative state entry:

| method | dt = 0.5 | dt = 0.1 | dt = 0.05 |
|---|---|---|---|
| 0 shipped CN + floors | C_b +611%, C_brain → 0 / floored | +55% / floored | +29.5% / floored |
| 1 CN, no floors | C_b +37 741%, C_brain < 0 / −1.02 | +50%, C_brain < 0 / −0.79 | +11.0% / −0.49 |
| 2 backward Euler | +37.0% / 0 | +6.9% / 0 | +3.4% / 0 |
| 3 CN + Rannacher start (2 steps → 4 BE half-steps) | +0.039% / −2.0e-10 | +0.0014% / −3e-15 | +0.0003% / −5e-17 |
| 4 TR-BDF2, γ = 2 − √2 | −0.33% / 0 | −0.013% / 0 | −0.0033% / 0 |

Every clamp-free method (1–4) recovers AUC_blood = 0.403226 to six significant
figures and closes the discrete mass identity to ≤ 6 × 10⁻¹³. Semaglutide is much
less stiff; there TR-BDF2 keeps every endpoint within 1 × 10⁻⁷ relative even at
dt = 0.5 h.

**Recommendation:** replace the CN step with **TR-BDF2** (Bank et al. 1985;
Hosea & Shampine 1996), and **delete the three floors**.

- TR-BDF2 is L-stable (R(∞) = 0), second-order, one-step, and conservative, because
  it is linear. With γ = 2 − √2, both stages solve with the **same** matrix
  I − σA, σ = γ·dt/2 = d·dt. The Schur coefficients b_v, b_t and the blood-row
  denominator are therefore computed once per step, and only the right-hand side
  changes: stage 1 is a CN step of length γ·dt, and stage 2 is the BDF2 combination.
  The O(N) cost and the Madaros workaround structure (`pbpk28_cn_apply_schur`
  crossing a function boundary) carry over.
- In every measured cell it stayed non-negative with no floor. Bolley–Crouzeix still
  forbids a *guarantee*, so the fix must **fail closed** on a negative entry below a
  roundoff threshold, not floor it.
- Rannacher start-up (method 3) is the smaller diff and is more accurate at dt = 0.5.
  However, it must re-trigger after every dosing discontinuity (XR, oral depot, SC
  depot, repeated doses). It also leaves roundoff-level negatives, and after start-up
  the step is plain CN again: a weakly damped R ≈ −1 mode that any mid-run forcing
  jump re-excites.
- Every consumer that accumulates an AUC must switch to the method's quadrature
  weights: TR-BDF2 b-weights (w, w, d), w = 1/(2(2−γ)), d = (1−γ)/(2−γ), applied to
  (x_n, x_γ, x_{n+1}). That means the step API has to expose the stage value, or
  return the quadrature increment.
- **Out of the scope of this dispatch:** the scenario-level TMDD/PD compositions,
  `semaglutide_sc_depot`, the venlafaxine formation coupling, and the parity refs,
  which carry their own copies of the floored kernel.

### Witness and gate to land with the fix

1. **Closed-system witness** (new `tests/run-pass/`, `//@ requires: madaros`, and
   also run under lean_single): all four profiles, CL = 0, rel = 0,
   dt ∈ {0.5, 0.1, 0.05}. Assert |M_n − M_0|/M_0 ≤ tol at every step, and no
   state entry below −tol·C_0. Today this fails at step 1 for every profile.
2. **Exact discrete identity** at the production dt, replacing `ep28` TEST 7:
   |M(T) + CL·AUC_quad − Dose| / Dose ≤ tol. Today this measures 0.2816 (28%)
   at dt = 0.05.
3. **Parity-gate case 4:** implement the promised ∫CL·C_b dt check, or delete the
   claim from the header. Run it at the production dt, not dt = 0.001.
4. **TEST 9:** gate the production dt against the reference, not only the finest dt.

The tolerance is a roundoff bound, not a fitted value. The identity is exact in real
arithmetic, so only accumulated floating-point error remains. Measured clamp-free
maxima: 6 × 10⁻¹³ at ≤ 3360 steps and 4.0 × 10⁻¹² at 168 000 steps. A bound linear
in the step count n, tol = max(1e-12, n · 1e-15) (about 5 ulp of 1.0 per step), covers both
with margin. Fix the exact constant when the fix lands, and never widen it to admit
an observed failure (principle 6).

### Required follow-up once a fix lands (not authorised by this dispatch)

- Regenerate `pbpk28_epistemic_runs_v1.txt`, `pbpk28_epistemic_v1.md`, the MC
  cross-validation, prior-sweep and Sobol / PCE results, and anything cited from
  them in the dissertation text. Principle 6 applies: re-derive and do not patch
  numbers. The +28% / +52% / +140% column mismatch between the GUM, Hessian and MC
  layers should disappear. If it does not, that is a finding.
- A math-review offload for the fix commit (CLAUDE.md §10), and a clinical-pathway
  review wherever the regenerated numbers reach clinical-facing text.

## Observations deliberately not pursued

- **Madaros exit-182 on the blast probe.** Under Madaros, `pbpk28_cn_clamp_blast_probe.sio`
  printed 14 of 20 result lines, all matching lean_single, then died with
  `madaros: handles full` in the 168 000-step venlafaxine run. That is the known
  handle-lifetime wall
  (`docs/audit/MADAROS_HANDLE_TABLE_182_LIFETIME_DISPATCH_2026-08-17.md`), not this
  defect. The other two probes ran to completion under Madaros. Their outputs match
  lean_single in every substantive field and differ only in the last digit of
  ~1e-13 roundoff residuals.
- **Flow topology.** Σ_{i=1..13} Q_i = 750 L/h, and the lung (Q = 350 L/h, one full
  cardiac output) sits in parallel with the systemic organs. That doubles blood-row
  stiffness and total circulating flow relative to a series lung. It is a model-form
  question, not a solver question, and is not examined here.

## Literature

Cited from the author's knowledge; the scite connector was unavailable in this
session. Verify before quoting externally.

- Bolley C, Crouzeix M. Conservation de la positivité lors de la discrétisation des problèmes d'évolution paraboliques. *RAIRO Anal. Numér.* 12 (1978) 237–245.
- Shampine LF. Conservation laws and the numerical solution of ODEs. *Comput. Math. Appl.* 12B (1986) 1287–1296.
- Bank RE, Coughran WM, Fichtner W, Grosse EH, Rose DJ, Smith RK. Transient simulation of silicon devices and circuits. *IEEE Trans. CAD* 4 (1985) 436–451.
- Hosea ME, Shampine LF. Analysis and implementation of TR-BDF2. *Appl. Numer. Math.* 20 (1996) 21–37.
- Rannacher R. Finite element solution of diffusion problems with irregular data. *Numer. Math.* 43 (1984) 309–327.
- Hairer E, Wanner G. *Solving Ordinary Differential Equations II*, 2nd ed., Springer 1996, §IV.3 (A- and L-stability).

## Reproduce

On the workspace, from a worktree at the base commit:

```bash
make build-madaros
export SOUNIO_STDLIB_PATH=$PWD/stdlib
ulimit -s 524288
artifacts/self-hosted/madaros build docs/audit/repro/pbpk28_cn_clamp_mass_probe.sio /tmp/m.elf && /tmp/m.elf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile docs/audit/repro/pbpk28_cn_clamp_blast_probe.sio -o /tmp/b.elf && /tmp/b.elf
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc compile docs/audit/repro/pbpk28_cn_fix_candidates_probe.sio -o /tmp/f.elf && /tmp/f.elf
```

Unset `SOUC_BIN SOUNIO_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN` first when launching
from the workspace tmux, whose global environment points at `/workspace/sounio`.
Wall-clock time: mass probe < 1 s; blast probe ≈ 110 s on lean_single; fix probe ≈ 60 s.
