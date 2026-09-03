<!-- docs:meta
topic_id: repo.docs.audit.pbpk14-modelform-stiff-repair-2026-06-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.pbpk14-modelform-stiff-repair-2026-06-14
-->

# PBPK-14 model-form repair via a stiff integrator (future-work / qual P1.4)

**Date:** 2026-06-14
**Artifact:** `examples/ode/pbpk14_stiff_backward_euler.sio` (self-contained, runs on
`bin/souc-linux-x86_64 <src> <out>`)
**Closes:** future-work item "model-form / RK4 (stiff integrator)" and qualification
task P1.4 ("RK4 model-form repair").

## The problem (model-form distortion to suit the solver)

`stdlib/ode/pbpk14_rk4.sio` integrates the 14-compartment PBPK with **explicit RK4** and
openly **halves every blood flow** — `"REDUCED by 50% for stability"`, cardiac output
350 → 175 L/h (unphysiological) — to keep the explicit method stable. PBPK is **stiff**:
the fastest mode is `≈ -q_lung/(v_lung·kp_lung) ≈ -875/h` (real) / `-437/h` (reduced).
RK4 is stable only for `dt·λ ≳ -2.78`; at the clinical `dt = 0.1 h` that is `dt·λ ≈ -87`
(real) / `-44` (reduced) — **both far outside** the stability region.

**Empirically confirmed:** running the existing reduced-flow demo at dt=0.1h already
**blows up** — every fast blood-flow compartment is NaN and the demo's own self-test
prints `TEST FAILED — Arterial/Venous/Lung invalid`. So the flow-distortion does **not**
even buy stability; the physiological-fidelity cost is paid for nothing. (The module
header's hopeful *"should prevent NaN issues"* is falsified.)

## The repair

PBPK linear kinetics ⇒ `dy/dt = A·y` exactly (A constant). **Backward Euler**
`(I − dt·A) y_{n+1} = y_n` is unconditionally **L-stable**: one 14×14 linear solve per
step handles the stiffness at the **real** flows, no distortion. `A` is recovered by
**probing** the derivative on unit vectors (`deriv` is linear & homogeneous, so
`deriv(e_j)` is exactly column j) — the physics is written once.

## Results (reproducible)

| check | result | meaning |
| --- | --- | --- |
| `max|A·t − deriv(t)|` | `4.5e-13` | probe reconstructs A correctly |
| mass conservation (elim off) | `500.000000` | solver conserves mass exactly |
| RK4, real flows, dt=0.1h | **invalid (blows up)** | explicit method unusable |
| BE, real flows, dt=0.1h | **stable, physiological** | the repair |
| BE@0.1 vs converged (RK4 dt=5e-4) | `Δliver = 0.0025 mg` | BE first-order integrator error (~5.6%) |
| **real vs halved-flow exposure** | **`Δliver = 0.0336 mg (~72%)`** | **model-form error** |

**The payload:** the model-form error from the flow distortion (~72% on liver exposure)
is **>13× larger** than the integrator error (~5.6%). Because a stiff solver makes *both*
the real-flow and distorted-flow models stable, this structural gap is finally
**measurable** — it is the honest epistemic interval (the G-α-δ / uncertainty-as-default
connection) that the reduced-flow hack silently buries.

## Honest caveats

- Backward Euler is **first-order**; both exposures carry ~5% numerical damping, biased
  the same direction, so it largely cancels in the difference — but a higher-order
  L-stable scheme (implicit midpoint / Radau IIA) would tighten the integrator term.
  The model-form gap (72%) dwarfs it either way.
- A **compiler bug surfaced and was worked around:** the self-hosted compiler
  miscompiles NaN equality (`NaN == NaN` is wrongly true — the `fcmp` self-compare bug
  class, here on x86). The `x != x` NaN idiom is therefore unreliable; the example uses a
  bounded test `x ∈ (−1e6, 1e6)` (robust because unordered comparisons both yield false
  for NaN/inf). This is a concrete instance of the same defect family tracked in
  `PR289_A64_REGRESSIONS_HONEST_SCOPE_2026-06-13.md`.

## Out of scope (other declared future-work, not local)

GPU PBPK14 Tsit5 (needs GPU), clinical cohort (needs patient data), identifiability,
Lean proof offload — PENDING, deferred for resource/scope reasons.
