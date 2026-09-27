<!-- docs:meta
topic_id: repo.docs.audit.venlafaxine-portal-parity-dt-convergence-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-27
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.venlafaxine-portal-parity-dt-convergence-2026-09-26
-->

# Venlafaxine XR portal model: parity port and time-step convergence

Date: 2026-09-26. Branch `claude/vfx-portal-parity`, on top of `03769363f`
(portal first-pass model, `claude/competent-mcclintock-b137de`). The operator
chose the portal model as the canonical venlafaxine absorption model on
2026-09-26.

## What was wrong

`scenarios/venlafaxine_xr.sio` used to absorb from the gut pool as
`absorb = F·G·(1 − e^(−ka·dt)); G −= absorb`. The (1 − F) share of the mass
leaving the lumen stayed in the pool and was absorbed again on later steps, so
the whole released dose reached the parent: effective F = 1, and an effective
absorption rate of −ln(1 − F(1 − e^(−ka·dt)))/dt (0.259, 0.271, 0.277 and
0.280 h⁻¹ at dt = 0.5, 0.25, 0.125 and 0.0625 h), which tends to F·ka = 0.2835
rather than ka = 0.63. `9a830bfad` fixed the gut split and `03769363f` replaced
the fixed F with portal input. Neither commit updated the parity surface:

- `website/src/lib/pbpk28_core.mjs` (`runVenlafaxineScenario`, which the Node
  runner imports) still had the F-leak, a CN kernel with negativity floors, the
  explicit post-step formation and the old systemic clearances.
- `tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio` was a
  self-contained copy of that same old model.
- `scripts/ci/dissertation_pbpk28_parity_gate.sh` still described F as "an
  absorption-rate scalar".

Cases 10–13 therefore compared two copies of the superseded model with each
other and passed, while witnessing nothing about the scenario the dissertation
uses.

## What changed

1. **JS mirror** (`pbpk28_core.mjs`). An independent reimplementation of:
   - the portal scenario: the hepatic resolution (`vfxHepaticEm`), the exact
     inversion of the liver sink (`vfxLiverClSink`), the gut split
     (`vfxGutAbsorbStep`) and the closed-form predictions;
   - the TR-BDF2 routed-sink step of `theta_pbpk28.sio`, with its ledger.

   The operation order follows the Sounio source.
2. **Sounio ref**. It now drives the stdlib scenario's own step functions
   (`vfx_gut_absorb_step`, `vfx_parent_step_with_formation`, `vfx_odv_step`) in
   the same order as `vfx_strang_step`. The Sounio side of the parity is
   therefore the canonical model. The ref keeps its own states and ledgers only
   because the scenario struct's fields are private.
3. **Gate**:
   - Case 13 covers both the total-mass ratio and the blood AUC ratio.
   - The old "parent + ODV ≤ released" check is replaced by a mass account on
     both engines' output: gut balance, the F_abs split, cumulative portal input
     ≤ F_abs·released (the bound the old model broke, checked as the
     full-precision slack F_abs·released − portal = F_abs·G ≥ 0), both ledger identities,
     body ≤ portal input, negative mass ≤ 5e-7 mg, and monotone release.
   - A closed-form check at 120 h on both engines, plus a Node ↔ Sounio
     comparison at print resolution.
   - Residuals are printed ×1e12, because `println(f64)` under Madaros is fixed
     at 6 decimals and prints every rounding-level residual as `0.000000`.
   - The residual tolerance is the ledger budget max(1e-12, steps·1e-15)
     relative to the released dose (`pbpk28_mass_tol_rel`). It is a heuristic
     rounding budget, as that function documents: not fitted to the observed
     residuals, and not a proven floating-point bound.

## Parity (Madaros ELF md5 57c015c1, Node 22, full gate rc = 0, with the exact gut step)

| Surface | Result |
|---|---|
| Parent cavg, 14 organs × 12 samples | RMSE ≤ 1.1e-8 (from print resolution alone: Madaros prints values < 5e-7 as `0.000000`) |
| ODV cavg, 14 × 12 | RMSE 0 |
| ODV/parent total-mass ratio at 96 h | 170107.171549 on both engines |
| Blood AUC ratio at 120 h, NM | 3.571424 on both (closed form 3.571429, within the truncation bound) |
| Oral CL/F at 120 h, NM | 100.000000 on both (closed form 100) |
| Mass account, both engines | at 96 h: gut, split and bound residuals < 1e-18 mg, ledger residuals ≤ 4.8e-12 mg (tolerance 75e-12 mg); portal input 61.989796 mg ≤ F_abs·75 = 61.989796 mg; negative mass 0 |
| Mutation: old gut update (pool keeps the unabsorbed share) in both engines | parity still passes (both engines share the defect); mass account FAILS on every sample: gut balance −15.7 mg, portal input exceeds F_abs·released by up to 13.0 mg |

JS all-phenotype run (dt = 0.5 h, 120 h), matching the figures in `03769363f`:

| | PM | IM | NM | UM |
|---|---:|---:|---:|---:|
| oral CL/F (L/h) | 23.0145 | 44.9072 | 100.0000 | 264.7971 |
| F_oral = F_abs·F_H | 0.7126 | 0.6112 | 0.4500 | 0.2515 |
| AUC ODV/parent | 0.2588 | 1.2008 | 3.5714 | 10.6625 |

## Time-step convergence (NM)

Probe: `docs/audit/repro/venlafaxine_portal_dt_convergence.sio`. It drives the
stdlib step functions under Madaros, and the JS engine agrees with it to about
12 significant digits at every dt. Blood concentrations are in mg/L.

### Before: release added to the pool at the start of the step

| dt (h) | C_b,parent 2 h | C_b,parent 24 h | C_b,ODV 24 h | portal input 4 h (mg) | AUC ratio 120 h |
|---:|---:|---:|---:|---:|---:|
| 0.5     | 0.0467352 | 0.00179676 | 0.0437279 | 22.78852 | 3.57142425 |
| 0.25    | 0.0453844 | 0.00185359 | 0.0442903 | 22.16936 | 3.57142418 |
| 0.125   | 0.0447084 | 0.00188318 | 0.0445900 | 21.84202 | 3.57142415 |
| 0.0625  | 0.0443706 | 0.00189830 | 0.0447449 | 21.67350 | 3.57142413 |
| 0.03125 | 0.0442018 | 0.00190594 | 0.0448237 | 21.58794 | 3.57142413 |

Successive differences halved with each halving of dt (ratios 1.998, 2.001,
2.002 at 2 h; 1.920, 1.958, 1.978 at 24 h; 1.891, 1.942, 1.970 for portal
input): **first order**. At the production dt = 0.5 h, Richardson extrapolation
put the error at about +6.1 % for C_b,parent at 2 h, −6.1 % at 24 h, −2.6 % for
C_b,ODV at 24 h, and +6.0 % for cumulative portal input at 4 h.

The cause was the Lie splitting of the release. Step 1 added the whole step's
release `matrix_step_amount(t, dt)` to the pool at the start of the step, so
that mass decayed for a full dt instead of dt/2 on average and left the lumen
early by about ka·dt/2 of each step's release. The gut decay itself was exact,
and TR-BDF2 is second order for input held constant over a step.

### After: exact gut step for release at a constant rate (applied, operator decision 2026-09-26)

`vfx_gut_absorb_step(gut, ΔR, dt)` now integrates dG/dt = R − ka·G exactly
with R = ΔR/dt over the step, where x = ka·dt:

    G₁ = G₀·e^(−x) + ΔR·(1 − e^(−x))/x,   leaving = G₀ + ΔR − G₁.

Only the exact solution has the semigroup property: one step of dt with ΔR
equals two steps of dt/2 with ΔR/2. `darwin_venlafaxine_gut_first_pass.sio`
now asserts this to 1e-12 mg. The residual is 1.8e-15 mg on both engines. With
the release-at-start update restored, the assertion fails by 0.311 mg, which
matches the predicted ΔR·x/4 order.

| dt (h) | C_b,parent 2 h | C_b,parent 24 h | C_b,ODV 24 h | portal input 4 h (mg) | AUC ratio 120 h |
|---:|---:|---:|---:|---:|---:|
| 0.5     | 0.0441861 | 0.00191473 | 0.0449544 | 21.45981 | 3.57142412 |
| 0.25    | 0.0440913 | 0.00191406 | 0.0449193 | 21.48888 | 3.57142412 |
| 0.125   | 0.0440561 | 0.00191381 | 0.0449086 | 21.49754 | 3.57142412 |
| 0.0625  | 0.0440426 | 0.00191371 | 0.0449052 | 21.50016 | 3.57142412 |
| 0.03125 | 0.0440373 | 0.00191367 | 0.0449041 | 21.50100 | 3.57142412 |

At dt = 0.5 h the Richardson error is now about +0.34 % (C_b,parent at 2 h),
+0.46 % (at 4 h), +0.06 % (at 24 h), +0.11 % (C_b,ODV at 24 h) and −0.19 %
(portal input at 4 h). That is 13–110 times smaller than before.

Successive-difference ratios over the halvings (observed order p = log₂ ratio):

| quantity | ratios | p |
|---|---|---|
| C_b,parent 4 h / 8 h / 12 h | 3.26–3.27 / 3.35–3.49 / 3.52–3.64 | 1.6–1.9 |
| C_b,ODV 24 h, portal input 4 h | 3.10–3.36 | 1.6–1.75 |
| C_b,parent 2 h / 24 h | 2.58–2.70 / 2.51–2.66 | 1.3–1.4 |

**The order is not 2.** The limits come from the release input, not from the
gut step or TR-BDF2. They were measured by extending the JS sweep to
dt = 1/256 h, both with the stdlib release curve and with an exact one
(`Math.pow` in place of `matrix_er`'s `mer_pow`, in a scratch copy only).
Successive-difference ratios, halvings from dt = 0.5 h to 1/256 h:

| quantity | stdlib release curve | exact release curve |
|---|---|---|
| gut mass / portal input at 2 h | 3.20 3.16 3.02 2.66 2.35 2.19 (p → 1.1) | 3.20 3.16 3.15 3.14 3.136 3.136 (p → 1.65) |
| C_b,parent 4 h | 3.26 3.27 3.12 2.73 2.39 2.20 | 3.25 3.27 3.25 3.21 3.19 3.18 (p → 1.67) |
| C_b,parent 2 h | 2.70 2.61 2.58 2.31 2.12 2.06 | 2.69 2.61 2.74 2.84 2.92 2.97 (p rising, 1.57) |
| C_b,ODV 24 h | 3.25 3.21 3.10 2.82 2.69 2.51 | 3.23 3.19 3.14 3.08 3.78 3.21 |
| portal input 24 h | 2.65 2.39 2.22 2.11 … | 2.36 2.19 2.10 2.05 … (p → 1) |

- **Release-rate singularity (confirmed).** The Korsmeyer–Peppas rate
  ∝ t^(n−1) = t^(−0.35) is singular at t = 0, and a constant rate per step
  cannot resolve it. On the first step the local error is Θ(h^(1+n)). Later
  steps contribute h³·t^(n−2); their sum converges because n − 2 < −1, and it
  is dominated by the first steps. That gives global order 1 + n = 1.65. With
  the exact curve, the gut and portal input at 2 h converge to ratio 3.136
  = 2^1.65.
- **`matrix_er` transcendental floor (first order at small dt).** With the stdlib
  curve the same gut quantity drifts to p ≈ 1. `mer_pow`'s series floors the K–P
  fraction at small t: 0.714 mg is released by t = 0.0039 h against an exact
  0.406 mg (+76 %; +9.7 % at 0.0156 h), which is effectively a ~0.6 mg
  instantaneous release at t = 0⁺. A constant rate over the first step
  mis-times that impulse by ~h/2, which is a first-order term. It dominates
  below dt ≈ 0.06 h and is small at 0.5 h. The same series is also −0.1 % low
  near saturation (74.919 mg at 11.99 h), which moves the switch-off from
  11.986 h to 12.010 h. This is
  bold-robinson's dispatch
  (docs/audit/MATRIX_ER_TRANSCENDENTAL_ACCURACY_DISPATCH_2026-09-26.md, PR
  #2722, `matrix_er` → `math::pure`), not addressed here.
- **Release switch-off.** The release rate drops to zero inside a step at
  t* = (1/k)^(1/n): 11.986 h analytically, 12.010 h on `matrix_er`'s series.
  A constant rate over that step mis-times a moment of order R(t*)·δ·(h − δ),
  where δ is t*'s offset in its step. The effect is small at production dt
  (Grok estimated it at ~50× below the start-up term at 0.5 h), but it made
  the late-time ratios erratic once dt was comparable to δ. With the exact
  curve, cumulative portal input at 24 h converged at first order. **Fixed by
  the split below.**

Whether the 2 h concentrations at production dt sit at 1.3–1.4 because of
solver start-up on the non-smooth input (Grok's 2 − n = 1.35 suggestion) is not
settled. With the exact curve the 2 h order climbs to 1.57 and has not levelled
off by dt = 1/256 h.

### Release window split at matrix saturation (applied, operator decision 2026-09-27)

`vfx_gut_release_step(gut, rel, t, dt)` now confines the step's release to
`vfx_release_window(rel, t, dt)`. That is dt, except in the one step where the
matrix saturates. There the window ends at the saturation time, found by
bisecting the public `matrix_fraction` on the step (64 halvings), so it matches
the curve the release amounts come from, 12.0099 h on the current series. The
pool is solved exactly through the window (`vfx_gut_absorb_step_window`) and
then decays for the rest of the step. `MatrixReleaseModel`'s private fields
are not needed. The scenario, the formation test, the parity ref and the probe
all call `vfx_gut_release_step`. The JS mirror follows in the same operation
order.

`darwin_venlafaxine_gut_first_pass.sio` asserts:
- that a windowed step equals a step of the window length with the release
  followed by a release-free step for the remainder. The residual is 0 on
  Madaros; with the post-window decay removed, it fails by 2.3 mg;
- that exactly one step of the canonical run has a window shorter than dt
  (12.0 h, window 0.009873 h on both engines);
- that the matrix fraction reaches 1 at the window's end and not before.

Successive-difference ratios, dt = 0.5 h down to 1/256 h, exact release curve:

| quantity | before the split | after the split |
|---|---|---|
| C_b,parent 24 h | 2.55 2.55 2.47 2.36 6.41 2.66 | 2.85 3.02 3.08 3.09 3.10 3.12 (p → 1.64) |
| C_b,ODV 24 h | 3.23 3.19 3.14 3.08 3.78 3.21 | 3.29 3.28 3.28 3.29 3.30 3.30 (p 1.72) |
| portal input 24 h | 2.36 2.19 2.10 2.05 12.4 2.13 (p → 1) | 3.93 3.91 3.88 3.85 3.81 3.80 (p 1.92–1.98) |

The late-time orders are now regular. They sit at the release singularity's
1 + n (concentrations) or near 2 (cumulative portal input, which integrates
the input). Nothing before 12 h changes. At dt = 0.5 h the values move little,
as expected (C_b,parent at 24 h: +0.056 % → about +0.04 %). The stdlib curve
still carries the `matrix_er` floor term below dt ≈ 0.06 h, which #2722
addresses. C_b,parent at 12 h is sampled 0.014 h after the analytic t*; there
its successive differences change sign, so its ratios do not measure an order.

Replacing the release transcendentals (#2722) is still not done here: it is
another lane's open PR.

The run-integrated readouts remain dt-independent: the oral CL/F is
100.0000000, AUC_parent(0, 120 h) is 0.75, and the AUC ratio is 3.5714241 at
every dt. These are exact integrated identities of the linear model. The
ledger's b-weighted quadrature is the scheme's own mass identity, so they hold
for the discrete solution regardless of when the input arrives. The AUC ratio
now moves by 1e-9 relative across dt (3.5e-9 absolute), against 1.3e-7 before.

## Not addressed

- The parent half-life that emerges from the Kp table and the portal
  clearances is about 2.8 h (C_b,parent falls ~20-fold between 24 h and 36 h).
  The label gives about 5 h. `03769363f` lists Kp/Vss as not addressed, and it
  is still open.
- The CYP2D6 phenotype scale is still defined from the literature ratios, whose
  provenance is unverified. Matching those ratios is not independent evidence.

## Math review (bin/llm-offload -t math-review, 2026-09-26)

Run from `origin/chore/llm-offload-llmgateway-grok47` (the canonical gateway
route) on the workspace. Legs with a verdict come from two independent vendors:

- **Grok 4.7 (xAI direct, after the gateway leg timed out at 900 s).** OK on
  the hepatic closed form, the sink inversion, the TR-BDF2 mass identity, the
  gut split and the old-model leak, the dt attribution, and the dt-independence
  of the integrated readouts.
  - DEFECT on the truncation bound: parent input still to come from the gut was
    missing from M_p,left, and the residual budget is not a proved rounding
    bound. **Fixed**: both engines now use M_p,left + F_abs·G, which also makes
    the CL/F bound rigorous, because F_abs·G/admin_∞ ≤ d_p. The gate and this
    note now call the budget a heuristic.
  - Its remark that max(d_p, d_o) is a tighter certificate is correct; the
    scenario's looser d_p + d_o + d_p·d_o is kept because it is also rigorous.
  - Its caveat on the second-order proposal stands: the proposal is a model
    change relative to impulse-at-start release, and the kink where the
    Korsmeyer–Peppas fraction saturates at 1 (t ≈ 12 h for k = 0.199, n = 0.65)
    can still cut the observed order.
- **Qwen 3 235B (OpenRouter).** OK on the F_abs closed form, the TR-BDF2 mass
  identity, the dt attribution and proposal, and the dt-independence. Three
  claimed defects are rejected:
  - *Sink inversion discriminant should be 4aC.* Multiplying
    X = cl·(a + b·PS/(PS/Kp + b·cl)) by (PS/Kp + b·cl) gives
    ab·cl² + (a·PS/Kp + b·PS − X·b)·cl − X·PS/Kp = 0. The leading coefficient is
    ab, so the discriminant is B² + 4abC; the round trip X(cl(X)) returns
    75.30612244897961 exactly.
  - *The old update tends to F_abs·released with rate → ka.* The old pool loses
    only F·L per step, so its retention 1 − F(1 − e^(−ka·dt)) < 1 drains it
    completely. Cumulative input reached 75.000000 mg (measured in `9a830bfad`,
    and reproduced by the mutation run above), and
    −ln(1 − F(1 − e^(−ka·dt)))/dt → F·ka.
  - *The ratio bound is not rigorous.* With AUC_∞ = AUC_T·(1 + α) and
    0 ≤ α ≤ d, the ratio R_T/R_∞ = (1 + α_p)/(1 + α_o) lies in
    [1/(1 + d_o), 1 + d_p], so |R_T/R_∞ − 1| ≤ max(d_p, d_o) ≤ d_p + d_o + d_p·d_o.
- **Kimi K3 (gateway) and DeepSeek V4 Pro (gateway).** No verdict: both spent
  the whole token budget on reasoning and returned empty content (Kimi at 8k and
  16k; a 32k request was rejected with HTTP 400). Not counted. Kimi's reasoning
  trace raised the same truncation-bound gap as Grok. GLM (Z.AI) was
  rate-limited (1313), and the local leg was down.

`.claude/llm_offload_log.md` was absent from main when this round ran (deleted
wholesale in `3944ff825`), so the record was kept here. #2714 has since
restored the log, and all three rounds are also recorded there.

### Round 2: exact gut step (2026-09-27)

Packet: the new step, the semigroup test, the convergence explanation and the
dt-independence claim. Three vendors returned verdicts.

- **Gemini 2.5 Pro (OpenRouter).** OK on all four questions: the exact solution,
  leaving ≥ 0, the semigroup property and the size of the old defect, order
  1 + n from the singularity, and dt-independence.
- **Qwen 3 235B.** OK on all four. Its Q3(a) justification calls t^(n−2)
  integrable at 0, which is false. The conclusion still matches the h^(1+n)
  sum above.
- **Grok 4.7 (xAI direct, after two 600 s timeouts).** OK on Q1, Q2 and Q4.
  - Q2 note, accepted: "about ΔR·x/4" is the leading term (0.394 mg). The exact
    defect ΔR/2·e^(−x/2)(1 − e^(−x/2)) = 0.311 mg is what the test measures.
  - Q4 note, accepted: dt-independence is the discrete identity plus a small
    horizon tail, not an identity that ignores the tail.
  - DEFECT on Q3, accepted: the release switch-off is too small to explain the
    24 h order at coarse dt. The finer sweep above was run to check. It
    confirms the h^(1+n) singularity term exactly with an exact release curve,
    and it finds the `matrix_er` floor as the first-order term at small dt. It
    does not bear out Grok's alternative 2 − n = 1.35 at 2 h: that order climbs
    past 1.5 with the exact curve. The audit text above was rewritten to match
    the measurements.

### Round 3: release window split (2026-09-27)

- **Gemini 2.5 Pro.** OK on the window step (exact, and bitwise equal to the
  plain step when τ = dt), on bisecting the capped fraction with the model's own
  t*, and on the late-time interpretation. It adds the reason portal input at
  24 h converges near second order:
  - portal(T) = F_abs·(released(T) − G(T)), and released(T) is exact on the grid;
  - so the portal error is −F_abs·err(G(T));
  - err(G) obeys e′ = −ka·e + (R − R̃), so the early start-up error decays like
    e^(−ka·t) and only the smooth, O(h²) error near the switch-off is left by
    24 h;
  - concentrations are a convolution over the whole error history, so they keep
    the early 1 + n term.
- **Qwen 3 235B.** OK on the window step and the bisection. Its Q3 "overreach"
  is rejected: it reads portal input as converging with ratios → 2, which are
  the before-split numbers. After the split the ratios are 3.80–3.93. Its own
  argument, that integrating the input gains one order capped at 2, supports
  the near-second-order reading.
