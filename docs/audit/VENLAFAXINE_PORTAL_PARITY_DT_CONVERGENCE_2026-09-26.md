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

## Parity (Madaros ELF md5 57c015c1, Node 22, full gate rc = 0)

| Surface | Result |
|---|---|
| Parent cavg, 14 organs × 12 samples | RMSE ≤ 1.0e-8 (from print resolution alone: Madaros prints values < 5e-7 as `0.000000`) |
| ODV cavg, 14 × 12 | RMSE 0 |
| ODV/parent total-mass ratio at 96 h | 175719.742698 on both engines |
| Blood AUC ratio at 120 h, NM | 3.571424 on both (closed form 3.571429, within the truncation bound) |
| Oral CL/F at 120 h, NM | 100.000000 on both (closed form 100) |
| Mass account, both engines | at 96 h: gut residual 0.04e-12 mg, ledger residuals ≤ 4.7e-12 mg (tolerance 75e-12 mg); portal input 61.989796 mg ≤ F_abs·75 = 61.989796 mg; negative mass 0 |
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

| dt (h) | C_b,parent 2 h | C_b,parent 24 h | C_b,ODV 24 h | portal input 4 h (mg) | AUC ratio 120 h |
|---:|---:|---:|---:|---:|---:|
| 0.5     | 0.0467352 | 0.00179676 | 0.0437279 | 22.78852 | 3.57142425 |
| 0.25    | 0.0453844 | 0.00185359 | 0.0442903 | 22.16936 | 3.57142418 |
| 0.125   | 0.0447084 | 0.00188318 | 0.0445900 | 21.84202 | 3.57142415 |
| 0.0625  | 0.0443706 | 0.00189830 | 0.0447449 | 21.67350 | 3.57142413 |
| 0.03125 | 0.0442018 | 0.00190594 | 0.0448237 | 21.58794 | 3.57142413 |

Successive differences halve with each halving of dt (ratios 1.998, 2.001,
2.002 at 2 h; 1.920, 1.958, 1.978 at 24 h; 1.891, 1.942, 1.970 for portal
input). **The scheme is first order in dt.** Richardson extrapolation puts the
error at the production dt = 0.5 h at about +6.1 % for C_b,parent at 2 h,
−6.1 % at 24 h, −2.6 % for C_b,ODV at 24 h, and +6.0 % for cumulative portal
input at 4 h.

The run-integrated readouts are dt-independent. The oral CL/F is 100.0000000
at every dt, AUC_parent(0, 120 h) is 0.75 at every dt, and the AUC ratio moves
by 1.3e-7 relative. These readouts are exact integrated identities of the
linear model (Dose/AUC and Ae/AUC), and they do not depend on when the input
arrives.

**Where the first order comes from.** The gut decay is exact:
G → G·e^(−ka·dt). TR-BDF2 is second order for input held constant over a step.
What remains is the Lie splitting of the release. Step 1 adds the whole step's
release `matrix_step_amount(t, dt)` to the pool at the start of the step, so
that mass decays for a full dt instead of dt/2 on average. Mass therefore leaves
the lumen early by O(ka·dt/2) of each step's release, which fits the
overestimated early portal input and the phase-shifted concentrations in the
table.

For release at a constant rate over the step, integrating dG/dt = R − ka·G
exactly gives

    G_{n+1} = G_n·e^(−ka·dt) + (ΔR/(ka·dt))·(1 − e^(−ka·dt)),
    leaving = G_n + ΔR − G_{n+1},

where ΔR = `matrix_step_amount`. With that update the gut step is exact for
piecewise-constant release. The expected overall order is then 2, limited by
holding the portal input constant over the step, which is still second order.
That expectation is not yet measured.

**This change was not made here.** It alters the scenario numerics in
`venlafaxine_xr.sio`, which belongs to another lane, and it is a
dissertation-facing modelling choice. It is recorded as a proposal for the
operator.

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

`.claude/llm_offload_log.md` is absent from main (deleted wholesale in
`3944ff825`), so the review record is kept here.
