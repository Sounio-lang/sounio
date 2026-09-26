# LLM offload log

## 2026-09-26T18:34Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, PBPK28 portal/hepatic-sink kernel

| 2026-09-26 | — | math-review | pbpk28_hepatic.sio, darwin_pbpk28_hepatic_gates.sio (CN step with portal topology + interstitial metabolic sink; extended-clearance-model steady-state identities) | WAIVED | No offload provider configured in this container (`bin/llm-offload --status`: keys file not found). Independent verification by exact closed-form gates plus sabotage controls; narrative below. |

**Trigger**: new hand-derived PK mathematics — (1) a Crank–Nicolson Schur
closure extended with portal routing (splanchnic organs affine in C_b, liver
inflow therefore affine in C_b, single scalar blood equation still closes the
step) and an interstitial first-order sink k_t = CLint_u·fu_b/K_p; (2) the
steady-state identities used as gates: with elimination confined to liver
tissue, X = fu·CLint·PS/(PS + fu·CLint) (extended clearance model, PS→∞ gives
well-stirred), IV infusion C_b = R/CL_H with CL_H = Q_L·X/(Q_L + X), portal
infusion C_b = R/X, hence F_H = Q_L/(Q_L + X) emerges.

**Attempted**: `bin/llm-offload --status` — no provider reachable.

**Outcome**: WAIVED. Independent verification instead:

1. **V0 reduction** — with k_t = 0, in_t = 0, portal = false the new step is
   bit-identical to `pbpk28_full_cn_step` over 3000 steps including a nonzero
   blood input (0 of 3000 steps differ). The sink/input terms are added as
   separate +0.0 terms precisely so this holds; a sabotage that fuses the sink
   into h·(PS/K_p + k)/V_t (mathematically equal) makes V0 fail on 3000/3000
   steps, proving the gate detects float reassociation.
2. **V1 mass balance** — M + E − D < 1e-12 relative (portal + liver/gut sinks +
   residual central CL + gut-tissue input; and portal with no elimination).
3. **V2 steady state vs closed form** — IV and portal C_b, and emergent F_H
   (0.819820 = closed form), all < 1e-12 relative; also a no-portal variant
   that routes the liver sink through the main organ loop.
4. **Sabotage controls** (kernel restored byte-identical after each):
   dropping the sink from the implicit diagonal fails V1 and V2; letting
   splanchnic organs drain to blood as well as to the liver (double-counted
   outflow) fails V1 and V2. A first version of the gates let the
   main-loop-sink sabotage pass V2 (the portal liver uses a separate code path)
   and hid NaN in the running maximum; both gaps were closed before commit.
5. Identical results under lean_single and the default Madaros engine.

**Flagged for re-review** once a provider is configured.

## 2026-09-26T19:05Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, PBPK28 θ-step (Rannacher start-up support)

| 2026-09-26 | — | math-review | pbpk28_hepatic.sio θ-scheme step + θ-weighted elimination accounting; darwin_pbpk28_hepatic_gates.sio VT | WAIVED | No offload provider configured (`bin/llm-offload --status`). Verified by exact reduction and mass-balance gates, below. |

**Trigger**: the CN step generalised to the θ-scheme (implicit weight θ·dt,
explicit (1−θ)·dt; θ = 1 backward Euler, L-stable) so a dosing event can be
followed by backward-Euler half steps (Rannacher 1984). Motivation measured: a
10 mg IV bolus at dt = 0.02 h under plain CN excites the stiff exchange modes
into sign-alternating oscillation and the negativity clamp creates mass
(residual 1.8e-3).

**Outcome**: WAIVED. VT gate: θ = ½ reproduces the CN step to 1e-12 relative;
θ = 1 with a 10 mg bolus at dt = 0.02 conserves mass (M + E − D) to 1e-12 with
θ-consistent elimination accounting. Both engines. Flagged for re-review.

## 2026-09-26T20:16Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, PBPK28 multi-drug coupling driver + allocation-free kernel steps

| 2026-09-26 | — | math-review | ddi/multidrug28.sio (competitive multi-inhibitor factor, Lie / predictor-corrector / iterated coupling, Rannacher start-up, θ-consistent AUC, clamp accounting); pbpk28_hepatic.sio *_ws steps; darwin_pbpk28_multidrug_gates.sio M/V3/V4/V5/V6 | WAIVED (orthogonal) + same-provider adversarial review | `bin/llm-offload --status`: no provider configured. An independent adversarial review was run by a separate Claude agent (different model, read-only, own probes); it is NOT an orthogonal-provider review. Flagged for re-review once a provider is configured. |

**Trigger**: new hand-derived numerics and PK mathematics: (1) competitive
factor f = 1/(1 + L + Σ_{j≠i} C_u,j/Ki_j) with C_u from the interstitial layer
at liver and gut; (2) coupling order claims (Lie 1, PC 2); (3) the V3 static
closed form with gut-wall sink; (4) the V4 site-convention closed form.

**Adversarial review — findings and disposition**
- M1 (accepted, my claim was wrong): I had attributed the erratic PC ratios
  on the physiological transport to CN order reduction by stiffness and moved
  the order gate to a transport scaled by 0.01. The reviewer refuted it with
  controls (backward-Euler transport and a 100x smaller lung PS leave the
  ratios unchanged; a 10x weaker perpetrator dose gives clean order two at
  s = 1). Reproduced here: dose 10 mg, PC 3.87 → 3.93, Lie 1.92 → 2.00;
  dose 100 mg, PC 3.54 → 3.78 → 3.89 at dt 0.0025 → 0.000625. The
  pre-asymptotic range is set by the interaction nonlinearity. V5 now runs on
  the physiological transport; headers corrected.
- M2 (accepted): the enzyme-site convention was ungated (gut-site inhibition
  dropped and self-inhibition included both passed every gate). New V4:
  perpetrator at infusion steady state, closed-form site concentrations,
  victim AUC vs closed form (rel err 2.4e-11). Sabotages re-run here: gut site
  dropped → V4 rel err 0.76 FAIL; self-inhibition → 4.3e-4 FAIL; corrector at
  f* → V5c 1.71/1.85 FAIL.
- M3 (accepted): the discrete AUC/mass identity holds only while the
  negativity clamp is silent; it fires at dt >= 0.05 h. Clamped mass is now
  accounted (PBPK28Work.clamped, md_clamped) and gated against a round-off
  bound n_steps·27·eps·dose. A first version required exactly zero and failed
  V3 on 1.3e-12 mg of terminal-phase round-off; the bound replaced it, derived
  from eps and the step count (V3 bound 3.6e-10), not from the observed value;
  the reviewer's oscillation case (3e-10 relative at dt 0.05) exceeds it.
- M4 (accepted): the gate file died under Madaros (exit 182, handle table).
  Kernel and driver steps made allocation-free (persistent scratch; PBPK28Work;
  portal block writes through &! instead of returning a tuple of copies).
- m1 (accepted): the iterated fixed point is the midpoint-coefficient CN
  coupling, not the implicit trapezoidal rule (O(dt^3) apart); renamed.
- m2-m4: V5c now self-converges the default PC scheme on the physiological
  fixture (3.58, 3.77); V6 labelled controls; gut-flow fixture caveat stated.

**Refactor equivalence** (allocation-free vs the reviewed implementation,
kept as renamed reference modules in scratch): 5 scenarios (portal on/off,
iters 0/1/12, three drugs, external load, oral + IV bolus + infusion
start/stop): Madaros `==` on every state and accounting field, 0 mismatches;
lean_single 2645 fingerprints (17 significant digits) identical.
V0 bit-identity of the kernel preserved on both engines.

**Measurement hazard found on the way**: lean_single ignores
SOUNIO_STDLIB_PATH and resolves `stdlib/` relative to the cwd; lean_single
`souc run` exits 1 silently on a compile error. An early "identical" diff in
this work compared old code with old code because of it and was discarded.

## 2026-09-26T20:36Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, PBPK28 closed-form calibration (ECM inverse, asymmetric gut wall)

| 2026-09-26 | — | math-review | pbpk28_calibration.sio (liver ECM inverse; gut-wall (PS_g, fu_g·CLint_g) from (F_G, E_sys)); darwin_pbpk28_calibration_gates.sio C0-C3 | WAIVED | No offload provider configured (`bin/llm-offload --status`). Verified by exact round trips on the dynamic model and sabotage controls, below. |

**Trigger**: new hand-derived PK identities used to calibrate drug profiles:
(1) inverse of the extended clearance model, fu·CLint = X·PS/(PS − X) with
X = Q·CL_H/(Q − CL_H); (2) the gut-wall asymmetry of the kernel — absorbed drug
enters the enzyme compartment directly, arterial drug crosses PS_g — giving
F_G = a·q/(q + X_g), E_sys = X_g/(q + X_g) with a = u/(u + k), X_g = PS·k/(u + k),
u = PS/Kp, and the inverse X_g = q·E_sys/(1 − E_sys), a = F_G/(1 − E_sys),
PS_g = X_g/(1 − a), fu_g·CLint_g = PS_g(1 − a)/a (independent of Kp_g).
Motivation measured in the literature (checked on PubMed): Paine 1996,
doi:10.1016/S0009-9236(96)90162-9, midazolam intestinal extraction 0.43 ± 0.18
of the absorbed dose vs 0.08 ± 0.11 of arterial drug per passage (anhepatic
liver-transplant recipients, n = 5 + 5).

**Outcome**: WAIVED. Gates, both engines: C0 algebraic round trips 0 (x1e12);
C1 IV AUC = D/(CL_H + CL_c) to 1e-11; C2 F_G and E_sys recovered on the dynamic
model to < 1e-12; C3 F = F_G·F_H to 1.8e-11. Sabotages: liver inverse replaced
by the forward form -> C0, C1, C3 FAIL; gut inverse a/(1 − a) -> C0, C2, C3
FAIL. Flagged for re-review once a provider is configured.

_Timestamps of the four PBPK28 entries above corrected on 2026-09-26 to the
commit times of 0eea0ba6, 829e0753, bdf495a8 and f4ae82c3; the first versions
carried estimated, future-dated times (review of sounio-lang/sounio#2695)._
