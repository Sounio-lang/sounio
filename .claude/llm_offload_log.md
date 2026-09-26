# LLM offload log

## 2026-09-26T19:10Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, PBPK28 portal/hepatic-sink kernel

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

## 2026-09-26T21:30Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, PBPK28 θ-step (Rannacher start-up support)

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
