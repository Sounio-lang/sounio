// Pure-JS, Node-and-browser-portable core of the 28-state permeability-limited
// PBPK with **Strang operator splitting** (Strang 1968; Hundsdorfer & Verwer 2003).
//
// State: 14 organs × {vascular Cv[i], interstitial Ct[i]}. Blood is the central
// compartment kept in Cv[0]; Ct[0] is unused.
//
// Equations:
//   V_blood · dC_blood/dt = sum_{i≥1} Q_i·(C_v,i - C_blood) + release - cl_hep·C_blood
//   V_v,i   · dC_v,i/dt   = Q_i·(C_blood - C_v,i) - PS_i·(C_v,i - C_t,i/Kp_i)     [i≥1]
//   V_t,i   · dC_t,i/dt   = PS_i·(C_v,i - C_t,i/Kp_i)                              [i≥1]
//
// Stiffness: at literature rapamycin PS, the eigenvalue PS·(1/V_v + 1/(V_t·Kp))
// is O(1e3-1e4) for kidney/gut/lung — far past explicit RK4's stability boundary
// for dt = 0.01 h. Solution: split each step as
//   exp(L_PS · dt/2) → RK4(L_slow, dt) → exp(L_PS · dt/2)
// where L_PS is the (per-organ, 2-state) PS-coupling operator, integrated by its
// closed-form linear-ODE solution, and L_slow is the remaining (Q-transport,
// clearance, release) operator integrated by classical RK4. Strang splitting is
// second-order accurate; PS-relaxation is exact; the composite scheme is
// unconditionally stable in the stiff term.
//
// At PS → ∞ each organ's PS-relaxation drives (C_v - C_t/Kp) → 0 instantly,
// reproducing the well-stirred limit when V_v → 0. This is the mathematical
// theorem verified numerically by Case 2 of the parity gate.
//
// References:
//   - Strang G., SIAM J Numer Anal 1968;5:506-17 (operator splitting).
//   - Brown RP et al., Toxicol Ind Health 1997;13:407 (organ vascular fractions).
//   - Rodgers T, Rowland M, J Pharm Sci 2006;95:1238-57 (perm-limited PBPK).
//   - Poulin P, Theil F-P, J Pharm Sci 2002;91:1358-70 (Kp prediction).
//   - JCGM 100:2008 §4.3 (Type B model-form uncertainty).

export { KP, Q, V_REF, CL_HEP_DEFAULT, HIGUCHI_KH_DEFAULT, N } from './pbpk14_core.mjs';
import { KP, Q, V_REF, CL_HEP_DEFAULT, HIGUCHI_KH_DEFAULT, N } from './pbpk14_core.mjs';

// Vascular volume fraction per organ (V_v,i / V_i). Brown 1997 Table 7.
export const VASC_FRAC = Object.freeze([
  1.000,  // 0 blood (central; entire compartment is vascular)
  0.210,  // 1 liver
  0.160,  // 2 kidney
  0.037,  // 3 brain (BBB)
  0.262,  // 4 heart
  0.262,  // 5 lung
  0.040,  // 6 muscle
  0.050,  // 7 adipose
  0.024,  // 8 gut
  0.019,  // 9 skin
  0.041,  // 10 bone
  0.282,  // 11 spleen
  0.180,  // 12 pancreas
  0.050,  // 13 other
]);

// Permeability-surface-area product PS_i (L/h) for rapamycin. logP ≈ 4.3, freely
// permeable across most capillary endothelium; BBB-limited at brain.
// Rodgers-Rowland 2006 + literature meta-review for sirolimus/tacrolimus
// (Möller et al. 2011 in CPT:PSP for related immunosuppressants).
export const PS_RAPAMYCIN = Object.freeze([
  0.0,      // 0 blood — unused
  1800.0,   // 1 liver  (PS/Q = 20, sinusoidal endothelium very permeable)
  1110.0,   // 2 kidney (PS/Q = 15)
    88.0,   // 3 brain  (PS/Q =  2, BBB-limited)
   160.0,   // 4 heart  (PS/Q = 10)
 10500.0,   // 5 lung   (PS/Q = 30, alveolar surface)
   336.0,   // 6 muscle (PS/Q =  8)
    40.0,   // 7 adipose(PS/Q =  4, lipophilic partitioning slow)
   660.0,   // 8 gut    (PS/Q = 12)
   126.0,   // 9 skin   (PS/Q =  6)
    51.0,   // 10 bone  (PS/Q =  3)
   150.0,   // 11 spleen(PS/Q = 15)
    96.0,   // 12 pancr (PS/Q = 12)
    78.0,   // 13 other (PS/Q =  6)
]);

export const DEFAULT_PARAMS_RAPAMYCIN = Object.freeze({
  clHep: CL_HEP_DEFAULT,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: 0.0,
  stentActive: true,
  ps: PS_RAPAMYCIN,
  vascFrac: VASC_FRAC,
  kp: KP,
});

/**
 * Stent release (Higuchi). Identical to pbpk14_core for cross-model parity.
 */
export function releaseRateAt(tHours, params) {
  if (!params.stentActive) return 0;
  const t = Math.max(0.1, tHours);
  return params.higuchiScale * HIGUCHI_KH_DEFAULT / (2 * Math.sqrt(t));
}

/**
 * Exact analytical solve of the per-organ PS-coupling ODE for duration τ at
 * frozen C_blood (which doesn't enter the PS sub-operator at all):
 *   V_v · dC_v/dt = - PS · (C_v - C_t/Kp)
 *   V_t · dC_t/dt = + PS · (C_v - C_t/Kp)
 *
 * Conserved: M = V_v·C_v + V_t·C_t  (mass in organ).
 * Decay rate: λ = PS · (1/V_v + 1/(V_t·Kp))
 * Equilibrium: Cv_eq = M / (V_v + V_t·Kp), Ct_eq = Kp · Cv_eq.
 * Solution: C(τ) = C_eq + (C(0) - C_eq) · exp(-λ·τ).
 *
 * Mutates Cv[i], Ct[i] in place for one organ.
 */
function psRelaxOrgan(Cv, Ct, i, Vv, Vt, Kp, PS, dt) {
  if (PS <= 0 || Vv <= 0 || Vt <= 0 || Kp <= 0) return;
  // Equilibrium: Cv = Ct/Kp ⇒ Ct_eq = Kp · Cv_eq.
  // Mass conservation: Vv·Cv + Vt·Ct = M ⇒ Cv_eq = M / (Vv + Vt·Kp).
  const mass = Vv * Cv[i] + Vt * Ct[i];
  const denom = Vv + Vt * Kp;
  const cvEq = mass / denom;
  const ctEq = Kp * cvEq;
  // Decay rate of z = Cv - Ct/Kp: λ = PS · (1/Vv + 1/(Vt·Kp)).
  const lambda = PS * (1.0 / Vv + 1.0 / (Vt * Kp));
  const decay = Math.exp(-lambda * dt);
  Cv[i] = cvEq + (Cv[i] - cvEq) * decay;
  Ct[i] = ctEq + (Ct[i] - ctEq) * decay;
}

/**
 * Crank-Nicolson step on the SLOW sub-operator (Q-transport + clearance +
 * release; Ct held by the Strang split). The slow operator is the 14-state
 * linear ODE dx/dt = M·x + f where x = [C_blood, Cv_1, ..., Cv_13] and M
 * is an arrow matrix:
 *   M[0,0] = -(Σ Q_i + cl_hep) / V_b
 *   M[0,i] =  Q_i / V_b           [i≥1]
 *   M[i,0] =  Q_i / V_v_i         [i≥1]
 *   M[i,i] = -Q_i / V_v_i         [i≥1]
 * f[0] = release/V_b; f[i] = 0.
 *
 * CN: (I - dt·M/2)·x_new = (I + dt·M/2)·x_old + dt·f_mid
 * Closed-form arrow-matrix solve in O(N):
 *   α_i = (dt/2)·Q_i/V_v_i,  β_i = (dt/2)·Q_i/V_b,  s = (dt/2)·(ΣQ+cl_hep)/V_b
 *   b[0] = (1 - s)·C_blood_old + (dt/(2·V_b))·Σ Q_i·Cv_i_old + dt·release/V_b
 *   b[i] = α_i·C_blood_old + (1 - α_i)·Cv_i_old
 *   γ_i = β_i / (1 + α_i)
 *   x_new[0] = (b[0] + Σ γ_i·b[i]) / ((1 + s) - Σ γ_i·α_i)
 *   x_new[i] = (b[i] + α_i·x_new[0]) / (1 + α_i)
 *
 * CN is A-stable (handles V_v→0 stiffness without sub-stepping) and 2nd-order
 * accurate. Together with the analytical exp() PS-relaxation, the full Strang
 * scheme exp(L_PS·dt/2) ∘ CN(L_slow, dt) ∘ exp(L_PS·dt/2) is unconditionally
 * stable in both stiff terms.
 */
function cnSlowStep(Cv, dt, params, scratch) {
  const half = 0.5 * dt;
  const Vblood = V_REF[0] * params.vdScale;
  const release = releaseRateAt(scratch._tCenter, params);

  let sumQ = 0;
  for (let i = 1; i < N; i++) sumQ += Q[i];
  const s = half * (sumQ + params.clHep) / Vblood;

  const cBloodOld = Cv[0];
  let sumQCv = 0;
  for (let i = 1; i < N; i++) sumQCv += Q[i] * Cv[i];
  const b0 = (1 - s) * cBloodOld + half * sumQCv / Vblood + dt * release / Vblood;

  const alpha = scratch.cnAlpha;
  const bi = scratch.cnBi;
  let denomAcc = 0;
  let numAcc = 0;
  for (let i = 1; i < N; i++) {
    const Vi = V_REF[i] * params.vdScale;
    const Vv = Math.max(Vi * params.vascFrac[i], 1e-30);
    const a = half * Q[i] / Vv;
    alpha[i] = a;
    bi[i] = a * cBloodOld + (1 - a) * Cv[i];
    const beta = half * Q[i] / Vblood;
    const gamma = beta / (1 + a);
    denomAcc += gamma * a;
    numAcc += gamma * bi[i];
  }
  const x0 = (b0 + numAcc) / ((1 + s) - denomAcc);
  Cv[0] = x0 < 0 ? 0 : x0;
  for (let i = 1; i < N; i++) {
    const xi = (bi[i] + alpha[i] * x0) / (1 + alpha[i]);
    Cv[i] = xi < 0 ? 0 : xi;
  }
  return release * dt;
}

/**
 * One Strang-split step: exp(L_PS·dt/2) · CN(L_slow, dt) · exp(L_PS·dt/2).
 * Mutates Cv, Ct in place. Returns mg released this step.
 */
export function stepStrang(Cv, Ct, tHours, dt, params, scratch) {
  scratch._tCenter = tHours + 0.5 * dt;
  const halfDt = 0.5 * dt;

  for (let i = 1; i < N; i++) {
    const Vi = V_REF[i] * params.vdScale;
    const Vv = Vi * params.vascFrac[i];
    const Vt = Vi * (1 - params.vascFrac[i]);
    psRelaxOrgan(Cv, Ct, i, Vv, Vt, params.kp[i], params.ps[i], halfDt);
  }

  const released = cnSlowStep(Cv, dt, params, scratch);

  for (let i = 1; i < N; i++) {
    const Vi = V_REF[i] * params.vdScale;
    const Vv = Vi * params.vascFrac[i];
    const Vt = Vi * (1 - params.vascFrac[i]);
    psRelaxOrgan(Cv, Ct, i, Vv, Vt, params.kp[i], params.ps[i], halfDt);
  }

  return released;
}

export function makeScratch() {
  const a = () => new Float64Array(N);
  return {
    cnAlpha: a(),
    cnBi: a(),
    _tCenter: 0,
  };
}

/**
 * Bolus initial state. Drug enters Cv[0] = C_blood; interstitial sub-comps
 * start at zero. (Ct[0] is unused and kept at zero.)
 */
export function initialState(params) {
  const Cv = new Float64Array(N);
  const Ct = new Float64Array(N);
  Cv[0] = params.bolusMg / (V_REF[0] * params.vdScale);
  return { Cv, Ct };
}

/**
 * Organ-average concentration C_avg,i = vasc_frac·C_v + (1-vasc_frac)·C_t.
 * Comparable quantity to pbpk14_core's C[i] for cross-model display.
 */
export function organAverage(Cv, Ct, params) {
  const out = new Float64Array(N);
  out[0] = Cv[0];
  for (let i = 1; i < N; i++) {
    const vF = params.vascFrac[i];
    out[i] = vF * Cv[i] + (1 - vF) * Ct[i];
  }
  return out;
}

/**
 * Degenerate-config helper for Case 2 of the parity gate. Drives the
 * vascular fraction → 0 and PS → ∞ asymptotic limit. As (eps→0, psScale→∞)
 * the resulting trajectory must converge to PBPK14 well-stirred (with C_t →
 * organ-average).
 */
export function degenerateParams(base, { eps = 1e-3, psScale = 1e4 } = {}) {
  const vasc = new Float64Array(N);
  const ps = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    vasc[i] = i === 0 ? 1.0 : eps;
    ps[i] = base.ps[i] * psScale;
  }
  return { ...base, vascFrac: vasc, ps };
}
