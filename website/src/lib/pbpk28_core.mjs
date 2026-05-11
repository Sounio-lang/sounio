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

// Rapamycin molecular weight (g/mol) — needed to convert PBPK28 mass
// concentrations (mg/L) to molar concentrations (nmol/L) for the TMDD layer.
export const MW_RAPAMYCIN = 914.17;

// TMDD organ indices for rapamycin → FKBP12 / mTORC1 binding.
// Liver and gut are major FKBP12 reservoirs; heart is the Cypher-stent
// clinical-endpoint organ (vascular smooth muscle proliferation).
export const TMDD_ORGANS_RAPAMYCIN = Object.freeze([1, 4, 8]);  // liver, heart, gut

// Per-TMDD-organ parameters, indexed by full 14-organ index (0..13).
// Non-TMDD organs map to null. Literature: Mager 2004 (sirolimus FKBP12 TMDD;
// FKBP12 cellular levels Schreiber 1991; receptor turnover Bohnacker 1992).
//
// Units:
//   k_on   [L / (nmol · h)]    binding rate constant
//   k_off  [1/h]                dissociation rate constant
//   k_int  [1/h]                bound-complex internalization
//   k_syn  [nmol / (L · h)]    receptor synthesis
//   k_deg  [1/h]                free-receptor degradation
//   r_total_0 [nmol/L]         baseline R_free at steady state = k_syn / k_deg
function tmddParams(rTotal0_nM, kDeg_per_h, kOn_per_nM_per_h, Kd_nM, kInt_per_h) {
  return Object.freeze({
    rTotal0: rTotal0_nM,
    kDeg:    kDeg_per_h,
    kSyn:    kDeg_per_h * rTotal0_nM,         // steady-state: k_syn = k_deg · R_total_0
    kOn:     kOn_per_nM_per_h,
    kOff:    kOn_per_nM_per_h * Kd_nM,        // K_d = k_off / k_on
    kInt:    kInt_per_h,
  });
}

export const TMDD_PARAMS_RAPAMYCIN = Object.freeze({
  1: tmddParams(/*R_total*/ 50.0,  /*k_deg*/ 0.05, /*k_on*/ 0.10, /*K_d*/ 0.10, /*k_int*/ 0.010),  // liver
  4: tmddParams(                   25.0,            0.05,         0.10,        0.10,            0.010),  // heart (coronary SMC)
  8: tmddParams(                   30.0,            0.05,         0.10,        0.10,            0.010),  // gut
});

export const DEFAULT_PARAMS_RAPAMYCIN = Object.freeze({
  clHep: CL_HEP_DEFAULT,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: 0.0,
  stentActive: true,
  ps: PS_RAPAMYCIN,
  vascFrac: VASC_FRAC,
  kp: KP,
  mw: MW_RAPAMYCIN,
  tmddOrgans: TMDD_ORGANS_RAPAMYCIN,
  tmddParams: TMDD_PARAMS_RAPAMYCIN,
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
 * Fully-coupled Crank-Nicolson step on the **27-state** PBPK28 system.
 * No operator splitting — both the PS coupling and the Q transport are
 * integrated implicitly in a single CN step.
 *
 * State vector x = [C_blood, Cv_1, Ct_1, Cv_2, Ct_2, ..., Cv_13, Ct_13].
 * M is a block-arrow matrix: a 1×1 blood diagonal, 13 organ 2×2 diagonal
 * blocks, and only the C_blood ↔ Cv_i off-diagonal coupling. Block structure
 * lets the CN linear system (I - dt·M/2)·x_new = (I + dt·M/2)·x_old + dt·f
 * be solved in O(N) via per-organ Schur-complement elimination:
 *
 *   Per organ i (h = dt/2):
 *     p_i = 1 + h·(Q_i + PS_i)/V_v_i           (Cv self in A = I - h·M)
 *     q_i = h·PS_i/(V_v_i·Kp_i)                (Cv ← Ct coupling)
 *     r_i = h·PS_i/V_t_i                       (Ct ← Cv coupling)
 *     s_i = 1 + h·PS_i/(V_t_i·Kp_i)            (Ct self in A)
 *     D_i = p_i·s_i - q_i·r_i                  (organ-block determinant)
 *     γ_i = h·Q_i/V_v_i                        (Cv ← Cb coupling)
 *     β_i = h·Q_i/V_b                          (Cb ← Cv coupling)
 *
 *   RHS = B·x_old + dt·f  where B = I + h·M:
 *     rhs_0   = (1-S)·Cb_old + Σ β_i·Cv_i_old + dt·release/V_b   (S = h·(ΣQ+cl)/V_b)
 *     rhs_v_i = γ_i·Cb_old + (1 - h(Q+PS)/V_v)·Cv_i_old + q_i·Ct_i_old
 *     rhs_t_i = r_i·Cv_i_old + (1 - h·PS/(V_t·Kp))·Ct_i_old
 *
 *   Per-organ Schur elimination (organ block A_i is 2×2 invertible):
 *     Cv_i_new = a_v_i + b_v_i·Cb_new  with  a_v_i = (s_i·rhs_v + q_i·rhs_t)/D_i,
 *                                            b_v_i =  s_i·γ_i / D_i
 *     Ct_i_new = a_t_i + b_t_i·Cb_new  with  a_t_i = (r_i·rhs_v + p_i·rhs_t)/D_i,
 *                                            b_t_i =  r_i·γ_i / D_i
 *
 *   Scalar reduction for blood (row 0 of A · x_new = rhs):
 *     (1+S)·Cb_new - Σ β_i·(a_v_i + b_v_i·Cb_new) = rhs_0
 *     Cb_new = (rhs_0 + Σ β_i·a_v_i) / ((1+S) - Σ β_i·b_v_i)
 *
 *   Back-substitute for Cv_i_new, Ct_i_new.
 *
 * Properties:
 *   - A-stable (any stiff eigenvalue of M is integrated stably).
 *   - 2nd-order accurate.
 *   - Reduces to PBPK14 well-stirred in the (V_v→0, PS→∞) limit because
 *     no splitting introduces non-commutativity error.
 *   - O(N) per step (13 organ-block inversions + 1 scalar solve + 13 back-subs).
 */
export function stepStrang(Cv, Ct, tHours, dt, params, scratch, Rfree, DR) {
  scratch._tCenter = tHours + 0.5 * dt;
  const h = 0.5 * dt;
  const Vb = V_REF[0] * params.vdScale;
  const release = releaseRateAt(scratch._tCenter, params);

  let sumQ = 0;
  for (let i = 1; i < N; i++) sumQ += Q[i];
  const S = h * (sumQ + params.clHep) / Vb;

  const cbOld = Cv[0];
  let rhs0 = (1 - S) * cbOld + dt * release / Vb;

  const Av = scratch.aV;
  const Bv = scratch.bV;
  const At = scratch.aT;
  const Bt = scratch.bT;
  let sumBetaA = 0;
  let sumBetaB = 0;

  for (let i = 1; i < N; i++) {
    const Vi = V_REF[i] * params.vdScale;
    const Vv = Math.max(Vi * params.vascFrac[i], 1e-30);
    const Vt = Math.max(Vi * (1 - params.vascFrac[i]), 1e-30);
    const Kp = params.kp[i];
    const PS = params.ps[i];
    const Qi = Q[i];

    const beta = h * Qi / Vb;
    const gamma = h * Qi / Vv;
    const p = 1 + h * (Qi + PS) / Vv;
    const q = h * PS / (Vv * Kp);
    const r = h * PS / Vt;
    const s = 1 + h * PS / (Vt * Kp);
    const D = p * s - q * r;

    // B · x_old contributions
    rhs0 += beta * Cv[i];
    const rhsV = gamma * cbOld + (1 - h * (Qi + PS) / Vv) * Cv[i] + q * Ct[i];
    const rhsT = r * Cv[i] + (1 - h * PS / (Vt * Kp)) * Ct[i];

    // Linear-in-Cb_new coefficients
    const aV = (s * rhsV + q * rhsT) / D;
    const bV = (s * gamma) / D;
    const aT = (r * rhsV + p * rhsT) / D;
    const bT = (r * gamma) / D;

    Av[i] = aV;
    Bv[i] = bV;
    At[i] = aT;
    Bt[i] = bT;

    sumBetaA += beta * aV;
    sumBetaB += beta * bV;
  }

  const cbNew = (rhs0 + sumBetaA) / ((1 + S) - sumBetaB);
  Cv[0] = cbNew < 0 ? 0 : cbNew;
  for (let i = 1; i < N; i++) {
    const cv = Av[i] + Bv[i] * cbNew;
    const ct = At[i] + Bt[i] * cbNew;
    Cv[i] = cv < 0 ? 0 : cv;
    Ct[i] = ct < 0 ? 0 : ct;
  }

  // ─── TMDD passive observer (G-β-1) ────────────────────────────────────────
  // After the PBPK28 CN step, per-TMDD-organ receptor-binding ODE evolves on
  // (R_free, DR) at frozen C_t_new. Linear at fixed C_t ⇒ analytical 2×2 CN
  // per organ. v1 is a *passive observer* (no -k_on·C_t·R_free back-reaction
  // on C_t) — adequate when [drug] >> [receptor]; back-reaction added in G-β-2.
  //
  //   dR_free/dt = k_syn - k_deg·R_free - k_on·C_t·R_free + k_off·DR
  //   dDR/dt     = k_on·C_t·R_free - (k_off + k_int)·DR
  //
  //   2×2 matrix at fixed C_t (in nmol/L; Ct_mass[mg/L] → Ct_nM via MW):
  //     M_TMDD = [-k_deg - k_on·C_t_nM,  +k_off                 ]
  //              [+k_on·C_t_nM,           -(k_off + k_int)       ]
  //     f      = [k_syn; 0]
  //   CN: (I - dt·M/2)·x_new = (I + dt·M/2)·x_old + dt·f_mid
  if (Rfree && DR && params.tmddOrgans && params.tmddParams && params.mw) {
    const h = 0.5 * dt;
    const mass_to_nM = 1.0e6 / params.mw;  // mg/L → nmol/L
    for (const j of params.tmddOrgans) {
      const tp = params.tmddParams[j];
      if (!tp) continue;
      const ctNM = Math.max(Ct[j], 0) * mass_to_nM;
      const kCt = tp.kOn * ctNM;
      // A = I - h·M
      const a11 =  1 + h * (tp.kDeg + kCt);
      const a12 = -h * tp.kOff;
      const a21 = -h * kCt;
      const a22 =  1 + h * (tp.kOff + tp.kInt);
      // RHS = (I + h·M)·x_old + dt·f
      const rhsR = (1 - h * (tp.kDeg + kCt)) * Rfree[j] + (h * tp.kOff) * DR[j] + dt * tp.kSyn;
      const rhsD = (h * kCt) * Rfree[j] + (1 - h * (tp.kOff + tp.kInt)) * DR[j];
      const det = a11 * a22 - a12 * a21;
      const rNew = (rhsR * a22 - rhsD * a12) / det;
      const dNew = (a11 * rhsD - a21 * rhsR) / det;
      Rfree[j] = rNew < 0 ? 0 : rNew;
      DR[j]    = dNew < 0 ? 0 : dNew;
    }
  }

  return release * dt;
}

export function makeScratch() {
  const a = () => new Float64Array(N);
  return {
    aV: a(),
    bV: a(),
    aT: a(),
    bT: a(),
    _tCenter: 0,
  };
}

/**
 * Bolus initial state. Drug enters Cv[0] = C_blood; interstitial sub-comps
 * start at zero. TMDD organs start at R_free = R_total_0 (free-receptor
 * baseline at no-drug steady state), DR = 0.
 */
export function initialState(params) {
  const Cv = new Float64Array(N);
  const Ct = new Float64Array(N);
  const Rfree = new Float64Array(N);
  const DR = new Float64Array(N);
  Cv[0] = params.bolusMg / (V_REF[0] * params.vdScale);
  if (params.tmddOrgans && params.tmddParams) {
    for (const j of params.tmddOrgans) {
      const tp = params.tmddParams[j];
      if (tp) Rfree[j] = tp.rTotal0;
    }
  }
  return { Cv, Ct, Rfree, DR };
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
