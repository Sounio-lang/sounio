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

// ─── PD layer: mTORC1 activity + neointimal proliferation (G-γ) ───────────
// PD organs: heart (i=4) as coronary smooth-muscle (coronary_smc) proxy.
// G-γ-2 will refine by carving a 5%-of-heart-mass sub-compartment from
// heart for the true coronary_smc; G-γ-1 uses the whole heart and is
// directionally correct for the dissertation message.
//
//   A(t)  mTORC1 active fraction in target tissue (dimensionless, 0..1).
//         At baseline (no drug): A = 1.0 (full activity).
//         Under drug: A → R_free / R_total_0 (fraction of FKBP12 unbound).
//
//   N(t)  Neointimal proliferation index (dimensionless, normalized 0..1).
//         dN/dt = k_prolif · A - k_apo · (1 - A) · N
//         At A=1 (no drug): dN/dt = k_prolif (linear neointimal growth →
//                                  restenosis if untreated).
//         At A=0 (full inhibition): dN/dt = -k_apo · N (regression).
//
// PD parameters from Mehilli 2003 (Cypher restenosis kinetics) and Nakazawa
// 2010 (sirolimus VSMC turnover). Times converted from days⁻¹ to hours⁻¹.
export const PD_ORGANS_RAPAMYCIN = Object.freeze([4]);   // heart / coronary_smc

function pdParams(kA_per_h, kProlif_per_h, kApo_per_h, rTotal0_nM) {
  return Object.freeze({
    kA:       kA_per_h,        // mTORC1 signaling-cascade rate (~1.0 /h, ~40 min half-life)
    kProlif:  kProlif_per_h,   // neointimal proliferation (~5e-4 /h, ~1.2%/day)
    kApo:     kApo_per_h,      // apoptosis under mTOR inhibition (~3e-3 /h, ~7%/day)
    rTotal0:  rTotal0_nM,      // reference R_total for normalizing A
  });
}

export const PD_PARAMS_RAPAMYCIN = Object.freeze({
  4: pdParams(/*k_a*/ 1.0, /*k_prolif*/ 5.0e-4, /*k_apo*/ 3.0e-3, /*R_total_0*/ 25.0),
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
  releaseModel: 'higuchi',                  // Cypher stent diffusion (Cordis 2003)
  releaseKH: HIGUCHI_KH_DEFAULT,
  tmddOrgans: TMDD_ORGANS_RAPAMYCIN,
  tmddParams: TMDD_PARAMS_RAPAMYCIN,
  pdOrgans:   PD_ORGANS_RAPAMYCIN,
  pdParams:   PD_PARAMS_RAPAMYCIN,
  cVisualMax: 1.3e-3,                       // mg/L, viewer organ-color saturation anchor
  drugLabel: 'Rapamycin (Cypher stent)',
  releaseSourceLabel: 'Cypher stent (Cordis 2003 Higuchi diffusion)',
});

// ─── Semaglutide profile (G-δ): GLP-1 receptor agonist, peptide, SC depot ──
// Overgaard 2019 (Clin Pharmacokinet) — semaglutide PK in T2D patients:
//   MW = 4113.58 g/mol         (peptide, vs 914 for rapamycin)
//   V_d ≈ 9 L (70 kg)          (vascular-confined; FcRn-mediated albumin
//                              recycling extends half-life to ~165 h)
//   CL_proteolytic ≈ 0.077 L/h  (very low; receptor-mediated + DPP-IV cleavage)
//   t1/2_abs ≈ 60 h            (slow SC absorption from depot, k_a = ln2/t)
//   F = 0.89                   (subcutaneous bioavailability)
//
// PBPK Kp for a peptide is dominated by tissue-vascular partitioning; almost
// all values are well below 1 (peptide stays in plasma/interstitial fluid).
// PS is much smaller than rapamycin (peptide endothelial transit slow,
// especially across BBB).
export const MW_SEMAGLUTIDE = 4113.58;

export const KP_SEMAGLUTIDE = Object.freeze([
  1.00,  // 0 blood (definition)
  0.50,  // 1 liver
  0.60,  // 2 kidney
  0.05,  // 3 brain  (BBB extremely restrictive)
  0.40,  // 4 heart
  0.30,  // 5 lung
  0.20,  // 6 muscle
  0.10,  // 7 adipose
  0.80,  // 8 gut   (enteroendocrine GLP-1R distribution)
  0.30,  // 9 skin
  0.20,  // 10 bone
  0.40,  // 11 spleen
  0.70,  // 12 pancreas (β-cell GLP-1R target)
  0.30,  // 13 other
]);

// Peptide PS: literature scarce; estimate from MW-based scaling of Rodgers &
// Rowland 2006 + biologic PBPK (Chen 2014, Zhao 2020): PS scales roughly as
// 1/sqrt(MW), so semaglutide PS ≈ rapamycin PS · sqrt(914/4114) ≈ 0.47×, but
// peptides also suffer additional steric exclusion at the endothelium —
// reduce another 5–10×.
export const PS_SEMAGLUTIDE = Object.freeze([
   0.0,    // 0 blood — unused
  50.0,    // 1 liver  (sinusoidal endothelium still relatively permeable)
  30.0,    // 2 kidney
   1.0,    // 3 brain  (BBB nearly impermeable to peptide)
   5.0,    // 4 heart
 100.0,    // 5 lung   (alveolar surface, larger area)
  10.0,    // 6 muscle
   2.0,    // 7 adipose
  30.0,    // 8 gut    (GLP-1R distribution)
   5.0,    // 9 skin   (also SC depot exit, but modelled separately)
   2.0,    // 10 bone
  10.0,    // 11 spleen
   8.0,    // 12 pancreas
   5.0,    // 13 other
]);

// Semaglutide subcutaneous depot release: dQ/dt = k_a · Q_depot,
// Q_depot(t) = F · Dose · exp(-k_a · t), giving rate = F · Dose · k_a · exp(-k_a · t).
// Default k_a = ln(2) / 60 h ≈ 0.01155 /h, F = 0.89.
export const SEMA_KA_DEFAULT = Math.LN2 / 60.0;
export const SEMA_F_DEFAULT  = 0.89;

// ─── Semaglutide TMDD: GLP-1R binding (pancreas, gut, brain) — G-δ-2 ──────
// GLP-1R distribution and density from Knudsen 2019 (semaglutide vs liraglutide
// receptor occupancy) + Lau 2015 (sema-GLP-1R affinity K_d ≈ 0.4 nM).
// Pancreatic β-cells are the highest-density site; enteroendocrine L-cells in
// gut and arcuate-nucleus neurons in brain are lower but clinically relevant
// (gut → satiety signaling; brain → central anorectic effect).
export const TMDD_ORGANS_SEMAGLUTIDE = Object.freeze([3, 8, 12]);  // brain, gut, pancreas

export const TMDD_PARAMS_SEMAGLUTIDE = Object.freeze({
  3:  tmddParams(/*R_total*/ 1.0, /*k_deg*/ 0.05, /*k_on*/ 0.50, /*K_d*/ 0.40, /*k_int*/ 0.050),  // brain
  8:  tmddParams(                 2.0,            0.05,         0.50,        0.40,            0.050),  // gut
  12: tmddParams(                 5.0,            0.05,         0.50,        0.40,            0.050),  // pancreas
});

// ─── Semaglutide PD: glucose-insulin Bergman minimal model (G-δ-3) ────────
// Couples pancreas GLP-1R occupancy to a 2-state linearized glucose-insulin
// system. States (stored at PD organ index = pancreas):
//   A[12]  =  ΔG  (mg/dL)  plasma glucose deviation from basal Gb
//   Ni[12] =  ΔI  (mU/L)   plasma insulin deviation from basal Ib
//
// Insulin secretion is augmented by GLP-1R activation in pancreas β-cells:
//   dΔI/dt = k_base · α · occ - kI · ΔI
//     where occ = DR_panc / (R_free_panc + DR_panc),
//           k_base = kI · Ib (steady-state secretion at no drug),
//           α = 1 (full GLP-1R occupancy doubles insulin secretion).
//
// Glucose dynamics (linearized at G ≈ Gb to keep 2-state linear):
//   dΔG/dt = -(SG + SI · Ib) · ΔG - SI · Gb · ΔI
//
// Bergman 1979 + UVA/Padova 2013 minimal parameters, scaled from /min to /h.
export const PD_ORGANS_SEMAGLUTIDE = Object.freeze([12]);    // pancreas as host

function pdParamsGlucoseInsulin(Gb, Ib, SG_h, SI_h, kI_h, alphaGLP1) {
  return Object.freeze({
    pdModel: 'glucose_insulin',
    Gb: Gb,                            // basal glucose [mg/dL]
    Ib: Ib,                            // basal insulin [mU/L]
    SG: SG_h,                          // glucose effectiveness [1/h]
    SI: SI_h,                          // insulin sensitivity [1/(mU·L·h)]
    kI: kI_h,                          // insulin elimination [1/h]
    kBase: kI_h * Ib,                  // baseline secretion [mU/(L·h)]
    alpha: alphaGLP1,                  // GLP-1R relative augmentation (1 = doubles at full occ)
  });
}

export const PD_PARAMS_SEMAGLUTIDE = Object.freeze({
  12: pdParamsGlucoseInsulin(/*Gb*/ 100.0, /*Ib*/ 10.0, /*SG*/ 1.5,
                             /*SI*/ 0.020, /*kI*/ 12.6, /*α*/ 1.0),
});

// Mark the rapamycin PD params as the mTORC1_neointima model so the
// step-dispatcher can distinguish (other arms might add new PD models later).
// Reassign PD_PARAMS_RAPAMYCIN inline above? — simpler: dispatch on the
// presence of `pdModel`. Default for rapamycin pdParams (no pdModel field)
// is mTORC1+neointima; semaglutide explicitly tags itself.

export const DEFAULT_PARAMS_SEMAGLUTIDE = Object.freeze({
  clHep: 0.077,                             // L/h — proteolytic clearance (Overgaard 2019)
  higuchiScale: 1.0,                        // unused for SC depot release
  vdScale: 1.0,
  bolusMg: 0.0,
  stentActive: true,                        // re-used as "depot active" toggle
  ps: PS_SEMAGLUTIDE,
  vascFrac: VASC_FRAC,
  kp: KP_SEMAGLUTIDE,
  mw: MW_SEMAGLUTIDE,
  releaseModel: 'sc_depot',
  releaseKa: SEMA_KA_DEFAULT,
  releaseF:  SEMA_F_DEFAULT,
  releaseDoseMg: 1.0,                       // typical weekly dose
  tmddOrgans: TMDD_ORGANS_SEMAGLUTIDE,
  tmddParams: TMDD_PARAMS_SEMAGLUTIDE,
  pdOrgans:   PD_ORGANS_SEMAGLUTIDE,
  pdParams:   PD_PARAMS_SEMAGLUTIDE,
  cVisualMax: 3.0e-5,                       // mg/L, peak blood ~25 ng/mL ≈ 2.5e-5
  drugLabel: 'Semaglutide (SC depot)',
  releaseSourceLabel: 'Subcutaneous depot (k_a = ln 2 / 60 h, F = 0.89)',
});

/**
 * Stent release (Higuchi). Identical to pbpk14_core for cross-model parity.
 */
export function releaseRateAt(tHours, params) {
  if (!params.stentActive) return 0;
  // Dispatch on releaseModel; default to Higuchi for back-compat.
  const model = params.releaseModel || 'higuchi';
  if (model === 'sc_depot') {
    // SC depot: rate = F · Dose · k_a · exp(-k_a · t)   (1st-order absorption)
    // higuchiScale is reused as the dose-multiplier slider (G-ε-4).
    const ka = params.releaseKa;
    const F  = params.releaseF;
    const D  = params.releaseDoseMg * (params.higuchiScale ?? 1);
    return F * D * ka * Math.exp(-ka * Math.max(0, tHours));
  }
  // Higuchi (default): dQ/dt = K_H / (2·√t), clipped at t < 0.1 h
  const t = Math.max(0.1, tHours);
  const kH = params.releaseKH || HIGUCHI_KH_DEFAULT;
  return params.higuchiScale * kH / (2 * Math.sqrt(t));
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
export function stepStrang(Cv, Ct, tHours, dt, params, scratch, Rfree, DR, A, Ni) {
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

  // ─── TMDD layer with back-reaction (G-β-2) ────────────────────────────────
  // Full per-TMDD-organ 3-state implicit step on (C_t_nM, R_free, DR). All
  // states in nmol/L; C_t is converted from mass at start and back to mass at
  // end. The bilinear k_on·C_t_nM·R_free term is Newton-linearized at the
  // start-of-step values (C_t_old, R_free_old), giving a single-iteration
  // implicit Crank-Nicolson solve. For the rapamycin/FKBP12 regime (K_d ≪
  // R_total, moderate dt = 1e-3 h), the residual non-linear error is below
  // 1% per step — well inside the gate threshold.
  //
  //   dC_t_nM/dt = -k_on·C_t·R_free + k_off·DR
  //   dR_free/dt =  k_syn - k_deg·R_free - k_on·C_t·R_free + k_off·DR
  //   dDR/dt    =  k_on·C_t·R_free - (k_off + k_int)·DR
  //
  // Newton linearization at (C_t_old, R_free_old):
  //   k_on·C_t·R_free  ≈  k_on·(C_t_old·R_free + R_free_old·C_t - C_t_old·R_free_old)
  // The constant -k_on·C_t_old·R_free_old is moved to the forcing f.
  //
  // 3×3 linear ODE per organ, 3×3 CN invert per step.
  // Mass conservation: drug leaves Ct only via PBPK clearance (already
  // handled) or internalization (-k_int·DR moves bound drug out of the
  // interstitial pool permanently — modelled as drug loss).
  if (Rfree && DR && params.tmddOrgans && params.tmddParams && params.mw) {
    const h = 0.5 * dt;
    const mass_to_nM = 1.0e6 / params.mw;     // mg/L → nmol/L
    const nM_to_mass = params.mw * 1.0e-6;    // nmol/L → mg/L
    for (const j of params.tmddOrgans) {
      const tp = params.tmddParams[j];
      if (!tp) continue;

      const ctOld_nM = Math.max(Ct[j], 0) * mass_to_nM;
      const rfOld    = Rfree[j];
      const drOld    = DR[j];

      const kOnR = tp.kOn * rfOld;            // coefficient on Ct in bilinear
      const kOnCt = tp.kOn * ctOld_nM;        // coefficient on R_free in bilinear
      const bilOld = tp.kOn * ctOld_nM * rfOld;   // Newton constant offset

      // M (3×3) — Newton-linearized at (ctOld_nM, rfOld):
      //   row 0 (Ct):     [-kOnR,        -kOnCt,                  +k_off    ]
      //   row 1 (R_free): [-kOnR,        -k_deg - kOnCt,          +k_off    ]
      //   row 2 (DR):     [+kOnR,        +kOnCt,                  -(k_off+k_int)]
      // f (forcing): [+bilOld; k_syn + bilOld; -bilOld]
      //
      // A = I - h·M; B = I + h·M.
      const m00 = -kOnR,           m01 = -kOnCt,                       m02 =  tp.kOff;
      const m10 = -kOnR,           m11 = -tp.kDeg - kOnCt,             m12 =  tp.kOff;
      const m20 =  kOnR,           m21 =  kOnCt,                       m22 = -(tp.kOff + tp.kInt);
      const f0 =  bilOld;
      const f1 =  tp.kSyn + bilOld;
      const f2 = -bilOld;

      // RHS = (I + h·M)·x_old + dt·f
      const rhs0 = (1 + h*m00) * ctOld_nM + (h*m01) * rfOld + (h*m02) * drOld + dt * f0;
      const rhs1 = (h*m10) * ctOld_nM + (1 + h*m11) * rfOld + (h*m12) * drOld + dt * f1;
      const rhs2 = (h*m20) * ctOld_nM + (h*m21) * rfOld + (1 + h*m22) * drOld + dt * f2;

      // A = I - h·M
      const a00 = 1 - h*m00, a01 = -h*m01, a02 = -h*m02;
      const a10 = -h*m10,    a11 = 1 - h*m11, a12 = -h*m12;
      const a20 = -h*m20,    a21 = -h*m21, a22 = 1 - h*m22;

      // 3×3 invert via cofactor expansion.
      const c00 = a11*a22 - a12*a21;
      const c01 = a12*a20 - a10*a22;
      const c02 = a10*a21 - a11*a20;
      const det = a00*c00 + a01*c01 + a02*c02;

      const c10 = a02*a21 - a01*a22;
      const c11 = a00*a22 - a02*a20;
      const c12 = a01*a20 - a00*a21;
      const c20 = a01*a12 - a02*a11;
      const c21 = a02*a10 - a00*a12;
      const c22 = a00*a11 - a01*a10;

      const ctNew_nM = (c00*rhs0 + c10*rhs1 + c20*rhs2) / det;
      const rfNew    = (c01*rhs0 + c11*rhs1 + c21*rhs2) / det;
      const drNew    = (c02*rhs0 + c12*rhs1 + c22*rhs2) / det;

      Ct[j]    = ctNew_nM < 0 ? 0 : ctNew_nM * nM_to_mass;
      Rfree[j] = rfNew    < 0 ? 0 : rfNew;
      DR[j]    = drNew    < 0 ? 0 : drNew;
    }
  }

  // ─── PD layer: mTORC1 activity A + neointimal proliferation N (G-γ) ───────
  // Per PD organ at frozen (R_free, DR) from the TMDD step:
  //   target_A = R_free / R_total_0      (free-receptor fraction)
  //   dA/dt    = k_a · (target_A - A)    1st-order tracker (signaling lag)
  //   dN/dt    = k_prolif · A - k_apo · (1-A) · N
  // Both 1-state implicit CN per organ. A solves first at frozen target; N
  // then solves at frozen A_new (Lie split — accuracy ample for the slow PD
  // timescales of 1.2%/day proliferation and 7%/day apoptosis).
  if (A && Ni && params.pdOrgans && params.pdParams && Rfree) {
    const h = 0.5 * dt;
    for (const j of params.pdOrgans) {
      const pp = params.pdParams[j];
      if (!pp) continue;

      if (pp.pdModel === 'glucose_insulin') {
        // ── Semaglutide PD: Bergman glucose-insulin (G-δ-3) ──────────────
        // A[j] = ΔG (mg/dL), Ni[j] = ΔI (mU/L)
        // Occupancy = DR / (Rfree + DR) at this organ (pancreas).
        const denom = Rfree[j] + DR[j];
        const occ = denom > 1e-9 ? DR[j] / denom : 0;

        // CN on dΔI/dt = k_base · α · occ - kI · ΔI
        const diNew = ((1 - h * pp.kI) * Ni[j] + dt * pp.kBase * pp.alpha * occ) / (1 + h * pp.kI);
        Ni[j] = diNew;

        // CN on dΔG/dt = -λG · ΔG - SI · Gb · ΔI_new   (linear at G ≈ Gb)
        const lambdaG = pp.SG + pp.SI * pp.Ib;
        const dgNew = ((1 - h * lambdaG) * A[j] - dt * pp.SI * pp.Gb * Ni[j]) / (1 + h * lambdaG);
        A[j] = dgNew;
      } else {
        // ── Rapamycin PD: mTORC1 activity + neointimal index (G-γ) ───────
        const targetA = Rfree[j] / pp.rTotal0;
        const aNew = ((1 - h * pp.kA) * A[j] + dt * pp.kA * targetA) / (1 + h * pp.kA);
        A[j] = aNew < 0 ? 0 : (aNew > 1 ? 1 : aNew);

        const oneMinusA = 1 - A[j];
        const gammaN = pp.kApo * oneMinusA;
        const nNew = ((1 - h * gammaN) * Ni[j] + dt * pp.kProlif * A[j]) / (1 + h * gammaN);
        Ni[j] = nNew < 0 ? 0 : nNew;
      }
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
  // PD states: A starts at 1.0 (full mTORC1 activity, no drug), N at 0.
  const A = new Float64Array(N);
  const Nidx = new Float64Array(N);
  Cv[0] = params.bolusMg / (V_REF[0] * params.vdScale);
  if (params.tmddOrgans && params.tmddParams) {
    for (const j of params.tmddOrgans) {
      const tp = params.tmddParams[j];
      if (tp) Rfree[j] = tp.rTotal0;
    }
  }
  if (params.pdOrgans && params.pdParams) {
    for (const j of params.pdOrgans) {
      const pp = params.pdParams[j];
      if (!pp) continue;
      if (pp.pdModel === 'glucose_insulin') {
        // ΔG and ΔI both start at 0 (system at baseline before drug)
        A[j] = 0.0;
        Nidx[j] = 0.0;
      } else {
        // Rapamycin: A starts at 1.0 (baseline mTORC1 activity), N at 0.
        A[j] = 1.0;
      }
    }
  }
  return { Cv, Ct, Rfree, DR, A, N: Nidx };
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

// ════════════════════════════════════════════════════════════════════════════
// VENLAFAXINE XR — controlled-release witness (SISTEMA 2: Korsmeyer-Peppas matrix)
//
// Third canonical drug. Two coupled PBPK28 compartments: parent (venlafaxine)
// and active metabolite ODV (O-desmethylvenlafaxine), bridged by hepatic CYP2D6
// formation. Oral XR input is gated by a Korsmeyer-Peppas erodible matrix.
//
// Independent JS reimplementation of stdlib/darwin_pbpk/scenarios/venlafaxine_xr.sio
// (portal first-pass model) and of the TR-BDF2 routed-sink step of
// stdlib/darwin_pbpk/theta_pbpk28.sio. Companion of
// tests/run-pass/dissertation_pbpk28_parity_ref_venlafaxine.sio, which drives the
// stdlib scenario functions themselves; scripts/ci/dissertation_pbpk28_parity_gate.sh
// cases 10-13 diff the two. Operation order follows the Sounio source so the two
// engines agree to f64 rounding, not merely within the 1 % gate.
//
// Per step (Lie-Trotter):
//   1-2. gut pool dG/dt = R − ka·G solved exactly over the step, the matrix
//      release entering at a constant rate over its release window (the whole
//      step, or up to matrix saturation); F_abs of the mass leaving the lumen
//      enters the PORTAL vein (liver vascular space), 1 − F_abs never does
//   3. parent TR-BDF2 step, input routed to the liver, all hepatic clearance a
//      liver sink inside the implicit solve; ODV formation = removed · X_form/X
//   4. ODV TR-BDF2 step, formed mass into blood
// F_H = Q/(Q + X) and presystemic ODV formation emerge from step 3, so the
// overall F_oral = F_abs·F_H is an output (0.45 at NM by construction).
//
// Hepatic resolution (well-stirred, portal input, CL_R on blood; see the Sounio
// header): Dose/AUC_b = (c·X + CL_R)/F_abs, Ae_i/AUC_b = c·X_i, c = (Q + CL_R)/Q,
// F = F_abs·Q/(Q + X). Lessard 1999 (EM CL/F 100, CL_m 43, quinidine 17, CL_R 4)
// and Patat 1998 (F 0.45) resolve F_abs, X, X_form, X_non2D6 in closed form.
//
// Sources: Gohel 2008 (matrix n=0.65/k=0.199), Wang 2022 (ka 0.63), Patat 1998
// (F 0.45), Lessard 1999 (oral clearances), Wyeth label / Klamerus 1996 (ODV
// apparent CL 28 L/h). CYP2D6 phenotype scale = literature ODV/V ratio / 3.45
// (provenance unverified; pgx/cyp2d6_venlafaxine.sio).
// ════════════════════════════════════════════════════════════════════════════

export const VFX_MATRIX_GOHEL2008 = Object.freeze({ totalDose: 75.0, k: 0.199, n: 0.65 });
export const VFX_F_ORAL_XR = 0.45;          // Patat 1998, F = F_abs·F_H
export const VFX_KA_ABS    = 0.63;          // Wang 2022, h⁻¹
export const VFX_CL_ORAL_NM = 100.0;        // Lessard 1999, EM Dose/AUC (L/h)
export const VFX_CL_FORM_ODV_NM = 43.0;     // Lessard 1999, EM Ae(ODV)/AUC (L/h)
export const VFX_CL_ORAL_QUINIDINE_EM = 17.0; // Lessard 1999, CYP2D6 blocked (L/h)
export const VFX_CL_RENAL = 4.0;            // Lessard 1999 (Troy 1996 ≈ 3.7), L/h
export const VFX_CL_ODV_APPARENT = 28.0;    // Wyeth label 0.4 L/h/kg × 70 kg, Dose/AUC_ODV
export const VFX_LIVER = 1;                 // portal input + hepatic sink organ

export const VFX_KP_PARENT = Object.freeze([1.00, 4.20, 3.50, 1.20, 2.00, 2.80, 1.80, 1.20, 2.40, 1.50, 0.80, 2.00, 1.60, 0.90]);
export const VFX_PS_PARENT = Object.freeze([0.0, 900.0, 600.0, 120.0, 200.0, 8000.0, 250.0, 80.0, 500.0, 100.0, 40.0, 120.0, 80.0, 60.0]);
export const VFX_KP_ODV    = Object.freeze([1.00, 3.50, 3.00, 1.00, 1.60, 2.20, 1.40, 0.90, 2.00, 1.20, 0.70, 1.60, 1.30, 0.70]);
export const VFX_PS_ODV    = Object.freeze([0.0, 700.0, 450.0, 90.0, 150.0, 6000.0, 200.0, 60.0, 400.0, 80.0, 30.0, 100.0, 70.0, 50.0]);

// CYP2D6 scale s = literature ODV/V ratio / NM ratio (PM, IM, NM, UM).
const VFX_ODV_RATIO_LIT = Object.freeze([0.25, 1.16, 3.45, 10.3]);
export function vfxCyp2d6Scale(pheno) { return VFX_ODV_RATIO_LIT[pheno] / VFX_ODV_RATIO_LIT[2]; }

// EM hepatic resolution (mirrors vfx_hepatic_em). Venous-referenced, L/h.
export function vfxHepaticEm() {
  const q = Q[VFX_LIVER];
  const clR = VFX_CL_RENAL;
  const f = VFX_F_ORAL_XR;
  const clf = VFX_CL_ORAL_NM;
  const c = (q + clR) / q;
  const fAbs = f * (1.0 - clR / (c * q)) / (1.0 - f * clf / (c * q));
  const x = (fAbs * clf - clR) / c;
  const xForm = VFX_CL_FORM_ODV_NM / c;
  const xNon = (fAbs * VFX_CL_ORAL_QUINIDINE_EM - clR) / c;
  return { fAbs, xForm, xOther: x - xForm - xNon, xNon2d6: xNon, c, fmOdv: VFX_CL_FORM_ODV_NM / clf };
}
export function vfxXHep(pheno) {
  const h = vfxHepaticEm();
  return h.xNon2d6 + vfxCyp2d6Scale(pheno) * (h.xForm + h.xOther);
}
export function vfxFormFraction(pheno) {
  const h = vfxHepaticEm();
  return vfxCyp2d6Scale(pheno) * h.xForm / vfxXHep(pheno);
}
export function vfxClOdvSystemic() { return vfxHepaticEm().fmOdv * VFX_CL_ODV_APPARENT; }

// Venous-referenced clearance realised at the liver's steady state by sink
// parameter cl (acts on C_avg,liver), and its exact inverse (positive root of
// a·b·cl² + (a·PS/Kp + b·PS − X·b)·cl − X·PS/Kp = 0, a = V_v/V, b = V_t/V).
export function vfxLiverXOfSink(cl) {
  const k = VFX_LIVER;
  const v = V_REF[k];
  const vv = v * VASC_FRAC[k];
  const vt = v * (1.0 - VASC_FRAC[k]);
  const ps = VFX_PS_PARENT[k];
  return (cl / v) * (vv + vt * ps / (ps / VFX_KP_PARENT[k] + cl * vt / v));
}
export function vfxLiverClSink(x) {
  const k = VFX_LIVER;
  const a = VASC_FRAC[k];
  const b = 1.0 - a;
  const ps = VFX_PS_PARENT[k];
  const kp = VFX_KP_PARENT[k];
  const bb = a * ps / kp + b * ps - x * b;
  const cc = x * ps / kp;
  return 2.0 * cc / (bb + Math.sqrt(bb * bb + 4.0 * a * b * cc));
}

// Closed-form predictions (vfx_*_predicted in the Sounio scenario).
export function vfxOralClearancePredicted(pheno) {
  const h = vfxHepaticEm();
  return (h.c * vfxXHep(pheno) + VFX_CL_RENAL) / h.fAbs;
}
export function vfxAucRatioPredicted(pheno) {
  const h = vfxHepaticEm();
  return vfxCyp2d6Scale(pheno) * h.c * h.xForm / vfxClOdvSystemic();
}
export function vfxFOralPredicted(pheno) {
  const q = Q[VFX_LIVER];
  return vfxHepaticEm().fAbs * q / (q + vfxXHep(pheno));
}

// ─── Matrix transcendentals — verbatim ports of release/matrix_er.sio ────────
function merLnUnit(x) {                       // ln(x) for x in (0,2], artanh series
  if (x <= 0.0) return -1.0e6;
  const y = (x - 1.0) / (x + 1.0);
  const y2 = y * y;
  let term = y, sum = term;
  for (let k = 1; k < 20; k++) { term = term * y2; sum = sum + term / (2.0 * k + 1.0); }
  return 2.0 * sum;
}
function merExp(x) {                          // exp via (1+x/1024)^1024
  let r = 1.0 + x / 1024.0;
  for (let i = 0; i < 10; i++) r = r * r;
  return r;
}
function merPow(t, n) {                        // t^n via exp(n·ln t), t>0
  if (t <= 0.0) return 0.0;
  if (Math.abs(n - 1.0) < 1.0e-12) return t;
  if (Math.abs(n - 0.5) < 1.0e-12) return Math.sqrt(t);
  let lnT = 0.0;
  if (t > 0.0) lnT = (t > 2.0) ? (0.6931471805599453 + merLnUnit(t / 2.0)) : merLnUnit(t);
  return merExp(n * lnT);
}
function merExpNeg(x) {   // exp(x) for x<0: range-reduced 20-term Taylor (mer_exp_neg)
  if (x >= 0.0) return 1.0;
  if (x < 0.0 - 700.0) return 0.0;
  let r = x, m = 0;
  while (r < 0.0 - 0.5) { r = r * 0.5; m = m + 1; }
  let y = 1.0, term = 1.0;
  for (let k = 1; k < 20; k++) { term = term * r / k; y = y + term; }
  for (let i = 0; i < m; i++) y = y * y;
  return y;
}
function merPhi(x) {      // (1 − e^(−x))/x for x > 0, series below 0.0625 (mer_phi)
  if (x < 0.0625) {
    let sum = 1.0, term = 1.0;
    for (let k = 1; k < 12; k++) { term = term * (0.0 - x) / (k + 1); sum = sum + term; }
    return sum;
  }
  return (1.0 - merExpNeg(0.0 - x)) / x;
}

export function vfxMatrixFraction(rel, t) {
  if (t <= 0.0) return 0.0;
  const f = rel.k * merPow(t, rel.n);
  return f > 1.0 ? 1.0 : f;
}
export function vfxMatrixCumulative(rel, t) { return rel.totalDose * vfxMatrixFraction(rel, t); }
export function vfxMatrixStepAmount(rel, t, dt) {
  return vfxMatrixCumulative(rel, t + dt) - vfxMatrixCumulative(rel, t);
}

// ─── TR-BDF2 routed-sink PBPK28 step — port of theta_pbpk28.sio ──────────────
// State {cv, ct} (Float64Array(14)) is advanced in place. No negativity floor:
// the θ-step conserves mass exactly, so a negative state is booked in the
// ledger (minConc / negMass), never repaired.
export function vfxLedgerZero() {
  return { administered: 0.0, eliminated: 0.0, sinkRemoved: 0.0, auc: new Float64Array(N),
           minConc: 0.0, negMass: 0.0, steps: 0 };
}

// Kernel organ averages (V_v·C_v + V_t·C_t)/V_i; blood C_v[0].
function vfxKernelAvgs(st) {
  const out = new Float64Array(N);
  out[0] = st.cv[0];
  for (let i = 1; i < N; i++) {
    const vi = V_REF[i], vf = VASC_FRAC[i];
    out[i] = (vi * vf * st.cv[i] + vi * (1.0 - vf) * st.ct[i]) / vi;
  }
  return out;
}

function vfxAccum(lg, avgs, w, clCentral, sinkOrgan, clSink) {
  for (let i = 0; i < N; i++) lg.auc[i] = lg.auc[i] + w * avgs[i];
  lg.eliminated = lg.eliminated + clCentral * (w * avgs[0]);
  let sink = 0.0;
  if (sinkOrgan >= 1 && sinkOrgan < N) sink = clSink * (w * avgs[sinkOrgan]);
  lg.sinkRemoved = lg.sinkRemoved + sink;
  return sink;
}

function vfxNegativity(lg, st) {
  let neg = 0.0, cmin = lg.minConc;
  const cb = st.cv[0];
  if (cb < 0.0) { neg = neg + V_REF[0] * cb; if (cb < cmin) cmin = cb; }
  for (let i = 1; i < N; i++) {
    const vi = V_REF[i], vf = VASC_FRAC[i];
    const cv = st.cv[i], ct = st.ct[i];
    if (cv < 0.0) { neg = neg + vi * vf * cv; if (cv < cmin) cmin = cv; }
    if (ct < 0.0) { neg = neg + vi * (1.0 - vf) * ct; if (ct < cmin) cmin = ct; }
  }
  lg.minConc = cmin;
  const negAbs = 0.0 - neg;
  if (negAbs > lg.negMass) lg.negMass = negAbs;
}

// One θ-step (pbpk28_theta_step_routed_sink_mut). pr = {kp, ps, clCentral}.
function vfxThetaStep(st, pr, relMid, inputOrgan, dt, theta, sinkOrgan, clSink, lg) {
  const oldAvg = vfxKernelAvgs(st);
  const aV = new Float64Array(N), bV = new Float64Array(N);
  const aT = new Float64Array(N), bT = new Float64Array(N);
  const hi = theta * dt;
  const he = (1.0 - theta) * dt;
  const vb = V_REF[0];
  const hasSink = sinkOrgan >= 1 && sinkOrgan < N && clSink > 0.0;
  const routed = inputOrgan >= 1 && inputOrgan < N;
  let sumQ = 0.0;
  for (let i = 1; i < N; i++) sumQ = sumQ + Q[i];
  const k0 = (sumQ + pr.clCentral) / vb;
  const bigS = hi * k0;
  const cbOld = st.cv[0];
  let rhs0 = routed ? (1.0 - he * k0) * cbOld : (1.0 - he * k0) * cbOld + dt * relMid / vb;
  let sumBetaA = 0.0, sumBetaB = 0.0;
  for (let i = 1; i < N; i++) {
    const vi = V_REF[i], vf = VASC_FRAC[i];
    const vvRaw = vi * vf, vtRaw = vi * (1.0 - vf);
    const vv = vvRaw < 1.0e-30 ? 1.0e-30 : vvRaw;
    const vt = vtRaw < 1.0e-30 ? 1.0e-30 : vtRaw;
    const kp = pr.kp[i], ps = pr.ps[i], qi = Q[i];
    const ks = (hasSink && i === sinkOrgan) ? clSink / vi : 0.0;
    const kv = (qi + ps) / vv;
    const kvt = ps / (vv * kp);
    const ktv = ps / vt;
    const kt = ps / (vt * kp);
    const betaI = hi * qi / vb;
    const gammaI = hi * qi / vv;
    const pp = 1.0 + hi * (kv + ks);
    const qq = hi * kvt;
    const rr = hi * ktv;
    const sb = 1.0 + hi * (kt + ks);
    const det = pp * sb - qq * rr;
    const cvOld = st.cv[i], ctOld = st.ct[i];
    rhs0 = rhs0 + (he * qi / vb) * cvOld;
    const rhsV0 = (he * qi / vv) * cbOld + (1.0 - he * (kv + ks)) * cvOld + (he * kvt) * ctOld;
    const rhsV = (routed && i === inputOrgan) ? rhsV0 + dt * relMid / vv : rhsV0;
    const rhsT = (he * ktv) * cvOld + (1.0 - he * (kt + ks)) * ctOld;
    const av = (sb * rhsV + qq * rhsT) / det;
    const bv = (sb * gammaI) / det;
    aV[i] = av;
    bV[i] = bv;
    aT[i] = (rr * rhsV + pp * rhsT) / det;
    bT[i] = (rr * gammaI) / det;
    sumBetaA = sumBetaA + betaI * av;
    sumBetaB = sumBetaB + betaI * bv;
  }
  const cbNew = (rhs0 + sumBetaA) / ((1.0 + bigS) - sumBetaB);
  st.cv[0] = cbNew;
  for (let i = 1; i < N; i++) {
    st.cv[i] = aV[i] + bV[i] * cbNew;
    st.ct[i] = aT[i] + bT[i] * cbNew;
  }
  const newAvg = vfxKernelAvgs(st);
  const sinkOrg = hasSink ? sinkOrgan : 0;
  const sOld = vfxAccum(lg, oldAvg, he, pr.clCentral, sinkOrg, clSink);
  const sNew = vfxAccum(lg, newAvg, hi, pr.clCentral, sinkOrg, clSink);
  vfxNegativity(lg, st);
  lg.administered = lg.administered + dt * relMid;
  lg.steps = lg.steps + 1;
  return sOld + sNew;
}

const VFX_TRBDF2_GAMMA = 0.5857864376269049;  // 2 − √2
const VFX_TRBDF2_D     = 0.2928932188134524;  // 1 − 1/√2
const VFX_TRBDF2_W     = 0.3535533905932738;  // 1/(2√2)
const VFX_TRBDF2_A     = 1.2071067811865475;  // (1 + √2)/2

// One TR-BDF2 step (pbpk28_trbdf2_step_routed_sink_mut): trapezoid stage over
// γ·dt, then BDF2 as a backward-Euler θ-step of length d·dt on a·x_γ − (a−1)·x_n.
// The ledger is booked with the b-weights (w, w, d); stage bookings are scratch.
function vfxTrbdf2Step(st, pr, relMid, inputOrgan, dt, sinkOrgan, clSink, lg) {
  const oldAvg = vfxKernelAvgs(st);
  const xnV = Float64Array.from(st.cv), xnT = Float64Array.from(st.ct);
  const scratch = vfxLedgerZero();
  vfxThetaStep(st, pr, relMid, inputOrgan, VFX_TRBDF2_GAMMA * dt, 0.5, sinkOrgan, clSink, scratch);
  const midAvg = vfxKernelAvgs(st);
  const am1 = VFX_TRBDF2_A - 1.0;
  for (let i = 0; i < N; i++) {
    st.cv[i] = VFX_TRBDF2_A * st.cv[i] - am1 * xnV[i];
    st.ct[i] = VFX_TRBDF2_A * st.ct[i] - am1 * xnT[i];
  }
  vfxThetaStep(st, pr, relMid, inputOrgan, VFX_TRBDF2_D * dt, 1.0, sinkOrgan, clSink, scratch);
  const newAvg = vfxKernelAvgs(st);
  const hasSink = sinkOrgan >= 1 && sinkOrgan < N && clSink > 0.0;
  const sinkOrg = hasSink ? sinkOrgan : 0;
  const k0 = vfxAccum(lg, oldAvg, VFX_TRBDF2_W * dt, pr.clCentral, sinkOrg, clSink);
  const k1 = vfxAccum(lg, midAvg, VFX_TRBDF2_W * dt, pr.clCentral, sinkOrg, clSink);
  const k2 = vfxAccum(lg, newAvg, VFX_TRBDF2_D * dt, pr.clCentral, sinkOrg, clSink);
  vfxNegativity(lg, st);
  lg.administered = lg.administered + dt * relMid;
  lg.steps = lg.steps + scratch.steps;
  return k0 + k1 + k2;
}

function vfxOrganAverage(cv, ct, i) {
  if (i === 0) return cv[0];
  const vf = VASC_FRAC[i];
  return vf * cv[i] + (1 - vf) * ct[i];
}
function vfxTotalMass(cv, ct) {
  let total = V_REF[0] * cv[0];
  for (let i = 1; i < N; i++) {
    const vi = V_REF[i], vf = VASC_FRAC[i];
    total += vi * vf * cv[i] + vi * (1 - vf) * ct[i];
  }
  return total;
}

// Ledger residual (mg): body + eliminated + sink − administered, from zero.
function vfxLedgerResidual(lg, st) {
  return vfxTotalMass(st.cv, st.ct) + lg.eliminated + lg.sinkRemoved - lg.administered;
}

// Gut pool step (vfx_gut_absorb_step): dG/dt = R − ka·G solved exactly over the
// step with the release ΔR entering at a constant rate, x = ka·dt:
// G₁ = G₀·e^(−x) + ΔR·(1 − e^(−x))/x, leaving = G₀ + ΔR − G₁. F_abs of what
// leaves reaches the portal vein.
// vfxGutAbsorbStepWindow confines the release to the first relDt of the step
// (vfx_gut_absorb_step_window): pool carried through the window, then decays
// for dt − relDt.
export function vfxGutAbsorbStepWindow(gutMg, relMg, relDt, dt) {
  const ka = VFX_KA_ABS;
  const x = ka * relDt;
  const decay = merExpNeg(0.0 - x);
  const relKept = x > 0.0 ? relMg * merPhi(x) : relMg;
  const gutEndRelease = gutMg * decay + relKept;
  const gutNew = gutEndRelease * merExpNeg(0.0 - ka * (dt - relDt));
  const leaveMg = (gutMg + relMg) - gutNew;
  const toParent = vfxHepaticEm().fAbs * leaveMg;
  return { gutMg: gutNew, toParentMg: toParent, lostMg: leaveMg - toParent };
}
export function vfxGutAbsorbStep(gutMg, relMg, dt) { return vfxGutAbsorbStepWindow(gutMg, relMg, dt, dt); }

// Release window in [t, t + dt] (vfx_release_window): dt, except in the step
// where the matrix saturates, where it ends at the saturation time found by
// bisecting vfxMatrixFraction on the step (64 halvings).
export function vfxReleaseWindow(rel, tStart, dt) {
  if (vfxMatrixFraction(rel, tStart + dt) < 1.0) return dt;
  if (vfxMatrixFraction(rel, tStart) >= 1.0) return dt;
  let lo = tStart, hi = tStart + dt;
  for (let i = 0; i < 64; i++) {
    const mid = 0.5 * (lo + hi);
    if (vfxMatrixFraction(rel, mid) >= 1.0) hi = mid; else lo = mid;
  }
  return hi - tStart;
}

// Steps 1-2 (vfx_gut_release_step).
export function vfxGutReleaseStep(gutMg, rel, tStart, dt) {
  const relAmt = vfxMatrixStepAmount(rel, tStart, dt);
  return vfxGutAbsorbStepWindow(gutMg, relAmt, vfxReleaseWindow(rel, tStart, dt), dt);
}

// One Lie-Trotter step (vfx_strang_step). Mutates `sc` in place.
function vfxStrangStep(sc, rel, tStart, dt) {
  const relAmt = vfxMatrixStepAmount(rel, tStart, dt);
  sc.released = sc.released + relAmt;
  const gs = vfxGutReleaseStep(sc.gut, rel, tStart, dt);
  sc.gut = gs.gutMg;
  sc.portal = sc.portal + gs.toParentMg;
  sc.lost = sc.lost + gs.lostMg;
  const parentInput = gs.toParentMg / dt;
  const clSink = vfxLiverClSink(vfxXHep(sc.pheno));
  const removed = vfxTrbdf2Step(sc.parent, sc.prP, parentInput, VFX_LIVER, dt, VFX_LIVER, clSink, sc.lgP);
  const formMg = removed * vfxFormFraction(sc.pheno);
  vfxTrbdf2Step(sc.odv, sc.prO, formMg / dt, 0, dt, 0, 0.0, sc.lgO);
}

function vfxScenarioInit(pheno) {
  return {
    pheno,
    parent: { cv: new Float64Array(N), ct: new Float64Array(N) },
    odv:    { cv: new Float64Array(N), ct: new Float64Array(N) },
    prP: { kp: VFX_KP_PARENT, ps: VFX_PS_PARENT, clCentral: VFX_CL_RENAL },
    prO: { kp: VFX_KP_ODV, ps: VFX_PS_ODV, clCentral: vfxClOdvSystemic() },
    lgP: vfxLedgerZero(), lgO: vfxLedgerZero(),
    gut: 0.0, released: 0.0, portal: 0.0, lost: 0.0,
  };
}

// Mass account and closed-form check of a scenario state (vfx_self_check_run).
function vfxAccount(sc) {
  const mp = vfxTotalMass(sc.parent.cv, sc.parent.ct);
  const mo = vfxTotalMass(sc.odv.cv, sc.odv.ct);
  const aucP = sc.lgP.auc[0], aucO = sc.lgO.auc[0];
  const h = vfxHepaticEm();
  const clf = aucP > 0.0 ? sc.lgP.administered / h.fAbs / aucP : 0.0;
  const aucRatio = aucP > 0.0 ? aucO / aucP : 0.0;
  // Truncation bounds: after T the parent still receives F_abs·G from the gut
  // and loses at least its renal share; all of it may still become ODV.
  const mpLeft = mp + h.fAbs * sc.gut;
  const tailP = aucP > 0.0 ? mpLeft / (VFX_CL_RENAL * aucP) : 0.0;
  const tailO = aucO > 0.0 ? (mo + mpLeft) / (sc.prO.clCentral * aucO) : 0.0;
  return {
    pMass: mp, oMass: mo, aucP, aucO, aucRatio, clf,
    clfPred: vfxOralClearancePredicted(sc.pheno), aucRatioPred: vfxAucRatioPredicted(sc.pheno),
    fOralPred: vfxFOralPredicted(sc.pheno), tailP, tailO,
    residGut: sc.released - (sc.gut + sc.portal + sc.lost),
    residSplit: sc.portal - h.fAbs * (sc.portal + sc.lost),
    boundSlack: h.fAbs * sc.released - sc.portal,
    residP: vfxLedgerResidual(sc.lgP, sc.parent), residO: vfxLedgerResidual(sc.lgO, sc.odv),
    negMass: Math.max(sc.lgP.negMass, sc.lgO.negMass),
    steps: sc.lgP.steps, fAbs: h.fAbs,
    gut: sc.gut, released: sc.released, portal: sc.portal, lost: sc.lost,
    administered: sc.lgP.administered, formed: sc.lgO.administered,
  };
}

/**
 * Integrate the venlafaxine XR portal scenario at a CYP2D6 phenotype (default
 * 2 = NM, the parity reference) and sample parent + ODV organ-average
 * trajectories. Each record carries {cv,ct,avg}[14] per drug, cumulative matrix
 * release, the total-body ODV/parent mass ratio, and the mass/closed-form
 * account (`vfxAccount`). Default dt = 0.5 h matches the stdlib scenario.
 */
export function runVenlafaxineScenario(sampleTimes, { dt = 0.5, pheno = 2 } = {}) {
  const rel = VFX_MATRIX_GOHEL2008;
  const sc = vfxScenarioInit(pheno);
  const out = [];
  let t = 0.0;
  for (const target of sampleTimes) {
    while (t + 0.5 * dt < target) {
      vfxStrangStep(sc, rel, t, dt);
      t += dt;
    }
    const pAvg = new Float64Array(N), oAvg = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      pAvg[i] = vfxOrganAverage(sc.parent.cv, sc.parent.ct, i);
      oAvg[i] = vfxOrganAverage(sc.odv.cv, sc.odv.ct, i);
    }
    const acc = vfxAccount(sc);
    out.push({
      t: target,
      pCv: Float64Array.from(sc.parent.cv), pCt: Float64Array.from(sc.parent.ct), pAvg,
      oCv: Float64Array.from(sc.odv.cv), oCt: Float64Array.from(sc.odv.ct), oAvg,
      released: vfxMatrixCumulative(rel, target),
      ratio: acc.pMass < 1.0e-12 ? 0.0 : acc.oMass / acc.pMass,
      ...acc,
    });
  }
  return out;
}

// ════════════════════════════════════════════════════════════════════════════
// HALOPERIDOL (CASO II) — PBPK14 well-stirred + detailed BBB + D2 occupancy.
//
// Fourth canonical drug. Its validated science is PBPK14 + BBB(ISF/ICF) + D2
// (no citable PBPK28 perm-limited model — empirical-first), so the canonical
// parity surface is plasma PK · BBB Kpuu · D2 occupancy. The production scenario
// integrates the systemic PBPK14 with adaptive Tsit5; to make Sounio↔Node parity
// tractable, BOTH this engine and the parity ref use a FIXED-STEP RK4 systemic
// integrator at the same dt — parity validates exactly what the viewer runs.
// Mirrors tests/run-pass/dissertation_pbpk28_parity_ref_haloperidol.sio to f64.
// Constants verbatim from drugs/haloperidol.sio + bbb/bbb_core.sio + pd/d2_occupancy.sio.
// ════════════════════════════════════════════════════════════════════════════

// index 0=blood,1=liver,2=kidney,3=brain,4=heart,5=lung,6=muscle,7=adipose,
// 8=gut,9=skin,10=bone,11=spleen,12=pancreas,13=other.
export const HALO_V  = Object.freeze([5.2, 1.69, 0.31, 1.37, 0.31, 1.17, 29.0, 18.2, 1.65, 7.8, 10.5, 0.19, 0.14, 4.5]);
export const HALO_Q  = Object.freeze([0.0, 87.0, 72.0, 44.0, 14.4, 0.0, 73.8, 18.0, 57.6, 18.0, 21.6, 10.8, 4.5, 18.0]);
export const HALO_KP = Object.freeze([1.0, 18.0, 8.0, 15.0, 6.0, 12.0, 35.0, 12.0, 6.0, 8.0, 6.0, 8.0, 5.0, 5.0]);
export const HALO_CL_HEPATIC = 40.0, HALO_CL_RENAL = 0.5, HALO_FU_PLASMA = 0.08, HALO_RB = 1.0;
export const HALO_BBB = Object.freeze({ vIsf: 0.270, vIcf: 1.080, psBbb: 2.0, psMem: 8.0, fuIsf: 0.035, fuIcf: 0.010, kpuuBrain: 3.0, kpuuCell: 0.5 });
export const HALO_D2_KD = 0.000564;
export const HALO_ABS = Object.freeze({ ka: 0.9, f: 0.65, tlag: 0.25 });

// exp(-x) — verbatim port of absorption.sio abs_exp_neg.
function haloAbsExpNeg(x) {
  if (x <= 0.0) return 1.0;
  if (x > 0.5) { const half = haloAbsExpNeg(x * 0.5); return half * half; }
  let term = 1.0, sum = 1.0;
  for (let k = 1; k < 16; k++) { term = term * (-x) / k; sum = sum + term; }
  return sum;
}

// PBPK14 well-stirred RHS (pbpk_ode), array form.
function haloSysOde(c) {
  const cPlasma = c[0] / HALO_RB;
  const cUnbound = cPlasma * HALO_FU_PLASMA;
  const d = new Float64Array(14);
  let dBlood = 0.0;
  for (let i = 1; i < 14; i++) {
    const flux = (HALO_Q[i] / HALO_V[i]) * (cPlasma - c[i] / HALO_KP[i]);
    d[i] = flux;
    dBlood = dBlood - (HALO_Q[i] / HALO_V[0]) * HALO_RB * (cPlasma - c[i] / HALO_KP[i]);
  }
  dBlood = dBlood - (HALO_CL_HEPATIC / HALO_V[0]) * cUnbound * HALO_RB;
  dBlood = dBlood - (HALO_CL_RENAL / HALO_V[0]) * cUnbound * HALO_RB;
  d[0] = dBlood;
  return d;
}
function haloSysRk4(c, dt) {
  const axpy = (y, k, a) => { const r = new Float64Array(14); for (let i = 0; i < 14; i++) r[i] = y[i] + a * k[i]; return r; };
  const k1 = haloSysOde(c);
  const k2 = haloSysOde(axpy(c, k1, 0.5 * dt));
  const k3 = haloSysOde(axpy(c, k2, 0.5 * dt));
  const k4 = haloSysOde(axpy(c, k3, dt));
  const r = new Float64Array(14);
  const sixth = dt / 6.0, third = dt / 3.0;
  for (let i = 0; i < 14; i++) r[i] = c[i] + sixth * k1[i] + third * k2[i] + third * k3[i] + sixth * k4[i];
  return r;
}

// BBB RHS + RK4 (bbb_ode / bbb_rk4_step), c_plasma constant across the step.
function haloBbbOde(isf, icf, cPlasma) {
  const cpu = HALO_FU_PLASMA * cPlasma;
  const ciu = HALO_BBB.fuIsf * isf;
  const ccu = HALO_BBB.fuIcf * icf;
  const bbbFlux = HALO_BBB.psBbb * (cpu - ciu / HALO_BBB.kpuuBrain);
  const memFlux = HALO_BBB.psMem * (ciu - ccu / HALO_BBB.kpuuCell);
  return { isf: (bbbFlux - memFlux) / HALO_BBB.vIsf, icf: memFlux / HALO_BBB.vIcf };
}
function haloBbbRk4(isf, icf, cPlasma, dt) {
  const k1 = haloBbbOde(isf, icf, cPlasma);
  const k2 = haloBbbOde(isf + 0.5 * dt * k1.isf, icf + 0.5 * dt * k1.icf, cPlasma);
  const k3 = haloBbbOde(isf + 0.5 * dt * k2.isf, icf + 0.5 * dt * k2.icf, cPlasma);
  const k4 = haloBbbOde(isf + dt * k3.isf, icf + dt * k3.icf, cPlasma);
  const sixth = dt / 6.0, third = dt / 3.0;
  return {
    isf: isf + sixth * k1.isf + third * k2.isf + third * k3.isf + sixth * k4.isf,
    icf: icf + sixth * k1.icf + third * k2.icf + third * k3.icf + sixth * k4.icf,
  };
}
function haloD2Occ(cIsfFree) { return cIsfFree <= 0.0 ? 0.0 : cIsfFree / (cIsfFree + HALO_D2_KD); }

function haloStep(st, t, dt) {
  let aGut = st.aGut;
  let c = st.sys;
  if (t + dt > HALO_ABS.tlag) {
    const tActiveStart = t < HALO_ABS.tlag ? HALO_ABS.tlag : t;
    const dtActive = (t + dt) - tActiveStart;
    if (dtActive > 0.0) {
      const decay = haloAbsExpNeg(HALO_ABS.ka * dtActive);
      const aAfter = aGut * decay;
      const toBlood = HALO_ABS.f * (aGut - aAfter);
      aGut = aAfter;
      c = Float64Array.from(c);
      c[0] = c[0] + toBlood / HALO_V[0];
    }
  }
  const b0 = c[0];
  const s1 = haloSysRk4(c, dt);
  const cMid = 0.5 * (b0 + s1[0]);
  const b1 = haloBbbRk4(st.bbb.isf, st.bbb.icf, cMid, dt);
  return { sys: s1, bbb: b1, aGut };
}

/**
 * Integrate the haloperidol oral scenario (single dose, fixed-step RK4) and
 * sample plasma / ISF_free / ICF_free / Kpuu / D2-occupancy at the given times.
 * dt default 0.01 h, dose 5 mg — matches the parity ref.
 */
export function runHaloperidolScenario(sampleTimes, { dt = 0.002, doseMg = 5.0 } = {}) {
  let st = { sys: new Float64Array(14), bbb: { isf: 0.0, icf: 0.0 }, aGut: doseMg };
  const out = [];
  let t = 0.0;
  for (const target of sampleTimes) {
    while (t + 0.5 * dt < target) { st = haloStep(st, t, dt); t += dt; }
    const plasma = st.sys[0] / HALO_RB;
    const isfFree = HALO_BBB.fuIsf * st.bbb.isf;
    const icfFree = HALO_BBB.fuIcf * st.bbb.icf;
    const plasmaFree = HALO_FU_PLASMA * plasma;
    const kpuu = plasmaFree > 1.0e-12 ? isfFree / plasmaFree : 0.0;
    out.push({ t: target, plasma, brain: st.sys[3], isfFree, icfFree, kpuu, d2occ: haloD2Occ(isfFree) });
  }
  return out;
}

// ============================================================================
// MIDAZOLAM — fifth canonical drug: oral CYP3A drug–drug interaction (DDI).
//
// Novel surface (none of the four prior drugs model enzyme inhibition):
// mechanistic well-stirred oral first-pass (F = Fa·FG·FH) + competitive CYP3A
// inhibition. A SINGLE hepatic intrinsic clearance CLint_h drives BOTH the
// first-pass survival FH = Qh/(Qh+fu·CLint_h) AND the systemic well-stirred
// clearance CL_h = Qh·fu·CLint_h/(Qh+fu·CLint_h). A separate gut CLint_g drives
// FG = Qg/(Qg+fu_g·CLint_g). Competitive inhibition divides the INTRINSIC
// clearances by (1 + I/Ki). Because FH and CL_h come from one CLint_h,
// inhibition raises FH and lowers CL_h *consistently* — no double-count — so the
// iconic oral midazolam+ketoconazole AUC rise (~15×) emerges honestly
// (AUC = F·Dose/CL_h; validated in stdlib/darwin_pbpk/validation/midazolam_ddi.sio).
//
// Inhibitor is held STATIC at its steady-state unbound concentration (matches how
// clinical DDI studies pre-dose the perpetrator). Systemic distribution Kp are
// nominal midazolam-plausible values (Vss ~1.4 L/kg); the *validated* DDI
// quantities are F, CL_h and AUCR, which are distribution-independent.
// Units: mg, L, L/h, h, mg/L throughout.
//
// Sources: Heizmann 1984 (oral F~0.4), Thummel 1996 (gut+hepatic first-pass),
// Gorski 1998 / Olkkola 1994,1996 (ketoconazole DDI ~15×), Yang 2007 (Qgut model).
// ============================================================================

// Generic 70 kg human physiology (volumes L, flows L/h) — reused; Kp are midazolam.
export const MDZ_V  = Object.freeze([5.2, 1.69, 0.31, 1.37, 0.31, 1.17, 29.0, 18.2, 1.65, 7.8, 10.5, 0.19, 0.14, 4.5]);
export const MDZ_Q  = Object.freeze([0.0, 90.0, 72.0, 44.0, 14.4, 0.0, 73.8, 18.0, 57.6, 18.0, 21.6, 10.8, 4.5, 18.0]);
export const MDZ_KP = Object.freeze([1.0, 2.0, 1.5, 1.5, 1.2, 1.5, 1.0, 2.0, 1.5, 1.2, 0.6, 1.2, 1.2, 1.0]);
export const MDZ_FU_PLASMA = 0.03, MDZ_RB = 1.0;
export const MDZ_CL_RENAL = 0.0;  // midazolam: <1% renal — elimination is CYP3A-hepatic.
// First-pass + CYP3A metabolism: one CLint_h -> {FH, CL_h}; CLint_g -> FG.
export const MDZ_FP = Object.freeze({ qh: 90.0, qg: 18.0, fu: 0.03, fuGut: 1.0, fa: 0.95, clintH: 1090.0, clintG: 14.7 });
// Absorption: rapid oral (Tmax ~0.9 h), negligible lag.
export const MDZ_ABS = Object.freeze({ ka: 3.0, tlag: 0.0 });
// Ketoconazole perpetrator, static steady-state UNBOUND conc (mg/L): I/Ki = 8 -> R = 9.
export const MDZ_KETO = Object.freeze({ ki: 0.008, iSteadyState: 0.064 });

// exp(-x) — verbatim port of absorption.sio abs_exp_neg (shared low-accuracy arithmetic).
function mdzAbsExpNeg(x) {
  if (x <= 0.0) return 1.0;
  if (x > 0.5) { const half = mdzAbsExpNeg(x * 0.5); return half * half; }
  let term = 1.0, sum = 1.0;
  for (let k = 1; k < 16; k++) { term = term * (-x) / k; sum = sum + term; }
  return sum;
}

// Competitive inhibition on CYP3A intrinsic clearance: factor in [0,1].
function mdzInhFactor(I) { return 1.0 / (1.0 + I / MDZ_KETO.ki); }
// Gut-wall survival FG, hepatic first-pass survival FH, systemic CL_h — each from
// a single (inhibited) intrinsic clearance. Same enzyme/inhibitor -> same factor.
function mdzFG(I)  { const b = MDZ_FP.fuGut * MDZ_FP.clintG * mdzInhFactor(I); return MDZ_FP.qg / (MDZ_FP.qg + b); }
function mdzFH(I)  { const a = MDZ_FP.fu   * MDZ_FP.clintH * mdzInhFactor(I); return MDZ_FP.qh / (MDZ_FP.qh + a); }
function mdzCLh(I) { const a = MDZ_FP.fu   * MDZ_FP.clintH * mdzInhFactor(I); return MDZ_FP.qh * a / (MDZ_FP.qh + a); }
function mdzF(I)   { return MDZ_FP.fa * mdzFG(I) * mdzFH(I); }

// PBPK14 well-stirred RHS; clH is the (inhibitor-dependent) systemic hepatic
// clearance acting on plasma — it already embeds fu via the well-stirred form,
// so it is NOT multiplied by fu again here.
function mdzSysOde(c, clH) {
  const cPlasma = c[0] / MDZ_RB;
  const d = new Float64Array(14);
  let dBlood = 0.0;
  for (let i = 1; i < 14; i++) {
    const flux = (MDZ_Q[i] / MDZ_V[i]) * (cPlasma - c[i] / MDZ_KP[i]);
    d[i] = flux;
    dBlood = dBlood - (MDZ_Q[i] / MDZ_V[0]) * MDZ_RB * (cPlasma - c[i] / MDZ_KP[i]);
  }
  dBlood = dBlood - (clH / MDZ_V[0]) * cPlasma * MDZ_RB;
  dBlood = dBlood - (MDZ_CL_RENAL / MDZ_V[0]) * cPlasma * MDZ_RB;
  d[0] = dBlood;
  return d;
}
function mdzSysRk4(c, dt, clH) {
  const axpy = (y, k, a) => { const r = new Float64Array(14); for (let i = 0; i < 14; i++) r[i] = y[i] + a * k[i]; return r; };
  const k1 = mdzSysOde(c, clH);
  const k2 = mdzSysOde(axpy(c, k1, 0.5 * dt), clH);
  const k3 = mdzSysOde(axpy(c, k2, 0.5 * dt), clH);
  const k4 = mdzSysOde(axpy(c, k3, dt), clH);
  const r = new Float64Array(14);
  const sixth = dt / 6.0, third = dt / 3.0;
  for (let i = 0; i < 14; i++) r[i] = c[i] + sixth * k1[i] + third * k2[i] + third * k3[i] + sixth * k4[i];
  return r;
}

// One fixed step: operator-split oral absorption (delivers Fa·FG·FH of the released
// mass to blood) -> systemic RK4. clH, Foral are constant (static inhibitor).
function mdzStep(st, t, dt, clH, Foral) {
  let aGut = st.aGut;
  let c = st.sys;
  if (t + dt > MDZ_ABS.tlag) {
    const tActiveStart = t < MDZ_ABS.tlag ? MDZ_ABS.tlag : t;
    const dtActive = (t + dt) - tActiveStart;
    if (dtActive > 0.0) {
      const decay = mdzAbsExpNeg(MDZ_ABS.ka * dtActive);
      const aAfter = aGut * decay;
      const toBlood = Foral * (aGut - aAfter);
      aGut = aAfter;
      c = Float64Array.from(c);
      c[0] = c[0] + toBlood / MDZ_V[0];
    }
  }
  return { sys: mdzSysRk4(c, dt, clH), aGut };
}

/**
 * Integrate the oral midazolam scenario (single dose, fixed-step RK4) under a
 * static unbound inhibitor concentration `I` (mg/L; 0 = solo). Samples plasma
 * (mg/L) at the given times. dt default 0.002 h, dose 5 mg — matches the ref.
 */
export function runMidazolamScenario(sampleTimes, { dt = 0.002, doseMg = 5.0, I = 0.0 } = {}) {
  const clH = mdzCLh(I);
  const Foral = mdzF(I);
  let st = { sys: new Float64Array(14), aGut: doseMg };
  const out = [];
  let t = 0.0;
  for (const target of sampleTimes) {
    while (t + 0.5 * dt < target) { st = mdzStep(st, t, dt, clH, Foral); t += dt; }
    out.push({ t: target, plasma: st.sys[0] / MDZ_RB });
  }
  return out;
}

/**
 * Analytic DDI dose-response across a grid of I/Ki ratios: per point returns oral
 * F, systemic CL_h, and AUC ratio vs solo (AUCR = (F/F0)·(CL_h0/CL_h)). This is
 * the inhibition-curve parity series (case 19) — closed form, no integration.
 */
export function runMidazolamDDIResponse(iOverKiGrid) {
  const F0 = mdzF(0.0), clH0 = mdzCLh(0.0);
  return iOverKiGrid.map((r) => {
    const I = r * MDZ_KETO.ki;
    const F = mdzF(I), clH = mdzCLh(I);
    return { iOverKi: r, F, clH, aucr: (F / F0) * (clH0 / clH) };
  });
}
