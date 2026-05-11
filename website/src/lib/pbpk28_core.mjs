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
    const ka = params.releaseKa;
    const F  = params.releaseF;
    const D  = params.releaseDoseMg;
    return F * D * ka * Math.exp(-ka * Math.max(0, tHours));
  }
  // Higuchi (default): dQ/dt = K_H / (2·√t), clipped at t < 0.1 h
  const t = Math.max(0.1, tHours);
  const kH = params.releaseKH || HIGUCHI_KH_DEFAULT;
  return params.higuchiScale * kH / (2 * Math.sqrt(t));
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
