// Pure-JS, Node-and-browser-portable core of the 14-compartment well-stirred
// PBPK using **arrow-matrix Crank-Nicolson** (G-α-δ; commit 33015d8f superseded
// the prior explicit-RK4-dt=0.01 form that violated mass conservation by 172×
// over 30 h — brain eigenvalue Q/(V·Kp)=303 h⁻¹ exceeded RK4's stability
// boundary 2.785).
//
// Both `website/src/hooks/usePBPK14.ts` (the React adapter) and
// `scripts/dissertation/run_pbpk14_node.mjs` (the parity-gate runner) consume
// this module. Keeping it ESM .mjs with no TS-only syntax lets Node import it
// directly without a build step.
//
// ODE: dx/dt = M·x + f, x = [C_b, C_1, ..., C_13]
//   M[0,0] = -(Σ Q_i + cl_hep)/V_b
//   M[0,i] =  Q_i / (V_b · Kp_i)        (venous return at C_i/Kp_i)
//   M[i,0] =  Q_i / V_i                 (organ supply at C_b)
//   M[i,i] = -Q_i / (V_i · Kp_i)
//   f[0] = release/V_b
//
// CN: (I - dt·M/2)·x_new = (I + dt·M/2)·x_old + dt·f_mid
// Arrow-matrix Schur-complement solve in O(N):
//   h     = dt/2
//   S     = h·(Σ Q + cl_hep)/V_b                  (Cb self in A)
//   α_i   = h·Q_i / V_i                           (Cb ← Ci  ; A[i,0] = -α_i)
//   α'_i  = h·Q_i / (V_i · Kp_i)                  (Ci self in A: 1+α'_i)
//   β_i   = h·Q_i / (V_b · Kp_i)                  (Ci ← Cb in row 0)
//   γ_i   = β_i / (1 + α'_i)
//   b[0]  = (1-S)·C_b_old + Σ β_i·C_i_old + dt·release/V_b
//   b[i]  = α_i·C_b_old + (1 - α'_i)·C_i_old
//   x_new[0] = (b[0] + Σ γ_i·b[i]) / ((1+S) - Σ γ_i·α_i)
//   x_new[i] = (b[i] + α_i·x_new[0]) / (1 + α'_i)
//
// A-stable, 2nd-order, mass-conserving (dM/dt = -cl_hep·C_b exactly).
//
// Compartment indexing matches `website/src/components/dissertation/compartments.ts`
// and is *different* from `tests/run-pass/dissertation_pbpk14_gum.sio` — the
// parity reference fixture `tests/run-pass/dissertation_frontend_parity_ref.sio`
// is hard-bound to THIS indexing.

// 0=blood 1=liver 2=kidney 3=brain 4=heart 5=lung 6=muscle 7=adipose
// 8=gut   9=skin  10=bone   11=spleen 12=pancreas 13=other
export const KP = Object.freeze([1.00, 5.40, 4.20, 0.10, 2.30, 3.10, 0.50, 0.30, 2.60, 0.80, 0.20, 2.10, 1.80, 0.40]);
export const Q  = Object.freeze([350,   90,   74,   44,   16,  350,   42,   10,   55,   21,   17,   10,    8,   13]);
export const V_REF = Object.freeze([5.0, 1.8, 0.31, 1.45, 0.33, 1.1, 28.0, 20.0, 1.2, 3.6, 6.6, 0.18, 0.09, 3.7]);

export const CL_HEP_DEFAULT = 12.4;     // L/h, rapamycin typical (Ferron 1997)
export const HIGUCHI_KH_DEFAULT = 0.00417; // mg / √h, Cypher Cordis 2003

export const N = 14;

/**
 * @typedef {{ clHep: number, higuchiScale: number, vdScale: number, bolusMg: number, stentActive: boolean }} PBPKParams
 */

export const DEFAULT_PARAMS = Object.freeze({
  clHep: CL_HEP_DEFAULT,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: 0.0,
  stentActive: true,
});

/**
 * Stent release rate at time t. Higuchi diffusion clipped at t < 0.1 h to
 * avoid 1/√0 divergence.
 */
export function releaseRateAt(tHours, params) {
  if (!params.stentActive) return 0;
  const t = Math.max(0.1, tHours);
  return params.higuchiScale * HIGUCHI_KH_DEFAULT / (2 * Math.sqrt(t));
}

/**
 * One arrow-matrix CN step on the 14-state well-stirred PBPK14. Mutates `C`
 * in place. Returns mg released this step. A-stable, 2nd-order, mass-
 * conserving (replaces the prior explicit-RK4 step which violated stability
 * because brain λ = Q/(V·Kp) = 303 h⁻¹ exceeded RK4's boundary at dt = 0.01).
 */
export function stepCN(C, tHours, dtHours, params, scratch) {
  const h = 0.5 * dtHours;
  const Vb = V_REF[0] * params.vdScale;
  const release = releaseRateAt(tHours + h, params);

  // Populate `k1` with the natural ODE rhs at start of step. This is used by
  // `usePBPK14.ts` as a |dC/dt| surrogate for the GUM-cone uncertainty
  // visual (uncertNorm in the React adapter). Not used by the CN solve.
  if (scratch.k1) {
    const k1 = scratch.k1;
    const cBlood = C[0];
    let bloodFluxOut = 0;
    for (let i = 1; i < N; i++) {
      const Vi = V_REF[i] * params.vdScale;
      const flux = Q[i] * (cBlood - C[i] / KP[i]);
      k1[i] = flux / Vi;
      bloodFluxOut += flux;
    }
    k1[0] = (-bloodFluxOut + release - params.clHep * cBlood) / Vb;
  }

  let sumQ = 0;
  for (let i = 1; i < N; i++) sumQ += Q[i];
  const S = h * (sumQ + params.clHep) / Vb;

  const cbOld = C[0];
  let rhs0 = (1 - S) * cbOld + dtHours * release / Vb;
  for (let i = 1; i < N; i++) {
    const beta = h * Q[i] / (Vb * KP[i]);
    rhs0 += beta * C[i];
  }

  const alpha = scratch.alpha;
  const alphaP = scratch.alphaP;
  const bi = scratch.bi;
  let denomAcc = 0;
  let numAcc = 0;
  for (let i = 1; i < N; i++) {
    const Vi = V_REF[i] * params.vdScale;
    const a = h * Q[i] / Vi;
    const ap = a / KP[i];
    alpha[i] = a;
    alphaP[i] = ap;
    bi[i] = a * cbOld + (1 - ap) * C[i];
    const beta = h * Q[i] / (Vb * KP[i]);
    const gamma = beta / (1 + ap);
    denomAcc += gamma * a;
    numAcc += gamma * bi[i];
  }
  const cbNew = (rhs0 + numAcc) / ((1 + S) - denomAcc);
  C[0] = cbNew < 0 ? 0 : cbNew;
  for (let i = 1; i < N; i++) {
    const ci = (bi[i] + alpha[i] * cbNew) / (1 + alphaP[i]);
    C[i] = ci < 0 ? 0 : ci;
  }
  return release * dtHours;
}

// Backward-compat alias for any caller still using the old explicit-RK4 name.
// The new step uses fully-implicit arrow-CN and is mass-conserving.
export const stepRK4 = stepCN;

export function makeScratch() {
  return {
    alpha:  new Float64Array(N),
    alphaP: new Float64Array(N),
    bi:     new Float64Array(N),
    // k1 retained for the React adapter's |dC/dt| GUM-cone surrogate (the
    // CN step populates it with the natural ODE rhs at start of step).
    k1:     new Float64Array(N),
  };
}

/**
 * Initial state from a bolus.
 *
 * @param {PBPKParams} params
 * @returns {Float64Array}
 */
export function initialState(params) {
  const C = new Float64Array(N);
  C[0] = params.bolusMg / (V_REF[0] * params.vdScale);
  return C;
}
