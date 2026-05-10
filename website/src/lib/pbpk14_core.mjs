// Pure-JS, Node-and-browser-portable core of the 14-compartment PBPK RK4.
//
// Both `website/src/hooks/usePBPK14.ts` (the React adapter) and
// `scripts/dissertation/run_pbpk14_node.mjs` (the parity-gate runner) consume
// this module. Keeping it ESM .mjs with no TS-only syntax lets Node import it
// directly without a build step.
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
 * Right-hand side of dC/dt. Writes into `out` in place.
 *
 * @param {Float64Array} C        Length-14 concentrations, mg/L
 * @param {Float64Array} out      Length-14 derivative target, mg/(L·h)
 * @param {PBPKParams}   params
 * @param {number}       releaseRate  Drug release into blood, mg/h
 */
export function rhs(C, out, params, releaseRate) {
  const cBlood = C[0];
  let bloodFluxOut = 0; // mg/h leaving blood toward organs
  for (let i = 1; i < N; i++) {
    const v = V_REF[i] * params.vdScale;
    const flux = Q[i] * (cBlood - C[i] / KP[i]); // mg/h delivered to organ i
    out[i] = flux / v;
    bloodFluxOut += flux;
  }
  const vBlood = V_REF[0] * params.vdScale;
  out[0] = (-bloodFluxOut + releaseRate - params.clHep * cBlood) / vBlood;
}

/**
 * Stent release rate at time t. Higuchi diffusion clipped at t < 0.1 h to
 * avoid 1/√0 divergence.
 *
 * @param {number} tHours
 * @param {PBPKParams} params
 * @returns {number} mg/h released into blood
 */
export function releaseRateAt(tHours, params) {
  if (!params.stentActive) return 0;
  const t = Math.max(0.1, tHours);
  return params.higuchiScale * HIGUCHI_KH_DEFAULT / (2 * Math.sqrt(t));
}

/**
 * One RK4 step. Mutates `C` in place. Returns the new cumulative released mg
 * (caller accumulates).
 *
 * @param {Float64Array} C
 * @param {number} tHours    Time at the start of the step
 * @param {number} dtHours
 * @param {PBPKParams} params
 * @param {{k1: Float64Array, k2: Float64Array, k3: Float64Array, k4: Float64Array, tmp: Float64Array}} scratch
 * @returns {number} drug mass released during this step (mg)
 */
export function stepRK4(C, tHours, dtHours, params, scratch) {
  const { k1, k2, k3, k4, tmp } = scratch;
  // Approximation: hold releaseRate constant across the step (release is a
  // slow-varying source vs ODE timescale; secondary-effect on parity).
  const release = releaseRateAt(tHours, params);
  rhs(C, k1, params, release);
  for (let i = 0; i < N; i++) tmp[i] = C[i] + 0.5 * dtHours * k1[i];
  rhs(tmp, k2, params, release);
  for (let i = 0; i < N; i++) tmp[i] = C[i] + 0.5 * dtHours * k2[i];
  rhs(tmp, k3, params, release);
  for (let i = 0; i < N; i++) tmp[i] = C[i] + dtHours * k3[i];
  rhs(tmp, k4, params, release);
  for (let i = 0; i < N; i++) {
    C[i] += (dtHours / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    if (C[i] < 0) C[i] = 0;
  }
  return release * dtHours;
}

export function makeScratch() {
  return {
    k1: new Float64Array(N),
    k2: new Float64Array(N),
    k3: new Float64Array(N),
    k4: new Float64Array(N),
    tmp: new Float64Array(N),
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
