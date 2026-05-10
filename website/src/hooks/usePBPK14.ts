import { useEffect, useRef, useState } from 'react';
import { COMPARTMENTS, STENT } from '../components/dissertation/compartments';

/**
 * 14-compartment flow-limited PBPK ODE driver for rapamycin / Cypher DES.
 *
 * Reference: stdlib/darwin_pbpk/tsit5_pbpk14.sio (compartment list, indices),
 * stdlib/darwin_pbpk/drugs/rapamycin.sio (kinetics), and
 * stdlib/darwin_pbpk/release/biomaterial_release.sio (Higuchi release).
 *
 * Browser-side this is a fixed-step RK4. A CI parity gate
 * (scripts/ci/dissertation_frontend_parity_gate.sh) compares 12 sample points
 * to the Sounio Tsit5 reference solver and requires < 1% RMSE before any
 * deploy — see the Lane 9 plan document.
 *
 * Mass balance (per organ i, flow-limited):
 *   V_i · dC_i/dt = Q_i · ( C_blood - C_i / Kp_i )
 *
 * Blood compartment additionally receives:
 *   + Cypher stent release: dQ_drug/dt = K_H / (2·√t)  for t > 0
 *   - Hepatic clearance:    -CL_hep · C_blood
 *
 * State vector C ∈ ℝ¹⁴ in mg/L; cumulative released mass tracked separately.
 */

export interface PBPKParams {
  /** Hepatic clearance, L/h. Default = rapamycin typical (12.4 L/h, 70 kg). */
  clHep: number;
  /** Multiplier on Higuchi K_H to vary patient absorption variability. */
  higuchiScale: number;
  /** Distribution volume scale (1.0 = typical, 0.7 = lean, 1.4 = obese). */
  vdScale: number;
  /** Initial dose released as IV bolus at t=0, mg. 0 = stent only. */
  bolusMg: number;
}

export const DEFAULT_PARAMS: PBPKParams = {
  clHep: 12.4,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: 0.0,
};

// Reference volumes V_i, L, 70 kg adult. Order matches COMPARTMENTS[].
// Sources: ICRP, scaled to standard man. Blood = central plasma + RBC.
const V_REF: readonly number[] = [
  5.0,   // blood
  1.8,   // liver
  0.31,  // kidney
  1.45,  // brain
  0.33,  // heart
  1.1,   // lung
  28.0,  // muscle
  20.0,  // adipose
  1.2,   // gut
  3.6,   // skin
  6.6,   // bone
  0.18,  // spleen
  0.09,  // pancreas
  3.7,   // other
];

export interface PBPKState {
  /** Hours since stent implantation. */
  timeHours: number;
  /** Normalized [0..1] concentrations (clamped to a per-organ reference C_max). */
  concentrations: Float32Array;
  /** Normalized [0..1] uncertainty cone radius — proxy: |dC/dt| · σ_param. */
  uncertainties: Float32Array;
  /** Cumulative drug mass released from stent, mg. */
  releasedMg: number;
}

// Normalization anchors. The peak rapamycin blood concentration after Cypher
// implant is ≈ 0.85 ng/mL → 8.5e-4 mg/L (Ferron 1997 / Kahan 2001 literature).
// We use 1.5 × that as the visual saturation point.
const C_VISUAL_MAX = 1.3e-3; // mg/L

function rhs(
  C: Float32Array,
  out: Float32Array,
  params: PBPKParams,
  releaseRate: number, // mg/h
) {
  const cBlood = C[0];
  // Start by initialising blood derivative; we will accumulate venous returns.
  out[0] = 0;
  for (let i = 1; i < 14; i++) {
    const c = COMPARTMENTS[i];
    const v = V_REF[i] * params.vdScale;
    const ci = C[i];
    const flux = c.q * (cBlood - ci / c.kp); // mg/h delivered to organ i
    out[i] = flux / v;
    out[0] -= flux; // mass conservation: leaves blood, enters organ
  }
  // Blood: convert mass/h back to concentration/h, add stent release, subtract hepatic clearance.
  const vBlood = V_REF[0] * params.vdScale;
  out[0] = out[0] / vBlood + releaseRate / vBlood - (params.clHep * cBlood) / vBlood;
}

export function useSimulation(params: PBPKParams) {
  const [state, setState] = useState<PBPKState>(() => {
    const C = new Float32Array(14);
    C[0] = params.bolusMg / (V_REF[0] * params.vdScale);
    return {
      timeHours: 0,
      concentrations: new Float32Array(14),
      uncertainties: new Float32Array(14),
      releasedMg: 0,
    };
  });

  // Persistent integrator state (mutable, kept outside React state to avoid
  // O(N) Float32Array allocations per frame).
  const integrator = useRef({
    C: new Float32Array(14),
    k1: new Float32Array(14),
    k2: new Float32Array(14),
    k3: new Float32Array(14),
    k4: new Float32Array(14),
    tmp: new Float32Array(14),
    t: 0,
    released: 0,
  });

  // Reset on param change (bolus is initial-condition-bearing).
  useEffect(() => {
    const I = integrator.current;
    I.C.fill(0);
    I.C[0] = params.bolusMg / (V_REF[0] * params.vdScale);
    I.t = 0;
    I.released = 0;
    setState({
      timeHours: 0,
      concentrations: new Float32Array(14),
      uncertainties: new Float32Array(14),
      releasedMg: 0,
    });
  }, [params.bolusMg, params.clHep, params.higuchiScale, params.vdScale]);

  // Advance the simulation by `dtHours` and emit a new state.
  const step = (dtHours: number) => {
    const I = integrator.current;
    const t0 = Math.max(1e-3, I.t);
    const releaseRate = params.higuchiScale * STENT.higuchiKH / (2 * Math.sqrt(t0));
    // RK4
    rhs(I.C, I.k1, params, releaseRate);
    for (let i = 0; i < 14; i++) I.tmp[i] = I.C[i] + 0.5 * dtHours * I.k1[i];
    rhs(I.tmp, I.k2, params, releaseRate);
    for (let i = 0; i < 14; i++) I.tmp[i] = I.C[i] + 0.5 * dtHours * I.k2[i];
    rhs(I.tmp, I.k3, params, releaseRate);
    for (let i = 0; i < 14; i++) I.tmp[i] = I.C[i] + dtHours * I.k3[i];
    rhs(I.tmp, I.k4, params, releaseRate);
    for (let i = 0; i < 14; i++) {
      I.C[i] += (dtHours / 6) * (I.k1[i] + 2 * I.k2[i] + 2 * I.k3[i] + I.k4[i]);
      if (I.C[i] < 0) I.C[i] = 0;
    }
    I.t += dtHours;
    I.released += releaseRate * dtHours;

    // Normalize to [0..1] for visual encoding, with a single reference anchor.
    const concNorm = new Float32Array(14);
    const uncertNorm = new Float32Array(14);
    for (let i = 0; i < 14; i++) {
      const ci = I.C[i] / C_VISUAL_MAX;
      concNorm[i] = Math.max(0, Math.min(1, ci));
      // Uncertainty proxy: σ_CL × |∂C/∂CL| is dominant; use |k1| · σ_rel as cheap surrogate.
      uncertNorm[i] = Math.max(0, Math.min(1, Math.abs(I.k1[i]) / C_VISUAL_MAX * 0.25));
    }
    setState({
      timeHours: I.t,
      concentrations: concNorm,
      uncertainties: uncertNorm,
      releasedMg: I.released,
    });
  };

  return { state, step };
}
