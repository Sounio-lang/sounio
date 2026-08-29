import { useEffect, useRef, useState } from 'react';
// Pure-JS core also imported by scripts/dissertation/run_pbpk14_node.mjs
// and exercised by scripts/ci/dissertation_frontend_parity_gate.sh.
import { initialState, makeScratch, stepRK4, N } from '../lib/pbpk14_core.mjs';

export interface PBPKParams {
  /** Hepatic clearance, L/h. Default = rapamycin typical (12.4 L/h, 70 kg). */
  clHep: number;
  /** Multiplier on Higuchi K_H to vary patient absorption variability. */
  higuchiScale: number;
  /** Distribution volume scale (1.0 = typical, 0.7 = lean, 1.4 = obese). */
  vdScale: number;
  /** Initial dose released as IV bolus at t=0, mg. 0 = stent only. */
  bolusMg: number;
  /** Whether the Cypher stent is releasing drug. */
  stentActive: boolean;
}

export const DEFAULT_PARAMS: PBPKParams = {
  clHep: 12.4,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: 0.0,
  stentActive: true,
};

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

// Normalization anchor — peak rapamycin blood concentration after Cypher
// implant ≈ 0.85 ng/mL → 8.5e-4 mg/L (Ferron 1997 / Kahan 2001). We use 1.5×
// that as the visual saturation point. This is a *display* constant only;
// trajectories themselves are pure-JS in pbpk14_core and parity-gated.
const C_VISUAL_MAX = 1.3e-3;

export function useSimulation(params: PBPKParams) {
  const [state, setState] = useState<PBPKState>(() => ({
    timeHours: 0,
    concentrations: new Float32Array(N),
    uncertainties: new Float32Array(N),
    releasedMg: 0,
  }));

  // Persistent integrator state (mutable, kept outside React state to avoid
  // O(N) allocations per frame).
  const integratorRef = useRef<{
    C: Float64Array;
    scratch: ReturnType<typeof makeScratch>;
    t: number;
    released: number;
  } | null>(null);
  if (integratorRef.current === null) {
    integratorRef.current = {
      C: initialState(params),
      scratch: makeScratch(),
      t: 0,
      released: 0,
    };
  }

  // Reset on param change (bolus is initial-condition-bearing; clHep / vd
  // affect both IC and ODE; higuchiScale only affects the source — but the
  // simplest correct policy is full restart).
  useEffect(() => {
    const I = integratorRef.current!;
    const fresh = initialState(params);
    for (let i = 0; i < N; i++) I.C[i] = fresh[i];
    I.t = 0;
    I.released = 0;
    setState({
      timeHours: 0,
      concentrations: new Float32Array(N),
      uncertainties: new Float32Array(N),
      releasedMg: 0,
    });
  }, [params.bolusMg, params.clHep, params.higuchiScale, params.vdScale, params.stentActive]);

  const step = (dtHours: number) => {
    const I = integratorRef.current!;
    const dRel = stepRK4(I.C, I.t, dtHours, params, I.scratch);
    I.t += dtHours;
    I.released += dRel;

    const concNorm = new Float32Array(N);
    const uncertNorm = new Float32Array(N);
    // |dC/dt| · σ_rel is a cheap surrogate for the GUM cone radius. The
    // committed gum_budget.csv values are exact for the standard profile;
    // this in-browser proxy widens visibly when clHep moves off-typical,
    // which is the dissertation's contribution-#1 money shot.
    for (let i = 0; i < N; i++) {
      concNorm[i] = Math.max(0, Math.min(1, I.C[i] / C_VISUAL_MAX));
      // |k1[i]| · 0.25 normalised the same way
      uncertNorm[i] = Math.max(0, Math.min(1, Math.abs(I.scratch.k1[i]) / C_VISUAL_MAX * 0.25));
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
