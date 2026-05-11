// React adapter for the PBPK28 permeability-limited + TMDD + PD core
// (website/src/lib/pbpk28_core.mjs). Replaces the rapamycin-only PBPK14
// hook with a drug-agnostic interface that selects between rapamycin
// (Cypher stent / Higuchi release / FKBP12/mTORC1 / neointimal index) and
// semaglutide (SC depot / GLP-1R / glucose-insulin Bergman PD) via a
// `drug` argument. Same Float32Array(N) state shape as usePBPK14 so the
// existing organ-color rendering picks up both drugs without per-drug code.
//
// Trajectory math is the fully-coupled Crank-Nicolson on the 27-state
// block-arrow matrix (commit c574ee81), plus the per-organ analytical TMDD
// (G-β-2) and PD (G-γ / G-δ-3) sub-steps. All bit-identical to the Sounio
// reference at < 1% RMSE per gate Cases 1, 5–9.

import { useEffect, useRef, useState } from 'react';
import {
  initialState, makeScratch, stepStrang, organAverage,
  DEFAULT_PARAMS_RAPAMYCIN, DEFAULT_PARAMS_SEMAGLUTIDE, N,
} from '../lib/pbpk28_core.mjs';

export type DrugId = 'rapamycin' | 'semaglutide';

/**
 * PBPKParams — keeps the same shape as the legacy usePBPK14.PBPKParams
 * (required fields) so that components consuming it (ConfidenceGate,
 * GumBudgetBar, etc.) don't need updating. Drug-specific semantics:
 *   rapamycin:   clHep = hepatic clearance L/h; higuchiScale = Cypher K_H mult
 *   semaglutide: clHep = proteolytic clearance L/h; higuchiScale = unused
 */
export interface PBPKParams {
  /** Hepatic / proteolytic clearance, L/h. */
  clHep: number;
  /** Multiplier on Higuchi K_H (rapamycin only; ignored for SC depot). */
  higuchiScale: number;
  /** Distribution-volume scale (1.0 = typical, 0.7 = lean, 1.4 = obese). */
  vdScale: number;
  /** Initial IV bolus (mg). Rapamycin uses for IV; semaglutide ignores. */
  bolusMg: number;
  /** Whether the release source (stent or SC depot) is active. */
  stentActive: boolean;
}

export interface PBPKState {
  /** Hours since release-source activation. */
  timeHours: number;
  /** Normalized [0..1] organ-average concentrations (clamped to drug-specific c_visual_max). */
  concentrations: Float32Array;
  /** Normalized [0..1] uncertainty cone radius — |dC/dt|·σ_param surrogate. */
  uncertainties: Float32Array;
  /** Cumulative drug mass released, mg. */
  releasedMg: number;
  /** Per-TMDD-organ free receptor (nmol/L). Length N; non-TMDD organs = 0. */
  rfree: Float32Array;
  /** Per-TMDD-organ drug-receptor complex (nmol/L). */
  dr: Float32Array;
  /** PD readout state. Interpretation depends on drug:
   *   rapamycin: A[heart]=mTORC1 active fraction (0..1), Ni[heart]=neointimal idx
   *   semaglutide: A[panc]=ΔG (mg/dL), Ni[panc]=ΔI (mU/L) */
  pdA: Float32Array;
  pdN: Float32Array;
}

function mergeParams(drug: DrugId, params: PBPKParams) {
  const base = drug === 'semaglutide' ? DEFAULT_PARAMS_SEMAGLUTIDE : DEFAULT_PARAMS_RAPAMYCIN;
  return {
    ...base,
    clHep:        params.clHep,
    higuchiScale: params.higuchiScale,
    vdScale:      params.vdScale,
    bolusMg:      params.bolusMg,
    stentActive:  params.stentActive,
  };
}

export function useSimulation(drug: DrugId, params: PBPKParams) {
  const [state, setState] = useState<PBPKState>(() => ({
    timeHours: 0,
    concentrations: new Float32Array(N),
    uncertainties:  new Float32Array(N),
    releasedMg: 0,
    rfree: new Float32Array(N),
    dr:    new Float32Array(N),
    pdA:   new Float32Array(N),
    pdN:   new Float32Array(N),
  }));

  const integratorRef = useRef<{
    profile: any;
    Cv: Float64Array;
    Ct: Float64Array;
    Rfree: Float64Array;
    DR: Float64Array;
    A: Float64Array;
    Ni: Float64Array;
    scratch: ReturnType<typeof makeScratch>;
    t: number;
    released: number;
  } | null>(null);

  const profile = mergeParams(drug, params);

  // (Re)initialize integrator whenever drug or any param changes.
  useEffect(() => {
    const fresh = initialState(profile);
    integratorRef.current = {
      profile,
      Cv: fresh.Cv,
      Ct: fresh.Ct,
      Rfree: fresh.Rfree,
      DR: fresh.DR,
      A: fresh.A,
      Ni: fresh.N,
      scratch: makeScratch(),
      t: 0,
      released: 0,
    };
    setState({
      timeHours: 0,
      concentrations: new Float32Array(N),
      uncertainties:  new Float32Array(N),
      releasedMg: 0,
      rfree: new Float32Array(N),
      dr:    new Float32Array(N),
      pdA:   new Float32Array(N),
      pdN:   new Float32Array(N),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drug, params.bolusMg, params.clHep, params.higuchiScale, params.vdScale, params.stentActive]);

  const step = (dtHours: number) => {
    const I = integratorRef.current;
    if (!I) return;
    const dRel = stepStrang(I.Cv, I.Ct, I.t, dtHours, I.profile, I.scratch, I.Rfree, I.DR, I.A, I.Ni);
    I.t += dtHours;
    I.released += dRel;

    const avg = organAverage(I.Cv, I.Ct, I.profile);
    const cMax = I.profile.cVisualMax || 1.3e-3;
    const concNorm   = new Float32Array(N);
    const uncertNorm = new Float32Array(N);
    const rfreeArr   = new Float32Array(N);
    const drArr      = new Float32Array(N);
    const pdAArr     = new Float32Array(N);
    const pdNArr     = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      concNorm[i]   = Math.max(0, Math.min(1, avg[i] / cMax));
      // Crude |dC/dt|·σ surrogate: difference between Cv and Ct (out-of-equilibrium proxy)
      const oof = Math.abs(I.Cv[i] - I.Ct[i] / (I.profile.kp[i] || 1));
      uncertNorm[i] = Math.max(0, Math.min(1, oof / cMax * 0.25));
      rfreeArr[i]   = I.Rfree[i];
      drArr[i]      = I.DR[i];
      pdAArr[i]     = I.A[i];
      pdNArr[i]     = I.Ni[i];
    }
    setState({
      timeHours: I.t,
      concentrations: concNorm,
      uncertainties:  uncertNorm,
      releasedMg: I.released,
      rfree: rfreeArr,
      dr:    drArr,
      pdA:   pdAArr,
      pdN:   pdNArr,
    });
  };

  return { state, step, drug, profile };
}

/** DEFAULT_PARAMS — sliders' "typical-patient" anchor (rapamycin units;
 *  cosmetically reused when semaglutide is selected, then ignored where the
 *  field doesn't apply — e.g. higuchiScale, which doesn't drive SC depot). */
export const DEFAULT_PARAMS: PBPKParams = {
  clHep: 12.4,
  higuchiScale: 1.0,
  vdScale: 1.0,
  bolusMg: 0.0,
  stentActive: true,
};
