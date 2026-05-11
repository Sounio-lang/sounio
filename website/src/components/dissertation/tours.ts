// Scripted tour definitions for the dissertation viewer. Each tour is a list
// of keyframes the CameraDirector lerps through. Optional `profile` /
// `higuchiScale` fields cause the viewer to apply a scene-state change at
// that frame (the third tour relies on this to make the GUM cone breathe).
//
// All times in seconds since tour start. Camera coordinates in the same scene
// space as `compartments.ts` (Y up, +Z forward, ≈ half-body heights).

import type { PBPKParams } from '../../hooks/usePBPK14';

export interface TourKeyframe {
  t: number;                       // seconds from tour start
  camera: [number, number, number];
  target: [number, number, number];
  profile?: keyof typeof PATIENT_KNOBS;
  higuchiScale?: number;
  narration?: string;              // sentence shown under the canvas
}

export interface Tour {
  id: string;
  title: string;
  duration: number;                // seconds, must equal last keyframe.t
  keyframes: TourKeyframe[];
}

// Mirror of PATIENT_PROFILES in DissertationViewer so the tours can be
// authored declaratively without importing the viewer.
export const PATIENT_KNOBS: Record<string, Partial<PBPKParams>> = {
  typical: { clHep: 12.4, vdScale: 1.0, higuchiScale: 1.0 },
  'low CL': { clHep: 6.2 },
  'high CL': { clHep: 24.8 },
  lean: { vdScale: 0.7 },
  obese: { vdScale: 1.4 },
};

export const TOURS: Tour[] = [
  {
    id: 'cypher',
    title: 'Cypher → blood → liver',
    duration: 30,
    keyframes: [
      { t: 0,  camera: [6, 3.5, 8],   target: [0, 2.5, 0],
        narration: 'Default anterior view — 14 compartments arranged around the central blood pool.' },
      { t: 4,  camera: [1.2, 3.2, 2.4], target: [0, 3.0, 0.55],
        narration: '140 µg of rapamycin is embedded in the Cypher stent coating, in the coronary location.' },
      { t: 9,  camera: [1.5, 3.6, 3.0], target: [0, 3.2, 0.0],
        narration: 'Higuchi diffusion releases the drug into blood at rate K_H / (2√t) — the yellow particles trace the early-fast / late-slow envelope.' },
      { t: 15, camera: [4.5, 3.0, 5.0], target: [0.0, 2.8, 0.0],
        narration: 'Blood (central compartment) carries the rapamycin systemically; flow ≈ 350 L/h.' },
      { t: 22, camera: [3.5, 2.2, 4.0], target: [0.75, 2.0, 0.35],
        narration: 'Liver receives ~90 L/h. Kp = 5.4 means tissue concentration steady-states ~5× plasma. Hepatic clearance is the dominant elimination route.' },
      { t: 28, camera: [5.5, 3.0, 7.0], target: [0, 2.5, 0],
        narration: 'Pulling back to overview — note red organ saturation tracking the blood-driven distribution kinetics.' },
      { t: 30, camera: [6, 3.5, 8],   target: [0, 2.5, 0] },
    ],
  },
  {
    id: 'bbb',
    title: 'BBB closeup — Kp = 0.10',
    duration: 20,
    keyframes: [
      { t: 0,  camera: [6, 3.5, 8],   target: [0, 2.5, 0],
        narration: 'Brain has the lowest partition coefficient in this model.' },
      { t: 4,  camera: [1.5, 5.6, 2.5], target: [0, 5.4, 0.0],
        narration: 'Kp_brain = 0.10 — P-gp efflux at the blood-brain barrier (Lampen 1998).' },
      { t: 10, camera: [0.0, 5.4, 2.4], target: [0, 5.4, 0.0],
        narration: 'The semi-transparent GUM shell is thin here: little drug accumulates, so little to be uncertain about.' },
      { t: 15, camera: [-1.6, 5.5, 2.5], target: [0, 5.4, 0.0],
        narration: 'BPR (brain-plasma ratio) stays < 0.15 in the dissertation gate — clinically, no CNS toxicity window opens.' },
      { t: 20, camera: [6, 3.5, 8],   target: [0, 2.5, 0] },
    ],
  },
  {
    id: 'gum-cone',
    title: 'GUM cone widening under CL_hep variability',
    duration: 20,
    keyframes: [
      { t: 0,  camera: [4.0, 2.2, 5.0], target: [0.0, 2.0, 0.0],
        narration: 'Focusing on liver + kidney — the elimination organs whose uncertainty the GUM cone visualises.',
        profile: 'typical' },
      { t: 3,  camera: [3.5, 2.2, 4.5], target: [0.0, 2.0, 0.0],
        profile: 'typical' },
      { t: 7,  camera: [3.5, 2.2, 4.5], target: [0.0, 2.0, 0.0],
        narration: 'Switching to CYP3A4 poor metaboliser (low CL). Watch the translucent shells around liver and kidney expand.',
        profile: 'low CL' },
      { t: 12, camera: [3.5, 2.2, 4.5], target: [0.0, 2.0, 0.0],
        profile: 'low CL' },
      { t: 14, camera: [3.5, 2.2, 4.5], target: [0.0, 2.0, 0.0],
        narration: 'Back to typical — the cones shrink. The GUM-through-ODE propagator is producing the visible width here.',
        profile: 'typical' },
      { t: 18, camera: [5.0, 3.0, 6.5], target: [0, 2.5, 0],
        profile: 'typical' },
      { t: 20, camera: [6, 3.5, 8],   target: [0, 2.5, 0],
        profile: 'typical' },
    ],
  },
];

export const CAMERA_PRESETS = {
  anterior: { camera: [6, 3.5, 8] as [number, number, number], target: [0, 2.5, 0] as [number, number, number] },
  stent:    { camera: [1.5, 3.6, 3.0] as [number, number, number], target: [0, 3.2, 0.0] as [number, number, number] },
  bbb:      { camera: [1.5, 5.6, 2.5] as [number, number, number], target: [0, 5.4, 0.0] as [number, number, number] },
};

/** Lerp helper used by the CameraDirector during playback. */
export function lerpKeyframe(a: TourKeyframe, b: TourKeyframe, time: number) {
  const span = b.t - a.t;
  const u = span <= 0 ? 1 : Math.max(0, Math.min(1, (time - a.t) / span));
  // Ease in/out (smoothstep) — softer than linear, no overshoot.
  const e = u * u * (3 - 2 * u);
  return {
    camera: [
      a.camera[0] + (b.camera[0] - a.camera[0]) * e,
      a.camera[1] + (b.camera[1] - a.camera[1]) * e,
      a.camera[2] + (b.camera[2] - a.camera[2]) * e,
    ] as [number, number, number],
    target: [
      a.target[0] + (b.target[0] - a.target[0]) * e,
      a.target[1] + (b.target[1] - a.target[1]) * e,
      a.target[2] + (b.target[2] - a.target[2]) * e,
    ] as [number, number, number],
  };
}
