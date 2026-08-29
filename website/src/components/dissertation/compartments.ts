// 14-compartment topology for the rapamycin PBPK dissertation.
// Indices and ordering match stdlib/darwin_pbpk/tsit5_pbpk14.sio:6-12.
// Anatomical positions in scene-space (Y up, +Z forward, units ≈ half-body
// heights). Kp values are population-typical rapamycin (Ferron 1997, Kahan 2001).

export interface Compartment {
  index: number;
  name: string;
  label: string;
  family: 'central' | 'eliminating' | 'distribution' | 'tissue' | 'bulk';
  position: [number, number, number];
  radius: number;
  kp: number; // partition coefficient (tissue:plasma at steady state)
  q: number;  // organ blood flow, L/h, 70 kg adult
  color: string; // base hex for the organ family
}

export const COMPARTMENTS: Compartment[] = [
  { index: 0,  name: 'blood',    label: 'Blood (central)',  family: 'central',      position: [0,  3.2, 0.0],   radius: 0.55, kp: 1.00, q: 350, color: '#dc2626' },
  { index: 1,  name: 'liver',    label: 'Liver',            family: 'eliminating',  position: [0.75, 2.0, 0.35], radius: 0.55, kp: 5.40, q:  90, color: '#a16207' },
  { index: 2,  name: 'kidney',   label: 'Kidney',           family: 'eliminating',  position: [-0.85, 1.6, -0.35], radius: 0.36, kp: 4.20, q:  74, color: '#854d0e' },
  { index: 3,  name: 'brain',    label: 'Brain',            family: 'tissue',       position: [0,  5.4, 0.05],  radius: 0.65, kp: 0.10, q:  44, color: '#9333ea' },
  { index: 4,  name: 'heart',    label: 'Heart',            family: 'tissue',       position: [-0.05, 3.05, 0.55], radius: 0.42, kp: 2.30, q:  16, color: '#ef4444' },
  { index: 5,  name: 'lung',     label: 'Lung',             family: 'central',      position: [0,  3.6, -0.25], radius: 0.78, kp: 3.10, q: 350, color: '#fb7185' },
  { index: 6,  name: 'muscle',   label: 'Muscle',           family: 'bulk',         position: [1.9, 1.6, 0.0],  radius: 0.62, kp: 0.50, q:  42, color: '#b45309' },
  { index: 7,  name: 'adipose',  label: 'Adipose',          family: 'bulk',         position: [-1.9, 1.4, 0.0], radius: 0.58, kp: 0.30, q:  10, color: '#f59e0b' },
  { index: 8,  name: 'gut',      label: 'Gut',              family: 'distribution', position: [0,  1.45, 0.55], radius: 0.55, kp: 2.60, q:  55, color: '#d97706' },
  { index: 9,  name: 'skin',     label: 'Skin',             family: 'bulk',         position: [0,  0.4, 1.05],  radius: 0.48, kp: 0.80, q:  21, color: '#fbbf24' },
  { index: 10, name: 'bone',     label: 'Bone',             family: 'bulk',         position: [0,  0.4, -1.05], radius: 0.42, kp: 0.20, q:  17, color: '#e7e5e4' },
  { index: 11, name: 'spleen',   label: 'Spleen',           family: 'distribution', position: [-0.85, 2.1, 0.4], radius: 0.30, kp: 2.10, q:  10, color: '#7c2d12' },
  { index: 12, name: 'pancreas', label: 'Pancreas',         family: 'distribution', position: [-0.20, 2.05, 0.45], radius: 0.28, kp: 1.80, q:   8, color: '#f97316' },
  { index: 13, name: 'other',    label: 'Other tissues',    family: 'bulk',         position: [1.6, -1.0, 0.0], radius: 0.45, kp: 0.40, q:  13, color: '#71717a' },
];

// Vessel topology: directed flow edges (source compartment → sink compartment)
// at the 1st-pass-through-the-heart approximation. Blood (0) and lung (5) form
// the central loop; everything else is fed by blood and drains back to blood
// (with liver downstream of gut/spleen/pancreas via the portal vein).
export interface FlowEdge {
  from: number;
  to: number;
  width: number; // visual weight ~ ln(Q)
  portal?: boolean; // true if part of the hepatic portal axis
}

export const FLOW_EDGES: FlowEdge[] = [
  // Pulmonary loop
  { from: 0, to: 5, width: 1.4 },
  { from: 5, to: 0, width: 1.4 },
  // Systemic supply (blood → organs)
  { from: 0, to: 3,  width: 0.6 },           // brain
  { from: 0, to: 4,  width: 0.5 },           // heart
  { from: 0, to: 2,  width: 0.8 },           // kidney
  { from: 0, to: 6,  width: 0.8 },           // muscle
  { from: 0, to: 7,  width: 0.4 },           // adipose
  { from: 0, to: 8,  width: 0.7, portal: true }, // gut
  { from: 0, to: 9,  width: 0.5 },           // skin
  { from: 0, to: 10, width: 0.4 },           // bone
  { from: 0, to: 11, width: 0.4, portal: true }, // spleen
  { from: 0, to: 12, width: 0.4, portal: true }, // pancreas
  { from: 0, to: 13, width: 0.5 },           // other
  // Portal axis (gut/spleen/pancreas → liver)
  { from: 8,  to: 1, width: 0.7, portal: true },
  { from: 11, to: 1, width: 0.4, portal: true },
  { from: 12, to: 1, width: 0.4, portal: true },
  // Direct hepatic artery
  { from: 0, to: 1, width: 0.4 },
  // Venous return (organs → blood)
  { from: 1,  to: 0, width: 1.0 }, // post-liver
  { from: 2,  to: 0, width: 0.8 },
  { from: 3,  to: 0, width: 0.6 },
  { from: 4,  to: 0, width: 0.5 },
  { from: 6,  to: 0, width: 0.8 },
  { from: 7,  to: 0, width: 0.4 },
  { from: 9,  to: 0, width: 0.5 },
  { from: 10, to: 0, width: 0.4 },
  { from: 13, to: 0, width: 0.5 },
];

// Cypher drug-eluting stent — Cordis 2003.
// Source: stdlib/darwin_pbpk/release/biomaterial_release.sio:16-26
// 140 µg rapamycin total, ~80% released over 30 days.
// Higuchi diffusion model: dQ/dt = K_H / (2 * sqrt(t))   with K_H ≈ 0.00417 / √h
export const STENT = {
  position: [0.0, 3.0, 0.55] as [number, number, number],
  innerRadius: 0.07,
  outerRadius: 0.10,
  length: 0.34,
  totalDoseMicrograms: 140,
  releaseFraction30d: 0.80,
  higuchiKH: 0.00417, // mg / √h
} as const;
