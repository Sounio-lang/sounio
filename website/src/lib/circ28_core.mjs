// website/src/lib/circ28_core.mjs
//
// Independent second implementation of stdlib/darwin_pbpk/circ28.sio (28-state
// closed-circulation whole-body PBPK kernel), written from the equations in
// that file's header and from the structure of its circulation, NOT
// transliterated from circ28_fill_rate_matrix / circ28_prepare / circ28_step.
// Parity gate: scripts/ci/circ28_parity_gate.sh.
//
// Differences in route from the Sounio kernel (so that agreement is evidence):
//   * the system is assembled in MASS FORM from a list of transfer edges
//     (src state -> dst state or sink, coefficient c in L/h, mass flux =
//     c * x_src). K is the mass-rate matrix (mg/h per mg/L), V = diag(volume).
//     Sounio builds A = V^-1 K row by row with hand-written entries.
//   * the SDIRK2 stages are solved for the STAGE VALUES
//         (V - h K) Y1 = V x + h u
//         (V - h K) Y2 = V x + (1 - g) dt (K Y1 + u) + h u        (h = g dt)
//     whereas Sounio solves for slopes k_i = (I - h A)^-1 (A y + u).
//   * (V - h K) is factored by LU with PARTIAL PIVOTING on a Float64Array;
//     Sounio uses Doolittle LU without pivoting on (I - h A).
//   * gamma is computed as 1 - 1/sqrt(2) (Sounio holds a decimal literal).
//   * the mass-conservation property (column sums of K equal minus the sink
//     rates) is asserted when the matrix is built.
// State layout: x[0] venous, x[1] arterial, x[2+2(i-1)] vascular space of
// organ i, x[3+2(i-1)] tissue space of organ i, i = 1..13
// (1 liver, 2 kidney, 3 brain, 4 heart, 5 lung, 6 muscle, 7 adipose, 8 gut,
// 9 skin, 10 bone, 11 spleen, 12 pancreas, 13 rest).

export const NS = 28;
export const GAMMA = 1 - 1 / Math.SQRT2;

export const iv = (i) => 2 + 2 * (i - 1);
export const it = (i) => 3 + 2 * (i - 1);
const PORTAL = new Set([8, 11, 12]);

// Reference adult male (ICRP Publication 89, 2002: 73 kg, cardiac output
// 6.5 L/min = 390 L/h, blood volume 5.3 L).
//   v    organ volume including its own blood: ICRP 89 Table 2.8 masses over
//        tissue densities (ICRU 46 as tabulated in Phys Med Biol 2018,
//        PMID 29999494, Tables 4-5) plus the organ's blood. Gut is stomach,
//        small and large intestine walls; rest closes the body mass.
//   vf   blood of the organ's nutrient bed (small arteries, capillaries,
//        small veins) over v: ICRP 89 Table 2.14 (regional blood content, %
//        of blood volume) times 5.3 L. This is the in-vivo blood through
//        which the organ flow passes, the definition that makes v vf / q the
//        vascular transit time; with it the Leggett-Williams circulation
//        model reproduces observed appearance times (mesenteric artery to
//        portal vein 9.3 s against 8.5 s observed). Fractions measured on
//        excised tissue (0.025 for gut in PK-Sim) give transit times ten
//        times shorter and are not used. Lung holds the pulmonary blood only;
//        bronchial blood and flow are in rest.
//   q    ICRP 89 Tables 2.39-2.40 (% of cardiac output); q[1] is the hepatic
//        artery (6.5 %), the portal organs are gut 15 %, spleen 3 %, pancreas
//        1 %; the arterial flows add up to 390 L/h.
//   pools  venous = large veins 18 % + right heart 4.5 % of blood volume;
//        arterial = aorta and large arteries 6 % + left heart 4.5 %
//        (ICRP 89 Tables 2.13-2.14). Pools + organ blood = 5.300 L.
// The widest literature ranges are for the liver and small-intestine blood
// contents (4-12 % and 0.9-4.4 % of blood volume, Leggett and Williams).
// kp, bp, fu and the clearances are placeholders for a drug; ps holds
// generic permeability-surface products carried over from pbpk28_params_*
// and has no source of its own: a drug set must supply its own.
export function physiology() {
  return {
    v_ven: 1.1925,
    v_art: 0.5565,
    v:  [0.0, 2.228, 0.401, 1.458, 0.373, 1.057, 28.361, 15.528, 1.529, 3.159, 8.448, 0.216, 0.166, 4.315],
    vf: [0.0, 0.237882, 0.264339, 0.043621, 0.142091, 0.526490, 0.026163, 0.017066, 0.242642, 0.050332, 0.043916, 0.343519, 0.191566, 0.052816],
    q:  [0.0, 25.35, 74.10, 46.80, 15.60, 390.0, 66.30, 19.50, 58.50, 19.50, 19.50, 11.70, 3.90, 29.25],
    kp: [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ps: [0.0, 1800.0, 1110.0, 88.0, 160.0, 10500.0, 336.0, 40.0, 660.0, 126.0, 51.0, 150.0, 96.0, 78.0],
    bp: 1.0,
    fu: 1.0,
    clint: 0.0,
    clform: 0.0,
    clint_gut: 0.0,
    cl_renal: 0.0,
    cl_plasma: 0.0,
  };
}

// Same drug-independent test set as ck_set() in circ28_kernel_identities.sio.
export function ckSet() {
  const p = physiology();
  p.kp = [0.0, 4.0, 3.0, 0.5, 2.0, 5.0, 1.5, 3.0, 2.5, 1.0, 0.8, 2.0, 1.8, 1.2];
  p.bp = 0.8;
  p.fu = 0.1;
  return p;
}

export function cloneParams(p) {
  return { ...p, v: p.v.slice(), vf: p.vf.slice(), q: p.q.slice(), kp: p.kp.slice(), ps: p.ps.slice() };
}

// ---- geometry -------------------------------------------------------------
export function cardiacOutput(p) {
  let co = 0;
  for (let i = 1; i <= 13; i++) if (i !== 5) co += p.q[i];
  return co;
}
export const qHepatic = (p) => p.q[1] + p.q[8] + p.q[11] + p.q[12];

export function stateVolumes(p) {
  const V = new Float64Array(NS);
  V[0] = p.v_ven;
  V[1] = p.v_art;
  for (let i = 1; i <= 13; i++) {
    V[iv(i)] = p.v[i] * p.vf[i];
    V[it(i)] = p.v[i] * (1 - p.vf[i]);
  }
  return V;
}
export function bloodVolume(p) {
  let vb = p.v_ven + p.v_art;
  for (let i = 1; i <= 13; i++) vb += p.v[i] * p.vf[i];
  return vb;
}
export function vssPlasma(p) {
  let s = 0;
  for (let i = 1; i <= 13; i++) s += p.v[i] * (1 - p.vf[i]) * p.kp[i];
  return p.bp * bloodVolume(p) + s;
}
export function totalMass(x, p) {
  const V = stateVolumes(p);
  let m = 0;
  for (let k = 0; k < NS; k++) m += V[k] * x[k];
  return m;
}

// ---- closed forms ---------------------------------------------------------
// Series combination of a permeability clearance ps and an intrinsic one ci.
const series = (ps, ci) => (ps * ci) / (ps + ci);

export function hepaticExtraction(p) {
  const ci = (p.clint + p.clform) * p.fu;
  if (!(ci > 0)) return 0;
  const clEff = series(p.ps[1], ci) / p.bp;
  return clEff / (qHepatic(p) + clEff);
}
export function gutAvailability(p) {
  const cg = p.clint_gut * p.fu;
  if (!(cg > 0)) return 1;
  const qb = p.q[8] * p.bp;
  const qg = series(qb, p.ps[8]);        // qb ps / (ps + qb)
  return qg / (qg + cg);
}
export function gutExtraction(p) {
  const cg = p.clint_gut * p.fu;
  if (!(cg > 0)) return 0;
  const clEff = series(p.ps[8], cg) / p.bp;
  return clEff / (p.q[8] + clEff);
}
export function splanchnicClearance(p) {
  const eg = gutExtraction(p);
  const eh = hepaticExtraction(p);
  return p.q[8] * eg + eh * (qHepatic(p) - p.q[8] * eg);
}

// ---- mass-form assembly ---------------------------------------------------
// Edge: mass flux (mg/h) = c * x[src]; dst = -1 means it leaves the model.
export function edges(p) {
  const E = [];
  const add = (src, dst, c) => { E.push({ src, dst, c }); };
  const co = cardiacOutput(p);
  const qh = qHepatic(p);
  const VEN = 0, ART = 1;

  add(VEN, -1, p.cl_plasma / p.bp);          // plasma clearance from venous pool
  add(VEN, iv(5), co);                        // whole cardiac output through the lung
  add(iv(5), ART, co);

  for (let i = 1; i <= 13; i++) {
    // vascular <-> tissue exchange ps (C/bp - T/kp)
    add(iv(i), it(i), p.ps[i] / p.bp);
    add(it(i), iv(i), p.ps[i] / p.kp[i]);
    if (i === 5) continue;                    // lung is fed from the venous pool
    if (i === 1) {
      add(ART, iv(1), p.q[1]);                // hepatic artery
      add(iv(1), VEN, qh);                    // hepatic venous outflow
      add(it(1), -1, (p.clint + p.clform) * p.fu / p.kp[1]);
      continue;
    }
    if (i === 2) {
      // arterial inflow Q, of which cl_renal/bp of the plasma-referred drug is cleared
      add(ART, iv(2), p.q[2] - p.cl_renal / p.bp);
      add(ART, -1, p.cl_renal / p.bp);
    } else {
      add(ART, iv(i), p.q[i]);
    }
    add(iv(i), PORTAL.has(i) ? iv(1) : VEN, p.q[i]);
    if (i === 8) add(it(8), -1, p.clint_gut * p.fu / p.kp[8]);
  }
  return E;
}

// Returns { K (mass-rate matrix, row-major, mg/h per mg/L), V (volumes),
//           sink[k] (loss rate coefficient of state k, L/h) }.
export function buildMassMatrix(p) {
  const V = stateVolumes(p);
  const K = new Float64Array(NS * NS);
  const sink = new Float64Array(NS);
  for (const { src, dst, c } of edges(p)) {
    K[src * NS + src] -= c;
    if (dst >= 0) K[dst * NS + src] += c; else sink[src] += c;
  }
  // conservation: sum over rows of column k equals minus the sink coefficient
  for (let k = 0; k < NS; k++) {
    let s = 0;
    for (let r = 0; r < NS; r++) s += K[r * NS + k];
    if (Math.abs(s + sink[k]) > 1e-9 * (1 + Math.abs(K[k * NS + k]))) {
      throw new Error(`circ28 mass-form column ${k} does not conserve mass: ${s} vs ${-sink[k]}`);
    }
  }
  return { K, V, sink };
}

// ---- LU with partial pivoting --------------------------------------------
export function luFactor(M, n = NS) {
  const lu = Float64Array.from(M);
  const piv = new Int32Array(n);
  for (let j = 0; j < n; j++) {
    let best = j, bv = Math.abs(lu[j * n + j]);
    for (let r = j + 1; r < n; r++) {
      const a = Math.abs(lu[r * n + j]);
      if (a > bv) { bv = a; best = r; }
    }
    if (!(bv > 0)) throw new Error('circ28 LU: singular matrix');
    piv[j] = best;
    if (best !== j) {
      for (let c = 0; c < n; c++) {
        const t = lu[j * n + c]; lu[j * n + c] = lu[best * n + c]; lu[best * n + c] = t;
      }
    }
    const d = lu[j * n + j];
    for (let r = j + 1; r < n; r++) {
      const f = lu[r * n + j] / d;
      lu[r * n + j] = f;
      if (f !== 0) for (let c = j + 1; c < n; c++) lu[r * n + c] -= f * lu[j * n + c];
    }
  }
  return { lu, piv, n };
}
export function luSolve(F, b) {
  const { lu, piv, n } = F;
  const x = Float64Array.from(b);
  for (let j = 0; j < n; j++) {
    const pj = piv[j];
    if (pj !== j) { const t = x[j]; x[j] = x[pj]; x[pj] = t; }
    for (let r = j + 1; r < n; r++) x[r] -= lu[r * n + j] * x[j];
  }
  for (let r = n - 1; r >= 0; r--) {
    let acc = x[r];
    for (let c = r + 1; c < n; c++) acc -= lu[r * n + c] * x[c];
    x[r] = acc / lu[r * n + r];
  }
  return x;
}

// ---- SDIRK2 (Alexander) ---------------------------------------------------
export function prepare(p, dt) {
  const { K, V } = buildMassMatrix(p);
  const h = GAMMA * dt;
  const M = new Float64Array(NS * NS);
  for (let r = 0; r < NS; r++) {
    for (let c = 0; c < NS; c++) M[r * NS + c] = (r === c ? V[r] : 0) - h * K[r * NS + c];
  }
  return {
    F: luFactor(M), K, V, dt, p,
    // loss coefficients per tally (mass rate = coefficient * state):
    kHepOnly: p.clint * p.fu / p.kp[1],
    kForm: p.clform * p.fu / p.kp[1],
    kRen: p.cl_renal / p.bp,
    kPla: p.cl_plasma / p.bp,
    kGut: p.clint_gut * p.fu / p.kp[8],
  };
}

export const stateZero = () => new Float64Array(NS);
export const elimZero = () => ({ hepatic: 0, formed: 0, renal: 0, plasma: 0, gut: 0, auc_ven: 0, auc_art: 0 });

// One step. ivRate (mg/h) goes into the venous pool, oralRate (mg/h) into gut
// tissue; both constant over the step. x is updated in place.
export function step(x, sys, ivRate, oralRate, el) {
  const { F, K, V, dt, p } = sys;
  const g = GAMMA, h = g * dt;
  const u = new Float64Array(NS);               // mass input rate, mg/h
  u[0] = ivRate;
  u[it(8)] = oralRate;                          // gut tissue state, index 17

  // stage 1: (V - h K) Y1 = V x + h u
  const r1 = new Float64Array(NS);
  for (let k = 0; k < NS; k++) r1[k] = V[k] * x[k] + h * u[k];
  const Y1 = luSolve(F, r1);

  // stage 2: (V - h K) Y2 = V x + (1-g) dt (K Y1 + u) + h u
  const r2 = new Float64Array(NS);
  for (let k = 0; k < NS; k++) {
    let f = u[k];
    for (let c = 0; c < NS; c++) f += K[k * NS + c] * Y1[c];
    r2[k] = V[k] * x[k] + (1 - g) * dt * f + h * u[k];
  }
  const Y2 = luSolve(F, r2);

  // quadrature with the scheme's weights b = (1-g, g)
  const w1 = (1 - g) * dt, w2 = g * dt;
  const q = (k) => w1 * Y1[k] + w2 * Y2[k];
  el.auc_ven += q(0);
  el.auc_art += q(1);
  const qTl = q(it(1)), qTg = q(it(8));
  el.hepatic += sys.kHepOnly * qTl;
  el.formed += sys.kForm * qTl;
  el.renal += sys.kRen * q(1);
  el.plasma += sys.kPla * q(0);
  el.gut += sys.kGut * qTg;
  for (let k = 0; k < NS; k++) x[k] = Y2[k];
}

// ---- scenario driver (shared by the runner and its negative control) ------
// spec: { p, dt, horizon, route: 'iv'|'oral', dose, dur (iv), ka (oral) }
export function runScenario(spec) {
  const { p, dt, horizon, route, dose } = spec;
  const sys = prepare(p, dt);
  const x = stateZero();
  const el = elimZero();
  const nTot = Math.round(horizon / dt);
  const nInf = route === 'iv' ? Math.round(spec.dur / dt) : 0;
  const sampleSteps = (spec.sampleTimes || []).map((t) => Math.round(t / dt));
  const samples = [];
  let input = 0;
  for (let n = 0; n < nTot; n++) {
    let ivRate = 0, oral = 0;
    if (route === 'iv') {
      if (n < nInf) { ivRate = dose / spec.dur; input += ivRate * dt; }
    } else {
      oral = dose * Math.exp(-spec.ka * (n * dt + 0.5 * dt));
      input += oral * dt;
    }
    step(x, sys, ivRate, oral, el);
    const j = sampleSteps.indexOf(n + 1);
    if (j >= 0) samples[j] = { ven: x[0], art: x[1] };
  }
  return { x, el, samples, input, massEnd: totalMass(x, p) };
}
