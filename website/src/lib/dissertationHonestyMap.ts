/**
 * Defense map: each of the 19 pbpk_suite FAILs named onto one of
 * the five epistemic states.
 *
 * The suite family (toolchain / resource-ceiling / science) lives in
 * dissertationHonestyNow.ts and is what grok-cli2 measured. This file
 * is the other axis — the one the defense needs and that until now
 * existed only as prose across six audit notes:
 *
 *   12 toolchain ≠ 12 refusals.
 *   Ten never started science (compiler refused).
 *   Two started and printed a fabricated numeral (cluster C).
 *   Seven started science and were cut by rc=182.
 *   Zero are a science or model defect.
 *
 * Source of the 19 names and evidence strings (do not invent a later
 * run): docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md
 * finished 2026-08-17T22:21:35Z. Same ledger as PBPK_SUITE_NOW.
 */

import { PBPK_SUITE_NOW, measureMayPrint } from './dissertationHonestyNow';

export const DEFENSE_STATES = [
  'verified',
  'uncertain',
  'refused',
  'unbounded',
  'fabricated',
] as const;

export type DefenseState = (typeof DEFENSE_STATES)[number];

export type FailKind = 'compile' | 'silent-zero' | 'bit-pattern' | 'interrupted';

export type FailFamilyId = 'toolchain' | 'resource-ceiling';

export type FailRow = {
  n: number;
  name: string;
  family: FailFamilyId;
  state: Extract<DefenseState, 'refused' | 'fabricated'>;
  kind: FailKind;
  evidence: string;
};

/**
 * Order follows the gate registration table in the 2026-08-17 remasure.
 * Evidence strings are the "Evidence (this run)" column, shortened to
 * the diagnostic, not paraphrased into adjectives.
 */
export const PBPK_SUITE_FAILS: readonly FailRow[] = [
  {
    n: 4,
    name: 'rapamycin_epistemic_adaptive',
    family: 'toolchain',
    state: 'fabricated',
    kind: 'silent-zero',
    evidence: 'var(blood/brain/periph)=0.000000; sibling #3 still PASS',
  },
  {
    n: 7,
    name: 'rapamycin_clinical',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: 'PART A clinical PASS; PART B GUM → handles full; rc=182',
  },
  {
    n: 8,
    name: 'gum_vs_mc',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: 'GUM SD printed; MC → handles full; rc=182',
  },
  {
    n: 10,
    name: 'rapamycin_pop_sim',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: '20-patient header; handles full; rc=182',
  },
  {
    n: 13,
    name: 'd2_gum',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: 'native emit; GUM header; handles full; rc=182',
  },
  {
    n: 14,
    name: 'd2_voi',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: 'native emit; VoI header; handles full; rc=182',
  },
  {
    n: 16,
    name: 'dissertation_oral_pd',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E175 private drugs/rapamycin::rapamycin_mean_params',
  },
  {
    n: 17,
    name: 'dissertation_steady_state',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E175 private + E008 return type + E137 print_i64',
  },
  {
    n: 18,
    name: 'dissertation_steady_state_fullvd',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'same family as #17; E175 + E008 + E137',
  },
  {
    n: 19,
    name: 'dissertation_scenario_gate',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E175 rapamycin_mean_params; E137 print_i64 in bbb_voi',
  },
  {
    n: 35,
    name: 'halo_pgx_gate',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E175 private math/pure::sqrt from aggregate_confidence',
  },
  {
    n: 36,
    name: 'halo_pgx_gate_pass',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E170 .value needs with Epistemic / acknowledge',
  },
  {
    n: 39,
    name: 'epistemic_pbpk28',
    family: 'toolchain',
    state: 'fabricated',
    kind: 'bit-pattern',
    evidence: '8/9 internal PASS; TEST 6 AUC confidence 4604219396932172800',
  },
  {
    n: 41,
    name: 'pbpk28_sobol_pce',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E009 fn-type mismatch; E035 missing Epistemic',
  },
  {
    n: 42,
    name: 'pbpk28_mc_cross_validation',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: 'u_Hessian=0.295160; MC N=2000 → handles full; rc=182',
  },
  {
    n: 43,
    name: 'pbpk28_mc_prior_family_sweep',
    family: 'resource-ceiling',
    state: 'fabricated',
    kind: 'interrupted',
    evidence: 'family 0 N=2000 → handles full; rc=182',
  },
  {
    n: 44,
    name: 'rapamycin_kaxi_fuse_prior',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E011/E013/E137 acknowledge on kaxi path',
  },
  {
    n: 48,
    name: 'dissertation_pgx_demo',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E175 math/pure::sqrt (same as #35)',
  },
  {
    n: 52,
    name: 'pbpk28_rapamycin_clinical',
    family: 'toolchain',
    state: 'refused',
    kind: 'compile',
    evidence: 'E011 ontology/model methods; never reaches data path',
  },
];

export type StateCounts = Record<DefenseState, number>;

export function countStates(rows: readonly FailRow[]): StateCounts {
  const counts: StateCounts = {
    verified: 0,
    uncertain: 0,
    refused: 0,
    unbounded: 0,
    fabricated: 0,
  };
  for (const row of rows) {
    counts[row.state] += 1;
  }
  return counts;
}

export function countFamilies(rows: readonly FailRow[]): {
  toolchain: number;
  'resource-ceiling': number;
  science: number;
} {
  const out = { toolchain: 0, 'resource-ceiling': 0, science: 0 };
  for (const row of rows) {
    out[row.family] += 1;
  }
  return out;
}

export function countKinds(rows: readonly FailRow[]): Record<FailKind, number> {
  const out: Record<FailKind, number> = {
    compile: 0,
    'silent-zero': 0,
    'bit-pattern': 0,
    interrupted: 0,
  };
  for (const row of rows) {
    out[row.kind] += 1;
  }
  return out;
}

export function mapCloses(rows: readonly FailRow[]): boolean {
  if (!measureMayPrint(PBPK_SUITE_NOW)) return false;
  if (rows.length !== PBPK_SUITE_NOW.fail) return false;
  const fam = countFamilies(rows);
  const kinds = countKinds(rows);
  const states = countStates(rows);
  const names = new Set(rows.map((r) => r.name));
  const nums = new Set(rows.map((r) => r.n));
  return (
    fam.toolchain === 12 &&
    fam['resource-ceiling'] === 7 &&
    fam.science === 0 &&
    kinds.compile === 10 &&
    kinds['silent-zero'] === 1 &&
    kinds['bit-pattern'] === 1 &&
    kinds.interrupted === 7 &&
    states.refused === 10 &&
    states.fabricated === 9 &&
    states.verified === 0 &&
    states.uncertain === 0 &&
    states.unbounded === 0 &&
    names.size === rows.length &&
    nums.size === rows.length
  );
}

if (!mapCloses(PBPK_SUITE_FAILS)) {
  throw new Error(
    'dissertationHonestyMap: the 19 named fails do not close against the dated ledger — refuse to print a defense sentence the parts do not sum to',
  );
}

export const FAIL_STATE_COUNTS = countStates(PBPK_SUITE_FAILS);
export const FAIL_KIND_COUNTS = countKinds(PBPK_SUITE_FAILS);
