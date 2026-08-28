/**
 * Dated dissertation-suite ledger for /honesty.
 *
 * A 6/6 from June survived two months because the page printed the
 * number and not the date. A later local remasure of the same gate
 * printed 52/53 PASS under lean_single. Both measurements are true.
 * What separates them is the engine. A panel that shows a numeral
 * without the engine is lying by omission — even with a date, even
 * with "not in CI" glued on.
 *
 * The ledger therefore refuses to print unless three things close:
 * the parts sum, the reachability label is present, and the engine
 * is named. Reachability without engine is the same class as an
 * undated 6/6, in a new place.
 *
 * This number never ran in GitHub Actions. The gate hangs off
 * native_v2_cpu_compiler_umbrella_gate.sh, which is itself
 * reachable=no (cursor-2 census 2026-08-18, SHA 465008a76b).
 * Fifteen children inherit that invisibility. Nobody chose this.
 * The intermediate node was never wired; there was nowhere to notice.
 *
 * Source of truth (do not invent a later run):
 *   docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-18.md
 *   author kimi-cli1 · landed #1914 as 7eced5d22d
 *   measured on source-built Madaros at c240e848bf
 *   main at 2026-08-18T18:27Z
 * Supersedes the 19-FAIL photograph (#1818, d0c798e4ed,
 * 2026-08-17T22:21:35Z). That face aged honestly — the date was
 * visible — in under 24 hours. The panel still cannot notice aging.
 *
 * PEND is its own category. This remasure closes 16 + 37 = 53
 * with zero PEND. Do not hide a future PEND inside PASS.
 * #1880 wired confidence + frontend_parity into ci.yml. Those two
 * are REACHABLE instances in dissertationWiredHonesty.ts. This file
 * is still the pbpk_suite instance. Do not copy their binding here.
 * suiteFaceParts refuses through claimMayPrint — the same predicate
 * as U2. A second throw path would be a second render path.
 */

export const MS_HOUR = 3_600_000;
export const MS_DAY = 86_400_000;

export type FailFamily = {
  n: number;
  id: 'toolchain' | 'resource-ceiling' | 'science';
};

import {
  COMPILER_ENGINES,
  REACHABILITIES,
  claimMayPrint,
  isCompilerEngine,
  isReachability,
  reachabilityComplete,
  reachabilityFace,
  type CompilerEngine,
  type MeasurementClaim,
  type Reachability,
  type ReachabilityBinding,
} from './measurementClaim';

export {
  COMPILER_ENGINES,
  REACHABILITIES,
  isCompilerEngine,
  isReachability,
  reachabilityFace,
};
export type { CompilerEngine, Reachability };

/**
 * Face of a reachability binding. `not in CI` is the unreachable
 * gloss. A reachable claim must carry the workflow line or it is
 * incomplete — a label that can only say "no" is a constant.
 */
export function REACHABILITY_FACE_OF(b: ReachabilityBinding): string {
  return reachabilityFace(b);
}

export function suiteAsClaim(m: SuiteMeasure): MeasurementClaim {
  return {
    id: 'pbpk-suite',
    kind: 'outcome',
    gate: m.gate,
    engine: m.engine,
    ...bindingOf(m),
    measuredAt: m.measuredAt,
    artifact: m.doc,
    parts: {
      pass: m.pass,
      fail: m.fail,
      pend: m.pend,
      skip: m.skip,
      unknown: m.unknown,
      registered: m.registered,
    },
  };
}

export function priorAsClaim(m: SuiteMeasure): MeasurementClaim {
  return {
    id: 'pbpk-suite-prior',
    kind: 'outcome',
    gate: m.gate,
    engine: m.prior.engine,
    ...bindingOf(m.prior),
    measuredAt: m.prior.measuredOn,
    artifact: `job ${m.prior.job}`,
    parts: {
      pass: m.prior.pass,
      fail: m.prior.fail,
      pend: m.prior.pend,
      registered: m.registered,
    },
  };
}

export function suitePartsClose(c: MeasurementClaim): boolean {
  return (
    c.parts.pass + c.parts.fail + c.parts.pend + (c.parts.skip || 0) + (c.parts.unknown || 0) ===
    c.parts.registered
  );
}

export type MeasureProvenance = {
  engine: CompilerEngine;
} & ReachabilityBinding;

export type SuiteMeasure = {
  gate: string;
  measuredAt: string;
  sourceSha: string;
  docSha: string;
  author: string;
  doc: string;
  pr: number;
  engine: CompilerEngine;
  reachability: Reachability;
  workflow?: string;
  line?: number;
  registered: number;
  pass: number;
  fail: number;
  pend: number;
  skip: number;
  unknown: number;
  pendName: string;
  families: readonly [FailFamily, FailFamily, FailFamily];
  prior: {
    measuredOn: string;
    job: string;
    sourceSha: string;
    engine: CompilerEngine;
    reachability: Reachability;
    workflow?: string;
    line?: number;
    pass: number;
    fail: number;
    pend: number;
  };
};

export const PBPK_SUITE_NOW: SuiteMeasure = {
  gate: 'scripts/ci/dissertation_pbpk_suite_gate.sh',
  measuredAt: '2026-08-18T18:27:00Z',
  sourceSha: 'c240e848bf',
  docSha: '7eced5d22d',
  author: 'kimi-cli1',
  doc: 'docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-18.md',
  pr: 1914,
  engine: 'Madaros',
  reachability: 'WORKFLOW-UNREACHABLE',
  registered: 53,
  pass: 37,
  fail: 16,
  pend: 0,
  skip: 0,
  unknown: 0,
  pendName: 'none this remasure',
  families: [
    { n: 9, id: 'toolchain' },
    { n: 7, id: 'resource-ceiling' },
    { n: 0, id: 'science' },
  ],
  prior: {
    measuredOn: '2026-08-17T22:21:35Z',
    job: '#1818',
    sourceSha: 'd0c798e4ed',
    engine: 'Madaros',
    reachability: 'WORKFLOW-UNREACHABLE',
    pass: 33,
    fail: 19,
    pend: 1,
  },
};

export function outcomesSum(m: SuiteMeasure): number {
  return m.pass + m.fail + m.pend + m.skip + m.unknown;
}

export function familiesSum(m: SuiteMeasure): number {
  return m.families[0].n + m.families[1].n + m.families[2].n;
}

export function ledgerCloses(m: SuiteMeasure): boolean {
  return outcomesSum(m) === m.registered && familiesSum(m) === m.fail;
}

export function bindingOf(m: {
  reachability: Reachability;
  workflow?: string;
  line?: number;
}): ReachabilityBinding {
  if (m.reachability === 'WORKFLOW-REACHABLE') {
    return {
      reachability: 'WORKFLOW-REACHABLE',
      workflow: m.workflow ?? '',
      line: m.line ?? 0,
    };
  }
  return { reachability: 'WORKFLOW-UNREACHABLE' };
}

export function provenanceComplete(m: {
  engine: string;
  reachability: Reachability;
  workflow?: string;
  line?: number;
}): boolean {
  return isCompilerEngine(m.engine) && reachabilityComplete(bindingOf(m));
}

export function measureMayPrint(m: SuiteMeasure): boolean {
  return (
    ledgerCloses(m) &&
    claimMayPrint(suiteAsClaim(m), suitePartsClose) &&
    claimMayPrint(priorAsClaim(m), suitePartsClose)
  );
}

/**
 * cursor-2 #1874, `docs/audit/CI_GATE_UMBRELLA_CLOSURE_2026-08-18.tsv`.
 * Leftover hangers on the dead umbrella, including the node itself.
 * Direct children stay 15; leftover closure is 25 of the 390
 * mention-orphan upper bound. Not most. Do not recount 468 here.
 */
export const UMBRELLA_LEFTOVER_HANGERS = 25 as const;
export const MENTION_ORPHAN_UPPER_BOUND = 390 as const;
export const UMBRELLA_CLOSURE_DOC =
  'docs/audit/CI_GATE_UMBRELLA_CLOSURE_2026-08-18.tsv';
export const UMBRELLA_CLOSURE_SHA = '0ff0b39764';
export const UMBRELLA_CLOSURE_PR = 1874;

export function umbrellaFace(): string {
  return `dead umbrella · ${UMBRELLA_LEFTOVER_HANGERS} leftover hangers of ${MENTION_ORPHAN_UPPER_BOUND} mention-orphans · not most`;
}

export type SuiteFaceParts = {
  numeral: string;
  gloss: string;
  face: string;
};

/**
 * Canonical face. The numeral is not allowed out without both
 * the engine and the reachability gloss. Parts exist so the large
 * fail count cannot render on its own.
 */
export function suiteFaceParts(m: SuiteMeasure): SuiteFaceParts {
  if (!measureMayPrint(m)) {
    throw new Error(
      'measurementClaim: refuse to print pbpk-suite without reachability and engine',
    );
  }
  const numeral = `${m.fail} FAIL / ${m.registered}`;
  const gloss = `${m.engine} · ${reachabilityFace(bindingOf(m))} · operator remeasure ${m.measuredAt} · #${m.pr}`;
  return { numeral, gloss, face: `${numeral} · ${gloss}` };
}

export function suiteFace(m: SuiteMeasure): string {
  return suiteFaceParts(m).face;
}

export function priorFace(m: SuiteMeasure): string {
  if (!measureMayPrint(m)) {
    throw new Error(
      'dissertationHonestyNow: refuse to print the prior pbpk_suite numeral without reachability and engine',
    );
  }
  return `${m.prior.fail} FAIL / ${m.prior.pass} PASS / ${m.prior.pend} PEND · ${m.prior.engine} · ${reachabilityFace(bindingOf(m.prior))} · ${m.prior.measuredOn} · job ${m.prior.job}`;
}

if (!measureMayPrint(PBPK_SUITE_NOW)) {
  throw new Error(
    'dissertationHonestyNow: pbpk_suite ledger does not close, or is missing reachability or engine — refuse to print the numeral',
  );
}

export function ageMs(measuredAt: string, nowMs: number): number {
  const t = Date.parse(measuredAt);
  if (Number.isNaN(t)) return Number.NaN;
  return nowMs - t;
}

export type AgeBand = 'lt-48h' | 'lt-30d' | 'ge-30d';

export function ageBand(ms: number): AgeBand | 'invalid' {
  if (!Number.isFinite(ms) || ms < 0) return 'invalid';
  if (ms < 48 * MS_HOUR) return 'lt-48h';
  if (ms < 30 * MS_DAY) return 'lt-30d';
  return 'ge-30d';
}

export type FormattedAge = { value: number; unit: 'h' | 'd' };

export function formatAge(ms: number): FormattedAge | null {
  if (!Number.isFinite(ms) || ms < 0) return null;
  if (ms < 48 * MS_HOUR) {
    return { value: Math.floor(ms / MS_HOUR), unit: 'h' };
  }
  return { value: Math.floor(ms / MS_DAY), unit: 'd' };
}

export function familyShare(n: number, fail: number): number {
  if (fail <= 0 || n <= 0) return 0;
  return n / fail;
}
