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
 *   docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md
 *   author grok-cli2 · landed #1818 as 2016efb8e4
 *   measured on source-built Madaros at d0c798e4ed
 *   finished 2026-08-17T22:21:35Z
 *
 * PEND is its own category. It is not a pass. Hiding it would make
 * 19 + 33 look like 53.
 * Do not treat #1874 (wire two leftover greens) as current main.
 */

export const MS_HOUR = 3_600_000;
export const MS_DAY = 86_400_000;

export type FailFamily = {
  n: number;
  id: 'toolchain' | 'resource-ceiling' | 'science';
};

/**
 * Machine label from the cursor-2 reachability census.
 * Face is `not in CI`. Do not print `manual-by-design` — that sounds
 * like a choice. Nobody chose this. The chain is dead.
 */
export const REACHABILITIES = ['WORKFLOW-UNREACHABLE'] as const;
export type Reachability = (typeof REACHABILITIES)[number];

export const REACHABILITY_FACE = {
  'WORKFLOW-UNREACHABLE': 'not in CI',
} as const;

export const COMPILER_ENGINES = ['Madaros', 'lean_single'] as const;
export type CompilerEngine = (typeof COMPILER_ENGINES)[number];

export type MeasureProvenance = {
  engine: CompilerEngine;
  reachability: Reachability;
};

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
    pass: number;
    fail: number;
    pend: number;
  };
};

export const PBPK_SUITE_NOW: SuiteMeasure = {
  gate: 'scripts/ci/dissertation_pbpk_suite_gate.sh',
  measuredAt: '2026-08-17T22:21:35Z',
  sourceSha: 'd0c798e4ed',
  docSha: '2016efb8e4',
  author: 'grok-cli2',
  doc: 'docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md',
  pr: 1818,
  engine: 'Madaros',
  reachability: 'WORKFLOW-UNREACHABLE',
  registered: 53,
  pass: 33,
  fail: 19,
  pend: 1,
  skip: 0,
  unknown: 0,
  pendName: 'pbpk28_semaglutide_clinical',
  families: [
    { n: 12, id: 'toolchain' },
    { n: 7, id: 'resource-ceiling' },
    { n: 0, id: 'science' },
  ],
  prior: {
    measuredOn: '2026-08-16',
    job: '9908',
    sourceSha: '6f2c4e2461',
    engine: 'Madaros',
    reachability: 'WORKFLOW-UNREACHABLE',
    pass: 28,
    fail: 24,
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

export function isCompilerEngine(value: string): value is CompilerEngine {
  return (COMPILER_ENGINES as readonly string[]).includes(value);
}

export function isReachability(value: string): value is Reachability {
  return (REACHABILITIES as readonly string[]).includes(value);
}

export function provenanceComplete(m: MeasureProvenance): boolean {
  return isCompilerEngine(m.engine) && isReachability(m.reachability);
}

export function measureMayPrint(m: SuiteMeasure): boolean {
  return ledgerCloses(m) && provenanceComplete(m) && provenanceComplete(m.prior);
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
 * 19 cannot render on its own.
 */
export function suiteFaceParts(m: SuiteMeasure): SuiteFaceParts {
  if (!measureMayPrint(m)) {
    throw new Error(
      'dissertationHonestyNow: refuse to print a pbpk_suite numeral without reachability and engine',
    );
  }
  const numeral = `${m.fail} FAIL / ${m.registered}`;
  const gloss = `${m.engine} · ${REACHABILITY_FACE[m.reachability]} · operator remeasure ${m.measuredAt} · #${m.pr}`;
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
  return `${m.prior.fail} FAIL / ${m.prior.pass} PASS / ${m.prior.pend} PEND · ${m.prior.engine} · ${REACHABILITY_FACE[m.prior.reachability]} · ${m.prior.measuredOn} · job ${m.prior.job}`;
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
