/**
 * Dated dissertation-suite ledger for /honesty.
 *
 * A 6/6 from June survived two months because the page printed the
 * number and not the date. This module is the opposite: every figure
 * is bound to the measurement that produced it. Editing one count
 * without the others must not ship — the ledger throws.
 *
 * Source of truth (do not invent a later run):
 *   docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md
 *   author grok-cli2 · landed #1818 as 2016efb8e4
 *   measured on source-built Madaros at d0c798e4ed
 *   finished 2026-08-17T22:21:35Z
 *
 * PEND is its own category. It is not a pass. Hiding it would make
 * 19 + 33 look like 53.
 */

export const MS_HOUR = 3_600_000;
export const MS_DAY = 86_400_000;

export type FailFamily = {
  n: number;
  id: 'toolchain' | 'resource-ceiling' | 'science';
};

export type SuiteMeasure = {
  gate: string;
  measuredAt: string;
  sourceSha: string;
  docSha: string;
  author: string;
  doc: string;
  pr: number;
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

if (!ledgerCloses(PBPK_SUITE_NOW)) {
  throw new Error(
    'dissertationHonestyNow: pbpk_suite ledger does not close — refuse to print a total that the parts do not sum to',
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
