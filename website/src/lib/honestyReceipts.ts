/**
 * Receipts for every numeral /honesty is allowed to show.
 *
 * A panel that asserts 16, or 37, or var=0.000000 without pointing at
 * the measurement that produced it is the same class as an undated
 * 6/6 — only this time the lie is in HTML. This module is the index.
 * If a figure has no receipt here, it does not belong on the page.
 */

export type Receipt = {
  id: string;
  pr: number;
  sha: string;
  href: string;
  doc?: string;
  docHref?: string;
  measuredAt?: string;
};

const GH = 'https://github.com/Sounio-lang/sounio';

export const RECEIPT = {
  pbpkSuite: {
    id: 'pbpk-suite',
    pr: 1914,
    sha: '7eced5d22d',
    href: `${GH}/pull/1914`,
    doc: 'docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-18.md',
    docHref: `${GH}/blob/7eced5d22d/docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-18.md`,
    measuredAt: '2026-08-18T18:27:00Z',
  },
  /**
   * 24 FAIL / 28 PASS / 1 PEND on 2026-08-16 job 9908.
   * The remasure names a triage file that is not in git; the counts
   * live in the remasure's own before/after table. Link that table.
   */
  pbpkSuitePrior: {
    id: 'pbpk-suite-prior',
    pr: 1818,
    sha: '2016efb8e4',
    href: `${GH}/pull/1818`,
    doc: 'docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md',
    docHref: `${GH}/blob/2016efb8e4/docs/audit/DISSERTATION_PBPK_SUITE_REMEASURE_2026-08-17.md`,
    measuredAt: '2026-08-16',
  },
  truncatedDiagnostics: {
    id: 'truncated-diagnostics',
    pr: 1794,
    sha: 'f6d2188d46',
    href: `${GH}/pull/1794`,
    doc: 'docs/audit/LONG_STRING_LITERAL_DIAGNOSTIC_CENSUS_2026-08-17.md',
    docHref: `${GH}/blob/f6d2188d46/docs/audit/LONG_STRING_LITERAL_DIAGNOSTIC_CENSUS_2026-08-17.md`,
  },
  silentZero: {
    id: 'silent-zero',
    pr: 1792,
    sha: 'a62b1da29c',
    href: `${GH}/pull/1792`,
    doc: 'docs/dissertation/results/pbpk28_epistemic_v1.md',
    docHref: `${GH}/blob/a62b1da29c/docs/dissertation/results/pbpk28_epistemic_v1.md`,
  },
} as const satisfies Record<string, Receipt>;

export type ReceiptKey = keyof typeof RECEIPT;

export function receiptHref(key: ReceiptKey, prefer: 'pr' | 'doc' = 'pr'): string {
  const r = RECEIPT[key];
  if (prefer === 'doc' && r.docHref) return r.docHref;
  return r.href;
}

export function receiptLabel(key: ReceiptKey): string {
  return `#${RECEIPT[key].pr}`;
}
