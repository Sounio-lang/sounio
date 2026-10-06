/**
 * Honesty kernel for the fifth epistemic state: FABRICATED.
 *
 * Verified, uncertain, and refused already have a face. A diagnostic has a
 * face. Fabricated is the state that looks like a given and is not. The
 * meter (#1797) refuses to print that numeral. This module is the other
 * half: it will show the printed face only as a printed face, never as
 * the actual.
 *
 * Witnesses are strings. The AUC integer 4604219396932172800 exceeds
 * JS 2^53; Number() would fabricate a different integer. This file does
 * not recover IEEE bits of 0.671038 in the browser — that conversion
 * does not match the measured decimal, and claiming it would.
 */

import {
  HISTORICAL_E219_HELP,
  PRE_FIX_PRINT_CAP,
  splitAtPrintCap,
} from './diagnosticHonesty';

export type FabricationKind = 'silent-zero' | 'bit-pattern' | 'truncated';

export type FabricationReading =
  | {
      state: 'fabricated';
      kind: FabricationKind;
      printed: string;
      actual: string;
    }
  | {
      state: 'vacuous';
      kind: FabricationKind;
      reason: 'faces-identical' | 'empty-face';
    };

export function facesAreDistinct(printed: string, actual: string): boolean {
  return printed.length > 0 && actual.length > 0 && printed !== actual;
}

export function readFabrication(
  kind: FabricationKind,
  printed: string,
  actual: string,
): FabricationReading {
  if (printed.length === 0 || actual.length === 0) {
    return { state: 'vacuous', kind, reason: 'empty-face' };
  }
  if (printed === actual) {
    return { state: 'vacuous', kind, reason: 'faces-identical' };
  }
  return { state: 'fabricated', kind, printed, actual };
}

/** #1792 — Madaros printed var=0.000000; lean_single shows ~1e-5. */
export const SILENT_ZERO_PRINTED = '0.000000';
export const SILENT_ZERO_ACTUAL = '~1e-5';

/**
 * #1792 — Madaros printed this decimal as AUC confidence.
 * Documented as IEEE-bit-pattern-as-decimal of lean_single 0.671038
 * in PR #1792 / pbpk28_epistemic_v1.md. Stored as text.
 */
export const BIT_PATTERN_PRINTED = '4604219396932172800';
export const BIT_PATTERN_ACTUAL = '0.671038';

export function truncatedHistoricalE219Help(): FabricationReading {
  const split = splitAtPrintCap(HISTORICAL_E219_HELP, PRE_FIX_PRINT_CAP);
  if (!split.wouldTruncate) {
    return { state: 'vacuous', kind: 'truncated', reason: 'faces-identical' };
  }
  return readFabrication('truncated', split.kept, split.dropped);
}

export const KIND_LABEL: Record<FabricationKind, string> = {
  'silent-zero': 'Silent zero',
  'bit-pattern': 'Bit-pattern ε',
  truncated: 'Truncated diagnostic',
};
