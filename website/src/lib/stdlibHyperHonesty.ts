/**
 * U3 — stdlib hyper lanes 7/7. May 2026. Never in CI.
 *
 * Same morning as the 251/251 reliability JSON (15:35 vs 15:38).
 * `scripts/stdlib/stdlib_hyper_execution_gate.sh` is not under
 * scripts/ci/ and is not named in .github/. Zero workflow matches.
 * The JSON names no Madaros and no lean_single. Do not invent an
 * engine. A sync is not a remasure.
 *
 * Instance of MeasurementClaim, not a fourth kernel. If this
 * numeral could not be a claim, the kernel would be the wrong
 * shape — it is not. Same green-count as U1 and U2.
 */

import {
  assertReachabilityComplete,
  claimFace,
  claimMayPrint,
  claimRefusal,
  type MeasurementClaim,
} from './measurementClaim';

export const STDLIB_HYPER: MeasurementClaim = {
  id: 'stdlib-hyper-lanes',
  kind: 'green-count',
  gate: 'scripts/stdlib/stdlib_hyper_execution_gate.sh',
  engine: null,
  reachability: 'WORKFLOW-UNREACHABLE',
  measuredAt: '2026-05-12T15:35:29Z',
  artifact: 'artifacts/stdlib/stdlib_hyper_execution_status.v1.json',
  parts: { pass: 7, fail: 0, skip: 0, total: 7 },
};

export function hyperPartsClose(c: MeasurementClaim): boolean {
  return (
    c.parts.pass + c.parts.fail + c.parts.skip === c.parts.total &&
    c.parts.total === 7
  );
}

export function hyperMayPrint(): boolean {
  return claimMayPrint(STDLIB_HYPER, hyperPartsClose);
}

export function hyperFace(): string {
  return claimFace(
    STDLIB_HYPER,
    hyperPartsClose,
    `${STDLIB_HYPER.parts.pass} pass / ${STDLIB_HYPER.parts.total}`,
  );
}

export function hyperRefusal(): string {
  return claimRefusal(STDLIB_HYPER, 'stdlib hyper lanes');
}

export function hyperShortRefusal(): string {
  return '—';
}

export function copyLeaksHyperNumeral(text: string): boolean {
  return /\b7\s*\/\s*7\b/.test(text);
}

assertReachabilityComplete(STDLIB_HYPER, STDLIB_HYPER.id);
