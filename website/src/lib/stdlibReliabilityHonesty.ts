/**
 * U1 — stdlib reliability 251/251. May 2026. Never in CI.
 *
 * Homepage 251/251 is the twin of the June 6/6. The committed
 * artifact is 2026-05-12. `npm run sync:artifacts` on 2026-08-17
 * copied it into artifactStatus.ts. That is not a remasure.
 * The script lives at scripts/stdlib_reliability_gate.sh →
 * scripts/dev/, not under scripts/ci/, and is not named in
 * .github/. It is outside the cursor-2 468-gate census. Nobody
 * chose this. There was nowhere to notice.
 *
 * The JSON names `bin/souc` 1.0.0-beta.5 on the science-pipeline
 * arm. That is a launcher, not Madaros or lean_single. Do not
 * invent an engine.
 *
 * Instance of MeasurementClaim, not a second kernel. Do not edit
 * artifactStatus.ts — the next sync will overwrite it. Every
 * surface that used to print 251/0 must come through here.
 */

import {
  claimFace,
  claimMayPrint,
  claimRefusal,
  type MeasurementClaim,
} from './measurementClaim';

export const STDLIB_RELIABILITY: MeasurementClaim = {
  id: 'stdlib-reliability',
  kind: 'green-count',
  gate: 'scripts/stdlib_reliability_gate.sh',
  engine: null,
  reachability: 'WORKFLOW-UNREACHABLE',
  measuredAt: '2026-05-12T15:38:17Z',
  artifact: 'artifacts/stdlib/stdlib_reliability_status.v1.json',
  parts: { pass: 251, fail: 0, skip: 0, total: 251 },
};

/** Sync timestamp and launcher name are not print conditions. */
export const STDLIB_RELIABILITY_META = {
  syncAt: '2026-08-17T02:17:14.465Z',
  namedLauncher: 'bin/souc 1.0.0-beta.5',
};

export function reliabilityPartsClose(c: MeasurementClaim): boolean {
  return (
    c.parts.pass + c.parts.fail + c.parts.skip === c.parts.total &&
    c.parts.total > 0
  );
}

export function reliabilityMayPrint(m: MeasurementClaim = STDLIB_RELIABILITY): boolean {
  return claimMayPrint(m, reliabilityPartsClose);
}

/**
 * Canonical face. Throws if engine or reachability is missing.
 * Callers that want a fallback must use reliabilityRefusal().
 */
export function reliabilityFace(m: MeasurementClaim = STDLIB_RELIABILITY): string {
  return claimFace(
    m,
    reliabilityPartsClose,
    `${m.parts.pass} pass / ${m.parts.total}`,
  );
}

/**
 * What a surface may show when the numeral is forbidden.
 * Must not contain pass, fail, or total counts.
 */
export function reliabilityRefusal(m: MeasurementClaim = STDLIB_RELIABILITY): string {
  return claimRefusal(m, 'stdlib reliability');
}

export function reliabilityShortRefusal(): string {
  return '—';
}

/**
 * True when copy would leak the bound counts (the generated
 * honestStatus line "251 / 251 stdlib reliability tests pass").
 */
export function copyLeaksReliabilityNumeral(text: string): boolean {
  const { pass, total } = STDLIB_RELIABILITY.parts;
  const passRe = new RegExp(`\\b${pass}\\b`);
  const totalRe = new RegExp(`\\b${total}\\b`);
  return passRe.test(text) && totalRe.test(text);
}

export function filterReliabilityLeaks<T extends { title: string; detail: string }>(
  items: readonly T[],
): T[] {
  return items.filter(
    (item) => !copyLeaksReliabilityNumeral(item.title) && !copyLeaksReliabilityNumeral(item.detail),
  );
}
