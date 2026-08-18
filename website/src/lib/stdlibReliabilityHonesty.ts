/**
 * Homepage 251/251 is the twin of the June 6/6.
 *
 * The committed artifact is 2026-05-12. `npm run sync:artifacts` on
 * 2026-08-17 copied it into artifactStatus.ts. That is not a remasure.
 * The script lives at scripts/stdlib_reliability_gate.sh → scripts/dev/,
 * not under scripts/ci/, and is not named in .github/. It is outside
 * the cursor-2 468-gate census. Nobody chose this. There was nowhere
 * to notice.
 *
 * The JSON names `bin/souc` 1.0.0-beta.5 on the science-pipeline arm.
 * That is not Madaros and not lean_single. Do not invent an engine.
 * Same rule as pbpk_suite: no engine and no reachability, no numeral.
 *
 * This file is the kernel. Do not edit artifactStatus.ts — the next
 * sync will overwrite it. Every surface that used to print 251/0
 * must come through here.
 */

import {
  REACHABILITY_FACE,
  isCompilerEngine,
  isReachability,
  type CompilerEngine,
  type Reachability,
} from './dissertationHonestyNow';

export type ReliabilityMeasure = {
  gate: string;
  measuredAt: string;
  artifact: string;
  syncAt: string;
  pass: number;
  fail: number;
  skip: number;
  total: number;
  /**
   * Null until a remasure names Madaros or lean_single.
   * `bin/souc` is a launcher, not an engine.
   */
  engine: CompilerEngine | null;
  reachability: Reachability;
  namedLauncher: string;
};

export const STDLIB_RELIABILITY: ReliabilityMeasure = {
  gate: 'scripts/stdlib_reliability_gate.sh',
  measuredAt: '2026-05-12T15:38:17Z',
  artifact: 'artifacts/stdlib/stdlib_reliability_status.v1.json',
  syncAt: '2026-08-17T02:17:14.465Z',
  pass: 251,
  fail: 0,
  skip: 0,
  total: 251,
  engine: null,
  reachability: 'WORKFLOW-UNREACHABLE',
  namedLauncher: 'bin/souc 1.0.0-beta.5',
};

export function reliabilityLedgerCloses(m: ReliabilityMeasure): boolean {
  return m.pass + m.fail + m.skip === m.total && m.total > 0;
}

export function reliabilityProvenanceComplete(m: ReliabilityMeasure): boolean {
  return m.engine !== null && isCompilerEngine(m.engine) && isReachability(m.reachability);
}

export function reliabilityMayPrint(m: ReliabilityMeasure = STDLIB_RELIABILITY): boolean {
  return reliabilityLedgerCloses(m) && reliabilityProvenanceComplete(m);
}

/**
 * Canonical face. Throws if engine or reachability is missing.
 * Callers that want a fallback must use reliabilityRefusal().
 */
export function reliabilityFace(m: ReliabilityMeasure = STDLIB_RELIABILITY): string {
  if (!reliabilityMayPrint(m) || m.engine === null) {
    throw new Error(
      'stdlibReliabilityHonesty: refuse to print the reliability numeral without reachability and engine',
    );
  }
  return `${m.pass} pass / ${m.total} · ${m.engine} · ${REACHABILITY_FACE[m.reachability]} · artifact ${m.measuredAt} · ${m.artifact}`;
}

/**
 * What a surface may show when the numeral is forbidden.
 * Must not contain pass, fail, or total counts.
 */
export function reliabilityRefusal(m: ReliabilityMeasure = STDLIB_RELIABILITY): string {
  return `stdlib reliability · ${REACHABILITY_FACE[m.reachability]} · artifact ${m.measuredAt.slice(0, 10)} · engine unnamed`;
}

export function reliabilityShortRefusal(): string {
  return '—';
}

/**
 * True when copy would leak the bound counts (the generated
 * honestStatus line "251 / 251 stdlib reliability tests pass").
 */
export function copyLeaksReliabilityNumeral(text: string): boolean {
  const { pass, total } = STDLIB_RELIABILITY;
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
