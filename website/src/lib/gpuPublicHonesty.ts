/**
 * U2 — GPU public contract 13/13. March 2026. Never in CI.
 *
 * This is the same class as the June 6/6 and the May 251/251: a green
 * count that survived because nobody ran it. The artifact names
 * `souc-linux-x86_64-gpu` / `souc 1.0.0-beta.4`. That is a launcher,
 * not Madaros or lean_single. Do not invent an engine.
 *
 * Instance of MeasurementClaim, not a third kernel.
 */

import {
  assertReachabilityComplete,
  claimFace,
  claimMayPrint,
  claimRefusal,
  type MeasurementClaim,
} from './measurementClaim';

export const GPU_PUBLIC: MeasurementClaim = {
  id: 'gpu-public-contract',
  kind: 'green-count',
  gate: null,
  engine: null,
  reachability: 'WORKFLOW-UNREACHABLE',
  measuredAt: '2026-03-09T21:56:35Z',
  artifact: 'artifacts/omega/gpu_public_contract.v1.json',
  parts: { pass: 13, total: 13, notPublic: 3 },
};

export function gpuPartsClose(c: MeasurementClaim): boolean {
  return c.parts.pass === c.parts.total && c.parts.total === 13 && c.parts.notPublic === 3;
}

export function gpuMayPrint(): boolean {
  return claimMayPrint(GPU_PUBLIC, gpuPartsClose);
}

export function gpuFace(): string {
  return claimFace(
    GPU_PUBLIC,
    gpuPartsClose,
    `${GPU_PUBLIC.parts.pass} pass / ${GPU_PUBLIC.parts.total}`,
  );
}

export function gpuRefusal(): string {
  return claimRefusal(GPU_PUBLIC, 'GPU public contract');
}

export function gpuShortRefusal(): string {
  return '—';
}

export function copyLeaksGpuNumeral(text: string): boolean {
  return /\b13\s*\/\s*13\b/.test(text);
}

assertReachabilityComplete(GPU_PUBLIC, GPU_PUBLIC.id);
