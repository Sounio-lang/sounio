/**
 * #1880 wired two leftover dissertation greens into Contracts.
 *
 * Until these exist as MeasurementClaim instances, reachability can
 * only say "not in CI". That is a constant, not a label. The suite
 * (pbpk_suite) is still UNREACHABLE — it hangs off the dead umbrella.
 * These two do not. Do not copy their REACHABLE binding onto the suite.
 *
 * Neither has a remasure that names Madaros or lean_single. The
 * numeral therefore stays refused. The face is allowed to say yes:
 * in CI · .github/workflows/ci.yml:LINE.
 *
 * Lines measured on origin/main after #1880 (12ebda238d). If ci.yml
 * moves the steps, this file must move with them or construction
 * still says a line that is no longer the wire.
 */

import {
  assertReachabilityComplete,
  claimMayPrint,
  claimRefusal,
  type MeasurementClaim,
} from './measurementClaim';

export const CI_WORKFLOW = '.github/workflows/ci.yml';
export const CI_WIRE_PR = 1880;
export const CI_WIRE_SHA = '12ebda238d';
export const CI_WIRED_AT = '2026-08-18T17:41:05Z';

export const DISSERTATION_CONFIDENCE_GATE: MeasurementClaim = {
  id: 'dissertation-confidence-gate',
  kind: 'outcome',
  gate: 'scripts/ci/dissertation_confidence_gate_gate.sh',
  engine: null,
  reachability: 'WORKFLOW-REACHABLE',
  workflow: CI_WORKFLOW,
  line: 110,
  measuredAt: CI_WIRED_AT,
  artifact: CI_WORKFLOW,
  parts: {},
};

export const DISSERTATION_FRONTEND_PARITY: MeasurementClaim = {
  id: 'dissertation-frontend-parity',
  kind: 'outcome',
  gate: 'scripts/ci/dissertation_frontend_parity_gate.sh',
  engine: null,
  reachability: 'WORKFLOW-REACHABLE',
  workflow: CI_WORKFLOW,
  line: 117,
  measuredAt: CI_WIRED_AT,
  artifact: CI_WORKFLOW,
  parts: {},
};

export function wiredPartsClose(_c: MeasurementClaim): boolean {
  return true;
}

export function confidenceGateMayPrint(): boolean {
  return claimMayPrint(DISSERTATION_CONFIDENCE_GATE, wiredPartsClose);
}

export function frontendParityMayPrint(): boolean {
  return claimMayPrint(DISSERTATION_FRONTEND_PARITY, wiredPartsClose);
}

export function confidenceGateRefusal(): string {
  return claimRefusal(DISSERTATION_CONFIDENCE_GATE, 'dissertation confidence gates');
}

export function frontendParityRefusal(): string {
  return claimRefusal(DISSERTATION_FRONTEND_PARITY, 'dissertation frontend parity');
}

assertReachabilityComplete(DISSERTATION_CONFIDENCE_GATE, DISSERTATION_CONFIDENCE_GATE.id);
assertReachabilityComplete(DISSERTATION_FRONTEND_PARITY, DISSERTATION_FRONTEND_PARITY.id);

if (confidenceGateMayPrint() || frontendParityMayPrint()) {
  throw new Error(
    'dissertationWiredHonesty: #1880 names no engine — refuse to print a numeral for a wire',
  );
}
