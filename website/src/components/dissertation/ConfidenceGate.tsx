import type { PBPKParams } from '../../hooks/usePBPK14';

/**
 * Phase J compile-time confidence gate, visualised as a traffic light.
 *
 * The committed Phase J gate
 * (`scripts/ci/kretikos_kaxi_phase_j_gate.sh`) enforces evidence-quality
 * thresholds on the kernel. Here we mirror that contract in the viewer:
 * given the current patient + dose configuration, would the dissertation's
 * confidence gate pass or trip?
 *
 * Thresholds:
 *  - clHep must remain within ±50% of the typical 12.4 L/h (covers
 *    CYP3A4 fast/poor metaboliser bands documented in Ferron 1997).
 *  - higuchiScale within [0.7, 1.3] (coating-thickness CV in
 *    biomaterial_release.sio is ~15% per Cordis 2003 IFU).
 *  - vdScale within [0.7, 1.4] (BMI-normal envelope).
 */

const TYPICAL_CL = 12.4;

function checkBand(value: number, anchor: number, frac: number) {
  return Math.abs(value - anchor) / anchor <= frac;
}

export interface GateVerdict {
  status: 'pass' | 'warn' | 'fail';
  reason: string;
}

export function evaluateGate(params: PBPKParams): GateVerdict {
  if (!checkBand(params.clHep, TYPICAL_CL, 0.5)) {
    return {
      status: 'fail',
      reason: `Hepatic clearance ${params.clHep.toFixed(1)} L/h exceeds ±50% band around typical (Ferron 1997, CYP3A4 CV=58%).`,
    };
  }
  if (params.higuchiScale < 0.7 || params.higuchiScale > 1.3) {
    return {
      status: 'fail',
      reason: `Stent release scale ${params.higuchiScale.toFixed(2)}× outside Cypher coating-thickness CV (±30%, Cordis 2003 IFU).`,
    };
  }
  if (params.vdScale < 0.7 || params.vdScale > 1.4) {
    return {
      status: 'fail',
      reason: `Body composition scale ${params.vdScale.toFixed(2)}× outside BMI-normal envelope.`,
    };
  }
  // Soft warnings near the band edge.
  if (!checkBand(params.clHep, TYPICAL_CL, 0.3) || params.higuchiScale < 0.85 || params.higuchiScale > 1.15) {
    return {
      status: 'warn',
      reason: 'Patient profile sits near the confidence-gate boundary; widen the GUM cone before clinical interpretation.',
    };
  }
  return {
    status: 'pass',
    reason: 'Population priors well within evidence-quality thresholds; Phase J would compile the kernel.',
  };
}

export function ConfidenceGate({ params }: { params: PBPKParams }) {
  const verdict = evaluateGate(params);
  const colors = {
    pass: { bg: '#16a34a', glow: 'rgba(34,197,94,0.5)', label: 'PASS' },
    warn: { bg: '#f59e0b', glow: 'rgba(245,158,11,0.5)', label: 'WARN' },
    fail: { bg: '#dc2626', glow: 'rgba(220,38,38,0.55)', label: 'TRIPPED' },
  };
  const c = colors[verdict.status];

  return (
    <div className="text-sm">
      <div className="flex items-baseline justify-between mb-2">
        <h4 className="text-[var(--color-text-primary)] font-semibold">Phase J confidence gate</h4>
        <span className="text-[0.7rem] opacity-60">compile-time</span>
      </div>
      <div className="flex items-center gap-3">
        <div
          className="w-9 h-9 rounded-full flex-shrink-0"
          style={{
            backgroundColor: c.bg,
            boxShadow: `0 0 14px ${c.glow}, inset 0 0 5px rgba(0,0,0,0.35)`,
          }}
        />
        <div>
          <div className="font-semibold tracking-wide">{c.label}</div>
          <div className="text-[0.75rem] opacity-80 mt-0.5 leading-snug">{verdict.reason}</div>
        </div>
      </div>
    </div>
  );
}
