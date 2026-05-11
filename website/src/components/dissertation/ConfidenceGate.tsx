import type { PBPKParams, DrugId } from '../../hooks/usePBPK';
import { InfoPopover } from './InfoPopover';

/**
 * Phase J compile-time confidence gate, visualised as a traffic light.
 *
 * The committed Phase J gate
 * (`scripts/ci/kretikos_kaxi_phase_j_gate.sh`) enforces evidence-quality
 * thresholds on the kernel. Here we mirror that contract in the viewer,
 * per drug, since the evidence bands are different:
 *
 *   rapamycin   — Ferron 1997 (n=24, CL CV=58%) + Cordis 2003 IFU
 *                 (coating-thickness CV ~30%). Band: clHep within ±50%,
 *                 higuchiScale within [0.7, 1.3], vdScale within [0.7, 1.4].
 *   semaglutide — Overgaard 2019 (n=72, CL CV=15%) + Carlsson 2020
 *                 (SC absorption F=0.89 ± 0.05, k_a CV=22%). Band:
 *                 clHep within ±40%, higuchiScale (= SC dose multiplier)
 *                 within [0.75, 1.25], vdScale within [0.7, 1.4].
 */

interface DrugBand {
  typicalCL: number;
  clBandFrac: number;        // hard-fail
  clSoftFrac: number;        // warn
  releaseLo: number;
  releaseHi: number;
  releaseSoftLo: number;
  releaseSoftHi: number;
  vdLo: number;
  vdHi: number;
  releaseLabel: string;      // human-readable name for the higuchiScale knob
  citation: string;          // shown in info popover
}

const BANDS: Record<DrugId, DrugBand> = {
  rapamycin: {
    typicalCL: 12.4,
    clBandFrac: 0.5,
    clSoftFrac: 0.3,
    releaseLo: 0.7,
    releaseHi: 1.3,
    releaseSoftLo: 0.85,
    releaseSoftHi: 1.15,
    vdLo: 0.7,
    vdHi: 1.4,
    releaseLabel: 'Stent release scale',
    citation: 'Ferron 1997 (n=24, CV=58%) + Cordis 2003 IFU (coating CV ~30%).',
  },
  semaglutide: {
    typicalCL: 0.077,
    clBandFrac: 0.4,
    clSoftFrac: 0.25,
    releaseLo: 0.75,
    releaseHi: 1.25,
    releaseSoftLo: 0.9,
    releaseSoftHi: 1.1,
    vdLo: 0.7,
    vdHi: 1.4,
    releaseLabel: 'SC depot dose',
    citation: 'Overgaard 2019 (n=72, CL CV=15%) + Carlsson 2020 (F=0.89 ± 0.05, k_a CV=22%).',
  },
};

function checkBand(value: number, anchor: number, frac: number) {
  return Math.abs(value - anchor) / anchor <= frac;
}

export interface GateVerdict {
  status: 'pass' | 'warn' | 'fail';
  reason: string;
}

export function evaluateGate(drug: DrugId, params: PBPKParams): GateVerdict {
  const b = BANDS[drug];
  if (!checkBand(params.clHep, b.typicalCL, b.clBandFrac)) {
    const clearanceName = drug === 'rapamycin' ? 'Hepatic clearance' : 'Proteolytic clearance';
    return {
      status: 'fail',
      reason: `${clearanceName} ${params.clHep.toFixed(drug === 'rapamycin' ? 1 : 3)} L/h exceeds ±${(b.clBandFrac * 100).toFixed(0)}% band around typical (${b.citation}).`,
    };
  }
  if (params.higuchiScale < b.releaseLo || params.higuchiScale > b.releaseHi) {
    return {
      status: 'fail',
      reason: `${b.releaseLabel} ${params.higuchiScale.toFixed(2)}× outside [${b.releaseLo}, ${b.releaseHi}] band (${b.citation}).`,
    };
  }
  if (params.vdScale < b.vdLo || params.vdScale > b.vdHi) {
    return {
      status: 'fail',
      reason: `Body composition scale ${params.vdScale.toFixed(2)}× outside BMI-normal envelope.`,
    };
  }
  // Soft warnings near the band edge.
  if (
    !checkBand(params.clHep, b.typicalCL, b.clSoftFrac) ||
    params.higuchiScale < b.releaseSoftLo ||
    params.higuchiScale > b.releaseSoftHi
  ) {
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

export function ConfidenceGate({ drug, params }: { drug: DrugId; params: PBPKParams }) {
  const verdict = evaluateGate(drug, params);
  const b = BANDS[drug];
  const colors = {
    pass: { bg: '#16a34a', glow: 'rgba(34,197,94,0.5)', label: 'PASS' },
    warn: { bg: '#f59e0b', glow: 'rgba(245,158,11,0.5)', label: 'WARN' },
    fail: { bg: '#dc2626', glow: 'rgba(220,38,38,0.55)', label: 'TRIPPED' },
  };
  const c = colors[verdict.status];

  return (
    <div className="text-sm">
      <div className="flex items-baseline justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <h4 className="text-[var(--color-text-primary)] font-semibold">Phase J confidence gate</h4>
          <InfoPopover ariaLabel="What is the confidence gate?">
            <p className="mb-1.5">
              <strong>What you're seeing.</strong> A compile-time check that
              refuses to run the simulation if the population priors fall
              outside their documented evidence band. For the active drug
              ({drug}), the band is sourced from {b.citation}
            </p>
            <p>
              <strong>Why it matters.</strong> Conventional PBPK tools will
              happily extrapolate past their evidence base. The gate makes
              that <em>impossible</em>: kernels outside the confidence band
              compile to a <code>CONF_REJECT</code> with the failure reason
              attached, before any patient simulation runs. Each drug carries
              its own evidence band — semaglutide's is tighter than
              rapamycin's because Overgaard 2019 had a larger, more uniform
              cohort.
            </p>
          </InfoPopover>
        </div>
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
