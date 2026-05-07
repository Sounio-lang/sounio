import { useState, useMemo } from 'react';
import { useAudience } from '../../lib/useAudience';

interface Measurement {
  label: string;
  value: number;
  uncertainty: number;
  prov: string;
}

const PRESETS: Record<string, { a: Measurement; b: Measurement; operation: string; resultLabel: string }> = {
  bmi: {
    a: { label: 'Weight', value: 75.3, uncertainty: 1.5, prov: 'clinical_scale_001' },
    b: { label: 'Height\u00B2', value: 3.3124, uncertainty: 0.04, prov: 'stadiometer_001' },
    operation: 'divide',
    resultLabel: 'BMI',
  },
  dosing: {
    a: { label: 'Creatinine Clearance', value: 95.0, uncertainty: 8.0, prov: 'jaffe_assay_lab3' },
    b: { label: 'Body Weight', value: 70.0, uncertainty: 2.0, prov: 'ward_scale_B' },
    operation: 'multiply_scaled',
    resultLabel: 'Adjusted CrCl',
  },
  fmri: {
    a: { label: 'BOLD Signal', value: 0.82, uncertainty: 0.12, prov: 'siemens_3T_run04' },
    b: { label: 'Baseline', value: 0.45, uncertainty: 0.05, prov: 'siemens_3T_rest01' },
    operation: 'subtract',
    resultLabel: 'Activation',
  },
};

function gumPropagate(a: Measurement, b: Measurement, op: string) {
  let value: number;
  let relUncA = a.uncertainty / Math.abs(a.value || 1);
  let relUncB = b.uncertainty / Math.abs(b.value || 1);
  let combinedRelUnc: number;

  switch (op) {
    case 'divide':
      value = a.value / b.value;
      combinedRelUnc = Math.sqrt(relUncA ** 2 + relUncB ** 2);
      break;
    case 'multiply_scaled':
      value = (a.value * b.value) / 100;
      combinedRelUnc = Math.sqrt(relUncA ** 2 + relUncB ** 2);
      break;
    case 'subtract':
      value = a.value - b.value;
      const absUnc = Math.sqrt(a.uncertainty ** 2 + b.uncertainty ** 2);
      combinedRelUnc = absUnc / Math.abs(value || 1);
      break;
    default:
      value = a.value * b.value;
      combinedRelUnc = Math.sqrt(relUncA ** 2 + relUncB ** 2);
  }

  const absoluteUnc = Math.abs(value) * combinedRelUnc;
  const epsilon = Math.max(0, Math.min(1, 1 - combinedRelUnc));

  return { value, absoluteUnc, combinedRelUnc, epsilon };
}

export function UncertaintyCalculator() {
  const [preset, setPreset] = useState<keyof typeof PRESETS>('bmi');
  const [aVal, setAVal] = useState(PRESETS.bmi.a.value);
  const [aUnc, setAUnc] = useState(PRESETS.bmi.a.uncertainty);
  const [bVal, setBVal] = useState(PRESETS.bmi.b.value);
  const [bUnc, setBUnc] = useState(PRESETS.bmi.b.uncertainty);
  const [threshold, setThreshold] = useState(0.95);
  const [showCode, setShowCode] = useState(false);
  const { audience } = useAudience();

  const p = PRESETS[preset];

  const switchPreset = (key: keyof typeof PRESETS) => {
    setPreset(key);
    const pr = PRESETS[key];
    setAVal(pr.a.value);
    setAUnc(pr.a.uncertainty);
    setBVal(pr.b.value);
    setBUnc(pr.b.uncertainty);
    setShowCode(false);
  };

  const result = useMemo(() => {
    const a = { ...p.a, value: aVal, uncertainty: aUnc };
    const b = { ...p.b, value: bVal, uncertainty: bUnc };
    return gumPropagate(a, b, p.operation);
  }, [aVal, aUnc, bVal, bUnc, p]);

  const gateResult = result.epsilon >= threshold;
  const epsilonColor = result.epsilon >= 0.95 ? 'text-emerald-400' : result.epsilon >= 0.80 ? 'text-yellow-400' : 'text-red-400';
  const gateBorderColor = gateResult
    ? 'border-emerald-500/40 bg-emerald-500/5'
    : 'border-red-500/40 bg-red-500/5';

  const sounioCode = `let ${p.a.label.toLowerCase().replace(/[^a-z]/g, '_')}: Knowledge<f64> = Knowledge(
    ${aVal.toFixed(1)},
    \u03B5=${(1 - aUnc / Math.abs(aVal || 1)).toFixed(2)},
    prov="${p.a.prov}"
)

let ${p.b.label.toLowerCase().replace(/[^a-z]/g, '_')}: Knowledge<f64> = Knowledge(
    ${bVal.toFixed(1)},
    \u03B5=${(1 - bUnc / Math.abs(bVal || 1)).toFixed(2)},
    prov="${p.b.prov}"
)

let ${p.resultLabel.toLowerCase().replace(/[^a-z]/g, '_')}: Knowledge<f64> = ${p.a.label.toLowerCase().replace(/[^a-z]/g, '_')} ${p.operation === 'divide' ? '/' : p.operation === 'subtract' ? '-' : '*'} ${p.b.label.toLowerCase().replace(/[^a-z]/g, '_')}

if ${p.resultLabel.toLowerCase().replace(/[^a-z]/g, '_')}.\u03B5 >= ${threshold.toFixed(2)} {
    approve(${p.resultLabel.toLowerCase().replace(/[^a-z]/g, '_')})
} else {
    request_remeasurement(${p.resultLabel.toLowerCase().replace(/[^a-z]/g, '_')})
}`;

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[var(--color-bg)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            Try It: Live Uncertainty Propagation
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[68ch]">
            {audience === 'scientist'
              ? 'Change the measurements below and watch how uncertainty flows through the calculation. This is what Sounio does automatically with every computation.'
              : 'Real-time GUM-compliant uncertainty propagation. Change inputs to see \u03B5 and provenance update live.'}
          </p>
        </div>

        {/* Preset selector */}
        <div className="flex justify-center gap-2 mb-8">
          {Object.entries(PRESETS).map(([key]) => (
            <button
              key={key}
              onClick={() => switchPreset(key as keyof typeof PRESETS)}
              className={`px-4 py-2 rounded-full text-sm font-semibold transition-all duration-200 ${
                preset === key
                  ? 'bg-[var(--color-accent-gold)] text-[#10233e] shadow-lg shadow-[var(--color-accent-gold)]/20'
                  : 'bg-[rgba(255,255,255,0.06)] text-[var(--color-text-secondary)] border border-[var(--glass-border)] hover:border-[var(--color-accent-gold)]/40'
              }`}
            >
              {key === 'bmi' ? 'BMI Calculation' : key === 'dosing' ? 'Drug Dosing' : 'fMRI Signal'}
            </button>
          ))}
        </div>

        <div className="max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-6 items-start">
          {/* Input panel */}
          <div className="glass glass-specular rounded-2xl p-6">
            <h3 className="text-sm font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider mb-4">Measurements</h3>

            <div className="grid gap-4">
              <div>
                <label className="text-sm text-[var(--color-text-secondary)] mb-1 block">{p.a.label}</label>
                <div className="flex gap-2">
                  <input type="number" value={aVal} onChange={(e) => setAVal(+e.target.value || 0)} step="0.1"
                    className="flex-1 bg-[rgba(0,0,0,0.3)] border border-[var(--glass-border)] rounded-lg px-3 py-2 text-[var(--color-text-primary)] text-sm font-mono focus:border-[var(--color-accent-gold)]/60 focus:outline-none" />
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-[var(--color-text-tertiary)]">\u00B1</span>
                    <input type="number" value={aUnc} onChange={(e) => setAUnc(+e.target.value || 0)} step="0.1" min="0"
                      className="w-20 bg-[rgba(0,0,0,0.3)] border border-[var(--glass-border)] rounded-lg px-3 py-2 text-[var(--color-text-primary)] text-sm font-mono focus:border-[var(--color-accent-gold)]/60 focus:outline-none" />
                  </div>
                </div>
                <span className="text-xs text-[var(--color-accent-teal)] mt-1 block font-mono">prov: {p.a.prov}</span>
              </div>

              <div>
                <label className="text-sm text-[var(--color-text-secondary)] mb-1 block">{p.b.label}</label>
                <div className="flex gap-2">
                  <input type="number" value={bVal} onChange={(e) => setBVal(+e.target.value || 0)} step="0.1"
                    className="flex-1 bg-[rgba(0,0,0,0.3)] border border-[var(--glass-border)] rounded-lg px-3 py-2 text-[var(--color-text-primary)] text-sm font-mono focus:border-[var(--color-accent-gold)]/60 focus:outline-none" />
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-[var(--color-text-tertiary)]">\u00B1</span>
                    <input type="number" value={bUnc} onChange={(e) => setBUnc(+e.target.value || 0)} step="0.1" min="0"
                      className="w-20 bg-[rgba(0,0,0,0.3)] border border-[var(--glass-border)] rounded-lg px-3 py-2 text-[var(--color-text-primary)] text-sm font-mono focus:border-[var(--color-accent-gold)]/60 focus:outline-none" />
                  </div>
                </div>
                <span className="text-xs text-[var(--color-accent-teal)] mt-1 block font-mono">prov: {p.b.prov}</span>
              </div>

              <div className="mt-2">
                <label className="text-sm text-[var(--color-text-secondary)] mb-1 block">
                  Confidence threshold (\u03B5 \u2265 {threshold.toFixed(2)})
                </label>
                <input type="range" min="0.50" max="0.99" step="0.01" value={threshold}
                  onChange={(e) => setThreshold(+e.target.value)}
                  className="w-full accent-[var(--color-accent-gold)]" />
                <div className="flex justify-between text-xs text-[var(--color-text-tertiary)]">
                  <span>Lenient (0.50)</span>
                  <span>Strict (0.99)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Arrow */}
          <div className="hidden lg:flex items-center justify-center pt-16">
            <svg className="w-10 h-10 text-[var(--color-accent-gold)]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 4.5L21 12m0 0l-7.5 7.5M21 12H3" />
            </svg>
          </div>

          {/* Result panel */}
          <div className="glass glass-specular rounded-2xl overflow-hidden">
            <div className="p-6">
              <h3 className="text-sm font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider mb-4">
                {audience === 'scientist' ? 'What Sounio Computes' : 'Epistemic Output'}
              </h3>

              {/* The result */}
              <div className="bg-[rgba(0,0,0,0.3)] rounded-xl p-5 mb-4 font-mono">
                <div className="text-2xl font-bold text-[var(--color-text-primary)] mb-2">
                  {p.resultLabel} = {result.value.toFixed(2)}{' '}
                  <span className="text-lg text-[var(--color-text-secondary)]">\u00B1 {result.absoluteUnc.toFixed(2)}</span>
                </div>
                <div className="grid gap-1 text-sm">
                  <div>
                    <span className="text-[var(--color-text-tertiary)]">\u03B5 = </span>
                    <span className={`font-semibold ${epsilonColor}`}>{result.epsilon.toFixed(4)}</span>
                  </div>
                  <div>
                    <span className="text-[var(--color-text-tertiary)]">prov: </span>
                    <span className="text-[var(--color-accent-teal)]">{p.a.prov} \u00D7 {p.b.prov}</span>
                  </div>
                </div>
              </div>

              {/* Gate decision */}
              <div className={`rounded-xl p-4 border ${gateBorderColor} transition-all duration-300`}>
                <div className="flex items-center gap-3">
                  {gateResult ? (
                    <svg className="w-6 h-6 text-emerald-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  ) : (
                    <svg className="w-6 h-6 text-red-400 shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                    </svg>
                  )}
                  <div>
                    <div className={`font-semibold text-sm ${gateResult ? 'text-emerald-400' : 'text-red-400'}`}>
                      {gateResult ? 'GATE PASSED' : 'GATE FAILED'}: \u03B5 {gateResult ? '\u2265' : '<'} {threshold.toFixed(2)}
                    </div>
                    <div className="text-xs text-[var(--color-text-tertiary)] mt-0.5">
                      {gateResult
                        ? audience === 'scientist'
                          ? 'Confidence is high enough to proceed with this result.'
                          : `Validated<${p.resultLabel}> \u2014 safe for downstream use.`
                        : audience === 'scientist'
                          ? 'Confidence too low. Sounio would request remeasurement before proceeding.'
                          : `Knowledge<${p.resultLabel}> blocked \u2014 does not meet \u03B5 threshold.`}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Show code toggle */}
            <div className="border-t border-[var(--glass-border)]">
              <button onClick={() => setShowCode(!showCode)}
                className="w-full px-6 py-3 text-sm font-semibold text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] transition-colors flex items-center justify-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={showCode ? "M19 9l-7 7-7-7" : "M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5"} />
                </svg>
                {showCode ? 'Hide Sounio code' : 'Show how this was computed'}
              </button>
              {showCode && (
                <pre className="px-6 pb-6 text-xs leading-relaxed text-emerald-400/90 font-mono overflow-x-auto">
                  {sounioCode}
                </pre>
              )}
            </div>
          </div>
        </div>

        {audience === 'scientist' && (
          <p className="text-center text-sm text-[var(--color-text-tertiary)] mt-8 max-w-xl mx-auto">
            Try increasing the uncertainty on either measurement and watch the confidence (\u03B5) drop.
            When it falls below your threshold, the system refuses to approve the result.
            This is GUM-compliant uncertainty propagation, enforced automatically.
          </p>
        )}
      </div>
    </section>
  );
}
