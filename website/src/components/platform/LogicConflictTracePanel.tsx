import { useState } from 'react';

export function LogicConflictTracePanel() {
  const [activeLine, setActiveLine] = useState<number | null>(null);

  const errorTrace = [
    {
      line: 42,
      file: 'models/oncology/tumor_growth.sio',
      code: `let efficacy: Reliable<f64> = measure_drug_response()`,
      note: 'Type requires: confidence >= 0.95',
      isSource: true
    },
    {
      line: 118,
      file: 'stdlib/medical/measurement.sio',
      code: `fn measure_drug_response() -> Knowledge<f64> { ... }`,
      note: 'Returns: confidence = 0.92 (due to historical variance)',
      isSource: false
    }
  ];

  return (
    <div className="w-full max-w-4xl mx-auto font-sans">
      <div className="flex flex-col gap-2 mb-6">
        <label className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
          Logic Conflict Traceability
        </label>
        <p className="text-sm text-[var(--color-text-tertiary)]">
          When the prover finds a contradiction, it traces exactly where the conflicting boundaries were established.
        </p>
      </div>

      <div className="glass rounded-xl border border-red-500/20 overflow-hidden bg-[var(--color-surface)]">
        {/* Error Header */}
        <div className="flex items-start gap-4 p-5 bg-red-500/5 border-b border-red-500/20">
          <div className="mt-1 w-6 h-6 rounded-full bg-red-500/20 text-red-400 flex items-center justify-center shrink-0">
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <div>
            <h3 className="text-[0.95rem] font-semibold text-red-400">
              Epistemic Refinement Conflict (E0421)
            </h3>
            <p className="text-sm text-[var(--color-text-secondary)] mt-1">
              Cannot prove that <code className="text-xs bg-red-500/10 px-1 py-0.5 rounded text-red-300">measure_drug_response()</code> satisfies the confidence bounds of <code className="text-xs bg-[var(--color-surface-secondary)] px-1 py-0.5 rounded border border-[var(--glass-border)] text-[var(--color-text-primary)]">Reliable&lt;f64&gt;</code>.
            </p>
          </div>
        </div>

        {/* Trace body */}
        <div className="p-0">
          {errorTrace.map((trace, idx) => (
            <div 
              key={idx}
              className={`relative border-b border-[var(--glass-border)] last:border-0 transition-colors duration-200 cursor-pointer ${
                activeLine === trace.line ? 'bg-[var(--color-surface-hover)]' : ''
              }`}
              onMouseEnter={() => setActiveLine(trace.line)}
              onMouseLeave={() => setActiveLine(null)}
            >
              <div className="flex">
                {/* Line number gutter */}
                <div className="w-12 shrink-0 border-r border-[var(--glass-border)] bg-[var(--color-surface-secondary)]/50 flex flex-col items-end py-4 pr-3 text-xs text-[var(--color-text-tertiary)] font-mono">
                  {trace.line}
                </div>
                
                {/* Code and Note */}
                <div className="flex-1 py-4 pl-4 pr-6 overflow-x-auto">
                  <div className="flex justify-between items-center mb-2 text-xs text-[var(--color-text-tertiary)] font-mono">
                    <span>{trace.file}</span>
                    {trace.isSource && (
                      <span className="text-red-400/80 bg-red-500/10 px-2 py-0.5 rounded">Target</span>
                    )}
                  </div>
                  
                  <pre className="text-[0.85rem] leading-loose text-[var(--color-text-primary)]">
                    <code>{trace.code}</code>
                  </pre>
                  
                  <div className="mt-3 flex items-start gap-2 text-sm">
                    <svg className="w-4 h-4 text-red-400 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span className="text-red-300/90">{trace.note}</span>
                  </div>
                </div>
              </div>

              {/* Connecting line between traces */}
              {idx === 0 && (
                <div className="absolute left-[1.45rem] bottom-0 w-px h-full bg-[var(--glass-border)] z-10 translate-y-1/2 hidden md:block"></div>
              )}
            </div>
          ))}
        </div>

        {/* Action Bar */}
        <div className="bg-[var(--color-surface-secondary)] p-4 flex justify-between items-center border-t border-[var(--glass-border)]">
          <div className="text-xs text-[var(--color-text-tertiary)] flex gap-4">
            <span>Z3 Model: <span className="font-mono text-[var(--color-text-secondary)]">UNSAT</span></span>
            <span>Proof Time: <span className="font-mono text-[var(--color-text-secondary)]">12ms</span></span>
          </div>
          <button className="text-xs font-semibold bg-[var(--color-surface)] border border-[var(--glass-border)] hover:border-[var(--color-accent-gold)] text-[var(--color-text-primary)] px-4 py-2 rounded-lg transition-colors">
            Open in Editor
          </button>
        </div>
      </div>
    </div>
  );
}