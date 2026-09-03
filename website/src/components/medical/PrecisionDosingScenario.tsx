import { useState, useEffect } from 'react';

export default function PrecisionDosingScenario() {
  const [renalMin, setRenalMin] = useState(40);
  const [renalMax, setRenalMax] = useState(60);
  const [dose, setDose] = useState(1000);
  const [verifying, setVerifying] = useState(false);
  const [result, setResult] = useState<null | 'safe' | 'unsafe'>(null);

  const Vd = 49;
  const targetTroughMin = 10;
  const targetTroughMax = 20;

  useEffect(() => {
    setVerifying(true);
    const timer = setTimeout(() => {
      const ke_min = 0.00083 * renalMin + 0.0044;
      const ke_max = 0.00083 * renalMax + 0.0044;
      
      const troughMax = (dose / Vd) * Math.exp(-ke_min * 12);
      const troughMin = (dose / Vd) * Math.exp(-ke_max * 12);
      
      if (troughMin >= targetTroughMin && troughMax <= targetTroughMax) {
        setResult('safe');
      } else {
        setResult('unsafe');
      }
      setVerifying(false);
    }, 800);
    return () => clearTimeout(timer);
  }, [renalMin, renalMax, dose]);

  return (
    <div className="flex flex-col gap-8 text-white">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <h3 className="text-xl font-bold">Vancomycin Precision Dosing</h3>
            <p className="text-sm text-gray-400">
              Verify therapeutic trough levels (10-20 mg/L) under bounded renal function uncertainty.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-white/10 bg-white/5 flex flex-col gap-5">
            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-center">
                <label className="text-xs font-mono text-gray-300 uppercase">Renal Function [CrCl mL/min]</label>
                <span className="text-xs font-mono text-[var(--color-accent-blue)]">
                  Range: {renalMin} - {renalMax}
                </span>
              </div>
              
              <div className="flex gap-4 items-center">
                <div className="flex flex-col w-full">
                  <span className="text-[10px] text-gray-500 mb-1">Lower Bound</span>
                  <input 
                    type="range" min="10" max="120" value={renalMin}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      if (val <= renalMax) setRenalMin(val);
                    }}
                    className="w-full accent-[var(--color-accent-blue)]"
                  />
                </div>
                <div className="flex flex-col w-full">
                  <span className="text-[10px] text-gray-500 mb-1">Upper Bound</span>
                  <input 
                    type="range" min="10" max="120" value={renalMax}
                    onChange={(e) => {
                      const val = parseInt(e.target.value);
                      if (val >= renalMin) setRenalMax(val);
                    }}
                    className="w-full accent-[var(--color-accent-blue)]"
                  />
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-3 pt-4 border-t border-white/10">
              <div className="flex justify-between items-center">
                <label className="text-xs font-mono text-gray-300 uppercase">Dose [mg Q12H]</label>
                <span className="text-xs font-mono text-white">{dose} mg</span>
              </div>
              <input 
                type="range" min="250" max="2000" step="250" value={dose}
                onChange={(e) => setDose(parseInt(e.target.value))}
                className="w-full accent-white"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-end border-b border-white/10 pb-2">
            <h3 className="text-sm font-semibold uppercase tracking-widest text-gray-400">Formal Verification</h3>
            {verifying ? (
              <span className="text-xs font-mono text-[var(--color-accent-blue)] animate-pulse">Computing...</span>
            ) : (
              <span className="text-xs font-mono text-gray-500">Idle</span>
            )}
          </div>

          <div className={`p-6 rounded-xl border flex flex-col items-center justify-center text-center gap-4 transition-all duration-500 min-h-[220px] ${
            verifying 
              ? 'border-white/10 bg-black/40' 
              : result === 'safe'
                ? 'border-[var(--color-accent-green)] bg-[var(--color-accent-green)]/10 shadow-[0_0_30px_rgba(34,197,94,0.15)]'
                : 'border-red-500/50 bg-red-500/10 shadow-[0_0_30px_rgba(239,68,68,0.15)]'
          }`}>
            {verifying ? (
              <>
                <div className="w-12 h-12 rounded-full border-2 border-t-[var(--color-accent-blue)] border-white/10 animate-spin"></div>
                <p className="font-mono text-xs text-gray-400">Evaluating bounded parameter space...</p>
              </>
            ) : result === 'safe' ? (
              <>
                <div className="w-16 h-16 rounded-full bg-[var(--color-accent-green)]/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-[var(--color-accent-green)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xl font-bold text-white mb-1">Safety Guaranteed</h4>
                  <p className="text-sm text-gray-300">
                    Mathematically proven: C_trough remains within [10.0, 20.0] mg/L across all renal clearances.
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xl font-bold text-white mb-1">Invariant Violated</h4>
                  <p className="text-sm text-red-200">
                    Counter-example found: Trough levels drift outside therapeutic window.
                  </p>
                </div>
              </>
            )}
          </div>

          <div className="bg-black/60 rounded-xl p-4 border border-white/10 mt-2">
            <h4 className="text-[10px] font-mono text-gray-500 uppercase mb-2">Sounio Verification Trace</h4>
            <pre className="text-[10px] font-mono text-gray-400 overflow-x-auto whitespace-pre-wrap">
{verifying ? `[Trace running...]` : result === 'safe' ? `[Trace complete]
Proof constraints satisfied:
∀ ke ∈ [f(CrCl_min), f(CrCl_max)]:
  C(12h, ke) >= 10.0 mg/L (True)
  C(12h, ke) <= 20.0 mg/L (True)
Conclusion: PROVEN (Confidence: 1.0)` : `[Trace complete]
Proof failed: Invariant violated.
Conclusion: CONTRAINDICATED`}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}
