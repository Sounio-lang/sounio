import { useState, useEffect } from 'react';

export function ProverStatusBadge() {
  const [status, setStatus] = useState<'idle' | 'evaluating' | 'verified' | 'failed'>('idle');

  // Cycle through states for the demo
  useEffect(() => {
    let timeout: NodeJS.Timeout;
    
    const runCycle = () => {
      setStatus('evaluating');
      timeout = setTimeout(() => {
        setStatus('verified');
        timeout = setTimeout(() => {
          setStatus('idle');
          timeout = setTimeout(runCycle, 2000);
        }, 5000);
      }, 1500);
    };

    timeout = setTimeout(runCycle, 1000);
    return () => clearTimeout(timeout);
  }, []);

  return (
    <div className="flex flex-col gap-6 w-full font-sans">
      <div className="flex flex-col gap-2">
        <label className="text-sm font-semibold text-[var(--color-text-secondary)] uppercase tracking-wider">
          Compile-Time Refinement Engine (Z3)
        </label>
        <p className="text-sm text-[var(--color-text-tertiary)] mb-2">
          Mathematical guarantee of confidence bounds across entire system execution paths.
        </p>
      </div>

      <div className="relative rounded-xl border border-[var(--glass-border)] bg-[var(--color-surface)] overflow-hidden flex flex-col md:flex-row shadow-lg">
        {/* Code side */}
        <div className="p-5 flex-1 font-mono text-sm leading-relaxed overflow-x-auto bg-[#0d1117] border-r border-[var(--glass-border)]">
          <div className="flex text-[#8b949e] mb-2">// Refinement type constraint</div>
          <div><span className="text-[#ff7b72]">type</span> <span className="text-[#79c0ff]">Reliable</span>&lt;<span className="text-[#a5d6ff]">T</span>&gt; = <span className="text-[#d2a8ff]">Knowledge</span>[T]</div>
          <div className="pl-4 mb-4"><span className="text-[#ff7b72]">where</span> <span className="text-[#79c0ff]">confidence</span> &gt;= <span className="text-[#a5d6ff]">0.95</span></div>
          
          <div className="flex text-[#8b949e] mb-2">// Z3 Prover checks this pipeline mathematically</div>
          <div><span className="text-[#ff7b72]">fn</span> <span className="text-[#d2a8ff]">analyze_dosing</span>(patient_data: <span className="text-[#d2a8ff]">Knowledge</span>[<span className="text-[#79c0ff]">f64</span>]) -&gt; <span className="text-[#79c0ff]">Reliable</span>&lt;<span className="text-[#79c0ff]">f64</span>&gt; {'{'}</div>
          <div className="pl-4 relative">
            <div className={`absolute -left-2 top-0 bottom-0 w-1 transition-colors duration-500 ${
              status === 'evaluating' ? 'bg-[var(--color-accent-orange)] animate-pulse' :
              status === 'verified' ? 'bg-[var(--color-accent-green)]' : 'bg-transparent'
            }`}></div>
            <span className="text-[#ff7b72]">let</span> <span className="text-[#79c0ff]">clearance</span> = patient_data * <span className="text-[#a5d6ff]">0.4</span><br/>
            <span className="text-[#ff7b72]">let</span> <span className="text-[#79c0ff]">dosing</span> = compute_auc(clearance)<br/>
            <span className="text-[#ff7b72]">return</span> dosing <span className="text-[#8b949e]"> // Must prove: dosing.ε &gt;= 0.95</span>
          </div>
          <div>{'}'}</div>
        </div>

        {/* Status side */}
        <div className="w-full md:w-64 p-5 flex flex-col justify-center gap-4 bg-[var(--color-surface-secondary)]">
          <div className="text-xs font-semibold text-[var(--color-text-tertiary)] uppercase tracking-wider">
            SMT Solver Status
          </div>
          
          <div className={`
            flex items-center gap-3 p-3 rounded-lg border transition-all duration-300
            ${status === 'idle' ? 'bg-[var(--color-surface)] border-[var(--glass-border)] text-[var(--color-text-secondary)]' : ''}
            ${status === 'evaluating' ? 'bg-[var(--color-accent-orange)]/10 border-[var(--color-accent-orange)]/30 text-[var(--color-accent-orange)]' : ''}
            ${status === 'verified' ? 'bg-[var(--color-accent-green)]/10 border-[var(--color-accent-green)]/30 text-[var(--color-accent-green)] shadow-[0_0_15px_rgba(46,160,67,0.15)]' : ''}
          `}>
            <div className="shrink-0">
              {status === 'idle' && (
                <svg className="w-5 h-5 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
              {status === 'evaluating' && (
                <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              )}
              {status === 'verified' && (
                <svg className="w-5 h-5 drop-shadow-[0_0_8px_rgba(46,160,67,0.5)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                </svg>
              )}
            </div>
            <div className="font-mono text-sm font-medium">
              {status === 'idle' && 'Awaiting AST'}
              {status === 'evaluating' && 'Proving Bounds...'}
              {status === 'verified' && 'Q.E.D. Verified'}
            </div>
          </div>

          <div className="flex flex-col gap-2 mt-2">
            <div className="flex justify-between text-xs text-[var(--color-text-tertiary)]">
              <span>Constraints:</span>
              <span className="font-mono">1</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-tertiary)]">
              <span>Variables:</span>
              <span className="font-mono">4</span>
            </div>
            <div className="flex justify-between text-xs text-[var(--color-text-tertiary)]">
              <span>Time:</span>
              <span className="font-mono">{status === 'verified' ? '4.1ms' : '--'}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}