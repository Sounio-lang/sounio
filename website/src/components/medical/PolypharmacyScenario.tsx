import { useState } from 'react';

const DRUGS = [
  { id: 'warfarin', name: 'Warfarin', type: 'Substrate', pathway: 'CYP2C9', risk: 'High', index: 'Narrow' },
  { id: 'simvastatin', name: 'Simvastatin', type: 'Substrate', pathway: 'CYP3A4', risk: 'Moderate', index: 'Standard' },
  { id: 'amiodarone', name: 'Amiodarone', type: 'Inhibitor', pathway: 'CYP2C9', risk: 'High', strength: 0.8 },
  { id: 'clarithromycin', name: 'Clarithromycin', type: 'Inhibitor', pathway: 'CYP3A4', risk: 'High', strength: 0.9 },
  { id: 'rifampin', name: 'Rifampin', type: 'Inducer', pathway: 'CYP3A4', risk: 'High', strength: -0.7 },
  { id: 'acetaminophen', name: 'Acetaminophen', type: 'Substrate', pathway: 'CYP2E1', risk: 'Low', index: 'Standard' },
];

export default function PolypharmacyScenario() {
  const [substrateId, setSubstrateId] = useState('warfarin');
  const [interactorId, setInteractorId] = useState('amiodarone');
  const [verifying, setVerifying] = useState(false);
  
  const substrate = DRUGS.find(d => d.id === substrateId)!;
  const interactor = DRUGS.find(d => d.id === interactorId)!;

  const hasConflict = substrate.pathway === interactor.pathway && interactor.type !== 'Substrate';
  const severity = hasConflict && substrate.index === 'Narrow' ? 'Critical' : hasConflict ? 'Warning' : 'Safe';

  const runVerification = () => {
    setVerifying(true);
    setTimeout(() => setVerifying(false), 1200);
  };

  return (
    <div className="flex flex-col gap-8 text-white">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <h3 className="text-xl font-bold">Metabolic Interference Check</h3>
            <p className="text-sm text-gray-400">
              Prove co-administration safety by modeling competitive inhibition across cytochrome P450 pathways.
            </p>
          </div>

          <div className="p-5 rounded-xl border border-white/10 bg-white/5 flex flex-col gap-5">
            <div className="flex flex-col gap-3">
              <label className="text-xs font-mono text-gray-300 uppercase">Primary Medication (Substrate)</label>
              <select 
                className="w-full bg-black/60 border border-white/20 rounded-lg p-3 text-white appearance-none outline-none focus:border-[var(--color-accent-blue)]"
                value={substrateId}
                onChange={(e) => { setSubstrateId(e.target.value); setVerifying(false); }}
              >
                {DRUGS.filter(d => d.type === 'Substrate').map(d => (
                  <option key={d.id} value={d.id}>{d.name} ({d.pathway})</option>
                ))}
              </select>
              <div className="flex justify-between">
                <span className="text-[10px] text-gray-500 font-mono">Index: {substrate.index}</span>
                <span className="text-[10px] text-[var(--color-accent-blue)] font-mono">{substrate.pathway}</span>
              </div>
            </div>

            <div className="h-px w-full bg-white/10"></div>

            <div className="flex flex-col gap-3">
              <label className="text-xs font-mono text-gray-300 uppercase">Concomitant Medication</label>
              <select 
                className="w-full bg-black/60 border border-white/20 rounded-lg p-3 text-white appearance-none outline-none focus:border-[var(--color-accent-blue)]"
                value={interactorId}
                onChange={(e) => { setInteractorId(e.target.value); setVerifying(false); }}
              >
                {DRUGS.filter(d => d.id !== substrateId).map(d => (
                  <option key={d.id} value={d.id}>{d.name} ({d.type})</option>
                ))}
              </select>
            </div>

            <button 
              onClick={runVerification}
              disabled={verifying}
              className="mt-4 w-full py-3 bg-[var(--color-accent-blue)] hover:bg-blue-600 text-black font-bold rounded-lg transition-colors font-mono uppercase text-xs"
            >
              {verifying ? 'Analyzing...' : 'Verify Interaction'}
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <div className="flex justify-between items-end border-b border-white/10 pb-2">
            <h3 className="text-sm font-semibold uppercase tracking-widest text-gray-400">Interaction Proof</h3>
          </div>

          <div className={`p-6 rounded-xl border flex flex-col items-center justify-center text-center gap-4 transition-all duration-500 min-h-[280px] ${
            verifying 
              ? 'border-white/10 bg-black/40' 
              : severity === 'Safe'
                ? 'border-[var(--color-accent-green)] bg-[var(--color-accent-green)]/10'
                : severity === 'Critical'
                  ? 'border-red-500/50 bg-red-500/10'
                  : 'border-yellow-500/50 bg-yellow-500/10'
          }`}>
            {verifying ? (
              <>
                <div className="w-12 h-12 rounded-full border-2 border-t-[var(--color-accent-blue)] border-white/10 animate-spin"></div>
                <p className="font-mono text-xs text-gray-400">Solving competitive enzyme kinetics...</p>
              </>
            ) : severity === 'Safe' ? (
              <>
                <div className="w-16 h-16 rounded-full bg-[var(--color-accent-green)]/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-[var(--color-accent-green)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xl font-bold text-white mb-1">Safe to Co-administer</h4>
                  <p className="text-sm text-gray-300">Independent clearance pathways. No toxic accumulation risk.</p>
                </div>
              </>
            ) : severity === 'Critical' ? (
              <>
                <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xl font-bold text-white mb-1">CRITICAL CONTRAINDICATION</h4>
                  <p className="text-sm text-red-200">{interactor.name} strongly inhibits {substrate.pathway}. Severe toxicity risk for {substrate.name}.</p>
                </div>
              </>
            ) : (
              <>
                <div className="w-16 h-16 rounded-full bg-yellow-500/20 flex items-center justify-center">
                  <svg className="w-8 h-8 text-yellow-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-xl font-bold text-white mb-1">Dose Adjustment Required</h4>
                  <p className="text-sm text-yellow-200">Shared pathway. Clearance will be altered.</p>
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
