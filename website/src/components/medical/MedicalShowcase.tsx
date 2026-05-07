import { useState } from 'react';
import PrecisionDosingScenario from './PrecisionDosingScenario';
import PolypharmacyScenario from './PolypharmacyScenario';
import CausalEfficacyScenario from './CausalEfficacyScenario';

const SCENARIOS = [
  {
    id: 'precision-dosing',
    title: 'Precision Dosing Guarantee',
    category: 'Pharmacokinetics',
    description: 'Verify therapeutic window safety for narrow-index drugs across a bounded range of patient organ function uncertainty.',
    icon: 'M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10'
  },
  {
    id: 'polypharmacy',
    title: 'Metabolic Interference Check',
    category: 'Polypharmacy Safety',
    description: 'Mathematically prove that co-administration of competitive inhibitors will not breach toxic accumulation thresholds.',
    icon: 'M13 10V3L4 14h7v7l9-11h-7z'
  },
  {
    id: 'causal-efficacy',
    title: 'True Efficacy Verification',
    category: 'Causal Inference',
    description: "Resolve Simpson's Paradox in clinical cohorts by isolating the true causal effect of treatment from observational confounders.",
    icon: 'M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'
  }
];

export default function MedicalShowcase() {
  const [activeScenario, setActiveScenario] = useState(SCENARIOS[0].id);

  const renderActiveScenario = () => {
    switch (activeScenario) {
      case 'precision-dosing':
        return <PrecisionDosingScenario />;
      case 'polypharmacy':
        return <PolypharmacyScenario />;
      case 'causal-efficacy':
        return <CausalEfficacyScenario />;
      default:
        return null;
    }
  };

  return (
    <div className="w-full max-w-7xl mx-auto p-6 flex flex-col gap-8">
      <div className="flex flex-col gap-2 border-l-4 border-[var(--color-accent-blue)] pl-6">
        <h2 className="text-3xl font-bold text-white tracking-tight">Clinical Verification Engine</h2>
        <p className="text-gray-400 max-w-3xl leading-relaxed">
          Sounio guarantees patient safety by lifting biological uncertainty into the type system. Rather than 
          relying on point estimates, our formal methods mathematically prove that treatments remain within 
          therapeutic windows across all potential physiological variations.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Left Navigation: Scenarios */}
        <div className="flex flex-col gap-3">
          {SCENARIOS.map(s => (
            <button
              key={s.id}
              onClick={() => setActiveScenario(s.id)}
              className={`flex flex-col items-start p-5 rounded-xl border transition-all text-left group ${
                activeScenario === s.id 
                  ? 'glass border-[var(--color-accent-blue)] bg-[var(--color-accent-blue)]/10 shadow-[0_0_15px_rgba(59,130,246,0.15)]' 
                  : 'border-white/10 hover:border-white/30 bg-black/40'
              }`}
            >
              <div className="flex items-center gap-3 mb-2 w-full">
                <svg className={`w-5 h-5 ${activeScenario === s.id ? 'text-[var(--color-accent-blue)]' : 'text-gray-500 group-hover:text-gray-300'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d={s.icon} />
                </svg>
                <span className="text-[10px] font-mono text-gray-400 uppercase tracking-widest">{s.category}</span>
              </div>
              <h3 className={`font-semibold mb-2 ${activeScenario === s.id ? 'text-white' : 'text-gray-300'}`}>{s.title}</h3>
              <p className="text-xs text-gray-500 leading-relaxed line-clamp-3">{s.description}</p>
            </button>
          ))}
          
          <div className="mt-4 p-4 rounded-xl bg-black/40 border border-white/5 border-dashed">
            <div className="flex items-center gap-2 mb-2">
              <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
              <span className="text-xs font-mono text-red-400 uppercase">Live Demo Mode</span>
            </div>
            <p className="text-[10px] text-gray-500">
              Values computed via WebAssembly bridge to Sounio native verification engine.
            </p>
          </div>
        </div>

        {/* Right Dashboard Area */}
        <div className="lg:col-span-3">
          <div className="glass border border-white/10 rounded-2xl overflow-hidden shadow-2xl">
            {/* Clinical Header Bar */}
            <div className="bg-black/60 border-b border-white/10 px-6 py-4 flex justify-between items-center">
              <div className="flex items-center gap-4">
                <div className="w-8 h-8 rounded-full bg-[var(--color-accent-blue)]/20 flex items-center justify-center">
                  <svg className="w-4 h-4 text-[var(--color-accent-blue)]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-white">Sounio Clinical Sandbox</h3>
                  <p className="text-[10px] font-mono text-gray-400">Formal Verification Environment</p>
                </div>
              </div>
              <button className="px-4 py-1.5 text-xs font-mono text-white bg-white/5 hover:bg-white/10 border border-white/10 rounded-lg transition-colors flex items-center gap-2">
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export Report
              </button>
            </div>

            {/* Scenario Content */}
            <div className="p-6">
              {renderActiveScenario()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
