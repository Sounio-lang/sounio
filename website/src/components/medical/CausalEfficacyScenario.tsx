import { useState } from 'react';

export default function CausalEfficacyScenario() {
  const [adjustAge, setAdjustAge] = useState(false);
  const [adjustSeverity, setAdjustSeverity] = useState(false);
  const [verifying, setVerifying] = useState(false);

  const naiveTreatment = 75;
  const naiveControl = 60;
  const naiveDiff = naiveTreatment - naiveControl;

  const calculateCausal = () => {
    let effect = naiveDiff;
    if (adjustAge) effect -= 8;
    if (adjustSeverity) effect -= 10;
    return effect;
  };

  const trueEffect = calculateCausal();
  const isCausal = trueEffect > 5;
  const isNeutral = trueEffect > -5 && trueEffect <= 5;
  
  const runVerification = () => {
    setVerifying(true);
    setTimeout(() => setVerifying(false), 900);
  };

  return (
    <div className="flex flex-col gap-8 text-white">
      <div className="flex flex-col gap-2">
        <h3 className="text-xl font-bold">Simpson's Paradox & True Efficacy</h3>
        <p className="text-sm text-gray-400 max-w-3xl">
          Observational data often suggests a drug works when healthier patients were simply more likely to receive it. Select confounders to isolate the true treatment effect.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        
        <div className="p-6 rounded-xl border border-white/10 bg-black/60 flex flex-col gap-6">
          <h4 className="text-sm font-semibold uppercase tracking-widest text-gray-300 border-b border-white/10 pb-3">Structural Causal Model</h4>
          
          <div className="flex flex-col gap-4">
            <label className="flex items-center gap-3 cursor-pointer group">
              <input type="checkbox" checked={true} disabled className="w-5 h-5 accent-gray-500 opacity-50" />
              <div className="flex flex-col">
                <span className="text-sm text-gray-400">Treatment Assigned (X)</span>
              </div>
            </label>

            <label className="flex items-center gap-3 cursor-pointer group">
              <input 
                type="checkbox" 
                checked={adjustAge} 
                onChange={(e) => { setAdjustAge(e.target.checked); runVerification(); }}
                className="w-5 h-5 accent-[var(--color-accent-blue)]" 
              />
              <div className="flex flex-col">
                <span className="text-sm text-white">Adjust for Patient Age (C1)</span>
                <span className="text-[10px] font-mono text-gray-400">Younger patients more likely treated.</span>
              </div>
            </label>

            <label className="flex items-center gap-3 cursor-pointer group">
              <input 
                type="checkbox" 
                checked={adjustSeverity} 
                onChange={(e) => { setAdjustSeverity(e.target.checked); runVerification(); }}
                className="w-5 h-5 accent-[var(--color-accent-blue)]" 
              />
              <div className="flex flex-col">
                <span className="text-sm text-white">Adjust for Disease Severity (C2)</span>
                <span className="text-[10px] font-mono text-gray-400">Mild cases biased toward treatment.</span>
              </div>
            </label>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          
          <div className="flex flex-col gap-3 p-5 rounded-xl border border-white/10 bg-black/40">
            <h4 className="text-xs font-semibold uppercase tracking-widest text-gray-400 mb-2">Raw Observational Data</h4>
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">Treatment Survival:</span>
              <span className="font-mono text-white">{naiveTreatment}%</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">Control Survival:</span>
              <span className="font-mono text-white">{naiveControl}%</span>
            </div>
            <div className="h-px w-full bg-white/10 my-1"></div>
            <div className="flex justify-between text-sm font-bold">
              <span className="text-white">Apparent Benefit:</span>
              <span className="font-mono text-green-400">+{naiveDiff}%</span>
            </div>
          </div>

          <div className={`p-6 rounded-xl border flex flex-col justify-center gap-4 transition-all duration-500 flex-1 ${
            verifying 
              ? 'border-white/10 bg-black/40' 
              : isCausal
                ? 'border-[var(--color-accent-green)] bg-[var(--color-accent-green)]/10'
                : isNeutral
                  ? 'border-yellow-500/50 bg-yellow-500/10'
                  : 'border-red-500/50 bg-red-500/10'
          }`}>
            {verifying ? (
              <div className="flex flex-col items-center justify-center h-full">
                <div className="w-8 h-8 rounded-full border-2 border-t-[var(--color-accent-blue)] border-white/10 animate-spin mb-3"></div>
                <p className="font-mono text-xs text-gray-400">Computing backdoor adjustment...</p>
              </div>
            ) : (
              <>
                <h4 className="text-xs font-semibold uppercase tracking-widest text-gray-400">True Causal Effect</h4>
                
                <div className="flex items-end gap-3 mb-2">
                  <span className={`text-4xl font-mono font-bold ${
                    isCausal ? 'text-[var(--color-accent-green)]' : isNeutral ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {trueEffect > 0 ? '+' : ''}{trueEffect}%
                  </span>
                  <span className="text-sm text-gray-400 pb-1">Absolute Risk Reduction</span>
                </div>

                <div className="text-sm">
                  {isCausal ? (
                    <span className="text-gray-200">Treatment remains effective after bias control.</span>
                  ) : isNeutral ? (
                    <span className="text-yellow-100 font-bold">FALSE POSITIVE IDENTIFIED.</span>
                  ) : (
                    <span className="text-red-100 font-bold">HARMFUL EFFECT HIDDEN.</span>
                  )}
                  {!isCausal && (
                    <p className="text-gray-400 text-xs mt-2">
                      The {naiveDiff}% apparent benefit was driven by confounding.
                    </p>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

      </div>
    </div>
  );
}
