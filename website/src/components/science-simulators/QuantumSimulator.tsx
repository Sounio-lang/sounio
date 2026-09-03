import { useState, useEffect, useMemo } from 'react';
import { SimulatorShell } from './SimulatorShell';

export function QuantumSimulator() {
  const [noise, setNoise] = useState(10);
  const [isHalted, setIsHalted] = useState(false);
  const [logLines, setLogLines] = useState<{ text: string; type: 'info' | 'warn' | 'error' | 'success' }[]>([]);

  const MAX_STEPS = 50;
  const TARGET_ENERGY = -1.14; // e.g. Hartree
  const START_ENERGY = 0.5;
  const VARIANCE_THRESHOLD = 0.08;

  const optimizationData = useMemo(() => {
    const steps = [];
    let movingVariance = 0;
    let haltedStep = -1;
    
    // Fixed seed
    const pseudoRandom = (i: number) => {
      const n = Math.sin(i * 12.9898) * 43758.5453;
      return (n - Math.floor(n)) * 2 - 1; // -1 to 1
    };

    let prevEnergies: number[] = [];

    for (let step = 0; step <= MAX_STEPS; step++) {
      // Ideal exponential decay to target
      const idealE = TARGET_ENERGY + (START_ENERGY - TARGET_ENERGY) * Math.exp(-0.12 * step);
      
      // Noise introduces variance. Higher noise -> larger random jumps
      // The optimizer path gets corrupted.
      const noiseAmplitude = (noise / 100) * 1.5;
      const actualE = idealE + pseudoRandom(step) * noiseAmplitude;

      prevEnergies.push(actualE);
      if (prevEnergies.length > 5) prevEnergies.shift();

      // Calculate recent variance
      if (prevEnergies.length === 5) {
        const mean = prevEnergies.reduce((a,b) => a+b, 0) / 5;
        movingVariance = prevEnergies.reduce((a,b) => a + Math.pow(b - mean, 2), 0) / 5;
      }

      if (movingVariance > VARIANCE_THRESHOLD && haltedStep === -1 && step > 10) {
        haltedStep = step;
      }

      steps.push({
        step,
        ideal: idealE,
        actual: actualE,
        variance: movingVariance
      });
    }

    return { steps, haltedStep, finalVariance: movingVariance };
  }, [noise]);

  useEffect(() => {
    const halted = optimizationData.haltedStep !== -1;
    setIsHalted(halted);

    const lines: { text: string; type: 'info' | 'warn' | 'error' | 'success' }[] = [
      { text: `Initializing VQE circuit optimization loop...`, type: 'info' },
      { text: `Simulated Hardware Shot Noise: ${noise}%`, type: 'info' },
    ];

    if (halted) {
      lines.push({ text: `WARN: Energy expectation variance exceeded ${VARIANCE_THRESHOLD} at step ${optimizationData.haltedStep}.`, type: 'warn' });
      lines.push({ text: `FATAL: Optimizer divergence detected. Objective landscape corrupted by measurement noise.`, type: 'error' });
      lines.push({ text: `HALT: VQE loop aborted early. Saved 4,200 unused QPU shots.`, type: 'error' });
    } else {
      lines.push({ text: `Variance stable at ${optimizationData.finalVariance.toFixed(3)}.`, type: 'success' });
      lines.push({ text: `Convergence to ground state (${TARGET_ENERGY} Ha) successful.`, type: 'success' });
    }

    setLogLines(lines);
  }, [noise, optimizationData.haltedStep, optimizationData.finalVariance]);

  // Coordinate mapping for SVG
  const mapX = (step: number) => (step / MAX_STEPS) * 500;
  const mapY = (energy: number) => {
    // Range roughly from -2.0 to 1.0
    // Y=0 at energy 1.0, Y=300 at energy -2.0
    // span = 3.0
    const normalized = (1.0 - energy) / 3.0;
    return normalized * 300;
  };

  const idealPath = `M ${optimizationData.steps.map(p => `${mapX(p.step)},${mapY(p.ideal)}`).join(' L ')}`;
  
  // Cut off actual path if halted
  const visibleSteps = optimizationData.haltedStep !== -1 
    ? optimizationData.steps.slice(0, optimizationData.haltedStep + 1)
    : optimizationData.steps;

  const actualPath = `M ${visibleSteps.map(p => `${mapX(p.step)},${mapY(p.actual)}`).join(' L ')}`;

  const controls = (
    <>
      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Hardware Shot Noise
          <span className="text-[var(--color-text-primary)]">{noise}%</span>
        </label>
        <input 
          type="range" 
          min="0" 
          max="80" 
          value={noise} 
          onChange={(e) => setNoise(Number(e.target.value))}
          className="w-full accent-[var(--color-accent-blue)]"
        />
        <p className="text-[0.8rem] text-[var(--color-text-tertiary)] leading-snug mt-1">
          Increase measurement error to simulate noisy intermediate-scale quantum (NISQ) hardware.
        </p>
      </div>
      
      <div className="flex flex-col gap-2 mt-4">
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-[2px] bg-[var(--color-accent-blue)]"></div>
          Measured Optimization Path
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-[2px] bg-[var(--color-accent-blue)] opacity-30 border-dashed border-[var(--color-accent-blue)]"></div>
          Ideal Ground State Trajectory
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 border border-red-500 rounded-full"></div>
          Divergence Halt Trigger
        </div>
      </div>
    </>
  );

  const visualization = (
    <div className="w-full h-full min-h-[300px] flex items-center justify-center relative">
      <svg viewBox="0 0 500 300" className="w-full h-full overflow-visible">
        {/* Ideal path */}
        <path d={idealPath} fill="none" stroke="var(--color-accent-blue)" strokeWidth="2" opacity="0.3" strokeDasharray="4 4" />
        
        {/* Actual path */}
        <path d={actualPath} fill="none" stroke="var(--color-accent-blue)" strokeWidth="2" />
        
        {/* Halt Marker */}
        {optimizationData.haltedStep !== -1 && (
          <circle 
            cx={mapX(optimizationData.haltedStep)} 
            cy={mapY(visibleSteps[visibleSteps.length - 1].actual)} 
            r="6" 
            fill="none" 
            stroke="rgb(239, 68, 68)" 
            strokeWidth="2" 
          />
        )}

        {/* Target Energy Line */}
        <line x1="0" y1={mapY(TARGET_ENERGY)} x2="500" y2={mapY(TARGET_ENERGY)} stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
        <text x="10" y={mapY(TARGET_ENERGY) - 5} fill="var(--color-text-tertiary)" fontSize="10" fontFamily="monospace">Target Ground State</text>
      </svg>
    </div>
  );

  return (
    <SimulatorShell
      title="Quantum Chemistry: VQE Optimization"
      description="Adjust shot noise to see how Sounio tracks execution variance and aborts expensive quantum loops when NISQ hardware noise corrupts the optimizer landscape."
      controls={controls}
      visualization={visualization}
      logLines={logLines}
      isHalted={isHalted}
    />
  );
}
