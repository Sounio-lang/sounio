import { useState, useEffect, useMemo } from 'react';
import { SimulatorShell } from './SimulatorShell';

export function PharmaSimulator() {
  const [noise, setNoise] = useState(20);
  const [isHalted, setIsHalted] = useState(false);
  const [logLines, setLogLines] = useState<{ text: string; type: 'info' | 'warn' | 'error' | 'success' }[]>([]);

  const TOXICITY_THRESHOLD = 60;
  
  // Calculate PK curve
  const points = useMemo(() => {
    const pts = [];
    let maxUpper = 0;
    
    for (let t = 0; t <= 50; t++) {
      // Basic one-compartment oral absorption model
      const c = 120 * (Math.exp(-0.05 * t) - Math.exp(-0.2 * t));
      
      // Uncertainty proportional to concentration and noise slider
      const uncertainty = c * (noise / 100) * 0.7;
      
      const upper = c + uncertainty;
      const lower = Math.max(0, c - uncertainty);
      
      if (upper > maxUpper) {
        maxUpper = upper;
      }
      
      pts.push({ x: t * 10, y: 300 - (c * 3), upperY: 300 - (upper * 3), lowerY: 300 - (lower * 3), c });
    }
    return { pts, maxUpper };
  }, [noise]);

  useEffect(() => {
    const halted = points.maxUpper > TOXICITY_THRESHOLD;
    setIsHalted(halted);

    const confidence = Math.max(0, 100 - noise).toFixed(1);
    
    const lines: { text: string; type: 'info' | 'warn' | 'error' | 'success' }[] = [
      { text: `Evaluating patient dosing model...`, type: 'info' },
      { text: `Current CV (Noise): ${noise}%`, type: 'info' },
      { text: `Calculated confidence: ${confidence}%`, type: 'info' }
    ];

    if (halted) {
      lines.push({ text: `WARN: Confidence dropped below safe minimum bounds.`, type: 'warn' });
      lines.push({ text: `FATAL: Upper uncertainty bound intersects toxicity threshold (${TOXICITY_THRESHOLD} mg/L).`, type: 'error' });
      lines.push({ text: `HALT: Execution halted safely to prevent adverse decision.`, type: 'error' });
    } else {
      lines.push({ text: `Bounds check passed. Peak remains below toxicity threshold.`, type: 'success' });
      lines.push({ text: `Execution approved. Dosing schedule compiled.`, type: 'success' });
    }

    setLogLines(lines);
  }, [noise, points.maxUpper]);

  const curvePath = `M ${points.pts.map(p => `${p.x},${p.y}`).join(' L ')}`;
  const areaPath = `M ${points.pts.map(p => `${p.x},${p.upperY}`).join(' L ')} L ${points.pts.map(p => `${p.x},${p.lowerY}`).reverse().join(' L ')} Z`;

  const controls = (
    <>
      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Patient Data Noise (CV)
          <span className="text-[var(--color-text-primary)]">{noise}%</span>
        </label>
        <input 
          type="range" 
          min="0" 
          max="100" 
          value={noise} 
          onChange={(e) => setNoise(Number(e.target.value))}
          className="w-full accent-[var(--color-accent-teal)]"
        />
        <p className="text-[0.8rem] text-[var(--color-text-tertiary)] leading-snug mt-1">
          Simulate higher uncertainty in patient sensor data or population variability.
        </p>
      </div>
      
      <div className="flex flex-col gap-2 mt-4">
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 rounded-full bg-red-500/50 border border-red-500"></div>
          Toxicity Threshold (60 mg/L)
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 bg-[var(--color-accent-teal)] opacity-20 border border-[var(--color-accent-teal)]"></div>
          95% Confidence Envelope
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-[2px] bg-[var(--color-accent-teal)]"></div>
          Predicted Concentration
        </div>
      </div>
    </>
  );

  const visualization = (
    <div className="w-full h-full min-h-[300px] flex items-center justify-center">
      <svg viewBox="0 0 500 300" className="w-full h-full overflow-visible">
        {/* Toxicity Threshold Line */}
        <line x1="0" y1={300 - (TOXICITY_THRESHOLD * 3)} x2="500" y2={300 - (TOXICITY_THRESHOLD * 3)} stroke="rgba(239, 68, 68, 0.5)" strokeWidth="2" strokeDasharray="6 4" />
        <text x="10" y={300 - (TOXICITY_THRESHOLD * 3) - 10} fill="rgba(239, 68, 68, 0.8)" fontSize="12" fontFamily="monospace">Toxicity Limit</text>
        
        {/* Uncertainty Envelope */}
        <path d={areaPath} fill="var(--color-accent-teal)" opacity="0.15" />
        <path d={areaPath} fill="none" stroke="var(--color-accent-teal)" strokeWidth="1" opacity="0.3" strokeDasharray="4 2" />
        
        {/* Main Curve */}
        <path d={curvePath} fill="none" stroke="var(--color-accent-teal)" strokeWidth="3" />
        
        {/* Axis */}
        <line x1="0" y1="300" x2="500" y2="300" stroke="var(--glass-border)" strokeWidth="2" />
        <line x1="0" y1="0" x2="0" y2="300" stroke="var(--glass-border)" strokeWidth="2" />
      </svg>
    </div>
  );

  return (
    <SimulatorShell
      title="Pharmacometrics: PK/PD Confidence Bounds"
      description="Adjust the patient data noise to see how Sounio evaluates the expanding uncertainty envelope against the rigid toxicity threshold in real-time."
      controls={controls}
      visualization={visualization}
      logLines={logLines}
      isHalted={isHalted}
    />
  );
}
