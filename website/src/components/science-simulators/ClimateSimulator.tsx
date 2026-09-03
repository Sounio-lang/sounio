import { useState, useEffect, useMemo } from 'react';
import { SimulatorShell } from './SimulatorShell';

export function ClimateSimulator() {
  const [uncertainty, setUncertainty] = useState(25);
  const [isHalted, setIsHalted] = useState(false);
  const [logLines, setLogLines] = useState<{ text: string; type: 'info' | 'warn' | 'error' | 'success' }[]>([]);

  const PRESENT_YEAR = 2020;
  const START_YEAR = 1850;
  const END_YEAR = 2100;

  const projectionData = useMemo(() => {
    const historical = [];
    const projections = []; // Central, Upper, Lower
    let maxSpread = 0;
    let haltedYear = -1;

    // Simulate historical data with some embedded noise
    for (let year = START_YEAR; year <= PRESENT_YEAR; year += 2) {
      const anomaly = (year - START_YEAR) * 0.005 + Math.sin(year * 0.1) * 0.1;
      const histUncertainty = (uncertainty / 100) * 0.15;
      
      historical.push({
        year,
        val: anomaly,
        upper: anomaly + histUncertainty,
        lower: anomaly - histUncertainty
      });
    }

    const lastHist = historical[historical.length - 1];

    // Project future based on historical variance
    for (let year = PRESENT_YEAR; year <= END_YEAR; year += 2) {
      const yearsOut = year - PRESENT_YEAR;
      // Exponential divergence based on input uncertainty
      const spread = (uncertainty / 100) * 1.5 * Math.pow(yearsOut / 10, 1.4);
      
      // Central trend
      const val = lastHist.val + (yearsOut * 0.02);
      
      const upper = val + spread;
      const lower = val - spread;
      const currentSpread = upper - lower;

      if (currentSpread > maxSpread) maxSpread = currentSpread;

      // Truncation threshold
      // For visual mapping, let's say spread > 4.5 is too wide
      if (currentSpread > 4.5 && haltedYear === -1) {
        haltedYear = year;
      }

      projections.push({
        year,
        val,
        upper,
        lower,
        spread: currentSpread
      });
    }

    return { historical, projections, haltedYear, maxSpread };
  }, [uncertainty]);

  useEffect(() => {
    const halted = projectionData.haltedYear !== -1;
    setIsHalted(halted);

    const lines: { text: string; type: 'info' | 'warn' | 'error' | 'success' }[] = [
      { text: `Running ensemble climate projection model...`, type: 'info' },
      { text: `Ingesting historical data (1850-${PRESENT_YEAR})...`, type: 'info' },
      { text: `Pre-1950 Sparse Sensor Noise: ${uncertainty}%`, type: 'info' }
    ];

    if (halted) {
      lines.push({ text: `WARN: Variance compounding. Ensemble spread exceeded safe utility bounds at year ${projectionData.haltedYear}.`, type: 'warn' });
      lines.push({ text: `FATAL: Long-term projection (2100) confidence collapsed to near-zero.`, type: 'error' });
      lines.push({ text: `HALT: Projection truncated to prevent actionable policy on speculative data.`, type: 'error' });
    } else {
      lines.push({ text: `Ensemble projection spread remains within policy-actionable bounds.`, type: 'success' });
      lines.push({ text: `Execution approved. 2100 projection generated successfully.`, type: 'success' });
    }

    setLogLines(lines);
  }, [uncertainty, projectionData.haltedYear]);

  // Coordinate mapping for SVG
  const mapX = (year: number) => ((year - START_YEAR) / (END_YEAR - START_YEAR)) * 500;
  const mapY = (anomaly: number) => {
    // Anomaly ranges roughly from -0.5 to 5.0
    // Y=300 is -1.0, Y=0 is +6.0
    const normalized = (6.0 - anomaly) / 7.0;
    return normalized * 300;
  };

  const histPath = `M ${projectionData.historical.map(p => `${mapX(p.year)},${mapY(p.val)}`).join(' L ')}`;
  const histArea = `M ${projectionData.historical.map(p => `${mapX(p.year)},${mapY(p.upper)}`).join(' L ')} L ${projectionData.historical.map(p => `${mapX(p.year)},${mapY(p.lower)}`).reverse().join(' L ')} Z`;

  const visibleProjections = projectionData.haltedYear !== -1
    ? projectionData.projections.filter(p => p.year <= projectionData.haltedYear)
    : projectionData.projections;

  const projPath = `M ${visibleProjections.map(p => `${mapX(p.year)},${mapY(p.val)}`).join(' L ')}`;
  const projArea = `M ${visibleProjections.map(p => `${mapX(p.year)},${mapY(p.upper)}`).join(' L ')} L ${visibleProjections.map(p => `${mapX(p.year)},${mapY(p.lower)}`).reverse().join(' L ')} Z`;

  const controls = (
    <>
      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Historical Sensor Uncertainty
          <span className="text-[var(--color-text-primary)]">{uncertainty}%</span>
        </label>
        <input 
          type="range" 
          min="0" 
          max="80" 
          value={uncertainty} 
          onChange={(e) => setUncertainty(Number(e.target.value))}
          className="w-full accent-[var(--color-accent-green)]"
        />
        <p className="text-[0.8rem] text-[var(--color-text-tertiary)] leading-snug mt-1">
          Increase to simulate wider confidence intervals in sparse pre-1950 surface temperature records.
        </p>
      </div>
      
      <div className="flex flex-col gap-2 mt-4">
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-[2px] bg-[var(--color-text-primary)]"></div>
          Historical Record
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-[2px] bg-[var(--color-accent-green)]"></div>
          Future Projection
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 bg-[var(--color-accent-green)] opacity-20 border border-[var(--color-accent-green)]"></div>
          Ensemble Spread (Uncertainty)
        </div>
      </div>
    </>
  );

  const visualization = (
    <div className="w-full h-full min-h-[300px] flex items-center justify-center relative">
      <svg viewBox="0 0 500 300" className="w-full h-full overflow-visible">
        {/* Axes */}
        <line x1="0" y1={mapY(0)} x2="500" y2={mapY(0)} stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="4 4" />
        <text x="10" y={mapY(0) - 5} fill="var(--color-text-tertiary)" fontSize="10" fontFamily="monospace">Pre-industrial Baseline (0°C)</text>

        <line x1={mapX(PRESENT_YEAR)} y1="0" x2={mapX(PRESENT_YEAR)} y2="300" stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="2 2" />
        <text x={mapX(PRESENT_YEAR) + 5} y="290" fill="var(--color-text-tertiary)" fontSize="10" fontFamily="monospace">Present</text>

        {/* Historical Area & Line */}
        <path d={histArea} fill="var(--color-text-primary)" opacity="0.1" />
        <path d={histPath} fill="none" stroke="var(--color-text-primary)" strokeWidth="2" />

        {/* Projection Area & Line */}
        <path d={projArea} fill="var(--color-accent-green)" opacity="0.15" />
        <path d={projArea} fill="none" stroke="var(--color-accent-green)" strokeWidth="1" opacity="0.4" strokeDasharray="2 2" />
        <path d={projPath} fill="none" stroke="var(--color-accent-green)" strokeWidth="2" />
        
        {/* Halt Marker */}
        {projectionData.haltedYear !== -1 && (
          <line 
            x1={mapX(projectionData.haltedYear)} 
            y1={0} 
            x2={mapX(projectionData.haltedYear)} 
            y2={300} 
            stroke="rgb(239, 68, 68)" 
            strokeWidth="2"
            opacity="0.5"
          />
        )}
      </svg>
    </div>
  );

  return (
    <SimulatorShell
      title="Climate Science: Ensemble Projection Truncation"
      description="Adjust historical data uncertainty to see Sounio refuse to output a 2100 projection when compounded variance makes the ensemble useless."
      controls={controls}
      visualization={visualization}
      logLines={logLines}
      isHalted={isHalted}
    />
  );
}
