import { useState, useEffect, useMemo } from 'react';
import { SimulatorShell } from './SimulatorShell';

export function NeuroSimulator() {
  const [sparsity, setSparsity] = useState(15);
  const [isHalted, setIsHalted] = useState(false);
  const [logLines, setLogLines] = useState<{ text: string; type: 'info' | 'warn' | 'error' | 'success' }[]>([]);

  const GRID_SIZE = 16;
  const CORE_RADIUS = 5;
  const CENTER_X = 8;
  const CENTER_Y = 8;
  const CONFIDENCE_THRESHOLD = 0.55;

  // Generate the BOLD signal heatmap grid
  const grid = useMemo(() => {
    const cells = [];
    let coreActiveCount = 0;
    let coreTotalCount = 0;

    // Use a fixed pseudo-random seed approach so sliding back and forth is consistent
    const pseudoRandom = (x: number, y: number) => {
      const n = Math.sin(x * 12.9898 + y * 78.233) * 43758.5453;
      return n - Math.floor(n);
    };

    for (let y = 0; y < GRID_SIZE; y++) {
      for (let x = 0; x < GRID_SIZE; x++) {
        const dist = Math.sqrt(Math.pow(x - CENTER_X, 2) + Math.pow(y - CENTER_Y, 2));
        const isCore = dist <= CORE_RADIUS;
        
        let baseActivation = 0;
        if (isCore) {
          coreTotalCount++;
          // Core is active, drops off near edges
          baseActivation = 1 - (dist / CORE_RADIUS) * 0.5;
        } else {
          // Background noise
          baseActivation = pseudoRandom(x, y) * 0.3;
        }

        // Apply sparsity dropout
        const rand = pseudoRandom(x + 10, y + 10);
        const dropoutThreshold = sparsity / 100;
        const droppedOut = rand < dropoutThreshold;

        const finalActivation = droppedOut ? 0 : baseActivation;

        if (isCore && finalActivation > 0) {
          coreActiveCount++;
        }

        cells.push({
          x,
          y,
          activation: finalActivation,
          isCore,
          droppedOut
        });
      }
    }

    const regionalConfidence = coreActiveCount / coreTotalCount;
    return { cells, regionalConfidence };
  }, [sparsity]);

  useEffect(() => {
    const halted = grid.regionalConfidence < CONFIDENCE_THRESHOLD;
    setIsHalted(halted);

    const lines: { text: string; type: 'info' | 'warn' | 'error' | 'success' }[] = [
      { text: `Processing fMRI BOLD signal matrix...`, type: 'info' },
      { text: `Voxel Sparsity Rate: ${sparsity}%`, type: 'info' },
      { text: `Calculated regional confidence: ${(grid.regionalConfidence * 100).toFixed(1)}%`, type: 'info' }
    ];

    if (halted) {
      lines.push({ text: `WARN: Confidence dropped below diagnostic threshold (${(CONFIDENCE_THRESHOLD * 100).toFixed(0)}%).`, type: 'warn' });
      lines.push({ text: `FATAL: Spatial cluster is too sparse to guarantee provenance-aware findings.`, type: 'error' });
      lines.push({ text: `HALT: Execution halted. Diagnostic cluster discarded safely.`, type: 'error' });
    } else {
      lines.push({ text: `Spatial cluster density verified.`, type: 'success' });
      lines.push({ text: `Execution approved. Extracting diagnostic biomarker.`, type: 'success' });
    }

    setLogLines(lines);
  }, [sparsity, grid.regionalConfidence]);

  const controls = (
    <>
      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Voxel Dropout Rate
          <span className="text-[var(--color-text-primary)]">{sparsity}%</span>
        </label>
        <input 
          type="range" 
          min="0" 
          max="80" 
          value={sparsity} 
          onChange={(e) => setSparsity(Number(e.target.value))}
          className="w-full accent-[var(--color-accent-blue)]"
        />
        <p className="text-[0.8rem] text-[var(--color-text-tertiary)] leading-snug mt-1">
          Increase spatial sparsity to simulate motion artifacts or scanner signal degradation.
        </p>
      </div>
      
      <div className="flex flex-col gap-2 mt-4">
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 bg-red-500 rounded-sm"></div>
          High BOLD Activation
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 bg-blue-500 rounded-sm opacity-50"></div>
          Background Noise
        </div>
        <div className="flex items-center gap-2 text-sm text-[var(--color-text-secondary)]">
          <div className="w-3 h-3 bg-black border border-[var(--glass-border)] rounded-sm"></div>
          Signal Dropout
        </div>
      </div>
    </>
  );

  const visualization = (
    <div className="w-full h-full min-h-[300px] flex items-center justify-center">
      <div 
        className="grid gap-[2px] p-2 bg-[rgba(255,255,255,0.02)] border border-[var(--glass-border)] rounded-lg shadow-lg"
        style={{ gridTemplateColumns: `repeat(${GRID_SIZE}, minmax(0, 1fr))` }}
      >
        {grid.cells.map((cell, i) => {
          let bgColor = 'transparent';
          if (!cell.droppedOut && cell.activation > 0) {
            // Map activation 0-1 to a color scale (blue -> purple -> red -> yellow)
            if (cell.activation > 0.8) bgColor = 'rgb(250, 204, 21)'; // yellow
            else if (cell.activation > 0.5) bgColor = 'rgb(239, 68, 68)'; // red
            else if (cell.activation > 0.3) bgColor = 'rgb(168, 85, 247)'; // purple
            else bgColor = 'rgba(59, 130, 246, 0.4)'; // blue
          } else if (cell.droppedOut) {
            bgColor = 'rgba(0, 0, 0, 0.4)';
          }

          return (
            <div 
              key={i} 
              className="w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 rounded-sm transition-colors duration-200"
              style={{ backgroundColor: bgColor }}
            ></div>
          );
        })}
      </div>
    </div>
  );

  return (
    <SimulatorShell
      title="Neuroimaging: BOLD Signal Thresholds"
      description="Adjust the voxel dropout rate to simulate how Sounio prevents generating false diagnostic clusters when the underlying spatial confidence collapses."
      controls={controls}
      visualization={visualization}
      logLines={logLines}
      isHalted={isHalted}
    />
  );
}
