import { useState, useEffect, useRef } from 'react';

// ---------------------------------------------------------------------------
// EpistemicFlowViz
// Animated SVG/CSS visualisation of the Sounio epistemic chain:
//   measure → Knowledge<T> → lift_knowledge → Contest<T>
//   → prove_robust → Robust<T> → validate_manifest → Validated<T>
//
// Each transformation is zero-cost: = MOV dst,src1 at machine level.
// ---------------------------------------------------------------------------

interface ChainStep {
  id: string;
  kind: 'transform' | 'type';
  label: string;
  sublabel?: string;
  color: string;
  textColor: string;
  borderColor: string;
}

const CHAIN: ChainStep[] = [
  {
    id: 'measure',
    kind: 'transform',
    label: 'measure(v, ε)',
    sublabel: 'source value + uncertainty bound',
    color: 'rgba(99,102,241,0.15)',
    textColor: '#a5b4fc',
    borderColor: '#6366f1',
  },
  {
    id: 'knowledge',
    kind: 'type',
    label: 'Knowledge<T>',
    sublabel: 'epistemic wrapper',
    color: 'rgba(99,102,241,0.25)',
    textColor: '#c7d2fe',
    borderColor: '#818cf8',
  },
  {
    id: 'lift',
    kind: 'transform',
    label: 'lift_knowledge',
    sublabel: 'promote to contestable',
    color: 'rgba(16,185,129,0.15)',
    textColor: '#6ee7b7',
    borderColor: '#10b981',
  },
  {
    id: 'contest',
    kind: 'type',
    label: 'Contest<T>',
    sublabel: 'open to challenge',
    color: 'rgba(16,185,129,0.25)',
    textColor: '#a7f3d0',
    borderColor: '#34d399',
  },
  {
    id: 'prove',
    kind: 'transform',
    label: 'prove_robust(c, p)',
    sublabel: 'attach proof obligation',
    color: 'rgba(245,158,11,0.15)',
    textColor: '#fcd34d',
    borderColor: '#f59e0b',
  },
  {
    id: 'robust',
    kind: 'type',
    label: 'Robust<T>',
    sublabel: 'proof-carrying value',
    color: 'rgba(245,158,11,0.25)',
    textColor: '#fde68a',
    borderColor: '#fbbf24',
  },
  {
    id: 'validate',
    kind: 'transform',
    label: 'validate_manifest',
    sublabel: 'discharge all obligations',
    color: 'rgba(239,68,68,0.15)',
    textColor: '#fca5a5',
    borderColor: '#ef4444',
  },
  {
    id: 'validated',
    kind: 'type',
    label: 'Validated<T>',
    sublabel: 'audit-ready, zero uncertainty',
    color: 'rgba(239,68,68,0.25)',
    textColor: '#fecaca',
    borderColor: '#f87171',
  },
];

// Particle that flows through the chain
interface Particle {
  id: number;
  progress: number; // 0..1 over the full chain width
  speed: number;
}

export default function EpistemicFlowViz() {
  const [activeStep, setActiveStep] = useState<string | null>(null);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [running, setRunning] = useState(true);
  const nextId = useRef(0);
  const raf = useRef<number>(0);
  const lastSpawn = useRef(0);

  useEffect(() => {
    if (!running) return;

    let prev = performance.now();

    function tick(now: number) {
      const dt = (now - prev) / 1000;
      prev = now;

      // Spawn a new particle every ~1.8 s
      if (now - lastSpawn.current > 1800) {
        lastSpawn.current = now;
        const id = nextId.current++;
        setParticles(ps => [
          ...ps.filter(p => p.progress < 1),
          { id, progress: 0, speed: 0.06 + Math.random() * 0.04 },
        ]);
      }

      setParticles(ps =>
        ps
          .map(p => ({ ...p, progress: p.progress + p.speed * dt }))
          .filter(p => p.progress <= 1.05),
      );

      raf.current = requestAnimationFrame(tick);
    }

    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [running]);

  const STEP_W = 110;
  const ARROW_W = 36;
  const STEP_H = 72;
  const TOTAL_W = CHAIN.length * STEP_W + (CHAIN.length - 1) * ARROW_W;
  const PAD_X = 24;
  const PAD_Y = 20;
  const SVG_W = TOTAL_W + PAD_X * 2;
  const SVG_H = STEP_H + PAD_Y * 2 + 32; // +32 for zero-cost label

  // Map particle progress [0,1] to SVG x coordinate along the chain midline
  function particleX(progress: number): number {
    return PAD_X + progress * TOTAL_W;
  }

  const MIDLINE_Y = PAD_Y + STEP_H / 2;

  return (
    <div className="w-full flex flex-col items-center gap-4 select-none">
      {/* Controls */}
      <div className="flex items-center gap-3 text-sm font-mono">
        <button
          onClick={() => setRunning(r => !r)}
          className="px-3 py-1 rounded border border-white/20 text-white/70 hover:text-white hover:border-white/40 transition-colors"
        >
          {running ? 'Pause' : 'Play'}
        </button>
        <span className="text-white/40 text-xs">
          Click a step to inspect · values flow left → right
        </span>
      </div>

      {/* Main SVG */}
      <div className="w-full overflow-x-auto">
        <svg
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          width={SVG_W}
          height={SVG_H}
          style={{ minWidth: SVG_W, display: 'block', margin: '0 auto' }}
          aria-label="Sounio epistemic chain visualisation"
        >
          {/* Arrows between steps */}
          {CHAIN.map((_step, i) => {
            if (i === CHAIN.length - 1) return null;
            const x = PAD_X + i * (STEP_W + ARROW_W) + STEP_W;
            const y = MIDLINE_Y;
            return (
              <g key={`arrow-${i}`}>
                <line
                  x1={x}
                  y1={y}
                  x2={x + ARROW_W}
                  y2={y}
                  stroke="#ffffff22"
                  strokeWidth={1.5}
                />
                <polygon
                  points={`${x + ARROW_W},${y} ${x + ARROW_W - 7},${y - 4} ${x + ARROW_W - 7},${y + 4}`}
                  fill="#ffffff33"
                />
              </g>
            );
          })}

          {/* Step boxes */}
          {CHAIN.map((step, i) => {
            const x = PAD_X + i * (STEP_W + ARROW_W);
            const y = PAD_Y;
            const isActive = activeStep === step.id;
            return (
              <g
                key={step.id}
                onClick={() => setActiveStep(activeStep === step.id ? null : step.id)}
                style={{ cursor: 'pointer' }}
              >
                <rect
                  x={x}
                  y={y}
                  width={STEP_W}
                  height={STEP_H}
                  rx={8}
                  fill={step.color}
                  stroke={isActive ? step.borderColor : step.borderColor + '66'}
                  strokeWidth={isActive ? 2 : 1}
                />
                <text
                  x={x + STEP_W / 2}
                  y={y + 26}
                  textAnchor="middle"
                  fill={step.textColor}
                  fontSize={step.kind === 'type' ? 11 : 10}
                  fontFamily="monospace"
                  fontWeight={step.kind === 'type' ? 'bold' : 'normal'}
                >
                  {step.label}
                </text>
                <text
                  x={x + STEP_W / 2}
                  y={y + 42}
                  textAnchor="middle"
                  fill={step.textColor + 'aa'}
                  fontSize={8.5}
                  fontFamily="sans-serif"
                >
                  {step.sublabel}
                </text>
                {/* kind badge */}
                <rect
                  x={x + STEP_W / 2 - 22}
                  y={y + STEP_H - 18}
                  width={44}
                  height={13}
                  rx={4}
                  fill={step.borderColor + '33'}
                />
                <text
                  x={x + STEP_W / 2}
                  y={y + STEP_H - 8}
                  textAnchor="middle"
                  fill={step.textColor + 'cc'}
                  fontSize={7.5}
                  fontFamily="monospace"
                >
                  {step.kind === 'type' ? 'type' : 'transform'}
                </text>
              </g>
            );
          })}

          {/* Zero-cost annotation */}
          <text
            x={SVG_W / 2}
            y={PAD_Y + STEP_H + 22}
            textAnchor="middle"
            fill="#ffffff33"
            fontSize={9}
            fontFamily="monospace"
          >
            all transforms = MOV dst,src1 at machine level (zero runtime cost)
          </text>

          {/* Flowing particles */}
          {particles.map(p => {
            const px = particleX(Math.min(p.progress, 1));
            const opacity = p.progress < 0.05
              ? p.progress / 0.05
              : p.progress > 0.9
              ? (1 - p.progress) / 0.1
              : 1;
            return (
              <circle
                key={p.id}
                cx={px}
                cy={MIDLINE_Y}
                r={4}
                fill="#818cf8"
                opacity={opacity * 0.85}
              />
            );
          })}
        </svg>
      </div>

      {/* Detail panel for active step */}
      {activeStep && (() => {
        const step = CHAIN.find(s => s.id === activeStep);
        if (!step) return null;
        return (
          <div
            className="w-full max-w-lg p-4 rounded-lg border text-sm font-mono"
            style={{
              background: step.color,
              borderColor: step.borderColor + '88',
              color: step.textColor,
            }}
          >
            <div className="font-bold mb-1">{step.label}</div>
            <div className="text-xs opacity-80 mb-2">{step.sublabel}</div>
            {step.kind === 'transform' && (
              <div className="text-xs opacity-60">
                Compile-time proof obligation — zero runtime overhead.<br />
                Lowers to: <span className="opacity-90">MOV dst, src1</span>
              </div>
            )}
            {step.kind === 'type' && (
              <div className="text-xs opacity-60">
                Phantom wrapper type — no runtime representation.<br />
                Size of {step.label} = size of T.
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}
