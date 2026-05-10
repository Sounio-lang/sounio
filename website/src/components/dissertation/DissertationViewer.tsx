import { Suspense, useCallback, useEffect, useMemo, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import { Compartments } from './Compartments';
import { BloodFlowEdges } from './BloodFlowEdges';
import { Stent } from './Stent';
import { Silhouette } from './Silhouette';
import { COMPARTMENTS } from './compartments';
import { DEFAULT_PARAMS, useSimulation } from '../../hooks/usePBPK14';
import type { PBPKParams } from '../../hooks/usePBPK14';

const PATIENT_PROFILES: Record<string, PBPKParams> = {
  typical: { ...DEFAULT_PARAMS },
  'low CL': { ...DEFAULT_PARAMS, clHep: 6.2 },     // CYP3A4 poor metabolizer
  'high CL': { ...DEFAULT_PARAMS, clHep: 24.8 },   // CYP3A4 ultrarapid
  lean: { ...DEFAULT_PARAMS, vdScale: 0.7 },
  obese: { ...DEFAULT_PARAMS, vdScale: 1.4 },
};

const MAX_SIM_HOURS = 30 * 24; // 30-day Cypher elution window
const REAL_SECONDS_PER_30D = 60; // 1 minute of wall-clock spans the entire 30-day window at speed=1

interface SimulationBridgeProps {
  params: PBPKParams;
  playing: boolean;
  speed: number;
  onTick: (timeHours: number, conc: Float32Array, uncert: Float32Array, releasedMg: number) => void;
}

function SimulationBridge({ params, playing, speed, onTick }: SimulationBridgeProps) {
  const { state, step } = useSimulation(params);

  // Emit state outward whenever it changes.
  useEffect(() => {
    onTick(state.timeHours, state.concentrations, state.uncertainties, state.releasedMg);
  }, [state, onTick]);

  useFrame((_, delta) => {
    if (!playing) return;
    const simHoursPerWallSec = (MAX_SIM_HOURS / REAL_SECONDS_PER_30D) * speed;
    const dt = Math.min(0.5, delta * simHoursPerWallSec); // cap RK4 step at 30 min sim time
    if (dt <= 0) return;
    if (state.timeHours + dt > MAX_SIM_HOURS) return;
    step(dt);
  });

  return null;
}

export default function DissertationViewer() {
  const [params, setParams] = useState<PBPKParams>(PATIENT_PROFILES.typical);
  const [profile, setProfile] = useState<keyof typeof PATIENT_PROFILES>('typical');
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState<number | null>(null);

  interface SimSnapshot {
    timeHours: number;
    concentrations: Float32Array;
    uncertainties: Float32Array;
    releasedMg: number;
  }
  const [simState, setSimState] = useState<SimSnapshot>(() => ({
    timeHours: 0,
    concentrations: new Float32Array(14),
    uncertainties: new Float32Array(14),
    releasedMg: 0,
  }));
  const onTick = useCallback(
    (timeHours: number, concentrations: Float32Array, uncertainties: Float32Array, releasedMg: number) => {
      setSimState({ timeHours, concentrations, uncertainties, releasedMg });
    },
    [],
  );

  // Re-apply profile.
  useEffect(() => {
    setParams(PATIENT_PROFILES[profile]);
  }, [profile]);

  const dayString = useMemo(() => {
    const d = simState.timeHours / 24;
    if (d < 1) return `${simState.timeHours.toFixed(1)} h`;
    return `${d.toFixed(2)} d`;
  }, [simState.timeHours]);

  const selectedCompartment = selected !== null ? COMPARTMENTS[selected] : null;
  const selectedConc = selected !== null ? simState.concentrations[selected] : 0;

  return (
    <div className="dissertation-viewer rounded-[var(--radius-xl)] overflow-hidden border border-[var(--glass-border)] bg-[rgba(0,0,0,0.35)] mt-[2rem]">
      {/* 3D canvas */}
      <div className="relative h-[640px] w-full">
        <Canvas
          shadows
          dpr={[1, 2]}
          gl={{ antialias: true, alpha: true }}
        >
          <color attach="background" args={['#0b1020']} />
          <fog attach="fog" args={['#0b1020', 12, 30]} />
          <PerspectiveCamera makeDefault position={[6, 3.5, 8]} fov={42} near={0.1} far={100} />
          <OrbitControls
            target={[0, 2.5, 0]}
            enableDamping
            dampingFactor={0.08}
            minDistance={4}
            maxDistance={20}
          />
          <ambientLight intensity={0.55} />
          <directionalLight position={[6, 10, 8]} intensity={0.85} castShadow />
          <directionalLight position={[-6, 4, -6]} intensity={0.25} color="#60a5fa" />
          <Suspense fallback={null}>
            <Silhouette />
            <BloodFlowEdges />
            <Compartments
              concentrations={simState.concentrations}
              uncertainties={simState.uncertainties}
              selected={selected}
              onSelect={(i) => setSelected(i === selected ? null : i)}
            />
            <Stent timeHours={simState.timeHours} speed={speed} />
            <SimulationBridge params={params} playing={playing} speed={speed} onTick={onTick} />
          </Suspense>
        </Canvas>

        {/* HUD: time + released mass + Phase J gate */}
        <div className="absolute top-4 left-4 flex flex-col gap-1 text-sm text-white pointer-events-none">
          <div className="bg-black/55 px-3 py-1.5 rounded">
            <span className="opacity-75">t = </span>
            <span className="font-mono font-semibold">{dayString}</span>
          </div>
          <div className="bg-black/55 px-3 py-1.5 rounded">
            <span className="opacity-75">Released: </span>
            <span className="font-mono">{(simState.releasedMg * 1000).toFixed(1)} µg</span>
            <span className="opacity-50"> / 140 µg</span>
          </div>
        </div>

        {/* Per-organ detail (Phase D will replace with KaTeX modal) */}
        {selectedCompartment && (
          <div className="absolute top-4 right-4 max-w-[260px] bg-black/65 backdrop-blur p-3 rounded text-white text-sm">
            <div className="text-base font-semibold mb-1">{selectedCompartment.label}</div>
            <div className="opacity-75 text-xs mb-2">{selectedCompartment.family}</div>
            <div className="font-mono text-xs leading-relaxed">
              Kp = {selectedCompartment.kp.toFixed(2)}<br />
              Q&nbsp; = {selectedCompartment.q} L/h<br />
              C&nbsp; ≈ {(selectedConc * 100).toFixed(1)} % of visual max
            </div>
            <button
              type="button"
              onClick={() => setSelected(null)}
              className="mt-2 text-xs underline opacity-70 hover:opacity-100"
            >
              close
            </button>
          </div>
        )}
      </div>

      {/* Bottom control strip */}
      <div className="grid grid-cols-1 md:grid-cols-[1fr_1fr_1fr_220px] gap-4 p-4 border-t border-[var(--glass-border)] bg-[rgba(0,0,0,0.25)]">
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-wide opacity-70">Patient profile</label>
          <select
            value={profile}
            onChange={(e) => setProfile(e.target.value as keyof typeof PATIENT_PROFILES)}
            className="bg-black/35 border border-white/15 rounded px-2 py-1.5 text-sm"
          >
            {Object.keys(PATIENT_PROFILES).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <p className="text-[0.7rem] opacity-60">
            CYP3A4 metabolizer status changes the hepatic clearance, widening the GUM cone.
          </p>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-wide opacity-70">
            Stent release scale ({params.higuchiScale.toFixed(2)}×)
          </label>
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={params.higuchiScale}
            onChange={(e) =>
              setParams({ ...params, higuchiScale: Number(e.target.value) })
            }
          />
          <p className="text-[0.7rem] opacity-60">
            Higuchi K_H multiplier; reflects coating thickness & solubility uncertainty.
          </p>
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-wide opacity-70">
            Speed ({speed.toFixed(1)}×)
          </label>
          <input
            type="range"
            min={0.1}
            max={4}
            step={0.1}
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          />
          <p className="text-[0.7rem] opacity-60">
            Wall-clock playback rate; full 30-day window takes ~60 s at 1×.
          </p>
        </div>

        <div className="flex flex-col gap-2 justify-end">
          <button
            type="button"
            onClick={() => setPlaying(!playing)}
            className="px-3 py-1.5 rounded bg-amber-500/85 text-black font-semibold text-sm hover:bg-amber-500"
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            onClick={() => {
              setParams({ ...params }); // triggers useEffect reset (params reference change)
              setProfile(profile);
            }}
            className="px-3 py-1.5 rounded bg-white/10 text-white text-sm hover:bg-white/15"
          >
            Restart at t = 0
          </button>
        </div>
      </div>
    </div>
  );
}
