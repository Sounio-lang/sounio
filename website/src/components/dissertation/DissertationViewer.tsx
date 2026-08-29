import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import type { WebGLRenderer } from 'three';
import { Compartments } from './Compartments';
import { BloodFlowEdges } from './BloodFlowEdges';
import { Stent } from './Stent';
import { SCDepot } from './SCDepot';
import { Silhouette } from './Silhouette';
import { GumBudgetBar } from './GumBudgetBar';
import { ConfidenceGate } from './ConfidenceGate';
import { HessianHeatmap } from './HessianHeatmap';
import { TmddPanel } from './TmddPanel';
import { PdReadoutPanel } from './PdReadoutPanel';
import { OrganDetailModal } from './OrganDetailModal';
import { TimeScrubber } from './TimeScrubber';
import { CameraDirector } from './CameraDirector';
import { TourControls } from './TourControls';
import { TOURS, type TourKeyframe } from './tours';
import { DEFAULT_PARAMS, useSimulation } from '../../hooks/usePBPK';
import type { PBPKParams, DrugId } from '../../hooks/usePBPK';
import { DEFAULT_PARAMS_RAPAMYCIN, DEFAULT_PARAMS_SEMAGLUTIDE } from '../../lib/pbpk28_core.mjs';
import { DrugSelector } from './DrugSelector';
import { useReducedMotion } from '../../hooks/useReducedMotion';

// Per-drug "typical patient" anchor + envelope perturbations.
//   rapamycin: clHep is CYP3A4 hepatic clearance (L/h), Ferron 1997 envelope
//              ± 2× covers PM ↔ ultrarapid.
//   semaglutide: clHep is proteolytic + renal clearance (L/h), Overgaard 2019
//              envelope is much tighter (CV ≈ 15%); lean/obese affects V_d more.
const PATIENT_PROFILES_BY_DRUG: Record<DrugId, Record<string, PBPKParams>> = {
  rapamycin: {
    typical: { ...DEFAULT_PARAMS },
    'low CL': { ...DEFAULT_PARAMS, clHep: 6.2 },
    'high CL': { ...DEFAULT_PARAMS, clHep: 24.8 },
    lean: { ...DEFAULT_PARAMS, vdScale: 0.7 },
    obese: { ...DEFAULT_PARAMS, vdScale: 1.4 },
  },
  semaglutide: {
    typical: { ...DEFAULT_PARAMS, clHep: 0.077 },
    'slow CL': { ...DEFAULT_PARAMS, clHep: 0.054 },
    'fast CL': { ...DEFAULT_PARAMS, clHep: 0.108 },
    lean: { ...DEFAULT_PARAMS, clHep: 0.077, vdScale: 0.7 },
    obese: { ...DEFAULT_PARAMS, clHep: 0.077, vdScale: 1.4 },
  },
};

const MAX_SIM_HOURS = 30 * 24;        // 30-day Cypher elution window
const REAL_SECONDS_PER_30D = 60;      // 1 minute wall-clock spans the full window at speed=1
// Drug-aware visual-saturation anchor: rapamycin and semaglutide differ by ~43×
// in plasma concentration scale, so the absolute-concentration display in the
// OrganDetailModal (and the mass-mg readout) must rescale on drug switch.
const C_VISUAL_MAX_BY_DRUG: Record<DrugId, number> = {
  rapamycin:   DEFAULT_PARAMS_RAPAMYCIN.cVisualMax,
  semaglutide: DEFAULT_PARAMS_SEMAGLUTIDE.cVisualMax,
};

interface SimSnapshot {
  timeHours: number;
  concentrations: Float32Array;
  uncertainties: Float32Array;
  releasedMg: number;
  // G-ε-1: TMDD + PD state available (rendering hooks to come in G-ε-2)
  rfree?: Float32Array;
  dr?: Float32Array;
  pdA?: Float32Array;
  pdN?: Float32Array;
}

interface SimulationBridgeProps {
  drug: DrugId;
  params: PBPKParams;
  playing: boolean;
  speed: number;
  /** Non-null sets a seek target (hours). The bridge advances toward it
   *  catching up to the requested time, then clears its internal latch. */
  seekTo: number | null;
  onSeekDone: () => void;
  onTick: (snapshot: SimSnapshot) => void;
}

function TourElapsedClock({ active, onTick }: { active: boolean; onTick: (s: number) => void }) {
  const elapsed = useRef(0);
  useEffect(() => {
    if (!active) elapsed.current = 0;
  }, [active]);
  useFrame((_, delta) => {
    if (!active) return;
    elapsed.current += delta;
    onTick(elapsed.current);
  });
  return null;
}

function SimulationBridge({ drug, params, playing, speed, seekTo, onSeekDone, onTick }: SimulationBridgeProps) {
  const { state, step } = useSimulation(drug, params);
  const lastEmitRef = useRef(state);

  useEffect(() => {
    lastEmitRef.current = state;
    onTick({ ...state });
  }, [state, onTick]);

  useFrame((_, delta) => {
    // Honour seek requests first — they catch the integrator up before play
    // resumes. We cap each frame's catch-up so the JS event loop stays
    // responsive even when the user drags by many days.
    if (seekTo !== null) {
      const remaining = seekTo - state.timeHours;
      if (Math.abs(remaining) < 1e-6) {
        onSeekDone();
        return;
      }
      if (remaining < 0) {
        // Backward seek requires a restart — handled at the React layer by
        // toggling params; here just clear the latch.
        onSeekDone();
        return;
      }
      const maxPerFrame = 6.0; // up to 6 simulation hours per frame
      const chunk = Math.min(remaining, maxPerFrame);
      // Use 0.1 h fixed steps for stability during fast catch-up.
      const dt = 0.1;
      let advanced = 0;
      while (advanced + dt <= chunk) {
        step(dt);
        advanced += dt;
      }
      if (chunk - advanced > 1e-6) {
        step(chunk - advanced);
      }
      return;
    }
    if (!playing) return;
    const simHoursPerWallSec = (MAX_SIM_HOURS / REAL_SECONDS_PER_30D) * speed;
    const dt = Math.min(0.5, delta * simHoursPerWallSec);
    if (dt <= 0) return;
    if (state.timeHours + dt > MAX_SIM_HOURS) return;
    step(dt);
  });

  return null;
}

export default function DissertationViewer() {
  const [drug, setDrug] = useState<DrugId>('rapamycin');
  const activeProfiles = PATIENT_PROFILES_BY_DRUG[drug];
  const [params, setParams] = useState<PBPKParams>(PATIENT_PROFILES_BY_DRUG.rapamycin.typical);
  const [profile, setProfile] = useState<string>('typical');
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [logScale, setLogScale] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  /** Forces a sim restart by changing the params reference. */
  const [restartNonce, setRestartNonce] = useState(0);

  // Stage E — tours, snapshot, reduced motion.
  const reducedMotion = useReducedMotion();
  const [activeTourId, setActiveTourId] = useState<string | null>(null);
  const [tourNarration, setTourNarration] = useState<string | null>(null);
  const [tourElapsed, setTourElapsed] = useState(0);
  const controlsRef = useRef<{ target: import('three').Vector3; update: () => void } | null>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const activeTour = useMemo(() => TOURS.find((t) => t.id === activeTourId) ?? null, [activeTourId]);
  const drugTours = useMemo(() => TOURS.filter((t) => t.drug === drug), [drug]);

  const handleTourKeyframe = useCallback((k: TourKeyframe) => {
    setTourNarration(k.narration ?? null);
    if (k.profile && k.profile in activeProfiles) {
      setProfile(k.profile);
    }
    if (typeof k.higuchiScale === 'number') {
      setParams((p) => ({ ...p, higuchiScale: k.higuchiScale! }));
    }
  }, []);
  const handleTourElapsedTick = useCallback((s: number) => setTourElapsed(s), []);
  const handleTourComplete = useCallback(() => {
    setActiveTourId(null);
    setTourNarration(null);
    setTourElapsed(0);
  }, []);
  const handleTourStart = (id: string) => {
    setActiveTourId(id);
    setTourElapsed(0);
    setTourNarration(null);
  };
  const handleTourStop = handleTourComplete;

  const handleSnapshot = () => {
    if (!rendererRef.current) return;
    // toDataURL only works with preserveDrawingBuffer:true on the Canvas.
    const url = rendererRef.current.domElement.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = url;
    a.download = `dissertation-${drug}-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const [simState, setSimState] = useState<SimSnapshot>(() => ({
    timeHours: 0,
    concentrations: new Float32Array(14),
    uncertainties: new Float32Array(14),
    releasedMg: 0,
  }));
  const onTick = useCallback((snap: SimSnapshot) => setSimState(snap), []);
  const onSeekDone = useCallback(() => setSeekTo(null), []);

  // Re-apply profile.
  useEffect(() => {
    setParams({ ...activeProfiles[profile] });
  }, [profile, restartNonce]);

  // Keyboard shortcuts for advisor/committee usability.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Ignore when typing in a form control (organ modal escape handled there).
      const tag = (e.target as HTMLElement | null)?.tagName?.toLowerCase();
      if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
      if (e.key === ' ') { e.preventDefault(); setPlaying((p) => !p); }
      else if (e.key.toLowerCase() === 'r') setRestartNonce((n) => n + 1);
      else if (e.key.toLowerCase() === 's') handleSnapshot();
      else if (e.key.toLowerCase() === 'd') {
        // Cycle drug A↔B.
        const next: DrugId = drug === 'rapamycin' ? 'semaglutide' : 'rapamycin';
        setDrug(next);
        setParams({ ...PATIENT_PROFILES_BY_DRUG[next].typical });
        setProfile('typical');
        setRestartNonce((n) => n + 1);
        if (activeTourId) handleTourStop();
      }
      else if (e.key.toLowerCase() === 't') {
        // Cycle tours within the active drug.
        const tours = TOURS.filter((tr) => tr.drug === drug);
        if (tours.length === 0) return;
        const idx = tours.findIndex((tr) => tr.id === activeTourId);
        const nextId = tours[(idx + 1) % tours.length].id;
        if (activeTourId) handleTourStop();
        handleTourStart(nextId);
      }
      else if (e.key === 'Escape' && activeTourId) handleTourStop();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeTourId, drug]);

  const handleSeek = (target: number) => {
    if (target < simState.timeHours) {
      // Backward seek: restart from t=0, then catch up forward.
      setRestartNonce((n) => n + 1);
      // Defer the forward catch-up until the restart effect lands.
      setTimeout(() => setSeekTo(target), 0);
      setPlaying(false);
    } else {
      setSeekTo(target);
      setPlaying(false);
    }
  };

  const handleReset = () => {
    setRestartNonce((n) => n + 1);
    setSeekTo(null);
    setPlaying(true);
  };

  const handleParamChange = (patch: Partial<PBPKParams>) => {
    setParams((p) => ({ ...p, ...patch }));
  };

  const selectedConcNorm = selected !== null ? simState.concentrations[selected] : 0;
  const selectedConcAbs = selectedConcNorm * C_VISUAL_MAX_BY_DRUG[drug];

  const dayString = useMemo(() => {
    const d = simState.timeHours / 24;
    if (d < 1) return `${simState.timeHours.toFixed(1)} h`;
    return `${d.toFixed(2)} d`;
  }, [simState.timeHours]);

  // Mass currently in the selected organ = C × V_ref (V_ref from compartments.ts
  // is implicit in the visual max; we use the same anchor here for consistency).
  const selectedMassMg = useMemo(() => {
    if (selected === null) return 0;
    // V_ref values, length 14, ICRP standard man. Local copy to avoid wide deps.
    const V_REF = [5.0, 1.8, 0.31, 1.45, 0.33, 1.1, 28.0, 20.0, 1.2, 3.6, 6.6, 0.18, 0.09, 3.7];
    return selectedConcAbs * V_REF[selected] * params.vdScale;
  }, [selected, selectedConcAbs, params.vdScale]);

  return (
    <div className="dissertation-viewer rounded-[var(--radius-xl)] overflow-hidden border border-[var(--glass-border)] bg-[rgba(0,0,0,0.35)] mt-[2rem]">
      <p className="lg:hidden px-3 sm:px-4 py-2.5 text-[0.82rem] leading-relaxed text-[var(--color-text-secondary)] border-b border-[var(--glass-border)] bg-[rgba(0,0,0,0.25)]">
        Demo PBPK interactiva — toque nos órgãos, avance o tempo e altere o perfil do doente nos controlos abaixo.
      </p>
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] xl:grid-cols-[1fr_320px]">
        {/* Left: 3D canvas */}
        <div className="relative h-[380px] sm:h-[500px] lg:h-[640px] w-full max-w-full overflow-hidden">
          <DrugSelector
            drug={drug}
            onChange={(d) => {
              setDrug(d);
              // Reset patient-overrides on drug switch — clHep / higuchiScale
              // are calibrated per drug (rapamycin CYP3A4 ↔ semaglutide
              // proteolytic) and would corrupt the other drug if carried over.
              setParams({ ...PATIENT_PROFILES_BY_DRUG[d].typical });
              setProfile('typical');
              setRestartNonce((n) => n + 1);
              if (activeTourId) handleTourStop();
            }}
          />
          <Canvas
            shadows
            dpr={[1, 2]}
            gl={{ antialias: true, alpha: true, preserveDrawingBuffer: true }}
            onCreated={({ gl }) => { rendererRef.current = gl; }}
          >
            <color attach="background" args={['#0b1020']} />
            <fog attach="fog" args={['#0b1020', 12, 30]} />
            <PerspectiveCamera makeDefault position={[6, 3.5, 8]} fov={42} near={0.1} far={100} />
            <OrbitControls
              ref={controlsRef as never}
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
              {drug === 'rapamycin' ? (
                <Stent
                  timeHours={simState.timeHours}
                  speed={speed}
                  reducedMotion={reducedMotion}
                />
              ) : (
                <SCDepot
                  timeHours={simState.timeHours}
                  speed={speed}
                  reducedMotion={reducedMotion}
                />
              )}
              <SimulationBridge
                drug={drug}
                params={params}
                playing={playing}
                speed={speed}
                seekTo={seekTo}
                onSeekDone={onSeekDone}
                onTick={onTick}
              />
              <CameraDirector
                tour={activeTour}
                paused={false}
                controlsRef={controlsRef}
                onKeyframe={handleTourKeyframe}
                onComplete={handleTourComplete}
              />
              {/* Tick the elapsed clock for the tour controls. */}
              <TourElapsedClock active={!!activeTour} onTick={handleTourElapsedTick} />
            </Suspense>
          </Canvas>

          {/* HUD: time + released mass (modal handles per-organ detail now). */}
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

          {/* Top-right: snapshot export */}
          <button
            type="button"
            onClick={handleSnapshot}
            title="Export the current frame as a PNG suitable for committee handouts"
            className="absolute top-4 right-4 bg-black/55 hover:bg-black/70 text-white text-xs px-3 py-1.5 rounded border border-white/15"
          >
            📸 Snapshot PNG
          </button>

          {/* Bottom-of-canvas narration overlay during tours */}
          {tourNarration && (
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 max-w-[80%] text-center text-sm text-white bg-black/65 backdrop-blur px-4 py-2 rounded shadow-lg">
              {tourNarration}
            </div>
          )}
        </div>

        {/* Right: side panel — tour controls, GUM bar, confidence gate, Hessian */}
        <aside className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-3 sm:gap-4 lg:gap-4 p-3 sm:p-4 lg:p-6 border-t lg:border-t-0 lg:border-l border-[var(--glass-border)] bg-[rgba(0,0,0,0.2)] text-white">
          <TourControls
            tours={drugTours}
            activeTourId={activeTourId}
            narration={tourNarration}
            elapsedSec={tourElapsed}
            onStart={handleTourStart}
            onStop={handleTourStop}
          />
          <hr className="hidden lg:block border-white/10" />
          <TmddPanel drug={drug} rfree={simState.rfree} dr={simState.dr} />
          <hr className="hidden lg:block border-white/10" />
          <PdReadoutPanel drug={drug} pdA={simState.pdA} pdN={simState.pdN} />
          <hr className="hidden lg:block border-white/10" />
          <ConfidenceGate drug={drug} params={params} />
          <hr className="hidden lg:block border-white/10" />
          <GumBudgetBar />
          <hr className="hidden lg:block border-white/10" />
          <HessianHeatmap />
        </aside>
      </div>

      {/* Bottom control strip */}
      <div className="grid grid-cols-1 md:grid-cols-[1.4fr_1fr_1fr] gap-4 p-4 border-t border-[var(--glass-border)] bg-[rgba(0,0,0,0.25)] text-white">
        <TimeScrubber
          timeHours={simState.timeHours}
          maxHours={MAX_SIM_HOURS}
          playing={playing}
          speed={speed}
          logScale={logScale}
          onSeek={handleSeek}
          onTogglePlay={() => setPlaying(!playing)}
          onSpeedChange={setSpeed}
          onToggleScale={() => setLogScale(!logScale)}
          onReset={handleReset}
        />
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-wide opacity-70">Patient profile</label>
          <select
            value={profile}
            onChange={(e) => setProfile(e.target.value)}
            className="bg-black/35 border border-white/15 rounded px-2 py-1.5 text-sm"
          >
            {Object.keys(activeProfiles).map((k) => (
              <option key={k} value={k}>{k}</option>
            ))}
          </select>
          <p className="text-[0.7rem] opacity-60">
            {drug === 'rapamycin'
              ? 'CYP3A4 metaboliser status changes hepatic clearance and widens the GUM cone.'
              : 'Proteolytic-clearance phenotype (Overgaard 2019 ±30%) and body-mass envelope (lean ↔ obese V_d).'}
          </p>
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-xs uppercase tracking-wide opacity-70">
            {drug === 'rapamycin'
              ? `Stent release scale (${params.higuchiScale.toFixed(2)}×)`
              : `SC depot dose (${(params.higuchiScale * 1.0).toFixed(2)} mg)`}
          </label>
          <input
            type="range"
            min={0.5}
            max={1.5}
            step={0.05}
            value={params.higuchiScale}
            onChange={(e) => handleParamChange({ higuchiScale: Number(e.target.value) })}
            className={drug === 'rapamycin' ? 'accent-amber-400' : 'accent-cyan-400'}
          />
          <p className="text-[0.7rem] opacity-60">
            {drug === 'rapamycin'
              ? <>Higuchi K<sub>H</sub> multiplier; reflects coating-thickness &amp; solubility CV per Cordis 2003 IFU.</>
              : <>Dose multiplier on the 1 mg weekly SC injection (Overgaard 2019 PK envelope, F = 0.89, k<sub>a</sub> = ln 2 / 60 h).</>}
          </p>
        </div>
      </div>

      {selected !== null && (
        <OrganDetailModal
          drug={drug}
          organIndex={selected}
          concentrationNorm={selectedConcNorm}
          concentrationMgPerL={selectedConcAbs}
          massMg={selectedMassMg}
          rfree={simState.rfree}
          dr={simState.dr}
          pdA={simState.pdA}
          pdN={simState.pdN}
          onClose={() => setSelected(null)}
        />
      )}
    </div>
  );
}
