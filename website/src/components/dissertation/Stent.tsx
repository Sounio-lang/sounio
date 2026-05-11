import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import type { InstancedMesh } from 'three';
import { Object3D, Color } from 'three';
import { STENT } from './compartments';

const PARTICLE_COUNT = 240;

interface StentProps {
  /** Simulation time in hours since stent implantation. */
  timeHours: number;
  /** Master playback speed scalar (0..N). */
  speed: number;
  /** When true, suppress the per-frame particle stream (prefers-reduced-motion). */
  reducedMotion?: boolean;
}

/**
 * Cypher drug-eluting stent at the coronary location.
 * - Tubular strut mesh (visible at the heart compartment).
 * - Particle emitter representing rapamycin elution.
 *   Higuchi diffusion: instantaneous rate ∝ K_H / (2·√t).
 *   Particle density along the upward stream encodes current release rate.
 */
export function Stent({ timeHours, speed, reducedMotion = false }: StentProps) {
  const instRef = useRef<InstancedMesh>(null);
  const dummy = useMemo(() => new Object3D(), []);
  const seeds = useMemo(
    () =>
      Array.from({ length: PARTICLE_COUNT }, () => ({
        phase: Math.random(),
        radialOffset: 0.05 + Math.random() * 0.06,
        angle: Math.random() * Math.PI * 2,
      })),
    [],
  );

  useFrame((_, delta) => {
    if (!instRef.current) return;
    if (reducedMotion) {
      // Park all particle instances at sub-pixel scale; the strut mesh remains
      // visible so the anatomical context is preserved.
      seeds.forEach((_s, i) => {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        instRef.current!.setMatrixAt(i, dummy.matrix);
      });
      instRef.current.instanceMatrix.needsUpdate = true;
      return;
    }
    // Release rate: high early, decays as 1/√t (Higuchi). Clamp at t<0.1 h
    // so the very-early divergence doesn't dominate the visual.
    const t = Math.max(0.1, timeHours);
    const rate = STENT.higuchiKH / (2 * Math.sqrt(t)); // mg / h
    const intensity = Math.min(1, rate / (STENT.higuchiKH / (2 * Math.sqrt(0.1))));

    seeds.forEach((s, i) => {
      s.phase += delta * speed * (0.25 + intensity * 0.85);
      if (s.phase > 1) s.phase -= 1;
      const y = s.phase * 1.6; // rises from stent up into the blood compartment above
      const x = STENT.position[0] + Math.cos(s.angle) * s.radialOffset;
      const z = STENT.position[2] + Math.sin(s.angle) * s.radialOffset;
      dummy.position.set(x, STENT.position[1] - STENT.length * 0.5 + y, z);
      const scale = 0.012 + (1 - s.phase) * 0.012;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      instRef.current!.setMatrixAt(i, dummy.matrix);
    });
    instRef.current.instanceMatrix.needsUpdate = true;
  });

  const ringColor = useMemo(() => new Color('#94a3b8'), []);
  const drugColor = useMemo(() => new Color('#fde047'), []);

  return (
    <group position={STENT.position}>
      {/* Strut mesh — a thin cylindrical lattice approximation */}
      <mesh>
        <cylinderGeometry
          args={[STENT.outerRadius, STENT.outerRadius, STENT.length, 16, 1, true]}
        />
        <meshStandardMaterial
          color={ringColor}
          metalness={0.85}
          roughness={0.25}
          emissive={ringColor}
          emissiveIntensity={0.05}
          transparent
          opacity={0.9}
        />
      </mesh>
      {/* Inner lumen (visual cue) */}
      <mesh>
        <cylinderGeometry
          args={[STENT.innerRadius, STENT.innerRadius, STENT.length * 1.05, 16, 1, true]}
        />
        <meshBasicMaterial color="#1e293b" transparent opacity={0.4} />
      </mesh>
      {/* Drug elution particles */}
      <instancedMesh ref={instRef} args={[undefined, undefined, PARTICLE_COUNT]}>
        <sphereGeometry args={[1, 8, 8]} />
        <meshBasicMaterial color={drugColor} transparent opacity={0.85} />
      </instancedMesh>
      {/* Callout */}
      <Html position={[0.35, 0.18, 0]} distanceFactor={8} style={{ pointerEvents: 'none' }}>
        <span
          style={{
            fontSize: '0.65rem',
            color: '#fde047',
            background: 'rgba(0,0,0,0.55)',
            padding: '2px 6px',
            borderRadius: 3,
            whiteSpace: 'nowrap',
            letterSpacing: '0.02em',
          }}
        >
          Cypher stent — 140 µg rapamycin
        </span>
      </Html>
    </group>
  );
}
