// G-ε-3: subcutaneous depot for semaglutide. Lower-abdomen ellipsoid with a
// cyan first-order release particle stream toward the blood compartment.
// Mirrors Stent.tsx's structure (instanced particles, reduced-motion guard,
// Html callout) but with semaglutide's SC depot kinetics:
//   release_rate(t) = F · Dose · k_a · exp(-k_a · t)    mg/h
// Default: F = 0.89, Dose = 1 mg, k_a = ln 2 / 60 h ≈ 0.01155 /h.

import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import type { InstancedMesh } from 'three';
import { Object3D, Color } from 'three';

const PARTICLE_COUNT = 200;
const KA = Math.LN2 / 60.0;          // /h, ln 2 / 60 h
const F  = 0.89;
const DOSE_MG = 1.0;

const DEPOT_POSITION: [number, number, number] = [0.55, 0.7, 0.95];  // lower-right abdomen surface
const DEPOT_RX = 0.18;
const DEPOT_RY = 0.10;
const DEPOT_RZ = 0.18;

interface SCDepotProps {
  /** Simulation time, hours since SC injection. */
  timeHours: number;
  /** Master playback speed scalar. */
  speed: number;
  /** prefers-reduced-motion suppresses particle motion. */
  reducedMotion?: boolean;
}

export function SCDepot({ timeHours, speed, reducedMotion = false }: SCDepotProps) {
  const instRef = useRef<InstancedMesh>(null);
  const dummy = useMemo(() => new Object3D(), []);
  const seeds = useMemo(
    () =>
      Array.from({ length: PARTICLE_COUNT }, () => ({
        phase: Math.random(),
        radialOffset: 0.05 + Math.random() * 0.07,
        angle: Math.random() * Math.PI * 2,
      })),
    [],
  );

  useFrame((_, delta) => {
    if (!instRef.current) return;
    if (reducedMotion) {
      seeds.forEach((_s, i) => {
        dummy.position.set(0, 0, 0);
        dummy.scale.setScalar(0);
        dummy.updateMatrix();
        instRef.current!.setMatrixAt(i, dummy.matrix);
      });
      instRef.current.instanceMatrix.needsUpdate = true;
      return;
    }
    // First-order absorption: rate(0) = F·Dose·ka ≈ 0.0103 mg/h; rate(60h) ≈ half.
    const t = Math.max(0, timeHours);
    const rate = F * DOSE_MG * KA * Math.exp(-KA * t);
    const rate0 = F * DOSE_MG * KA;
    const intensity = Math.min(1, rate / rate0);

    seeds.forEach((s, i) => {
      s.phase += delta * speed * (0.18 + intensity * 0.55);
      if (s.phase > 1) s.phase -= 1;
      // Particles drift medially-up toward blood compartment.
      const y = s.phase * 1.8;
      const xWobble = Math.cos(s.angle) * s.radialOffset * (1 - 0.4 * s.phase);
      const zWobble = Math.sin(s.angle) * s.radialOffset * (1 - 0.4 * s.phase);
      // Slight inward drift (toward x=0) as particles rise.
      const xDrift = -s.phase * 0.4;
      const zDrift = -s.phase * 0.4;
      dummy.position.set(
        DEPOT_POSITION[0] + xWobble + xDrift,
        DEPOT_POSITION[1] + y,
        DEPOT_POSITION[2] + zWobble + zDrift,
      );
      const scale = 0.014 + (1 - s.phase) * 0.012;
      dummy.scale.setScalar(scale);
      dummy.updateMatrix();
      instRef.current!.setMatrixAt(i, dummy.matrix);
    });
    instRef.current.instanceMatrix.needsUpdate = true;
  });

  const skinColor = useMemo(() => new Color('#fbbf24'), []);
  const drugColor = useMemo(() => new Color('#22d3ee'), []);

  return (
    <group position={DEPOT_POSITION}>
      {/* SC depot — ellipsoid bulge under the skin */}
      <mesh scale={[DEPOT_RX, DEPOT_RY, DEPOT_RZ]}>
        <sphereGeometry args={[1, 24, 16]} />
        <meshStandardMaterial
          color={skinColor}
          roughness={0.7}
          metalness={0.05}
          transparent
          opacity={0.55}
        />
      </mesh>
      {/* Inner reservoir cue */}
      <mesh scale={[DEPOT_RX * 0.6, DEPOT_RY * 0.6, DEPOT_RZ * 0.6]}>
        <sphereGeometry args={[1, 16, 12]} />
        <meshBasicMaterial color="#0e7490" transparent opacity={0.55} />
      </mesh>
      {/* Particle stream */}
      <instancedMesh ref={instRef} args={[undefined, undefined, PARTICLE_COUNT]}>
        <sphereGeometry args={[1, 8, 8]} />
        <meshBasicMaterial color={drugColor} transparent opacity={0.85} />
      </instancedMesh>
      <Html position={[0.4, 0.05, 0]} distanceFactor={8} style={{ pointerEvents: 'none' }}>
        <span
          style={{
            fontSize: '0.65rem',
            color: '#22d3ee',
            background: 'rgba(0,0,0,0.55)',
            padding: '2px 6px',
            borderRadius: 3,
            whiteSpace: 'nowrap',
            letterSpacing: '0.02em',
          }}
        >
          SC depot — 1 mg semaglutide (k_a = ln 2 / 60 h)
        </span>
      </Html>
    </group>
  );
}
