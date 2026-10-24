import { useMemo, useRef } from 'react';
import { Html } from '@react-three/drei';
import type { Mesh } from 'three';
import { Color } from 'three';
import { COMPARTMENTS } from './compartments';
import type { Compartment } from './compartments';

interface CompartmentSphereProps {
  c: Compartment;
  concentration: number;       // 0..1 normalized
  uncertaintyRadius: number;   // GUM cone half-width, in scene units
  selected: boolean;
  onSelect: (index: number) => void;
}

function CompartmentSphere({ c, concentration, uncertaintyRadius, selected, onSelect }: CompartmentSphereProps) {
  const meshRef = useRef<Mesh>(null);

  // Color blend: base organ color (empty) → red (peak concentration).
  const color = useMemo(() => {
    const base = new Color(c.color);
    const peak = new Color('#dc2626');
    return base.lerp(peak, Math.min(1, Math.max(0, concentration)));
  }, [c.color, concentration]);

  return (
    <group position={c.position}>
      {/* GUM uncertainty cone — translucent shell */}
      {uncertaintyRadius > 0 && (
        <mesh>
          <sphereGeometry args={[c.radius + uncertaintyRadius, 24, 24]} />
          <meshBasicMaterial color={color} transparent opacity={0.10} depthWrite={false} />
        </mesh>
      )}
      {/* Organ sphere */}
      <mesh
        ref={meshRef}
        onClick={(e) => { e.stopPropagation(); onSelect(c.index); }}
        onPointerOver={(e) => { e.stopPropagation(); document.body.style.cursor = 'pointer'; }}
        onPointerOut={() => { document.body.style.cursor = 'auto'; }}
      >
        <sphereGeometry args={[c.radius, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={selected ? 0.45 : 0.18}
          roughness={0.42}
          metalness={0.05}
          transparent
          opacity={selected ? 0.95 : 0.82}
        />
      </mesh>
      {/* Label */}
      <Html position={[0, c.radius + 0.18, 0]} center distanceFactor={8} style={{ pointerEvents: 'none' }}>
        <span
          style={{
            fontSize: '0.7rem',
            fontWeight: 600,
            color: selected ? '#fbbf24' : 'rgba(255,255,255,0.85)',
            textShadow: '0 1px 2px rgba(0,0,0,0.85)',
            whiteSpace: 'nowrap',
            letterSpacing: '0.02em',
          }}
        >
          {c.label}
        </span>
      </Html>
    </group>
  );
}

interface CompartmentsProps {
  concentrations: Float32Array; // length 14, 0..1 normalized
  uncertainties: Float32Array;  // length 14, 0..1 normalized
  selected: number | null;
  onSelect: (index: number) => void;
}

export function Compartments({ concentrations, uncertainties, selected, onSelect }: CompartmentsProps) {
  return (
    <>
      {COMPARTMENTS.map((c) => (
        <CompartmentSphere
          key={c.index}
          c={c}
          concentration={concentrations[c.index] ?? 0}
          uncertaintyRadius={(uncertainties[c.index] ?? 0) * 0.45}
          selected={selected === c.index}
          onSelect={onSelect}
        />
      ))}
    </>
  );
}
