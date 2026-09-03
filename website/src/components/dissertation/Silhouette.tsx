import { useMemo } from 'react';
import { CatmullRomCurve3, Vector3, Color } from 'three';

/**
 * Stylised translucent human silhouette giving spatial context to the 14
 * compartment spheres. Built entirely from primitives — no licensed meshes.
 *
 * Convention: Y up, +Z forward (anterior view by default). Coordinates are in
 * the same scene units as `compartments.ts` (≈ half-body heights).
 */
export function Silhouette() {
  const skinColor = useMemo(() => new Color('#cbd5e1'), []);
  const spineCurve = useMemo(() => {
    return new CatmullRomCurve3(
      [
        new Vector3(0, 6.4, -0.05),
        new Vector3(0, 5.1, -0.10),
        new Vector3(0, 3.0, -0.10),
        new Vector3(0, 1.0, -0.05),
        new Vector3(0, -1.2, 0.0),
      ],
      false,
      'catmullrom',
      0.3,
    );
  }, []);

  return (
    <group>
      {/* Head */}
      <mesh position={[0, 6.4, 0]}>
        <sphereGeometry args={[0.85, 32, 24]} />
        <meshStandardMaterial
          color={skinColor}
          transparent
          opacity={0.10}
          roughness={0.7}
          depthWrite={false}
        />
      </mesh>
      {/* Neck */}
      <mesh position={[0, 5.6, 0]}>
        <cylinderGeometry args={[0.32, 0.42, 0.5, 16]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.10} roughness={0.7} depthWrite={false} />
      </mesh>
      {/* Torso */}
      <mesh position={[0, 3.0, 0]}>
        <cylinderGeometry args={[1.45, 1.25, 4.4, 24, 1, false]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.08} roughness={0.7} depthWrite={false} />
      </mesh>
      {/* Pelvis */}
      <mesh position={[0, 0.4, 0]}>
        <cylinderGeometry args={[1.25, 1.05, 1.0, 24]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.10} roughness={0.7} depthWrite={false} />
      </mesh>
      {/* Upper arms */}
      <mesh position={[1.8, 2.6, 0]} rotation={[0, 0, -0.05]}>
        <cylinderGeometry args={[0.30, 0.32, 2.6, 12]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.08} roughness={0.7} depthWrite={false} />
      </mesh>
      <mesh position={[-1.8, 2.6, 0]} rotation={[0, 0, 0.05]}>
        <cylinderGeometry args={[0.30, 0.32, 2.6, 12]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.08} roughness={0.7} depthWrite={false} />
      </mesh>
      {/* Thighs */}
      <mesh position={[0.55, -1.4, 0]}>
        <cylinderGeometry args={[0.38, 0.34, 2.4, 12]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.08} roughness={0.7} depthWrite={false} />
      </mesh>
      <mesh position={[-0.55, -1.4, 0]}>
        <cylinderGeometry args={[0.38, 0.34, 2.4, 12]} />
        <meshStandardMaterial color={skinColor} transparent opacity={0.08} roughness={0.7} depthWrite={false} />
      </mesh>
      {/* Spine (visual cue, also implicit "bone" compartment anchor) */}
      <mesh>
        <tubeGeometry args={[spineCurve, 48, 0.05, 8, false]} />
        <meshStandardMaterial color="#e5e7eb" transparent opacity={0.25} roughness={0.6} />
      </mesh>
    </group>
  );
}
