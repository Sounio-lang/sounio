import { useMemo } from 'react';
import { CatmullRomCurve3, Vector3, Color } from 'three';
import { COMPARTMENTS, FLOW_EDGES } from './compartments';

function curveBetween(a: [number, number, number], b: [number, number, number]) {
  const start = new Vector3(...a);
  const end = new Vector3(...b);
  // Bend the midpoint outward + slightly upward so vessels arc instead of
  // intersecting the body silhouette and so opposing supply/return aren't co-planar.
  const mid = start.clone().lerp(end, 0.5);
  const bend = new Vector3(0, 0.18, 0).add(
    mid.clone().normalize().multiplyScalar(0.35),
  );
  mid.add(bend);
  return new CatmullRomCurve3([start, mid, end], false, 'catmullrom', 0.4);
}

export function BloodFlowEdges() {
  const tubes = useMemo(() => {
    return FLOW_EDGES.map((edge, i) => {
      const from = COMPARTMENTS[edge.from];
      const to = COMPARTMENTS[edge.to];
      const curve = curveBetween(from.position, to.position);
      const radius = 0.012 + edge.width * 0.012;
      // Arterial supply tinted red, venous return cooler, portal axis amber.
      const color = edge.portal ? new Color('#f59e0b') : edge.from === 0 || edge.from === 5
        ? new Color('#ef4444')
        : new Color('#60a5fa');
      return { key: `${edge.from}-${edge.to}-${i}`, curve, radius, color };
    });
  }, []);

  return (
    <>
      {tubes.map((t) => (
        <mesh key={t.key}>
          <tubeGeometry args={[t.curve, 48, t.radius, 8, false]} />
          <meshStandardMaterial
            color={t.color}
            emissive={t.color}
            emissiveIntensity={0.25}
            roughness={0.6}
            metalness={0.0}
            transparent
            opacity={0.55}
          />
        </mesh>
      ))}
    </>
  );
}
