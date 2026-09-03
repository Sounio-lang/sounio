import { useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import type { Tour, TourKeyframe } from './tours';
import { lerpKeyframe } from './tours';
import { Vector3 } from 'three';

interface CameraDirectorProps {
  /** When set, drives the camera + OrbitControls.target along the tour. */
  tour: Tour | null;
  /** Callback fired when the tour passes its last keyframe. */
  onComplete: () => void;
  /** Called per keyframe boundary with the keyframe's optional scene-state hints. */
  onKeyframe: (k: TourKeyframe) => void;
  /** Ref to the drei OrbitControls instance — controls.target is what we lerp. */
  controlsRef: React.RefObject<{ target: Vector3; update: () => void } | null>;
  /** Pause / freeze the elapsed-time accumulator. */
  paused: boolean;
}

/**
 * In-Canvas component (must live inside <Canvas>) that walks the camera along
 * a Tour's keyframes. Smoothstep-interpolates camera position + OrbitControls
 * target. Honours prefers-reduced-motion via the parent's `paused` flag during
 * the catch-up moments and emits per-keyframe scene-state hints upward.
 */
export function CameraDirector({ tour, onComplete, onKeyframe, controlsRef, paused }: CameraDirectorProps) {
  const { camera } = useThree();
  const elapsedRef = useRef(0);
  const lastKeyframeRef = useRef(-1);

  // Reset elapsed on tour change.
  useEffect(() => {
    elapsedRef.current = 0;
    lastKeyframeRef.current = -1;
  }, [tour?.id]);

  useFrame((_, delta) => {
    if (!tour || paused) return;
    elapsedRef.current += delta;
    const t = elapsedRef.current;

    if (t >= tour.duration) {
      // Snap to last keyframe and complete.
      const last = tour.keyframes[tour.keyframes.length - 1];
      camera.position.set(...last.camera);
      if (controlsRef.current) {
        controlsRef.current.target.set(...last.target);
        controlsRef.current.update();
      }
      onComplete();
      return;
    }

    // Find bracketing keyframes.
    let i = 0;
    while (i < tour.keyframes.length - 1 && tour.keyframes[i + 1].t <= t) i += 1;
    const a = tour.keyframes[i];
    const b = tour.keyframes[Math.min(i + 1, tour.keyframes.length - 1)];

    // Fire keyframe callback on crossing (i increased).
    if (i !== lastKeyframeRef.current) {
      lastKeyframeRef.current = i;
      onKeyframe(a);
    }

    const { camera: cam, target } = lerpKeyframe(a, b, t);
    camera.position.set(...cam);
    if (controlsRef.current) {
      controlsRef.current.target.set(...target);
      controlsRef.current.update();
    }
  });

  return null;
}
