import { useEffect, useRef, useState } from 'react';

type Mode = 'loading' | 'ready' | 'fallback';

const shaderCode = /* wgsl */ `
struct Uniforms {
  resolution: vec2f,
  time: f32,
  density: f32,
  pointer: vec2f,
  pulse: f32,
  pad: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOut {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -3.0),
    vec2f(-1.0, 1.0),
    vec2f(3.0, 1.0)
  );

  let pos = positions[vertexIndex];
  var out: VertexOut;
  out.position = vec4f(pos, 0.0, 1.0);
  out.uv = pos * 0.5 + vec2f(0.5, 0.5);
  return out;
}

fn ring(uv: vec2f, center: vec2f, radius: f32, width: f32) -> f32 {
  let d = abs(length(uv - center) - radius);
  return 1.0 - smoothstep(width, width * 2.2, d);
}

fn glow(uv: vec2f, center: vec2f, spread: f32) -> f32 {
  return exp(-length(uv - center) * spread);
}

@fragment
fn fsMain(input: VertexOut) -> @location(0) vec4f {
  let res = max(u.resolution, vec2f(1.0, 1.0));
  var uv = input.uv * 2.0 - vec2f(1.0, 1.0);
  uv.x *= res.x / res.y;

  let t = u.time * 0.42;
  let pointer = vec2f((u.pointer.x * 2.0 - 1.0) * (res.x / res.y), u.pointer.y * 2.0 - 1.0);

  let orbitA = vec2f(cos(t * 1.3) * 0.42, sin(t * 1.1) * 0.24);
  let orbitB = vec2f(cos(t * 0.8 + 1.6) * 0.28, sin(t * 1.5 + 0.9) * 0.36);
  let orbitC = vec2f(cos(t * 1.7 + 3.14) * 0.18, sin(t * 0.9 + 2.4) * 0.2);

  let lattice =
    sin((uv.x + t * 0.08) * 7.5) * 0.18 +
    cos((uv.y - t * 0.11) * 8.8) * 0.17 +
    sin((uv.x + uv.y) * 4.2 - t * 0.9) * 0.14;

  let contour =
    ring(uv, orbitA, 0.18 + sin(t * 1.1) * 0.03, 0.012) +
    ring(uv, orbitB, 0.14 + cos(t * 1.3) * 0.03, 0.01) +
    ring(uv, orbitC, 0.1 + sin(t * 1.7) * 0.02, 0.008);

  let energy =
    glow(uv, orbitA, 9.5) +
    glow(uv, orbitB, 11.5) * 0.9 +
    glow(uv, orbitC, 15.0) * 0.7 +
    glow(uv, pointer * 0.4, 8.0) * 0.55;

  let wave = sin(length(uv) * 14.0 - t * 5.0 + atan2(uv.y, uv.x) * 2.0);
  let radial = smoothstep(1.22, 0.06, length(uv));
  let veil = smoothstep(1.18, 0.12, length(uv - pointer * 0.12));

  let navy = vec3f(0.02, 0.05, 0.13);
  let deep = vec3f(0.04, 0.11, 0.23);
  let teal = vec3f(0.16, 0.65, 0.7);
  let gold = vec3f(0.84, 0.71, 0.35);
  let ivory = vec3f(0.98, 0.96, 0.93);

  let field = clamp(energy * 0.7 + contour * 0.5 + lattice * 0.5 + wave * 0.08, 0.0, 1.6);
  var color = mix(navy, deep, radial);
  color = mix(color, teal, clamp(field * 0.5, 0.0, 0.75));
  color = mix(color, gold, clamp(contour * 0.55 + energy * 0.32 + u.pulse * 0.08, 0.0, 0.68));
  color += ivory * glow(uv, pointer * 0.35, 16.0) * 0.11;
  color *= radial * veil;
  color += vec3f(0.012, 0.018, 0.03);

  return vec4f(color, 1.0);
}
`;

export default function WebGPUHeroCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [mode, setMode] = useState<Mode>('loading');

  useEffect(() => {
    let mounted = true;
    let frame = 0;
    let resizeObserver: ResizeObserver | null = null;
    const cleanupHandlers: Array<() => void> = [];

    const canvas = canvasRef.current;
    if (!canvas) return;

    const gpu = (navigator as Navigator & { gpu?: any }).gpu;
    if (!gpu) {
      setMode('fallback');
      return;
    }

    const setup = async () => {
      try {
        const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (!adapter) throw new Error('adapter-unavailable');

        const device = await adapter.requestDevice();
        const context = canvas.getContext('webgpu') as any;
        if (!context) throw new Error('context-unavailable');

        const format = gpu.getPreferredCanvasFormat();
        const uniformBuffer = device.createBuffer({
          size: 32,
          usage: 0x40 | 0x04, // UNIFORM | COPY_DST
        });

        const shaderModule = device.createShaderModule({ code: shaderCode });
        const pipeline = device.createRenderPipeline({
          layout: 'auto',
          vertex: {
            module: shaderModule,
            entryPoint: 'vsMain',
          },
          fragment: {
            module: shaderModule,
            entryPoint: 'fsMain',
            targets: [{ format }],
          },
          primitive: {
            topology: 'triangle-list',
          },
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
        });

        const pointer = { x: 0.78, y: 0.32 };
        const start = performance.now();

        const resize = () => {
          const rect = canvas.getBoundingClientRect();
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          const width = Math.max(1, Math.floor(rect.width * dpr));
          const height = Math.max(1, Math.floor(rect.height * dpr));

          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }

          context.configure({
            device,
            format,
            alphaMode: 'opaque',
          });
        };

        resizeObserver = new ResizeObserver(resize);
        resizeObserver.observe(canvas);
        resize();

        const handlePointer = (event: PointerEvent) => {
          const rect = canvas.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0) return;
          pointer.x = (event.clientX - rect.left) / rect.width;
          pointer.y = (event.clientY - rect.top) / rect.height;
        };

        canvas.addEventListener('pointermove', handlePointer, { passive: true });
        cleanupHandlers.push(() => canvas.removeEventListener('pointermove', handlePointer));

        const render = (now: number) => {
          if (!mounted) return;

          const elapsed = (now - start) / 1000;
          const pulse = 0.5 + Math.sin(elapsed * 0.9) * 0.5;

          const uniforms = new Float32Array([
            canvas.width,
            canvas.height,
            elapsed,
            1,
            pointer.x,
            pointer.y,
            pulse,
            0,
          ]);

          device.queue.writeBuffer(uniformBuffer, 0, uniforms);

          const encoder = device.createCommandEncoder();
          const view = context.getCurrentTexture().createView();
          const pass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0.015, g: 0.03, b: 0.075, a: 1 },
              },
            ],
          });

          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);
          pass.draw(3, 1, 0, 0);
          pass.end();

          device.queue.submit([encoder.finish()]);
          frame = window.requestAnimationFrame(render);
        };

        if (mounted) {
          setMode('ready');
          frame = window.requestAnimationFrame(render);
        }
      } catch {
        if (mounted) setMode('fallback');
      }
    };

    void setup();

    return () => {
      mounted = false;
      window.cancelAnimationFrame(frame);
      resizeObserver?.disconnect();
      for (const handler of cleanupHandlers) handler();
    };
  }, []);

  return (
    <div className="relative h-full w-full overflow-hidden rounded-[1.9rem] bg-[radial-gradient(circle_at_20%_15%,rgba(214,179,90,0.16),transparent_34%),linear-gradient(140deg,#030a16,#071a35_44%,#0a2141)]">
      <canvas
        ref={canvasRef}
        className="h-full w-full"
        aria-label="Live WebGPU field"
      />
      <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(180deg,rgba(255,255,255,0.08),transparent_16%,transparent_80%,rgba(255,255,255,0.08))]" />
      <div className="pointer-events-none absolute inset-0 opacity-30 [background-image:linear-gradient(rgba(255,255,255,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.05)_1px,transparent_1px)] [background-size:56px_56px]" />

      {mode !== 'ready' && (
        <div className="pointer-events-none absolute inset-0 flex items-end justify-start p-5">
          <div className="rounded-full border border-white/12 bg-[#07132a]/70 px-4 py-2 text-[0.72rem] uppercase tracking-[0.18em] text-[var(--color-accent-gold-soft)] backdrop-blur-xl">
            {mode === 'loading' ? 'Initializing WebGPU' : 'WebGPU unavailable'}
          </div>
        </div>
      )}
    </div>
  );
}
