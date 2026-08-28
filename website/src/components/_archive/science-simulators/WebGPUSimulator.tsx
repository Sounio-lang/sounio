import { useEffect, useMemo, useRef, useState } from 'react';
import { SimulatorShell } from './SimulatorShell';

type LogLine = { text: string; type: 'info' | 'warn' | 'error' | 'success' };

interface WebGPUSimulatorProps {
  variant?: 'full' | 'hero';
}

function drawFallback(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  time: number,
  density: number,
  uncertainty: number,
  pulse: number,
) {
  ctx.clearRect(0, 0, width, height);

  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#06111b');
  gradient.addColorStop(0.45, '#0d2232');
  gradient.addColorStop(1, '#140c18');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  const cols = Math.max(16, Math.floor(density / 2));
  const rows = Math.max(10, Math.floor(density / 3));
  const cellW = width / cols;
  const cellH = height / rows;
  const uncertaintyNorm = uncertainty / 100;
  const pulseNorm = pulse / 100;

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const nx = x / cols;
      const ny = y / rows;
      const wave =
        Math.sin(nx * 8.0 + time * 0.0009 * (0.8 + pulseNorm)) *
        Math.cos(ny * 7.0 - time * 0.0007);
      const swirl = Math.sin((nx * nx + ny * ny) * 18.0 - time * 0.0012);
      const confidence = Math.max(0, 1 - uncertaintyNorm * 0.9 + wave * 0.25 + swirl * 0.18);
      const alpha = 0.08 + confidence * 0.6;
      const red = Math.round(22 + confidence * 118 + pulseNorm * 28);
      const green = Math.round(72 + confidence * 138);
      const blue = Math.round(136 + confidence * 98);

      ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${alpha})`;
      ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
    }
  }

  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  for (let x = 0; x <= cols; x += 1) {
    const px = x * cellW;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
  }
  for (let y = 0; y <= rows; y += 1) {
    const py = y * cellH;
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(width, py);
    ctx.stroke();
  }
}

export function WebGPUSimulator({ variant = 'full' }: WebGPUSimulatorProps) {
  const heroVariant = variant === 'hero';
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [uncertainty, setUncertainty] = useState(heroVariant ? 22 : 34);
  const [density, setDensity] = useState(heroVariant ? 68 : 48);
  const [pulse, setPulse] = useState(heroVariant ? 76 : 58);
  const [mode, setMode] = useState<'probing' | 'webgpu' | 'fallback'>('probing');
  const [logLines, setLogLines] = useState<LogLine[]>([]);

  const summary = useMemo(() => {
    const uncertaintyLabel =
      uncertainty >= 72 ? 'Epistemic widening visible' : uncertainty >= 48 ? 'Guard bands elevated' : 'Attested window stable';
    const laneCount = Math.max(32, density * 4);
    return {
      uncertaintyLabel,
      laneCount,
      lattice: `${Math.max(16, Math.floor(density / 2))}×${Math.max(10, Math.floor(density / 3))}`,
    };
  }, [density, uncertainty]);

  useEffect(() => {
    const lines: LogLine[] = [
      { text: 'Bootstrapping GPU proof visual.', type: 'info' },
      {
        text:
          mode === 'webgpu'
            ? 'WebGPU adapter acquired. Rendering on the browser GPU.'
            : mode === 'fallback'
              ? 'WebGPU unavailable. Using deterministic canvas fallback.'
              : 'Probing browser graphics capabilities.',
        type: mode === 'webgpu' ? 'success' : mode === 'fallback' ? 'warn' : 'info',
      },
      { text: `Lattice ${summary.lattice}, projected lanes ${summary.laneCount}.`, type: 'info' },
      { text: `Uncertainty gate ${uncertainty}% — ${summary.uncertaintyLabel}.`, type: uncertainty >= 72 ? 'warn' : 'success' },
      { text: `Tensor pulse ${pulse}% controlling wave transport intensity.`, type: 'info' },
    ];
    setLogLines(lines);
  }, [mode, pulse, summary, uncertainty]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let cancelled = false;
    let frame = 0;
    let observer: ResizeObserver | null = null;
    let context2d: CanvasRenderingContext2D | null = null;
    let device: any = null;
    let webgpuContext: any = null;
    let uniformBuffer: any = null;
    let pipeline: any = null;
    let bindGroup: any = null;

    const applyCanvasSize = () => {
      const parent = canvas.parentElement;
      if (!parent) return;
      const width = Math.max(320, Math.floor(parent.clientWidth));
      const height = Math.max(260, Math.floor(parent.clientHeight));
      const ratio = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    };

    const renderFallback = (time: number) => {
      if (cancelled || !context2d) return;
      drawFallback(context2d, canvas.width, canvas.height, time, density, uncertainty, pulse);
      frame = window.requestAnimationFrame(renderFallback);
    };

    const renderWebGPU = (time: number) => {
      if (cancelled || !device || !webgpuContext || !uniformBuffer || !pipeline || !bindGroup) return;
      const payload = new Float32Array([
        canvas.width,
        canvas.height,
        time * 0.001,
        density / 100,
        uncertainty / 100,
        pulse / 100,
        0,
        0,
      ]);
      device.queue.writeBuffer(uniformBuffer, 0, payload);

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: webgpuContext.getCurrentTexture().createView(),
            clearValue: { r: 0.02, g: 0.05, b: 0.08, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(6);
      pass.end();
      device.queue.submit([encoder.finish()]);
      frame = window.requestAnimationFrame(renderWebGPU);
    };

    const boot = async () => {
      applyCanvasSize();
      observer = new ResizeObserver(() => {
        applyCanvasSize();
        if (webgpuContext && device) {
          const gpu = (navigator as Navigator & { gpu?: any }).gpu;
          const format = gpu?.getPreferredCanvasFormat?.() ?? 'bgra8unorm';
          webgpuContext.configure({
            device,
            format,
            alphaMode: 'opaque',
          });
        }
      });
      observer.observe(canvas);

      const gpu = (navigator as Navigator & { gpu?: any }).gpu;
      if (gpu?.requestAdapter) {
        try {
          const adapter = await gpu.requestAdapter();
          if (adapter && !cancelled) {
            device = await adapter.requestDevice();
            webgpuContext = canvas.getContext('webgpu' as any) as any;
            if (webgpuContext) {
              const format = gpu.getPreferredCanvasFormat?.() ?? 'bgra8unorm';
              webgpuContext.configure({
                device,
                format,
                alphaMode: 'opaque',
              });

              const shaderModule = device.createShaderModule({
                code: `
struct Uniforms {
  resolution : vec2<f32>,
  time : f32,
  density : f32,
  uncertainty : f32,
  pulse : f32,
  _pad0 : f32,
  _pad1 : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var<uniform> u : Uniforms;

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index : u32) -> VertexOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0),
  );
  let pos = positions[vertex_index];
  var out : VertexOut;
  out.position = vec4<f32>(pos, 0.0, 1.0);
  out.uv = pos * 0.5 + vec2<f32>(0.5, 0.5);
  return out;
}

fn field(uv: vec2<f32>) -> f32 {
  let scale = 10.0 + u.density * 26.0;
  let p = uv * scale;
  let t = u.time;
  let wave = sin(p.x + t * (0.7 + u.pulse * 1.6)) * cos(p.y - t * 0.6);
  let swirl = sin(length(p - vec2<f32>(4.0, 3.0)) * 2.1 - t * 1.4);
  let ridge = cos((p.x * 0.7 - p.y * 0.45) + t * 0.45 + u.pulse * 2.2);
  return wave * 0.45 + swirl * 0.35 + ridge * 0.2;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let f = field(in.uv);
  let uncertaintyMask = 1.0 - u.uncertainty * 0.78;
  let confidence = clamp(0.5 + f * uncertaintyMask, 0.0, 1.0);
  let lattice = abs(fract(in.uv.x * (8.0 + u.density * 24.0)) - 0.5)
    + abs(fract(in.uv.y * (6.0 + u.density * 18.0)) - 0.5);
  let grid = smoothstep(0.14, 0.02, lattice);

  let base = vec3<f32>(0.03, 0.08, 0.12);
  let teal = vec3<f32>(0.09, 0.62, 0.74);
  let gold = vec3<f32>(0.89, 0.69, 0.28);
  let rose = vec3<f32>(0.71, 0.24, 0.46);

  var color = mix(base, teal, confidence);
  color = mix(color, gold, grid * 0.34);
  color = mix(color, rose, clamp(u.uncertainty * 0.65 - confidence * 0.18, 0.0, 0.5));

  return vec4<f32>(color, 1.0);
}
                `,
              });

              uniformBuffer = device.createBuffer({
                size: 8 * 4,
                usage: 64 | 8,
              });

              pipeline = device.createRenderPipeline({
                layout: 'auto',
                vertex: {
                  module: shaderModule,
                  entryPoint: 'vs_main',
                },
                fragment: {
                  module: shaderModule,
                  entryPoint: 'fs_main',
                  targets: [{ format }],
                },
                primitive: {
                  topology: 'triangle-list',
                },
              });

              bindGroup = device.createBindGroup({
                layout: pipeline.getBindGroupLayout(0),
                entries: [
                  {
                    binding: 0,
                    resource: { buffer: uniformBuffer },
                  },
                ],
              });

              if (!cancelled) {
                setMode('webgpu');
                frame = window.requestAnimationFrame(renderWebGPU);
                return;
              }
            }
          }
        } catch {
          // Fall through to deterministic canvas rendering.
        }
      }

      context2d = canvas.getContext('2d', { alpha: false });
      if (!context2d) return;
      if (!cancelled) {
        setMode('fallback');
        frame = window.requestAnimationFrame(renderFallback);
      }
    };

    void boot();

    return () => {
      cancelled = true;
      if (observer) observer.disconnect();
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, [density, pulse, uncertainty]);

  const controls = (
    <>
      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Uncertainty gate
          <span className="text-[var(--color-text-primary)]">{uncertainty}%</span>
        </label>
        <input
          type="range"
          min="8"
          max="92"
          value={uncertainty}
          onChange={(event) => setUncertainty(Number(event.target.value))}
          className="w-full accent-[var(--color-accent-gold)]"
        />
        <p className="text-[0.8rem] text-[var(--color-text-tertiary)] leading-snug mt-1">
          Push the field from an attested regime into visible epistemic widening.
        </p>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Lattice density
          <span className="text-[var(--color-text-primary)]">{summary.lattice}</span>
        </label>
        <input
          type="range"
          min="20"
          max="88"
          value={density}
          onChange={(event) => setDensity(Number(event.target.value))}
          className="w-full accent-[var(--color-accent-blue)]"
        />
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-[var(--color-text-secondary)] text-sm font-semibold flex justify-between">
          Tensor pulse
          <span className="text-[var(--color-text-primary)]">{pulse}%</span>
        </label>
        <input
          type="range"
          min="12"
          max="96"
          value={pulse}
          onChange={(event) => setPulse(Number(event.target.value))}
          className="w-full accent-[var(--color-accent-green)]"
        />
      </div>

      <div className="grid grid-cols-2 gap-3 mt-2">
        <div className="rounded-[var(--radius-md)] border border-[var(--glass-border)] bg-[rgba(255,255,255,0.03)] p-3">
          <div className="text-[0.72rem] uppercase tracking-[0.08em] text-[var(--color-text-tertiary)]">Mode</div>
          <div className="mt-1 text-sm font-semibold text-[var(--color-text-primary)]">
            {mode === 'webgpu' ? 'WebGPU live' : mode === 'fallback' ? 'Canvas fallback' : 'Probing'}
          </div>
        </div>
        <div className="rounded-[var(--radius-md)] border border-[var(--glass-border)] bg-[rgba(255,255,255,0.03)] p-3">
          <div className="text-[0.72rem] uppercase tracking-[0.08em] text-[var(--color-text-tertiary)]">Projected lanes</div>
          <div className="mt-1 text-sm font-semibold text-[var(--color-text-primary)]">{summary.laneCount}</div>
        </div>
      </div>
    </>
  );

  const visualization = (
    <div className="w-full h-full min-h-[320px] flex items-center justify-center relative">
      <div className="absolute left-4 top-4 z-10 rounded-full border border-[rgba(255,255,255,0.12)] bg-[rgba(7,10,18,0.72)] px-3 py-1 text-[0.72rem] font-semibold uppercase tracking-[0.08em] text-[var(--color-text-secondary)]">
        {mode === 'webgpu' ? 'Browser GPU active' : mode === 'fallback' ? 'Graceful software fallback' : 'Capability probe'}
      </div>
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full rounded-[var(--radius-lg)]" />
      <div className="absolute bottom-4 right-4 z-10 rounded-[var(--radius-md)] border border-[rgba(255,255,255,0.12)] bg-[rgba(7,10,18,0.72)] px-3 py-2 text-[0.76rem] text-[var(--color-text-secondary)]">
        Confidence gradients, provenance lattice, tensor pulse
      </div>
    </div>
  );

  if (heroVariant) {
    return (
      <div className="hero-gpu-shell glass glass-specular rounded-[var(--radius-2xl)] overflow-hidden border border-[var(--glass-border)] relative z-10">
        <div className="relative min-h-[420px] overflow-hidden bg-[linear-gradient(160deg,rgba(4,14,24,0.92),rgba(8,19,34,0.82))]">
          <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" />
          <div className="absolute left-4 top-4 z-10 rounded-full border border-[rgba(255,255,255,0.12)] bg-[rgba(7,10,18,0.72)] px-3 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-[var(--color-text-secondary)]">
            {mode === 'webgpu' ? 'Browser GPU active' : mode === 'fallback' ? 'Canvas fallback' : 'Capability probe'}
          </div>
          <div className="absolute inset-x-0 bottom-0 z-10 bg-[linear-gradient(180deg,transparent,rgba(3,8,16,0.94))] p-5">
            <div className="grid gap-4 md:grid-cols-[1.2fr_0.8fr] md:items-end">
              <div className="grid gap-2">
                <span className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-[var(--color-accent-gold-soft)]">
                  Live proof object
                </span>
                <h3 className="text-[1.35rem] font-semibold text-[var(--color-text-primary)]">
                  Provenance field with visible uncertainty transport
                </h3>
                <p className="max-w-[42ch] text-[0.9rem] text-[var(--color-text-secondary)]">
                  A storefront-grade GPU object: WebGPU when the browser permits it, deterministic fallback when it does not.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-[var(--radius-lg)] border border-[rgba(255,255,255,0.12)] bg-[rgba(5,10,18,0.68)] px-3 py-3">
                  <div className="text-[0.68rem] uppercase tracking-[0.1em] text-[var(--color-text-tertiary)]">Lattice</div>
                  <div className="mt-1 text-[0.95rem] font-semibold text-[var(--color-text-primary)]">{summary.lattice}</div>
                </div>
                <div className="rounded-[var(--radius-lg)] border border-[rgba(255,255,255,0.12)] bg-[rgba(5,10,18,0.68)] px-3 py-3">
                  <div className="text-[0.68rem] uppercase tracking-[0.1em] text-[var(--color-text-tertiary)]">Projected lanes</div>
                  <div className="mt-1 text-[0.95rem] font-semibold text-[var(--color-text-primary)]">{summary.laneCount}</div>
                </div>
                <div className="rounded-[var(--radius-lg)] border border-[rgba(255,255,255,0.12)] bg-[rgba(5,10,18,0.68)] px-3 py-3">
                  <div className="text-[0.68rem] uppercase tracking-[0.1em] text-[var(--color-text-tertiary)]">Gate</div>
                  <div className="mt-1 text-[0.95rem] font-semibold text-[var(--color-text-primary)]">{uncertainty}%</div>
                </div>
                <div className="rounded-[var(--radius-lg)] border border-[rgba(255,255,255,0.12)] bg-[rgba(5,10,18,0.68)] px-3 py-3">
                  <div className="text-[0.68rem] uppercase tracking-[0.1em] text-[var(--color-text-tertiary)]">Pulse</div>
                  <div className="mt-1 text-[0.95rem] font-semibold text-[var(--color-text-primary)]">{pulse}%</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <SimulatorShell
      title="WebGPU Provenance Field"
      description="A live browser-side field that turns uncertainty, lattice density, and tensor pulse into an inspectable GPU surface. On supported browsers it runs through WebGPU; elsewhere it falls back without breaking the showcase."
      controls={controls}
      visualization={visualization}
      logLines={logLines}
      isHalted={false}
    />
  );
}
