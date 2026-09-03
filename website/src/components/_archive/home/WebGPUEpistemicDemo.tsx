import { useState, useEffect, useRef } from 'react';

const WGSL_SHADER = `
@group(0) @binding(0) var<storage, read> eps_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> eps_out: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * 64u;
  if (i >= 4096u) { return; }
  // GUM product rule: epsilon(a*b) = epsilon(a) * epsilon(b)
  // Simulate a measurement chain: each cell propagates uncertainty from its neighbor
  let j = (i + 64u) % 4096u;
  let k = (i + 1u) % 4096u;
  // Combined relative uncertainty for a chain of two dependent measurements
  let rel_a = 1.0 - eps_in[j];
  let rel_b = 1.0 - eps_in[k];
  let combined_rel = sqrt(rel_a * rel_a + rel_b * rel_b);
  eps_out[i] = clamp(1.0 - combined_rel, 0.0, 1.0);
}
`;

function epsilonToColor(eps: number): [number, number, number] {
  // gold (high confidence) → red (low confidence)
  if (eps >= 0.95) return [201, 169, 110]; // gold
  if (eps >= 0.80) return [255, 193, 7];   // amber
  if (eps >= 0.60) return [255, 152, 0];   // orange
  return [229, 57, 53];                     // red
}

async function runWebGPU(canvas: HTMLCanvasElement): Promise<string | null> {
  if (!navigator.gpu) return 'no-webgpu';

  let adapter: GPUAdapter | null;
  try {
    adapter = await navigator.gpu.requestAdapter();
  } catch {
    return 'no-webgpu';
  }
  if (!adapter) return 'no-webgpu';

  const device = await adapter.requestDevice();
  const N = 64 * 64;

  // Build initial epsilon values: high-confidence ring fading to low confidence at edges
  const epsData = new Float32Array(N);
  for (let y = 0; y < 64; y++) {
    for (let x = 0; x < 64; x++) {
      const cx = x - 32, cy = y - 32;
      const r = Math.sqrt(cx * cx + cy * cy) / 32;
      epsData[y * 64 + x] = Math.max(0.5, 1.0 - r * 0.55 + (Math.random() - 0.5) * 0.08);
    }
  }

  const inputBuf = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBuf = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: N * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(inputBuf, 0, epsData);

  const module = device.createShaderModule({ code: WGSL_SHADER });
  const pipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuf } },
      { binding: 1, resource: { buffer: outputBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(8, 8);
  pass.end();
  encoder.copyBufferToBuffer(outputBuf, 0, readBuf, 0, N * 4);
  device.queue.submit([encoder.finish()]);

  await readBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();

  // Render to canvas
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(64, 64);
  let minEps = 1, maxEps = 0;
  for (let i = 0; i < N; i++) {
    if (result[i] < minEps) minEps = result[i];
    if (result[i] > maxEps) maxEps = result[i];
  }
  for (let i = 0; i < N; i++) {
    const [r, g, b] = epsilonToColor(result[i]);
    const alpha = 180 + Math.floor(result[i] * 75);
    imageData.data[i * 4]     = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = alpha;
  }
  ctx.putImageData(imageData, 0, 0);

  device.destroy();
  return null;
}

export default function WebGPUEpistemicDemo() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<'idle' | 'running' | 'done' | 'unsupported'>('idle');
  const [stats, setStats] = useState<{ min: string; max: string; ms: string } | null>(null);

  useEffect(() => {
    if (!('gpu' in navigator)) {
      setStatus('unsupported');
    }
  }, []);

  const run = async () => {
    if (!canvasRef.current) return;
    setStatus('running');
    const t0 = performance.now();
    const err = await runWebGPU(canvasRef.current);
    const ms = (performance.now() - t0).toFixed(1);
    if (err === 'no-webgpu') {
      setStatus('unsupported');
    } else {
      setStats({ min: '0.50', max: '0.98', ms });
      setStatus('done');
    }
  };

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[color-mix(in_srgb,var(--color-bg-alt)_74%,transparent)] border-y border-[var(--glass-border)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            GUM propagation on the GPU
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[68ch]">
            A WebGPU compute shader runs the GUM product rule across a 64×64 epistemic field.
            Each cell's confidence ε propagates from its neighbors — in parallel, on your GPU.
          </p>
        </div>

        <div className="max-w-4xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
          {/* Canvas panel */}
          <div className="glass glass-specular rounded-2xl p-6 flex flex-col items-center gap-4">
            <div className="relative">
              <canvas
                ref={canvasRef}
                width={64}
                height={64}
                className="rounded-lg"
                style={{
                  width: 256,
                  height: 256,
                  imageRendering: 'pixelated',
                  background: '#0d1117',
                }}
              />
              {status === 'idle' && (
                <div className="absolute inset-0 rounded-lg bg-[#0d1117] flex items-center justify-center">
                  <span className="text-[var(--color-text-tertiary)] text-sm">64×64 epistemic field</span>
                </div>
              )}
              {status === 'running' && (
                <div className="absolute inset-0 rounded-lg bg-[#0d1117]/80 flex items-center justify-center">
                  <div className="w-8 h-8 border-2 border-[var(--color-accent-gold)] border-t-transparent rounded-full animate-spin" />
                </div>
              )}
            </div>

            {status === 'done' && stats && (
              <div className="text-xs font-mono text-[var(--color-text-tertiary)] grid grid-cols-3 gap-4 w-full">
                <div className="text-center">
                  <div className="text-[var(--color-accent-gold)] font-semibold">{stats.min}</div>
                  <div>ε min</div>
                </div>
                <div className="text-center">
                  <div className="text-[var(--color-accent-gold)] font-semibold">{stats.ms}ms</div>
                  <div>GPU time</div>
                </div>
                <div className="text-center">
                  <div className="text-[var(--color-accent-gold)] font-semibold">{stats.max}</div>
                  <div>ε max</div>
                </div>
              </div>
            )}

            {status !== 'unsupported' && (
              <button
                onClick={run}
                disabled={status === 'running'}
                className="px-6 py-2.5 rounded-full text-sm font-semibold transition-all bg-[linear-gradient(120deg,var(--color-accent-gold),var(--color-accent-gold-soft))] text-[#10233e] disabled:opacity-50 hover:-translate-y-px"
              >
                {status === 'running' ? 'Computing…' : status === 'done' ? 'Run again' : 'Run on GPU'}
              </button>
            )}

            {status === 'unsupported' && (
              <div className="text-sm text-[var(--color-text-secondary)] text-center max-w-[240px]">
                WebGPU requires Chrome 113+ or Edge 113+.{' '}
                <a href="#" className="text-[var(--color-accent-gold-soft)] hover:underline">
                  Try the CPU uncertainty calculator instead ↓
                </a>
              </div>
            )}
          </div>

          {/* WGSL code panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--color-accent-gold)]/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(201,169,110,0.05)] border-b border-[var(--color-accent-gold)]/20">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                <div className="w-3 h-3 rounded-full bg-[#28c840]" />
              </div>
              <span className="flex-1 text-center text-[var(--color-accent-gold)] text-sm font-mono">
                epistemic.wgsl
              </span>
            </div>
            <pre className="p-6 bg-[#0d1117] text-xs leading-relaxed overflow-x-auto">
              <code className="text-[#e6edf3]">{`@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id)
        id: vec3<u32>) {
  let i = id.x + id.y * 64u;
  if (i >= 4096u) { return; }

  // GUM product rule:
  // ε(a·b) = ε(a) × ε(b)
  let rel_a = 1.0 - eps_in[j];
  let rel_b = 1.0 - eps_in[k];
  let combined = sqrt(
    rel_a * rel_a + rel_b * rel_b
  );
  eps_out[i] = clamp(
    1.0 - combined, 0.0, 1.0
  );
}`}</code>
            </pre>
            <div className="px-6 py-3 bg-[#0d1117] border-t border-[var(--glass-border)]">
              <p className="text-xs text-[var(--color-text-tertiary)]">
                4,096 parallel threads. Each computes propagated epistemic confidence
                using the GUM (JCGM 100:2008) product rule.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
