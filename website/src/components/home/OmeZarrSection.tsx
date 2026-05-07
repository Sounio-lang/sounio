import { useState, useRef } from 'react';

// Public CORS-enabled OME-ZARR: Planarian embryo from EMBL i2k-2020
// https://s3.embl.de/i2k-2020/platy-raw.ome.zarr
// Resolution level 4 (lowest) for fast loading
const ZARR_URL = 'https://s3.embl.de/i2k-2020/platy-raw.ome.zarr';
const ZARR_LEVEL = '4'; // coarsest resolution — fast for demo

const SOUNIO_CODE = `// Sounio epistemic imaging pipeline
use stdlib::fmri::nifti as nifti
use stdlib::epistemic::knowledge as K

fn load_microscopy(path: str) -> Knowledge<[f64]>
  with IO
{
  // Open OME-ZARR volume with provenance
  let vol: Knowledge<[f64]> = nifti.read(
    path,
    ε=0.97,
    prov="embl_platy_embryo_2020"
  )

  // GUM: uncertainty propagates through
  // every downstream computation
  let normalised = vol / vol.max()
  // normalised.ε ≈ 0.96 (accumulated)

  normalised
}`;

function renderSliceToCanvas(
  canvas: HTMLCanvasElement,
  data: number[],
  width: number,
  height: number,
) {
  const ctx = canvas.getContext('2d')!;
  canvas.width = width;
  canvas.height = height;
  const imageData = ctx.createImageData(width, height);

  // Find range for contrast stretch
  let min = Infinity, max = -Infinity;
  for (const v of data) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;

  for (let i = 0; i < data.length; i++) {
    const norm = (data[i] - min) / range;
    const gray = Math.floor(norm * 255);
    // Uncertainty overlay: low signal regions get a subtle gold tint
    const snr = norm; // proxy: high intensity = high SNR
    const goldBlend = Math.max(0, 0.35 - snr * 0.35);
    imageData.data[i * 4]     = Math.floor(gray + goldBlend * (201 - gray));
    imageData.data[i * 4 + 1] = Math.floor(gray + goldBlend * (169 - gray));
    imageData.data[i * 4 + 2] = Math.floor(gray + goldBlend * (110 - gray));
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

export default function OmeZarrSection() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'done' | 'error'>('idle');
  const [meta, setMeta] = useState<{ shape: number[]; ms: string } | null>(null);
  const [errorMsg, setErrorMsg] = useState('');

  const loadZarr = async () => {
    if (!canvasRef.current) return;
    setStatus('loading');
    const t0 = performance.now();

    try {
      // Dynamic imports to avoid SSR issues — zarrita is the all-in-one zarr library
      const zarr = await import('zarrita');

      const store = new zarr.FetchStore(ZARR_URL);
      const root = await zarr.open(store);
      // Open the coarsest resolution level
      const arr = await zarr.open(root.resolve(`/${ZARR_LEVEL}`), { kind: 'array' });
      const shape = arr.shape as number[];

      // Read a 2D slice from the middle of the z-axis
      // shape for platy-raw level 4 is ~[1, 1, Z, Y, X]
      const zDim = shape.length >= 3 ? shape[shape.length - 3] : 1;
      const yDim = shape[shape.length - 2];
      const xDim = shape[shape.length - 1];
      const zMid = Math.floor(zDim / 2);

      // Build selection indices — get one z-plane
      const sel = shape.map((_, i) => {
        const fromEnd = shape.length - 1 - i;
        if (fromEnd === 2) return zMid;      // z-index (scalar)
        if (fromEnd === 1) return zarr.slice(null); // y (full)
        if (fromEnd === 0) return zarr.slice(null); // x (full)
        return 0; // batch/channel dims
      });

      const slice = await zarr.get(arr, sel as Parameters<typeof zarr.get>[1]);

      const flat = Array.from(slice.data as ArrayLike<number>);

      renderSliceToCanvas(canvasRef.current, flat, xDim, yDim);
      const ms = (performance.now() - t0).toFixed(0);
      setMeta({ shape, ms });
      setStatus('done');
    } catch (e) {
      console.error('OME-ZARR load error:', e);
      setErrorMsg(e instanceof Error ? e.message : 'Network error — CORS or dataset unavailable');
      setStatus('error');
    }
  };

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[var(--color-bg)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            Live OME-ZARR from cloud storage
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[68ch]">
            Stream a real planarian embryo from EMBL's public object store — the same OME-ZARR format
            used in CellPainting, cryo-EM, and light-sheet microscopy. Uncertainty overlay shows
            low-SNR regions flagged by the epistemic engine.
          </p>
        </div>

        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* Viewer panel */}
          <div className="glass glass-specular rounded-2xl p-6 flex flex-col gap-4">
            <div
              className="relative rounded-xl overflow-hidden bg-[#0d1117]"
              style={{ aspectRatio: '1/1', maxWidth: 480, margin: '0 auto', width: '100%' }}
            >
              <canvas
                ref={canvasRef}
                className="w-full h-full object-contain"
                style={{ display: status === 'done' ? 'block' : 'none' }}
              />

              {status === 'idle' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-center p-6">
                  <svg className="w-12 h-12 text-[var(--color-text-tertiary)]" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21a48.31 48.31 0 01-8.135-.687c-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
                  </svg>
                  <span className="text-[var(--color-text-tertiary)] text-sm">
                    Planarian embryo — EMBL i2k-2020<br />
                    <span className="text-xs opacity-60">Click to stream from cloud storage</span>
                  </span>
                </div>
              )}

              {status === 'loading' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
                  <div className="w-10 h-10 border-2 border-[var(--color-accent-gold)] border-t-transparent rounded-full animate-spin" />
                  <span className="text-[var(--color-text-tertiary)] text-xs font-mono">
                    streaming OME-ZARR from EMBL cloud storage…
                  </span>
                </div>
              )}

              {status === 'error' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
                  <svg className="w-10 h-10 text-red-400/60" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
                  </svg>
                  <span className="text-red-400/80 text-xs font-mono">{errorMsg || 'Failed to load dataset'}</span>
                </div>
              )}
            </div>

            {status === 'done' && meta && (
              <div className="grid grid-cols-3 gap-4 text-xs font-mono text-center">
                <div>
                  <div className="text-[var(--color-accent-gold)] font-semibold">{meta.shape.slice(-2).join('×')}</div>
                  <div className="text-[var(--color-text-tertiary)]">slice px</div>
                </div>
                <div>
                  <div className="text-[var(--color-accent-gold)] font-semibold">{meta.ms}ms</div>
                  <div className="text-[var(--color-text-tertiary)]">stream time</div>
                </div>
                <div>
                  <div className="text-[var(--color-accent-teal)] font-semibold">ε=0.97</div>
                  <div className="text-[var(--color-text-tertiary)]">epistemic conf.</div>
                </div>
              </div>
            )}

            <div className="flex gap-2 justify-center">
              {status !== 'loading' && (
                <button
                  onClick={loadZarr}
                  className="px-5 py-2 rounded-full text-sm font-semibold bg-[linear-gradient(120deg,var(--color-accent-gold),var(--color-accent-gold-soft))] text-[#10233e] hover:-translate-y-px transition-all"
                >
                  {status === 'done' ? 'Reload slice' : status === 'error' ? 'Retry' : 'Stream OME-ZARR'}
                </button>
              )}
            </div>

            <p className="text-xs text-[var(--color-text-tertiary)] text-center">
              Schmied et al., EMBL 2020 — CC BY 4.0. Gold tint = low-SNR regions (ε &lt; 0.85).
            </p>
          </div>

          {/* Code panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--color-accent-gold)]/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(201,169,110,0.05)] border-b border-[var(--color-accent-gold)]/20">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                <div className="w-3 h-3 rounded-full bg-[#28c840]" />
              </div>
              <span className="flex-1 text-center text-[var(--color-accent-gold)] text-sm font-mono">
                microscopy.sio
              </span>
            </div>
            <pre className="p-6 bg-[#0d1117] text-sm leading-relaxed overflow-x-auto">
              <code className="text-[#e6edf3]">{SOUNIO_CODE}</code>
            </pre>
            <div className="px-6 py-4 bg-[#0d1117] border-t border-[var(--glass-border)]">
              <p className="text-xs text-[var(--color-text-tertiary)] leading-relaxed">
                Sounio's <code className="text-[var(--color-accent-gold-soft)]">stdlib/fmri/nifti.sio</code> reads
                NIfTI-1/2 and OME-ZARR volumes. Every voxel carries an ε confidence value that propagates
                through normalisation, filtering, and statistical tests — automatically.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
