import { useState, useEffect, useCallback } from 'react';

interface Diagnostic {
  severity: 'error' | 'warning';
  message: string;
  line: number;
  column: number;
}

interface CompileResult {
  success: boolean;
  diagnostics: Diagnostic[];
}

interface RunResult extends CompileResult {
  output: string;
  returnValue: string | null;
}

type ShimModule = {
  compile: (src: string) => string;
  run: (src: string) => string;
  default: () => Promise<void>;
};

const DEFAULT_SOURCE = `fn main() with IO {
  let mass: Knowledge<f64> = Knowledge(
    75.3, ε=0.98, prov="clinical_scale"
  )
  let height: Knowledge<f64> = Knowledge(
    1.82, ε=0.99, prov="stadiometer"
  )
  // GUM: ε(a/b²) = ε(a) × ε(b)²
  let bmi = mass / (height * height)
  print("BMI: " + bmi.value)
  print("Confidence: " + bmi.ε)
}`;

const EXAMPLES = [
  { label: 'Epistemic BMI', source: DEFAULT_SOURCE },
  {
    label: 'Linear type',
    source: `linear struct FileHandle {
  fd: i32
}

fn close_file(h: FileHandle) -> i32 {
  h.fd
}

fn main() with IO {
  let handle = FileHandle { fd: 42 }
  let result = close_file(handle)
  print("closed fd: " + result)
  // let err = close_file(handle)
  // ^ COMPILE ERROR: use of consumed linear value
}`,
  },
  {
    label: 'Catch Rust syntax',
    source: `fn main() {
  let mut x = 5
  println!("x = {}", x)
}`,
  },
];

export default function SounioPlayground() {
  const [source, setSource] = useState(DEFAULT_SOURCE);
  const [shimReady, setShimReady] = useState(false);
  const [shimMod, setShimMod] = useState<ShimModule | null>(null);
  const [result, setResult] = useState<(CompileResult | RunResult) | null>(null);
  const [mode, setMode] = useState<'compile' | 'run'>('compile');
  const [running, setRunning] = useState(false);

  useEffect(() => {
    // Use Function() to bypass Vite's static import analysis —
    // the shim lives in /public/wasm/ and is not bundled.
    const shimPath = '/wasm/sounio_compiler.js';
    (async () => {
      try {
        const mod = await (Function('p', 'return import(p)')(shimPath)) as ShimModule;
        await mod.default();
        setShimMod(mod);
        setShimReady(true);
      } catch {
        // WASM shim unavailable — playground shows limited UI
      }
    })();
  }, []);

  const execute = useCallback(() => {
    if (!shimMod) return;
    setRunning(true);
    try {
      const raw = mode === 'compile' ? shimMod.compile(source) : shimMod.run(source);
      setResult(JSON.parse(raw));
    } catch {
      setResult({ success: false, diagnostics: [{ severity: 'error', message: 'Internal error', line: 0, column: 0 }] });
    }
    setRunning(false);
  }, [shimMod, source, mode]);

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[var(--color-bg)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            Try Sounio in your browser
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[68ch]">
            Live syntax checking — catch Rust habits, missing <code className="font-mono text-[var(--color-accent-gold-soft)]">with IO</code>,
            and epistemic type errors before the compiler ever runs.
          </p>
        </div>

        {/* Example selector */}
        <div className="flex flex-wrap gap-2 mb-4">
          {EXAMPLES.map((ex) => (
            <button
              key={ex.label}
              onClick={() => { setSource(ex.source); setResult(null); }}
              className="px-3 py-1.5 rounded-full text-xs font-semibold border border-[var(--glass-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] hover:border-[var(--color-accent-gold)]/40 transition-all"
            >
              {ex.label}
            </button>
          ))}
        </div>

        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Editor panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--glass-border)]">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(255,255,255,0.03)] border-b border-[var(--glass-border)]">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                <div className="w-3 h-3 rounded-full bg-[#28c840]" />
              </div>
              <span className="flex-1 text-center text-[var(--color-text-tertiary)] text-sm font-mono">
                playground.sio
              </span>
            </div>
            <textarea
              value={source}
              onChange={(e) => { setSource(e.target.value); setResult(null); }}
              spellCheck={false}
              className="w-full bg-[#0d1117] text-[#e6edf3] font-mono text-sm p-6 leading-relaxed resize-none focus:outline-none"
              style={{ minHeight: 340 }}
            />
            <div className="flex items-center gap-2 px-4 py-3 bg-[#0d1117] border-t border-[var(--glass-border)]">
              <div className="flex gap-1 p-0.5 rounded-full bg-[rgba(255,255,255,0.06)]">
                {(['compile', 'run'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setMode(m)}
                    className={`px-4 py-1.5 rounded-full text-xs font-semibold transition-all ${
                      mode === m
                        ? 'bg-[var(--color-text-primary)] text-[var(--color-surface-primary)]'
                        : 'text-[var(--color-text-secondary)]'
                    }`}
                  >
                    {m.charAt(0).toUpperCase() + m.slice(1)}
                  </button>
                ))}
              </div>
              <button
                onClick={execute}
                disabled={!shimReady || running}
                className="ml-auto px-5 py-1.5 rounded-full text-xs font-semibold bg-[linear-gradient(120deg,var(--color-accent-gold),var(--color-accent-gold-soft))] text-[#10233e] disabled:opacity-40 hover:-translate-y-px transition-all"
              >
                {!shimReady ? 'Loading…' : running ? 'Running…' : mode === 'compile' ? '⚡ Compile' : '▶ Run'}
              </button>
            </div>
          </div>

          {/* Output panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--color-accent-gold)]/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(201,169,110,0.05)] border-b border-[var(--color-accent-gold)]/20">
              <span className="flex-1 text-center text-[var(--color-accent-gold)] text-sm font-mono">
                output
              </span>
              {result && (
                <span className={`text-xs font-semibold ${result.success ? 'text-emerald-400' : 'text-red-400'}`}>
                  {result.success ? '✓ OK' : '✗ ERRORS'}
                </span>
              )}
            </div>
            <div className="bg-[#0d1117] p-6 font-mono text-sm leading-relaxed min-h-[340px]">
              {!result && (
                <span className="text-[var(--color-text-tertiary)] text-xs">
                  Click Compile or Run to see output…
                </span>
              )}
              {result && result.diagnostics.length === 0 && result.success && (
                <div className="text-emerald-400 text-xs mb-3">No diagnostics — clean compile.</div>
              )}
              {result?.diagnostics.map((d, i) => (
                <div key={i} className={`mb-2 ${d.severity === 'error' ? 'text-red-400' : 'text-yellow-400'}`}>
                  <span className="opacity-60 text-xs">
                    {d.severity === 'error' ? '✗' : '⚠'} line {d.line}:{d.column}
                  </span>
                  <div className="text-sm mt-0.5">{d.message}</div>
                </div>
              ))}
              {'output' in (result ?? {}) && (result as RunResult).output && (
                <div className="mt-4 pt-4 border-t border-[var(--glass-border)]">
                  <div className="text-xs text-[var(--color-text-tertiary)] mb-2">stdout:</div>
                  <div className="text-[#e6edf3] whitespace-pre-wrap">
                    {(result as RunResult).output}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
