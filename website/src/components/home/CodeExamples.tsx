import { useState } from 'react';

interface CodeExample {
  id: string;
  label: string;
  code: string;
}

const examples: CodeExample[] = [
  {
    id: 'uncertainty',
    label: 'Uncertainty',
    code: `// Epistemic computing with native uncertainty
let mass: Knowledge<kg> = measure(
    value: 75.3,
    uncertainty: 0.5,
    source: "clinical_scale_001"
)

let height: Knowledge<m> = measure(
    value: 1.82,
    uncertainty: 0.01,
    source: "stadiometer_001"
)

// Uncertainty propagates automatically (GUM-compliant)
let bmi = mass / (height * height)
// bmi.uncertainty computed via error propagation

// Confidence gates control execution
if bmi.confidence > 0.95 {
    report_bmi(bmi)
} else {
    request_remeasurement()
}`,
  },
  {
    id: 'effects',
    label: 'Effects',
    code: `// Algebraic effects for controlled side effects
fn read_sensor() -> f64 with IO, Fail {
    let raw = perform IO::read("/dev/sensor0")?
    let value = parse_float(raw)?
    value
}

// Effect handlers provide the implementation
fn main() with IO {
    handle read_sensor() {
        IO::read(path) => resume(fs::read_to_string(path)),
        Fail::error(msg) => {
            log_error(msg)
            resume(0.0)  // Default fallback
        }
    }
}

// Effects are tracked in types—no hidden side effects`,
  },
  {
    id: 'gpu',
    label: 'GPU Kernels',
    code: `// Native GPU kernel syntax with automatic memory management
@kernel
fn matrix_multiply(
    a: &Tensor<f32, [M, K]>,
    b: &Tensor<f32, [K, N]>,
    c: &!Tensor<f32, [M, N]>
) with GPU {
    let row = gpu::thread_idx_y()
    let col = gpu::thread_idx_x()

    var sum: f32 = 0.0
    for k in 0..K {
        sum += a[row, k] * b[k, col]
    }
    c[row, col] = sum
}

// Launch with automatic grid/block sizing
let result = matrix_multiply<<<grid, block>>>(a, b)`,
  },
  {
    id: 'units',
    label: 'Units',
    code: `// Type-safe units of measure
let dose: mg = 500.0
let volume: mL = 250.0
let concentration: mg/mL = dose / volume  // 2.0 mg/mL

// Compile-time dimension checking
let mass: kg = 1.5
// let error = mass + volume  // Compile error: kg ≠ mL

// Unit conversions are explicit
let mass_mg: mg = mass.to::<mg>()  // 1500.0 mg

// Scientific constants with units
let c: m/s = SPEED_OF_LIGHT
let h: J*s = PLANCK_CONSTANT
let energy: J = mass * c * c`,
  },
];

export default function CodeExamples() {
  const [activeTab, setActiveTab] = useState('uncertainty');

  const activeExample = examples.find((e) => e.id === activeTab) || examples[0];

  return (
    <section id="code" className="py-24 bg-[var(--color-bg)]">
      <div className="container">
        <h2 className="text-4xl font-semibold text-center mb-4 text-[var(--color-navy-900)] dark:text-white">
          Hello, Uncertainty
        </h2>
        <p className="text-lg text-center text-[var(--color-text-muted)] mb-12 max-w-3xl mx-auto">
          See how Sounio handles real scientific computing challenges with native language support.
        </p>

        {/* Tabs */}
        <div className="flex flex-wrap gap-2 mb-6 justify-center">
          {examples.map((example) => (
            <button
              key={example.id}
              onClick={() => setActiveTab(example.id)}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
                activeTab === example.id
                  ? 'bg-[var(--color-navy-900)] text-white shadow-lg'
                  : 'bg-transparent text-[var(--color-navy-900)] dark:text-white border-2 border-[var(--color-navy-900)] dark:border-white/30 hover:bg-[var(--color-navy-900)]/10'
              }`}
            >
              {example.label}
            </button>
          ))}
        </div>

        {/* Code block */}
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            <pre className="bg-[var(--color-navy-900)] text-[#e2e8f0] p-6 rounded-xl overflow-x-auto text-sm leading-relaxed shadow-2xl">
              <code>{activeExample.code}</code>
            </pre>
            <button
              onClick={() => navigator.clipboard.writeText(activeExample.code)}
              className="absolute top-4 right-4 px-3 py-1.5 bg-white/10 hover:bg-white/20 text-white/80 hover:text-white rounded text-sm transition-colors"
            >
              Copy
            </button>
          </div>

          <div className="mt-6 text-center">
            <a
              href="/playground"
              className="inline-flex items-center gap-2 text-[var(--color-navy-900)] dark:text-[var(--color-gold-500)] hover:underline font-medium"
            >
              Try in Playground
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
              </svg>
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
