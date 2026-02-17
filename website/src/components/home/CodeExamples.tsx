import { useState } from 'react';

interface TabExample {
  id: string;
  label: string;
  python: string;
  sounio: string;
}

const examples: TabExample[] = [
  {
    id: 'uncertainty',
    label: 'Uncertainty',
    python: `# Python: bare floats, no uncertainty
mass = 75.3        # kg... probably?
height = 1.82      # meters... maybe?

bmi = mass / (height ** 2)
# bmi = 22.74
# No idea how confident we are
# No idea where the data came from

if bmi < 25:
    print("Normal weight")
# What if the scale was off by 2kg?`,
    sounio: `// Sounio: epistemic computing
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

// Uncertainty propagates automatically
let bmi = mass / (height * height)
// bmi = 22.74 \u00b1 0.31 (confidence: 97.1%)

if bmi.confidence > 0.95 {
    report_bmi(bmi)
} else {
    request_remeasurement()
}`,
  },
  {
    id: 'effects',
    label: 'Effects',
    python: `# Python: hidden side effects
def read_sensor():
    raw = open("/dev/sensor0").read()
    value = float(raw)
    return value
    # What can go wrong? Everything.
    # File IO? Network? Parse errors?
    # The signature doesn't tell you.

def main():
    try:
        v = read_sensor()
    except Exception as e:
        print(f"Error: {e}")
        v = 0.0
    # Catching 'Exception' \u2014 hoping for the best`,
    sounio: `// Sounio: effects are tracked in types
fn read_sensor() -> f64 with IO, Fail {
    let raw = perform IO::read("/dev/sensor0")?
    let value = parse_float(raw)?
    value
}

// Effect handlers provide the implementation
fn main() with IO {
    handle read_sensor() {
        IO::read(path) => resume(fs::read(path)),
        Fail::error(msg) => {
            log_error(msg)
            resume(0.0)
        }
    }
}
// The type system guarantees: no hidden effects`,
  },
  {
    id: 'gpu',
    label: 'GPU',
    python: `# Python: string-based GPU code
import cupy as cp

kernel = cp.RawKernel(r'''
extern "C" __global__
void matmul(const float* a, const float* b,
            float* c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += a[row*K+k] * b[k*N+col];
    c[row*N+col] = sum;
}
''', 'matmul')  # No type checking inside string`,
    sounio: `// Sounio: native GPU kernel syntax
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
// Type-safe. Dimension-checked. No strings.`,
  },
];

export default function CodeExamples() {
  const [activeTab, setActiveTab] = useState('uncertainty');

  const activeExample = examples.find((e) => e.id === activeTab) || examples[0];

  return (
    <section id="code" className="py-32 bg-[var(--color-surface-primary)]">
      <div className="container px-4">
        <div className="text-center mb-16">
          <h2 className="text-4xl md:text-5xl font-bold text-[var(--color-text-primary)] mb-4">
            See the Difference
          </h2>
          <p className="text-lg text-[var(--color-text-secondary)] max-w-2xl mx-auto">
            Python computes the number. Sounio computes what you can trust.
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex justify-center mb-8">
          <div className="inline-flex gap-1 p-1 rounded-full glass">
            {examples.map((example) => (
              <button
                key={example.id}
                onClick={() => setActiveTab(example.id)}
                className={`px-6 py-2.5 rounded-full text-sm font-medium transition-all duration-200 ${
                  activeTab === example.id
                    ? 'bg-[var(--color-text-primary)] text-[var(--color-surface-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
              >
                {example.label}
              </button>
            ))}
          </div>
        </div>

        {/* Side-by-side comparison */}
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Python panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--glass-border)]">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(255,255,255,0.03)] border-b border-[var(--glass-border)]">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]"></div>
                <div className="w-3 h-3 rounded-full bg-[#febc2e]"></div>
                <div className="w-3 h-3 rounded-full bg-[#28c840]"></div>
              </div>
              <span className="flex-1 text-center text-[var(--color-text-tertiary)] text-sm font-mono">
                example.py
              </span>
            </div>
            <pre className="p-6 bg-[#0d1117] text-sm leading-relaxed overflow-x-auto min-h-[400px]">
              <code className="text-[#e6edf3]">{activeExample.python}</code>
            </pre>
          </div>

          {/* Sounio panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--color-accent-gold)]/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(201,169,110,0.05)] border-b border-[var(--color-accent-gold)]/20">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]"></div>
                <div className="w-3 h-3 rounded-full bg-[#febc2e]"></div>
                <div className="w-3 h-3 rounded-full bg-[#28c840]"></div>
              </div>
              <span className="flex-1 text-center text-[var(--color-accent-gold)] text-sm font-mono">
                example.sio
              </span>
            </div>
            <pre className="p-6 bg-[#0d1117] text-sm leading-relaxed overflow-x-auto min-h-[400px]">
              <code className="text-[#e6edf3]">{activeExample.sounio}</code>
            </pre>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-12 text-center">
          <a
            href="/playground"
            className="inline-flex items-center gap-2 text-[var(--color-accent-gold)] font-medium hover:gap-3 transition-all"
          >
            Try in Playground
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </a>
        </div>
      </div>
    </section>
  );
}
