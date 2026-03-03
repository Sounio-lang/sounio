import os

filepath = "website/src/components/home/CodeExamples.tsx"

content = """import { useState } from 'react';

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
// \u03B5 is the confidence level (0.0 to 1.0)
let mass: Knowledge[f64] = Knowledge(
    75.3, 
    \u03B5=0.98, 
    prov="clinical_scale_001"
)

let height: Knowledge[f64] = Knowledge(
    1.82, 
    \u03B5=0.99, 
    prov="stadiometer_001"
)

// GUM Uncertainty propagates automatically:
// \u03B5(a/b) = \u03B5(a) \u00d7 \u03B5(b)
let bmi: Knowledge[f64] = mass / (height * height)
// bmi.value = 22.74, bmi.\u03B5 = 0.96

if bmi.\u03B5 >= 0.95 {
    println("Confidence sufficient for diagnosis")
} else {
    println("Request remeasurement")
}`,
  },
  {
    id: 'causality',
    label: 'Causal Models',
    python: `# Python: implicit causality
# A model where Genetics confound Smoking and Cancer
def predict_cancer(genetics, smoking):
    tar = 0.8 * smoking
    cancer = 0.6 * tar + 0.3 * genetics
    return cancer

# The structure is hidden in code execution.
# Hard to analyze counterfactuals or interventions.
# No standard way to express "Smoking causes Tar".
`,
    sounio: `// Sounio: native causal directed acyclic graphs
causal model SmokingCancer {
    // Declare causal variables
    nodes: [Smoking, Tar, Cancer, Genetics]

    // Define causal relationships (edges)
    Genetics -> Smoking
    Genetics -> Cancer
    Smoking -> Tar
    Tar -> Cancer

    // Structural causal equations
    equations: {
        Smoking = 0.5 * Genetics,
        Tar = 0.8 * Smoking,
        Cancer = 0.6 * Tar + 0.3 * Genetics
    }
}
// The compiler understands the causal graph natively.`,
  },
  {
    id: 'resources',
    label: 'Resource Safety',
    python: `# Python: garbage collection
class FileHandle:
    def __init__(self, fd):
        self.fd = fd

def close_file(h):
    # Closes the file descriptors
    pass

h = FileHandle(42)
close_file(h)
close_file(h) # Oops, double free!
# The type system doesn't stop you from using 
# a closed resource or forgetting to close it.`,
    sounio: `// Sounio: linear types for strict ownership
linear struct FileHandle {
    fd: i32
}

fn close_file(h: FileHandle) -> i32 {
    h.fd // consumed exactly once
}

fn main() {
    let handle = FileHandle { fd: 42 }
    
    // Consumes the linear resource
    let result = close_file(handle)
    
    // let err = close_file(handle) 
    // ^ COMPILE ERROR: use of consumed linear value!
}
// Enforces no-cloning and no-dropping at compile time.`,
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

      </div>
    </section>
  );
}
"""

with open(filepath, "w") as f:
    f.write(content)
print("CodeExamples.tsx rewritten.")
