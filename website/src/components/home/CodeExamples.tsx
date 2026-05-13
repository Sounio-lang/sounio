import { useState } from 'react';
import { useAudience } from '../../lib/useAudience';
import { codeExplanations } from '../../lib/audienceContent';

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
let mass: Knowledge<f64> = Knowledge(
    75.3,
    \u03B5=0.98,
    prov="clinical_scale_001"
)

let height: Knowledge<f64> = Knowledge(
    1.82,
    \u03B5=0.99,
    prov="stadiometer_001"
)

// GUM Uncertainty propagates automatically:
// \u03B5(a/b) = \u03B5(a) \u00D7 \u03B5(b)
let bmi: Knowledge<f64> = mass / (height * height)
// bmi.value = 22.74, bmi.\u03B5 = 0.96

if bmi.\u03B5 >= 0.95 {
    println("Confidence sufficient for diagnosis")
} else {
    println("Request remeasurement")
}`,
  },
  {
    id: 'causality',
    label: 'Causal Types',
    python: `# Python: implicit causality, no type safety
import dowhy

# Intervention is a plain dict \u2014 no type check
model = CausalModel(data=df,
    graph="digraph {Education -> Earnings}")
do_result = model.do({"Education": 1})

# Counterfactual: manual estimate, no confidence
cf_earnings = predict_cf(do_result)
# Was cf_earnings reliable? The type won't say.
# No way to gate on causal confidence.`,
    sounio: `// Sounio: Intervention<T> and Counterfactual<T>
// are language-level types (compiler keywords)

// Build a causal DAG with epistemic edge uncertainty
var dag = causal_dag_new()
dag = dag_add_node(dag, 0)  // treatment
dag = dag_add_node(dag, 1)  // outcome
dag = dag_add_edge(dag, 0, 1,
    ec_beta_new(8.0, 2.0), 0.5, 0.1)

// Pearl's do-operator removes confounding
let intervened = do_intervention(dag, 1)

// Average treatment effect with epistemic uncertainty
let ate = average_treatment_effect(dag, 0, 1)`,
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
  {
    id: 'algebra',
    label: 'Algebra',
    python: `# Python: no algebraic laws in the type system
import numpy as np

# Octonions: multiplication is non-associative.
# Python has no way to express or enforce this.
# The optimizer will reorder operations assuming
# standard floating-point associativity — wrong.

def oct_mul(a, b):
    # 64-element Cayley table... hope it's correct
    result = np.zeros(8)
    # ... manual table lookup ...
    return result

x = oct_mul(a, oct_mul(b, c))  # (a*(b*c))?
y = oct_mul(oct_mul(a, b), c)  # (a*b)*c?
# x != y — but NumPy doesn't warn you.`,
    sounio: `// Sounio: algebraic laws are part of the type
algebra Octonion over f64 {
    add: commutative, associative
    mul: alternative, non_commutative
    reassociate: fano_selective
}

// The e-graph optimizer only rewrites using
// laws you declared. Fano-selective means it
// respects the Fano plane symmetry for octonions.

fn norm_sq(x0: f64, x1: f64, x2: f64, x3: f64,
           x4: f64, x5: f64, x6: f64, x7: f64) -> f64 {
    x0*x0 + x1*x1 + x2*x2 + x3*x3 +
    x4*x4 + x5*x5 + x6*x6 + x7*x7
}
// Optimizer applies only algebra-safe rewrites.
// Non-associative operations are never reordered.`,
  },
];

export default function CodeExamples() {
  const [activeTab, setActiveTab] = useState('uncertainty');
  const [codeExpanded, setCodeExpanded] = useState<Record<string, boolean>>({});
  const { audience } = useAudience();

  const activeExample = examples.find((e) => e.id === activeTab) || examples[0];
  const isScientist = audience === 'scientist';
  const isExpanded = codeExpanded[activeTab] ?? !isScientist;
  const explanation = codeExplanations[activeTab];

  const toggleCode = () => {
    setCodeExpanded((prev) => ({ ...prev, [activeTab]: !isExpanded }));
  };

  return (
    <section id="code" className="py-[clamp(3.5rem,7vw,6rem)] bg-[color-mix(in_srgb,var(--color-bg-alt)_74%,transparent)] border-y border-[var(--glass-border)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            See the difference
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[68ch]">
            {isScientist
              ? 'Your current tools compute the number. Sounio computes what you can trust about that number.'
              : 'Python computes the number. Sounio computes what you can trust.'}
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
            <div className={isScientist ? `code-collapsible ${isExpanded ? 'expanded' : 'collapsed'}` : ''}>
              <pre className="p-6 bg-[#0d1117] text-sm leading-relaxed overflow-x-auto min-h-[400px]">
                <code className="text-[#e6edf3]">{activeExample.python}</code>
              </pre>
            </div>
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
            <div className={isScientist ? `code-collapsible ${isExpanded ? 'expanded' : 'collapsed'}` : ''}>
              <pre className="p-6 bg-[#0d1117] text-sm leading-relaxed overflow-x-auto min-h-[400px]">
                <code className="text-[#e6edf3]">{activeExample.sounio}</code>
              </pre>
            </div>
            {isScientist && (
              <div className="px-4 py-3 bg-[#0d1117] border-t border-[var(--glass-border)] flex justify-center">
                <button onClick={toggleCode} className="code-toggle-btn">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={isExpanded ? "M19 9l-7 7-7-7" : "M9 5l7 7-7 7"} />
                  </svg>
                  {isExpanded ? 'Hide code' : 'Show code'}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Explanation panel for scientist mode */}
        {isScientist && explanation && (
          <div className="explanation-panel rounded-2xl mt-4 max-w-6xl mx-auto">
            <strong>In plain language: </strong>{explanation}
          </div>
        )}

      </div>
    </section>
  );
}
