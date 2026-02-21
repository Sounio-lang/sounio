import React, { useState } from 'react';

interface Scenario {
  id: string;
  label: string;
  dimension: string;
  traditional: {
    lang: string;
    code: string;
    lines: number;
    issues: string[];
  };
  sounio: {
    code: string;
    lines: number;
    advantages: string[];
  };
}

const scenarios: Scenario[] = [
  {
    id: 'uncertainty',
    label: 'Uncertainty Propagation',
    dimension: 'Every measurement carries doubt. How does the language handle it?',
    traditional: {
      lang: 'Python + NumPy',
      code: `import numpy as np

value = 9.8
std_dev = 0.3

# Manual propagation
derived = value * 2.5
derived_err = std_dev * 2.5  # hope this is right

# Manual threshold check
if derived_err / derived < 0.05:
    approve(derived)
else:
    # What was the original source?
    # Who measured it? When?
    remeasure()  # no provenance`,
      lines: 14,
      issues: [
        'Uncertainty is a separate variable — easy to lose',
        'Propagation rules are manual and error-prone',
        'No provenance: source identity is gone',
      ],
    },
    sounio: {
      code: `let c: Knowledge<mg_per_l> = measure(
  value: 9.8,
  uncertainty: 0.3,
  confidence: 0.97,
  source: "lab_run_042"
)

let derived = c * 2.5  // uncertainty auto-propagates

if derived.confidence > 0.95 {
  approve(derived)  // provenance travels with it
}`,
      lines: 11,
      advantages: [
        'Uncertainty is part of the type — impossible to forget',
        'Propagation is automatic through all operations',
        'Provenance tracks the full measurement chain',
      ],
    },
  },
  {
    id: 'units',
    label: 'Unit Safety',
    dimension: 'Physical units must be correct. When does the language catch mistakes?',
    traditional: {
      lang: 'C++ (raw doubles)',
      code: `double distance_km = 400.0;
double time_hr = 2.5;
double speed = distance_km / time_hr;

// Later, someone passes meters...
double dist_m = 400000.0;
double bad_speed = dist_m / time_hr;
// Compiles fine. Units are wrong.
// Result: 160,000 "km/h" — silent bug

// Mars Climate Orbiter (1999):
// $327M lost to this exact class of error`,
      lines: 10,
      issues: [
        'Units exist only in comments — compiler ignores them',
        'Mixing km and m compiles without any warning',
        'Bugs surface at runtime (or in orbit)',
      ],
    },
    sounio: {
      code: `let distance: km = 400.0
let time: hr = 2.5
let speed: km_per_hr = distance / time

// This would be a compile-time error:
// let bad: km_per_hr = meters(400000) / time
//   ^ ERROR: dimension mismatch (m/hr ≠ km/hr)`,
      lines: 6,
      advantages: [
        'Units are first-class types checked at compile time',
        'Dimension mismatches are caught before code ships',
        'No runtime overhead — erased after verification',
      ],
    },
  },
  {
    id: 'effects',
    label: 'Effect Transparency',
    dimension: 'Side effects hide bugs. How visible are they in function signatures?',
    traditional: {
      lang: 'Python',
      code: `def analyze(data):
    # Does this read files? Hit the network?
    # Allocate GPU memory? Mutate state?
    # The signature tells you nothing.
    
    result = expensive_model(data)  # GPU?
    db.save(result)                 # I/O?
    cache[data.id] = result         # mutation?
    return result

# Caller has no idea what just happened
x = analyze(my_data)`,
      lines: 10,
      issues: [
        'Function signature hides all side effects',
        'Caller cannot know if GPU, I/O, or mutation occurs',
        'Debugging requires reading every implementation',
      ],
    },
    sounio: {
      code: `fn analyze(data: Dataset)
  -> Result<Analysis> ! Gpu, Io, Mut {
  
  let result = expensive_model(data)  // ! Gpu
  db.save(result)                     // ! Io
  cache.set(data.id, result)          // ! Mut
  result
}

// Caller sees EXACTLY what effects are involved`,
      lines: 9,
      advantages: [
        'Effects declared in the function signature',
        'Caller knows precisely what resources are touched',
        'Compiler enforces effect boundaries at every call site',
      ],
    },
  },
];

export function ComparisonMatrix() {
  const [activeId, setActiveId] = useState(scenarios[0].id);
  const active = scenarios.find((s) => s.id === activeId)!;

  return (
    <div className="comparison-matrix">
      <div className="flex flex-wrap gap-2 mb-8">
        {scenarios.map((s) => (
          <button
            key={s.id}
            onClick={() => setActiveId(s.id)}
            className={`
              px-4 py-2 rounded-full text-sm font-semibold transition-all duration-200
              ${activeId === s.id
                ? 'bg-[var(--color-accent-gold)] text-[#10233e] shadow-lg shadow-[var(--color-accent-gold)]/20'
                : 'bg-[rgba(255,255,255,0.06)] text-[var(--color-text-secondary)] border border-[var(--glass-border)] hover:border-[var(--color-accent-gold)]/40'
              }
            `}
          >
            {s.label}
          </button>
        ))}
      </div>

      <p className="text-[var(--color-text-secondary)] text-[0.95rem] mb-6 max-w-[60ch]">
        {active.dimension}
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Traditional Side */}
        <div className="glass rounded-[var(--radius-xl)] overflow-hidden border border-red-500/20">
          <div className="flex items-center justify-between px-5 py-3 bg-red-500/10 border-b border-red-500/20">
            <span className="text-sm font-semibold text-red-400">{active.traditional.lang}</span>
            <span className="text-xs text-[var(--color-text-tertiary)]">{active.traditional.lines} lines</span>
          </div>
          <pre className="p-5 overflow-x-auto text-[0.82rem] leading-relaxed text-[var(--color-text-secondary)] bg-[rgba(0,0,0,0.2)]">
            <code>{active.traditional.code}</code>
          </pre>
          <div className="p-5 border-t border-[var(--glass-border)]">
            <ul className="list-none p-0 m-0 flex flex-col gap-2">
              {active.traditional.issues.map((issue, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-red-400/80">
                  <svg className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                  {issue}
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Sounio Side */}
        <div className="glass glass-specular rounded-[var(--radius-xl)] overflow-hidden border border-[var(--color-accent-green)]/30">
          <div className="flex items-center justify-between px-5 py-3 bg-[var(--color-accent-green)]/10 border-b border-[var(--color-accent-green)]/20">
            <span className="text-sm font-semibold text-[var(--color-accent-green)]">Sounio</span>
            <span className="text-xs text-[var(--color-text-tertiary)]">{active.sounio.lines} lines</span>
          </div>
          <pre className="p-5 overflow-x-auto text-[0.82rem] leading-relaxed text-[var(--color-text-primary)] bg-[rgba(0,0,0,0.15)]">
            <code>{active.sounio.code}</code>
          </pre>
          <div className="p-5 border-t border-[var(--glass-border)]">
            <ul className="list-none p-0 m-0 flex flex-col gap-2">
              {active.sounio.advantages.map((adv, i) => (
                <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-accent-green)]">
                  <svg className="w-4 h-4 mt-0.5 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                  {adv}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
