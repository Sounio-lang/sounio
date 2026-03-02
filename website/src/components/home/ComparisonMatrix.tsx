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
      code: `let c: Knowledge[f64] = Knowledge(
  9.8, 
  ε=0.97, 
  prov="lab_run_042"
)

let derived: Knowledge[f64] = c * 2.5  // uncertainty auto-propagates

if derived.ε > 0.95 {
  approve(derived)  // provenance travels with it
}`,
      lines: 10,
      advantages: [
        'Uncertainty is part of the type — impossible to forget',
        'Propagation is automatic through all operations',
        'Provenance tracks the full measurement chain',
      ],
    },
  },
  {
    id: 'causal',
    label: 'Causal Integration',
    dimension: 'Statistical correlation is not causation. Can the language express interventions?',
    traditional: {
      lang: 'Python (NetworkX)',
      code: `import networkx as nx

# Purely library level, opaque to the compiler
model = nx.DiGraph()
model.add_edges_from([
    ("Genetics", "Smoking"),
    ("Genetics", "Cancer"),
    ("Smoking", "Tar"),
    ("Tar", "Cancer")
])

# Compiler doesn't understand the DAG
# Typos in string names cause silent failures`,
      lines: 12,
      issues: [
        'DAG exists only as string pairs at runtime',
        'Compiler cannot validate node connections',
        'Refactoring node names breaks graphs silently',
      ],
    },
    sounio: {
      code: `causal model SmokingCancer {
    nodes: [Smoking, Tar, Cancer, Genetics]
    Genetics -> Smoking
    Genetics -> Cancer
    Smoking -> Tar
    Tar -> Cancer
}

// The DAG is parsed as a native language construct
// Fully validated and typed at compile time`,
      lines: 10,
      advantages: [
        'Causal graphs are first-class language keywords',
        'Nodes and edges are validated by the compiler',
        'Impossible to connect non-existent nodes',
      ],
    },
  },
  {
    id: 'resources',
    label: 'Resource Safety',
    dimension: 'Forgetting to close a file or release a lock causes leaks. How are resources managed?',
    traditional: {
      lang: 'C (Manual)',
      code: `void process(int fd) {
    FILE* f = fdopen(fd, "r");
    
    if (error_condition) {
        // Forgot to fclose(f) before return!
        // Resource leak in error path.
        return; 
    }
    
    fclose(f);
}`,
      lines: 10,
      issues: [
        'Resource cleanup is purely convention',
        'Early returns easily bypass cleanup logic',
        'Double-free and use-after-free bugs common',
      ],
    },
    sounio: {
      code: `linear struct FileHandle { fd: i32 }

fn process(h: FileHandle) {
    if error_condition {
        // Compiler ERROR: \`h\` must be consumed!
        // Cannot return without closing or moving \`h\`
        return
    }
    
    close(h) // Consumes the linear value
}`,
      lines: 10,
      advantages: [
        'Linear structs must be consumed exactly once',
        'Compiler prevents leaks on all code paths',
        'Cannot accidentally use a resource after closing',
      ],
    },
  },
  {
    id: 'ontology',
    label: 'Ontology Integration',
    dimension: 'Scientific data requires standardized terminologies. How are domain concepts validated?',
    traditional: {
      lang: 'Python / JSON',
      code: `# Stringly-typed concepts, no domain validation
patient_data = {
    "diagnosis": "SNOMED:22298006", # Myocardial infarction
    "lab_test": "LOINC:718-7"      # Hemoglobin
}

def analyze_cardiac_event(dx_code):
    # Just a string check. Is this code a specific type 
    # of heart disease? The compiler doesn't know.
    if dx_code == "SNOMED:22298006":
        run_protocol()`,
      lines: 11,
      issues: [
        'Ontology IDs are opaque strings prone to typos',
        'No compile-time validation of SNOMED/LOINC codes',
        'Subsumption (Is-A) hierarchy is lost in the language',
      ],
    },
    sounio: {
      code: `import medical::snomed::{Disease, MyocardialInfarction}

// Compiler natively resolves the 15M+ term hierarchy
fn analyze_cardiac_event(dx: Disease) {
    match dx {
        // MyocardialInfarction is type-checked as a valid Disease
        MyocardialInfarction => run_protocol(),
        _ => default_care()
    }
}

let dx = MyocardialInfarction
analyze_cardiac_event(dx)`,
      lines: 12,
      advantages: [
        '15M+ scientific entities bound directly as language types',
        'Compile-time validation guarantees valid ontology terms',
        'Type system understands subsumption (child/parent relations)',
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
