import { useAudience } from '../../lib/useAudience';
import { useState } from 'react';

const scenarios = [
  {
    id: 'epistemic',
    scientistTitle: 'Numbers Without Context Are Dangerous',
    technicalTitle: 'Epistemic Uncertainty & Provenance',
    problem: 'Traditional tools lose where your data came from and how reliable it really is.',
    scientistExplanation:
      'Sounio automatically tracks provenance and uncertainty for every measurement. When confidence drops too low, it warns you \u2014 before you draw the wrong conclusion.',
    researchImpact: [
      'Automatic GUM-compliant error propagation',
      'Full audit trail of every data source',
      'Safety gates that prevent overconfident results',
      'Reproducible and defensible findings',
    ],
    status: 'Fully implemented',
    code: `let weight: Knowledge<f64> = Knowledge(
    75.3,
    \u03B5=0.98,
    prov="clinical_scale_001"
)
let height: Knowledge<f64> = Knowledge(
    1.82,
    \u03B5=0.99,
    prov="stadiometer_002"
)

let bmi: Knowledge<f64> = weight / (height * height)
// bmi.value = 22.74, bmi.\u03B5 = 0.96, provenance tracked

if bmi.\u03B5 >= 0.95 {
    approve_for_diagnosis(bmi)
} else {
    request_remeasurement(bmi)
}`,
  },
  {
    id: 'causal',
    scientistTitle: 'Correlation Is Not Causation',
    technicalTitle: 'Causal Reasoning',
    problem: 'Most code cannot tell the difference between association and real cause.',
    scientistExplanation:
      'Sounio lets you build causal models and apply interventions that remove confounding. You get the true effect size \u2014 not just statistical correlation.',
    researchImpact: [
      'True treatment effect quantification',
      'Automatic confounder removal',
      'Stronger causal claims in papers',
      'Better clinical and policy decisions',
    ],
    status: 'Stdlib \u2014 actively maturing',
    code: `var dag = causal_dag_new()
dag = dag_add_node(dag, 0)  // treatment
dag = dag_add_node(dag, 1)  // outcome
dag = dag_add_edge(dag, 0, 1,
    ec_beta_new(8.0, 2.0), 0.5, 0.1)

// Pearl's do-operator removes confounding
let intervened = do_intervention(dag, 1)
let ate = average_treatment_effect(dag, 0, 1)`,
  },
  {
    id: 'knowledge',
    scientistTitle: 'Knowledge With Guardrails',
    technicalTitle: 'Knowledge<T> + Safety Gates',
    problem: 'Scientific data always carries uncertainty that normal code silently ignores.',
    scientistExplanation:
      'Sounio wraps your data in Knowledge<T> and enforces confidence thresholds. It refuses to let unsafe or unreliable results proceed.',
    researchImpact: [
      'Prevents dangerous overconfidence',
      "Enforces your lab\u2019s risk standards",
      'Clear compliance and audit trail',
      'Built-in protection for patients and conclusions',
    ],
    status: 'Fully implemented',
    code: `let dose: Knowledge<f64> = Knowledge(
    500.0,
    \u03B5=0.92,
    prov="ASHP_2020_RCT"
)

let validated: Validated<f64> = validate(
    dose, threshold: 0.85
)

if dose.\u03B5 >= 0.85 {
    approve_dose(dose)
} else {
    request_remeasurement(dose)  // safety gate triggered
}`,
  },
];

export default function ComparisonMatrix() {
  const { audience } = useAudience();
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const isScientist = audience === 'scientist';

  return (
    <div className="grid gap-8">
      {scenarios.map((s) => (
        <div key={s.id} className="glass glass-specular rounded-[var(--radius-xl)] p-[clamp(1.1rem,3vw,2rem)]">
          <h3 className="text-[clamp(1.3rem,3vw,1.7rem)] font-bold text-[var(--color-text-primary)] mb-3">
            {isScientist ? s.scientistTitle : s.technicalTitle}
          </h3>

          <p className="text-[var(--color-text-secondary)] mb-6">{s.problem}</p>

          {isScientist && (
            <div className="mb-8 p-5 bg-[rgba(0,0,0,0.25)] rounded-xl border border-[var(--glass-border)]">
              <p className="text-[1.02rem] leading-relaxed text-[var(--color-text-primary)]">
                {s.scientistExplanation}
              </p>
            </div>
          )}

          <div className="mb-8">
            <p className="uppercase text-xs tracking-[0.12em] text-[var(--color-accent-blue)] font-semibold mb-3">
              What this means for your research
            </p>
            <ul className="grid md:grid-cols-2 gap-2.5 text-sm list-none p-0 m-0">
              {s.researchImpact.map((item, i) => (
                <li key={i} className="flex gap-2.5 text-[var(--color-text-secondary)]">
                  <span className="text-[var(--color-accent-blue)] shrink-0">\u2192</span> {item}
                </li>
              ))}
            </ul>
          </div>

          <button
            onClick={() => setExpanded((prev) => ({ ...prev, [s.id]: !prev[s.id] }))}
            className="w-full flex items-center justify-between bg-[rgba(0,0,0,0.2)] hover:bg-[rgba(0,0,0,0.3)] px-5 py-3.5 rounded-xl border border-[var(--glass-border)] transition-colors text-left"
          >
            <span className="font-semibold text-sm text-[var(--color-text-secondary)]">
              {isScientist ? 'Show the actual Sounio code' : 'Show code comparison'}
            </span>
            <span
              className="text-[var(--color-text-tertiary)] transition-transform duration-300"
              style={{ transform: expanded[s.id] ? 'rotate(180deg)' : 'rotate(0deg)' }}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
              </svg>
            </span>
          </button>

          {expanded[s.id] && (
            <pre className="mt-2 px-5 py-5 bg-[#0d1117] rounded-xl text-[0.82rem] leading-relaxed text-emerald-400/90 font-mono overflow-x-auto">
              {s.code}
            </pre>
          )}

          <div className="mt-4">
            <span className="px-3 py-1 rounded-full text-[0.68rem] font-semibold tracking-[0.06em] bg-[rgba(255,255,255,0.06)] text-[var(--color-text-tertiary)] border border-[var(--glass-border)]">
              {s.status}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
