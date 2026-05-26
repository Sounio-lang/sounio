import { useState } from 'react';
import { useAudience } from '../../lib/useAudience';
import { codeExplanations } from '../../lib/audienceContent';

interface TabExample {
  id: string;
  label: string;
  python: string;
  sounio: string;
}

const scientistExamples: TabExample[] = [
  {
    id: 'uncertainty',
    label: 'Uncertainty (ISO GUM)',
    python: `# Python: bare floats, no uncertainty tracking
mass = 75.3        # kg (measurement from standard scale)
height = 1.82      # meters (measurement from laser)

bmi = mass / (height ** 2)
# bmi = 22.74
# Problem: We have no idea how accurate this result is.
# If the scale had a 1.5kg drift and stadiometer was uncalibrated,
# is this patient still safely below the overweight threshold?
# Traditional programming leaves you in the dark.`,
    sounio: `// Sounio: built-in uncertainty (ISO GUM propagation)
// ε is the epistemic confidence level (0.00 to 1.00)
let mass: Knowledge<f64> = Knowledge(
    75.3,
    ε=0.98,
    prov="clinical_scale_model_B"
)

let height: Knowledge<f64> = Knowledge(
    1.82,
    ε=0.99,
    prov="laser_stadiometer_calibrated"
)

// The mathematical division automatically propagates uncertainty:
let bmi: Knowledge<f64> = mass / (height * height)
// bmi.value = 22.74, bmi.ε = 0.96 (computed rigorously!)

if bmi.ε >= 0.95 {
    println("Confidence high enough for diagnostic decision.")
} else {
    println("WARNING: Uncertainty is too high. Please remeasure.")
}`,
  },
  {
    id: 'causality',
    label: 'Causal Decisions',
    python: `# Python: plain numbers representing interventions
# No guarantee of causal consistency or confounding controls.
def calculate_treatment_effect(data):
    # Just a mathematical average, has no concept of confounding
    return data['outcome'].mean() - data['control'].mean()

# Problem: Traditional programming cannot distinguish
# correlation from causation. You can easily mistake an association
# for a clinical treatment effect and prescribe wrong treatments.`,
    sounio: `// Sounio: Pearl's Causal do-calculus built into the type system
// Intervention<T> and Counterfactual<T> are language keywords.

fn main() -> i32 with IO {
    // 1. Build a causal DAG representing biological pathways
    var dag = causal_dag_new()
    dag = dag_add_node(dag, 0) // Drug treatment node
    dag = dag_add_node(dag, 1) // Patient recovery node
    
    // 2. Add causal influence edge with epistemic uncertainty
    dag = dag_add_edge(dag, 0, 1, beta(8.0, 2.0), uncertainty: 0.1)

    // 3. Apply Pearl's do-operator to mathematically isolate 
    // the causal effect from confounding variables
    let effect: Counterfactual<f64> = do_intervention(dag, treatment: 0, outcome: 1)
    
    println("Rigorous, confounder-free causal effect computed.")
    0
}`,
  },
  {
    id: 'provenance',
    label: 'Clinical Provenance',
    python: `# Python: arbitrary data from database, no trace of source device
patient_record = db.query("SELECT * FROM labs WHERE patient_id = 901")
creatinine = patient_record['creatinine']

# Problem: Is this data from a calibrated Abbott or Beckman analyzer?
# What calibration offset was applied? Did someone edit this manually?
# Standard programming treats this value as a raw, untraceable number.`,
    sounio: `// Sounio: provenance metadata travels with Knowledge<T>
struct LabResult {
    creatinine: f64
}

fn process_lab(res: Knowledge<LabResult>) with IO, Panic {
    // res.prov records source instrument / run id (string metadata)
    // res.epsilon tracks measurement confidence after GUM propagation
    
    if res.prov == "lab_run_042" && res.epsilon >= 0.90 {
        let val = res.value.creatinine
        println("Laboratory result with auditable metadata processed.")
    } else {
        panic("Insufficient confidence or missing provenance. Rejected.")
    }
}`,
  },
];

const technicalExamples: TabExample[] = [
  {
    id: 'uncertainty',
    label: 'Uncertainty',
    python: `# Python: bare floats, no uncertainty tracking
mass = 75.3        # kg (measurement from standard scale)
height = 1.82      # meters (measurement from laser)

bmi = mass / (height ** 2)
# bmi = 22.74
# Problem: We have no idea how accurate this result is.
# If the scale had a 1.5kg drift and stadiometer was uncalibrated,
# is this patient still safely below the overweight threshold?
# Traditional programming leaves you in the dark.`,
    sounio: `// Sounio: built-in uncertainty (ISO GUM propagation)
// ε is the epistemic confidence level (0.00 to 1.00)
let mass: Knowledge<f64> = Knowledge(
    75.3,
    ε=0.98,
    prov="clinical_scale_model_B"
)

let height: Knowledge<f64> = Knowledge(
    1.82,
    ε=0.99,
    prov="laser_stadiometer_calibrated"
)

// The mathematical division automatically propagates uncertainty:
let bmi: Knowledge<f64> = mass / (height * height)
// bmi.value = 22.74, bmi.ε = 0.96 (computed rigorously!)

if bmi.ε >= 0.95 {
    println("Confidence high enough for diagnostic decision.")
} else {
    println("WARNING: Uncertainty is too high. Please remeasure.")
}`,
  },
  {
    id: 'causality',
    label: 'Causal Safety',
    python: `# Python: plain numbers representing interventions
# No guarantee of causal consistency or confounding controls.
def calculate_treatment_effect(data):
    # Just a mathematical average, has no concept of confounding
    return data['outcome'].mean() - data['control'].mean()

# Problem: Traditional programming cannot distinguish
# correlation from causation. You can easily mistake an association
# for a clinical treatment effect and prescribe wrong treatments.`,
    sounio: `// Sounio: Pearl's Causal do-calculus built into the type system
// Intervention<T> and Counterfactual<T> are language keywords.

fn main() -> i32 with IO {
    // 1. Build a causal DAG representing biological pathways
    var dag = causal_dag_new()
    dag = dag_add_node(dag, 0) // Drug treatment node
    dag = dag_add_node(dag, 1) // Patient recovery node
    
    // 2. Add causal influence edge with epistemic uncertainty
    dag = dag_add_edge(dag, 0, 1, beta(8.0, 2.0), uncertainty: 0.1)

    // 3. Apply Pearl's do-operator to mathematically isolate 
    // the causal effect from confounding variables
    let effect: Counterfactual<f64> = do_intervention(dag, treatment: 0, outcome: 1)
    
    println("Rigorous, confounder-free causal effect computed.")
    0
}`,
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

interface CodeExamplesProps {
  locale?: string;
}

const tabLabelsByLocale: Record<string, Record<string, string>> = {
  en: {
    'Uncertainty (ISO GUM)': 'Uncertainty (ISO GUM)',
    'Causal Decisions': 'Causal Decisions',
    'Clinical Provenance': 'Clinical Provenance',
    Uncertainty: 'Uncertainty',
    Resources: 'Resources',
    Algebra: 'Algebra',
  },
  pt: {
    'Uncertainty (ISO GUM)': 'Incerteza (ISO GUM)',
    'Causal Decisions': 'Decisões Causais',
    'Clinical Provenance': 'Proveniência Clínica',
    Uncertainty: 'Incerteza',
    Resources: 'Recursos',
    Algebra: 'Álgebra',
  },
  es: {
    'Uncertainty (ISO GUM)': 'Incertidumbre (ISO GUM)',
    'Causal Decisions': 'Decisiones Causales',
    'Clinical Provenance': 'Procedencia Clínica',
    Uncertainty: 'Incertidumbre',
    Resources: 'Recursos',
    Algebra: 'Álgebra',
  },
  el: {
    'Uncertainty (ISO GUM)': 'Αβεβαιότητα (ISO GUM)',
    'Causal Decisions': 'Αιτιώδεις Αποφάσεις',
    'Clinical Provenance': 'Κλινική Προέλευση',
    Uncertainty: 'Αβεβαιότητα',
    Resources: 'Πόροι',
    Algebra: 'Άλγεβρα',
  },
  zh: {
    'Uncertainty (ISO GUM)': '不确定性 (ISO GUM)',
    'Causal Decisions': '因果决策',
    'Clinical Provenance': '临床溯源',
    Uncertainty: '不确定性',
    Resources: '资源',
    Algebra: '代数',
  },
  ja: {
    'Uncertainty (ISO GUM)': '不確実性 (ISO GUM)',
    'Causal Decisions': '因果的決定',
    'Clinical Provenance': '臨床プロベナンス',
    Uncertainty: '不確実性',
    Resources: 'リソース',
    Algebra: '代数',
  },
};

function localizeTabLabel(label: string, locale: string): string {
  const map = tabLabelsByLocale[locale] ?? tabLabelsByLocale.en;
  return map[label] ?? label;
}

export default function CodeExamples({ locale = 'en' }: CodeExamplesProps) {
  const [activeTab, setActiveTab] = useState('uncertainty');
  const [codeExpanded, setCodeExpanded] = useState<Record<string, boolean>>({});
  const { audience } = useAudience();

  const isScientist = audience === 'scientist';
  const examplesList = isScientist ? scientistExamples : technicalExamples;
  
  // If activeTab is not in active list, fall back to first item of active list
  const currentTabIsValid = examplesList.some((e) => e.id === activeTab);
  const effectiveTab = currentTabIsValid ? activeTab : examplesList[0].id;
  const activeExample = examplesList.find((e) => e.id === effectiveTab) || examplesList[0];

  const isExpanded = codeExpanded[effectiveTab] ?? !isScientist;

  const ptExplanations: Record<string, string> = {
    uncertainty:
      'O que isto significa: Cada medição médica carrega seu próprio nível de confiança epistêmica (ε) e sua proveniência (prov - ex: o modelo do aparelho ou balança). Ao calcular o IMC, o Sounio propaga e calcula de forma matematicamente rigorosa a incerteza combinada final. Se a incerteza for muito alta para garantir uma tomada de decisão médica segura, o sistema alerta o profissional para refazer a medição.',
    causality:
      'O que isto significa: Ao contrário das linguagens comuns que tratam números causais como meros floats isolados, o Sounio integra o do-calculus de Pearl nativamente. Você define grafos causais de caminhos biológicos e, ao aplicar a intervenção, o tipo Counterfactual garante o controle de confundidores e calcula a incerteza real do efeito do tratamento antes de aplicá-lo ao paciente.',
    provenance:
      'O que isto significa: Valores clínicos podem carregar metadados de proveniência (prov) e confiança epistémica (ε) no tipo Knowledge<T>. Isto permite trilhas de auditoria reproduzíveis no código — não promete integração hospitalar, blockchain institucional ou conformidade regulatória pronta a usar.',
    resources:
      'O que isto significa: Ao lidar com recursos críticos (como conexões a prontuários, monitores ou arquivos), os tipos lineares do Sounio garantem que um recurso seja fechado exatamente uma vez. Se você esquecer de liberar, ou tentar usar após fechar, o compilador rejeita o código na hora, evitando travamentos e vazamentos no hospital.',
    algebra:
      'O que isto significa: O Sounio permite expressar leis matemáticas diretamente nos tipos (ex: a não-associatividade de redes neurais baseadas em octoniões). O compilador impede otimizações matemáticas automáticas incorretas que outras ferramentas (como NumPy ou PyTorch) fazem por assumir que floats tradicionais são sempre associativos.',
  };

  const enExplanations: Record<string, string> = {
    ...codeExplanations,
    provenance:
      'What this means: Clinical values can carry provenance metadata (prov) and epistemic confidence (ε) on Knowledge<T>. This supports auditable, reproducible trails in code — it does not promise hospital integration, institutional blockchain, or turnkey regulatory compliance.'
  };

  const explanations = locale === 'pt' ? ptExplanations : enExplanations;
  const explanation = explanations[effectiveTab];

  const trans = {
    en: {
      title: 'See the difference',
      descScientist: 'Your current tools compute the number. Sounio computes what you can trust about that number.',
      descTechnical: 'Python computes the number. Sounio computes what you can trust.',
      showCode: 'Show Sounio code',
      hideCode: 'Hide code',
      plainLanguage: 'In plain language: '
    },
    pt: {
      title: 'Veja a diferença prática',
      descScientist: 'As suas ferramentas atuais calculam apenas o número. O Sounio calcula o quanto você pode confiar nesse número.',
      descTechnical: 'O Python calcula o número. O Sounio calcula a confiança.',
      showCode: 'Mostrar código Sounio',
      hideCode: 'Ocultar código',
      plainLanguage: 'Em termos simples: '
    }
  };

  const d = trans[locale === 'pt' ? 'pt' : 'en'];

  const toggleCode = () => {
    setCodeExpanded((prev) => ({ ...prev, [effectiveTab]: !isExpanded }));
  };

  return (
    <section id="code" className="py-[clamp(3.5rem,7vw,6rem)] bg-[color-mix(in_srgb,var(--color-bg-alt)_74%,transparent)] border-y border-[var(--glass-border)]">
      <div className="container">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            {d.title}
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[68ch]">
            {isScientist ? d.descScientist : d.descTechnical}
          </p>
        </div>

        {/* Tab bar */}
        <div className="flex justify-start md:justify-center mb-8 overflow-x-auto no-scrollbar max-w-full px-4 md:px-0">
          <div className="inline-flex gap-1 p-1 rounded-full glass whitespace-nowrap">
            {examplesList.map((example) => (
              <button
                key={example.id}
                onClick={() => setActiveTab(example.id)}
                className={`px-4 md:px-6 py-2 md:py-2.5 rounded-full text-xs md:text-sm font-medium transition-all duration-200 ${
                  effectiveTab === example.id
                    ? 'bg-[var(--color-text-primary)] text-[var(--color-surface-primary)]'
                    : 'text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]'
                }`}
              >
                {localizeTabLabel(example.label, locale)}
              </button>
            ))}
          </div>
        </div>

        {/* Side-by-side comparison */}
        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Python panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--glass-border)] min-w-0">
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
              <pre className="p-4 md:p-6 bg-[#0d1117] text-xs md:text-sm leading-relaxed overflow-x-auto min-h-[320px] md:min-h-[400px]">
                <code className="text-[#e6edf3]">{activeExample.python}</code>
              </pre>
            </div>
          </div>

          {/* Sounio panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--color-accent-gold)]/20 min-w-0">
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
              <pre className="p-4 md:p-6 bg-[#0d1117] text-xs md:text-sm leading-relaxed overflow-x-auto min-h-[320px] md:min-h-[400px]">
                <code className="text-[#e6edf3]">{activeExample.sounio}</code>
              </pre>
            </div>
            {isScientist && (
              <div className="px-4 py-3 bg-[#0d1117] border-t border-[var(--glass-border)] flex justify-center">
                <button onClick={toggleCode} className="code-toggle-btn">
                  <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d={isExpanded ? "M19 9l-7 7-7-7" : "M9 5l7 7-7 7"} />
                  </svg>
                  {isExpanded ? d.hideCode : d.showCode}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Explanation panel for scientist mode */}
        {isScientist && explanation && (
          <div className="explanation-panel rounded-2xl mt-4 max-w-6xl mx-auto">
            <strong>{d.plainLanguage}</strong>{explanation}
          </div>
        )}

      </div>
    </section>
  );
}
