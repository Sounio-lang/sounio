import type { Audience } from './audience';

// ---------------------------------------------------------------------------
// CinematicHero
// ---------------------------------------------------------------------------

export const heroContent: Record<Audience, {
  badge: string;
  h1Line1: string;
  h1Line2: string;
  subtitle: string;
  cta1Label: string;
  cta1Href: string;
  cta2Label: string;
  cta2Href: string;
}> = {
  scientist: {
    badge: 'FOR RESEARCHERS \u00B7 PHARMACOLOGY \u00B7 NEUROSCIENCE \u00B7 CLIMATE \u00B7 QUANTUM',
    h1Line1: 'Your Calculations',
    h1Line2: 'Hide Uncertainty',
    subtitle:
      'When you compute a drug dose, an fMRI threshold, or a climate projection, the answer is a number. But how confident is that number? Where did the input data come from? What happens when the measurement error matters? Sounio is a programming language that tracks all of this automatically.',
    cta1Label: 'See How Labs Use Sounio',
    cta1Href: '/science',
    cta2Label: 'Explore a Dosing Example',
    cta2Href: '/docs/vancomycin-uncertainty',
  },
  technical: {
    badge: 'CURRENT CONTRACT \u00B7 BETA COMPILER + GATED STDLIB',
    h1Line1: 'Scientific Code',
    h1Line2: 'That Shows Its Work',
    subtitle:
      'Sounio combines explicit uncertainty, provenance-aware types, and gate-backed compiler workflows. The current repo is strongest as a check-first platform with validated scientific lanes, not as a fully finished all-backends release.',
    cta1Label: 'Read the Current Contract',
    cta1Href: '/learn/getting-started',
    cta2Label: 'Open Playground',
    cta2Href: '/playground',
  },
};

// ---------------------------------------------------------------------------
// ScrollStorySection
// ---------------------------------------------------------------------------

interface ScrollSegment {
  heading: string;
  headingHighlight: string;
  body: string;
}

export const scrollStoryContent: Record<Audience, [ScrollSegment, ScrollSegment, ScrollSegment]> = {
  scientist: [
    {
      heading: 'Numbers Without Context Are',
      headingHighlight: 'Dangerous.',
      body: 'Your Python script says the dose is 500\u2009mg. But was the patient\u2019s kidney function measured today or last week? How much did the scale drift? These questions have answers \u2014 your current tools just don\u2019t track them.',
    },
    {
      heading: 'What If the Computer',
      headingHighlight: 'Tracked It For You?',
      body: 'Imagine every measurement carrying its confidence level, its source instrument, and its date. When you multiply two uncertain values, the uncertainty propagates automatically. No manual error bars. No spreadsheet formulas.',
    },
    {
      heading: 'Decisions You Can',
      headingHighlight: 'Defend.',
      body: 'When confidence is high enough, approve the dose. When it\u2019s not, the system itself requests remeasurement. Every decision traces back to the evidence that supports it.',
    },
  ],
  technical: [
    {
      heading: 'Syntax is not',
      headingHighlight: 'Semantics.',
      body: 'Traditional tools stop at the AST. We traverse deeper, encoding your intent into the type system so the compiler can enforce it.',
    },
    {
      heading: 'Enter the',
      headingHighlight: 'Type Checker.',
      body: 'Epistemic constraints are tracked through the type system. If your confidence levels conflict, the compiler refuses to build.',
    },
    {
      heading: 'Compile to',
      headingHighlight: 'Certainty.',
      body: 'We generate native binaries from the self-hosted compiler. Your deployment reflects the epistemic guarantees checked at compile time.',
    },
  ],
};

// ---------------------------------------------------------------------------
// Scientist-mode code explanations for ComparisonMatrix
// ---------------------------------------------------------------------------

export const comparisonExplanations: Record<string, string> = {
  uncertainty:
    'What this means: Sounio tracks where each measurement came from (the \u201cprov\u201d field) and how confident it is (the \u03B5 value). When you divide two uncertain values, it automatically calculates how confident the result is. If confidence drops below your threshold, the system flags it.',
  causal:
    'What this means: Instead of treating causal claims as regular numbers, Sounio wraps them in special types (Intervention, Counterfactual) that carry confidence levels. You build a causal graph, apply Pearl\u2019s do-operator, and the system tracks how confident the causal estimate is before you can act on it.',
  resources:
    'What this means: When your code opens a file, connection, or instrument handle, Sounio guarantees you close it exactly once. If you forget, or try to use it after closing, the compiler stops you before the program ever runs.',
  ontology:
    'What this means: Scientific domain terms (diseases, genes, phenotypes) can be represented as compiler-checked types instead of plain text strings. Typos and invalid codes are caught at compile time. Full ontology integration with SNOMED/GO/HPO is planned for 2026.',
};

// ---------------------------------------------------------------------------
// Scientist-mode code explanations for CodeExamples
// ---------------------------------------------------------------------------

export const codeExplanations: Record<string, string> = {
  uncertainty:
    'What this means: Every measurement carries its confidence level (\u03B5) and where it came from (prov). When you compute BMI from weight and height, the system automatically calculates how confident the result is. If confidence is high enough, you can proceed. If not, it tells you to remeasure.',
  causality:
    'What this means: Instead of representing interventions as plain dictionaries, Sounio uses a typed causal DAG. You add nodes (treatment, outcome), edges with uncertainty, and apply Pearl\u2019s do-operator. The system tracks how confident the causal effect estimate is.',
  resources:
    'What this means: A linear struct can only be used once. After you pass it to close_file(), the compiler prevents you from using it again. This eliminates double-free bugs and resource leaks at compile time, not at runtime.',
  algebra:
    'What this means: Sounio lets you declare the mathematical laws your data obeys \u2014 like \u201coctonion multiplication is non-associative\u201d \u2014 and the compiler enforces those laws in every optimization. Tools like NumPy assume floating-point associativity everywhere; Sounio respects what your algebra actually allows.',
};

// ---------------------------------------------------------------------------
// Capability cards (index.astro "Why Sounio" section)
// ---------------------------------------------------------------------------

interface CapabilityCard {
  title: string;
  body: string;
  accent: string;
}

export const capabilityCards: Record<Audience, CapabilityCard[]> = {
  scientist: [
    {
      title: 'Uncertainty Tracking',
      body: 'Every measurement in your analysis carries its confidence level. When you combine measurements, Sounio automatically calculates how confident the result is.',
      accent: 'var(--color-accent-gold)',
    },
    {
      title: 'Data Provenance',
      body: 'Know exactly which instrument, which run, and which calibration produced each number. Essential for reproducible science and regulatory compliance.',
      accent: 'var(--color-accent-teal)',
    },
    {
      title: 'Automatic Error Propagation',
      body: 'No more manual uncertainty formulas in spreadsheets. Sounio follows the GUM standard (JCGM 100:2008) to propagate measurement uncertainty through all your calculations — including through ODEs and neural networks.',
      accent: 'var(--color-accent-blue)',
    },
    {
      title: 'Safety Gates',
      body: 'Set minimum confidence thresholds. If a calculation\u2019s confidence drops too low, the system prevents it from being used in downstream decisions.',
      accent: 'var(--color-accent-green)',
    },
  ],
  technical: [
    {
      title: 'Epistemic Core',
      body: 'Knowledge<T> with GUM propagation, Unobserved<T> for observation boundaries, and epistemic closures that auto-infer the Epistemic effect when capturing uncertain values.',
      accent: 'var(--color-accent-gold)',
    },
    {
      title: 'Algebra System',
      body: 'Declare algebraic laws on types (commutative, alternative, Fano-selective). The e-graph optimizer only applies rewrites permitted by your declared algebra — Quaternion, Octonion, Clifford, G2.',
      accent: 'var(--color-accent-teal)',
    },
    {
      title: 'Self-Hosted Stack v2.0',
      body: 'gen2==gen3 fixed point (bit-identical, 230KB ELF). Ownership tracking with use-after-move and move-while-borrowed enforced at compile time.',
      accent: 'var(--color-accent-blue)',
    },
    {
      title: 'Gate-Tracked Stdlib',
      body: '743 stdlib files, 257 run-pass tests, 1,003+ e-graph rewrite rules. Repo artifacts track which stdlib lanes are callable today and which are still gated.',
      accent: 'var(--color-accent-green)',
    },
  ],
};

// ---------------------------------------------------------------------------
// Bottom CTA (index.astro final section)
// ---------------------------------------------------------------------------

export const bottomCta: Record<Audience, {
  heading: string;
  body: string;
  cta1Label: string;
  cta1Href: string;
  cta2Label: string;
  cta2Href: string;
  showShellCommand: boolean;
}> = {
  scientist: {
    heading: 'Ready to stop guessing?',
    body: 'Start with the vancomycin dosing case study \u2014 a real example of how Sounio tracks confidence from raw measurement to clinical decision.',
    cta1Label: 'See the Vancomycin Case Study',
    cta1Href: '/docs/vancomycin-uncertainty',
    cta2Label: 'Browse Scientific Domains',
    cta2Href: '/science',
    showShellCommand: false,
  },
  technical: {
    heading: 'Ship scientific software that explains itself.',
    body: 'Start with the parts of Sounio that are already verified, then expand into the self-hosted and domain-specific lanes with the gate artifacts in hand.',
    cta1Label: 'Enter the Learning Hub',
    cta1Href: '/learn',
    cta2Label: 'Explore on GitHub',
    cta2Href: 'https://github.com/sounio-lang/sounio',
    showShellCommand: true,
  },
};

// ---------------------------------------------------------------------------
// "The Sounio Difference" section subtitle (index.astro)
// ---------------------------------------------------------------------------

export const differenceSubtitle: Record<Audience, string> = {
  scientist:
    'See how Sounio handles real scientific problems differently from traditional tools. Click "Show Sounio code" to see the actual syntax.',
  technical:
    'The current repository focus is explicit uncertainty, auditable semantics, and evidence that lives next to the code.',
};

// ---------------------------------------------------------------------------
// "Built For" section heading (index.astro)
// ---------------------------------------------------------------------------

export const builtForContent: Record<Audience, { heading: string; body: string }> = {
  scientist: {
    heading: 'Built For Researchers Who Need Trustworthy Results',
    body: 'Sounio is designed for teams where a wrong calculation has real consequences \u2014 a miscalculated drug dose, a misinterpreted brain scan, a climate projection that misleads policy.',
  },
  technical: {
    heading: 'Built For High-Consequence Scientific Systems',
    body: 'Sounio is strongest where teams need to prove what a program checked, what was only sketched, and which outputs are backed by reproducible gate artifacts.',
  },
};
