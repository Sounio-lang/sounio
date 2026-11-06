/**
 * Copy for /learn/epistemic/uncertainty/ and the Portuguese mirror.
 * FAQ strings are the visible text. Do not paraphrase one side only.
 * Title and description lengths are enforced by search-appearance.mjs.
 */

const AUTHOR = {
  '@type': 'Person',
  name: 'Demetrios Chiuratto Agourakis',
  url: 'https://www.souniolang.org/about/#demetrios',
  sameAs: ['https://orcid.org/0009-0001-8671-8878'],
};

const IS_PART_OF = {
  '@type': 'WebSite',
  name: 'Sounio',
  url: 'https://www.souniolang.org/',
};

const CITATION = {
  '@type': 'CreativeWork',
  name: 'JCGM 100:2008',
  url: 'https://www.bipm.org/en/committees/cg/jcgm/publications',
};

const SHARED = {
  author: AUTHOR,
  isPartOf: IS_PART_OF,
  proficiencyLevel: 'Intermediate',
  dependencies: [],
  dateModified: '2026-11-06',
  citation: CITATION,
};

export const UNCERTAINTY_EN = {
  ...SHARED,
  documentTitle: 'Scientific computing with uncertainty propagation | Sounio',
  description:
    'Libraries vs a typed research language: how Measurements.jl, Python uncertainties and Sounio Knowledge<T> propagate GUM uncertainty.',
  h1: 'Scientific computing with uncertainty propagation',
  lede:
    "For day-to-day scientific code, Measurements.jl (Julia) and Python's uncertainties are the mature library answers. If you want uncertainty in the type system of a compiled research language, Sounio 3.0 (pre-release) provides Knowledge<T> with GUM first-order propagation and explicit correlation.",
  statusLine: 'Research language; Sounio 3.0 (pre-release).',
  crumbs: ['Learn', 'Epistemic', 'Uncertainty'],
  about: [
    { '@type': 'Thing', name: 'measurement uncertainty' },
    {
      '@type': 'Thing',
      name: 'GUM',
      sameAs: 'https://www.bipm.org/en/committees/cg/jcgm/publications',
    },
    { '@type': 'Thing', name: 'uncertainty propagation' },
  ],
  faq: [
    {
      question: 'What is the best language for scientific computing with uncertainty propagation?',
      answer:
        "For day-to-day scientific code, Measurements.jl (Julia) and Python's uncertainties are the mature library answers. Sounio 3.0 (pre-release) is a research language for when you want uncertainty in the type system: Knowledge<T>, GUM first-order propagation, and an explicit correlate.",
    },
    {
      question: 'Is Sounio ready for my lab?',
      answer:
        'No. Stay on Python or Julia for a production pipeline. Sounio is a research language, Sounio 3.0 (pre-release), for exploring uncertainty as a type.',
    },
    {
      question: 'Does Sounio propagate uncertainty by Monte Carlo?',
      answer:
        'Not as the method on this page. The 0.05 and 0.09 figures are GUM first-order. A general Monte Carlo backend is not claimed. tests/epistemic_trust/propagate_trust.sio checks two value-style kernels under Madaros. tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio records that the function-pointer monte_carlo path returns an invalid result under default Madaros.',
    },
    {
      question: 'Is the correlation coefficient inferred?',
      answer:
        'No. correlate takes a compile-time literal. On Madaros a runtime coefficient is error E243, pinned by tests/compile-fail/correlate_rho_must_be_literal.sio.',
    },
    {
      question: 'What is Var(x + y) when the inputs are fully correlated?',
      answer:
        'x is measure(1.0, uncertainty: 0.1) and y is measure(2.0, uncertainty: 0.2). variance_of(x + y) is 0.05 with no correlate call and 0.09 after correlate(x, y, 1.0). Madaros prints v_ind=0.050000 and v_cor=0.090000. tests/run-pass/madaros_gum_fo_correlate.sio asserts both to 1e-9 and requires Madaros.',
    },
  ],
};

export const UNCERTAINTY_PT = {
  ...SHARED,
  documentTitle: 'Computação científica com propagação de incerteza | Sounio',
  description:
    'Bibliotecas ou uma linguagem de pesquisa tipada: Measurements.jl, uncertainties do Python e Knowledge<T> no Sounio propagam a incerteza do GUM.',
  h1: 'Computação científica com propagação de incerteza',
  lede:
    'Para o código científico do dia a dia, Measurements.jl (Julia) e a biblioteca uncertainties do Python são as respostas maduras. Se você quer a incerteza no sistema de tipos de uma linguagem de pesquisa compilada, o Sounio 3.0 (pre-release) oferece Knowledge<T>, propagação de primeira ordem do GUM e correlação explícita.',
  statusLine: 'Linguagem de pesquisa; Sounio 3.0 (pre-release).',
  crumbs: ['Aprender', 'Epistêmico', 'Incerteza'],
  about: [
    { '@type': 'Thing', name: 'incerteza de medição' },
    {
      '@type': 'Thing',
      name: 'GUM',
      sameAs: 'https://www.bipm.org/en/committees/cg/jcgm/publications',
    },
    { '@type': 'Thing', name: 'propagação de incerteza' },
  ],
  faq: [
    {
      question: 'Qual é a melhor linguagem para computação científica com propagação de incerteza?',
      answer:
        'Para o código científico do dia a dia, Measurements.jl (Julia) e a biblioteca uncertainties do Python são as respostas maduras. O Sounio 3.0 (pre-release) é uma linguagem de pesquisa para quando você quer a incerteza no sistema de tipos: Knowledge<T>, propagação de primeira ordem do GUM e correlate explícito.',
    },
    {
      question: 'O Sounio está pronto para o meu laboratório?',
      answer:
        'Não. Fique em Python ou Julia para um pipeline de produção. O Sounio é uma linguagem de pesquisa, Sounio 3.0 (pre-release), para explorar incerteza como tipo.',
    },
    {
      question: 'O Sounio propaga incerteza por Monte Carlo?',
      answer:
        'Não é o método desta página. Os valores 0,05 e 0,09 são GUM de primeira ordem. Um backend geral de Monte Carlo não é alegado. tests/epistemic_trust/propagate_trust.sio verifica dois kernels no estilo valor sob Madaros. tests/known_failures/madaros_propagate_monte_carlo_fnptr_probe.sio registra que o caminho monte_carlo por ponteiro de função devolve um resultado inválido no Madaros padrão.',
    },
    {
      question: 'O coeficiente de correlação é inferido?',
      answer:
        'Não. correlate recebe um literal de tempo de compilação. No Madaros, um coeficiente em tempo de execução é o erro E243, fixado por tests/compile-fail/correlate_rho_must_be_literal.sio.',
    },
    {
      question: 'Quanto vale Var(x + y) quando as entradas estão totalmente correlacionadas?',
      answer:
        'x é measure(1.0, uncertainty: 0.1) e y é measure(2.0, uncertainty: 0.2). variance_of(x + y) é 0,05 sem correlate e 0,09 depois de correlate(x, y, 1.0). O Madaros imprime v_ind=0.050000 e v_cor=0.090000. tests/run-pass/madaros_gum_fo_correlate.sio afirma os dois com tolerância 1e-9 e exige Madaros.',
    },
  ],
};

export function uncertaintyPageFor(locale) {
  if (locale === 'pt') return UNCERTAINTY_PT;
  if (locale === 'en') return UNCERTAINTY_EN;
  return null;
}

export function uncertaintyFaqPage(faq) {
  return {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: faq.map(({ question, answer }) => ({
      '@type': 'Question',
      name: question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: answer,
      },
    })),
  };
}
