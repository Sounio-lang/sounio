/**
 * Visible FAQ and FAQPage JSON-LD for /learn/effects/.
 * The strings are the page text. Do not paraphrase one side only.
 */
export const EFFECTS_FAQ = [
  {
    question: 'What is an effect system?',
    answer: "An effect system tracks a function's side effects in its type so the compiler can check them.",
  },
  {
    question: 'Does Sounio have algebraic effects?',
    answer:
      'Effects in the signature are checked for a closed built-in set; user-declared effects and handlers are parsed but not implemented yet.',
  },
  {
    question: 'Which effects are checked?',
    answer:
      'A closed list of 30 built-in names (ids 0-29). IO, Mut, Div and Panic are the common ones. User-declared effects are parsed but not implemented.',
  },
  {
    question: 'Is it row-polymorphic like Koka?',
    answer:
      'No. TyEffectVar and TyEffectRow exist and nothing calls them. Madaros checks a set of at most eight effects per function. lean_single checks a bitmask.',
  },
  {
    question: 'Can a pure function call an IO function?',
    answer:
      'No. The caller must declare IO. Both engines report E035, "effect not declared in function signature".',
  },
  {
    question: 'Are effects inferred?',
    answer:
      'Named functions are not inferred by default. A missing with clause is E035 on both engines. Closure effects are inferred onto the closure signature. Cross-function inference for named functions runs only when SOUNIO_EFFECT_INFER=1, and that flag is off by default.',
  },
  {
    question: 'Do handlers resume (one-shot / multi-shot)?',
    answer:
      'Handlers are parsed, not implemented. Resumption is not specified (SOUNIO-SPEC-07 status is "undefined").',
  },
  {
    question: 'Where is the implementation?',
    answer:
      'The call check is effects_subset in self-hosted/check/effects.sio, called from self-hosted/check/check.sio. self-hosted/effects/ has no use effects:: importer under self-hosted/.',
  },
];

export function effectsFaqPage() {
  return {
    '@context': 'https://schema.org',
    '@type': 'FAQPage',
    mainEntity: EFFECTS_FAQ.map(({ question, answer }) => ({
      '@type': 'Question',
      name: question,
      acceptedAnswer: {
        '@type': 'Answer',
        text: answer,
      },
    })),
  };
}
