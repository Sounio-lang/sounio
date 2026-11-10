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
    question: 'Is the compiler internals page the effect system?',
    answer:
      'No. This page is the effect system for learners. The effect checker internals page, at /learn/compiler/effect-system/, is the source map of the checker.',
  },
  {
    question: 'Does Sounio have algebraic effects?',
    answer:
      'A declared effect is checked in the signature when the name is not held. Madaros handlers are deep and linear one-shot, with the limits named on this page. Multi-shot handlers are not a goal.',
  },
  {
    question: 'Which effects are checked?',
    answer:
      'A closed list of 30 built-in names (ids 0-29). IO, Mut, Div and Panic are the common ones. A declared effect outside that list is checked too, unless the name is held.',
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
      'On Madaros, handlers are deep and a continuation is linear one-shot. Resume at most once on each path. Zero uses are allowed. Twice is a compile error. Multi-shot continuations are not a goal.',
  },
  {
    question: 'Where is the implementation?',
    answer:
      'The call check is effects_subset in self-hosted/check/effects.sio. Declared effects are registered by checker_collect_effect_def_inplace in self-hosted/check/check.sio. The source map is /learn/compiler/effect-system/.',
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
