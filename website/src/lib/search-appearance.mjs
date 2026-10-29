/**
 * English <title> and meta description for the pages that Search Console
 * showed at positions 5–7 with almost no clicks (audit 27 Oct 2026).
 * Visible headings and localized copy stay where they are; BaseLayout
 * applies this map only when the page locale is English.
 *
 * Lengths are code units, the same count the search-appearance check uses.
 */
export const ENGLISH_SEARCH = {
  '/': {
    title: 'Sounio language: units, effects, and uncertainty types',
    description:
      'Sounio is a self-hosted systems language for uncertainty, physical units, and declared effects. Follow a program back to the compiler sources.',
  },
  '/learn/': {
    title: 'Sounio guides: effect system, units, and uncertainty',
    description:
      'Language guides for Sounio: units of measure, the effect system, refinement types, uncertainty propagation, and the self-hosted compiler.',
  },
  '/learn/getting-started/': {
    title: 'Sounio getting started: Linux prebuilt or macOS build',
    description:
      'Clone Sounio, then check and run a .sio program. Linux x86-64 uses the committed prebuilt; macOS ARM64 builds the compiler from source.',
  },
  '/learn/epistemic/': {
    title: 'Uncertainty propagation types in the Sounio language',
    description:
      'Epistemic types treat uncertainty, confidence, and provenance as program data. The guide shows Knowledge values and the checked unwrap boundary.',
  },
  '/learn/compiler/self-hosted/': {
    title: 'Sounio self-hosted compiler: x86-64 ELF without LLVM',
    description:
      "Sounio's compiler is written in Sounio: it type-checks, optimizes and emits static Linux x86-64 ELF binaries directly, with no LLVM or Cranelift.",
  },
  '/about/roadmap/': {
    title: 'Sounio roadmap: language, runtime, and what is open',
    description:
      'The Sounio roadmap tracks language, runtime, and ecosystem milestones, including runtime and backend work that is still in progress.',
  },
  '/learn/effects/': {
    title: 'Effect System in Sounio: Side Effects in the Signature',
    description:
      "How Sounio's effect system works: with IO, Mut, Div and Panic in every signature, custom effects, and what handlers do today. Code from real tests.",
  },
};

/** Approved S1 copy for the English self-hosted compiler page. */
export const SELF_HOSTED_PAGE = {
  h1: 'Sounio self-hosted compiler',
  lede: 'The Sounio compiler is written in Sounio (.sio). It parses, type-checks, lowers to an IR, runs its optimization passes and emits a static Linux x86-64 ELF binary directly, with no LLVM or Cranelift on that path. Linux x86-64 is the default shipped target.',
  crumbs: ['Learn', 'Compiler', 'Self-hosted compiler'],
  author: {
    '@type': 'Person',
    name: 'Demetrios Chiuratto Agourakis',
    url: 'https://www.souniolang.org/about/',
    sameAs: ['https://orcid.org/0009-0001-8671-8878'],
  },
  about: {
    '@type': 'SoftwareApplication',
    name: 'Sounio',
  },
};

/** Approved copy for the English effect-system hub. The search title stays in ENGLISH_SEARCH. */
export const EFFECTS_PAGE = {
  h1: 'Effect system: side effects in the function signature',
  lede: "An effect system tracks what a function may do besides returning a value (I/O, mutation, panics, division) in its type. In Sounio, effects are written in the signature with `with`, and a caller that does not declare a callee's effect is rejected at compile time.",
  author: {
    '@type': 'Person',
    name: 'Demetrios Chiuratto Agourakis',
    url: 'https://www.souniolang.org/about/',
    sameAs: ['https://orcid.org/0009-0001-8671-8878'],
  },
  about: [
    {
      '@type': 'Thing',
      name: 'Effect system',
      sameAs: 'https://en.wikipedia.org/wiki/Effect_system',
    },
  ],
  isPartOf: {
    '@type': 'WebSite',
    name: 'Sounio',
    url: 'https://www.souniolang.org/',
  },
  proficiencyLevel: 'Intermediate',
  dependencies: [],
  dateModified: '2026-10-29',
};

export function englishSearchFor(barePath, locale) {
  if (locale !== 'en') return undefined;
  return ENGLISH_SEARCH[barePath];
}

const COMPILER_VERSION = /v0\.80|v0\.99|v0\.100|v1\.0/;

for (const [page, copy] of Object.entries(ENGLISH_SEARCH)) {
  if (copy.title.length < 50 || copy.title.length > 60) {
    throw new Error(`${page} title length ${copy.title.length} is outside 50–60`);
  }
  if (copy.description.length < 120 || copy.description.length > 155) {
    throw new Error(`${page} description length ${copy.description.length} is outside 120–155`);
  }
  if (COMPILER_VERSION.test(copy.title) || COMPILER_VERSION.test(copy.description)) {
    throw new Error(`${page} title or description names a compiler version`);
  }
}
