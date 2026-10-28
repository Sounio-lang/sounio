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
    title: 'Sounio self-hosted compiler, from source to x86-64',
    description:
      'Architecture of the self-hosted Sounio compiler: parse, type-check, lower to IR, and emit a native Linux x86-64 ELF binary.',
  },
  '/about/roadmap/': {
    title: 'Sounio roadmap: language, runtime, and what is open',
    description:
      'The Sounio roadmap tracks language, runtime, and ecosystem milestones, including runtime and backend work that is still in progress.',
  },
};

export function englishSearchFor(barePath, locale) {
  if (locale !== 'en') return undefined;
  return ENGLISH_SEARCH[barePath];
}

for (const [page, copy] of Object.entries(ENGLISH_SEARCH)) {
  if (copy.title.length < 50 || copy.title.length > 60) {
    throw new Error(`${page} title length ${copy.title.length} is outside 50–60`);
  }
  if (copy.description.length < 120 || copy.description.length > 155) {
    throw new Error(`${page} description length ${copy.description.length} is outside 120–155`);
  }
}
