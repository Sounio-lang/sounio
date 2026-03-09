export interface DocsNavSection {
  titleKey: string;
  heroSlug: string;
  slugs: string[];
  icon: string;
}

export const docsNavSections: DocsNavSection[] = [
  {
    titleKey: 'docs_sidebar_getting_started',
    heroSlug: 'getting-started',
    slugs: ['getting-started', 'getting-started/hello-world', 'getting-started/first-program'],
    icon: 'getting-started',
  },
  {
    titleKey: 'docs_sidebar_language',
    heroSlug: 'language',
    slugs: ['language', 'language/variables', 'language/functions', 'language/control-flow'],
    icon: 'language',
  },
  {
    titleKey: 'docs_sidebar_epistemic',
    heroSlug: 'epistemic',
    slugs: ['epistemic', 'epistemic/knowledge', 'epistemic/uncertainty', 'epistemic/provenance'],
    icon: 'epistemic',
  },
  {
    titleKey: 'docs_sidebar_effects',
    heroSlug: 'effects',
    slugs: ['effects', 'effects/io', 'effects/custom'],
    icon: 'effects',
  },
  {
    titleKey: 'docs_sidebar_runtime',
    heroSlug: 'feature-status',
    slugs: ['feature-status', 'gpu', 'units', 'stdlib', 'tooling', 'tooling/lsp'],
    icon: 'runtime',
  },
  {
    titleKey: 'docs_sidebar_examples',
    heroSlug: 'examples',
    slugs: ['examples', 'vancomycin-uncertainty'],
    icon: 'examples',
  },
  {
    titleKey: 'docs_sidebar_compiler',
    heroSlug: 'compiler',
    slugs: [
      'compiler',
      'compiler/lexer-parser',
      'compiler/type-system',
      'compiler/effect-system',
      'compiler/codegen',
      'compiler/scientific-features',
    ],
    icon: 'compiler',
  },
  {
    titleKey: 'docs_sidebar_spec',
    heroSlug: 'spec',
    slugs: ['spec', 'spec/language-spec'],
    icon: 'spec',
  },
];

export const docsLinearOrder = docsNavSections.flatMap((section) => section.slugs);
