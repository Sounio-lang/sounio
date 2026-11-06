export interface DocsNavLink {
  title: string;
  href: string;
}

export interface DocsNavSection {
  titleKey: string;
  heroSlug: string;
  slugs: string[];
  icon: string;
  /** English-only shell pages. Not part of the doc linear order. */
  extra?: DocsNavLink[];
}

/** Insert shell links after the uncertainty guide, before later epistemic pages. */
export function withEpistemicExtras(items: DocsNavLink[], extra: DocsNavLink[] | undefined): DocsNavLink[] {
  if (!extra || extra.length === 0) return items;
  const index = items.findIndex((item) => item.href.endsWith('/epistemic/uncertainty/'));
  if (index === -1) return [...items, ...extra];
  return [...items.slice(0, index + 1), ...extra, ...items.slice(index + 1)];
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
    slugs: ['language', 'language/variables', 'language/functions', 'language/control-flow', 'language/units-and-uncertainty'],
    icon: 'language',
  },
  {
    titleKey: 'docs_sidebar_epistemic',
    heroSlug: 'epistemic',
    slugs: ['epistemic', 'epistemic/knowledge', 'epistemic/uncertainty', 'epistemic/provenance'],
    icon: 'epistemic',
    extra: [
      {
        title: 'Uncertainty tools compared',
        href: '/learn/epistemic/uncertainty-tools-compared/',
      },
    ],
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
