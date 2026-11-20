export const CODE_REPOSITORY = 'https://github.com/Sounio-lang/sounio';

/** Former public name of the language. Not a second language. */
export const FORMER_LANGUAGE_NAME = 'Demetrios';

/** Record URL for sameAs. This repository does not otherwise mention 18040325. */
export const FORMER_RELEASE_URL = 'https://zenodo.org/records/18040325';

export function isoDateOrNothing(value) {
  if (typeof value !== 'string') return undefined;
  const match = /^(\d{4}-\d{2}-\d{2})$/.exec(value.trim());
  if (!match) return undefined;
  const date = match[1];
  const parsed = new Date(`${date}T00:00:00Z`);
  if (Number.isNaN(parsed.getTime())) return undefined;
  if (parsed.toISOString().slice(0, 10) !== date) return undefined;
  return date;
}

export function softwareApplication({ description, siteOrigin }) {
  const origin = String(siteOrigin || 'https://www.souniolang.org').replace(/\/$/, '');
  return {
    '@context': 'https://schema.org',
    '@type': 'SoftwareApplication',
    name: 'Sounio',
    alternateName: FORMER_LANGUAGE_NAME,
    sameAs: [FORMER_RELEASE_URL],
    applicationCategory: 'DeveloperApplication',
    operatingSystem: 'Linux x86_64 (committed prebuilt); macOS arm64 (source-built)',
    programmingLanguage: 'Sounio',
    license: 'https://opensource.org/licenses/Apache-2.0',
    downloadUrl: CODE_REPOSITORY,
    codeRepository: CODE_REPOSITORY,
    author: {
      '@type': 'Person',
      name: 'Demetrios Chiuratto Agourakis',
      url: origin,
    },
    description,
    url: origin,
    offers: {
      '@type': 'Offer',
      price: '0',
      priceCurrency: 'USD',
    },
  };
}

export function softwareSourceCode() {
  return {
    '@context': 'https://schema.org',
    '@type': 'SoftwareSourceCode',
    name: 'Sounio',
    alternateName: FORMER_LANGUAGE_NAME,
    sameAs: [FORMER_RELEASE_URL],
    codeRepository: CODE_REPOSITORY,
    programmingLanguage: 'Sounio',
    license: 'https://opensource.org/licenses/Apache-2.0',
    url: CODE_REPOSITORY,
  };
}

/** Stable id for the language page. Not the About person ids, and not a software node. */
export const COMPUTER_LANGUAGE_ID = 'https://www.souniolang.org/language/#computer-language';

export function computerLanguage({ description } = {}) {
  const node = {
    '@context': 'https://schema.org',
    '@type': 'ComputerLanguage',
    '@id': COMPUTER_LANGUAGE_ID,
    name: 'Sounio',
    alternateName: FORMER_LANGUAGE_NAME,
    sameAs: [FORMER_RELEASE_URL],
    url: 'https://www.souniolang.org/language/',
  };
  if (typeof description === 'string' && description.length > 0) node.description = description;
  return node;
}

export function techArticle({
  headline,
  description,
  url,
  inLanguage,
  dateModified,
  about,
  author,
  isPartOf,
  proficiencyLevel,
  dependencies,
  citation,
}) {
  const article = {
    '@context': 'https://schema.org',
    '@type': 'TechArticle',
    headline,
    description,
    inLanguage,
    url,
  };
  const modified = isoDateOrNothing(dateModified);
  if (modified) article.dateModified = modified;
  if (about) article.about = about;
  // Science and tutorial pages do not pass an author. Learn pages that do
  // not name one use the same person. A supplied author wins.
  article.author = author ?? SITE_AUTHOR;
  if (isPartOf) article.isPartOf = isPartOf;
  if (typeof proficiencyLevel === 'string' && proficiencyLevel.length > 0) {
    article.proficiencyLevel = proficiencyLevel;
  }
  if (dependencies !== undefined) article.dependencies = dependencies;
  if (citation) article.citation = citation;
  return article;
}

/** Article author already used on learn pages. No honorific, no credential. */
export const SITE_AUTHOR = {
  '@type': 'Person',
  name: 'Demetrios Chiuratto Agourakis',
  url: 'https://www.souniolang.org/about/',
  sameAs: ['https://orcid.org/0009-0001-8671-8878'],
};

/**
 * JSON-LD author for a visible byline.
 * A Person is emitted only for the site author, spelled exactly.
 * The team byline, and any other byline, is an Organization whose name
 * is the byline text, so the structured name stays the visible one.
 */
export function authorFromByline(byline) {
  const name = String(byline ?? '').replace(/\s+/g, ' ').trim();
  if (name === SITE_AUTHOR.name) return SITE_AUTHOR;
  return {
    '@type': 'Organization',
    name,
  };
}

/**
 * Schema kind for pages that previously emitted no JSON-LD.
 * `article` is TechArticle, `blog` is BlogPosting, `collection` is a hub,
 * `crumbs` is BreadcrumbList only.
 */
export function pageSchemaKind(bare) {
  if (
    bare === '/science/'
    || bare === '/tutorials/'
    || bare === '/insights/'
    || bare === '/releases/'
    || bare === '/packages/'
    || bare === '/showcases/'
  ) {
    return 'collection';
  }
  if (/^\/tutorials\/[^/]+\/$/.test(bare)) return 'collection';
  if (bare === '/install/') return 'article';
  if (/^\/science\/[^/]+\/$/.test(bare)) return 'article';
  if (/^\/tutorials\/[^/]+\/[^/]+\/$/.test(bare)) return 'article';
  if (/^\/insights\/[^/]+\/$/.test(bare)) return 'blog';
  if (
    bare === '/credits/'
    || bare === '/dissertation/'
    || bare === '/scientist/'
    || /^\/packages\/[^/]+\/$/.test(bare)
  ) {
    return 'crumbs';
  }
  return null;
}

export function collectionPage({ name, description, url, inLanguage }) {
  return {
    '@context': 'https://schema.org',
    '@type': 'CollectionPage',
    name,
    description,
    url,
    inLanguage,
  };
}

export function blogPosting({
  headline,
  description,
  url,
  inLanguage,
  datePublished,
  dateModified,
  author,
}) {
  const post = {
    '@context': 'https://schema.org',
    '@type': 'BlogPosting',
    headline,
    description,
    inLanguage,
    url,
  };
  if (author) post.author = author;
  const published = isoDateOrNothing(datePublished);
  if (published) post.datePublished = published;
  const modified = isoDateOrNothing(dateModified);
  if (modified) post.dateModified = modified;
  return post;
}

export function breadcrumbList(crumbs, absolutize) {
  return {
    '@context': 'https://schema.org',
    '@type': 'BreadcrumbList',
    itemListElement: crumbs.map((crumb, index) => ({
      '@type': 'ListItem',
      position: index + 1,
      name: crumb.name,
      item: absolutize(crumb.href),
    })),
  };
}

export function buildPageSchemas({
  bare,
  title,
  description,
  canonicalUrl,
  inLanguage,
  siteOrigin,
  breadcrumbs,
  dateModified,
  extra,
  absolutize,
  headline,
  about,
  author,
  isPartOf,
  proficiencyLevel,
  dependencies,
  citation,
  omitSoftwareApplication,
  datePublished,
  pageTitle,
}) {
  const schemas = [];
  if (bare === '/' && !omitSoftwareApplication) schemas.push(softwareApplication({ description, siteOrigin }));
  if (bare === '/' || bare === '/learn/getting-started/') schemas.push(softwareSourceCode());
  if (bare.startsWith('/learn/')) {
    schemas.push(techArticle({
      headline: headline || title,
      description,
      url: canonicalUrl,
      inLanguage,
      dateModified,
      about,
      author,
      isPartOf,
      proficiencyLevel,
      dependencies,
      citation,
    }));
  } else {
    const kind = pageSchemaKind(bare);
    const articleHeadline = headline || pageTitle || title;
    if (kind === 'article') {
      schemas.push(techArticle({
        headline: articleHeadline,
        description,
        url: canonicalUrl,
        inLanguage,
        author,
      }));
    } else if (kind === 'blog') {
      schemas.push(blogPosting({
        headline: articleHeadline,
        description,
        url: canonicalUrl,
        inLanguage,
        datePublished,
        dateModified: dateModified || datePublished,
        author,
      }));
    } else if (kind === 'collection') {
      schemas.push(collectionPage({
        name: articleHeadline,
        description,
        url: canonicalUrl,
        inLanguage,
      }));
    }
  }
  if (Array.isArray(breadcrumbs) && breadcrumbs.length > 0) {
    schemas.push(breadcrumbList(breadcrumbs, absolutize));
  }
  for (const block of extra ?? []) schemas.push(block);
  return schemas;
}

export function jsonForScript(value) {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}
