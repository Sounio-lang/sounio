export const CODE_REPOSITORY = 'https://github.com/Sounio-lang/sounio';

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
    codeRepository: CODE_REPOSITORY,
    programmingLanguage: 'Sounio',
    license: 'https://opensource.org/licenses/Apache-2.0',
    url: CODE_REPOSITORY,
  };
}

export function techArticle({ headline, description, url, inLanguage, dateModified, about, author }) {
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
  if (author) article.author = author;
  return article;
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
}) {
  const schemas = [];
  if (bare === '/') schemas.push(softwareApplication({ description, siteOrigin }));
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
    }));
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
