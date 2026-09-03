/**
 * Internationalization utilities for Sounio website
 */

export const locales = ['en', 'pt', 'el', 'zh', 'ja', 'es', 'zh-hk'] as const;
export type Locale = (typeof locales)[number];

export const localeNames: Record<Locale, string> = {
  en: 'English',
  pt: 'Português',
  el: 'Ελληνικά',
  zh: '中文',
  ja: '日本語',
  es: 'Español',
  'zh-hk': '香港粵語',
};

export const defaultLocale: Locale = 'en';
export const fullyLocalizedLocales = new Set<Locale>(['en']);

// Translation type (inferred from en.json structure)
export type Translations = typeof import('../i18n/en.json');

// Cache for loaded translations
const translationCache = new Map<Locale, Translations>();

/**
 * Load translations for a locale
 */
export async function loadTranslations(locale: Locale): Promise<Translations> {
  if (translationCache.has(locale)) {
    return translationCache.get(locale)!;
  }

  try {
    const translations = await import(`../i18n/${locale}.json`);
    translationCache.set(locale, translations.default || translations);
    return translations.default || translations;
  } catch {
    // Fallback to English if locale not found
    console.warn(`Translations for ${locale} not found, falling back to English`);
    return loadTranslations('en');
  }
}

/**
 * Get the current locale from URL
 */
export function getLocaleFromUrl(url: URL): Locale {
  const [, lang] = url.pathname.split('/');
  if (locales.includes(lang as Locale)) {
    return lang as Locale;
  }
  return defaultLocale;
}

/**
 * Get the path for a different locale
 */
export function getLocalizedPath(path: string, locale: Locale): string {
  // Remove any existing locale prefix
  const cleanPath = path.replace(/^\/(en|pt|el|zh-hk|zh|ja|es)/, '');

  // Don't prefix default locale
  if (locale === defaultLocale) {
    return cleanPath || '/';
  }

  return `/${locale}${cleanPath}`;
}

/**
 * Create a translation function for a specific locale
 */
export function createTranslator(translations: Translations) {
  return function t(key: keyof Translations): string {
    return (translations[key] as string) || key;
  };
}

/**
 * Whether a locale has fully rewritten V2 content.
 */
export function isLocaleFullyLocalized(locale: Locale): boolean {
  return fullyLocalizedLocales.has(locale);
}

/**
 * Strip a leading `/{locale}` prefix from a pathname.
 * Example: `/pt/learn/foo` becomes `/learn/foo`, and `/pt` becomes `/`.
 */
export function getPathWithoutLocalePrefix(pathname: string): string {
  for (const loc of locales) {
    if (loc === defaultLocale) continue;
    const prefix = `/${loc}`;
    if (pathname === prefix) {
      return '/';
    }
    if (pathname.startsWith(`${prefix}/`)) {
      return pathname.slice(prefix.length);
    }
  }
  return pathname;
}

/**
 * BCP 47 tag for Open Graph (aligned with @astrojs/sitemap i18n mapping).
 * The short URL prefix `pt` maps to Brazilian Portuguese because astro.config.mjs
 * already publishes `pt-BR` in the sitemap locale table.
 */
export function ogLocaleTag(locale: Locale): string {
  const map: Record<Locale, string> = {
    en: 'en_US',
    pt: 'pt_BR',
    el: 'el',
    zh: 'zh_CN',
    ja: 'ja',
    es: 'es',
    'zh-hk': 'zh_HK',
  };
  return map[locale];
}

/**
 * Whether to show the "translation may be incomplete / English is authoritative" banner.
 * Policy: only on long-form documentation surfaces (`/learn`, `/tutorials`) so marketing
 * and shell pages stay clean for `/pt`, `/ja`, etc. See website/docs/I18N_POLICY.md.
 */
export function shouldShowLocaleTranslationNotice(pathname: string): boolean {
  const p = getPathWithoutLocalePrefix(pathname);
  if (p === '/' || p === '') return false;
  if (p.startsWith('/learn')) return true;
  if (p.startsWith('/tutorials')) return true;
  return false;
}
