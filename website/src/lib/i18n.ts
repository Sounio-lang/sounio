/**
 * Internationalization utilities for Sounio website
 */

export const locales = ['en', 'pt', 'el', 'zh', 'ja', 'es'] as const;
export type Locale = (typeof locales)[number];

export const localeNames: Record<Locale, string> = {
  en: 'English',
  pt: 'Português',
  el: 'Ελληνικά',
  zh: '中文',
  ja: '日本語',
  es: 'Español',
};

export const defaultLocale: Locale = 'en';

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
  const cleanPath = path.replace(/^\/(en|pt|el|zh|ja|es)/, '');

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
