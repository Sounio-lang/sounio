export const AUDIENCES = ['scientist', 'technical'] as const;
export type Audience = (typeof AUDIENCES)[number];
export const DEFAULT_AUDIENCE: Audience = 'scientist';
export const STORAGE_KEY = 'sounio-audience';

export function getStoredAudience(): Audience | null {
  if (typeof localStorage === 'undefined') return null;
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === 'scientist' || stored === 'technical') return stored;
  return null;
}

export function setStoredAudience(audience: Audience): void {
  localStorage.setItem(STORAGE_KEY, audience);
  document.documentElement.setAttribute('data-audience', audience);
  window.dispatchEvent(new CustomEvent('audience-change', { detail: audience }));
}

export function getEffectiveAudience(): Audience {
  return getStoredAudience() ?? DEFAULT_AUDIENCE;
}
