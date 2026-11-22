import { useState, useEffect } from 'react';

type Theme = 'light' | 'dark' | 'system';

interface Props {
  strings?: { toggle: string; light: string; dark: string; system: string };
}

export default function ThemeToggle({ strings = { toggle: 'Toggle theme', light: 'Light', dark: 'Dark', system: 'System' } }: Props = {}) {
  const [theme, setTheme] = useState<Theme>('dark');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    const synchronize = (event: Event) => {
      const value = (event as CustomEvent<Theme>).detail;
      if (value === 'dark' || value === 'light' || value === 'system') setTheme(value);
    };
    window.addEventListener('sounio-theme-change', synchronize);
    return () => window.removeEventListener('sounio-theme-change', synchronize);
  }, []);

  useEffect(() => {
    setMounted(true);

    const stored = localStorage.getItem('theme') as Theme | null;
    if (stored === 'light' || stored === 'dark' || stored === 'system') {
      setTheme(stored);
      return;
    }

    // Dark-first default when no user preference exists.
    setTheme('dark');
    localStorage.setItem('theme', 'dark');
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const root = document.documentElement;
    const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const resolvedTheme = theme === 'system' ? (systemDark ? 'dark' : 'light') : theme;

    root.classList.remove('dark', 'light');
    root.classList.add(resolvedTheme);
    localStorage.setItem('theme', theme);
  }, [theme, mounted]);

  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleChange = () => {
      if (theme !== 'system') return;
      const root = document.documentElement;
      root.classList.remove('dark', 'light');
      root.classList.add(mediaQuery.matches ? 'dark' : 'light');
    };

    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, [theme]);

  const cycleTheme = () => {
    const themes: Theme[] = ['dark', 'light', 'system'];
    const currentIndex = themes.indexOf(theme);
    const nextTheme = themes[(currentIndex + 1) % themes.length];
    setTheme(nextTheme);
    window.dispatchEvent(new CustomEvent('sounio-theme-change', { detail: nextTheme }));
  };

  if (!mounted) {
    return (
      <button
        type="button"
        className="theme-toggle-btn"
        aria-label={strings.toggle}
        disabled
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
          />
        </svg>
      </button>
    );
  }

  return (
    <button
      type="button"
      onClick={cycleTheme}
      className="theme-toggle-btn"
      aria-label={`${strings.toggle}: ${strings[theme]}`}
      title={strings[theme]}
    >
      {theme === 'light' && (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
          />
        </svg>
      )}
      {theme === 'dark' && (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
          />
        </svg>
      )}
      {theme === 'system' && (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
          />
        </svg>
      )}
    </button>
  );
}
