import { useState, useEffect } from 'react';
import { getEffectiveAudience, setStoredAudience, type Audience, STORAGE_KEY } from './audience';

export function useAudience() {
  const [audience, setAudience] = useState<Audience>(getEffectiveAudience());
  const [hasExplicitChoice, setHasExplicitChoice] = useState(false);

  useEffect(() => {
    setHasExplicitChoice(localStorage.getItem(STORAGE_KEY) !== null);

    const handler = (e: Event) => {
      const detail = (e as CustomEvent<Audience>).detail;
      setAudience(detail);
      setHasExplicitChoice(true);
    };
    window.addEventListener('audience-change', handler);
    return () => window.removeEventListener('audience-change', handler);
  }, []);

  const choose = (a: Audience) => {
    setStoredAudience(a);
    setAudience(a);
    setHasExplicitChoice(true);
  };

  return { audience, choose, hasExplicitChoice };
}
