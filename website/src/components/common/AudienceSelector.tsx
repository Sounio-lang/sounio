import { useState, useEffect } from 'react';
import { useAudience } from '../../lib/useAudience';
import type { Audience } from '../../lib/audience';

interface AudienceStrings {
  scientist: string;
  technical: string;
  choose: string;
}

const defaultStrings: AudienceStrings = {
  scientist: 'Scientist',
  technical: 'Technical',
  choose: 'Choose your view',
};

interface Props {
  strings?: AudienceStrings;
}

export default function AudienceSelector({ strings = defaultStrings }: Props) {
  const { audience, choose, hasExplicitChoice } = useAudience();
  const [showTooltip, setShowTooltip] = useState(false);
  const [mounted, setMounted] = useState(false);

  const labels: Record<Audience, string> = {
    scientist: strings.scientist,
    technical: strings.technical,
  };

  useEffect(() => {
    setMounted(true);
    if (!hasExplicitChoice) {
      const timer = setTimeout(() => setShowTooltip(true), 2000);
      const hide = setTimeout(() => setShowTooltip(false), 8000);
      return () => { clearTimeout(timer); clearTimeout(hide); };
    }
  }, [hasExplicitChoice]);

  const toggle = () => {
    const next: Audience = audience === 'scientist' ? 'technical' : 'scientist';
    choose(next);
    setShowTooltip(false);
  };

  if (!mounted) {
    return (
      <button type="button" className="audience-toggle-btn" aria-label="Toggle audience view">
        <span className="audience-label">{strings.scientist}</span>
      </button>
    );
  }

  return (
    <div className="relative">
      <button
        type="button"
        onClick={toggle}
        className="audience-toggle-btn"
        aria-label={`Viewing as ${labels[audience]}. Click to switch.`}
        title={`View: ${labels[audience]}`}
      >
        <svg className="w-3.5 h-3.5 opacity-60" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          {audience === 'scientist' ? (
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23.693L5 14.5m14.8.8l1.402 1.402c1.232 1.232.65 3.318-1.067 3.611A48.309 48.309 0 0112 21c-2.773 0-5.491-.235-8.135-.687-1.718-.293-2.3-2.379-1.067-3.61L5 14.5" />
          ) : (
            <path strokeLinecap="round" strokeLinejoin="round" d="M17.25 6.75L22.5 12l-5.25 5.25m-10.5 0L1.5 12l5.25-5.25m7.5-3l-4.5 16.5" />
          )}
        </svg>
        <span className="audience-label">{labels[audience]}</span>
      </button>

      {showTooltip && !hasExplicitChoice && (
        <div className="absolute top-full right-0 mt-2 px-3 py-1.5 rounded-lg bg-[var(--color-accent-gold)] text-[#10233e] text-xs font-semibold whitespace-nowrap shadow-lg z-50 animate-[fadeIn_0.3s_ease]">
          {strings.choose}
          <div className="absolute -top-1 right-4 w-2 h-2 bg-[var(--color-accent-gold)] rotate-45" />
        </div>
      )}
    </div>
  );
}
