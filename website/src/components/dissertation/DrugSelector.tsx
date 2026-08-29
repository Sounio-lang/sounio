// Drug A/B selector chip row (G-ε-1). Top-of-canvas overlay.
// Press `D` to cycle through drugs. Each chip shows the drug's clinical
// label and (when active) a brief release-source descriptor.

import { useEffect } from 'react';
import type { DrugId } from '../../hooks/usePBPK';

const DRUGS: { id: DrugId; label: string; release: string; sourceColor: string }[] = [
  {
    id: 'rapamycin',
    label: 'Rapamycin',
    release: 'Cypher stent (Higuchi)',
    sourceColor: '#f59e0b',
  },
  {
    id: 'semaglutide',
    label: 'Semaglutide',
    release: 'SC depot (k_a = ln 2 / 60 h)',
    sourceColor: '#06b6d4',
  },
];

export function DrugSelector({
  drug,
  onChange,
}: {
  drug: DrugId;
  onChange: (d: DrugId) => void;
}) {
  // Keyboard: `D` cycles drugs.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'd' && e.key !== 'D') return;
      const idx = DRUGS.findIndex((d) => d.id === drug);
      const next = DRUGS[(idx + 1) % DRUGS.length];
      onChange(next.id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [drug, onChange]);

  const active = DRUGS.find((d) => d.id === drug) ?? DRUGS[0];

  return (
    <div
      role="radiogroup"
      aria-label="Selected drug"
      style={{
        position: 'absolute',
        top: 12,
        left: '50%',
        transform: 'translateX(-50%)',
        display: 'flex',
        gap: 8,
        zIndex: 10,
        background: 'rgba(15, 23, 42, 0.55)',
        padding: '6px 10px',
        borderRadius: 999,
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        boxShadow: '0 4px 16px rgba(0,0,0,0.25)',
      }}
    >
      {DRUGS.map((d) => {
        const isActive = d.id === drug;
        return (
          <button
            key={d.id}
            type="button"
            role="radio"
            aria-checked={isActive}
            onClick={() => onChange(d.id)}
            style={{
              border: 'none',
              borderRadius: 999,
              padding: '4px 14px',
              fontSize: 13,
              fontWeight: isActive ? 600 : 400,
              cursor: 'pointer',
              color: isActive ? '#0f172a' : '#e2e8f0',
              background: isActive ? d.sourceColor : 'transparent',
              transition: 'all 0.15s ease',
            }}
          >
            {d.label}
          </button>
        );
      })}
      <div
        aria-hidden
        className="hidden sm:block"
        style={{
          fontSize: 11,
          color: '#94a3b8',
          alignSelf: 'center',
          marginLeft: 6,
          maxWidth: 200,
          whiteSpace: 'nowrap',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
        }}
        title={active.release}
      >
        {active.release}  ·  press <kbd style={{ fontFamily: 'inherit' }}>D</kbd> to switch
      </div>
    </div>
  );
}
