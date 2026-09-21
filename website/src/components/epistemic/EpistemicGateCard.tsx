import type { CSSProperties } from 'react';
import './EpistemicGateCard.css';

export const GATE_LEVELS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'] as const;
export type GateLevel = (typeof GATE_LEVELS)[number];
export type GateVerdict = 'TRUSTWORTHY' | 'UNBOUNDED' | 'REFUSED';

export const GATE_LABEL: Record<GateLevel, string> = {
  E0: 'Syntax',
  E1: 'HIR Type',
  E2: 'Effect Lattice',
  E3: 'GUM Variance Bounds',
  E4: 'Lean 4 Formal Proof',
  E5: 'Closed-form Theorem',
};

export type EpistemicGateCardProps = {
  claim: string;
  ceiling: GateLevel;
  verdict: GateVerdict;
  reason: string;
  sha: string;
  href: string;
  hrefLabel: string;
  className?: string;
};

const TOKEN: Record<
  GateVerdict,
  { ink: string; surface: string; border: string; text: string; glyph: string }
> = {
  TRUSTWORTHY: {
    ink: 'var(--color-epistemic-verified)',
    surface: 'var(--color-epistemic-verified-surface)',
    border: 'var(--color-epistemic-verified-border)',
    text: 'var(--color-epistemic-verified-text)',
    glyph: '◈',
  },
  UNBOUNDED: {
    ink: 'var(--color-epistemic-uncertain)',
    surface: 'var(--color-epistemic-uncertain-surface)',
    border: 'var(--color-epistemic-uncertain-border)',
    text: 'var(--color-epistemic-uncertain-text)',
    glyph: '△',
  },
  REFUSED: {
    ink: 'var(--color-epistemic-refused)',
    surface: 'var(--color-epistemic-refused-surface)',
    border: 'var(--color-epistemic-refused-border)',
    text: 'var(--color-epistemic-refused-text)',
    glyph: '⊘',
  },
};

function stepState(level: GateLevel, ceiling: GateLevel): 'reached' | 'ceiling' | 'unreached' {
  const i = GATE_LEVELS.indexOf(level);
  const c = GATE_LEVELS.indexOf(ceiling);
  if (i < c) return 'reached';
  if (i === c) return 'ceiling';
  return 'unreached';
}

export function EpistemicGateCard({
  claim,
  ceiling,
  verdict,
  reason,
  sha,
  href,
  hrefLabel,
  className,
}: EpistemicGateCardProps) {
  const token = TOKEN[verdict];
  const classes = ['egc', className].filter(Boolean).join(' ');

  return (
    <article
      className={classes}
      data-verdict={verdict}
      data-ceiling={ceiling}
      style={
        {
          '--egc-ink': token.ink,
          '--egc-surface': token.surface,
          '--egc-border': token.border,
          '--egc-text': token.text,
        } as CSSProperties
      }
    >
      <header className="egc-header">
        <p className="egc-claim">{claim}</p>
        <p className="egc-verdict">
          <span className="egc-glyph" aria-hidden="true">
            {token.glyph}
          </span>
          {verdict}
        </p>
      </header>

      <ol className="egc-ladder" aria-label={`Gate ladder, ceiling ${ceiling}`}>
        {GATE_LEVELS.map((level) => {
          const state = stepState(level, ceiling);
          return (
            <li
              key={level}
              className="egc-step"
              data-state={state}
              data-verdict={state === 'ceiling' ? verdict : undefined}
            >
              <span className="egc-step-id">{level}</span>
              <span className="egc-step-label">{GATE_LABEL[level]}</span>
            </li>
          );
        })}
      </ol>

      <p className="egc-reason">{reason}</p>

      <footer className="egc-footer">
        <code className="egc-sha">{sha}</code>
        <a className="egc-link" href={href} target="_blank" rel="noopener noreferrer">
          {hrefLabel}
        </a>
      </footer>
    </article>
  );
}
