import { useId } from 'react';
import './EpistemicGateCard.css';

export const GATE_LEVELS = ['E0', 'E1', 'E2', 'E3', 'E4', 'E5'] as const;
export type GateLevel = (typeof GATE_LEVELS)[number];
export type GateVerdict = 'TRUSTWORTHY' | 'UNBOUNDED' | 'REFUSED';
export type EpistemicGateCardSize = 'default' | 'compact';

export const GATE_LABEL: Record<GateLevel, string> = {
  E0: 'Syntax',
  E1: 'HIR Type',
  E2: 'Effect Lattice',
  E3: 'GUM Variance Bounds',
  E4: 'Lean 4 Formal Proof',
  E5: 'Closed-form Theorem',
};

export const VERDICT_LABEL: Record<GateVerdict, string> = {
  TRUSTWORTHY: 'Trustworthy',
  UNBOUNDED: 'Unbounded',
  REFUSED: 'Refused',
};

export const VERDICT_GLYPH: Record<GateVerdict, string> = {
  TRUSTWORTHY: '◈',
  UNBOUNDED: '△',
  REFUSED: '⊘',
};

export type EpistemicGateCardProps = {
  claim: string;
  ceiling: GateLevel;
  verdict: GateVerdict;
  reason: string;
  sha: string;
  href: string;
  hrefLabel: string;
  size?: EpistemicGateCardSize;
  className?: string;
};

type StepState = 'reached' | 'ceiling' | 'unreached';

function stepState(level: GateLevel, ceiling: GateLevel): StepState {
  const i = GATE_LEVELS.indexOf(level);
  const c = GATE_LEVELS.indexOf(ceiling);
  if (i < c) return 'reached';
  if (i === c) return 'ceiling';
  return 'unreached';
}

function stepAriaLabel(
  level: GateLevel,
  state: StepState,
  verdict: GateVerdict,
): string {
  const label = GATE_LABEL[level];
  if (state === 'reached') return `${level} ${label}, passed`;
  if (state === 'ceiling') {
    return `${level} ${label}, ceiling gate, ${VERDICT_LABEL[verdict].toLowerCase()}`;
  }
  return `${level} ${label}, not reached`;
}

export function EpistemicGateCard({
  claim,
  ceiling,
  verdict,
  reason,
  sha,
  href,
  hrefLabel,
  size = 'default',
  className,
}: EpistemicGateCardProps) {
  const uid = useId();
  const claimId = `${uid}-claim`;
  const summaryId = `${uid}-summary`;
  const classes = ['egc', className].filter(Boolean).join(' ');

  return (
    <article
      className={classes}
      data-verdict={verdict}
      data-ceiling={ceiling}
      data-size={size}
      aria-labelledby={`${summaryId} ${claimId}`}
    >
      <p id={summaryId} className="egc-sr-only">
        {VERDICT_LABEL[verdict]} at gate {ceiling} ({GATE_LABEL[ceiling]}). {claim}
      </p>

      <header className="egc-header">
        <p id={claimId} className="egc-claim">
          {claim}
        </p>
        <p className="egc-verdict" role="status">
          <span className="egc-glyph" aria-hidden="true">
            {VERDICT_GLYPH[verdict]}
          </span>
          {VERDICT_LABEL[verdict]}
        </p>
      </header>

      <ol
        className="egc-ladder"
        aria-label={`Evidence ladder, ceiling ${ceiling} (${GATE_LABEL[ceiling]})`}
      >
        {GATE_LEVELS.map((level) => {
          const state = stepState(level, ceiling);
          return (
            <li
              key={level}
              className="egc-step"
              data-state={state}
              data-verdict={state === 'ceiling' ? verdict : undefined}
              aria-label={stepAriaLabel(level, state, verdict)}
              aria-current={state === 'ceiling' ? 'step' : undefined}
            >
              <span className="egc-step-id">{level}</span>
              <span className="egc-step-label">{GATE_LABEL[level]}</span>
            </li>
          );
        })}
      </ol>

      <p className="egc-reason">{reason}</p>

      <footer className="egc-footer">
        <code className="egc-sha" aria-label={`Commit ${sha}`}>
          {sha}
        </code>
        <a
          className="egc-link"
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          aria-label={`${hrefLabel} (opens in new tab)`}
        >
          {hrefLabel}
        </a>
      </footer>
    </article>
  );
}
