import type { CSSProperties } from 'react';
import './EffectLatticeBadge.css';

export const ALGEBRAIC_EFFECTS = [
  'IO',
  'Mut',
  'Div',
  'Panic',
  'Alloc',
  'Async',
  'GPU',
  'Prob',
  'Observe',
] as const;

export type AlgebraicEffect = (typeof ALGEBRAIC_EFFECTS)[number];
export type LatticeMark = AlgebraicEffect | 'Linear';
export type PurityLevel = 'Declared' | 'Inferred' | 'Handled';
export type EffectLatticeSize = 'sm' | 'md';

export type EffectLatticeBadgeProps = {
  effect: LatticeMark;
  purityLevel?: PurityLevel;
  size?: EffectLatticeSize;
  className?: string;
};

const TOKEN: Record<LatticeMark, string> = {
  IO: 'var(--color-effect-io)',
  Mut: 'var(--color-effect-mut)',
  Div: 'var(--color-effect-div)',
  Panic: 'var(--color-effect-panic)',
  Alloc: 'var(--color-effect-alloc)',
  Async: 'var(--color-effect-async)',
  GPU: 'var(--color-effect-gpu)',
  Prob: 'var(--color-effect-prob)',
  Observe: 'var(--color-effect-observe)',
  Linear: 'var(--color-type-linear)',
};

const GLYPH: Record<LatticeMark, string> = {
  IO: '⌁',
  Mut: '&!',
  Div: '÷',
  Panic: '⊥',
  Alloc: '⊞',
  Async: '∿',
  GPU: '∇',
  Prob: '∼',
  Observe: '◐',
  Linear: '1',
};

export function EffectLatticeBadge({
  effect,
  purityLevel = 'Declared',
  size = 'md',
  className,
}: EffectLatticeBadgeProps) {
  const ink = TOKEN[effect];
  const kind = effect === 'Linear' ? 'linear' : 'effect';
  const classes = ['elb', className].filter(Boolean).join(' ');

  return (
    <span
      className={classes}
      data-effect={effect}
      data-kind={kind}
      data-purity={purityLevel}
      data-size={size}
      style={{ '--elb-ink': ink } as CSSProperties}
      title={`${effect} · ${purityLevel}`}
      aria-label={`${effect}, ${purityLevel.toLowerCase()}`}
    >
      <span className="elb-glyph" aria-hidden="true">
        {GLYPH[effect]}
      </span>
      <span className="elb-label">{effect}</span>
    </span>
  );
}
