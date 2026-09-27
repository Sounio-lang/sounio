import type { CSSProperties } from 'react';
import {
  formatFixed,
  readKnowledge,
  type EpistemicState,
  type KnowledgeInput,
} from '../../lib/epistemicHonesty';
import './KnowledgeRefusalMeter.css';

export type KnowledgeRefusalMeterProps = KnowledgeInput & {
  signature: string;
  unit?: string;
  provenance?: string;
  coverageK?: number;
  windowLow?: number | null;
  windowHigh?: number | null;
  guard?: string;
  diagnostic?: string;
  density?: 'full' | 'compact';
  className?: string;
};

const TOKEN: Record<
  EpistemicState,
  { ink: string; surface: string; border: string; text: string; glyph: string; label: string }
> = {
  verified: {
    ink: 'var(--color-epistemic-verified)',
    surface: 'var(--color-epistemic-verified-surface)',
    border: 'var(--color-epistemic-verified-border)',
    text: 'var(--color-epistemic-verified-text)',
    glyph: '◈',
    label: 'Verified',
  },
  uncertain: {
    ink: 'var(--color-epistemic-uncertain)',
    surface: 'var(--color-epistemic-uncertain-surface)',
    border: 'var(--color-epistemic-uncertain-border)',
    text: 'var(--color-epistemic-uncertain-text)',
    glyph: '△',
    label: 'Uncertain',
  },
  refused: {
    ink: 'var(--color-epistemic-refused)',
    surface: 'var(--color-epistemic-refused-surface)',
    border: 'var(--color-epistemic-refused-border)',
    text: 'var(--color-epistemic-refused-text)',
    glyph: '⊘',
    label: 'Refused',
  },
};

function pct(value: number, lo: number, hi: number): number {
  if (hi <= lo) return 50;
  return Math.min(100, Math.max(0, ((value - lo) / (hi - lo)) * 100));
}

function domainOf(values: Array<number | null | undefined>): [number, number] | null {
  const xs = values.filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v));
  if (xs.length === 0) return null;
  const lo = Math.min(...xs);
  const hi = Math.max(...xs);
  if (lo === hi) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.12;
  return [lo - pad, hi + pad];
}

function epsilonLabel(
  status: ReturnType<typeof readKnowledge>['epsilonStatus'],
  epsilon: number | null,
): string {
  if (status === 'readable' && epsilon !== null) return `ε = ${formatFixed(epsilon, 2)}`;
  if (status === 'missing') return 'ε absent';
  return 'ε unreadable';
}

function varianceLabel(
  status: ReturnType<typeof readKnowledge>['varianceStatus'],
  variance: number | null,
  expandedU: number | null,
  unit: string,
  coverageK: number,
): string | null {
  if (status === 'calibrated' && variance !== null) {
    const uPart =
      expandedU !== null
        ? ` ± ${formatFixed(expandedU, 3)} (k=${coverageK}, GUM)`
        : ` var = ${formatFixed(variance, 6)}`;
    return `${unit}${uPart}`.trim();
  }
  if (status === 'uncalibrated') return 'var uncalibrated';
  if (status === 'invalid') return 'var unreadable';
  return null;
}

function defaultBanner(
  state: EpistemicState,
  reason: ReturnType<typeof readKnowledge>['reason'],
  guard?: string,
): { title: string; body: string } {
  if (reason === 'fabrication-shape') {
    return {
      title: `${TOKEN.refused.glyph} Refusal — fabrication refused`,
      body: 'A stub zero paired with a non-unit confidence is not a Knowledge value. The control prints no numeral.',
    };
  }
  if (reason === 'unreadable-epsilon') {
    return {
      title: `${TOKEN.refused.glyph} Refusal — ε is not a unit interval`,
      body: 'Confidence outside [0, 1] is not rendered as ε. A raw integer is a bit pattern, not a degree of belief.',
    };
  }
  if (reason === 'guard-failed') {
    return {
      title: `${TOKEN.refused.glyph} Refusal — guard`,
      body: guard
        ? `The static constraint ${guard} does not hold. The program does not produce a downstream value.`
        : 'The confidence guard does not hold. The program does not produce a downstream value.',
    };
  }
  if (state === 'verified') {
    return {
      title: `${TOKEN.verified.glyph} Verified`,
      body: guard
        ? `Support sits inside the gate ${guard}. Variance, if uncalibrated, is still not printed as zero.`
        : 'The payload is readable and the caller claims verification.',
    };
  }
  return {
    title: `${TOKEN.uncertain.glyph} Uncertain`,
    body: 'The measurement stands, with its doubt attached. This is not an error state.',
  };
}

export function KnowledgeRefusalMeter({
  signature,
  unit = '',
  provenance,
  coverageK = 2,
  windowLow,
  windowHigh,
  guard,
  diagnostic,
  density = 'full',
  className,
  ...input
}: KnowledgeRefusalMeterProps) {
  const reading = readKnowledge(input, coverageK);
  const tone = TOKEN[reading.state];
  const banner = diagnostic
    ? {
        title: `${tone.glyph} ${tone.label}${reading.reason === 'fabrication-shape' ? ' — fabrication' : ''}`,
        body: diagnostic,
      }
    : defaultBanner(reading.state, reading.reason, guard);

  const hasInterval = reading.boundLow !== null && reading.boundHigh !== null;
  const scale = domainOf([
    reading.value,
    reading.boundLow,
    reading.boundHigh,
    windowLow,
    windowHigh,
    reading.value !== null && reading.expandedU !== null ? reading.value - reading.expandedU : null,
    reading.value !== null && reading.expandedU !== null ? reading.value + reading.expandedU : null,
  ]);

  const envelopeLow =
    reading.boundLow ??
    (reading.value !== null && reading.expandedU !== null ? reading.value - reading.expandedU : null);
  const envelopeHigh =
    reading.boundHigh ??
    (reading.value !== null && reading.expandedU !== null ? reading.value + reading.expandedU : null);
  const showRibbon =
    scale !== null && envelopeLow !== null && envelopeHigh !== null && envelopeLow <= envelopeHigh;

  const primary = hasInterval
    ? `[${formatFixed(reading.boundLow as number, 2)}, ${formatFixed(reading.boundHigh as number, 2)}]`
    : reading.value !== null
      ? formatFixed(reading.value, reading.value >= 10 ? 2 : 3)
      : `${tone.glyph}`;

  const unitLine = varianceLabel(
    reading.varianceStatus,
    reading.variance,
    reading.expandedU,
    unit,
    coverageK,
  );
  const unitFallback = unit && !unitLine ? unit : null;

  const [domainLo, domainHi] = scale ?? [0, 1];
  const envLeft = showRibbon ? pct(envelopeLow as number, domainLo, domainHi) : 0;
  const envRight = showRibbon ? pct(envelopeHigh as number, domainLo, domainHi) : 0;
  const needle =
    showRibbon && reading.value !== null ? pct(reading.value, domainLo, domainHi) : null;
  const windowLeft =
    showRibbon && windowLow != null && windowHigh != null
      ? pct(windowLow, domainLo, domainHi)
      : null;
  const windowRight =
    showRibbon && windowLow != null && windowHigh != null
      ? pct(windowHigh, domainLo, domainHi)
      : null;

  return (
    <article
      className={['krm', className].filter(Boolean).join(' ')}
      data-state={reading.state}
      data-density={density}
      data-reason={reading.reason}
      style={
        {
          '--krm-ink': tone.ink,
          '--krm-surface': tone.surface,
          '--krm-border': tone.border,
          '--krm-text': tone.text,
        } as CSSProperties
      }
      aria-label={`${signature}: ${tone.label}`}
    >
      <header className="krm-header">
        <span className="krm-signature">{signature}</span>
        <div className="krm-meta">
          {provenance ? <span className="krm-provenance">{provenance}</span> : null}
          <span className="krm-epsilon">{epsilonLabel(reading.epsilonStatus, reading.epsilon)}</span>
        </div>
      </header>

      <div className="krm-readout">
        <span className="krm-nominal" data-empty={reading.value === null && !hasInterval}>
          {primary}
        </span>
        {unitLine ? <span className="krm-unit">{unitLine}</span> : null}
        {!unitLine && unitFallback ? <span className="krm-unit">{unitFallback}</span> : null}
        <span className="krm-spacer" />
        <span className="krm-verdict">{tone.label}</span>
      </div>

      {showRibbon ? (
        <div className="krm-ribbon" aria-hidden="true">
          <div className="krm-gauge">
            <div className="krm-axis" />
            {windowLeft !== null && windowRight !== null ? (
              <div
                className="krm-window"
                style={{ left: `${windowLeft}%`, width: `${Math.max(0, windowRight - windowLeft)}%` }}
              />
            ) : null}
            <div
              className="krm-envelope"
              style={{ left: `${envLeft}%`, width: `${Math.max(1.5, envRight - envLeft)}%` }}
            />
            {needle !== null ? <div className="krm-needle" style={{ left: `${needle}%` }} /> : null}
          </div>
          <div className="krm-scale">
            <span>{formatFixed(envelopeLow as number, 2)}</span>
            {windowLow != null && windowHigh != null ? (
              <span className="krm-scale-mid">
                window [{formatFixed(windowLow, 0)}, {formatFixed(windowHigh, 0)}]
              </span>
            ) : (
              <span />
            )}
            <span>{formatFixed(envelopeHigh as number, 2)}</span>
          </div>
        </div>
      ) : (
        <p className="krm-absent">
          {reading.varianceStatus === 'uncalibrated'
            ? 'No GUM ribbon — variance left uncalibrated rather than printed as 0.000.'
            : 'No GUM ribbon — no calibrated dispersion was supplied.'}
        </p>
      )}

      {density === 'full' ? (
        <footer className="krm-banner">
          <div className="krm-banner-head">
            <span className="krm-banner-title">
              <span className="krm-glyph" aria-hidden="true">
                {tone.glyph}
              </span>
              {banner.title.replace(/^[◈△⊘]\s/, '')}
            </span>
          </div>
          <p className="krm-banner-body">{banner.body}</p>
        </footer>
      ) : null}
    </article>
  );
}
