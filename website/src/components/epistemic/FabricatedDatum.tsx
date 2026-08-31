import type { CSSProperties } from 'react';
import {
  KIND_LABEL,
  readFabrication,
  type FabricationKind,
} from '../../lib/fabricationHonesty';
import './FabricatedDatum.css';

export type FabricatedDatumProps = {
  kind: FabricationKind;
  signature: string;
  printed: string;
  actual: string;
  printedLabel: string;
  actualLabel: string;
  reason: string;
  sha: string;
  href: string;
  hrefLabel: string;
  className?: string;
};

export function FabricatedDatum({
  kind,
  signature,
  printed,
  actual,
  printedLabel,
  actualLabel,
  reason,
  sha,
  href,
  hrefLabel,
  className,
}: FabricatedDatumProps) {
  const reading = readFabrication(kind, printed, actual);
  const classes = ['fd', className].filter(Boolean).join(' ');

  if (reading.state === 'vacuous') {
    return (
      <article
        className={classes}
        data-state="vacuous"
        data-kind={kind}
        aria-label={`${signature}: not a fabrication witness`}
      >
        <header className="fd-header">
          <span className="fd-signature">{signature}</span>
          <span className="fd-chip">Vacuous</span>
        </header>
        <p className="fd-reason">
          {reading.reason === 'faces-identical'
            ? 'Printed and actual are the same string. That is not a fabrication witness — showing a tear here would fabricate one.'
            : 'A fabrication witness needs both faces. One of them is empty.'}
        </p>
      </article>
    );
  }

  const printedIsLong = reading.printed.length > 48;
  const actualIsLong = reading.actual.length > 48;

  return (
    <article
      className={classes}
      data-state="fabricated"
      data-kind={kind}
      style={
        {
          '--fd-ink': 'var(--color-epistemic-refused)',
          '--fd-surface': 'var(--color-epistemic-refused-surface)',
          '--fd-border': 'var(--color-epistemic-refused-border)',
          '--fd-text': 'var(--color-epistemic-refused-text)',
        } as CSSProperties
      }
      aria-label={`${signature}: fabricated — ${KIND_LABEL[kind]}`}
    >
      <header className="fd-header">
        <span className="fd-signature">{signature}</span>
        <span className="fd-chip">
          <span className="fd-glyph" aria-hidden="true">
            ≠
          </span>
          Fabricated
        </span>
      </header>

      <div className="fd-faces" data-kind={kind}>
        <div className="fd-face fd-printed">
          <p className="fd-face-kicker">{printedLabel}</p>
          <p className="fd-face-label">Printed as given</p>
          <pre className="fd-numeral" data-long={printedIsLong}>
            {reading.printed}
          </pre>
        </div>

        <div className="fd-tear" aria-hidden="true">
          <span className="fd-tear-rule" />
          <span className="fd-tear-mark">tear</span>
        </div>

        <div className="fd-face fd-actual">
          <p className="fd-face-kicker">{actualLabel}</p>
          <p className="fd-face-label">
            {kind === 'truncated' ? 'Dropped after 127' : 'The other reading'}
          </p>
          <pre className="fd-numeral" data-long={actualIsLong}>
            {reading.actual}
          </pre>
        </div>
      </div>

      <p className="fd-reason">{reason}</p>

      <footer className="fd-footer">
        <code className="fd-sha">{sha}</code>
        <a className="fd-link" href={href} target="_blank" rel="noopener noreferrer">
          {hrefLabel}
        </a>
      </footer>
    </article>
  );
}
