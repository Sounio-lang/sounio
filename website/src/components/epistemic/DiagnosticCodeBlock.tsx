import {
  E219_HELP,
  E219_MESSAGE,
  E219_NOTE,
  PRE_FIX_PRINT_CAP,
  splitAtPrintCap,
} from '../../lib/diagnosticHonesty';
import './DiagnosticCodeBlock.css';

export type DiagnosticSpan = {
  file: string;
  line: number;
  column: number;
  endColumn: number;
};

export type DiagnosticLine = {
  number: number;
  text: string;
};

export type DiagnosticCodeBlockProps = {
  filename: string;
  engine?: string;
  effects?: string[];
  lines: DiagnosticLine[];
  span: DiagnosticSpan;
  code?: string;
  message: string;
  help?: string;
  note?: string;
  /** Mark the pre-#1784 127-character print cap inside a long help literal. */
  markPrintCap?: boolean;
  className?: string;
};

function caretFor(line: string, column: number, endColumn: number): string {
  const start = Math.min(line.length, Math.max(0, column - 1));
  const end = Math.min(line.length, Math.max(start + 1, endColumn - 1));
  return `${' '.repeat(start)}${'^'.repeat(Math.max(1, end - start))}`;
}

function renderLineText(
  text: string,
  hit: boolean,
  column: number,
  endColumn: number,
) {
  if (!hit) return text;
  const start = Math.max(0, column - 1);
  const end = Math.min(text.length, Math.max(start + 1, endColumn - 1));
  return (
    <>
      {text.slice(0, start)}
      <span className="dcb-hit">{text.slice(start, end)}</span>
      {text.slice(end)}
    </>
  );
}

export function DiagnosticCodeBlock({
  filename,
  engine = 'Madaros',
  effects = [],
  lines,
  span,
  code = 'E219',
  message,
  help,
  note,
  markPrintCap = true,
  className,
}: DiagnosticCodeBlockProps) {
  const helpSplit = help ? splitAtPrintCap(help) : null;
  const showCut = Boolean(markPrintCap && helpSplit?.wouldTruncate);

  return (
    <article
      className={['dcb', className].filter(Boolean).join(' ')}
      data-code={code}
      aria-label={`${code} ${message}`}
    >
      <header className="dcb-header">
        <span className="dcb-file">{filename}</span>
        <div className="dcb-meta-row">
          <span className="dcb-engine">{engine}</span>
          {effects.map((effect) => (
            <span key={effect} className="dcb-effect">
              {effect}
            </span>
          ))}
          <span className="dcb-code-chip">{code}</span>
          <span className="dcb-verdict">Refused</span>
        </div>
      </header>

      <pre className="dcb-body">
        {lines.map((line) => {
          const hit = line.number === span.line;
          return (
            <div key={line.number}>
              <div className="dcb-line" data-hit={hit}>
                <span className="dcb-gutter">{line.number}</span>
                <code className="dcb-src">
                  {renderLineText(line.text, hit, span.column, span.endColumn)}
                </code>
              </div>
              {hit ? (
                <div className="dcb-line dcb-caret-row" aria-hidden="true">
                  <span className="dcb-gutter" />
                  <span className="dcb-caret">
                    {caretFor(line.text, span.column, span.endColumn)}
                  </span>
                </div>
              ) : null}
            </div>
          );
        })}
      </pre>

      <section className="dcb-ribbon">
        <div className="dcb-ribbon-head">
          <span className="dcb-code-chip">{code}</span>
          <span className="dcb-span">
            {'--> '}
            {span.file}:{span.line}:{span.column}
          </span>
        </div>
        <p className="dcb-diag-body">
          error[{code}]: {message}
        </p>
        {helpSplit ? (
          <p className="dcb-help">
            {showCut ? (
              <>
                <span className="dcb-help-kept">{helpSplit.kept}</span>
                <span
                  className="dcb-cut"
                  title={`Pre-#1784 print cap: first ${PRE_FIX_PRINT_CAP} characters kept, ${helpSplit.droppedCount} dropped`}
                >
                  <span className="dcb-cut-rule" />
                  <span className="dcb-cut-label">cut {PRE_FIX_PRINT_CAP}</span>
                </span>
                <span className="dcb-help-dropped">{helpSplit.dropped}</span>
              </>
            ) : (
              help
            )}
          </p>
        ) : null}
        {note ? <p className="dcb-note">{note}</p> : null}
      </section>

      {helpSplit ? (
        <footer className="dcb-footer">
          <p className="dcb-meta">
            Help literal {helpSplit.length} characters
            {helpSplit.wouldTruncate ? (
              <>
                {' · '}
                pre-#1784 would keep {PRE_FIX_PRINT_CAP} and drop{' '}
                <strong>{helpSplit.droppedCount}</strong>
              </>
            ) : (
              <> · under the old 127-character cap</>
            )}
          </p>
        </footer>
      ) : null}
    </article>
  );
}

export const E219_WITNESS_LINES: DiagnosticLine[] = [
  { number: 19, text: 'extern "C" {' },
  { number: 20, text: '    fn malloc(size: i64) -> i64;' },
  { number: 21, text: '    fn abs(x: i64) -> i64;' },
  { number: 22, text: '}' },
  { number: 23, text: '' },
  { number: 24, text: 'fn main() -> i64 with IO {' },
  { number: 25, text: '    let p = malloc(64)' },
  { number: 26, text: '    print_int(p)' },
  { number: 27, text: '    print_int(abs(0 - 7))' },
  { number: 28, text: '    0' },
  { number: 29, text: '}' },
];

// P0-F allow-listed malloc. The live E219 in this fixture is abs — the
// call that used to compile to a fabricated 0 (#1622) and that #1801
// now infects as ty_error() rather than the declared i64.
export const E219_WITNESS_SPAN: DiagnosticSpan = {
  file: 'tests/compile-fail/extern_c_unimplemented_builtin.sio',
  line: 27,
  column: 15,
  endColumn: 18,
};

export function E219DiagnosticCodeBlock(props: { className?: string }) {
  return (
    <DiagnosticCodeBlock
      filename="tests/compile-fail/extern_c_unimplemented_builtin.sio"
      engine="Madaros"
      effects={['IO']}
      lines={E219_WITNESS_LINES}
      span={E219_WITNESS_SPAN}
      code="E219"
      message={E219_MESSAGE}
      help={E219_HELP}
      note={E219_NOTE}
      markPrintCap
      className={props.className}
    />
  );
}
