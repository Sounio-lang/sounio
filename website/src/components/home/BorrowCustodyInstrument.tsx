import { useState } from 'react';
import './BorrowCustodyInstrument.css';

interface Props { locale?: string; }

const cases = [
  {
    id: 'shared',
    fixture: 'tests/run-pass/borrow_reborrow.sio',
    source: 'let x: i64 = 42',
    live: ['a = &x', 'b = &x'],
    request: 'read_ref(a) + read_ref(b)',
    result: '42 + 42 = 84',
    exit: 0,
    receipt: 'borrow reborrow: PASS',
  },
  {
    id: 'exclusive',
    fixture: 'tests/compile-fail/borrow_call_conflict_explicit.sio',
    source: 'var x: i64 = 42',
    live: ['hold = &x'],
    request: 'read_exclusive(&!x)',
    result: 'stopped at call boundary',
    exit: 1,
    receipt: 'E038 · cannot borrow exclusively while other borrows are active',
  },
] as const;

const copy = {
  en: {
    eyebrow: 'OWNERSHIP CONTRACT / BORROW CUSTODY',
    heading: 'Custody stays visible.',
    body: 'A reference is not an invisible alias. Inspect one fixture where two shared borrows coexist, and one where an exclusive request is refused while shared custody remains live.',
    labels: ['Shared custody', 'Exclusive conflict'],
    descriptions: [
      'Two shared references to x remain live together and both read the same value.',
      'A shared reference to x is still live when an exclusive borrow is requested at a call boundary.',
    ],
    sourceValue: 'owned value', liveCustody: 'live custody ledger', requested: 'requested operation', outcome: 'checker outcome',
    admitted: 'coexists', refused: 'refused', pass: 'PASS', refuse: 'REFUSE', output: 'compiler receipt',
    source: 'Open this fixture', manifest: 'Open the conformance case', gate: 'TARGETED SUITE', gateResult: '2 / 2 FIXTURES PASS',
    boundary: 'Claim boundary',
    boundaryText: 'These fixtures show shared reborrowing and one explicit shared-versus-exclusive conflict at a call boundary. They do not establish Rust-equivalent ownership, general lifetime inference, concurrency safety, or formal soundness of the complete borrow system.',
  },
  pt: {
    eyebrow: 'CONTRATO DE OWNERSHIP / CUSTÓDIA DE BORROWS',
    heading: 'A custódia permanece visível.',
    body: 'Uma referência não é um alias invisível. Inspecione um fixture em que dois borrows compartilhados coexistem e outro em que um pedido exclusivo é recusado enquanto a custódia compartilhada continua viva.',
    labels: ['Custódia compartilhada', 'Conflito exclusivo'],
    descriptions: [
      'Duas referências compartilhadas para x permanecem vivas e ambas leem o mesmo valor.',
      'Uma referência compartilhada para x ainda está viva quando um borrow exclusivo é solicitado na fronteira da chamada.',
    ],
    sourceValue: 'valor possuído', liveCustody: 'ledger de custódia viva', requested: 'operação solicitada', outcome: 'resultado do checker',
    admitted: 'coexiste', refused: 'recusada', pass: 'PASSA', refuse: 'RECUSA', output: 'recibo do compilador',
    source: 'Abrir este fixture', manifest: 'Abrir o caso de conformidade', gate: 'SUÍTE DIRECIONADA', gateResult: '2 / 2 FIXTURES PASSAM',
    boundary: 'Fronteira da alegação',
    boundaryText: 'Estes fixtures mostram reborrow compartilhado e um conflito explícito entre borrow compartilhado e exclusivo na fronteira de uma chamada. Eles não estabelecem ownership equivalente ao Rust, inferência geral de lifetimes, segurança concorrente ou soundness formal do sistema completo de borrows.',
  },
};

export default function BorrowCustodyInstrument({ locale = 'en' }: Props) {
  const [selected, setSelected] = useState(0);
  const d = copy[locale === 'pt' ? 'pt' : 'en'];
  const active = cases[selected];
  const refused = active.id === 'exclusive';
  const sourceUrl = `https://github.com/Sounio-lang/sounio/blob/main/${active.fixture}`;

  return (
    <section className="bc-section" id="borrow-custody" aria-labelledby="bc-title">
      <div className="bc-atmosphere" aria-hidden="true" />
      <div className="bc-inner">
        <header className="bc-header">
          <div><p className="bc-eyebrow">{d.eyebrow}</p><h2 id="bc-title">{d.heading}</h2></div>
          <div className="bc-intro">
            <p>{d.body}</p>
            <div className="bc-links">
              <a href={sourceUrl} target="_blank" rel="noreferrer">{d.source} <span aria-hidden="true">↗</span></a>
              <a href="https://github.com/Sounio-lang/sounio/blob/main/tests/conformance/manifest.v1.tsv#L11-L12" target="_blank" rel="noreferrer">{d.manifest} <span aria-hidden="true">↗</span></a>
            </div>
          </div>
        </header>

        <div className={`bc-instrument ${refused ? 'is-refused' : 'is-pass'}`}>
          <div className="bc-tabs" role="tablist" aria-label="Borrow custody cases">
            {cases.map((item, index) => (
              <button
                type="button"
                role="tab"
                id={`bc-tab-${item.id}`}
                aria-controls="bc-case-panel"
                aria-selected={selected === index}
                key={item.id}
                onClick={() => setSelected(index)}
              >
                <span>0{index + 1}</span><strong>{d.labels[index]}</strong><small>{index === 0 ? '&x + &x' : '&x / &!x'}</small>
              </button>
            ))}
          </div>

          <div
            className="bc-stage"
            id="bc-case-panel"
            role="tabpanel"
            aria-labelledby={`bc-tab-${active.id}`}
            aria-live="polite"
          >
            <div className="bc-ledger">
              <p className="bc-description">{d.descriptions[selected]}</p>
              <div className="bc-ledger-row bc-owned"><span>{d.sourceValue}</span><code>{active.source}</code><strong>x</strong></div>
              <div className="bc-ledger-row bc-live"><span>{d.liveCustody}</span><div>{active.live.map((borrow) => <code key={borrow}>{borrow}</code>)}</div><strong>{active.live.length} shared</strong></div>
              <div className="bc-ledger-row bc-request"><span>{d.requested}</span><code>{active.request}</code><strong>{refused ? d.refused : d.admitted}</strong></div>
              <div className="bc-outcome"><span>{d.outcome}</span><strong>{active.result}</strong></div>
            </div>

            <aside className="bc-receipt">
              <div className="bc-verdict"><span>{refused ? d.refuse : d.pass}</span><strong>exit {active.exit}</strong></div>
              <div className="bc-output"><span>{d.output}</span><code>{active.receipt}</code></div>
              <dl>
                <div><dt>owner</dt><dd>x: i64 = 42</dd></div>
                <div><dt>live</dt><dd>{active.live.join(' · ')}</dd></div>
                <div><dt>request</dt><dd>{refused ? '&!x exclusive' : 'two shared reads'}</dd></div>
                <div><dt>fixture</dt><dd>{active.fixture.split('/').at(-1)}</dd></div>
              </dl>
              <div className="bc-gate"><span>{d.gate}</span><strong>{d.gateResult}</strong><code>run_sio_test_suite.sh</code></div>
            </aside>
          </div>
        </div>

        <footer className="bc-claim"><strong>{d.boundary}</strong><span>{d.boundaryText}</span><code>ownership.borrowing · partially_executable</code></footer>
      </div>
    </section>
  );
}
