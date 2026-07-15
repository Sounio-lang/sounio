import { useState } from 'react';
import './EffectBoundaryInstrument.css';

interface Props {
  locale?: string;
}

type CaseId = 'pure' | 'covered' | 'refused';

type EffectCase = {
  id: CaseId;
  caller: string;
  callerEffects: string[];
  callee: string;
  calleeEffects: string[];
  fixture: string;
  exit: number;
  receipt: string;
};

const cases: EffectCase[] = [
  {
    id: 'pure',
    caller: 'main',
    callerEffects: ['IO', 'Mut', 'Div'],
    callee: 'pure_add',
    calleeEffects: [],
    fixture: 'tests/run-pass/effect_superset_ok.sio',
    exit: 0,
    receipt: 'effect superset: PASS',
  },
  {
    id: 'covered',
    caller: 'main',
    callerEffects: ['IO', 'Mut', 'Div'],
    callee: 'needs_io',
    calleeEffects: ['IO'],
    fixture: 'tests/run-pass/effect_superset_ok.sio',
    exit: 0,
    receipt: 'effect superset: PASS',
  },
  {
    id: 'refused',
    caller: 'load_data',
    callerEffects: [],
    callee: 'read_file',
    calleeEffects: ['IO'],
    fixture: 'tests/compile-fail/effect_missing.sio',
    exit: 1,
    receipt: 'E035 · effect not declared in function signature (missing: IO)',
  },
];

const copy = {
  en: {
    eyebrow: 'LANGUAGE CONTRACT / EFFECT ROWS',
    heading: 'Effects cannot cross invisibly.',
    body: 'Sounio puts computational effects in the signature. Explore three calls from the current conformance spine: an empty requirement, a covered IO requirement, and the exact boundary the compiler refuses.',
    labels: ['Pure callee', 'Covered IO', 'Refused IO'],
    descriptions: [
      'A caller may contain more effects than a pure callee requires.',
      'The callee requires IO and the caller declares IO, Mut, and Div.',
      'The callee requires IO, but the intermediate caller declares no effects.',
    ],
    caller: 'caller signature',
    callee: 'callee signature',
    declared: 'declared row',
    required: 'required row',
    empty: 'pure / empty row',
    boundary: 'call boundary',
    admitted: 'admitted',
    stopped: 'stopped before code generation',
    output: 'compiler receipt',
    source: 'Open this fixture',
    manifest: 'Open the conformance manifest',
    pass: 'PASS',
    refuse: 'REFUSE',
    claim: 'Claim boundary',
    claimText: 'Two current conformance fixtures demonstrate static effect-row calls and one missing-IO diagnostic. They do not prove custom handlers, effect polymorphism, inference completeness, runtime isolation, or soundness of the entire effect system.',
    gate: 'CURRENT CONFORMANCE RUN',
    gateResult: '16 / 16 CASES PASS',
  },
  pt: {
    eyebrow: 'CONTRATO DA LINGUAGEM / EFFECT ROWS',
    heading: 'Efeitos não atravessam invisíveis.',
    body: 'O Sounio coloca efeitos computacionais na assinatura. Explore três chamadas da conformance spine atual: uma exigência vazia, uma exigência de IO coberta e a fronteira exata que o compilador recusa.',
    labels: ['Callee puro', 'IO coberto', 'IO recusado'],
    descriptions: [
      'Um caller pode conter mais efeitos do que um callee puro exige.',
      'O callee exige IO e o caller declara IO, Mut e Div.',
      'O callee exige IO, mas o caller intermediário não declara efeitos.',
    ],
    caller: 'assinatura do caller',
    callee: 'assinatura do callee',
    declared: 'row declarada',
    required: 'row exigida',
    empty: 'pura / row vazia',
    boundary: 'fronteira da chamada',
    admitted: 'admitida',
    stopped: 'interrompida antes do codegen',
    output: 'recibo do compilador',
    source: 'Abrir este fixture',
    manifest: 'Abrir o manifesto de conformidade',
    pass: 'PASS',
    refuse: 'RECUSA',
    claim: 'Fronteira da alegação',
    claimText: 'Dois fixtures atuais de conformidade demonstram chamadas com effect rows estáticas e um diagnóstico de IO ausente. Eles não provam handlers customizados, polimorfismo de efeitos, completude de inferência, isolamento em runtime ou soundness do sistema inteiro.',
    gate: 'EXECUÇÃO ATUAL DE CONFORMIDADE',
    gateResult: '16 / 16 CASOS PASSAM',
  },
};

function signature(name: string, effects: string[]) {
  return `fn ${name}()${effects.length ? ` with ${effects.join(', ')}` : ''}`;
}

export default function EffectBoundaryInstrument({ locale = 'en' }: Props) {
  const [selected, setSelected] = useState(1);
  const language = locale === 'pt' ? 'pt' : 'en';
  const d = copy[language];
  const active = cases[selected];
  const refused = active.id === 'refused';
  const sourceUrl = `https://github.com/Sounio-lang/sounio/blob/main/${active.fixture}`;

  return (
    <section className="eb-section" id="effect-boundaries" aria-labelledby="eb-title">
      <div className="eb-atmosphere" aria-hidden="true" />
      <div className="eb-inner">
        <header className="eb-header">
          <div>
            <p className="eb-eyebrow">{d.eyebrow}</p>
            <h2 id="eb-title">{d.heading}</h2>
          </div>
          <div className="eb-intro">
            <img src="/assets/stamps/stamp_monochrome_on_navy.png" alt="" aria-hidden="true" />
            <p>{d.body}</p>
            <div className="eb-links">
              <a href={sourceUrl} target="_blank" rel="noreferrer">{d.source} <span aria-hidden="true">↗</span></a>
              <a href="https://github.com/Sounio-lang/sounio/blob/main/tests/conformance/manifest.v1.tsv" target="_blank" rel="noreferrer">{d.manifest} <span aria-hidden="true">↗</span></a>
            </div>
          </div>
        </header>

        <div className={`eb-instrument ${refused ? 'is-refused' : 'is-pass'}`}>
          <div className="eb-tabs" role="group" aria-label="Effect boundary cases">
            {cases.map((item, index) => (
              <button
                type="button"
                key={item.id}
                aria-pressed={selected === index}
                onClick={() => setSelected(index)}
              >
                <span>0{index + 1}</span>
                <strong>{d.labels[index]}</strong>
              </button>
            ))}
          </div>

          <div className="eb-stage" aria-live="polite">
            <div className="eb-contract">
              <p className="eb-case-description">{d.descriptions[selected]}</p>

              <div className="eb-signature">
                <span>{d.caller}</span>
                <code>{signature(active.caller, active.callerEffects)}</code>
                <div className="eb-row">
                  <small>{d.declared}</small>
                  {active.callerEffects.length > 0
                    ? active.callerEffects.map((effect) => <b key={effect}>{effect}</b>)
                    : <em>{d.empty}</em>}
                </div>
              </div>

              <div className="eb-crossing">
                <span>{d.boundary}</span>
                <i aria-hidden="true" />
                <strong>{refused ? d.stopped : d.admitted}</strong>
              </div>

              <div className="eb-signature eb-signature-callee">
                <span>{d.callee}</span>
                <code>{signature(active.callee, active.calleeEffects)}</code>
                <div className="eb-row">
                  <small>{d.required}</small>
                  {active.calleeEffects.length > 0
                    ? active.calleeEffects.map((effect) => <b key={effect}>{effect}</b>)
                    : <em>{d.empty}</em>}
                </div>
              </div>
            </div>

            <aside className="eb-receipt">
              <div className="eb-verdict">
                <span>{active.exit === 0 ? d.pass : d.refuse}</span>
                <strong>exit {active.exit}</strong>
              </div>
              <div className="eb-output">
                <span>{d.output}</span>
                <code>{active.receipt}</code>
              </div>
              <dl>
                <div><dt>caller</dt><dd>{`{${active.callerEffects.join(', ')}}` || '{}'}</dd></div>
                <div><dt>callee</dt><dd>{`{${active.calleeEffects.join(', ')}}` || '{}'}</dd></div>
                <div><dt>fixture</dt><dd>{active.fixture.split('/').at(-1)}</dd></div>
              </dl>
              <div className="eb-gate">
                <span>{d.gate}</span>
                <strong>{d.gateResult}</strong>
                <code>serious_language_conformance_gate.sh</code>
              </div>
            </aside>
          </div>
        </div>

        <footer className="eb-claim">
          <strong>{d.claim}</strong>
          <span>{d.claimText}</span>
          <code>effects.subtyping · effects.diagnostics</code>
        </footer>
      </div>
    </section>
  );
}
