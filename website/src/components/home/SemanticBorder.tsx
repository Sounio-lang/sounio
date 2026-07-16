import { useState } from 'react';
import './SemanticBorder.css';

interface Props {
  locale?: string;
}

type Crossing = 'accept' | 'refuse';

const cases = {
  accept: {
    runtime: 'GO:0007166',
    nominal: 'GO_0007166',
    requested: 'GO_0008150',
    relation: 'GO_0007166 <: GO_0007165 <: GO_0008150',
    source: 'tests/run-pass/ontology_typed_bridge_go.sio',
    receipt: 'ontology typed bridge go OK',
  },
  refuse: {
    runtime: 'GO:0008150',
    nominal: 'GO_0008150',
    requested: 'GO_0007166',
    relation: 'GO_0008150 </: GO_0007166',
    source: 'tests/compile-fail/ontology_typed_bridge_go_reject.sio',
    receipt: 'E152: cannot prove ontology subsumption at this call site',
  },
} as const;

const copy = {
  en: {
    eyebrow: 'THE SEMANTIC BORDER / GO SUBSUMPTION',
    heading: 'When an identifier becomes a type.',
    body: 'A runtime Gene Ontology IRI crosses into a nominal Sounio type. From there, the call site must prove the direction of the hierarchy: descendants satisfy ancestors; ancestors cannot impersonate descendants.',
    acceptTab: 'Descendant to ancestor',
    refuseTab: 'Root to descendant',
    runtime: 'runtime identity',
    border: 'semantic border',
    static: 'static obligation',
    passport: 'IRI passport',
    materialized: 'nominal type',
    requires: 'callee requires',
    accepted: 'ACCEPT',
    refused: 'REFUSE',
    acceptedText: 'The descendant satisfies the ancestor.',
    refusedText: 'The hierarchy cannot prove the reverse.',
    lineage: 'Gene Ontology lineage in this bundle',
    process: 'biological process',
    signaling: 'signaling',
    transduction: 'signal transduction',
    gate: 'RECORDED GATE',
    command: 'command',
    engine: 'compiler path',
    result: 'result',
    witness: 'witness',
    boundary: 'Claim boundary',
    boundaryText: 'This receipt proves one generated GO bundle, one Linux x86-64 compatibility path, and static subsumption at these call sites. The IRI cast trusts its numeric ID; it is not arbitrary OWL/DL reasoning, multi-bundle composition, or proof of the public Madaros compile subcommand.',
    openGate: 'Open gate',
    openWitness: 'Open witness',
  },
  pt: {
    eyebrow: 'A FRONTEIRA SEMÂNTICA / SUBSUNÇÃO GO',
    heading: 'Quando um identificador vira tipo.',
    body: 'Um IRI da Gene Ontology em runtime atravessa para um tipo nominal de Sounio. A partir daí, o call site precisa provar a direção da hierarquia: descendentes satisfazem ancestrais; ancestrais não podem se passar por descendentes.',
    acceptTab: 'Descendente para ancestral',
    refuseTab: 'Raiz para descendente',
    runtime: 'identidade em runtime',
    border: 'fronteira semântica',
    static: 'obrigação estática',
    passport: 'passaporte IRI',
    materialized: 'tipo nominal',
    requires: 'callee exige',
    accepted: 'ACEITA',
    refused: 'RECUSA',
    acceptedText: 'O descendente satisfaz o ancestral.',
    refusedText: 'A hierarquia não consegue provar o inverso.',
    lineage: 'Linhagem da Gene Ontology neste bundle',
    process: 'processo biológico',
    signaling: 'sinalização',
    transduction: 'transdução de sinal',
    gate: 'GATE REGISTRADO',
    command: 'comando',
    engine: 'caminho do compilador',
    result: 'resultado',
    witness: 'testemunho',
    boundary: 'Fronteira da alegação',
    boundaryText: 'Este recibo prova um bundle GO gerado, um caminho de compatibilidade Linux x86-64 e subsunção estática nestes call sites. O cast do IRI confia no ID numérico; não é raciocínio OWL/DL arbitrário, composição de múltiplos bundles ou prova do subcomando público compile de Madaros.',
    openGate: 'Abrir gate',
    openWitness: 'Abrir testemunho',
  },
};

const repo = 'https://github.com/Sounio-lang/sounio/blob/website/living-observatory-20260713';

export default function SemanticBorder({ locale = 'en' }: Props) {
  const [selected, setSelected] = useState<Crossing>('accept');
  const d = copy[locale === 'pt' ? 'pt' : 'en'];
  const active = cases[selected];
  const accepted = selected === 'accept';

  return (
    <section id="semantic-border" className={`sb-section ${accepted ? 'is-accept' : 'is-refuse'}`} aria-labelledby="semantic-border-title">
      <div className="sb-shell">
        <header className="sb-header">
          <div>
            <p className="sb-eyebrow">{d.eyebrow}</p>
            <h2 id="semantic-border-title">{d.heading}</h2>
          </div>
          <p className="sb-intro">{d.body}</p>
        </header>

        <div className="sb-controls" role="group" aria-label="Ontology subsumption cases">
          <button type="button" aria-pressed={accepted} onClick={() => setSelected('accept')}>
            <span>01</span>{d.acceptTab}<b>ACCEPT</b>
          </button>
          <button type="button" aria-pressed={!accepted} onClick={() => setSelected('refuse')}>
            <span>02</span>{d.refuseTab}<b>E152</b>
          </button>
        </div>

        <div className="sb-instrument" aria-live="polite">
          <aside className="sb-lineage" aria-label={d.lineage}>
            <p>{d.lineage}</p>
            <ol>
              <li className={accepted ? 'is-lit' : ''}>
                <code>GO_0008150</code><span>{d.process}</span>
              </li>
              <li className={accepted ? 'is-lit' : ''}>
                <code>GO_0007165</code><span>{d.signaling}</span>
              </li>
              <li className={accepted ? 'is-lit is-origin' : ''}>
                <code>GO_0007166</code><span>{d.transduction}</span>
              </li>
            </ol>
          </aside>

          <div className="sb-crossing">
            <div className="sb-runtime">
              <span>{d.runtime}</span>
              <div className="sb-passport">
                <small>{d.passport}</small>
                <strong>{active.runtime}</strong>
                <i>iri.id</i>
              </div>
            </div>

            <div className="sb-border" aria-hidden="true">
              <span>{d.border}</span>
              <i />
              <b>as GO type</b>
            </div>

            <div className="sb-static">
              <span>{d.static}</span>
              <div className="sb-type">
                <small>{d.materialized}</small>
                <strong>{active.nominal}</strong>
              </div>
              <div className="sb-requirement">
                <small>{d.requires}</small>
                <code>{active.requested}</code>
              </div>
            </div>
          </div>

          <aside className="sb-decision">
            <div className="sb-verdict">
              <span>{accepted ? d.accepted : d.refused}</span>
              <strong>{accepted ? 'exit 0' : 'exit 1 · E152'}</strong>
            </div>
            <code className="sb-relation">{active.relation}</code>
            <p>{accepted ? d.acceptedText : d.refusedText}</p>
            <div className="sb-diagnostic">
              <small>{d.result}</small>
              <code>{active.receipt}</code>
            </div>
          </aside>
        </div>

        <div className="sb-ledger">
          <div className="sb-ledger-title"><span>{d.gate}</span><strong>PASS · E152 observed</strong></div>
          <dl>
            <div><dt>{d.command}</dt><dd><code>bash scripts/ci/ontology_typed_bridge_gate.sh</code></dd></div>
            <div><dt>{d.engine}</dt><dd><code>lean_single · legacy positional · Linux/x86-64</code></dd></div>
            <div><dt>{d.result}</dt><dd><code>[ontology-typed-bridge] PASS: bridge round-trips and bad subsumption is rejected (E152)</code></dd></div>
            <div><dt>{d.witness}</dt><dd><code>{active.source}</code></dd></div>
          </dl>
          <div className="sb-ledger-links">
            <a href={`${repo}/scripts/ci/ontology_typed_bridge_gate.sh`} target="_blank" rel="noreferrer">{d.openGate}<span aria-hidden="true">↗</span></a>
            <a href={`${repo}/${active.source}`} target="_blank" rel="noreferrer">{d.openWitness}<span aria-hidden="true">↗</span></a>
          </div>
        </div>

        <footer className="sb-claim">
          <strong>{d.boundary}</strong>
          <span>{d.boundaryText}</span>
          <code>GO bundle 1/compile-unit · trusted iri.id · no OWL/DL claim</code>
        </footer>
      </div>
    </section>
  );
}
