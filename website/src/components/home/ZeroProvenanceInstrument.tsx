import ZeroEventObservatory from './ZeroEventObservatory';
import './ZeroProvenanceInstrument.css';

interface Props {
  locale?: string;
}

const copy = {
  en: {
    eyebrow: 'EPISTEMIC RECEIPT / ZERO PROVENANCE',
    title: 'Not every zero means nothing.',
    body: 'Seven computation paths arrive at the same surface value. Sounio keeps their evidence distinct until an explicit discharge chooses to forget it.',
    source: 'Open the library source',
    gate: 'Open the verification gate',
    traceLabel: 'Evidence paths',
    traceTitle: 'same surface / different provenance',
    convergedLabel: 'surface value 0.0',
    receipt: 'CURRENT REPOSITORY GATE',
    pass: 'ZERO EVENT GATE PASS',
    proof: 'receipts · evidence · explicit discharge · EISA flags',
    boundary: 'Claim boundary',
    boundaryCopy: 'These are computational evidence categories implemented by stdlib/epistemic/zero_event.sio. A receipt does not establish a biological, clinical, physical, or metaphysical cause, and the gate does not claim that every scalar zero requires this representation.',
  },
  pt: {
    eyebrow: 'RECIBO EPISTÊMICO / PROVENIÊNCIA DO ZERO',
    title: 'Nem todo zero significa ausência.',
    body: 'Sete caminhos computacionais chegam ao mesmo valor de superfície. O Sounio mantém suas evidências distintas até que uma descarga explícita escolha esquecê-las.',
    source: 'Abrir o código da biblioteca',
    gate: 'Abrir o gate de verificação',
    traceLabel: 'Caminhos de evidência',
    traceTitle: 'mesma superfície / proveniência diferente',
    convergedLabel: 'valor de superfície 0,0',
    receipt: 'GATE ATUAL DO REPOSITÓRIO',
    pass: 'ZERO EVENT GATE PASS',
    proof: 'recibos · evidência · descarga explícita · flags EISA',
    boundary: 'Fronteira da alegação',
    boundaryCopy: 'Estas são categorias de evidência computacional implementadas por stdlib/epistemic/zero_event.sio. Um recibo não estabelece uma causa biológica, clínica, física ou metafísica, e o gate não afirma que todo zero escalar exige esta representação.',
  },
};

export default function ZeroProvenanceInstrument({ locale = 'en' }: Props) {
  const language = locale === 'pt' ? 'pt' : 'en';
  const d = copy[language];

  return (
    <section className="zp-section" id="zero-provenance" aria-labelledby="zp-title">
      <div className="zp-atmosphere" aria-hidden="true" />
      <div className="zp-inner">
        <header className="zp-header">
          <div>
            <p className="zp-eyebrow">{d.eyebrow}</p>
            <h2 id="zp-title">{d.title}</h2>
          </div>
          <div className="zp-intro">
            <img src="/assets/stamps/stamp_monochrome_on_navy.png" alt="" />
            <p>{d.body}</p>
            <div className="zp-links">
              <a href="https://github.com/Sounio-lang/sounio/blob/main/stdlib/epistemic/zero_event.sio">{d.source} ↗</a>
              <a href="https://github.com/Sounio-lang/sounio/blob/main/scripts/ci/zero_event_gate.sh">{d.gate} ↗</a>
            </div>
          </div>
        </header>

        <div className="zp-instrument">
          <ZeroEventObservatory
            locale={language}
            traceLabel={d.traceLabel}
            traceTitle={d.traceTitle}
            convergedLabel={d.convergedLabel}
          />
          <div className="zp-receipt" aria-label={d.receipt}>
            <span>{d.receipt}</span>
            <strong>{d.pass}</strong>
            <code>{d.proof}</code>
          </div>
        </div>

        <footer className="zp-boundary">
          <strong>{d.boundary}</strong>
          <span>{d.boundaryCopy}</span>
          <code>bash scripts/ci/zero_event_gate.sh</code>
        </footer>
      </div>
    </section>
  );
}
