import { useState, type KeyboardEvent } from 'react';
import './SounioObservatoryStage.css';

type ObservatoryDomain = {
  file: string;
  domain: string;
  title: string;
  claim: string;
  boundary: string;
  href: string;
  publicPath: string;
  width: number;
  height: number;
  source: string;
  sourceHref: string;
  command: string;
  engine: string;
  gate: string;
  gateHref?: string;
  receipt: string;
  sha256?: string;
  renderSha256?: string;
  verification?: string;
};

interface Props {
  locale?: string;
  domains: ObservatoryDomain[];
  compilerArtifact: string;
}

const copy = {
  en: {
    eyebrow: 'SOUNIO OBSERVATORY / EXECUTABLE IMAGES',
    heading: 'The claim stops where the receipt stops.',
    intro: 'Four windows into one discipline: Sounio source becomes a deterministic artifact, the gate records what passed, and the boundary keeps the picture from saying more than the program proved.',
    specimen: 'active specimen',
    rendered: 'Sounio-rendered artifact',
    claim: 'compiled witness',
    boundary: 'where the claim stops',
    openDomain: 'Enter domain',
    openCustody: 'Open custody record',
    closeCustody: 'Close custody record',
    source: 'source',
    command: 'command',
    engine: 'engine',
    gate: 'gate',
    receipt: 'receipt',
    dimensions: 'dimensions',
    integrity: 'artifact sha-256',
    renderIntegrity: 'render sha-256',
    compiler: 'compiler entrypoint',
    sourceLink: 'Open source',
    gateLink: 'Open gate',
    custody: 'SOURCE → ARTIFACT → GATE',
    registered: '4 / 4 MANIFEST-BACKED',
  },
  pt: {
    eyebrow: 'OBSERVATÓRIO SOUNIO / IMAGENS EXECUTÁVEIS',
    heading: 'O claim termina onde o recibo termina.',
    intro: 'Quatro janelas para uma disciplina: fonte Sounio vira artefato determinístico, o gate registra o que passou e a fronteira impede que a imagem diga mais do que o programa provou.',
    specimen: 'espécime ativo',
    rendered: 'artefato renderizado por Sounio',
    claim: 'testemunho compilado',
    boundary: 'onde o claim para',
    openDomain: 'Entrar no domínio',
    openCustody: 'Abrir cadeia de custódia',
    closeCustody: 'Fechar cadeia de custódia',
    source: 'fonte',
    command: 'comando',
    engine: 'engine',
    gate: 'gate',
    receipt: 'recibo',
    dimensions: 'dimensões',
    integrity: 'sha-256 do artefato',
    renderIntegrity: 'sha-256 do render',
    compiler: 'entrada do compilador',
    sourceLink: 'Abrir fonte',
    gateLink: 'Abrir gate',
    custody: 'FONTE → ARTEFATO → GATE',
    registered: '4 / 4 NO MANIFESTO',
  },
};

export default function SounioObservatoryStage({ locale = 'en', domains, compilerArtifact }: Props) {
  const [selected, setSelected] = useState(0);
  const [custodyOpen, setCustodyOpen] = useState(false);
  const d = copy[locale === 'pt' ? 'pt' : 'en'];
  const active = domains[selected];

  const select = (index: number) => {
    setSelected(index);
    setCustodyOpen(false);
  };

  const selectFromKeyboard = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next = index;
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') next = (index + 1) % domains.length;
    else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') next = (index - 1 + domains.length) % domains.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = domains.length - 1;
    else return;
    event.preventDefault();
    select(next);
    document.getElementById(`so-tab-${next}`)?.focus();
  };

  return (
    <section className="so-section" id="sounio-observatory" aria-labelledby="so-title">
      <div className="so-shell">
        <header className="so-header">
          <div className="so-brand">
            <img src="/assets/stamps/stamp_monochrome_on_navy.png" alt="" aria-hidden="true" width="92" height="92" />
            <p>{d.eyebrow}</p>
          </div>
          <h2 id="so-title">{d.heading}</h2>
          <div className="so-intro"><p>{d.intro}</p><span>{d.registered}</span></div>
        </header>

        <div className="so-observatory">
          <div className="so-domain-rail" role="tablist" aria-label="Sounio observatory domains">
            {domains.map((domain, index) => (
              <button
                type="button"
                role="tab"
                id={`so-tab-${index}`}
                aria-controls="so-panel"
                aria-selected={selected === index}
                tabIndex={selected === index ? 0 : -1}
                key={domain.file}
                onClick={() => select(index)}
                onKeyDown={(event) => selectFromKeyboard(event, index)}
              >
                <span>{String(index + 1).padStart(2, '0')}</span>
                <strong>{domain.domain}</strong>
                <small>{domain.receipt ? 'PASS' : 'RECORDED'}</small>
              </button>
            ))}
          </div>

          <div className="so-stage" id="so-panel" role="tabpanel" aria-labelledby={`so-tab-${selected}`}>
            <figure className="so-figure">
              <div className="so-frame-meta"><span>{d.specimen}</span><code>{active.file}</code></div>
              <div className="so-image-field">
                <img src={active.publicPath} alt={`${active.title} ${active.claim}`} width={active.width} height={active.height} />
              </div>
              <figcaption><span>{d.rendered}</span><code>{active.width} × {active.height} px</code></figcaption>
            </figure>

            <aside className="so-narrative">
              <div className="so-domain-index"><span>0{selected + 1}</span><strong>{active.domain}</strong></div>
              <h3>{active.title}</h3>
              <dl>
                <div><dt>{d.claim}</dt><dd>{active.claim}</dd></div>
                <div className="so-boundary"><dt>{d.boundary}</dt><dd>{active.boundary}</dd></div>
              </dl>
              <a href={active.href}>{d.openDomain}<span aria-hidden="true">↗</span></a>
            </aside>
          </div>

          <button
            type="button"
            className="so-custody-toggle"
            aria-expanded={custodyOpen}
            aria-controls="so-custody"
            onClick={() => setCustodyOpen((open) => !open)}
          >
            <span>{d.custody}</span>
            <strong>{custodyOpen ? d.closeCustody : d.openCustody}</strong>
            <b aria-hidden="true">{custodyOpen ? '×' : '+'}</b>
          </button>

          <div className="so-custody" id="so-custody" hidden={!custodyOpen}>
            <div className="so-custody-column">
              <span>01 / {d.source}</span>
              <dl>
                <div><dt>{d.source}</dt><dd><code>{active.source}</code></dd></div>
                <div><dt>{d.command}</dt><dd><code>{active.command}</code></dd></div>
                <div><dt>{d.compiler}</dt><dd><code>{compilerArtifact}</code></dd></div>
              </dl>
              <a href={active.sourceHref} target="_blank" rel="noreferrer">{d.sourceLink}<span aria-hidden="true">↗</span></a>
            </div>
            <div className="so-custody-column">
              <span>02 / artifact</span>
              <dl>
                <div><dt>{d.dimensions}</dt><dd><code>{active.width} × {active.height}</code></dd></div>
                {active.sha256 && <div><dt>{d.integrity}</dt><dd><code>{active.sha256}</code></dd></div>}
                {active.renderSha256 && <div><dt>{d.renderIntegrity}</dt><dd><code>{active.renderSha256}</code></dd></div>}
              </dl>
            </div>
            <div className="so-custody-column is-gate">
              <span>03 / {d.gate}</span>
              <dl>
                <div><dt>{d.engine}</dt><dd><code>{active.engine}</code></dd></div>
                <div><dt>{d.gate}</dt><dd><code>{active.gate}</code></dd></div>
                <div><dt>{d.receipt}</dt><dd><code>{active.receipt}</code></dd></div>
              </dl>
              {active.gateHref && <a href={active.gateHref} target="_blank" rel="noreferrer">{d.gateLink}<span aria-hidden="true">↗</span></a>}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
