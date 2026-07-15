import { useState, type KeyboardEvent } from 'react';
import './RenderAtelier.css';

export type RenderReceipt = {
  assetFile: string;
  assetPath: string;
  publicPath: string;
  example: string;
  command: string;
  width: number;
  height: number;
  sourceHref: string;
  title: string;
  body: string;
};

interface Props {
  locale?: string;
  receipts: RenderReceipt[];
  compilerArtifact: string;
  registrySize: number;
}

const copy = {
  en: {
    eyebrow: 'SOUNIO RENDER ATELIER / MANIFESTED OUTPUT',
    heading: 'Images with provenance.',
    body: 'Not decorative science graphics. Select a checked-in render and inspect the Sounio source, exact command, dimensions, and manifest path attached to it.',
    manifest: 'generated manifest', compiler: 'compiler entrypoint', registry: 'registered renders',
    source: 'source program', command: 'render command', dimensions: 'raster dimensions', asset: 'manifest asset',
    inspect: 'Inspect the Sounio source', frame: 'active compiled visual receipt', status: 'CHECKED-IN RENDER',
    boundary: 'Generated artifact boundary',
    boundaryText: 'This atelier displays checked-in render assets and their manifest records. It does not claim that the current default compiler regenerated them during this page build; the current quality gate retained them after an explicit render-check skip.',
    buildState: 'render check: explicit SKIP · pre-rendered assets retained',
  },
  pt: {
    eyebrow: 'ATELIER DE RENDER SOUNIO / SAÍDA MANIFESTADA',
    heading: 'Imagens com proveniência.',
    body: 'Não são gráficos científicos decorativos. Selecione um render checked-in e inspecione o código Sounio, comando exato, dimensões e caminho de manifesto ligados a ele.',
    manifest: 'manifesto gerado', compiler: 'entrada do compilador', registry: 'renders registrados',
    source: 'programa fonte', command: 'comando de render', dimensions: 'dimensões raster', asset: 'asset no manifesto',
    inspect: 'Inspecionar o código Sounio', frame: 'recibo visual compilado ativo', status: 'RENDER CHECKED-IN',
    boundary: 'Fronteira do artefato gerado',
    boundaryText: 'Este atelier exibe assets de render checked-in e seus registros no manifesto. Ele não afirma que o compilador padrão atual os regenerou durante este build; o gate de qualidade os preservou após um render-check explicitamente ignorado.',
    buildState: 'render check: SKIP explícito · assets pré-renderizados preservados',
  },
};

export default function RenderAtelier({ locale = 'en', receipts, compilerArtifact, registrySize }: Props) {
  const [selected, setSelected] = useState(0);
  const d = copy[locale === 'pt' ? 'pt' : 'en'];
  const active = receipts[selected];

  const selectFromKeyboard = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % receipts.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + receipts.length) % receipts.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = receipts.length - 1;
    else return;
    event.preventDefault();
    setSelected(next);
    document.getElementById(`ra-tab-${next}`)?.focus();
  };

  return (
    <section className="ra-section" id="render-atelier" aria-labelledby="ra-title">
      <div className="ra-inner">
        <header className="ra-header">
          <div className="ra-title-block">
            <p className="ra-eyebrow">{d.eyebrow}</p>
            <h2 id="ra-title">{d.heading}</h2>
          </div>
          <div className="ra-intro">
            <img src="/assets/stamps/stamp_monochrome_on_navy.png" alt="" aria-hidden="true" />
            <p>{d.body}</p>
            <dl>
              <div><dt>{d.manifest}</dt><dd><code>render/manifest.json</code></dd></div>
              <div><dt>{d.compiler}</dt><dd><code>{compilerArtifact}</code></dd></div>
              <div><dt>{d.registry}</dt><dd>{registrySize}</dd></div>
            </dl>
          </div>
        </header>

        <div className="ra-atelier">
          <div className="ra-tabs" role="tablist" aria-label="Compiled visual receipts">
            {receipts.map((receipt, index) => (
              <button
                type="button"
                role="tab"
                id={`ra-tab-${index}`}
                aria-controls="ra-panel"
                aria-selected={selected === index}
                tabIndex={selected === index ? 0 : -1}
                key={receipt.assetFile}
                onClick={() => setSelected(index)}
                onKeyDown={(event) => selectFromKeyboard(event, index)}
              >
                <span>{String(index + 1).padStart(2, '0')}</span>
                <strong>{receipt.title}</strong>
                <small>{receipt.width} × {receipt.height}</small>
              </button>
            ))}
          </div>

          <div className="ra-stage" id="ra-panel" role="tabpanel" aria-labelledby={`ra-tab-${selected}`}>
            <figure className="ra-render">
              <div className="ra-frame-label"><span>{d.frame}</span><code>{active.assetFile}</code></div>
              <div className={`ra-canvas ${active.width > active.height ? 'is-wide' : 'is-square'}`}>
                <img src={active.publicPath} alt={`${active.title}. ${active.body}`} width={active.width} height={active.height} />
              </div>
              <figcaption><span>{d.status}</span><code>{active.width} × {active.height} px</code></figcaption>
            </figure>

            <aside className="ra-receipt">
              <div className="ra-receipt-heading"><span>{String(selected + 1).padStart(2, '0')}</span><h3>{active.title}</h3></div>
              <p>{active.body}</p>
              <dl>
                <div><dt>{d.source}</dt><dd><code>{active.example}</code></dd></div>
                <div><dt>{d.command}</dt><dd><code>{active.command}</code></dd></div>
                <div><dt>{d.dimensions}</dt><dd><code>{active.width} × {active.height}</code></dd></div>
                <div><dt>{d.asset}</dt><dd><code>{active.assetPath}</code></dd></div>
              </dl>
              <a href={active.sourceHref} target="_blank" rel="noreferrer">{d.inspect} <span aria-hidden="true">↗</span></a>
              <div className="ra-build-state"><span>{d.buildState}</span><code>npm run check:render-assets</code></div>
            </aside>
          </div>
        </div>

        <footer className="ra-boundary">
          <strong>{d.boundary}</strong><span>{d.boundaryText}</span><code>website/public/assets/generated/render/manifest.json</code>
        </footer>
      </div>
    </section>
  );
}
