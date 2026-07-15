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
  sourceRef?: string;
  sourceAsset?: string;
  gateHref?: string;
  engine?: string;
  sha256?: string;
  renderSha256?: string;
  verification?: string;
  gate?: string;
  receipt?: string;
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
    sourceLayer: 'Source', artifactLayer: 'Artifact', gateLayer: 'Gate', sourceRef: 'source ref', sourceAsset: 'checked-in image',
    engine: 'verified engine', integrity: 'receipt sha-256', renderIntegrity: 'PPM sha-256', verification: 'determinism', gate: 'executable gate', receipt: 'pass receipt',
    inspect: 'Open source', inspectGate: 'Open gate', copyHash: 'Copy hash', copied: 'Copied', frame: 'active compiled visual receipt', status: 'CHECKED-IN RENDER',
    boundary: 'Generated artifact boundary',
    boundaryText: 'Verified receipts regenerate with their declared engine, reproduce the recorded PPM hash, and execute their gate during the website check. Historical receipts remain checked-in assets and retain an explicit fallback boundary.',
    buildState: 'render check: explicit SKIP · pre-rendered assets retained',
    verifiedState: 'lean_single verified · deterministic receipt',
  },
  pt: {
    eyebrow: 'ATELIER DE RENDER SOUNIO / SAÍDA MANIFESTADA',
    heading: 'Imagens com proveniência.',
    body: 'Não são gráficos científicos decorativos. Selecione um render checked-in e inspecione o código Sounio, comando exato, dimensões e caminho de manifesto ligados a ele.',
    manifest: 'manifesto gerado', compiler: 'entrada do compilador', registry: 'renders registrados',
    source: 'programa fonte', command: 'comando de render', dimensions: 'dimensões raster', asset: 'asset no manifesto',
    sourceLayer: 'Fonte', artifactLayer: 'Artefato', gateLayer: 'Gate', sourceRef: 'referência da fonte', sourceAsset: 'imagem checked-in',
    engine: 'engine verificado', integrity: 'sha-256 do recibo', renderIntegrity: 'sha-256 do PPM', verification: 'determinismo', gate: 'gate executável', receipt: 'recibo de aprovação',
    inspect: 'Abrir fonte', inspectGate: 'Abrir gate', copyHash: 'Copiar hash', copied: 'Copiado', frame: 'recibo visual compilado ativo', status: 'RENDER CHECKED-IN',
    boundary: 'Fronteira do artefato gerado',
    boundaryText: 'Recibos verificados são regenerados com o engine declarado, reproduzem o hash PPM registrado e executam seu gate durante o check do website. Recibos históricos continuam como assets checked-in e preservam uma fronteira explícita de fallback.',
    buildState: 'render check: SKIP explícito · assets pré-renderizados preservados',
    verifiedState: 'lean_single verificado · recibo determinístico',
  },
};

export default function RenderAtelier({ locale = 'en', receipts, compilerArtifact, registrySize }: Props) {
  const [selected, setSelected] = useState(0);
  const [copied, setCopied] = useState<string | null>(null);
  const d = copy[locale === 'pt' ? 'pt' : 'en'];
  const active = receipts[selected];

  const copyHash = async (kind: string, value: string) => {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      const field = document.createElement('textarea');
      field.value = value;
      field.style.position = 'fixed';
      field.style.opacity = '0';
      document.body.appendChild(field);
      field.select();
      (document as unknown as { execCommand: (command: string) => boolean }).execCommand('copy');
      field.remove();
    }
    setCopied(kind);
    window.setTimeout(() => setCopied((current) => current === kind ? null : current), 1600);
  };

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
              <div className={`ra-canvas ${active.width > active.height ? 'is-wide' : 'is-square'} ${active.sha256 ? 'is-verified' : ''}`}>
                <img src={active.publicPath} alt={`${active.title}. ${active.body}`} width={active.width} height={active.height} />
              </div>
              <figcaption><span>{d.status}</span><code>{active.width} × {active.height} px</code></figcaption>
            </figure>

            <aside className="ra-receipt">
              <div className="ra-receipt-heading"><span>{String(selected + 1).padStart(2, '0')}</span><h3>{active.title}</h3></div>
              <p>{active.body}</p>
              <div className="ra-chain" aria-label={locale === 'pt' ? 'Cadeia de custódia do artefato' : 'Artifact custody chain'}>
                <section className="ra-chain-step">
                  <header><span>01</span><strong>{d.sourceLayer}</strong></header>
                  <dl>
                    <div><dt>{d.source}</dt><dd><code>{active.example}</code></dd></div>
                    {active.sourceRef && <div><dt>{d.sourceRef}</dt><dd><code>{active.sourceRef}</code></dd></div>}
                    <div><dt>{d.command}</dt><dd><code>{active.command}</code></dd></div>
                  </dl>
                  <a href={active.sourceHref} target="_blank" rel="noreferrer">{d.inspect} <span aria-hidden="true">↗</span></a>
                </section>

                <section className={`ra-chain-step ${active.sha256 ? 'is-verified' : ''}`}>
                  <header><span>02</span><strong>{d.artifactLayer}</strong>{active.sha256 && <em>SHA</em>}</header>
                  <dl>
                    <div><dt>{d.dimensions}</dt><dd><code>{active.width} × {active.height}</code></dd></div>
                    <div><dt>{d.asset}</dt><dd><code>{active.assetPath}</code></dd></div>
                    {active.sourceAsset && <div><dt>{d.sourceAsset}</dt><dd><code>{active.sourceAsset}</code></dd></div>}
                    {active.sha256 && <div className="ra-hash-row"><dt>{d.integrity}</dt><dd><code>{active.sha256}</code><button type="button" aria-label={`${d.copyHash}: ${d.integrity}`} aria-live="polite" onClick={() => copyHash('asset', active.sha256!)}>{copied === 'asset' ? d.copied : d.copyHash}</button></dd></div>}
                    {active.renderSha256 && <div className="ra-hash-row"><dt>{d.renderIntegrity}</dt><dd><code>{active.renderSha256}</code><button type="button" aria-label={`${d.copyHash}: ${d.renderIntegrity}`} aria-live="polite" onClick={() => copyHash('render', active.renderSha256!)}>{copied === 'render' ? d.copied : d.copyHash}</button></dd></div>}
                  </dl>
                </section>

                <section className={`ra-chain-step ${active.receipt ? 'is-verified' : ''}`}>
                  <header><span>03</span><strong>{d.gateLayer}</strong>{active.receipt && <em>PASS</em>}</header>
                  <dl>
                    {active.engine && <div><dt>{d.engine}</dt><dd><code>{active.engine}</code></dd></div>}
                    {active.verification && <div><dt>{d.verification}</dt><dd><code>{active.verification}</code></dd></div>}
                    {active.gate && <div><dt>{d.gate}</dt><dd><code>{active.gate}</code></dd></div>}
                    {active.receipt && <div><dt>{d.receipt}</dt><dd><code className="ra-pass-receipt">{active.receipt}</code></dd></div>}
                  </dl>
                  {active.gateHref && <a href={active.gateHref} target="_blank" rel="noreferrer">{d.inspectGate} <span aria-hidden="true">↗</span></a>}
                  <div className={`ra-build-state ${active.sha256 ? 'is-verified' : ''}`}><span>{active.sha256 ? d.verifiedState : d.buildState}</span><code>{active.receipt ?? 'npm run check:render-assets'}</code></div>
                </section>
              </div>
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
