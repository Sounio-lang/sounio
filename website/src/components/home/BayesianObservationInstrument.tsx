import { useEffect, useRef, useState } from 'react';
import './BayesianObservationInstrument.css';

type Locale = 'en' | 'pt' | 'es' | 'el' | 'zh' | 'zh-hk' | 'ja';

interface Props {
  locale?: Locale;
}

type Distribution = {
  id: 'prior' | 'measurement' | 'posterior';
  mean: number;
  variance: number;
  color: string;
  dash: number[];
};

const distributions: Distribution[] = [
  { id: 'prior', mean: 37, variance: 0.04, color: '#d6b35a', dash: [] },
  { id: 'measurement', mean: 37.2, variance: 0.25, color: '#a8bbc9', dash: [7, 6] },
  { id: 'posterior', mean: 37.027586, variance: 0.034482, color: '#73d5dc', dash: [] },
];

const copy = {
  en: {
    eyebrow: 'EXECUTABLE CALCULATION / NORMAL-NORMAL UPDATE',
    heading: 'This observation changes the shape of doubt.',
    body: 'A self-contained Sounio program combines a Normal prior with an illustrative thermometer measurement. Step through the fixed fixture and watch the posterior move between them while its variance contracts.',
    source: 'Open the source',
    stages: ['Prior', 'Measurement', 'Posterior'],
    stageCopy: [
      'The fixture begins with mean 37.0 and variance 0.04.',
      'A reading of 37.2 enters with variance 0.25.',
      'The emitted posterior mean is 37.027586 and variance is 0.034482.',
    ],
    mean: 'mean',
    variance: 'variance',
    standardDeviation: 'standard deviation',
    receipt: 'CURRENT MADAROS RECEIPT',
    boundary: 'Claim boundary',
    boundaryText: 'Illustrative scalar normal-normal update authored directly over f64. The fixture checks one example and three inequalities. It is not automatic compiler inference, a general Bayesian proof, Observe-effect semantics, or a clinical claim.',
    prior: 'prior',
    measurement: 'measurement',
    posterior: 'posterior',
  },
  pt: {
    eyebrow: 'CÁLCULO EXECUTÁVEL / UPDATE NORMAL-NORMAL',
    heading: 'Esta observação muda a forma da dúvida.',
    body: 'Um programa Sounio autocontido combina uma priori Normal com uma medição ilustrativa de termômetro. Percorra o fixture fixo e veja a posterior mover-se entre as duas enquanto sua variância contrai.',
    source: 'Abrir o código-fonte',
    stages: ['Priori', 'Medição', 'Posterior'],
    stageCopy: [
      'O fixture começa com média 37,0 e variância 0,04.',
      'Uma leitura de 37,2 entra com variância 0,25.',
      'A média posterior emitida é 37,027586 e a variância é 0,034482.',
    ],
    mean: 'média',
    variance: 'variância',
    standardDeviation: 'desvio padrão',
    receipt: 'RECIBO ATUAL DO MADAROS',
    boundary: 'Fronteira da afirmação',
    boundaryText: 'Update normal-normal escalar e ilustrativo, escrito diretamente sobre f64. O fixture verifica um exemplo e três desigualdades. Não é inferência automática do compilador, prova Bayesiana geral, semântica do efeito Observe ou alegação clínica.',
    prior: 'priori',
    measurement: 'medição',
    posterior: 'posterior',
  },
};

function density(x: number, mean: number, variance: number) {
  const denominator = Math.sqrt(2 * Math.PI * variance);
  return Math.exp(-((x - mean) ** 2) / (2 * variance)) / denominator;
}

function labelFor(id: Distribution['id'], d: typeof copy.en) {
  if (id === 'prior') return d.prior;
  if (id === 'measurement') return d.measurement;
  return d.posterior;
}

function drawDistributions(
  canvas: HTMLCanvasElement,
  stage: number,
  d: typeof copy.en,
) {
  const context = canvas.getContext('2d');
  if (!context) return;

  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);

  const inset = { top: 44, right: 24, bottom: 48, left: 48 };
  const plotWidth = width - inset.left - inset.right;
  const plotHeight = height - inset.top - inset.bottom;
  const xMin = 36.1;
  const xMax = 38.1;
  const yMax = 2.4;
  const mapX = (x: number) => inset.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
  const mapY = (y: number) => inset.top + (1 - y / yMax) * plotHeight;

  context.strokeStyle = 'rgba(235, 214, 162, 0.13)';
  context.lineWidth = 1;
  [36.5, 37, 37.5, 38].forEach((tick) => {
    const x = mapX(tick);
    context.beginPath();
    context.moveTo(x, inset.top);
    context.lineTo(x, inset.top + plotHeight);
    context.stroke();
    context.fillStyle = 'rgba(235, 214, 162, 0.48)';
    context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillText(tick.toFixed(1), x - 14, height - 22);
  });
  context.beginPath();
  context.moveTo(inset.left, inset.top + plotHeight);
  context.lineTo(inset.left + plotWidth, inset.top + plotHeight);
  context.strokeStyle = 'rgba(235, 214, 162, 0.34)';
  context.stroke();

  distributions.slice(0, stage + 1).forEach((distribution, index) => {
    const points = Array.from({ length: 220 }, (_, pointIndex) => {
      const x = xMin + ((xMax - xMin) * pointIndex) / 219;
      return { x, y: density(x, distribution.mean, distribution.variance) };
    });
    const isCurrent = index === stage;

    context.save();
    context.globalAlpha = isCurrent ? 1 : 0.5;
    context.fillStyle = `${distribution.color}16`;
    context.beginPath();
    context.moveTo(mapX(xMin), mapY(0));
    points.forEach((point) => context.lineTo(mapX(point.x), mapY(point.y)));
    context.lineTo(mapX(xMax), mapY(0));
    context.closePath();
    context.fill();

    context.strokeStyle = distribution.color;
    context.lineWidth = isCurrent ? 2.5 : 1.5;
    context.setLineDash(distribution.dash);
    context.beginPath();
    points.forEach((point, pointIndex) => {
      if (pointIndex === 0) context.moveTo(mapX(point.x), mapY(point.y));
      else context.lineTo(mapX(point.x), mapY(point.y));
    });
    context.stroke();
    context.setLineDash([]);

    const peakX = mapX(distribution.mean);
    const peakY = mapY(density(distribution.mean, distribution.mean, distribution.variance));
    context.fillStyle = distribution.color;
    context.font = `${isCurrent ? '600 ' : ''}11px ui-monospace, SFMono-Regular, Menlo, monospace`;
    const label = labelFor(distribution.id, d);
    const labelWidth = context.measureText(label).width;
    const crowdedPeakOffset = stage === 2 && distribution.id !== 'measurement'
      ? distribution.id === 'prior' ? -labelWidth - 8 : 8
      : -labelWidth / 2;
    const labelX = Math.min(Math.max(inset.left, peakX + crowdedPeakOffset), width - inset.right - labelWidth);
    context.fillText(label, labelX, Math.max(16, peakY - 13));
    context.restore();
  });
}

export default function BayesianObservationInstrument({ locale = 'en' }: Props) {
  const [stage, setStage] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const d = locale === 'pt' ? copy.pt : copy.en;
  const active = distributions[stage];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => drawDistributions(canvas, stage, d);
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [stage, d]);

  return (
    <section className="bo-section" id="bayesian-observation" aria-labelledby="bo-title">
      <div className="bo-atmosphere" aria-hidden="true" />
      <div className="bo-inner">
        <header className="bo-header">
          <div className="bo-heading-wrap">
            <p className="bo-eyebrow">{d.eyebrow}</p>
            <h2 id="bo-title">{d.heading}</h2>
          </div>
          <div className="bo-intro">
            <img src="/assets/stamps/stamp_monochrome_on_navy.png" alt="" aria-hidden="true" />
            <p>{d.body}</p>
            <a
              href="https://github.com/Sounio-lang/sounio/blob/website/living-observatory-20260713/tests/run-pass/bayesian_observe.sio"
              target="_blank"
              rel="noreferrer"
            >
              {d.source} <span aria-hidden="true">↗</span>
            </a>
          </div>
        </header>

        <div className="bo-instrument">
          <div className="bo-stages" role="group" aria-label="Bayesian update stages">
            {d.stages.map((label, index) => (
              <button
                type="button"
                key={label}
                aria-pressed={stage === index}
                onClick={() => setStage(index)}
              >
                <span>0{index + 1}</span>
                <strong>{label}</strong>
              </button>
            ))}
          </div>

          <div className="bo-workbench">
            <div className="bo-plot">
              <div className="bo-stage-copy" aria-live="polite">
                <span>{d.stages[stage]}</span>
                <p>{d.stageCopy[stage]}</p>
              </div>
              <canvas
                ref={canvasRef}
                role="img"
                aria-label={`${d.stages[stage]}: mean ${active.mean.toFixed(6)}, variance ${active.variance.toFixed(6)}. Visible curves: ${d.stages.slice(0, stage + 1).join(', ')}.`}
              />
            </div>

            <aside className="bo-ledger">
              <p className="bo-ledger-label">NORMAL / NORMAL</p>
              <dl>
                {distributions.map((distribution) => (
                  <div className={distribution.id === active.id ? 'is-active' : ''} key={distribution.id}>
                    <dt><i style={{ backgroundColor: distribution.color }} />{labelFor(distribution.id, d)}</dt>
                    <dd>
                      <span>{d.mean}<code>{distribution.mean.toFixed(6)}</code></span>
                      <span>{d.variance}<code>{distribution.variance.toFixed(6)}</code></span>
                      <span>{d.standardDeviation}<code>{Math.sqrt(distribution.variance).toFixed(6)}</code></span>
                    </dd>
                  </div>
                ))}
              </dl>
              <div className="bo-equation">
                <code>var_post = (0.04 * 0.25) / (0.04 + 0.25)</code>
                <strong>= 0.034482</strong>
              </div>
              <div className="bo-pass">
                <span>{d.receipt}</span>
                <strong>3 INVARIANTS PASS</strong>
                <code>ALL PASS: observation narrows uncertainty</code>
              </div>
            </aside>
          </div>
        </div>

        <footer className="bo-boundary">
          <strong>{d.boundary}</strong>
          <p>{d.boundaryText}</p>
          <code>./bin/souc run tests/run-pass/bayesian_observe.sio</code>
        </footer>
      </div>
    </section>
  );
}
