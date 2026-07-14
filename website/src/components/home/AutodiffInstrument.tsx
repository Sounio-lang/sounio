import { useEffect, useMemo, useRef, useState } from 'react';
import './AutodiffInstrument.css';

type Locale = 'en' | 'pt' | 'es' | 'el' | 'zh' | 'zh-hk' | 'ja';

interface Props {
  locale?: Locale;
}

type Specimen = {
  id: string;
  short: string;
  expression: string;
  derivative: string;
  x: number;
  value: number;
  dot: number;
  tolerance: string;
  domain: [number, number];
  evaluate: (x: number) => number;
};

const specimens: Specimen[] = [
  {
    id: 'T1',
    short: 'x^2',
    expression: 'f(x) = x^2',
    derivative: "f'(x) = 2x",
    x: 3,
    value: 9,
    dot: 6,
    tolerance: '< 0.001',
    domain: [-1, 4],
    evaluate: (x) => x * x,
  },
  {
    id: 'T2',
    short: 'x^3',
    expression: 'f(x) = x^3',
    derivative: "f'(x) = 3x^2",
    x: 2,
    value: 8,
    dot: 12,
    tolerance: '< 0.001',
    domain: [-2.2, 2.5],
    evaluate: (x) => x * x * x,
  },
  {
    id: 'T3',
    short: 'poly',
    expression: 'f(x) = x^2 + 2x + 1',
    derivative: "f'(x) = 2x + 2",
    x: 1,
    value: 4,
    dot: 4,
    tolerance: '< 0.001',
    domain: [-2.5, 3],
    evaluate: (x) => x * x + 2 * x + 1,
  },
  {
    id: 'T4',
    short: 'exp',
    expression: 'f(x) = exp(x)',
    derivative: "f'(x) = exp(x)",
    x: 0,
    value: 1,
    dot: 1,
    tolerance: '< 0.001',
    domain: [-2.5, 1.5],
    evaluate: (x) => Math.exp(x),
  },
  {
    id: 'T5',
    short: 'sin',
    expression: 'f(x) = sin(x)',
    derivative: "f'(x) = cos(x)",
    x: 0,
    value: 0,
    dot: 1,
    tolerance: '< 0.001',
    domain: [-Math.PI, Math.PI],
    evaluate: (x) => Math.sin(x),
  },
  {
    id: 'T6',
    short: 'chain',
    expression: 'f(x) = exp(x^2)',
    derivative: "f'(x) = 2x exp(x^2)",
    x: 1,
    value: Math.E,
    dot: 2 * Math.E,
    tolerance: '< 0.01',
    domain: [-1.35, 1.35],
    evaluate: (x) => Math.exp(x * x),
  },
];

const copy = {
  en: {
    eyebrow: 'EXECUTABLE LANGUAGE WITNESS / FORWARD MODE',
    heading: 'Every value has a tangent.',
    body: 'A dual number carries the value and its derivative through the same computation. Select one of the six coordinates exercised by the current run-pass fixture.',
    source: 'Open the source',
    input: 'seed',
    output: 'result',
    value: 'value lane',
    tangent: 'tangent lane',
    point: 'checked coordinate',
    tolerance: 'tolerance',
    receipt: 'CURRENT MADAROS RECEIPT',
    boundary: 'Claim boundary',
    boundaryText: 'This proves forward dual-number differentiation in this self-contained Sounio fixture. It does not claim compiler-native AD or the currently unstable imported stdlib path.',
  },
  pt: {
    eyebrow: 'TESTEMUNHO EXECUTÁVEL / MODO FORWARD',
    heading: 'Todo valor tem uma tangente.',
    body: 'Um número dual carrega o valor e sua derivada pela mesma computação. Selecione uma das seis coordenadas exercitadas pelo fixture run-pass atual.',
    source: 'Abrir o codigo-fonte',
    input: 'semente',
    output: 'resultado',
    value: 'trilha de valor',
    tangent: 'trilha tangente',
    point: 'coordenada verificada',
    tolerance: 'tolerância',
    receipt: 'RECIBO ATUAL DO MADAROS',
    boundary: 'Fronteira da afirmação',
    boundaryText: 'Isto prova diferenciação forward por números duais neste fixture autocontido em Sounio. Não afirma AD nativa do compilador nem estabilidade da rota importada do stdlib.',
  },
};

function formatNumber(value: number) {
  if (Math.abs(value) < 0.0000001) return '0.000000';
  return value.toFixed(6);
}

function drawInstrument(canvas: HTMLCanvasElement, specimen: Specimen) {
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

  const inset = { top: 24, right: 22, bottom: 34, left: 42 };
  const plotWidth = width - inset.left - inset.right;
  const plotHeight = height - inset.top - inset.bottom;
  const [xMin, xMax] = specimen.domain;
  const sampleCount = 180;
  const points = Array.from({ length: sampleCount + 1 }, (_, index) => {
    const x = xMin + ((xMax - xMin) * index) / sampleCount;
    return { x, y: specimen.evaluate(x) };
  });
  const tangent = (x: number) => specimen.value + specimen.dot * (x - specimen.x);
  const ys = points.flatMap(({ x, y }) => [y, tangent(x)]).filter(Number.isFinite);
  let yMin = Math.min(...ys);
  let yMax = Math.max(...ys);
  const yPad = Math.max(0.4, (yMax - yMin) * 0.12);
  yMin -= yPad;
  yMax += yPad;

  const mapX = (x: number) => inset.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
  const mapY = (y: number) => inset.top + (1 - (y - yMin) / (yMax - yMin)) * plotHeight;

  context.lineWidth = 1;
  context.strokeStyle = 'rgba(235, 214, 162, 0.12)';
  for (let i = 0; i <= 6; i += 1) {
    const x = inset.left + (plotWidth * i) / 6;
    context.beginPath();
    context.moveTo(x, inset.top);
    context.lineTo(x, inset.top + plotHeight);
    context.stroke();
  }
  for (let i = 0; i <= 4; i += 1) {
    const y = inset.top + (plotHeight * i) / 4;
    context.beginPath();
    context.moveTo(inset.left, y);
    context.lineTo(inset.left + plotWidth, y);
    context.stroke();
  }

  context.strokeStyle = 'rgba(235, 214, 162, 0.36)';
  if (xMin <= 0 && xMax >= 0) {
    context.beginPath();
    context.moveTo(mapX(0), inset.top);
    context.lineTo(mapX(0), inset.top + plotHeight);
    context.stroke();
  }
  if (yMin <= 0 && yMax >= 0) {
    context.beginPath();
    context.moveTo(inset.left, mapY(0));
    context.lineTo(inset.left + plotWidth, mapY(0));
    context.stroke();
  }

  context.strokeStyle = '#73d5dc';
  context.lineWidth = 2;
  context.beginPath();
  points.forEach((point, index) => {
    const x = mapX(point.x);
    const y = mapY(point.y);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();

  context.strokeStyle = '#d6b35a';
  context.lineWidth = 1.5;
  context.setLineDash([7, 6]);
  context.beginPath();
  context.moveTo(mapX(xMin), mapY(tangent(xMin)));
  context.lineTo(mapX(xMax), mapY(tangent(xMax)));
  context.stroke();
  context.setLineDash([]);

  const pointX = mapX(specimen.x);
  const pointY = mapY(specimen.value);
  context.fillStyle = '#061426';
  context.strokeStyle = '#f2d88f';
  context.lineWidth = 2;
  context.beginPath();
  context.arc(pointX, pointY, 7, 0, Math.PI * 2);
  context.fill();
  context.stroke();

  context.fillStyle = 'rgba(235, 214, 162, 0.72)';
  context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
  context.fillText(`x=${specimen.x}`, Math.min(pointX + 12, width - 70), Math.max(18, pointY - 12));
}

export default function AutodiffInstrument({ locale = 'en' }: Props) {
  const [activeId, setActiveId] = useState(specimens[0].id);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const active = useMemo(
    () => specimens.find((specimen) => specimen.id === activeId) ?? specimens[0],
    [activeId],
  );
  const d = locale === 'pt' ? copy.pt : copy.en;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const draw = () => drawInstrument(canvas, active);
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [active]);

  return (
    <section className="ad-section" id="autodiff-forward" aria-labelledby="ad-title">
      <div className="ad-atmosphere" aria-hidden="true" />
      <div className="ad-inner">
        <header className="ad-header">
          <div>
            <p className="ad-eyebrow">{d.eyebrow}</p>
            <h2 id="ad-title">{d.heading}</h2>
          </div>
          <div className="ad-intro">
            <p>{d.body}</p>
            <a
              href="https://github.com/Sounio-lang/sounio/blob/website/living-observatory-20260713/tests/run-pass/autodiff_forward_basic.sio"
              target="_blank"
              rel="noreferrer"
            >
              {d.source} <span aria-hidden="true">↗</span>
            </a>
          </div>
        </header>

        <div className="ad-instrument">
          <div className="ad-specimen-tabs" role="group" aria-label="Verified derivative specimens">
            {specimens.map((specimen) => (
              <button
                key={specimen.id}
                type="button"
                aria-pressed={specimen.id === active.id}
                className={specimen.id === active.id ? 'is-active' : ''}
                onClick={() => setActiveId(specimen.id)}
              >
                <span>{specimen.id}</span>
                <code>{specimen.short}</code>
              </button>
            ))}
          </div>

          <div className="ad-workbench">
            <div className="ad-plot">
              <div className="ad-plot-head">
                <div>
                  <strong>{active.expression}</strong>
                  <span>{active.derivative}</span>
                </div>
                <div className="ad-legend" aria-label="Plot legend">
                  <span><i className="value" />{d.value}</span>
                  <span><i className="tangent" />{d.tangent}</span>
                </div>
              </div>
              <canvas
                ref={canvasRef}
                role="img"
                aria-label={`${active.expression}; at x = ${active.x}, value = ${formatNumber(active.value)}, derivative = ${formatNumber(active.dot)}`}
              />
            </div>

            <aside className="ad-ledger" aria-live="polite">
              <p className="ad-ledger-label">DUAL / {active.id}</p>
              <div className="ad-dual">
                <span>{d.input}</span>
                <code>{`{ val: ${formatNumber(active.x)}, dot: 1.000000 }`}</code>
              </div>
              <div className="ad-rail" aria-hidden="true">
                <span />
                <b>dual_{active.short}</b>
                <span />
              </div>
              <div className="ad-dual output">
                <span>{d.output}</span>
                <code>{`{ val: ${formatNumber(active.value)}, dot: ${formatNumber(active.dot)} }`}</code>
              </div>
              <dl>
                <div><dt>{d.point}</dt><dd>x = {active.x}</dd></div>
                <div><dt>{d.tolerance}</dt><dd>{active.tolerance}</dd></div>
              </dl>
              <div className="ad-pass">
                <span>{d.receipt}</span>
                <strong>{active.id} PASS</strong>
                <code>autodiff_forward_basic: ALL PASS</code>
              </div>
            </aside>
          </div>
        </div>

        <footer className="ad-boundary">
          <strong>{d.boundary}</strong>
          <p>{d.boundaryText}</p>
          <code>./bin/souc run tests/run-pass/autodiff_forward_basic.sio</code>
        </footer>
      </div>
    </section>
  );
}
