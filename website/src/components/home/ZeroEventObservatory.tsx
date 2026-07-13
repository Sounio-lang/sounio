import { useEffect, useMemo, useRef, useState } from 'react';
import './ZeroEventObservatory.css';

type Locale = 'en' | 'pt';

interface Props {
  locale: Locale;
  traceLabel: string;
  traceTitle: string;
  convergedLabel: string;
}

interface ZeroTrace {
  id: string;
  en: string;
  pt: string;
  color: string;
  witness: string;
  detailEn: string;
  detailPt: string;
}

const traces: ZeroTrace[] = [
  {
    id: 'absent',
    en: 'Absent observation',
    pt: 'Observação ausente',
    color: '#D6B35A',
    witness: 'ZE_ABSENT',
    detailEn: 'No observation was made. The zero is not a measurement.',
    detailPt: 'Nenhuma observação foi feita. O zero não é uma medição.',
  },
  {
    id: 'cancelled',
    en: 'Cancellation',
    pt: 'Cancelamento',
    color: '#EBD6A2',
    witness: 'ZE_CANCELLED',
    detailEn: 'Non-zero terms cancelled while their provenance remained distinct.',
    detailPt: 'Termos não nulos se cancelaram, mantendo proveniências distintas.',
  },
  {
    id: 'annihilated',
    en: 'Annihilation',
    pt: 'Aniquilação',
    color: '#B99B4A',
    witness: 'ZE_ANNIHILATED',
    detailEn: 'The computation produced zero through an annihilating operation.',
    detailPt: 'A computação produziu zero por uma operação aniquiladora.',
  },
  {
    id: 'resolution',
    en: 'Below resolution',
    pt: 'Abaixo da resolução',
    color: '#2BA6B3',
    witness: 'ZE_SUBRESOLUTION',
    detailEn: 'The signal exists but falls below the declared measurement resolution.',
    detailPt: 'O sinal existe, mas está abaixo da resolução de medição declarada.',
  },
  {
    id: 'rounded',
    en: 'Rounded to zero',
    pt: 'Arredondado a zero',
    color: '#7BA7B5',
    witness: 'ZE_ROUNDED',
    detailEn: 'A representable non-zero value became zero under explicit rounding.',
    detailPt: 'Um valor não nulo representável virou zero por arredondamento explícito.',
  },
  {
    id: 'gated',
    en: 'Confidence gated',
    pt: 'Bloqueado por confiança',
    color: '#C9B37A',
    witness: 'ZE_GATED',
    detailEn: 'A decision boundary suppressed a value whose confidence was insufficient.',
    detailPt: 'Um limite de decisão suprimiu um valor com confiança insuficiente.',
  },
  {
    id: 'unknown',
    en: 'Unknown origin',
    pt: 'Origem desconhecida',
    color: '#8C9AA8',
    witness: 'ZE_UNKNOWN',
    detailEn: 'The origin cannot be reconstructed, so the uncertainty remains explicit.',
    detailPt: 'A origem não pode ser reconstruída, então a incerteza permanece explícita.',
  },
];

export default function ZeroEventObservatory({ locale, traceLabel, traceTitle, convergedLabel }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameRef = useRef<number | null>(null);
  const [selectedId, setSelectedId] = useState(traces[0].id);
  const selected = useMemo(
    () => traces.find((trace) => trace.id === selectedId) ?? traces[0],
    [selectedId],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const context = canvas.getContext('2d');
    if (!context) return;

    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let startedAt = performance.now();

    const draw = (now: number) => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.max(1, Math.floor(rect.width));
      const height = Math.max(1, Math.floor(rect.height));

      if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
        canvas.width = width * ratio;
        canvas.height = height * ratio;
      }

      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, width, height);

      context.strokeStyle = 'rgba(214, 179, 90, 0.11)';
      context.lineWidth = 1;
      const gridX = Math.max(44, width / 10);
      const gridY = Math.max(42, height / 8);
      for (let x = 0; x <= width; x += gridX) {
        context.beginPath();
        context.moveTo(x, 0);
        context.lineTo(x, height);
        context.stroke();
      }
      for (let y = 0; y <= height; y += gridY) {
        context.beginPath();
        context.moveTo(0, y);
        context.lineTo(width, y);
        context.stroke();
      }

      const startX = Math.max(18, width * 0.04);
      const bendX = width * 0.68;
      const endX = width * 0.88;
      const centerY = height * 0.5;
      const spacing = Math.min(55, (height - 74) / traces.length);
      const topY = centerY - (spacing * (traces.length - 1)) / 2;
      const progress = reducedMotion ? 1 : Math.min(1, (now - startedAt) / 1250);

      traces.forEach((trace, index) => {
        const y = topY + index * spacing;
        const active = trace.id === selectedId;
        const path = new Path2D();
        path.moveTo(startX, y);
        path.lineTo(bendX, y);
        path.bezierCurveTo(width * 0.79, y, width * 0.78, centerY, endX, centerY);

        context.save();
        context.globalAlpha = active ? 1 : 0.52;
        context.strokeStyle = trace.color;
        context.lineWidth = active ? 2 : 1.25;
        context.setLineDash(active ? [] : [3, 6]);
        context.lineDashOffset = reducedMotion ? 0 : -(now / 80 + index * 5);
        context.stroke(path);
        context.restore();

        const nodes = 11;
        for (let node = 0; node < nodes; node += 1) {
          const nodeProgress = node / (nodes - 1);
          const x = startX + (bendX - startX) * nodeProgress;
          if (nodeProgress > progress) continue;
          context.beginPath();
          context.fillStyle = trace.color;
          context.globalAlpha = active ? 0.95 : 0.55;
          context.arc(x, y, active ? 2.3 : 1.7, 0, Math.PI * 2);
          context.fill();
        }
        context.globalAlpha = 1;
      });

      context.beginPath();
      context.fillStyle = selected.color;
      context.shadowColor = selected.color;
      context.shadowBlur = 16;
      context.arc(endX, centerY, 6, 0, Math.PI * 2);
      context.fill();
      context.shadowBlur = 0;

      context.beginPath();
      context.strokeStyle = 'rgba(214, 179, 90, 0.58)';
      context.lineWidth = 1;
      context.arc(endX, centerY, 15, 0, Math.PI * 2);
      context.stroke();
      context.beginPath();
      context.strokeStyle = 'rgba(214, 179, 90, 0.22)';
      context.arc(endX, centerY, 27, 0, Math.PI * 2);
      context.stroke();

      if (!reducedMotion) {
        frameRef.current = window.requestAnimationFrame(draw);
      }
    };

    frameRef.current = window.requestAnimationFrame(draw);
    const observer = new ResizeObserver(() => {
      if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
      startedAt = performance.now() - 1250;
      frameRef.current = window.requestAnimationFrame(draw);
    });
    observer.observe(canvas);

    return () => {
      observer.disconnect();
      if (frameRef.current !== null) window.cancelAnimationFrame(frameRef.current);
    };
  }, [selectedId, selected.color]);

  return (
    <div className="zero-event-observatory">
      <div className="zero-event-head">
        <span>{traceLabel} (7)</span>
        <strong>{traceTitle}</strong>
        <span>{convergedLabel}</span>
      </div>

      <div className="zero-event-body">
        <ul className="zero-event-list" aria-label={traceLabel}>
          {traces.map((trace) => {
            const active = trace.id === selectedId;
            return (
              <li key={trace.id}>
                <button
                  type="button"
                  aria-pressed={active}
                  className={active ? 'zero-trace active' : 'zero-trace'}
                  style={{ borderLeftColor: trace.color }}
                  onClick={() => setSelectedId(trace.id)}
                  onPointerEnter={() => setSelectedId(trace.id)}
                >
                  <span>{locale === 'pt' ? trace.pt : trace.en}</span>
                  <code>{trace.witness}</code>
                </button>
              </li>
            );
          })}
        </ul>

        <div className="zero-event-canvas-wrap">
          <canvas ref={canvasRef} aria-hidden="true" />
          <div className="zero-event-value" aria-live="polite">
            <strong>0.0</strong>
            <span>{selected.witness}</span>
          </div>
        </div>
      </div>

      <div className="zero-event-detail" aria-live="polite">
        <span style={{ color: selected.color }}>{selected.witness}</span>
        <p>{locale === 'pt' ? selected.detailPt : selected.detailEn}</p>
      </div>
    </div>
  );
}
