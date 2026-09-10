import type { Artifact, Proportion } from '../evidence/types';
import { EVIDENCE_STATS } from '../evidence/claim';

const { widest, narrowest } = EVIDENCE_STATS.proportions;

/** 0 = a amostra restringe · 1 = a amostra mal restringe. */
export function tension(p: Proportion): number {
  const span = widest - narrowest;
  return span > 0 ? (p.width - narrowest) / span : 0;
}

/**
 * O que a amostra sustenta, desenhado.
 *
 * A cor vem da largura, não do veredito — verde é medição que restringe,
 * âmbar é medição que mal restringe. A interpolação é feita em CSS com
 * color-mix, para acompanhar o tema sem recalcular nada em JS.
 */
export function Interval({ a, showId = true }: { a: Artifact; showId?: boolean }) {
  const p = a.proportion;
  if (!p) return null;

  const failed = a.level === 'refused';
  // --t vive no contêiner: a banda, o ponto e o texto do intervalo leem dele.
  const container = { '--t': tension(p) } as React.CSSProperties;

  return (
    <div className="iv" data-fail={failed || undefined} style={container}>
      <span className="iv-n">{p.passed}<i>/</i>{p.total}</span>
      <span className="iv-track">
        <span className="iv-band"
              style={{ left: `${p.lo * 100}%`,
                       width: `${Math.max(p.width * 100, 0.35)}%` }} />
        <span className="iv-point" style={{ left: `${p.point * 100}%` }} />
      </span>
      <span className="iv-text">[{p.lo.toFixed(3)}, {p.hi.toFixed(3)}]</span>
      {showId && <span className="iv-id">{a.id}</span>}
    </div>
  );
}
