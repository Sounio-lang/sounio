import type { Artifact } from '../evidence/types';

/** Deterministic pass fraction, displayed as counts; no confidence interval. */
export function Interval({ a, showId = true }: { a: Artifact; showId?: boolean }) {
  const p = a.proportion;
  if (!p) return null;
  return (
    <div className="iv" data-fail={a.level === 'refused' || undefined}>
      <span className="iv-n">{p.passed}<i>/</i>{p.total}</span>
      <span className="iv-text">checks passed</span>
      {showId && <span className="iv-id">{a.id}</span>}
    </div>
  );
}
