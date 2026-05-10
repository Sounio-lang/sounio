import { useEffect, useState } from 'react';

interface GumRow {
  component: string;
  source: string;
  type: string;
  u: number;
  contribution_pct: number;
}

interface GumBudget {
  rows: GumRow[];        // per-component rows (Type A / Type B sources)
  combined_u: number;    // u_c
  expanded_u95: number;  // U95 = k·u_c
}

async function fetchBudget(url: string): Promise<GumBudget> {
  const text = await fetch(url).then((r) => r.text());
  const lines = text.split('\n').filter((l) => l && !l.startsWith('#'));
  // First non-comment line is the header.
  const rows: GumRow[] = [];
  let combined_u = NaN;
  let expanded_u95 = NaN;
  for (let i = 1; i < lines.length; i++) {
    const [component, source, type, uStr, , pctStr] = lines[i].split(',');
    const u = Number(uStr);
    if (component === 'combined') { combined_u = u; continue; }
    if (component === 'expanded_k2') { expanded_u95 = u; continue; }
    rows.push({
      component,
      source,
      type,
      u,
      contribution_pct: Number(pctStr),
    });
  }
  return { rows, combined_u, expanded_u95 };
}

const SOURCE_COLOR: Record<string, string> = {
  dosing_variability: '#fbbf24',   // amber — Type B priors (literature)
  population_variation: '#60a5fa', // blue — Type A (population)
};

export function GumBudgetBar() {
  const [budget, setBudget] = useState<GumBudget | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchBudget('/dissertation/gum_budget.csv').then(setBudget).catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="text-xs text-red-300 p-2">GUM budget unavailable: {error}</div>
    );
  }
  if (!budget) {
    return <div className="text-xs opacity-60 p-2">Loading GUM budget…</div>;
  }

  return (
    <div className="text-sm">
      <div className="flex items-baseline justify-between mb-2">
        <h4 className="text-[var(--color-text-primary)] font-semibold">GUM uncertainty budget</h4>
        <span className="text-[0.7rem] opacity-60">ISO 17025 · Phase Y · n=10⁷</span>
      </div>

      {/* Stacked horizontal bar */}
      <div className="flex h-6 rounded overflow-hidden border border-white/10">
        {budget.rows.map((r) => (
          <div
            key={r.component}
            title={`${r.component} · ${r.source} (Type ${r.type}) — ${r.contribution_pct.toFixed(2)}%`}
            style={{
              width: `${r.contribution_pct}%`,
              backgroundColor: SOURCE_COLOR[r.source] ?? '#9ca3af',
            }}
            className="flex items-center justify-center text-[0.65rem] text-black/85 font-semibold"
          >
            {r.contribution_pct.toFixed(0)}%
          </div>
        ))}
      </div>

      {/* Legend */}
      <ul className="text-[0.7rem] mt-2 space-y-1">
        {budget.rows.map((r) => (
          <li key={r.component} className="flex items-center gap-2">
            <span
              className="inline-block w-2.5 h-2.5 rounded-sm"
              style={{ backgroundColor: SOURCE_COLOR[r.source] ?? '#9ca3af' }}
            />
            <span className="font-mono opacity-85">{r.component}</span>
            <span className="opacity-55">— Type {r.type}</span>
            <span className="ml-auto font-mono">{r.contribution_pct.toFixed(1)}%</span>
          </li>
        ))}
      </ul>

      <div className="mt-2 pt-2 border-t border-white/10 text-[0.75rem] font-mono">
        <div>
          <span className="opacity-65">u_c </span>
          <span>= {budget.combined_u.toFixed(4)} mg/L</span>
        </div>
        <div>
          <span className="opacity-65">U95 (k=2) </span>
          <span>= {budget.expanded_u95.toFixed(4)} mg/L</span>
        </div>
      </div>
    </div>
  );
}
