import { useEffect, useMemo, useState } from 'react';
import { InfoPopover } from './InfoPopover';

// hessian_budget.csv schema:
//   kind=ref       i=0  j=0  value=<reference AUC mean>, units=ng·h/mL
//   kind=mean_corr i=0  j=0  value=<Hessian-corrected mean>
//   kind=var_o1   i=0  j=0  value=<1st-order GUM variance>
//   kind=var_o2   i=0  j=0  value=<2nd-order GUM variance>
//   kind=sens     i=<param> j=0  value=<∂AUC/∂param>
//   kind=hess     i=<param> j=<param> value=<∂²AUC/∂param∂param>
//
// The two scanned parameters in the Lane 8a budget are CL_hep and fu_plasma.

const PARAM_LABELS = ['CL_hep', 'fu_plasma'] as const;
type Param = (typeof PARAM_LABELS)[number];

interface HessianBudget {
  ref: number;
  meanCorrected: number;
  varO1: number;
  varO2: number;
  sens: Partial<Record<Param, number>>;
  hess: Partial<Record<`${Param}_${Param}`, number>>;
}

async function fetchHessian(url: string): Promise<HessianBudget> {
  const text = await fetch(url).then((r) => r.text());
  const lines = text.split('\n').filter((l) => l && !l.startsWith('#'));
  const out: HessianBudget = {
    ref: NaN,
    meanCorrected: NaN,
    varO1: NaN,
    varO2: NaN,
    sens: {},
    hess: {},
  };
  for (let n = 1; n < lines.length; n++) {
    const [kind, iStr, jStr, valueStr] = lines[n].split(',');
    const i = Math.round(Number(iStr));
    const j = Math.round(Number(jStr));
    const v = Number(valueStr);
    if (kind === 'ref') out.ref = v;
    else if (kind === 'mean_corr') out.meanCorrected = v;
    else if (kind === 'var_o1') out.varO1 = v;
    else if (kind === 'var_o2') out.varO2 = v;
    else if (kind === 'sens' && PARAM_LABELS[i]) out.sens[PARAM_LABELS[i]] = v;
    else if (kind === 'hess' && PARAM_LABELS[i] && PARAM_LABELS[j])
      out.hess[`${PARAM_LABELS[i]}_${PARAM_LABELS[j]}`] = v;
  }
  return out;
}

interface HessianHeatmapProps {
  /** Which compartment / parameter the user has selected upstream (highlights the matching row/col). */
  highlightedParam?: Param;
  /** Called when the user clicks a Hessian cell, with one of the two param names. */
  onSelectParam?: (p: Param) => void;
}

export function HessianHeatmap({ highlightedParam, onSelectParam }: HessianHeatmapProps) {
  const [budget, setBudget] = useState<HessianBudget | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHessian('/dissertation/hessian_budget.csv').then(setBudget).catch((e) => setError(String(e)));
  }, []);

  // Colour scale on absolute Hessian magnitude.
  const cells = useMemo(() => {
    if (!budget) return [];
    const out: { i: Param; j: Param; value: number | null; pct: number }[] = [];
    let maxAbs = 0;
    for (const i of PARAM_LABELS) for (const j of PARAM_LABELS) {
      const v = budget.hess[`${i}_${j}`] ?? budget.hess[`${j}_${i}`];
      if (v !== undefined) maxAbs = Math.max(maxAbs, Math.abs(v));
    }
    for (const i of PARAM_LABELS) for (const j of PARAM_LABELS) {
      const v = budget.hess[`${i}_${j}`] ?? budget.hess[`${j}_${i}`] ?? null;
      const pct = v === null ? 0 : maxAbs === 0 ? 0 : Math.min(1, Math.abs(v) / maxAbs);
      out.push({ i, j, value: v, pct });
    }
    return out;
  }, [budget]);

  if (error) {
    return <div className="text-xs text-red-300 p-2">Hessian budget unavailable: {error}</div>;
  }
  if (!budget) {
    return <div className="text-xs opacity-60 p-2">Loading Hessian budget…</div>;
  }

  return (
    <div className="text-sm">
      <div className="flex items-baseline justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <h4 className="text-[var(--color-text-primary)] font-semibold">2nd-order Hessian</h4>
          <InfoPopover ariaLabel="What is the Hessian budget?">
            <p className="mb-1.5">
              <strong>What you're seeing.</strong> The matrix of second
              derivatives of AUC with respect to CL<sub>hep</sub> and
              f<sub>u,plasma</sub>. Diagonal cells are pure curvature; the
              off-diagonal captures parameter interaction.
            </p>
            <p>
              <strong>Why it matters.</strong> First-order GUM truncates the
              Taylor expansion; for non-linear PBPK that can <em>understate</em>{' '}
              uncertainty. Var(O₂) − Var(O₁) is the magnitude of the correction
              the second-order term adds. Per JCGM 100:2008 §F.
            </p>
          </InfoPopover>
        </div>
        <span className="text-[0.7rem] opacity-60">Lane 8a · CL × fu_plasma</span>
      </div>
      {/* Heatmap grid: 3×3 with header row + header col */}
      <div className="grid grid-cols-[60px_1fr_1fr] grid-rows-[20px_1fr_1fr] gap-px text-[0.7rem]">
        <div />
        {PARAM_LABELS.map((p) => (
          <div key={`col-${p}`} className="text-center opacity-65">{p}</div>
        ))}
        {PARAM_LABELS.map((rowParam) => (
          <>
            <div key={`row-${rowParam}`} className="opacity-65 text-right pr-1.5 self-center">{rowParam}</div>
            {PARAM_LABELS.map((colParam) => {
              const cell = cells.find((c) => c.i === rowParam && c.j === colParam);
              const highlight =
                highlightedParam === rowParam || highlightedParam === colParam;
              return (
                <button
                  key={`cell-${rowParam}-${colParam}`}
                  type="button"
                  onClick={() => onSelectParam?.(colParam)}
                  className="aspect-[2/1] flex items-center justify-center font-mono rounded-sm transition-all"
                  style={{
                    background: `rgba(244,114,182,${cell?.pct ?? 0})`,
                    border: highlight ? '1px solid #fbbf24' : '1px solid rgba(255,255,255,0.08)',
                  }}
                  title={cell?.value === null ? 'no data' : `${cell?.value.toExponential(2)}`}
                >
                  {cell?.value === null ? '—' : cell?.value.toExponential(1)}
                </button>
              );
            })}
          </>
        ))}
      </div>

      <div className="mt-3 pt-2 border-t border-white/10 text-[0.72rem] font-mono space-y-0.5">
        <div><span className="opacity-65">AUC ref          </span>= {budget.ref.toFixed(4)} ng·h/mL</div>
        <div><span className="opacity-65">AUC (Hess-corr)  </span>= {budget.meanCorrected.toFixed(4)} ng·h/mL</div>
        <div><span className="opacity-65">Var O(1)         </span>= {budget.varO1.toExponential(2)} (ng·h/mL)²</div>
        <div><span className="opacity-65">Var O(2)         </span>= {budget.varO2.toExponential(2)} (ng·h/mL)²</div>
      </div>
    </div>
  );
}
