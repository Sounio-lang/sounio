// G-ε-2: PD readout panel. Dispatches on drug:
//   rapamycin   → A = mTORC1 active fraction (0..1), N = neointimal index
//                  (dissertation's Cypher-stent endpoint — late lumen loss
//                  at 90 d).
//   semaglutide → ΔG = plasma glucose deviation (mg/dL, A[12])
//                  ΔI = plasma insulin deviation (mU/L, N[12])
//                  (Bergman minimal model, pancreas-host index).

import type { DrugId } from '../../hooks/usePBPK';

export function PdReadoutPanel({
  drug,
  pdA,
  pdN,
}: {
  drug: DrugId;
  pdA?: Float32Array;
  pdN?: Float32Array;
}) {
  if (drug === 'semaglutide') {
    const dG = pdA?.[12] ?? 0;
    const dI = pdN?.[12] ?? 0;
    const Gb = 100;
    const Ib = 10;
    return (
      <div>
        <h3 className="text-sm font-semibold text-white mb-1">
          PD readout · <span className="text-slate-400 font-normal">glucose / insulin (Bergman)</span>
        </h3>
        <dl className="text-xs grid grid-cols-2 gap-y-1 text-slate-300">
          <dt>Plasma glucose</dt>
          <dd className="tabular-nums text-right">
            {(Gb + dG).toFixed(1)} mg/dL{' '}
            <span className={dG < 0 ? 'text-emerald-400' : 'text-slate-500'}>
              (ΔG = {dG.toFixed(2)})
            </span>
          </dd>
          <dt>Plasma insulin</dt>
          <dd className="tabular-nums text-right">
            {(Ib + dI).toFixed(2)} mU/L{' '}
            <span className={dI > 0 ? 'text-amber-300' : 'text-slate-500'}>
              (ΔI = {dI.toFixed(2)})
            </span>
          </dd>
        </dl>
        <p className="text-[10px] text-slate-500 mt-1 leading-snug">
          Linearized Bergman minimal model at G ≈ G<sub>b</sub>. Pancreatic
          GLP-1R occupancy drives insulin secretion (α=1 → 2× at full occ),
          which suppresses plasma glucose. Clinical endpoint: fasting glucose
          reduction over weekly dosing.
        </p>
      </div>
    );
  }

  // Rapamycin: mTORC1 + neointima
  const A = pdA?.[4] ?? 1;
  const N = pdN?.[4] ?? 0;
  return (
    <div>
      <h3 className="text-sm font-semibold text-white mb-1">
        PD readout · <span className="text-slate-400 font-normal">mTORC1 + neointima</span>
      </h3>
      <dl className="text-xs grid grid-cols-2 gap-y-1 text-slate-300">
        <dt>mTORC1 active fraction</dt>
        <dd className="tabular-nums text-right">
          {A.toFixed(3)}{' '}
          <span className={A < 0.9 ? 'text-emerald-400' : 'text-slate-500'}>
            ({(A * 100).toFixed(1)}%)
          </span>
        </dd>
        <dt>Neointimal index</dt>
        <dd className="tabular-nums text-right">
          {N.toExponential(3)}
        </dd>
      </dl>
      <p className="text-[10px] text-slate-500 mt-1 leading-snug">
        dA/dt = k<sub>a</sub>·(R<sub>free</sub>/R<sub>tot</sub> − A);
        dN/dt = k<sub>prolif</sub>·A − k<sub>apo</sub>·(1−A)·N. Clinical
        endpoint: N(t = 90 d) — late lumen loss (Cypher restenosis suppression).
      </p>
    </div>
  );
}
