// G-ε-2: TMDD occupancy panel. Reads simState.rfree / simState.dr (per-organ
// nmol/L) and renders horizontal bars showing receptor occupancy (DR/R_total)
// for the active drug's TMDD organs. Organs sourced from drug profile so
// rapamycin shows liver/heart/gut (FKBP12) and semaglutide shows brain/gut/
// pancreas (GLP-1R).

import type { DrugId } from '../../hooks/usePBPK';

const ORGAN_NAMES: Record<number, string> = {
  0: 'Blood', 1: 'Liver', 2: 'Kidney', 3: 'Brain', 4: 'Heart',
  5: 'Lung', 6: 'Muscle', 7: 'Adipose', 8: 'Gut', 9: 'Skin',
  10: 'Bone', 11: 'Spleen', 12: 'Pancreas', 13: 'Other',
};

const TMDD_ORGANS_BY_DRUG: Record<DrugId, number[]> = {
  rapamycin: [1, 4, 8],     // FKBP12: liver, heart, gut
  semaglutide: [3, 8, 12],  // GLP-1R: brain, gut, pancreas
};

const RECEPTOR_BY_DRUG: Record<DrugId, string> = {
  rapamycin: 'FKBP12 / mTORC1',
  semaglutide: 'GLP-1 receptor',
};

export function TmddPanel({
  drug,
  rfree,
  dr,
}: {
  drug: DrugId;
  rfree?: Float32Array;
  dr?: Float32Array;
}) {
  const organs = TMDD_ORGANS_BY_DRUG[drug];
  const label = RECEPTOR_BY_DRUG[drug];
  return (
    <div>
      <h3 className="text-sm font-semibold text-white mb-1">
        Receptor occupancy · <span className="text-slate-400 font-normal">{label}</span>
      </h3>
      <div className="space-y-1.5">
        {organs.map((i) => {
          const rf = rfree?.[i] ?? 0;
          const d = dr?.[i] ?? 0;
          const total = rf + d;
          const occ = total > 1e-9 ? d / total : 0;
          const pctOcc = (occ * 100).toFixed(1);
          return (
            <div key={i} className="text-xs">
              <div className="flex justify-between text-slate-300">
                <span>{ORGAN_NAMES[i]}</span>
                <span className="tabular-nums">
                  {pctOcc}% &nbsp;
                  <span className="text-slate-500">
                    DR={d.toFixed(2)} / R<sub>tot</sub>={total.toFixed(2)} nM
                  </span>
                </span>
              </div>
              <div
                className="h-1.5 rounded-full bg-slate-800 overflow-hidden mt-0.5"
                role="progressbar"
                aria-valuenow={parseFloat(pctOcc)}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <div
                  className="h-full bg-gradient-to-r from-emerald-400 to-amber-400"
                  style={{ width: `${Math.min(100, occ * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
