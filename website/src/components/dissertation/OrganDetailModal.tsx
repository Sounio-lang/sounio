import { useEffect } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { COMPARTMENTS } from './compartments';
import type { DrugId } from '../../hooks/usePBPK';

interface OrganDetailModalProps {
  drug: DrugId;
  organIndex: number;
  /** Normalised concentration in [0..1] used by the visual encoding. */
  concentrationNorm: number;
  /** Absolute concentration on the visual reference scale, mg/L. */
  concentrationMgPerL: number;
  /** Mass currently in the organ, mg (concentrationMgPerL × V_ref). */
  massMg: number;
  /** Per-organ TMDD state (nmol/L). length-14 arrays; 0 for non-TMDD organs. */
  rfree?: Float32Array;
  dr?: Float32Array;
  /** Per-organ PD state. For rapamycin (heart): A=mTORC1, N=neointima.
   *  For semaglutide (pancreas): A=ΔG mg/dL, N=ΔI mU/L. */
  pdA?: Float32Array;
  pdN?: Float32Array;
  onClose: () => void;
}

const TMDD_ORGANS_BY_DRUG: Record<DrugId, number[]> = {
  rapamycin: [1, 4, 8],
  semaglutide: [3, 8, 12],
};
const PD_ORGAN_BY_DRUG: Record<DrugId, number> = {
  rapamycin: 4,
  semaglutide: 12,
};

function KatexBlock({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, {
    throwOnError: false,
    displayMode: true,
  });
  return <div className="text-white katex-block" dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * Click-an-organ pop-out with the KaTeX-rendered mass-balance ODE, current
 * concentration, Kp / Q, and a one-line clinical reading. Drug-aware: blood
 * ODE shows Higuchi + CL_hep for rapamycin, SC depot + CL_proteolytic for
 * semaglutide; TMDD and PD organs append their per-organ ODE blocks.
 */
export function OrganDetailModal({
  drug,
  organIndex,
  concentrationNorm,
  concentrationMgPerL,
  massMg,
  rfree,
  dr,
  pdA,
  pdN,
  onClose,
}: OrganDetailModalProps) {
  const c = COMPARTMENTS[organIndex];

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  if (!c) return null;

  const isBlood = organIndex === 0;
  const isTmddOrgan = TMDD_ORGANS_BY_DRUG[drug].includes(organIndex);
  const isPdOrgan = PD_ORGAN_BY_DRUG[drug] === organIndex;

  // Blood ODE — drug-aware release source + clearance route.
  const bloodTex = drug === 'rapamycin'
    ? String.raw`V_{\text{blood}}\frac{dC_{\text{blood}}}{dt} = -\sum_{i\ge 1} Q_i\!\left(C_{\text{blood}} - \frac{C_i}{K_{p,i}}\right) + R_{\text{stent}}(t) - \mathrm{CL}_{\text{hep}}\,C_{\text{blood}}`
    : String.raw`V_{\text{blood}}\frac{dC_{\text{blood}}}{dt} = -\sum_{i\ge 1} Q_i\!\left(C_{\text{blood}} - \frac{C_i}{K_{p,i}}\right) + R_{\text{SC}}(t) - \mathrm{CL}_{\text{prot}}\,C_{\text{blood}}`;
  const orgTex = String.raw`V_{${c.name}}\,\frac{dC_{${c.name}}}{dt} = Q_{${c.name}}\!\left(C_{\text{blood}} - \frac{C_{${c.name}}}{K_{p,${c.name}}}\right)`;
  const tex = isBlood ? bloodTex : orgTex;

  const tmddTex = String.raw`\begin{aligned}
\frac{dR_{\text{free}}}{dt} &= k_{\text{syn}} - k_{\text{deg}}\,R_{\text{free}} - k_{\text{on}}\,C_t R_{\text{free}} + k_{\text{off}}\,DR \\
\frac{d(DR)}{dt} &= k_{\text{on}}\,C_t R_{\text{free}} - (k_{\text{off}}+k_{\text{int}})\,DR
\end{aligned}`;

  const pdTexRapamycin = String.raw`\begin{aligned}
\frac{dA}{dt} &= k_a\!\left(\tfrac{R_{\text{free}}}{R_{\text{tot}}} - A\right) \\
\frac{dN}{dt} &= k_{\text{prolif}}\,A - k_{\text{apo}}\,(1-A)\,N
\end{aligned}`;
  const pdTexSemaglutide = String.raw`\begin{aligned}
\frac{d\Delta I}{dt} &= k_{\text{base}}\,\alpha\,\mathrm{occ} - k_I\,\Delta I \\
\frac{d\Delta G}{dt} &= -S_I\,\Delta I - S_G\,\Delta G
\end{aligned}`;

  const rf = rfree?.[organIndex] ?? 0;
  const drv = dr?.[organIndex] ?? 0;
  const occ = rf + drv > 1e-9 ? drv / (rf + drv) : 0;
  const a = pdA?.[organIndex] ?? 0;
  const n = pdN?.[organIndex] ?? 0;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/65 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="max-w-xl w-full bg-[#0e1325] border border-white/15 rounded-lg shadow-2xl p-5 text-white"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-baseline justify-between mb-3">
          <div>
            <h3 className="text-lg font-semibold">{c.label}</h3>
            <p className="text-xs opacity-60 mt-0.5">
              compartment {organIndex} · {c.family}
              {isTmddOrgan && <span className="ml-2 text-emerald-400">· TMDD</span>}
              {isPdOrgan && <span className="ml-2 text-amber-300">· PD readout</span>}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-xs opacity-60 hover:opacity-100 px-2 py-1"
          >
            close (esc)
          </button>
        </div>

        <div className="bg-black/40 rounded p-3 mb-3 overflow-x-auto text-[0.95rem]">
          <KatexBlock tex={tex} />
        </div>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <div className="opacity-65 text-xs uppercase tracking-wider">Partition</div>
            <div className="font-mono text-base">K<sub>p</sub> = {c.kp.toFixed(2)}</div>
          </div>
          <div>
            <div className="opacity-65 text-xs uppercase tracking-wider">Blood flow</div>
            <div className="font-mono text-base">Q = {c.q} L/h</div>
          </div>
          <div>
            <div className="opacity-65 text-xs uppercase tracking-wider">C (current)</div>
            <div className="font-mono text-base">{(concentrationMgPerL * 1e6).toFixed(2)} ng/L</div>
            <div className="font-mono text-[0.7rem] opacity-65">
              {(concentrationNorm * 100).toFixed(1)}% of visual max
            </div>
          </div>
          <div>
            <div className="opacity-65 text-xs uppercase tracking-wider">Mass</div>
            <div className="font-mono text-base">{(massMg * 1e9).toFixed(2)} pg</div>
          </div>
        </div>

        {isTmddOrgan && (
          <div className="mt-4 pt-3 border-t border-white/10">
            <div className="text-xs uppercase tracking-wider opacity-70 mb-1.5">
              TMDD · {drug === 'rapamycin' ? 'FKBP12 / mTORC1' : 'GLP-1R'}
            </div>
            <div className="bg-black/40 rounded p-2 mb-2 overflow-x-auto text-[0.85rem]">
              <KatexBlock tex={tmddTex} />
            </div>
            <div className="grid grid-cols-3 gap-2 text-[0.75rem] font-mono">
              <div><span className="opacity-60">R_free </span>{rf.toFixed(2)} nM</div>
              <div><span className="opacity-60">DR </span>{drv.toFixed(2)} nM</div>
              <div><span className="opacity-60">occ </span>{(occ * 100).toFixed(1)}%</div>
            </div>
          </div>
        )}

        {isPdOrgan && (
          <div className="mt-4 pt-3 border-t border-white/10">
            <div className="text-xs uppercase tracking-wider opacity-70 mb-1.5">
              PD readout · {drug === 'rapamycin' ? 'mTORC1 + neointima' : 'glucose-insulin (Bergman)'}
            </div>
            <div className="bg-black/40 rounded p-2 mb-2 overflow-x-auto text-[0.85rem]">
              <KatexBlock tex={drug === 'rapamycin' ? pdTexRapamycin : pdTexSemaglutide} />
            </div>
            <div className="grid grid-cols-2 gap-2 text-[0.75rem] font-mono">
              {drug === 'rapamycin' ? (
                <>
                  <div><span className="opacity-60">A (mTORC1) </span>{a.toFixed(3)}</div>
                  <div><span className="opacity-60">N </span>{n.toExponential(2)}</div>
                </>
              ) : (
                <>
                  <div><span className="opacity-60">ΔG </span>{a.toFixed(2)} mg/dL</div>
                  <div><span className="opacity-60">ΔI </span>{n.toFixed(2)} mU/L</div>
                </>
              )}
            </div>
          </div>
        )}

        <div className="mt-4 pt-3 border-t border-white/10 text-[0.78rem] opacity-80 leading-relaxed">
          {isBlood ? (
            drug === 'rapamycin' ? (
              <>
                Central compartment. Receives <span className="text-yellow-300">stent elution</span>{' '}
                (Higuchi 1/√t source), distributes to all tissues, and clears via hepatic CL.
              </>
            ) : (
              <>
                Central compartment. Receives <span className="text-cyan-300">SC absorption</span>{' '}
                (1st-order F·Dose·k<sub>a</sub>·e<sup>−k<sub>a</sub>t</sup> source), distributes
                systemically with very slow proteolytic clearance (~0.077 L/h).
              </>
            )
          ) : c.kp < 0.2 ? (
            <>
              Low partition (K<sub>p</sub> &lt; 0.2) — tissue exposure capped by membrane/transporter
              efflux (e.g. P-gp at the BBB for brain).
            </>
          ) : c.kp > 3 ? (
            drug === 'rapamycin' ? (
              <>
                High partition (K<sub>p</sub> &gt; 3) — significant tissue accumulation; relevant for
                30-day Cypher elution AUC.
              </>
            ) : (
              <>
                High partition (K<sub>p</sub> &gt; 3) — tissue accumulation over the weekly dosing
                interval drives the chronic exposure envelope.
              </>
            )
          ) : (
            <>Moderate partition; tissue tracks plasma with modest lag.</>
          )}
        </div>
      </div>
    </div>
  );
}
