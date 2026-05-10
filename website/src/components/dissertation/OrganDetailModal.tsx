import { useEffect } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';
import { COMPARTMENTS } from './compartments';

interface OrganDetailModalProps {
  organIndex: number;
  /** Normalised concentration in [0..1] used by the visual encoding. */
  concentrationNorm: number;
  /** Absolute concentration on the visual reference scale, mg/L. */
  concentrationMgPerL: number;
  /** Mass currently in the organ, mg (concentrationMgPerL × V_ref). */
  massMg: number;
  onClose: () => void;
}

function KatexBlock({ tex }: { tex: string }) {
  const html = katex.renderToString(tex, {
    throwOnError: false,
    displayMode: true,
  });
  return <div className="text-white katex-block" dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * Click-an-organ pop-out with the KaTeX-rendered mass-balance ODE, current
 * concentration, Kp / Q, and a one-line clinical reading.
 *
 * Replaces the inline HUD card that the MVP shipped with — same data, richer
 * presentation suitable for the defense.
 */
export function OrganDetailModal({
  organIndex,
  concentrationNorm,
  concentrationMgPerL,
  massMg,
  onClose,
}: OrganDetailModalProps) {
  const c = COMPARTMENTS[organIndex];

  // Escape on close.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  if (!c) return null;

  // Mass-balance ODE
  //   V_i dC_i/dt = Q_i ( C_blood - C_i / Kp_i )
  // and for blood the central balance.
  const isBlood = organIndex === 0;
  const tex = isBlood
    ? String.raw`V_{\text{blood}}\frac{dC_{\text{blood}}}{dt} = -\sum_{i\ge 1} Q_i\!\left(C_{\text{blood}} - \frac{C_i}{K_{p,i}}\right) + R_{\text{stent}}(t) - \mathrm{CL}_{\text{hep}}\,C_{\text{blood}}`
    : String.raw`V_{${c.name}}\,\frac{dC_{${c.name}}}{dt} = Q_{${c.name}}\!\left(C_{\text{blood}} - \frac{C_{${c.name}}}{K_{p,${c.name}}}\right)`;

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

        <div className="mt-4 pt-3 border-t border-white/10 text-[0.78rem] opacity-80 leading-relaxed">
          {isBlood ? (
            <>
              Central compartment. Receives <span className="text-yellow-300">stent elution</span>{' '}
              (Higuchi 1/√t source), distributes to all tissues, and clears via hepatic CL.
            </>
          ) : c.kp < 0.2 ? (
            <>
              Low partition (K<sub>p</sub> &lt; 0.2) — tissue exposure capped by membrane/transporter
              efflux (e.g. P-gp at the BBB for brain).
            </>
          ) : c.kp > 3 ? (
            <>
              High partition (K<sub>p</sub> &gt; 3) — significant tissue accumulation; relevant for
              30-day Cypher elution AUC.
            </>
          ) : (
            <>Moderate partition; tissue tracks plasma with modest lag.</>
          )}
        </div>
      </div>
    </div>
  );
}
