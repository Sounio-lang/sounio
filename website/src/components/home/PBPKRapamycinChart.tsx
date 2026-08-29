import { useState, useRef } from 'react';
import { MCQBlock } from '../common/MCQBlock';

const PBPK_QUESTIONS = [
  {
    question: 'A 2025 Phase 1 trial of oral rapamycin (7 mg weekly) in Alzheimer\'s patients (Communications Medicine, doi:10.1038/s43856-025-00904-9) measured CSF drug levels alongside whole-blood concentrations. What was found?',
    options: [
      'CSF rapamycin reached ~15% of whole-blood trough, consistent with passive-diffusion estimates from logP ~4.3',
      'CSF levels correlated with dose but stayed below the IC₅₀ for mTORC1 inhibition in neurons',
      'Rapamycin was undetectable in CSF at every time point, yet CSF tau, neurofilament light, and Aβ40 all changed significantly',
      'CSF levels were measurable only in patients with elevated baseline CSF/plasma albumin ratios',
    ],
    correct: 2,
    explanation: 'Despite sirolimus being highly lipophilic (logP ~4.3, which normally predicts CNS penetration), CSF drug levels were below the assay detection limit at all time points. Yet CSF biomarkers of neurodegeneration changed — total tau, NfL, and Aβ40 increased significantly. This implies rapamycin\'s CNS effects operate through a peripheral mechanism (systemic mTOR inhibition, BBB regulation) rather than direct brain drug exposure. A PBPK model with no brain compartment would fail to capture this dissociation.',
  },
  {
    question: 'The foundational sirolimus PBPK model (Emoto et al. 2013) uses a Qgut framework to explain bioavailability variability. Which anatomical locus does the model identify as the dominant source?',
    options: [
      'Hepatic first-pass via P-glycoprotein efflux at the canalicular membrane',
      'Intestinal CYP3A activity (CYP3A4 > CYP3A5 > CYP2C8) in enterocytes',
      'Renal tubular secretion competing with glomerular filtration',
      'Adipose tissue sequestration reducing the effective central compartment concentration',
    ],
    correct: 1,
    explanation: 'Although sirolimus is clinically described as "hepatically metabolized," the PBPK model shows intestinal CYP3A as the primary driver of first-pass variability. The Qgut model predicts ~15% oral bioavailability — matching observed values — with CYP3A4 > CYP3A5 > CYP2C8 ranked by contribution. Adipose Vd (~12 L/kg) explains distribution depth, not bioavailability variance. This is the mechanistic reason why co-administration of intestinal CYP3A inhibitors (e.g. ketoconazole) has such large effects on sirolimus AUC.',
  },
  {
    question: 'Real-world rapamycin monitoring in longevity cohorts (GeroScience 2025, n=67) revealed a surprising finding about Tmax for whole-blood concentration. What is correct?',
    options: [
      'Tmax is 1–2 hours post-dose, consistent with the immediate-release gut-absorption peak',
      'Tmax is bimodal at 2 h and 36 h due to lymphatic absorption and enterohepatic recirculation',
      'Tmax is approximately 48 hours post-dose — the standard 24-hour trough sample is taken on the ascending limb',
      'Tmax is 12 hours, dominated by slow release from the Rapamune microemulsion formulation',
    ],
    correct: 2,
    explanation: 'Sirolimus binds ~95% to erythrocytes. The rapid gut-absorption peak at 1–2 hours (seen in plasma) is followed by slow redistribution into RBCs and deep tissue compartments, pushing the whole-blood Cmax to ~48 hours. Clinical 24-hour "trough" draws therefore sample the ascending phase, systematically underestimating AUC. Compounded preparations showed 31% the bioavailability of commercial Rapamune per mg, with a coefficient of variation of 0.28–0.40 across patients — exactly the kind of uncertainty that Sounio\'s GUM-propagating Knowledge<T> type is designed to make compile-time-visible.',
  },
];

// Golden fixture from tests/fixtures/darwin_pbpk/brain_plasma_tac_golden.v1.json
const ROWS = [
  { time: 0,    blood_value: 1.0,      blood_uncertainty: 0.0,      brain_value: 0.0,      brain_uncertainty: 0.0 },
  { time: 0.25, blood_value: 0.010225, blood_uncertainty: 0.001044, brain_value: 0.00067,  brain_uncertainty: 0.000976 },
  { time: 1,    blood_value: 0.003112, blood_uncertainty: 0.004049, brain_value: 0.000673, brain_uncertainty: 0.000978 },
  { time: 4,    blood_value: 0.000539, blood_uncertainty: 0.001544, brain_value: 0.000285, brain_uncertainty: 0.000981 },
  { time: 8,    blood_value: 0.000101, blood_uncertainty: 0.008361, brain_value: 0.000067, brain_uncertainty: 0.000977 },
  { time: 24,   blood_value: 0.00001,  blood_uncertainty: 0.007694, brain_value: 0.0,      brain_uncertainty: 0.000976 },
  { time: 72,   blood_value: 0.000017, blood_uncertainty: 0.001565, brain_value: 0.0,      brain_uncertainty: 0.000976 },
  { time: 168,  blood_value: 0.000026, blood_uncertainty: 0.008859, brain_value: 0.0,      brain_uncertainty: 0.000976 },
];

const SOUNIO_CODE = `// Darwin PBPK — 14-compartment rapamycin model
// GUM uncertainty propagates through RK4 ODE solver

let dose: Knowledge<mg> = Knowledge(
  5.0, ε=0.99, prov="iv_bolus_ref"
)

// Compile-time confidence gate:
// fn administer(c: Knowledge<mg_per_L>)
//   where c.ε >= 0.85 { ... }

let conc: Knowledge<mg_per_L> =
  darwin_pbpk_solve(
    drug     = rapamycin,
    dose     = dose,
    route    = IVBolus,
    solver   = RK4,
    t_span   = [0.0, 168.0]
  )
// conc.ε degrades as ODE uncertainty
// accumulates — tracked by GUM rules`;

function epsilonColor(eps: number): string {
  if (eps >= 0.90) return 'var(--color-accent-gold)';
  if (eps >= 0.80) return '#f59e0b';
  return '#ef4444';
}

export default function PBPKRapamycinChart() {
  const [hovered, setHovered] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Chart dimensions
  const W = 480, H = 260;
  const margin = { top: 20, right: 20, bottom: 40, left: 58 };
  const cw = W - margin.left - margin.right;
  const ch = H - margin.top - margin.bottom;

  // Log scale for Y (blood spans 1.0 down to ~1e-5)
  const yMin = Math.log10(0.000008);
  const yMax = Math.log10(1.1);
  const yScale = (v: number) => ch - ((Math.log10(Math.max(v, 1e-7)) - yMin) / (yMax - yMin)) * ch;

  // X scale: positions for each time point (non-linear spacing matches log-time feel)
  const times = ROWS.map(r => r.time);
  const xPositions = times.map((_t, i) => (i / (times.length - 1)) * cw);

  const bloodPath = ROWS.map((r, i) =>
    `${i === 0 ? 'M' : 'L'}${xPositions[i].toFixed(1)},${yScale(r.blood_value).toFixed(1)}`
  ).join(' ');

  const brainPath = ROWS.filter(r => r.brain_value > 0).map((r, i) => {
    const xi = ROWS.findIndex(row => row.time === r.time);
    return `${i === 0 ? 'M' : 'L'}${xPositions[xi].toFixed(1)},${yScale(r.brain_value).toFixed(1)}`;
  }).join(' ');

  // Uncertainty band for blood
  const bloodBandTop = ROWS.map((r, i) =>
    `${i === 0 ? 'M' : 'L'}${xPositions[i].toFixed(1)},${yScale(Math.max(r.blood_value + r.blood_uncertainty, 1e-7)).toFixed(1)}`
  ).join(' ');
  const bloodBandBottom = [...ROWS].reverse().map((r, i) =>
    `${i === 0 ? 'L' : 'L'}${xPositions[ROWS.length-1-i].toFixed(1)},${yScale(Math.max(r.blood_value - r.blood_uncertainty, 1e-7)).toFixed(1)}`
  ).join(' ');

  const yticks = [-5, -4, -3, -2, -1, 0];

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[color-mix(in_srgb,var(--color-bg-alt)_74%,transparent)] border-y border-[var(--glass-border)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            Rapamycin PBPK: GUM uncertainty through a 14-compartment ODE
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[72ch] leading-[1.75]">
            Physiologically-based pharmacokinetic (PBPK) modelling requires propagating
            parameter uncertainty through coupled ODEs — a task that classical tools handle
            by Monte Carlo sampling (thousands of runs) or by discarding uncertainty
            entirely. Sounio's <strong className="text-[var(--color-text-primary)]">epistemic ODE solver</strong> propagates
            ISO GUM uncertainty analytically through every RK4 integration step: each
            concentration is a <code className="text-[var(--color-accent-gold-soft)] text-sm">Knowledge&lt;mg_per_L&gt;</code> value
            carrying both a point estimate and a confidence ε ∈ [0,1].
          </p>
          <p className="text-[clamp(0.88rem,1.8vw,1rem)] text-[var(--color-text-secondary)] max-w-[72ch] leading-[1.75]">
            The chart shows a 5 mg IV bolus of rapamycin in the Darwin 14-compartment model
            (Ferron et al. 1997, CL/F = 12.4 L/h; Kahan et al. 2001, t½ = 62 ± 16 h).
            Gold bands are GUM ±σ on blood-plasma concentration; teal dashes track the
            brain-tissue compartment across the blood-brain barrier. Epistemic confidence
            degrades over the 168-hour integration window as uncertainty accumulates through
            sequential RK4 steps — a formally tracked degradation, not a silent one.
            A function declared <code className="text-[var(--color-accent-gold-soft)] text-sm">where c.ε ≥ 0.85</code>
            would reject the low-confidence late time-points at compile time.
          </p>
        </div>

        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-start">
          {/* SVG Chart */}
          <div className="glass glass-specular rounded-2xl p-6">
            <div className="flex items-center gap-6 mb-4 text-xs font-mono">
              <div className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-[var(--color-accent-gold)]" />
                <span className="text-[var(--color-text-secondary)]">Blood plasma</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-6 h-0.5 bg-[var(--color-accent-teal)]" />
                <span className="text-[var(--color-text-secondary)]">Brain tissue</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-3 bg-[var(--color-accent-gold)] opacity-20 rounded-sm" />
                <span className="text-[var(--color-text-secondary)]">±σ (GUM)</span>
              </div>
            </div>

            <svg
              ref={svgRef}
              viewBox={`0 0 ${W} ${H}`}
              className="w-full"
              style={{ maxHeight: 260 }}
              onMouseLeave={() => setHovered(null)}
            >
              <g transform={`translate(${margin.left},${margin.top})`}>
                {/* Y grid */}
                {yticks.map(tick => (
                  <g key={tick}>
                    <line x1={0} y1={yScale(Math.pow(10, tick))} x2={cw} y2={yScale(Math.pow(10, tick))}
                      stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
                    <text x={-6} y={yScale(Math.pow(10, tick))} textAnchor="end" dominantBaseline="middle"
                      fontSize={9} fill="rgba(255,255,255,0.35)" fontFamily="monospace">
                      10{tick < 0 ? tick : '⁰'}
                    </text>
                  </g>
                ))}

                {/* X axis labels */}
                {ROWS.map((r, i) => (
                  <text key={i} x={xPositions[i]} y={ch + 18} textAnchor="middle"
                    fontSize={9} fill="rgba(255,255,255,0.35)" fontFamily="monospace">
                    {r.time < 1 ? r.time : `${r.time}h`}
                  </text>
                ))}
                <text x={cw / 2} y={ch + 34} textAnchor="middle"
                  fontSize={10} fill="rgba(255,255,255,0.4)" fontFamily="monospace">
                  time (hours)
                </text>
                <text x={-ch/2} y={-44} textAnchor="middle"
                  fontSize={10} fill="rgba(255,255,255,0.4)" fontFamily="monospace"
                  transform="rotate(-90)">
                  concentration (mg/L)
                </text>

                {/* Blood uncertainty band */}
                <path
                  d={`${bloodBandTop} ${bloodBandBottom} Z`}
                  fill="var(--color-accent-gold)" fillOpacity={0.12} />

                {/* Blood plasma line */}
                <path d={bloodPath} fill="none"
                  stroke="var(--color-accent-gold)" strokeWidth={2.5} strokeLinejoin="round" />

                {/* Brain tissue line */}
                <path d={brainPath} fill="none"
                  stroke="var(--color-accent-teal)" strokeWidth={2} strokeLinejoin="round"
                  strokeDasharray="5 3" />

                {/* Data points + hover */}
                {ROWS.map((r, i) => {
                  const eps = r.blood_uncertainty > 0 && r.blood_value > 0
                    ? Math.max(0, 1 - r.blood_uncertainty / r.blood_value)
                    : 0.99;
                  return (
                    <g key={i}>
                      <circle
                        cx={xPositions[i]} cy={yScale(r.blood_value)} r={hovered === i ? 7 : 5}
                        fill={epsilonColor(eps)}
                        stroke="#0d1117" strokeWidth={1.5}
                        className="cursor-pointer transition-all"
                        onMouseEnter={() => setHovered(i)}
                      />
                    </g>
                  );
                })}
              </g>
            </svg>

            {/* Tooltip */}
            {hovered !== null && (
              <div className="mt-3 rounded-lg bg-[rgba(0,0,0,0.5)] px-4 py-3 font-mono text-xs grid grid-cols-2 gap-x-6 gap-y-1">
                <div><span className="text-[var(--color-text-tertiary)]">t = </span><span className="text-[var(--color-text-primary)]">{ROWS[hovered].time}h</span></div>
                <div>
                  <span className="text-[var(--color-text-tertiary)]">ε = </span>
                  <span style={{ color: epsilonColor(ROWS[hovered].blood_uncertainty > 0 && ROWS[hovered].blood_value > 0
                    ? Math.max(0, 1 - ROWS[hovered].blood_uncertainty / ROWS[hovered].blood_value) : 0.99) }}>
                    {ROWS[hovered].blood_value > 0 && ROWS[hovered].blood_uncertainty > 0
                      ? (1 - ROWS[hovered].blood_uncertainty / ROWS[hovered].blood_value).toFixed(3)
                      : '0.990'}
                  </span>
                </div>
                <div><span className="text-[var(--color-accent-gold)]">blood</span><span className="text-[var(--color-text-secondary)]"> = {ROWS[hovered].blood_value.toExponential(3)} ± {ROWS[hovered].blood_uncertainty.toExponential(2)}</span></div>
                <div><span className="text-[var(--color-accent-teal)]">brain</span><span className="text-[var(--color-text-secondary)]"> = {ROWS[hovered].brain_value.toExponential(3)} ± {ROWS[hovered].brain_uncertainty.toExponential(2)}</span></div>
              </div>
            )}
          </div>

          {/* Code panel */}
          <div className="rounded-2xl overflow-hidden border border-[var(--color-accent-gold)]/20">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(201,169,110,0.05)] border-b border-[var(--color-accent-gold)]/20">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                <div className="w-3 h-3 rounded-full bg-[#28c840]" />
              </div>
              <span className="flex-1 text-center text-[var(--color-accent-gold)] text-sm font-mono">
                pbpk_rapamycin.sio
              </span>
            </div>
            <pre className="p-6 bg-[#0d1117] text-xs leading-relaxed overflow-x-auto">
              <code className="text-[#e6edf3]">{SOUNIO_CODE}</code>
            </pre>
            <div className="px-6 py-4 bg-[#0d1117] border-t border-[var(--glass-border)]">
              <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                <div>
                  <div className="text-[var(--color-accent-gold)] font-semibold">CL/F = 12.4 L/h</div>
                  <div className="text-[var(--color-text-tertiary)]">Ferron et al. 1997</div>
                </div>
                <div>
                  <div className="text-[var(--color-accent-teal)] font-semibold">t½ = 62 ± 16h</div>
                  <div className="text-[var(--color-text-tertiary)]">Kahan et al. 2001</div>
                </div>
                <div>
                  <div className="text-[var(--color-accent-gold)] font-semibold">14 compartments</div>
                  <div className="text-[var(--color-text-tertiary)]">Darwin PBPK model</div>
                </div>
                <div>
                  <div className="text-emerald-400 font-semibold">GUM tracked</div>
                  <div className="text-[var(--color-text-tertiary)]">through RK4 steps</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <MCQBlock questions={PBPK_QUESTIONS} />
      </div>
    </section>
  );
}
