import { useState } from 'react';

const STEPS = [
  {
    id: 'scene',
    label: 'The scenario',
    accent: 'rgba(255,255,255,0.15)',
    content: {
      type: 'scene' as const,
      title: 'Ward C, 07:14.',
      body: [
        'A nurse enters patient weight into a dosing calculator for vancomycin — a narrow-therapeutic-index antibiotic where the difference between 15 mg/kg and 18 mg/kg is the difference between treatment and nephrotoxicity.',
        'The bedside scale passed visual inspection. Its last calibration certificate is three months old. The software has no way to know this.',
        'The calculator runs. The output looks reasonable. The dose is prepared.',
      ],
      tag: null,
    },
  },
  {
    id: 'python',
    label: 'The code',
    accent: 'rgba(255,255,255,0.15)',
    content: {
      type: 'code' as const,
      lang: 'Python',
      filename: 'dose_calculator.py',
      code: `def vancomycin_dose(weight_kg: float,
                       creatinine: float) -> float:
    # AUC/MIC target: 400-600 mg·h/L
    # Cockcroft-Gault for clearance estimate
    crcl = (140 - 45) * weight_kg / (72 * creatinine)
    daily_dose = crcl * 15          # mg/kg/day × CrCl
    return daily_dose

# Nurse enters: 82 kg  (scale reads 82, true weight: 77)
dose = vancomycin_dose(82, 0.9)
# Output: 1,923 mg/day
# Should be: 1,806 mg/day
# Difference: +6.5%  →  cumulative nephrotoxicity risk

print(f"Administer: {dose:.0f} mg/day")  # looks fine`,
      tag: null,
    },
  },
  {
    id: 'failure',
    label: 'The outcome',
    accent: '#ef4444',
    content: {
      type: 'failure' as const,
      output: 'Administer: 1923 mg/day',
      points: [
        { icon: '⚠', text: 'Scale last calibrated 94 days ago. Reads +5 kg high at this range.' },
        { icon: '⚠', text: 'Actual weight: 77 kg. Dose should be 1,806 mg/day.' },
        { icon: '⚠', text: 'Overshoot: +6.5%. Over 5 days, cumulative AUC exceeds nephrotoxicity threshold.' },
        { icon: '⚠', text: 'No error. No warning. The type system has no concept of measurement quality.' },
      ],
      tag: 'SILENT FAILURE',
    },
  },
  {
    id: 'sounio',
    label: 'Sounio version',
    accent: 'var(--color-accent-gold)',
    content: {
      type: 'code' as const,
      lang: 'Sounio',
      filename: 'dose_calculator.sio',
      code: `// Scale calibration cert: 94 days old → ε degraded
let weight: Knowledge<kg> = Knowledge(
  82.0,
  ε=0.71,                     // calibration age penalty
  prov="ward_c_scale_A3",
)

// Creatinine from lab: fresh assay, high confidence
let creatinine: Knowledge<mmol_per_L> = Knowledge(
  0.9, ε=0.97, prov="lab_jaffe_run_441"
)

// Confidence gate on the dosing function:
fn compute_dose(w: Knowledge<kg>, cr: Knowledge<mmol_per_L>)
  -> Knowledge<mg_per_day>
  where w.ε >= 0.82            // ← requires calibrated scale
  with IO
{
  let crcl = (140.0 - 45.0) * w / (72.0 * cr)
  crcl * 15.0
}

let dose = compute_dose(weight, creatinine)`,
      tag: null,
    },
  },
  {
    id: 'gate',
    label: 'Compiler rejects',
    accent: '#22c55e',
    content: {
      type: 'gate' as const,
      error: `error[E0421]: epistemic confidence gate not satisfied
  --> dose_calculator.sio:22:13
   |
22 |   let dose = compute_dose(weight, creatinine)
   |              ^^^^^^^^^^^^ ε=0.71 < required ε=0.82
   |
note: function declared with \`where w.ε >= 0.82\`
  --> dose_calculator.sio:14:3
   |
14 |   where w.ε >= 0.82
   |   ^^^^^^^^^^^^^^^^^ confidence gate defined here
   |
help: the scale at prov="ward_c_scale_A3" has degraded
      confidence due to calibration age (94 days).
      Request recalibration or use a scale with ε ≥ 0.82.

error: could not compile \`dose_calculator.sio\``,
      points: [
        { icon: '✓', text: 'Binary not produced. The wrong dose is never administered.' },
        { icon: '✓', text: 'Error points to the calibration source — not a vague type mismatch.' },
        { icon: '✓', text: 'The gate ε ≥ 0.82 is part of the function\'s public contract, visible to every caller.' },
        { icon: '✓', text: 'No change to runtime. No overhead. This is purely compile-time.' },
      ],
      tag: 'CAUGHT AT COMPILE TIME',
    },
  },
];

export default function NarrativeDemo() {
  const [step, setStep] = useState(0);
  const current = STEPS[step];
  const { content } = current;

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[#08111f] border-y border-[var(--glass-border)]">
      <div className="container px-4">

        {/* Header */}
        <div className="mb-[2.4rem] grid gap-[0.5rem] max-w-[58ch]">
          <h2 className="font-sans text-[clamp(1.5rem,3.8vw,2.6rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            A real scenario. See what happens.
          </h2>
          <p className="text-[0.9rem] text-[var(--color-text-secondary)] leading-[1.65]">
            Walk through a clinical dosing calculation — first the way every other language handles it, then with Sounio.
          </p>
        </div>

        {/* Step indicator */}
        <div className="flex items-center gap-3 mb-8">
          {STEPS.map((s, i) => (
            <button
              key={s.id}
              onClick={() => setStep(i)}
              className="flex items-center gap-2 group"
            >
              <div
                className="w-7 h-7 rounded-full flex items-center justify-center text-[0.7rem] font-[700] transition-all border"
                style={{
                  background: i === step ? current.accent : 'transparent',
                  borderColor: i === step ? current.accent : i < step ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.1)',
                  color: i === step ? (step === 2 ? '#fff' : '#0d1117') : i < step ? 'rgba(255,255,255,0.5)' : 'rgba(255,255,255,0.2)',
                }}
              >
                {i < step ? '✓' : i + 1}
              </div>
              <span
                className="text-[0.72rem] font-[600] hidden sm:block transition-colors"
                style={{ color: i === step ? 'var(--color-text-primary)' : 'var(--color-text-tertiary)' }}
              >
                {s.label}
              </span>
              {i < STEPS.length - 1 && (
                <div
                  className="hidden sm:block h-[1px] w-6 transition-all"
                  style={{ background: i < step ? 'rgba(255,255,255,0.3)' : 'rgba(255,255,255,0.08)' }}
                />
              )}
            </button>
          ))}
        </div>

        {/* Content card */}
        <div
          className="rounded-2xl overflow-hidden border transition-all duration-300"
          style={{ borderColor: current.accent }}
        >
          {/* Tag banner */}
          {content.tag && (
            <div
              className="px-6 py-3 flex items-center gap-3"
              style={{
                background: step === 2 ? 'rgba(239,68,68,0.15)' : 'rgba(34,197,94,0.12)',
              }}
            >
              <div
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ background: current.accent }}
              />
              <span
                className="text-[0.72rem] font-[800] tracking-[0.15em] uppercase"
                style={{ color: current.accent }}
              >
                {content.tag}
              </span>
            </div>
          )}

          {/* Scene */}
          {content.type === 'scene' && (
            <div className="p-8 grid gap-5 bg-[rgba(255,255,255,0.02)]">
              <h3 className="text-[1.4rem] font-[750] text-[var(--color-text-primary)]">
                {content.title}
              </h3>
              {content.body.map((para, i) => (
                <p key={i} className="text-[0.92rem] text-[var(--color-text-secondary)] leading-[1.75] max-w-[70ch]">
                  {para}
                </p>
              ))}
              <div className="mt-2 flex items-center gap-3 text-[0.8rem] text-[var(--color-text-tertiary)]">
                <span className="inline-block w-3 h-3 rounded-full bg-[rgba(239,68,68,0.6)]" />
                Vancomycin therapeutic index: 15–20 mg/kg/day. Nephrotoxicity threshold: &gt;20 mg/kg/day sustained.
              </div>
            </div>
          )}

          {/* Code */}
          {content.type === 'code' && (
            <div className="bg-[#0d1117]">
              <div className="flex items-center gap-2 px-5 py-3 border-b border-[var(--glass-border)]">
                <div className="flex gap-1.5">
                  <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                  <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                  <div className="w-3 h-3 rounded-full bg-[#28c840]" />
                </div>
                <span
                  className="ml-2 text-[0.8rem] font-mono px-2 py-0.5 rounded"
                  style={{
                    color: step === 3 ? 'var(--color-accent-gold)' : 'rgba(255,255,255,0.4)',
                    background: step === 3 ? 'rgba(201,169,110,0.08)' : 'transparent',
                  }}
                >
                  {content.filename}
                </span>
                <span className="ml-auto text-[0.72rem] font-mono text-[var(--color-text-tertiary)]">
                  {content.lang}
                </span>
              </div>
              <pre className="p-6 text-[0.82rem] leading-[1.85] overflow-x-auto">
                <code
                  className="font-mono"
                  style={{ color: step === 3 ? '#e6edf3' : '#8b949e' }}
                >
                  {content.code.split('\n').map((line, i) => {
                    // Highlight comment lines in a subtler way
                    const isComment = line.trimStart().startsWith('#') || line.trimStart().startsWith('//');
                    const isKey = step === 3 && (
                      line.includes('Knowledge') || line.includes('ε=') ||
                      line.includes('where') || line.includes('prov=')
                    );
                    return (
                      <span
                        key={i}
                        className="block"
                        style={{
                          color: isKey
                            ? 'var(--color-accent-gold)'
                            : isComment
                              ? 'rgba(139,148,158,0.6)'
                              : undefined,
                        }}
                      >
                        {line || '\u00A0'}
                      </span>
                    );
                  })}
                </code>
              </pre>
            </div>
          )}

          {/* Failure */}
          {content.type === 'failure' && (
            <div className="bg-[rgba(239,68,68,0.05)]">
              <div className="p-6 border-b border-[rgba(239,68,68,0.2)]">
                <div className="text-[0.72rem] text-[rgba(239,68,68,0.7)] font-mono mb-2">stdout</div>
                <pre className="font-mono text-[1rem] text-[#ef4444] font-[600]">
                  {content.output}
                </pre>
                <div className="mt-2 text-[0.78rem] text-[rgba(239,68,68,0.6)] font-mono">
                  Exit code: 0  — no errors reported
                </div>
              </div>
              <div className="p-6 grid gap-3">
                {content.points.map((p, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="text-[#ef4444] text-[1rem] mt-[1px] shrink-0">{p.icon}</span>
                    <p className="text-[0.87rem] text-[rgba(239,68,68,0.85)] leading-[1.6]">{p.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Gate */}
          {content.type === 'gate' && (
            <div className="bg-[#0d1117]">
              <pre className="p-6 text-[0.79rem] leading-[1.8] overflow-x-auto font-mono">
                {content.error.split('\n').map((line, i) => {
                  const isError = line.startsWith('error');
                  const isNote = line.trim().startsWith('note') || line.trim().startsWith('-->');
                  const isHelp = line.trim().startsWith('help');
                  const isCarets = line.includes('^^^') || line.includes('^^^');
                  return (
                    <span
                      key={i}
                      className="block"
                      style={{
                        color: isError
                          ? '#ef4444'
                          : isNote
                            ? 'rgba(139,148,158,0.7)'
                            : isHelp
                              ? 'var(--color-accent-teal)'
                              : isCarets
                                ? '#ef4444'
                                : '#8b949e',
                      }}
                    >
                      {line || '\u00A0'}
                    </span>
                  );
                })}
              </pre>
              <div className="p-6 border-t border-[rgba(34,197,94,0.2)] grid gap-3">
                {content.points.map((p, i) => (
                  <div key={i} className="flex items-start gap-3">
                    <span className="text-[#22c55e] shrink-0 font-[700]">{p.icon}</span>
                    <p className="text-[0.87rem] text-[rgba(34,197,94,0.85)] leading-[1.6]">{p.text}</p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Navigation */}
        <div className="flex items-center justify-between mt-6">
          <button
            onClick={() => setStep(s => Math.max(0, s - 1))}
            disabled={step === 0}
            className="rounded-full px-5 py-2.5 text-[0.85rem] font-[600] border border-[var(--glass-border)] text-[var(--color-text-secondary)] hover:bg-[rgba(255,255,255,0.05)] disabled:opacity-30 disabled:cursor-default transition-all"
          >
            ← Back
          </button>

          <span className="text-[0.78rem] text-[var(--color-text-tertiary)] font-mono">
            {step + 1} / {STEPS.length}
          </span>

          {step < STEPS.length - 1 ? (
            <button
              onClick={() => setStep(s => Math.min(STEPS.length - 1, s + 1))}
              className="rounded-full px-5 py-2.5 text-[0.85rem] font-[600] transition-all hover:-translate-y-px"
              style={{
                background: `linear-gradient(120deg, ${current.accent}, ${current.accent})`,
                color: step === 2 ? '#fff' : '#0d1117',
              }}
            >
              {step === 1 ? 'See what goes wrong →' : step === 2 ? 'See the fix →' : 'Continue →'}
            </button>
          ) : (
            <a
              href="/proof#compile-fail-vancomycin"
              className="rounded-full px-5 py-2.5 text-[0.85rem] font-[600] text-[#0d1117] hover:-translate-y-px transition-all"
              style={{ background: '#22c55e' }}
            >
              Read the full proof →
            </a>
          )}
        </div>
      </div>
    </section>
  );
}
