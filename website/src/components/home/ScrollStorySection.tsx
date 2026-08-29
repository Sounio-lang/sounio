import { useRef, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useAudience } from '../../lib/useAudience';
import { scrollStoryContent } from '../../lib/audienceContent';

export function ScrollStorySection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });
  const { audience } = useAudience();
  const [codeExpanded, setCodeExpanded] = useState(false);

  const segments = scrollStoryContent[audience];

  const textOpacity1 = useTransform(scrollYProgress, [0, 0.1, 0.25, 0.35], [0, 1, 1, 0]);
  const textOpacity2 = useTransform(scrollYProgress, [0.35, 0.45, 0.6, 0.7], [0, 1, 1, 0]);
  const textOpacity3 = useTransform(scrollYProgress, [0.7, 0.8, 0.95, 1], [0, 1, 1, 0]);

  const ring1Scale = useTransform(scrollYProgress, [0, 1], [0.5, 2.5]);
  const ring1Rotate = useTransform(scrollYProgress, [0, 1], [0, 180]);
  const ring2Scale = useTransform(scrollYProgress, [0, 1], [0.8, 3.5]);
  const ring2Rotate = useTransform(scrollYProgress, [0, 1], [0, -180]);

  const codeClipPath = useTransform(
    scrollYProgress,
    [0.75, 0.9],
    ["inset(0 0 100% 0)", "inset(0 0 0% 0)"]
  );

  const scientistCode = `// Confidence-gated vancomycin dosing
let dose: Knowledge<f64> = Knowledge(
    500.0,
    \u03B5=0.92,
    prov="ASHP_2020_RCT"
)

if dose.\u03B5 >= 0.82 {
    approve_dose(dose)
} else {
    request_remeasurement(dose)
}`;

  const technicalCode = `// Effect-annotated epistemic pipeline
fn analyze(dose: Knowledge<mg>) -> Validated<mg>
    with IO, Mut {
  let v: Validated<mg> = validate(
    dose, threshold: 0.95
  )
  v
}

// \u2192 self-hosted native ELF (x86-64 SSE2)
0x1000: push   rbp
0x1001: mov    rbp, rsp
0x1004: movsd  xmm0, [rip+0x200]
0x100b: pop    rbp
0x100c: ret`;

  return (
    <section ref={containerRef} className="relative h-[400vh] bg-[#020617] text-white">
      <div className="sticky top-0 h-screen w-full flex flex-col items-center justify-center overflow-hidden">

        {/* Abstract graphical rings */}
        <div className="absolute inset-0 opacity-20 flex items-center justify-center pointer-events-none mix-blend-screen">
           <motion.div
             className="w-[80vw] h-[80vw] md:w-[50vw] md:h-[50vw] rounded-full border border-white/10 shadow-[0_0_100px_rgba(255,255,255,0.05)]"
             style={{ scale: ring1Scale, rotate: ring1Rotate }}
           />
           <motion.div
             className="absolute w-[60vw] h-[60vw] md:w-[35vw] md:h-[35vw] rounded-full border border-[var(--color-accent-teal)]/20 shadow-[0_0_80px_rgba(45,212,191,0.1)]"
             style={{ scale: ring2Scale, rotate: ring2Rotate }}
           />
        </div>

        <div className="absolute inset-0 bg-gradient-to-b from-[#020617] via-transparent to-[#020617] pointer-events-none" />

        {/* Segment 1 */}
        <motion.div className="absolute text-center max-w-4xl px-6 pointer-events-none" style={{ opacity: textOpacity1 }}>
          <h2 className="text-[clamp(2.5rem,5vw,5rem)] font-extrabold tracking-tighter mb-6 leading-[1.1]">
            {segments[0].heading} <br className="md:hidden" /><span className="text-transparent bg-clip-text bg-gradient-to-r from-[var(--color-accent-gold)] to-white">{segments[0].headingHighlight}</span>
          </h2>
          <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 font-light mx-auto max-w-[40ch]">
            {segments[0].body}
          </p>
        </motion.div>

        {/* Segment 2 */}
        <motion.div className="absolute text-center max-w-4xl px-6 pointer-events-none" style={{ opacity: textOpacity2 }}>
          <h2 className="text-[clamp(2.5rem,5vw,5rem)] font-extrabold tracking-tighter mb-6 leading-[1.1]">
            {segments[1].heading} <span className="text-transparent bg-clip-text bg-gradient-to-r from-[var(--color-accent-teal)] to-blue-400">{segments[1].headingHighlight}</span>
          </h2>
          <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 font-light mx-auto max-w-[40ch]">
            {segments[1].body}
          </p>
        </motion.div>

        {/* Segment 3 */}
        <motion.div className="absolute flex flex-col md:flex-row gap-12 items-center justify-between max-w-6xl px-6 w-full" style={{ opacity: textOpacity3 }}>
          <div className="flex-1 md:pr-12">
            <h2 className="text-[clamp(2.5rem,5vw,5rem)] font-extrabold tracking-tighter mb-6 leading-[1.1]">
              {segments[2].heading} <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200">{segments[2].headingHighlight}</span>
            </h2>
            <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 font-light">
              {segments[2].body}
            </p>
          </div>
          <div className="flex-1 w-full border border-white/10 rounded-2xl bg-black/60 backdrop-blur-xl overflow-hidden relative shadow-2xl">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[var(--color-accent-teal)] to-emerald-400 opacity-50" />
            {audience === 'scientist' ? (
              <div className="pointer-events-auto">
                <div className={`code-collapsible ${codeExpanded ? 'expanded' : 'collapsed'}`}>
                  <pre className="p-8 text-xs md:text-sm font-mono text-emerald-400/90 leading-relaxed">
                    {scientistCode}
                  </pre>
                </div>
                <div className="p-4 border-t border-white/10 flex justify-center">
                  <button
                    onClick={() => setCodeExpanded(!codeExpanded)}
                    className="code-toggle-btn"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d={codeExpanded ? "M19 9l-7 7-7-7" : "M9 5l7 7-7 7"} />
                    </svg>
                    {codeExpanded ? 'Hide Sounio code' : 'Show Sounio code'}
                  </button>
                </div>
              </div>
            ) : (
              <motion.pre
                className="p-8 text-xs md:text-sm font-mono text-emerald-400/90 leading-relaxed"
                style={{ clipPath: codeClipPath }}
              >
                {technicalCode}
              </motion.pre>
            )}
          </div>
        </motion.div>

      </div>
    </section>
  );
}
