import React, { useRef } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';

export function ScrollStorySection() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start start", "end end"]
  });

  // Opacities for the three main story segments
  const textOpacity1 = useTransform(scrollYProgress, [0, 0.1, 0.25, 0.35], [0, 1, 1, 0]);
  const textOpacity2 = useTransform(scrollYProgress, [0.35, 0.45, 0.6, 0.7], [0, 1, 1, 0]);
  const textOpacity3 = useTransform(scrollYProgress, [0.7, 0.8, 0.95, 1], [0, 1, 1, 0]);

  // Scaling/rotation for the background abstract data rings
  const ring1Scale = useTransform(scrollYProgress, [0, 1], [0.5, 2.5]);
  const ring1Rotate = useTransform(scrollYProgress, [0, 1], [0, 180]);
  const ring2Scale = useTransform(scrollYProgress, [0, 1], [0.8, 3.5]);
  const ring2Rotate = useTransform(scrollYProgress, [0, 1], [0, -180]);

  // Typewriter code clip paths mapping progress
  const codeClipPath = useTransform(
    scrollYProgress,
    [0.75, 0.9],
    ["inset(0 0 100% 0)", "inset(0 0 0% 0)"]
  );

  return (
    <section ref={containerRef} className="relative h-[400vh] bg-[#020617] text-white">
      <div className="sticky top-0 h-screen w-full flex flex-col items-center justify-center overflow-hidden">
        
        {/* Abstract graphical representation of pipeline */}
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

        {/* Global gradient to merge into dark theme */}
        <div className="absolute inset-0 bg-gradient-to-b from-[#020617] via-transparent to-[#020617] pointer-events-none" />

        {/* Segment 1: Syntax vs Semantics */}
        <motion.div className="absolute text-center max-w-4xl px-6 pointer-events-none" style={{ opacity: textOpacity1 }}>
          <h2 className="text-[clamp(2.5rem,5vw,5rem)] font-extrabold tracking-tighter mb-6 leading-[1.1]">
            Syntax is not <br className="md:hidden" /><span className="text-transparent bg-clip-text bg-gradient-to-r from-[var(--color-accent-gold)] to-white">Semantics.</span>
          </h2>
          <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 font-light mx-auto max-w-[40ch]">
            Traditional tools stop at the AST. We traverse deeper, translating your intent into a mathematical proof.
          </p>
        </motion.div>

        {/* Segment 2: Z3 Integration */}
        <motion.div className="absolute text-center max-w-4xl px-6 pointer-events-none" style={{ opacity: textOpacity2 }}>
          <h2 className="text-[clamp(2.5rem,5vw,5rem)] font-extrabold tracking-tighter mb-6 leading-[1.1]">
            Enter the <span className="text-transparent bg-clip-text bg-gradient-to-r from-[var(--color-accent-teal)] to-blue-400">Prover.</span>
          </h2>
          <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 font-light mx-auto max-w-[40ch]">
            Every constraint is extracted and formally verified. If your domain rules conflict, the compiler categorically refuses to build.
          </p>
        </motion.div>

        {/* Segment 3: Compilation & Code */}
        <motion.div className="absolute flex flex-col md:flex-row gap-12 items-center justify-between max-w-6xl px-6 w-full" style={{ opacity: textOpacity3 }}>
          <div className="flex-1 md:pr-12">
            <h2 className="text-[clamp(2.5rem,5vw,5rem)] font-extrabold tracking-tighter mb-6 leading-[1.1]">
              Compile to <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-teal-200">Certainty.</span>
            </h2>
            <p className="text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 font-light">
              We generate native binaries stripped of uncertainty. Your deployment runs exactly as proved, executing pure scientific intent.
            </p>
          </div>
          <div className="flex-1 w-full border border-white/10 rounded-2xl bg-black/60 backdrop-blur-xl p-8 overflow-hidden relative shadow-2xl">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-[var(--color-accent-teal)] to-emerald-400 opacity-50" />
            <motion.pre 
              className="text-xs md:text-sm font-mono text-emerald-400/90 leading-relaxed"
              style={{ clipPath: codeClipPath }}
            >
{`import Mathlib.Data.Real.Basic

theorem uncertainty_bounds (x : ℝ) :
  let confidence := compute_interval x
  confidence.lower > 0.95 := by
  -- Z3 Prover Output Extraction
  exact rigorous_bound.apply rfl
  
// Machine Codegen Phase
0x1000: mov rax, 0x1
0x1008: mov rcx, [rsp+0x8]
0x1010: mul rcx, 0x3F800000
0x1018: ret`}
            </motion.pre>
          </div>
        </motion.div>

      </div>
    </section>
  );
}
