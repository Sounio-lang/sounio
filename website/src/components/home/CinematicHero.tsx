import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  localePath: (path: string) => string;
}

export function CinematicHero({ localePath }: Props) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    // Small delay to ensure the browser has painted the initial frame
    const timer = setTimeout(() => {
      setIsLoaded(true);
    }, 100);

    const handleMouseMove = (e: MouseEvent) => {
      // Normalize mouse position between -1 and 1
      setMousePosition({
        x: (e.clientX / window.innerWidth) * 2 - 1,
        y: (e.clientY / window.innerHeight) * 2 - 1
      });
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      clearTimeout(timer);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  const textVariants = {
    hidden: { opacity: 0, y: 50, filter: 'blur(10px)' },
    visible: (custom: number) => ({
      opacity: 1,
      y: 0,
      filter: 'blur(0px)',
      transition: {
        delay: custom * 0.15 + 0.8,
        duration: 1.2,
        ease: [0.16, 1, 0.3, 1], // Custom cubic-bezier for cinematic feel
      },
    }),
  };

  const line1 = "Scientific Code".split(" ");
  const line2 = "That Shows Its Work".split(" ");

  return (
    <section className="relative h-[100vh] w-full overflow-hidden bg-[#020617] flex items-center justify-center">
      {/* Background Media */}
      <motion.div 
        className="absolute inset-0 w-full h-full scale-[1.05]"
        initial={{ scale: 1.1, opacity: 0 }}
        animate={{ 
          scale: 1.05, 
          opacity: isLoaded ? 1 : 0,
          x: mousePosition.x * -25,
          y: mousePosition.y * -25
        }}
        transition={{ 
          opacity: { duration: 2.5, ease: "easeOut", delay: 0.2 },
          scale: { duration: 2.5, ease: "easeOut", delay: 0.2 },
          x: { type: "spring", stiffness: 40, damping: 25 },
          y: { type: "spring", stiffness: 40, damping: 25 }
        }}
      >
        <picture>
          <source srcSet="/assets/original-artworks/pappou_hero.webp" type="image/webp" />
          <img 
            src="/assets/original-artworks/pappou_hero.png" 
            alt="Sounio Heritage" 
            className="w-[110%] h-[110%] -ml-[5%] -mt-[5%] object-cover object-center opacity-40 mix-blend-luminosity pointer-events-none"
          />
        </picture>
        {/* Radial Vignette Mask */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,#020617_80%)] pointer-events-none" />
        {/* Linear Gradient for Text Readability */}
        <div className="absolute inset-0 bg-gradient-to-t from-[#020617] via-transparent to-transparent opacity-80 pointer-events-none" />
      </motion.div>

      {/* Content Container */}
      <div className="relative z-10 container flex flex-col items-center justify-center text-center px-4">
        
        {/* Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
          transition={{ delay: 0.6, duration: 1, ease: "easeOut" }}
          className="mb-8"
        >
          <span className="px-4 py-1.5 rounded-full border border-white/10 bg-white/5 backdrop-blur-md text-xs font-mono tracking-[0.2em] text-[var(--color-accent-gold-soft)] uppercase">
            Current Contract · Beta Compiler + Gated Stdlib
          </span>
        </motion.div>

        {/* Massive Typography */}
        <h1 className="text-[clamp(3rem,8vw,8rem)] leading-[0.9] font-extrabold tracking-tighter text-white flex flex-col gap-2">
          <div className="overflow-hidden flex flex-wrap justify-center gap-x-[clamp(0.5rem,2vw,2rem)]">
            {line1.map((word, i) => (
              <motion.span
                key={i}
                custom={i}
                variants={textVariants}
                initial="hidden"
                animate={isLoaded ? "visible" : "hidden"}
                className="inline-block"
              >
                {word}
              </motion.span>
            ))}
          </div>
          <div className="overflow-hidden flex flex-wrap justify-center gap-x-[clamp(0.5rem,2vw,2rem)]">
            {line2.map((word, i) => (
              <motion.span
                key={i}
                custom={i + line1.length}
                variants={textVariants}
                initial="hidden"
                animate={isLoaded ? "visible" : "hidden"}
                className="inline-block bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-white/50"
              >
                {word}
              </motion.span>
            ))}
          </div>
        </h1>

        {/* Subtitle */}
        <motion.p 
          className="mt-10 text-[clamp(1.1rem,2vw,1.5rem)] text-white/60 max-w-[42ch] font-light leading-relaxed"
          initial={{ opacity: 0 }}
          animate={{ opacity: isLoaded ? 1 : 0 }}
          transition={{ delay: 2.2, duration: 1.5 }}
        >
          Sounio combines explicit uncertainty, provenance-aware types, and gate-backed compiler workflows. The current repo is strongest as a check-first platform with validated scientific lanes, not as a fully finished all-backends release.
        </motion.p>

        {/* Actions */}
        <motion.div 
          className="mt-12 flex flex-wrap items-center justify-center gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: isLoaded ? 1 : 0, y: isLoaded ? 0 : 20 }}
          transition={{ delay: 2.5, duration: 1, ease: "easeOut" }}
        >
          <a
            href={localePath('/learn/getting-started')}
            className="group relative px-8 py-4 bg-white text-black font-semibold rounded-full overflow-hidden transition-transform hover:scale-105 duration-300"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-[var(--color-accent-gold)] to-[var(--color-accent-teal)] opacity-0 group-hover:opacity-20 transition-opacity duration-500" />
            <span className="relative z-10">Read the Current Contract</span>
          </a>
          
          <a
            href={localePath('/playground')}
            className="px-8 py-4 text-white font-medium border border-white/20 rounded-full hover:bg-white/5 transition-colors duration-300 backdrop-blur-sm"
          >
            Open Playground
          </a>
        </motion.div>
      </div>

      {/* Scroll Indicator */}
      <motion.div 
        className="absolute bottom-12 left-1/2 -translate-x-1/2 flex flex-col items-center gap-3 opacity-50"
        initial={{ opacity: 0 }}
        animate={{ opacity: isLoaded ? 0.5 : 0 }}
        transition={{ delay: 3, duration: 1 }}
      >
        <span className="text-[10px] uppercase tracking-[0.3em] font-mono text-white/50">Scroll</span>
        <motion.div 
          className="w-px h-16 bg-gradient-to-b from-white/50 to-transparent origin-top"
          animate={{ scaleY: [0, 1, 0], opacity: [0, 1, 0], y: [0, 20, 40] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        />
      </motion.div>

      {/* Initial Solid Mask */}
      <AnimatePresence>
        {!isLoaded && (
          <motion.div 
            className="absolute inset-0 bg-[#020617] z-50"
            exit={{ opacity: 0 }}
            transition={{ duration: 0.8, ease: "easeInOut" }}
          />
        )}
      </AnimatePresence>
    </section>
  );
}
