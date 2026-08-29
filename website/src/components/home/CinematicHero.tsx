import { useEffect, useState } from 'react';

interface Props {
  localePath: (path: string) => string;
}

const audiences = [
  'Clinical and scientific research',
  'Data, ML, and AI systems',
  'Review, governance, and audit',
];

const proofPoints = [
  {
    label: 'Built by its own compiler',
    value: 'Self-hosted fixed point',
  },
  {
    label: 'Kretikos GPU',
    value: 'GPU compiler lane',
  },
  {
    label: 'Pressure',
    value: 'PBPK • climate • neuro',
  },
];

export function CinematicHero({ localePath }: Props) {
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setIsLoaded(true), 80);
    return () => window.clearTimeout(timer);
  }, []);

  const line1 = ['Scientific', 'and', 'AI', 'software'];
  const line2 = ['that', 'keeps', 'its', 'honesty', 'under', 'pressure.'];

  return (
    <section className="relative flex min-h-[100vh] w-full items-center justify-center overflow-hidden bg-[#020617]">
      <div
        className="absolute inset-0 h-full w-full scale-[1.05] transition-[opacity,transform] duration-[2200ms] ease-out"
        style={{
          opacity: isLoaded ? 1 : 0,
          transform: isLoaded ? 'scale(1.05)' : 'scale(1.1)',
        }}
      >
        <picture>
          <source srcSet="/assets/original-artworks/pappou_hero.webp" type="image/webp" />
          <img
            src="/assets/original-artworks/pappou_hero.png"
            alt="Sounio Heritage"
            className="pointer-events-none h-[110%] w-[110%] -ml-[5%] -mt-[5%] object-cover object-center opacity-40 mix-blend-luminosity"
            loading="eager"
            decoding="async"
            fetchPriority="high"
          />
        </picture>
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_0%,#020617_80%)]" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#020617] via-transparent to-transparent opacity-80" />
      </div>

      <div className="relative z-10 container flex flex-col items-center justify-center px-4 py-24 text-center">
        <div
          className="mb-8 transition-[opacity,transform] duration-1000 ease-out"
          style={{
            opacity: isLoaded ? 1 : 0,
            transform: isLoaded ? 'translateY(0)' : 'translateY(20px)',
            transitionDelay: '120ms',
          }}
        >
          <span className="rounded-full border border-white/10 bg-white/5 px-4 py-1.5 font-mono text-xs uppercase tracking-[0.2em] text-[var(--color-accent-gold-soft)] backdrop-blur-md">
            UNCERTAINTY • PROVENANCE • DECISION BOUNDARIES
          </span>
        </div>

        <h1 className="flex flex-col gap-2 text-[clamp(3rem,8vw,8rem)] font-extrabold leading-[0.9] tracking-tighter text-white">
          <div className="flex flex-wrap justify-center gap-x-[clamp(0.5rem,2vw,2rem)] overflow-hidden">
            {line1.map((word, index) => (
              <span
                key={`hero-1-${index}`}
                className="inline-block font-[var(--font-display)] transition-[opacity,transform,filter] duration-[1200ms] [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]"
                style={{
                  opacity: isLoaded ? 1 : 0,
                  transform: isLoaded ? 'translateY(0)' : 'translateY(50px)',
                  filter: isLoaded ? 'blur(0px)' : 'blur(10px)',
                  transitionDelay: `${index * 150 + 220}ms`,
                }}
              >
                {word}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap justify-center gap-x-[clamp(0.5rem,2vw,2rem)] overflow-hidden">
            {line2.map((word, index) => (
              <span
                key={`hero-2-${index}`}
                className="inline-block bg-gradient-to-r from-white via-white to-white/55 bg-clip-text text-transparent transition-[opacity,transform,filter] duration-[1200ms] [transition-timing-function:cubic-bezier(0.16,1,0.3,1)]"
                style={{
                  opacity: isLoaded ? 1 : 0,
                  transform: isLoaded ? 'translateY(0)' : 'translateY(50px)',
                  filter: isLoaded ? 'blur(0px)' : 'blur(10px)',
                  transitionDelay: `${(index + line1.length) * 150 + 220}ms`,
                }}
              >
                {word}
              </span>
            ))}
          </div>
        </h1>

        <p
          className="mt-8 max-w-[46ch] text-[clamp(1.08rem,2vw,1.42rem)] font-light leading-relaxed text-white/70 transition-opacity duration-[1400ms] ease-out"
          style={{
            opacity: isLoaded ? 1 : 0,
            transitionDelay: '1400ms',
          }}
        >
          Sounio is for teams building high-consequence systems where the result is not enough. The software must also preserve confidence, provenance, and the point where it should stop bluffing.
        </p>

        <div
          className="mt-8 flex flex-wrap items-center justify-center gap-3 transition-[opacity,transform] duration-1000 ease-out"
          style={{
            opacity: isLoaded ? 1 : 0,
            transform: isLoaded ? 'translateY(0)' : 'translateY(24px)',
            transitionDelay: '1600ms',
          }}
        >
          {audiences.map((audience) => (
            <span
              key={audience}
              className="rounded-full border border-white/12 bg-white/[0.035] px-4 py-2 text-[0.76rem] uppercase tracking-[0.14em] text-white/72 backdrop-blur-md"
            >
              {audience}
            </span>
          ))}
        </div>

        <div
          className="mt-10 flex flex-wrap items-center justify-center gap-6 transition-[opacity,transform] duration-1000 ease-out"
          style={{
            opacity: isLoaded ? 1 : 0,
            transform: isLoaded ? 'translateY(0)' : 'translateY(20px)',
            transitionDelay: '1780ms',
          }}
        >
          <a
            href={localePath('/science')}
            className="group relative overflow-hidden rounded-full bg-white px-8 py-4 font-semibold text-black transition-transform duration-300 hover:scale-105"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-[var(--color-accent-gold)] to-[var(--color-accent-teal)] opacity-0 transition-opacity duration-500 group-hover:opacity-20" />
            <span className="relative z-10">Enter the proof surfaces</span>
          </a>

          <a
            href={localePath('/learn/getting-started')}
            className="rounded-full border border-white/20 px-8 py-4 font-medium text-white backdrop-blur-sm transition-colors duration-300 hover:bg-white/5"
          >
            Read the current contract
          </a>
        </div>

        <div
          className="mt-12 grid w-full max-w-4xl gap-3 md:grid-cols-3 transition-[opacity,transform] duration-1000 ease-out"
          style={{
            opacity: isLoaded ? 1 : 0,
            transform: isLoaded ? 'translateY(0)' : 'translateY(20px)',
            transitionDelay: '1900ms',
          }}
        >
          {proofPoints.map((point) => (
            <div
              key={point.label}
              className="rounded-[1.2rem] border border-white/10 bg-black/15 px-4 py-4 text-left backdrop-blur-md"
            >
              <p className="text-[0.7rem] uppercase tracking-[0.14em] text-white/45">{point.label}</p>
              <p className="mt-2 text-[0.98rem] font-medium text-white/84">{point.value}</p>
            </div>
          ))}
        </div>
      </div>

      <div
        className="absolute bottom-12 left-1/2 flex -translate-x-1/2 flex-col items-center gap-3 opacity-50 transition-opacity duration-1000 ease-out"
        style={{ opacity: isLoaded ? 0.5 : 0, transitionDelay: '2100ms' }}
      >
        <span className="text-[10px] uppercase tracking-[0.3em] text-white/50">Scroll</span>
        <div className="h-16 w-px origin-top animate-pulse bg-gradient-to-b from-white/50 to-transparent" />
      </div>

      <div
        className="pointer-events-none absolute inset-0 z-50 bg-[#020617] transition-opacity duration-700 ease-in-out"
        style={{ opacity: isLoaded ? 0 : 1 }}
      />
    </section>
  );
}
