import { useEffect, useRef, type ReactNode } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
gsap.registerPlugin(ScrollTrigger);

export interface SimulatorShellProps {
  title: string;
  description: string;
  controls: ReactNode;
  visualization: ReactNode;
  logLines: { text: string; type: 'info' | 'warn' | 'error' | 'success' }[];
  isHalted: boolean;
}

export function SimulatorShell({
  title,
  description,
  controls,
  visualization,
  logLines,
  isHalted,
}: SimulatorShellProps) {
  const shellRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = shellRef.current;
    if (!el) return;
    const mm = gsap.matchMedia();
    mm.add('(prefers-reduced-motion: no-preference)', () => {
      gsap.from(el, {
        y: 36, opacity: 0, duration: 0.8, ease: 'power2.out',
        scrollTrigger: { trigger: el, start: 'top 82%', once: true },
      });
    });
    return () => mm.revert();
  }, []);

  return (
    <div ref={shellRef} className="simulator-shell glass glass-specular rounded-[var(--radius-xl)] overflow-hidden flex flex-col mt-[3rem] mb-[2rem] border border-[var(--glass-border)] relative z-10">
      <div className="p-[1.5rem] border-b border-[var(--glass-border)] bg-[rgba(255,255,255,0.02)]">
        <h3 className="text-[1.25rem] font-[600] text-[var(--color-text-primary)] mb-[0.25rem]">
          {title}
        </h3>
        <p className="text-[var(--color-text-secondary)] text-[0.95rem]">
          {description}
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[280px_1fr] flex-grow">
        <div className="p-[1.5rem] border-r border-[var(--glass-border)] bg-[rgba(0,0,0,0.15)] flex flex-col gap-6">
          {controls}
        </div>

        <div className="p-[1.5rem] bg-[rgba(0,0,0,0.05)] relative min-h-[300px] flex items-center justify-center overflow-hidden">
          {isHalted && (
            <div className="absolute inset-0 z-10 flex items-center justify-center bg-[var(--color-bg)]/60 backdrop-blur-sm transition-all duration-300">
              <div className="flex flex-col items-center gap-3">
                <div className="w-16 h-16 rounded-full bg-red-500/20 flex items-center justify-center border border-red-500/50">
                  <svg className="w-8 h-8 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <span className="text-red-500 font-bold tracking-widest uppercase text-sm">Execution Halted</span>
              </div>
            </div>
          )}
          {visualization}
        </div>
      </div>

      <div className="p-[1rem] bg-[rgba(0,0,0,0.4)] border-t border-[var(--glass-border)] font-mono text-[0.85rem] h-[160px] overflow-y-auto">
        <div className="flex items-center gap-2 mb-3 text-[var(--color-text-tertiary)] text-[0.75rem] uppercase tracking-wider">
          <span className="w-2 h-2 rounded-full bg-[var(--color-accent-green)] animate-pulse"></span>
          Sounio WASM Runtime
        </div>
        <div className="flex flex-col gap-1">
          {logLines.map((line, i) => (
            <div
              key={i}
              className={`
                ${line.type === 'error' ? 'text-red-400' : ''}
                ${line.type === 'warn' ? 'text-amber-400' : ''}
                ${line.type === 'success' ? 'text-[var(--color-accent-green)]' : ''}
                ${line.type === 'info' ? 'text-[var(--color-text-secondary)]' : ''}
              `}
            >
              <span className="text-[var(--color-text-tertiary)] mr-2">&gt;</span>
              {line.text}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
