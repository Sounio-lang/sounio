import { useEffect, useRef, useState } from 'react';
import { publicContract } from '../../data/artifactStatus';

const { bootstrap } = publicContract;
const hashLabel = bootstrap.recordedSha256
  ? `SHA-256 ${bootstrap.recordedSha256.slice(0, 12)}…`
  : 'gen2 == gen3 parity verified';

const STAGES = [
  {
    id: 'source',
    label: 'Source',
    sub: '.sio file',
    color: 'var(--color-accent-gold)',
    icon: '{ }',
    desc: 'UTF-8 text with Sounio syntax. Units, effects, Knowledge<T>, algebra declarations, linear types — all in one file.',
    example: 'let dose: Knowledge<mg> = …',
  },
  {
    id: 'lexer',
    label: 'Lexer',
    sub: 'tokenizer',
    color: 'var(--color-accent-teal)',
    icon: 'T',
    desc: 'Hand-written tokenizer. Produces a flat token stream with byte-offset spans. Unicode-aware; Greek letters (ε, α, Ω) are valid identifiers.',
    example: 'Knowledge · < · mg · > · = · …',
  },
  {
    id: 'parser',
    label: 'Parser',
    sub: 'recursive descent',
    color: 'var(--color-accent-gold)',
    icon: '⊢',
    desc: 'Recursive-descent parser with no backtracking. Produces a typed AST with full source locations for every node — used by the error reporter and LSP.',
    example: 'LetDecl { name, ty: KnowledgeTy, …}',
  },
  {
    id: 'check',
    label: 'Type Check',
    sub: 'bidirectional + effects',
    color: '#ef4444',
    icon: 'ε',
    desc: 'Bidirectional type inference with algebraic effect rows. Evaluates confidence gates (where c.ε ≥ 0.85) statically. Rejects programs at this stage — before codegen ever runs.',
    example: 'ERROR: ε=0.71 < gate 0.85 at call site',
  },
  {
    id: 'hir',
    label: 'HIR',
    sub: 'high-level IR',
    color: 'var(--color-accent-teal)',
    icon: '◇',
    desc: 'Desugared, monomorphised IR. Epistemic metadata (ε, provenance) is still structurally attached. This is where trait resolution and algebra law-binding happens.',
    example: 'HirCall { callee, args, ε_bound: 0.85 }',
  },
  {
    id: 'sir',
    label: 'SIR / HLIR',
    sub: 'SSA + e-graph',
    color: 'var(--color-accent-purple)',
    icon: '⟳',
    desc: 'Static single-assignment form. The e-graph optimizer applies 1,000+ rewrite rules — constrained by declared algebra laws (fano_selective, non_commutative). All rules have FAIL=0.',
    example: '(x & C) | (x & ~C) → x  [pass]',
  },
  {
    id: 'codegen',
    label: 'Codegen',
    sub: 'x86-64',
    color: 'var(--color-accent-gold)',
    icon: '⚙',
    desc: 'Register allocation + instruction selection targeting x86-64 System V ABI via the self-hosted native backend. No LLVM required for the default bin/souc workflow.',
    example: 'mov rax, [rsp+8]  ; load Knowledge value',
  },
  {
    id: 'elf',
    label: 'ELF Binary',
    sub: 'verified fixed point',
    color: '#22c55e',
    icon: '✓',
    desc: `Self-contained x86-64 ELF. The compiler compiles itself: ${bootstrap.fixedPointChain}. Recorded ${hashLabel}. A fixed-point self-hosting chain is the strongest evidence of semantic consistency.`,
    example: 'sha256(gen2.elf) == sha256(gen3.elf)',
  },
];

export default function CompilerPipeline() {
  const [visible, setVisible] = useState<Set<string>>(new Set());
  const [active, setActive] = useState<string>('source');
  const refs = useRef<Map<string, HTMLDivElement>>(new Map());

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            const id = (e.target as HTMLElement).dataset.id;
            if (id) setVisible(prev => new Set([...prev, id]));
          }
        });
      },
      { threshold: 0.25 }
    );
    refs.current.forEach(el => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const activeStage = STAGES.find(s => s.id === active)!;

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[var(--color-bg)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem] max-w-[56ch]">
          <h2 className="font-sans text-[clamp(1.5rem,3.8vw,2.6rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            How the compiler works.
          </h2>
          <p className="text-[0.9rem] text-[var(--color-text-secondary)] leading-[1.65]">
            Eight stages, all written in Sounio. Click any stage to see what it does.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-[1fr_380px]">

          {/* Pipeline column */}
          <div className="relative">
            {/* Connecting spine */}
            <div
              className="absolute left-[27px] top-8 bottom-8 w-[2px]"
              style={{ background: 'linear-gradient(to bottom, var(--color-accent-gold), #22c55e)' }}
              aria-hidden="true"
            />

            <div className="grid gap-3">
              {STAGES.map((stage, i) => {
                const isVisible = visible.has(stage.id);
                const isActive = active === stage.id;
                return (
                  <div
                    key={stage.id}
                    data-id={stage.id}
                    ref={el => { if (el) refs.current.set(stage.id, el); }}
                    className="transition-all duration-500"
                    style={{
                      opacity: isVisible ? 1 : 0,
                      transform: isVisible ? 'translateX(0)' : 'translateX(-24px)',
                      transitionDelay: `${i * 60}ms`,
                    }}
                  >
                    <button
                      onClick={() => setActive(stage.id)}
                      className={`w-full flex items-center gap-4 rounded-xl px-4 py-3 text-left transition-all ${
                        isActive
                          ? 'bg-[rgba(255,255,255,0.07)] border border-[rgba(255,255,255,0.12)]'
                          : 'hover:bg-[rgba(255,255,255,0.04)] border border-transparent'
                      }`}
                    >
                      {/* Node circle */}
                      <div
                        className="shrink-0 w-[54px] h-[54px] rounded-full flex items-center justify-center text-[0.75rem] font-mono font-bold border-2 transition-all z-10 relative bg-[var(--color-bg)]"
                        style={{
                          borderColor: isActive ? stage.color : 'rgba(255,255,255,0.15)',
                          color: isActive ? stage.color : 'var(--color-text-tertiary)',
                        }}
                      >
                        {stage.icon}
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-baseline gap-2">
                          <span
                            className="text-[0.95rem] font-[650] transition-colors"
                            style={{ color: isActive ? stage.color : 'var(--color-text-primary)' }}
                          >
                            {stage.label}
                          </span>
                          <span className="text-[0.73rem] text-[var(--color-text-tertiary)] font-mono">
                            {stage.sub}
                          </span>
                        </div>
                        <div className="text-[0.78rem] text-[var(--color-text-tertiary)] mt-[2px] truncate">
                          {stage.example}
                        </div>
                      </div>

                      {/* Stage number */}
                      <span className="shrink-0 text-[0.72rem] font-mono text-[var(--color-text-tertiary)] opacity-50">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Detail panel */}
          <div className="lg:sticky lg:top-[5rem] h-fit">
            <div
              className="glass glass-specular rounded-2xl p-6 grid gap-4 transition-all duration-300"
              style={{ borderLeft: `3px solid ${activeStage.color}` }}
            >
              <div>
                <div
                  className="text-[0.72rem] font-[700] uppercase tracking-[0.12em] mb-2"
                  style={{ color: activeStage.color }}
                >
                  Stage {STAGES.findIndex(s => s.id === active) + 1} of {STAGES.length}
                </div>
                <div className="text-[1.3rem] font-[700] text-[var(--color-text-primary)] leading-[1.2]">
                  {activeStage.label}
                </div>
                <div className="text-[0.8rem] text-[var(--color-text-tertiary)] font-mono mt-[2px]">
                  {activeStage.sub}
                </div>
              </div>

              <p className="text-[0.88rem] text-[var(--color-text-secondary)] leading-[1.7]">
                {activeStage.desc}
              </p>

              <div className="rounded-lg bg-[rgba(0,0,0,0.4)] px-4 py-3 font-mono text-[0.78rem]"
                style={{ color: activeStage.color }}>
                {activeStage.example}
              </div>

              {/* Navigation */}
              <div className="flex gap-2 pt-2">
                <button
                  onClick={() => {
                    const i = STAGES.findIndex(s => s.id === active);
                    if (i > 0) setActive(STAGES[i - 1].id);
                  }}
                  disabled={active === STAGES[0].id}
                  className="flex-1 rounded-lg border border-[var(--glass-border)] py-2 text-[0.78rem] text-[var(--color-text-secondary)] hover:bg-[rgba(255,255,255,0.05)] disabled:opacity-30 disabled:cursor-default transition-all"
                >
                  ← Prev
                </button>
                <button
                  onClick={() => {
                    const i = STAGES.findIndex(s => s.id === active);
                    if (i < STAGES.length - 1) setActive(STAGES[i + 1].id);
                  }}
                  disabled={active === STAGES[STAGES.length - 1].id}
                  className="flex-1 rounded-lg border border-[var(--glass-border)] py-2 text-[0.78rem] text-[var(--color-text-secondary)] hover:bg-[rgba(255,255,255,0.05)] disabled:opacity-30 disabled:cursor-default transition-all"
                >
                  Next →
                </button>
              </div>
            </div>

            {/* Self-hosting badge */}
            <div className="mt-4 glass rounded-xl px-4 py-3 flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-[#22c55e] animate-pulse shrink-0" />
              <p className="text-[0.78rem] text-[var(--color-text-secondary)] leading-[1.5]">
                The entire pipeline is written in Sounio.{' '}
                <span className="text-[var(--color-text-primary)] font-[600]">
                  stage2 == stage3
                </span>
                {' '}— byte-identical self-compilation verified.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
