import { useState } from 'react';

type Seg = { t: 'plain'; s: string } | { t: 'token'; key: string; display: string };

type Ann = { title: string; body: string; color: string };

const ANNOTATIONS: Record<string, Ann> = {
  'Knowledge<mg>': {
    title: 'Epistemic type',
    body: 'A generic wrapper that carries a value, a confidence ε ∈ [0,1], and a provenance tag. GUM rules propagate ε automatically through every arithmetic operation — you never lose track of how uncertain your data is.',
    color: 'var(--color-accent-gold)',
  },
  'ε=0.99': {
    title: 'Confidence scalar',
    body: 'Epistemic confidence following ISO GUM (Guide to the Expression of Uncertainty in Measurement). ε = 1.0 is perfect certainty; ε = 0.0 is total ignorance. The compiler tracks this through every calculation.',
    color: 'var(--color-accent-gold)',
  },
  'prov="iv_bolus_ref"': {
    title: 'Provenance tag',
    body: 'A compile-time string identifying the measurement origin. In a safety-critical pipeline, you can always trace any output back to its raw inputs — sensor ID, instrument, protocol, timestamp.',
    color: 'var(--color-accent-teal)',
  },
  'where c.ε >= 0.85': {
    title: 'Compile-time confidence gate',
    body: 'A type constraint that the compiler evaluates statically. If the caller passes a Knowledge value with ε < 0.85, the type-checker rejects the program. Not a lint, not a runtime warning — the binary is never produced.',
    color: '#ef4444',
  },
  'with IO': {
    title: 'Effect annotation',
    body: 'Declares that this function performs I/O. Sounio tracks algebraic effects in the type signature — a pure function can never accidentally call an I/O-effectful one without the type system catching it.',
    color: 'var(--color-accent-purple)',
  },
  'algebra Octonion': {
    title: 'First-class algebra declaration',
    body: 'Declares a named algebraic structure with its laws. The compiler uses these laws to constrain the e-graph optimizer — only rewrites that respect the declared laws are ever applied.',
    color: 'var(--color-accent-gold)',
  },
  'non_commutative': {
    title: 'Commutativity law',
    body: 'Tells the optimizer that e₁·e₂ ≠ e₂·e₁ — argument order cannot be swapped. This prevents incorrect vectorization or code motion across non-commutative products like quaternion or octonion multiplication.',
    color: 'var(--color-accent-teal)',
  },
  'fano_selective': {
    title: 'Fano-plane constraint',
    body: 'Restricts reassociation to the 175 associative triples. The 168 triples lying on Fano lines are non-associative and the optimizer never reorders them. The first compiler with rewrite rules constrained by projective plane geometry.',
    color: 'var(--color-accent-gold)',
  },
  'linear struct': {
    title: 'Linear type',
    body: 'A value that must be used exactly once. The compiler statically prevents double-free, use-after-free, and forgotten resource cleanup — without a garbage collector or runtime overhead.',
    color: '#f59e0b',
  },
  'fd: i32': {
    title: 'File descriptor field',
    body: 'A raw integer wrapped in a linear struct. The linearity guarantee means the OS resource is always released — even across error paths, without try/finally or RAII destructors.',
    color: 'var(--color-accent-teal)',
  },
};

// Code represented as a flat segment list so we can render tokens inline
const SEGMENTS: Seg[] = [
  { t: 'plain', s: '// 1. Epistemic measurements\nlet dose: ' },
  { t: 'token', key: 'Knowledge<mg>', display: 'Knowledge<mg>' },
  { t: 'plain', s: ' = Knowledge(\n  5.0, ' },
  { t: 'token', key: 'ε=0.99', display: 'ε=0.99' },
  { t: 'plain', s: ', ' },
  { t: 'token', key: 'prov="iv_bolus_ref"', display: 'prov="iv_bolus_ref"' },
  { t: 'plain', s: '\n)\n\n// 2. Compile-time confidence gate\nfn administer(c: Knowledge<mg_per_L>)\n  ' },
  { t: 'token', key: 'where c.ε >= 0.85', display: 'where c.ε ≥ 0.85' },
  { t: 'plain', s: ' ' },
  { t: 'token', key: 'with IO', display: 'with IO' },
  { t: 'plain', s: ' {\n  // gate passed at compile time — safe to run\n}\n\n// 3. Non-associative algebra\n' },
  { t: 'token', key: 'algebra Octonion', display: 'algebra Octonion' },
  { t: 'plain', s: ' over f64 {\n  mul: alternative, ' },
  { t: 'token', key: 'non_commutative', display: 'non_commutative' },
  { t: 'plain', s: '\n  reassociate: ' },
  { t: 'token', key: 'fano_selective', display: 'fano_selective' },
  { t: 'plain', s: '\n}\n\n// 4. Linear resource types\n' },
  { t: 'token', key: 'linear struct', display: 'linear struct' },
  { t: 'plain', s: ' Handle {\n  ' },
  { t: 'token', key: 'fd: i32', display: 'fd: i32' },
  { t: 'plain', s: '\n}  // compiler guarantees: used exactly once' },
];

const ORDERED_KEYS = [
  'Knowledge<mg>', 'ε=0.99', 'prov="iv_bolus_ref"',
  'where c.ε >= 0.85', 'with IO',
  'algebra Octonion', 'non_commutative', 'fano_selective',
  'linear struct', 'fd: i32',
];

export default function AnnotatedCode() {
  const [active, setActive] = useState<string | null>(null);

  const ann = active ? ANNOTATIONS[active] : null;

  return (
    <section className="py-[clamp(3rem,6vw,5rem)] bg-[var(--color-bg-alt)]">
      <div className="container px-4">
        <div className="mb-[2rem] grid gap-[0.5rem] max-w-[52ch]">
          <h2 className="font-sans text-[clamp(1.5rem,3.5vw,2.4rem)] font-[750] leading-[1.15] tracking-[-0.02em] text-[var(--color-text-primary)]">
            Click anything you don't understand.
          </h2>
          <p className="text-[0.9rem] text-[var(--color-text-secondary)] leading-[1.6]">
            Every Sounio-specific token below is interactive. The code is real — it compiles.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
          {/* Code block */}
          <div className="rounded-2xl overflow-hidden border border-[var(--glass-border)] bg-[#0d1117]">
            <div className="flex items-center gap-2 px-4 py-3 bg-[rgba(255,255,255,0.03)] border-b border-[var(--glass-border)]">
              <div className="flex gap-1.5">
                <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
                <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
                <div className="w-3 h-3 rounded-full bg-[#28c840]" />
              </div>
              <span className="flex-1 text-center text-[var(--color-text-tertiary)] text-[0.8rem] font-mono">
                sounio_in_90_seconds.sio
              </span>
              <span className="text-[0.72rem] text-[var(--color-text-tertiary)] font-mono opacity-60">
                click highlighted tokens
              </span>
            </div>
            <pre className="p-6 text-[0.82rem] leading-[1.9] overflow-x-auto font-mono">
              {SEGMENTS.map((seg, i) => {
                if (seg.t === 'plain') {
                  return (
                    <span key={i} className="text-[#8b949e] whitespace-pre">
                      {seg.s}
                    </span>
                  );
                }
                const isActive = active === seg.key;
                const color = ANNOTATIONS[seg.key]?.color ?? 'var(--color-accent-gold)';
                return (
                  <button
                    key={i}
                    onClick={() => setActive(isActive ? null : seg.key)}
                    className={`relative font-mono text-[0.82rem] leading-[1.9] rounded px-[2px] -mx-[2px] transition-all cursor-pointer border-b-2 ${
                      isActive
                        ? 'bg-[rgba(255,255,255,0.1)]'
                        : 'hover:bg-[rgba(255,255,255,0.06)]'
                    }`}
                    style={{
                      color: isActive ? '#ffffff' : color,
                      borderBottomColor: isActive ? color : 'transparent',
                    }}
                  >
                    {seg.display}
                  </button>
                );
              })}
            </pre>
          </div>

          {/* Annotation panel */}
          <div className="flex flex-col gap-4">
            {/* Active annotation */}
            <div
              className={`glass rounded-2xl p-5 grid gap-3 transition-all duration-200 ${
                ann ? 'opacity-100' : 'opacity-40'
              }`}
              style={{ borderLeft: ann ? `3px solid ${ann.color}` : '3px solid rgba(255,255,255,0.1)' }}
            >
              {ann ? (
                <>
                  <div className="text-[0.72rem] font-[700] uppercase tracking-[0.1em]" style={{ color: ann.color }}>
                    {ann.title}
                  </div>
                  <p className="text-[0.87rem] text-[var(--color-text-secondary)] leading-[1.65]">
                    {ann.body}
                  </p>
                </>
              ) : (
                <>
                  <div className="text-[0.72rem] font-[700] uppercase tracking-[0.1em] text-[var(--color-text-tertiary)]">
                    No token selected
                  </div>
                  <p className="text-[0.87rem] text-[var(--color-text-tertiary)] leading-[1.65]">
                    Click any highlighted token in the code block to see what it does and why it matters.
                  </p>
                </>
              )}
            </div>

            {/* Token index */}
            <div className="glass rounded-2xl p-4 grid gap-2">
              <div className="text-[0.72rem] font-[700] uppercase tracking-[0.1em] text-[var(--color-text-tertiary)] mb-1">
                All tokens ({ORDERED_KEYS.length})
              </div>
              {ORDERED_KEYS.map(key => {
                const a = ANNOTATIONS[key];
                const isActive = active === key;
                return (
                  <button
                    key={key}
                    onClick={() => setActive(isActive ? null : key)}
                    className={`flex items-start gap-2.5 w-full text-left rounded-lg px-2.5 py-2 transition-all ${
                      isActive ? 'bg-[rgba(255,255,255,0.07)]' : 'hover:bg-[rgba(255,255,255,0.04)]'
                    }`}
                  >
                    <span
                      className="shrink-0 mt-[3px] w-2 h-2 rounded-full"
                      style={{ background: isActive ? a.color : 'rgba(255,255,255,0.2)' }}
                    />
                    <div>
                      <div className="text-[0.76rem] font-mono leading-none mb-[3px]"
                        style={{ color: isActive ? a.color : 'var(--color-text-secondary)' }}>
                        {key}
                      </div>
                      <div className="text-[0.72rem] text-[var(--color-text-tertiary)] leading-none">
                        {a.title}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
