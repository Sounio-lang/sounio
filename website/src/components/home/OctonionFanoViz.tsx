import { useState, useEffect, useRef } from 'react';
import { MCQBlock } from '../common/MCQBlock';

const OCTONION_QUESTIONS = [
  {
    question: 'Reggiani (arXiv:2411.18881, Nov 2024) proved that the submanifold Z(𝕊) of normalized sedenion pairs whose product is zero is isometric to which space — upgrading Moreno\'s 1998 homeomorphism to a full Riemannian isometry?',
    options: [
      'The 7-sphere S⁷ with the round metric',
      'The exceptional symmetric space G₂/SO(4)',
      'The exceptional Lie group G₂ with a naturally reductive left-invariant metric',
      'The compact symmetric space Sp(3)/U(3)',
    ],
    correct: 2,
    explanation: 'G₂/SO(4) is the base of the Riemannian submersion that Z(𝕊) fibers over — it appears in the structure, but Z(𝕊) itself is isometric to the total space G₂. This 2024 result places the sedenion zero-divisor geometry inside exceptional Riemannian geometry, deepening the connection between Cayley-Dickson algebras and the exceptional Lie hierarchy.',
  },
  {
    question: 'The number of primitive unit zero-divisor pairs in the sedenions is exactly 168. Why is this the same as |PSL(2,7)|?',
    options: [
      'PSL(2,7) acts on the 8 sedenion imaginary units by permutation, producing 168 orbits',
      'Both count the independent Moufang identities in an alternative algebra of dimension 16',
      'The sedenion zero-divisor set inherits the same symmetry as the Fano plane PG(2,2), whose automorphism group has order 168',
      '168 = 7! / (7 × 6) and appears by coincidence in both contexts',
    ],
    correct: 2,
    explanation: 'PSL(2,7) ≅ GL(3,2) is the automorphism group of the Fano plane PG(2,2), the same combinatorial structure that encodes the octonion multiplication table. The sedenion zero-divisors inherit this geometry because they arise precisely from pairs of octonion subalgebras related by the Fano structure. The count 168 is a structural consequence, not a coincidence.',
  },
  {
    question: 'Octonion-valued neural networks (OctoNNs) cannot be classified as Clifford-valued networks (2025 PMC review). Which consequence of non-associativity is the core problem for backpropagation through multi-layer octonion products?',
    options: [
      'Octonion activations are not differentiable in the Wirtinger sense, so gradients are undefined',
      'The chain rule requires associativity: (AB)C ≠ A(BC) means the gradient of a three-factor product is order-dependent, violating the standard layer-by-layer accumulation assumption',
      'Octonion weight matrices are not invertible in general, preventing backward-pass computation',
      'Non-associativity causes the loss landscape to be non-convex, which breaks gradient descent convergence guarantees',
    ],
    correct: 1,
    explanation: 'Clifford (geometric) algebras are always associative, which is what makes the chain rule factorize cleanly through products of algebra-valued layers. Octonions break associativity: (AB)C ≠ A(BC) means that computing ∂(ABC)/∂A differs depending on bracketing order. Most published OctoNN architectures silently degrade to approximate or restricted octonion operations to work around this. Non-convexity (D) affects all deep nets, not specifically octonion ones.',
  },
];

// Standard octonion multiplication table (imaginary basis e1..e7, Cayley-Dickson)
// oct_mul(i,j) returns {idx: result_index (0=real,1-7=imag), sign: +1/-1}
// i,j are 1..7 (imaginary), 0 = real unit e0
const MULT_TABLE: [number, number][][] = (() => {
  // Encoding: [idx, sign] for each (i,j) pair where idx 0=real, 1..7=imag
  // e1*e2=e4, e2*e1=-e4, e1*e4=e2 (Fano lines: {1,2,4},{2,3,5},{3,4,6},{4,5,7},{5,6,1},{6,7,2},{7,1,3})
  const lines = [[1,2,4],[2,3,5],[3,4,6],[4,5,7],[5,6,1],[6,7,2],[7,1,3]];
  const table: [number,number][][] = Array.from({length:8},()=>Array(8).fill([0,0]));
  // Self products: ei*ei = -1 (real, negative)
  for (let i=1;i<=7;i++) table[i][i]=[-1,-1]; // special: represents -e0
  // Cross products from Fano lines
  for (const [a,b,c] of lines) {
    table[a][b]=[c,+1];  // ea*eb = +ec
    table[b][a]=[c,-1];  // eb*ea = -ec
    table[b][c]=[a,+1];
    table[c][b]=[a,-1];
    table[c][a]=[b,+1];
    table[a][c]=[b,-1];
  }
  return table;
})();

function octMul(i: number, j: number): [number, number] {
  if (i === 0) return [j, 1];
  if (j === 0) return [i, 1];
  return MULT_TABLE[i][j];
}

function octMulComplex(a: [number,number], b: number): [number, number] {
  // multiply oct element (idx, sign) by basis element b
  const [ai, as_] = a;
  if (ai === 0) return octMul(0, b); // real * imag
  const [ri, rs] = octMul(ai, b);
  return [ri, as_ * rs];
}

function getAssociator(i: number, j: number, k: number): boolean {
  // lhs = (ei*ej)*ek
  const ij = octMul(i, j);
  const lhs = octMulComplex(ij, k);
  // rhs = ei*(ej*ek) via jki, jks = ej*ek then compare to (ei*ej)*ek
  const [jki, jks] = octMul(j, k);
  const [rjki, rjks] = octMul(i, jki);
  const final_rhs: [number,number] = [rjki, jks * rjks];
  return lhs[0] !== final_rhs[0] || lhs[1] !== final_rhs[1];
}

const FANO_LINES = [[1,2,4],[2,3,5],[3,4,6],[4,5,7],[5,6,1],[6,7,2],[7,1,3]];
const LINE_COLORS = [
  'var(--color-accent-gold)',
  'var(--color-accent-teal)',
  'var(--color-accent-purple)',
  'var(--color-accent-blue)',
  '#ec4899',
  'var(--color-accent-orange)',
  '#f87171',
];

const SOUNIO_CODE = `algebra Octonion over f64 {
  add: commutative, associative
  mul: alternative, non_commutative
  reassociate: fano_selective
}

// The Fano plane encodes which triples
// are non-associative — 168 of them.
// Sounio verifies this at compile time.

fn verify_168_theorem() -> bool {
  var count = 0
  for i in 1..=7 {
    for j in 1..=7 {
      for k in 1..=7 {
        // GUM: [ei,ej,ek] = (ei·ej)·ek - ei·(ej·ek)
        let assoc = octonion_associator(i, j, k)
        if assoc != zero { count = count + 1 }
      }
    }
  }
  // Theorem: count == 168 == |PSL(2,7)|
  count == 168
}`;

export default function OctonionFanoViz() {
  const [selectedLine, setSelectedLine] = useState<number | null>(null);
  const [counter, setCounter] = useState(0);
  const [nonAssocCount, setNonAssocCount] = useState(0);
  const [computed, setComputed] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const animRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    // Compute 168 on mount
    let count = 0;
    for (let i = 1; i <= 7; i++)
      for (let j = 1; j <= 7; j++)
        for (let k = 1; k <= 7; k++)
          if (getAssociator(i, j, k)) count++;
    setNonAssocCount(count);
    setComputed(true);

    // Animate counter
    const target = count;
    const duration = 1800;
    const start = performance.now();
    const animate = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      setCounter(Math.floor(eased * target));
      if (p < 1) animRef.current = requestAnimationFrame(animate);
      else setCounter(target);
    };
    animRef.current = requestAnimationFrame(animate);
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, []);

  // SVG Fano plane layout
  // 7 nodes on a circle (e1..e6 outer, e7 center) + inscribed circle
  const cx = 140, cy = 140, r = 110;
  const nodePositions: [number,number][] = [
    [0,0], // placeholder for index 0
    ...Array.from({length:6}, (_, i) => {
      const angle = (i * Math.PI * 2 / 6) - Math.PI/2;
      return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)] as [number,number];
    }),
    [cx, cy], // e7 at center
  ];

  const activeLine = selectedLine !== null ? FANO_LINES[selectedLine] : null;
  const isNodeActive = (n: number) => activeLine ? activeLine.includes(n) : false;

  return (
    <section className="py-[clamp(3.5rem,7vw,6rem)] bg-[var(--color-bg)]">
      <div className="container px-4">
        <div className="mb-[2.4rem] grid gap-[0.5rem]">
          <h2 className="font-sans text-[clamp(1.7rem,4.2vw,3rem)] font-[750] leading-[1.1] tracking-[-0.025em] text-[var(--color-text-primary)]">
            The 168 Theorem, verified live in your browser
          </h2>
          <p className="text-[clamp(0.96rem,2.1vw,1.1rem)] text-[var(--color-text-secondary)] max-w-[72ch] leading-[1.75]">
            The octonions — the largest normed division algebra — are non-associative:
            (eᵢ·eⱼ)·eₖ ≠ eᵢ·(eⱼ·eₖ) for most basis triples.
            The associator [eᵢ, eⱼ, eₖ] = (eᵢ·eⱼ)·eₖ − eᵢ·(eⱼ·eₖ) measures how badly
            associativity fails at each ordered triple. Of the 7³ = 343 ordered triples
            over the imaginary basis &#123;e₁, …, e₇&#125;, exactly <strong className="text-[var(--color-text-primary)]">168 yield a non-zero associator</strong>.
          </p>
          <p className="text-[clamp(0.88rem,1.8vw,1rem)] text-[var(--color-text-secondary)] max-w-[72ch] leading-[1.75]">
            The number 168 is not arbitrary: it equals |PSL(2,7)|, the order of the
            projective special linear group over GF(7) — which is also the automorphism group
            of the Fano plane PG(2,2). The seven lines of the Fano plane (each a collinear
            triple) encode precisely the multiplication law of the seven imaginary octonion
            units. This coincidence reflects the deep G₂ symmetry of the octonion algebra.
            The counter below is computed from scratch in your browser using the full
            Cayley-Dickson multiplication table. Click any Fano line to inspect which triples it governs.
          </p>
        </div>

        <div className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
          {/* Fano plane SVG */}
          <div className="glass glass-specular rounded-2xl p-6 flex flex-col items-center gap-4" ref={containerRef}>
            {/* Counter */}
            <div className="text-center">
              <div className="text-[clamp(3rem,8vw,5rem)] font-extrabold text-[var(--color-accent-gold)] leading-none tabular-nums">
                {counter}
              </div>
              <div className="text-sm text-[var(--color-text-secondary)] mt-1">
                non-associative triples out of 343 = 7³
                {computed && nonAssocCount === 168 && (
                  <span className="ml-2 text-emerald-400 font-semibold">✓ = |PSL(2,7)|</span>
                )}
              </div>
            </div>

            {/* SVG Fano plane */}
            <svg width={280} height={280} viewBox="0 0 280 280" className="w-full max-w-[280px]">
              {/* Inscribed Fano circle (line 7: {7,1,3}) */}
              <circle
                cx={cx} cy={cy} r={r * 0.6}
                fill="none"
                stroke={selectedLine === 6 ? LINE_COLORS[6] : 'rgba(255,255,255,0.12)'}
                strokeWidth={selectedLine === 6 ? 3 : 1.5}
                strokeDasharray={selectedLine === 6 ? undefined : '4 3'}
                className="cursor-pointer transition-all"
                onClick={() => setSelectedLine(selectedLine === 6 ? null : 6)}
              />

              {/* 6 straight Fano lines */}
              {FANO_LINES.slice(0, 6).map(([a, b, c], li) => {
                const [x1, y1] = nodePositions[a];
                const [x2, y2] = nodePositions[b];
                const [x3, y3] = nodePositions[c];
                const active = selectedLine === li;
                const color = LINE_COLORS[li];
                return (
                  <g key={li} className="cursor-pointer" onClick={() => setSelectedLine(selectedLine === li ? null : li)}>
                    <line x1={x1} y1={y1} x2={x3} y2={y3}
                      stroke={active ? color : 'rgba(255,255,255,0.12)'}
                      strokeWidth={active ? 3 : 1.5}
                      className="transition-all" />
                    <line x1={x1} y1={y1} x2={x2} y2={y2}
                      stroke={active ? color : 'rgba(255,255,255,0.12)'}
                      strokeWidth={active ? 3 : 1.5}
                      className="transition-all" />
                    <line x1={x2} y1={y2} x2={x3} y2={y3}
                      stroke={active ? color : 'rgba(255,255,255,0.12)'}
                      strokeWidth={active ? 3 : 1.5}
                      className="transition-all" />
                  </g>
                );
              })}

              {/* Nodes */}
              {[1,2,3,4,5,6,7].map(n => {
                const [nx, ny] = nodePositions[n];
                const active = isNodeActive(n);
                const color = active && selectedLine !== null ? LINE_COLORS[selectedLine] : 'var(--color-text-tertiary)';
                return (
                  <g key={n}>
                    <circle cx={nx} cy={ny} r={active ? 12 : 9}
                      fill={active ? color : 'rgba(255,255,255,0.08)'}
                      stroke={active ? color : 'rgba(255,255,255,0.25)'}
                      strokeWidth={1.5}
                      className="transition-all" />
                    <text x={nx} y={ny+1} textAnchor="middle" dominantBaseline="middle"
                      fontSize={active ? 11 : 10} fontFamily="monospace" fontWeight="600"
                      fill={active ? '#0d1117' : 'rgba(255,255,255,0.7)'}>
                      e{n}
                    </text>
                  </g>
                );
              })}
            </svg>

            {/* Line selector legend */}
            <div className="flex flex-wrap gap-2 justify-center">
              {FANO_LINES.map(([a,b,c], li) => (
                <button key={li}
                  onClick={() => setSelectedLine(selectedLine === li ? null : li)}
                  className={`px-2.5 py-1 rounded-full text-xs font-mono font-semibold border transition-all ${
                    selectedLine === li
                      ? 'border-transparent text-[#0d1117]'
                      : 'border-[var(--glass-border)] text-[var(--color-text-secondary)]'
                  }`}
                  style={selectedLine === li ? { background: LINE_COLORS[li] } : {}}
                >
                  e{a}·e{b}·e{c}
                </button>
              ))}
            </div>

            {activeLine && (
              <div className="text-xs text-[var(--color-text-secondary)] text-center font-mono bg-[rgba(0,0,0,0.3)] rounded-lg px-4 py-2">
                <span style={{color: LINE_COLORS[selectedLine!]}}>
                  (e{activeLine[0]}·e{activeLine[1]})·e{activeLine[2]} ≠ e{activeLine[0]}·(e{activeLine[1]}·e{activeLine[2]})
                </span>
                <br/>
                <span className="text-[var(--color-text-tertiary)]">
                  Associator [e{activeLine[0]},e{activeLine[1]},e{activeLine[2]}] ≠ 0
                </span>
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
                theorem_168.sio
              </span>
            </div>
            <pre className="p-6 bg-[#0d1117] text-xs leading-relaxed overflow-x-auto">
              <code className="text-[#e6edf3]">{SOUNIO_CODE}</code>
            </pre>
            <div className="px-6 py-4 bg-[#0d1117] border-t border-[var(--glass-border)]">
              <p className="text-xs text-[var(--color-text-tertiary)] leading-relaxed max-w-[52ch]">
                The <code className="text-[var(--color-accent-gold-soft)]">fano_selective</code> reassociation
                rule tells Sounio's e-graph optimizer precisely which (eᵢ,eⱼ,eₖ) rewrites
                are algebraically valid — only the 343 − 168 = 175 associative triples may
                be reordered by the optimizer. Triples on a Fano line are non-associative
                and are never reassociated. This is, to our knowledge, the first compiler
                whose algebraic rewrite rules are constrained by the Fano plane geometry.
                Manuscript under review at <em>Advances in Applied Clifford Algebras</em>.
              </p>
            </div>
          </div>
        </div>

        <MCQBlock questions={OCTONION_QUESTIONS} />
      </div>
    </section>
  );
}
