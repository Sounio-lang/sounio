import { useState, type KeyboardEvent } from 'react';
import './ParticleLoom.css';

interface Props { locale?: string; }
type Mode = 'ladder' | 'spectrum';

const copy = {
  en: {
    chapter: 'I · Exact algebra flagship',
    eyebrow: 'THE PARTICLE LOOM / EXACT LADDER WITNESS',
    heading: 'Exact algebra, woven into an eight-state spectrum.',
    body: 'Sounio executes an established construction associated with Furey: three exact ladder operators satisfy 18 ordered anticommutator checks over Gaussian integers, then their occupation count forms the multiplicities 1 · 3 · 3 · 1.',
    ladderTab: '18 exact checks',
    spectrumTab: '8-state spectrum',
    operators: 'three scaled ladder operators',
    ordinary: 'ordinary anticommutator',
    adjoint: 'adjoint anticommutator',
    relation: 'relation',
    sealed: '18 ordered checks covered',
    spectrum: 'positive occupation spectrum',
    occupation: 'occupied modes',
    charge: 'Q = N / 3',
    multiplicity: 'multiplicity',
    totalStates: 'eight positive states',
    exact: 'exact arithmetic',
    exactValue: 'Z[i] · 8×8 left-action matrices · † = conjugate-transpose',
    oracle: 'cross-toolchain receipt',
    identical: 'DIFF = IDENTICAL',
    sounio: 'SOUNIO',
    python: 'PYTHON ORACLE',
    pass: 'FUREY OK',
    command: 'bash scripts/ci/furey_octonion_gate.sh',
    source: 'Open Sounio witness',
    gate: 'Open cross-verification gate',
    boundary: 'Claim boundary',
    boundaryText: 'This executes an established Furey-style result in Sounio\'s cd_sigma convention. It verifies 18 ordered matrix checks and emits the positive eight-state combinatorial spectrum. It does not prove uniqueness, completeness, phenomenology, the conjugate negative ideal, or a compiler-native algebra feature.',
    focusLadder: 'Every cell represents a checked 8 × 8 complex-integer matrix identity.',
    focusSpectrum: 'The binomial occupation counts C(3,N) produce eight states across four charge tracks.',
    nextChapter: 'Next flagship · executable images',
    nextHref: '#sounio-observatory',
  },
  pt: {
    chapter: 'I · Flagship de álgebra exata',
    eyebrow: 'O TEAR DE PARTÍCULAS / WITNESS DE ESCADA EXATO',
    heading: 'Álgebra exata, tecida em um espectro de oito estados.',
    body: 'Sounio executa uma construção estabelecida associada a Furey: três operadores de escada exatos satisfazem 18 verificações anticomutadoras ordenadas sobre inteiros gaussianos; depois, a contagem de ocupação forma as multiplicidades 1 · 3 · 3 · 1.',
    ladderTab: '18 verificações exatas',
    spectrumTab: 'espectro de 8 estados',
    operators: 'três operadores de escada escalados',
    ordinary: 'anticomutador ordinário',
    adjoint: 'anticomutador adjunto',
    relation: 'relação',
    sealed: '18 verificações ordenadas cobertas',
    spectrum: 'espectro positivo de ocupação',
    occupation: 'modos ocupados',
    charge: 'Q = N / 3',
    multiplicity: 'multiplicidade',
    totalStates: 'oito estados positivos',
    exact: 'aritmética exata',
    exactValue: 'Z[i] · matrizes de ação à esquerda 8×8 · † = transposta conjugada',
    oracle: 'recibo entre toolchains',
    identical: 'DIFF = IDÊNTICO',
    sounio: 'SOUNIO',
    python: 'ORÁCULO PYTHON',
    pass: 'FUREY OK',
    command: 'bash scripts/ci/furey_octonion_gate.sh',
    source: 'Abrir witness Sounio',
    gate: 'Abrir gate de verificação cruzada',
    boundary: 'Fronteira da alegação',
    boundaryText: 'Isto executa um resultado estabelecido no estilo de Furey sob a convenção cd_sigma de Sounio. Verifica 18 checagens matriciais ordenadas e emite o espectro combinatório positivo de oito estados. Não prova unicidade, completude, fenomenologia, o ideal conjugado negativo ou um recurso de álgebra nativo do compilador.',
    focusLadder: 'Cada célula representa uma identidade matricial complexa inteira 8 × 8 verificada.',
    focusSpectrum: 'As contagens binomiais C(3,N) produzem oito estados distribuídos em quatro trilhas de carga.',
    nextChapter: 'Próximo flagship · imagens executáveis',
    nextHref: '#sounio-observatory',
  },
};

const operators = [
  { id: 'A₁', value: '−L₁ + iL₂', pair: '(e₁, e₂)' },
  { id: 'A₂', value: '−L₃ + iL₄', pair: '(e₃, e₄)' },
  { id: 'A₃', value: '−L₅ + iL₆', pair: '(e₅, e₆)' },
];

const tracks = [
  { n: 0, q: '0', q3: '0', count: 1 },
  { n: 1, q: '1/3', q3: '1', count: 3 },
  { n: 2, q: '2/3', q3: '2', count: 3 },
  { n: 3, q: '1', q3: '3', count: 1 },
];

const REPO_PIN = '435376dcb5c3100fc69aee51d059ff91f67ea626';
const repo = `https://github.com/Sounio-lang/sounio/blob/${REPO_PIN}`;

export default function ParticleLoom({ locale = 'en' }: Props) {
  const [mode, setMode] = useState<Mode>('ladder');
  const [selectedRelation, setSelectedRelation] = useState({ family: 'ordinary', i: 0, j: 0 });
  const d = copy[locale === 'pt' ? 'pt' : 'en'];
  const ladderActive = mode === 'ladder';
  const selectedDiagonal = selectedRelation.family === 'adjoint' && selectedRelation.i === selectedRelation.j;
  const selectedResult = selectedDiagonal ? '4I₈' : '0₈×₈';
  const selectedFormula = selectedRelation.family === 'adjoint'
    ? `{A${selectedRelation.i + 1}, A${selectedRelation.j + 1}†} = 4δ${selectedRelation.i + 1}${selectedRelation.j + 1}I₈`
    : `{A${selectedRelation.i + 1}, A${selectedRelation.j + 1}} = 0₈×₈`;
  const handleTabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    let nextMode: Mode | null = null;
    if (event.key === 'ArrowRight' || event.key === 'End') nextMode = 'spectrum';
    if (event.key === 'ArrowLeft' || event.key === 'Home') nextMode = 'ladder';
    if (!nextMode) return;
    event.preventDefault();
    setMode(nextMode);
    document.getElementById(`pl-${nextMode}-tab`)?.focus();
  };

  return (
    <section className={`pl-section ${ladderActive ? 'is-ladder' : 'is-spectrum'}`} id="particle-loom" aria-labelledby="pl-title">
      <div className="pl-shell">
        <p className="pl-chapter">{d.chapter}</p>
        <header className="pl-header">
          <div className="pl-brand">
            <img src="/assets/stamps/stamp_monochrome_on_navy.png" alt="" aria-hidden="true" width="72" height="72" />
            <p>{d.eyebrow}</p>
          </div>
          <h2 id="pl-title">{d.heading}</h2>
          <p>{d.body}</p>
        </header>

        <div className="pl-modes" role="tablist" aria-label="Particle loom views">
          <button id="pl-ladder-tab" type="button" role="tab" aria-selected={ladderActive} aria-controls="pl-relations" tabIndex={ladderActive ? 0 : -1} onKeyDown={handleTabKey} onClick={() => setMode('ladder')}><span>01</span>{d.ladderTab}<b>18 / 18</b></button>
          <button id="pl-spectrum-tab" type="button" role="tab" aria-selected={!ladderActive} aria-controls="pl-spectrum" tabIndex={ladderActive ? -1 : 0} onKeyDown={handleTabKey} onClick={() => setMode('spectrum')}><span>02</span>{d.spectrumTab}<b>1 · 3 · 3 · 1</b></button>
        </div>

        <div className="pl-loom" aria-live="polite">
          <aside className="pl-operators">
            <p>{d.operators}</p>
            <ol>
              {operators.map((operator) => (
                <li key={operator.id}><strong>{operator.id}</strong><code>{operator.value}</code><span>{operator.pair}</span></li>
              ))}
            </ol>
            <dl><div><dt>{d.exact}</dt><dd>{d.exactValue}</dd></div></dl>
          </aside>

          <div className="pl-relations" id="pl-relations" role="tabpanel" aria-labelledby="pl-ladder-tab">
            <div className="pl-relation-head"><span>{d.relation} A</span><strong>{'{Aᵢ, Aⱼ} = 0'}</strong><small>{d.ordinary}</small></div>
            <div className="pl-matrix" aria-label="Nine ordinary anticommutator checks">
              {operators.flatMap((left, i) => operators.map((right, j) => (
                <button
                  type="button"
                  key={`ordinary-${i}-${j}`}
                  aria-pressed={selectedRelation.family === 'ordinary' && selectedRelation.i === i && selectedRelation.j === j}
                  aria-label={`{A${i + 1}, A${j + 1}} equals zero 8 by 8 complex matrix; covered by aggregate check`}
                  onClick={() => setSelectedRelation({ family: 'ordinary', i, j })}
                ><span>{left.id}{right.id}</span><strong>0₈×₈</strong><small>COVERED</small></button>
              )))}
            </div>
            <div className="pl-relation-head"><span>{d.relation} B</span><strong>{'{Aᵢ, Aⱼ†} = 4δᵢⱼI'}</strong><small>{d.adjoint}</small></div>
            <div className="pl-matrix" aria-label="Nine adjoint anticommutator checks">
              {operators.flatMap((left, i) => operators.map((right, j) => (
                <button
                  type="button"
                  key={`adjoint-${i}-${j}`}
                  aria-pressed={selectedRelation.family === 'adjoint' && selectedRelation.i === i && selectedRelation.j === j}
                  aria-label={`{A${i + 1}, A${j + 1} dagger} equals ${i === j ? '4 times identity 8 by 8' : 'zero 8 by 8 complex matrix'}; covered by aggregate check`}
                  onClick={() => setSelectedRelation({ family: 'adjoint', i, j })}
                ><span>{left.id}{right.id}†</span><strong>{i === j ? '4I₈' : '0₈×₈'}</strong><small>COVERED</small></button>
              )))}
            </div>
            <div className="pl-relation-inspector">
              <span>SELECTED ORDERED CHECK</span>
              <strong>{selectedFormula}</strong>
              <dl>
                <div><dt>expected</dt><dd>{selectedResult}</dd></div>
                <div><dt>real</dt><dd>{selectedResult}</dd></div>
                <div><dt>imaginary</dt><dd>0₈×₈</dd></div>
              </dl>
              <code>covered by aggregate LADDER_OK 1</code>
            </div>
            <div className="pl-sealed"><span>18</span><strong>{d.sealed}</strong><code>{d.focusLadder}</code></div>
          </div>

          <aside className="pl-spectrum" id="pl-spectrum" role="tabpanel" aria-labelledby="pl-spectrum-tab">
            <div className="pl-spectrum-head"><span>{d.spectrum}</span><strong>Σ = 8</strong></div>
            <div className="pl-weave" aria-hidden="true">
              <i /><i /><i /><i />
            </div>
            <div className="pl-tracks">
              {tracks.map((track) => (
                <div className="pl-track" key={track.n} data-n={track.n}>
                  <div className="pl-particles" aria-label={`${track.count} states`}>
                    {Array.from({ length: track.count }, (_, index) => <i key={index} />)}
                  </div>
                  <strong>{track.count}</strong>
                  <dl>
                    <div><dt>N</dt><dd>{track.n}</dd></div>
                    <div><dt>Q</dt><dd>{track.q}</dd></div>
                    <div><dt>Q×3</dt><dd>{track.q3}</dd></div>
                  </dl>
                </div>
              ))}
            </div>
            <div className="pl-spectrum-note">
              <span>{d.multiplicity}</span>
              <strong>1 · 3 · 3 · 1</strong>
              <code>{d.focusSpectrum}</code>
              <em>{d.totalStates}</em>
            </div>
          </aside>
        </div>

        <div className="pl-crosscheck">
          <div className="pl-tape"><span>{d.sounio}</span><code>LADDER_OK 1 · CHARGE3_0..3 1 3 3 1 · FUREY OK</code></div>
          <div className="pl-convergence" aria-hidden="true"><i /><b>=</b><i /></div>
          <div className="pl-tape"><span>{d.python}</span><code>LADDER_OK 1 · CHARGE3_0..3 1 3 3 1 · FUREY OK</code></div>
          <div className="pl-verdict"><span>{d.oracle}</span><strong>{d.identical}</strong><code>{d.pass}</code></div>
        </div>

        <footer className="pl-footer">
          <div><strong>{d.boundary}</strong><span>{d.boundaryText}</span></div>
          <div className="pl-links">
            <code>{d.command}</code>
            <a href={`${repo}/tests/run-pass/furey_octonion_generation.sio`} target="_blank" rel="noreferrer">{d.source}<span aria-hidden="true">↗</span></a>
            <a href={`${repo}/scripts/ci/furey_octonion_gate.sh`} target="_blank" rel="noreferrer">{d.gate}<span aria-hidden="true">↗</span></a>
            <a className="pl-next" href={d.nextHref}>{d.nextChapter}<span aria-hidden="true">↓</span></a>
          </div>
        </footer>
      </div>
    </section>
  );
}
