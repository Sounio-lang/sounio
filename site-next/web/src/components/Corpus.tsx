import { claim, hasClaim } from '../evidence/claim';
import './corpus.css';

/** O registro que mede o corpus. Registro, não portão: não há veredito. */
export const CORPUS = 'site/corpus.v1';

/**
 * Os domínios do corpus escrito, na ordem em que o campo os empilha.
 *
 * A ordem é semântica, não decrescente: primeiro o que sustenta a tese — o
 * compilador escrito na própria linguagem, a biblioteca que ele compila, a
 * semântica verificada em máquina — e depois o material de apoio. Ordenar por
 * tamanho poria os testes à frente do Lean e desfaria o argumento.
 */
const DOMAINS = [
  { key: 'compiler', label: 'the compiler, written in Sounio' },
  { key: 'stdlib',   label: 'standard library' },
  { key: 'lean',     label: 'machine-checked semantics, Lean 4' },
  { key: 'examples', label: 'examples and benchmarks' },
  { key: 'tests',    label: 'test corpus' },
  { key: 'tooling',  label: 'tooling and experiments' },
  { key: 'docs',     label: 'documentation programs' },
] as const;

/** As árvores excluídas por serem geradas ou arquivadas. */
const TREES = [
  { key: 'artifacts', label: 'generated scale probes' },
  { key: 'archive',   label: 'superseded bootstrap images' },
  { key: 'bootstrap', label: 'the generated bootstrap image' },
] as const;

type Row = { key: string; label: string; lines: number; files: number; marks: number };

/** A escala do campo vem do artefato: os dois geradores leem a mesma. */
const perMark = claim(CORPUS, 'mark_lines').value;

const rows: Row[] = DOMAINS
  .filter(d => hasClaim(CORPUS, `${d.key}.lines`))
  .map(d => ({
    key: d.key,
    label: d.label,
    lines: claim(CORPUS, `${d.key}.lines`).value,
    files: claim(CORPUS, `${d.key}.files`).value,
    marks: Math.round(claim(CORPUS, `${d.key}.lines`).value / perMark),
  }));

const totalMarks = rows.reduce((n, r) => n + r.marks, 0);
const n = (v: number) => v.toLocaleString('en-GB');

/**
 * O campo: o corpus escrito inteiro, uma marca por fatia igual de linhas.
 *
 * Dimensionado por LINHA, não por arquivo. Por arquivo o corpus de testes tem
 * mais entradas que todo o resto somado e domina a imagem — o que desenha o
 * resíduo de CI, não a linguagem. Por linha, o que ocupa a maior área é o
 * compilador escrito em Sounio, que é o objeto.
 *
 * Aqui a cor codifica DOMÍNIO. É outra grandeza da que as bandas codificam
 * (largura de intervalo), e por isso este gráfico traz a sua própria legenda.
 */
export function CorpusField() {
  return (
    <>
      <div className="field" role="img"
           aria-label={`The written corpus: ${n(claim(CORPUS, 'written.lines').value)} lines across ${rows.length} domains.`}>
        {rows.map(r => (
          <span className="field-run" key={r.key} data-d={r.key}>
            {Array.from({ length: r.marks }, (_, i) => <i key={i} data-d={r.key} />)}
          </span>
        ))}
      </div>
      <p className="field-cap">
        {n(totalMarks)} marks · 1 mark = {n(perMark)} written lines · grouped by what they are
      </p>
      <ul className="legend">
        {rows.map(r => (
          <li key={r.key} data-d={r.key}>
            <b>{n(r.lines)}</b>
            <span>{r.label}</span>
            <em>{n(r.files)} files</em>
          </li>
        ))}
      </ul>
    </>
  );
}

/** A mesma medição, achatada numa barra — para páginas onde o campo é demais. */
export function CorpusBar() {
  const written = claim(CORPUS, 'written.lines').value;
  return (
    <>
      <div className="cbar" role="img"
           aria-label={`Written corpus by domain, ${n(written)} lines in total.`}>
        {rows.map(r => (
          <span key={r.key} data-d={r.key} style={{ flex: r.lines }}
                title={`${r.label}: ${n(r.lines)} lines`} />
        ))}
      </div>
      <ul className="ckey">
        {rows.map(r => (
          <li key={r.key} data-d={r.key}><b>{n(r.lines)}</b> {r.key}</li>
        ))}
      </ul>
    </>
  );
}

/** O que ficou de fora, na mesma escala — e por quê. */
export function Excluded() {
  const trees = TREES
    .filter(t => hasClaim(CORPUS, `tree.${t.key}.lines`))
    .map(t => ({ ...t, lines: claim(CORPUS, `tree.${t.key}.lines`).value }));

  return (
    <>
      <div className="strip" role="img" aria-label="Excluded trees, by size.">
        {trees.map(t => (
          <span key={t.key} data-x={t.key} style={{ flex: t.lines }}
                title={`${t.key}: ${n(t.lines)} lines`} />
        ))}
      </div>
      <ul className="excl">
        {trees.map(t => (
          <li key={t.key}><b>{n(t.lines)}</b> {t.key} — {t.label}</li>
        ))}
      </ul>
    </>
  );
}
