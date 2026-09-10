#!/usr/bin/env node
/**
 * measure-corpus.mjs — mede o tamanho do corpus escrito e emite um ARTEFATO.
 *
 * Não emite HTML nem números para o site diretamente. Escreve
 * `artifacts/site/corpus.v1.json`, que o indexador lê como qualquer outro
 * artefato. Isso é deliberado: se os números do corpus entrassem no site por
 * um caminho especial, a garantia que o site anuncia — todo numeral vem de
 * artifacts/**\/*.json — teria um buraco do tamanho da própria manchete.
 *
 * É um REGISTRO, não um portão: não tem status porque não há nada a passar ou
 * falhar. É medição com procedência.
 *
 * Três árvores são EXCLUÍDAS por serem geradas ou arquivadas. Contá-las
 * inflaria o corpus escrito em 1,3 milhão de linhas — e publicar saída gerada
 * como trabalho escrito é a primeira coisa que este projeto recusa.
 */
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '../..');
const REF = process.env.CORPUS_REF ?? 'origin/main';
const OUT = join(REPO, 'artifacts/site/corpus.v1.json');

const git = (...a) => execFileSync('git', ['-C', REPO, ...a], {
  encoding: 'utf-8', maxBuffer: 1 << 30,
});

/** Gerado ou arquivado. Nunca conta como escrito. */
const EXCLUDED = ['artifacts/', 'archive/', 'bootstrap/'];

/** Prefixo -> domínio. A ordem importa: o primeiro que casar ganha. */
const DOMAINS = [
  ['self-hosted/', 'compiler'],
  ['stdlib/',      'stdlib'],
  ['tests/',       'tests'],
  ['examples/',    'examples'],
  ['benchmarks/',  'examples'],
  ['docs/',        'docs'],
];

const domainOf = p => DOMAINS.find(([pre]) => p.startsWith(pre))?.[1] ?? 'tooling';

const paths = git('ls-tree', '-r', '--name-only', REF).split('\n').filter(Boolean);

// Uma chamada de git por arquivo seria ~9k processos. `git cat-file --batch`
// resolve tudo num fluxo só.
function lineCounts(list) {
  const input = list.map(p => `${REF}:${p}`).join('\n') + '\n';
  const buf = execFileSync('git', ['-C', REPO, 'cat-file', '--batch'],
                           { input, maxBuffer: 1 << 30 });
  const out = [];
  let off = 0;
  for (const p of list) {
    const nl = buf.indexOf(0x0a, off);
    const header = buf.subarray(off, nl).toString();
    const size = Number(header.split(' ')[2]);
    if (!Number.isFinite(size)) throw new Error(`cat-file inesperado para ${p}: ${header}`);
    const body = buf.subarray(nl + 1, nl + 1 + size);
    let n = 0;
    for (let i = 0; i < body.length; i++) if (body[i] === 0x0a) n++;
    // arquivo sem quebra final ainda tem uma última linha
    if (size > 0 && body[size - 1] !== 0x0a) n++;
    out.push([p, n]);
    off = nl + 1 + size + 1;   // +1 pelo \n que fecha o registro
  }
  return out;
}

const source = paths.filter(p => p.endsWith('.sio') || p.endsWith('.lean'));
const counted = lineCounts(source);

const written = {};   // domínio -> {lines, files}
const excluded = {};  // árvore   -> {lines, files}
let biggest = { path: null, lines: 0 };

for (const [p, n] of counted) {
  const tree = EXCLUDED.find(pre => p.startsWith(pre));
  if (tree) {
    const k = tree.slice(0, -1);
    excluded[k] ??= { lines: 0, files: 0 };
    excluded[k].lines += n; excluded[k].files++;
    if (n > biggest.lines) biggest = { path: p, lines: n };
    continue;
  }
  const d = p.endsWith('.lean') ? 'lean' : domainOf(p);
  written[d] ??= { lines: 0, files: 0 };
  written[d].lines += n; written[d].files++;
}

const sum = (o, k) => Object.values(o).reduce((a, x) => a + x[k], 0);

// A escala do campo de marcas viaja DENTRO do artefato, não no código da
// página. Duas razões: ela vira uma afirmação com procedência como qualquer
// outra (o portão de evidência não precisa abrir excepção para ela), e os dois
// geradores — TypeScript e Swift — passam a ler a mesma escala em vez de cada
// um escolher a sua.
const TARGET_MARKS = 8400;
const markLines = Math.max(
  50, Math.round(sum(written, 'lines') / TARGET_MARKS / 50) * 50);

const artifact = {
  schema: 'sounio.site.corpus.v1',
  generated_at: new Date().toISOString(),
  // sem `status`: é registro, não portão. Não há veredito a declarar.
  reason:
    `Medido em ${REF} lendo cada blob .sio e .lean rastreado. ` +
    `As árvores ${EXCLUDED.map(e => e.slice(0, -1)).join(', ')} são geradas ou ` +
    `arquivadas e ficam fora de toda contagem de linhas escritas.`,
  ref: REF,
  commit: git('rev-parse', REF).trim(),
  excluded_trees: EXCLUDED.map(e => e.slice(0, -1)),
  biggest_generated_file: biggest.path,
  metrics: {
    written: { lines: sum(written, 'lines'), files: sum(written, 'files') },
    excluded: { lines: sum(excluded, 'lines'), files: sum(excluded, 'files') },
    biggest_generated: biggest.lines,
    // o que a manchete teria dito se o gerado fosse contado como escrito
    total_tracked: sum(written, 'lines') + sum(excluded, 'lines'),
    mark_lines: markLines,
    commits: Number(git('rev-list', '--count', REF).trim()),
    ...written,
    tree: excluded,
  },
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(artifact, null, 2) + '\n');

console.log(`escrito ${OUT}`);
console.log(`escala do campo: 1 marca = ${markLines} linhas`);
console.log(`escrito:  ${artifact.metrics.written.lines.toLocaleString('en-GB')} linhas em ${artifact.metrics.written.files.toLocaleString('en-GB')} arquivos`);
for (const [d, v] of Object.entries(written).sort((a, b) => b[1].lines - a[1].lines))
  console.log(`  ${d.padEnd(10)}${v.lines.toLocaleString('en-GB').padStart(11)}  ${String(v.files).padStart(6)} arquivos`);
console.log(`excluído: ${artifact.metrics.excluded.lines.toLocaleString('en-GB')} linhas (gerado ou arquivado)`);
for (const [d, v] of Object.entries(excluded).sort((a, b) => b[1].lines - a[1].lines))
  console.log(`  ${d.padEnd(10)}${v.lines.toLocaleString('en-GB').padStart(11)}`);
