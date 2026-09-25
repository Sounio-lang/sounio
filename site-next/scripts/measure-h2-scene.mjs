#!/usr/bin/env node
/**
 * measure-h2-scene.mjs — extrai a cena de combustão de hidrogénio.
 *
 * A tese desta cena é diferente da cena da dose. Lá: um ponto não cruza uma
 * linha. Aqui: **uma banda que muda quando você muda o passo de integração
 * está a reportar o integrador, não a química.**
 *
 * Tudo vem de `benchmarks/chemistry/RESULTS.md`, secções 5.1 e 5.4 — medições
 * de três implementações independentes (Sounio nativo, réplica Python, e um
 * cross-check em C++23 escrito do protocolo publicado, não traduzido). O
 * script falha alto se as tabelas não estiverem lá: uma cena sobre não
 * inventar números não pode ter os seus próprios inventados.
 *
 * As razões são normalizadas ao próprio valor em dt = 4e-9, para os dois lados
 * serem razões MEDIDAS e nenhum precisar de uma âncora absoluta que este
 * documento não publica.
 */
import { execFileSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '../..');
// HEAD, não origin/main: o site descreve a árvore de que foi construído, e
// qualquer pessoa pode fazer checkout desse commit e repetir a medição. Medir
// um ramo que se move faria os números descreverem uma árvore que não é a
// publicada, e reprovaria a guarda de deriva a cada commit alheio.
const REF = process.env.CORPUS_REF ?? 'HEAD';
const SRC = 'benchmarks/chemistry/RESULTS.md';
const OUT = join(REPO, 'artifacts/site/h2_scene.v1.json');

const git = (...a) => execFileSync('git', ['-C', REPO, ...a],
  { encoding: 'utf-8', maxBuffer: 1 << 30 });

const doc = git('show', `${REF}:${SRC}`);

function section(from, to, what) {
  const a = doc.indexOf(from);
  const b = doc.indexOf(to, a + 1);
  if (a < 0 || b < 0) throw new Error(`não encontrei a secção ${what} em ${SRC}`);
  return doc.slice(a, b);
}

const num = s => Number(String(s).replace(/\*/g, '').trim());

/** Tabela 5.1: razão da banda de quadratura por espécie, três pares de dt. */
const s51 = section('### 5.1 Scaling in dt', '### 5.2', '5.1');
const quad = {};
for (const m of s51.matchAll(
  /^\|\s*(H2O2|H2O|HO2|OH|H2|O2|H|O)\s*\|([^|]+)\|([^|]+)\|([^|]+)\|/gm)) {
  quad[m[1]] = { half1: num(m[2]), half2: num(m[3]), quarter: num(m[4]) };
}
if (Object.keys(quad).length !== 8) {
  throw new Error(`5.1 devia dar 8 espécies, deu ${Object.keys(quad).length}`);
}

/** Tabela 5.4: bandas nativas absolutas, três dt. Viram razões. */
const s54 = section('### 5.4 The Sounio native band', '### 5.5', '5.4');
const native = {};
for (const m of s54.matchAll(
  /^\|\s*(H2O2|H2O|HO2|OH|H2|O2|H|O)\s*\|\s*([\d.e+-]+)\s*\|\s*([\d.e+-]+)\s*\|\s*([\d.e+-]+)\s*\|/gm)) {
  const [a, b, c] = [Number(m[2]), Number(m[3]), Number(m[4])];
  native[m[1]] = { half1: a / b, half2: b / c, quarter: a / c, band: a };
}
if (Object.keys(native).length !== 8) {
  throw new Error(`5.4 devia dar 8 espécies, deu ${Object.keys(native).length}`);
}

/** O contraste em três implementações, sob um fator 4 em dt. */
const contrast = section('This is the whole contrast', '### 5.5', 'contraste');
function row(label, what) {
  const re = new RegExp(`\\|[^|\\n]*${label}[^|\\n]*\\|\\s*\\*\\*([\\d.]+)\\s*[–-]\\s*([\\d.]+)\\*\\*`);
  const m = re.exec(contrast);
  if (!m) throw new Error(`não encontrei a linha de ${what} na tabela de contraste`);
  return { lo: Number(m[1]), hi: Number(m[2]) };
}
const replica = row('Python replica', 'réplica Python');
const cpp     = row('C\\+\\+23 cross-check', 'cross-check C++23');
const sounio  = row('Sounio native', 'Sounio nativo');

const round = (v, n = 6) => Number(v.toFixed(n));

// espécies cuja banda NÃO acumula: dominada pela incerteza inicial de 1%.
// A secção 5.2 diz explicitamente que a lei √dt é falsa para elas.
const nonAccumulating = Object.keys(quad).filter(k => Math.abs(quad[k].quarter - 1) < 0.01);

const artifact = {
  schema: 'sounio.site.h2_scene.v1',
  generated_at: new Date().toISOString(),
  reason:
    `Extraído de ${SRC} (secções 5.1, 5.4) em ${REF}. As razões da banda são ` +
    `normalizadas ao valor da própria implementação em dt = 4e-9, para os dois ` +
    `lados serem medições e não precisarem de âncora absoluta. Três ` +
    `implementações independentes; a de C++23 foi escrita do protocolo ` +
    `publicado, não traduzida.`,
  ref: REF,
  source: SRC,
  commit: git('rev-parse', REF).trim(),
  species_order: ['H', 'O', 'OH', 'H2O', 'HO2', 'H2O2', 'H2', 'O2'],
  non_accumulating: nonAccumulating,
  reactions_note: 'GRI-Mech 3.0 H/O sub-mechanism',
  metrics: {
    species: 10, reactions: 29,
    // o contraste, sob um fator 4 no passo
    replica_lo: replica.lo, replica_hi: replica.hi,
    cpp_lo: cpp.lo, cpp_hi: cpp.hi,
    native_lo: sounio.lo, native_hi: sounio.hi,
    predicted_quarter: 2, predicted_half: round(Math.SQRT2),
    non_accumulating_count: nonAccumulating.length,
    species_reported: Object.keys(quad).length,
    quadrature: Object.fromEntries(Object.entries(quad).map(
      ([k, v]) => [k, { half1: v.half1, half2: v.half2, quarter: v.quarter }])),
    native: Object.fromEntries(Object.entries(native).map(
      ([k, v]) => [k, { half1: round(v.half1), half2: round(v.half2), quarter: round(v.quarter) }])),
  },
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(artifact, null, 2) + '\n');
console.log(`escrito ${OUT}`);
console.log(`sob um fator 4 no passo dt:`);
console.log(`  réplica Python   ${replica.lo} – ${replica.hi}`);
console.log(`  cross-check C++  ${cpp.lo} – ${cpp.hi}`);
console.log(`  Sounio nativo    ${sounio.lo} – ${sounio.hi}`);
console.log(`espécies que não acumulam (banda dominada pela condição inicial): ${nonAccumulating.join(', ')}`);
