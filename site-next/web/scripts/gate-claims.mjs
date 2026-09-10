#!/usr/bin/env node
/**
 * gate-claims.mjs — recusa o build quando o conteúdo afirma um número sem procedência.
 *
 * A regra: um numeral que aparece na prosa do site é uma afirmação factual.
 * Ou ele vem de `claim(artefato, métrica)` — e portanto carrega o portão que
 * o sustenta — ou o build para. Não há terceira opção, e não há lista de
 * exceções por arquivo: a saída é sempre trocar o numeral por uma evidência.
 *
 * O que NÃO é afirmação, e por isso passa: versões, anos, índices, valores de
 * layout, números dentro de código, e numerais de um dígito (ordinais de
 * lista, contagens triviais que o leitor confere na própria página).
 */
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEB = join(HERE, '..');
const ROOTS = ['src', 'content'].map(d => join(WEB, d)).filter(existsSync);

const SCAN = /\.(tsx|mdx|md)$/;

// numerais que são afirmação: 2+ dígitos, percentagens, razões, decimais
const CLAIMISH = /(?<![\w.#-])(\d{2,}(?:[.,]\d+)?%?|\d+\s*\/\s*\d+|\d+[.,]\d+)(?![\w%-])/g;

// Trechos que não são afirmação. São MASCARADOS, não usados para pular a
// linha: o resto da linha continua sendo verificado.
const EXEMPT_LINE = [
  /claim\s*\([^)]*\)/,        // já tem procedência
  /<Claim\b[^>]*>/,           // idem, na forma de componente
  /https?:\/\//,              // URLs
  /\b(px|rem|em|vh|vw|ms|deg|fr)\b/,   // layout
  /#[0-9a-fA-F]{3,8}\b/,      // cores
  /\bv?\d+\.\d+\.\d+\b/,      // versões semânticas
  /\b(19|20)\d{2}\b/,         // anos
  /\bE[0-5]\b/,               // níveis de portão

  // Constantes do MÉTODO, não resultados de medição. O nível de confiança e o
  // z que o produz descrevem como o intervalo é calculado; não são afirmações
  // sobre o compilador. A lista é deliberadamente mínima — se ela crescer,
  // provavelmente alguém está usando-a para contrabandear uma afirmação.
  /\b\d{1,2}:\d{2}\b/,       // horas do relógio
  /\b\d{1,2}(?:[–-]\d{1,2})?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b/i,   // datas
  /\b95\s*%/,
  /\bz\s*=\s*1\.96\b/,

  // Versões de NOMES PRÓPRIOS de padrões e ferramentas. A lista é explícita
  // em vez de uma regra genérica ("maiúscula seguida de número") de propósito:
  // uma regra genérica isentaria "Madaros 4.71" e viraria exactamente o
  // contrabando que este portão existe para impedir. Cresce um nome de cada
  // vez, e só para nomes próprios.
  /\bGRI-Mech\s+\d+(?:\.\d+)+/,
  /\bCantera\s+\d+(?:\.\d+)+/,
  /\bNASA-\d\b/,
];

function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name.startsWith('.')) continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, acc);
    else if (SCAN.test(name) && !name.endsWith('.generated.ts')) acc.push(p);
  }
  return acc;
}

// remove blocos e trechos de código: números lá dentro são código, não afirmação.
// Cobre cercas ```, crases inline e o elemento <code> do JSX/MDX — este último
// faltava, e fazia `var=0.000` (que é código) ser lido como afirmação.
function stripCode(text) {
  const blank = m => m.replace(/[^\n]/g, ' ');
  return text
    .replace(/```[\s\S]*?```/g, blank)
    .replace(/<code\b[^>]*>[\s\S]*?<\/code>/g, blank)
    .replace(/`[^`\n]*`/g, blank);
}

const violations = [];
for (const root of ROOTS) {
  for (const file of walk(root)) {
    const raw = readFileSync(file, 'utf-8');
    const lines = stripCode(raw).split('\n');
    lines.forEach((line, i) => {
      // Mascara os trechos isentos e varre o resto. Pular a linha inteira
      // porque ela contém uma versão ou um ano deixaria passar qualquer
      // numeral inventado na mesma linha.
      const masked = EXEMPT_LINE.reduce(
        (acc, re) => acc.replace(new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g'), m => '·'.repeat(m.length)),
        line);
      if (/^\s*(import|export)\b/.test(line) || /^\s*(\/\/|\*)/.test(line)) return;
      for (const m of masked.matchAll(CLAIMISH)) {
        violations.push({
          file: relative(WEB, file), line: i + 1,
          numeral: m[0], text: line.trim().slice(0, 96),
        });
      }
    });
  }
}

if (violations.length === 0) {
  console.log(`gate-claims: nenhum numeral sem procedência (${ROOTS.length} raiz(es) varrida(s)).`);
  process.exit(0);
}

console.error(`\ngate-claims: ${violations.length} afirmação(ões) numérica(s) sem evidência.\n`);
for (const v of violations) {
  console.error(`  ${v.file}:${v.line}  "${v.numeral}"`);
  console.error(`    ${v.text}`);
}
console.error(`
Cada uma precisa virar claim(artefato, métrica) — ou sair do texto.
Os artefatos disponíveis estão em src/evidence/index.generated.ts
(regenere com: npm run evidence).
`);
process.exit(1);
