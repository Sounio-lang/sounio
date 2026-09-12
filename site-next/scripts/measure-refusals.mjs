#!/usr/bin/env node
/**
 * measure-refusals.mjs — mede o corpus de programas que o compilador RECUSA.
 *
 * `tests/compile-fail/` é o melhor argumento que esta linguagem tem e o site
 * não o usava. Cada arquivo ali é um programa que outras linguagens compilam
 * sem dizer nada, e que Sounio se recusa a construir.
 *
 * Um teste marcado `//@ ignore` está DESLIGADO. Ele não conta na manchete:
 * uma recusa que está desligada não é uma recusa. São contados à parte, e
 * publicados à parte.
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
const OUT = join(REPO, 'artifacts/site/refusals.v1.json');
const DIR = 'tests/compile-fail';

const git = (...a) => execFileSync('git', ['-C', REPO, ...a],
  { encoding: 'utf-8', maxBuffer: 1 << 30 });

const paths = git('ls-tree', '-r', '--name-only', REF, '--', DIR)
  .split('\n').filter(p => p.endsWith('.sio'));

// um único fluxo em vez de um processo git por arquivo
const buf = execFileSync('git', ['-C', REPO, 'cat-file', '--batch'],
  { input: paths.map(p => `${REF}:${p}`).join('\n') + '\n', maxBuffer: 1 << 30 });

let off = 0;
const live = [], ignored = [];
const codes = new Map();

for (const p of paths) {
  const nl = buf.indexOf(0x0a, off);
  const size = Number(buf.subarray(off, nl).toString().split(' ')[2]);
  const body = buf.subarray(nl + 1, nl + 1 + size).toString('utf-8');
  off = nl + 1 + size + 1;

  const head = body.split('\n').slice(0, 12).join('\n');
  const name = p.slice(DIR.length + 1).replace(/\.sio$/, '');
  if (/^\/\/@\s*ignore\b/m.test(head)) { ignored.push(name); continue; }

  live.push(name);
  const m = /error-pattern:\s*(E\d+)/.exec(head);
  if (m) codes.set(m[1], (codes.get(m[1]) ?? 0) + 1);
}

const artifact = {
  schema: 'sounio.site.refusals.v1',
  generated_at: new Date().toISOString(),
  // registro: mede um corpus, não declara veredito sobre ele
  reason:
    `Contado em ${REF} sobre ${DIR}. Um teste marcado "//@ ignore" está ` +
    `desligado e não entra na contagem viva: uma recusa desligada não é ` +
    `uma recusa.`,
  ref: REF,
  commit: git('rev-parse', REF).trim(),
  error_codes: [...codes.keys()].sort(),
  metrics: {
    live: live.length,
    ignored: ignored.length,
    total: paths.length,
    error_codes: codes.size,
  },
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(artifact, null, 2) + '\n');
console.log(`escrito ${OUT}`);
console.log(`  vivos     ${live.length}`);
console.log(`  desligados ${ignored.length}`);
console.log(`  classes de erro declaradas ${codes.size}`);
