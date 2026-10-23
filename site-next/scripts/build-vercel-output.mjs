#!/usr/bin/env node
/**
 * build-vercel-output.mjs — monta .vercel/output a partir do dist do Swift.
 *
 * Por que existe: o gerador é escrito em Swift, e o contêiner de build da
 * Vercel não tem toolchain Swift. A saída é construir no CI (que pode instalar
 * o Swift) e enviar pronta com `vercel deploy --prebuilt`, que não roda build
 * nenhum do lado da Vercel.
 *
 * Formato: Build Output API v3.
 * https://vercel.com/docs/build-output-api/v3
 */
import { cpSync, mkdirSync, writeFileSync, rmSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SRC = join(ROOT, 'swift/dist');
const OUT = join(ROOT, '.vercel/output');

if (!existsSync(SRC)) {
  console.error(`Nada em ${relative(ROOT, SRC)}. Rode o gerador antes:`);
  console.error('  cd site-next/web && npm run build:islands');
  console.error('  cd site-next/swift && swift run sitegen');
  process.exit(1);
}

rmSync(OUT, { recursive: true, force: true });
mkdirSync(join(OUT, 'static'), { recursive: true });
cpSync(SRC, join(OUT, 'static'), { recursive: true });

// O site é estático: sem funções, sem middleware, sem imagens otimizadas.
// `cleanUrls` deixa /proof servir proof.html — os links internos usam .html,
// e ambos passam a funcionar.
const config = {
  version: 3,
  cleanUrls: true,
  trailingSlash: false,
  headers: [
    {
      // Os assets de marca são imutáveis dentro de um deploy.
      source: '/brand/(.*)',
      headers: [{ key: 'Cache-Control', value: 'public, max-age=31536000, immutable' }],
    },
    {
      // A ilha muda a cada build e é minúscula: revalidação barata.
      source: '/islands.js',
      headers: [{ key: 'Cache-Control', value: 'public, max-age=0, must-revalidate' }],
    },
    {
      source: '/(.*)',
      headers: [
        { key: 'X-Content-Type-Options', value: 'nosniff' },
        { key: 'Referrer-Policy', value: 'strict-origin-when-cross-origin' },
      ],
    },
  ],
};
writeFileSync(join(OUT, 'config.json'), JSON.stringify(config, null, 2) + '\n');

function walk(dir, acc = []) {
  for (const n of readdirSync(dir)) {
    const p = join(dir, n);
    statSync(p).isDirectory() ? walk(p, acc) : acc.push(p);
  }
  return acc;
}
const files = walk(join(OUT, 'static'));
const bytes = files.reduce((n, f) => n + statSync(f).size, 0);

console.log(`.vercel/output montado: ${files.length} arquivos, ${(bytes / 1024).toFixed(0)} KB`);
for (const f of files.filter(f => f.endsWith('.html'))) {
  console.log(`  ${relative(join(OUT, 'static'), f)}`);
}
