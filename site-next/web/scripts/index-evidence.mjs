#!/usr/bin/env node
/**
 * index-evidence.mjs — indexa os artefatos do repositório numa tabela tipada.
 *
 * Esta é a base da tese do site: nenhum número é escrito à mão. Todo numeral
 * exibido tem de vir daqui, com o status do portão que o sustenta. O gerador
 * roda antes do build; o portão (gate-claims.mjs) recusa o build se sobrar
 * numeral solto no conteúdo.
 */
import { readFileSync, writeFileSync, readdirSync, statSync, mkdirSync } from 'node:fs';
import { join, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, '../../..');   // site-next/web/scripts -> raiz do repo
const ARTIFACTS = join(REPO, 'artifacts');
const OUT = join(HERE, '../src/evidence/index.generated.ts');

function walk(dir, acc = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p, acc);
    else if (name.endsWith('.json')) acc.push(p);
  }
  return acc;
}

// Duas espécies, não uma com metade quebrada.
//
// Um PORTÃO declara veredito: passa, falha, fica sem cota. Um REGISTRO traz
// medição, pesquisa ou procedência — e não tem veredito porque não é portão.
// Tratar registro como portão sem status era erro de categoria: `seed-refresh/`
// são recibos com input_seed/output_seed/fixed_point, `research/` são
// resultados científicos. Nenhum deles falhou em declarar nada.
const VERDICTS = { pass: 'verified', fail: 'refused',
                   partial: 'unbounded', beta: 'unbounded', active: 'unbounded' };

function classify(status) {
  const v = VERDICTS[String(status ?? '').toLowerCase()];
  return v ? { kind: 'gate', level: v } : { kind: 'record', level: null };
}

// Intervalo de Wilson 95%.
//
// Todo portão que reporta passed/total é uma proporção estimada de uma amostra
// finita. "Todos passaram" com n=1 sustenta [0.207, 1.000] — quase nada. O site
// desenha esse intervalo porque é o que a amostra realmente diz, e porque um
// valor nesta linguagem é uma banda, não um ponto.
function wilson(k, n, z = 1.96) {
  if (!(n > 0)) return null;
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z / d) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  const lo = Math.max(0, c - h), hi = Math.min(1, c + h);
  return { passed: k, total: n, point: p, lo, hi, width: hi - lo };
}

// O território é o primeiro segmento do id, com uma normalização: as pastas
// sprintNN são 88 territórios de um artefato cada — fragmentam a página sem
// informar nada. Viram um território só, ordenado por número.
function territory(id) {
  const i = id.indexOf('/');
  const head = i === -1 ? '(raiz)' : id.slice(0, i);
  if (/^sprint\d+$/.test(head)) return 'sprints';
  return head;
}

// achata metrics em caminhos: "passed", "cases.total", ...
function flatten(obj, prefix = '', out = {}) {
  for (const [k, v] of Object.entries(obj ?? {})) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v && typeof v === 'object' && !Array.isArray(v)) flatten(v, key, out);
    else if (typeof v === 'number') out[key] = v;
  }
  return out;
}

const files = walk(ARTIFACTS);
const entries = [];
let skipped = 0;

for (const f of files) {
  let d;
  try { d = JSON.parse(readFileSync(f, 'utf-8')); }
  catch { skipped++; continue; }
  if (!d || typeof d !== 'object' || Array.isArray(d)) { skipped++; continue; }

  const id = relative(ARTIFACTS, f).replace(/\.json$/, '').replace(/\\/g, '/');
  const metrics = flatten(d.metrics ?? {});
  // alguns artefatos põem números na raiz
  for (const [k, v] of Object.entries(d)) {
    if (typeof v === 'number' && !(k in metrics)) metrics[k] = v;
  }
  // nem todo artefato respeita a forma: status/reason às vezes vêm como objeto
  const str = v => (typeof v === 'string' ? v : v == null ? null : JSON.stringify(v).slice(0, 200));
  const rawStatus = typeof d.status === 'string' ? d.status
                  : typeof d.gate === 'string' ? d.gate : null;

  const { kind, level } = classify(rawStatus);
  const prop = (typeof metrics.passed === 'number' && typeof metrics.total === 'number')
    ? wilson(metrics.passed, metrics.total) : null;

  entries.push({
    id,
    kind,
    proportion: prop,
    territory: territory(id),
    schema: str(d.schema),
    status: rawStatus,
    level,
    reason: str(d.reason),
    generatedAt: str(d.generated_at ?? d.timestamp ?? d.generated_at_utc ?? d.receipt_utc),
    metrics,
    claims: Array.isArray(d.novel_claims) ? d.novel_claims.length : 0,
  });
}

entries.sort((a, b) => a.id.localeCompare(b.id));

const byLevel = entries.reduce((a, e) => {
  if (e.level) a[e.level] = (a[e.level] ?? 0) + 1;
  return a;
}, {});
const byKind = entries.reduce((a, e) => ((a[e.kind] = (a[e.kind] ?? 0) + 1), a), {});
const byTerritory = entries.reduce((a, e) => ((a[e.territory] = (a[e.territory] ?? 0) + 1), a), {});
const metricCount = entries.reduce((n, e) => n + Object.keys(e.metrics).length, 0);
const withProp = entries.filter(e => e.proportion);
const allPass = withProp.filter(e => e.proportion.passed === e.proportion.total);
const widths = withProp.map(e => e.proportion.width).sort((a, b) => a - b);

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT,
`// GERADO por scripts/index-evidence.mjs — não edite à mão.
// ${entries.length} artefatos, ${metricCount} métricas numéricas indexadas.
// Gerado a partir de artifacts/**/*.json do repositório.

import type { Artifact } from './types';

export const EVIDENCE: Record<string, Artifact> = ${JSON.stringify(
  Object.fromEntries(entries.map(e => [e.id, e])), null, 1)} as const;

export const EVIDENCE_STATS = ${JSON.stringify({
  artifacts: entries.length, metrics: metricCount,
  byKind, byLevel, byTerritory,
  proportions: {
    count: withProp.length,
    allPass: allPass.length,
    widest: widths.length ? widths[widths.length - 1] : 0,
    median: widths.length ? widths[Math.floor(widths.length / 2)] : 0,
    narrowest: widths.length ? widths[0] : 0,
  },
}, null, 1)} as const;
`);

console.log(`indexados ${entries.length} artefatos (${skipped} ignorados)`);
console.log(`métricas numéricas: ${metricCount}`);
console.log('por espécie:', byKind);
console.log('portões por veredito:', byLevel);
console.log('territórios:', Object.keys(byTerritory).length);
console.log(`proporções: ${withProp.length} (${allPass.length} relatam todos-passaram)`);
