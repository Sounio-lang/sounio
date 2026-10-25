#!/usr/bin/env node
/** SeedReceipt metadata writer/checker. No numerical compiler implementation. */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import {fileURLToPath} from 'node:url';

const truth = v => v !== null && v !== undefined && v !== false && v !== 0 && v !== '' && (!Array.isArray(v) || v.length !== 0) && (typeof v !== 'object' || Array.isArray(v) || Object.keys(v).length !== 0);
const lower = v => (v || '').toLowerCase();
const pyString = v => v === true ? 'True' : v === false ? 'False' : v == null ? 'None' : String(v);
export const asciiJSON = v => JSON.stringify(v, null, 2).replace(/[\u007f-\uffff]/g, c => '\\u' + c.charCodeAt(0).toString(16).padStart(4, '0')) + '\n';
export function digestFile(p) {
  const b = fs.readFileSync(p);
  return {path: p, sha256: crypto.createHash('sha256').update(b).digest('hex'), md5: crypto.createHash('md5').update(b).digest('hex'), bytes: b.length};
}
function integer(s) {
  if (!/^[+-]?\d(?:_?\d)*$/.test(s)) throw new Error(`invalid integer: ${s}`);
  const n = Number(s.replaceAll('_', ''));
  if (!Number.isSafeInteger(n)) throw new Error(`integer outside exact metadata range: ${s}`);
  return n;
}
export function parseGensTSV(p) {
  const rows = [];
  for (const [i, raw] of fs.readFileSync(p, 'utf8').split(/\r?\n/).entries()) {
    const line = raw.trim();
    if (!line || line.startsWith('#')) continue;
    const parts = line.split('\t');
    if (parts.length < 3) throw new Error(`gens.tsv line ${i + 1}: need gen md5 sha256 [path]`);
    const g = parts[0].trim(), label = g.startsWith('g') ? g : `g${g}`;
    let index;
    try { index = integer(label.slice(1)); } catch { throw new Error(`gens.tsv line ${i + 1}: bad gen ${JSON.stringify(g)}`); }
    rows.push({gen: label, gen_index: index, md5: parts[1].trim().toLowerCase(), sha256: parts[2].trim().toLowerCase(), path: (parts[3] || '').trim()});
  }
  return rows.sort((a, b) => a.gen_index - b.gen_index);
}
export function buildFixedPoint(gens, k) {
  const byIndex = new Map(gens.map(g => [g.gen_index, g]));
  const a = byIndex.get(k), b = byIndex.get(k + 1);
  if (!a || !b) throw new Error(`settle-k=${k} requires g${k} and g${k + 1} in gens.tsv; have ${gens.map(g => g.gen).join(', ')}`);
  const em = a.md5 === b.md5, es = a.sha256 === b.sha256;
  return {criterion: 'md5(g_k) == md5(g_{k+1}) AND sha256(g_k) == sha256(g_{k+1})', k, k_plus_1: k + 1, gk_label: a.gen, gk_plus1_label: b.gen, gk_md5: a.md5, gk_plus1_md5: b.md5, gk_sha256: a.sha256, gk_plus1_sha256: b.sha256, md5_equal: em, sha256_equal: es, verified: em && es && !!a.md5 && !!a.sha256, eyeball_md5: `${a.md5}\n${b.md5}`, eyeball_sha256: `${a.sha256}\n${b.sha256}`};
}
export function receiptInternalErrors(data) {
  const e = [];
  if (data.schema != null && data.schema !== 'sounio.SeedReceipt') e.push(`schema must be sounio.SeedReceipt, got ${JSON.stringify(data.schema)}`);
  const fp = data.fixed_point || {};
  if (!truth(fp)) { e.push('missing fixed_point object'); return e; }
  if (!truth(fp.verified)) e.push('fixed_point.verified is not true');
  const m = lower(fp.gk_md5), m1 = lower(fp.gk_plus1_md5), s = lower(fp.gk_sha256), s1 = lower(fp.gk_plus1_sha256);
  if (!m || !m1) e.push('fixed_point missing gk_md5 / gk_plus1_md5 (must be written side by side)');
  else if (m !== m1) e.push(`md5 mismatch by eye (provenance refuse):\n  gk_md5:       ${m}\n  gk_plus1_md5: ${m1}`);
  if (!s || !s1) e.push('fixed_point missing gk_sha256 / gk_plus1_sha256');
  else if (s !== s1) e.push(`sha256 mismatch by eye:\n  gk_sha256:       ${s}\n  gk_plus1_sha256: ${s1}`);
  const gens = data.generations || [];
  if (gens.length < 2) e.push('generations[] must list at least g_k and g_{k+1} (need ≥2 entries)');
  if (m && !new Set(gens.map(g => lower(g.md5))).has(m)) e.push(`fixed_point.gk_md5 ${m} not present in generations[]`);
  if (!(data.source?.sha256 || data.source?.md5)) e.push('source missing sha256/md5');
  if (!(data.output_seed?.sha256 || data.output_seed?.md5)) e.push('output_seed missing sha256/md5');
  if (!(data.input_seed?.sha256 || data.input_seed?.md5)) e.push('input_seed missing sha256/md5 (g0 of the chain)');
  return e;
}
export function checkReceiptAgainstTree(data, src, seed) {
  const e = receiptInternalErrors(data);
  if (!fs.existsSync(src) || !fs.statSync(src).isFile()) { e.push(`source file missing: ${src}`); return e; }
  if (!fs.existsSync(seed) || !fs.statSync(seed).isFile()) { e.push(`seed ELF missing: ${seed}`); return e; }
  const a = digestFile(src), b = digestFile(seed);
  const ss = lower(data.source?.sha256), sm = lower(data.source?.md5), bs = lower(data.output_seed?.sha256), bm = lower(data.output_seed?.md5);
  if (ss && ss !== a.sha256) e.push(`PROVENANCE FAIL: receipt source.sha256 does not match committed lean_single.sio\n  receipt:   ${ss}\n  committed: ${a.sha256}\n  (A receipt that names another source cannot vouch for this tree.)`);
  else if (sm && !ss && sm !== a.md5) e.push(`PROVENANCE FAIL: receipt source.md5 does not match committed lean_single.sio\n  receipt:   ${sm}\n  committed: ${a.md5}`);
  else if (!ss && !sm) e.push('receipt source has neither sha256 nor md5');
  if (bs && bs !== b.sha256) e.push(`PROVENANCE FAIL: receipt output_seed.sha256 does not match committed seed ELF\n  receipt:   ${bs}\n  committed: ${b.sha256}`);
  else if (bm && !bs && bm !== b.md5) e.push(`PROVENANCE FAIL: receipt output_seed.md5 does not match committed seed ELF\n  receipt:   ${bm}\n  committed: ${b.md5}`);
  else if (!bs && !bm) e.push('receipt output_seed has neither sha256 nor md5');
  const fp = data.fixed_point || {};
  if (fp.gk_md5 && lower(fp.gk_md5) !== b.md5) e.push(`PROVENANCE FAIL: fixed_point.gk_md5 does not match committed seed ELF md5\n  gk_md5:    ${lower(fp.gk_md5)}\n  seed md5:  ${b.md5}`);
  if (fp.gk_sha256 && lower(fp.gk_sha256) !== b.sha256) e.push(`PROVENANCE FAIL: fixed_point.gk_sha256 does not match committed seed ELF sha256\n  gk_sha256:   ${lower(fp.gk_sha256)}\n  seed sha256: ${b.sha256}`);
  return e;
}
export function humanText(r) {
  const fp = r.fixed_point, env = r.environment;
  const lines = ['SeedReceipt', `schema_version: ${r.schema_version}`, `receipt_utc: ${r.receipt_utc}`, `git_commit: ${r.git_commit || ''}`, `git_branch: ${r.git_branch || ''}`, '', '## source (lean_single.sio)'];
  const addIdentity = x => { for (const key of ['path', 'sha256', 'md5', 'bytes']) lines.push(`${key}: ${x[key]}`); };
  addIdentity(r.source);lines.push('', '## input seed (g0 / bootstrap ELF used to start the chain)');addIdentity(r.input_seed);
  lines.push('', '## generations');
  for (const g of r.generations) lines.push(`  ${g.gen}: md5=${g.md5}  sha256=${g.sha256}  path=${g.path || ''}`);
  lines.push('', '## fixed_point (FIELD — confirm equality by eye; if you cannot, receipt proves nothing)', `criterion: ${fp.criterion}`, `k: ${fp.k}`, `k_plus_1: ${fp.k_plus_1}`, `gk_label: ${fp.gk_label}`, `gk_plus1_label: ${fp.gk_plus1_label}`, '--- md5 side-by-side (must be identical) ---', `gk_md5:       ${fp.gk_md5}`, `gk_plus1_md5: ${fp.gk_plus1_md5}`, `md5_equal: ${String(fp.md5_equal).toLowerCase()}`, '--- sha256 side-by-side (must be identical) ---', `gk_sha256:       ${fp.gk_sha256}`, `gk_plus1_sha256: ${fp.gk_plus1_sha256}`, `sha256_equal: ${String(fp.sha256_equal).toLowerCase()}`, `verified: ${String(fp.verified).toLowerCase()}`, '', '## output seed (installed / to commit)');
  addIdentity(r.output_seed);lines.push('', '## environment');
  for (const k of ['placement', 'hostname', 'slurm_partition', 'slurm_time', 'slurm_job_id', 'slurm_nodelist']) lines.push(`${k}: ${env[k] || ''}`);
  lines.push('', '## post checks', `canonical_compiler_gate: ${pyString(r.checks.canonical_compiler_gate)}`, `verify_lean_seed: ${pyString(r.checks.verify_lean_seed)}`, '', '## what this receipt does NOT prove', r.limits.provenance_note, '');
  return lines.join('\n');
}
const provenanceNote = 'This receipt proves the installed ELF is a self-reproducing fixed point of the recorded source bytes (g_k == g_{k+1}, self-repro). It does NOT by itself prove the ELF was *derived from* that source rather than substituted from another generation that happens to self-reproduce the same source. canonical_compiler_gate.sh checks md5(committed)==md5(self-compile) — stability, not provenance. Provenance (ELF came FROM this source via this chain) is what the generation table + input_seed hashes are for; a future gate should require a SeedReceipt, not only the self-repro equality.';
export function main(argv = process.argv.slice(2)) {
  const allowed = new Set(['verify-receipt', 'check-against-tree', 'seed-elf', 'out-dir', 'gens-tsv', 'source', 'input-seed', 'output-seed', 'settle-k', 'placement', 'slurm-partition', 'slurm-time', 'slurm-job-id', 'slurm-nodelist', 'hostname', 'git-commit', 'git-branch', 'canonical-gate', 'verify-lean-seed', 'schema-version']);
  const args = {placement: 'unknown', 'slurm-partition': '', 'slurm-time': '', 'slurm-job-id': '', 'slurm-nodelist': '', hostname: '', 'git-commit': '', 'git-branch': '', 'canonical-gate': 'skip', 'verify-lean-seed': 'skip', 'schema-version': '1'};
  for (let i = 0; i < argv.length; ++i) {
    if (argv[i] === '--help' || argv[i] === '-h') { console.log('SeedReceipt metadata writer/checker\n' + [...allowed].map(k => `  --${k} VALUE`).join('\n'));return 0; }
    const match = /^--([^=]+)(?:=(.*))?$/.exec(argv[i]);
    if (!match || !allowed.has(match[1])) throw new Error(`unrecognized argument: ${argv[i]}`);
    const value = match[2] ?? argv[++i];if (value === undefined) throw new Error(`missing argument for --${match[1]}`);args[match[1]] = value;
  }
  for (const [k, choices] of [['placement', ['slurm','local-locked','check-only','unknown']], ['canonical-gate', ['pass','fail','skip']], ['verify-lean-seed', ['pass','fail','skip']]]) if (!choices.includes(args[k])) throw new Error(`invalid --${k}: ${args[k]}`);
  args['schema-version'] = integer(args['schema-version']);
  if (args['settle-k'] !== undefined) args['settle-k'] = integer(args['settle-k']);
  if (args['verify-receipt']) {
    const p = args['verify-receipt'], data = JSON.parse(fs.readFileSync(p, 'utf8')), e = receiptInternalErrors(data);
    if (e.length) { console.error(`[seed-receipt] FAIL ${p}`);for (const s of e) console.error(`  - ${s}`);return 1; }
    console.log(`[seed-receipt] PASS fixed_point verified by eye fields in ${p}\n  gk_md5:       ${data.fixed_point.gk_md5}\n  gk_plus1_md5: ${data.fixed_point.gk_plus1_md5}`);return 0;
  }
  if (args['check-against-tree']) {
    const p = args['check-against-tree'], src = args.source, seed = args['seed-elf'] || args['output-seed'];
    if (!src || !seed) throw new Error('--check-against-tree requires --source and --seed-elf (or --output-seed)');
    const e = checkReceiptAgainstTree(JSON.parse(fs.readFileSync(p, 'utf8')), path.resolve(src), path.resolve(seed));
    if (e.length) { console.error(`[seed-receipt] FAIL provenance check ${p}`);for (const s of e) console.error(`  - ${s}`);return 1; }
    console.log(`[seed-receipt] PASS provenance: receipt matches source+seed on disk\n  source=${src}\n  seed=${seed}`);return 0;
  }
  for (const k of ['out-dir', 'gens-tsv', 'source', 'input-seed', 'output-seed', 'settle-k']) if (args[k] === undefined) throw new Error('emit mode requires --out-dir --gens-tsv --source --input-seed --output-seed --settle-k');
  for (const k of ['source', 'input-seed', 'output-seed', 'gens-tsv']) if (!fs.statSync(args[k]).isFile()) throw new Error(`missing file: ${args[k]}`);
  const gens = parseGensTSV(args['gens-tsv']), fp = buildFixedPoint(gens, args['settle-k']);
  if (!fp.verified) throw new Error(`refusing to emit receipt: fixed_point not verified (md5_equal=${pyString(fp.md5_equal)} sha256_equal=${pyString(fp.sha256_equal)})`);
  const utc = new Date().toISOString().slice(0, 19) + 'Z', stamp = utc.replaceAll('-', '').replaceAll(':', '');
  const receipt = {schema: 'sounio.SeedReceipt', schema_version: args['schema-version'], receipt_utc: utc, git_commit: args['git-commit'], git_branch: args['git-branch'], source: digestFile(fs.realpathSync(args.source)), input_seed: {...digestFile(fs.realpathSync(args['input-seed'])), role: 'g0_start_of_chain'}, generations: gens, fixed_point: fp, output_seed: digestFile(fs.realpathSync(args['output-seed'])), environment: {placement: args.placement, hostname: args.hostname || os.hostname(), slurm_partition: args['slurm-partition'], slurm_time: args['slurm-time'], slurm_job_id: args['slurm-job-id'] || process.env.SLURM_JOB_ID || '', slurm_nodelist: args['slurm-nodelist'] || process.env.SLURM_NODELIST || ''}, checks: {canonical_compiler_gate: args['canonical-gate'], verify_lean_seed: args['verify-lean-seed']}, limits: {provenance_note: provenanceNote}};
  if (receipt.output_seed.md5 !== fp.gk_md5) throw new Error(`output seed md5 ${receipt.output_seed.md5} != settled gk_md5 ${fp.gk_md5}`);
  fs.mkdirSync(args['out-dir'], {recursive: true});
  const base = path.join(args['out-dir'], `SeedReceipt-${stamp}`), json = asciiJSON(receipt), txt = humanText(receipt);
  fs.writeFileSync(`${base}.json`, json);fs.writeFileSync(`${base}.txt`, txt);fs.writeFileSync(path.join(args['out-dir'], 'SeedReceipt.latest.json'), json);fs.writeFileSync(path.join(args['out-dir'], 'SeedReceipt.latest.txt'), txt);
  console.log(`[seed-receipt] wrote ${base}.json\n[seed-receipt] wrote ${base}.txt\n[seed-receipt] fixed_point.verified=${pyString(fp.verified)}\n[seed-receipt] gk_md5:       ${fp.gk_md5}\n[seed-receipt] gk_plus1_md5: ${fp.gk_plus1_md5}`);return 0;
}
let isMain = false;
try { isMain = !!process.argv[1] && fs.realpathSync(process.argv[1]) === fileURLToPath(import.meta.url); } catch { /* imports from stdin have no script file */ }
if (isMain) {
  try { process.exitCode = main(); } catch (e) { console.error(`[seed-receipt] ${e.message}`);process.exitCode = 1; }
}
