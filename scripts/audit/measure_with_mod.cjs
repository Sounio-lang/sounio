#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Structural source census, not a type checker or an executable effect oracle.
const fs = require('node:fs');
const path = require('node:path');

function maskCommentsAndStrings(text) {
  let mode = 'code', depth = 0, out = '';
  for (let i = 0; i < text.length; i++) {
    const ch = text[i], next = text[i + 1];
    if (mode === 'line') { if (ch === '\n') mode = 'code'; out += ch === '\n' ? '\n' : ' '; continue; }
    if (mode === 'block') {
      if (ch === '/' && next === '*') { depth++; out += '  '; i++; }
      else if (ch === '*' && next === '/') { depth--; out += '  '; i++; if (!depth) mode = 'code'; }
      else out += ch === '\n' ? '\n' : ' ';
      continue;
    }
    if (mode === 'string') {
      if (ch === '\\' && next !== undefined) { out += '  '; i++; }
      else { if (ch === '"') mode = 'code'; out += ch === '\n' ? '\n' : ' '; }
      continue;
    }
    if (ch === '/' && next === '/') { mode = 'line'; out += '  '; i++; }
    else if (ch === '/' && next === '*') { mode = 'block'; depth = 1; out += '  '; i++; }
    else if (ch === '"') { mode = 'string'; out += ' '; }
    else out += ch;
  }
  return out;
}

function* walk(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      if (['node_modules', '.git', 'archive', 'bootstrap', 'target'].includes(ent.name)) continue;
      yield* walk(full);
    } else if (ent.isFile() && ent.name.endsWith('.sio')) yield full;
  }
}

function extractFunctionBodies(text, file) {
  const masked = maskCommentsAndStrings(text), out = [], re = /\bfn\s+[A-Za-z_]\w*/g;
  let match;
  while ((match = re.exec(masked))) {
    const open = masked.indexOf('{', re.lastIndex);
    if (open < 0) break;
    const signature = masked.slice(match.index, open);
    // A declaration with no body must not consume the following definition.
    if (/\bfn\s+[A-Za-z_]\w*/.test(signature.slice(match[0].length))) continue;
    const clauses = [...signature.matchAll(/\bwith\s+([A-Z][A-Za-z0-9]*(?:\s*,\s*[A-Z][A-Za-z0-9]*)*)/g)];
    const effects = clauses.flatMap(m => m[1].split(',').map(x => x.trim()));
    let close = open + 1, depth = 1;
    while (close < masked.length && depth) { if (masked[close] === '{') depth++; if (masked[close] === '}') depth--; close++; }
    if (depth) continue;
    out.push({ file, line: text.slice(0, match.index).split('\n').length, effects, body: masked.slice(open + 1, close - 1) });
    re.lastIndex = close;
  }
  return out;
}

function classify(body) {
  const fields = /[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\]]+\])\s*=(?!=)/.test(body);
  const refBang = /&!/.test(body), vars = [...body.matchAll(/\bvar\s+([A-Za-z_]\w*)/g)];
  const reassigned = vars.some(m => {
    const after = body.slice(m.index + m[0].length);
    return (after.match(new RegExp(`\\b${m[1]}\\s*=(?!=)`, 'g')) || []).length > 0;
  });
  const writes = fields || refBang || reassigned, mod = /%/.test(body);
  return { writes, mod, varOnly: vars.length > 0 && !writes };
}

function selftest() {
  const assert = require('node:assert/strict');
  const sample = 'fn a() -> i64 with Div, Mod { var x = 0 var y = 1 y = 2 "} % with Fake" /* { */ x }\nfn b() with Mod, IO { 1 % 2 }\nfn c() with\n Mut, Div, Mod\n{ 4 }\n// fn fake() with Mod { 0 }';
  const f = extractFunctionBodies(sample, 'fixture');
  assert.equal(f.length, 3); assert.deepEqual(f.map(x => x.effects), [['Div','Mod'],['Mod','IO'],['Mut','Div','Mod']]);
  assert.equal(classify(f[0].body).writes, true); assert.equal(classify(f[0].body).mod, false);
  assert.equal(classify(f[1].body).mod, true); assert.equal(classify(f[2].body).writes, false);
  assert.equal(maskCommentsAndStrings('"// with Mod"\n/* outer /* inner */ with Fake */ with IO').includes('Fake'), false);
  console.log('WITH_MOD_STRUCTURAL_SELFTEST_PASS');
}

function main(root) {
  const counts = { declarations: 0, writesAndModulo: 0, writesOnly: 0, moduloOnly: 0, varOnly: 0, neither: 0 }, coOccurrence = new Map();
  let files = 0;
  for (const file of walk(root)) {
    files++;
    for (const f of extractFunctionBodies(fs.readFileSync(file, 'utf8'), file)) {
      if (!f.effects.includes('Mod')) continue;
      counts.declarations++;
      for (const effect of new Set(f.effects)) if (effect !== 'Mod') coOccurrence.set(effect, (coOccurrence.get(effect) || 0) + 1);
      const c = classify(f.body);
      counts[c.writes && c.mod ? 'writesAndModulo' : c.writes ? 'writesOnly' : c.mod ? 'moduloOnly' : c.varOnly ? 'varOnly' : 'neither']++;
    }
  }
  console.log(JSON.stringify({ kind: 'structural-source-heuristic', files, counts, coOccurrence: Object.fromEntries([...coOccurrence].sort()), limits: ['Text patterns are not semantic mutation/effect proofs.', 'Function-signature clauses may include nested function-type effects.', 'Walk excludes archive/bootstrap/target/node_modules and symlinks; not every tracked source file.'] }, null, 2));
}
module.exports = { maskCommentsAndStrings, walk, extractFunctionBodies, classify };
if (require.main === module) { if (process.argv[2] === '--selftest') selftest(); else main(process.argv[2] || '.'); }
