#!/usr/bin/env node
// SPDX-License-Identifier: MIT
//
// measure_with_neighbors.cjs — generalize the "with Mod" question to ALL
// `with X` names in the tree. The dispatch's question 4: how many other
// `with` names are at edit distance 1 or 2 from a builtin?
//
// Builtin effects (23): IO, Mut, Alloc, Panic, Div, GPU, Async, Prob,
// Epistemic, Causal, Network, Sensor, Render, Observe, NonAssoc, Audit,
// Hypothesis, MultiTest, ZD, Witness, Temporal, Learn, Chaotic.
//
// User-declared effects (from `effect X { ... }`) are searched separately
// and considered as "declared", not "near-builtin".
//
// Output:
//   1. The set of all distinct `with X` names found, with their count.
//   2. For each name, the closest builtin by Levenshtein distance.
//   3. Names at distance 1 (typo candidate).
//   4. Names at distance 2 (typo candidate).
//   5. Names with no builtin nearby (distance > 2) AND no `effect X` in tree
//      (orphan effect — neither typo nor declared).
//
// Companion script to docs/audit/EFFECT_MOD_MEANING_MEASUREMENT_2026-08-19.md.

const fs = require('node:fs');
const path = require('node:path');

const ROOT = process.argv[2] || '.';

const BUILTINS = [
  'IO', 'Mut', 'Alloc', 'Panic', 'Div', 'GPU', 'Async', 'Prob',
  'Epistemic', 'Causal', 'Network', 'Sensor', 'Render', 'Observe',
  'NonAssoc', 'Audit', 'Hypothesis', 'MultiTest', 'ZD', 'Witness',
  'Temporal', 'Learn', 'Chaotic',
];

function* walk(dir) {
  for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      if (ent.name === 'node_modules' || ent.name === '.git' || ent.name === 'archive' || ent.name === 'bootstrap' || ent.name === 'target') continue;
      yield* walk(full);
    } else if (ent.isFile() && ent.name.endsWith('.sio')) {
      yield full;
    }
  }
}

function lev(a, b) {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      if (a[i-1] === b[j-1]) dp[i][j] = dp[i-1][j-1];
      else dp[i][j] = 1 + Math.min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]);
    }
  }
  return dp[m][n];
}

// Strip Sounio-style comments. Both // line comments and /* block */ comments.
// Also strip string literals to avoid matching `with Foo` inside a string.
function stripCommentsAndStrings(text) {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/\/\/[^\n]*/g, '')
    .replace(/"(?:[^"\\]|\\.)*"/g, '""');
}

// Parse every `with X[, Y, ...]` clause in the tree. Returns Map<name, count>.
function collectWithNames(text) {
  const cleaned = stripCommentsAndStrings(text);
  const names = new Map();
  // Match `with <ident>` where ident is a PascalCase token.
  const re = /\bwith\s+([A-Z][A-Za-z0-9]*)/g;
  let m;
  while ((m = re.exec(cleaned)) !== null) {
    const name = m[1];
    names.set(name, (names.get(name) || 0) + 1);
  }
  return names;
}

// Find every `effect X` (or `pub effect X`) declaration in the tree.
function collectDeclaredEffects(text) {
  const cleaned = stripCommentsAndStrings(text);
  const out = new Set();
  const re = /\b(?:pub\s+)?effect\s+([A-Z][A-Za-z0-9]*)/g;
  let m;
  while ((m = re.exec(cleaned)) !== null) out.add(m[1]);
  return out;
}

(async () => {
  const allWithNames = new Map();
  const allDeclared = new Set();
  for (const f of walk(ROOT)) {
    const text = fs.readFileSync(f, 'utf8');
    const names = collectWithNames(text);
    for (const [n, c] of names) allWithNames.set(n, (allWithNames.get(n) || 0) + c);
    for (const d of collectDeclaredEffects(text)) allDeclared.add(d);
  }

  // For each non-builtin `with X`, find closest builtin.
  const distance = new Map();
  for (const name of allWithNames.keys()) {
    if (BUILTINS.includes(name)) continue;  // skip builtins themselves
    let best = { d: Infinity, builtin: null };
    for (const b of BUILTINS) {
      const d = lev(name, b);
      if (d < best.d) best = { d, builtin: b };
    }
    distance.set(name, best);
  }

  const sortedByDistance = [...distance.entries()]
    .sort((a, b) => a[1].d - b[1].d || a[0].localeCompare(b[0]));

  console.log(`Total distinct \`with X\` names (excluding builtins): ${distance.size}`);
  console.log(`Total builtin effects in scope: ${BUILTINS.length}`);
  console.log('');

  // Distance-1 (typo candidate)
  const d1 = sortedByDistance.filter(([_, v]) => v.d === 1);
  console.log(`Names at edit distance 1 from a builtin (TYPO CANDIDATES): ${d1.length}`);
  for (const [name, { builtin, d }] of d1) {
    const count = allWithNames.get(name);
    const declared = allDeclared.has(name) ? ' (DECLARED)' : '';
    console.log(`  ${name.padEnd(15)} -> ${builtin.padEnd(12)} (distance ${d}, ${count} uses)${declared}`);
  }
  console.log('');

  // Distance-2
  const d2 = sortedByDistance.filter(([_, v]) => v.d === 2);
  console.log(`Names at edit distance 2 from a builtin: ${d2.length}`);
  for (const [name, { builtin, d }] of d2) {
    const count = allWithNames.get(name);
    const declared = allDeclared.has(name) ? ' (DECLARED)' : '';
    console.log(`  ${name.padEnd(15)} -> ${builtin.padEnd(12)} (distance ${d}, ${count} uses)${declared}`);
  }
  console.log('');

  // Orphans: distance > 2 AND not declared AND not in BUILTINS
  const orphans = sortedByDistance
    .filter(([name, v]) => v.d > 2 && !allDeclared.has(name) && !BUILTINS.includes(name));
  console.log(`ORPHAN effect names (no nearby builtin, not declared): ${orphans.length}`);
  for (const [name, { builtin, d }] of orphans) {
    const count = allWithNames.get(name);
    console.log(`  ${name.padEnd(20)} (closest: ${builtin} at distance ${d}, ${count} uses)`);
  }
  console.log('');

  // Top 10 most-used `with` names that aren't builtins
  const top = [...allWithNames.entries()]
    .filter(([n]) => !BUILTINS.includes(n))
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15);
  console.log('Top 15 most-used non-builtin `with X` names:');
  for (const [name, count] of top) {
    const dist = distance.get(name);
    const distStr = dist ? `dist=${dist.d}->${dist.builtin}` : 'n/a';
    const declared = allDeclared.has(name) ? 'DECLARED' : 'orphan';
    console.log(`  ${name.padEnd(20)} ${count.toString().padStart(5)} uses  [${declared}] [${distStr}]`);
  }
})();