#!/usr/bin/env node
// SPDX-License-Identifier: MIT
// Structural name-neighbour census; edit distance does not prove a typo.
const fs = require('node:fs');
const path = require('node:path');
const { maskCommentsAndStrings, walk } = require('./measure_with_mod.cjs');

function collectWithNames(text) {
  const names = new Map(), cleaned = maskCommentsAndStrings(text);
  for (const m of cleaned.matchAll(/\bwith\s+([A-Z][A-Za-z0-9]*(?:\s*,\s*[A-Z][A-Za-z0-9]*)*)/g)) {
    for (const name of m[1].split(',').map(x => x.trim())) names.set(name, (names.get(name) || 0) + 1);
  }
  return names;
}
function recognizedNames(root) {
  const source = fs.readFileSync(path.join(root, 'self-hosted/check/effects.sio'), 'utf8');
  const enumBody = maskCommentsAndStrings(source).match(/enum EffectKind\s*\{([^}]+)\}/)?.[1];
  if (!enumBody) throw Error('missing checker EffectKind enum');
  const names = [...enumBody.matchAll(/\bEff([A-Z][A-Za-z0-9]*)\b/g)].map(m => m[1]).filter(n => n !== 'Var');
  // The mapping explicitly admits Confidence as an Epistemic alias.
  if (!source.includes('Confidence is an alias of Epistemic')) throw Error('checker alias contract changed; review scanner');
  return [...names, 'Confidence'];
}
function lev(a,b) {
  let previous = Array.from({length:b.length+1},(_,i)=>i);
  for(let i=1;i<=a.length;i++){const current=[i];for(let j=1;j<=b.length;j++)current[j]=a[i-1]===b[j-1]?previous[j-1]:1+Math.min(previous[j],current[j-1],previous[j-1]);previous=current}
  return previous[b.length];
}
function selftest() {
  const assert = require('node:assert/strict');
  assert.deepEqual([...collectWithNames('fn a() -> i64 with Div, Mod, IO { 0 }\nfn b() with Mod, Mut { 0 }\n"with Fake" // with Wrong').entries()], [['Div',1],['Mod',2],['IO',1],['Mut',1]]);
  assert.equal(lev('Mod','Mut'),2);
  const names = recognizedNames(path.resolve(__dirname,'../..'));
  for(const name of ['Approx','NaturalityG2','Deterministic','Perturbative','NarrowWidthApproximation','NonUnitary','Correlated','Confidence'])assert(names.includes(name));
  assert(!names.includes('Var')); assert(!names.includes('Mod'));
  console.log('WITH_NEIGHBORS_STRUCTURAL_SELFTEST_PASS');
}
function main(root) {
  const builtins = recognizedNames(root), all = new Map(), declared = new Set();
  for(const file of walk(root)){const text=fs.readFileSync(file,'utf8');for(const[n,c]of collectWithNames(text))all.set(n,(all.get(n)||0)+c);for(const m of maskCommentsAndStrings(text).matchAll(/\b(?:pub\s+)?effect\s+([A-Z][A-Za-z0-9]*)/g))declared.add(m[1])}
  const neighbours=[...all].filter(([n])=>!builtins.includes(n)).map(([name,count])=>{let best={distance:Infinity,builtin:null};for(const builtin of builtins){const distance=lev(name,builtin);if(distance<best.distance)best={distance,builtin}}return{name,count,declared:declared.has(name),...best}}).sort((a,b)=>a.distance-b.distance||a.name.localeCompare(b.name));
  console.log(JSON.stringify({kind:'structural-source-heuristic',recognizedNames:builtins,neighbours,limits:['Near spelling is not evidence of typo or intended effect.','Counts are textual with-clauses, not typechecker results.','Walk excludes archive/bootstrap/target/node_modules and symlinks.']},null,2));
}
if(process.argv[2]==='--selftest')selftest();else main(process.argv[2]||'.');
