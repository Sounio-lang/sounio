#!/usr/bin/env node
// Text-only scanner controls: these snippets are not compiled/executed Sounio.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { bareFunctionTypes } from './fn_type_effect_scan.mjs';

const cases = [
  ['bare scalar', 'fn d(f: fn(f64) -> f64, x:f64) -> f64 with Div {0.0}', 1],
  ['annotated scalar', 'fn d(f: fn(f64) -> f64 with Div, x:f64) -> f64 {0.0}', 0],
  ['named declaration', 'fn d(x:f64) -> f64 with Div {0.0}', 0],
  ['line comment', '// d(f: fn(f64) -> f64)\nfn k()->i64 {0}', 0],
  ['return position', 'fn k(x:i64) -> fn(i64) -> i64 with Mut {f}', 0],
  ['annotated unit', 'fn k(f: fn(f64,&[f64;64],&![f64;64])->() with Mut,Div,Panic, x:f64) { }', 0],
  ['bare unit', 'fn k(f: fn(f64,&[f64;64],&![f64;64])->(), x:f64) with Mut,Div,Panic { }', 1],
  ['annotated tuple', 'fn k(f: fn(f64)->(f64,(i64,f64)) with Div, x:f64) { }', 0],
  ['bare tuple', 'fn k(f: fn(f64)->(f64,(i64,f64)), x:f64) with Div { }', 1],
  ['annotated array', 'fn k(f: fn(i64)->[f64;64] with Panic, x:i64) { }', 0],
  ['separate annotations', 'fn k(f: fn(i64)->() with Mut, g: fn(f64)->(), x:i64) with Div { }', 1],
  ['annotated generic', 'fn k(f: fn(i64)->Result<f64,Error> with Div, x:i64) { }', 0],
  ['bare generic', 'fn k(f: fn(i64)->Result<f64,Error>, x:i64) with Div { }', 1],
  ['bare unit next annotated parameter', 'fn k(f: fn(i64)->(), g: fn(f64)->f64 with Div) { }', 1],
  ['outer effects not owned', 'fn k(f: fn(i64)->()) -> i64 with Mut {0}', 1],
];
for (const [name, source, expected] of cases) {
  assert.equal(bareFunctionTypes(source).length, expected, name);
}
// Reproduce the old instrument's exact root failure on an annotated Unit return.
const annotated = cases.find(([name]) => name === 'annotated unit')[1];
const oldHits = annotated.match(/:\s*fn\([^)]*\)\s*->[^,)]*/g) ?? [];
assert.equal(oldHits.filter(hit => !/\bwith\s+[A-Za-z]/.test(hit)).length, 1);
assert.equal(bareFunctionTypes(annotated).length, 0);
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'fn-type-scan-'));
try {
  const bare = path.join(directory, 'bare.txt');
  const annotatedFile = path.join(directory, 'annotated.txt');
  fs.writeFileSync(bare, cases.find(([name]) => name === 'bare unit')[1]);
  fs.writeFileSync(annotatedFile, annotated);
  const scanner = fileURLToPath(new URL('./fn_type_effect_scan.mjs', import.meta.url));
  const direct = execFileSync(process.execPath, [scanner, bare, annotatedFile], { encoding: 'utf8' });
  assert.ok(direct.startsWith(bare + '\t'));
  assert.equal(direct.trimEnd().split('\n').length, 1);
  const nul = execFileSync(process.execPath, [scanner, '--files0'], {
    input: bare + '\0' + annotatedFile + '\0', encoding: 'utf8',
  });
  assert.equal(nul, direct, 'NUL corpus CLI has same non-vacuous result');
} finally {
  fs.rmSync(directory, { recursive: true });
}
console.log('FN_TYPE_EFFECT_SCAN_CONTROLS_PASS: 15 lexical controls + direct/NUL CLI; legacy Unit false-positive reproduced');
