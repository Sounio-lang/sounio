#!/usr/bin/env node
// SPDX-License-Identifier: MIT
//
// measure_with_mod.cjs — classify every `with Mod` function body in the
// Sounio corpus to test the three hypotheses from the 2026-08-19 dispatch:
//
//   H1 (typo for Mut):       function WRITES to memory (var / &! / arr[i]= / .field=)
//   H2 (modular arithmetic):  function uses `%` (modulo)
//   H3 (decorative):          function does neither — declaration is unused
//
// Counts are exact (not sampled). Sample size = every declaration in the
// tree at scan time, with the body extracted by brace-matching from the
// `with Mod` line.
//
// Output goes to stdout in plain text. Companion script to docs/audit/
// EFFECT_MOD_MEANING_MEASUREMENT_2026-08-19.md.

const fs = require('node:fs');
const path = require('node:path');

const ROOT = process.argv[2] || '.';

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

// Extract every function whose signature contains the literal token
// `with Mod` (as one of the comma-separated effects in a `with X, Y, Z`
// clause). Returns [{file, line, body}], where body is the text from the
// function signature line through the matching closing brace.
function extractWithModBodies(text, filePath) {
  const lines = text.split('\n');
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    // Match `with Mod` as a token in a `with X[, Y, ...]` effect clause.
    // The line must look like a function signature: contains `fn` OR ends with
    // `with Mod` (followed by space, brace, or comma), and the tokens between
    // `with` and `Mod` and between `Mod` and end-of-line must look like effect
    // identifiers (PascalCase, no `=`, no `(`, no unbalanced parens).
    const m = lines[i].match(/\bwith\s+([^()]*?)\bMod\b([^()]*?)\s*$/);
    if (!m) continue;
    const left = (m[1] || '').trim();
    const right = (m[2] || '').trim();
    // Effects are PascalCase identifiers with optional commas and spaces.
    // Anything else (function calls, string interpolation, arg defaults)
    // disqualifies.
    // DEBUG: log first 3 rejections to understand
    if (process.env.DEBUG_MOD) {
      console.error(`L${i+1}: left=${JSON.stringify(left)} right=${JSON.stringify(right)} | line=${JSON.stringify(lines[i].slice(0,120))}`);
    }
    if (left && !/^[A-Z][A-Za-z0-9]*(\s*,\s*[A-Z][A-Za-z0-9]*)*$/.test(left)) {
      if (process.env.DEBUG_MOD) console.error(`  rejected: left`);
      continue;
    }
    // Right side: empty (signature continues to next line) OR ends with `{`,
    // optionally preceded by `, Effect` and an identifier like `, Mod, ...`.
    if (right) {
      const rstrip = right.replace(/\s*\{$/, '');
      if (rstrip && !/^[A-Z][A-Za-z0-9]*(\s*,\s*[A-Z][A-Za-z0-9]*)*$/.test(rstrip)) {
        if (process.env.DEBUG_MOD) console.error(`  rejected: right=${JSON.stringify(right)}`);
        continue;
      }
    }
    if (process.env.DEBUG_MOD) console.error(`  ACCEPTED`);

    // Extract body via brace-matching from the end of this line.
    let j = i, depth = 0, started = false;
    const lineHasBrace = lines[i].includes('{');
    if (lineHasBrace) {
      for (const ch of lines[i]) {
        if (ch === '{') { depth++; started = true; }
        else if (ch === '}') { depth--; }
      }
      if (started && depth === 0) {
        out.push({ file: filePath, line: i+1, body: lines[i] });
        continue;
      }
      j = i + 1;
    } else {
      // Multi-line signature: signature continues until we find `{`.
      j = i + 1;
      while (j < lines.length && !lines[j].includes('{')) j++;
    }
    while (j < lines.length) {
      for (const ch of lines[j]) {
        if (ch === '{') { depth++; started = true; }
        else if (ch === '}') { depth--; }
      }
      if (started && depth === 0) break;
      j++;
    }
    if (started) {
      out.push({ file: filePath, line: i+1, body: lines.slice(i, j+1).join('\n') });
    }
  }
  return out;
}

// Classify a body. Returns:
//   { writes, mod, writesAndMod, neither }
// where each is boolean indicating the property.
function classify(body) {
  // Strip comment-only lines (avoid false positives from `// Mod-related thing`).
  const code = body
    .split('\n')
    .filter(l => !l.trim().startsWith('//'))
    .join('\n');

  // H1 (writes): `var X`, `X.field =`, `arr[i] =`, `&!` call site.
  //   Conservative regex: any `=` that is NOT ==, !=, <=, >=, =>
  //   and is preceded by an identifier or `.` or `]` (an lvalue).
  const hasVarDecl = /\bvar\s+[A-Za-z_]\w*\b/.test(code);
  const hasFieldOrIndexAssign = /[A-Za-z_]\w*(?:\.[A-Za-z_]\w*|\[[^\]]+\])\s*=(?!=)/.test(code);
  const hasRefBang = /&!/.test(code);
  const writes = hasVarDecl || hasFieldOrIndexAssign || hasRefBang;

  // Real write = either field/array assignment OR a var that is reassigned
  // (not just declared and read). Detect reassignment by searching for
  // `IDENT =` after the var declaration line.
  let realWrites = hasFieldOrIndexAssign || hasRefBang;
  if (hasVarDecl && !realWrites) {
    const varMatch = code.match(/\bvar\s+([A-Za-z_]\w*)/);
    if (varMatch) {
      const vname = varMatch[1];
      // Find a later line assigning to vname (not `==`, not in a comment).
      const reassignRe = new RegExp(`\\b${vname}\\s*=(?!=)`, 'g');
      const all = code.match(reassignRe) || [];
      // The first match is the declaration's `var X = ...`. If there's a
      // second match, it's a real reassignment.
      if (all.length >= 2) realWrites = true;
    }
  }

  // H2 (modular arithmetic): at least one `%` operator outside of strings.
  //   Sounio uses `%` for integer modulo. False positives would be in
  //   format strings ("%d"); we strip string literals before checking.
  const noStrings = code.replace(/"(?:[^"\\]|\\.)*"/g, '""');
  const mod = /%/.test(noStrings);

  return {
    writes,
    realWrites,
    mod,
    writesAndMod: writes && mod,
    neither: !writes && !mod,
    varDeclaredNotMutated: hasVarDecl && !realWrites && !hasFieldOrIndexAssign && !hasRefBang,
  };
}

(async () => {
  const files = [];
  for (const f of walk(ROOT)) files.push(f);
  console.error(`walked ${files.length} files`);

  let total = 0;
  let regexTotal = 0;
  const buckets = {
    realWritesOnly: 0,    // genuinely mutates (var reassigned, field/arr/&!)
    realWritesAndMod: 0,  // mutates AND uses %
    modOnly: 0,           // pure-mod (no real writes)
    varDeclaredNotMutated: 0,  // `var X = ...; return X` — var declared but never reassigned
    neither: 0,           // no var, no field-write, no %
  };
  const coOccur = { mut: 0, div: 0, panic: 0, alloc: 0, io: 0 }; // co-occurrence counts with other effects on the same signature
  const examples = { realWritesOnly: [], realWritesAndMod: [], modOnly: [], varDeclaredNotMutated: [], neither: [] };

  for (const f of files) {
    const text = fs.readFileSync(f, 'utf8');
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(/\bwith\s+([^()]*?)\bMod\b([^()]*?)\s*$/);
      if (m) regexTotal++;
    }
    const bodies = extractWithModBodies(text, f);
    for (const b of bodies) {
      total++;
      const c = classify(b.body);
      const sigLine = b.body.split('\n')[0];
      // co-occurrence with other effects
      const sigOnly = sigLine.replace(/->.*$/, '').replace(/fn\s+[A-Za-z_]\w*\s*\(/, '(');
      const effects = (sigOnly.match(/\bwith\s+(.+)$/) || [,''])[1];
      if (effects.includes('Mut')) coOccur.mut++;
      if (effects.includes('Div')) coOccur.div++;
      if (effects.includes('Panic')) coOccur.panic++;
      if (effects.includes('Alloc')) coOccur.alloc++;
      if (effects.includes('IO')) coOccur.io++;
      // bucket
      if (c.realWrites && c.mod) {
        buckets.realWritesAndMod++;
        if (examples.realWritesAndMod.length < 3) examples.realWritesAndMod.push({ file: f.replace(ROOT + '/', ''), line: b.line });
      } else if (c.realWrites) {
        buckets.realWritesOnly++;
        if (examples.realWritesOnly.length < 3) examples.realWritesOnly.push({ file: f.replace(ROOT + '/', ''), line: b.line });
      } else if (c.mod) {
        buckets.modOnly++;
        if (examples.modOnly.length < 3) examples.modOnly.push({ file: f.replace(ROOT + '/', ''), line: b.line });
      } else if (c.varDeclaredNotMutated) {
        buckets.varDeclaredNotMutated++;
        if (examples.varDeclaredNotMutated.length < 3) examples.varDeclaredNotMutated.push({ file: f.replace(ROOT + '/', ''), line: b.line });
      } else {
        buckets.neither++;
        if (examples.neither.length < 3) examples.neither.push({ file: f.replace(ROOT + '/', ''), line: b.line });
      }
    }
  }

  console.log(`Total \`with Mod\` declarations in the tree: ${total}`);
  console.log(`(raw regex matches before filtering: ${regexTotal})`);
  console.log('');
  console.log('Classification of function bodies:');
  console.log(`  REAL_WRITES_ONLY   (var reassigned / field / arr[i]= / &!)   : ${buckets.realWritesOnly}`);
  console.log(`  REAL_WRITES_AND_MOD (both)                                  : ${buckets.realWritesAndMod}`);
  console.log(`  MOD_ONLY            (% present, no real writes)             : ${buckets.modOnly}`);
  console.log(`  VAR_NO_REASSIGN     (\`var X = ...; return X\`, no mutation)  : ${buckets.varDeclaredNotMutated}`);
  console.log(`  NEITHER             (no var, no field-write, no %)           : ${buckets.neither}`);
  console.log('');
  console.log('Co-occurrence with other effects on the SAME signature:');
  console.log(`  with Mut on the same line  : ${coOccur.mut}`);
  console.log(`  with Div on the same line  : ${coOccur.div}`);
  console.log(`  with Panic on the same line: ${coOccur.panic}`);
  console.log(`  with Alloc on the same line: ${coOccur.alloc}`);
  console.log(`  with IO on the same line   : ${coOccur.io}`);
  console.log('');
  console.log('Examples (3 per category):');
  for (const k of ['realWritesOnly', 'realWritesAndMod', 'modOnly', 'varDeclaredNotMutated', 'neither']) {
    console.log(`  ${k}:`);
    for (const e of examples[k]) console.log(`    ${e.file}:${e.line}`);
  }
})();