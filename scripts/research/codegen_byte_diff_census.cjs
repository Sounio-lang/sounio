#!/usr/bin/env node
// SPDX-License-Identifier: MIT
//
// codegen_byte_diff_census.cjs — byte-diff classification of every homonym
// pair across self-hosted/native/codegen.sio and codegen_x86_linux.sio.
//
// Categories (four, in order of decreasing leniency):
//
//   IDENTICAL              bodies byte-equal
//   PUB_SWAP_ONLY          bodies byte-equal after stripping `pub`
//   COMMENT_ONLY_DIVERGENT bodies byte-equal after stripping `pub` AND
//                          comment-only lines — only human-written
//                          documentation (sync-provenance, char-count
//                          markers, etc.) differs between the two copies.
//                          The LOGIC is byte-equal; the census was
//                          mislabeling these as SUBSTANTIVE_DIVERGENT.
//   SUBSTANTIVE_DIVERGENT  real logic differences — at least one byte-cmp,
//                          byte-cascade, control-flow branch, etc. is
//                          missing from one copy or present in a different
//                          form. This is the only category that can
//                          indicate a runtime divergence between the two
//                          backends.
//
// Usage:
//   node codegen_byte_diff_census.cjs <codegen.sio> <codegen_x86_linux.sio>
//
// See docs/audit/CODEGEN_DEAD_BODY_DELETION_2026-08-18.md for the worked
// example that motivated adding the COMMENT_ONLY_DIVERGENT category
// (name_is_get_arg_count after the 2026-08-18 sync from codegen_x86_linux.sio).

const fs = require('node:fs/promises');

function extractSymbols(text) {
  const lines = text.split('\n');
  const symbols = new Map();
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(/);
    if (!m) continue;
    const name = m[1];
    let j = i, depth = 0, started = false;
    while (j < lines.length) {
      for (const ch of lines[j]) {
        if (ch === '{') { depth++; started = true; }
        else if (ch === '}') { depth--; }
      }
      if (started && depth === 0) break;
      j++;
    }
    const body = lines.slice(i, j+1).join('\n');
    if (!symbols.has(name)) symbols.set(name, []);
    symbols.get(name).push({ line: i+1, body });
  }
  return symbols;
}

function stripPub(b) {
  return b.replace(/^pub\s+/m, '');
}

// Strip comment-only lines (lines whose first non-blank token is `//`) and
// normalise whitespace runs so two bodies that differ ONLY in comments
// (sync provenance, char-count markers, etc.) classify as COMMENT_ONLY_DIVERGENT
// rather than SUBSTANTIVE_DIVERGENT.
//
// Sounio doesn't currently use /* ... */ block comments in this corpus, but
// if that ever changes the same trick applies: drop the comment lines before
// comparing. The point is "logic bytes equal" — what the code DOES, not what
// the human wrote around it.
function stripComments(b) {
  return b
    .split('\n')
    .filter(l => !l.trim().startsWith('//'))
    .join('\n')
    .replace(/[ \t]+/g, ' ');
}

(async () => {
  const codegenText = await fs.readFile(process.argv[2], 'utf8');
  const x86Text = await fs.readFile(process.argv[3], 'utf8');

  const codegen = extractSymbols(codegenText);
  const x86 = extractSymbols(x86Text);

  const common = [];
  for (const [name] of codegen) {
    if (x86.has(name)) common.push(name);
  }

  console.log(`codegen.sio symbols: ${codegen.size}`);
  console.log(`codegen_x86_linux.sio symbols: ${x86.size}`);
  console.log(`Common homonym symbols: ${common.length}`);
  console.log('');

  let identical = 0, pubSwap = 0, commentOnly = 0, divergent = 0;
  const results = [];
  for (const name of common) {
    const a = codegen.get(name)[0];
    const b = x86.get(name)[0];
    if (a.body === b.body) {
      identical++;
      results.push({ name, class: 'IDENTICAL', la: a.line, lb: b.line });
    } else if (stripPub(a.body) === stripPub(b.body)) {
      pubSwap++;
      results.push({ name, class: 'PUB_SWAP_ONLY', la: a.line, lb: b.line });
    } else if (stripComments(stripPub(a.body)) === stripComments(stripPub(b.body))) {
      commentOnly++;
      const al = a.body.split('\n').length;
      const bl = b.body.split('\n').length;
      results.push({ name, class: 'COMMENT_ONLY_DIVERGENT', la: a.line, lb: b.line, sa: al, sb: bl });
    } else {
      divergent++;
      const al = a.body.split('\n').length;
      const bl = b.body.split('\n').length;
      results.push({ name, class: 'SUBSTANTIVE_DIVERGENT', la: a.line, lb: b.line, sa: al, sb: bl });
    }
  }

  console.log(`IDENTICAL: ${identical}`);
  console.log(`PUB_SWAP_ONLY: ${pubSwap}`);
  console.log(`COMMENT_ONLY_DIVERGENT: ${commentOnly}`);
  console.log(`SUBSTANTIVE_DIVERGENT: ${divergent}`);
  console.log('');

  console.log('--- IDENTICAL ---');
  for (const r of results.filter(r => r.class === 'IDENTICAL').sort((a,b)=>a.name.localeCompare(b.name))) {
    console.log(`  ${r.name}: codegen L${r.la} / x86 L${r.lb}`);
  }
  console.log('');
  console.log('--- PUB_SWAP_ONLY ---');
  for (const r of results.filter(r => r.class === 'PUB_SWAP_ONLY').sort((a,b)=>a.name.localeCompare(b.name))) {
    console.log(`  ${r.name}: codegen L${r.la} / x86 L${r.lb}`);
  }
  console.log('');
  console.log('--- COMMENT_ONLY_DIVERGENT (logic byte-equal; only comments differ) ---');
  for (const r of results.filter(r => r.class === 'COMMENT_ONLY_DIVERGENT').sort((a,b)=>a.name.localeCompare(b.name))) {
    console.log(`  ${r.name}: codegen L${r.la} (${r.sa} lines) vs x86 L${r.lb} (${r.sb} lines)`);
  }
  console.log('');
  console.log('--- SUBSTANTIVE_DIVERGENT (real logic differences) ---');
  for (const r of results.filter(r => r.class === 'SUBSTANTIVE_DIVERGENT').sort((a,b)=>a.name.localeCompare(b.name))) {
    console.log(`  ${r.name}: codegen L${r.la} (${r.sa} lines) vs x86 L${r.lb} (${r.sb} lines)`);
  }
})();
