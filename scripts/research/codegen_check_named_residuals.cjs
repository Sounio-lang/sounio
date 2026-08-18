#!/usr/bin/env node
// SPDX-License-Identifier: MIT
//
// codegen_check_named_residuals.cjs — re-classify the 25 residual homonyms
// (24 IDENTICAL + 1 SUBSTANTIVE_DIVERGENT) of the 2026-08-17 census using
// the refined four-category scheme from codegen_byte_diff_census.cjs.
//
// After the COMMENT_ONLY_DIVERGENT category was introduced, the previously
// lone SUBSTANTIVE_DIVERGENT residual (`name_is_get_arg_count`) reclassifies
// as COMMENT_ONLY_DIVERGENT — its byte cascades are byte-equal between the
// two files; the entire 5-line size delta is in inline comments (sync
// provenance + char-count labels). The other 24 stay IDENTICAL.
//
// Usage:
//   node codegen_check_named_residuals.cjs <codegen.sio> <codegen_x86_linux.sio>

const fs = require('node:fs/promises');

// The 25 residual homonyms from the 2026-08-17 census.
//   24 IDENTICAL (pure duplication debt)
//    1 COMMENT_ONLY_DIVERGENT after the 2026-08-18 sync — was
//      SUBSTANTIVE_DIVERGENT under the three-category byte census, but the
//      diff is entirely comments and the logic is byte-equal.
const RESIDUALS = [
  'ARCH_RISCV64', 'ARCH_UNKNOWN', 'ARCH_X86_64',
  'ERR_BACKEND_NOT_IMPLEMENTED', 'ERR_INVALID_MATRIX', 'ERR_INVALID_TARGET',
  'ERR_OK', 'ERR_TARGET_FLAGS_REQUIRE_NATIVE', 'ERR_TRACE_WRITE_FAILED',
  'FORMAT_ELF64', 'FORMAT_MACHO64', 'FORMAT_NONE', 'FORMAT_PE64',
  'MACOS_SYS_OPEN',
  'MATRIX_AME', 'MATRIX_APPLE_AMX', 'MATRIX_AUTO', 'MATRIX_IME',
  'MATRIX_INTEL_AMX', 'MATRIX_OFF', 'MATRIX_VME',
  'OS_LINUX', 'OS_UNKNOWN', 'OS_WINDOWS',
  'name_is_get_arg_count',  // COMMENT_ONLY_DIVERGENT after the 2026-08-18 sync
];

function extractBody(text, name) {
  const lines = text.split('\n');
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z_0-9]*)\s*\(/);
    if (m && m[1] === name) {
      let j = i, depth = 0, started = false;
      while (j < lines.length) {
        for (const ch of lines[j]) {
          if (ch === '{') { depth++; started = true; }
          else if (ch === '}') { depth--; }
        }
        if (started && depth === 0) break;
        j++;
      }
      return { line: i+1, body: lines.slice(i, j+1).join('\n') };
    }
  }
  return null;
}

function stripPub(b) { return b.replace(/^pub\s+/m, ''); }

// Strip comment-only lines and normalise whitespace. Two bodies that differ
// ONLY in inline comments (sync-provenance, char-count markers like
// `// "arg_count" = 9 chars`) classify as COMMENT_ONLY_DIVERGENT rather than
// SUBSTANTIVE_DIVERGENT — the logic is byte-equal, only the documentation
// around it differs.
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

  let identical = 0, pubSwap = 0, commentOnly = 0, divergent = 0, missing = [];
  const results = [];
  for (const name of RESIDUALS) {
    const a = extractBody(codegenText, name);
    const b = extractBody(x86Text, name);
    if (!a || !b) {
      missing.push({ name, a: !!a, b: !!b });
      continue;
    }
    if (a.body === b.body) {
      identical++;
      results.push({ name, class: 'IDENTICAL', la: a.line, lb: b.line });
    } else if (stripPub(a.body) === stripPub(b.body)) {
      pubSwap++;
      results.push({ name, class: 'PUB_SWAP_ONLY', la: a.line, lb: b.line });
    } else if (stripComments(stripPub(a.body)) === stripComments(stripPub(b.body))) {
      commentOnly++;
      results.push({ name, class: 'COMMENT_ONLY_DIVERGENT', la: a.line, lb: b.line,
                     sa: a.body.split('\n').length, sb: b.body.split('\n').length });
    } else {
      divergent++;
      results.push({ name, class: 'SUBSTANTIVE_DIVERGENT', la: a.line, lb: b.line,
                     sa: a.body.split('\n').length, sb: b.body.split('\n').length });
    }
  }

  console.log(`Of the 25 residual names:`);
  console.log(`  IDENTICAL: ${identical}`);
  console.log(`  PUB_SWAP_ONLY: ${pubSwap}`);
  console.log(`  COMMENT_ONLY_DIVERGENT: ${commentOnly}`);
  console.log(`  SUBSTANTIVE_DIVERGENT: ${divergent}`);
  console.log(`  MISSING: ${missing.length}`);
  if (missing.length) console.log(`    ${JSON.stringify(missing)}`);
  console.log('');
  for (const r of results.sort((a,b) => a.name.localeCompare(b.name))) {
    if (r.class === 'SUBSTANTIVE_DIVERGENT' || r.class === 'COMMENT_ONLY_DIVERGENT') {
      console.log(`  ${r.name}: ${r.class}  codegen L${r.la} (${r.sa} lines) / x86 L${r.lb} (${r.sb} lines)`);
    } else {
      console.log(`  ${r.name}: ${r.class}  codegen L${r.la} / x86 L${r.lb}`);
    }
  }
})();
