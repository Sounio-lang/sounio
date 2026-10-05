/**
 * License rule: classify who may assert, not which digits look like
 * a ratio. A live page may print a measurement well only through
 * claimFace / claimRefusal / suiteFaceParts, or through claimEscape.
 *
 * claimEscape is the visible door. The table prints on every run.
 * Growing the table is a review signal. A silent skip is a refuse.
 *
 * Positive control: scripts/fixtures/mutant-unlicensed-fact.astro
 * asserts a new fact (scienceLanes, 9/9) without license. The
 * denylist misses it — confirmed before this rule shipped. If the
 * mutant produces no finding, exit 1: POSITIVE CONTROL DEAD.
 *
 * DEBT: CLOSED_PATTERNS is the old denylist. It stays until the
 * license rule is the sole finding on one green Website CI run on
 * main AND the positive control still fires. Do not add rows. Two
 * mechanisms for the same thing is how neither is owned. Retire
 * this block in the first follow-up after that green — not in the
 * same change that invents the rule.
 */
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const websiteRoot = path.resolve(process.cwd());
const srcRoot = path.join(websiteRoot, 'src');
const mutantPath = path.join(websiteRoot, 'scripts/fixtures/mutant-unlicensed-fact.astro');

const SKIP_DIR = new Set(['_archive', 'content', 'node_modules']);

const ESCAPE_CLASSES = new Set(['inventory', 'literature', 'identity', 'threshold', 'version']);

/**
 * Fact wells. A new `{metrics.scienceLanes}` on a live page must
 * fail until it is licensed. Tailwind `w-1/2` is not a well.
 * Do not add CLOSED_PATTERNS rows for new greens — add a well or
 * a claimEscape.
 */
const WELLS = [
  { id: 'metrics.stdlibInventoryFiles', re: /metrics\.stdlibInventoryFiles/ },
  { id: 'metrics.scienceLanes', re: /metrics\.scienceLanes/ },
  { id: 'metrics.hyperLanes', re: /metrics\.hyperLanes/ },
  { id: 'metrics.fullTestSuite', re: /metrics\.fullTestSuite/ },
  { id: 'metrics.selfHostedSourceLines', re: /metrics\.selfHostedSourceLines/ },
  { id: 'metrics.selfHostedSourceFiles', re: /metrics\.selfHostedSourceFiles/ },
  { id: 'metrics.stdlibReliabilityGate.pass', re: /metrics\.stdlibReliabilityGate\.(pass|total|fail)/ },
  { id: 'proof.stdlib.activeModuleEntrypoints', re: /proof(?:Hub)?\.stdlib\.activeModuleEntrypoints/ },
  { id: 'proof.stdlib.hyperPassCount', re: /proof(?:Hub)?\.stdlib\.hyperPassCount/ },
  { id: 'proof.stdlib.hyperTotalCount', re: /proof(?:Hub)?\.stdlib\.hyperTotalCount/ },
  { id: 'proof.stdlib.scienceStatus', re: /proof(?:Hub)?\.stdlib\.scienceStatus/ },
  { id: 'proof.gpu.attestedTargetCount', re: /proof(?:Hub)?\.gpu\.attestedTargetCount/ },
  { id: 'proof.gpu.publicPassCount', re: /proof(?:Hub)?\.gpu\.publicPassCount/ },
  { id: 'proof.gpu.publicCheckCount', re: /proof(?:Hub)?\.gpu\.publicCheckCount/ },
  { id: 'proof.release.version', re: /proof(?:Hub)?\.release\.version/ },
  { id: 'fullSuite.pass', re: /fullSuite\.(pass|total)/ },
  { id: 'full.pass', re: /\bfull\.(pass|total)\b/ },
  { id: 'stdlib.pass', re: /\bstdlib\.(pass|total)\b/ },
  { id: 'versions.checkedArtifact', re: /versions\.checkedArtifact/ },
  { id: 'versions.readmeBadge', re: /versions\.readmeBadge/ },
];

const CLOSED_PATTERNS = [
  { id: 'U2-13/13', re: /\b13\s*\/\s*13\b/ },
  { id: 'U1-251/251', re: /\b251\s*\/\s*251\b/ },
  { id: 'U3-7/7', re: /\b7\s*\/\s*7\b/ },
  { id: 'raw-hyperPassCount', re: /hyperPassCount/ },
  { id: 'raw-hyperTotalCount', re: /hyperTotalCount/ },
  { id: 'raw-publicPassCount', re: /publicPassCount/ },
  { id: 'raw-hyperLanes', re: /hyperLanes/ },
];

const KERNEL_DIR = path.join(srcRoot, 'lib');

async function walk(dir, out = []) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (SKIP_DIR.has(entry.name)) continue;
      await walk(full, out);
      continue;
    }
    if (!/\.(astro|ts|tsx)$/.test(entry.name)) continue;
    out.push(full);
  }
  return out;
}

function isKernelFile(file) {
  const rel = path.relative(KERNEL_DIR, file);
  if (rel.startsWith('..') || rel.startsWith(`..${path.sep}`)) return false;
  return /Honesty|measurementClaim|proofData|artifactStatus/.test(rel);
}

function isClaimEscapesFile(file) {
  return path.basename(file) === 'claimEscapes.ts';
}

function findMatchingParen(src, openIdx) {
  let depth = 0;
  let quote = null;
  for (let i = openIdx; i < src.length; i += 1) {
    const c = src[i];
    if (quote) {
      if (c === '\\') {
        i += 1;
        continue;
      }
      if (c === quote) quote = null;
      continue;
    }
    if (c === '"' || c === "'" || c === '`') {
      quote = c;
      continue;
    }
    if (c === '(') depth += 1;
    else if (c === ')') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function eachClaimEscapeCall(src, onCall) {
  let i = 0;
  while (i < src.length) {
    const idx = src.indexOf('claimEscape(', i);
    if (idx === -1) break;
    const open = idx + 'claimEscape'.length;
    const close = findMatchingParen(src, open);
    if (close === -1) {
      onCall({ start: idx, end: src.length, body: src.slice(idx), incomplete: true });
      break;
    }
    onCall({ start: idx, end: close + 1, body: src.slice(idx, close + 1), incomplete: false });
    i = close + 1;
  }
}

function stripClassAttrs(src) {
  return src.replace(/\s(?:class|className)=(?:"[^"]*"|'[^']*'|\{[^}]*\})/g, '');
}

function stripClaimEscapeCalls(src) {
  const pieces = [];
  let last = 0;
  eachClaimEscapeCall(src, ({ start, end }) => {
    pieces.push(src.slice(last, start));
    pieces.push(' CLAIM_ESCAPE ');
    last = end;
  });
  pieces.push(src.slice(last));
  return pieces.join('');
}

function extractQuoted(body, key) {
  const re = new RegExp(`${key}:\\s*['"]([^'"]+)['"]`);
  const m = body.match(re);
  return m ? m[1] : null;
}

function parseEscapes(src, file) {
  const rows = [];
  const errors = [];
  const rel = path.relative(websiteRoot, file);
  eachClaimEscapeCall(src, ({ body, incomplete }) => {
    if (incomplete) {
      errors.push(`${rel}: claimEscape call is unclosed`);
      return;
    }
    const id = extractQuoted(body, 'id');
    const cls = extractQuoted(body, 'class');
    const reason = extractQuoted(body, 'reason');
    if (!id || !cls || !reason) {
      errors.push(`${rel}: claimEscape missing id, class, or reason`);
      return;
    }
    if (!ESCAPE_CLASSES.has(cls)) {
      errors.push(`${rel}: claimEscape ${id} has unknown class "${cls}"`);
      return;
    }
    if (reason.trim().length < 20) {
      errors.push(`${rel}: claimEscape ${id} reason is too short to audit`);
      return;
    }
    rows.push({ file: rel, id, class: cls, reason: reason.trim() });
  });
  return { rows, errors };
}

function scanWells(src, file) {
  const rel = path.relative(websiteRoot, file);
  const findings = [];
  for (const { id, re } of WELLS) {
    if (re.test(src)) findings.push(`${id} ${rel}`);
  }
  return findings;
}

function scanDenylist(src, file) {
  const rel = path.relative(websiteRoot, file);
  const findings = [];
  for (const { id, re } of CLOSED_PATTERNS) {
    if (re.test(src)) findings.push(`${id} ${rel}`);
  }
  return findings;
}

function printEscapeTable(rows) {
  const unique = new Map();
  for (const row of rows) {
    if (!unique.has(row.id)) unique.set(row.id, row);
  }
  console.log(`claimEscape table (${unique.size} ids, ${rows.length} annotated calls):`);
  const sorted = [...unique.values()].sort((a, b) => a.class.localeCompare(b.class) || a.id.localeCompare(b.id));
  for (const row of sorted) {
    console.log(`  ${row.class.padEnd(11)} ${row.id.padEnd(32)} ${row.reason}`);
  }
}

async function run() {
  const liveFiles = [
    ...(await walk(path.join(srcRoot, 'pages'))),
    ...(await walk(path.join(srcRoot, 'components'))),
  ];

  const escapeFiles = [path.join(srcRoot, 'lib', 'claimEscapes.ts'), ...liveFiles];

  const escapeRows = [];
  const escapeErrors = [];
  for (const file of escapeFiles) {
    let text;
    try {
      text = await readFile(file, 'utf8');
    } catch {
      if (isClaimEscapesFile(file)) {
        escapeErrors.push('claimEscapes.ts is missing — the escape table cannot be empty by deletion');
      }
      continue;
    }
    const parsed = parseEscapes(text, file);
    escapeRows.push(...parsed.rows);
    escapeErrors.push(...parsed.errors);
  }

  printEscapeTable(escapeRows);

  const unlicensed = [];
  const denylistLeaks = [];
  for (const file of liveFiles) {
    if (isKernelFile(file) && !isClaimEscapesFile(file)) continue;
    let text;
    try {
      text = await readFile(file, 'utf8');
    } catch {
      if (isClaimEscapesFile(file)) unlicensed.push('claimEscapes.ts missing');
      continue;
    }
    const stripped = stripClaimEscapeCalls(stripClassAttrs(text));
    unlicensed.push(...scanWells(stripped, file));
    if (!isClaimEscapesFile(file)) {
      denylistLeaks.push(...scanDenylist(text, file));
    }
  }

  const mutantText = await readFile(mutantPath, 'utf8');
  const mutantFindings = scanWells(stripClaimEscapeCalls(stripClassAttrs(mutantText)), mutantPath);

  let failed = false;

  if (escapeErrors.length > 0) {
    failed = true;
    console.error('claimEscape annotation errors:');
    for (const row of escapeErrors) console.error(`- ${row}`);
  }

  if (unlicensed.length > 0) {
    failed = true;
    console.error('Unlicensed measurement wells:');
    for (const row of unlicensed) console.error(`- ${row}`);
  } else {
    console.log('OK: live pages have no unlicensed measurement wells.');
  }

  if (mutantFindings.length === 0) {
    failed = true;
    console.error(
      'POSITIVE CONTROL DEAD: mutant-unlicensed-fact.astro produced no unlicensed well. The instrument no longer sees a new fact.',
    );
  } else {
    console.log(
      `OK: positive control — mutant produced ${mutantFindings.length} unlicensed well(s) as required.`,
    );
  }

  if (denylistLeaks.length > 0) {
    failed = true;
    console.error('Unguarded closed numerals (denylist debt):');
    for (const row of denylistLeaks) console.error(`- ${row}`);
  } else {
    console.log(
      'OK: denylist debt still silent on live pages. Retire CLOSED_PATTERNS after one green Website CI run on main where this license rule is the only finding and the positive control still fires.',
    );
  }

  if (failed) process.exit(1);
}

await run();
