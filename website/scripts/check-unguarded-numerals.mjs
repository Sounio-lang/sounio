/**
 * Closed green-counts must not interpolate outside the throw functions.
 * A leak here means the kernel is a list of sites, not an invariant.
 *
 * Closed: U1 251/251, U2 13/13, U3 7/7.
 * Archive and docs locales are not live faces.
 */
import { readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd(), 'src');

const SKIP_DIR = new Set(['_archive', 'content', 'node_modules']);

const CLOSED_PATTERNS = [
  { id: 'U2-13/13', re: /\b13\s*\/\s*13\b/ },
  { id: 'U1-251/251', re: /\b251\s*\/\s*251\b/ },
  { id: 'U3-7/7', re: /\b7\s*\/\s*7\b/ },
  { id: 'raw-hyperPassCount', re: /hyperPassCount/ },
  { id: 'raw-hyperTotalCount', re: /hyperTotalCount/ },
  { id: 'raw-publicPassCount', re: /publicPassCount/ },
  { id: 'raw-hyperLanes', re: /hyperLanes/ },
];

const KERNEL_DIR = path.join(root, 'lib');

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
  if (rel.startsWith('..')) return false;
  return /Honesty|measurementClaim|proofData|artifactStatus/.test(rel);
}

async function run() {
  const files = [
    ...(await walk(path.join(root, 'pages'))),
    ...(await walk(path.join(root, 'components'))),
  ];

  const leaks = [];
  for (const file of files) {
    if (isKernelFile(file)) continue;
    const text = await readFile(file, 'utf8');
    for (const { id, re } of CLOSED_PATTERNS) {
      if (re.test(text)) {
        leaks.push(`${id} ${path.relative(process.cwd(), file)}`);
      }
    }
  }

  if (leaks.length > 0) {
    console.error('Unguarded closed numerals:');
    for (const row of leaks) console.error(`- ${row}`);
    process.exit(1);
  }

  console.log('OK: closed numerals (U1/U2/U3) do not interpolate outside the kernel.');
}

await run();
