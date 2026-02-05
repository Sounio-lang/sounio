import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const ROOT = path.resolve(process.cwd());
const I18N_DIR = path.join(ROOT, 'src', 'i18n');

function sortedKeys(obj) {
  return Object.keys(obj).sort((a, b) => a.localeCompare(b));
}

function setDiff(a, b) {
  const bSet = new Set(b);
  return a.filter((k) => !bSet.has(k));
}

async function main() {
  const entries = (await fs.readdir(I18N_DIR)).filter((f) => f.endsWith('.json')).sort();
  if (!entries.includes('en.json')) {
    throw new Error(`Missing en.json in ${I18N_DIR}`);
  }

  const enPath = path.join(I18N_DIR, 'en.json');
  const en = JSON.parse(await fs.readFile(enPath, 'utf8'));
  const enKeys = sortedKeys(en);

  let ok = true;

  for (const file of entries) {
    if (file === 'en.json') continue;
    const localePath = path.join(I18N_DIR, file);
    const data = JSON.parse(await fs.readFile(localePath, 'utf8'));

    const keys = sortedKeys(data);
    const missing = setDiff(enKeys, keys);
    const extra = setDiff(keys, enKeys);

    if (missing.length === 0 && extra.length === 0) continue;

    ok = false;
    const locale = file.replace(/\.json$/, '');
    console.error(`\n${locale}: key mismatch`);
    if (missing.length) console.error(`  Missing (${missing.length}): ${missing.join(', ')}`);
    if (extra.length) console.error(`  Extra   (${extra.length}): ${extra.join(', ')}`);
  }

  if (!ok) {
    console.error('\nFAIL: i18n key parity check failed.');
    process.exit(1);
  }

  console.log(`OK: ${entries.length} locales match en.json keys.`);
}

await main();

