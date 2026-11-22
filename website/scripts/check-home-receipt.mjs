/**
 * Re-run the two programs the homepage shows, against the committed
 * ./bin/souc. Expected lines live in src/fixtures/home/receipts.json.
 * The Madaros banner is not part of the receipt.
 */
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const websiteRoot = process.cwd();
const repoRoot = path.resolve(websiteRoot, '..');
const souc = path.join(repoRoot, 'bin/souc');
const fixtureDir = path.join(websiteRoot, 'src/fixtures/home');
const receipts = JSON.parse(await readFile(path.join(fixtureDir, 'receipts.json'), 'utf8'));

function fail(message) {
  console.error(`check-home-receipt: ${message}`);
  process.exitCode = 1;
}

for (const name of ['speed_and_knowledge', 'metres_plus_seconds']) {
  const spec = receipts[name];
  const program = path.join(fixtureDir, `${name}.sio`);
  const hero = await readFile(path.join(websiteRoot, 'src/components/home/Hero.astro'), 'utf8');
  if (!hero.includes(`fixtures/home/${name}.sio`)) fail(`Hero.astro does not read ${name}.sio`);
  for (const needle of spec.expect) {
    if (!hero.includes(needle)) fail(`Hero.astro does not show ${JSON.stringify(needle)}`);
  }
  const result = spawnSync(souc, [spec.mode === 'run' ? 'run' : 'check', program], {
    cwd: repoRoot,
    env: { ...process.env, SOUNIO_STDLIB_PATH: path.join(repoRoot, 'stdlib') },
    encoding: 'utf8',
  });
  const combined = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
  const status = result.status ?? 1;
  if (spec.mode === 'run' && status !== 0) fail(`${name} exited ${status}`);
  if (spec.mode !== 'run' && status === 0) fail(`${name} was refused by the receipt and exited 0`);
  for (const needle of spec.expect) {
    if (!combined.includes(needle)) fail(`${name} did not print ${JSON.stringify(needle)}`);
  }
}

if (!process.exitCode) console.log('check-home-receipt: homepage programs match the committed compiler');
