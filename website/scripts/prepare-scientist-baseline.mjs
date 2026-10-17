import { execFileSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const catalog = JSON.parse(await readFile(path.join(repository, 'website/src/data/scientistWorkflows.json'), 'utf8'));
const revision = catalog.source_revision;
if (!/^[a-f0-9]{40}$/.test(revision ?? '')) throw new Error('Invalid scientist source revision');
const options = { cwd: repository, stdio: 'pipe', env: { ...process.env, GIT_NO_LAZY_FETCH: '1' } };
try { execFileSync('git', ['cat-file', '-e', `${revision}^{commit}`], options); }
catch {
  console.log(`Materializing pinned scientist source commit ${revision} from origin; no checkout or compiler execution.`);
  execFileSync('git', ['fetch', '--no-tags', '--depth=1', 'origin', revision], { ...options, stdio: 'inherit' });
  execFileSync('git', ['cat-file', '-e', `${revision}^{commit}`], options);
}
