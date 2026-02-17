import { readFile } from 'node:fs/promises';
import path from 'node:path';

const htmlPath = path.resolve(process.cwd(), 'dist/index.html');

try {
  const html = await readFile(htmlPath, 'utf8');
  if (!html.includes('data-pagefind-body')) {
    console.error('Missing data-pagefind-body marker in dist/index.html');
    process.exit(1);
  }
  console.log('OK: pagefind body marker present.');
} catch (err) {
  console.error(`Unable to validate pagefind marker at ${htmlPath}:`, err.message);
  process.exit(1);
}
