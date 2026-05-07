#!/usr/bin/env node
/**
 * Fail if both src/pages/<name>.astro and src/pages/<name>/index.astro exist.
 * Astro warns that this maps two files to the same URL (build-time collision).
 */
import { existsSync, readdirSync } from 'node:fs';
import path from 'node:path';

const pagesDir = path.resolve(process.cwd(), 'src/pages');

function scan() {
  const errors = [];
  if (!existsSync(pagesDir)) {
    console.error('Missing src/pages directory');
    process.exit(1);
  }

  const entries = readdirSync(pagesDir, { withFileTypes: true });

  for (const ent of entries) {
    if (!ent.isDirectory()) continue;
    if (ent.name.startsWith('_')) continue;

    const dirPath = path.join(pagesDir, ent.name);
    const indexFile = path.join(dirPath, 'index.astro');
    const siblingFile = path.join(pagesDir, `${ent.name}.astro`);

    if (existsSync(indexFile) && existsSync(siblingFile)) {
      errors.push(
        `Route collision: both exist — src/pages/${ent.name}.astro and src/pages/${ent.name}/index.astro → same URL "/${ent.name}"`,
      );
    }
  }

  return errors;
}

const errors = scan();
if (errors.length > 0) {
  console.error('Duplicate page route validation failed:');
  for (const e of errors) console.error(`- ${e}`);
  process.exit(1);
}

console.log('OK: no duplicate *.astro vs */index.astro routes under src/pages.');
