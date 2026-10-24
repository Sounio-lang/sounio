import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { scientistContext } from '../src/lib/scientist-context.mjs';
import { scientistLanguages } from '../src/lib/scientist-prompt.mjs';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const dependencies = ['website/src/data/scientistWorkflows.json', 'website/src/lib/scientist-context.mjs', 'scripts/docs/sounio_task_context.mjs', 'docs/llm-guide/task-map.json', 'docs/llm-guide/example-manifest.json', 'docs/llm-guide/example-results.json'];
const catalog = JSON.parse(await readFile(path.join(repository, dependencies[0]), 'utf8'));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const inputs = Object.fromEntries(await Promise.all(dependencies.map(async source => [source, hash(await readFile(path.join(repository, source)))])));
const contexts = Object.fromEntries(scientistLanguages.map(language => [language, scientistContext(catalog, language)]));
const artifact = { schema: 'sounio.scientist-contexts.v1', source_revision: catalog.source_revision, catalog_sha256: hash(JSON.stringify(catalog)), compiler_invoked: false, receipt_authenticated: false, inputs, contexts };
await writeFile(path.join(repository, 'website/src/data/scientistContexts.json'), `${JSON.stringify(artifact, null, 2)}\n`);
console.log(`Prepared ${catalog.workflows.length * scientistLanguages.length} source-bound instruction contexts; no compiler or LLM invoked.`);
