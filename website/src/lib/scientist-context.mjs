import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync, realpathSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { validateScientistCatalog, scientistLanguages } from './scientist-prompt.mjs';
import { renderPack } from '../../../scripts/docs/sounio_task_context.mjs';

const repository = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const packs = new Map();

export function scientistContext(catalog, language) {
  if (!scientistLanguages.includes(language)) throw new Error('Unsupported scientist language');
  const taskMap = JSON.parse(readFileSync(path.join(repository, 'docs/llm-guide/task-map.json'), 'utf8'));
  const errors = validateScientistCatalog(catalog, taskMap.tasks.map(task => task.id));
  if (errors.length) throw new Error(errors.join('\n'));
  const contexts = {};
  for (const workflow of catalog.workflows) {
    const key = `${language}:${workflow.starter_task}`;
    if (!packs.has(key)) {
      const taskPack = JSON.parse(execFileSync(process.execPath, ['scripts/docs/sounio_task_context.mjs', 'show', workflow.starter_task, '--lang', language, '--json'], { cwd: repository, encoding: 'utf8', maxBuffer: 1024 * 1024, env: { ...process.env, GIT_NO_LAZY_FETCH: '1' } }));
      packs.set(key, { task_pack: taskPack, language_pack: renderPack(taskPack) });
    }
    const sources = workflow.sources.map(source => {
      const absolute = realpathSync(path.join(repository, source));
      const relative = path.relative(repository, absolute);
      if (relative.startsWith('..') || path.isAbsolute(relative)) throw new Error('Source escapes repository');
      const bytes = readFileSync(absolute);
      const baseline = execFileSync('git', ['show', `${catalog.source_revision}:${source}`], { cwd: repository, maxBuffer: 1024 * 1024, env: { ...process.env, GIT_NO_LAZY_FETCH: '1' } });
      if (!bytes.equals(baseline)) throw new Error(`Domain source drift: ${source}`);
      return { path: source, sha256: createHash('sha256').update(bytes).digest('hex'), revision: catalog.source_revision, url: `https://github.com/Sounio-lang/sounio/blob/${catalog.source_revision}/${source}` };
    });
    contexts[workflow.id] = { sources, ...packs.get(key) };
  }
  return contexts;
}
