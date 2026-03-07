import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import {
  ACCEPTANCE_RELATIVE_PATH,
  MATRIX_RELATIVE_PATH,
  REGISTRY_RELATIVE_PATH,
  buildGovernedTopicRegistry,
  formatAcceptanceReport,
  formatAuthorityMatrix,
  formatHistoricalStatusNote,
  formatRepoMetadataBlock,
  metadataFieldsForTopic,
  stripStatusNote,
} from './governance_registry.mjs';

const rootDir = path.resolve(process.cwd());

function ensureTrailingNewline(content) {
  return content.endsWith('\n') ? content : `${content}\n`;
}

function syncRepoMetadata(content, topic, relPath) {
  const metadataBlock = `${formatRepoMetadataBlock(topic)}\n\n`;
  let next = content;

  if (next.startsWith('<!-- docs:meta\n')) {
    next = next.replace(/^<!-- docs:meta\n[\s\S]*?\n-->\n*/m, metadataBlock);
  } else {
    next = `${metadataBlock}${next}`;
  }

  next = stripStatusNote(next).replace(/^\n+/, '');

  const statusNote = formatHistoricalStatusNote(topic, relPath);
  if (statusNote) {
    next = next.replace(/^<!-- docs:meta\n[\s\S]*?\n-->\n*/m, (match) => `${match}${statusNote}\n\n`);
  }

  return ensureTrailingNewline(next);
}

function formatFrontmatterValue(value) {
  return `"${String(value).replace(/"/g, '\\"')}"`;
}

function syncFrontmatter(content, topic) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) {
    throw new Error('Expected frontmatter block at file start.');
  }

  const fields = metadataFieldsForTopic(topic);
  const body = match[1];
  let lines = body.split('\n');

  for (const [key, value] of Object.entries(fields)) {
    const formatted = `${key}: ${formatFrontmatterValue(value)}`;
    const index = lines.findIndex((line) => line.trimStart().startsWith(`${key}:`));
    if (index >= 0) {
      lines[index] = formatted;
    } else {
      lines.push(formatted);
    }
  }

  while (lines.length > 0 && lines[lines.length - 1].trim() === '') {
    lines.pop();
  }

  const frontmatter = `---\n${lines.join('\n')}\n---\n`;
  return ensureTrailingNewline(`${frontmatter}${content.slice(match[0].length)}`);
}

async function main() {
  const registry = await buildGovernedTopicRegistry(rootDir);
  const registryAbsPath = path.join(rootDir, REGISTRY_RELATIVE_PATH);
  const matrixAbsPath = path.join(rootDir, MATRIX_RELATIVE_PATH);
  const acceptanceAbsPath = path.join(rootDir, ACCEPTANCE_RELATIVE_PATH);

  await mkdir(path.dirname(registryAbsPath), { recursive: true });
  await mkdir(path.dirname(matrixAbsPath), { recursive: true });
  await mkdir(path.dirname(acceptanceAbsPath), { recursive: true });

  await writeFile(registryAbsPath, `${JSON.stringify(registry, null, 2)}\n`, 'utf8');
  await writeFile(matrixAbsPath, ensureTrailingNewline(formatAuthorityMatrix(registry)), 'utf8');
  await writeFile(acceptanceAbsPath, ensureTrailingNewline(formatAcceptanceReport(registry)), 'utf8');

  const repoTopics = registry.topics.filter((topic) => topic.repo_doc_path);
  const websiteTopics = registry.topics.filter((topic) => Object.keys(topic.website_paths ?? {}).length > 0);

  for (const topic of repoTopics) {
    const absPath = path.join(rootDir, topic.repo_doc_path);
    const current = await readFile(absPath, 'utf8');
    const next = syncRepoMetadata(current, topic, topic.repo_doc_path);
    if (next !== current) {
      await writeFile(absPath, next, 'utf8');
    }
  }

  for (const topic of websiteTopics) {
    for (const relPath of Object.values(topic.website_paths)) {
      const absPath = path.join(rootDir, relPath);
      const current = await readFile(absPath, 'utf8');
      const next = syncFrontmatter(current, topic);
      if (next !== current) {
        await writeFile(absPath, next, 'utf8');
      }
    }
  }

  console.log(`Synced docs governance metadata for ${repoTopics.length} repo docs and ${websiteTopics.length} website topics.`);
}

await main();
