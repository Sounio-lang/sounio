import { readFile, stat } from 'node:fs/promises';
import { writeSync } from 'node:fs';
import path from 'node:path';
import {
  ACCEPTANCE_RELATIVE_PATH,
  LOCALES,
  REGISTRY_RELATIVE_PATH,
  buildGovernedTopicRegistry,
  formatAcceptanceReport,
  metadataFieldsForTopic,
  parseFrontmatter,
  parseRepoMetadata,
  readRegistryFile,
} from './governance_registry.mjs';

const rootDir = path.resolve(process.cwd());
const markdownLinkPattern = /\[[^\]]+\]\(([^)]+)\)/g;

function compareRegistry(expected, actual) {
  return JSON.stringify(expected) === JSON.stringify(actual);
}

function fail(errors) {
  if (errors.length === 0) {
    console.log('Docs registry check passed.');
    return;
  }

  writeSync(2, `Docs registry check failed:\n${errors.map((error) => `- ${error}`).join('\n')}\n`);
  process.exitCode = 1;
}

function normalizeLinkTarget(target) {
  return target.split('#')[0].split('?')[0];
}

function shouldCheckLink(target) {
  if (!target) {
    return false;
  }
  return !/^(https?:|mailto:|data:|tel:|#)/.test(target);
}

function shouldValidateRepoLinks(topic) {
  const relPath = topic.repo_doc_path ?? '';
  if (topic.authority === 'historical' || topic.authority === 'archived') {
    return false;
  }

  // Keep link enforcement focused on front-door and current-canon surfaces.
  // Historical/deep implementation docs stay governed by metadata and status
  // notes without forcing a whole-repo legacy link cleanup in the same gate.
  return (
    relPath === 'README.md' ||
    relPath === 'docs/README.md' ||
    relPath === 'docs/codebase_overview.md' ||
    relPath === 'paper/README.md' ||
    relPath === 'spec/LANGUAGE_SPECIFICATION.md' ||
    relPath === 'examples/README.md' ||
    relPath.startsWith('docs/guide/') ||
    relPath.startsWith('docs/contributor-guide/') ||
    relPath.startsWith('docs/reference/') ||
    relPath.startsWith('docs/stdlib/') ||
    relPath.startsWith('docs/features/') ||
    relPath === 'docs/research/vancomycin-uncertainty.md'
  );
}

async function validateRepoLinks(topic, errors) {
  if (!topic.repo_doc_path || !shouldValidateRepoLinks(topic)) {
    return;
  }

  const absPath = path.join(rootDir, topic.repo_doc_path);
  const content = await readFile(absPath, 'utf8');
  markdownLinkPattern.lastIndex = 0;
  let match;
  while ((match = markdownLinkPattern.exec(content)) !== null) {
    const rawTarget = match[1].trim();
    if (!shouldCheckLink(rawTarget)) {
      continue;
    }

    const target = normalizeLinkTarget(rawTarget);
    if (!target) {
      continue;
    }

    const resolved = path.resolve(path.dirname(absPath), target);
    try {
      const resolvedStat = await stat(resolved);
      if (resolvedStat.isDirectory() || resolvedStat.isFile()) {
        continue;
      }
    } catch {
      // fall through
    }

    try {
      await readFile(resolved);
      continue;
    } catch {
      // fall through
    }

    try {
      const maybeIndex = path.join(resolved, 'index.md');
      await readFile(maybeIndex);
      continue;
    } catch {
      // fall through
    }

    try {
      const maybeReadme = path.join(resolved, 'README.md');
      await readFile(maybeReadme);
      continue;
    } catch {
      // fall through
    }

    errors.push(`Broken front-door/current-canon relative link in ${topic.repo_doc_path}: ${rawTarget}`);
  }
}

function metadataMismatch(expectedFields, actualFields, context, errors) {
  for (const [key, expected] of Object.entries(expectedFields)) {
    if ((actualFields?.[key] ?? '') !== expected) {
      errors.push(`${context} metadata mismatch for ${key}: expected "${expected}"`);
    }
  }
}

async function main() {
  const errors = [];
  const expectedRegistry = await buildGovernedTopicRegistry(rootDir);
  const actualRegistry = await readRegistryFile(rootDir);

  if (!compareRegistry(expectedRegistry, actualRegistry)) {
    errors.push(`Checked-in ${REGISTRY_RELATIVE_PATH} is stale. Re-run node scripts/docs/sync_governance_metadata.mjs`);
  }

  try {
    const expectedAcceptance = `${formatAcceptanceReport(expectedRegistry).trimEnd()}\n`;
    const actualAcceptance = await readFile(path.join(rootDir, ACCEPTANCE_RELATIVE_PATH), 'utf8');
    if (actualAcceptance !== expectedAcceptance) {
      errors.push(`Checked-in ${ACCEPTANCE_RELATIVE_PATH} is stale. Re-run node scripts/docs/sync_governance_metadata.mjs`);
    }
  } catch {
    errors.push(`Missing acceptance report: ${ACCEPTANCE_RELATIVE_PATH}`);
  }

  for (const topic of actualRegistry.topics) {
    if (topic.repo_doc_path) {
      const absPath = path.join(rootDir, topic.repo_doc_path);
      try {
        const content = await readFile(absPath, 'utf8');
        const actualMeta = parseRepoMetadata(content);
        metadataMismatch(metadataFieldsForTopic(topic), actualMeta, topic.repo_doc_path, errors);
        if ((topic.authority === 'historical' || topic.authority === 'archived') && !content.includes('Docs status:')) {
          errors.push(`${topic.repo_doc_path} is missing a visible ${topic.authority} status note`);
        }
      } catch {
        errors.push(`Missing repo doc path from registry: ${topic.repo_doc_path}`);
      }
    }

    for (const [locale, relPath] of Object.entries(topic.website_paths ?? {})) {
      try {
        const content = await readFile(path.join(rootDir, relPath), 'utf8');
        const actualMeta = parseFrontmatter(content);
        metadataMismatch(metadataFieldsForTopic(topic), actualMeta, relPath, errors);
        if (topic.collection === 'docs' && topic.locale_status?.[locale] !== 'present') {
          errors.push(`${relPath} exists but registry locale status for ${locale} is ${topic.locale_status?.[locale]}`);
        }
      } catch {
        errors.push(`Missing website content path from registry: ${relPath}`);
      }
    }

    for (const locale of LOCALES) {
      if (topic.collection === 'docs' && topic.website_slug && locale !== 'en' && topic.locale_status?.[locale] !== 'present') {
        errors.push(`Docs topic ${topic.topic_id} is missing localized coverage for ${locale}`);
      }
    }

    for (const artifact of topic.related_artifacts ?? []) {
      const absArtifact = path.join(rootDir, artifact);
      try {
        await stat(absArtifact);
      } catch {
        errors.push(`Missing related artifact: ${artifact}`);
      }
    }

    await validateRepoLinks(topic, errors);
  }

  fail(errors);
}

await main();
