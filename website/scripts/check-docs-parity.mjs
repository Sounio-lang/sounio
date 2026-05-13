import path from 'node:path';
import { writeSync } from 'node:fs';
import { buildGovernedTopicRegistry, readRegistryFile } from '../../scripts/docs/governance_registry.mjs';

const repoRoot = path.resolve(process.cwd(), '..');

function fail(errors) {
  if (errors.length === 0) {
    console.log('OK: docs parity validated.');
    return;
  }

  writeSync(2, `Docs parity validation failed:\n${errors.map((error) => `- ${error}`).join('\n')}\n`);
  process.exitCode = 1;
}

function compareByTopicId(topics) {
  return new Map(topics.map((topic) => [topic.topic_id, topic]));
}

const expectedRegistry = await buildGovernedTopicRegistry(repoRoot);
const actualRegistry = await readRegistryFile(repoRoot);
const errors = [];

const expectedDocsTopics = expectedRegistry.topics.filter((topic) => topic.collection === 'docs');
const actualDocsTopics = actualRegistry.topics.filter((topic) => topic.collection === 'docs');
const expectedById = compareByTopicId(expectedDocsTopics);
const actualById = compareByTopicId(actualDocsTopics);

for (const topic of expectedDocsTopics) {
  const actual = actualById.get(topic.topic_id);
  if (!actual) {
    errors.push(`Missing docs topic in checked-in registry: ${topic.topic_id}`);
    continue;
  }

  if (actual.website_slug !== topic.website_slug) {
    errors.push(`Website slug drift for ${topic.topic_id}: expected ${topic.website_slug}, found ${actual.website_slug}`);
  }

  for (const [locale, status] of Object.entries(topic.locale_status)) {
    if (actual.locale_status?.[locale] !== status) {
      errors.push(`Locale status drift for ${topic.topic_id} ${locale}: expected ${status}, found ${actual.locale_status?.[locale]}`);
    }
  }
}

for (const topic of actualDocsTopics) {
  if (!expectedById.has(topic.topic_id)) {
    errors.push(`Orphan docs topic in checked-in registry: ${topic.topic_id}`);
  }
}

fail(errors);
