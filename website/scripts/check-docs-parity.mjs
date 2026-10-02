import path from 'node:path';
import { writeSync } from 'node:fs';
import { access } from 'node:fs/promises';
import { buildGovernedTopicRegistry, LOCALES } from '../../scripts/docs/governance_registry.mjs';

const repoRoot = path.resolve(process.cwd(), '..');

function fail(errors) {
  if (errors.length === 0) {
    console.log('OK: docs parity validated.');
    return;
  }

  writeSync(2, `Docs parity validation failed:\n${errors.map((error) => `- ${error}`).join('\n')}\n`);
  process.exitCode = 1;
}

async function pathExists(absPath) {
  try {
    await access(absPath);
    return true;
  } catch {
    return false;
  }
}

// Generate-then-validate: do not compare against the checked-in registry.
// That serialisation races every concurrent PR that adds a website doc.
// The generated topic list is the scan of website/src/content/docs; we
// still refuse missing locale pages and broken website_paths.
const registry = await buildGovernedTopicRegistry(repoRoot);
const errors = [];

for (const topic of registry.topics.filter((entry) => entry.collection === 'docs')) {
  for (const [locale, relPath] of Object.entries(topic.website_paths ?? {})) {
    if (!(await pathExists(path.join(repoRoot, relPath)))) {
      errors.push(`Missing website content path from registry: ${relPath}`);
    }
    if (topic.locale_status?.[locale] !== 'present') {
      errors.push(`${relPath} exists but registry locale status for ${locale} is ${topic.locale_status?.[locale]}`);
    }
  }

  for (const locale of LOCALES) {
    if (locale !== 'en' && topic.locale_status?.[locale] !== 'present') {
      errors.push(`Docs topic ${topic.topic_id} is missing localized coverage for ${locale}`);
    }
  }
}

fail(errors);
