import { readFile, stat } from 'node:fs/promises';
import { writeSync } from 'node:fs';
import path from 'node:path';
import {
  ACCEPTANCE_RELATIVE_PATH,
  LOCALES,
  MATRIX_RELATIVE_PATH,
  REGISTRY_RELATIVE_PATH,
  buildGovernedTopicRegistry,
  formatAcceptanceReportStub,
  isRealValidationDate,
  LEGACY_PLACEHOLDER_DATE,
  UNVALIDATED_BY,
  UNVALIDATED_DATE,
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
  if (placeholderPaths.length > 0) {
    console.log(`Notice: stale generated snapshot ${MATRIX_RELATIVE_PATH} still carries the retired ${LEGACY_PLACEHOLDER_DATE} placeholder, which records no validation (#1240).`);
  }
  if (unvalidatedCount > 0) {
    console.log(`Notice: ${unvalidatedCount} governed documents carry no validation record (last_validated: ${UNVALIDATED_DATE}).`);
  }
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
  // Provenance is checked by provenanceFormatErrors. Only these structural
  // fields are registry-authoritative; comparing generated provenance here
  // would make a stale snapshot block an unrelated PR.
  for (const key of ['topic_id', 'authority', 'audience', 'source_of_truth']) {
    const expected = expectedFields[key];
    if ((actualFields?.[key] ?? '') !== expected) {
      errors.push(`${context} metadata mismatch for ${key}: expected "${expected}"`);
    }
  }
}

// The provenance pair is preserve-per-document (a real record survives the
// sync), so equality against a generator constant is no longer the contract
// for these two fields. The contract that REPLACES it is shape: a real
// YYYY-MM-DD calendar date (isRealValidationDate -- '2026-13-45' is shaped
// like one and is not a day) and a non-empty validator. This keeps the gate
// meaningful for the pair -- it still rejects a malformed or blanked record --
// without rejecting the true values the old constant-equality enforcement used
// to fail on (the defect the self-falsifying lineage recorded as R22/R23).
//
// The one other accepted shape is the explicit no-record marker
// (`unvalidated` / `none`, #1240). The retired 2026-03-07 placeholder is
// shaped like a record and is not one; the generator no longer writes it and
// the checker counts the documents that still carry it (the generated
// DOCS_AUTHORITY_MATRIX.md snapshot until its next refresh).
let unvalidatedCount = 0;
const placeholderPaths = [];

function provenanceFormatErrors(actualFields, context, errors) {
  if (!actualFields) {
    return;
  }
  const date = String(actualFields.last_validated ?? '').trim();
  const validator = String(actualFields.validated_by ?? '').trim();
  if (date === UNVALIDATED_DATE) {
    unvalidatedCount += 1;
    if (validator !== UNVALIDATED_BY) {
      errors.push(`${context} metadata mismatch for validated_by: an unvalidated document names no validator, expected "${UNVALIDATED_BY}", got "${validator}"`);
    }
    return;
  }
  if (date === LEGACY_PLACEHOLDER_DATE) {
    const isExactStaleMatrixPair = context === MATRIX_RELATIVE_PATH && validator === 'A1';
    if (isExactStaleMatrixPair) {
      if (process.env.SOUNIO_DOCS_REGISTRY_STRICT_SNAPSHOT === '1') {
        errors.push(`${context} has stale generated snapshot provenance; refresh it with --snapshot`);
      } else {
        placeholderPaths.push(context);
      }
      return;
    }
    errors.push(`${context} metadata mismatch for last_validated: retired placeholder is valid only in the stale authority-matrix snapshot`);
  }
  if (!isRealValidationDate(date)) {
    errors.push(`${context} metadata mismatch for last_validated: expected a YYYY-MM-DD date, got "${date}"`);
  }
  if (!validator) {
    errors.push(`${context} metadata mismatch for validated_by: expected a non-empty validator, got ""`);
  }
}

async function main() {
  const errors = [];
  const expectedRegistry = await buildGovernedTopicRegistry(rootDir);
  const actualRegistry = await readRegistryFile(rootDir);

  // The checked-in registry is a snapshot (see sync_governance_metadata.mjs):
  // every check below runs against the registry rebuilt from the tree, so a
  // stale snapshot cannot hide a defect and is only reported. Requiring it to
  // be fresh made every docs PR regenerate it, and any two such PRs conflict.
  // SOUNIO_DOCS_REGISTRY_STRICT_SNAPSHOT=1 restores the hard check, for the
  // snapshot-refresh PR itself.
  if (!compareRegistry(expectedRegistry, actualRegistry)) {
    if (process.env.SOUNIO_DOCS_REGISTRY_STRICT_SNAPSHOT === '1') {
      errors.push(`Checked-in ${REGISTRY_RELATIVE_PATH} is stale. Re-run node scripts/docs/sync_governance_metadata.mjs --snapshot`);
    } else {
      console.log(`Notice: ${REGISTRY_RELATIVE_PATH} snapshot differs from the tree (expected between snapshot refreshes; not an error).`);
    }
  }

  // Deliberately NOT a function of expectedRegistry: the acceptance report is
  // a static stub (see formatAcceptanceReportStub), so this check can never
  // race against a concurrent PR that adds or removes an unrelated governed
  // doc. It still catches real drift -- a hand-edited or bit-rotted stub --
  // because the stub is a fixed string, not a corpus scan.
  try {
    const expectedAcceptance = `${formatAcceptanceReportStub().trimEnd()}\n`;
    const actualAcceptance = await readFile(path.join(rootDir, ACCEPTANCE_RELATIVE_PATH), 'utf8');
    if (actualAcceptance !== expectedAcceptance) {
      errors.push(`Checked-in ${ACCEPTANCE_RELATIVE_PATH} is stale. Re-run node scripts/docs/sync_governance_metadata.mjs`);
    }
  } catch {
    errors.push(`Missing acceptance report: ${ACCEPTANCE_RELATIVE_PATH}`);
  }

  for (const topic of expectedRegistry.topics) {
    if (topic.repo_doc_path) {
      const absPath = path.join(rootDir, topic.repo_doc_path);
      try {
        const content = await readFile(absPath, 'utf8');
        const actualMeta = parseRepoMetadata(content);
        // Provenance (last_validated / validated_by) is the document's own
        // record and is preserve-by-default (see preservedProvenance); the
        // structural four fields remain registry-authoritative. A document
        // with NO header still fails against the full default field set.
        metadataMismatch(metadataFieldsForTopic(topic, actualMeta), actualMeta, topic.repo_doc_path, errors);
        provenanceFormatErrors(actualMeta, topic.repo_doc_path, errors);
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
        metadataMismatch(metadataFieldsForTopic(topic, actualMeta), actualMeta, relPath, errors);
        provenanceFormatErrors(actualMeta, relPath, errors);
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
