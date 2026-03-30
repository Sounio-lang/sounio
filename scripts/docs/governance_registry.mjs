import { access, readdir, readFile } from 'node:fs/promises';
import path from 'node:path';

export const REGISTRY_RELATIVE_PATH = 'docs/governance/topic-registry.v1.json';
export const MATRIX_RELATIVE_PATH = 'docs/governance/DOCS_AUTHORITY_MATRIX.md';
export const ACCEPTANCE_RELATIVE_PATH = 'docs/governance/DOCS_ACCEPTANCE_REPORT.md';
export const LOCALES = ['en', 'pt', 'el', 'zh', 'ja', 'es'];
export const NON_ENGLISH_LOCALES = LOCALES.filter((locale) => locale !== 'en');
export const COLLECTIONS = ['docs', 'tutorials', 'showcases', 'blog'];
export const REPO_META_MARKER = '<!-- docs:meta';
export const STATUS_NOTE_START = '<!-- docs:status-note:start -->';
export const STATUS_NOTE_END = '<!-- docs:status-note:end -->';

const ACTIVE_IMPLEMENTATION_DOCS = new Set([
  'docs/implementation/BASELINE_STEWARDSHIP_PLAN.md',
  'docs/implementation/BASELINE_STEWARDSHIP_PR_NOTE.md',
  'docs/implementation/BASELINE_STEWARDSHIP_MAINTAINER_MEMO.md',
  'docs/implementation/GPU_COMPILER_CONTRACTS.md',
  'docs/implementation/MV_CORE_CHECKLIST.md',
  'docs/implementation/PAPER_ARTIFACT_PACKAGING_SPEC.md',
  'docs/implementation/PRODUCTION_READYNESS.md',
  'docs/implementation/SELFHOST_AUTHORITY_MODEL.md',
  'docs/implementation/SELFHOST_CHECKLIST.md',
  'docs/implementation/SELFHOST_DEBT_REGISTER.md',
  'docs/implementation/SELFHOST_RELEASE_TRAIN.md',
  'docs/implementation/SELFHOST_REPRODUCIBILITY_REPORT.md',
  'docs/implementation/SELF_HOSTED_COMPILER.md',
  'docs/implementation/SELF_HOSTING_PHASES.md',
  'docs/implementation/TOOLING_SUMMARY.md',
]);

const ACTIVE_FEATURE_DOCS = new Set([
  'docs/features/GENOMICS_FEATURES.md',
  'docs/features/GPU_RUNTIME.md',
  'docs/features/HOT_RELOAD.md',
  'docs/features/PLATFORM_SUPPORT.md',
  'docs/features/SMT_INTEGRATION_GUIDE.md',
  'docs/features/VISUAL_SHOWCASE_INTEGRATION.md',
]);

const ACTIVE_RESEARCH_DOCS = new Set([
  'docs/research/RESEARCH_VALIDATION_SUMMARY.md',
  'docs/research/epistemic_algebra_review.md',
  'docs/research/vancomycin-uncertainty.md',
]);

const WEBSITE_DOC_OVERRIDES = {
  compiler: {
    repo_doc_path: 'docs/compiler/COMPILER_ARCHITECTURE_OVERVIEW.md',
    audience: 'contributors',
    owner_agent: 'A4',
  },
  'compiler/codegen': {
    repo_doc_path: 'docs/compiler/CODE_GENERATION_ARCHITECTURE.md',
    audience: 'contributors',
    owner_agent: 'A4',
  },
  'compiler/effect-system': {
    repo_doc_path: 'docs/compiler/EFFECT_SYSTEM_ARCHITECTURE.md',
    audience: 'contributors',
    owner_agent: 'A4',
  },
  'compiler/lexer-parser': {
    repo_doc_path: 'docs/compiler/LEXER_PARSER_ARCHITECTURE.md',
    audience: 'contributors',
    owner_agent: 'A4',
  },
  'compiler/scientific-features': {
    repo_doc_path: 'docs/compiler/SCIENTIFIC_FEATURES_ARCHITECTURE.md',
    audience: 'contributors',
    owner_agent: 'A4',
  },
  'compiler/type-system': {
    repo_doc_path: 'docs/compiler/TYPE_SYSTEM_ARCHITECTURE.md',
    audience: 'contributors',
    owner_agent: 'A4',
  },
  epistemic: {
    repo_doc_path: 'docs/reference/KNOWLEDGE_REFERENCE.md',
    audience: 'users',
    owner_agent: 'A3',
  },
  examples: {
    repo_doc_path: 'examples/README.md',
    audience: 'users',
    owner_agent: 'A5',
  },
  'getting-started': {
    repo_doc_path: 'docs/guide/getting-started.md',
    audience: 'users',
    owner_agent: 'A2',
  },
  gpu: {
    repo_doc_path: 'docs/features/GPU_RUNTIME.md',
    audience: 'users',
    owner_agent: 'A6',
    related_artifacts: ['artifacts/omega/gpu_runtime_attest_gate.v1.json'],
  },
  stdlib: {
    repo_doc_path: 'docs/stdlib/STDLIB_REFERENCE.md',
    audience: 'users',
    owner_agent: 'A3',
  },
  'spec/language-spec': {
    repo_doc_path: 'spec/LANGUAGE_SPECIFICATION.md',
    audience: 'users',
    owner_agent: 'A3',
  },
  'vancomycin-uncertainty': {
    repo_doc_path: 'docs/research/vancomycin-uncertainty.md',
    audience: 'researchers',
    owner_agent: 'A6',
    related_artifacts: ['website/public/docs/assets/vancomycin-ship/check_pass.png'],
  },
};

const MANUAL_REPO_TOPIC_OVERRIDES = {
  'README.md': {
    topic_id: 'repo.frontdoor.readme',
    audience: 'users',
    owner_agent: 'A2',
    authority: 'repo_only',
    related_artifacts: [],
  },
  'docs/README.md': {
    topic_id: 'repo.frontdoor.docs-index',
    audience: 'users',
    owner_agent: 'A2',
    authority: 'repo_only',
    related_artifacts: ['docs/governance/DOCS_AUTHORITY_MATRIX.md'],
  },
  'docs/codebase_overview.md': {
    topic_id: 'repo.contributor.codebase-overview',
    audience: 'contributors',
    owner_agent: 'A5',
    authority: 'repo_only',
    related_artifacts: [],
  },
  'docs/governance/DOCS_AUTHORITY_MATRIX.md': {
    topic_id: 'repo.governance.authority-matrix',
    audience: 'maintainers',
    owner_agent: 'A1',
    authority: 'repo_only',
    related_artifacts: [REGISTRY_RELATIVE_PATH],
  },
  'docs/governance/DOCS_ACCEPTANCE_REPORT.md': {
    topic_id: 'repo.governance.acceptance-report',
    audience: 'maintainers',
    owner_agent: 'A0',
    authority: 'repo_only',
    related_artifacts: [REGISTRY_RELATIVE_PATH, MATRIX_RELATIVE_PATH],
  },
  'paper/README.md': {
    topic_id: 'repo.paper.readme',
    audience: 'researchers',
    owner_agent: 'A6',
    authority: 'repo_only',
    related_artifacts: [
      'paper/reproduce.sh',
      'scripts/paper/paper_repro_gate.sh',
      'scripts/paper/paper_submission_pack.sh',
    ],
  },
  'spec/LANGUAGE_SPECIFICATION.md': {
    topic_id: 'repo.spec.language-spec',
    audience: 'users',
    owner_agent: 'A3',
    authority: 'repo_only',
    related_artifacts: [],
  },
};

const SKIP_REPO_PATHS = new Set([
  'docs/CNAME',
  'docs/favicon.ico',
  'docs/index.html',
  REGISTRY_RELATIVE_PATH,
]);

function normalizeRelative(relPath) {
  return relPath.replace(/\\/g, '/').replace(/^\.\//, '');
}

function stripExtension(relPath) {
  return relPath.replace(/\.[^.]+$/, '');
}

function topicIdFromPath(relPath, prefix = 'repo') {
  const stem = stripExtension(relPath)
    .replace(/[^A-Za-z0-9/.-]+/g, '-')
    .replace(/\//g, '.')
    .replace(/_+/g, '-')
    .replace(/\.+/g, '.')
    .replace(/^-+/, '')
    .replace(/-+$/, '')
    .toLowerCase();
  return `${prefix}.${stem}`;
}

function topicIdFromSlug(collection, slug) {
  return `website.${collection}.${slug.replace(/\//g, '.').toLowerCase()}`;
}

async function pathExists(absPath) {
  try {
    await access(absPath);
    return true;
  } catch {
    return false;
  }
}

async function listFilesRecursive(absDir, predicate) {
  const results = [];
  if (!(await pathExists(absDir))) {
    return results;
  }

  async function walk(currentDir) {
    const entries = await readdir(currentDir, { withFileTypes: true });
    entries.sort((a, b) => a.name.localeCompare(b.name));

    for (const entry of entries) {
      const absPath = path.join(currentDir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === 'assets' && currentDir.endsWith(`${path.sep}docs`)) {
          continue;
        }
        await walk(absPath);
      } else if (predicate(absPath)) {
        results.push(absPath);
      }
    }
  }

  await walk(absDir);
  return results;
}

function inferOwnerAgentFromSlug(slug) {
  if (slug.startsWith('compiler/')) {
    return 'A4';
  }
  if (slug === 'compiler') {
    return 'A4';
  }
  if (slug.startsWith('getting-started')) {
    return 'A2';
  }
  if (slug.startsWith('tooling') || slug === 'examples') {
    return 'A5';
  }
  if (slug === 'gpu' || slug === 'vancomycin-uncertainty') {
    return 'A6';
  }
  if (slug === 'stdlib' || slug.startsWith('spec') || slug === 'epistemic') {
    return 'A3';
  }
  return 'A2';
}

function inferAudienceFromSlug(slug) {
  if (slug.startsWith('compiler/')) {
    return 'contributors';
  }
  if (slug === 'vancomycin-uncertainty') {
    return 'researchers';
  }
  return 'users';
}

function defaultWebsiteValidation(collection, ownerAgent, websiteSlug) {
  const commands = ['bash scripts/check_docs_registry.sh'];

  if (collection === 'docs') {
    commands.push('node website/scripts/check-docs-parity.mjs');
  }

  commands.push('npm --prefix website run check:quality');

  if (ownerAgent === 'A6' && websiteSlug === 'vancomycin-uncertainty') {
    commands.push('bash paper/reproduce.sh');
  }

  return dedupe(commands);
}

function dedupe(values) {
  return [...new Set(values)];
}

function defaultLocaleStatus(collection, locale, exists) {
  if (collection === 'docs') {
    return exists ? 'present' : 'missing';
  }
  if (locale === 'en') {
    return exists ? 'present' : 'missing';
  }
  return 'english_only';
}

function inferRepoTopicDetails(relPath) {
  const override = MANUAL_REPO_TOPIC_OVERRIDES[relPath];
  if (override) {
    return {
      topic_id: override.topic_id,
      audience: override.audience,
      owner_agent: override.owner_agent,
      authority: override.authority,
      related_artifacts: override.related_artifacts ?? [],
    };
  }

  if (relPath.startsWith('docs/archived/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'maintainers',
      owner_agent: 'A7',
      authority: 'archived',
      related_artifacts: [],
    };
  }

  if (relPath.startsWith('docs/implementation/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'maintainers',
      owner_agent: 'A7',
      authority: ACTIVE_IMPLEMENTATION_DOCS.has(relPath) ? 'repo_only' : 'historical',
      related_artifacts:
        relPath === 'docs/implementation/PAPER_ARTIFACT_PACKAGING_SPEC.md'
          ? ['scripts/paper/package_paper_artifacts.sh', 'scripts/paper/paper_submission_pack.sh']
          : [],
    };
  }

  if (relPath.startsWith('docs/compiler/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'contributors',
      owner_agent: 'A4',
      authority: 'repo_only',
      related_artifacts: [],
    };
  }

  if (relPath.startsWith('docs/reference/') || relPath.startsWith('docs/stdlib/') || relPath.startsWith('spec/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'users',
      owner_agent: 'A3',
      authority: 'repo_only',
      related_artifacts: [],
    };
  }

  if (relPath.startsWith('docs/guide/') || relPath.startsWith('docs/contributor-guide/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: relPath.startsWith('docs/contributor-guide/') ? 'contributors' : 'users',
      owner_agent: 'A5',
      authority: 'repo_only',
      related_artifacts: [],
    };
  }

  if (relPath.startsWith('docs/features/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'users',
      owner_agent: 'A6',
      authority: ACTIVE_FEATURE_DOCS.has(relPath) ? 'repo_only' : 'historical',
      related_artifacts:
        relPath === 'docs/features/GPU_RUNTIME.md'
          ? ['artifacts/omega/gpu_runtime_attest_gate.v1.json']
          : [],
    };
  }

  if (relPath.startsWith('docs/research/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'researchers',
      owner_agent: 'A6',
      authority: ACTIVE_RESEARCH_DOCS.has(relPath) ? 'repo_only' : 'historical',
      related_artifacts: [],
    };
  }

  if (relPath === 'docs/governance/DOCS_AUTHORITY_MATRIX.md') {
    return {
      topic_id: 'repo.governance.authority-matrix',
      audience: 'maintainers',
      owner_agent: 'A1',
      authority: 'repo_only',
      related_artifacts: [REGISTRY_RELATIVE_PATH],
    };
  }

  if (relPath.startsWith('docs/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'users',
      owner_agent: 'A2',
      authority: 'repo_only',
      related_artifacts: [],
    };
  }

  if (relPath.startsWith('examples/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'users',
      owner_agent: 'A5',
      authority: 'repo_only',
      related_artifacts: [],
    };
  }

  if (relPath.startsWith('paper/')) {
    return {
      topic_id: topicIdFromPath(relPath),
      audience: 'researchers',
      owner_agent: 'A6',
      authority: 'repo_only',
      related_artifacts: ['paper/reproduce.sh', 'scripts/paper/paper_repro_gate.sh'],
    };
  }

  return {
    topic_id: topicIdFromPath(relPath),
    audience: 'contributors',
    owner_agent: 'A2',
    authority: 'repo_only',
    related_artifacts: [],
  };
}

function defaultRepoValidation(relPath, ownerAgent, authority) {
  const commands = ['bash scripts/check_docs_registry.sh'];

  if (authority !== 'historical' && authority !== 'archived') {
    commands.push('bash scripts/check_docs_consistency.sh');
  }

  if (ownerAgent === 'A4' || ownerAgent === 'A5') {
    commands.push('bash scripts/fast_gate.sh');
  }

  if (ownerAgent === 'A6' && relPath.startsWith('paper/')) {
    commands.push('bash paper/reproduce.sh');
    commands.push('bash scripts/paper/paper_repro_gate.sh');
    commands.push('bash scripts/paper/paper_submission_pack.sh');
  }

  return dedupe(commands);
}

function localeStatusFromPaths(pathsByLocale, collection) {
  return Object.fromEntries(
    LOCALES.map((locale) => [locale, defaultLocaleStatus(collection, locale, Boolean(pathsByLocale[locale]))])
  );
}

function localeSummary(localeStatus) {
  return LOCALES.map((locale) => `${locale}:${localeStatus[locale]}`).join(', ');
}

function buildTopicRecord({
  topicId,
  collection,
  repoDocPath = null,
  websiteSlug = null,
  websitePaths = {},
  audience,
  authority,
  ownerAgent,
  localeStatus,
  validationCommands,
  relatedArtifacts,
}) {
  return {
    topic_id: topicId,
    collection,
    repo_doc_path: repoDocPath,
    website_slug: websiteSlug,
    website_paths: websitePaths,
    audience,
    authority,
    owner_agent: ownerAgent,
    locale_status: localeStatus,
    validation_commands: validationCommands,
    related_artifacts: relatedArtifacts,
  };
}

async function scanWebsiteDocTopics(rootDir) {
  const docsRoot = path.join(rootDir, 'website/src/content/docs');
  const topics = [];
  const claimedRepoPaths = new Set();
  const englishFiles = await listFilesRecursive(path.join(docsRoot, 'en'), (absPath) => absPath.endsWith('.mdx'));

  for (const englishAbsPath of englishFiles) {
    const relWithinLocale = normalizeRelative(path.relative(path.join(docsRoot, 'en'), englishAbsPath));
    const slug = stripExtension(relWithinLocale);
    const override = WEBSITE_DOC_OVERRIDES[slug] ?? {};
    const ownerAgent = override.owner_agent ?? inferOwnerAgentFromSlug(slug);
    const audience = override.audience ?? inferAudienceFromSlug(slug);
    const repoDocPath = override.repo_doc_path ?? null;
    const websitePaths = {};

    for (const locale of LOCALES) {
      const absPath = path.join(docsRoot, locale, relWithinLocale);
      if (await pathExists(absPath)) {
        websitePaths[locale] = normalizeRelative(path.relative(rootDir, absPath));
      }
    }

    if (repoDocPath) {
      claimedRepoPaths.add(repoDocPath);
    }

    topics.push(
      buildTopicRecord({
        topicId: topicIdFromSlug('docs', slug),
        collection: 'docs',
        repoDocPath,
        websiteSlug: slug,
        websitePaths,
        audience,
        authority: repoDocPath ? 'dual' : 'website_only',
        ownerAgent,
        localeStatus: localeStatusFromPaths(websitePaths, 'docs'),
        validationCommands: override.validation_commands ?? defaultWebsiteValidation('docs', ownerAgent, slug),
        relatedArtifacts: override.related_artifacts ?? [],
      })
    );
  }

  return { topics, claimedRepoPaths };
}

async function scanEnglishOnlyCollection(rootDir, collection) {
  const collectionRoot = path.join(rootDir, `website/src/content/${collection}`);
  const files = await listFilesRecursive(collectionRoot, (absPath) => absPath.endsWith('.mdx'));
  return files.map((absPath) => {
    const relWithinCollection = normalizeRelative(path.relative(collectionRoot, absPath));
    const slug = stripExtension(relWithinCollection);
    const ownerAgent = collection === 'showcases' ? 'A6' : 'A2';
    const audience = collection === 'blog' ? 'users' : collection === 'showcases' ? 'researchers' : 'users';
    const websitePaths = {
      en: normalizeRelative(path.relative(rootDir, absPath)),
    };

    return buildTopicRecord({
      topicId: topicIdFromSlug(collection, slug),
      collection,
      repoDocPath: null,
      websiteSlug: `${collection}/${slug}`,
      websitePaths,
      audience,
      authority: 'website_only',
      ownerAgent,
      localeStatus: localeStatusFromPaths(websitePaths, collection),
      validationCommands: defaultWebsiteValidation(collection, ownerAgent, `${collection}/${slug}`),
      relatedArtifacts: [],
    });
  });
}

async function scanRepoTopics(rootDir, claimedRepoPaths) {
  const repoFiles = [];
  const docsFiles = await listFilesRecursive(path.join(rootDir, 'docs'), (absPath) => absPath.endsWith('.md'));
  repoFiles.push(...docsFiles.map((absPath) => normalizeRelative(path.relative(rootDir, absPath))));
  repoFiles.push(MATRIX_RELATIVE_PATH);
  repoFiles.push(ACCEPTANCE_RELATIVE_PATH);

  const rootCandidates = ['README.md', 'spec/LANGUAGE_SPECIFICATION.md'];
  for (const relPath of rootCandidates) {
    if (await pathExists(path.join(rootDir, relPath))) {
      repoFiles.push(relPath);
    }
  }

  const exampleFiles = await listFilesRecursive(path.join(rootDir, 'examples'), (absPath) => absPath.endsWith(`${path.sep}README.md`));
  repoFiles.push(...exampleFiles.map((absPath) => normalizeRelative(path.relative(rootDir, absPath))));

  const paperFiles = await listFilesRecursive(path.join(rootDir, 'paper'), (absPath) => absPath.endsWith('.md'));
  repoFiles.push(...paperFiles.map((absPath) => normalizeRelative(path.relative(rootDir, absPath))));

  const uniqueRepoFiles = dedupe(
    repoFiles.filter((relPath) => !SKIP_REPO_PATHS.has(relPath)).sort((a, b) => a.localeCompare(b))
  );

  return uniqueRepoFiles
    .filter((relPath) => !claimedRepoPaths.has(relPath))
    .map((relPath) => {
      const details = inferRepoTopicDetails(relPath);
      return buildTopicRecord({
        topicId: details.topic_id,
        collection: 'repo',
        repoDocPath: relPath,
        websiteSlug: null,
        websitePaths: {},
        audience: details.audience,
        authority: details.authority,
        ownerAgent: details.owner_agent,
        localeStatus: Object.fromEntries(LOCALES.map((locale) => [locale, 'n/a'])),
        validationCommands: defaultRepoValidation(relPath, details.owner_agent, details.authority),
        relatedArtifacts: details.related_artifacts,
      });
    });
}

export async function buildGovernedTopicRegistry(rootDir) {
  const { topics: websiteDocsTopics, claimedRepoPaths } = await scanWebsiteDocTopics(rootDir);
  const tutorialTopics = await scanEnglishOnlyCollection(rootDir, 'tutorials');
  const showcaseTopics = await scanEnglishOnlyCollection(rootDir, 'showcases');
  const blogTopics = await scanEnglishOnlyCollection(rootDir, 'blog');
  const repoTopics = await scanRepoTopics(rootDir, claimedRepoPaths);

  const topics = [...websiteDocsTopics, ...tutorialTopics, ...showcaseTopics, ...blogTopics, ...repoTopics].sort((a, b) =>
    a.topic_id.localeCompare(b.topic_id)
  );

  return {
    version: 1,
    topics,
  };
}

export function topicSourceOfTruth(topicId) {
  return `${REGISTRY_RELATIVE_PATH}#${topicId}`;
}

export function formatRepoMetadataBlock(topic) {
  return [
    '<!-- docs:meta',
    `topic_id: ${topic.topic_id}`,
    `authority: ${topic.authority}`,
    `audience: ${topic.audience}`,
    'last_validated: 2026-03-07',
    `validated_by: ${topic.owner_agent}`,
    `source_of_truth: ${topicSourceOfTruth(topic.topic_id)}`,
    '-->',
  ].join('\n');
}

export function parseRepoMetadata(content) {
  const match = content.match(/^<!-- docs:meta\n([\s\S]*?)\n-->\n?/);
  if (!match) {
    return null;
  }

  const data = {};
  for (const rawLine of match[1].split('\n')) {
    const line = rawLine.trim();
    if (!line) {
      continue;
    }
    const separator = line.indexOf(':');
    if (separator === -1) {
      continue;
    }
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    data[key] = value;
  }
  return data;
}

export function formatHistoricalStatusNote(topic, relPath) {
  if (topic.authority !== 'historical' && topic.authority !== 'archived') {
    return null;
  }

  const matrixRel = normalizeRelative(path.relative(path.dirname(relPath), MATRIX_RELATIVE_PATH)) || '.';
  const docsIndexRel = normalizeRelative(path.relative(path.dirname(relPath), 'docs/README.md')) || '.';
  const matrixLink = matrixRel.startsWith('.') ? matrixRel : `./${matrixRel}`;
  const docsIndexLink = docsIndexRel.startsWith('.') ? docsIndexRel : `./${docsIndexRel}`;
  const label = topic.authority === 'archived' ? 'archived' : 'historical';

  return [
    STATUS_NOTE_START,
    `> Docs status: \`${label}\``,
    `> This page is preserved for lineage. Start at [Docs Authority Matrix](${matrixLink}) and [docs index](${docsIndexLink}) for the current canonical surface for this topic.`,
    STATUS_NOTE_END,
  ].join('\n');
}

export function stripStatusNote(content) {
  return content.replace(new RegExp(`${STATUS_NOTE_START}[\\s\\S]*?${STATUS_NOTE_END}\\n?`, 'm'), '');
}

export function parseFrontmatter(content) {
  const match = content.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!match) {
    return null;
  }

  const data = {};
  for (const rawLine of match[1].split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) {
      continue;
    }
    const separator = line.indexOf(':');
    if (separator === -1) {
      continue;
    }
    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim().replace(/^"(.*)"$/, '$1');
    data[key] = value;
  }
  return data;
}

export function metadataFieldsForTopic(topic) {
  return {
    topic_id: topic.topic_id,
    authority: topic.authority,
    audience: topic.audience,
    last_validated: '2026-03-07',
    validated_by: topic.owner_agent,
    source_of_truth: topicSourceOfTruth(topic.topic_id),
  };
}

export function formatAuthorityMatrix(registry) {
  const lines = [
    formatRepoMetadataBlock({
      topic_id: 'repo.governance.authority-matrix',
      authority: 'repo_only',
      audience: 'maintainers',
      owner_agent: 'A1',
    }),
    '',
    '# Docs Authority Matrix',
    '',
    'This matrix is the human-readable companion to `docs/governance/topic-registry.v1.json`.',
    '',
    '## Dual-Canon Policy',
    '',
    '- Website pages are canonical for public IA, onboarding, localized docs, tutorials, and concise summaries.',
    '- Repo docs are canonical for deep technical guidance, executable commands, evidence packs, paper workflows, and historical lineage.',
    '- Topics marked `dual` must keep their website summary and repo detail synchronized through the registry rather than by copying long-form prose.',
    '- Topics marked `historical` or `archived` are retained for lineage and must point readers back to the current canonical surface.',
    '',
    '## Topic Matrix',
    '',
    '| Topic ID | Authority | Repo doc | Website slug | Owner | Locales |',
    '| --- | --- | --- | --- | --- | --- |',
  ];

  for (const topic of registry.topics) {
    lines.push(
      `| ${topic.topic_id} | ${topic.authority} | ${topic.repo_doc_path ?? '-'} | ${topic.website_slug ?? '-'} | ${topic.owner_agent} | ${localeSummary(topic.locale_status)} |`
    );
  }

  lines.push('');
  return lines.join('\n');
}

function countBy(values, keyFn) {
  const counts = new Map();
  for (const value of values) {
    const key = keyFn(value);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]));
}

export function formatAcceptanceReport(registry) {
  const totalTopics = registry.topics.length;
  const repoTopics = registry.topics.filter((topic) => topic.repo_doc_path);
  const websiteTopics = registry.topics.filter((topic) => Object.keys(topic.website_paths ?? {}).length > 0);
  const dualTopics = registry.topics.filter((topic) => topic.authority === 'dual');
  const docsTopics = registry.topics.filter((topic) => topic.collection === 'docs');
  const fullyLocalizedDocs = docsTopics.filter((topic) =>
    NON_ENGLISH_LOCALES.every((locale) => topic.locale_status?.[locale] === 'present')
  );
  const missingLocalizedDocs = docsTopics.filter((topic) =>
    NON_ENGLISH_LOCALES.some((locale) => topic.locale_status?.[locale] !== 'present')
  );
  const authorityCounts = countBy(registry.topics, (topic) => topic.authority);
  const ownerCounts = countBy(registry.topics, (topic) => topic.owner_agent);
  const evidenceTopics = registry.topics.filter((topic) => (topic.related_artifacts ?? []).length > 0);
  const validationCommands = dedupe(registry.topics.flatMap((topic) => topic.validation_commands ?? [])).sort((a, b) =>
    a.localeCompare(b)
  );
  const englishOnlyCollections = ['tutorials', 'showcases', 'blog'].map((collection) => {
    const count = registry.topics.filter((topic) => topic.collection === collection).length;
    return `${collection}: ${count}`;
  });

  const lines = [
    formatRepoMetadataBlock({
      topic_id: 'repo.governance.acceptance-report',
      authority: 'repo_only',
      audience: 'maintainers',
      owner_agent: 'A0',
    }),
    '',
    '# Docs Acceptance Report',
    '',
    'This is the editor-in-chief acceptance snapshot generated from `docs/governance/topic-registry.v1.json`.',
    '',
    '## Verdict',
    '',
    '- Status: accepted for the current documentation-governance wave when the listed validation surfaces pass.',
    '- Dual-canon sync contract is active across repo docs, website docs, and localized docs metadata.',
    '- Historical and archived repo docs are labeled and redirected back to the current canonical surface through the authority matrix.',
    '',
    '## Scope Summary',
    '',
    `- Total governed topics: ${totalTopics}`,
    `- Repo-backed topics: ${repoTopics.length}`,
    `- Website-backed topics: ${websiteTopics.length}`,
    `- Dual-canon topics: ${dualTopics.length}`,
  ];

  for (const [authority, count] of authorityCounts) {
    lines.push(`- Authority count \`${authority}\`: ${count}`);
  }

  lines.push('', '## Ownership Summary', '');
  for (const [owner, count] of ownerCounts) {
    lines.push(`- ${owner}: ${count} topics`);
  }

  lines.push('', '## Locale Acceptance', '');
  lines.push(`- Docs collection topics with full six-locale coverage: ${fullyLocalizedDocs.length}/${docsTopics.length}`);
  if (missingLocalizedDocs.length === 0) {
    lines.push('- All governed website docs topics are present in `en`, `pt`, `el`, `zh`, `ja`, and `es`.');
  } else {
    lines.push(`- Missing locale coverage remains on: ${missingLocalizedDocs.map((topic) => topic.topic_id).join(', ')}`);
  }
  lines.push(
    `- English-only website collections allowed by policy and marked in the registry: ${englishOnlyCollections.join('; ')}`
  );

  lines.push('', '## Evidence-Bearing Topics', '');
  for (const topic of evidenceTopics) {
    lines.push(`- ${topic.topic_id}: ${(topic.related_artifacts ?? []).join(', ')}`);
  }

  lines.push('', '## Validation Surfaces', '');
  for (const command of validationCommands) {
    lines.push(`- \`${command}\``);
  }

  lines.push('');
  return lines.join('\n');
}

export async function readRegistryFile(rootDir) {
  const absPath = path.join(rootDir, REGISTRY_RELATIVE_PATH);
  const raw = await readFile(absPath, 'utf8');
  return JSON.parse(raw);
}
