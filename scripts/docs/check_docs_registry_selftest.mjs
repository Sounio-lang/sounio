import { execFile } from 'node:child_process';
import { cp, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(scriptDir, '..', '..');
const syncScript = path.join(repoRoot, 'scripts/docs/sync_governance_metadata.mjs');
const registryCheckScript = path.join(repoRoot, 'scripts/docs/check_docs_registry.mjs');
const parityCheckScript = path.join(repoRoot, 'website/scripts/check-docs-parity.mjs');
const locales = ['en', 'pt', 'el', 'zh', 'ja', 'es'];

async function runNode(scriptPath, cwd) {
  try {
    const { stdout, stderr } = await execFileAsync(process.execPath, [scriptPath], {
      cwd,
      maxBuffer: 128 * 1024 * 1024,
    });
    return { ok: true, stdout, stderr, code: 0 };
  } catch (error) {
    return {
      ok: false,
      stdout: error.stdout ?? '',
      stderr: [
        error.stderr ?? '',
        error.message ?? '',
        `code=${error.code ?? ''} signal=${error.signal ?? ''}`,
      ].filter(Boolean).join('\n'),
      code: error.code ?? 1,
    };
  }
}

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

async function writeFixtureFile(rootDir, relPath, content) {
  const absPath = path.join(rootDir, relPath);
  await mkdir(path.dirname(absPath), { recursive: true });
  await writeFile(absPath, content, 'utf8');
}

async function createBaseFixture() {
  const fixtureRoot = await mkdtemp(path.join(os.tmpdir(), 'sounio-docs-registry-selftest-'));

  await writeFixtureFile(
    fixtureRoot,
    'README.md',
    '# Fixture Repo\n\n- [Docs](docs/README.md)\n- [Paper](paper/README.md)\n'
  );

  await writeFixtureFile(
    fixtureRoot,
    'docs/README.md',
    [
      '# Fixture Docs',
      '',
      '- [Getting Started](guide/getting-started.md)',
      '- [GPU Runtime](features/GPU_RUNTIME.md)',
      '- [History](archived/HISTORY.md)',
      '- [Paper](../paper/README.md)',
      '',
    ].join('\n')
  );

  await writeFixtureFile(
    fixtureRoot,
    'docs/guide/getting-started.md',
    '# Getting Started\n\nThis is the repo-native getting started guide.\n'
  );
  await writeFixtureFile(
    fixtureRoot,
    'docs/features/GPU_RUNTIME.md',
    '# GPU Runtime\n\nThis page references tracked runtime evidence.\n'
  );
  await writeFixtureFile(
    fixtureRoot,
    'docs/archived/HISTORY.md',
    '# History\n\nLegacy notes preserved for lineage.\n'
  );

  await writeFixtureFile(
    fixtureRoot,
    'paper/README.md',
    '# Paper\n\nThis paper entrypoint is evidence-backed.\n'
  );
  await writeFixtureFile(fixtureRoot, 'paper/reproduce.sh', '#!/usr/bin/env bash\nexit 0\n');
  await writeFixtureFile(fixtureRoot, 'scripts/paper/paper_repro_gate.sh', '#!/usr/bin/env bash\nexit 0\n');
  await writeFixtureFile(fixtureRoot, 'scripts/paper/paper_submission_pack.sh', '#!/usr/bin/env bash\nexit 0\n');
  await writeFixtureFile(fixtureRoot, 'artifacts/omega/gpu_runtime_attest_gate.v1.json', '{"status":"pass"}\n');

  for (const locale of locales) {
    await writeFixtureFile(
      fixtureRoot,
      `website/src/content/docs/${locale}/getting-started.mdx`,
      ['---', 'title: "Getting Started"', 'description: "Start here."', '---', '', '# Getting Started', ''].join('\n')
    );
    await writeFixtureFile(
      fixtureRoot,
      `website/src/content/docs/${locale}/gpu.mdx`,
      ['---', 'title: "GPU Runtime"', 'description: "GPU docs."', '---', '', '# GPU Runtime', ''].join('\n')
    );
  }

  const syncResult = await runNode(syncScript, fixtureRoot);
  assert(syncResult.ok, `Fixture sync failed:\n${syncResult.stderr || syncResult.stdout}`);

  return fixtureRoot;
}

async function cloneFixture(baseDir, label) {
  const scenarioRoot = await mkdtemp(path.join(os.tmpdir(), `sounio-docs-registry-${label}-`));
  await rm(scenarioRoot, { recursive: true, force: true });
  await cp(baseDir, scenarioRoot, { recursive: true });
  return scenarioRoot;
}

async function expectFailure(result, expectedSubstring, scenarioName) {
  assert(!result.ok, `${scenarioName} unexpectedly passed`);
  const output = `${result.stdout}\n${result.stderr}`;
  assert(
    output.includes(expectedSubstring),
    `${scenarioName} failed for the wrong reason. Expected to find "${expectedSubstring}" in:\n${output}`
  );
}

async function main() {
  const baseDir = await createBaseFixture();
  const cleanupPaths = [baseDir];

  try {
    const baselineRegistry = await runNode(registryCheckScript, baseDir);
    assert(baselineRegistry.ok, `Baseline registry check failed:\n${baselineRegistry.stderr || baselineRegistry.stdout}`);

    const baselineParity = await runNode(parityCheckScript, path.join(baseDir, 'website'));
    assert(baselineParity.ok, `Baseline parity check failed:\n${baselineParity.stderr || baselineParity.stdout}`);

    const brokenLinkDir = await cloneFixture(baseDir, 'broken-link');
    cleanupPaths.push(brokenLinkDir);
    await writeFixtureFile(
      brokenLinkDir,
      'docs/README.md',
      `${await readFile(path.join(brokenLinkDir, 'docs/README.md'), 'utf8')}\n- [Broken](missing.md)\n`
    );
    await expectFailure(
      await runNode(registryCheckScript, brokenLinkDir),
      'Broken front-door/current-canon relative link in docs/README.md: missing.md',
      'broken repo-doc link detection'
    );

    const slugDriftDir = await cloneFixture(baseDir, 'slug-drift');
    cleanupPaths.push(slugDriftDir);
    const slugRegistryPath = path.join(slugDriftDir, 'docs/governance/topic-registry.v1.json');
    const slugRegistry = JSON.parse(await readFile(slugRegistryPath, 'utf8'));
    const gettingStarted = slugRegistry.topics.find((topic) => topic.topic_id === 'website.docs.getting-started');
    gettingStarted.website_slug = 'getting-started-stale';
    await writeFile(slugRegistryPath, `${JSON.stringify(slugRegistry, null, 2)}\n`, 'utf8');
    await expectFailure(
      await runNode(parityCheckScript, path.join(slugDriftDir, 'website')),
      'Website slug drift for website.docs.getting-started',
      'stale website alias detection'
    );

    const missingLocaleDir = await cloneFixture(baseDir, 'missing-locale');
    cleanupPaths.push(missingLocaleDir);
    await rm(path.join(missingLocaleDir, 'website/src/content/docs/ja/gpu.mdx'));
    await expectFailure(
      await runNode(parityCheckScript, path.join(missingLocaleDir, 'website')),
      'Locale status drift for website.docs.gpu ja: expected missing, found present',
      'missing locale page detection'
    );

    const historicalDir = await cloneFixture(baseDir, 'historical-status');
    cleanupPaths.push(historicalDir);
    const archivedPath = path.join(historicalDir, 'docs/archived/HISTORY.md');
    const archivedContent = await readFile(archivedPath, 'utf8');
    await writeFile(
      archivedPath,
      archivedContent.replace(/^> Docs status: .*?\n/m, ''),
      'utf8'
    );
    await expectFailure(
      await runNode(registryCheckScript, historicalDir),
      'docs/archived/HISTORY.md is missing a visible archived status note',
      'historical status labeling detection'
    );

    const missingArtifactDir = await cloneFixture(baseDir, 'missing-artifact');
    cleanupPaths.push(missingArtifactDir);
    await rm(path.join(missingArtifactDir, 'artifacts/omega/gpu_runtime_attest_gate.v1.json'));
    await expectFailure(
      await runNode(registryCheckScript, missingArtifactDir),
      'Missing related artifact: artifacts/omega/gpu_runtime_attest_gate.v1.json',
      'missing evidence artifact detection'
    );

    console.log('Docs registry selftest passed (5 failure scenarios + baseline).');
  } finally {
    await Promise.all(cleanupPaths.map((target) => rm(target, { recursive: true, force: true })));
  }
}

await main();
