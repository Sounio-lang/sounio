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

    // 2026-08-16 regression scenarios: the sync must PRESERVE a real
    // provenance record instead of regressing it to the placeholder
    // (.claude/llm_offload_log.md 2026-08-16 WAIVED row), a headerless doc
    // must still GET a header, and the checker must still reject a malformed
    // record now that it no longer enforces the generator constant.
    const preserveDir = await cloneFixture(baseDir, 'preserve-header');
    cleanupPaths.push(preserveDir);
    const preservePath = path.join(preserveDir, 'docs/features/GPU_RUNTIME.md');
    const realHeaderContent = (await readFile(preservePath, 'utf8'))
      .replace(/^last_validated: .*$/m, 'last_validated: 2026-08-13')
      .replace(/^validated_by: .*$/m, 'validated_by: claude');
    await writeFile(preservePath, realHeaderContent, 'utf8');
    const preserveSync = await runNode(syncScript, preserveDir);
    assert(preserveSync.ok, `Preserve-scenario sync failed:\n${preserveSync.stderr || preserveSync.stdout}`);
    const afterSync = await readFile(preservePath, 'utf8');
    assert(
      afterSync === realHeaderContent,
      'sync rewrote a real provenance header -- the 2026-08-16 regression is back'
    );
    assert(
      afterSync.includes('last_validated: 2026-08-13') && afterSync.includes('validated_by: claude'),
      'sync dropped the preserved provenance values'
    );
    const preserveCheck = await runNode(registryCheckScript, preserveDir);
    assert(preserveCheck.ok, `Checker rejected a preserved real header:\n${preserveCheck.stderr || preserveCheck.stdout}`);
    const preserveResync = await runNode(syncScript, preserveDir);
    assert(preserveResync.ok, `Preserve-scenario re-sync failed:\n${preserveResync.stderr || preserveResync.stdout}`);
    assert(
      (await readFile(preservePath, 'utf8')) === realHeaderContent,
      'second sync was not idempotent on a preserved header'
    );

    const headerlessDir = await cloneFixture(baseDir, 'headerless-stamped');
    cleanupPaths.push(headerlessDir);
    const headerlessPath = path.join(headerlessDir, 'docs/guide/getting-started.md');
    const stripped = (await readFile(headerlessPath, 'utf8')).replace(/^<!-- docs:meta\n[\s\S]*?\n-->\n\n/m, '');
    await writeFile(headerlessPath, stripped, 'utf8');
    const headerlessSync = await runNode(syncScript, headerlessDir);
    assert(headerlessSync.ok, `Headerless-scenario sync failed:\n${headerlessSync.stderr || headerlessSync.stdout}`);
    const restamped = await readFile(headerlessPath, 'utf8');
    assert(restamped.startsWith('<!-- docs:meta\n'), 'a genuinely headerless doc was not given a header');
    assert(
      restamped.includes('last_validated: 2026-03-07'),
      'a headerless doc was not stamped with the default provenance placeholder'
    );
    const headerlessCheck = await runNode(registryCheckScript, headerlessDir);
    assert(headerlessCheck.ok, `Checker rejected a freshly stamped headerless doc:\n${headerlessCheck.stderr || headerlessCheck.stdout}`);

    const malformedDir = await cloneFixture(baseDir, 'malformed-provenance');
    cleanupPaths.push(malformedDir);
    const malformedPath = path.join(malformedDir, 'docs/features/GPU_RUNTIME.md');
    const malformedContent = (await readFile(malformedPath, 'utf8'))
      .replace(/^last_validated: .*$/m, 'last_validated: yesterday');
    await writeFile(malformedPath, malformedContent, 'utf8');
    await expectFailure(
      await runNode(registryCheckScript, malformedDir),
      'metadata mismatch for last_validated: expected a YYYY-MM-DD date, got "yesterday"',
      'malformed provenance record detection'
    );

    // A date that is SHAPED like YYYY-MM-DD but is not a calendar day must be
    // rejected too: '2026-13-45' passed the regex the inversion started with
    // and was found by the inverted R22 instrument on its first run. An
    // impossible date carries no more information than the placeholder did.
    const impossibleDir = await cloneFixture(baseDir, 'impossible-date');
    cleanupPaths.push(impossibleDir);
    const impossiblePath = path.join(impossibleDir, 'docs/features/GPU_RUNTIME.md');
    const impossibleContent = (await readFile(impossiblePath, 'utf8'))
      .replace(/^last_validated: .*$/m, 'last_validated: 2026-13-45');
    await writeFile(impossiblePath, impossibleContent, 'utf8');
    await expectFailure(
      await runNode(registryCheckScript, impossibleDir),
      'expected a YYYY-MM-DD date',
      'impossible-calendar-date detection'
    );
    const impossibleSync = await runNode(syncScript, impossibleDir);
    assert(impossibleSync.ok, `Impossible-date sync failed:\n${impossibleSync.stderr || impossibleSync.stdout}`);
    const impossibleAfterSync = await readFile(impossiblePath, 'utf8');
    assert(
      impossibleAfterSync.includes('last_validated: 2026-03-07'),
      'the sync PRESERVED an impossible date instead of falling back to the default'
    );
    const impossibleCheckAfterSync = await runNode(registryCheckScript, impossibleDir);
    assert(
      impossibleCheckAfterSync.ok,
      `Checker rejected the fallback-stamped doc:\n${impossibleCheckAfterSync.stderr || impossibleCheckAfterSync.stdout}`
    );

    // 2026-08-18 corpus-race regression: DOCS_ACCEPTANCE_REPORT.md used to
    // carry whole-corpus counts (total governed topics, per-authority/owner
    // breakdowns, evidence-bearing topics, validation-surface union) that were
    // a pure function of every governed doc present at scan time. Two PRs that
    // each added an unrelated governed doc would both pass their own CI, then
    // the moment the second one landed beside the first, whichever PR hadn't
    // merged yet carried counts that were off by the topics the other one
    // added -- and every PR open at that moment inherited the failure. This
    // recurred same-day (#1804, then #1839 eight hours later).
    //
    // Arm 1 (the old gate sees the bug) is the historical record: both #1804
    // and #1839 are exactly that failure, independently reproduced against a
    // live two-branch merge before this fix (see the commit message). Arms 2
    // and 3 below are the permanent regression control: growing the corpus
    // must never touch the acceptance report (2), and a hand-corrupted stub
    // must still be caught (3) -- proving the gate was narrowed, not deleted.
    const corpusGrowthDir = await cloneFixture(baseDir, 'corpus-growth');
    cleanupPaths.push(corpusGrowthDir);
    const acceptanceReportPath = 'docs/governance/DOCS_ACCEPTANCE_REPORT.md';
    const beforeGrowth = await readFile(path.join(corpusGrowthDir, acceptanceReportPath), 'utf8');
    await writeFixtureFile(
      corpusGrowthDir,
      'docs/guide/second-unrelated-doc.md',
      '# Second Unrelated Doc\n\nSimulates a concurrent PR landing an unrelated governed doc.\n'
    );
    const corpusGrowthSync = await runNode(syncScript, corpusGrowthDir);
    assert(corpusGrowthSync.ok, `Corpus-growth sync failed:\n${corpusGrowthSync.stderr || corpusGrowthSync.stdout}`);
    const afterGrowth = await readFile(path.join(corpusGrowthDir, acceptanceReportPath), 'utf8');
    assert(
      afterGrowth === beforeGrowth,
      'adding an unrelated governed doc changed DOCS_ACCEPTANCE_REPORT.md -- the corpus-count race is back'
    );
    const corpusGrowthCheck = await runNode(registryCheckScript, corpusGrowthDir);
    assert(
      corpusGrowthCheck.ok,
      `Registry check failed after adding an unrelated doc and resyncing:\n${corpusGrowthCheck.stderr || corpusGrowthCheck.stdout}`
    );

    const corruptedStubDir = await cloneFixture(baseDir, 'corrupted-stub');
    cleanupPaths.push(corruptedStubDir);
    const corruptedStubPath = path.join(corruptedStubDir, acceptanceReportPath);
    const corruptedStubContent = (await readFile(corruptedStubPath, 'utf8')).replace(
      '## Verdict',
      '## Verdict (hand-edited, does not match the generator)'
    );
    await writeFile(corruptedStubPath, corruptedStubContent, 'utf8');
    await expectFailure(
      await runNode(registryCheckScript, corruptedStubDir),
      `Checked-in ${acceptanceReportPath} is stale`,
      'hand-corrupted acceptance-report stub detection'
    );

    console.log(
      'Docs registry selftest passed (5 failure scenarios + baseline + 4 provenance-preserve scenarios + 2 corpus-race scenarios).'
    );
  } finally {
    await Promise.all(cleanupPaths.map((target) => rm(target, { recursive: true, force: true })));
  }
}

await main();
