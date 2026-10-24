import { readFile, realpath, access, mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import { scanMarkdown } from './validate_llm_guide.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const coverageFile = 'docs/el/coverage.json';
const greekText = /\p{Script=Greek}/u;
const safePath = value => typeof value === 'string' && !/[\\\0\r\n:]/.test(value) && !path.posix.isAbsolute(value) && value.split('/').every(part => part && part !== '.' && part !== '..' && part !== '.git');

async function readRepository(file) {
  if (!safePath(file)) throw new Error(`Unsafe path: ${file}`);
  const resolved = await realpath(path.join(root, file));
  const relative = path.relative(root, resolved);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error(`Escaping path: ${file}`);
  return readFile(resolved, 'utf8');
}

function shellContainsSounio(body) {
  let fence = null;
  for (const rawLine of body.split('\n')) {
    const line = rawLine.replace(/^(?:[ \t]*>[ \t]?)+/, '');
    if (fence) {
      const closing = /^[ \t]*(`{3,}|~{3,})[ \t]*$/.exec(line);
      if (closing && closing[1][0] === fence.delimiter[0] && closing[1].length >= fence.delimiter.length) fence = null;
      else if (fence.shell && /^[ \t]*(?:fn|struct)\s+\w+/.test(line)) return true;
      continue;
    }
    const opening = /^[ \t]*(`{3,}|~{3,})(.*)$/.exec(line);
    if (opening) fence = { delimiter: opening[1], shell: /^(?:sh|bash)(?:\s|$)/i.test(opening[2].trim()) };
  }
  return false;
}

function coverageIssues(coverage, documents, expectedSources) {
  const errors = [];
  if (coverage.schema !== 'sounio.greek-journey.v1' || coverage.locale !== 'el' || coverage.scope !== 'public-user-journey') errors.push('invalid coverage identity');
  if (!/^[a-f0-9]{40}$/.test(coverage.source_revision ?? '')) errors.push('invalid source revision');
  for (const field of ['full_specification_translated', 'entire_website_translated_and_reviewed', 'new_compiler_execution']) if (coverage[field] !== false) errors.push(`unsupported claim: ${field}`);
  const files = [...(coverage.documents ?? []), ...(coverage.website_pages ?? [])];
  if (!files.length || files.some(file => !safePath(file)) || new Set(files).size !== files.length) errors.push('invalid covered file set');
  const required = ['README.el.md', 'INSTALL.el.md', 'docs/el/README.md', 'docs/el/getting-started.md', 'docs/el/applications-chemistry.md', 'docs/el/applications-ai.md', 'docs/el/developing.md', 'docs/el/contributing.md', 'docs/el/language-reference.md', 'docs/el/review-receipt.md', 'docs/guide/LLM_PROGRAMMING_GUIDE_EL.md'];
  for (const file of required) if (!files.includes(file)) errors.push(`missing journey surface: ${file}`);
  for (const page of ['hello-world', 'first-program']) {
    const file = `website/src/content/docs/el/getting-started/${page}.mdx`;
    if (!coverage.website_pages?.includes(file)) errors.push(`missing canonical onboarding surface: ${file}`);
  }
  for (const file of files) {
    const body = documents.get(file);
    if (typeof body !== 'string') { errors.push(`missing document: ${file}`); continue; }
    const parsed = scanMarkdown(body, file);
    const heading = file.startsWith('website/')
      ? /^title: (.+)$/m.exec(body)?.[1] ?? ''
      : /^# (.+)$/m.exec(parsed.prose)?.[1] ?? '';
    if (!greekText.test(heading)) errors.push(`missing Greek heading: ${file}`);
    if (file.startsWith('website/')) {
      for (const field of ['title', 'description']) if (!greekText.test(new RegExp(`^${field}: (.+)$`, 'm').exec(body)?.[1] ?? '')) errors.push(`untranslated ${field}: ${file}`);
    }
    for (const snippet of parsed.snippets) {
      if (![...expectedSources.values()].includes(snippet.source.replace(/\n$/, ''))) errors.push(`unbound Sounio source: ${file}`);
    }
    if (/```(?:sh|bash)\n[^]*?\bsounio run\b/.test(body)) errors.push(`obsolete launcher: ${file}`);
    if (shellContainsSounio(body)) errors.push(`Sounio source in shell fence: ${file}`);
  }
  const guide = documents.get(coverage.llm_guide) ?? '';
  const topics = scanMarkdown(guide, coverage.llm_guide ?? '').sections;
  if (topics.length !== 15 || new Set(topics).size !== 15) errors.push('Greek manual must retain fifteen distinct topics');
  const rootReadme = documents.get('README.el.md') ?? '';
  for (const target of ['getting-started.md', 'applications-chemistry.md', 'applications-ai.md', 'developing.md', 'contributing.md', 'LLM_PROGRAMMING_GUIDE_EL.md']) if (!rootReadme.includes(target)) errors.push(`missing frontdoor destination: ${target}`);
  return errors;
}

async function linkIssues(documents) {
  const errors = [];
  let links = 0;
  for (const [file, body] of documents) {
    const prose = scanMarkdown(body, file).prose;
    for (const match of prose.matchAll(/\[[^\]]*\]\(([^\s)]+)\)/g)) {
      links += 1;
      let target = match[1].split('#')[0];
      if (!target) continue;
      const repositoryUrl = /^https:\/\/github\.com\/Sounio-lang\/sounio\/blob\/main\/(.+)$/.exec(target);
      if (repositoryUrl) target = repositoryUrl[1];
      else if (/^[a-z][a-z0-9+.-]*:/i.test(target)) continue;
      else if (target.startsWith('/el/')) continue;
      else target = path.posix.normalize(path.posix.join(path.posix.dirname(file), target));
      try {
        target = decodeURIComponent(target).replace(/\/+$/, '');
        if (!safePath(target)) throw new Error('unsafe link');
        await access(path.join(root, target));
      } catch { errors.push(`missing local link: ${file} -> ${match[1]}`); }
    }
  }
  return { links, errors };
}

function firstSuccessBlock(body) {
  const block = /```sh\n(bash <<'SH'\n[\s\S]*?\nSH)\n```/.exec(body)?.[1];
  if (!block) throw new Error('Missing executable Greek first-success block');
  return block;
}

async function mockPreflightChecks(body) {
  const directory = await mkdtemp(path.join(tmpdir(), 'sounio-greek-preflight-mock-'));
  try {
    await mkdir(path.join(directory, 'bin'));
    await mkdir(path.join(directory, 'mock-tools'));
    const hash = 'a'.repeat(64);
    await writeFile(path.join(directory, 'mock-compiler'), 'NOT A COMPILER\n');
    await writeFile(path.join(directory, 'bin/madaros-linux-x86_64.sha256'), `${hash}  mock-compiler\n`);
    const tools = {
      uname: '#!/bin/bash\nprintf "%s\\n" "${GREEK_TEST_HOST:-Linux x86_64}"\n',
      file: '#!/bin/bash\nprintf "mock inspection only\\n"\n',
      sha256sum: `#!/bin/bash\nif [ "\${GREEK_TEST_HASH_FAIL:-0}" = 1 ]; then exit 1; fi\nprintf '%s  mock-compiler\\n' "\${GREEK_TEST_DIGEST:-${hash}}"\n`,
    };
    for (const [name, source] of Object.entries(tools)) await writeFile(path.join(directory, 'mock-tools', name), source, { mode: 0o755 });
    await writeFile(path.join(directory, 'bin/souc'), `#!/bin/bash
printf '%s\\n' "$*" >> "$GREEK_TEST_LOG"
case "$1" in
  info) if [ "\${GREEK_TEST_INFO_FAIL:-0}" = 1 ]; then exit 1; fi; printf 'raw_elf: %s\\n' "$PWD/\${GREEK_TEST_RAW:-mock-compiler}" ;;
  --version) printf 'mock compiler identity only\\n' ;;
  check) if [ "\${GREEK_TEST_CHECK_FAIL:-0}" = 1 ]; then exit 1; fi ;;
  run) if [ "\${GREEK_TEST_RUN_FAIL:-0}" = 1 ]; then printf 'preserve-me\\n'; exit 3; fi; if [ "\${GREEK_TEST_NO_GREETING:-0}" = 1 ]; then printf 'missing greeting\\n'; else printf 'Hello, Sounio\\n'; fi ;;
  *) exit 2 ;;
esac
`, { mode: 0o755 });
    const block = firstSuccessBlock(body);
    const syntax = spawnSync('/bin/bash', ['-n'], { input: block, encoding: 'utf8' });
    assert.equal(syntax.status, 0, syntax.stderr);
    const cases = [
      { label: 'success', env: {}, status: 0, run: true },
      { label: 'host mismatch', env: { GREEK_TEST_HOST: 'Darwin arm64' }, status: 1, run: false },
      { label: 'engine mismatch', env: { SOUNIO_SOUC_ENGINE: 'lean_single' }, status: 1, run: false },
      { label: 'digest mismatch', env: { GREEK_TEST_DIGEST: 'b'.repeat(64) }, status: 1, run: false },
      { label: 'check failure', env: { GREEK_TEST_CHECK_FAIL: '1' }, status: 1, run: false },
      { label: 'run failure retains stdout', env: { GREEK_TEST_RUN_FAIL: '1' }, status: 1, run: true, stdout: 'preserve-me' },
      { label: 'missing greeting', env: { GREEK_TEST_NO_GREETING: '1' }, status: 1, run: true },
      { label: 'info failure', env: { GREEK_TEST_INFO_FAIL: '1' }, status: 1, run: false },
      { label: 'digest read failure', env: { GREEK_TEST_HASH_FAIL: '1' }, status: 1, run: false },
      { label: 'missing raw artifact', env: { GREEK_TEST_RAW: 'missing' }, status: 1, run: false },
    ];
    for (const [index, entry] of cases.entries()) {
      const log = path.join(directory, `case-${index}.log`);
      await writeFile(log, '');
      const result = spawnSync('/bin/bash', ['-c', block], {
        cwd: directory, input: '', encoding: 'utf8', timeout: 10000,
        env: { PATH: `${directory}/mock-tools:/usr/bin:/bin`, HOME: directory, GREEK_TEST_LOG: log, ...entry.env },
      });
      assert.equal(result.status, entry.status, `${entry.label}: ${result.stdout} ${result.stderr}`);
      const commands = await readFile(log, 'utf8');
      assert.equal(commands.includes('run examples/hello.sio'), entry.run, entry.label);
      if (entry.stdout) assert.ok(result.stdout.includes(entry.stdout), entry.label);
    }
    return cases.length;
  } finally { await rm(directory, { recursive: true, force: true }); }
}

async function main() {
  const args = new Set(process.argv.slice(2));
  if ([...args].some(argument => !['--self-test', '--json'].includes(argument))) throw new Error('Unknown option');
  const coverage = JSON.parse(await readRepository(coverageFile));
  const documents = new Map();
  for (const file of [...coverage.documents, ...coverage.website_pages]) documents.set(file, await readRepository(file));
  const expectedSources = new Map();
  for (const file of ['examples/hello.sio', 'examples/native/struct_basic.sio', 'tests/run-pass/generics_multi_param.sio', 'tests/run-pass/effect_superset_ok.sio']) expectedSources.set(file, (await readRepository(file)).replace(/\n$/, ''));
  const errors = coverageIssues(coverage, documents, expectedSources);
  const links = await linkIssues(documents);
  errors.push(...links.errors);
  const result = { schema: 'sounio.greek-journey.validation.v1', scope: args.has('--self-test') ? 'structural-and-mocked-preflight-only' : 'structural-only', documents: documents.size, local_link_checks: links.links, compiler_invoked: false, deployed: false, errors };
  if (args.has('--self-test')) {
    let mutations = 0;
    const mutate = (change, expected) => {
      const candidate = structuredClone(coverage);
      const changed = new Map(documents);
      change(candidate, changed);
      assert.ok(coverageIssues(candidate, changed, expectedSources).some(error => error.includes(expected)), expected);
      mutations += 1;
    };
    for (const field of ['full_specification_translated', 'entire_website_translated_and_reviewed', 'new_compiler_execution']) mutate(value => value[field] = true, 'unsupported claim');
    mutate(value => value.locale = 'en', 'identity');
    mutate(value => value.source_revision = 'invalid', 'revision');
    mutate(value => value.documents = value.documents.filter(file => file !== 'INSTALL.el.md'), 'missing journey surface');
    mutate((value, changed) => changed.delete('README.el.md'), 'missing document');
    mutate((value, changed) => changed.set('README.el.md', '# English heading'), 'Greek heading');
    const websiteEntry = 'website/src/content/docs/el/getting-started.mdx';
    mutate((value, changed) => changed.set(websiteEntry, changed.get(websiteEntry).replace(/^title: .+$/m, 'title: "English heading"')), 'Greek heading');
    mutate((value, changed) => changed.set(websiteEntry, changed.get(websiteEntry).replace(/^title: .+$/m, '')), 'Greek heading');
    const withoutBodyHeading = new Map(documents);
    withoutBodyHeading.set(websiteEntry, documents.get(websiteEntry).replace(/^# .+\n/m, ''));
    assert.deepEqual(coverageIssues(coverage, withoutBodyHeading, expectedSources), []);
    mutate((value, changed) => changed.set('README.el.md', '# Ελληνικά'), 'frontdoor destination');
    mutate((value, changed) => changed.set(value.llm_guide, changed.get(value.llm_guide).replace('llm:section syntax', 'missing-marker')), 'fifteen distinct');
    mutate((value, changed) => changed.set(value.llm_guide, changed.get(value.llm_guide).replace('Hello, Sounio', 'Altered source')), 'unbound Sounio');
    mutate(value => value.documents.push('../secret'), 'covered file set');
    for (const page of ['hello-world', 'first-program']) mutate(value => value.website_pages = value.website_pages.filter(file => !file.endsWith(`/getting-started/${page}.mdx`)), 'missing canonical onboarding surface');
    for (const block of ['```bash\nfn main() {}\n```', '  ```sh\n  struct Wrong {}\n  ```', '~~~bash\nfn main() {}\n~~~~', '````bash\n```\nfn main() {}\n````']) mutate((value, changed) => changed.set('README.el.md', `${changed.get('README.el.md')}\n${block}\n`), 'Sounio source in shell fence');
    assert.equal(shellContainsSounio('````text\n```bash\nfn main() {}\n```\n````'), false);
    result.mutation_controls = mutations;
    result.mock_preflight_controls = await mockPreflightChecks(documents.get('docs/el/getting-started.md'));
  }
  console.log(JSON.stringify(result, null, 2));
  if (errors.length) process.exitCode = 1;
}

main().catch(error => { console.error(`Greek journey validation failed: ${error.message}`); process.exitCode = 1; });
