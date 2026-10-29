#!/usr/bin/env node
/**
 * IndexNow ping for https://www.souniolang.org.
 *
 * After a production deploy, compare the live sitemap with the previous
 * run's snapshot and POST added, lastmod-changed, and removed URLs to
 * https://api.indexnow.org/indexnow. The key is the public file
 * website/public/<32 hex>.txt. A non-200/202 answer is logged. The snapshot
 * is written only after every batch is accepted.
 *
 *   node website/scripts/indexnow-ping.mjs --dry-run
 *   node website/scripts/indexnow-ping.mjs --self-test
 *   node website/scripts/indexnow-ping.mjs --ci
 */
import { execFile } from 'node:child_process';
import { appendFile, chmod, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const execFileAsync = promisify(execFile);

const websiteDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = path.dirname(websiteDir);
const publicDir = path.join(websiteDir, 'public');
const snapshotPath = path.join(websiteDir, 'indexnow-snapshot', 'sitemap-0.xml');

export const HOST = 'www.souniolang.org';
export const SITEMAP_URL = 'https://www.souniolang.org/sitemap-0.xml';
export const ENDPOINT = 'https://api.indexnow.org/indexnow';
export const MAX_URLS_PER_REQUEST = 10_000;
export const RETRY_DELAY_MS = 60_000;
const ARTIFACT_NAME = 'indexnow-sitemap';
const KEY_FILE_RE = /^[0-9a-f]{32}\.txt$/;
const REPO_RE = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;

function decodeXml(value) {
  return value
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&apos;', "'");
}

export function parseSitemap(xml) {
  const urls = new Map();
  const blocks = xml.matchAll(/<url\b[^>]*>([\s\S]*?)<\/url>/g);
  for (const block of blocks) {
    const loc = block[1].match(/<loc>([^<]*)<\/loc>/);
    if (!loc) continue;
    const lastmod = block[1].match(/<lastmod>([^<]*)<\/lastmod>/);
    urls.set(decodeXml(loc[1].trim()), lastmod ? lastmod[1].trim() : '');
  }
  return urls;
}

export function keepWww(urls) {
  const kept = new Map();
  const dropped = [];
  for (const [url, lastmod] of urls) {
    let parsed;
    try {
      parsed = new URL(url);
    } catch {
      dropped.push(url);
      continue;
    }
    if (parsed.protocol === 'https:' && parsed.hostname === HOST) kept.set(url, lastmod);
    else dropped.push(url);
  }
  return { urls: kept, dropped };
}

export function diffSitemap(previous, current) {
  if (previous == null) {
    return {
      urlList: [...current.keys()],
      added: current.size,
      changed: 0,
      removed: 0,
      unchanged: 0,
      baseline: false,
    };
  }
  const urlList = [];
  let added = 0;
  let changed = 0;
  let unchanged = 0;
  let removed = 0;
  for (const [url, lastmod] of current) {
    if (!previous.has(url)) {
      added += 1;
      urlList.push(url);
    } else if (previous.get(url) !== lastmod) {
      changed += 1;
      urlList.push(url);
    } else {
      unchanged += 1;
    }
  }
  for (const url of previous.keys()) {
    if (!current.has(url)) {
      removed += 1;
      urlList.push(url);
    }
  }
  return { urlList, added, changed, removed, unchanged, baseline: true };
}

export function chunk(list, size) {
  if (size < 1) throw new Error('batch size must be positive');
  const batches = [];
  for (let index = 0; index < list.length; index += size) batches.push(list.slice(index, index + size));
  return batches;
}

export function pickArtifact(artifacts, currentRunId) {
  const rows = (artifacts ?? [])
    .filter((artifact) => {
      if (!artifact || artifact.expired === true || artifact.name !== ARTIFACT_NAME) return false;
      const runId = artifact.workflow_run?.id;
      return String(runId ?? '') !== String(currentRunId ?? '');
    })
    .sort((left, right) => Date.parse(right.created_at ?? 0) - Date.parse(left.created_at ?? 0));
  return rows[0] ?? null;
}

export async function loadKey(dir = publicDir) {
  const names = (await readdir(dir)).filter((name) => KEY_FILE_RE.test(name)).sort();
  if (names.length !== 1) {
    throw new Error(`expected one IndexNow key file in ${dir}, found ${names.length}`);
  }
  const fileName = names[0];
  const key = await readFile(path.join(dir, fileName), 'utf8');
  const expected = fileName.slice(0, -'.txt'.length);
  if (key !== expected) throw new Error('IndexNow key file must contain only the key');
  return {
    key,
    fileName,
    keyLocation: `https://${HOST}/${fileName}`,
  };
}

async function markSaved() {
  if (!process.env.GITHUB_OUTPUT) return;
  await appendFile(process.env.GITHUB_OUTPUT, 'saved=true\n');
}

export async function submitUrlList({
  urlList,
  key,
  keyLocation,
  host = HOST,
  endpoint = ENDPOINT,
  fetchImpl = globalThis.fetch,
  sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms)),
  retryDelayMs = RETRY_DELAY_MS,
  maxPerRequest = MAX_URLS_PER_REQUEST,
  log = console.log,
}) {
  if (process.env.INDEXNOW_FORBID_POST === '1') throw new Error('POST forbidden');
  const batches = chunk(urlList, maxPerRequest);
  const statuses = [];
  if (batches.length === 0) {
    log('indexnow: no URL changes');
    return { ok: true, statuses, submitted: 0 };
  }
  for (const batch of batches) {
    if (batch.length > MAX_URLS_PER_REQUEST) throw new Error('urlList exceeds 10000');
    const body = JSON.stringify({ host, key, keyLocation, urlList: batch });
    let attempt = 0;
    while (true) {
      attempt += 1;
      let response;
      try {
        response = await fetchImpl(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json; charset=utf-8' },
          body,
        });
      } catch (error) {
        log(`indexnow: POST api.indexnow.org failed: ${error.message}`);
        return { ok: false, statuses, submitted: 0, error: error.message };
      }
      const status = response.status;
      if (status === 200 || status === 202) {
        log(`indexnow: POST api.indexnow.org → ${status} (${batch.length} URLs)`);
        statuses.push(status);
        break;
      }
      let detail = '';
      try {
        detail = String(await response.text()).slice(0, 500);
      } catch {
        detail = '';
      }
      log(`indexnow: POST api.indexnow.org → ${status} (${batch.length} URLs)${detail ? ` ${detail}` : ''}`);
      if (status === 429 && attempt === 1) {
        log('indexnow: HTTP 429; retrying once in 60s');
        await sleep(retryDelayMs);
        continue;
      }
      return { ok: false, statuses: [...statuses, status], submitted: 0, error: `HTTP ${status}` };
    }
  }
  return { ok: true, statuses, submitted: urlList.length };
}

export async function runPing({
  sitemapXml,
  previousXml = null,
  key,
  keyLocation,
  snapshotOut = null,
  dryRun = false,
  fetchImpl,
  sleep,
  retryDelayMs,
  maxPerRequest,
  log = console.log,
}) {
  const current = keepWww(parseSitemap(sitemapXml));
  const previous = previousXml == null ? null : keepWww(parseSitemap(previousXml)).urls;
  if (current.dropped.length) log(`indexnow: dropped ${current.dropped.length} URL(s) not on https://${HOST}`);
  const diff = diffSitemap(previous, current.urls);
  if (!diff.baseline) log(`indexnow: no previous snapshot; submitting all ${diff.urlList.length} URLs`);
  else log(`indexnow: added=${diff.added} changed=${diff.changed} removed=${diff.removed} unchanged=${diff.unchanged}`);
  if (dryRun) {
    log('indexnow: dry-run; no POST');
    return { ok: true, dryRun: true, ...diff, saved: false };
  }
  const submitted = await submitUrlList({
    urlList: diff.urlList,
    key,
    keyLocation,
    fetchImpl,
    sleep,
    retryDelayMs,
    maxPerRequest,
    log,
  });
  if (!submitted.ok || !snapshotOut) return { ...submitted, ...diff, saved: false };
  await mkdir(path.dirname(snapshotOut), { recursive: true });
  await writeFile(snapshotOut, sitemapXml);
  return { ...submitted, ...diff, saved: true };
}

const EXTRACT_PY = `
import sys, zipfile
from pathlib import Path
zip_path, dest = sys.argv[1], Path(sys.argv[2])
dest.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    names = [n for n in zf.namelist() if Path(n).name == 'sitemap-0.xml' and '..' not in Path(n).parts]
    if len(names) != 1:
        sys.exit(2)
    target = dest / 'sitemap-0.xml'
    with zf.open(names[0]) as src, target.open('wb') as out:
        out.write(src.read())
`;

export async function extractSitemapZip(zipPath, destDir, execImpl = execFileAsync) {
  await execImpl('python3', ['-c', EXTRACT_PY, zipPath, destDir]);
  return readFile(path.join(destDir, 'sitemap-0.xml'), 'utf8');
}

function safeError(error) {
  return String(error?.message ?? error).replace(/Bearer\s+\S+/gi, 'Bearer [redacted]');
}

async function curlApi({ url, token, output, execImpl }) {
  const dir = await mkdtemp(path.join(tmpdir(), 'indexnow-curl-'));
  const config = path.join(dir, 'curl.cfg');
  await writeFile(config, [
    `header = "Authorization: Bearer ${token}"`,
    'header = "Accept: application/vnd.github+json"',
    'header = "X-GitHub-Api-Version: 2022-11-28"',
    'fail',
    'silent',
    'show-error',
    'location',
    'max-time = 60',
    output ? `output = "${output}"` : '',
    `url = "${url}"`,
    '',
  ].join('\n'));
  await chmod(config, 0o600);
  try {
    return await execImpl('curl', ['--config', config], {
      encoding: output ? 'buffer' : 'utf8',
      maxBuffer: 8 * 1024 * 1024,
    });
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

export async function restoreSnapshot({
  repository,
  token,
  runId,
  destFile,
  execImpl = execFileAsync,
  log = console.log,
}) {
  if (!repository || !token) {
    log('indexnow: no GitHub token; submitting all URLs');
    return null;
  }
  if (!REPO_RE.test(repository)) {
    log('indexnow: refusing unexpected GITHUB_REPOSITORY; submitting all URLs');
    return null;
  }
  let listed;
  try {
    const { stdout } = await curlApi({
      url: `https://api.github.com/repos/${repository}/actions/artifacts?name=${ARTIFACT_NAME}&per_page=100`,
      token,
      execImpl,
    });
    listed = JSON.parse(stdout);
  } catch (error) {
    log(`indexnow: snapshot list failed (${safeError(error)}); submitting all URLs`);
    return null;
  }
  const chosen = pickArtifact(listed.artifacts, runId);
  if (!chosen || !Number.isInteger(chosen.id)) {
    log('indexnow: no previous snapshot; submitting all URLs');
    return null;
  }
  const dir = await mkdtemp(path.join(tmpdir(), 'indexnow-'));
  const zipPath = path.join(dir, 'snapshot.zip');
  try {
    await curlApi({
      url: `https://api.github.com/repos/${repository}/actions/artifacts/${chosen.id}/zip`,
      token,
      output: zipPath,
      execImpl,
    });
    const xml = await extractSitemapZip(zipPath, dir, execImpl);
    await mkdir(path.dirname(destFile), { recursive: true });
    await writeFile(destFile, xml);
    log(`indexnow: restored snapshot artifact ${chosen.id} (${xml.length} bytes)`);
    return xml;
  } catch (error) {
    log(`indexnow: snapshot download failed (${safeError(error)}); submitting all URLs`);
    return null;
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

function applyRoutes(pathname, routes) {
  for (const route of routes) {
    const match = new RegExp(route.src).exec(pathname);
    if (!match) continue;
    const locationTemplate = route.headers?.Location;
    if (!locationTemplate) return null;
    let location = locationTemplate;
    for (let index = match.length - 1; index >= 1; index -= 1) {
      location = location.split(`$${index}`).join(match[index] ?? '');
    }
    return { status: route.status ?? 308, location };
  }
  return null;
}

function urlset(entries) {
  const body = entries.map(([loc, lastmod]) => {
    const when = lastmod ? `<lastmod>${lastmod}</lastmod>` : '';
    return `<url><loc>${loc}</loc>${when}</url>`;
  }).join('');
  return `<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">${body}</urlset>`;
}

async function selfTest() {
  const logs = [];
  const log = (line) => logs.push(line);
  const same = Array.from({ length: 18 }, (_, index) => [`https://${HOST}/same-${index}/`, '2026-10-01']);
  const previousXml = urlset([
    ...same,
    [`https://${HOST}/changed/`, '2026-10-01'],
    [`https://${HOST}/removed/`, '2026-10-01'],
  ]);
  const currentXml = urlset([
    ...same,
    [`https://${HOST}/changed/`, '2026-10-02'],
    [`https://${HOST}/added/`, '2026-10-02'],
    ['https://souniolang.org/apex/', '2026-10-02'],
    ['https://example.com/elsewhere/', '2026-10-02'],
  ]);
  const diff = await runPing({
    sitemapXml: currentXml,
    previousXml,
    key: 'a'.repeat(32),
    keyLocation: `https://${HOST}/${'a'.repeat(32)}.txt`,
    dryRun: true,
    fetchImpl: () => { throw new Error('dry-run must not fetch'); },
    log,
  });
  assert.equal(diff.urlList.length, 3);
  assert.equal(diff.added, 1);
  assert.equal(diff.changed, 1);
  assert.equal(diff.removed, 1);
  assert.equal(diff.unchanged, 18);
  assert.deepEqual(new Set(diff.urlList), new Set([
    `https://${HOST}/changed/`,
    `https://${HOST}/added/`,
    `https://${HOST}/removed/`,
  ]));
  assert.ok(logs.some((line) => line.startsWith('indexnow: dropped 2')));
  assert.ok(logs.some((line) => line === 'indexnow: dry-run; no POST'));

  const bare = parseSitemap(`<urlset><url><loc>https://${HOST}/</loc></url><url><loc>https://${HOST}/new/</loc></url></urlset>`);
  const barePrev = parseSitemap(`<urlset><url><loc>https://${HOST}/</loc></url></urlset>`);
  const bareDiff = diffSitemap(barePrev, bare);
  assert.deepEqual(bareDiff.urlList, [`https://${HOST}/new/`]);
  assert.equal(bareDiff.unchanged, 1);

  const decoded = parseSitemap(`<url><loc>https://${HOST}/a&amp;b/</loc></url>`);
  assert.ok(decoded.has(`https://${HOST}/a&b/`));

  const noBaseline = diffSitemap(null, bare);
  assert.equal(noBaseline.baseline, false);
  assert.equal(noBaseline.urlList.length, 2);

  const dir = await mkdtemp(path.join(tmpdir(), 'indexnow-self-'));
  try {
    const failedOut = path.join(dir, 'failed.xml');
    const failed = await runPing({
      sitemapXml: currentXml,
      previousXml,
      key: 'b'.repeat(32),
      keyLocation: `https://${HOST}/${'b'.repeat(32)}.txt`,
      snapshotOut: failedOut,
      fetchImpl: async () => ({ status: 500, text: async () => 'down' }),
      log,
    });
    assert.equal(failed.ok, false);
    assert.equal(failed.saved, false);
    await assert.rejects(readFile(failedOut, 'utf8'));
    assert.ok(logs.some((line) => line.includes('POST api.indexnow.org → 500')));

    const networkOut = path.join(dir, 'network.xml');
    const network = await runPing({
      sitemapXml: currentXml,
      previousXml,
      key: 'b'.repeat(32),
      keyLocation: `https://${HOST}/${'b'.repeat(32)}.txt`,
      snapshotOut: networkOut,
      fetchImpl: async () => { throw new Error('socket hang up'); },
      log,
    });
    assert.equal(network.ok, false);
    await assert.rejects(readFile(networkOut, 'utf8'));

    const delays = [];
    let posts = 0;
    const retried = await runPing({
      sitemapXml: currentXml,
      previousXml,
      key: 'c'.repeat(32),
      keyLocation: `https://${HOST}/${'c'.repeat(32)}.txt`,
      snapshotOut: path.join(dir, 'retry.xml'),
      sleep: async (ms) => { delays.push(ms); },
      fetchImpl: async () => {
        posts += 1;
        return { status: posts === 1 ? 429 : 200, text: async () => '' };
      },
      log,
    });
    assert.equal(retried.ok, true);
    assert.equal(retried.saved, true);
    assert.deepEqual(delays, [RETRY_DELAY_MS]);
    assert.equal(await readFile(path.join(dir, 'retry.xml'), 'utf8'), currentXml);

    posts = 0;
    const stillLimited = await runPing({
      sitemapXml: currentXml,
      previousXml,
      key: 'c'.repeat(32),
      keyLocation: `https://${HOST}/${'c'.repeat(32)}.txt`,
      snapshotOut: path.join(dir, 'limited.xml'),
      sleep: async () => { delays.push('second'); },
      fetchImpl: async () => {
        posts += 1;
        return { status: 429, text: async () => 'slow down' };
      },
      log,
    });
    assert.equal(stillLimited.ok, false);
    assert.equal(posts, 2);
    await assert.rejects(readFile(path.join(dir, 'limited.xml'), 'utf8'));

    const accepted = [];
    const many = Array.from({ length: 10_001 }, (_, index) => [`https://${HOST}/n-${index}/`, '']);
    const batched = await runPing({
      sitemapXml: urlset(many),
      previousXml: null,
      key: 'd'.repeat(32),
      keyLocation: `https://${HOST}/${'d'.repeat(32)}.txt`,
      fetchImpl: async (_url, init) => {
        const payload = JSON.parse(init.body);
        accepted.push(payload.urlList.length);
        assert.ok(payload.urlList.length <= MAX_URLS_PER_REQUEST);
        assert.equal(payload.host, HOST);
        assert.equal(payload.keyLocation, `https://${HOST}/${'d'.repeat(32)}.txt`);
        return { status: accepted.length === 1 ? 200 : 202, text: async () => '' };
      },
      log,
    });
    assert.equal(batched.ok, true);
    assert.deepEqual(accepted, [10_000, 1]);
    assert.ok(logs.some((line) => line.includes('→ 200 (10000 URLs)')));
    assert.ok(logs.some((line) => line.includes('→ 202 (1 URLs)')));

    const keyInfo = await loadKey(publicDir);
    assert.match(keyInfo.key, /^[0-9a-f]{32}$/);
    assert.equal(keyInfo.fileName, `${keyInfo.key}.txt`);
    assert.equal(keyInfo.keyLocation, `https://${HOST}/${keyInfo.fileName}`);
    const keyBytes = await readFile(path.join(publicDir, keyInfo.fileName));
    assert.equal(keyBytes.length, 32);
    assert.equal(keyBytes.toString('utf8'), keyInfo.key);

    const vercel = JSON.parse(await readFile(path.join(websiteDir, 'vercel.json'), 'utf8'));
    const keyPath = `/${keyInfo.fileName}`;
    assert.equal(applyRoutes(keyPath, vercel.routes), null);
    assert.deepEqual(applyRoutes(`${keyPath}/`, vercel.routes), { status: 308, location: keyPath });
    assert.equal(applyRoutes('/robots.txt', vercel.routes), null);

    const workflow = await readFile(path.join(repoRoot, '.github/workflows/vercel-website.yml'), 'utf8');
    const pingAt = workflow.indexOf('name: Submit changed URLs to IndexNow');
    const deployAt = workflow.indexOf('name: Deploy production');
    const saveAt = workflow.indexOf('name: Save IndexNow sitemap snapshot');
    assert.ok(deployAt !== -1 && pingAt > deployAt && saveAt > pingAt);
    assert.match(workflow, /permissions:[\s\S]*contents: read[\s\S]*actions: write/);
    assert.doesNotMatch(workflow, /contents: write/);
    const pingStep = workflow.slice(pingAt, saveAt);
    assert.match(pingStep, /steps\.deploy_production\.outcome == 'success'/);
    assert.match(pingStep, /inputs\.target == 'production'/);
    assert.match(pingStep, /github\.event_name == 'push'/);
    assert.match(pingStep, /continue-on-error: true/);
    assert.match(pingStep, /node website\/scripts\/indexnow-ping\.mjs --ci \|\| true/);
    const saveStep = workflow.slice(saveAt);
    assert.match(saveStep, /continue-on-error: true/);
    assert.match(saveStep, /uses: actions\/upload-artifact@v4/);
    assert.match(saveStep, /name: indexnow-sitemap/);
    assert.match(saveStep, /steps\.indexnow\.outputs\.saved == 'true'/);
    assert.match(saveStep, /steps\.deploy_production\.outcome == 'success'/);

    const chosen = pickArtifact([
      { id: 1, name: ARTIFACT_NAME, expired: false, created_at: '2026-10-01T00:00:00Z', workflow_run: { id: 10 } },
      { id: 2, name: ARTIFACT_NAME, expired: true, created_at: '2026-10-03T00:00:00Z', workflow_run: { id: 11 } },
      { id: 3, name: ARTIFACT_NAME, expired: false, created_at: '2026-10-02T00:00:00Z', workflow_run: { id: 99 } },
      { id: 4, name: 'other', expired: false, created_at: '2026-10-04T00:00:00Z', workflow_run: { id: 12 } },
    ], 99);
    assert.equal(chosen.id, 1);

    const zipDir = path.join(dir, 'zip');
    await mkdir(zipDir);
    const zipPath = path.join(zipDir, 'artifact.zip');
    await execFileAsync('python3', ['-c', `
import zipfile, sys
with zipfile.ZipFile(sys.argv[1], 'w') as zf:
    zf.writestr('website/indexnow-snapshot/sitemap-0.xml', '<urlset><url><loc>https://${HOST}/</loc></url></urlset>')
`, zipPath]);
    const extracted = await extractSitemapZip(zipPath, path.join(zipDir, 'out'));
    assert.match(extracted, new RegExp(`https://${HOST}/`));

    const token = 'ghp_secretvalue';
    let sawTokenInArgv = false;
    let configHadToken = false;
    await restoreSnapshot({
      repository: 'Sounio-lang/sounio',
      token,
      runId: '1',
      destFile: path.join(dir, 'prev.xml'),
      log,
      execImpl: async (_cmd, args) => {
        const rendered = args.join(' ');
        if (rendered.includes(token)) sawTokenInArgv = true;
        const config = args[args.indexOf('--config') + 1];
        const text = await readFile(config, 'utf8');
        if (text.includes(`Bearer ${token}`)) configHadToken = true;
        return { stdout: JSON.stringify({ artifacts: [] }) };
      },
    });
    assert.equal(sawTokenInArgv, false);
    assert.equal(configHadToken, true);
    const leaked = [];
    await restoreSnapshot({
      repository: 'Sounio-lang/sounio',
      token,
      runId: '1',
      destFile: path.join(dir, 'prev.xml'),
      log: (line) => leaked.push(line),
      execImpl: async () => {
        throw new Error('Command failed: curl --header Authorization: Bearer ghp_secretvalue');
      },
    });
    assert.ok(leaked.some((line) => line.includes('Bearer [redacted]')));
    assert.ok(leaked.every((line) => !line.includes('ghp_secretvalue')));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
  console.log('indexnow: self-test ok');
}

async function fetchText(url) {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new Error(`GET ${url} → ${response.status}`);
  return response.text();
}

async function main() {
  const args = new Set(process.argv.slice(2));
  if (args.has('--self-test')) {
    await selfTest();
    return 0;
  }
  const dryRun = args.has('--dry-run');
  const ci = args.has('--ci');
  if (dryRun === ci) {
    console.log('usage: indexnow-ping.mjs --dry-run | --self-test | --ci');
    return 2;
  }
  const { key, keyLocation } = await loadKey();
  console.log(`indexnow: host ${HOST}`);
  console.log(`indexnow: keyLocation ${keyLocation}`);
  let previousXml = null;
  if (ci) {
    previousXml = await restoreSnapshot({
      repository: process.env.GITHUB_REPOSITORY ?? '',
      token: process.env.GH_TOKEN || process.env.GITHUB_TOKEN || '',
      runId: process.env.GITHUB_RUN_ID ?? '',
      destFile: path.join(tmpdir(), 'indexnow-previous.xml'),
    });
  }
  const sitemapXml = await fetchText(SITEMAP_URL);
  const result = await runPing({
    sitemapXml,
    previousXml,
    key,
    keyLocation,
    snapshotOut: ci ? snapshotPath : null,
    dryRun,
  });
  if (result.saved) await markSaved();
  if (!result.ok) return 1;
  if (dryRun) {
    const offHost = result.urlList.filter((url) => {
      try {
        return new URL(url).hostname !== HOST;
      } catch {
        return true;
      }
    });
    console.log(`indexnow: dry-run would submit ${result.urlList.length} URL(s); off-host ${offHost.length}`);
  }
  return 0;
}

const invoked = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (invoked) {
  try {
    const code = await main();
    process.exit(code);
  } catch (error) {
    console.error(error.stack || error.message);
    process.exit(1);
  }
}
