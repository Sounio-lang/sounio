#!/usr/bin/env node
// GitHub preflight is read-only. Local leases are not a distributed lock.
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
import fs from 'node:fs';
import {createHash} from 'node:crypto';

export function run(command, args) {
  return execFileSync(command, args, {encoding: 'utf8', timeout: 60000,
    maxBuffer: 32 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe']}).trim();
}
function api(args) {
  const value = JSON.parse(run('gh', ['api', ...args]));
  if (value.errors) throw new Error(JSON.stringify(value.errors));
  return value;
}
function pages(endpoint, stop = () => false) {
  const result = [];
  for (let page = 1; ; page++) {
    const batch = api([`${endpoint}${endpoint.includes('?') ? '&' : '?'}per_page=100&page=${page}`]);
    if (!Array.isArray(batch)) throw new Error(`Incomplete response: ${endpoint}`);
    result.push(...batch);
    if (batch.length < 100 || stop(batch)) return result;
  }
}
export function normalize(file, root) {
  if (path.isAbsolute(file)) file = path.relative(root, file);
  file = path.posix.normalize(file.replaceAll('\\', '/')).replace(/\/$/, '');
  if (!file || file === '..' || file.startsWith('../') || /[\n\r\t]/.test(file))
    throw new Error(`Invalid write path: ${file}`);
  return file;
}
export function overlaps(a, b) {
  if (a === '.' || b === '.') return true;
  // Deliberately conservative: a glob reserves its literal directory prefix.
  const prefix = s => {
    const i = s.search(/[*?[{]/);
    if (i < 0) return s;
    const literal = s.slice(0, i);
    return literal.slice(0, literal.lastIndexOf('/') + 1).replace(/\/$/, '');
  };
  a = prefix(a); b = prefix(b);
  return !a || !b || a === b || a.startsWith(`${b}/`) || b.startsWith(`${a}/`);
}
function declarations(body, field) {
  return [...(body || '').matchAll(new RegExp(`^Sounio-Coord-${field}: (.+)$`, 'gm'))]
    .flatMap(m => m[1].trim().split(/\s+/));
}
export function conflicts(candidate, files, symbols) {
  const paths = [...candidate.files, ...declarations(candidate.body, 'Files').map(p => path.posix.normalize(p).replace(/\/$/, ''))];
  return [...files.filter(f => paths.some(p => overlaps(f, p))).map(f => `path:${f}`),
    ...symbols.filter(s => declarations(candidate.body, 'Symbols').includes(s) ||
      (candidate.patch || '').includes(s)).map(s => `symbol:${s}`)];
}
export function snapshot(repo, brief = false) {
  const [owner, name] = repo.split('/');
  let cursor = null;
  const prs = [];
  do {
    const query = `query($owner:String!,$name:String!,$cursor:String) {
      repository(owner:$owner,name:$name) { defaultBranchRef { name target { oid } }
        pullRequests(first:100,after:$cursor,states:OPEN,orderBy:{field:UPDATED_AT,direction:DESC}) {
          pageInfo {hasNextPage endCursor} nodes { number title url body changedFiles headRefName headRefOid
            headRepository { nameWithOwner } files(first:100) { nodes {path changeType} pageInfo {hasNextPage} } }
        }
      }
    }`;
    const args = ['graphql', '-f', `query=${query}`, '-f', `owner=${owner}`, '-f', `name=${name}`];
    if (cursor) args.push('-f', `cursor=${cursor}`);
    const data = api(args).data?.repository;
    if (!data?.defaultBranchRef || !data.pullRequests?.nodes) throw new Error('Incomplete PR inventory');
    for (const pr of data.pullRequests.nodes) {
      const files = !brief && !(pr.changedFiles >= 3000) && (pr.files.pageInfo.hasNextPage || pr.files.nodes.some(f => f.changeType === 'RENAMED')) ?
        pages(`repos/${repo}/pulls/${pr.number}/files`).flatMap(f => [f.filename, f.previous_filename].filter(Boolean)) :
        pr.files.nodes.map(f => f.path);

      prs.push({...pr, id: `pr:${pr.number}`, sha: pr.headRefOid, files, incomplete: pr.changedFiles >= 3000 || files.length >= 3000});
    }
    cursor = data.pullRequests.pageInfo.hasNextPage ? data.pullRequests.pageInfo.endCursor : null;
    if (data.pullRequests.pageInfo.hasNextPage && !cursor) throw new Error('Missing PR cursor');
    snapshot.base = data.defaultBranchRef;
  } while (cursor);
  return prs;
}
export function check(options, execute = run) {
  const root = execute('git', ['rev-parse', '--show-toplevel']);
  const remotes = execute('git', ['remote']);
  if (!remotes) { console.log('REMOTE_CHECK=LOCAL_ONLY no configured remotes'); return; }
  const origin = execute('git', ['remote', 'get-url', 'origin']);
  const match = origin.match(/^(?:https:\/\/github\.com\/|git@github\.com:|ssh:\/\/git@github\.com\/)([^/]+\/[^/]+?)(?:\.git)?$/);
  if (!match) throw new Error('origin must identify a GitHub repository; cannot establish shared visibility');
  const repo = match[1];
  if (repo.toLowerCase() !== 'sounio-lang/sounio')
    throw new Error('Shared coordination requires canonical Sounio-lang/sounio origin; a fork-only origin is not shared visibility');
  const files = options.files.map(f => normalize(f, root));

  const branch = execute('git', ['branch', '--show-current']);
  const head = execute('git', ['rev-parse', 'HEAD']);
  const receiptFile = options.receipt && path.join(options.receipt, createHash('sha256').update(root).digest('hex') + '.json');
  const identity = {root, repo, branch, head};
  if (options.reuse && files.length) {
    try {
      const receipt = JSON.parse(fs.readFileSync(receiptFile, 'utf8'));
      if (Object.entries(identity).every(([k,v]) => receipt[k] === v) &&
          Date.now() >= receipt.time && Date.now() - receipt.time < 120000 &&
          files.every(f => receipt.files.includes(f)) &&
          options.symbols.every(s => receipt.symbols.includes(s))) {
        console.log('REMOTE_CHECK=PASS receipt_age_under_120s'); return;
      }
    } catch { /* no valid receipt is never permission */ }
    throw new Error('No fresh remote receipt: run bin/sounio-coord remote-check with the full write set before scope/structured writes');
  }
  if (receiptFile && !options.brief) fs.rmSync(receiptFile, {force: true});
  const candidates = snapshot(repo, options.brief);
  console.log(`REMOTE_SNAPSHOT repo=${repo} utc=${new Date().toISOString()} open_prs=${candidates.length}`);
  if (options.brief) {
    for (const pr of candidates.slice(0, 12)) console.log(`REMOTE_PR ${pr.url} head=${pr.sha} ${JSON.stringify(pr.title)}`);
    console.log('REMOTE_BRIEF=INVENTORY_ONLY claim/scope performs the write-set check');
    return;
  }
  if (!files.length) return; // Presence-only scope is not write permission.
  const cutoff = Date.now() - 7 * 24 * 60 * 60 * 1000;
  // All open PRs, regardless of age; recent closed PRs retain discovery evidence.
  for (const pr of pages(`repos/${repo}/pulls?state=closed&sort=updated&direction=desc`, batch => batch.some(p => Date.parse(p.updated_at) < cutoff)).filter(p => Date.parse(p.updated_at) >= cutoff)) {
    console.log(`REMOTE_RECENT_CLOSED ${pr.html_url} head=${pr.head.sha} merged=${Boolean(pr.merged_at)}`);
  }
  let cursor = null;
  do {
    const query = `query($owner:String!,$name:String!,$cursor:String) { repository(owner:$owner,name:$name) {
      refs(refPrefix:"refs/heads/",first:100,after:$cursor) { pageInfo {hasNextPage endCursor}
        nodes { name target {oid ... on Commit {committedDate}}} }
    } }`;
    const [owner, name] = repo.split('/');
    const args = ['graphql', '-f', `query=${query}`, '-f', `owner=${owner}`, '-f', `name=${name}`];
    if (cursor) args.push('-f', `cursor=${cursor}`);
    const refs = api(args).data?.repository?.refs;
    if (!refs?.nodes) throw new Error('Incomplete branch inventory');
    for (const ref of refs.nodes) {
      if (!ref.target?.oid || !Number.isFinite(Date.parse(ref.target.committedDate))) throw new Error('Incomplete branch head');
      if (ref.name === snapshot.base.name || Date.parse(ref.target.committedDate) < cutoff ||
        candidates.some(p => p.headRepository?.nameWithOwner?.toLowerCase() === repo.toLowerCase() && p.headRefName === ref.name && p.sha === ref.target.oid)) continue;
      const refKey = `branch:${ref.name}@${ref.target.oid}`;
      if ((ref.name === branch && ref.target.oid === head) ||
          (options.reviewed.includes(refKey) && options.reason.trim())) {
        console.log(`REMOTE_REVIEWED ${refKey} reason=${JSON.stringify(options.reason || 'current exact head')}`);
        continue;
      }
      const comparison = api([`repos/${repo}/compare/${snapshot.base.target.oid}...${ref.target.oid}`]);
      if (!Array.isArray(comparison.files)) throw new Error(`Incomplete branch diff: ${ref.name}`);
      candidates.push({id: `branch:${ref.name}`, sha: ref.target.oid, incomplete: comparison.files.length >= 300, files: comparison.files.flatMap(f => [f.filename, f.previous_filename].filter(Boolean)),
        patch: comparison.files.map(f => f.patch || '').join('\n'), url: `https://github.com/${repo}/tree/${encodeURIComponent(ref.name)}`});
    }
    cursor = refs.pageInfo.hasNextPage ? refs.pageInfo.endCursor : null;
    if (refs.pageInfo.hasNextPage && !cursor) throw new Error('Missing branch cursor');
  } while (cursor);
  let blocked = false;
  for (const candidate of candidates) {
    // Same branch name alone is insufficient (forks and changed remote heads).
    const own = candidate.id === `branch:${branch}` ||
      (candidate.headRefName === branch && candidate.headRepository?.nameWithOwner?.toLowerCase() === repo.toLowerCase());
    if (own && candidate.sha === head) continue;
    if (options.symbols.length && candidate.number) {
      candidate.patch = run('gh', ['api', `repos/${repo}/pulls/${candidate.number}`, '-H', 'Accept: application/vnd.github.diff']);
    }
    const hits = conflicts(candidate, files, options.symbols);
    if (candidate.incomplete) hits.push('incomplete-diff:requires-full-review');
    if (!hits.length) continue;
    const key = `${candidate.id}@${candidate.sha}`;
    if (options.reviewed.includes(key) && options.reason.trim()) {
      console.log(`REMOTE_REVIEWED ${key} reason=${JSON.stringify(options.reason)} hits=${hits.join(',')}`);
    } else {
      console.error(`REMOTE_OVERLAP ${key} ${candidate.url} ${hits.join(',')}`);
      blocked = true;
    }
  }
  if (blocked) throw new Error('Remote overlap: join the carrier or obtain a scoped handoff; --reviewed ID@SHA --review-reason TEXT records an explicit reviewed exception');
  if (receiptFile) {
    fs.mkdirSync(options.receipt, {recursive: true, mode: 0o700});
    const tmp = receiptFile + '.' + process.pid;
    fs.writeFileSync(tmp, JSON.stringify({...identity, files, symbols: options.symbols, reviewed: options.reviewed, reason: options.reason, time: Date.now()}), {mode: 0o600});
    fs.renameSync(tmp, receiptFile);
  }
  console.log('REMOTE_CHECK=PASS (snapshot, not a distributed lock)');
}
export function parse(args) {
  const options = {files: [], symbols: [], reviewed: [], reason: '', brief: false};
  while (args.length) {
    const arg = args.shift();
    if (arg === '--reuse-receipt') options.reuse = true;
    else if (arg === '--receipt-dir') { options.receipt = args.shift(); if (!options.receipt) throw new Error('Missing receipt directory'); }
    else if (arg === '--brief') options.brief = true;
    else if (arg === '--files') { options.files = args; break; }
    else if (['--symbol', '--reviewed', '--review-reason'].includes(arg)) {
      const value = args.shift();
      if (!value || value.startsWith('--')) throw new Error(`Missing value for ${arg}`);
      if (arg === '--symbol') options.symbols.push(value);
      else if (arg === '--reviewed') options.reviewed.push(value);
      else options.reason = value;
    } else throw new Error(`Unknown remote-check option: ${arg}`);
  }
  if (options.reviewed.length && !options.reason.trim()) throw new Error('Reviewed overlaps require --review-reason');
  return options;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { check(parse(process.argv.slice(2))); }
  catch (error) { console.error(`REMOTE_CHECK=BLOCKED ${error.message}`); process.exitCode = 3; }
}
