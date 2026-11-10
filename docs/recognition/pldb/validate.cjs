#!/usr/bin/env node
// Data-format/source validation only; no Sounio program is compiled or executed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { createRequire } = require('node:module');
const { execFileSync } = require('node:child_process');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const git = (root, ...args) => execFileSync('git', ['-C', root, ...args]);
const root = path.resolve(__dirname, '../../..');
const pldb = process.argv[2] && fs.realpathSync(process.argv[2]);
assert(pldb, 'usage: node docs/recognition/pldb/validate.cjs /path/to/pinned-pldb');
const sources = JSON.parse(fs.readFileSync(path.join(__dirname, 'sources.json')));
assert.equal(git(pldb, 'rev-parse', 'HEAD').toString().trim(), sources.pldbCommit, 'PLDB revision drift');
const requirePldb = createRequire(path.join(pldb, 'package.json'));
assert.equal(requirePldb('scroll-cli/package.json').version, '178.2.3');
assert.equal(requirePldb('scrollsdk/package.json').version, '107.0.1');
const entry = fs.readFileSync(path.join(__dirname, 'sounio.scroll'), 'utf8');
assert(entry.startsWith('../code/conceptPage.scroll\n\n'), 'unexpected template preamble');
const rows = entry.trimEnd().split('\n').slice(2).map(line => {
  const split = line.indexOf(' ');
  assert(split > 0, 'each field needs a value');
  return [line.slice(0, split), line.slice(split + 1)];
});
for (const [field, value] of rows) {
  assert(sources.fields.some(item => item.field === field && item.value === value), `unwitnessed field ${field}`);
}
for (const item of sources.fields) {
  assert(rows.some(([field, value]) => field === item.field && value === item.value), 'stale field witness');
  const bytes = git(root, 'show', `${sources.sourceCommit}:${item.path}`);
  assert.equal(sha256(bytes), item.sha256, `source hash mismatch: ${item.path}`);
  assert(bytes.toString().includes(item.witness), `source text missing: ${item.path}`);
  assert.equal(item.url, `https://github.com/Sounio-lang/sounio/blob/${sources.sourceCommit}/${item.path}`);
}
assert(entry.includes('research language (pre-1.0); APIs may change; not recommended for production use.'));
// Check the upstream source dataset, not a separately generated JSON export.
for (const filename of fs.readdirSync(path.join(pldb, 'concepts'))) {
  if (!filename.endsWith('.scroll')) continue;
  const original = fs.readFileSync(path.join(pldb, 'concepts', filename), 'utf8');
  assert(!/^id (sounio|demetrios)$/mi.test(original), `existing id needs reconciliation: ${filename}`);
  assert(!/^name (Sounio|Demetrios)$/mi.test(original), `existing name needs reconciliation: ${filename}`);
  assert(!/^aka .*\bsounio\b/mi.test(original), `existing alias needs reconciliation: ${filename}`);
  assert(!/^githubRepo https:\/\/github\.com\/sounio-lang\/sounio\/?$/mi.test(original), `existing repository: ${filename}`);
}
const { ScrollCli } = requirePldb('scroll-cli/scroll.js');
async function errors(text) {
  const file = new ScrollCli().sfs.newFile(text, path.join(pldb, 'concepts', 'sounio.scroll'));
  await file.singlePassFuse();
  await file.scrollProgram.load();
  return file.scrollProgram.getAllErrors().map(error => error.toObject());
}
(async () => {
  assert.deepEqual(await errors(entry), [], 'upstream parser rejected the entry');
  assert((await errors(entry.replace('appeared 2025', 'appeared not-a-year'))).length, 'invalid year control was not rejected');
  assert((await errors(entry + 'notARealPldbMeasure 1\n')).length, 'unknown field control was not rejected');
  console.log(JSON.stringify({ result: 'PASS', scope: 'pinned upstream parser, source witnesses, duplicate search and two rejection controls', sourceCommit: sources.sourceCommit, pldbCommit: sources.pldbCommit, entrySha256: sha256(entry), parserVersions: { scrollCli: '178.2.3', scrollsdk: '107.0.1' }, lockSha256: sha256(fs.readFileSync(path.join(pldb, 'package-lock.json'))) }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });
