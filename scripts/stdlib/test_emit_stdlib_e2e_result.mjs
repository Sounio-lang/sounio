#!/usr/bin/env node
// Metadata-only contract tests; no compiler, scientific calculation or Python.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdtempSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { decodeDetail, makeDocument, stringifyDocument, timestampUtc } from './emit_stdlib_e2e_result.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const helper = join(root, 'scripts/stdlib/emit_stdlib_e2e_result.mjs');
const context = { root_dir: '/répo', souc_bin: '/repo/bin/souc', stdlib_path: '/repo/stdlib', filter: 'test_"quoted"' };
const counts = { pass: '1', fail: '1', skip: '2' };
const stamp = '2026-10-25T20:00:00.123456Z';
const encode = text => Buffer.from(text).toString('base64');
const record = (...fields) => fields.join('\t');
const parse = document => JSON.parse(stringifyDocument(document));
let checks = 0;
function test(name, action) { action(); checks++; console.log('PASS ' + name); }

test('base64 canonical and empty', () => {
  for (const [raw, expected] of [['', ''], ['Zg==', 'f'], ['Zm8=', 'fo'], ['Zm9v', 'foo']]) assert.equal(decodeDetail(raw), expected);
});
test('base64 non-strict legacy alphabet and padding', () => {
  assert.equal(decodeDetail('Y!W-J_j'), 'abc');
  assert.equal(decodeDetail('=Zg==ignored'), 'f');
  assert.equal(decodeDetail('Zm9v=YmFy'), 'foobar');
  for (const invalid of ['A', 'Zg', 'Zm8', '☃']) assert.throws(() => decodeDetail(invalid));
});
test('UTF8 replacement, BOM and embedded detail characters', () => {
  assert.equal(decodeDetail(Buffer.from([0x66, 0x80, 0xe2, 0x82]).toString('base64')), 'f\uFFFD\uFFFD');
  assert.equal(decodeDetail(Buffer.from([0xef, 0xbb, 0xbf, 0x41]).toString('base64')), '\uFEFFA');
  const text = 'detail "quote" \\ slash\n\té🧪';
  assert.equal(decodeDetail(encode(text)), text);
});
const rows = [
  record('pass', 'tests/a.sio', 'run-pass', '', '', '', '0', ''),
  record('fail', 'tests/b.sio', 'check', 'check-failed', '', '', '124', encode('bad\n\t"é"')),
  record('skip', 'tests/c.sio', 'ignore', 'blocked', 'équipe', 'fix imported module', '0', ''),
  record('skip', 'tests/d.sio', 'no-annotation', 'no-annotation', '', '', '0', ''),
];
test('eight fields, pass/fail/skip, ignored and context', () => {
  const obj = parse(makeDocument(rows.join('\n'), context, counts, stamp));
  assert.equal(obj.schema, 'sounio.stdlib.e2e.result.v1');
  assert.equal(obj.command, 'bash scripts/stdlib/run_stdlib_e2e.sh');
  assert.deepEqual(obj.context, context);
  assert.deepEqual(obj.totals, { pass: 1, fail: 1, skip: 2, total: 4 });
  assert.equal(obj.results.length, 4);
  assert.deepEqual(obj.failures, [obj.results[1]]);
  assert.deepEqual(obj.ignored, [{ test_path: 'tests/c.sio', reason: 'blocked', owner: 'équipe', unblock_condition: 'fix imported module' }]);
  assert.equal(obj.generated_at_utc, stamp);
  assert.equal(obj.results[1].exit_code, 124);
  assert.equal(obj.results[1].detail_excerpt, 'bad\n\t"é"');
});
test('malformed field-count rows remain ignored', () => {
  const bad = ['too\tfew', record('pass', 'extra', 'run', '', '', '', '0', '', 'ninth')];
  const obj = parse(makeDocument(['', ...bad, ...rows, ''].join('\r\n'), context, counts, stamp));
  assert.equal(obj.results.length, 4);
  assert.deepEqual(obj.totals, { pass: 1, fail: 1, skip: 2, total: 4 });
  assert.throws(() => makeDocument(record('fail', 'x', 'check', '', '', '', 'invalid', ''), context, counts));
});
test('Python splitlines boundaries and exact decimal integer tokens', () => {
  const obj = makeDocument(rows.join('\u2028'), context, counts, stamp);
  assert.equal(obj.results.length, 4);
  const wide = makeDocument(record('pass', 'wide', 'run-pass', '', '', '', '9223372036854775807', ''), context,
    { pass: '9223372036854775807', fail: '0', skip: '0' }, stamp);
  const text = stringifyDocument(wide);
  assert.ok(text.includes('"pass": 9223372036854775807'));
  assert.ok(text.includes('"exit_code": 9223372036854775807'));
});
test('UTC timestamp fraction compatibility', () => {
  assert.equal(timestampUtc(new Date('2026-10-25T20:01:02.123Z')), '2026-10-25T20:01:02.123000Z');
  assert.equal(timestampUtc(new Date('2026-10-25T20:01:02.000Z')), '2026-10-25T20:01:02Z');
});
test('committed historical JSON exact text roundtrip is a fixture only', () => {
  const historicalPath = process.env.STDLIB_E2E_JSON_FIXTURE || join(root, 'artifacts/stdlib/stdlib_e2e_result.v1.json');
  const text = readFileSync(historicalPath, 'utf8');
  assert.equal(createHash('sha256').update(text).digest('hex'), 'c02d2ac60c5d2a6ec8b8161e7f1d244ccf47412db3e9943c283493cdac551fe2');
  const historical = JSON.parse(text);
  const tsv = historical.results.map(r => record(r.status, r.test_path, r.mode, r.reason, r.owner, r.unblock_condition, String(r.exit_code), encode(r.detail_excerpt))).join('\n');
  const result = makeDocument(tsv, historical.context, historical.totals, historical.generated_at_utc);
  assert.equal(stringifyDocument(result) + '\n', text);
  console.log('HISTORICAL_FIXTURE_ONLY results=' + historical.results.length + ' timestamp=' + historical.generated_at_utc);
});
test('runner changes only the emit_json function body', () => {
  const path = 'scripts/stdlib/run_stdlib_e2e.sh';
  const after = readFileSync(join(root, path), 'utf8');
  const strip = text => {
    const start = text.indexOf('\nemit_json() {');
    const end = text.indexOf('\necho "=== Stdlib E2E Test Suite ==="', start);
    assert.ok(start > 0 && end > start);
    return text.slice(0, start) + text.slice(end);
  };
  // Pinned from the unchanged prefix/suffix of main8eca, so an extracted
  // source packet needs no .git directory or subprocess for this assertion.
  assert.equal(createHash('sha256').update(strip(after)).digest('hex'), 'ea9386ef7929b8177d7978aa7f97fd0821b0cfdaa8587fac2be27bdf16485ef9');
  assert.ok(!after.includes('python3'));
});
test('CLI emits fresh JSON and rejects invalid UTF8 without output', () => {
  const temporary = mkdtempSync(join(tmpdir(), 'sounio-e2e-json-node-'));
  try {
    const input = join(temporary, 'input.tsv'), output = join(temporary, 'fresh.json');
    writeFileSync(input, rows.join('\n'));
    const args = [helper, input, output, '1', '1', '2', context.root_dir, context.souc_bin, context.stdlib_path, context.filter];
    const proc = spawnSync(process.execPath, args, { encoding: 'utf8' });
    assert.equal(proc.status, 0, proc.stderr);
    const doc = JSON.parse(readFileSync(output, 'utf8'));
    assert.equal(doc.results.length, 4);
    assert.match(doc.generated_at_utc, /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{6})?Z$/);
    const invalidInput = join(temporary, 'invalid.tsv'), absentOutput = join(temporary, 'absent.json');
    writeFileSync(invalidInput, Buffer.from([0xff]));
    const bad = spawnSync(process.execPath, [helper, invalidInput, absentOutput, ...args.slice(3)], { encoding: 'utf8' });
    assert.equal(bad.status, 1);
    assert.ok(bad.stderr.includes('STDLIB_E2E_JSON_ERROR:'));
    assert.equal(existsSync(absentOutput), false);
  } finally { rmSync(temporary, { recursive: true, force: true }); }
});
console.log('STDLIB_E2E_JSON_NODE_SELFTEST_PASS cases=' + checks);
