// Operational hash, filesystem and metadata-parser controls; no numerical oracle.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  canonical_json,
  sha256_bytes,
  sha256_file,
  receipt_identity,
  write_json_atomic,
  read_json,
  read_utf8,
  py_strip,
  py_str,
  python_sort,
  resolve_path,
  within_root,
  safe_relative_text,
  load_toml,
} from './core.mjs';
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'sounio-boundary-core-')),
  cases = [];
const test = (name, body) => {
  body();
  cases.push(name);
};
const file = (name, data) => {
  const p = path.join(work, name);
  fs.writeFileSync(p, data);
  return p;
};
test('SHA256 published abc vector', () =>
  assert.equal(
    sha256_bytes(Buffer.from('abc')),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
  ));
test('canonical ASCII DEL and Unicode codepoint key order', () =>
  assert.equal(
    canonical_json({ '\u{10000}': 2, '\ue000': 1, del: '\u007f' }).toString(),
    ' {"del":"\\u007f","\\ue000":1,"\\ud800\\udc00":2}'.slice(1)
  ));
test('canonical nested sorted dictionary', () =>
  assert.equal(
    canonical_json({ z: [{ b: true, a: null }], a: 'x\n' }).toString(),
    '{"a":"x\\n","z":[{"a":null,"b":true}]}'
  ));
test('canonical unsafe number refuses', () =>
  assert.throws(() => canonical_json({ capacity: 9007199254740992 })));
test('canonical nonintegral receipt number refuses', () =>
  assert.throws(() => canonical_json({ capacity: 1.5 })));
test('Python Unicode strip keeps BOM', () => {
  assert.equal(py_strip('\u0085\u001c x\u001f'), 'x');
  assert.equal(py_strip('\ufeffx\ufeff'), '\ufeffx\ufeff');
});
test('UTF8 invalid byte refuses', () =>
  assert.throws(() => read_utf8(file('bad-utf8', Buffer.from([255])))));
test('UTF8 BOM remains visible', () =>
  assert.equal(read_utf8(file('bom', Buffer.from([239, 187, 191, 65]))), '\ufeffA'));
test('safe path normalizes empty components', () =>
  assert.equal(safe_relative_text('a//./b/'), 'a/b'));
test('safe path refuses parent and absolute', () => {
  assert.equal(safe_relative_text('a/../b'), null);
  assert.equal(safe_relative_text('/etc/passwd'), null);
});
fs.mkdirSync(path.join(work, 'real/sub'), { recursive: true });
fs.symlinkSync('real/sub', path.join(work, 'link'));
test('physical symlink then parent resolution', () =>
  assert.equal(resolve_path(path.join(work, 'link') + '/../leaf'), path.join(work, 'real/leaf')));
test('physical missing suffix remains resolvable', () =>
  assert.equal(
    resolve_path(path.join(work, 'link') + '/missing/../new'),
    path.join(work, 'real/sub/new')
  ));
test('root containment rejects sibling prefix', () =>
  assert.equal(within_root(work + '-other/x', work), false));
fs.symlinkSync('loop2', path.join(work, 'loop1'));
fs.symlinkSync('loop1', path.join(work, 'loop2'));
test('symlink loops refuse', () => assert.throws(() => resolve_path(path.join(work, 'loop1'))));
test('Python tuple sorting', () =>
  assert.deepEqual(
    python_sort([
      ['b', 'a'],
      ['a', 'z'],
      ['a', 'a'],
    ]),
    [
      ['a', 'a'],
      ['a', 'z'],
      ['b', 'a'],
    ]
  ));
test('TOML multiline and exact large integer', () => {
  const d = load_toml(file('valid.toml', 'name="""first\nsecond"""\nbig=9007199254740993\n'));
  assert.equal(d.name, 'first\nsecond');
  assert.equal(d.big, 9007199254740993n);
});
test('TOML repeated key refuses', () =>
  assert.throws(() => load_toml(file('duplicate.toml', 'a=1\na=2\n'))));
test('TOML invalid calendar refuses', () =>
  assert.throws(() => load_toml(file('calendar.toml', 'date=2023-02-30\n'))));
test('TOML zero year refuses', () =>
  assert.throws(() => load_toml(file('zero-year.toml', 'date=0000-01-01\n'))));
test('TOML escaped surrogate refuses', () =>
  assert.throws(() => load_toml(file('surrogate.toml', 's="\\uD800"\n'))));
test('TOML invalid UTF8 refuses', () =>
  assert.throws(() => load_toml(file('utf8.toml', Buffer.from([0x73, 0x3d, 0x22, 0xff, 0x22])))));
test('TOML temporal fraction and authored offset retained', () => {
  const d = load_toml(file('time.toml', 'date=1979-05-27T07:32:00.123456789-08:00\n'));
  assert.equal(py_str(d.date), '1979-05-27 07:32:00.123456-08:00');
});
test('TOML UTC and local time coercions', () => {
  const d = load_toml(file('utc.toml', 'a=1979-05-27T07:32:00Z\nb=07:32:00.000001\n'));
  assert.equal(py_str(d.a), '1979-05-27 07:32:00+00:00');
  assert.equal(py_str(d.b), '07:32:00.000001');
});
test('TOML prototype keys do not pollute host', () => {
  load_toml(file('proto.toml', '__proto__={intrusion=true}\n'));
  assert.equal({}.intrusion, undefined);
});
test('TOML negative zero timezone normalizes', () => {
  const d = load_toml(file('negative-zone.toml', 'date=1979-05-27T07:32:00-00:00\n'));
  assert.equal(py_str(d.date), '1979-05-27 07:32:00+00:00');
});
test('TOML float metadata string formatting', () => {
  const d = load_toml(file('float-metadata.toml', 'small=0.00001\nlarge=1e16\nfixed=1.0\n'));
  assert.equal(py_str(d.small), '1e-05');
  assert.equal(py_str(d.large), '1e+16');
  assert.equal(py_str(d.fixed), '1.0');
});
test('explicit raw pin does not materialize fallback', () => {
  const wrapper = fs.readFileSync(
      fileURLToPath(new URL('../../bin/madaros', import.meta.url)),
      'utf8'
    ),
    start = wrapper.indexOf('_resolve_modular_elf() {'),
    end = wrapper.indexOf('\nif ! RAW_MADAROS=', start);
  assert(start >= 0 && end > start);
  const raw = file('raw-pin', Buffer.from([127, 69, 76, 70])),
    fallback = file('raw-fallback', Buffer.from([127, 69, 76, 70]));
  fs.chmodSync(raw, 0o755);
  fs.chmodSync(fallback, 0o755);
  const marker = path.join(work, 'unexpected-fallback');
  const script =
    'ROOT_DIR=$1; export MADAROS_RAW_BIN=$2; fallback=$3\nsounio_madaros_host_raw(){ touch \"$ROOT_DIR/unexpected-fallback\"; printf \"%s\\n\" \"$fallback\"; }\n' +
    wrapper.slice(start, end) +
    '\n_resolve_modular_elf\n';
  const r = spawnSync('bash', ['-c', script, '_', work, raw, fallback], { encoding: 'utf8' });
  assert.equal(r.status, 0);
  assert.equal(r.stdout.trim(), raw);
  assert.equal(fs.existsSync(marker), false);
});
const receipt = {
  z: '\u{10000}',
  hashes: { source_bundle_sha256: '0'.repeat(64) },
  graph: { capacity: 1 },
};
const out = path.join(work, 'receipt.json');
test('atomic ASCII receipt identity and permissions', () => {
  write_json_atomic(out, receipt);
  const bytes = fs.readFileSync(out);
  assert(bytes.every((b) => b <= 127));
  assert.equal(fs.statSync(out).mode & 0o777, 0o600);
  const r = read_json(out, { ascii: true });
  assert.equal(r.hashes.receipt_identity_sha256, receipt_identity(r));
  assert(bytes.toString().startsWith('{\n  "graph":'));
  assert(!fs.readdirSync(work).some((x) => x.startsWith('.receipt.json.')));
});
test('failed replacement preserves prior receipt', () => {
  const before = sha256_file(out);
  assert.throws(() => write_json_atomic(out, { capacity: 1.5 }));
  assert.equal(sha256_file(out), before);
});
const cli = fileURLToPath(new URL('./manifest_value.mjs', import.meta.url));
const manifest = path.join(work, 'sounio.toml');
const query = (key) => spawnSync(process.execPath, [cli, manifest, key], { encoding: 'utf8' });
test('project manifest name and first bin', () => {
  fs.writeFileSync(manifest, '[package]\nname="sample"\n[[bin]]\npath="app/main.sio"\n');
  assert.equal(query('package.name').stdout, 'sample\n');
  assert.equal(query('entry').stdout, 'app/main.sio\n');
});
test('project manifest defaults and project-before-lib', () => {
  fs.writeFileSync(manifest, '[project]\nentry="custom.sio"\n[lib]\npath="lib.sio"\n');
  assert.equal(query('package.name').stdout, 'sounio-project\n');
  assert.equal(query('entry').stdout, 'custom.sio\n');
});
test('manifest malformed TOML refuses', () => {
  fs.writeFileSync(manifest, '[project\n');
  assert.equal(query('entry').status, 3);
});
test('manifest scalar lib refuses even with empty bin marker', () => {
  fs.writeFileSync(manifest, 'lib=42\n[[bin]]\nname="without-path"\n');
  assert.equal(query('entry').status, 3);
});
test('container diagnostic repr quotes and control escapes', () => {
  assert.equal(py_str(["it's"]), '["it\'s"]');
  assert.equal(py_str(['\u0001']), "['\\x01']");
});
fs.writeFileSync(
  path.join(work, 'RESULTS.json'),
  JSON.stringify({ scope: 'operational metadata only', passed: cases.length, cases }, null, 2) +
    '\n'
);
console.log(`SCIENCE_BOUNDARY_CORE_PASS ${cases.length} receipts=${work}`);
