// CLI and artifact identity controls only; fixture compiler/ELF bytes never execute.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { TSV_COLUMNS, CLAIM_SCHEMA, sha256_file, write_json_atomic, read_json } from './core.mjs';
const cli = fileURLToPath(new URL('./attestor.mjs', import.meta.url));
const work = fs.mkdtempSync(path.join(os.tmpdir(), 'sounio-boundary-attestor-'));
let serial = 0;
const cases = [];
function fixture() {
  const root = path.join(work, 'fixture-' + ++serial);
  fs.mkdirSync(root);
  const source = path.join(root, 'main.sio'),
    compiler = path.join(root, 'compiler'),
    elf = path.join(root, 'program.elf'),
    report = path.join(root, 'raw.tsv'),
    pre = path.join(root, 'pre.json'),
    final = path.join(root, 'final.json');
  fs.writeFileSync(source, '// Existing boundary identity fixture; not compiled by this suite.\n');
  fs.writeFileSync(compiler, 'compiler identity fixture\n');
  fs.writeFileSync(elf, Buffer.from([127, 69, 76, 70, 1, 2, 3]));
  const row = [
    '.',
    'pl-core',
    'declared',
    'metadata-only',
    'public',
    'strict',
    'identity',
    'algorithmic|empirical|gum',
    'source:main.sio',
    'fixture',
    'fixture',
    'draft',
  ];
  fs.writeFileSync(
    path.join(root, 'science-rings.tsv'),
    TSV_COLUMNS.join('\t') + '\n' + row.join('\t') + '\n'
  );
  fs.writeFileSync(
    report,
    'SOUNIO_BOUNDARY_CLOSURE_V1\nstatus\tcomplete\ncapacity\t256\nsaturated\tfalse\nparse_failed\tfalse\nnode\tmain.sio\n'
  );
  return { root, source, compiler, elf, report, pre, final };
}
function run(args, expected) {
  const r = spawnSync(process.execPath, [cli, ...args], { encoding: 'utf8' });
  const id = String(++serial);
  fs.writeFileSync(
    path.join(work, id + '.command.json'),
    JSON.stringify([process.execPath, cli, ...args]) + '\n'
  );
  fs.writeFileSync(path.join(work, id + '.stdout'), r.stdout);
  fs.writeFileSync(path.join(work, id + '.stderr'), r.stderr);
  fs.writeFileSync(path.join(work, id + '.status'), String(r.status) + '\n');
  assert.equal(r.status, expected, r.stderr);
  return r;
}
function evaluate(f, expected = 0, extra = []) {
  return run(
    [
      'evaluate',
      '--source',
      f.source,
      '--mode',
      'strict',
      '--receipt',
      f.pre,
      '--compiler',
      f.compiler,
      '--engine-identity',
      'metadata-fixture',
      '--closure-report',
      f.report,
      ...extra,
    ],
    expected
  );
}
function finish(f, expected = 0, extra = []) {
  return run(
    [
      'finalize',
      '--preflight-receipt',
      f.pre,
      '--source',
      f.source,
      '--compiler',
      f.compiler,
      '--elf',
      f.elf,
      '--artifact-label',
      'build/program.elf',
      '--receipt',
      f.final,
      ...extra,
    ],
    expected
  );
}
function verify(f, expected = 0, extra = []) {
  return run(
    [
      'verify',
      '--receipt',
      f.final,
      '--root',
      f.root,
      '--compiler',
      f.compiler,
      '--elf',
      f.elf,
      ...extra,
    ],
    expected
  );
}
const test = (name, body) => {
  body();
  cases.push(name);
};
test('strict raw closure evaluates OK and emits identity-only receipt', () => {
  const f = fixture();
  evaluate(f);
  const r = read_json(f.pre);
  assert.equal(r.verdict, 'OK');
  assert.equal(r.engine.boundary_collector, 'madaros-raw-ast-v1');
  assert.equal(r.assurance_level, 'identity-only');
  assert(r.limitations.includes('does_not_assert_scientific_truth'));
});
test('strict receipt finalizes and verifies source compiler policy ELF', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  verify(f);
  const r = read_json(f.final);
  assert.equal(r.hashes.elf_sha256, sha256_file(f.elf));
  assert.equal(r.graph.root_artifact, 'main.sio');
});
test('evaluation is deterministic independent of output filename', () => {
  const f = fixture();
  evaluate(f);
  evaluate(f, 0, ['--receipt', f.final]);
  assert.equal(sha256_file(f.pre), sha256_file(f.final));
});
test('host syntax closure cannot grant authoritative strict OK', () => {
  const f = fixture();
  evaluate(f, 21, ['--closure-report', '']);
  assert.equal(read_json(f.pre).verdict, 'UNKNOWN');
  finish(f, 2);
  assert.equal(fs.existsSync(f.final), false);
});
test('missing raw report remains UNKNOWN with receipt', () => {
  const f = fixture();
  evaluate(f, 21, ['--closure-report', path.join(f.root, 'missing.tsv')]);
  assert.equal(read_json(f.pre).verdict, 'UNKNOWN');
});
test('contradictory complete saturated report is UNKNOWN', () => {
  const f = fixture();
  fs.appendFileSync(f.report, 'saturated\ttrue\n');
  evaluate(f, 21);
  finish(f, 2);
});
test('invalid policy header remains UNKNOWN', () => {
  const f = fixture();
  fs.writeFileSync(path.join(f.root, 'science-rings.tsv'), 'bad\theader\n');
  evaluate(f, 21);
});
test('forbidden authoritative ring edge rejects', () => {
  const f = fixture();
  fs.writeFileSync(path.join(f.root, 'dependency.sio'), '// dependency identity\n');
  const data = fs.readFileSync(path.join(f.root, 'science-rings.tsv'), 'utf8');
  const row = [
    'dependency.sio',
    'research',
    'declared',
    'metadata-only',
    'public',
    'strict',
    'identity',
    'algorithmic',
    'source:dependency.sio',
    'fixture',
    'fixture',
    'draft',
  ];
  fs.writeFileSync(path.join(f.root, 'science-rings.tsv'), data + row.join('\t') + '\n');
  fs.appendFileSync(f.report, 'node\tdependency.sio\nedge\tmain.sio\tdependency.sio\n');
  evaluate(f, 20);
  assert(read_json(f.pre).diagnostics.some((x) => x.code === 'E-SRB-001'));
  finish(f, 2);
});
test('public to protected authoritative edge rejects', () => {
  const f = fixture();
  fs.writeFileSync(path.join(f.root, 'protected.sio'), '// protected identity\n');
  const row = [
    'protected.sio',
    'pl-core',
    'declared',
    'metadata-only',
    'protected',
    'strict',
    'identity',
    'algorithmic',
    'source:protected.sio',
    'fixture',
    'fixture',
    'draft',
  ];
  fs.appendFileSync(path.join(f.root, 'science-rings.tsv'), row.join('\t') + '\n');
  fs.appendFileSync(f.report, 'node\tprotected.sio\nedge\tmain.sio\tprotected.sio\n');
  evaluate(f, 20);
  assert(read_json(f.pre).diagnostics.some((x) => x.code === 'E-SRB-002'));
});
test('off mode remains explicit UNKNOWN and cannot become OK', () => {
  const f = fixture();
  evaluate(f, 0, ['--mode', 'off']);
  assert.equal(read_json(f.pre).mode, 'off');
  assert.equal(read_json(f.pre).verdict, 'UNKNOWN');
});
test('malformed claim evidence table retains UNKNOWN receipt', () => {
  const f = fixture(),
    claim = path.join(f.root, 'claim.toml');
  fs.writeFileSync(
    claim,
    `schema="${CLAIM_SCHEMA}"\nclaim-id="x"\nrequested-class="algorithmic"\ncontext-of-use="metadata-only"\nroot-artifact="main.sio"\n[evidence]\ntype="source"\n`
  );
  evaluate(f, 21, ['--claim-contract', claim]);
  assert.equal(read_json(f.pre).verdict, 'UNKNOWN');
  assert(read_json(f.pre).diagnostics.some((x) => x.message.includes('array of tables')));
});
test('source mutation refuses finalization without output', () => {
  const f = fixture();
  evaluate(f);
  fs.appendFileSync(f.source, '// mutation\n');
  finish(f, 2);
  assert.equal(fs.existsSync(f.final), false);
});
test('different source cannot reuse OK preflight', () => {
  const f = fixture();
  evaluate(f);
  const other = path.join(f.root, 'other.sio');
  fs.writeFileSync(other, '// other identity\n');
  finish(f, 2, ['--source', other]);
  assert.equal(fs.existsSync(f.final), false);
});
test('policy mutation refuses finalization', () => {
  const f = fixture();
  evaluate(f);
  fs.appendFileSync(path.join(f.root, 'science-rings.tsv'), '# changed\n');
  finish(f, 2);
});
test('compiler mutation refuses finalization', () => {
  const f = fixture();
  evaluate(f);
  fs.appendFileSync(f.compiler, 'changed\n');
  finish(f, 2);
});
test('recorded compiler cannot be omitted at finalization', () => {
  const f = fixture();
  evaluate(f);
  finish(f, 2, ['--compiler', '']);
});
test('empty artifact refuses finalization', () => {
  const f = fixture();
  evaluate(f);
  fs.writeFileSync(f.elf, '');
  finish(f, 2);
});
test('ELF mutation refuses verification', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  fs.appendFileSync(f.elf, 'mutation');
  verify(f, 1);
});
test('source mutation refuses verification', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  fs.appendFileSync(f.source, 'mutation');
  verify(f, 1);
});
test('compiler omission refuses verification', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  verify(f, 1, ['--compiler', '']);
});
test('ELF omission refuses verification', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  verify(f, 1, ['--elf', '']);
});
test('rehashed extra authority field refuses schema', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  const r = read_json(f.final);
  r.clinical_authority = true;
  write_json_atomic(f.final, r);
  verify(f, 1);
});
test('rehashed malformed graph array refuses', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  const r = read_json(f.final);
  r.graph.nodes = {};
  write_json_atomic(f.final, r);
  verify(f, 1);
});
test('rehashed traversal graph path refuses', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  const r = read_json(f.final);
  r.graph.nodes[0].path = '../outside.sio';
  write_json_atomic(f.final, r);
  verify(f, 1);
});
test('rehashed invalid digest refuses', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  const r = read_json(f.final);
  r.hashes.elf_sha256 = 'g'.repeat(64);
  write_json_atomic(f.final, r);
  verify(f, 1);
});
test('modified identity without rehash refuses', () => {
  const f = fixture();
  evaluate(f);
  finish(f);
  const r = read_json(f.final);
  r.mode = 'off';
  fs.writeFileSync(f.final, JSON.stringify(r));
  verify(f, 1);
});
test('invalid CLI option refuses', () => run(['evaluate', '--unknown', 'x'], 2));
test('missing required CLI source refuses', () => run(['evaluate'], 2));
fs.writeFileSync(
  path.join(work, 'RESULTS.json'),
  JSON.stringify(
    {
      scope: 'operational metadata only; no compiler/Python execution',
      passed: cases.length,
      cases,
    },
    null,
    2
  ) + '\n'
);
console.log(`SCIENCE_BOUNDARY_ATTESTOR_PASS ${cases.length} receipts=${work}`);
