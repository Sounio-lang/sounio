// Metadata-only refusal/identity controls; never invokes Python or a compiler.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import {
  Policy,
  PolicyRow,
  SCIENCE_SCHEMA,
  CLAIM_SCHEMA,
  TSV_COLUMNS,
  sha256_file,
} from './core.mjs';
import {
  validate_policy_row,
  load_tsv_policy,
  load_toml_policy,
  policy_has_science,
  discover_policy,
  classify_path,
  validate_scientific_examples,
} from './policy.mjs';
import {
  load_claim_contract,
  claim_diagnostics,
  source_bundle_hash,
  verify_claim_evidence_bindings,
} from './claims.mjs';

function fixture(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'boundary-policy-claims-'));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  fs.mkdirSync(path.join(root, 'src'));
  fs.writeFileSync(path.join(root, 'src', 'code.sio'), '// identity fixture only\n');
  fs.writeFileSync(path.join(root, 'compiler'), 'identity-only compiler fixture\n');
  return root;
}
function row(overrides = {}) {
  return new PolicyRow({
    path: '.',
    ring: 'pl-core',
    evidence_status: 'declared',
    context_of_use: 'metadata-only',
    visibility: 'public',
    enforcement: 'strict',
    next_gate: 'identity',
    allowed_claim_classes: ['algorithmic', 'empirical', 'gum'],
    evidence_refs: ['source:src/code.sio'],
    declared_by: 'test',
    declared_at: 'fixture',
    review_state: 'draft',
    ...overrides,
  });
}
const quoted = (x) => (/[\t\r\n"]/.test(x) ? '"' + x.replaceAll('"', '""') + '"' : x);
function tsv(rows) {
  return (
    TSV_COLUMNS.join('\t') +
    '\n' +
    rows
      .map((r) =>
        TSV_COLUMNS.map((k) => quoted(Array.isArray(r[k]) ? r[k].join('|') : String(r[k]))).join(
          '\t'
        )
      )
      .join('\n') +
    '\n'
  );
}
function write_policy(root, rows = [row()]) {
  const file = path.join(root, 'science-rings.tsv');
  fs.writeFileSync(file, tsv(rows));
  return file;
}
const messages = (diagnostics) => diagnostics.map((d) => d.message);
const codes = (diagnostics) => diagnostics.map((d) => d.code);
function manifest(extra = '') {
  return `[science]\nschema = "${SCIENCE_SCHEMA}"\nring = "pl-core"\nevidence-status = "declared"\ncontext-of-use = "metadata-only"\nvisibility = "public"\nevidence-refs = ["source:src/code.sio"]\nallowed-claim-classes = ["algorithmic"]\n${extra}`;
}
function claim_fixture(root, requested = 'algorithmic') {
  const file = write_policy(root),
    source = path.join(root, 'src/code.sio'),
    compiler = path.join(root, 'compiler'),
    policy = load_tsv_policy(file, 'strict');
  const contract = {
    schema: CLAIM_SCHEMA,
    'claim-id': 'identity',
    'requested-class': requested,
    'context-of-use': 'metadata-only',
    'root-artifact': 'src/code.sio',
    evidence: [
      { type: 'source', ref: 'src/code.sio', sha256: sha256_file(source) },
      { type: 'package', ref: 'science-rings.tsv', sha256: sha256_file(file) },
      { type: 'compiler', ref: 'compiler', sha256: sha256_file(compiler) },
    ],
  };
  return {
    source,
    compiler,
    policy,
    contract,
    run: () =>
      claim_diagnostics(contract, policy.rows[0], 'src/code.sio', source, policy, compiler),
  };
}
function add_evidence(root, contract, type) {
  const ref = `evidence-${type}.txt`;
  fs.writeFileSync(path.join(root, ref), `named ${type} identity\n`);
  contract.evidence.push({ type, ref, sha256: sha256_file(path.join(root, ref)) });
}

test('quoted TSV tabs, escaped quotes, CRLF, multiline and physical comment filter', (t) => {
  const root = fixture(t),
    context = 'first\t"quoted"\r\n#filtered\r\nlast';
  const p = write_policy(root, [row({ context_of_use: context })]);
  const loaded = load_tsv_policy(p, 'strict');
  assert.equal(loaded.rows[0].context_of_use, 'first\t"quoted"\r\nlast');
  assert.deepEqual(loaded.diagnostics, []);
});
test('TSV follows non-strict CSV quote handling and ignores excess DictReader cells', (t) => {
  const root = fixture(t),
    p = write_policy(root);
  let text = fs.readFileSync(p, 'utf8');
  text = text.replace('metadata-only', '"metadata"tail').replace(/draft\n$/, '"draft');
  fs.writeFileSync(p, text);
  const loaded = load_tsv_policy(p, 'strict');
  assert.equal(loaded.rows[0].context_of_use, 'metadatatail');
  assert.equal(loaded.rows[0].review_state, 'draft');
  assert.deepEqual(loaded.diagnostics, []);
  fs.writeFileSync(p, tsv([row()]).trimEnd() + '\textra\tignored\n');
  assert.deepEqual(load_tsv_policy(p, 'strict').diagnostics, []);
});
test('TSV exact header, BOM, blank first record and missing declarations refuse', (t) => {
  const root = fixture(t),
    p = write_policy(root);
  for (const text of [
    '\uFEFF' + tsv([row()]),
    '\n' + tsv([row()]),
    TSV_COLUMNS.slice(1).join('\t') + '\n',
  ]) {
    fs.writeFileSync(p, text);
    assert(
      messages(load_tsv_policy(p, 'strict').diagnostics).some((x) =>
        x.includes('header does not match')
      )
    );
  }
  fs.writeFileSync(p, TSV_COLUMNS.join('\t') + '\n\n');
  assert(
    messages(load_tsv_policy(p, 'strict').diagnostics).includes(
      'science-rings.tsv has no declarations'
    )
  );
});
test('TSV duplicate normalized paths and short rows retain explicit diagnostics', (t) => {
  const root = fixture(t),
    p = write_policy(root, [row({ path: 'src' }), row({ path: './src' })]);
  assert(
    messages(load_tsv_policy(p, 'strict').diagnostics).includes('duplicate policy path on line 3')
  );
  fs.writeFileSync(p, TSV_COLUMNS.join('\t') + '\nsrc\tpl-core\n');
  const loaded = load_tsv_policy(p, 'strict');
  assert.equal(loaded.rows.length, 1);
  assert(messages(loaded.diagnostics).includes('missing context_of_use'));
});
test('TSV fatal UTF8 and field-size refusal preserve prior parsed rows', (t) => {
  const root = fixture(t),
    p = write_policy(root);
  fs.writeFileSync(p, Buffer.from([0xff]));
  assert(
    messages(load_tsv_policy(p, 'strict').diagnostics).some((x) =>
      x.startsWith('cannot parse science-rings.tsv:')
    )
  );
  fs.writeFileSync(
    p,
    tsv([row()]) +
      tsv([row({ context_of_use: 'x'.repeat(131073) })])
        .split('\n')
        .slice(1)
        .join('\n')
  );
  const loaded = load_tsv_policy(p, 'strict');
  assert.equal(loaded.rows.length, 1);
  assert(messages(loaded.diagnostics).some((x) => x.includes('field larger than field limit')));
});
test('policy paths reject traversal/symlink escape and nonexistent paths', (t) => {
  const root = fixture(t),
    outside = fs.mkdtempSync(path.join(os.tmpdir(), 'boundary-outside-'));
  t.after(() => fs.rmSync(outside, { recursive: true, force: true }));
  fs.symlinkSync(outside, path.join(root, 'escape'));
  assert(
    messages(validate_policy_row(row({ path: 'escape' }), root, 'strict')).includes(
      'policy path escapes its root'
    )
  );
  assert(
    messages(validate_policy_row(row({ path: 'missing' }), root, 'strict')).includes(
      'policy path does not exist'
    )
  );
  const p = write_policy(root, [row({ path: '../escape' })]);
  assert(
    messages(load_tsv_policy(p, 'strict').diagnostics).some((x) =>
      x.includes('invalid policy path')
    )
  );
});
test('ring/evidence/required fields refuse with strict/advisory severity preserved', (t) => {
  const root = fixture(t),
    invalid = row({
      ring: 'unknown',
      visibility: 'unknown',
      enforcement: 'unknown',
      next_gate: '',
    });
  const strict = validate_policy_row(invalid, root, 'strict'),
    advisory = validate_policy_row(invalid, root, 'advisory');
  assert.equal(strict.length, 4);
  assert(strict.every((x) => x.severity === 'error'));
  assert(advisory.every((x) => x.severity === 'warning'));
  assert(
    codes(validate_policy_row(row({ evidence_refs: [] }), root, 'strict')).includes('E-SRB-003')
  );
});
test('classification selects deepest physical declaration and deterministic path tie', (t) => {
  const root = fixture(t);
  fs.symlinkSync('src', path.join(root, 'alias'));
  const a = row({ path: '.' }),
    b = row({ path: 'src' }),
    c = row({ path: 'alias' }),
    policy = new Policy({ root, source: null, kind: 'test', rows: [a, b, c] });
  assert.equal(classify_path(path.join(root, 'src/code.sio'), policy), c);
  assert.equal(classify_path(path.join(root, 'compiler'), policy), a);
  assert.equal(classify_path(path.dirname(root), policy), null);
});
test('TOML quoted multiline values, aliases, legacy warning and temporal string conversion', (t) => {
  const root = fixture(t),
    p = path.join(root, 'sounio.toml');
  fs.writeFileSync(
    p,
    manifest('declared-at = 1979-05-27T07:32:00.1234Z\n[epistemic]\nscore = 1\n')
  );
  const loaded = load_toml_policy(p, 'strict');
  assert.equal(loaded.legacy_present, true);
  assert.equal(loaded.rows[0].declared_at, '1979-05-27 07:32:00.123400+00:00');
  assert.equal(loaded.rows[0].enforcement, 'advisory');
  assert.deepEqual(codes(loaded.diagnostics), ['W-SRB-LEGACY-001']);
  fs.writeFileSync(
    p,
    manifest().replace('context-of-use = "metadata-only"', 'context_of_use = """first\nsecond"""')
  );
  assert.equal(load_toml_policy(p, 'strict').rows[0].context_of_use, 'first\nsecond');
});
test('TOML invalid schema/date refuses and malformed file remains discoverable', (t) => {
  const root = fixture(t),
    p = path.join(root, 'sounio.toml');
  fs.writeFileSync(p, manifest().replace(SCIENCE_SCHEMA, 'bad'));
  assert(
    messages(load_toml_policy(p, 'strict').diagnostics).some((x) =>
      x.includes('[science].schema must be')
    )
  );
  fs.writeFileSync(p, manifest('declared-at = 2023-02-29T00:00:00Z\n'));
  assert(
    messages(load_toml_policy(p, 'strict').diagnostics).some((x) =>
      x.startsWith('cannot parse sounio.toml:')
    )
  );
  fs.writeFileSync(p, '[science');
  assert.equal(policy_has_science(p), true);
  assert.equal(discover_policy(path.join(root, 'src/code.sio'), '', 'strict').kind, 'sounio-toml');
});
test('discovery conflicts, explicit absence and legacy-only manifest are distinct', (t) => {
  const root = fixture(t),
    p = write_policy(root),
    manifest_file = path.join(root, 'sounio.toml'),
    source = path.join(root, 'src/code.sio');
  fs.writeFileSync(manifest_file, manifest());
  assert.equal(discover_policy(source, '', 'strict').kind, 'conflicting-declarations');
  fs.writeFileSync(manifest_file, '[epistemic]\nscore = 1\n');
  assert.equal(discover_policy(source, '', 'strict').source, p);
  assert.equal(discover_policy(source, path.join(root, 'absent.toml'), 'strict').kind, 'missing');
});
test('scientific examples require typed empirical evidence and safe paths', (t) => {
  const root = fixture(t),
    example = {
      name: 'declared',
      maturity: 'validated',
      'context-of-use': 'metadata-only',
      path: 'src/code.sio',
      'evidence-refs': ['source:x'],
    };
  assert(
    messages(validate_scientific_examples({ example: [example] }, root, 'strict')).some((x) =>
      x.includes('dataset,diagnostics,gate,review,split')
    )
  );
  example['evidence-refs'] = ['dataset:x', 'split:x', 'diagnostics:x', 'gate:x', 'review:x'];
  assert.deepEqual(validate_scientific_examples({ example: [example] }, root, 'strict'), []);
  example.path = '../outside';
  assert(
    messages(validate_scientific_examples({ example: [example] }, root, 'strict')).some((x) =>
      x.includes('invalid path')
    )
  );
  assert.equal(validate_scientific_examples({ example: {} }, root, 'strict')[0].code, 'E-SRB-000');
});
test('claim loader validates schema, required fields, evidence table and lowercase digest', (t) => {
  const root = fixture(t),
    p = path.join(root, 'claim.toml');
  assert.deepEqual(load_claim_contract('', 'strict'), [null, []]);
  fs.writeFileSync(p, `schema = "bad"\nevidence = [1]\n`);
  const [loaded, ds] = load_claim_contract(p, 'strict');
  assert(loaded);
  assert.equal(ds.length, 6);
  fs.writeFileSync(
    p,
    `schema = "${CLAIM_SCHEMA}"\nclaim-id = "x"\nrequested-class = "algorithmic"\ncontext-of-use = "metadata-only"\nroot-artifact = "src/code.sio"\n[[evidence]]\ntype = "source"\nref = "src/code.sio"\nsha256 = "${'A'.repeat(64)}"\n`
  );
  assert(
    messages(load_claim_contract(p, 'strict')[1]).some((x) => x.includes('lowercase SHA-256'))
  );
});
test('actual source/package/compiler bindings pass then each content mutation refuses', (t) => {
  const root = fixture(t),
    f = claim_fixture(root);
  assert.deepEqual(f.run(), []);
  for (const file of [f.source, f.policy.source, f.compiler]) {
    const old = fs.readFileSync(file);
    fs.appendFileSync(file, 'changed');
    assert(codes(f.run()).includes('E-SRB-006'));
    fs.writeFileSync(file, old);
  }
  assert.deepEqual(f.run(), []);
});
test('source normalization does not weaken exact package ref and traversal is refused', (t) => {
  const root = fixture(t),
    f = claim_fixture(root);
  f.contract.evidence[0].ref = './src/code.sio';
  assert.deepEqual(f.run(), []);
  f.contract.evidence[1].ref = './science-rings.tsv';
  assert(messages(f.run()).some((x) => x.endsWith('package:science-rings.tsv')));
  f.contract.evidence[1].ref = '../science-rings.tsv';
  assert(codes(f.run()).includes('E-SRB-006'));
});
test('empirical/GUM claims need additional named bound evidence beyond runtime', (t) => {
  const root = fixture(t),
    f = claim_fixture(root, 'empirical');
  assert(codes(f.run()).includes('E-SRB-005'));
  for (const type of [
    'model',
    'data-manifest',
    'dataset',
    'split',
    'diagnostics',
    'gate',
    'review',
  ])
    add_evidence(root, f.contract, type);
  assert.deepEqual(f.run(), []);
  f.contract['requested-class'] = 'gum';
  assert(messages(f.run()).some((x) => x.includes('method,witness')));
  for (const type of ['method', 'witness']) add_evidence(root, f.contract, type);
  assert.deepEqual(f.run(), []);
});
test('legacy scalar evidence cannot grant authority; context/root/class mismatches remain errors', (t) => {
  const root = fixture(t),
    f = claim_fixture(root);
  add_evidence(root, f.contract, 'score');
  assert(messages(f.run()).some((x) => x.includes('legacy scalar metadata')));
  f.contract.evidence.pop();
  f.contract['context-of-use'] = 'other';
  assert(messages(f.run()).includes('claim context-of-use does not match the root declaration'));
  f.contract['root-artifact'] = 'compiler';
  assert(messages(f.run()).some((x) => x.startsWith('claim root-artifact must bind')));
  f.contract['requested-class'] = 'unapproved';
  assert(codes(f.run()).includes('E-SRB-007'));
});
test('additional evidence symlink escape refuses despite a matching hash', (t) => {
  const root = fixture(t),
    f = claim_fixture(root),
    outside = fs.mkdtempSync(path.join(os.tmpdir(), 'boundary-binding-'));
  t.after(() => fs.rmSync(outside, { recursive: true, force: true }));
  fs.writeFileSync(path.join(outside, 'data'), 'identity');
  fs.symlinkSync(path.join(outside, 'data'), path.join(root, 'external'));
  f.contract.evidence.push({
    type: 'dataset',
    ref: 'external',
    sha256: sha256_file(path.join(root, 'external')),
  });
  assert(codes(f.run()).includes('E-SRB-006'));
});
test('receipt evidence verification detects stale source/compiler and invalid paths', (t) => {
  const root = fixture(t),
    f = claim_fixture(root),
    receipt = { claim_contract: { evidence_bindings: structuredClone(f.contract.evidence) } };
  verify_claim_evidence_bindings(receipt, root, f.compiler);
  fs.appendFileSync(f.source, 'changed');
  assert.throws(
    () => verify_claim_evidence_bindings(receipt, root, f.compiler),
    /claim evidence hash mismatch: source/
  );
  receipt.claim_contract.evidence_bindings = [f.contract.evidence[2]];
  assert.throws(
    () => verify_claim_evidence_bindings(receipt, root, null),
    /claim compiler evidence hash mismatch/
  );
  receipt.claim_contract.evidence_bindings = [
    { type: 'dataset', ref: '../outside', sha256: '0'.repeat(64) },
  ];
  assert.throws(
    () => verify_claim_evidence_bindings(receipt, root, f.compiler),
    /claim evidence path is invalid/
  );
});
test('bundle hashing uses Unicode code-point path order, explicit NUL/newline framing and ASCII digests', () => {
  const nodes = [
    { path: '\u{10000}', sha256: 'b' },
    { path: '\uE000', sha256: 'a' },
  ];
  const expected = crypto
    .createHash('sha256')
    .update(Buffer.from('\uE000\0a\n\u{10000}\0b\n', 'utf8'))
    .digest('hex');
  assert.equal(source_bundle_hash(nodes), expected);
  assert.equal(source_bundle_hash([...nodes].reverse()), expected);
  assert.throws(() => source_bundle_hash([{ path: 'x', sha256: '\u00ff' }]), /ASCII/);
});

test('policy fields use Python whitespace without trimming a BOM', (t) => {
  const root = fixture(t),
    p = write_policy(root, [row({ context_of_use: '\u0085metadata-only\u001c' })]);
  assert.equal(load_tsv_policy(p, 'strict').rows[0].context_of_use, 'metadata-only');
  write_policy(root, [row({ context_of_use: '\uFEFFmetadata-only' })]);
  assert.equal(load_tsv_policy(p, 'strict').rows[0].context_of_use, '\uFEFFmetadata-only');
});
test('raw absolute policy row cannot be silently rebased inside root', (t) => {
  const root = fixture(t),
    outside = fs.mkdtempSync(path.join(os.tmpdir(), 'boundary-absolute-'));
  t.after(() => fs.rmSync(outside, { recursive: true, force: true }));
  assert(
    messages(validate_policy_row(row({ path: outside }), root, 'strict')).includes(
      'policy path escapes its root'
    )
  );
  const policy = new Policy({ root, source: null, kind: 'test', rows: [row({ path: outside })] });
  assert.equal(classify_path(outside, policy), policy.rows[0]);
});
test('source bundle refuses lone surrogates instead of replacing their UTF8 identity', () => {
  assert.throws(() => source_bundle_hash([{ path: '\uD800', sha256: 'a' }]), /Unicode|surrogate/);
});

function set_science_field(text, key, value) {
  const escaped = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
    re = new RegExp('^' + escaped + ' = .*$', 'm');
  return re.test(text) ? text.replace(re, key + ' = ' + value) : text + key + ' = ' + value + '\n';
}
function claim_text(extra = '') {
  return `schema = "${CLAIM_SCHEMA}"\nclaim-id = "identity"\nrequested-class = "algorithmic"\ncontext-of-use = "metadata-only"\nroot-artifact = "src/code.sio"\n${extra}`;
}

test('TOML scalar metadata containers are schema UNKNOWN before admission', (t) => {
  const root = fixture(t),
    p = path.join(root, 'sounio.toml');
  for (const key of [
    'ring',
    'evidence-status',
    'context-of-use',
    'visibility',
    'enforcement',
    'next-gate',
    'declared-by',
    'declared-at',
    'review-state',
  ])
    for (const value of ['["x"]', '{ x = "y" }']) {
      fs.writeFileSync(p, set_science_field(manifest(), key, value));
      const loaded = load_toml_policy(p, 'strict');
      assert(
        messages(loaded.diagnostics).some((x) => x.includes('must be scalar')),
        key + ':' + value
      );
      assert.equal(loaded.valid, false);
    }
});
test('claim schema and required scalar containers refuse even when nonempty', (t) => {
  const root = fixture(t),
    p = path.join(root, 'claim.toml');
  for (const key of ['schema', 'claim-id', 'requested-class', 'context-of-use', 'root-artifact'])
    for (const value of ['["x"]', '{ x = "y" }']) {
      const text = claim_text().replace(new RegExp('^' + key + ' = .*$', 'm'), key + ' = ' + value);
      fs.writeFileSync(p, text);
      assert(
        messages(load_claim_contract(p, 'strict')[1]).some((x) => x.includes('must be scalar')),
        key + ':' + value
      );
    }
});
test('policy intended string lists reject nested containers and nonstring array members', (t) => {
  const root = fixture(t),
    p = path.join(root, 'sounio.toml');
  for (const key of ['allowed-claim-classes', 'evidence-refs'])
    for (const value of ['[["x"]]', '[{ x = "y" }]', '{ x = "y" }', '[1]']) {
      fs.writeFileSync(p, set_science_field(manifest(), key, value));
      assert(
        messages(load_toml_policy(p, 'strict').diagnostics).some((x) =>
          x.includes('must be a flat string list')
        ),
        key + ':' + value
      );
    }
});
test('different dictionary contexts cannot collapse to an admitted policy/claim pair', (t) => {
  const root = fixture(t),
    p = path.join(root, 'sounio.toml'),
    c = path.join(root, 'claim.toml');
  fs.writeFileSync(
    p,
    set_science_field(manifest(), 'context-of-use', '{ "2" = "two", "1" = "one" }')
  );
  fs.writeFileSync(
    c,
    claim_text().replace(
      'context-of-use = "metadata-only"',
      'context-of-use = { "1" = "one", "2" = "two" }'
    )
  );
  const policy = load_toml_policy(p, 'strict'),
    [, diagnostics] = load_claim_contract(c, 'strict');
  assert.equal(policy.valid, false);
  assert(codes(diagnostics).includes('E-SRB-000'));
});
test('direct claim evaluator refuses scalar containers independently of loader', (t) => {
  const root = fixture(t),
    f = claim_fixture(root);
  f.contract['context-of-use'] = { a: 'x' };
  assert(codes(f.run()).includes('E-SRB-000'));
});
test('valid scalar metadata coercions and flat string-list syntax remain accepted', (t) => {
  const root = fixture(t),
    p = path.join(root, 'sounio.toml'),
    c = path.join(root, 'claim.toml');
  fs.writeFileSync(p, set_science_field(manifest(), 'context-of-use', '1.0'));
  const loaded = load_toml_policy(p, 'strict');
  assert.equal(loaded.valid, true);
  assert.equal(loaded.rows[0].context_of_use, '1.0');
  fs.writeFileSync(
    c,
    claim_text().replace('context-of-use = "metadata-only"', 'context-of-use = 1.0')
  );
  assert.deepEqual(load_claim_contract(c, 'strict')[1], []);
  fs.writeFileSync(
    p,
    set_science_field(manifest(), 'allowed-claim-classes', '["gum", "algorithmic", "gum"]')
  );
  assert.deepEqual(load_toml_policy(p, 'strict').rows[0].allowed_claim_classes, [
    'algorithmic',
    'gum',
  ]);
});
test('scientific example evidence-reference lists also reject nested containers', (t) => {
  const root = fixture(t),
    example = {
      name: 'x',
      maturity: 'prototype',
      path: 'src/code.sio',
      'context-of-use': 'metadata-only',
      'evidence-refs': [['source:x']],
    };
  assert(
    messages(validate_scientific_examples({ example: [example] }, root, 'strict')).some((x) =>
      x.includes('must be a flat string list')
    )
  );
});
