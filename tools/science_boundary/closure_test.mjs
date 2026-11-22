#!/usr/bin/env node
// Operational source/graph/path fixtures only. No compiler or Python execution.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import crypto from 'node:crypto';
import { ImportSpec, resolve_path } from './core.mjs';
import {
  tokenize_import_surface,
  parse_imports,
  candidate_import_paths,
  resolve_import,
  collect_closure,
  closure_report_path,
  load_raw_ast_closure_report,
} from './closure.mjs';

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'sounio-boundary-closure-'));
const root = path.join(work, 'root');
fs.mkdirSync(root);
const results = [];
function write(rel, content = '// OPERATIONAL PARSER FIXTURE ONLY\n') {
  const file = path.join(root, rel);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
  return file;
}
function test(name, fn) {
  try {
    fn();
    results.push({ name, status: 'PASS' });
    console.log('PASS ' + name);
  } catch (error) {
    results.push({ name, status: 'FAIL', reason: error.message });
    throw error;
  }
}
function env(name, value, fn) {
  const old = process.env[name];
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
  try {
    return fn();
  } finally {
    if (old === undefined) delete process.env[name];
    else process.env[name] = old;
  }
}
const specs = (file) =>
  parse_imports(file)[0].map((spec) => [spec.segments, spec.line, spec.display]);
const main = write('main.sio');
const dep = write('dep.sio');
const raw = (records) => 'SOUNIO_BOUNDARY_CLOSURE_V1\n' + records.join('\n') + '\n';
function report(records, name = 'report.tsv') {
  const file = path.join(work, name);
  fs.writeFileSync(file, raw(records));
  return load_raw_ast_closure_report(file, main, root)[0];
}
function failures(closure, text) {
  assert.ok(
    closure.parse_failures.some(([, message]) => message.includes(text)),
    text
  );
}
const complete = [
  'status\tcomplete',
  'capacity\t256',
  'saturated\tfalse',
  'parse_failed\tfalse',
  'node\tmain.sio',
];

try {
  test('token tuples and exact import punctuation', () => {
    assert.deepEqual(tokenize_import_surface('use a::b;\n'), [
      [
        ['ident', 'use', 1],
        ['ident', 'a', 1],
        ['symbol', '::', 1],
        ['ident', 'b', 1],
        ['symbol', ';', 1],
        ['newline', '\n', 1],
      ],
      [],
    ]);
  });
  test('top-level pub/use/import and grouped surface', () => {
    const file = write(
      'surface.sio',
      'pub use a::b;\nimport c::{x, y}\nfn f() { use hidden::body }\nuse d::*\n'
    );
    assert.deepEqual(specs(file), [
      [['a', 'b'], 1, 'a::b'],
      [['c'], 2, 'c'],
      [['d'], 4, 'd'],
    ]);
  });
  test('nested comments and quoted text cannot manufacture imports', () => {
    const file = write(
      'comments.sio',
      '/* use fake /* import inner */ tail */\n"use ignored::text"; \'use ignored::char\'; // import another\n# use ignored::hash\nuse real::dep\n'
    );
    assert.deepEqual(specs(file), [[['real', 'dep'], 4, 'real::dep']]);
  });
  test('escaped quotes preserve the syntax boundary', () => {
    const file = write('escaped.sio', '"a \\" use fake::dep"; use real::dep;\n');
    assert.deepEqual(specs(file), [[['real', 'dep'], 1, 'real::dep']]);
  });
  test('Unicode letters astral identifiers and numeric continuation', () => {
    const file = write('unicode.sio', 'use α::漢字\nuse \u{10400}::a²\n');
    assert.deepEqual(specs(file), [
      [['α', '漢字'], 1, 'α::漢字'],
      [['\u{10400}', 'a²'], 2, '\u{10400}::a²'],
    ]);
  });
  test('unterminated comments and strings retain explicit errors', () => {
    assert.deepEqual(tokenize_import_surface('\n/* never')[1], [
      'unterminated block comment starting on line 2',
    ]);
    assert.deepEqual(tokenize_import_surface('\n"never')[1], [
      'unterminated string starting on line 2',
    ]);
  });
  test('invalid import retains line and no invented segment', () => {
    assert.deepEqual(parse_imports(write('invalid-import.sio', '\nuse ;\n'))[1], [
      'invalid import on line 2',
    ]);
  });
  test('read_text universal CRLF and bare CR handling', () => {
    assert.deepEqual(specs(write('cr.sio', 'use first\ruse second\r\nuse third\n')), [
      [['first'], 1, 'first'],
      [['second'], 2, 'second'],
      [['third'], 3, 'third'],
    ]);
  });
  test('invalid UTF8 is an incomplete parse rather than replacement text', () => {
    const file = write('invalid-utf8.sio', Buffer.from([0xff]));
    assert.match(parse_imports(file)[1][0], /^cannot read source:/);
  });
  test('missing source is an incomplete parse', () => {
    assert.match(parse_imports(path.join(root, 'absent.sio'))[1][0], /^cannot read source:/);
  });
  test('import search order is caller ancestors environment stdlib package', () => {
    const file = write('caller/main.sio');
    const external = path.join(work, 'external');
    env('SOUNIO_STDLIB_PATH', external, () => {
      assert.deepEqual(
        [...candidate_import_paths(file, new ImportSpec(['foo'], 1), root)],
        [
          path.join(root, 'caller/foo.sio'),
          path.join(root, 'caller/foo/mod.sio'),
          path.join(root, 'foo.sio'),
          path.join(root, 'foo/mod.sio'),
          path.join(external, 'foo.sio'),
          path.join(external, 'foo/mod.sio'),
          path.join(root, 'stdlib/foo.sio'),
          path.join(root, 'stdlib/foo/mod.sio'),
          path.join(root, 'packages/foo/src/lib.sio'),
        ]
      );
    });
  });
  test('first existing physical candidate wins', () => {
    const file = write('priority/main.sio');
    write('pick.sio');
    const nearest = write('priority/pick.sio');
    assert.equal(resolve_import(file, new ImportSpec(['pick'], 1), root), nearest);
  });
  test('module mod.sio and package name normalization', () => {
    const module = write('module_only/mod.sio');
    assert.equal(resolve_import(main, new ImportSpec(['module_only'], 1), root), module);
    const pkg = write('packages/pkg-name/src/lib.sio');
    const sub = write('packages/pkg-name/src/sub.sio');
    assert.equal(resolve_import(main, new ImportSpec(['pkg_name'], 1), root), pkg);
    assert.equal(resolve_import(main, new ImportSpec(['pkg_name', 'sub'], 1), root), sub);
  });
  test('finite import cycle closes and repeated edges deduplicate', () => {
    const a = write('cycle/a.sio', 'use b\nuse b\n');
    const b = write('cycle/b.sio', 'use a\n');
    const closure = collect_closure(a, root);
    assert.deepEqual(closure.nodes, [a, b]);
    assert.deepEqual(closure.edges, [
      [a, b],
      [b, a],
    ]);
    assert.deepEqual(closure.unresolved, []);
    assert.equal(closure.saturated, false);
    assert.equal(closure.collector, 'sounio-host-syntax-v1');
  });
  test('unresolved import includes source and original line', () => {
    const file = write('unresolved.sio', '\nuse nowhere::module\n');
    assert.deepEqual(collect_closure(file, root).unresolved, [[file, 'nowhere::module@2']]);
  });
  test('symlink dependency escape stays outside closure', () => {
    const outside = path.join(work, 'outside.sio');
    fs.writeFileSync(outside, '// OUTSIDE OPERATIONAL FIXTURE\n');
    fs.symlinkSync(outside, path.join(root, 'escape.sio'));
    const file = write('escape-main.sio', 'use escape\n');
    const closure = collect_closure(file, root);
    assert.deepEqual(closure.nodes, [file]);
    assert.deepEqual(closure.unresolved, [[file, 'root-escape:escape@1']]);
    assert.equal(closure_report_path('escape.sio', root), null);
  });
  test('environment dependency outside policy root is unresolved', () => {
    const external = path.join(work, 'env-stdlib');
    fs.mkdirSync(external);
    fs.writeFileSync(path.join(external, 'external_dep.sio'), '// OUTSIDE\n');
    const file = write('env-main.sio', 'use external_dep\n');
    const closure = env('SOUNIO_STDLIB_PATH', external, () => collect_closure(file, root));
    assert.deepEqual(closure.unresolved, [[file, 'root-escape:external_dep@1']]);
  });
  test('capacity saturation retains seen edge but refuses complete closure', () => {
    const file = write('cap.sio', 'use dep\n');
    const closure = env('SOUNIO_SCIENCE_BOUNDARY_MAX_NODES', '1', () =>
      collect_closure(file, root)
    );
    assert.equal(closure.capacity, 1);
    assert.equal(closure.saturated, true);
    assert.deepEqual(closure.nodes, [file]);
    assert.deepEqual(closure.edges, [[file, dep]]);
  });
  for (const [text, expected] of [
    ['0', 1],
    ['-2', 1],
    ['4097', 4096],
    ['+2', 2],
    [' 3 ', 3],
    ['1_0', 10],
    ['१२', 12],
    ['𝟙𝟚', 12],
    ['bogus', 256],
    ['', 256],
    ['\ufeff2', 256],
    ['\u00852\u0085', 2],
    ['\x1c2', 256],
    ['2\x1f', 256],
    ['9'.repeat(4301), 256],
    ['999999999999999999999999999999999999999999', 4096],
  ]) {
    test(
      'capacity environment ' +
        (text.length > 64 ? 'decimal digits=' + text.length : JSON.stringify(text)),
      () => {
        assert.equal(
          env('SOUNIO_SCIENCE_BOUNDARY_MAX_NODES', text, () => collect_closure(main, root))
            .capacity,
          expected
        );
      }
    );
  }
  test('raw complete closure binds report bytes and authoritative collector', () => {
    const records = [...complete, 'node\tdep.sio', 'edge\tmain.sio\tdep.sio'];
    const file = path.join(work, 'full.tsv');
    fs.writeFileSync(file, 'Madaros banner\n' + raw(records));
    const [closure, diagnostics] = load_raw_ast_closure_report(file, main, root);
    assert.deepEqual(closure.nodes, [dep, main]);
    assert.deepEqual(closure.edges, [[main, dep]]);
    assert.deepEqual(closure.parse_failures, []);
    assert.deepEqual(diagnostics, []);
    assert.equal(closure.collector, 'madaros-raw-ast-v1');
    assert.equal(
      closure.report_sha256,
      crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')
    );
  });
  test('raw missing header remains invalid', () => {
    const file = path.join(work, 'missing-header.tsv');
    fs.writeFileSync(file, 'status\tcomplete\n');
    failures(
      load_raw_ast_closure_report(file, main, root)[0],
      'invalid raw AST closure report header'
    );
  });
  test('UTF8 BOM is retained rather than making a bad raw header valid', () => {
    const file = path.join(work, 'bom.tsv');
    fs.writeFileSync(file, '\ufeff' + raw(complete));
    failures(
      load_raw_ast_closure_report(file, main, root)[0],
      'invalid raw AST closure report header'
    );
  });
  test('raw invalid UTF8 remains parse failure', () => {
    const file = path.join(work, 'invalid-report.tsv');
    fs.writeFileSync(file, Buffer.from([0xff]));
    failures(
      load_raw_ast_closure_report(file, main, root)[0],
      'cannot read raw AST closure report:'
    );
  });
  test('missing raw report hash error propagates for caller UNKNOWN fallback', () => {
    assert.throws(() => load_raw_ast_closure_report(path.join(work, 'missing.tsv'), main, root));
  });
  test('raw root artifact is mandatory', () => {
    failures(
      report(complete.filter((line) => !line.startsWith('node'))),
      'does not contain the root source'
    );
  });
  test('raw edges require declared nodes', () => {
    failures(report([...complete, 'edge\tmain.sio\tdep.sio']), 'references an undeclared node');
  });
  test('raw complete cannot coexist with unresolved imports', () => {
    const closure = report([...complete, 'unresolved\tmain.sio\tmissing::dep@1']);
    assert.deepEqual(closure.unresolved, [[main, 'missing::dep@1']]);
    failures(closure, 'claims complete with unresolved or saturated state');
  });
  test('raw complete cannot coexist with saturation', () => {
    failures(
      report(complete.map((line) => (line === 'saturated\tfalse' ? 'saturated\ttrue' : line))),
      'claims complete with unresolved or saturated state'
    );
  });
  test('raw parser failure is retained', () => {
    failures(
      report(
        complete.map((line) => (line === 'parse_failed\tfalse' ? 'parse_failed\ttrue' : line))
      ),
      'raw AST parser reported failure'
    );
  });
  test('raw incomplete status cannot become a complete closure', () => {
    failures(
      report(complete.map((line) => (line === 'status\tcomplete' ? 'status\tincomplete' : line))),
      'raw AST closure is incomplete'
    );
  });
  test('raw invalid or absent status cannot become complete', () => {
    failures(
      report(complete.filter((line) => !line.startsWith('status'))),
      'status is missing or invalid'
    );
    failures(
      report(complete.map((line) => (line === 'status\tcomplete' ? 'status\tinvented' : line))),
      'status is missing or invalid'
    );
  });
  test('raw unsafe node edge and unresolved caller refuse root escape', () => {
    failures(report([...complete, 'node\t../outside.sio']), 'invalid raw AST node');
    failures(report([...complete, 'edge\tmain.sio\t../outside.sio']), 'edge escapes root');
    failures(
      report([...complete, 'unresolved\t../outside.sio\tunknown']),
      'unresolved caller escapes root'
    );
  });
  test('raw missing node file and directory node refuse', () => {
    failures(report([...complete, 'node\tabsent.sio']), 'invalid raw AST node');
    failures(report([...complete, 'node\tcaller']), 'invalid raw AST node');
  });
  test('raw unsupported record and invalid capacity remain parser failures', () => {
    failures(report([...complete, 'invented\tfalse']), 'invalid raw AST record');
    failures(
      report(complete.map((line) => (line === 'capacity\t256' ? 'capacity\t1e2' : line))),
      'invalid capacity'
    );
    assert.equal(
      report(complete.map((line) => (line === 'capacity\t256' ? 'capacity\t9_999' : line)))
        .capacity,
      4096
    );
  });
  test('raw capacity rejects BOM ASCII controls and excessive digit count', () => {
    for (const value of ['\ufeff2', '\x1c2', '9'.repeat(4301)])
      failures(
        report(complete.map((line) => (line === 'capacity\t256' ? 'capacity\t' + value : line))),
        'invalid capacity'
      );
    failures(
      report(complete.map((line) => (line === 'capacity\t256' ? 'capacity\t\u00852\u0085' : line))),
      'invalid capacity'
    );
    assert.equal(
      report(complete.map((line) => (line === 'capacity\t256' ? 'capacity\t\u00a02\u00a0' : line)))
        .capacity,
      2
    );
  });
  test('raw duplicate nodes and edges follow legacy set semantics', () => {
    const closure = report([
      ...complete,
      'node\tmain.sio',
      'node\tdep.sio',
      'edge\tmain.sio\tdep.sio',
      'edge\tmain.sio\tdep.sio',
    ]);
    assert.deepEqual(closure.nodes, [dep, main]);
    assert.deepEqual(closure.edges, [[main, dep]]);
    assert.deepEqual(closure.parse_failures, []);
  });
  test('raw node ordering uses Unicode codepoints rather than UTF16 units', () => {
    const bmp = write('\ue000.sio'),
      astral = write('\u{10000}.sio');
    assert.deepEqual(report([...complete, 'node\t\u{10000}.sio', 'node\t\ue000.sio']).nodes, [
      main,
      bmp,
      astral,
    ]);
  });
  test('raw Python splitlines supports CRLF bareCR and Unicode separators', () => {
    const file = path.join(work, 'linebreaks.tsv');
    fs.writeFileSync(
      file,
      'SOUNIO_BOUNDARY_CLOSURE_V1\r\nstatus\tcomplete\rnode\tmain.sio\u2028capacity\t256\u0085saturated\tfalse\nparse_failed\tfalse\n'
    );
    assert.deepEqual(load_raw_ast_closure_report(file, main, root)[0].parse_failures, []);
  });
  test('physical root resolution never accepts a string prefix sibling', () => {
    const sibling = path.join(work, 'root-other');
    fs.mkdirSync(sibling);
    const file = path.join(sibling, 'dep.sio');
    fs.writeFileSync(file, '// sibling');
    assert.equal(closure_report_path(file, root), null);
    assert.equal(closure_report_path(main, root), resolve_path(main));
  });
  // Intentional malformed-report hardening beyond the permissive legacy parser.
  for (const kind of ['saturated', 'parse_failed']) {
    test('raw nonliteral boolean refuses ' + kind, () => {
      failures(
        report(complete.map((line) => (line.startsWith(kind + '\t') ? kind + '\tgarbage' : line))),
        'invalid ' + kind + ' boolean'
      );
    });
  }
  for (const kind of ['status', 'capacity', 'saturated', 'parse_failed']) {
    test('raw mandatory control refuses omission ' + kind, () => {
      failures(
        report(complete.filter((line) => !line.startsWith(kind + '\t'))),
        'missing raw AST control record: ' + kind
      );
    });
    test('raw repeated control refuses even identical ' + kind, () => {
      failures(
        report([...complete, complete.find((line) => line.startsWith(kind + '\t'))]),
        'duplicate raw AST control record: ' + kind
      );
    });
  }
  test('raw saturation true then false cannot erase incomplete authority', () => {
    failures(
      report([...complete.slice(0, 2), 'saturated\ttrue', ...complete.slice(2)]),
      'duplicate raw AST control record: saturated'
    );
  });
  test('raw status incomplete then complete cannot override authority', () => {
    failures(
      report(['status\tincomplete', ...complete]),
      'duplicate raw AST control record: status'
    );
  });
  test('raw parser true then false still records duplicate control refusal', () => {
    failures(
      report([...complete.slice(0, 2), 'parse_failed\ttrue', ...complete.slice(2)]),
      'duplicate raw AST control record: parse_failed'
    );
  });
  fs.writeFileSync(path.join(work, 'RESULTS.json'), JSON.stringify(results, null, 2) + '\n');
  console.log('SCIENCE_BOUNDARY_CLOSURE_OPERATIONAL_PASS tests=' + results.length);
  console.log('operational_fixtures=' + work);
} catch (error) {
  fs.writeFileSync(path.join(work, 'RESULTS.json'), JSON.stringify(results, null, 2) + '\n');
  console.error(error.stack);
  console.error('operational_fixtures=' + work);
  process.exitCode = 1;
}
