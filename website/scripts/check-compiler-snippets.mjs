/**
 * Run the Sounio fences this content refresh is willing to publish.
 *
 * A gated page may contain ```sio fences only when the first line is
 * `// souc:run`, `// souc:check`, or `// souc:fail`. Lines `// expect: ...`
 * must occur in the compiler's combined stdout and stderr. `run` and
 * `check` require exit 0. `fail` requires a non-zero exit.
 *
 * The compiler is the committed launcher, ../bin/souc. This script does
 * not build Madaros from source.
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import os from 'node:os';
import { mkdtempSync, writeFileSync } from 'node:fs';

const websiteRoot = process.cwd();
const repoRoot = path.resolve(websiteRoot, '..');
const souc = path.join(repoRoot, 'bin/souc');
const stdlib = path.join(repoRoot, 'stdlib');

const GATED = [
  'src/content/docs/en/language/units-and-uncertainty.mdx',
  'src/content/docs/en/language.mdx',
  'src/content/docs/en/epistemic/uncertainty.mdx',
  'src/content/docs/en/compiler/lexer-parser.mdx',
];

function fences(source) {
  const blocks = [];
  const re = /```sio\n([\s\S]*?)```/g;
  let match;
  while ((match = re.exec(source)) !== null) {
    blocks.push(match[1].replace(/\n$/, ''));
  }
  return blocks;
}

function fail(message) {
  console.error(`check-compiler-snippets: ${message}`);
  process.exitCode = 1;
}

const tmp = mkdtempSync(path.join(os.tmpdir(), 'sounio-snippets-'));
let ran = 0;

for (const relative of GATED) {
  const full = path.join(websiteRoot, relative);
  const source = await readFile(full, 'utf8');
  const blocks = fences(source);
  if (blocks.length === 0) {
    fail(`${relative}: no sio fence`);
    continue;
  }
  blocks.forEach((body, index) => {
    const lines = body.split('\n');
    const directive = lines[0]?.match(/^\/\/ souc:(run|check|fail)$/);
    if (!directive) {
      fail(`${relative}: fence ${index + 1} is not marked // souc:run|check|fail`);
      return;
    }
    const mode = directive[1];
    const expects = [];
    for (const line of lines.slice(1)) {
      const expected = line.match(/^\/\/ expect: (.*)$/);
      if (expected) expects.push(expected[1]);
      else break;
    }
    if (expects.length === 0) {
      fail(`${relative}: fence ${index + 1} has no // expect: line`);
      return;
    }
    const program = path.join(tmp, `snippet-${ran}.sio`);
    writeFileSync(program, `${body}\n`);
    const result = spawnSync(souc, [mode === 'run' ? 'run' : 'check', program], {
      cwd: repoRoot,
      env: { ...process.env, SOUNIO_STDLIB_PATH: stdlib },
      encoding: 'utf8',
    });
    ran += 1;
    const combined = `${result.stdout ?? ''}\n${result.stderr ?? ''}`;
    const status = result.status ?? 1;
    if (mode === 'fail') {
      if (status === 0) fail(`${relative}: fence ${index + 1} was expected to fail and exited 0`);
    } else if (status !== 0) {
      fail(`${relative}: fence ${index + 1} (${mode}) exited ${status}\n${combined.slice(0, 800)}`);
    }
    for (const needle of expects) {
      if (!combined.includes(needle)) {
        fail(`${relative}: fence ${index + 1} did not print ${JSON.stringify(needle)}`);
      }
    }
  });
}

if (!process.exitCode) {
  console.log(`check-compiler-snippets: ${ran} fences passed on ${souc}`);
}
