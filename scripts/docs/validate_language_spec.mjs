import { execFileSync } from 'node:child_process';
import { readFile, readdir, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const editionDir = path.join(root, 'docs/spec/2026-10');
const catalogHeader = ['rule_id', 'standing', 'source', 'symbol', 'positive', 'negative', 'gap'];
const standings = new Set(['source-observed', 'specified-contract', 'library-contract', 'target-contract', 'open-decision', 'editorial-contract']);
const argumentsSet = new Set(process.argv.slice(2));

function parseTable(text, header, label) {
  const lines = text.replaceAll('\r\n', '\n').trimEnd().split('\n');
  if (lines.shift() !== header.join('\t')) {
    throw new Error(`${label}: invalid TSV header`);
  }
  return lines.map((line, index) => {
    const cells = line.split('\t');
    if (cells.length !== header.length || cells.some(cell => !cell.trim())) {
      throw new Error(`${label}:${index + 2}: invalid or empty TSV cell`);
    }
    return Object.fromEntries(header.map((field, fieldIndex) => [field, cells[fieldIndex]]));
  });
}

function markdownProse(text) {
  return text.replace(/<!--[\s\S]*?-->/g, '').replace(/^ {0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?^ {0,3}\1[ \t]*$/gm, '');
}

function ruleDeclarations(body) {
  const identifier = '[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\\d{3}';
  const pattern = new RegExp('^ {0,3}(?:#{1,6}[ \\t]+)?(?:\\*\\*)?(?:`(' + identifier + ')`|(' + identifier + '))(?:\\*\\*)?[ \\t]+(?:[—–-](?:[ \\t]|$)|\\((?:\\*\\*)?(?:' + [...standings].join('|') + ')\\b)', 'gm');
  return [...markdownProse(body).matchAll(pattern)].map(match => match[1] ?? match[2]);
}

function catalogIssues(rows, body, prefixes) {
  const errors = [];
  const identifiers = new Set();
  const declarations = ruleDeclarations(body);
  const declared = new Set(declarations);
  if (!rows.length) errors.push('empty evidence catalog');
  if (!declared.size) errors.push('chapter has no rule declarations');
  if (declared.size !== declarations.length) errors.push('duplicate rule declaration');
  for (const row of rows) {
    if (!/^[A-Z][A-Z0-9]*(?:-[A-Z][A-Z0-9]*)*-\d{3}$/.test(row.rule_id)) {
      errors.push(`invalid rule identifier ${row.rule_id}`);
    }
    if (identifiers.has(row.rule_id)) errors.push(`duplicate rule ${row.rule_id}`);
    identifiers.add(row.rule_id);
    if (!standings.has(row.standing)) errors.push(`${row.rule_id}: unknown standing ${row.standing}`);
    if (!declared.has(row.rule_id)) errors.push(`${row.rule_id}: absent from chapter declarations`);
    if (!row.gap || row.gap === '-') errors.push(`${row.rule_id}: missing evidence boundary`);
    if (row.source === '-' && !['open-decision', 'editorial-contract'].includes(row.standing)) errors.push(`${row.rule_id}: missing primary source`);
    if (!prefixes.some(prefix => row.rule_id.startsWith(`${prefix}-`))) errors.push(`${row.rule_id}: unexpected chapter prefix`);
  }
  for (const identifier of declared) {
    if (!identifiers.has(identifier)) errors.push(`${identifier}: missing catalog row`);
  }
  return [...new Set(errors)];
}

function grammarIssues(text) {
  const tokens = [];
  const defined = new Set();
  const references = new Set();
  const errors = [];
  let offset = 0;
  let cursor = 0;
  const fail = message => { throw new Error(`grammar: ${message}`); };
  const peek = () => tokens[cursor]?.kind ?? 'EOF';
  const take = kind => {
    if (peek() !== kind) fail(`expected ${kind}, found ${peek()} at ${tokens[cursor]?.offset ?? text.length}`);
    return tokens[cursor++];
  };
  function primary() {
    if (peek() === 'identifier') {
      references.add(take('identifier').value);
    } else if (peek() === 'terminal' || peek() === 'special') {
      cursor += 1;
    } else if (['(', '[', '{'].includes(peek())) {
      const opening = tokens[cursor++].kind;
      const closing = { '(': ')', '[': ']', '{': '}' }[opening];
      choice();
      if (peek() !== closing) fail(`unclosed grouping: expected ${closing}, found ${peek()}`);
      take(closing);
    } else {
      fail(`expected nonempty factor, found ${peek()} at ${tokens[cursor]?.offset ?? text.length}`);
    }
  }
  function sequence() {
    primary();
    while (peek() === ',') {
      take(',');
      primary();
    }
  }
  function choice() {
    sequence();
    while (peek() === '|') {
      take('|');
      sequence();
    }
  }
  try {
    while (offset < text.length) {
      if (/\s/.test(text[offset])) { offset += 1; continue; }
      const start = offset;
      if (text.startsWith('(*', offset)) {
        let depth = 1;
        offset += 2;
        while (offset < text.length && depth) {
          if (text.startsWith('(*', offset)) { depth += 1; offset += 2; }
          else if (text.startsWith('*)', offset)) { depth -= 1; offset += 2; }
          else offset += 1;
        }
        if (depth) fail(`unterminated comment at ${start}`);
      } else if (text[offset] === '"' || text[offset] === "'") {
        const quote = text[offset++];
        while (offset < text.length && text[offset] !== quote) {
          if (text[offset] === '\\') offset += 2;
          else offset += 1;
        }
        if (offset >= text.length) fail(`unterminated terminal at ${start}`);
        offset += 1;
        tokens.push({ kind: 'terminal', offset: start });
      } else if (text[offset] === '?') {
        const end = text.indexOf('?', offset + 1);
        if (end < 0) fail(`unterminated special sequence at ${start}`);
        offset = end + 1;
        tokens.push({ kind: 'special', offset: start });
      } else {
        const identifier = /^[A-Za-z_][A-Za-z0-9_-]*/.exec(text.slice(offset));
        if (identifier) {
          tokens.push({ kind: 'identifier', value: identifier[0], offset });
          offset += identifier[0].length;
        } else if (text.startsWith('::=', offset)) {
          tokens.push({ kind: '=', offset });
          offset += 3;
        } else if ('=|,;()[]{}'.includes(text[offset])) {
          tokens.push({ kind: text[offset], offset });
          offset += 1;
        } else fail(`unexpected character ${JSON.stringify(text[offset])} at ${offset}`);
      }
    }
    while (peek() !== 'EOF') {
      const name = take('identifier').value;
      if (defined.has(name)) errors.push(`grammar: duplicate production ${name}`);
      defined.add(name);
      take('=');
      choice();
      take(';');
    }
  } catch (error) {
    errors.push(error.message);
  }
  if (!defined.size) errors.push('grammar: no productions');
  for (const reference of references) {
    if (!defined.has(reference)) errors.push(`grammar: undefined production ${reference}`);
  }
  return { errors: [...new Set(errors)], productions: defined.size };
}

function headingIds(text) {
  const ids = new Set();
  const occurrences = new Map();
  for (const match of markdownProse(text).matchAll(/^#{1,6}\s+(.+)$/gm)) {
    const slug = match[1].trim().toLowerCase().replace(/<[^>]*>/g, '').replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-');
    const occurrence = occurrences.get(slug) ?? 0;
    occurrences.set(slug, occurrence + 1);
    ids.add(occurrence ? `${slug}-${occurrence}` : slug);
  }
  for (const match of text.matchAll(/<(?:a|span)\s+[^>]*(?:id|name)=["']([^"']+)["']/g)) ids.add(match[1]);
  return ids;
}

function safeEditionPath(value) {
  return typeof value === 'string' && value.length > 0 && value !== '.' && value !== '..'
    && path.basename(value) === value && !/[\\/\0]/.test(value);
}

async function exists(relative, fileOnly = false) {
  if (path.isAbsolute(relative) || relative.split('/').includes('..')) return false;
  try {
    const entry = await stat(path.join(root, relative));
    return !fileOnly || entry.isFile();
  } catch {
    return false;
  }
}

async function evidenceIssues(row, sourceCache = new Map()) {
  const errors = [];
  for (const field of ['source', 'positive', 'negative']) {
    if (row[field] !== '-' && !(await exists(row[field], true))) errors.push(`${row.rule_id}: missing or non-file ${field} path ${row[field]}`);
  }
  if (row.symbol !== '-') {
    if (row.source === '-' || !(await exists(row.source, true))) {
      errors.push(`${row.rule_id}: symbol has no source file`);
    } else {
      if (!sourceCache.has(row.source)) sourceCache.set(row.source, await readFile(path.join(root, row.source), 'utf8'));
      if (!sourceCache.get(row.source).includes(row.symbol)) errors.push(`${row.rule_id}: missing literal source symbol ${row.symbol}`);
    }
  }
  return errors;
}

async function editionPathIssues(edition) {
  const errors = [];
  for (const field of ['grammar', 'coverage', 'compiler_execution_receipt']) {
    if (field === 'compiler_execution_receipt' && edition[field] === undefined) continue;
    if (!safeEditionPath(edition[field])) errors.push(`Unsafe edition ${field} path`);
    else if (!(await exists(path.relative(root, path.join(editionDir, edition[field])), true))) errors.push(`Edition ${field} is missing or not a file`);
  }
  return errors;
}

async function linkIssues(file, text) {
  const errors = [];
  let prose = markdownProse(text).replace(/`[^`\n]*`/g, '');
  const references = new Map();
  const targets = [];
  const normalizeLabel = label => label.trim().replace(/\s+/g, ' ').toLowerCase();
  prose = prose.replace(/^ {0,3}\[([^\]\n]+)\]:[ \t]*(<[^>\n]*>|\S+)(?:[ \t]+(?:"[^"\n]*"|'[^'\n]*'|\([^\)\n]*\)))?[ \t]*$/gm, (definition, label, target) => {
    const key = normalizeLabel(label);
    if (!references.has(key)) references.set(key, target);
    return '';
  });
  for (const match of prose.matchAll(/\[[^\]\n]*\]\((<[^>\n]*>|[^)\n]+)\)/g)) targets.push(match[1].trim().replace(/\s+(?:"[^"]*"|'[^']*')$/, ''));
  for (const match of prose.matchAll(/\[([^\]\n]+)\](?:\[([^\]\n]*)\])?/g)) {
    if (match[2] === undefined && prose[match.index + match[0].length] === '(') continue;
    const label = normalizeLabel(match[2] || match[1]);
    if (references.has(label)) targets.push(references.get(label));
    else if (match[2] !== undefined) errors.push(`${path.relative(root, file)}: undefined link reference ${label}`);
  }
  for (let target of targets) {
    target = target.replace(/^<|>$/g, '');
    if (/^(?:https?:|mailto:|data:|tel:)/.test(target)) continue;
    const [rawPath, rawAnchor] = target.split('#', 2);
    let decodedPath;
    let anchor;
    try {
      decodedPath = decodeURIComponent(rawPath.split('?')[0]);
      anchor = rawAnchor ? decodeURIComponent(rawAnchor) : '';
    } catch {
      errors.push(`${path.relative(root, file)}: invalid link encoding ${target}`);
      continue;
    }
    const resolved = rawPath ? path.resolve(path.dirname(file), decodedPath) : file;
    const relative = path.relative(root, resolved);
    if (relative.startsWith('..') || !(await exists(relative))) {
      errors.push(`${path.relative(root, file)}: broken or escaping link ${target}`);
      continue;
    }
    if (rawAnchor && path.extname(resolved) === '.md') {
      if (!headingIds(await readFile(resolved, 'utf8')).has(anchor)) errors.push(`${path.relative(root, file)}: missing anchor ${target}`);
    }
  }
  return errors;
}

function treeEntries(revision, directory = '', directoriesOnly = false) {
  const object = directory ? `${revision}:${directory}` : revision;
  const args = ['ls-tree', '--name-only'];
  if (directoriesOnly) args.push('-d');
  args.push(object);
  return execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim().split('\n').filter(Boolean).map(entry => directory ? `${directory}/${entry}` : entry);
}

async function selfTest() {
  const row = { rule_id: 'SPEC-001', standing: 'source-observed', source: 'AGENTS.md', symbol: '-', positive: '-', negative: '-', gap: 'Unrun.' };
  const declaration = '`SPEC-001` — A rule.';
  const chapterFile = path.join(editionDir, '02-types-generics-and-ownership.md');
  const validLink = '02-types-generics-and-ownership.md';
  const manifestPaths = { grammar: 'core-grammar.ebnf', coverage: 'repository-coverage.tsv' };
  const cases = [
    ['valid catalog', catalogIssues([row], declaration, ['SPEC']).length === 0],
    ['heading declaration', catalogIssues([row], '### SPEC-001 — A rule.', ['SPEC']).length === 0],
    ['quoted heading declaration', catalogIssues([row], '### `SPEC-001` — A rule.', ['SPEC']).length === 0],
    ['bold declaration', catalogIssues([row], '**`SPEC-001` — `specified-contract`.** A rule.', ['SPEC']).length === 0],
    ['parenthetical declaration', catalogIssues([row], '`SPEC-001` (**source-observed**, topic). A rule.', ['SPEC']).length === 0],
    ['duplicate ID refused', catalogIssues([row, row], declaration, ['SPEC']).some(error => error.includes('duplicate'))],
    ['duplicate declaration refused', catalogIssues([row], `${declaration}\n${declaration}`, ['SPEC']).some(error => error.includes('duplicate'))],
    ['empty catalog refused', catalogIssues([], '# Empty', ['SPEC']).some(error => error.includes('empty'))],
    ['bare reference refused', catalogIssues([row], 'See SPEC-001 for details.', ['SPEC']).some(error => error.includes('absent'))],
    ['comment declaration ignored', catalogIssues([row], `<!--\n${declaration}\n-->`, ['SPEC']).some(error => error.includes('absent'))],
    ['fenced declaration ignored', catalogIssues([row], `\`\`\`md\n${declaration}\n\`\`\``, ['SPEC']).some(error => error.includes('absent'))],
    ['tilde fenced declaration ignored', catalogIssues([row], `~~~md\n${declaration}\n~~~`, ['SPEC']).some(error => error.includes('absent'))],
    ['unknown standing refused', catalogIssues([{ ...row, standing: 'passed' }], declaration, ['SPEC']).some(error => error.includes('standing'))],
    ['missing prose refused', catalogIssues([row], '', ['SPEC']).some(error => error.includes('absent'))],
    ['uncatalogued declaration refused', catalogIssues([row], `${declaration}\n### SPEC-002 — Another rule.`, ['SPEC']).some(error => error.includes('missing catalog'))],
    ['ordinary reference ignored', catalogIssues([row], `${declaration}\nSee SPEC-002 for context.`, ['SPEC']).length === 0],
    ['missing source refused', catalogIssues([{ ...row, source: '-' }], declaration, ['SPEC']).some(error => error.includes('primary source'))],
    ['session editorial source allowed', catalogIssues([{ ...row, source: '-', standing: 'editorial-contract' }], declaration, ['SPEC']).length === 0],
    ['missing gap refused', catalogIssues([{ ...row, gap: '-' }], declaration, ['SPEC']).some(error => error.includes('boundary'))],
    ['valid source and no symbol', (await evidenceIssues(row)).length === 0],
    ['literal source symbol', (await evidenceIssues({ ...row, symbol: '# AGENTS.md' })).length === 0],
    ['invented symbol refused', (await evidenceIssues({ ...row, symbol: 'definitely_nonexistent_spec_symbol_20261016' })).some(error => error.includes('symbol'))],
    ['symbol alternatives not interpreted', (await evidenceIssues({ ...row, symbol: '# AGENTS.md|FOUNDER_INTENT.md' })).some(error => error.includes('symbol'))],
    ['symbol without source refused', (await evidenceIssues({ ...row, source: '-', symbol: 'main' })).some(error => error.includes('source file'))],
    ['missing evidence file refused', (await evidenceIssues({ ...row, positive: 'absent-spec-witness.sio' })).length > 0],
    ['source directory refused', (await evidenceIssues({ ...row, source: 'self-hosted/check' })).length > 0],
    ['positive directory refused', (await evidenceIssues({ ...row, positive: 'tests' })).length > 0],
    ['negative directory refused', (await evidenceIssues({ ...row, negative: 'tests' })).length > 0],
    ['valid grammar', grammarIssues('program = item ;\nitem = "x" ;').errors.length === 0],
    ['grouped grammar', grammarIssues('program = ( "x" | "y" ), [ "z" ], { "a" } ;').errors.length === 0],
    ['special sequence and nested comment', grammarIssues('(* outer (* nested *) *) program = ? predicate; [x] ? ;').errors.length === 0],
    ['comment spelling in terminal', grammarIssues('program = "(*", "*)", "?" ;').errors.length === 0],
    ['escaped terminal delimiter', grammarIssues(String.raw`program = "a\"b" ;`).errors.length === 0],
    ['single quoted terminal', grammarIssues(`program = '"', '\\n' ;`).errors.length === 0],
    ['undefined grammar refused', grammarIssues('program = missing ;').errors.some(error => error.includes('undefined'))],
    ['duplicate grammar refused', grammarIssues('program = "x" ;\nprogram = "y" ;').errors.some(error => error.includes('duplicate'))],
    ['unclosed grammar refused', grammarIssues('program = [ "x" ;').errors.some(error => error.includes('unclosed'))],
    ['missing grammar terminator refused', grammarIssues('program = "x"').errors.length > 0],
    ['unterminated terminal refused', grammarIssues('program = "program').errors.some(error => error.includes('unterminated terminal'))],
    ['unterminated single quote refused', grammarIssues("program = 'program").errors.length > 0],
    ['unterminated special sequence refused', grammarIssues('program = ? predicate').errors.length > 0],
    ['unterminated comment refused', grammarIssues('program = "x" ; (* open').errors.length > 0],
    ['empty leading alternative refused', grammarIssues('program = | "x" ;').errors.length > 0],
    ['empty trailing alternative refused', grammarIssues('program = "x" | ;').errors.length > 0],
    ['empty production refused', grammarIssues('program = ;').errors.length > 0],
    ['empty group refused', grammarIssues('program = ( ) ;').errors.length > 0],
    ['missing concatenation refused', grammarIssues('program = "x" "y" ;').errors.length > 0],
    ['trailing concatenation refused', grammarIssues('program = "x", ;').errors.length > 0],
    ['mismatched group refused', grammarIssues('program = [ "x" ) ;').errors.length > 0],
    ['invalid grammar character refused', grammarIssues('program = @ ;').errors.length > 0],
    ['heading anchors', headingIds('# Title\n## 2.7 Comments\n## Title').has('27-comments')],
    ['safe edition basename', safeEditionPath('core-grammar.ebnf')],
    ['unsafe grammar or coverage paths refused', ['../outside', '/outside', 'sub/file', 'sub\\file', '.', '..', '', null].every(value => !safeEditionPath(value))],
    ['safe receipt basename', safeEditionPath('15-bounded-conformance-receipt.md')],
    ['optional receipt absent', (await editionPathIssues(manifestPaths)).length === 0],
    ['existing receipt file allowed', (await editionPathIssues({ ...manifestPaths, compiler_execution_receipt: validLink })).length === 0],
    ['escaping grammar refused', (await editionPathIssues({ ...manifestPaths, grammar: '../outside.ebnf' })).some(error => error.includes('Unsafe edition grammar'))],
    ['escaping coverage refused', (await editionPathIssues({ ...manifestPaths, coverage: '../outside.tsv' })).some(error => error.includes('Unsafe edition coverage'))],
    ['escaping receipt refused', (await editionPathIssues({ ...manifestPaths, compiler_execution_receipt: '../outside.md' })).some(error => error.includes('Unsafe edition compiler_execution_receipt'))],
    ['missing receipt refused', (await editionPathIssues({ ...manifestPaths, compiler_execution_receipt: 'absent-spec-receipt.md' })).some(error => error.includes('missing or not a file'))],
    ['existing inline link', (await linkIssues(chapterFile, `[chapter](${validLink})`)).length === 0],
    ['broken inline link refused', (await linkIssues(chapterFile, '[missing](absent-spec-target.md)')).length > 0],
    ['reference link', (await linkIssues(chapterFile, `[chapter][target]\n\n[target]: ${validLink}`)).length === 0],
    ['case folded reference label', (await linkIssues(chapterFile, `[chapter][MY   TARGET]\n\n[my target]: ${validLink}`)).length === 0],
    ['collapsed reference link', (await linkIssues(chapterFile, `[target][]\n\n[target]: ${validLink}`)).length === 0],
    ['shortcut reference link', (await linkIssues(chapterFile, `[target]\n\n[target]: ${validLink}`)).length === 0],
    ['broken reference target refused', (await linkIssues(chapterFile, '[missing][gone]\n\n[gone]: absent-spec-target.md')).length > 0],
    ['undefined reference refused', (await linkIssues(chapterFile, '[missing][gone]')).length > 0],
    ['reference missing anchor refused', (await linkIssues(chapterFile, `[chapter][target]\n\n[target]: ${validLink}#absent-spec-anchor`)).length > 0],
    ['reference title and angle destination', (await linkIssues(chapterFile, `[chapter][target]\n\n[target]: <${validLink}> "Title"`)).length === 0],
    ['external angle reference allowed', (await linkIssues(chapterFile, '[site][target]\n\n[target]: <https://example.invalid/>')).length === 0],
    ['commented link ignored', (await linkIssues(chapterFile, '<!-- [bad](absent-spec-target.md) -->')).length === 0],
  ];
  let invalidHeaderRefused = false;
  try { parseTable('wrong\nvalue', catalogHeader, 'mutation'); } catch { invalidHeaderRefused = true; }
  cases.push(['invalid TSV refused', invalidHeaderRefused]);
  for (const [name, passed] of cases) {
    if (!passed) throw new Error(`Self-test failed: ${name}`);
  }
  console.log(`Language specification validator self-tests passed (${cases.length} cases).`);
}

async function main() {
  for (const argument of argumentsSet) {
    if (!['--self-test', '--json'].includes(argument)) throw new Error(`Unknown argument: ${argument}`);
  }
  if (argumentsSet.has('--self-test')) {
    await selfTest();
    return;
  }
  const edition = JSON.parse(await readFile(path.join(editionDir, 'edition.json'), 'utf8'));
  const errors = [];
  if (edition.schema !== 'sounio.language-specification.edition.v1') errors.push('Unknown edition schema');
  const manifestErrors = await editionPathIssues(edition);
  if (manifestErrors.length) throw new Error(manifestErrors.join('; '));
  if (!/^[a-f0-9]{40}$/.test(edition.source_revision)) throw new Error('Invalid source revision');
  const tree = execFileSync('git', ['rev-parse', `${edition.source_revision}^{tree}`], { cwd: root, encoding: 'utf8' }).trim();
  if (tree !== edition.source_tree) errors.push('Edition source tree does not match its revision');
  const globalRules = new Set();
  const referencedPaths = new Set();
  const chapterNames = new Set();
  const sourceCache = new Map();
  for (const chapter of edition.chapters) {
    if (!safeEditionPath(chapter.path) || !safeEditionPath(chapter.evidence)) throw new Error('Unsafe chapter or catalog path');
    if (chapterNames.has(chapter.path)) errors.push(`Duplicate chapter ${chapter.path}`);
    chapterNames.add(chapter.path);
    const body = await readFile(path.join(editionDir, chapter.path), 'utf8');
    const rows = parseTable(await readFile(path.join(editionDir, chapter.evidence), 'utf8'), catalogHeader, chapter.evidence);
    errors.push(...catalogIssues(rows, body, chapter.prefixes).map(error => `${chapter.evidence}: ${error}`));
    for (const row of rows) {
      if (globalRules.has(row.rule_id)) errors.push(`Globally duplicated rule ${row.rule_id}`);
      globalRules.add(row.rule_id);
      for (const field of ['source', 'positive', 'negative']) {
        if (row[field] === '-') continue;
        referencedPaths.add(row[field]);
      }
      errors.push(...await evidenceIssues(row, sourceCache));
    }
  }
  const grammar = grammarIssues(await readFile(path.join(editionDir, edition.grammar), 'utf8'));
  errors.push(...grammar.errors);
  const coverage = parseTable(await readFile(path.join(editionDir, edition.coverage), 'utf8'), ['path', 'chapter', 'role'], edition.coverage);
  const required = new Set(treeEntries(edition.source_revision));
  for (const directory of ['stdlib', 'self-hosted', 'tools', 'ecosystem']) {
    for (const entry of treeEntries(edition.source_revision, directory, true)) required.add(entry);
  }
  const covered = new Set();
  for (const row of coverage) {
    if (covered.has(row.path)) errors.push(`Duplicate coverage path ${row.path}`);
    covered.add(row.path);
    if (!required.has(row.path)) errors.push(`Coverage path outside baseline census: ${row.path}`);
    if (!chapterNames.has(row.chapter)) errors.push(`Coverage references missing chapter ${row.chapter}`);
  }
  for (const entry of required) if (!covered.has(entry)) errors.push(`Unmapped repository entry ${entry}`);
  const files = [path.join(root, 'docs/spec/LANGUAGE_SPECIFICATION.md')];
  for (const entry of await readdir(editionDir)) if (entry.endsWith('.md')) files.push(path.join(editionDir, entry));
  for (const file of files) errors.push(...await linkIssues(file, await readFile(file, 'utf8')));
  const result = {
    schema: 'sounio.language-specification.validation.v1',
    source_revision: edition.source_revision,
    scope: 'structural-only',
    chapters: chapterNames.size,
    rules: globalRules.size,
    evidence_paths: referencedPaths.size,
    grammar_productions: grammar.productions,
    repository_entries: covered.size,
    errors,
  };
  if (argumentsSet.has('--json')) console.log(JSON.stringify(result, null, 2));
  else if (errors.length) console.error(`Language specification validation failed:\n${errors.map(error => `- ${error}`).join('\n')}`);
  else console.log(`Language specification structural validation passed: ${result.chapters} chapters, ${result.rules} rules, ${result.evidence_paths} evidence paths, ${result.grammar_productions} grammar productions, ${result.repository_entries} mapped entries.`);
  if (errors.length) process.exitCode = 1;
}

main().catch(error => {
  console.error(`Language specification validation failed: ${error.message}`);
  process.exitCode = 1;
});
