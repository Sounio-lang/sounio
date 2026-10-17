import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFile, realpath } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const catalogPath = 'docs/llm-guide/task-map.json';
const manifestPath = 'docs/llm-guide/example-manifest.json';
const receiptPath = 'docs/llm-guide/example-results.json';
const languages = ['en', 'pt', 'el'];
const eligible = new Set(['run-passed', 'check-passed']);
const identifier = /^[a-z][a-z0-9-]*$/;
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const textPair = value => value && languages.every(language => typeof value[language] === 'string' && value[language].trim());
const listPair = value => value && languages.every(language => Array.isArray(value[language]) && value[language].length >= 2 && value[language].every(item => typeof item === 'string' && item.trim()));
const safePath = value => typeof value === 'string' && !/[\\\0\r\n:]/.test(value) && !path.posix.isAbsolute(value) && value.split('/').every(part => part && part !== '.' && part !== '..' && part !== '.git');

export function validateCatalog(catalog, manifest, receipt) {
  const errors = [];
  if (catalog?.schema !== 'sounio.llm-task-map.v1') errors.push('invalid catalog schema');
  if (manifest?.schema !== 'sounio.llm-guide.examples.v1') errors.push('invalid manifest schema');
  if (!/^[a-f0-9]{40}$/.test(catalog?.source_revision ?? '') || catalog.source_revision !== manifest?.source_revision) errors.push('catalog baseline mismatch');
  if (receipt?.source_revision !== manifest?.source_revision) errors.push('receipt baseline mismatch');
  if (receipt?.schema !== 'sounio.llm-guide.example-results.v1') errors.push('invalid receipt schema');
  if (receipt?.execution?.raw_compiler_sha256 !== manifest?.compiler_profile?.raw_sha256 || !/^[a-f0-9]{64}$/.test(manifest?.compiler_profile?.raw_sha256 ?? '')) errors.push('compiler digest mismatch');
  const examples = new Map();
  const sourceIdentities = new Map();
  for (const example of manifest?.examples ?? []) {
    if (examples.has(example.id)) errors.push(`duplicate example ${example.id}`);
    if (sourceIdentities.has(example.source) && sourceIdentities.get(example.source) !== example.source_sha256) errors.push(`conflicting source hashes ${example.source}`);
    sourceIdentities.set(example.source, example.source_sha256);
    examples.set(example.id, example);
  }
  const observations = new Map();
  for (const observation of receipt?.cases ?? []) {
    if (observations.has(observation.id)) errors.push(`duplicate receipt ${observation.id}`);
    observations.set(observation.id, observation);
  }
  if (!Array.isArray(catalog?.tasks) || !catalog.tasks.length) errors.push('missing tasks');
  const taskIds = new Set();
  for (const task of Array.isArray(catalog?.tasks) ? catalog.tasks : []) {
    if (!task || typeof task !== 'object') { errors.push('invalid task row'); continue; }
    if (!identifier.test(task.id ?? '') || taskIds.has(task.id)) errors.push(`invalid or duplicate task ${task.id}`);
    taskIds.add(task.id);
    if (!['starter', 'core', 'scientific', 'integration'].includes(task.level)) errors.push(`${task.id}: invalid level`);
    for (const field of ['title', 'intent']) if (!textPair(task[field])) errors.push(`${task.id}: missing bilingual ${field}`);
    for (const field of ['steps', 'acceptance', 'pitfalls']) if (!listPair(task[field])) errors.push(`${task.id}: missing bilingual ${field}`);
    if (!Array.isArray(task.controls) || !task.controls.length || new Set(task.controls).size !== task.controls.length || task.controls.includes(task.reference)) errors.push(`${task.id}: invalid controls`);
    for (const [index, id] of [task.reference, ...(Array.isArray(task.controls) ? task.controls : [])].entries()) {
      const example = examples.get(id);
      const observation = observations.get(id);
      if (!example || !observation) { errors.push(`${task.id}: missing example or receipt ${id}`); continue; }
      if (!safePath(example.source) || !example.source.endsWith('.sio') || !/^[a-f0-9]{64}$/.test(example.source_sha256 ?? '')) errors.push(`${id}: invalid source identity`);
      if (observation.source_sha256 !== example.source_sha256 || observation.source !== example.source) errors.push(`${id}: receipt source mismatch`);
      const state = example.verification?.state;
      if (observation.state !== state) errors.push(`${id}: verification state mismatch`);
      if (observation.reason !== 'ok') errors.push(`${id}: recorded failure reason`);
      if (index === 0 && (!eligible.has(state) || example.kind !== 'positive')) errors.push(`${task.id}: reference is not a passed positive example`);
      if (index !== 0 && !eligible.has(state) && state !== 'expected-refusal') errors.push(`${task.id}: quarantined or unexecuted control ${id}`);
      if (observation.mode !== example.mode || observation.expected_exit !== example.expected_exit) errors.push(`${id}: receipt contract mismatch`);
      if (eligible.has(state) && (example.kind !== 'positive' || !Number.isInteger(example.expected_exit) || example.expected_exit < 0 || example.expected_exit > 255 || !Number.isInteger(observation.actual_exit))) errors.push(`${id}: invalid positive exit or kind`);
      if (eligible.has(state) && observation.actual_exit !== example.expected_exit) errors.push(`${id}: recorded exit mismatch`);
      if (state === 'run-passed' && example.mode !== 'run') errors.push(`${id}: invalid run state`);
      if (state === 'check-passed' && example.mode !== 'check') errors.push(`${id}: invalid check state`);
      if (state === 'check-passed' && example.expected_exit !== 0) errors.push(`${id}: check-passed requires zero exit`);
      if (state === 'expected-refusal' && (example.kind !== 'negative' || example.mode !== 'check-fail' || !Number.isInteger(observation.actual_exit) || observation.actual_exit <= 0 || observation.actual_exit >= 124 || !example.diagnostic)) errors.push(`${id}: invalid refusal state`);
      if (state === 'expected-refusal') {
        if (example.expected_exit !== 'nonzero' && (!Number.isInteger(example.expected_exit) || example.expected_exit <= 0 || example.expected_exit >= 124 || example.expected_exit !== observation.actual_exit)) errors.push(`${id}: invalid refusal exit contract`);
        if (typeof example.diagnostic !== 'string' || ![example.stdout_contains, example.stderr_contains].some(marker => typeof marker === 'string' && marker.includes(example.diagnostic))) errors.push(`${id}: refusal diagnostic is not asserted on a stream`);
      }
      for (const [field, stream] of [['stdout_contains', 'stdout'], ['stderr_contains', 'stderr']]) {
        if (example[field] !== null && (typeof example[field] !== 'string' || !example[field].length || typeof observation[stream] !== 'string' || !observation[stream].includes(example[field]))) errors.push(`${id}: recorded ${field} mismatch`);
      }
    }
  }
  return [...new Set(errors)];
}

async function readRepository(file) {
  if (!safePath(file)) throw new Error(`Unsafe repository path: ${file}`);
  const absolute = await realpath(path.join(root, file));
  const relative = path.relative(root, absolute);
  if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) throw new Error(`Path escapes repository: ${file}`);
  return readFile(absolute);
}

async function load() {
  const [catalog, manifest, receipt] = await Promise.all([catalogPath, manifestPath, receiptPath].map(async file => JSON.parse(await readRepository(file))));
  const errors = validateCatalog(catalog, manifest, receipt);
  if (errors.length) throw new Error(errors.join('\n'));
  const selected = new Set(catalog.tasks.flatMap(task => [task.reference, ...task.controls]));
  const selectedExamples = manifest.examples.filter(example => selected.has(example.id));
  const selectedPaths = [...new Set(selectedExamples.map(example => example.source))];
  const batch = execFileSync('git', ['cat-file', '--batch'], {
    cwd: root, input: selectedPaths.map(file => `${manifest.source_revision}:${file}\n`).join(''),
    env: { ...process.env, GIT_NO_LAZY_FETCH: '1' }, stdio: ['pipe', 'pipe', 'pipe'], maxBuffer: 16 * 1024 * 1024,
  });
  const baselines = parseBaselineBatch(batch, selectedPaths);
  const sources = new Map();
  for (const example of selectedExamples) {
    if (sources.has(example.source)) continue;
    const bytes = await readRepository(example.source);
    if (digest(bytes) !== example.source_sha256) throw new Error(`Current source hash mismatch: ${example.source}`);
    const baseline = baselines.get(example.source);
    if (!bytes.equals(baseline)) throw new Error(`Source differs from recorded baseline: ${example.source}`);
    sources.set(example.source, bytes.toString('utf8'));
  }
  return { catalog, manifest, receipt, sources };
}

export function parseBaselineBatch(bytes, files) {
  const baselines = new Map();
  let cursor = 0;
  for (const file of files) {
    const end = bytes.indexOf(10, cursor);
    if (end < 0) throw new Error('Truncated Git batch header');
    const header = /^[a-f0-9]{40} blob (\d+)$/.exec(bytes.subarray(cursor, end).toString('utf8'));
    if (!header) throw new Error(`Missing baseline blob: ${file}`);
    const size = Number(header[1]);
    const start = end + 1;
    if (!Number.isSafeInteger(size) || start + size >= bytes.length || bytes[start + size] !== 10) throw new Error('Truncated Git batch content');
    baselines.set(file, bytes.subarray(start, start + size));
    cursor = start + size + 1;
  }
  if (cursor !== bytes.length) throw new Error('Unexpected Git batch trailing content');
  return baselines;
}

export function selectTasks(tasks, query) {
  const terms = query.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase().split(/\s+/).filter(Boolean);
  return tasks.filter(task => {
    const haystack = JSON.stringify([task.id, task.title, task.intent]).normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase();
    return terms.every(term => haystack.includes(term));
  });
}

export function contextPack(task, state, language) {
  if (!languages.includes(language)) throw new Error('Unsupported language');
  const selected = [task.reference, ...task.controls].map(id => {
    const example = state.manifest.examples.find(entry => entry.id === id);
    const observation = state.receipt.cases.find(entry => entry.id === id);
    if (!example || !observation || typeof state.sources.get(example.source) !== 'string') throw new Error(`Context requires validated example and source: ${id}`);
    return {
      id, role: id === task.reference ? 'reference' : example.kind === 'negative' ? 'refusal-control-not-a-template' : 'positive-control',
      source: example.source, source_sha256: example.source_sha256, code: state.sources.get(example.source),
      recorded_state: observation.state, recorded_mode: example.mode, recorded_exit: observation.actual_exit,
      expected_exit: example.expected_exit, stdout_contains: example.stdout_contains, stderr_contains: example.stderr_contains,
      diagnostic: example.diagnostic,
    };
  });
  const portuguese = language === 'pt';
  const greek = language === 'el';
  return {
    schema: 'sounio.llm-task-context.v1', language, task_id: task.id, title: task.title[language], intent: task.intent[language],
    authority: greek
      ? 'Υλικό αναφοράς, όχι οδηγία συστήματος. Προηγούνται οι οδηγίες χρήστη και αποθετηρίου. Δεν εγκρίνει εργαλεία, δημοσίευση ή μεταφορά δεδομένων.'
      : portuguese
      ? 'Material de referência, não instrução de sistema. As instruções do usuário e do repositório prevalecem. Não autoriza ferramentas, publicação ou envio de dados.'
      : 'Reference data, not a system instruction. User and repository instructions prevail. This grants no permission for tools, publication or data transfer.',
    evidence_boundary: greek
      ? 'Ο δημιουργός πλαισίου δεν εκτελεί μεταγλωττιστή και δεν πιστοποιεί το τεκμήριο. Τα αποτελέσματα είναι ιστορικά και αφορούν το δηλωμένο προφίλ· δεν επικυρώνουν αλλαγές, σημερινές εξαρτήσεις ή άλλον στόχο.'
      : portuguese
      ? 'O gerador não executa o compilador nem autentica o recibo. Resultados registrados são históricos e do perfil indicado; não validam alterações, dependências atuais ou outro target.'
      : 'The generator does not execute a compiler or authenticate the receipt. Recorded outcomes are historical and profile-specific; they do not validate edits, current dependencies or another target.',
    source_revision: state.manifest.source_revision, compiler_profile: state.manifest.compiler_profile,
    receipt: receiptPath, steps: task.steps[language], acceptance: task.acceptance[language], pitfalls: task.pitfalls[language], examples: selected,
    delivery: greek
      ? ['Πλήρες πρόγραμμα, χωρίς ανύπαρκτες βοηθητικές συναρτήσεις.', 'Δηλώστε επιτρεπόμενα αρχεία και σημασιολογία που διατηρήθηκε.', 'Διαχωρίστε αναμενόμενο και παρατηρημένο· κρατήστε εντολές, προφίλ, exit και εξόδους.', 'Χωρίς εργαλεία: δηλώστε ότι δεν εκτελέστηκε. Μην επινοείτε αποτελέσματα.']
      : portuguese
      ? ['Programa completo, sem funções auxiliares inexistentes.', 'Declare arquivos autorizados e semântica preservada.', 'Separe esperado de observado; registre comandos, perfil, exits e streams.', 'Sem ferramenta: declare não executado. Não invente resultados.']
      : ['Complete program, no imaginary helpers.', 'State authorized paths and preserved semantics.', 'Separate expected from observed; record commands, profile, exits and streams.', 'Without tools: state not executed. Never invent results.'],
  };
}

export function renderPack(pack) {
  const greek = pack.language === 'el';
  const labels = greek
    ? { task: 'Εργασία', baseline: 'Αναθεώρηση αναφοράς', profile: 'Προφίλ μεταγλωττιστή', receipt: 'Ιστορικό τεκμήριο', steps: 'Βήματα', acceptance: 'Κριτήρια αποδοχής', pitfalls: 'Παγίδες', source: 'Πηγή', recorded: 'Καταγεγραμμένο', expected: 'Αναμενόμενο', delivery: 'Παράδοση' }
    : { task: 'Task / tarefa', baseline: 'Source baseline', profile: 'Compiler profile', receipt: 'Historical receipt', steps: 'Steps / passos', acceptance: 'Acceptance / aceitacao', pitfalls: 'Pitfalls / armadilhas', source: 'Source', recorded: 'Recorded', expected: 'Expected', delivery: 'Delivery / entrega' };
  const lines = [`# ${pack.title}`, '', pack.authority, '', pack.evidence_boundary, '', `${labels.task}: ${pack.intent}`, '',
    `${labels.baseline}: ${pack.source_revision}`, `${labels.profile}: ${JSON.stringify(pack.compiler_profile)}`, `${labels.receipt}: ${pack.receipt}`, '',
    `## ${labels.steps}`, ...pack.steps.map((step, index) => `${index + 1}. ${step}`), '',
    `## ${labels.acceptance}`, ...pack.acceptance.map(item => `- ${item}`), '',
    `## ${labels.pitfalls}`, ...pack.pitfalls.map(item => `- ${item}`), ''];
  for (const example of pack.examples) {
    const delimiter = '`'.repeat(Math.max(3, ...(example.code.match(/`+/g) ?? []).map(run => run.length + 1)));
    const role = greek ? { reference: 'Αναφορά', 'positive-control': 'Θετικός έλεγχος', 'refusal-control-not-a-template': 'Αρνητικός έλεγχος — όχι πρότυπο' }[example.role] : example.role;
    lines.push(`## ${role}: ${example.id}`, `${labels.source}: ${example.source}`, `SHA-256: ${example.source_sha256}`,
      `${labels.recorded}: ${example.recorded_state}; mode=${example.recorded_mode}; exit=${example.recorded_exit}`,
      `${labels.expected}: ${JSON.stringify({ exit: example.expected_exit, stdout_contains: example.stdout_contains, stderr_contains: example.stderr_contains, diagnostic: example.diagnostic })}`,
      ...(example.role === 'refusal-control-not-a-template' ? [greek ? 'ΑΡΝΗΤΙΚΟΣ ΕΛΕΓΧΟΣ: σκόπιμη απόρριψη· ποτέ μην τον χρησιμοποιείτε ως λειτουργικό πρότυπο.' : 'NEGATIVE CONTROL: intentionally refused; never use as a working template. / CONTROLE NEGATIVO: recusa intencional; nao copiar como receita funcional.'] : []),
      `${delimiter}sio`, example.code.replace(/\n$/, ''), delimiter, '');
  }
  lines.push(`## ${labels.delivery}`, ...pack.delivery.map(item => `- ${item}`), '');
  return lines.join('\n');
}

function selfTest() {
  const translated = { en: 'Example', pt: 'Exemplo', el: 'Παράδειγμα' };
  const lists = { en: ['One', 'Two'], pt: ['Um', 'Dois'], el: ['Ένα', 'Δύο'] };
  const hash = 'a'.repeat(64);
  const source = 'fn main() -> i32 { return 0 }\n';
  const example = { id: 'hello', source: 'examples/hello.sio', source_sha256: hash, kind: 'positive', mode: 'run', expected_exit: 0, stdout_contains: 'OK', stderr_contains: null, verification: { state: 'run-passed' } };
  const observation = { id: 'hello', source: example.source, source_sha256: hash, state: 'run-passed', mode: 'run', expected_exit: 0, actual_exit: 0, reason: 'ok', stdout: 'OK', stderr: '' };
  const task = { id: 'hello-task', level: 'starter', title: translated, intent: { en: 'Print greeting', pt: 'Impressao inicial', el: 'Πρώτο μήνυμα' }, reference: 'hello', controls: ['control'], steps: lists, acceptance: lists, pitfalls: lists };
  const state = {
    catalog: { schema: 'sounio.llm-task-map.v1', source_revision: 'b'.repeat(40), tasks: [task] },
    manifest: { schema: 'sounio.llm-guide.examples.v1', source_revision: 'b'.repeat(40), compiler_profile: { raw_sha256: hash }, examples: [example, { ...example, id: 'control' }] },
    receipt: { schema: 'sounio.llm-guide.example-results.v1', source_revision: 'b'.repeat(40), execution: { raw_compiler_sha256: hash }, cases: [observation, { ...observation, id: 'control' }] },
    sources: new Map([[example.source, source]]),
  };
  let cases = 0;
  const check = (label, mutate, expected) => {
    const candidate = structuredClone(state);
    mutate(candidate);
    const errors = validateCatalog(candidate.catalog, candidate.manifest, candidate.receipt);
    assert.equal(errors.some(error => error.includes(expected)), true, `${label}: ${errors.join('; ')}`);
    cases += 1;
  };
  assert.deepEqual(validateCatalog(state.catalog, state.manifest, state.receipt), []); cases += 1;
  check('wrong schema', value => value.catalog.schema = 'unknown', 'schema');
  check('baseline drift', value => value.catalog.source_revision = 'c'.repeat(40), 'baseline mismatch');
  check('receipt baseline drift', value => value.receipt.source_revision = 'c'.repeat(40), 'receipt baseline');
  check('compiler drift', value => value.receipt.execution.raw_compiler_sha256 = 'd'.repeat(64), 'compiler digest');
  check('empty tasks', value => value.catalog.tasks = [], 'missing tasks');
  check('null row', value => value.catalog.tasks.push(null), 'invalid task row');
  check('duplicate task', value => value.catalog.tasks.push(value.catalog.tasks[0]), 'duplicate task');
  check('bad level', value => value.catalog.tasks[0].level = 'expert', 'level');
  for (const field of ['title', 'intent', 'steps', 'acceptance', 'pitfalls']) check(`missing PT ${field}`, value => delete value.catalog.tasks[0][field].pt, `bilingual ${field}`);
  for (const field of ['title', 'intent', 'steps', 'acceptance', 'pitfalls']) check(`missing EL ${field}`, value => delete value.catalog.tasks[0][field].el, `bilingual ${field}`);
  check('missing reference', value => value.catalog.tasks[0].reference = 'absent', 'missing example');
  check('no controls', value => value.catalog.tasks[0].controls = [], 'invalid controls');
  check('repeated control', value => value.catalog.tasks[0].controls.push('control'), 'invalid controls');
  check('self control', value => value.catalog.tasks[0].controls = ['hello'], 'invalid controls');
  check('quarantined reference', value => { value.manifest.examples[0].verification.state = 'unexpected-failure'; value.receipt.cases[0].state = 'unexpected-failure'; }, 'not a passed');
  check('quarantined control', value => { value.manifest.examples[1].verification.state = 'unexpected-failure'; value.receipt.cases[1].state = 'unexpected-failure'; }, 'quarantined');
  check('receipt state disagreement', value => value.receipt.cases[0].state = 'check-passed', 'state mismatch');
  check('receipt hash drift', value => value.receipt.cases[0].source_sha256 = 'f'.repeat(64), 'source mismatch');
  check('receipt path drift', value => value.receipt.cases[0].source = 'examples/other.sio', 'source mismatch');
  check('unsafe path', value => value.manifest.examples[0].source = '../secret.sio', 'invalid source');
  check('exit drift', value => value.receipt.cases[0].actual_exit = 1, 'exit mismatch');
  check('mode drift', value => value.manifest.examples[0].mode = 'check', 'run state');
  check('missing output', value => value.receipt.cases[0].stdout = '', 'stdout_contains');
  check('duplicate example', value => value.manifest.examples.push(value.manifest.examples[0]), 'duplicate example');
  check('duplicate observation', value => value.receipt.cases.push(value.receipt.cases[0]), 'duplicate receipt');
  check('alias source hash drift', value => value.manifest.examples[1].source_sha256 = 'e'.repeat(64), 'conflicting source hashes');
  check('negative kind cannot be a passed control', value => value.manifest.examples[1].kind = 'negative', 'positive exit or kind');
  for (const expectedExit of ['0', null, 0.5, -1, 256]) {
    check('positive exit must be a process status', value => { value.manifest.examples[0].expected_exit = expectedExit; value.receipt.cases[0].expected_exit = expectedExit; value.receipt.cases[0].actual_exit = expectedExit; }, 'positive exit or kind');
  }
  check('failure reason cannot be a passing observation', value => value.receipt.cases[0].reason = 'timeout', 'failure reason');
  check('nonzero check is not success', value => {
    Object.assign(value.manifest.examples[0], { mode: 'check', expected_exit: 42, verification: { state: 'check-passed' } });
    Object.assign(value.receipt.cases[0], { mode: 'check', expected_exit: 42, actual_exit: 42, state: 'check-passed' });
  }, 'check-passed requires zero exit');
  const refusal = structuredClone(state);
  Object.assign(refusal.manifest.examples[1], { kind: 'negative', mode: 'check-fail', expected_exit: 'nonzero', diagnostic: 'E001', stdout_contains: 'E001', verification: { state: 'expected-refusal' } });
  Object.assign(refusal.receipt.cases[1], { state: 'expected-refusal', mode: 'check-fail', expected_exit: 'nonzero', actual_exit: 1, stdout: 'E001' });
  assert.deepEqual(validateCatalog(refusal.catalog, refusal.manifest, refusal.receipt), []); cases += 1;
  for (const expectedExit of [0, 139, 'any', 2]) {
    const invalid = structuredClone(refusal);
    invalid.manifest.examples[1].expected_exit = expectedExit;
    invalid.receipt.cases[1].expected_exit = expectedExit;
    assert.ok(validateCatalog(invalid.catalog, invalid.manifest, invalid.receipt).some(error => error.includes('refusal exit contract'))); cases += 1;
  }
  const unasserted = structuredClone(refusal);
  unasserted.manifest.examples[1].stdout_contains = 'unrelated';
  assert.ok(validateCatalog(unasserted.catalog, unasserted.manifest, unasserted.receipt).some(error => error.includes('not asserted'))); cases += 1;
  refusal.receipt.cases[1].actual_exit = 139;
  assert.ok(validateCatalog(refusal.catalog, refusal.manifest, refusal.receipt).some(error => error.includes('refusal state'))); cases += 1;
  for (const language of languages) {
    const pack = contextPack(task, state, language);
    assert.equal(pack.examples[0].code, source);
    assert.equal(pack.examples[0].recorded_state, 'run-passed');
    assert.match(renderPack(pack), /fn main/);
    assert.match(pack.evidence_boundary, language === 'en' ? /does not execute/ : language === 'pt' ? /não executa/ : /δεν εκτελεί/);
    cases += 1;
  }
  assert.throws(() => contextPack(task, state, 'xx'), /Unsupported/); cases += 1;
  assert.throws(() => contextPack({ ...task, reference: 'missing' }, state, 'en'), /validated example/); cases += 1;
  assert.equal(selectTasks([task], 'impressão').length, 1); cases += 1;
  assert.equal(selectTasks([task], 'PRINT greeting').length, 1); cases += 1;
  assert.equal(selectTasks([task], 'print unknown').length, 0); cases += 1;
  const fenced = contextPack(task, state, 'en'); fenced.examples[0].code = '```\n';
  assert.match(renderPack(fenced), /````sio/); cases += 1;
  const batch = Buffer.from(`${'a'.repeat(40)} blob 3\nabc\n`);
  assert.equal(parseBaselineBatch(batch, ['example.sio']).get('example.sio').toString(), 'abc'); cases += 1;
  for (const bytes of [Buffer.from('missing\n'), Buffer.from('bad header'), batch.subarray(0, -2), Buffer.concat([batch, Buffer.from('extra')])]) {
    assert.throws(() => parseBaselineBatch(bytes, ['example.sio'])); cases += 1;
  }
  return { cases, compiler_invoked: false, receipt_authenticated: false };
}

async function main() {
  const argumentsList = process.argv.slice(2);
  if (!argumentsList.length || argumentsList.includes('--help')) {
    console.log('Sounio task context (read-only; never invokes the compiler)\nUsage: node scripts/docs/sounio_task_context.mjs list|search <terms>|show <task-id> [--lang en|pt|el] [--json]\n       node scripts/docs/sounio_task_context.mjs --validate|--self-test');
    return;
  }
  if (argumentsList.length === 1 && argumentsList[0] === '--self-test') { console.log(JSON.stringify(selfTest())); return; }
  const json = argumentsList.includes('--json');
  let language = 'en';
  const operands = [];
  for (let index = 0; index < argumentsList.length; index += 1) {
    const argument = argumentsList[index];
    if (argument === '--json') continue;
    if (argument === '--lang') { language = argumentsList[++index]; if (!languages.includes(language)) throw new Error('--lang requires en, pt or el'); continue; }
    if (argument.startsWith('--') && argument !== '--validate') throw new Error(`Unknown option: ${argument}`);
    operands.push(argument);
  }
  const [command, ...query] = operands;
  if (!['list', 'search', 'show', '--validate'].includes(command)) throw new Error(`Unknown command: ${command}`);
  if ((['list', '--validate'].includes(command) && query.length) || (command === 'show' && query.length !== 1) || (command === 'search' && !query.length)) throw new Error('Invalid command operands; use --help');
  const state = await load();
  if (command === '--validate') {
    let contextPacks = 0;
    for (const task of state.catalog.tasks) {
      for (const selectedLanguage of languages) {
        const pack = contextPack(task, state, selectedLanguage);
        const rendered = renderPack(pack);
        for (const example of pack.examples) {
          assert.equal(example.code, state.sources.get(example.source));
          assert.equal(digest(Buffer.from(example.code)), example.source_sha256);
          assert.ok(rendered.includes(example.code.replace(/\n$/, '')));
        }
        contextPacks += 1;
      }
    }
    console.log(JSON.stringify({ schema: 'sounio.llm-task-map.validation.v1', tasks: state.catalog.tasks.length, context_packs: contextPacks, source_files: state.sources.size, compiler_invoked: false, receipt_authenticated: false, errors: [] }, null, 2));
  } else if (command === 'show') {
    const task = state.catalog.tasks.find(entry => entry.id === query[0]);
    if (!task) throw new Error(`Unknown task: ${query[0]}; use list or search`);
    const pack = contextPack(task, state, language);
    console.log(json ? JSON.stringify(pack, null, 2) : renderPack(pack));
  } else {
    const tasks = command === 'search' ? selectTasks(state.catalog.tasks, query.join(' ')) : state.catalog.tasks;
    const rows = tasks.map(task => ({ id: task.id, level: task.level, title: task.title[language], reference: task.reference }));
    if (json) console.log(JSON.stringify(rows, null, 2));
    else console.log(rows.map(row => `${row.id}\t${row.level}\t${row.title}\t${row.reference}`).join('\n') || 'No matching task / nenhuma tarefa encontrada.');
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => { console.error(`Sounio task context: ${error.message}`); process.exitCode = 1; });
}
